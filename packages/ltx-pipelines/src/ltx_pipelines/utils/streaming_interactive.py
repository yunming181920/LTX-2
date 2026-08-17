"""Interactive (live-prompt) streaming TI2V driver — a generator over AR chunks.

A thin variant of the M1 driver :func:`ltx_pipelines.utils.streaming.streaming_generate_joint`
that exposes two things the offline driver hides inside its loop:

  * a **per-chunk text context** via ``context_resolver``. Text conditioning is
    cross-attention (see :class:`JointStreamingTwinDenoiser`), *not* part of the
    cached self-attention history (TwinCache snapshots) or the sink, so a prompt
    change between chunks is architecturally clean: it changes only the
    cross-attention conditioning of subsequent chunks and leaves the sliding-window
    history untouched. This is what makes live prompt injection during streaming
    generation possible.
  * the finalized latent **prefixes** after each chunk, so a caller can decode and
    emit video/audio incrementally (streaming output) instead of decoding once at
    the end.

It reuses every M1 primitive from :mod:`ltx_pipelines.utils.streaming` unchanged
(``_build_window_state`` / ``_build_audio_window_state`` /
:class:`JointStreamingTwinDenoiser` / ``cross_causal_attention_mask`` / the audio
window-alignment and patchify helpers) plus
:func:`ltx_pipelines.utils.samplers.euler_denoising_loop`. The chunk loop body is
:func:`streaming_generate_joint` verbatim except for the resolved context and the
per-chunk ``yield`` — so a ``context_resolver`` that returns a constant context
reproduces M1 exactly (covered by the parity assertion in
``tests/test_streaming_interactive.py``).

Prompt-agnostic: this module knows nothing about prompt strings or text encoders.
The caller decides what context each chunk gets (and re-encodes prompts on change,
caching the result); here we simply call ``context_resolver(i, num_chunks)`` once
per chunk and pass its ``(v_context, a_context)`` to that chunk's denoiser.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.model.transformer import CausalStreamingModel
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.streaming import (
    ChunkSnapshots,
    JointStreamingTwinDenoiser,
    _audio_chunk_frame_count,
    _audio_window_alignment,
    _build_audio_window_positions,
    _build_audio_window_state,
    _build_window_positions,
    _build_window_state,
    _patchify_frame_latent,
    _unpatchify_audio_tokens,
    _unpatchify_tokens,
    _window_pe,
    block_causal_attention_mask,
    cross_causal_attention_mask,
)

logger = logging.getLogger(__name__)

#: ``(v_context, a_context)`` provider invoked once per AR chunk.
ContextResolver = Callable[[int, int], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class StreamChunk:
    """One finalized AR chunk plus the accumulated latent prefixes for incremental decode.

    ``video_latent_prefix`` / ``audio_latent_prefix`` hold **only the frames filled
    so far** (the sink + every finalized chunk up to and including this one), so a
    caller can decode exactly the generated content and grow the output clip. The
    trailing zero-padded region of the full-size buffer is excluded.
    """

    chunk_index: int
    num_chunks: int
    video_latent_prefix: torch.Tensor  # (1, C, 1 + frames_generated, h, w)
    audio_latent_prefix: torch.Tensor  # (1, C, audio_generated, mel)
    new_video_frames: int
    new_audio_frames: int


def iter_streaming_chunks_joint(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    window_chunks: int,
    video_tools_full: VideoLatentTools,
    audio_tools_full: AudioLatentTools,
    sink_latent_unpatchified: torch.Tensor,
    context_resolver: ContextResolver,
    stepper: EulerDiffusionStep,
    transformer,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    causal_cross_attn: bool = True,
    cross_attn_lookahead_sec: float = 0.0,
) -> Iterator[StreamChunk]:
    """Autoregressive streaming TI2V generation yielding one :class:`StreamChunk` per AR chunk.

    Identical to :func:`streaming_generate_joint` (M1) except:

      * ``v_context`` / ``a_context`` are replaced by ``context_resolver``, called as
        ``context_resolver(i, num_chunks)`` at the top of each chunk to obtain that
        chunk's ``(v_context, a_context)``. Returning a different context for later
        chunks is the live-prompt-injection hook; it does not affect the cached
        history (text is cross-attention only).
      * after each chunk's clean latents are spliced into the full buffers, the
        filled prefixes are yielded (via :class:`StreamChunk`) so the caller can
        decode and emit incrementally.

    Video history: persistent first chunk (always clean) + rolling FIFO capped at
    ``window_chunks``. Audio history: rolling FIFO only (no sink, no persistent
    anchor). The time-causal AV cross-attention mask and the audio-window clock
    alignment are applied exactly as in M1.
    """
    patchifier = video_tools_full.patchifier
    audio_patchifier = AudioPatchifier(patch_size=1)
    h_lat = video_tools_full.target_shape.height
    w_lat = video_tools_full.target_shape.width
    channels = video_tools_full.target_shape.channels
    fps = video_tools_full.fps
    tokens_per_frame = h_lat * w_lat
    audio_channels = audio_tools_full.target_shape.channels
    audio_mel = audio_tools_full.target_shape.mel_bins

    total_latent_frames = video_tools_full.target_shape.frames
    full_video_latent = torch.zeros(
        (1, channels, total_latent_frames, h_lat, w_lat), device=device, dtype=dtype
    )
    full_video_latent[:, :, 0:1, :, :] = sink_latent_unpatchified[:, :, 0:1, :, :]
    total_audio_frames = audio_tools_full.target_shape.frames
    full_audio_latent = torch.zeros(
        (1, audio_channels, total_audio_frames, audio_mel), device=device, dtype=dtype
    )

    sink_tokens = _patchify_frame_latent(sink_latent_unpatchified, patchifier)
    if sink_tokens.shape[1] != tokens_per_frame:
        sink_tokens = sink_tokens[:, :tokens_per_frame, :].contiguous()

    num_steps = len(sigmas) - 1
    sigma_mid_step = max(1, num_steps // 2)
    first_ref: ChunkSnapshots | None = None  # video persistent first chunk
    rolling_video: deque[ChunkSnapshots] = deque(maxlen=window_chunks)
    rolling_audio: deque[ChunkSnapshots] = deque(maxlen=window_chunks)

    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames
    frames_generated_before = 0
    audio_generated_before = 0

    for i in range(num_chunks):
        current_video_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
        frames_through_chunk = frames_generated_before + current_video_frames
        current_audio_frames = _audio_chunk_frame_count(frames_through_chunk, audio_generated_before, fps)

        # --- live-prompt hook: resolve THIS chunk's text context (cross-attention only) ---
        v_context, a_context = context_resolver(i, num_chunks)

        # --- video window [sink | first | history | current] ---
        video_history = ([first_ref] if first_ref is not None else []) + list(rolling_video)
        video_state, sink_range, video_history_ranges, video_current_range = _build_window_state(
            video_tools=video_tools_full,
            sink_tokens=sink_tokens,
            history=video_history,
            current_frames=current_video_frames,
            tokens_per_frame=tokens_per_frame,
            noiser=noiser,
            device=device,
            dtype=dtype,
        )

        # --- audio window [history | current], clock-aligned to the video window ---
        audio_history = list(rolling_audio)
        a_abs_start, a_time_shift = _audio_window_alignment(
            audio_generated_before=audio_generated_before,
            audio_hist_frames=sum(s.frames for s in audio_history),
            video_abs_current_frame=1 + frames_generated_before,
            video_rel_current_frame=1 + sum(s.frames for s in video_history),
            fps=fps,
        )
        audio_state, audio_history_ranges, audio_current_range = _build_audio_window_state(
            audio_tools_full=audio_tools_full,
            history=audio_history,
            current_frames=current_audio_frames,
            noiser=noiser,
            device=device,
            dtype=dtype,
            abs_start_frame=a_abs_start,
            time_shift_sec=a_time_shift,
        )

        # --- time-causal AV cross-attention masks (window-relative positions) ---
        if causal_cross_attn:
            a2v_mask, v2a_mask = cross_causal_attention_mask(
                video_state.positions, audio_state.positions, cross_attn_lookahead_sec
            )
            video_state = replace(video_state, cross_attention_mask=a2v_mask)
            audio_state = replace(audio_state, cross_attention_mask=v2a_mask)

        denoiser = JointStreamingTwinDenoiser(
            v_context=v_context,
            a_context=a_context,
            video_sink_tokens=sink_tokens,
            video_sink_range=sink_range,
            video_history=video_history,
            video_history_ranges=video_history_ranges,
            video_current_range=video_current_range,
            audio_history=audio_history,
            audio_history_ranges=audio_history_ranges,
            audio_current_range=audio_current_range,
            sigma_mid_step=sigma_mid_step,
            num_steps=num_steps,
        )

        logger.info(
            "Interactive streaming AR chunk %d/%d (video=%d audio=%d frames, v-hist=%d a-hist=%d)",
            i + 1, num_chunks, current_video_frames, current_audio_frames,
            len(video_history), len(audio_history),
        )

        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

        # --- finalize video chunk ---
        vc0, vc1 = video_current_range
        video_clean_tokens = video_state.latent[:, vc0:vc1, :].clone()
        if first_ref is None:
            first_ref = ChunkSnapshots(
                tokens_noisy=video_clean_tokens, tokens_clean=video_clean_tokens, frames=current_video_frames
            )
        else:
            v_noisy = (
                denoiser.noisy_capture_video.clone()
                if denoiser.noisy_capture_video is not None
                else video_clean_tokens.clone()
            )
            rolling_video.append(
                ChunkSnapshots(tokens_noisy=v_noisy, tokens_clean=video_clean_tokens, frames=current_video_frames)
            )
        video_clean_unpatchified = _unpatchify_tokens(
            video_clean_tokens, current_video_frames, h_lat, w_lat, channels, patchifier
        )
        f0 = 1 + frames_generated_before
        full_video_latent[:, :, f0 : f0 + current_video_frames, :, :] = video_clean_unpatchified

        # --- finalize audio chunk ---
        ac0, ac1 = audio_current_range
        audio_clean_tokens = audio_state.latent[:, ac0:ac1, :].clone()
        a_noisy = (
            denoiser.noisy_capture_audio.clone()
            if denoiser.noisy_capture_audio is not None
            else audio_clean_tokens.clone()
        )
        rolling_audio.append(
            ChunkSnapshots(tokens_noisy=a_noisy, tokens_clean=audio_clean_tokens, frames=current_audio_frames)
        )
        audio_clean_unpatchified = _unpatchify_audio_tokens(
            audio_clean_tokens, current_audio_frames, audio_channels, audio_mel, audio_patchifier
        )
        a0 = audio_generated_before
        full_audio_latent[:, :, a0 : a0 + current_audio_frames, :] = audio_clean_unpatchified

        frames_generated_before += current_video_frames
        audio_generated_before += current_audio_frames

        # Yield the filled prefixes (sink + all finalized chunks so far) so the
        # caller can decode and emit incrementally. The audio prefix already has
        # exactly ``audio_generated_before`` filled frames (no trailing pad).
        video_prefix = full_video_latent[:, :, : 1 + frames_generated_before, :, :].contiguous()
        audio_prefix = full_audio_latent[:, :, :audio_generated_before, :].contiguous()

        yield StreamChunk(
            chunk_index=i,
            num_chunks=num_chunks,
            video_latent_prefix=video_prefix,
            audio_latent_prefix=audio_prefix,
            new_video_frames=current_video_frames,
            new_audio_frames=current_audio_frames,
        )


def iter_streaming_chunks_joint_cached(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    window_chunks: int,
    video_tools_full: VideoLatentTools,
    audio_tools_full: AudioLatentTools,
    sink_latent_unpatchified: torch.Tensor,
    context_resolver: ContextResolver,
    stepper: EulerDiffusionStep,
    transformer,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    causal_cross_attn: bool = True,
    cross_attn_lookahead_sec: float = 0.0,
    strategy: str = "twin",
) -> Iterator[StreamChunk]:
    """Interactive (live-prompt) variant of :func:`streaming_generate_joint_cached`.

    Identical to the offline M2 cached driver except ``v_context``/``a_context``
    are replaced by ``context_resolver`` (called per chunk) and one
    :class:`StreamChunk` (filled prefixes) is yielded after each chunk's finalize,
    so a caller can decode/emit incrementally. The :class:`CausalStreamingModel`
    wrapper is built here and detached in a ``finally`` (runs on generator
    exhaustion/close), so the strategy is selectable per generation. ``strategy``
    selects the TwinCache variant (``twin``/``clean``/``noisy_steps``).
    """
    patchifier = video_tools_full.patchifier
    audio_patchifier = AudioPatchifier(patch_size=1)
    h_lat = video_tools_full.target_shape.height
    w_lat = video_tools_full.target_shape.width
    channels = video_tools_full.target_shape.channels
    fps = video_tools_full.fps
    tokens_per_frame = h_lat * w_lat
    audio_channels = audio_tools_full.target_shape.channels
    audio_mel = audio_tools_full.target_shape.mel_bins
    audio_token_dim = audio_channels * audio_mel

    total_latent_frames = video_tools_full.target_shape.frames
    full_video_latent = torch.zeros((1, channels, total_latent_frames, h_lat, w_lat), device=device, dtype=dtype)
    full_video_latent[:, :, 0:1, :, :] = sink_latent_unpatchified[:, :, 0:1, :, :]
    total_audio_frames = audio_tools_full.target_shape.frames
    full_audio_latent = torch.zeros(
        (1, audio_channels, total_audio_frames, audio_mel), device=device, dtype=dtype
    )

    sink_tokens = _patchify_frame_latent(sink_latent_unpatchified, patchifier)
    if sink_tokens.shape[1] != tokens_per_frame:
        sink_tokens = sink_tokens[:, :tokens_per_frame, :].contiguous()

    num_steps = len(sigmas) - 1
    sigma_mid_step = max(1, num_steps // 2)
    wrapper = CausalStreamingModel(
        transformer,
        window_chunks,
        tokens_per_frame,
        cache_audio=True,
        audio_tokens_per_frame=1,
        strategy=strategy,
        num_steps=num_steps,
    )

    first_frames = 0
    rolling_frames: deque[int] = deque(maxlen=window_chunks)
    rolling_audio_frames: deque[int] = deque(maxlen=window_chunks)
    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames
    frames_generated_before = 0
    audio_generated_before = 0

    try:
        for i in range(num_chunks):
            current_video_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
            frames_through_chunk = frames_generated_before + current_video_frames
            current_audio_frames = _audio_chunk_frame_count(frames_through_chunk, audio_generated_before, fps)

            v_context, a_context = context_resolver(i, num_chunks)

            v_hist_frames = first_frames + sum(rolling_frames)
            v_sink_t = tokens_per_frame
            v_hist_t = v_hist_frames * tokens_per_frame
            v_cur_t = current_video_frames * tokens_per_frame
            full_positions, _, _, _ = _build_window_positions(
                video_tools_full, v_hist_frames, current_video_frames, device
            )
            v_window_pe = _window_pe(full_positions, wrapper, dtype)
            v_full_tokens = full_positions.shape[2]
            if v_full_tokens != v_sink_t + v_hist_t + v_cur_t:
                raise RuntimeError(
                    "Video window positions out of sync: "
                    f"positions={v_full_tokens}, sink+hist+cur={v_sink_t + v_hist_t + v_cur_t}."
                )
            v_full_window_frames = v_full_tokens // tokens_per_frame
            v_full_frame_indices = torch.arange(v_full_window_frames, device=device).repeat_interleave(
                tokens_per_frame
            )
            v_sink_rows = torch.arange(0, v_sink_t, device=device)
            v_current_rows = torch.arange(v_sink_t + v_hist_t, v_sink_t + v_hist_t + v_cur_t, device=device)
            v_query_rows = torch.cat([v_sink_rows, v_current_rows])
            v_query_mask = block_causal_attention_mask(
                v_full_frame_indices[v_query_rows], v_full_frame_indices
            )

            a_hist_t = sum(rolling_audio_frames)
            a_cur_t = current_audio_frames
            a_abs_start, a_time_shift = _audio_window_alignment(
                audio_generated_before=audio_generated_before,
                audio_hist_frames=a_hist_t,
                video_abs_current_frame=1 + frames_generated_before,
                video_rel_current_frame=1 + v_hist_frames,
                fps=fps,
            )
            a_full_positions, _, _ = _build_audio_window_positions(
                audio_tools_full, a_hist_t, current_audio_frames, device,
                abs_start_frame=a_abs_start, time_shift_sec=a_time_shift,
            )
            a_window_pe = _window_pe(a_full_positions, wrapper, dtype, audio=True)
            a_full_tokens = a_full_positions.shape[2]
            if a_full_tokens != a_hist_t + a_cur_t:
                raise RuntimeError(
                    "Audio window positions out of sync: "
                    f"positions={a_full_tokens}, hist+cur={a_hist_t + a_cur_t}."
                )
            a_full_frame_indices = torch.arange(a_full_tokens, device=device)
            a_current_rows = torch.arange(a_hist_t, a_hist_t + a_cur_t, device=device)
            a_query_mask = block_causal_attention_mask(
                a_full_frame_indices[a_current_rows], a_full_frame_indices
            )

            v_window_noise = torch.randn(
                (1, v_sink_t + v_hist_t + v_cur_t, channels), device=device, dtype=dtype, generator=noiser.generator
            )
            v_cur_noise = v_window_noise[:, v_sink_t + v_hist_t :, :]
            v_mod_latent = torch.cat([sink_tokens, v_cur_noise], dim=1)
            v_mod_clean = torch.zeros_like(v_mod_latent)
            v_mod_clean[:, :v_sink_t] = sink_tokens
            v_mod_mask = torch.zeros((1, v_mod_latent.shape[1], 1), device=device, dtype=torch.float32)
            v_mod_mask[:, v_sink_t:] = 1.0
            v_mod_positions = full_positions[:, :, v_query_rows, :]
            video_state = LatentState(
                latent=v_mod_latent, denoise_mask=v_mod_mask, positions=v_mod_positions,
                clean_latent=v_mod_clean, attention_mask=None,
            )

            a_window_noise = torch.randn(
                (1, a_hist_t + a_cur_t, audio_token_dim), device=device, dtype=dtype, generator=noiser.generator
            )
            a_cur_noise = a_window_noise[:, a_hist_t:, :]
            a_mod_latent = a_cur_noise
            a_mod_clean = torch.zeros_like(a_mod_latent)
            a_mod_mask = torch.ones((1, a_mod_latent.shape[1], 1), device=device, dtype=torch.float32)
            a_mod_positions = a_full_positions[:, :, a_current_rows, :]
            audio_state = LatentState(
                latent=a_mod_latent, denoise_mask=a_mod_mask, positions=a_mod_positions,
                clean_latent=a_mod_clean, attention_mask=None,
            )

            if causal_cross_attn:
                a2v_mask, v2a_mask = cross_causal_attention_mask(
                    video_state.positions, audio_state.positions, cross_attn_lookahead_sec
                )
                video_state = replace(video_state, cross_attention_mask=a2v_mask)
                audio_state = replace(audio_state, cross_attention_mask=v2a_mask)

            wrapper.prepare_chunk(
                window_pe=v_window_pe, query_mask=v_query_mask, hist_len=v_hist_t,
                audio_window_pe=a_window_pe, audio_query_mask=a_query_mask, audio_hist_len=a_hist_t,
            )

            logger.info(
                "Interactive streaming AR chunk %d/%d (cached %s; video=%d audio=%d frames)",
                i + 1, num_chunks, strategy, current_video_frames, current_audio_frames,
            )

            for step_idx in range(num_steps):
                if strategy == "noisy_steps":
                    wrapper.set_mode("noisy", step_idx=step_idx)
                elif strategy == "clean":
                    wrapper.set_mode("clean")
                else:
                    wrapper.set_mode("clean" if step_idx == num_steps - 1 else "noisy")
                pos_video = modality_from_latent_state(video_state, v_context, sigmas[step_idx])
                pos_audio = modality_from_latent_state(audio_state, a_context, sigmas[step_idx])
                denoised_video, denoised_audio = wrapper(video=pos_video, audio=pos_audio, perturbations=None)
                denoised_video = post_process_latent(
                    denoised_video, video_state.denoise_mask, video_state.clean_latent
                )
                denoised_audio = post_process_latent(
                    denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
                )
                if strategy == "noisy_steps":
                    wrapper.stash("noisy", step_idx=step_idx)
                elif step_idx == sigma_mid_step:
                    wrapper.stash("noisy")
                video_state = replace(
                    video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx)
                )
                audio_state = replace(
                    audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx)
                )

            if strategy == "noisy_steps":
                wrapper.commit()
            else:
                wrapper.set_mode("clean")
                zero_sigma = torch.zeros_like(sigmas[0])
                pos_video = modality_from_latent_state(video_state, v_context, zero_sigma)
                pos_audio = modality_from_latent_state(audio_state, a_context, zero_sigma)
                wrapper(video=pos_video, audio=pos_audio, perturbations=None)
                wrapper.stash("clean")
                wrapper.commit()
            if first_frames == 0:
                first_frames = current_video_frames
            else:
                rolling_frames.append(current_video_frames)
            rolling_audio_frames.append(current_audio_frames)

            v_clean_tokens = video_state.latent[:, v_sink_t : v_sink_t + v_cur_t, :].clone()
            v_clean_unpatchified = _unpatchify_tokens(
                v_clean_tokens, current_video_frames, h_lat, w_lat, channels, patchifier
            )
            f0 = 1 + frames_generated_before
            full_video_latent[:, :, f0 : f0 + current_video_frames, :, :] = v_clean_unpatchified

            a_clean_tokens = audio_state.latent.clone()
            a_clean_unpatchified = _unpatchify_audio_tokens(
                a_clean_tokens, current_audio_frames, audio_channels, audio_mel, audio_patchifier
            )
            a0 = audio_generated_before
            full_audio_latent[:, :, a0 : a0 + current_audio_frames, :] = a_clean_unpatchified

            frames_generated_before += current_video_frames
            audio_generated_before += current_audio_frames

            video_prefix = full_video_latent[:, :, : 1 + frames_generated_before, :, :].contiguous()
            audio_prefix = full_audio_latent[:, :, :audio_generated_before, :].contiguous()
            yield StreamChunk(
                chunk_index=i, num_chunks=num_chunks,
                video_latent_prefix=video_prefix, audio_latent_prefix=audio_prefix,
                new_video_frames=current_video_frames, new_audio_frames=current_audio_frames,
            )
    finally:
        wrapper.detach()


def iter_streaming_chunks_joint_image_cond(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    video_tools_full: VideoLatentTools,
    audio_tools_full: AudioLatentTools,
    sink_latent_unpatchified: torch.Tensor,
    context_resolver: ContextResolver,
    stepper: EulerDiffusionStep,
    transformer,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    causal_cross_attn: bool = True,
    cross_attn_lookahead_sec: float = 0.0,
) -> Iterator[StreamChunk]:
    """Interactive (live-prompt) variant of :func:`streaming_generate_joint_image_cond`.

    Ablation C as a generator: each chunk conditions on the previous chunk's last
    frame as the image reference (rotating sink), no attention history, no KV
    cache. ``context_resolver`` is called per chunk for the cross-attention text
    context; one :class:`StreamChunk` is yielded after each chunk.
    """
    patchifier = video_tools_full.patchifier
    audio_patchifier = AudioPatchifier(patch_size=1)
    h_lat = video_tools_full.target_shape.height
    w_lat = video_tools_full.target_shape.width
    channels = video_tools_full.target_shape.channels
    fps = video_tools_full.fps
    tokens_per_frame = h_lat * w_lat
    audio_channels = audio_tools_full.target_shape.channels
    audio_mel = audio_tools_full.target_shape.mel_bins

    total_latent_frames = video_tools_full.target_shape.frames
    full_video_latent = torch.zeros(
        (1, channels, total_latent_frames, h_lat, w_lat), device=device, dtype=dtype
    )
    full_video_latent[:, :, 0:1, :, :] = sink_latent_unpatchified[:, :, 0:1, :, :]
    total_audio_frames = audio_tools_full.target_shape.frames
    full_audio_latent = torch.zeros(
        (1, audio_channels, total_audio_frames, audio_mel), device=device, dtype=dtype
    )

    sink_tokens = _patchify_frame_latent(sink_latent_unpatchified, patchifier)
    if sink_tokens.shape[1] != tokens_per_frame:
        sink_tokens = sink_tokens[:, :tokens_per_frame, :].contiguous()

    num_steps = len(sigmas) - 1
    sigma_mid_step = max(1, num_steps // 2)
    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames
    frames_generated_before = 0
    audio_generated_before = 0

    for i in range(num_chunks):
        current_video_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
        frames_through_chunk = frames_generated_before + current_video_frames
        current_audio_frames = _audio_chunk_frame_count(frames_through_chunk, audio_generated_before, fps)

        v_context, a_context = context_resolver(i, num_chunks)

        video_state, sink_range, video_history_ranges, video_current_range = _build_window_state(
            video_tools=video_tools_full, sink_tokens=sink_tokens, history=[],
            current_frames=current_video_frames, tokens_per_frame=tokens_per_frame,
            noiser=noiser, device=device, dtype=dtype,
        )
        a_abs_start, a_time_shift = _audio_window_alignment(
            audio_generated_before=audio_generated_before, audio_hist_frames=0,
            video_abs_current_frame=1 + frames_generated_before, video_rel_current_frame=1, fps=fps,
        )
        audio_state, audio_history_ranges, audio_current_range = _build_audio_window_state(
            audio_tools_full=audio_tools_full, history=[], current_frames=current_audio_frames,
            noiser=noiser, device=device, dtype=dtype,
            abs_start_frame=a_abs_start, time_shift_sec=a_time_shift,
        )
        if causal_cross_attn:
            a2v_mask, v2a_mask = cross_causal_attention_mask(
                video_state.positions, audio_state.positions, cross_attn_lookahead_sec
            )
            video_state = replace(video_state, cross_attention_mask=a2v_mask)
            audio_state = replace(audio_state, cross_attention_mask=v2a_mask)

        denoiser = JointStreamingTwinDenoiser(
            v_context=v_context, a_context=a_context,
            video_sink_tokens=sink_tokens, video_sink_range=sink_range,
            video_history=[], video_history_ranges=video_history_ranges,
            video_current_range=video_current_range,
            audio_history=[], audio_history_ranges=audio_history_ranges,
            audio_current_range=audio_current_range,
            sigma_mid_step=sigma_mid_step, num_steps=num_steps,
        )

        logger.info(
            "Interactive streaming AR chunk %d/%d (image-cond; video=%d audio=%d frames)",
            i + 1, num_chunks, current_video_frames, current_audio_frames,
        )

        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas, video_state=video_state, audio_state=audio_state,
            stepper=stepper, transformer=transformer, denoiser=denoiser,
        )

        vc0, vc1 = video_current_range
        video_clean_tokens = video_state.latent[:, vc0:vc1, :].clone()
        video_clean_unpatchified = _unpatchify_tokens(
            video_clean_tokens, current_video_frames, h_lat, w_lat, channels, patchifier
        )
        f0 = 1 + frames_generated_before
        full_video_latent[:, :, f0 : f0 + current_video_frames, :, :] = video_clean_unpatchified
        sink_tokens = video_clean_tokens[:, -tokens_per_frame:, :].clone()

        ac0, ac1 = audio_current_range
        audio_clean_tokens = audio_state.latent[:, ac0:ac1, :].clone()
        audio_clean_unpatchified = _unpatchify_audio_tokens(
            audio_clean_tokens, current_audio_frames, audio_channels, audio_mel, audio_patchifier
        )
        a0 = audio_generated_before
        full_audio_latent[:, :, a0 : a0 + current_audio_frames, :] = audio_clean_unpatchified

        frames_generated_before += current_video_frames
        audio_generated_before += current_audio_frames

        video_prefix = full_video_latent[:, :, : 1 + frames_generated_before, :, :].contiguous()
        audio_prefix = full_audio_latent[:, :, :audio_generated_before, :].contiguous()
        yield StreamChunk(
            chunk_index=i, num_chunks=num_chunks,
            video_latent_prefix=video_prefix, audio_latent_prefix=audio_prefix,
            new_video_frames=current_video_frames, new_audio_frames=current_audio_frames,
        )
