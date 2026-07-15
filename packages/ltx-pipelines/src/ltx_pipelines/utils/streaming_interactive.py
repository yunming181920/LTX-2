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
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.streaming import (
    ChunkSnapshots,
    JointStreamingTwinDenoiser,
    _audio_chunk_frame_count,
    _audio_window_alignment,
    _build_audio_window_state,
    _build_window_state,
    _patchify_frame_latent,
    _unpatchify_audio_tokens,
    _unpatchify_tokens,
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
