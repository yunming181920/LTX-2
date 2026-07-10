"""Streaming, autoregressive, causal A2V inference primitives (Milestone 1).

Training-free reproduction of Vidu S1 §2.3.1 streaming inference on top of the
pretrained *bidirectional* LTX-2 checkpoint, used as-is as the "causal model".

Milestone 1 = correct-but-slow, NO core changes, NO KV cache:
  * block-causal self-attention mask on the temporal axis (routed through the
    existing ``Modality.attention_mask`` → ``TransformerArgs.self_attention_mask``
    channel — no DiT changes),
  * sliding-window decoding with a persistent reference context ("sink" = the
    encoded first-frame latent, fixed at window-relative frame 0),
  * latent-level TwinCache: each finalized history chunk stores a *noisy*
    snapshot (captured at a mid denoising step) and a *clean* snapshot (the
    final latent); intermediate denoising steps of the current chunk read the
    *noisy* history, the final step reads the *clean* history,
  * per-token ``denoise_mask`` keeps sink+history frozen (velocity == 0 under
    the Euler step) while the current chunk is denoised,
  * audio is a frozen control signal (growing time-aligned slice per AR step).

The AR unit is one latent video frame (= 8 pixel frames = ``H_lat * W_lat``
tokens); ``chunk_frames`` may generate a few latent frames per step.

Reused verbatim from the existing stack: ``euler_denoising_loop`` +
``_step_state`` + ``post_process_latent`` + ``modality_from_latent_state`` +
``GaussianNoiser`` + ``EulerDiffusionStep`` + ``VideoLatentTools`` (which yields
window-relative RoPE positions for free) + ``VideoDecoder``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser, Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.transformer import X0Model
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import DenoisedLatentResult

logger = logging.getLogger(__name__)

# Audio latent frames per second (16000 / 160 / 4 = 25).
AUDIO_LATENT_FRAMES_PER_SECOND = 25.0
# Pixel frames advanced per latent video frame (causal VAE temporal stride).
VIDEO_LATENT_FRAME_STRIDE = 8


def block_causal_attention_mask(frame_indices: torch.Tensor) -> torch.Tensor:
    """Block-causal self-attention mask on the temporal axis.

    ``frame_indices`` is a 1-D tensor of per-token frame indices ``(T,)``.
    Returns a ``(1, T, T)`` float mask in ``[0, 1]`` where
    ``mask[q, k] = 1`` iff ``frame_indices[k] <= frame_indices[q]``: a token may
    attend to every token of an earlier-or-equal frame (fully bidirectional
    *within* a frame's spatial block, causal *across* frames). This is the
    faithful interpretation of the paper's causal mask for a frame-major
    patchifier. The ``[0, 1]`` form plugs straight into the existing
    ``self_attention_mask`` additive-log-bias channel.
    """
    fi = frame_indices.to(torch.long)
    # (T, 1) <= (1, T) -> (T, T); key frame <= query frame.
    mask = (fi.unsqueeze(1) >= fi.unsqueeze(0)).to(torch.float32)
    return mask.unsqueeze(0)  # (1, T, T)


@dataclass
class ChunkSnapshots:
    """TwinCache snapshots for one finalized AR chunk (patchified tokens).

    ``tokens_noisy`` is the chunk's latent captured at an intermediate sigma
    (residual noise -> low-pass temporal prior, used as history during the
    *intermediate* denoising steps of later chunks). ``tokens_clean`` is the
    fully-denoised final latent (used as history during the *final* step).
    Both are patchified ``(1, frames * tokens_per_frame, C)`` so they can be
    written straight into a window state's latent/clean_latent token slices.
    """

    tokens_noisy: torch.Tensor
    tokens_clean: torch.Tensor
    frames: int


class StreamingTwinDenoiser:
    """SimpleDenoiser + per-step TwinCache history injection.

    Each call, before building the Modality, writes the TwinCache-selected
    snapshot (noisy at intermediate steps, clean at the final step) into the
    sink + history token ranges of ``video_state.latent`` *and*
    ``video_state.clean_latent`` in place. Setting ``latent == clean_latent``
    on those (mask=0) tokens makes the Euler velocity ``(latent - clean)/sigma``
    zero, so ``_step_state`` / ``post_process_latent`` / ``stepper.step`` leave
    them frozen at the injected snapshot — exactly the TwinCache behaviour.
    The current chunk is left untouched and denoises normally.

    At the configured mid step it also snapshots the current chunk's evolving
    latent as that chunk's *noisy* TwinCache entry (captured into
    ``self.noisy_capture``).
    """

    def __init__(
        self,
        v_context: torch.Tensor,
        a_context: torch.Tensor,
        history: list[ChunkSnapshots],
        sink_tokens: torch.Tensor,
        sink_range: tuple[int, int],
        history_ranges: list[tuple[int, int]],
        current_range: tuple[int, int],
        sigma_mid_step: int,
        num_steps: int,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.history = history
        self.sink_tokens = sink_tokens
        self.sink_range = sink_range
        self.history_ranges = history_ranges
        self.current_range = current_range
        self.sigma_mid_step = sigma_mid_step
        self.num_steps = num_steps
        self.noisy_capture: torch.Tensor | None = None

    def _inject_history(self, video_state: LatentState, step_index: int) -> None:
        """Overwrite sink + history latent & clean_latent with the selected snapshot."""
        is_final = step_index == self.num_steps - 1
        lat = video_state.latent
        clean = video_state.clean_latent

        # Sink: always the fixed reference latent.
        s0, s1 = self.sink_range
        lat[:, s0:s1, :] = self.sink_tokens
        clean[:, s0:s1, :] = self.sink_tokens

        # History chunks: noisy snapshot mid-denoising, clean at the final step.
        for rng, snap in zip(self.history_ranges, self.history, strict=True):
            h0, h1 = rng
            tokens = snap.tokens_clean if is_final else snap.tokens_noisy
            lat[:, h0:h1, :] = tokens
            clean[:, h0:h1, :] = tokens

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        if video_state is None:
            raise ValueError("StreamingTwinDenoiser requires a video state")

        # Capture the current chunk's noisy latent at the mid step (before any
        # mutation; current tokens are never mutated here, so ordering is safe).
        if step_index == self.sigma_mid_step and self.noisy_capture is None:
            c0, c1 = self.current_range
            self.noisy_capture = video_state.latent[:, c0:c1, :].clone()

        self._inject_history(video_state, step_index)

        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )


def _patchify_frame_latent(
    unpatchified: torch.Tensor, patchifier: VideoLatentPatchifier
) -> torch.Tensor:
    """``(1, C, F, H, W) -> (1, F*H*W, C)`` patchified tokens (patch size 1)."""
    return patchifier.patchify(unpatchified)


def _unpatchify_tokens(
    tokens: torch.Tensor, frames: int, h_lat: int, w_lat: int, channels: int, patchifier: VideoLatentPatchifier
) -> torch.Tensor:
    shape = VideoLatentShape(batch=1, channels=channels, frames=frames, height=h_lat, width=w_lat)
    return patchifier.unpatchify(tokens, output_shape=shape)


def assemble_audio_slice(
    audio_latent_full: torch.Tensor,
    frames_through_chunk: int,
    fps: float,
    audio_lookahead: int,
) -> torch.Tensor:
    """Time-aligned frozen audio latent slice for the AR chunk that has
    produced ``frames_through_chunk`` latent video frames so far.

    The slice covers audio latent frames from 0 up to the end of the current
    chunk's video time span plus ``audio_lookahead`` audio frames (so the model
    sees the audio aligned to the frame it is generating). Audio latent runs at
    25 frames/sec; video latent at ``fps / 8`` frames/sec. Frame 0 is the sink,
    so generated video latent frames start at index 1.
    """
    current_end_latent_frame = 1 + frames_through_chunk
    current_end_time_sec = current_end_latent_frame * VIDEO_LATENT_FRAME_STRIDE / fps
    n_audio = int(round(current_end_time_sec * AUDIO_LATENT_FRAMES_PER_SECOND)) + audio_lookahead
    n_audio = max(1, min(n_audio, audio_latent_full.shape[2]))
    return audio_latent_full[:, :, :n_audio, :]


def _build_window_state(
    *,
    video_tools: VideoLatentTools,
    sink_tokens: torch.Tensor,
    history: list[ChunkSnapshots],
    current_frames: int,
    tokens_per_frame: int,
    noiser: Noiser,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LatentState, tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    """Assemble the sliding-window video LatentState for one AR step.

    Layout (token order): ``[sink (1 frame) | history chunks ... | current]``.
    Sink + history: ``denoise_mask = 0`` (frozen; the denoiser injects the
    TwinCache snapshot per step). Current: ``denoise_mask = 1`` (denoised),
    initialised to pure noise by the noiser. Positions are window-relative
    (built by ``VideoLatentTools.create_initial_state`` on a window-sized shape
    -> the paper's RoPE repositioning). The block-causal attention mask is built
    from per-token frame indices over the whole window.
    """
    hist_frames = sum(s.frames for s in history)
    window_frames = 1 + hist_frames + current_frames
    h_lat = video_tools.target_shape.height
    w_lat = video_tools.target_shape.width
    channels = video_tools.target_shape.channels

    # Fresh window-shaped tools so positions/patchify match this window exactly.
    window_shape = VideoLatentShape(batch=1, channels=channels, frames=window_frames, height=h_lat, width=w_lat)
    window_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), window_shape, video_tools.fps)
    state = window_tools.create_initial_state(device=device, dtype=dtype)  # zeros latent/clean, ones mask, positions

    sink_range = (0, tokens_per_frame)
    cursor = tokens_per_frame
    history_ranges: list[tuple[int, int]] = []
    for snap in history:
        n = snap.frames * tokens_per_frame
        history_ranges.append((cursor, cursor + n))
        cursor += n
    current_range = (cursor, cursor + current_frames * tokens_per_frame)

    # denoise_mask: 0 on sink+history (frozen), 1 on current (denoised).
    mask = torch.zeros_like(state.denoise_mask)
    c0, c1 = current_range
    mask[:, c0:c1] = 1.0
    state = replace(state, denoise_mask=mask)

    # Noise only the current chunk (mask gates the noiser; sink+history stay 0
    # and are overwritten by the denoiser each step anyway).
    state = noiser(state, noise_scale=1.0)

    # Block-causal attention mask over the window's frame indices.
    frame_indices = torch.arange(window_frames, device=device).repeat_interleave(tokens_per_frame)
    state = replace(state, attention_mask=block_causal_attention_mask(frame_indices))
    return state, sink_range, history_ranges, current_range


def streaming_generate(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    window_chunks: int,
    video_tools_full: VideoLatentTools,
    audio_latent_full: torch.Tensor,
    audio_lookahead: int,
    sink_latent_unpatchified: torch.Tensor,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    stepper: EulerDiffusionStep,
    transformer: X0Model,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive streaming A2V generation (returns the full video latent).

    Generates ``num_generated_latent_frames`` latent video frames (frame 0 is
    the sink, so these are the frames *after* the reference). For each AR chunk
    of up to ``chunk_frames`` latent frames: build a time-aligned frozen audio
    slice; assemble the ``[sink | history | current]`` window state with a
    block-causal mask and window-relative positions; run
    ``euler_denoising_loop`` with :class:`StreamingTwinDenoiser` (which swaps
    the history snapshot noisy↔clean per step and captures the current chunk's
    noisy mid-snapshot); finalize the chunk's TwinCache snapshots; splice the
    chunk's clean latent into the full video latent. History is a FIFO ring
    capped at ``window_chunks``.

    Generation is streaming internally (per-step activation memory is O(window),
    independent of total length). The returned full latent is decoded by the
    caller (causal-VAE seamless decode) — decode is separate from the DiT and
    need not run under the transformer's model context.
    """
    patchifier = video_tools_full.patchifier
    h_lat = video_tools_full.target_shape.height
    w_lat = video_tools_full.target_shape.width
    channels = video_tools_full.target_shape.channels
    fps = video_tools_full.fps
    tokens_per_frame = h_lat * w_lat

    # Full video latent: frame 0 = sink, frames 1.. = generated chunks.
    total_latent_frames = video_tools_full.target_shape.frames
    full_latent = torch.zeros(
        (1, channels, total_latent_frames, h_lat, w_lat), device=device, dtype=dtype
    )
    full_latent[:, :, 0:1, :, :] = sink_latent_unpatchified[:, :, 0:1, :, :]

    sink_tokens = _patchify_frame_latent(sink_latent_unpatchified, patchifier)
    # sink_latent_unpatchified may carry >1 latent frame from the VAE; keep frame 0.
    if sink_tokens.shape[1] != tokens_per_frame:
        sink_tokens = sink_tokens[:, :tokens_per_frame, :].contiguous()

    num_steps = len(sigmas) - 1
    sigma_mid_step = max(1, num_steps // 2)
    history: deque[ChunkSnapshots] = deque(maxlen=window_chunks)

    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames
    frames_generated_before = 0

    for i in range(num_chunks):
        current_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
        frames_through_chunk = frames_generated_before + current_frames

        audio_slice = assemble_audio_slice(
            audio_latent_full, frames_through_chunk, fps, audio_lookahead
        )
        audio_shape = AudioLatentShape(
            batch=1, channels=audio_slice.shape[1], frames=audio_slice.shape[2], mel_bins=audio_slice.shape[3]
        )
        audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
        audio_state = audio_tools.create_initial_state(device=device, dtype=dtype, initial_latent=audio_slice)
        # Frozen audio control: zero the denoise mask so it never changes.
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

        history_list = list(history)
        window_state, sink_range, history_ranges, current_range = _build_window_state(
            video_tools=video_tools_full,
            sink_tokens=sink_tokens,
            history=history_list,
            current_frames=current_frames,
            tokens_per_frame=tokens_per_frame,
            noiser=noiser,
            device=device,
            dtype=dtype,
        )

        denoiser = StreamingTwinDenoiser(
            v_context=v_context,
            a_context=a_context,
            history=history_list,
            sink_tokens=sink_tokens,
            sink_range=sink_range,
            history_ranges=history_ranges,
            current_range=current_range,
            sigma_mid_step=sigma_mid_step,
            num_steps=num_steps,
        )

        logger.info(
            "Streaming AR chunk %d/%d (current_frames=%d, history=%d)",
            i + 1, num_chunks, current_frames, len(history_list),
        )

        video_state, _ = euler_denoising_loop(
            sigmas=sigmas,
            video_state=window_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

        # Finalize TwinCache snapshots for this chunk.
        c0, c1 = current_range
        clean_tokens = video_state.latent[:, c0:c1, :].clone()
        noisy_tokens = (
            denoiser.noisy_capture.clone() if denoiser.noisy_capture is not None else clean_tokens.clone()
        )
        history.append(ChunkSnapshots(tokens_noisy=noisy_tokens, tokens_clean=clean_tokens, frames=current_frames))

        # Splice the chunk's clean latent into the full video latent (seamless).
        clean_unpatchified = _unpatchify_tokens(clean_tokens, current_frames, h_lat, w_lat, channels, patchifier)
        f0 = 1 + frames_generated_before
        full_latent[:, :, f0 : f0 + current_frames, :, :] = clean_unpatchified

        frames_generated_before += current_frames

    return full_latent


# ---------------------------------------------------------------------------
# Milestone 2 — KV-cache + RoPE repositioning path (UNTESTED, see plan)
# ---------------------------------------------------------------------------


def _build_window_positions(
    video_tools: VideoLatentTools,
    history: list[ChunkSnapshots],
    current_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """Full-window positions ``[sink | history | current]`` (window-relative).

    Returns ``(positions, sink_tokens, hist_tokens, current_tokens)`` where the
    positions tensor is ``(1, 3, T, 2)`` over the whole window (frame-major), so
    that the window RoPE stays in the trained range (RoPE repositioning).
    """
    h_lat = video_tools.target_shape.height
    w_lat = video_tools.target_shape.width
    tokens_per_frame = h_lat * w_lat
    hist_frames = sum(s.frames for s in history)
    window_frames = 1 + hist_frames + current_frames
    channels = video_tools.target_shape.channels
    window_shape = VideoLatentShape(batch=1, channels=channels, frames=window_frames, height=h_lat, width=w_lat)
    window_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), window_shape, video_tools.fps)
    state = window_tools.create_initial_state(device=device, dtype=torch.float32)
    return state.positions, tokens_per_frame, hist_frames * tokens_per_frame, current_frames * tokens_per_frame


def _window_pe(positions: torch.Tensor, transformer, dtype: torch.dtype):
    """Build the full-window RoPE (cos, sin) via the model's own preprocessor helper."""
    preprocessor = transformer.x0.velocity_model.video_args_preprocessor
    simple = preprocessor.simple_preprocessor
    return simple._prepare_positional_embeddings(
        positions=positions,
        inner_dim=simple.inner_dim,
        max_pos=simple.max_pos,
        use_middle_indices_grid=simple.use_middle_indices_grid,
        num_attention_heads=simple.num_attention_heads,
        x_dtype=dtype,
    )


def streaming_generate_cached(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    window_chunks: int,
    video_tools_full: VideoLatentTools,
    audio_latent_full: torch.Tensor,
    audio_lookahead: int,
    sink_latent_unpatchified: torch.Tensor,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    stepper: EulerDiffusionStep,
    transformer,  # X0Model (will be wrapped)
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """M2: KV-cache + RoPE repositioning streaming generation.

    Wraps ``transformer`` with a :class:`CausalStreamingModel` (per-block
    video-self-attn KV cache). Each AR chunk carries only ``[sink | current]``
    in the modality (history K/V come from the cache); the wrapper builds the
    full-window RoPE ``window_pe`` and the block-causal ``query_mask`` (history
    query rows removed). TwinCache: history reads the ``noisy`` snapshot during
    intermediate steps, ``clean`` at the final step; the current chunk's K/V are
    captured (mid step → noisy, final step → clean) and committed to the cache.

    NOTE: this is a best-effort blind implementation (no local runtime — see the
    plan's risk section). Run the parity test before trusting it.
    """
    from ltx_core.model.transformer import CausalStreamingModel
    from ltx_pipelines.utils.helpers import post_process_latent

    patchifier = video_tools_full.patchifier
    h_lat = video_tools_full.target_shape.height
    w_lat = video_tools_full.target_shape.width
    channels = video_tools_full.target_shape.channels
    fps = video_tools_full.fps
    tokens_per_frame = h_lat * w_lat

    total_latent_frames = video_tools_full.target_shape.frames
    full_latent = torch.zeros((1, channels, total_latent_frames, h_lat, w_lat), device=device, dtype=dtype)
    full_latent[:, :, 0:1, :, :] = sink_latent_unpatchified[:, :, 0:1, :, :]
    sink_tokens = _patchify_frame_latent(sink_latent_unpatchified, patchifier)
    if sink_tokens.shape[1] != tokens_per_frame:
        sink_tokens = sink_tokens[:, :tokens_per_frame, :].contiguous()

    wrapper = CausalStreamingModel(transformer, window_chunks, tokens_per_frame)

    num_steps = len(sigmas) - 1
    sigma_mid_step = max(1, num_steps // 2)
    history: list[ChunkSnapshots] = []  # for position/mask layout only (K/V live in the cache)
    frames_generated_before = 0
    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames

    for i in range(num_chunks):
        current_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
        frames_through_chunk = frames_generated_before + current_frames
        current_tokens = current_frames * tokens_per_frame

        # Audio slice (frozen control) for this chunk.
        audio_slice = assemble_audio_slice(audio_latent_full, frames_through_chunk, fps, audio_lookahead)
        audio_shape = AudioLatentShape(
            batch=1, channels=audio_slice.shape[1], frames=audio_slice.shape[2], mel_bins=audio_slice.shape[3]
        )
        audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
        audio_state = audio_tools.create_initial_state(device=device, dtype=dtype, initial_latent=audio_slice)
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

        # Full-window positions [sink | history | current] -> window_pe + query_mask.
        full_positions, sink_t, hist_t, cur_t = _build_window_positions(
            video_tools_full, history, current_frames, device
        )
        window_pe = _window_pe(full_positions, wrapper, dtype)
        full_frame_indices = torch.arange(
            full_positions.shape[2], device=device
        ).repeat_interleave(tokens_per_frame)
        full_mask = block_causal_attention_mask(full_frame_indices)  # (1, T, T)
        # query_mask = full block-causal with history query rows removed -> [sink | current] rows.
        sink_rows = torch.arange(0, sink_t, device=device)
        current_rows = torch.arange(sink_t + hist_t, sink_t + hist_t + cur_t, device=device)
        query_rows = torch.cat([sink_rows, current_rows])
        query_mask = full_mask[:, query_rows, :]

        # Modality carries only [sink | current]; sink frozen, current noised.
        cur_noise = torch.randn((1, current_tokens, channels), device=device, dtype=dtype, generator=noiser.generator)
        mod_latent = torch.cat([sink_tokens, cur_noise], dim=1)
        mod_clean = torch.zeros_like(mod_latent)
        mod_clean[:, :sink_t] = sink_tokens
        mod_mask = torch.zeros((1, mod_latent.shape[1], 1), device=device, dtype=torch.float32)
        mod_mask[:, sink_t:] = 1.0
        # Modality positions = full positions' [sink | current] rows.
        mod_positions = full_positions[:, :, query_rows, :]
        video_state = LatentState(
            latent=mod_latent,
            denoise_mask=mod_mask,
            positions=mod_positions,
            clean_latent=mod_clean,
            attention_mask=None,  # cached attn1 uses query_mask via the cache
        )

        wrapper.prepare_chunk(window_pe=window_pe, query_mask=query_mask, hist_len=hist_t)

        logger.info(
            "Streaming AR chunk %d/%d (cached; current_frames=%d, history=%d)",
            i + 1, num_chunks, current_frames, len(history),
        )

        # Inline per-step loop: toggle TwinCache snapshot mode, run the cached
        # model, post-process + euler step, capture K/V at mid/final.
        for step_idx in range(num_steps):
            mode = "clean" if step_idx == num_steps - 1 else "noisy"
            wrapper.set_mode(mode)
            pos_video = modality_from_latent_state(video_state, v_context, sigmas[step_idx])
            pos_audio = modality_from_latent_state(audio_state, a_context, sigmas[step_idx])
            denoised_video, _ = wrapper(video=pos_video, audio=pos_audio, perturbations=None)
            denoised_video = post_process_latent(
                denoised_video, video_state.denoise_mask, video_state.clean_latent
            )
            if step_idx == sigma_mid_step:
                wrapper.stash("noisy")
            if step_idx == num_steps - 1:
                wrapper.stash("clean")
            video_state = replace(
                video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx)
            )
        wrapper.commit()

        # Splice the finalized current latent into the full video latent.
        clean_tokens = video_state.latent[:, sink_t : sink_t + cur_t, :].clone()
        history.append(ChunkSnapshots(tokens_noisy=clean_tokens.clone(), tokens_clean=clean_tokens, frames=current_frames))
        clean_unpatchified = _unpatchify_tokens(clean_tokens, current_frames, h_lat, w_lat, channels, patchifier)
        f0 = 1 + frames_generated_before
        full_latent[:, :, f0 : f0 + current_frames, :, :] = clean_unpatchified
        frames_generated_before += current_frames

    wrapper.reset()
    return full_latent
