"""Streaming, autoregressive, causal TI2V (joint video+audio) inference primitives.

Training-free reproduction of Vidu S1 §2.3.1 streaming inference on top of the
pretrained *bidirectional* LTX-2 checkpoint, used as-is as the "causal model".

Unlike an audio-to-video setup (audio frozen, video generated), TI2V has no
audio input: **both** video and audio are generated. This module drives them
chunk-by-chunk in lockstep so per-step activation memory stays O(window) for
*both* modalities:

  * **block-causal self-attention mask** on each modality's temporal axis
    (routed through ``Modality.attention_mask`` →
    ``TransformerArgs.self_attention_mask`` — no DiT changes). Built as a
    structured :class:`BlockCausalMask` (per query frame block, the visible
    contiguous key prefix), NOT a dense additive bias — the attention module
    serves it with exact unmasked prefix calls, so the causal paths run on
    FlashAttention backends (which have no additive-mask kernel),
  * **sliding-window decoding** with a persistent anchor per Vidu S1 §2.3.1:
    chunk 1 is a standard **bidirectional ti2v bootstrap** — the encoded
    first-frame latent replaces window frame 0 (frozen) and denoises with
    full attention, matching the base model's training — and the whole
    ``[image | chunk 1]`` output is pinned (always clean, never evicted) as
    the video anchor; later chunks stream causally over ``[anchor | rolling |
    current]``. Audio pins its first generated chunk the same way (no image
    frame — audio has no image conditioning),
  * **latent-level TwinCache** for *both* modalities: each finalized subsequent
    chunk stores a *noisy* snapshot (captured at a mid denoising step) and a
    *clean* snapshot (the final latent); intermediate denoising steps of the
    current chunk read the *noisy* history, the final step reads the *clean*
    history,
  * **per-token ``denoise_mask``** keeps sink+history frozen (velocity == 0
    under the Euler step) while the current chunk is denoised,
  * **time-causal video↔audio cross-attention mask** built from LTX-2's shared
    seconds-axis cross-attn RoPE positions. With the audio first chunk pinned
    (like the video window's), both windows are fresh window-relative grids
    whose rolling sections compress by the identical evicted span, so they
    share one clock with no realignment. The history-less ``image_cond`` path
    instead keeps a rotating sink and aligns its audio grid via
    :func:`_audio_window_alignment` (absolute start minus the video
    compression shift).

The AR unit is one latent video frame (= 8 pixel frames = ``H_lat * W_lat``
tokens); ``chunk_frames`` may generate a few latent frames per step, and each
video chunk also produces the time-aligned audio latent frames in lockstep.

Two paths:
  * **M1** (:func:`streaming_generate_joint`) — latent TwinCache, full per-step
    recompute of history features. Correct, recommended.
  * **M2** (:func:`streaming_generate_joint_cached`) — per-block KV cache + RoPE
    repositioning for *both* video and audio self-attention. Faster, but
    conceptual/unvalidated (run ``tests/test_streaming_joint_parity.py`` first).

Reused verbatim from the existing stack: ``euler_denoising_loop`` +
``_step_state`` + ``post_process_latent`` + ``modality_from_latent_state`` +
``GaussianNoiser`` + ``EulerDiffusionStep`` + ``VideoLatentTools`` /
``AudioLatentTools`` (which yield window-relative RoPE positions for free) +
``VideoDecoder`` / ``AudioDecoder``.
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
from ltx_core.model.transformer.masking import BlockCausalMask
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


def _video_latent_frame_bounds_sec(latent_frame_index: int, fps: float) -> tuple[float, float]:
    """Causal-VAE video latent frame bounds in seconds.

    Mirrors ``VideoLatentTools.create_initial_state`` + ``get_pixel_coords`` with
    ``causal_fix=True``: the sink frame spans one pixel frame, and every later
    latent frame advances by ``VIDEO_LATENT_FRAME_STRIDE`` pixel frames.
    """
    start_frame = max(
        latent_frame_index * VIDEO_LATENT_FRAME_STRIDE + 1 - VIDEO_LATENT_FRAME_STRIDE,
        0,
    )
    end_frame = max(
        (latent_frame_index + 1) * VIDEO_LATENT_FRAME_STRIDE + 1 - VIDEO_LATENT_FRAME_STRIDE,
        0,
    )
    return start_frame / fps, end_frame / fps


def block_causal_attention_mask(
    frame_indices: torch.Tensor, k_frame_indices: torch.Tensor | None = None
) -> BlockCausalMask:
    """Structured block-causal self-attention mask on the temporal axis.

    ``frame_indices`` is a 1-D tensor of per-token *query* frame indices
    ``(T_q,)``; ``k_frame_indices`` the per-token *key* frame indices ``(T_k,)``
    (defaults to the query indices: the square self-attention case). The
    block-causal rule is ``key frame <= query frame``: a token may attend to
    every token of an earlier-or-equal frame (fully bidirectional *within* a
    frame's spatial block, causal *across* frames) — the faithful
    interpretation of the paper's causal mask for a frame-major patchifier.

    Returns a :class:`BlockCausalMask` (per query frame block, the visible
    contiguous key-prefix length) instead of a dense ``(T_q, T_k)`` additive
    bias: the attention module serves it with exact **unmasked** per-block
    prefix calls, so the causal streaming paths run on FlashAttention backends
    (FA2/FA3/FA4 and torch SDPA's FLASH kernel), which have no additive-mask
    support.
    """
    return BlockCausalMask.from_frame_indices(
        frame_indices, frame_indices if k_frame_indices is None else k_frame_indices
    )


def cross_causal_attention_mask(
    video_positions: torch.Tensor,
    audio_positions: torch.Tensor,
    lookahead_sec: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Time-causal AV cross-attention masks from LTX-2's shared seconds-axis positions.

    LTX-2 builds cross-attention RoPE from ``positions[:, 0:1, :]`` for *both*
    modalities (``MultiModalTransformerArgsPreprocessor.prepare``): the temporal
    dim, evaluated at the middle of each patch's ``[start, end)`` bounds. Video's
    temporal dim is ``frame_index / fps`` and audio's is the spectrogram
    timestamp — **both in seconds** — so the two are directly comparable on one
    time axis (Vidu S1 §2.3 "causal attention mask on video-audio tokens"). The
    streaming driver builds both states window-relative, so they share an origin.

    Returns ``(a2v, v2a)`` as ``[0, 1]`` float masks (``1`` = attend):

    * ``a2v`` ``(B, T_v, T_a)`` — video query, audio key. Allow audio frame ``j``
      iff its start is within or before the video frame's end:
      ``audio_start_j <= video_end_i + lookahead``.
    * ``v2a`` ``(B, T_a, T_v)`` — audio query, video key (the transpose with the
      same lookahead): ``video_start_j <= audio_end_i + lookahead``.

    Using ``[start, end)`` bounds (not frame centers) avoids a video frame
    missing audio frames that lie within its own span. ``lookahead_sec=0`` is
    strict causal (paper-faithful: "conditions available up to frame i"); a small
    positive lookahead lets a video frame peek slightly into future audio (useful
    for lip-sync on a bidirectionally-trained model).

    A query row whose keys are *all* in the future (e.g. the video sink at
    t~0 once the window's earliest audio starts later) would otherwise become
    an all-zero row; through the additive log-bias channel that degenerates to
    *uniform* attention over every key (worse than any causal choice). Such
    rows fall back to the single earliest key — the minimum-leak option.
    """
    # temporal [start, end) per token, in seconds (dim 0 of the position grid).
    v_start = video_positions[:, 0, :, 0]  # (B, T_v)
    v_end = video_positions[:, 0, :, 1]  # (B, T_v)
    a_start = audio_positions[:, 0, :, 0]  # (B, T_a)
    a_end = audio_positions[:, 0, :, 1]  # (B, T_a)
    # a2v: query=video (T_v), key=audio (T_a). [i, j] = a_start[j] <= v_end[i] + la.
    a2v = (a_start[:, None, :] <= v_end[:, :, None] + lookahead_sec).to(torch.float32)
    # v2a: query=audio (T_a), key=video (T_v). [i, j] = v_start[j] <= a_end[i] + la.
    v2a = (v_start[:, None, :] <= a_end[:, :, None] + lookahead_sec).to(torch.float32)
    a2v = _fallback_empty_rows_to_earliest_key(a2v, a_start)
    v2a = _fallback_empty_rows_to_earliest_key(v2a, v_start)
    return a2v, v2a


def _fallback_empty_rows_to_earliest_key(mask: torch.Tensor, key_start: torch.Tensor) -> torch.Tensor:
    """Open the earliest key (min start time) for query rows with no visible key.

    ``mask`` is ``(B, T_q, T_k)`` in ``[0, 1]``; ``key_start`` is ``(B, T_k)``
    start times in seconds. An all-zero row turns into uniform attention after
    the log-bias conversion (all logits get the same ``finfo.min`` offset), so
    it must never reach the model; attending only the earliest key is the
    smallest possible causality leak.
    """
    empty = mask.sum(dim=-1) == 0  # (B, T_q)
    if not bool(empty.any()):
        return mask
    earliest = key_start.argmin(dim=-1)  # (B,)
    fallback = torch.zeros_like(mask)
    fallback.scatter_(-1, earliest.view(-1, 1, 1).expand(-1, mask.shape[1], 1), 1.0)
    return torch.where(empty.unsqueeze(-1), fallback, mask)


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


def _build_window_state(
    *,
    video_tools: VideoLatentTools,
    sink_tokens: torch.Tensor | None,
    history: list[ChunkSnapshots],
    current_frames: int,
    tokens_per_frame: int,
    noiser: Noiser,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool = True,
) -> tuple[LatentState, tuple[int, int] | None, list[tuple[int, int]], tuple[int, int]]:
    """Assemble the sliding-window video LatentState for one AR step.

    Layout (token order): ``[sink (optional, 1 frame) | history chunks ... |
    current]``. Sink + history: ``denoise_mask = 0`` (frozen; the denoiser
    injects the TwinCache snapshot per step). Current: ``denoise_mask = 1``
    (denoised), initialised to pure noise by the noiser. Positions are
    window-relative (built by ``VideoLatentTools.create_initial_state`` on a
    window-sized shape -> the paper's RoPE repositioning).

    ``sink_tokens=None`` builds the sink-less ``[history | current]`` layout
    (joint TI2V after the bootstrap: the image lives inside the pinned
    ``[image | chunk 1]`` anchor). ``causal=False`` skips the block-causal
    attention mask (bidirectional attention) — used by the joint TI2V
    bootstrap, which denoises its first chunk with standard ti2v semantics.
    """
    has_sink = sink_tokens is not None
    hist_frames = sum(s.frames for s in history)
    window_frames = (1 if has_sink else 0) + hist_frames + current_frames
    h_lat = video_tools.target_shape.height
    w_lat = video_tools.target_shape.width
    channels = video_tools.target_shape.channels

    # Fresh window-shaped tools so positions/patchify match this window exactly.
    window_shape = VideoLatentShape(batch=1, channels=channels, frames=window_frames, height=h_lat, width=w_lat)
    window_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), window_shape, video_tools.fps)
    state = window_tools.create_initial_state(device=device, dtype=dtype)  # zeros latent/clean, ones mask, positions

    sink_range = (0, tokens_per_frame) if has_sink else None
    cursor = tokens_per_frame if has_sink else 0
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

    if causal:
        # Block-causal attention mask over the window's frame indices.
        frame_indices = torch.arange(window_frames, device=device).repeat_interleave(tokens_per_frame)
        state = replace(state, attention_mask=block_causal_attention_mask(frame_indices))
    # causal=False leaves attention_mask unset -> bidirectional (bootstrap).
    return state, sink_range, history_ranges, current_range


# ---------------------------------------------------------------------------
# Shared window-position / RoPE helpers (used by the M2 joint cached path).
# ---------------------------------------------------------------------------


def _build_window_positions(
    video_tools: VideoLatentTools,
    hist_frames: int,
    current_frames: int,
    device: torch.device,
    sink_frames: int = 1,
) -> tuple[torch.Tensor, int, int, int]:
    """Full-window positions ``[sink? | first+history | current]`` (window-relative).

    ``hist_frames`` counts all cached frames (anchor — image + first chunk —
    plus rolling history). ``sink_frames=0`` builds the sink-less layout used
    after the joint TI2V bootstrap (the image lives inside the anchor).
    Returns ``(positions, sink_tokens, hist_tokens, current_tokens)`` where
    the positions tensor is ``(1, 3, T, 2)`` over the whole window
    (frame-major), so that the window RoPE stays in the trained range (RoPE
    repositioning).
    """
    h_lat = video_tools.target_shape.height
    w_lat = video_tools.target_shape.width
    tokens_per_frame = h_lat * w_lat
    window_frames = sink_frames + hist_frames + current_frames
    channels = video_tools.target_shape.channels
    window_shape = VideoLatentShape(batch=1, channels=channels, frames=window_frames, height=h_lat, width=w_lat)
    window_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), window_shape, video_tools.fps)
    state = window_tools.create_initial_state(device=device, dtype=torch.float32)
    return (
        state.positions,
        sink_frames * tokens_per_frame,
        hist_frames * tokens_per_frame,
        current_frames * tokens_per_frame,
    )


def _window_pe(positions: torch.Tensor, transformer, dtype: torch.dtype, *, audio: bool = False):
    """Build the full-window RoPE (cos, sin) via the model's own preprocessor helper.

    ``audio=False`` (default) uses the video args preprocessor; ``audio=True``
    uses the audio args preprocessor (different ``max_pos`` / grid settings), so
    the joint streaming TI2V path can build window-relative audio RoPE the same
    way it builds video RoPE.
    """
    velocity = transformer.x0.velocity_model
    preprocessor = velocity.audio_args_preprocessor if audio else velocity.video_args_preprocessor
    simple = preprocessor.simple_preprocessor
    return simple._prepare_positional_embeddings(
        positions=positions,
        inner_dim=simple.inner_dim,
        max_pos=simple.max_pos,
        use_middle_indices_grid=simple.use_middle_indices_grid,
        num_attention_heads=simple.num_attention_heads,
        x_dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Joint streaming TI2V drivers — generate BOTH video and audio causally in
# lockstep. TI2V has no audio input: both modalities are generated. Each AR
# video chunk also produces the time-aligned audio latent frames, and audio
# keeps its own sliding-window TwinCache history (no sink — audio has no image
# conditioning — but a pinned first chunk, the audio half of Vidu S1 §2.3.1's
# video-audio reference) so per-step memory stays O(window) for *both*
# modalities. Built on the shared primitives above (block-causal mask,
# cross-causal mask, ChunkSnapshots, _build_window_state, euler_denoising_loop).
# ---------------------------------------------------------------------------


def _audio_chunk_frame_count(frames_through_chunk: int, audio_generated_before: int, fps: float) -> int:
    """Latent audio frames to generate for the video chunk ending at ``frames_through_chunk``.

    Cumulative tiling: the chunk's video end maps to an audio-frame target, and
    this chunk fills the gap between what was already generated and that target.
    Summed across chunks this covers the full audio timeline exactly (modulo
    rounding, which the caller conforms to the exact audio shape at the end).
    """
    _, end_sec = _video_latent_frame_bounds_sec(frames_through_chunk, fps)
    target = int(round(end_sec * AUDIO_LATENT_FRAMES_PER_SECOND))
    return max(1, target - audio_generated_before)


def _audio_window_alignment(
    *,
    audio_generated_before: int,
    audio_hist_frames: int,
    video_abs_current_frame: int,
    video_rel_current_frame: int,
    fps: float,
) -> tuple[int, float]:
    """Align a history-less audio window's clock with the video window's clock.

    Used by the ``image_cond`` ablation (rotating sink, ``audio_hist_frames=0``
    — the only remaining caller). The joint M1/M2 paths instead pin the audio
    first chunk at the window head, which makes the audio window a fresh
    window-relative grid — the same construction as the video window's — and
    the two fresh grids already share a clock.

    The video window keeps the sink pinned at its head, so after eviction its
    rolling/current section sits at *compressed* (window-relative) times:
    ``rel = abs - shift`` with ``shift = start(video_abs_current_frame)
    - start(video_rel_current_frame)``. Without a pinned audio anchor the
    audio positions must be rebuilt starting at the audio rolling window's
    absolute frame and shifted back by that same compression span, or the
    time-causal AV cross-attention mask and the shared-seconds cross-attn
    RoPE break (audio current would see video current as *future*).

    Returns ``(abs_start_frame, time_shift_sec)``: build the audio window grid
    starting at absolute audio latent frame ``abs_start_frame`` (the
    ``AudioPatchifier.shift`` parameter) and subtract ``time_shift_sec`` from
    the resulting timestamps, landing audio on the same compressed clock as
    the video window's rolling/current section.
    """
    abs_start_frame = audio_generated_before - audio_hist_frames
    v_abs_start, _ = _video_latent_frame_bounds_sec(video_abs_current_frame, fps)
    v_rel_start, _ = _video_latent_frame_bounds_sec(video_rel_current_frame, fps)
    return abs_start_frame, v_abs_start - v_rel_start


def _patchify_audio_frame_latent(unpatchified: torch.Tensor, patchifier: AudioPatchifier) -> torch.Tensor:
    """``(1, C, T, mel) -> (1, T, C*mel)`` patchified audio tokens (1 token/frame)."""
    return patchifier.patchify(unpatchified)


def _unpatchify_audio_tokens(
    tokens: torch.Tensor, frames: int, channels: int, mel_bins: int, patchifier: AudioPatchifier
) -> torch.Tensor:
    shape = AudioLatentShape(batch=1, channels=channels, frames=frames, mel_bins=mel_bins)
    return patchifier.unpatchify(tokens, output_shape=shape)


def _conform_audio_latent(latent: torch.Tensor, expected_frames: int) -> torch.Tensor:
    """Trim or zero-pad an audio latent ``(1, C, T, mel)`` to ``expected_frames`` on dim 2."""
    actual = latent.shape[2]
    if actual > expected_frames:
        return latent[:, :, :expected_frames, :]
    if actual < expected_frames:
        shape = list(latent.shape)
        shape[2] = expected_frames - actual
        pad = torch.zeros(shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([latent, pad], dim=2)
    return latent


def _build_audio_window_state(
    *,
    audio_tools_full: AudioLatentTools,
    history: list[ChunkSnapshots],
    current_frames: int,
    noiser: Noiser,
    device: torch.device,
    dtype: torch.dtype,
    abs_start_frame: int = 0,
    time_shift_sec: float = 0.0,
    causal: bool = True,
) -> tuple[LatentState, list[tuple[int, int]], tuple[int, int]]:
    """Assemble the audio sliding-window LatentState for one AR step.

    Layout (token order): ``[first chunk (pinned) | history chunks ... | current]``
    (no sink — audio has no image conditioning; the first chunk is the audio
    half of Vidu S1 §2.3.1's persistent reference). History: ``denoise_mask = 0``
    (frozen; the denoiser injects the TwinCache snapshot per step). Current:
    ``denoise_mask = 1`` (denoised), initialised to pure noise by the noiser.
    The joint callers pass the first chunk inside ``history`` and use the
    default fresh window-relative grid (``abs_start_frame=0``,
    ``time_shift_sec=0``) — the same construction as the video window, so both
    fresh grids share one clock. The rotating-sink ``image_cond`` path instead
    passes ``abs_start_frame``/``time_shift_sec`` from
    :func:`_audio_window_alignment`. The block-causal attention mask is built
    over per-token audio frame indices (1 token per audio frame).
    """
    channels = audio_tools_full.target_shape.channels
    mel_bins = audio_tools_full.target_shape.mel_bins
    hist_frames = sum(s.frames for s in history)
    window_frames = hist_frames + current_frames

    window_shape = AudioLatentShape(batch=1, channels=channels, frames=window_frames, mel_bins=mel_bins)
    window_tools = AudioLatentTools(AudioPatchifier(patch_size=1, shift=abs_start_frame), window_shape)
    state = window_tools.create_initial_state(device=device, dtype=dtype)
    if time_shift_sec:
        state = replace(state, positions=state.positions - time_shift_sec)

    # 1 token per audio frame: token ranges == frame ranges.
    history_ranges: list[tuple[int, int]] = []
    cursor = 0
    for snap in history:
        history_ranges.append((cursor, cursor + snap.frames))
        cursor += snap.frames
    current_range = (cursor, cursor + current_frames)

    # denoise_mask: 0 on history (frozen), 1 on current (denoised).
    mask = torch.zeros_like(state.denoise_mask)
    c0, c1 = current_range
    mask[:, c0:c1] = 1.0
    state = replace(state, denoise_mask=mask)

    # Noise only the current chunk (GaussianNoiser respects the mask).
    state = noiser(state, noise_scale=1.0)

    # Block-causal attention mask over the window's audio frame indices
    # (skipped when causal=False -> bidirectional, joint TI2V bootstrap).
    if causal:
        frame_indices = torch.arange(window_frames, device=device)
        state = replace(state, attention_mask=block_causal_attention_mask(frame_indices))
    return state, history_ranges, current_range


class JointStreamingTwinDenoiser:
    """SimpleDenoiser + per-step TwinCache history injection for BOTH modalities.

    Each call, before building the Modalities, writes the TwinCache-selected
    snapshot (noisy at intermediate steps, clean at the final step) into the
    history token ranges of both ``video_state`` and ``audio_state`` (and the
    video sink range) — into ``latent`` *and* ``clean_latent`` in place. Setting
    ``latent == clean_latent`` on those (mask=0) tokens makes the Euler velocity
    ``(latent - clean)/sigma`` zero, so ``_step_state`` / ``post_process_latent``
    / ``stepper.step`` leave them frozen at the injected snapshot (the TwinCache
    behaviour). The current chunks are left untouched and denoise normally.

    At the configured mid step it also snapshots each current chunk's evolving
    latent as that chunk's *noisy* TwinCache entry.
    """

    def __init__(
        self,
        v_context: torch.Tensor,
        a_context: torch.Tensor,
        # video: [image | current] bootstrap window, then anchor + history + current
        video_sink_tokens: torch.Tensor | None,
        video_sink_range: tuple[int, int] | None,
        video_history: list[ChunkSnapshots],
        video_history_ranges: list[tuple[int, int]],
        video_current_range: tuple[int, int],
        # audio: first chunk (pinned) + history chunks + current (no sink)
        audio_history: list[ChunkSnapshots],
        audio_history_ranges: list[tuple[int, int]],
        audio_current_range: tuple[int, int],
        sigma_mid_step: int,
        num_steps: int,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_sink_tokens = video_sink_tokens
        self.video_sink_range = video_sink_range
        self.video_history = video_history
        self.video_history_ranges = video_history_ranges
        self.video_current_range = video_current_range
        self.audio_history = audio_history
        self.audio_history_ranges = audio_history_ranges
        self.audio_current_range = audio_current_range
        self.sigma_mid_step = sigma_mid_step
        self.num_steps = num_steps
        self.noisy_capture_video: torch.Tensor | None = None
        self.noisy_capture_audio: torch.Tensor | None = None

    @staticmethod
    def _inject(
        state: LatentState,
        is_final: bool,
        sink_tokens: torch.Tensor | None,
        sink_range: tuple[int, int] | None,
        history: list[ChunkSnapshots],
        history_ranges: list[tuple[int, int]],
    ) -> None:
        """Overwrite sink + history latent & clean_latent with the selected snapshot."""
        lat = state.latent
        clean = state.clean_latent
        if sink_tokens is not None and sink_range is not None:
            s0, s1 = sink_range
            lat[:, s0:s1, :] = sink_tokens
            clean[:, s0:s1, :] = sink_tokens
        for rng, snap in zip(history_ranges, history, strict=True):
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
        if video_state is None or audio_state is None:
            raise ValueError("JointStreamingTwinDenoiser requires both a video and an audio state")

        is_final = step_index == self.num_steps - 1

        # Capture the current chunks' noisy latent at the mid step (before any
        # mutation; current tokens are never mutated here, so ordering is safe).
        if step_index == self.sigma_mid_step:
            if self.noisy_capture_video is None:
                vc0, vc1 = self.video_current_range
                self.noisy_capture_video = video_state.latent[:, vc0:vc1, :].clone()
            if self.noisy_capture_audio is None:
                ac0, ac1 = self.audio_current_range
                self.noisy_capture_audio = audio_state.latent[:, ac0:ac1, :].clone()

        self._inject(
            video_state, is_final,
            self.video_sink_tokens, self.video_sink_range,
            self.video_history, self.video_history_ranges,
        )
        self._inject(
            audio_state, is_final,
            None, None,  # audio has no sink
            self.audio_history, self.audio_history_ranges,
        )

        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma)
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )


def streaming_generate_joint(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    window_chunks: int,
    video_tools_full: VideoLatentTools,
    audio_tools_full: AudioLatentTools,
    sink_latent_unpatchified: torch.Tensor,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    stepper: EulerDiffusionStep,
    transformer: X0Model,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    causal_cross_attn: bool = True,
    cross_attn_lookahead_sec: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive streaming TI2V generation (returns full video + audio latents).

    Both video and audio are generated (TI2V has no audio input). For each AR
    video chunk of up to ``chunk_frames`` latent frames: compute the
    time-aligned audio frame count for this chunk; assemble the video window —
    ``[image | current]`` for chunk 1, ``[anchor | history | current]``
    afterwards (reusing :func:`_build_window_state`) — and the audio window
    ``[first | history | current]`` (:func:`_build_audio_window_state`), both
    with block-causal masks and window-relative positions; set time-causal AV
    cross-attention masks; run
    :func:`euler_denoising_loop` with
    :class:`JointStreamingTwinDenoiser` (which swaps the history snapshot
    noisy<->clean per step for *both* modalities and captures each current
    chunk's noisy mid-snapshot); finalize both chunks' snapshots; splice both
    clean latents into the full video / audio latents.

    Chunk 1 is a standard **bidirectional ti2v bootstrap**: the reference image
    replaces latent frame 0 (frozen) and denoises with full attention (no
    causal self/cross masks), matching the base model's training. Its output,
    ``[image | chunk 1]``, is pinned as the anchor — Vidu S1 §2.3.1's
    persistent reference (first frame + first generated video-audio state) in
    one entry, always clean, never evicted. Later chunks stream causally:
    anchor + rolling FIFO capped at ``window_chunks`` (video), persistent first
    chunk + rolling FIFO (audio — no image conditioning, so no image frame).
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
    audio_first: ChunkSnapshots | None = None  # audio persistent first chunk
    rolling_video: deque[ChunkSnapshots] = deque(maxlen=window_chunks)
    rolling_audio: deque[ChunkSnapshots] = deque(maxlen=window_chunks)

    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames
    frames_generated_before = 0
    audio_generated_before = 0

    for i in range(num_chunks):
        current_video_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
        frames_through_chunk = frames_generated_before + current_video_frames
        current_audio_frames = _audio_chunk_frame_count(frames_through_chunk, audio_generated_before, fps)

        # Chunk 1 is a standard bidirectional ti2v bootstrap: the reference
        # image replaces latent frame 0 (frozen) and the chunk denoises with
        # full attention — no causal self/cross masks. Its output, [image |
        # chunk 1], becomes the pinned anchor of every later window; later
        # chunks stream causally exactly as before.
        is_bootstrap = first_ref is None

        # --- video window: [image | current] for the bootstrap, then
        # [anchor | rolling | current] (no separate sink — the image lives in
        # the anchor's frame 0) ---
        video_history = ([first_ref] if first_ref is not None else []) + list(rolling_video)
        video_state, sink_range, video_history_ranges, video_current_range = _build_window_state(
            video_tools=video_tools_full,
            sink_tokens=sink_tokens if is_bootstrap else None,
            history=video_history,
            current_frames=current_video_frames,
            tokens_per_frame=tokens_per_frame,
            noiser=noiser,
            device=device,
            dtype=dtype,
            causal=not is_bootstrap,
        )

        # --- audio window [first | history | current], clock-aligned to the
        # video window. With the audio first chunk pinned at the head (like the
        # video window's), the window is a fresh window-relative grid — the
        # same construction as the video window — and needs no clock
        # realignment: both rolling sections are compressed by the identical
        # evicted span, so the fresh grids stay on a shared clock. ---
        audio_history = ([audio_first] if audio_first is not None else []) + list(rolling_audio)
        audio_state, audio_history_ranges, audio_current_range = _build_audio_window_state(
            audio_tools_full=audio_tools_full,
            history=audio_history,
            current_frames=current_audio_frames,
            noiser=noiser,
            device=device,
            dtype=dtype,
            causal=not is_bootstrap,
        )

        # --- time-causal AV cross-attention masks (skipped on the
        # bidirectional bootstrap) ---
        if causal_cross_attn and not is_bootstrap:
            a2v_mask, v2a_mask = cross_causal_attention_mask(
                video_state.positions, audio_state.positions, cross_attn_lookahead_sec
            )
            video_state = replace(video_state, cross_attention_mask=a2v_mask)
            audio_state = replace(audio_state, cross_attention_mask=v2a_mask)

        denoiser = JointStreamingTwinDenoiser(
            v_context=v_context,
            a_context=a_context,
            video_sink_tokens=sink_tokens if is_bootstrap else None,
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
            "Joint streaming AR chunk %d/%d (video=%d audio=%d frames, v-hist=%d a-hist=%d)",
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
            # Anchor = [image | chunk 1]: the frozen image frame plus this
            # chunk's clean frames, pinned as one history entry (always clean).
            anchor_tokens = video_state.latent[:, 0 : tokens_per_frame * (1 + current_video_frames), :].clone()
            first_ref = ChunkSnapshots(
                tokens_noisy=anchor_tokens, tokens_clean=anchor_tokens, frames=1 + current_video_frames
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

        # --- finalize audio chunk (chunk 1 joins the persistent reference,
        # mirroring the video first chunk: its noisy slot holds the clean
        # snapshot, so it always reads clean at every step) ---
        ac0, ac1 = audio_current_range
        audio_clean_tokens = audio_state.latent[:, ac0:ac1, :].clone()
        if audio_first is None:
            audio_first = ChunkSnapshots(
                tokens_noisy=audio_clean_tokens, tokens_clean=audio_clean_tokens, frames=current_audio_frames
            )
        else:
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

    full_audio_latent = _conform_audio_latent(full_audio_latent, total_audio_frames)
    return full_video_latent, full_audio_latent


def streaming_generate_joint_image_cond(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    video_tools_full: VideoLatentTools,
    audio_tools_full: AudioLatentTools,
    sink_latent_unpatchified: torch.Tensor,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    stepper: EulerDiffusionStep,
    transformer: X0Model,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    causal_cross_attn: bool = True,
    cross_attn_lookahead_sec: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ablation C: AR image-conditioning chain (no KV cache, no attention history).

    Each AR chunk is a fresh TI2V denoise conditioned on the *previous chunk's
    last generated frame* as the image reference (a rotating sink). The window
    per chunk is just ``[sink | current]`` (video) / ``[current]`` (audio) — no
    persistent first chunk, no rolling history. Cross-chunk context is carried
    solely by the one rotating image-reference frame; there is no self-attention
    history and no KV cache (full recompute within the tiny per-chunk window).

    Chunk 0's sink is the user-supplied reference image (``sink_latent_unpatchified``,
    same as M1's sink). After each chunk, the sink is rotated to that chunk's
    last generated latent frame. Audio is generated jointly with each video
    chunk and has no cross-chunk history either.

    Structurally a stripped-down :func:`streaming_generate_joint` (M1) with
    ``history=[]`` on every chunk and a rotating sink. Reuses
    :func:`_build_window_state` / :class:`JointStreamingTwinDenoiser` /
    :func:`_build_audio_window_state` / :func:`_audio_window_alignment`.
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

        # --- video window [sink | current] (no history) ---
        video_state, sink_range, video_history_ranges, video_current_range = _build_window_state(
            video_tools=video_tools_full,
            sink_tokens=sink_tokens,
            history=[],  # no attention history across chunks
            current_frames=current_video_frames,
            tokens_per_frame=tokens_per_frame,
            noiser=noiser,
            device=device,
            dtype=dtype,
        )

        # --- audio window [current] (no history), clock-aligned to video ---
        a_abs_start, a_time_shift = _audio_window_alignment(
            audio_generated_before=audio_generated_before,
            audio_hist_frames=0,
            video_abs_current_frame=1 + frames_generated_before,
            video_rel_current_frame=1,  # no video history
            fps=fps,
        )
        audio_state, audio_history_ranges, audio_current_range = _build_audio_window_state(
            audio_tools_full=audio_tools_full,
            history=[],
            current_frames=current_audio_frames,
            noiser=noiser,
            device=device,
            dtype=dtype,
            abs_start_frame=a_abs_start,
            time_shift_sec=a_time_shift,
        )

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
            video_history=[],  # empty -> _inject is a no-op on history
            video_history_ranges=video_history_ranges,
            video_current_range=video_current_range,
            audio_history=[],
            audio_history_ranges=audio_history_ranges,
            audio_current_range=audio_current_range,
            sigma_mid_step=sigma_mid_step,
            num_steps=num_steps,
        )

        logger.info(
            "Joint streaming AR chunk %d/%d (image-cond; video=%d audio=%d frames)",
            i + 1, num_chunks, current_video_frames, current_audio_frames,
        )

        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

        # --- finalize video chunk + rotate sink to this chunk's last frame ---
        vc0, vc1 = video_current_range
        video_clean_tokens = video_state.latent[:, vc0:vc1, :].clone()
        video_clean_unpatchified = _unpatchify_tokens(
            video_clean_tokens, current_video_frames, h_lat, w_lat, channels, patchifier
        )
        f0 = 1 + frames_generated_before
        full_video_latent[:, :, f0 : f0 + current_video_frames, :, :] = video_clean_unpatchified
        # Rotate the sink: the next chunk's image reference is this chunk's
        # last generated latent frame (one latent frame's worth of tokens).
        sink_tokens = video_clean_tokens[:, -tokens_per_frame:, :].clone()

        # --- finalize audio chunk ---
        ac0, ac1 = audio_current_range
        audio_clean_tokens = audio_state.latent[:, ac0:ac1, :].clone()
        audio_clean_unpatchified = _unpatchify_audio_tokens(
            audio_clean_tokens, current_audio_frames, audio_channels, audio_mel, audio_patchifier
        )
        a0 = audio_generated_before
        full_audio_latent[:, :, a0 : a0 + current_audio_frames, :] = audio_clean_unpatchified

        frames_generated_before += current_video_frames
        audio_generated_before += current_audio_frames

    full_audio_latent = _conform_audio_latent(full_audio_latent, total_audio_frames)
    return full_video_latent, full_audio_latent


def _build_audio_window_positions(
    audio_tools: AudioLatentTools,
    hist_frames: int,
    current_frames: int,
    device: torch.device,
    abs_start_frame: int = 0,
    time_shift_sec: float = 0.0,
) -> tuple[torch.Tensor, int, int]:
    """Full-window audio positions ``[first | history | current]`` (no sink).

    Audio has 1 token per frame, so token counts == frame counts. Returns
    ``(positions, hist_tokens, current_tokens)`` with ``positions`` shaped
    ``(1, 1, T, 2)`` over the whole window. The joint M2 caller passes the
    pinned first chunk inside ``hist_frames`` and uses the default fresh
    window-relative grid — the same construction as the video window's, so
    both grids share one clock, stay in the trained positional range (RoPE
    repositioning) and keep the time-causal AV cross-attn mask /
    shared-seconds cross RoPE aligned. ``abs_start_frame``/``time_shift_sec``
    (from :func:`_audio_window_alignment`) remain for the rotating-sink
    ``image_cond`` path.
    """
    channels = audio_tools.target_shape.channels
    mel_bins = audio_tools.target_shape.mel_bins
    window_frames = hist_frames + current_frames
    window_shape = AudioLatentShape(batch=1, channels=channels, frames=window_frames, mel_bins=mel_bins)
    window_tools = AudioLatentTools(AudioPatchifier(patch_size=1, shift=abs_start_frame), window_shape)
    state = window_tools.create_initial_state(device=device, dtype=torch.float32)
    positions = state.positions - time_shift_sec if time_shift_sec else state.positions
    return positions, hist_frames, current_frames


def streaming_generate_joint_cached(  # noqa: PLR0913, PLR0915
    *,
    sigmas: torch.Tensor,
    num_generated_latent_frames: int,
    chunk_frames: int,
    window_chunks: int,
    video_tools_full: VideoLatentTools,
    audio_tools_full: AudioLatentTools,
    sink_latent_unpatchified: torch.Tensor,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    stepper: EulerDiffusionStep,
    transformer,  # X0Model (will be wrapped)
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    causal_cross_attn: bool = True,
    cross_attn_lookahead_sec: float = 0.0,
    strategy: str = "twin",
) -> tuple[torch.Tensor, torch.Tensor]:
    """M2: KV-cache + RoPE repositioning joint streaming TI2V generation.

    The KV-cache analogue of :func:`streaming_generate_joint`: caches audio
    self-attention too (audio is generated, so its self-attn must be cached for
    O(window) memory). Wraps ``transformer`` with a :class:`CausalStreamingModel`
    configured ``cache_audio=True`` and ``video_sink_tokens=0``: chunk 1 is a
    **bidirectional ti2v bootstrap** (the reference image replaces latent frame
    0, full attention — matching the base model's training), and its whole
    ``[image | chunk 1]`` output is committed as the pinned anchor in the video
    cache's permanent first slot. Later chunks carry only ``[current]``
    (video/audio) in the modalities (history K/V come from the caches); the
    driver builds the full-window RoPE ``window_pe`` and the block-causal
    ``query_mask`` (history query rows removed, structured ``BlockCausalMask``)
    for *both* modalities.

    The TwinCache noisy/clean snapshot selection and the permanent first-chunk
    slots (both modalities — Vidu S1 §2.3.1's persistent reference is the
    first generated video-audio state) follow Vidu S1 §2.3.1. The *clean* K/V
    snapshot is captured by one extra forward on the finalized (post-final-
    Euler-step) latents at sigma 0 per chunk — the paper's "clean cache
    obtained after the final denoising step" — rather than from the final
    step's still-noisy input.

    Strategies (ablation):
      * ``"twin"`` (default): intermediate steps read the history ``noisy``
        snapshot (captured at the mid step), the final step reads ``clean``.
      * ``"clean"`` (ablation A): every step reads the history ``clean``
        snapshot (pure-clean history).
      * ``"noisy_steps"`` (ablation B): step ``t`` reads the history's own
        step-``t`` noisy snapshot (noise-level matched). No clean snapshot;
        no extra sigma-0 forward.

    NOTE: conceptual/unvalidated — extends an already-untested M2 path to a
    second modality with a different (sink-less) layout. Run
    ``tests/test_streaming_joint_parity.py`` in a GPU env before trusting; until
    it passes, prefer :func:`streaming_generate_joint` (M1).
    """
    from ltx_core.model.transformer import CausalStreamingModel
    from ltx_pipelines.utils.helpers import post_process_latent

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
        video_sink_tokens=0,  # first committed chunk = [image | chunk 1] anchor
        cache_audio=True,
        audio_tokens_per_frame=1,
        strategy=strategy,
        num_steps=num_steps,
    )

    # Video layout bookkeeping (frame counts; K/V live in the caches).
    first_frames = 0
    rolling_frames: deque[int] = deque(maxlen=window_chunks)
    # Audio layout bookkeeping (frame counts == token counts; first chunk
    # pinned like video's, later chunks pure FIFO).
    first_audio_frames = 0
    rolling_audio_frames: deque[int] = deque(maxlen=window_chunks)
    frames_generated_before = 0
    audio_generated_before = 0
    num_chunks = (num_generated_latent_frames + chunk_frames - 1) // chunk_frames

    try:
        for i in range(num_chunks):
            current_video_frames = min(chunk_frames, num_generated_latent_frames - frames_generated_before)
            frames_through_chunk = frames_generated_before + current_video_frames
            current_audio_frames = _audio_chunk_frame_count(frames_through_chunk, audio_generated_before, fps)

            # Chunk 1 = bidirectional ti2v bootstrap ([image | current], no
            # causal masks); later chunks: [anchor | rolling | current], the
            # anchor = [image | chunk 1] pinned in the caches' _first slots.
            is_bootstrap = first_frames == 0
            v_hist_frames = first_frames + sum(rolling_frames)
            v_sink_t = tokens_per_frame if is_bootstrap else 0
            v_hist_t = v_hist_frames * tokens_per_frame
            v_cur_t = current_video_frames * tokens_per_frame
            full_positions, _, _, _ = _build_window_positions(
                video_tools_full, v_hist_frames, current_video_frames, device,
                sink_frames=1 if is_bootstrap else 0,
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
            if is_bootstrap:
                # Bidirectional bootstrap: the whole [image | current] window is
                # queried, full attention (no causal query mask).
                v_query_rows = torch.arange(0, v_sink_t + v_cur_t, device=device)
                v_query_mask = None
            else:
                # Structured block-causal mask (history query rows removed):
                # queries are the current chunk, keys the full window. Served by
                # unmasked prefix attention calls (FlashAttention-capable).
                v_query_rows = torch.arange(v_hist_t, v_hist_t + v_cur_t, device=device)
                v_query_mask = block_causal_attention_mask(
                    v_full_frame_indices[v_query_rows], v_full_frame_indices
                )

            # --- audio window positions/mask [first | history | current] (no
            # sink). The pinned audio first chunk puts the grid's head next to
            # the video window's persistent reference, so the window is a fresh
            # window-relative grid — the same construction as the video window
            # — and needs no clock realignment: both rolling sections compress
            # by the identical evicted span. ---
            a_hist_t = first_audio_frames + sum(rolling_audio_frames)
            a_cur_t = current_audio_frames  # 1 token per audio frame
            a_full_positions, _, _ = _build_audio_window_positions(
                audio_tools_full, a_hist_t, current_audio_frames, device
            )
            a_window_pe = _window_pe(a_full_positions, wrapper, dtype, audio=True)
            a_full_tokens = a_full_positions.shape[2]
            if a_full_tokens != a_hist_t + a_cur_t:
                raise RuntimeError(
                    "Audio window positions out of sync: "
                    f"positions={a_full_tokens}, hist+cur={a_hist_t + a_cur_t}."
                )
            a_full_frame_indices = torch.arange(a_full_tokens, device=device)  # 1 token/frame
            a_current_rows = torch.arange(a_hist_t, a_hist_t + a_cur_t, device=device)
            a_query_mask = (
                None
                if is_bootstrap
                else block_causal_attention_mask(a_full_frame_indices[a_current_rows], a_full_frame_indices)
            )

            # --- video modality: [image | current] on the bootstrap (image
            # frozen, current noised), [current] afterwards (history K/V come
            # from the caches). Draw noise over the FULL video window and slice
            # the current rows so the generator consumption matches M1 (same
            # seed => same chunk noise). ---
            v_window_noise = torch.randn(
                (1, v_sink_t + v_hist_t + v_cur_t, channels), device=device, dtype=dtype, generator=noiser.generator
            )
            v_cur_noise = v_window_noise[:, v_sink_t + v_hist_t :, :]
            if is_bootstrap:
                v_mod_latent = torch.cat([sink_tokens, v_cur_noise], dim=1)
                v_mod_clean = torch.zeros_like(v_mod_latent)
                v_mod_clean[:, :v_sink_t] = sink_tokens
            else:
                v_mod_latent = v_cur_noise
                v_mod_clean = torch.zeros_like(v_mod_latent)
            v_mod_mask = torch.zeros((1, v_mod_latent.shape[1], 1), device=device, dtype=torch.float32)
            v_mod_mask[:, v_sink_t:] = 1.0
            v_mod_positions = full_positions[:, :, v_query_rows, :]
            video_state = LatentState(
                latent=v_mod_latent,
                denoise_mask=v_mod_mask,
                positions=v_mod_positions,
                clean_latent=v_mod_clean,
                attention_mask=None,  # cached attn1 uses query_mask via the cache
            )

            # --- audio modality: carries [current] only; all noised ---
            a_window_noise = torch.randn(
                (1, a_hist_t + a_cur_t, audio_token_dim), device=device, dtype=dtype, generator=noiser.generator
            )
            a_cur_noise = a_window_noise[:, a_hist_t:, :]
            a_mod_latent = a_cur_noise
            a_mod_clean = torch.zeros_like(a_mod_latent)
            a_mod_mask = torch.ones((1, a_mod_latent.shape[1], 1), device=device, dtype=torch.float32)
            a_mod_positions = a_full_positions[:, :, a_current_rows, :]
            audio_state = LatentState(
                latent=a_mod_latent,
                denoise_mask=a_mod_mask,
                positions=a_mod_positions,
                clean_latent=a_mod_clean,
                attention_mask=None,  # cached audio_attn1 uses query_mask via the cache
            )

            if causal_cross_attn and not is_bootstrap:
                # Cross-attn is NOT cached (only self-attn is), so the time-causal
                # mask applies normally through the modality; set once per chunk.
                # The bidirectional bootstrap skips it entirely.
                a2v_mask, v2a_mask = cross_causal_attention_mask(
                    video_state.positions, audio_state.positions, cross_attn_lookahead_sec
                )
                video_state = replace(video_state, cross_attention_mask=a2v_mask)
                audio_state = replace(audio_state, cross_attention_mask=v2a_mask)

            wrapper.prepare_chunk(
                window_pe=v_window_pe,
                query_mask=v_query_mask,
                hist_len=v_hist_t,
                audio_window_pe=a_window_pe,
                audio_query_mask=a_query_mask,
                audio_hist_len=a_hist_t,
            )

            logger.info(
                "Joint streaming AR chunk %d/%d (cached; video=%d audio=%d frames)",
                i + 1, num_chunks, current_video_frames, current_audio_frames,
            )

            # Inline per-step loop: toggle TwinCache snapshot mode, run the cached
            # model, post-process + euler step, capture K/V per strategy.
            for step_idx in range(num_steps):
                if strategy == "noisy_steps":
                    # B: read history's own step-t noisy; capture current step-t.
                    wrapper.set_mode("noisy", step_idx=step_idx)
                elif strategy == "clean":
                    # A: read history clean at every step.
                    wrapper.set_mode("clean")
                else:  # twin
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
                # B: no clean snapshot — the per-step noisy K/V are the history.
                # Finalized latent is whatever the last Euler step produced.
                wrapper.commit()
            else:
                # twin / clean: Clean-KV refresh (Vidu S1 §2.3.1: the clean cache
                # is "obtained AFTER the final denoising step"). The loop's last
                # forward saw the current chunk at sigma[-2] (still noisy), so its
                # K/V are not the clean snapshot. Run one extra forward on the
                # finalized latents at sigma 0 — the exact condition under which
                # M1 presents this chunk as clean history (timestep-0 conditioning
                # tokens) — and stash that as the "clean" TwinCache entry. Output
                # is discarded.
                wrapper.set_mode("clean")
                zero_sigma = torch.zeros_like(sigmas[0])
                pos_video = modality_from_latent_state(video_state, v_context, zero_sigma)
                pos_audio = modality_from_latent_state(audio_state, a_context, zero_sigma)
                wrapper(video=pos_video, audio=pos_audio, perturbations=None)
                wrapper.stash("clean")
                wrapper.commit()
            if first_frames == 0:
                first_frames = 1 + current_video_frames  # anchor = image + chunk 1
            else:
                rolling_frames.append(current_video_frames)
            if first_audio_frames == 0:
                first_audio_frames = current_audio_frames  # joined the persistent reference
            else:
                rolling_audio_frames.append(current_audio_frames)

            # Splice the finalized current latents into the full latents.
            v_clean_tokens = video_state.latent[:, v_sink_t : v_sink_t + v_cur_t, :].clone()
            v_clean_unpatchified = _unpatchify_tokens(
                v_clean_tokens, current_video_frames, h_lat, w_lat, channels, patchifier
            )
            f0 = 1 + frames_generated_before
            full_video_latent[:, :, f0 : f0 + current_video_frames, :, :] = v_clean_unpatchified

            a_clean_tokens = audio_state.latent.clone()  # [current] only
            a_clean_unpatchified = _unpatchify_audio_tokens(
                a_clean_tokens, current_audio_frames, audio_channels, audio_mel, audio_patchifier
            )
            a0 = audio_generated_before
            full_audio_latent[:, :, a0 : a0 + current_audio_frames, :] = a_clean_unpatchified

            frames_generated_before += current_video_frames
            audio_generated_before += current_audio_frames
    finally:
        # Drop the caches and restore the wrapped model's standard forward path.
        wrapper.detach()
    full_audio_latent = _conform_audio_latent(full_audio_latent, total_audio_frames)
    return full_video_latent, full_audio_latent
