"""Unit tests for the joint streaming TI2V helpers.

Checks (pure-tensor, no checkpoint needed):
  * audio chunk-frame tiling sums to the full audio length,
  * ``_build_audio_window_state`` token counts / mask shape / block-causal prefix
    structure / denoise-mask values,
  * the joint cross-causal mask shapes ``(B, T_v, T_a)`` / ``(B, T_a, T_v)``.

Needs torch + the ltx env (the helper imports pull in ltx_core). Run:

    uv run python packages/ltx-pipelines/tests/test_streaming_joint.py
"""

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.streaming import (
    ChunkSnapshots,
    _audio_chunk_frame_count,
    _audio_window_alignment,
    _build_audio_window_state,
    _video_latent_frame_bounds_sec,
    cross_causal_attention_mask,
)

FPS = 30.0
AUDIO_FPS = 25.0


def _audio_token_snap(frames: int, device: torch.device) -> ChunkSnapshots:
    """Fake ChunkSnapshots: 1 token per audio frame, 128-dim (8 channels * 16 mel)."""
    toks = torch.randn(1, frames, 128, device=device)
    return ChunkSnapshots(tokens_noisy=toks, tokens_clean=toks.clone(), frames=frames)


def _is_prefix_of_ones(row: torch.Tensor) -> bool:
    nz = (row == 0).nonzero(as_tuple=True)[0]
    oz = (row == 1).nonzero(as_tuple=True)[0]
    if len(nz) == 0 or len(oz) == 0:
        return True
    return nz.min().item() > oz.max().item()


def test_audio_chunk_tiling() -> None:
    """Per-chunk audio frames tile to (approx) the full audio length."""
    device = torch.device("cpu")
    num_pixel_frames = 41  # -> 6 latent frames -> 5 generated (1 chunk each)
    pixel_shape = VideoPixelShape(1, num_pixel_frames, 512, 768, FPS)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    num_generated = v_shape.frames - 1
    chunk_frames = 1

    total = 0
    before = 0
    g = 0
    while g < num_generated:
        c = min(chunk_frames, num_generated - g)
        frames_through = g + c
        n = _audio_chunk_frame_count(frames_through, before, FPS)
        assert n >= 1, f"audio chunk must generate >= 1 frame, got {n}"
        total += n
        before += n
        g += c

    expected = a_shape.frames
    assert abs(total - expected) <= 2, f"audio tiling {total} vs expected {expected}"
    print(f"[tiling] {num_generated} video frames -> {total} audio frames (expected ~{expected}) OK")


def test_audio_window_state() -> None:
    device = torch.device("cpu")
    a_shape = AudioLatentShape(1, 8, 10, 16)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
    history = [_audio_token_snap(2, device), _audio_token_snap(1, device)]
    current = 3
    noiser = GaussianNoiser(torch.Generator(device=device).manual_seed(0))

    state, hist_ranges, cur_range = _build_audio_window_state(
        audio_tools_full=audio_tools,
        history=history,
        current_frames=current,
        noiser=noiser,
        device=device,
        dtype=torch.float32,
    )

    window_frames = 2 + 1 + 3  # 6
    assert state.latent.shape == (1, window_frames, 128), f"latent {state.latent.shape}"
    assert state.denoise_mask.shape == (1, window_frames, 1), f"mask {state.denoise_mask.shape}"
    assert state.positions.shape == (1, 1, window_frames, 2), f"positions {state.positions.shape}"
    assert state.attention_mask.shape == (1, window_frames, window_frames), f"attn {state.attention_mask.shape}"

    # denoise_mask: 0 on history (3 frames, frozen), 1 on current (denoised).
    assert torch.all(state.denoise_mask[0, :3, 0] == 0), "history must be frozen (mask 0)"
    assert torch.all(state.denoise_mask[0, 3:, 0] == 1), "current must be denoised (mask 1)"

    # block-causal: each row is a prefix of ones (allow past, mask future).
    for i in range(window_frames):
        assert _is_prefix_of_ones(state.attention_mask[0, i]), (
            f"audio row {i} not a causal prefix: {state.attention_mask[0, i].tolist()}"
        )

    assert hist_ranges == [(0, 2), (2, 3)], f"history ranges {hist_ranges}"
    assert cur_range == (3, 6), f"current range {cur_range}"
    print("[audio-window] shapes + causal mask + denoise mask + ranges OK")


def test_joint_cross_mask_shapes() -> None:
    device = torch.device("cpu")
    tv = 4  # video window: sink(1) + 2 history + 1 current
    vpos = torch.zeros(1, 3, tv, 2)
    vstart = torch.arange(tv, dtype=torch.float32) * 8.0 / FPS
    vpos[:, 0, :, 0] = vstart
    vpos[:, 0, :, 1] = vstart + 8.0 / FPS

    ta = 5  # audio window: 3 history + 2 current
    apos = torch.zeros(1, 1, ta, 2)
    astart = torch.arange(ta, dtype=torch.float32) / AUDIO_FPS
    apos[:, 0, :, 0] = astart
    apos[:, 0, :, 1] = astart + 1.0 / AUDIO_FPS

    a2v, v2a = cross_causal_attention_mask(vpos, apos, lookahead_sec=0.0)
    assert a2v.shape == (1, tv, ta), f"a2v {a2v.shape}"
    assert v2a.shape == (1, ta, tv), f"v2a {v2a.shape}"
    assert torch.all((a2v == 0) | (a2v == 1)) and torch.all((v2a == 0) | (v2a == 1))
    print(f"[cross-mask] a2v={tuple(a2v.shape)} v2a={tuple(v2a.shape)} OK")


def test_audio_window_clock_alignment() -> None:
    """After eviction the audio window's clock must match the video window's
    compressed clock: the audio *current* chunk must be able to see the video
    *current* frame under the strict time-causal v2a mask (this is exactly what
    breaks without the alignment — audio would see video current as future)."""
    device = torch.device("cpu")
    fps = 25.0
    window_chunks, chunk_frames = 4, 1
    audio_per_chunk = 8  # fps=25 -> 8 audio latent frames per video latent frame
    k = 10  # steady-state AR chunk index (evictions well underway)

    frames_generated_before = (k - 1) * chunk_frames
    audio_generated_before = 0
    # Replay the cumulative audio tiling up to chunk k.
    for g in range(1, k):
        audio_generated_before += _audio_chunk_frame_count(g * chunk_frames, audio_generated_before, fps)

    v_hist_frames = chunk_frames * (1 + window_chunks)  # first + rolling
    a_hist_frames = audio_per_chunk * window_chunks

    abs_start, time_shift = _audio_window_alignment(
        audio_generated_before=audio_generated_before,
        audio_hist_frames=a_hist_frames,
        video_abs_current_frame=1 + frames_generated_before,
        video_rel_current_frame=1 + v_hist_frames,
        fps=fps,
    )
    assert abs_start == audio_generated_before - a_hist_frames
    assert time_shift > 0, "steady state must have a positive video compression shift"

    # Video window positions (window-relative seconds), 1 token per frame.
    window_frames = 1 + v_hist_frames + chunk_frames
    vpos = torch.zeros(1, 3, window_frames, 2)
    for f in range(window_frames):
        s, e = _video_latent_frame_bounds_sec(f, fps)
        vpos[0, 0, f, 0], vpos[0, 0, f, 1] = s, e

    # Audio window positions on the aligned clock.
    a_window = a_hist_frames + audio_per_chunk
    patchifier = AudioPatchifier(patch_size=1, shift=abs_start)
    apos = patchifier.get_patch_grid_bounds(AudioLatentShape(1, 8, a_window, 16), device) - time_shift

    a2v, v2a = cross_causal_attention_mask(vpos, apos, lookahead_sec=0.0)
    v_cur = window_frames - 1
    a_cur0 = a_hist_frames
    # The tail of the audio current chunk overlaps the video current frame in
    # absolute time, so under the aligned clock it must see it.
    assert v2a[0, a_cur0 + audio_per_chunk - 1, v_cur] == 1, (
        "aligned clock: last audio current frame must see the video current frame"
    )
    # Without alignment (raw window-relative audio grid) it cannot — the bug.
    apos_raw = AudioPatchifier(patch_size=1).get_patch_grid_bounds(AudioLatentShape(1, 8, a_window, 16), device)
    _, v2a_raw = cross_causal_attention_mask(vpos, apos_raw, lookahead_sec=0.0)
    assert torch.all(v2a_raw[0, a_cur0:, v_cur] == 0), (
        "sanity: the unaligned clock should hide video current from audio current"
    )
    print(f"[clock-align] shift={time_shift:.3f}s abs_start={abs_start}: audio current sees video current OK")


def test_cross_mask_empty_row_fallback() -> None:
    """A query row with no causally-visible key must fall back to the earliest
    key (all-zero rows degenerate to uniform attention after the log-bias
    conversion)."""
    # Video sink at [0, 0.04); all audio starts at >= 1.0s (evicted early audio).
    vpos = torch.zeros(1, 3, 1, 2)
    vpos[0, 0, 0] = torch.tensor([0.0, 0.04])
    apos = torch.zeros(1, 1, 3, 2)
    apos[0, 0, :, 0] = torch.tensor([1.0, 1.04, 1.08])
    apos[0, 0, :, 1] = torch.tensor([1.04, 1.08, 1.12])

    a2v, v2a = cross_causal_attention_mask(vpos, apos, lookahead_sec=0.0)
    assert a2v[0, 0].sum() == 1, f"empty a2v row must open exactly one key, got {a2v[0, 0].tolist()}"
    assert a2v[0, 0, 0] == 1, "fallback must pick the earliest audio key"
    assert torch.all(v2a[0].sum(dim=-1) >= 1), "v2a rows must never be empty (sink key at t=0)"
    print("[empty-row] all-zero cross rows fall back to the earliest key OK")


def main() -> None:
    torch.manual_seed(0)
    test_audio_chunk_tiling()
    test_audio_window_state()
    test_joint_cross_mask_shapes()
    test_audio_window_clock_alignment()
    test_cross_mask_empty_row_fallback()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
