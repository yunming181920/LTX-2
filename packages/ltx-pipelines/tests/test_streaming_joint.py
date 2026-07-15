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
    _build_audio_window_state,
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

    # denoise_mask: 0 on history (frozen), 1 on current (denoised).
    assert torch.all(state.denoise_mask[0, :5, 0] == 0), "history must be frozen (mask 0)"
    assert torch.all(state.denoise_mask[0, 5:, 0] == 1), "current must be denoised (mask 1)"

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


def main() -> None:
    torch.manual_seed(0)
    test_audio_chunk_tiling()
    test_audio_window_state()
    test_joint_cross_mask_shapes()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
