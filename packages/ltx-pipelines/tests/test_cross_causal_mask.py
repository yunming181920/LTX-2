"""Unit tests for ``cross_causal_attention_mask`` (time-causal AV cross-attn).

Pure-tensor: builds representative LTX-2-style window-relative positions
(video temporal dim = frame_index/fps seconds; audio temporal dim = spectrogram
seconds) and checks the mask is shape-correct, causal (per-row prefix of 1s),
monotone across frames, widened by lookahead, and that the sink (video frame 0)
cannot see future audio.

Needs torch + the ltx env (the helper import pulls in ltx_core). Run:

    uv run python packages/ltx-pipelines/tests/test_cross_causal_mask.py
"""

import torch

from ltx_pipelines.utils.streaming import cross_causal_attention_mask

FPS = 30.0
VIDEO_STRIDE = 8  # pixel frames per latent video frame (causal VAE temporal stride)
AUDIO_FPS = 25.0  # audio latent frames per second


def _video_positions(num_frames: int, batch: int = 1) -> torch.Tensor:
    """``(B, 3, T_v, 2)`` with dim 0 = temporal [start, end) in seconds."""
    start = torch.arange(num_frames, dtype=torch.float32) * VIDEO_STRIDE / FPS
    end = start + VIDEO_STRIDE / FPS
    t = torch.stack([start, end], dim=-1)  # (T_v, 2)
    # dims 1,2 (height, width) unused by the mask; zeros.
    pos = torch.zeros(batch, 3, num_frames, 2)
    pos[:, 0, :, :] = t
    return pos


def _audio_positions(num_frames: int, batch: int = 1) -> torch.Tensor:
    """``(B, 1, T_a, 2)`` with dim 0 = temporal [start, end) in seconds."""
    start = torch.arange(num_frames, dtype=torch.float32) / AUDIO_FPS
    end = start + 1.0 / AUDIO_FPS
    pos = torch.zeros(batch, 1, num_frames, 2)
    pos[:, 0, :, :] = torch.stack([start, end], dim=-1)
    return pos


def _is_prefix_of_ones(row: torch.Tensor) -> bool:
    """True if ``row`` is a block of 1s followed by 0s (no 0-then-1)."""
    nz = (row == 0).nonzero(as_tuple=True)[0]
    oz = (row == 1).nonzero(as_tuple=True)[0]
    if len(nz) == 0 or len(oz) == 0:
        return True
    return nz.min().item() > oz.max().item()


def main() -> None:
    torch.manual_seed(0)
    tv, ta = 4, 10
    vpos = _video_positions(tv)
    apos = _audio_positions(ta)

    # --- shapes ---
    a2v, v2a = cross_causal_attention_mask(vpos, apos, lookahead_sec=0.0)
    assert a2v.shape == (1, tv, ta), f"a2v shape {a2v.shape} != (1,{tv},{ta})"
    assert v2a.shape == (1, ta, tv), f"v2a shape {v2a.shape} != (1,{ta},{tv})"
    assert a2v.dtype == torch.float32 and v2a.dtype == torch.float32
    print(f"[shapes] a2v={tuple(a2v.shape)} v2a={tuple(v2a.shape)} OK")

    # --- values in {0,1} ---
    assert torch.all((a2v == 0) | (a2v == 1)), "a2v has non-binary values"
    assert torch.all((v2a == 0) | (v2a == 1)), "v2a has non-binary values"

    # --- causal: each row is a prefix of 1s (allow past, mask future) ---
    for i in range(tv):
        assert _is_prefix_of_ones(a2v[0, i]), f"a2v row {i} is not a causal prefix: {a2v[0, i].tolist()}"
    for i in range(ta):
        assert _is_prefix_of_ones(v2a[0, i]), f"v2a row {i} is not a causal prefix: {v2a[0, i].tolist()}"
    print("[causal] all rows are 1-prefixes (past allowed, future masked) OK")

    # --- monotone: later video frames see >= audio frames (row sums non-decreasing) ---
    a2v_sums = a2v[0].sum(dim=1)
    assert torch.all(a2v_sums[1:] >= a2v_sums[:-1]), f"a2v row sums not non-decreasing: {a2v_sums.tolist()}"
    v2a_sums = v2a[0].sum(dim=1)
    assert torch.all(v2a_sums[1:] >= v2a_sums[:-1]), f"v2a row sums not non-decreasing: {v2a_sums.tolist()}"
    print(f"[monotone] a2v row sums={a2v_sums.tolist()} v2a row sums={v2a_sums.tolist()} OK")

    # --- sink (video frame 0) must NOT see late audio ---
    # video frame 0 spans [0, 8/fps) ≈ 0.267s -> audio up to ~frame 6; frame 9 is ~0.36-0.40s, future.
    assert a2v[0, 0, ta - 1] == 0, "sink (video frame 0) must not attend the last (future) audio frame"
    assert a2v[0, 0, 0] == 1, "sink must attend the first audio frame (same-time)"
    print("[sink] video frame 0 attends early audio, masks future audio OK")

    # --- lookahead widens the band (row sums >= strict) ---
    a2v_la, _ = cross_causal_attention_mask(vpos, apos, lookahead_sec=1.0)
    assert torch.all(a2v_la[0].sum(dim=1) >= a2v[0].sum(dim=1)), "lookahead did not widen a2v"
    # large enough lookahead => fully connected
    a2v_full, _ = cross_causal_attention_mask(vpos, apos, lookahead_sec=1e6)
    assert torch.all(a2v_full == 1), "huge lookahead should make a2v fully connected"
    print("[lookahead] widens band, huge lookahead -> fully connected OK")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
