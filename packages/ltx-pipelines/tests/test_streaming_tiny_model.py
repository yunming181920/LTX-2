"""Tiny-model M1 vs M2 validation on CPU (random weights, no checkpoint).

Complements ``test_streaming_joint_parity.py`` (which needs the full GPU
checkpoint): a 2-layer randomly initialized ``LTXModel`` is enough to check the
*plumbing* of both streaming paths end to end.

Phase 1: single-chunk strict parity — with no history the M2 cached path
(RoPE repositioning + query-mask + full-window noise accounting + clean-KV
refresh) must reproduce M1's standard attention numerically, for BOTH
modalities, with causal cross-attention off AND on.

Phase 2: multi-chunk smoke deep into eviction (8 chunks, window 2) with the
time-causal cross-attn ON — exercises the audio window clock alignment
(``_audio_window_alignment``), the empty-cross-row fallback, the persistent
first video chunk, FIFO eviction and the M2 per-chunk clean-KV refresh; checks
both paths produce finite latents. (With history the paths are NOT numerically
equal by design — M2 reuses cached K/V.)

Run:

    uv run python packages/ltx-pipelines/tests/test_streaming_tiny_model.py
"""

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.transformer.model import LTXModel, X0Model
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape
from ltx_pipelines.utils.streaming import streaming_generate_joint, streaming_generate_joint_cached

FPS = 25.0
H, W = 2, 3
CH = 8
ACH, MEL = 4, 16  # audio token dim = 64


def build_tiny() -> X0Model:
    torch.manual_seed(7)
    model = LTXModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=CH,
        out_channels=CH,
        num_layers=2,
        cross_attention_dim=16,  # == video inner_dim (context is viewed to inner_dim)
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_in_channels=ACH * MEL,
        audio_out_channels=ACH * MEL,
        audio_cross_attention_dim=16,  # == audio inner_dim
    )
    # Some params (e.g. scale_shift_table) are torch.empty at construction and
    # only filled from the checkpoint in production — reinitialize EVERYTHING
    # to small gaussians so the random model is finite and well-behaved.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    return X0Model(model).float().eval()


def run(x0: X0Model, num_latent_frames: int, *, use_kv_cache: bool, causal_cross_attn: bool, seed: int = 0):
    device = torch.device("cpu")
    v_shape = VideoLatentShape(1, CH, num_latent_frames, H, W)
    total_audio = int(round((num_latent_frames - 1) * 8 / FPS * 25)) + 1
    a_shape = AudioLatentShape(1, ACH, total_audio, MEL)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, FPS)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
    torch.manual_seed(11)
    kwargs = dict(
        sigmas=torch.linspace(1.0, 0.0, 5),
        num_generated_latent_frames=num_latent_frames - 1,
        chunk_frames=1,
        window_chunks=2,
        video_tools_full=video_tools,
        audio_tools_full=audio_tools,
        sink_latent_unpatchified=torch.randn(1, CH, 1, H, W),
        v_context=torch.randn(1, 4, 16),
        a_context=torch.randn(1, 4, 16),
        stepper=EulerDiffusionStep(),
        transformer=x0,
        noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(seed)),
        dtype=torch.float32,
        device=device,
        causal_cross_attn=causal_cross_attn,
        cross_attn_lookahead_sec=0.0,
    )
    fn = streaming_generate_joint_cached if use_kv_cache else streaming_generate_joint
    return fn(**kwargs)


def main() -> None:
    x0 = build_tiny()
    with torch.inference_mode():
        # Phase 1: single chunk (2 latent frames -> 1 generated), no history.
        for ccx in (False, True):
            m1 = run(x0, 2, use_kv_cache=False, causal_cross_attn=ccx)
            m2 = run(x0, 2, use_kv_cache=True, causal_cross_attn=ccx)
            dv = (m1[0] - m2[0]).abs().max().item()
            da = (m1[1] - m2[1]).abs().max().item()
            print(f"[phase1 ccx={ccx}] single-chunk parity: video max|diff|={dv:.3e} audio max|diff|={da:.3e}")
            assert dv < 1e-4 and da < 1e-4, f"single-chunk M2 must match M1 (ccx={ccx})"

        # Phase 2: 8 generated chunks, window 2 -> deep eviction; causal cross ON.
        for kv in (False, True):
            v, a = run(x0, 9, use_kv_cache=kv, causal_cross_attn=True)
            assert torch.isfinite(v).all() and torch.isfinite(a).all(), f"non-finite latents (kv={kv})"
            print(f"[phase2] multi-chunk kv={kv}: video {tuple(v.shape)} audio {tuple(a.shape)} finite OK")
    print("\nTINY M1/M2 VALIDATION PASSED")


if __name__ == "__main__":
    main()
