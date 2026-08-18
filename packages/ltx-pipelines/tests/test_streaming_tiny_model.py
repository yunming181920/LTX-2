"""Tiny-model streaming-strategy validation on CPU (random weights, no checkpoint).

A 2-layer randomly initialized ``LTXModel`` is enough to check the *plumbing*
of every streaming strategy end to end (no checkpoint needed).

Strategies (see ``--stream-strategy`` in ``ti2vid_streaming.py``):
  * ``kv_twin`` (default): KV cache, history reads noisy at mid steps + clean at
    the final step (Vidu S1 §2.3.1).
  * ``kv_clean`` (ablation A): KV cache, history reads clean at every step.
  * ``kv_noisy_steps`` (ablation B): KV cache, step t reads the history's own
    step-t noisy snapshot (noise-level matched; no clean, no sigma-0 forward).
  * ``image_cond`` (ablation C): no KV cache; each chunk conditions on the
    previous chunk's last frame as the image reference (rotating sink).

Phase 1: single-chunk strict parity — with NO history every cached path
(``kv_twin``/``kv_clean``/``kv_noisy_steps``) has an empty cache (``read()``
returns ``None``), so the three KV strategies are numerically identical to each
other, for BOTH modalities, cross-attn off/on. ``image_cond`` keeps its
all-causal rotating-sink path, so its single-chunk output legitimately differs
(the old latent-recompute parity reference was removed with the M1 path).

Phase 2: multi-chunk smoke deep into eviction (8 chunks, window 2) with the
time-causal cross-attn ON — exercises every strategy's per-chunk finalize /
cache-commit / sink-rotation; checks all produce finite latents of the right
shape. (With history the strategies are NOT numerically equal by design.)

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
from ltx_pipelines.utils.streaming import (
    streaming_generate_joint_cached,
    streaming_generate_joint_image_cond,
)

FPS = 25.0
H, W = 2, 3
CH = 8
ACH, MEL = 4, 16  # audio token dim = 64

# Map CLI strategy name -> (driver, cache-strategy-or-None)
_STRATEGIES = {
    "kv_twin": (streaming_generate_joint_cached, "twin"),
    "kv_clean": (streaming_generate_joint_cached, "clean"),
    "kv_noisy_steps": (streaming_generate_joint_cached, "noisy_steps"),
    "image_cond": (streaming_generate_joint_image_cond, None),
}


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


def run(x0: X0Model, num_latent_frames: int, *, strategy: str, causal_cross_attn: bool, chunk_frames: int = 1, seed: int = 0):
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
        chunk_frames=chunk_frames,
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
    fn, cache_strategy = _STRATEGIES[strategy]
    if cache_strategy is not None:
        return fn(**kwargs, window_chunks=2, strategy=cache_strategy)
    return fn(**kwargs)  # image_cond: no window_chunks


def main() -> None:
    x0 = build_tiny()
    with torch.inference_mode():
        # Phase 1: single chunk (2 latent frames -> 1 generated), no history.
        # The three KV strategies share the bidirectional ti2v bootstrap and
        # must agree bit-close (no history => identical paths). image_cond
        # keeps its all-causal rotating-sink path, so its single-chunk output
        # legitimately differs from the KV strategies — finite only.
        for ccx in (False, True):
            outs = {s: run(x0, 2, strategy=s, causal_cross_attn=ccx) for s in _STRATEGIES}
            ref_v, ref_a = outs["kv_twin"]
            for s, (v, a) in outs.items():
                dv = (ref_v - v).abs().max().item()
                da = (ref_a - a).abs().max().item()
                print(f"[phase1 ccx={ccx}] {s:15s} vs kv_twin: "
                      f"video max|diff|={dv:.3e} audio max|diff|={da:.3e}")
                if s == "image_cond":
                    assert torch.isfinite(v).all() and torch.isfinite(a).all(), f"non-finite latents ({s})"
                else:
                    assert dv < 1e-4 and da < 1e-4, f"single-chunk {s} must match kv_twin (ccx={ccx})"

        # Phase 2: 8 generated chunks, window 2 -> deep eviction; causal cross ON.
        for s in _STRATEGIES:
            v, a = run(x0, 9, strategy=s, causal_cross_attn=True)
            assert torch.isfinite(v).all() and torch.isfinite(a).all(), f"non-finite latents ({s})"
            print(f"[phase2] {s:15s} multi-chunk: video {tuple(v.shape)} audio {tuple(a.shape)} finite OK")

        # Phase 3: coarse 31-latent chunks (the pipeline default). 32 latent
        # frames total (sink + 31 generated) decode to 1 + 31*8 = 249 pixel
        # frames. Single-chunk bootstrap parity at the coarse granularity, a
        # 2-chunk smoke (62 generated = [31, 31]), and the causal-VAE
        # decode-math round trip (pixel <-> latent frame counts).
        for ccx in (False, True):
            outs = {s: run(x0, 32, strategy=s, causal_cross_attn=ccx, chunk_frames=31) for s in _STRATEGIES}
            ref_v, ref_a = outs["kv_twin"]
            for s, (v, a) in outs.items():
                dv = (ref_v - v).abs().max().item()
                da = (ref_a - a).abs().max().item()
                print(f"[phase3 ccx={ccx}] {s:15s} 31-latent chunk vs kv_twin: "
                      f"video max|diff|={dv:.3e} audio max|diff|={da:.3e}")
                if s == "image_cond":
                    assert torch.isfinite(v).all() and torch.isfinite(a).all(), f"non-finite latents ({s})"
                else:
                    assert dv < 1e-4 and da < 1e-4, f"31-latent-chunk {s} must match kv_twin (ccx={ccx})"
                assert v.shape == (1, CH, 32, H, W), f"unexpected video shape {tuple(v.shape)}"
        for s in _STRATEGIES:
            v, a = run(x0, 63, strategy=s, causal_cross_attn=True, chunk_frames=31)  # 62 generated -> [31, 31]
            assert torch.isfinite(v).all() and torch.isfinite(a).all(), f"non-finite latents ({s})"
            assert v.shape == (1, CH, 63, H, W), f"unexpected video shape {tuple(v.shape)}"
            print(f"[phase3] {s:15s} 31-latent chunks x2: video {tuple(v.shape)} audio {tuple(a.shape)} finite OK")

        # Default granularity (1 image + 6 latents per window = ~2 s at 24 fps):
        # 12 generated -> [6, 6].
        for s in _STRATEGIES:
            v, a = run(x0, 13, strategy=s, causal_cross_attn=True, chunk_frames=6)
            assert torch.isfinite(v).all() and torch.isfinite(a).all(), f"non-finite latents ({s})"
            assert v.shape == (1, CH, 13, H, W), f"unexpected video shape {tuple(v.shape)}"
            print(f"[phase3] {s:15s} 6-latent chunks x2:   video {tuple(v.shape)} audio {tuple(a.shape)} finite OK")

        # Causal-VAE decode arithmetic: 249 pixel frames <-> 1 + 31*8 (32
        # latent frames; the first latent frame covers 1 pixel frame, every
        # later one covers 8). Mirrors ConvVideoDecoder's F = 8*(F'-1) + 1.
        from ltx_core.types import VIDEO_SCALE_FACTORS, VideoPixelShape
        pixel = VideoPixelShape(batch=1, frames=249, height=H * 32, width=W * 32, fps=FPS)
        latent = VideoLatentShape.from_pixel_shape(pixel, scale_factors=VIDEO_SCALE_FACTORS)
        assert latent.frames == 32, f"249 pixel frames should map to 32 latent frames, got {latent.frames}"
        assert latent.upscale(VIDEO_SCALE_FACTORS).frames == 249
        print("[phase3] causal-VAE frame math: 249 pixel frames <-> 32 latent frames (1 + 31*8) OK")
    print("\nTINY STREAMING-STRATEGY VALIDATION PASSED")


if __name__ == "__main__":
    main()
