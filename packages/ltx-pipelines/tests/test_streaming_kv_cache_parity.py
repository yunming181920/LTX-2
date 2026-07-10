"""Streaming KV-cache validation: single-chunk parity + multi-chunk smoke.

WHY two phases:

* **Phase 1 — strict parity (single chunk).** While no history exists, the
  Milestone 2 cached path must reproduce Milestone 1's standard attention
  exactly (modulo bf16): same q/k norm + RoPE (window pe == modality pe for
  chunk 0), same log-space additive block-causal bias, and — because M2 draws
  its per-chunk noise over the full window and slices the current rows — the
  exact same initial noise. Any divergence here is a plumbing bug (RoPE
  reapplication, mask conversion, noise accounting).

* **Phase 2 — multi-chunk smoke.** Once history exists the two paths are NOT
  numerically equivalent by design: M1 recomputes history features every step
  as timestep-0 conditioning tokens, whereas M2 (per Vidu S1 §2.3.1) reuses
  K/V captured when each chunk was *current* (different AdaLN conditioning,
  different window layout at capture time). Phase 2 therefore only checks
  that the cached path runs through sliding-window eviction + the persistent
  first-chunk slot without shape/NaN blowups, and reports the divergence for
  information.

This is an INTEGRATION test: it loads the full LTX-2 checkpoint + Gemma and
needs a reference image + audio. Run it in the GPU environment:

    uv run python packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py \
        --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
        --gemma-root models/gemma-3-12b --image ref.jpg --audio-path ref.wav
"""

import argparse

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.streaming import streaming_generate, streaming_generate_cached


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--gemma-root", required=True)
    p.add_argument("--image", required=True, help="reference first-frame image (sink)")
    p.add_argument("--audio-path", required=True)
    p.add_argument("--prompt", default="a person talking")
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-inference-steps", type=int, default=6)
    p.add_argument("--window-chunks", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-1, help="phase-1 strict parity tolerance (bf16)")
    p.add_argument(
        "--smoke-num-frames",
        type=int,
        default=41,
        help="pixel frames for the multi-chunk smoke phase (default 41 -> 5 generated "
        "latent frames -> exercises the persistent first chunk + sliding eviction "
        "with window-chunks=2).",
    )
    return p.parse_args()


def _run_both(pipeline, common_static, *, num_frames, frame_rate, chunk_frames, window_chunks, seed, device):
    """Run M1 and M2 with independent, identically-seeded noisers."""
    pixel_shape = VideoPixelShape(1, num_frames, common_static["height"], common_static["width"], frame_rate)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    video_tools_full = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, frame_rate)

    kwargs = dict(
        sigmas=common_static["sigmas"],
        num_generated_latent_frames=v_shape.frames - 1,
        chunk_frames=chunk_frames,
        window_chunks=window_chunks,
        video_tools_full=video_tools_full,
        audio_latent_full=common_static["audio_latent_full"],
        audio_lookahead=8,
        sink_latent_unpatchified=common_static["sink_latent"],
        v_context=common_static["v_context"],
        a_context=common_static["a_context"],
        stepper=EulerDiffusionStep(),
        dtype=torch.bfloat16,
        device=device,
    )

    with pipeline.stage.model_context() as transformer:
        m1 = streaming_generate(
            transformer=transformer,
            noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(seed)),
            **kwargs,
        )
        m2 = streaming_generate_cached(
            transformer=transformer,
            noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(seed)),
            **kwargs,
        )
    return m1, m2


def main() -> None:
    args = _build_args()
    from ltx_pipelines.a2vid_streaming import A2VidStreamingPipeline  # local import (heavy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    pipeline = A2VidStreamingPipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=(),
        device=device,
    )

    images = [ImageConditioningInput(path=args.image, frame_idx=0, strength=1.0)]

    # Shared inputs.
    ctx_p, _ = pipeline.prompt_encoder([args.prompt, ""], enhance_prompt_image=images[0][0])
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_pipelines.utils.media_io import decode_audio_from_file, load_image_and_preprocess

    sink_latent = pipeline.image_conditioner(
        lambda enc: enc(
            load_image_and_preprocess(
                image_path=images[0].path, height=args.height, width=args.width,
                dtype=dtype, device=device, crf=images[0].crf,
            )
        )
    )
    decoded_audio = decode_audio_from_file(
        args.audio_path, device, 0.0, args.smoke_num_frames / args.frame_rate
    )
    audio_latent_full = pipeline.audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))

    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(dtype=torch.float32, device=device)
    common_static = dict(
        sigmas=sigmas,
        audio_latent_full=audio_latent_full,
        sink_latent=sink_latent,
        v_context=ctx_p.video_encoding,
        a_context=ctx_p.audio_encoding,
        height=args.height,
        width=args.width,
    )

    # ---- Phase 1: strict single-chunk parity (no history -> paths identical).
    # 9 pixel frames -> 2 latent frames -> 1 generated frame -> exactly 1 chunk.
    m1, m2 = _run_both(
        pipeline, common_static,
        num_frames=9, frame_rate=args.frame_rate, chunk_frames=1,
        window_chunks=args.window_chunks, seed=args.seed, device=device,
    )
    diff = (m1.float() - m2.float()).abs()
    print(f"[phase 1] single-chunk  max|diff|={diff.max().item():.4e}  mean|diff|={diff.mean().item():.4e}")
    assert diff.max().item() < args.tol, (
        f"M2 KV-cache path diverges from M1 on the history-free chunk "
        f"(max diff {diff.max().item():.4e} >= tol {args.tol}). "
        "Debug the RoPE reapplication / mask conversion / noise accounting."
    )
    print("[phase 1] PARITY OK: cached path ≡ standard path with no history.")

    # ---- Phase 2: multi-chunk smoke (persistent first chunk + sliding eviction).
    # M1 and M2 are NOT numerically equivalent with history (by design); only
    # check the cached path stays finite and report divergence for information.
    m1, m2 = _run_both(
        pipeline, common_static,
        num_frames=args.smoke_num_frames, frame_rate=args.frame_rate, chunk_frames=1,
        window_chunks=args.window_chunks, seed=args.seed, device=device,
    )
    assert torch.isfinite(m2.float()).all(), "M2 produced non-finite latents in the multi-chunk run."
    assert torch.isfinite(m1.float()).all(), "M1 produced non-finite latents in the multi-chunk run."
    diff = (m1.float() - m2.float()).abs()
    print(
        f"[phase 2] multi-chunk  max|diff|={diff.max().item():.4e}  mean|diff|={diff.mean().item():.4e} "
        "(nonzero expected: cached history K/V vs full recompute)"
    )
    print("[phase 2] SMOKE OK: multi-chunk cached run finished with finite latents.")


if __name__ == "__main__":
    main()
