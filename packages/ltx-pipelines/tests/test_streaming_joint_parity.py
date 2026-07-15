"""Joint streaming TI2V validation: single-chunk M1≡M2 parity + multi-chunk smoke.

WHY two phases:

* **Phase 1 — strict parity (single chunk).** While no history exists, the M2
  cached path must reproduce M1's standard attention exactly (modulo bf16) for
  BOTH video and audio: same q/k norm + RoPE (window pe == modality pe for chunk
  0), same log-space additive block-causal bias, and — because M2 draws its
  per-chunk noise over the full window and slices the current rows — the exact
  same initial noise. Any divergence here is a plumbing bug (RoPE reapplication,
  mask conversion, noise accounting, audio-cache layout).

* **Phase 2 — multi-chunk smoke.** Once history exists the two paths are NOT
  numerically equivalent by design (M1 recomputes history features every step as
  timestep-0 conditioning tokens, whereas M2 reuses K/V captured when each chunk
  was *current*). Phase 2 only checks that both paths run through sliding-window
  eviction + the persistent first-chunk slot (video) and the audio FIFO without
  shape/NaN blowups, and reports the divergence for information.

This is an INTEGRATION test: it loads the full LTX-2 checkpoint + Gemma and
needs a reference image (there is no audio input — audio is generated). Run it
in the GPU environment:

    uv run python packages/ltx-pipelines/tests/test_streaming_joint_parity.py \
        --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
        --gemma-root models/gemma-3-12b --image ref.jpg
"""

import argparse

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.streaming import streaming_generate_joint, streaming_generate_joint_cached


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--gemma-root", required=True)
    p.add_argument("--image", required=True, help="reference first-frame image (sink)")
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
        "latent frames -> exercises the persistent first video chunk + sliding eviction "
        "with window-chunks=2).",
    )
    return p.parse_args()


def _run_both(pipeline, common_static, *, num_frames, frame_rate, chunk_frames, window_chunks, seed, device):
    """Run M1 and M2 joint with independent, identically-seeded noisers."""
    pixel_shape = VideoPixelShape(1, num_frames, common_static["height"], common_static["width"], frame_rate)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    video_tools_full = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, frame_rate)
    audio_tools_full = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)

    kwargs = dict(
        sigmas=common_static["sigmas"],
        num_generated_latent_frames=v_shape.frames - 1,
        chunk_frames=chunk_frames,
        window_chunks=window_chunks,
        video_tools_full=video_tools_full,
        audio_tools_full=audio_tools_full,
        sink_latent_unpatchified=common_static["sink_latent"],
        v_context=common_static["v_context"],
        a_context=common_static["a_context"],
        stepper=EulerDiffusionStep(),
        dtype=torch.bfloat16,
        device=device,
        # Isolate the KV-cache comparison: disable the (default-ON) causal
        # cross-attention so this test measures M2-vs-M1 KV-cache parity only.
        causal_cross_attn=False,
    )

    with pipeline.stage.model_context() as transformer:
        m1_v, m1_a = streaming_generate_joint(
            transformer=transformer,
            noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(seed)),
            **kwargs,
        )
        m2_v, m2_a = streaming_generate_joint_cached(
            transformer=transformer,
            noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(seed)),
            **kwargs,
        )
    return (m1_v, m1_a), (m2_v, m2_a)


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    args = _build_args()
    from ltx_pipelines.ti2vid_streaming import TI2VidStreamingPipeline  # local import (heavy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    pipeline = TI2VidStreamingPipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=(),
        device=device,
    )

    images = [ImageConditioningInput(path=args.image, frame_idx=0, strength=1.0)]

    # Shared inputs (positive prompt only; no CFG in M1).
    ctx_p, _ = pipeline.prompt_encoder([args.prompt, ""])
    from ltx_pipelines.utils.media_io import load_image_and_preprocess

    sink_latent = pipeline.image_conditioner(
        lambda enc: enc(
            load_image_and_preprocess(
                image_path=images[0].path, height=args.height, width=args.width,
                dtype=dtype, device=device, crf=images[0].crf,
            )
        )
    )

    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(dtype=torch.float32, device=device)
    common_static = dict(
        sigmas=sigmas,
        sink_latent=sink_latent,
        v_context=ctx_p.video_encoding,
        a_context=ctx_p.audio_encoding,
        height=args.height,
        width=args.width,
    )

    # ---- Phase 1: strict single-chunk parity (no history -> paths identical).
    # 9 pixel frames -> 2 latent frames -> 1 generated video frame -> 1 chunk.
    m1, m2 = _run_both(
        pipeline, common_static,
        num_frames=9, frame_rate=args.frame_rate, chunk_frames=1,
        window_chunks=args.window_chunks, seed=args.seed, device=device,
    )
    dv = _maxdiff(m1[0], m2[0])
    da = _maxdiff(m1[1], m2[1])
    print(f"[phase 1] single-chunk  video max|diff|={dv:.4e}  audio max|diff|={da:.4e}")
    assert dv < args.tol, f"M2 video diverges from M1 on the history-free chunk ({dv:.4e} >= {args.tol})"
    assert da < args.tol, f"M2 audio diverges from M1 on the history-free chunk ({da:.4e} >= {args.tol})"
    print("[phase 1] PARITY OK: cached path ≡ standard path with no history (both modalities).")

    # ---- Phase 2: multi-chunk smoke (persistent first video chunk + sliding eviction + audio FIFO).
    m1, m2 = _run_both(
        pipeline, common_static,
        num_frames=args.smoke_num_frames, frame_rate=args.frame_rate, chunk_frames=1,
        window_chunks=args.window_chunks, seed=args.seed, device=device,
    )
    for name, t in [("m1_video", m1[0]), ("m1_audio", m1[1]), ("m2_video", m2[0]), ("m2_audio", m2[1])]:
        assert torch.isfinite(t.float()).all(), f"{name} produced non-finite latents"
    print(
        f"[phase 2] multi-chunk  video max|diff|={_maxdiff(m1[0], m2[0]):.4e}  "
        f"audio max|diff|={_maxdiff(m1[1], m2[1]):.4e} (nonzero expected: cached history K/V vs recompute)"
    )
    print("[phase 2] SMOKE OK: multi-chunk cached run finished with finite latents (both modalities).")


if __name__ == "__main__":
    main()
