"""Parity test: Milestone 2 (KV cache + RoPE repositioning) ≡ Milestone 1 (latent-level
TwinCache, full recompute) to bf16 tolerance.

WHY: M2 is a pure efficiency optimization — the cached history K/V + re-applied
RoPE must reproduce M1's full-recompute attention exactly (modulo bf16). A
silent shape/order bug in the RoPE reapplication or the query-mask slicing would
diverge, so this test is the gate before trusting M2.

This is an INTEGRATION test: it loads the full LTX-2 checkpoint + Gemma and
needs a reference image + audio. Run it in the GPU environment:

    uv run python -m pytest packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py -s \
        --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
        --gemma-root models/gemma-3-12b --image ref.jpg --audio-path ref.wav

or directly:

    uv run python packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py \
        --checkpoint-path ... --gemma-root ... --image ref.jpg --audio-path ref.wav

A passing run (max abs diff < 1e-1 in the bf16 latent) confirms the cache path
is correct. Expect small diffs from the clean-K/V capture timing (near-clean vs
clean) — if diffs are large, the RoPE/mask plumbing needs debugging.
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
from ltx_pipelines.utils.helpers import modality_from_latent_state  # noqa: F401
from ltx_pipelines.utils.streaming import streaming_generate, streaming_generate_cached


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--gemma-root", required=True)
    p.add_argument("--image", required=True, help="reference first-frame image (sink)")
    p.add_argument("--audio-path", required=True)
    p.add_argument("--prompt", default="a person talking")
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-inference-steps", type=int, default=6)
    p.add_argument("--window-chunks", type=int, default=2)
    p.add_argument("--chunk-frames", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-1)
    return p.parse_args()


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
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    from ltx_pipelines.utils.media_io import decode_audio_from_file, load_image_and_preprocess
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio

    sink_latent = pipeline.image_conditioner(
        lambda enc: enc(
            load_image_and_preprocess(
                image_path=images[0].path, height=args.height, width=args.width,
                dtype=dtype, device=device, crf=images[0].crf,
            )
        )
    )
    decoded_audio = decode_audio_from_file(args.audio_path, device, 0.0, args.num_frames / args.frame_rate)
    audio_latent_full = pipeline.audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))

    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(dtype=torch.float32, device=device)
    pixel_shape = VideoPixelShape(1, args.num_frames, args.height, args.width, args.frame_rate)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    video_tools_full = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, args.frame_rate)

    common = dict(
        sigmas=sigmas,
        num_generated_latent_frames=v_shape.frames - 1,
        chunk_frames=args.chunk_frames,
        window_chunks=args.window_chunks,
        video_tools_full=video_tools_full,
        audio_latent_full=audio_latent_full,
        audio_lookahead=8,
        sink_latent_unpatchified=sink_latent,
        v_context=v_context_p,
        a_context=a_context_p,
        stepper=EulerDiffusionStep(),
        noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(args.seed)),
        dtype=dtype,
        device=device,
    )

    # M1 (full recompute) and M2 (KV cache) under the same model context.
    with pipeline.stage.model_context() as transformer:
        m1 = streaming_generate(transformer=transformer, **common)
        m2 = streaming_generate_cached(transformer=transformer, **common)

    diff = (m1.float() - m2.float()).abs()
    print(f"M1 vs M2  max|diff|={diff.max().item():.4e}  mean|diff|={diff.mean().item():.4e}")
    assert diff.max().item() < args.tol, (
        f"M2 KV-cache path diverges from M1 (max diff {diff.max().item():.4e} >= tol {args.tol}). "
        "Debug the RoPE reapplication / query-mask slicing."
    )
    print("PARITY OK: M2 KV-cache ≡ M1 within tolerance.")


if __name__ == "__main__":
    main()
