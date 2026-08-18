"""Streaming, autoregressive, causal text/image-to-video+audio (TI2V) pipeline.

Training-free reproduction of Vidu S1 §2.3 streaming inference on top of the
pretrained *bidirectional* LTX-2 checkpoint (used as-is as the "causal model").

Unlike an audio-to-video (A2V) streaming setup (where audio is a frozen control
signal and only video is generated), TI2V has no audio input:
it generates **both** video and audio. This pipeline generates them chunk by
chunk in lockstep — each video AR chunk also produces its time-aligned audio
latent frames — with a sliding window and a persistent anchor per Vidu S1
§2.3.1: chunk 1 is a standard **bidirectional ti2v bootstrap** (the encoded
first-frame image replaces latent frame 0, frozen, full attention), and the
whole ``[image | chunk 1]`` output is pinned as the never-evicted anchor;
later chunks stream causally over ``[anchor | history | current]`` with
latent-level TwinCache (noisy/clean history snapshots swapped per denoising
step) for **both** modalities, block-causal self-attention masks on both
temporal axes, and a time-causal video↔audio cross-attention mask.

Audio keeps its own sliding-window history (no sink — audio has no image
conditioning — but a pinned first chunk, the audio half of Vidu S1 §2.3.1's
first generated video-audio reference), so per-step activation memory is
O(window) for both modalities. The full latents are decoded once at the end
(causal-VAE seamless video decode + audio decode) and returned.

The driver is the KV-cache path (``kv_twin`` by default): per-block KV cache +
RoPE repositioning for *both* video and audio self-attention. Faster than the
removed latent-recompute path; conceptual/unvalidated on real weights (a
training-free reproduction on a bidirectional checkpoint) — smoke-test at
checkpoint scale before trusting quality. Ablations: ``kv_clean`` /
``kv_noisy_steps`` (history-read variants) and ``image_cond`` (no cache).

No core (ltx-core) production changes; reuses ``DiffusionStage``'s transformer
context for the transformer lifecycle, ``PromptEncoder``/``ImageConditioner`` for IO,
``VideoDecoder``/``AudioDecoder`` for output, and the streaming primitives in
:mod:`ltx_pipelines.utils.streaming`.
"""

import argparse
import logging
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TileSizeConfig, TilingConfig
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_1_stage_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.media_io import encode_video, load_image_and_preprocess
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)


class TI2VidStreamingPipeline:
    """Single-stage streaming text/image-to-video+audio pipeline (causal AR).

    The pretrained LTX-2 (full, non-distilled) checkpoint is used as the causal
    model. Video and audio are generated chunk-by-chunk in lockstep, each with a
    sliding window + TwinCache history (see
    :func:`ltx_pipelines.utils.streaming.streaming_generate_joint_cached`).
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ):
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self._scheduler = LTX2Scheduler()
        self.diffvae_optimization = diffvae_optimization
        # Scale factors derived from the video VAE checkpoint (conv VAE -> the
        # 32x32x8 default; DiffVAE -> its pixel_scale). Used for VideoLatentShape
        # and tiling resolution, matching the 2.5 pipeline contract.
        self.video_scale_factors = tiling_scale_factors_for_vae(model_paths.video_vae())
        self.prompt_encoder = PromptEncoder(
            model_paths,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=model_paths.video_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            checkpoint_path=model_paths.transformer(),
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            scale_factors=self.video_scale_factors,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=model_paths.video_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=model_paths.audio_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        window_chunks: int = 1,
        chunk_frames: int = 3,
        use_kv_cache: bool = False,
        stream_strategy: str = "kv_twin",
        causal_cross_attn: bool = True,
        cross_attn_lookahead_sec: float = 0.0,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        sigmas: torch.Tensor | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Generate a streaming causal TI2V clip. Returns ``(video_frames, audio)``.

        ``images[0]`` (frame_idx 0) is the sink reference. Audio is *generated*
        jointly with the video (no audio input); both modalities are produced
        chunk-by-chunk with bounded O(window) per-step memory.
        """
        assert_resolution(height=height, width=width, is_two_stage=False)
        if not images:
            raise ValueError("TI2VidStreamingPipeline requires a reference image (frame_idx=0) as the sink.")
        if window_chunks < 1:
            raise ValueError(f"window_chunks must be >= 1, got {window_chunks}")
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = self.dtype

        # Text context (negative unused — no CFG in the streaming path).
        ctx_p, _ = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0],
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding

        # Sink: encode the reference first-frame image to a video latent.
        ref = images[0]
        sink_latent = self.image_conditioner(
            lambda enc: enc(
                load_image_and_preprocess(
                    image_path=ref.path,
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=self.device,
                    crf=ref.crf,
                )
            )
        )  # (1, C, F, H_lat, W_lat)

        sigmas = (sigmas if sigmas is not None else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32, device=self.device
        )

        pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate)
        v_shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=self.video_scale_factors)
        a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        video_tools_full = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), v_shape, frame_rate, scale_factors=self.video_scale_factors
        )
        audio_tools_full = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
        num_generated_latent_frames = v_shape.frames - 1  # frame 0 is the sink
        if num_generated_latent_frames <= 0:
            raise ValueError(f"num_frames={num_frames} yields no frames to generate beyond the sink.")

        # Resolve tiling (AUTO_TILING/default -> concrete per-VAE/per-resolution config)
        # before decode, the 2.5 pipeline contract.
        resolved_tiling = ensure_tiling_config(
            tiling_config,
            scale_factors=self.video_scale_factors,
            video_shape=pixel_shape,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            diffvae_optimization=self.diffvae_optimization,
            device=self.device,
        )

        # Generation runs under the transformer's model context (DiT required);
        # decode uses separate decoders and runs outside it.
        stepper = EulerDiffusionStep()
        # NOTE: 2.5 made DiffusionStage's transformer context private; the streaming
        # driver holds the DiT open across chunks via this ctx (no public equivalent).
        with self.stage._transformer_ctx() as transformer:
            _KV_STRATEGY = {"kv_twin": "twin", "kv_clean": "clean", "kv_noisy_steps": "noisy_steps"}
            if stream_strategy in _KV_STRATEGY:
                from ltx_pipelines.utils.streaming import streaming_generate_joint_cached

                full_video_latent, full_audio_latent = streaming_generate_joint_cached(
                    sigmas=sigmas,
                    num_generated_latent_frames=num_generated_latent_frames,
                    chunk_frames=chunk_frames,
                    window_chunks=window_chunks,
                    video_tools_full=video_tools_full,
                    audio_tools_full=audio_tools_full,
                    sink_latent_unpatchified=sink_latent,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    stepper=stepper,
                    transformer=transformer,
                    noiser=noiser,
                    dtype=dtype,
                    device=self.device,
                    causal_cross_attn=causal_cross_attn,
                    cross_attn_lookahead_sec=cross_attn_lookahead_sec,
                    strategy=_KV_STRATEGY[stream_strategy],
                )
            elif stream_strategy == "image_cond":
                from ltx_pipelines.utils.streaming import streaming_generate_joint_image_cond

                full_video_latent, full_audio_latent = streaming_generate_joint_image_cond(
                    sigmas=sigmas,
                    num_generated_latent_frames=num_generated_latent_frames,
                    chunk_frames=chunk_frames,
                    video_tools_full=video_tools_full,
                    audio_tools_full=audio_tools_full,
                    sink_latent_unpatchified=sink_latent,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    stepper=stepper,
                    transformer=transformer,
                    noiser=noiser,
                    dtype=dtype,
                    device=self.device,
                    causal_cross_attn=causal_cross_attn,
                    cross_attn_lookahead_sec=cross_attn_lookahead_sec,
                )
            else:
                raise ValueError(f"unknown stream strategy {stream_strategy!r}")

        decoded_video = self.video_decoder(full_video_latent, resolved_tiling, generator=generator)
        decoded_audio = self.audio_decoder(full_audio_latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = default_1_stage_arg_parser(params=params)
    parser.add_argument(
        "--window-chunks",
        type=int,
        default=1,
        help="Sliding-window rolling-history size in AR chunks (TwinCache FIFO cap; the "
        "persistent items — the [image | chunk 1] anchor and the audio first chunk — are "
        "not counted). Audio uses the same cap for its own FIFO history. Default 1.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=3,
        help="Latent video frames generated per AR step (default 3 = ~1 s of video at "
        "24 fps). With the causal VAE the window decodes as 1 + frames*8 pixel frames: "
        "the first latent frame (the reference image) decodes to 1 frame and every "
        "later latent frame to 8, so a 3-latent chunk (1 image + 3 = 4 latents in the "
        "window) yields 1 + 3*8 = 25 pixel frames (~1.04 s at 24 fps). The "
        "time-aligned audio latent frames for each chunk are generated in lockstep "
        "(~8/fps*25 audio frames per video latent frame).",
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="Legacy no-op alias (the KV-cache TwinCache path is now the default). "
        "Ignored if --stream-strategy is given explicitly.",
    )
    parser.add_argument(
        "--stream-strategy",
        choices=["kv_twin", "kv_clean", "kv_noisy_steps", "image_cond"],
        default=None,
        help="Streaming context strategy (ablation). kv_twin (default) = per-block KV "
        "cache + RoPE repositioning, history reads noisy at mid steps + clean at the "
        "final step (Vidu S1 §2.3.1). kv_clean (ablation A) = KV cache reading clean "
        "history at every step. kv_noisy_steps (ablation B) = KV cache reading, at step "
        "t, the history's own step-t noisy snapshot (noise-level matched; no clean, no "
        "sigma-0 forward; ~num_steps/2x the KV memory of kv_twin — reduce "
        "--num-inference-steps / --window-chunks or add --offload). image_cond (ablation "
        "C) = no KV cache; each chunk conditions on the previous chunk's last frame as "
        "the image reference (rotating sink), no attention history.",
    )
    parser.add_argument(
        "--causal-cross-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a time-causal mask to the AV cross-attention (a2v video->audio and "
        "v2a audio->video), per Vidu S1 §2.3. Default ON for paper-faithful streaming "
        "causality; pass --no-causal-cross-attn to revert to full bidirectional "
        "cross-attention.",
    )
    parser.add_argument(
        "--cross-attn-lookahead-seconds",
        type=float,
        default=0.0,
        help="Seconds of future audio a video frame may attend to under --causal-cross-attn "
        "(0.0 = strict causal). In TI2V there is no frozen 'future audio' beyond the "
        "current window, so this only relaxes causality within the visible window.",
    )
    args = parser.parse_args()

    # Resolve the streaming strategy: --stream-strategy (explicit) wins,
    # otherwise the KV-cache TwinCache default.
    stream_strategy = args.stream_strategy or "kv_twin"

    pipeline = TI2VidStreamingPipeline(
        model_paths=args.model_paths,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        diffvae_optimization=args.diffvae_optimization,
    )
    tiling_config = TileSizeConfig.default()

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        images=args.images,
        window_chunks=args.window_chunks,
        chunk_frames=args.chunk_frames,
        use_kv_cache=args.use_kv_cache,
        stream_strategy=stream_strategy,
        causal_cross_attn=args.causal_cross_attn,
        cross_attn_lookahead_sec=args.cross_attn_lookahead_seconds,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    from ltx_core.model.video_vae import get_video_chunks_number

    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
