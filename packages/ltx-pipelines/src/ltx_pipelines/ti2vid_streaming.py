"""Streaming, autoregressive, causal text/image-to-video+audio (TI2V) pipeline.

Training-free reproduction of Vidu S1 §2.3 streaming inference on top of the
pretrained *bidirectional* LTX-2 checkpoint (used as-is as the "causal model").

Unlike an audio-to-video (A2V) streaming setup (where audio is a frozen control
signal and only video is generated), TI2V has no audio input:
it generates **both** video and audio. This pipeline generates them chunk by
chunk in lockstep — each video AR chunk also produces its time-aligned audio
latent frames — with a sliding window, persistent reference context per Vidu S1
§2.3.1 (the encoded first-frame "sink" plus the first generated *video* chunk,
fixed and never evicted), latent-level TwinCache (noisy/clean history snapshots
swapped per denoising step) for **both** modalities, block-causal self-attention
masks on both temporal axes, and a time-causal video↔audio cross-attention mask.

Audio keeps its own sliding-window FIFO history (no sink / no persistent anchor
— audio has no image conditioning), so per-step activation memory is O(window)
for both modalities. The full latents are decoded once at the end (causal-VAE
seamless video decode + audio decode) and returned.

Two paths:
  * **M1** (default): latent TwinCache, full per-step recompute of history
    features — the correct, recommended path.
  * **M2** (``--use-kv-cache``): per-block KV cache + RoPE repositioning for
    *both* video and audio self-attention. Faster, but conceptual/unvalidated
    (extends an already-untested A2V M2 path to a second modality with a
    sink-less layout) — run ``tests/test_streaming_joint_parity.py`` in a GPU
    env before trusting.

No core (ltx-core) production changes; reuses ``DiffusionStage.model_context``
for the transformer lifecycle, ``PromptEncoder``/``ImageConditioner`` for IO,
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
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
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
from ltx_pipelines.utils.helpers import assert_resolution, get_device
from ltx_pipelines.utils.media_io import encode_video, load_image_and_preprocess
from ltx_pipelines.utils.streaming import streaming_generate_joint
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)


class TI2VidStreamingPipeline:
    """Single-stage streaming text/image-to-video+audio pipeline (causal AR).

    The pretrained LTX-2 (full, non-distilled) checkpoint is used as the causal
    model. Video and audio are generated chunk-by-chunk in lockstep, each with a
    sliding window + TwinCache history (see
    :func:`ltx_pipelines.utils.streaming.streaming_generate_joint`).
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ):
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self._scheduler = LTX2Scheduler()
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
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
        window_chunks: int = 4,
        chunk_frames: int = 1,
        use_kv_cache: bool = False,
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

        # Text context (negative unused — no CFG in M1, SimpleDenoiser logic).
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
        v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        video_tools_full = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, frame_rate)
        audio_tools_full = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
        num_generated_latent_frames = v_shape.frames - 1  # frame 0 is the sink
        if num_generated_latent_frames <= 0:
            raise ValueError(f"num_frames={num_frames} yields no frames to generate beyond the sink.")

        # Generation runs under the transformer's model context (DiT required);
        # decode uses separate decoders and runs outside it.
        stepper = EulerDiffusionStep()
        with self.stage.model_context() as transformer:
            if use_kv_cache:
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
                )
            else:
                full_video_latent, full_audio_latent = streaming_generate_joint(
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
                )

        decoded_video = self.video_decoder(full_video_latent, tiling_config, generator=generator)
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
        default=4,
        help="Sliding-window rolling-history size in AR chunks (TwinCache FIFO cap; "
        "the video sink and the first generated video chunk are persistent and not "
        "counted). Audio uses the same cap for its own FIFO history. Default 4.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=1,
        help="Latent video frames generated per AR step (default 1 = finest streaming "
        "granularity). The time-aligned audio latent frames for each chunk are generated "
        "in lockstep (~8/fps*25 audio frames per video latent frame).",
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="M2: use the KV-cache + RoPE-repositioning path for BOTH video and audio "
        "self-attention (faster; history K/V are reused from each chunk's own denoising "
        "pass per Vidu S1 §2.3.1, so results differ slightly from the default "
        "full-recompute path — run the joint parity/smoke test before trusting). Default "
        "off (M1 latent-level TwinCache).",
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

    pipeline = TI2VidStreamingPipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
    tiling_config = TilingConfig.default()

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
