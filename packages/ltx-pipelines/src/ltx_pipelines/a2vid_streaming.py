"""Streaming, autoregressive, causal audio-to-video (A2V) pipeline.

Training-free reproduction of Vidu S1 §2.3 streaming inference on top of the
pretrained *bidirectional* LTX-2 checkpoint (used as-is as the "causal model").

Milestone 1: block-causal self-attention mask + sliding-window decoding +
persistent reference context per Vidu S1 §2.3.1 (the encoded first-frame
latent "sink" plus the first generated chunk, both fixed and never evicted) +
latent-level TwinCache (noisy/clean history snapshots swapped per denoising
step) + frozen audio control (window-aligned slice, window-relative
positions). Generation is streaming internally (per-step activation memory is
O(window)); the full latent is decoded once at the end (causal-VAE seamless
decode) and streamed out.

No core (ltx-core) changes; reuses ``DiffusionStage.model_context`` for the
transformer lifecycle, ``PromptEncoder``/``ImageConditioner``/``AudioConditioner``
for IO, ``VideoDecoder`` for output, and the streaming primitives in
:mod:`ltx_pipelines.utils.streaming`.
"""

import logging
import math
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import VideoLatentTools
from ltx_core.types import Audio, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_1_stage_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.helpers import assert_resolution, get_device
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video, load_image_and_preprocess
from ltx_pipelines.utils.streaming import streaming_generate
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)


class A2VidStreamingPipeline:
    """Single-stage streaming audio-to-video pipeline (causal AR generation).

    The pretrained LTX-2 (full, non-distilled) checkpoint is used as the causal
    model. Audio is a frozen control signal; video is generated chunk-by-chunk
    with a sliding window, sink, and TwinCache (see
    :func:`ltx_pipelines.utils.streaming.streaming_generate`).
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
        self.audio_conditioner = AudioConditioner(
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
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        window_chunks: int = 4,
        chunk_frames: int = 1,
        audio_lookahead: int | None = None,
        use_kv_cache: bool = False,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        sigmas: torch.Tensor | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Generate a streaming causal A2V video. Returns (video_frames, audio).

        ``images[0]`` (frame_idx 0) is the sink reference. Audio is returned as
        the (frozen) input audio, matching the standard A2V convention.
        """
        assert_resolution(height=height, width=width, is_two_stage=False)
        if not images:
            raise ValueError("A2VidStreamingPipeline requires a reference image (frame_idx=0) as the sink.")
        if window_chunks < 1:
            raise ValueError(f"window_chunks must be >= 1, got {window_chunks}")
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = self.dtype

        # Text context (negative unused — no CFG in milestone 1, SimpleDenoiser logic).
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

        # Frozen audio control: full audio latent, sliced per AR chunk inside the loop.
        audio_max_duration = audio_max_duration if audio_max_duration is not None else num_frames / frame_rate
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}.")
        audio_latent_full = self.audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))

        sigmas = (sigmas if sigmas is not None else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32, device=self.device
        )

        pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate)
        v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        video_tools_full = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, frame_rate)
        num_generated_latent_frames = v_shape.frames - 1  # frame 0 is the sink
        if num_generated_latent_frames <= 0:
            raise ValueError(f"num_frames={num_frames} yields no frames to generate beyond the sink.")

        if audio_lookahead is None:
            audio_lookahead = math.ceil(200.0 / frame_rate)

        # Generation runs under the transformer's model context (DiT required);
        # decode uses a separate VideoDecoder and runs outside it.
        stepper = EulerDiffusionStep()
        with self.stage.model_context() as transformer:
            if use_kv_cache:
                from ltx_pipelines.utils.streaming import streaming_generate_cached

                full_latent = streaming_generate_cached(
                    sigmas=sigmas,
                    num_generated_latent_frames=num_generated_latent_frames,
                    chunk_frames=chunk_frames,
                    window_chunks=window_chunks,
                    video_tools_full=video_tools_full,
                    audio_latent_full=audio_latent_full,
                    audio_lookahead=audio_lookahead,
                    sink_latent_unpatchified=sink_latent,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    stepper=stepper,
                    transformer=transformer,
                    noiser=noiser,
                    dtype=dtype,
                    device=self.device,
                )
            else:
                full_latent = streaming_generate(
                    sigmas=sigmas,
                    num_generated_latent_frames=num_generated_latent_frames,
                    chunk_frames=chunk_frames,
                    window_chunks=window_chunks,
                    video_tools_full=video_tools_full,
                    audio_latent_full=audio_latent_full,
                    audio_lookahead=audio_lookahead,
                    sink_latent_unpatchified=sink_latent,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    stepper=stepper,
                    transformer=transformer,
                    noiser=noiser,
                    dtype=dtype,
                    device=self.device,
                )

        decoded_video = self.video_decoder(full_latent, tiling_config, generator=generator)
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate
        )
        return decoded_video, original_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = default_1_stage_arg_parser(params=params)
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file used to drive the video generation (frozen control signal).",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds. Defaults to the video duration (num_frames / frame_rate).",
    )
    parser.add_argument(
        "--window-chunks",
        type=int,
        default=4,
        help="Sliding-window rolling-history size in AR chunks (TwinCache FIFO cap; "
        "the sink and the first generated chunk are persistent and not counted). Default 4.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=1,
        help="Latent video frames generated per AR step (default 1 = finest streaming granularity).",
    )
    parser.add_argument(
        "--audio-lookahead",
        type=int,
        default=None,
        help="Extra audio latent frames visible ahead of the current chunk's end "
        "(default ceil(200/fps), one video-frame's worth).",
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="Milestone 2: use the KV-cache + RoPE-repositioning path (faster; history "
        "K/V are reused from each chunk's own denoising pass per Vidu S1 §2.3.1, so "
        "results differ slightly from the default full-recompute path — run the "
        "parity/smoke test before trusting). Default off (M1 latent-level TwinCache).",
    )
    args = parser.parse_args()

    pipeline = A2VidStreamingPipeline(
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
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=args.audio_max_duration,
        window_chunks=args.window_chunks,
        chunk_frames=args.chunk_frames,
        audio_lookahead=args.audio_lookahead,
        use_kv_cache=args.use_kv_cache,
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
