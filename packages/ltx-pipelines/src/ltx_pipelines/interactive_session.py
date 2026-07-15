"""Interactive streaming session for LTX-2 TI2V — live prompt injection + streaming A/V.

Wraps :class:`ltx_pipelines.ti2vid_streaming.TI2VidStreamingPipeline`'s building blocks
into a **long-lived** session so the 22B DiT, the 12B Gemma text encoder, and the
audio/video VAEs are built **once** and reused across chunks, across prompt changes,
and across multiple generations (no reload per chunk or per prompt).

Two responsibilities on top of the model lifecycle:

  * **Live prompt injection** (:class:`PromptSlot`). Text conditioning is
    cross-attention (see :mod:`ltx_pipelines.utils.streaming_interactive`), so
    changing the prompt mid-stream only rewrites the cross-attention conditioning of
    subsequent chunks and leaves the cached self-attention history / sink untouched.
    :meth:`InteractiveStreamingSession.submit_prompt` queues a new prompt; the run
    loop drains the queue at each chunk boundary and re-encodes via the kept-alive
    Gemma (:class:`LivePromptEncoder`, which caches the last prompt so an unchanged
    prompt is not re-encoded).
  * **Streaming output**. :meth:`InteractiveStreamingSession.run` is a generator
    that drives :func:`iter_streaming_chunks_joint` and, after each AR chunk,
    decodes the growing latent **prefix** through the kept-alive VAEs and re-encodes
    the assembled clip so far to a temp mp4 — yielding ``(video_path, audio_chunk,
    status)`` so a UI can show a growing video, stream live audio, and report
    progress + the active prompt.

Design notes / trade-offs (see the plan for detail):

  * **Growing-prefix decode** (O(n²) VAE): the full filled latent prefix is decoded
    each chunk so the result is seamless (``decode_video`` does temporal tiling with
    overlap internally). Decoding only the new frame is a future optimization.
  * **M1 path only** (the recommended, correct path). An M2 interactive variant can
    reuse this decode/prompt layer later.
  * **Single active generation**: one prompt slot, one GPU — not a multi-tenant
    server. The queue holds at most one pending prompt (the latest wins).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import decode_audio
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.helpers import assert_resolution, cleanup_memory, generate_enhanced_prompt, get_device
from ltx_pipelines.utils.media_io import encode_video, load_image_and_preprocess
from ltx_pipelines.utils.streaming_interactive import StreamChunk, iter_streaming_chunks_joint
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)


class PromptSlot:
    """Thread-safe single-slot queue for the latest pending prompt.

    The UI writes the live prompt textbox value here (`.change()`); the generation
    loop drains it at each chunk boundary. At most one prompt is pending — a later
    ``submit`` overwrites an earlier unread one (the newest prompt wins).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: str | None = None

    def submit(self, prompt: str) -> None:
        prompt = (prompt or "").strip()
        with self._lock:
            self._pending = prompt

    def drain(self) -> str | None:
        with self._lock:
            p = self._pending
            self._pending = None
            return p


class LivePromptEncoder:
    """Gemma text encoder + embeddings processor kept alive for the session.

    Reuses :class:`PromptEncoder`'s configured builders (so the gemma-root module
    ops, sd_ops and the embeddings-processor configurator are identical to the
    standard path) but builds each model once and holds it, then re-encodes prompts
    on demand — replicating ``PromptEncoder.__call__``'s encode logic
    (``text_encoder.encode`` → ``embeddings_processor.process_hidden_states``)
    without rebuilding/freeing. Caches the last encoded prompt so an unchanged
    prompt is not re-encoded (Gemma is 12B; re-encoding only on real change matters).
    """

    def __init__(
        self,
        prompt_encoder: PromptEncoder,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> None:
        self._prompt_encoder = prompt_encoder
        self._enhance_image = enhance_prompt_image
        self._enhance_seed = enhance_prompt_seed
        self._text_encoder = prompt_encoder._build_text_encoder()
        self._embeddings_processor = prompt_encoder._build_embeddings_processor()
        self._last_prompt: str | None = None
        self._last_context: tuple[torch.Tensor, torch.Tensor] | None = None

    def encode(self, prompt: str, *, enhance: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single positive prompt → ``(v_context, a_context)``.

        No CFG in the streaming M1 path (negative prompt unused), so only the
        positive prompt is encoded. ``enhance`` rewrites the prompt via Gemma
        (``generate_enhanced_prompt``) before encoding.
        """
        if prompt == self._last_prompt and self._last_context is not None and not enhance:
            return self._last_context

        prompts = [prompt]
        if enhance:
            prompts = [
                generate_enhanced_prompt(
                    self._text_encoder, prompt, self._enhance_image, seed=self._enhance_seed
                )
            ]
        raw_outputs = self._text_encoder.encode(prompts)
        (out,) = [self._embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]
        context = (out.video_encoding, out.audio_encoding)
        self._last_prompt = prompt
        self._last_context = context
        return context

    def free(self) -> None:
        for model in (self._text_encoder, self._embeddings_processor):
            if model is not None:
                model.to("meta")
        cleanup_memory()


@dataclass
class StreamUpdate:
    """One incremental emission from :meth:`InteractiveStreamingSession.run`.

    ``video_path`` points at a freshly re-encoded mp4 of the clip so far (growing);
    ``audio`` is ``(sample_rate, np.ndarray)`` of the *newly generated* trailing
    audio samples for live streaming (shape ``(samples, channels)``);
    ``status`` is a human-readable progress line.
    """

    video_path: str
    audio: tuple[int, np.ndarray]
    status: str


class InteractiveStreamingSession:
    """Long-lived interactive streaming session (build once, generate many).

    Construct with the same config as :class:`TI2VidStreamingPipeline`, then
    :meth:`start` to build all models (kept resident), :meth:`run` (a generator) to
    generate with live prompt injection + streaming output, and :meth:`stop` to free.
    The live prompt is fed via :meth:`submit_prompt` (thread-safe).
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
    ) -> None:
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

        self.prompt_slot = PromptSlot()
        self._live_encoder: LivePromptEncoder | None = None
        self._transformer_ctx = None
        self._transformer = None
        self._video_vae = None
        self._audio_vae_decoder = None
        self._vocoder = None
        self._active_prompt: str = ""
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    def start(self, *, enhance_prompt_image: str | None = None, enhance_prompt_seed: int = 42) -> None:
        """Build all models once and hold them resident for the session.

        Keeping the DiT, Gemma, and both VAEs resident avoids reloading between
        chunks, prompt changes, and generations. With ``offload_mode != NONE`` the
        DiT streams from CPU/disk but Gemma is still resident (12B) — on smaller
        GPUs prefer ``offload_mode=NONE`` + ``--quantization fp8-cast``.
        """
        if self._started:
            return
        logger.info("Building streaming session models (DiT + Gemma + VAEs)...")
        # DiT: hold the model_context() open for the whole session.
        self._transformer_ctx = self.stage.model_context()
        self._transformer = self._transformer_ctx.__enter__()
        self._live_encoder = LivePromptEncoder(
            self.prompt_encoder,
            enhance_prompt_image=enhance_prompt_image,
            enhance_prompt_seed=enhance_prompt_seed,
        )
        # VAEs: build once via the decoders' configured builders and hold.
        self._video_vae = self.video_decoder._decoder_builder.build(
            device=self.device, dtype=self.dtype
        ).eval()
        vocoder_dtype = torch.float32 if self.device.type == "mps" else self.dtype
        self._audio_vae_decoder = self.audio_decoder._decoder_builder.build(
            device=self.device, dtype=self.dtype
        ).eval()
        self._vocoder = self.audio_decoder._vocoder_builder.build(
            device=self.device, dtype=vocoder_dtype
        ).eval()
        self._started = True
        logger.info("Streaming session ready.")

    def stop(self) -> None:
        """Free all resident models (reverse order)."""
        for model in (self._vocoder, self._audio_vae_decoder, self._video_vae):
            if model is not None:
                model.to("meta")
        if self._live_encoder is not None:
            self._live_encoder.free()
        cleanup_memory()
        if self._transformer_ctx is not None:
            self._transformer_ctx.__exit__(None, None, None)
            self._transformer_ctx = None
            self._transformer = None
        self._started = False

    # -- live prompt ---------------------------------------------------------

    def submit_prompt(self, prompt: str) -> str:
        """Queue a new prompt for the next chunk boundary. Returns the queued text."""
        self.prompt_slot.submit(prompt)
        return (prompt or "").strip()

    @property
    def active_prompt(self) -> str:
        return self._active_prompt

    # -- generation ----------------------------------------------------------

    def run(  # noqa: PLR0913, PLR0915
        self,
        *,
        initial_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        image_path: str,
        image_crf: int = 15,
        window_chunks: int = 4,
        chunk_frames: int = 1,
        enhance_prompt: bool = False,
        causal_cross_attn: bool = True,
        cross_attn_lookahead_sec: float = 0.0,
        tiling_config: TilingConfig | None = None,
        output_dir: str | None = None,
    ) -> Generator[StreamUpdate, None, None]:
        """Generate with live prompt injection, yielding a growing clip + streaming audio.

        Yields one :class:`StreamUpdate` per AR chunk: a freshly re-encoded mp4 of
        the clip so far, the newly-added trailing audio samples, and a status line.
        At each chunk boundary the live prompt slot is drained; if a new prompt
        arrived it is re-encoded (cached otherwise) and applied to the next chunk's
        cross-attention only.
        """
        if not self._started or self._transformer is None or self._live_encoder is None:
            raise RuntimeError("Session not started — call start() before run().")
        live = self._live_encoder
        assert_resolution(height=height, width=width, is_two_stage=False)
        if not image_path:
            raise ValueError("A reference image (the sink) is required.")
        if window_chunks < 1:
            raise ValueError(f"window_chunks must be >= 1, got {window_chunks}")
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")

        # Keep inference_mode active across all yields (the generator frame persists).
        im_ctx = torch.inference_mode()
        im_ctx.__enter__()
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            dtype = self.dtype

            # Encode the initial prompt (sets the active prompt + caches it).
            self._active_prompt = initial_prompt.strip()
            v_context, a_context = live.encode(initial_prompt, enhance=enhance_prompt)

            # Sink: encode the reference first-frame image to a video latent.
            sink_latent = self.image_conditioner(
                lambda enc: enc(
                    load_image_and_preprocess(
                        image_path=image_path,
                        height=height,
                        width=width,
                        dtype=dtype,
                        device=self.device,
                        crf=image_crf,
                    )
                )
            )  # (1, C, 1, H_lat, W_lat)

            sigmas = self._scheduler.execute(steps=num_inference_steps).to(
                dtype=torch.float32, device=self.device
            )

            pixel_shape = VideoPixelShape(
                batch=1, frames=num_frames, height=height, width=width, fps=frame_rate
            )
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            video_tools_full = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, frame_rate)
            audio_tools_full = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
            num_generated_latent_frames = v_shape.frames - 1  # frame 0 is the sink
            if num_generated_latent_frames <= 0:
                raise ValueError(f"num_frames={num_frames} yields no frames to generate beyond the sink.")

            stepper = EulerDiffusionStep()

            # Live-prompt context resolver: drain the slot at each chunk boundary.
            # On change, re-encode (Gemma kept alive) and update the active prompt;
            # otherwise reuse the cached context (no re-encode). The initial encode
            # above guarantees ``live._last_context`` is set for the cached path.
            def context_resolver(i: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
                del i, n
                pending = self.prompt_slot.drain()
                if pending and pending != self._active_prompt:
                    logger.info("Live prompt change: %r -> %r", self._active_prompt, pending)
                    self._active_prompt = pending
                    return live.encode(pending)
                return live._last_context if live._last_context is not None else (v_context, a_context)

            out_dir = output_dir or tempfile.mkdtemp(prefix="ltx2_stream_")
            prev_video_path: str | None = None
            prev_audio_len = 0

            for chunk in iter_streaming_chunks_joint(
                sigmas=sigmas,
                num_generated_latent_frames=num_generated_latent_frames,
                chunk_frames=chunk_frames,
                window_chunks=window_chunks,
                video_tools_full=video_tools_full,
                audio_tools_full=audio_tools_full,
                sink_latent_unpatchified=sink_latent,
                context_resolver=context_resolver,
                stepper=stepper,
                transformer=self._transformer,
                noiser=noiser,
                dtype=dtype,
                device=self.device,
                causal_cross_attn=causal_cross_attn,
                cross_attn_lookahead_sec=cross_attn_lookahead_sec,
            ):
                update, prev_audio_len = self._emit_chunk(
                    chunk=chunk,
                    frame_rate=frame_rate,
                    tiling_config=tiling_config,
                    generator=generator,
                    out_dir=out_dir,
                    prev_video_path=prev_video_path,
                    prev_audio_len=prev_audio_len,
                )
                prev_video_path = update.video_path
                yield update
        finally:
            im_ctx.__exit__(None, None, None)

    # -- incremental decode + re-encode --------------------------------------

    def _emit_chunk(  # noqa: PLR0913
        self,
        *,
        chunk: StreamChunk,
        frame_rate: float,
        tiling_config: TilingConfig | None,
        generator: torch.Generator,
        out_dir: str,
        prev_video_path: str | None,
        prev_audio_len: int,
    ) -> tuple[StreamUpdate, int]:
        """Decode the growing prefixes, re-encode the assembled clip, build the update.

        Returns ``(update, cumulative_audio_len)`` so the caller tracks the audio
        cursor across chunks for streaming the trailing samples.
        """
        # Video: decode the filled latent prefix (seamless via tiled decode).
        frame_chunks = list(
            self._video_vae.decode_video(  # type: ignore[union-attr]
                chunk.video_latent_prefix, tiling_config, generator=generator
            )
        )
        all_frames = (
            torch.cat([f.to(torch.float32) for f in frame_chunks], dim=0)
            if frame_chunks
            else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        )  # (F, H, W, C) in [0, 1]

        # Audio: decode the filled latent prefix → full waveform so far.
        audio: Audio = decode_audio(
            chunk.audio_latent_prefix, self._audio_vae_decoder, self._vocoder  # type: ignore[arg-type]
        )
        wf = audio.waveform  # (channels, samples) stereo per _validate_audio_waveform
        if wf.dim() == 1:
            wf_for_mux = wf.unsqueeze(0).repeat(2, 1)
            new_samples = wf
        else:
            # Stereo may be (2, N) or (N, 2) — normalize to (2, N) for muxing.
            wf_for_mux = wf if wf.shape[0] <= wf.shape[1] else wf.t()
            new_samples = wf_for_mux
        cur_audio_len = new_samples.shape[-1]
        trailing = new_samples[..., prev_audio_len:]
        # gr.Audio(streaming=True) takes (sample_rate, data) with data (samples, channels).
        trailing_np = np.ascontiguousarray(trailing.t().cpu().numpy())

        # Re-encode the assembled clip so far (growing video, muxed accumulated audio).
        mux_audio = Audio(waveform=wf_for_mux, sampling_rate=audio.sampling_rate)
        out_path = os.path.join(out_dir, f"growing_{chunk.chunk_index:04d}.mp4")
        encode_video(
            video=all_frames,
            fps=int(frame_rate),
            audio=mux_audio,
            output_path=out_path,
            video_chunks_number=1,
        )
        # Best-effort cleanup of the previous growing file (Gradio has already served it).
        if prev_video_path is not None and prev_video_path != out_path and os.path.exists(prev_video_path):
            try:
                os.remove(prev_video_path)
            except OSError:
                pass

        status = (
            f"▶ chunk **{chunk.chunk_index + 1}/{chunk.num_chunks}** · "
            f"{int(all_frames.shape[0])} frames · active prompt: **{self._active_prompt}**"
        )
        update = StreamUpdate(
            video_path=out_path,
            audio=(audio.sampling_rate, trailing_np),
            status=status,
        )
        return update, cur_audio_len
