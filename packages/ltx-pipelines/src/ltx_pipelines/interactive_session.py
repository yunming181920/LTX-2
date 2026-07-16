"""Interactive streaming session for LTX-2 TI2V — pre-encoded prompt playlist + streaming A/V.

Wraps :class:`ltx_pipelines.ti2vid_streaming.TI2VidStreamingPipeline`'s building blocks
into a **long-lived** session so the 22B DiT, the 12B Gemma text encoder, and the
audio/video VAEs are built **once** and reused across chunks, across prompt switches,
and across multiple generations (no reload per chunk or per prompt).

Two responsibilities on top of the model lifecycle:

  * **Pre-encoded prompt playlist** (:class:`PromptBank`). Text conditioning is
    cross-attention (see :mod:`ltx_pipelines.utils.streaming_interactive`), so
    switching the prompt mid-stream only rewrites the cross-attention conditioning of
    subsequent chunks and leaves the cached self-attention history / sink untouched.
    :meth:`InteractiveStreamingSession.prefetch_prompts` pre-encodes a list of prompts
    once (single batched Gemma forward via :class:`LivePromptEncoder`) and caches them
    in the bank; :meth:`InteractiveStreamingSession.advance_prompt` (the UI's **Next
    Prompt** button) queues a one-step advance that the run loop applies at the next
    chunk boundary, then returns the bank's current cached context (no re-encode per
    chunk). Generation starts on prompt #1 and keeps running on the current prompt
    until the next advance; the last prompt is a hard clamp (no loop).
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
  * **Single active generation**: one playlist, one GPU — not a multi-tenant server.
    Advances apply at most one per chunk; rapid Next clicks queue instead of skipping.
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


class PromptBank:
    """Thread-safe ordered playlist of pre-encoded prompts.

    The UI hands :meth:`InteractiveStreamingSession.prefetch_prompts` a list of
    prompts up front; each is encoded once (via :class:`LivePromptEncoder`) and
    stored here as ``(prompt, (v_context, a_context))``. Generation starts at index
    0 (the first prompt) and keeps returning that prompt's cached context every
    chunk until an advance is applied. The UI's **Next Prompt** button calls
    :meth:`request_advance`; the run loop applies at most one queued advance per AR
    chunk via :meth:`drain_one_advance` (one step per click, so rapid double-clicks
    queue instead of skipping a prompt). The last entry is a hard clamp — no loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[tuple[str, tuple[torch.Tensor, torch.Tensor]]] = []
        self._index = 0
        self._pending = 0  # queued advances not yet applied at a chunk boundary

    def prefetch(self, entries: list[tuple[str, tuple[torch.Tensor, torch.Tensor]]]) -> int:
        """Load a fresh pre-encoded playlist and reset to the first prompt."""
        with self._lock:
            self._entries = list(entries)
            self._index = 0
            self._pending = 0
            return len(self._entries)

    def request_advance(self) -> None:
        """Queue one advance for the next chunk boundary (clamped at the last entry)."""
        with self._lock:
            if self._index + self._pending < len(self._entries) - 1:
                self._pending += 1

    def drain_one_advance(self) -> bool:
        """Apply at most one queued advance. Returns True if the index moved."""
        with self._lock:
            if self._pending > 0 and self._index < len(self._entries) - 1:
                self._pending -= 1
                self._index += 1
                return True
            # At the last entry: drop any leftover queued advances (clamped, no loop).
            self._pending = 0
            return False

    def current_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            return self._entries[self._index][1]

    def current_prompt(self) -> str:
        with self._lock:
            return self._entries[self._index][0] if self._entries else ""

    @property
    def index(self) -> int:
        with self._lock:
            return self._index

    @property
    def total(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def queued(self) -> int:
        with self._lock:
            return self._pending


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

    def encode_many(
        self, prompts: list[str], *, enhance: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-encode a list of positive prompts → ``[(v_context, a_context), ...]``.

        Runs a single batched Gemma forward (``text_encoder.encode`` batches) and
        processes every row through the embeddings processor — the same path as
        :meth:`encode`, just consuming all rows instead of the first. With
        ``enhance``, each prompt is first rewritten via ``generate_enhanced_prompt``.
        Used once at generation start to populate a :class:`PromptBank`.
        """
        if not prompts:
            return []
        to_encode = prompts
        if enhance:
            to_encode = [
                generate_enhanced_prompt(
                    self._text_encoder, p, self._enhance_image, seed=self._enhance_seed
                )
                for p in prompts
            ]
        raw_outputs = self._text_encoder.encode(to_encode)
        contexts: list[tuple[torch.Tensor, torch.Tensor]] = []
        for hs, mask in raw_outputs:
            out = self._embeddings_processor.process_hidden_states(hs, mask)
            contexts.append((out.video_encoding, out.audio_encoding))
        return contexts

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
    generate over a pre-encoded prompt playlist + streaming output, and :meth:`stop`
    to free. Prompts are pre-encoded via :meth:`prefetch_prompts` (called at the start
    of :meth:`run`); the UI advances the playlist via :meth:`advance_prompt`
    (thread-safe, safe during generation).
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

        self.prompt_bank = PromptBank()
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

    # -- prompt playlist -----------------------------------------------------

    def prefetch_prompts(self, prompts: list[str], *, enhance: bool = False) -> int:
        """Pre-encode a list of prompts once and cache them in the bank.

        Drops empty lines (order preserved), encodes all prompts in a single batched
        Gemma forward via :meth:`LivePromptEncoder.encode_many`, loads them into the
        :class:`PromptBank`, and resets the active prompt to the first one. Must be
        called after :meth:`start` and before / at the start of :meth:`run`.
        Returns the number of prompts cached.
        """
        if self._live_encoder is None:
            raise RuntimeError("Session not started — call start() before prefetch_prompts().")
        cleaned = [p.strip() for p in prompts if p and p.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty prompt is required.")
        contexts = self._live_encoder.encode_many(cleaned, enhance=enhance)
        self.prompt_bank.prefetch(list(zip(cleaned, contexts)))
        self._active_prompt = cleaned[0]
        logger.info("Pre-encoded %d prompts; starting with %r", len(cleaned), cleaned[0])
        return len(cleaned)

    def advance_prompt(self) -> str:
        """Queue a one-step advance to the next prompt (clamped at the last).

        Thread-safe — safe to call from a UI button while a generation is running.
        The advance is applied at the next AR chunk boundary. Returns a status line
        for the UI describing the current / queued state.
        """
        self.prompt_bank.request_advance()
        return self.prompt_status()

    def prompt_status(self) -> str:
        """Human-readable playlist state for the UI (current index / total / queued)."""
        total = self.prompt_bank.total
        if total == 0:
            return "Prompts: 0 loaded."
        idx = self.prompt_bank.index
        queued = self.prompt_bank.queued
        text = self.prompt_bank.current_prompt()
        head = f"▶ Prompt **{idx + 1}/{total}**"
        if queued:
            head += f"  (⏭ +{queued} queued → next: **{min(idx + queued, total - 1) + 1}/{total}**)"
        return f"{head}: {text}"

    @property
    def active_prompt(self) -> str:
        return self._active_prompt

    # -- generation ----------------------------------------------------------

    def run(  # noqa: PLR0913, PLR0915
        self,
        *,
        prompts: list[str],
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
        """Generate over a pre-encoded prompt playlist, yielding a growing clip + audio.

        Yields one :class:`StreamUpdate` per AR chunk: a freshly re-encoded mp4 of
        the clip so far, the newly-added trailing audio samples, and a status line.
        All ``prompts`` are pre-encoded once at the start and cached in the
        :class:`PromptBank`; generation runs on the first prompt. At each chunk
        boundary the bank applies at most one queued advance (from the UI's **Next
        Prompt** button) and returns the current prompt's cached cross-attention
        context — so the model keeps running on the current prompt until the next
        advance, and a switch affects only subsequent chunks.
        """
        if not self._started or self._transformer is None or self._live_encoder is None:
            raise RuntimeError("Session not started — call start() before run().")
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

            # Pre-encode ALL prompts once and cache them in the bank (12B Gemma, so
            # encode eagerly + once). Resets the active prompt to the first one.
            self.prefetch_prompts(prompts, enhance=enhance_prompt)

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

            # Playlist context resolver: apply at most one queued advance (from the
            # UI's Next Prompt button) at each chunk boundary, then return the bank's
            # current cached context. No advance pending → the current prompt's
            # context is reused every chunk (keeps running on it). All contexts are
            # pre-encoded, so this is a pure lookup — no Gemma call per chunk.
            def context_resolver(i: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
                del i, n
                if self.prompt_bank.drain_one_advance():
                    self._active_prompt = self.prompt_bank.current_prompt()
                    logger.info("Prompt advance -> %r", self._active_prompt)
                return self.prompt_bank.current_context()

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
