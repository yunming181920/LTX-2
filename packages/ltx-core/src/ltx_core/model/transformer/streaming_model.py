"""CausalStreamingModel — wraps an :class:`X0Model` to drive per-block KV caches.

Milestone 2 wrapper. Owns one :class:`StreamingKVCache` per transformer block's
self-attention and toggles them ``active`` around each forward so the cached
attention path (RoPE repositioning + TwinCache history) runs only for streaming,
leaving all production bidirectional pipelines untouched (their
``*.stream_cache`` stays ``None``).

Two modalities can be cached, selected at construction:

  * **Video** (always): one cache per block's video self-attention (``attn1``),
    with a 1-frame sink and a persistent first chunk (Vidu S1 §2.3.1). This is
    the original A2V behaviour — ``cache_audio=False`` leaves audio self-attention
    on the standard path, so A2V M2 is byte-identical to before.
  * **Audio** (opt-in, ``cache_audio=True``): one cache per block's audio
    self-attention (``audio_attn1``), with *no* sink and *no* persistent first
    chunk (audio has no image conditioning) — a pure FIFO ring. Required by the
    joint streaming TI2V path (M2), where audio is *generated* in lockstep and
    its self-attention must also be cached for O(window) memory.

The driver (``ltx_pipelines.utils.streaming``) calls:
  * :meth:`prepare_chunk` once per AR chunk with the full-window RoPE
    ``window_pe`` and block-causal ``query_mask`` (log-space additive bias, history
    query rows removed) for video — and, when caching audio, the equivalent pair
    for audio (no-sink layout ``[history | current]``);
  * :meth:`set_mode` per denoising step (``"noisy"`` mid-denoising, ``"clean"``
    final) — selects which TwinCache snapshot the rolling history reads;
  * :meth:`stash` at the mid step (``"noisy"``) and final step (``"clean"``),
    then :meth:`commit` to finalize the chunk in every cache;
  * :meth:`detach` when streaming ends, to remove the caches from the wrapped
    model's attention modules (restores the production forward path).

The query RoPE uses the modality's own ``[sink|current]`` (video) or ``[current]``
(audio) pe (passed to the self-attn as ``*.positional_embeddings``); only the key
RoPE uses the cache's full-window ``window_pe`` (RoPE repositioning).

The model's ``forward`` just activates the caches with the current chunk/mode
params, delegates to the wrapped ``X0Model``, then deactivates them.
"""

from __future__ import annotations

import torch

from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.streaming_cache import StreamingKVCache


class CausalStreamingModel(torch.nn.Module):
    """Wraps an X0Model and manages per-block self-attn KV caches.

    Caches video self-attention (``attn1``) always, and audio self-attention
    (``audio_attn1``) only when ``cache_audio=True``. A2V passes
    ``cache_audio=False`` and is byte-identical to the pre-audio-cache behaviour.
    """

    def __init__(
        self,
        x0_model: X0Model,
        window_chunks: int,
        tokens_per_frame: int,
        *,
        cache_audio: bool = False,
        audio_tokens_per_frame: int = 1,
    ) -> None:
        super().__init__()
        self.x0 = x0_model
        self.tokens_per_frame = tokens_per_frame
        self.audio_tokens_per_frame = audio_tokens_per_frame
        self.cache_audio = cache_audio
        blocks = self.x0.velocity_model.transformer_blocks
        # Video caches: 1-frame sink + persistent first chunk (Vidu S1 §2.3.1).
        self._caches: list[StreamingKVCache] = [
            StreamingKVCache(window_chunks, sink_tokens=tokens_per_frame, persistent_first=True)
            for _ in blocks
        ]
        for blk, cache in zip(blocks, self._caches):
            blk.attn1.stream_cache = cache
        # Audio caches: no sink, no persistent first chunk (pure FIFO ring).
        # Only attached when cache_audio=True; otherwise audio_attn1.stream_cache
        # stays None and audio self-attention runs the standard (uncached) path.
        self._audio_caches: list[StreamingKVCache] = []
        if cache_audio:
            self._audio_caches = [
                StreamingKVCache(window_chunks, sink_tokens=0, persistent_first=False)
                for _ in blocks
            ]
            for blk, cache in zip(blocks, self._audio_caches):
                blk.audio_attn1.stream_cache = cache
        # Per-chunk / per-step params (set by the driver).
        self._video_window_pe = None
        self._video_query_mask = None
        self._video_hist_len = 0
        self._audio_window_pe = None
        self._audio_query_mask = None
        self._audio_hist_len = 0
        self._mode: str = "clean"

    @property
    def num_blocks(self) -> int:
        return self.x0.num_blocks

    def reset(self) -> None:
        for cache in self._caches:
            cache.reset()
        for cache in self._audio_caches:
            cache.reset()
        self._video_window_pe = None
        self._video_query_mask = None
        self._video_hist_len = 0
        self._audio_window_pe = None
        self._audio_query_mask = None
        self._audio_hist_len = 0

    def detach(self) -> None:
        """Reset and remove the caches from the wrapped model's attn modules.

        After this, the wrapped ``X0Model`` is byte-identical to its
        pre-wrapping state (``attn1.stream_cache is None`` and, when audio was
        cached, ``audio_attn1.stream_cache is None`` -> standard path).
        """
        self.reset()
        for blk in self.x0.velocity_model.transformer_blocks:
            blk.attn1.stream_cache = None
            if self.cache_audio:
                blk.audio_attn1.stream_cache = None

    def prepare_chunk(
        self,
        *,
        window_pe,
        query_mask,
        hist_len: int,
        audio_window_pe=None,
        audio_query_mask=None,
        audio_hist_len: int | None = None,
    ) -> None:
        """Set per-AR-chunk RoPE/mask params (held until the next chunk).

        ``window_pe``/``query_mask``/``hist_len`` are the video params; the
        ``audio_*`` params are required only when ``cache_audio=True`` and are
        otherwise ignored (so the A2V call site is unchanged).
        """
        self._video_window_pe = window_pe
        self._video_query_mask = query_mask
        self._video_hist_len = hist_len
        if audio_window_pe is not None:
            self._audio_window_pe = audio_window_pe
        if audio_query_mask is not None:
            self._audio_query_mask = audio_query_mask
        if audio_hist_len is not None:
            self._audio_hist_len = audio_hist_len

    def set_mode(self, mode: str) -> None:
        """Select the TwinCache snapshot history reads (``"noisy"``/``"clean"``)."""
        self._mode = mode

    def stash(self, mode: str) -> None:
        """Snapshot every cache's current-chunk K/V into its pending entry."""
        for cache in self._caches:
            cache.stash(mode)
        for cache in self._audio_caches:
            cache.stash(mode)

    def commit(self) -> None:
        """Append every cache's pending TwinCache entry to its FIFO ring."""
        for cache in self._caches:
            cache.commit()
        for cache in self._audio_caches:
            cache.commit()

    def hist_len(self) -> int:
        """Current cached video-history token count (sink excluded; recomputed)."""
        return self._caches[0].hist_len if self._caches else 0

    def audio_hist_len(self) -> int:
        """Current cached audio-history token count (recomputed). 0 if not caching audio."""
        return self._audio_caches[0].hist_len if self._audio_caches else 0

    def forward(self, video, audio, perturbations):
        for cache in self._caches:
            cache.set_active(
                mode=self._mode,
                window_pe=self._video_window_pe,
                query_mask=self._video_query_mask,
                hist_len=self._video_hist_len,
                tokens_per_frame=self.tokens_per_frame,
            )
        for cache in self._audio_caches:
            cache.set_active(
                mode=self._mode,
                window_pe=self._audio_window_pe,
                query_mask=self._audio_query_mask,
                hist_len=self._audio_hist_len,
                tokens_per_frame=self.audio_tokens_per_frame,
            )
        try:
            return self.x0(video=video, audio=audio, perturbations=perturbations)
        finally:
            for cache in self._caches:
                cache.set_inactive()
            for cache in self._audio_caches:
                cache.set_inactive()
