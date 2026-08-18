"""CausalStreamingModel — wraps an :class:`X0Model` to drive per-block KV caches.

Milestone 2 wrapper. Owns one :class:`StreamingKVCache` per transformer block's
self-attention and toggles them ``active`` around each forward so the cached
attention path (RoPE repositioning + TwinCache history) runs only for streaming,
leaving all production bidirectional pipelines untouched (their
``*.stream_cache`` stays ``None``).

Two modalities can be cached, selected at construction:

  * **Video** (always): one cache per block's video self-attention (``attn1``)
    with a persistent first chunk. The sink-carrying default (``video_sink_tokens
    = None`` → one latent frame) is the original A2V behaviour —
    ``cache_audio=False`` leaves audio self-attention on the standard path, so
    A2V M2 is byte-identical to before. The joint TI2V path passes
    ``video_sink_tokens=0``: its first committed chunk is the ``[image |
    chunk 1]`` anchor from a bidirectional ti2v bootstrap, so there is no
    separate sink block to recompute.
  * **Audio** (opt-in, ``cache_audio=True``): one cache per block's audio
    self-attention (``audio_attn1``), with *no* sink (audio has no image
    conditioning) but **with** a persistent first chunk — Vidu S1 §2.3.1's
    persistent reference is the first generated *video-audio* state, so audio
    keeps its first chunk forever and only later chunks roll through the FIFO
    ring. Required by the joint streaming TI2V path (M2), where audio is
    *generated* in lockstep and its self-attention must also be cached for
    O(window) memory.

The driver (``ltx_pipelines.utils.streaming``) calls:
  * :meth:`prepare_chunk` once per AR chunk with the full-window RoPE
    ``window_pe`` and block-causal ``query_mask`` (structured ``BlockCausalMask``, history
    query rows removed; served by unmasked prefix calls, FlashAttention-capable) for video — and, when caching audio, the equivalent pair
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
    (``audio_attn1``) only when ``cache_audio=True``. Both cache flavours pin
    their first generated chunk. The joint TI2V path bootstraps its first chunk
    with standard bidirectional ti2v (image at frame 0) and pins the whole
    ``[image | chunk 1]`` anchor, so it passes ``video_sink_tokens=0`` — the
    video cache then uses the same no-sink pinned-first layout as audio. The
    sink-carrying default (``video_sink_tokens=None`` → one latent frame)
    preserves the A2V behaviour, which passes ``cache_audio=False``.
    """

    def __init__(
        self,
        x0_model: X0Model,
        window_chunks: int,
        tokens_per_frame: int,
        *,
        video_sink_tokens: int | None = None,
        cache_audio: bool = False,
        audio_tokens_per_frame: int = 1,
        strategy: str = "twin",
        num_steps: int = 0,
        cache_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.x0 = x0_model
        self.tokens_per_frame = tokens_per_frame
        self.audio_tokens_per_frame = audio_tokens_per_frame
        self.cache_audio = cache_audio
        self.cache_cross_attn = cache_cross_attn
        self.strategy = strategy
        self.num_steps = num_steps
        blocks = self.x0.velocity_model.transformer_blocks
        # Video caches: sink_tokens = one latent frame by default (A2V: 1-frame
        # sink + persistent first chunk, Vidu S1 §2.3.1). The joint TI2V path
        # passes video_sink_tokens=0: its first committed chunk is the
        # [image | chunk 1] anchor (bidirectional ti2v bootstrap), so no
        # separate sink block is carried.
        video_sink = tokens_per_frame if video_sink_tokens is None else video_sink_tokens
        self._caches: list[StreamingKVCache] = [
            StreamingKVCache(
                window_chunks,
                sink_tokens=video_sink,
                persistent_first=True,
                strategy=strategy,
                num_steps=num_steps,
            )
            for _ in blocks
        ]
        for blk, cache in zip(blocks, self._caches):
            blk.attn1.stream_cache = cache
        # Audio caches: no sink (audio has no image conditioning) but a
        # persistent first chunk — Vidu S1 §2.3.1's persistent reference is the
        # first generated video-audio state, so both modalities pin chunk 1 and
        # only later chunks roll through the FIFO ring. Only attached when
        # cache_audio=True; otherwise audio_attn1.stream_cache stays None and
        # audio self-attention runs the standard (uncached) path.
        self._audio_caches: list[StreamingKVCache] = []
        if cache_audio:
            self._audio_caches = [
                StreamingKVCache(
                    window_chunks,
                    sink_tokens=0,
                    persistent_first=True,
                    strategy=strategy,
                    num_steps=num_steps,
                )
                for _ in blocks
            ]
            for blk, cache in zip(blocks, self._audio_caches):
                blk.audio_attn1.stream_cache = cache
        # Cross-attention caches (opt-in, ``cache_cross_attn=True``): one per
        # block's a2v (Q=video, K=audio) and v2a (Q=audio, K=video) modules.
        # No sink (cross-attn has no image conditioning) but a persistent first
        # chunk, mirroring the audio self-attn layout. With ``sink_tokens=0``
        # the existing cached self-attn path (``_stream_cached_forward``)
        # splices ``[hist | cur]`` of the *other* modality and applies key
        # RoPE from the cache's ``window_pe`` (full-window cross-PE) and query
        # RoPE from the passed ``pe`` (current chunk's cross-PE), so no changes
        # to ``attention.py``/``streaming_cache.py`` are needed.
        self._a2v_caches: list[StreamingKVCache] = []
        self._v2a_caches: list[StreamingKVCache] = []
        if cache_cross_attn:
            cross_present = all(
                hasattr(blk, "audio_to_video_attn") and hasattr(blk, "video_to_audio_attn")
                for blk in blocks
            )
            if not cross_present:
                raise ValueError(
                    "cache_cross_attn=True requires joint video+audio transformer blocks "
                    "(audio_to_video_attn / video_to_audio_attn); the wrapped model has none."
                )
            self._a2v_caches = [
                StreamingKVCache(
                    window_chunks,
                    sink_tokens=0,
                    persistent_first=True,
                    strategy=strategy,
                    num_steps=num_steps,
                )
                for _ in blocks
            ]
            self._v2a_caches = [
                StreamingKVCache(
                    window_chunks,
                    sink_tokens=0,
                    persistent_first=True,
                    strategy=strategy,
                    num_steps=num_steps,
                )
                for _ in blocks
            ]
            for blk, a2v, v2a in zip(blocks, self._a2v_caches, self._v2a_caches):
                blk.audio_to_video_attn.stream_cache = a2v
                blk.video_to_audio_attn.stream_cache = v2a
        # Per-chunk / per-step params (set by the driver).
        self._video_window_pe = None
        self._video_query_mask = None
        self._video_hist_len = 0
        self._audio_window_pe = None
        self._audio_query_mask = None
        self._audio_hist_len = 0
        self._a2v_window_pe = None
        self._a2v_query_mask = None
        self._a2v_hist_len = 0
        self._v2a_window_pe = None
        self._v2a_query_mask = None
        self._v2a_hist_len = 0
        self._mode: str = "clean"
        self._step_idx: int = 0

    @property
    def num_blocks(self) -> int:
        return self.x0.num_blocks

    def reset(self) -> None:
        for cache in self._caches:
            cache.reset()
        for cache in self._audio_caches:
            cache.reset()
        for cache in self._a2v_caches:
            cache.reset()
        for cache in self._v2a_caches:
            cache.reset()
        self._video_window_pe = None
        self._video_query_mask = None
        self._video_hist_len = 0
        self._audio_window_pe = None
        self._audio_query_mask = None
        self._audio_hist_len = 0
        self._a2v_window_pe = None
        self._a2v_query_mask = None
        self._a2v_hist_len = 0
        self._v2a_window_pe = None
        self._v2a_query_mask = None
        self._v2a_hist_len = 0

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
            if self.cache_cross_attn:
                blk.audio_to_video_attn.stream_cache = None
                blk.video_to_audio_attn.stream_cache = None

    def prepare_chunk(
        self,
        *,
        window_pe,
        query_mask,
        hist_len: int,
        audio_window_pe=None,
        audio_query_mask=None,
        audio_hist_len: int | None = None,
        a2v_window_pe=None,
        a2v_query_mask=None,
        a2v_hist_len: int | None = None,
        v2a_window_pe=None,
        v2a_query_mask=None,
        v2a_hist_len: int | None = None,
    ) -> None:
        """Set per-AR-chunk RoPE/mask params (held until the next chunk).

        ``window_pe``/``query_mask``/``hist_len`` are the video params; the
        ``audio_*`` params are required only when ``cache_audio=True`` and are
        otherwise ignored (so the A2V call site is unchanged). The ``a2v_*`` /
        ``v2a_*`` params are required only when ``cache_cross_attn=True``:
        ``a2v_*`` configures the video-query/audio-key cross cache (window_pe
        is the full-window audio cross-PE), ``v2a_*`` the audio-query/video-key
        cross cache (window_pe is the full-window video cross-PE).
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
        if a2v_window_pe is not None:
            self._a2v_window_pe = a2v_window_pe
        if a2v_query_mask is not None:
            self._a2v_query_mask = a2v_query_mask
        if a2v_hist_len is not None:
            self._a2v_hist_len = a2v_hist_len
        if v2a_window_pe is not None:
            self._v2a_window_pe = v2a_window_pe
        if v2a_query_mask is not None:
            self._v2a_query_mask = v2a_query_mask
        if v2a_hist_len is not None:
            self._v2a_hist_len = v2a_hist_len

    def set_mode(self, mode: str, step_idx: int | None = None) -> None:
        """Select the TwinCache snapshot history reads (``"noisy"``/``"clean"``).

        For the ``noisy_steps`` strategy, ``step_idx`` selects which per-step
        snapshot the ring reads (the current denoising step).
        """
        self._mode = mode
        if step_idx is not None:
            self._step_idx = step_idx

    def stash(self, mode: str, step_idx: int | None = None) -> None:
        """Snapshot every cache's current-chunk K/V into its pending entry."""
        for cache in self._caches:
            cache.stash(mode, step_idx=step_idx)
        for cache in self._audio_caches:
            cache.stash(mode, step_idx=step_idx)
        for cache in self._a2v_caches:
            cache.stash(mode, step_idx=step_idx)
        for cache in self._v2a_caches:
            cache.stash(mode, step_idx=step_idx)

    def commit(self) -> None:
        """Append every cache's pending TwinCache entry to its FIFO ring."""
        for cache in self._caches:
            cache.commit()
        for cache in self._audio_caches:
            cache.commit()
        for cache in self._a2v_caches:
            cache.commit()
        for cache in self._v2a_caches:
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
                step_idx=self._step_idx,
            )
        for cache in self._audio_caches:
            cache.set_active(
                mode=self._mode,
                window_pe=self._audio_window_pe,
                query_mask=self._audio_query_mask,
                hist_len=self._audio_hist_len,
                tokens_per_frame=self.audio_tokens_per_frame,
                step_idx=self._step_idx,
            )
        for cache in self._a2v_caches:
            cache.set_active(
                mode=self._mode,
                window_pe=self._a2v_window_pe,
                query_mask=self._a2v_query_mask,
                hist_len=self._a2v_hist_len,
                tokens_per_frame=self.audio_tokens_per_frame,
                step_idx=self._step_idx,
            )
        for cache in self._v2a_caches:
            cache.set_active(
                mode=self._mode,
                window_pe=self._v2a_window_pe,
                query_mask=self._v2a_query_mask,
                hist_len=self._v2a_hist_len,
                tokens_per_frame=self.tokens_per_frame,
                step_idx=self._step_idx,
            )
        try:
            return self.x0(video=video, audio=audio, perturbations=perturbations)
        finally:
            for cache in self._caches:
                cache.set_inactive()
            for cache in self._audio_caches:
                cache.set_inactive()
            for cache in self._a2v_caches:
                cache.set_inactive()
            for cache in self._v2a_caches:
                cache.set_inactive()
