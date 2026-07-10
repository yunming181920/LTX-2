"""CausalStreamingModel — wraps an :class:`X0Model` to drive per-block KV caches.

Milestone 2 wrapper. Owns one :class:`StreamingKVCache` per transformer block's
video self-attention (``block.attn1``) — 48 caches — and toggles them
``active`` around each forward so the cached attention path (RoPE repositioning
+ TwinCache history) runs only for streaming, leaving all production
bidirectional pipelines untouched (their ``attn1.stream_cache`` stays ``None``).

The driver (``ltx_pipelines.utils.streaming``) calls:
  * :meth:`prepare_chunk` once per AR chunk with the full-window RoPE
    ``window_pe`` ([sink|history|current]) and the block-causal ``query_mask``
    (history query rows removed), the history token length;
  * :meth:`set_mode` per denoising step (``"noisy"`` mid-denoising,
    ``"clean"`` at the final step) — selects which TwinCache snapshot history
    reads;
  * :meth:`stash` at the mid step (``"noisy"``) and final step (``"clean"``),
    then :meth:`commit` to append the chunk's TwinCache K/V entry to every
    cache's FIFO ring.

The query RoPE uses the modality's own ``[sink|current]`` pe (passed to
``attn1`` as ``video.positional_embeddings``); only the key RoPE uses the
cache's full-window ``window_pe`` (RoPE repositioning).

The model's ``forward`` just activates the caches with the current chunk/mode
params, delegates to the wrapped ``X0Model``, then deactivates them.
"""

from __future__ import annotations

import torch

from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.streaming_cache import StreamingKVCache


class CausalStreamingModel(torch.nn.Module):
    """Wraps an X0Model and manages per-block video-self-attn KV caches."""

    def __init__(self, x0_model: X0Model, window_chunks: int, tokens_per_frame: int) -> None:
        super().__init__()
        self.x0 = x0_model
        self.tokens_per_frame = tokens_per_frame
        blocks = self.x0.velocity_model.transformer_blocks
        self._caches: list[StreamingKVCache] = [StreamingKVCache(window_chunks) for _ in blocks]
        for blk, cache in zip(blocks, self._caches):
            blk.attn1.stream_cache = cache
        # Per-chunk / per-step params (set by the driver).
        self._window_pe = None
        self._query_mask = None
        self._hist_len = 0
        self._mode: str = "clean"

    @property
    def num_blocks(self) -> int:
        return self.x0.num_blocks

    def reset(self) -> None:
        for cache in self._caches:
            cache.reset()
        self._window_pe = None
        self._query_mask = None
        self._hist_len = 0

    def prepare_chunk(self, *, window_pe, query_mask, hist_len: int) -> None:
        """Set per-AR-chunk RoPE/mask params (held until the next chunk)."""
        self._window_pe = window_pe
        self._query_mask = query_mask
        self._hist_len = hist_len

    def set_mode(self, mode: str) -> None:
        """Select the TwinCache snapshot history reads (``"noisy"``/``"clean"``)."""
        self._mode = mode

    def stash(self, mode: str) -> None:
        """Snapshot every cache's current-chunk K/V into its pending entry."""
        for cache in self._caches:
            cache.stash(mode)

    def commit(self) -> None:
        """Append every cache's pending TwinCache entry to its FIFO ring."""
        for cache in self._caches:
            cache.commit()

    def hist_len(self) -> int:
        """Current cached-history token count (sink excluded; recomputed)."""
        return self._caches[0].hist_len if self._caches else 0

    def forward(self, video, audio, perturbations):
        for cache in self._caches:
            cache.set_active(
                mode=self._mode,
                window_pe=self._window_pe,
                query_mask=self._query_mask,
                hist_len=self._hist_len,
                tokens_per_frame=self.tokens_per_frame,
            )
        try:
            return self.x0(video=video, audio=audio, perturbations=perturbations)
        finally:
            for cache in self._caches:
                cache.set_inactive()
