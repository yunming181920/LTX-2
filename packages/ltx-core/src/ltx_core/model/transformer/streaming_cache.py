"""Per-module KV cache for causal streaming (Milestone 2).

A :class:`StreamingKVCache` lives on one ``Attention`` module (the video
self-attention ``attn1`` of a single transformer block). It stores, for each
finalized AR chunk, the chunk's **pre-RoPE** key (post ``k_norm``, pre
``apply_rotary_emb``) and value (post ``to_v``) — *two snapshots per chunk*
(TwinCache: ``noisy`` captured at a mid denoising step, ``clean`` captured at
the final step). The first deque entry is the sink, added once and fixed.

The cache is read by the cached attention path (see
:func:`ltx_core.model.transformer.attention.Attention._stream_cached_forward`):
it concatenates the history snapshots selected by the current ``mode``
(``"noisy"`` during intermediate denoising steps, ``"clean"`` at the final
step) with the freshly-computed current-chunk K/V, then re-applies RoPE to the
assembled keys using the window-relative ``window_pe`` (RoPE repositioning).

Only **video self-attention** is cached. Audio↔video cross-attention uses
frozen audio (recomputed cheaply) and its history-side output is discarded
(audio is frozen), so it needs no cache.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _ChunkKV:
    """Pre-RoPE key + value for one chunk, in both TwinCache snapshots."""

    noisy: tuple[object, object] | None = None  # (k_pre_rope, v)
    clean: tuple[object, object] | None = None  # (k_pre_rope, v)


class StreamingKVCache:
    """Per-Attention-module KV cache with TwinCache (noisy + clean) snapshots.

    Lifecycle within one AR chunk's multi-step denoising:
      1. ``set_active(mode, window_pe, hist_len)`` before each forward — selects
         which snapshot history reads (``"noisy"`` mid-denoising, ``"clean"``
         final) and provides the full-window RoPE ``window_pe`` plus the history
         token length (sink + finalized chunks).
      2. The cached attention path calls :meth:`read` for history K/V and
         :meth:`set_current` to stash the freshly-computed current-chunk K/V.
      3. At the mid step the driver calls :meth:`stash` (``"noisy"``); at the
         final step :meth:`stash` (``"clean"``) then :meth:`commit` to append the
         finalized chunk's TwinCache entry to the FIFO ring.
    """

    def __init__(self, window_chunks: int) -> None:
        # +1 slot for the sink (permanent head); history chunks cap at window_chunks.
        self._entries: deque[_ChunkKV] = deque(maxlen=window_chunks + 1)
        self._pending: _ChunkKV = _ChunkKV()
        # Runtime state, set per forward by the driver via set_active(...).
        self.active: bool = False
        self.mode: str | None = None  # "noisy" | "clean"
        self.window_pe: tuple[object, object] | None = None  # full-window (cos, sin) [sink|hist|cur]
        self.query_mask: object | None = None  # (1, sink+cur, full) block-causal, hist query rows removed
        self.tokens_per_frame: int = 0  # sink token count (1 latent frame)
        self.hist_len: int = 0  # cached history token count (finalized chunks)
        # Current-chunk pre-RoPE K/V (stashed by the attention path each forward).
        self._cur_k = None
        self._cur_v = None

    # -- driver control ----------------------------------------------------------
    def set_active(
        self, *, mode: str, window_pe, query_mask, hist_len: int, tokens_per_frame: int
    ) -> None:
        self.active = True
        self.mode = mode
        self.window_pe = window_pe
        self.query_mask = query_mask
        self.hist_len = hist_len
        self.tokens_per_frame = tokens_per_frame

    def set_inactive(self) -> None:
        self.active = False

    def read(self):
        """Concatenate history K/V (pre-RoPE) for the current ``mode`` snapshot.

        Returns ``(k_hist, v_hist)`` along the token dim, or ``(None, None)`` if
        no history is cached yet (the first AR chunk has no generated history).
        The sink is NOT cached — it lives in the modality and is recomputed each
        step (its K/V depend on the per-chunk audio slice).
        """
        if not self._entries:
            return None, None
        import torch

        ks = []
        vs = []
        for entry in self._entries:
            kv = entry.noisy if self.mode == "noisy" else entry.clean
            if kv is None:
                # Snapshot not yet captured for this entry (e.g. mid-step before
                # any chunk finalized): fall back to whichever exists.
                kv = entry.clean or entry.noisy
            if kv is None:
                continue
            ks.append(kv[0])
            vs.append(kv[1])
        if not ks:
            return None, None
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    def set_current(self, k_pre_rope, v) -> None:
        """Stash the current chunk's pre-RoPE K/V (for snapshot capture)."""
        self._cur_k = k_pre_rope
        self._cur_v = v

    def stash(self, mode: str) -> None:
        """Snapshot the current K/V into the pending entry (copy, don't ref)."""
        if self._cur_k is None or self._cur_v is None:
            return
        kv = (self._cur_k.clone(), self._cur_v.clone())
        if mode == "noisy":
            self._pending.noisy = kv
        else:
            self._pending.clean = kv

    def commit(self) -> None:
        """Append the pending TwinCache entry to the FIFO ring and reset."""
        if self._pending.noisy is None or self._pending.clean is None:
            # Need both snapshots; if one is missing, duplicate the other.
            kv = self._pending.clean or self._pending.noisy
            self._pending = _ChunkKV(noisy=kv, clean=kv)
        self._entries.append(self._pending)
        self._pending = _ChunkKV()
        self._cur_k = None
        self._cur_v = None
        self.hist_len = self._token_len()

    def reset(self) -> None:
        self._entries.clear()
        self._pending = _ChunkKV()
        self._cur_k = None
        self._cur_v = None
        self.active = False
        self.mode = None
        self.window_pe = None
        self.query_mask = None
        self.tokens_per_frame = 0
        self.hist_len = 0

    # -- helpers ----------------------------------------------------------------
    def _token_len(self) -> int:
        """Total token count of all cached history entries."""
        total = 0
        for entry in self._entries:
            kv = entry.clean or entry.noisy
            if kv is not None:
                total += kv[0].shape[1]
        return total
