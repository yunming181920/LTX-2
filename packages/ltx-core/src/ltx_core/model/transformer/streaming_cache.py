"""Per-module KV cache for causal streaming (Milestone 2).

A :class:`StreamingKVCache` lives on one ``Attention`` module (the video
self-attention ``attn1`` of a single transformer block). It stores, for each
finalized AR chunk, the chunk's **pre-RoPE** key (post ``k_norm``, pre
``apply_rotary_emb``) and value (post ``to_v``).

Layout follows Vidu S1 §2.3.1:
  * The *first* generated chunk is part of the persistent reference context:
    it is committed once into the permanent ``_first`` slot (clean snapshot,
    never evicted, read in both TwinCache modes).
  * Every *subsequent* chunk keeps **two snapshots** (TwinCache: ``noisy``
    captured at a mid denoising step, ``clean`` captured at the final step)
    in a FIFO ring capped at ``window_chunks``.

The cache is read by the cached attention path (see
:func:`ltx_core.model.transformer.attention.Attention._stream_cached_forward`):
it concatenates ``_first`` and then the ring snapshots selected by the current
``mode`` (``"noisy"`` during intermediate denoising steps, ``"clean"`` at the
final step) with the freshly-computed current-chunk K/V, then re-applies RoPE
to the assembled keys using the window-relative ``window_pe`` (RoPE
repositioning).

Only **self-attention** is cached. For the video modality the sink (first-frame
latent) is NOT cached — it lives in the modality and is recomputed each step (its
K/V depend on the per-chunk audio slice via AV cross-attn). Audio↔video
cross-attention is recomputed each step (not KV-cached), so it needs no cache.

Two modality flavours, selected at construction:

  * **Video** (``sink_tokens = tokens_per_frame``, ``persistent_first = True``):
    the cached attention layout is ``[sink | first | history | current]`` — the
    sink tokens (1 latent frame) live in the modality, and the first generated
    chunk occupies the permanent ``_first`` slot (Vidu S1 §2.3.1 persistent
    reference). This is the original A2V/ti2v-video behaviour.
  * **Audio** (``sink_tokens = 0``, ``persistent_first = False``): audio has no
    image conditioning, so there is no sink and no permanent first chunk — every
    chunk goes to the rolling FIFO ring, and the cached layout collapses to
    ``[history | current]`` (an empty ``[0:0]`` sink slice). Used by the joint
    streaming TI2V path (M2), where audio is *generated* in lockstep and its
    self-attention must also be cached for O(window) memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class _ChunkKV:
    """Pre-RoPE key + value for one chunk, in both TwinCache snapshots."""

    noisy: tuple[torch.Tensor, torch.Tensor] | None = None  # (k_pre_rope, v)
    clean: tuple[torch.Tensor, torch.Tensor] | None = None  # (k_pre_rope, v)


class StreamingKVCache:
    """Per-Attention-module KV cache with a permanent first-chunk slot and a
    TwinCache (noisy + clean) FIFO ring for subsequent chunks.

    Lifecycle within one AR chunk's multi-step denoising:
      1. ``set_active(mode, window_pe, query_mask, hist_len, tokens_per_frame)``
         before each forward — selects which snapshot the ring reads
         (``"noisy"`` mid-denoising, ``"clean"`` final) and provides the
         full-window RoPE ``window_pe`` plus the block-causal
         ``query_mask`` (a structured ``BlockCausalMask``) and history token length.
      2. The cached attention path calls :meth:`read` for history K/V and
         :meth:`set_current` to stash the freshly-computed current-chunk K/V.
      3. At the mid step the driver calls :meth:`stash` (``"noisy"``); at the
         final step :meth:`stash` (``"clean"``) then :meth:`commit`. The first
         commit fills the permanent slot; later commits append to the ring.
    """

    def __init__(self, window_chunks: int, *, sink_tokens: int = 0, persistent_first: bool = False) -> None:
        if window_chunks < 1:
            raise ValueError(f"window_chunks must be >= 1, got {window_chunks}")
        # Modality flavour (fixed at construction):
        #  * video: sink_tokens = one latent frame, persistent_first = True.
        #  * audio: sink_tokens = 0, persistent_first = False (pure FIFO ring).
        self.sink_tokens = sink_tokens
        self.persistent_first = persistent_first
        # Persistent reference (first generated chunk, clean, never evicted).
        # Only used when persistent_first=True (video); audio leaves this None.
        self._first: tuple[torch.Tensor, torch.Tensor] | None = None
        # Rolling history of subsequent chunks (TwinCache entries).
        self._entries: deque[_ChunkKV] = deque(maxlen=window_chunks)
        self._pending: _ChunkKV = _ChunkKV()
        # Runtime state, set per forward by the driver via set_active(...).
        self.active: bool = False
        self.mode: str | None = None  # "noisy" | "clean"
        self.window_pe: tuple[torch.Tensor, torch.Tensor] | None = None  # (cos, sin) [sink|first|hist|cur]
        # Block-causal visibility of the [sink | current] query rows over the
        # full window: a structured BlockCausalMask (preferred; unmasked prefix
        # calls, FlashAttention-capable) or a legacy dense (1, sink+cur, full)
        # log-space additive bias.
        self.query_mask = None
        self.tokens_per_frame: int = 0  # sink token count (1 latent frame)
        self.hist_len: int = 0  # cached history token count (first + ring)
        # Current-chunk pre-RoPE K/V (stashed by the attention path each forward).
        self._cur_k: torch.Tensor | None = None
        self._cur_v: torch.Tensor | None = None

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

    def read(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Concatenate history K/V (pre-RoPE): permanent first chunk, then the
        ring snapshots for the current ``mode``.

        Returns ``(k_hist, v_hist)`` along the token dim, or ``(None, None)``
        if no history is cached yet (the first AR chunk has no history).
        """
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        if self._first is not None:
            ks.append(self._first[0])
            vs.append(self._first[1])
        for entry in self._entries:
            kv = entry.noisy if self.mode == "noisy" else entry.clean
            if kv is None:
                # Snapshot not captured for this entry (e.g. single-step
                # schedules): fall back to whichever exists.
                kv = entry.clean or entry.noisy
            if kv is None:
                continue
            ks.append(kv[0])
            vs.append(kv[1])
        if not ks:
            return None, None
        if len(ks) == 1:
            return ks[0], vs[0]
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    def set_current(self, k_pre_rope: torch.Tensor, v: torch.Tensor) -> None:
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
        """Finalize the pending chunk.

        For a ``persistent_first`` cache (video), the first committed chunk
        becomes the permanent reference slot (clean snapshot; Vidu S1's
        persistent reference context) and later chunks append to the FIFO ring.
        For a non-persistent cache (audio, no sink/anchor), every chunk appends
        straight to the FIFO ring.
        """
        if self._pending.noisy is None or self._pending.clean is None:
            # Need both snapshots; if one is missing, duplicate the other.
            kv = self._pending.clean or self._pending.noisy
            self._pending = _ChunkKV(noisy=kv, clean=kv)
        if self.persistent_first and self._first is None:
            # Persistent reference: always the clean snapshot (video only).
            self._first = self._pending.clean
        else:
            self._entries.append(self._pending)
        self._pending = _ChunkKV()
        self._cur_k = None
        self._cur_v = None
        self.hist_len = self._token_len()

    def reset(self) -> None:
        self._first = None
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
        """Total token count of all cached history entries (first + ring)."""
        total = 0
        if self._first is not None:
            total += self._first[0].shape[1]
        for entry in self._entries:
            kv = entry.clean or entry.noisy
            if kv is not None:
                total += kv[0].shape[1]
        return total
