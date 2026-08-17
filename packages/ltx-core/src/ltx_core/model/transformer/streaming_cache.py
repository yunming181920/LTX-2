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
  * **Audio** (``sink_tokens = 0``, ``persistent_first = True``): audio has no
    image conditioning, so there is no sink — but Vidu S1 §2.3.1's persistent
    reference is the first generated *video-audio* state, so the first audio
    chunk is still committed to the permanent ``_first`` slot (clean snapshot,
    never evicted) and only later chunks roll through the FIFO ring. The cached
    layout is ``[first | history | current]`` (an empty ``[0:0]`` sink slice).
    Used by the joint streaming TI2V path (M2), where audio is *generated* in
    lockstep and its self-attention must also be cached for O(window) memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class _ChunkKV:
    """Pre-RoPE key + value for one chunk, in both TwinCache snapshots.

    ``noisy``/``clean`` are used by the ``twin`` and ``clean`` strategies;
    ``noisy_steps`` (length ``num_steps``) is used by the ``noisy_steps``
    strategy (per-step matched noisy K/V, one snapshot per denoising step).
    """
    noisy: tuple[torch.Tensor, torch.Tensor] | None = None  # (k_pre_rope, v)
    clean: tuple[torch.Tensor, torch.Tensor] | None = None  # (k_pre_rope, v)
    noisy_steps: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None


class StreamingKVCache:
    """Per-Attention-module KV cache with a permanent first-chunk slot and a
    TwinCache (noisy + clean) FIFO ring for subsequent chunks.

    Lifecycle within one AR chunk's multi-step denoising:
      1. ``set_active(mode, window_pe, query_mask, hist_len, tokens_per_frame, step_idx)``
         before each forward — selects which snapshot the ring reads
         (``"noisy"`` mid-denoising, ``"clean"`` final) and provides the
         full-window RoPE ``window_pe`` plus the block-causal
         ``query_mask`` (a structured ``BlockCausalMask``) and history token length.
      2. The cached attention path calls :meth:`read` for history K/V and
         :meth:`set_current` to stash the freshly-computed current-chunk K/V.
      3. At the mid step the driver calls :meth:`stash` (``"noisy"``); at the
         final step :meth:`stash` (``"clean"``) then :meth:`commit`. The first
         commit fills the permanent slot; later commits append to the ring.

    Strategies (select at construction):
      * ``"twin"`` (default, Vidu S1 §2.3.1): ring reads ``noisy`` at
        intermediate steps and ``clean`` at the final step.
      * ``"clean"`` (ablation A): ring reads ``clean`` at *every* step
        (pure-clean history).
      * ``"noisy_steps"`` (ablation B): ring reads, at current step ``t``,
        the history's own step-``t`` ``noisy_steps[t]`` snapshot (per-step
        noise-level-matched history). No clean snapshot is used; the
        permanent first-chunk slot is per-step too (``_first_steps``).
    """

    def __init__(
        self,
        window_chunks: int,
        *,
        sink_tokens: int = 0,
        persistent_first: bool = False,
        strategy: str = "twin",
        num_steps: int = 0,
    ) -> None:
        if window_chunks < 1:
            raise ValueError(f"window_chunks must be >= 1, got {window_chunks}")
        if strategy not in ("twin", "clean", "noisy_steps"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if strategy == "noisy_steps" and num_steps < 1:
            raise ValueError("noisy_steps strategy requires num_steps >= 1")
        # Modality flavour (fixed at construction):
        #  * video: sink_tokens = one latent frame, persistent_first = True.
        #  * audio: sink_tokens = 0, persistent_first = True (first chunk still
        #    pinned; only the sink is video-specific).
        self.sink_tokens = sink_tokens
        self.persistent_first = persistent_first
        self.strategy = strategy
        self.num_steps = num_steps
        # Persistent reference (first generated chunk). twin/clean: a single
        # clean (k, v). noisy_steps: a per-step list of (k, v), one per step.
        # Only used when persistent_first=True (video); audio leaves these None.
        self._first: tuple[torch.Tensor, torch.Tensor] | None = None
        self._first_steps: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        # Rolling history of subsequent chunks (TwinCache entries).
        self._entries: deque[_ChunkKV] = deque(maxlen=window_chunks)
        self._pending: _ChunkKV = _ChunkKV()
        if strategy == "noisy_steps":
            self._pending.noisy_steps = [None] * num_steps
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
        self.step_idx: int = 0  # current denoising step (noisy_steps strategy)
        # Current-chunk pre-RoPE K/V (stashed by the attention path each forward).
        self._cur_k: torch.Tensor | None = None
        self._cur_v: torch.Tensor | None = None

    # -- driver control ----------------------------------------------------------
    def set_active(
        self, *, mode: str, window_pe, query_mask, hist_len: int, tokens_per_frame: int,
        step_idx: int = 0,
    ) -> None:
        self.active = True
        self.mode = mode
        self.window_pe = window_pe
        self.query_mask = query_mask
        self.hist_len = hist_len
        self.tokens_per_frame = tokens_per_frame
        self.step_idx = step_idx

    def set_inactive(self) -> None:
        self.active = False

    def _entry_snapshot(self, entry: _ChunkKV) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Select the history snapshot for ``entry`` per the active strategy/mode."""
        if self.strategy == "noisy_steps":
            steps = entry.noisy_steps
            if steps is None:
                return None
            kv = steps[self.step_idx] if self.step_idx < len(steps) else None
            if kv is None:
                # Fallback: a snapshot at any captured step (e.g. schedule
                # length changed across chunks should not happen, but be safe).
                kv = next((s for s in steps if s is not None), None)
            return kv
        if self.strategy == "clean":
            kv = entry.clean or entry.noisy
            return kv
        # twin
        kv = entry.noisy if self.mode == "noisy" else entry.clean
        if kv is None:
            # Snapshot not captured for this entry (e.g. single-step
            # schedules): fall back to whichever exists.
            kv = entry.clean or entry.noisy
        return kv

    def _first_snapshot(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Permanent first-chunk snapshot for the active strategy/mode."""
        if self.strategy == "noisy_steps":
            if self._first_steps is None:
                return None
            kv = self._first_steps[self.step_idx] if self.step_idx < len(self._first_steps) else None
            if kv is None:
                kv = next((s for s in self._first_steps if s is not None), None)
            return kv
        return self._first  # twin/clean: the clean snapshot

    def read(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Concatenate history K/V (pre-RoPE): permanent first chunk, then the
        ring snapshots for the current ``mode``/``step_idx``.

        Returns ``(k_hist, v_hist)`` along the token dim, or ``(None, None)``
        if no history is cached yet (the first AR chunk has no history).
        """
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        first_kv = self._first_snapshot()
        if first_kv is not None:
            ks.append(first_kv[0])
            vs.append(first_kv[1])
        for entry in self._entries:
            kv = self._entry_snapshot(entry)
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

    def stash(self, mode: str, step_idx: int | None = None) -> None:
        """Snapshot the current K/V into the pending entry (copy, don't ref).

        twin/clean: ``mode`` selects the ``noisy``/``clean`` slot.
        noisy_steps: ``step_idx`` selects the per-step slot (``mode`` ignored).
        """
        if self._cur_k is None or self._cur_v is None:
            return
        kv = (self._cur_k.clone(), self._cur_v.clone())
        if self.strategy == "noisy_steps":
            if self._pending.noisy_steps is None:
                self._pending.noisy_steps = [None] * self.num_steps
            idx = step_idx if step_idx is not None else self.step_idx
            if 0 <= idx < len(self._pending.noisy_steps):
                self._pending.noisy_steps[idx] = kv
            return
        if mode == "noisy":
            self._pending.noisy = kv
        else:
            self._pending.clean = kv

    def commit(self) -> None:
        """Finalize the pending chunk.

        For a ``persistent_first`` cache (both modalities in the joint TI2V
        path), the first committed chunk becomes the permanent reference slot
        (twin/clean: the clean snapshot; noisy_steps: the per-step list) and
        later chunks append to the FIFO ring.
        """
        if self.strategy == "noisy_steps":
            # Ensure the per-step list is populated; fill any gaps with an
            # existing step's snapshot so read() never returns None mid-run.
            steps = self._pending.noisy_steps
            if steps is None or not any(s is not None for s in steps):
                # Nothing captured (shouldn't happen in normal flow); abort.
                self._pending = _ChunkKV(noisy_steps=[None] * self.num_steps)
                return
            fill = next(s for s in steps if s is not None)
            steps = [s if s is not None else fill for s in steps]
            entry = _ChunkKV(noisy_steps=steps)
            if self.persistent_first and self._first_steps is None:
                self._first_steps = steps
            else:
                self._entries.append(entry)
            self._pending = _ChunkKV(noisy_steps=[None] * self.num_steps)
            self._cur_k = None
            self._cur_v = None
            self.hist_len = self._token_len()
            return
        # twin / clean
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
        self._first_steps = None
        self._entries.clear()
        self._pending = _ChunkKV(noisy_steps=[None] * self.num_steps if self.strategy == "noisy_steps" else None)
        self._cur_k = None
        self._cur_v = None
        self.active = False
        self.mode = None
        self.window_pe = None
        self.query_mask = None
        self.tokens_per_frame = 0
        self.hist_len = 0
        self.step_idx = 0

    # -- helpers ----------------------------------------------------------------
    def _token_len(self) -> int:
        """Total token count of all cached history entries (first + ring)."""
        total = 0
        first_kv = self._first_snapshot()
        if first_kv is not None:
            total += first_kv[0].shape[1]
        for entry in self._entries:
            kv = self._entry_snapshot(entry)
            if kv is not None:
                total += kv[0].shape[1]
        return total
