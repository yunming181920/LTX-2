"""Structured block-causal attention mask (streaming causal paths).

The streaming drivers' block-causal masks (bidirectional *within* a frame,
causal *across* frames, frame-major token order) have a special structure:
every query token of frame ``f`` attends exactly the contiguous **key prefix**
ending at the last key token of frame ``f``. :class:`BlockCausalMask` captures
that structure directly — per query block (one contiguous run of same-frame
query rows), the visible key-prefix length — instead of materializing a dense
``(T_q, T_k)`` additive bias.

Why: FlashAttention (FA2/FA3/FA4, and torch SDPA's FLASH backend — the FA2
kernel) has no additive-mask support, so the dense-bias form forces the
streaming causal attention onto slower mask-capable backends. With the
structured form, :meth:`BlockCausalMask.apply` computes the *exact* same
softmax by decomposing into one **unmasked** attention call per query block
over its sliced key/value prefix (softmax is row-independent, so splitting by
query rows is lossless). Each call runs on the configured unmasked backend —
FlashAttention included.

The object is immutable and batch-agnostic (no batch dim; the same structure
applies to every sample), and duck-types ``.clone()`` / ``.to()`` so it can
travel through the existing ``LatentState.attention_mask`` /
``Modality.attention_mask`` channels untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

# Unmasked attention callable: (q, k, v, heads) -> out. Structurally matches
# ltx_core.model.transformer.attention.AttentionCallable (not imported here to
# keep this module dependency-free; attention.py imports us).
UnmaskedAttention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class BlockCausalMask:
    """Per-query-block contiguous key-prefix visibility.

    ``q_block_sizes[i]`` query rows (contiguous, frame-major) attend the key
    prefix ``k[:, :k_prefix_lens[i]]``. ``sum(q_block_sizes)`` must equal the
    query token count; every prefix must be ``>= 1`` (an empty prefix would be
    a softmax over nothing).
    """

    q_block_sizes: tuple[int, ...]
    k_prefix_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.q_block_sizes) != len(self.k_prefix_lens):
            raise ValueError(
                f"q_block_sizes ({len(self.q_block_sizes)}) and k_prefix_lens "
                f"({len(self.k_prefix_lens)}) must have the same length"
            )
        if any(s < 1 for s in self.q_block_sizes):
            raise ValueError(f"q_block_sizes must be >= 1, got {self.q_block_sizes}")
        if any(p < 1 for p in self.k_prefix_lens):
            raise ValueError(f"k_prefix_lens must be >= 1, got {self.k_prefix_lens}")

    @property
    def num_q_tokens(self) -> int:
        return sum(self.q_block_sizes)

    @property
    def num_k_tokens(self) -> int:
        """Key tokens the mask expects: the largest visible prefix (the last
        query block of a causal layout sees the whole key sequence)."""
        return max(self.k_prefix_lens)

    @classmethod
    def from_frame_indices(cls, q_frame_indices: torch.Tensor, k_frame_indices: torch.Tensor) -> "BlockCausalMask":
        """Build from per-token frame indices (block-causal: key frame <= query frame).

        ``q_frame_indices`` (T_q,) and ``k_frame_indices`` (T_k,) are 1-D
        non-decreasing integer tensors (frame-major token order). Query rows are
        grouped into contiguous runs of equal frame index; each run's visible
        prefix is the count of key tokens with ``k_frame <= q_frame``. Every
        query frame must see at least one key (i.e. the key sequence must start
        at or before the earliest query frame).
        """
        qf = q_frame_indices.detach().to("cpu", torch.long)
        kf = k_frame_indices.detach().to("cpu", torch.long)
        if qf.ndim != 1 or kf.ndim != 1:
            raise ValueError("frame indices must be 1-D per-token tensors")
        if (qf[1:] < qf[:-1]).any() or (kf[1:] < kf[:-1]).any():
            raise ValueError("frame indices must be non-decreasing (frame-major token order)")
        block_frames, block_sizes = torch.unique_consecutive(qf, return_counts=True)
        prefix_lens = torch.searchsorted(kf, block_frames, right=True)
        return cls(
            q_block_sizes=tuple(int(s) for s in block_sizes),
            k_prefix_lens=tuple(int(p) for p in prefix_lens),
        )

    def apply(
        self, attention_function: UnmaskedAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int
    ) -> torch.Tensor:
        """Exact block-causal attention via unmasked per-block prefix calls.

        ``q``/``k``/``v`` are ``(B, T, heads * dim_head)``. Each query block is
        one unmasked ``attention_function`` call over its contiguous key/value
        prefix — no mask tensor reaches the kernel, so FlashAttention backends
        serve it directly. Outputs concatenate along the token dim.
        """
        if self.num_q_tokens != q.shape[1]:
            raise ValueError(f"mask covers {self.num_q_tokens} query tokens, got q with {q.shape[1]}")
        if self.num_k_tokens > k.shape[1]:
            raise ValueError(f"mask expects >= {self.num_k_tokens} key tokens, got k with {k.shape[1]}")
        outs: list[torch.Tensor] = []
        q0 = 0
        for size, prefix in zip(self.q_block_sizes, self.k_prefix_lens, strict=True):
            outs.append(attention_function(q[:, q0 : q0 + size], k[:, :prefix], v[:, :prefix], heads))
            q0 += size
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    def to_dense(self, device: torch.device | None = None) -> torch.Tensor:
        """Dense ``(1, T_q, T_k)`` float ``[0, 1]`` mask (tests / reference)."""
        t_q, t_k = self.num_q_tokens, self.num_k_tokens
        mask = torch.zeros(1, t_q, t_k, device=device)
        q0 = 0
        for size, prefix in zip(self.q_block_sizes, self.k_prefix_lens, strict=True):
            mask[:, q0 : q0 + size, :prefix] = 1.0
            q0 += size
        return mask

    # Duck-typed tensor conveniences so the object travels through existing
    # LatentState / Modality plumbing (clone-on-copy, device moves) untouched.
    def clone(self) -> "BlockCausalMask":
        return self  # immutable

    def to(self, *args, **kwargs) -> "BlockCausalMask":  # noqa: ARG002
        return self  # no tensors to move
