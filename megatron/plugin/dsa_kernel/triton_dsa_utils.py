# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton DSA/CSA utility functions.

Shared helpers for the Triton-based DSA sparse attention and indexer kernels,
including causal mask computation, online softmax primitives, and autotune configs.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton autotune configurations
# ---------------------------------------------------------------------------


def get_sparse_attn_autotune_configs(D: int) -> List[triton.Config]:
    """Generate autotune configs for sparse attention kernels based on head dim."""
    configs = []
    # For typical DSA head dims (D=512 or D=128)
    if D >= 256:
        configs.extend([
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        ])
    else:
        configs.extend([
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        ])
    return configs


def get_indexer_autotune_configs() -> List[triton.Config]:
    """Generate autotune configs for indexer scoring kernels."""
    return [
        triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ]


# ---------------------------------------------------------------------------
# PyTorch-level utilities
# ---------------------------------------------------------------------------


def compute_ratio_causal_mask(
    S_q: int,
    S_k: int,
    ratio: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a ratio-based causal mask.

    In DSA, compressed KV positions use a coarser causal constraint:
    query at position q can attend to compressed position k if
    ``k < ceil(q / ratio)`` (i.e., the compressed token was produced from
    original tokens all earlier than q).

    Args:
        S_q: number of query positions.
        S_k: number of KV positions (compressed).
        ratio: compression ratio.
        device: target device.
        dtype: output dtype (default float32).

    Returns:
        mask: ``(S_q, S_k)`` tensor, 0 for valid positions, ``-inf`` for masked.
    """
    q_idx = torch.arange(S_q, device=device, dtype=torch.int64)
    k_idx = torch.arange(S_k, device=device, dtype=torch.int64)
    # query q can attend to compressed position k if k < ceil((q+1) / ratio)
    # equivalently: k * ratio < q + 1, i.e., k * ratio <= q
    valid = k_idx.unsqueeze(0) * ratio <= q_idx.unsqueeze(1)  # (S_q, S_k)
    mask = torch.where(valid, torch.zeros(1, device=device, dtype=dtype),
                       torch.full((1,), float("-inf"), device=device, dtype=dtype))
    return mask


def topk_with_causal_mask(
    scores: torch.Tensor,
    k: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-K from scores with ratio-based causal masking.

    Args:
        scores: ``(B, S_q, S_k)`` float32 — raw indexer scores.
        k: number of top-K indices to select.
        ratio: compression ratio for causal masking.

    Returns:
        topk_indices: ``(B, S_q, k)`` int32 — indices into S_k, -1 for invalid.
        topk_length: ``(B, S_q)`` int32 — number of valid positions per query.
    """
    B, S_q, S_k = scores.shape
    device = scores.device

    # Apply ratio causal mask
    mask = compute_ratio_causal_mask(S_q, S_k, ratio, device, scores.dtype)
    masked_scores = scores + mask.unsqueeze(0)  # (B, S_q, S_k)

    # Effective k — cannot exceed S_k
    effective_k = min(k, S_k)

    # Top-K selection
    _, indices = masked_scores.topk(effective_k, dim=-1)  # (B, S_q, effective_k)
    indices = indices.to(torch.int32)

    # Mark invalid positions (those that were -inf before topk)
    # A position is invalid if its masked score is -inf
    gathered_scores = torch.gather(masked_scores, 2, indices.long())
    invalid = torch.isinf(gathered_scores) & (gathered_scores < 0)
    indices = torch.where(invalid, torch.full_like(indices, -1), indices)

    # Compute valid length per query
    topk_length = (~invalid).sum(dim=-1).to(torch.int32)  # (B, S_q)

    # Pad to requested k if effective_k < k
    if effective_k < k:
        pad_width = k - effective_k
        indices = torch.nn.functional.pad(indices, (0, pad_width), value=-1)

    return indices, topk_length


# ---------------------------------------------------------------------------
# Triton-level primitives (used inside kernels)
# ---------------------------------------------------------------------------


@triton.jit
def _online_softmax_update(
    m_prev,  # running max, shape matches a row
    l_prev,  # running sum of exp, shape matches a row
    acc_prev,  # running weighted accumulator
    m_new_block,  # max of new block scores
    scores_block,  # new block scores (before exp)
    v_block,  # new block V values
):
    """Online softmax update step (used inside attention kernel loops).

    Given:
        - Previous state: (m_prev, l_prev, acc_prev)
        - New block: scores_block, v_block

    Returns updated: (m_new, l_new, acc_new)
    """
    # New global max
    m_new = tl.maximum(m_prev, m_new_block)
    # Rescale previous accumulator
    alpha = tl.exp(m_prev - m_new)
    # Exp of new scores with new max
    p_block = tl.exp(scores_block - m_new[:, None])
    # Update sum
    l_new = l_prev * alpha + tl.sum(p_block, axis=1)
    # Update accumulator: rescale old + add new contribution
    acc_new = acc_prev * alpha[:, None] + tl.dot(p_block.to(v_block.dtype), v_block)
    return m_new, l_new, acc_new


@triton.jit
def _compute_ratio_mask_triton(
    q_pos,  # scalar or vector of query positions
    k_pos,  # scalar or vector of kv positions
    ratio: tl.constexpr,
):
    """Compute ratio-based causal mask in Triton.

    Returns True (valid) if k_pos * ratio <= q_pos.
    """
    return k_pos * ratio <= q_pos


@triton.jit
def _safe_load_with_mask(ptr, mask, other=0.0):
    """Load from pointer with mask, returning `other` for masked positions."""
    return tl.load(ptr, mask=mask, other=other)


# ---------------------------------------------------------------------------
# Block size selection
# ---------------------------------------------------------------------------


def select_block_sizes(D: int, TopK: int) -> Tuple[int, int]:
    """Select BLOCK_Q and BLOCK_K sizes based on head dim and topk.

    Heuristic: keep BLOCK_K * D fitting in shared memory, and BLOCK_Q small
    enough for good occupancy.

    Args:
        D: head dimension.
        TopK: number of top-K positions per query.

    Returns:
        (BLOCK_Q, BLOCK_K) tuple.
    """
    if D >= 512:
        # Large head dim — use smaller blocks
        BLOCK_Q = 16
        BLOCK_K = min(32, TopK) if TopK >= 32 else TopK
    elif D >= 128:
        BLOCK_Q = 32
        BLOCK_K = min(64, TopK) if TopK >= 64 else min(32, TopK)
    else:
        BLOCK_Q = 64
        BLOCK_K = min(128, TopK)
    # Ensure BLOCK_K is power of 2 and >= 16
    BLOCK_K = max(16, 1 << (BLOCK_K - 1).bit_length())
    return BLOCK_Q, BLOCK_K


# ---------------------------------------------------------------------------
# Padding / alignment helpers
# ---------------------------------------------------------------------------


def pad_to_multiple(x: torch.Tensor, dim: int, multiple: int, value=0) -> torch.Tensor:
    """Pad tensor along `dim` to the next multiple of `multiple`."""
    size = x.shape[dim]
    if size % multiple == 0:
        return x
    pad_size = multiple - (size % multiple)
    pad_widths = [0] * (2 * x.ndim)
    # F.pad uses reversed dim order
    pad_idx = 2 * (x.ndim - 1 - dim)
    pad_widths[pad_idx + 1] = pad_size
    return torch.nn.functional.pad(x, pad_widths, value=value)


def ceildiv(a: int, b: int) -> int:
    """Integer ceiling division."""
    return (a + b - 1) // b
