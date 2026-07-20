# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Optimized backward utilities for sparse attention (MLA/DSA).

Strategy: keep cuBLAS BMM for matmul (Tensor Core), use Triton only for
fusing small epilogue kernels that PyTorch launches separately.

Optimizations over the baseline PyTorch BMM backward:
  1. bf16 BMM: skip f32 upcast of kv_gathered (saves 384MB allocation +
     ~1.2GB DRAM bandwidth). cuBLAS bf16 matmul uses f32 accumulation
     internally, so precision is equivalent.
  2. Fused mask+scatter: merge masked_fill_ + scatter_add_ into one Triton
     kernel that reads dkv_gathered once, skips invalid entries, and does
     f32 atomic_add directly to the target buffer. Saves 384MB read pass.
  3. Fused exp+mask: merge exp(scores - lse) + masked_fill_(~valid, 0) into
     one Triton kernel (saves 24MB read/write + 1 kernel launch).

Key MLA/DSA properties exploited:
  - K == V == kv (single latent vector): 1 gather, dK+dV combine naturally
  - All heads share indices: gather is (total_Sq, TopK, d_kv), not (S, H, T, d)
  - kv_is_shared (d == d_v == d_kv): baddbmm fuses dK + dV in-place
"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernel: Fused mask + scatter_add
# ---------------------------------------------------------------------------


@triton.jit
def _fused_mask_scatter_kernel(
    SRC_ptr, DST_ptr, IDX_ptr, VALID_ptr,
    total_elements,
    d_kv: tl.constexpr,
    stride_src_row, stride_src_d,
    stride_dst_row, stride_dst_d,
    BLOCK_D: tl.constexpr,
):
    """One-pass fused masked_fill + scatter_add.

    For each element in flattened (total_Sq * TopK):
      - If valid: atomic_add src[element, :] to dst[idx[element], :]
      - If invalid: skip (no need to zero — we never read it again)

    This eliminates the separate masked_fill_ pass (384MB read+write)
    and the separate scatter_add_ pass (384MB read + random write).
    Combined: single 384MB sequential read + small random writes to L2-resident dst.

    Grid: (total_elements,) — one program per (query, topk_pos) pair.
    Each program handles one d_kv-dimensional row.
    """
    pid = tl.program_id(0)
    if pid >= total_elements:
        return

    # Check validity
    is_valid = tl.load(VALID_ptr + pid)
    if not is_valid:
        return

    # Load target index
    target_idx = tl.load(IDX_ptr + pid)

    # Load source row and atomic-add to destination
    d_range = tl.arange(0, BLOCK_D)
    mask = d_range < d_kv

    src_row = tl.load(
        SRC_ptr + pid * stride_src_row + d_range * stride_src_d,
        mask=mask, other=0.0
    )
    tl.atomic_add(
        DST_ptr + target_idx * stride_dst_row + d_range * stride_dst_d,
        src_row,
        mask=mask
    )


# ---------------------------------------------------------------------------
# Triton Kernel: Fused exp + mask (scores → P)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_exp_mask_kernel(
    SCORES_ptr, LSE_ptr, VALID_ptr, P_ptr,
    total_Sq, np_, TopK,
    stride_scores_s, stride_scores_h, stride_scores_k,
    stride_lse_s, stride_lse_h,
    stride_valid_s, stride_valid_k,
    stride_p_s, stride_p_h, stride_p_k,
    BLOCK_K: tl.constexpr,
):
    """Fused: P = exp(scores - lse) * valid_mask.

    Merges 3 ops into 1 kernel:
      1. scores - lse (broadcast subtraction)
      2. exp(...)
      3. masked_fill_(~valid, 0.0)

    Grid: (total_Sq, np_) — one program per (query, head) pair.
    Each program processes TopK values in tiles of BLOCK_K.
    """
    pid_s = tl.program_id(0)  # query index
    pid_h = tl.program_id(1)  # head index

    if pid_s >= total_Sq:
        return

    # Load LSE for this (query, head)
    lse_val = tl.load(LSE_ptr + pid_s * stride_lse_s + pid_h * stride_lse_h)

    # Process TopK in tiles
    for tile_start in range(0, TopK, BLOCK_K):
        k_offs = tile_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < TopK

        # Load scores
        score_ptrs = SCORES_ptr + pid_s * stride_scores_s + pid_h * stride_scores_h + k_offs * stride_scores_k
        scores = tl.load(score_ptrs, mask=k_mask, other=0.0)

        # Load validity
        valid_ptrs = VALID_ptr + pid_s * stride_valid_s + k_offs * stride_valid_k
        valid = tl.load(valid_ptrs, mask=k_mask, other=0).to(tl.int1)

        # Compute P = exp(score - lse) * valid
        p = tl.exp(scores - lse_val)
        p = tl.where(valid & k_mask, p, 0.0)

        # Store
        p_ptrs = P_ptr + pid_s * stride_p_s + pid_h * stride_p_h + k_offs * stride_p_k
        tl.store(p_ptrs, p, mask=k_mask)


# ---------------------------------------------------------------------------
# Python API: Fused scatter
# ---------------------------------------------------------------------------


def fused_mask_scatter_add(
    dkv_gathered: Tensor,
    flat_idxs: Tensor,
    valid_flat: Tensor,
    dkv_out: Tensor,
) -> None:
    """Fused masked_fill + scatter_add in one pass.

    Instead of:
        dkv_gathered.masked_fill_(~valid.unsqueeze(-1), 0.0)  # 384MB read+write
        dkv_out.scatter_add_(0, idx.expand(...), dkv_gathered.reshape(...))  # 384MB read

    This does a single pass: read each row, skip if invalid, atomic_add if valid.

    Args:
        dkv_gathered: (total_Sq, TopK, d_kv) f32 — gradient w.r.t. gathered KV.
        flat_idxs: (total_Sq * TopK,) int64 — scatter target indices.
        valid_flat: (total_Sq * TopK,) bool — validity mask.
        dkv_out: (total_Skv, d_kv) f32 — output buffer (modified in-place).
    """
    total_elements = flat_idxs.shape[0]
    d_kv = dkv_out.shape[-1]

    # Round up BLOCK_D to power of 2
    BLOCK_D = triton.next_power_of_2(d_kv)

    # Flatten source for row-major access
    src_flat = dkv_gathered.reshape(-1, d_kv)

    grid = (total_elements,)
    _fused_mask_scatter_kernel[grid](
        src_flat, dkv_out, flat_idxs, valid_flat,
        total_elements,
        d_kv,
        src_flat.stride(0), src_flat.stride(1),
        dkv_out.stride(0), dkv_out.stride(1),
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# Python API: Fused exp + mask
# ---------------------------------------------------------------------------


def fused_exp_mask(
    scores: Tensor,
    lse: Tensor,
    valid_shared: Tensor,
    np_: int,
) -> Tensor:
    """Fused P = exp(scores - lse) * valid, replacing 3 separate ops.

    Args:
        scores: (total_Sq, np_, TopK) f32 — raw attention scores (already scaled).
        lse: (total_Sq, np_) f32 — log-sum-exp from forward.
        valid_shared: (total_Sq, TopK) bool — validity mask (shared across heads).
        np_: number of heads.

    Returns:
        P: (total_Sq, np_, TopK) f32 — attention probabilities.
    """
    total_Sq, _, TopK = scores.shape

    P = torch.empty_like(scores)

    # valid_shared is (total_Sq, TopK) — shared across heads
    BLOCK_K = triton.next_power_of_2(TopK) if TopK <= 1024 else 1024

    grid = (total_Sq, np_)
    _fused_exp_mask_kernel[grid](
        scores, lse, valid_shared, P,
        total_Sq, np_, TopK,
        scores.stride(0), scores.stride(1), scores.stride(2),
        lse.stride(0), lse.stride(1),
        valid_shared.stride(0), valid_shared.stride(1),
        P.stride(0), P.stride(1), P.stride(2),
        BLOCK_K=BLOCK_K,
    )
    return P


# ---------------------------------------------------------------------------
# Adaptive routing (kept for backward compatibility with existing callers)
# ---------------------------------------------------------------------------

_TRITON_BWD_MEMORY_THRESHOLD = 64 * 1024 * 1024  # 64 MB


def should_use_triton_bwd(total_Sq: int, TopK: int, d_kv: int, H: int, shared_indices: bool) -> bool:
    """Determine whether to use optimized backward path.

    The optimized path uses bf16 BMM + Triton fused epilogues.
    Always returns True for now since the optimized path is strictly better
    (same cuBLAS matmul, less memory, fewer kernel launches).
    """
    # The bf16 BMM path is always better: same precision (f32 accumulation),
    # less memory (no f32 kv_gathered), fewer launches (fused exp+mask, fused scatter).
    # Only fall back for very small configs where kernel launch overhead dominates.
    if shared_indices:
        gather_bytes = total_Sq * TopK * d_kv * 4
    else:
        gather_bytes = total_Sq * H * TopK * d_kv * 4
    total_intermediate = gather_bytes * 2
    return total_intermediate > _TRITON_BWD_MEMORY_THRESHOLD
