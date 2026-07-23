# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

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
# Triton Kernel: Fused dQ (eliminates P, dov, dS materialization)
# ---------------------------------------------------------------------------


def _fused_dq_configs():
    """Autotune configs for fused dQ kernel."""
    configs = []
    for block_k in [16, 32, 64]:
        for nw in [4, 8]:
            configs.append(
                triton.Config({"BLOCK_K": block_k}, num_warps=nw, num_stages=2)
            )
    return configs


@triton.autotune(configs=_fused_dq_configs(), key=["TopK", "D", "H"])
@triton.jit
def _fused_dq_kernel(
    SCORES_ptr, LSE_ptr, DI_ptr, DO_ptr, KV_GATHERED_ptr, VALID_ptr, DQ_ptr,
    softmax_scale,
    total_Sq, H: tl.constexpr, TopK: tl.constexpr, D: tl.constexpr,
    stride_scores_s, stride_scores_h, stride_scores_k,
    stride_lse_s, stride_lse_h,
    stride_di_s, stride_di_h,
    stride_do_s, stride_do_h, stride_do_d,
    stride_kv_s, stride_kv_k, stride_kv_d,
    stride_valid_s, stride_valid_k,
    stride_dq_s, stride_dq_h, stride_dq_d,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused dQ kernel: computes dQ without materializing P, dov, dS.

    For each (query, head-block):
      dQ[q, h, :] = sum_k( dS[q, h, k] * K[q, k, :] )
    where dS = P * (dov - Di) * scale, P = exp(scores - lse) * valid,
    dov = dO @ K^T (since K==V in shared-latent case).

    Grid: (total_Sq, H // BLOCK_H)
    """
    pid_q = tl.program_id(0)
    pid_hb = tl.program_id(1)

    if pid_q >= total_Sq:
        return

    h_start = pid_hb * BLOCK_H
    h_range = h_start + tl.arange(0, BLOCK_H)  # (BLOCK_H,)
    d_range = tl.arange(0, D)  # (D,)

    # Load LSE and Di for this query's head-block
    lse_vals = tl.load(
        LSE_ptr + pid_q * stride_lse_s + h_range * stride_lse_h,
        mask=h_range < H, other=0.0,
    )  # (BLOCK_H,) f32
    di_vals = tl.load(
        DI_ptr + pid_q * stride_di_s + h_range * stride_di_h,
        mask=h_range < H, other=0.0,
    )  # (BLOCK_H,) f32

    # Load dO tile: (BLOCK_H, D) — loaded once, reused across all TopK tiles
    dO_tile = tl.load(
        DO_ptr + pid_q * stride_do_s + h_range[:, None] * stride_do_h + d_range[None, :] * stride_do_d,
        mask=(h_range[:, None] < H) & (d_range[None, :] < D),
        other=0.0,
    ).to(tl.float16)  # (BLOCK_H, D) f16 for tl.dot

    # Accumulator for dQ
    dQ_acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    # Tiled iteration over TopK
    for tile_start in range(0, TopK, BLOCK_K):
        k_range = tile_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
        k_valid_mask = k_range < TopK

        # Load validity (shared across heads)
        valid_tile = tl.load(
            VALID_ptr + pid_q * stride_valid_s + k_range * stride_valid_k,
            mask=k_valid_mask, other=0,
        ).to(tl.int1)  # (BLOCK_K,)

        # Load scores tile: (BLOCK_H, BLOCK_K)
        scores_tile = tl.load(
            SCORES_ptr + pid_q * stride_scores_s + h_range[:, None] * stride_scores_h + k_range[None, :] * stride_scores_k,
            mask=(h_range[:, None] < H) & k_valid_mask[None, :],
            other=float("-inf"),
        )  # (BLOCK_H, BLOCK_K) f32

        # Load K tile (K==V): (BLOCK_K, D)
        K_tile = tl.load(
            KV_GATHERED_ptr + pid_q * stride_kv_s + k_range[:, None] * stride_kv_k + d_range[None, :] * stride_kv_d,
            mask=k_valid_mask[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_K, D) f16

        # P = exp(scores - lse) * valid — in-register, never written to GMEM
        P_tile = tl.exp(scores_tile - lse_vals[:, None])  # (BLOCK_H, BLOCK_K)
        P_tile = tl.where(
            valid_tile[None, :] & k_valid_mask[None, :],
            P_tile, 0.0,
        )

        # dov = dO @ K^T — WGMMA (since K==V)
        dov_tile = tl.dot(dO_tile, tl.trans(K_tile))  # (BLOCK_H, BLOCK_K) f32

        # dS = P * (dov - Di) * scale — in-register
        dS_tile = P_tile * (dov_tile - di_vals[:, None]) * softmax_scale

        # dQ += dS @ K — WGMMA accumulation
        dQ_acc += tl.dot(dS_tile.to(tl.float16), K_tile)  # (BLOCK_H, D) f32

    # Store dQ
    tl.store(
        DQ_ptr + pid_q * stride_dq_s + h_range[:, None] * stride_dq_h + d_range[None, :] * stride_dq_d,
        dQ_acc,
        mask=(h_range[:, None] < H) & (d_range[None, :] < D),
    )


# ---------------------------------------------------------------------------
# Triton Kernel: Fused dKV (eliminates P, dov, dS materialization)
# ---------------------------------------------------------------------------


def _fused_dkv_configs():
    """Autotune configs for fused dKV kernel."""
    configs = []
    for block_h in [16, 32]:
        for nw in [4, 8]:
            configs.append(
                triton.Config({"BLOCK_H": block_h}, num_warps=nw, num_stages=2)
            )
    return configs


@triton.autotune(configs=_fused_dkv_configs(), key=["TopK", "D", "H"])
@triton.jit
def _fused_dkv_kernel(
    SCORES_ptr, LSE_ptr, DI_ptr, DO_ptr, Q_ptr, KV_GATHERED_ptr, VALID_ptr, DKV_ptr,
    softmax_scale,
    total_Sq, H: tl.constexpr, TopK: tl.constexpr, D: tl.constexpr,
    stride_scores_s, stride_scores_h, stride_scores_k,
    stride_lse_s, stride_lse_h,
    stride_di_s, stride_di_h,
    stride_do_s, stride_do_h, stride_do_d,
    stride_q_s, stride_q_h, stride_q_d,
    stride_kv_s, stride_kv_k, stride_kv_d,
    stride_valid_s, stride_valid_k,
    stride_dkv_s, stride_dkv_k, stride_dkv_d,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused dKV kernel: computes dKV_gathered without materializing P, dov, dS.

    For D==DV (shared latent) case:
      dKV[q, k, :] = sum_h( dS[q,h,k] * Q[q,h,:] + P[q,h,k] * dO[q,h,:] )

    Grid: (total_Sq, ceil(TopK / BLOCK_K))
    Each program reduces across all H heads for BLOCK_K KV positions.
    """
    pid_q = tl.program_id(0)
    pid_kb = tl.program_id(1)

    if pid_q >= total_Sq:
        return

    k_start = pid_kb * BLOCK_K
    k_range = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
    k_valid_mask = k_range < TopK
    d_range = tl.arange(0, D)  # (D,)

    # Load validity (shared across heads)
    valid_tile = tl.load(
        VALID_ptr + pid_q * stride_valid_s + k_range * stride_valid_k,
        mask=k_valid_mask, other=0,
    ).to(tl.int1)  # (BLOCK_K,)

    # Load K tile once (K==V, reused across all head tiles)
    K_tile = tl.load(
        KV_GATHERED_ptr + pid_q * stride_kv_s + k_range[:, None] * stride_kv_k + d_range[None, :] * stride_kv_d,
        mask=k_valid_mask[:, None] & (d_range[None, :] < D),
        other=0.0,
    ).to(tl.float16)  # (BLOCK_K, D) f16

    # Accumulator for dKV: (BLOCK_K, D) f32
    dKV_acc = tl.zeros([BLOCK_K, D], dtype=tl.float32)

    # Reduce across all heads in tiles of BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        h_range = h_start + tl.arange(0, BLOCK_H)  # (BLOCK_H,)
        h_mask = h_range < H

        # Load scores tile: (BLOCK_H, BLOCK_K) f32
        scores_tile = tl.load(
            SCORES_ptr + pid_q * stride_scores_s + h_range[:, None] * stride_scores_h + k_range[None, :] * stride_scores_k,
            mask=h_mask[:, None] & k_valid_mask[None, :],
            other=float("-inf"),
        )

        # Load per-head scalars
        lse_tile = tl.load(
            LSE_ptr + pid_q * stride_lse_s + h_range * stride_lse_h,
            mask=h_mask, other=0.0,
        )  # (BLOCK_H,)
        di_tile = tl.load(
            DI_ptr + pid_q * stride_di_s + h_range * stride_di_h,
            mask=h_mask, other=0.0,
        )  # (BLOCK_H,)

        # Load Q and dO tiles: (BLOCK_H, D)
        Q_tile = tl.load(
            Q_ptr + pid_q * stride_q_s + h_range[:, None] * stride_q_h + d_range[None, :] * stride_q_d,
            mask=h_mask[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_H, D) f16

        dO_tile = tl.load(
            DO_ptr + pid_q * stride_do_s + h_range[:, None] * stride_do_h + d_range[None, :] * stride_do_d,
            mask=h_mask[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_H, D) f16

        # P = exp(scores - lse) * valid — in-register
        P_tile = tl.exp(scores_tile - lse_tile[:, None])  # (BLOCK_H, BLOCK_K)
        P_tile = tl.where(
            valid_tile[None, :] & k_valid_mask[None, :] & h_mask[:, None],
            P_tile, 0.0,
        )

        # dov = dO @ K^T — WGMMA (K==V)
        dov_tile = tl.dot(dO_tile, tl.trans(K_tile))  # (BLOCK_H, BLOCK_K)

        # dS = P * (dov - Di) * scale — in-register
        dS_tile = P_tile * (dov_tile - di_tile[:, None]) * softmax_scale

        # dKV += dS^T @ Q + P^T @ dO (combined dK + dV for shared latent)
        dKV_acc += tl.dot(tl.trans(dS_tile.to(tl.float16)), Q_tile)   # (BLOCK_K, D)
        dKV_acc += tl.dot(tl.trans(P_tile.to(tl.float16)), dO_tile)   # (BLOCK_K, D)

    # Store dKV_gathered
    tl.store(
        DKV_ptr + pid_q * stride_dkv_s + k_range[:, None] * stride_dkv_k + d_range[None, :] * stride_dkv_d,
        dKV_acc,
        mask=k_valid_mask[:, None] & (d_range[None, :] < D),
    )


# ---------------------------------------------------------------------------
# Python API: Fused dQ
# ---------------------------------------------------------------------------


def fused_dq(
    scores: Tensor,
    lse: Tensor,
    Di: Tensor,
    dO: Tensor,
    kv_gathered: Tensor,
    valid_shared: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Fused dQ computation — eliminates P, dov, dS materialization.

    Replaces:
        P = exp(scores - lse) * valid          # (S, H, TopK) f32
        dov = BMM(dO, V^T)                     # (S, H, TopK) f32
        dS = P * (dov - Di) * scale            # (S, H, TopK) f32
        dQ = BMM(dS, K)                        # (S, H, D) f32

    With a single Triton kernel that keeps P, dov, dS in registers.
    Saves ~768MB intermediate memory for typical training configs.

    Args:
        scores: (total_Sq, H, TopK) f32 — pre-computed by cuBLAS BMM.
        lse: (total_Sq, H) f32 — from forward pass.
        Di: (total_Sq, H) f32 — sum(dO * O, dim=-1).
        dO: (total_Sq, H, D) bf16 — upstream gradient.
        kv_gathered: (total_Sq, TopK, D) bf16 — gathered KV (K==V).
        valid_shared: (total_Sq, TopK) bool — validity mask.
        softmax_scale: float.

    Returns:
        dQ: (total_Sq, H, D) f32.
    """
    total_Sq, H, TopK = scores.shape
    D = kv_gathered.shape[-1]

    dQ = torch.empty((total_Sq, H, D), dtype=torch.float32, device=scores.device)

    BLOCK_H = 16
    grid = (total_Sq, H // BLOCK_H)

    _fused_dq_kernel[grid](
        scores, lse, Di, dO, kv_gathered, valid_shared, dQ,
        softmax_scale,
        total_Sq, H, TopK, D,
        scores.stride(0), scores.stride(1), scores.stride(2),
        lse.stride(0), lse.stride(1),
        Di.stride(0), Di.stride(1),
        dO.stride(0), dO.stride(1), dO.stride(2),
        kv_gathered.stride(0), kv_gathered.stride(1), kv_gathered.stride(2),
        valid_shared.stride(0), valid_shared.stride(1),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        BLOCK_H=BLOCK_H,
    )
    return dQ


# ---------------------------------------------------------------------------
# Python API: Fused dKV
# ---------------------------------------------------------------------------


def fused_dkv(
    scores: Tensor,
    lse: Tensor,
    Di: Tensor,
    dO: Tensor,
    query: Tensor,
    kv_gathered: Tensor,
    valid_shared: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Fused dKV computation for shared-latent (D==DV) case.

    Replaces:
        P = exp(scores - lse) * valid
        dov = BMM(dO, V^T)
        dS = P * (dov - Di) * scale
        dKV = BMM(dS^T, Q) + BMM(P^T, dO)    # combined dK + dV

    With a single Triton kernel that keeps P, dov, dS in registers.

    Args:
        scores: (total_Sq, H, TopK) f32 — pre-computed by cuBLAS BMM.
        lse: (total_Sq, H) f32 — from forward pass.
        Di: (total_Sq, H) f32 — sum(dO * O, dim=-1).
        dO: (total_Sq, H, D) bf16 — upstream gradient.
        query: (total_Sq, H, D) bf16 — original query.
        kv_gathered: (total_Sq, TopK, D) bf16 — gathered KV (K==V).
        valid_shared: (total_Sq, TopK) bool — validity mask.
        softmax_scale: float.

    Returns:
        dKV_gathered: (total_Sq, TopK, D) f32.
    """
    total_Sq, H, TopK = scores.shape
    D = kv_gathered.shape[-1]

    dKV = torch.empty((total_Sq, TopK, D), dtype=torch.float32, device=scores.device)

    BLOCK_K = 16  # Fixed for dKV (TopK positions per program)
    grid = (total_Sq, triton.cdiv(TopK, BLOCK_K))

    _fused_dkv_kernel[grid](
        scores, lse, Di, dO, query, kv_gathered, valid_shared, dKV,
        softmax_scale,
        total_Sq, H, TopK, D,
        scores.stride(0), scores.stride(1), scores.stride(2),
        lse.stride(0), lse.stride(1),
        Di.stride(0), Di.stride(1),
        dO.stride(0), dO.stride(1), dO.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        kv_gathered.stride(0), kv_gathered.stride(1), kv_gathered.stride(2),
        valid_shared.stride(0), valid_shared.stride(1),
        dKV.stride(0), dKV.stride(1), dKV.stride(2),
        BLOCK_K=BLOCK_K,
    )
    return dKV


# ---------------------------------------------------------------------------
# Triton Kernel: Sorted scatter_add with local reduction
# ---------------------------------------------------------------------------


@triton.jit
def _sorted_scatter_add_kernel(
    SRC_ptr, DST_ptr,
    SORTED_TARGETS_ptr, SORT_PERM_ptr, RUN_STARTS_ptr, RUN_LENGTHS_ptr,
    num_runs,
    d_kv: tl.constexpr,
    stride_src_row, stride_src_d,
    stride_dst_row, stride_dst_d,
    BLOCK_D: tl.constexpr,
    MAX_RUN: tl.constexpr,
):
    """Sorted scatter_add with local reduction.

    Instead of one atomic_add per source row, this kernel:
    1. Groups source rows by target index (pre-sorted)
    2. Accumulates all rows targeting the same KV position in registers
    3. Issues a single atomic_add per target per program

    This reduces atomic contention by factor of avg_run_length.

    Grid: (num_runs,) — one program per contiguous run of same target.
    """
    pid = tl.program_id(0)
    if pid >= num_runs:
        return

    run_start = tl.load(RUN_STARTS_ptr + pid)
    run_len = tl.load(RUN_LENGTHS_ptr + pid)
    target_idx = tl.load(SORTED_TARGETS_ptr + run_start)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < d_kv

    # Local accumulator — reduces all rows in this run
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(MAX_RUN):
        # Triton does not support `break`; use masking to skip past run_len.
        # Clamp index to run_start to avoid OOB when i >= run_len.
        safe_offset = tl.where(i < run_len, run_start + i, run_start)
        src_idx = tl.load(SORT_PERM_ptr + safe_offset)
        # Load source row; mask gates whether it contributes to accumulator
        load_mask = d_mask & (i < run_len)
        row = tl.load(
            SRC_ptr + src_idx * stride_src_row + d_range * stride_src_d,
            mask=load_mask, other=0.0,
        )
        acc += row

    # Single atomic_add to destination
    tl.atomic_add(
        DST_ptr + target_idx * stride_dst_row + d_range * stride_dst_d,
        acc,
        mask=d_mask,
    )


# ---------------------------------------------------------------------------
# Python API: Sorted scatter_add
# ---------------------------------------------------------------------------


def sorted_scatter_add(
    dkv_gathered: Tensor,
    flat_idxs: Tensor,
    valid_flat: Tensor,
    dkv_out: Tensor,
    max_run: int = 64,
) -> None:
    """Scatter_add with sorted local reduction to minimize atomic contention.

    Pre-sorts source rows by target KV index, then accumulates within each
    run (consecutive rows targeting same KV position) before issuing atomics.
    Reduces atomic operations by ~avg_run_length factor.

    Falls back to `fused_mask_scatter_add` if sorting overhead exceeds benefit
    (small total_elements or low contention scenarios).

    Args:
        dkv_gathered: (total_Sq, TopK, D) f32 — gradient w.r.t. gathered KV.
        flat_idxs: (total_Sq * TopK,) int64 — scatter target indices.
        valid_flat: (total_Sq * TopK,) bool — validity mask.
        dkv_out: (total_Skv, D) f32 — output buffer (modified in-place).
        max_run: int — max run length per program (compile-time bound).
    """
    total_elements = flat_idxs.shape[0]
    d_kv = dkv_out.shape[-1]

    # Filter to valid elements only
    valid_indices = torch.where(valid_flat)[0]  # indices of valid entries
    if valid_indices.numel() == 0:
        return

    valid_targets = flat_idxs[valid_indices]  # target KV indices for valid rows

    # Sort by target to group writes to same KV position
    sorted_targets, sort_order = valid_targets.sort(stable=True)
    sort_perm = valid_indices[sort_order]  # maps sorted position → original flat index

    # Find run boundaries (where target changes)
    n_valid = sorted_targets.shape[0]
    if n_valid <= 1:
        # Single element — just use direct scatter
        fused_mask_scatter_add(dkv_gathered, flat_idxs, valid_flat, dkv_out)
        return

    changes = sorted_targets[1:] != sorted_targets[:-1]  # (n_valid-1,) bool
    run_boundary_indices = torch.where(changes)[0] + 1  # positions where new run starts
    run_starts = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=flat_idxs.device),
        run_boundary_indices,
    ])
    run_ends = torch.cat([
        run_boundary_indices,
        torch.tensor([n_valid], dtype=torch.int64, device=flat_idxs.device),
    ])
    run_lengths = (run_ends - run_starts).to(torch.int32)
    num_runs = run_starts.shape[0]

    # Determine if sorting is beneficial (avg run > 1.5)
    avg_run = n_valid / num_runs
    if avg_run < 1.5:
        # Low contention — sorting overhead not worth it, use direct scatter
        fused_mask_scatter_add(dkv_gathered, flat_idxs, valid_flat, dkv_out)
        return

    # Flatten source for row-major access
    src_flat = dkv_gathered.reshape(-1, d_kv).contiguous()
    BLOCK_D = triton.next_power_of_2(d_kv)

    # Clamp max_run to actual max for tighter loop bound
    actual_max_run = int(run_lengths.max().item())
    effective_max_run = min(max_run, actual_max_run)
    # Round up to power of 2 for Triton constexpr
    effective_max_run = triton.next_power_of_2(effective_max_run)

    grid = (num_runs,)
    _sorted_scatter_add_kernel[grid](
        src_flat, dkv_out,
        sorted_targets, sort_perm, run_starts, run_lengths.to(torch.int64),
        num_runs,
        d_kv,
        src_flat.stride(0), src_flat.stride(1),
        dkv_out.stride(0), dkv_out.stride(1),
        BLOCK_D=BLOCK_D,
        MAX_RUN=effective_max_run,
    )


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
