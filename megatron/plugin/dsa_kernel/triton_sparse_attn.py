# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton sparse attention forward and backward kernels for DSA.

Replaces the dependency on ``flash_mla.flash_mla_sparse_fwd`` and
``cudnn.DSA.sparse_attention_backward`` with pure Triton implementations.

The kernels operate on "flat" (unbatched) tensors where Q and KV are concatenated
across the batch dimension, and topk_idxs provides global indices into KV.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr, KV_ptr, IDX_ptr, OUT_ptr, LSE_ptr,
    SINK_ptr, LSE_IDX_ptr,
    softmax_scale: tl.constexpr,
    total_Sq, total_Skv, TopK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_kv_s, stride_kv_d,
    stride_idx_s, stride_idx_h, stride_idx_k,
    stride_out_s, stride_out_h, stride_out_d,
    stride_lse_s, stride_lse_h,
    HAS_SINK: tl.constexpr,
    HAS_LSE_IDX: tl.constexpr,
    INDEXER_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Sparse attention forward with tiled KV gather.

    Each program handles one (query, head) pair. KV positions are processed in
    tiles of BLOCK_K to amortize gather overhead and enable vectorized loads.
    Uses online softmax across tiles for numerical stability.
    """
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_q >= total_Sq:
        return

    # Load query vector: (D,)
    q_offset = pid_q * stride_q_s + pid_h * stride_q_h
    d_range = tl.arange(0, D)
    q = tl.load(Q_ptr + q_offset + d_range * stride_q_d, mask=d_range < D, other=0.0)
    q = (q * softmax_scale).to(tl.float32)

    # Online softmax state
    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.full([], 0.0, dtype=tl.float32)
    # Accumulator for output (DV,)
    dv_range = tl.arange(0, DV)
    acc = tl.zeros([DV], dtype=tl.float32)

    # Optional: bias-only sink
    if HAS_SINK:
        sink_bias = tl.load(SINK_ptr + pid_h)
        m_new = tl.maximum(m_i, sink_bias)
        l_i = l_i * tl.exp(m_i - m_new) + tl.exp(sink_bias - m_new)
        acc = acc * tl.exp(m_i - m_new)
        m_i = m_new

    # Indexer LSE state (tracks first INDEXER_TOPK positions)
    if HAS_LSE_IDX:
        m_idx = tl.full([], float("-inf"), dtype=tl.float32)
        l_idx = tl.full([], 0.0, dtype=tl.float32)

    # Process KV positions in tiles of BLOCK_K
    idx_base = pid_q * stride_idx_s + pid_h * stride_idx_h
    k_range = tl.arange(0, BLOCK_K)

    for tile_start in range(0, TopK, BLOCK_K):
        # Load a tile of indices: (BLOCK_K,)
        tile_offsets = tile_start + k_range
        idx_mask = tile_offsets < TopK
        kv_indices = tl.load(
            IDX_ptr + idx_base + tile_offsets * stride_idx_k,
            mask=idx_mask, other=-1
        )
        valid_mask = (kv_indices >= 0) & idx_mask
        safe_indices = tl.where(valid_mask, kv_indices, 0)

        # Gather K tile: (BLOCK_K, D) — each row is one KV position's key
        kv_bases = safe_indices * stride_kv_s  # (BLOCK_K,)
        k_tile = tl.load(
            KV_ptr + kv_bases[:, None] + d_range[None, :] * stride_kv_d,
            mask=valid_mask[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, D)

        # Compute scores: q (D,) @ k_tile^T (D, BLOCK_K) -> (BLOCK_K,)
        scores = tl.sum(q[None, :] * k_tile, axis=1)  # (BLOCK_K,)
        scores = tl.where(valid_mask, scores, float("-inf"))

        # Tile-level online softmax update
        tile_max = tl.max(scores)
        m_new = tl.maximum(m_i, tile_max)
        exp_old = tl.exp(m_i - m_new)
        exp_scores = tl.exp(scores - m_new)  # (BLOCK_K,)
        # Zero out invalid positions
        exp_scores = tl.where(valid_mask, exp_scores, 0.0)
        tile_sum = tl.sum(exp_scores)

        # Gather V tile: (BLOCK_K, DV)
        v_tile = tl.load(
            KV_ptr + kv_bases[:, None] + dv_range[None, :] * stride_kv_d,
            mask=valid_mask[:, None] & (dv_range[None, :] < DV),
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, DV)

        # Weighted sum: exp_scores (BLOCK_K,) @ v_tile (BLOCK_K, DV) -> (DV,)
        tile_out = tl.sum(exp_scores[:, None] * v_tile, axis=0)  # (DV,)

        # Update accumulators
        l_i = l_i * exp_old + tile_sum
        acc = acc * exp_old + tile_out
        m_i = m_new

        # Update indexer LSE for tiles within INDEXER_TOPK range
        if HAS_LSE_IDX:
            if tile_start < INDEXER_TOPK:
                # Only count positions within [0, INDEXER_TOPK)
                idx_valid = valid_mask & (tile_offsets < INDEXER_TOPK)
                idx_scores = tl.where(idx_valid, scores, float("-inf"))
                idx_tile_max = tl.max(idx_scores)
                m_idx_new = tl.maximum(m_idx, idx_tile_max)
                exp_idx_scores = tl.exp(idx_scores - m_idx_new)
                exp_idx_scores = tl.where(idx_valid, exp_idx_scores, 0.0)
                l_idx = l_idx * tl.exp(m_idx - m_idx_new) + tl.sum(exp_idx_scores)
                m_idx = m_idx_new

    # Finalize: out = acc / l_i
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l

    # Store output
    out_offset = pid_q * stride_out_s + pid_h * stride_out_h
    tl.store(OUT_ptr + out_offset + dv_range * stride_out_d, out.to(tl.bfloat16), mask=dv_range < DV)

    # Store LSE = m_i + log(l_i)
    lse_val = m_i + tl.log(safe_l)
    tl.store(LSE_ptr + pid_q * stride_lse_s + pid_h * stride_lse_h, lse_val)

    # Store indexer LSE if needed
    if HAS_LSE_IDX:
        safe_l_idx = tl.where(l_idx > 0.0, l_idx, 1.0)
        lse_idx_val = m_idx + tl.log(safe_l_idx)
        tl.store(LSE_IDX_ptr + pid_q * stride_lse_s + pid_h * stride_lse_h, lse_idx_val)


# ---------------------------------------------------------------------------
# 2D-Tiled Forward kernel (BLOCK_Q queries per program)
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_fwd_2d_kernel(
    Q_ptr, KV_ptr, IDX_ptr, OUT_ptr, LSE_ptr,
    SINK_ptr, LSE_IDX_ptr,
    softmax_scale: tl.constexpr,
    total_Sq, total_Skv, TopK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_kv_s, stride_kv_d,
    stride_idx_s, stride_idx_h, stride_idx_k,
    stride_out_s, stride_out_h, stride_out_d,
    stride_lse_s, stride_lse_h,
    HAS_SINK: tl.constexpr,
    HAS_LSE_IDX: tl.constexpr,
    INDEXER_TOPK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    """2D-tiled sparse attention forward.

    Each program handles BLOCK_Q queries for one head. Iterates over all TopK
    KV positions one at a time. For each position, loads one index per query
    (BLOCK_Q independent indices) and gathers the corresponding KV vectors.

    Benefits over 1D kernel:
    - BLOCK_Q× fewer programs = less dispatch overhead
    - Q block loaded once, reused across all TopK iterations
    - BLOCK_Q online-softmax states updated in parallel (better SIMD utilization)
    """
    pid_qblock = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Range of queries this program handles
    q_start = pid_qblock * BLOCK_Q
    q_range = tl.arange(0, BLOCK_Q)
    q_ids = q_start + q_range  # (BLOCK_Q,)
    q_valid = q_ids < total_Sq

    # Load Q block: (BLOCK_Q, D)
    d_range = tl.arange(0, D)
    q_offsets = q_ids[:, None] * stride_q_s + pid_h * stride_q_h + d_range[None, :] * stride_q_d
    q_block = tl.load(
        Q_ptr + q_offsets,
        mask=q_valid[:, None] & (d_range[None, :] < D),
        other=0.0,
    ).to(tl.float32) * softmax_scale  # (BLOCK_Q, D)

    # Online softmax state per query: (BLOCK_Q,)
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    # Output accumulator: (BLOCK_Q, DV)
    dv_range = tl.arange(0, DV)
    acc = tl.zeros([BLOCK_Q, DV], dtype=tl.float32)

    # Optional: bias-only sink (same for all queries in block, per head)
    if HAS_SINK:
        sink_bias = tl.load(SINK_ptr + pid_h)
        sink_vec = tl.full([BLOCK_Q], sink_bias, dtype=tl.float32)
        m_new = tl.maximum(m_i, sink_vec)
        l_i = l_i * tl.exp(m_i - m_new) + tl.exp(sink_vec - m_new)
        acc = acc * tl.exp(m_i - m_new)[:, None]
        m_i = m_new

    # Indexer LSE state
    if HAS_LSE_IDX:
        m_idx = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        l_idx = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Iterate over all TopK positions one at a time.
    # Each iteration: load one index per query → gather KV → update softmax.
    for k_pos in range(TopK):
        # Load index for each query at this KV position: (BLOCK_Q,)
        idx_offsets = q_ids * stride_idx_s + pid_h * stride_idx_h + k_pos * stride_idx_k
        kv_indices = tl.load(
            IDX_ptr + idx_offsets,
            mask=q_valid,
            other=-1,
        )  # (BLOCK_Q,)
        valid_q = (kv_indices >= 0) & q_valid  # (BLOCK_Q,)
        safe_idx = tl.where(valid_q, kv_indices, 0)  # (BLOCK_Q,)

        # Gather K for all BLOCK_Q queries: (BLOCK_Q, D)
        kv_bases = safe_idx * stride_kv_s  # (BLOCK_Q,)
        k_vec = tl.load(
            KV_ptr + kv_bases[:, None] + d_range[None, :] * stride_kv_d,
            mask=valid_q[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float32)  # (BLOCK_Q, D)

        # Score: dot product per query
        scores = tl.sum(q_block * k_vec, axis=1)  # (BLOCK_Q,)
        scores = tl.where(valid_q, scores, float("-inf"))

        # Online softmax update (vectorized over BLOCK_Q)
        m_new = tl.maximum(m_i, scores)
        exp_old = tl.exp(m_i - m_new)
        exp_scores = tl.exp(scores - m_new)  # (BLOCK_Q,)
        exp_scores = tl.where(valid_q, exp_scores, 0.0)

        # Gather V: (BLOCK_Q, DV)
        v_vec = tl.load(
            KV_ptr + kv_bases[:, None] + dv_range[None, :] * stride_kv_d,
            mask=valid_q[:, None] & (dv_range[None, :] < DV),
            other=0.0,
        ).to(tl.float32)  # (BLOCK_Q, DV)

        # Update accumulators
        l_i = l_i * exp_old + exp_scores
        acc = acc * exp_old[:, None] + exp_scores[:, None] * v_vec
        m_i = m_new

        # Indexer LSE update
        if HAS_LSE_IDX:
            if k_pos < INDEXER_TOPK:
                m_idx_new = tl.maximum(m_idx, scores)
                exp_idx = tl.exp(scores - m_idx_new)
                exp_idx = tl.where(valid_q, exp_idx, 0.0)
                l_idx = l_idx * tl.exp(m_idx - m_idx_new) + exp_idx
                m_idx = m_idx_new

    # Finalize output: (BLOCK_Q, DV)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]

    # Store output
    out_offsets = q_ids[:, None] * stride_out_s + pid_h * stride_out_h + dv_range[None, :] * stride_out_d
    tl.store(
        OUT_ptr + out_offsets,
        out.to(tl.bfloat16),
        mask=q_valid[:, None] & (dv_range[None, :] < DV),
    )

    # Store LSE: (BLOCK_Q,)
    lse_vals = m_i + tl.log(safe_l)
    lse_offsets = q_ids * stride_lse_s + pid_h * stride_lse_h
    tl.store(LSE_ptr + lse_offsets, lse_vals, mask=q_valid)

    # Store indexer LSE
    if HAS_LSE_IDX:
        safe_l_idx = tl.where(l_idx > 0.0, l_idx, 1.0)
        lse_idx_vals = m_idx + tl.log(safe_l_idx)
        tl.store(LSE_IDX_ptr + lse_offsets, lse_idx_vals, mask=q_valid)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_bwd_kernel(
    Q_ptr, KV_ptr, IDX_ptr, OUT_ptr, LSE_ptr, DO_ptr,
    DQ_ptr, DKV_ptr, DSINK_ptr, SINK_ptr,
    softmax_scale: tl.constexpr,
    total_Sq, total_Skv, TopK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_kv_s, stride_kv_d,
    stride_idx_s, stride_idx_h, stride_idx_k,
    stride_out_s, stride_out_h, stride_out_d,
    stride_lse_s, stride_lse_h,
    HAS_SINK: tl.constexpr,
    BLOCK_K: tl.constexpr = 32,
):
    """Sparse attention backward kernel.

    Recomputes attention weights from Q, KV, LSE, then computes:
    - dQ via dS @ K
    - dK via dS^T @ Q (atomic add since multiple queries may reference same KV)
    - dV via P^T @ dO (atomic add)
    where dS = P * (dO @ V^T - D_i), D_i = rowsum(dO * O).

    TopK is constexpr so that `range(0, TopK, BLOCK_K)` is fully resolved at
    compile time.  Previous versions had TopK as a runtime parameter with a
    two-level loop (`num_full_tiles = TopK // BLOCK_K`), but runtime integer
    division and runtime loop bounds in Triton can produce undefined behavior
    depending on the compiler version, leading to NaN gradients.
    """
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_q >= total_Sq:
        return

    # Load query
    q_offset = pid_q * stride_q_s + pid_h * stride_q_h
    d_range = tl.arange(0, D)
    dv_range = tl.arange(0, DV)
    q = tl.load(Q_ptr + q_offset + d_range * stride_q_d, mask=d_range < D, other=0.0).to(tl.float32)

    # Load dO and O for this position
    out_offset = pid_q * stride_out_s + pid_h * stride_out_h
    dO = tl.load(DO_ptr + out_offset + dv_range * stride_out_d, mask=dv_range < DV, other=0.0).to(tl.float32)
    O = tl.load(OUT_ptr + out_offset + dv_range * stride_out_d, mask=dv_range < DV, other=0.0).to(tl.float32)

    # Load LSE
    lse_val = tl.load(LSE_ptr + pid_q * stride_lse_s + pid_h * stride_lse_h)

    # D_i = sum(dO * O)
    Di = tl.sum(dO * O)

    # Accumulate dQ
    dq_acc = tl.zeros([D], dtype=tl.float32)

    # Process TopK KV positions in tiles of BLOCK_K (both TopK and BLOCK_K are
    # constexpr, so `range(0, TopK, BLOCK_K)` is fully resolved at compile time).
    idx_base = pid_q * stride_idx_s + pid_h * stride_idx_h

    for tile_start in range(0, TopK, BLOCK_K):
        for k_off in range(BLOCK_K):
            k_pos = tile_start + k_off
            if k_pos < TopK:
                kv_idx = tl.load(IDX_ptr + idx_base + k_pos * stride_idx_k)

                is_valid = kv_idx >= 0
                safe_idx = tl.where(is_valid, kv_idx, 0)

                kv_base = safe_idx * stride_kv_s
                k_vec = tl.load(KV_ptr + kv_base + d_range * stride_kv_d, mask=d_range < D, other=0.0).to(tl.float32)
                v_vec = tl.load(KV_ptr + kv_base + dv_range * stride_kv_d, mask=dv_range < DV, other=0.0).to(tl.float32)

                # Recompute attention probability
                s = tl.sum(q * k_vec) * softmax_scale
                p = tl.where(is_valid, tl.exp(s - lse_val), 0.0)

                # dS = P * (dO @ V^T - Di)
                dov = tl.sum(dO * v_vec)
                ds = p * (dov - Di) * softmax_scale

                # dQ += ds * K
                dq_acc += ds * k_vec

                # dK += ds * Q (atomic add to shared KV)
                tl.atomic_add(DKV_ptr + kv_base + d_range * stride_kv_d, ds * q, mask=(d_range < D) & is_valid)

                # dV += p * dO (atomic add)
                tl.atomic_add(DKV_ptr + kv_base + dv_range * stride_kv_d, p * dO, mask=(dv_range < DV) & is_valid)

    # Store dQ
    tl.store(DQ_ptr + q_offset + d_range * stride_q_d, dq_acc.to(tl.bfloat16), mask=d_range < D)


# ---------------------------------------------------------------------------
# Python wrapper functions
# ---------------------------------------------------------------------------


def _triton_sparse_attn_fwd(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Triton sparse attention forward (replaces flash_mla_sparse_fwd).

    Args:
        q: Query tensor ``(total_S_q, H, D)`` bf16.
        kv: KV tensor ``(total_S_kv, D_full)`` bf16 where D_full >= D.
            The first D elements are keys, the first d_v elements are values
            (in absorbed MLA, key and value share the latent space).
        topk_idxs: ``(total_S_q, H, TopK)`` int32 — global KV indices, -1 for invalid.
        softmax_scale: attention scale factor.
        d_v: value dimension (may differ from D for MLA).
        attn_sink: ``(H,)`` f32 — per-head bias-only sink. When provided, exp(bias) is
            added to the softmax denominator without attending to any KV token (no value
            contribution). This matches ``unfused_compressed_sparse_attn`` semantics.
        topk_length: ``(total_S_q, H)`` int32 — valid count per query (unused in Triton impl,
            we use -1 in topk_idxs to mark invalid positions).
        indexer_topk: if > 0, compute separate LSE for the first ``indexer_topk`` positions.

    Returns:
        out: ``(total_S_q, H, d_v)`` bf16.
        lse: ``(total_S_q, H)`` f32.
        lse_indexer: ``(total_S_q, H)`` f32 if indexer_topk > 0, else None.
    """
    total_Sq, H, D = q.shape
    total_Skv = kv.shape[0]
    TopK = topk_idxs.shape[-1]

    # Choose BLOCK_K: balance between register pressure and gather amortization
    if TopK <= 32:
        BLOCK_K = 16
    elif TopK <= 128:
        BLOCK_K = 32
    else:
        BLOCK_K = 64

    # Allocate outputs
    out = torch.empty((total_Sq, H, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((total_Sq, H), dtype=torch.float32, device=q.device)
    lse_indexer = None
    if indexer_topk > 0:
        lse_indexer = torch.empty((total_Sq, H), dtype=torch.float32, device=q.device)

    # Launch grid: one program per (query_token, head)
    grid = (total_Sq, H)

    _sparse_attn_fwd_kernel[grid](
        q, kv, topk_idxs, out, lse,
        attn_sink if attn_sink is not None else torch.empty(0, device=q.device),
        lse_indexer if lse_indexer is not None else torch.empty(0, device=q.device),
        softmax_scale,
        total_Sq, total_Skv, TopK, D, d_v,
        H,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # KV strides
        kv.stride(0), kv.stride(-1) if kv.dim() > 1 else 1,
        # IDX strides
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
        # OUT strides
        out.stride(0), out.stride(1), out.stride(2),
        # LSE strides
        lse.stride(0), lse.stride(1),
        # Compile-time flags
        HAS_SINK=(attn_sink is not None),
        HAS_LSE_IDX=(indexer_topk > 0),
        INDEXER_TOPK=indexer_topk if indexer_topk > 0 else 0,
        BLOCK_K=BLOCK_K,
    )

    # Handle edge case: if indexer_topk >= TopK, lse_indexer should equal lse
    if indexer_topk > 0 and indexer_topk >= TopK:
        lse_indexer = lse.clone()

    return out, lse, lse_indexer


def _pytorch_sparse_attn_fwd(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """PyTorch-native sparse attention forward using gather + batched matmul.

    Uses cuBLAS batched GEMM for score computation instead of the Triton kernel's
    per-position tiled loop. Computes attention scores in f32 for numerical
    precision (aligned with unfused reference and backward recomputation),
    while keeping the output weighted-sum in bf16 for performance.

    Same API as ``triton_sparse_attn_fwd``.
    """
    total_Sq, H, D = q.shape
    TopK = topk_idxs.shape[-1]
    d_kv = kv.shape[-1] if kv.dim() > 1 else D

    # Detect shared indices (MLA mode): stride=0 on head dim means all heads
    # share the same TopK indices. Gather once, broadcast to all heads.
    shared_indices = (topk_idxs.stride(1) == 0)

    if shared_indices:
        # Gather KV once with shape (total_Sq, TopK, d_kv), then use einsum for scores.
        # This avoids materializing (total_Sq, H, TopK, d_kv) which is H times larger.
        idxs_1h = topk_idxs[:, 0, :]  # (total_Sq, TopK) — single head slice
        valid_mask_1h = idxs_1h >= 0   # (total_Sq, TopK)
        safe_idxs_1h = idxs_1h.clamp(min=0).long()
        flat_idxs = safe_idxs_1h.reshape(-1)  # (total_Sq * TopK)
        kv_gathered_1h = kv[flat_idxs].reshape(total_Sq, TopK, d_kv)  # bf16, (S, T, D)

        # Scores via BMM in f32: Q(S,H,D) @ K(S,D,T) -> (S,H,T)
        # Using f32 for the dot-product accumulation eliminates O(D * eps_bf16)
        # error in attention scores, aligning with the unfused reference path
        # and backward recomputation which both use f32.
        k_1h_f = kv_gathered_1h[:, :, :D].float()  # (S, T, D) f32
        scores = torch.bmm(q.float(), k_1h_f.transpose(1, 2)) * softmax_scale  # (S, H, T) f32
        del k_1h_f
        valid_mask = valid_mask_1h.unsqueeze(1).expand(-1, H, -1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        # LSE with optional sink
        if attn_sink is not None:
            sink_expanded = attn_sink.unsqueeze(0).unsqueeze(-1).expand(total_Sq, -1, -1)
            scores_with_sink = torch.cat([scores, sink_expanded], dim=-1)
            lse = torch.logsumexp(scores_with_sink, dim=-1)
        else:
            lse = torch.logsumexp(scores, dim=-1)

        # Attention weights (f32)
        P = torch.exp(scores - lse.unsqueeze(-1))
        P = P.masked_fill(~valid_mask, 0.0)

        # Output via BMM in bf16: P(S,H,T) @ V(S,T,Dv) -> (S,H,Dv)
        # Keeping output bmm in bf16 is acceptable: the linear weighted sum has
        # bounded error O(TopK * eps_bf16) and matches the Triton kernel's behavior.
        # The output is stored as bf16 in save_for_backward regardless.
        v_1h = kv_gathered_1h[:, :, :d_v]  # (S, T, Dv) bf16
        out = torch.bmm(P.to(torch.bfloat16), v_1h)  # (S, H, Dv) bf16

        # Indexer LSE
        lse_indexer = None
        if indexer_topk > 0:
            if indexer_topk >= TopK:
                lse_indexer = lse.clone()
            else:
                idx_scores = scores[:, :, :indexer_topk]
                lse_indexer = torch.logsumexp(idx_scores, dim=-1)

        return out.to(torch.bfloat16), lse, lse_indexer

    else:
        # General path: per-head gather
        valid_mask = topk_idxs >= 0  # (total_Sq, H, TopK)
        safe_idxs = topk_idxs.clamp(min=0).long()
        flat_idxs = safe_idxs.reshape(-1)  # (total_Sq * H * TopK)
        kv_gathered = kv[flat_idxs].reshape(total_Sq, H, TopK, d_kv)  # bf16

        # Compute scores in f32 via bmm for precision (aligned with backward recomputation).
        # Reshape: (total_Sq * H, 1, D) @ (total_Sq * H, D, TopK) -> (total_Sq * H, 1, TopK)
        q_r = q.float().reshape(total_Sq * H, 1, D)  # (SH, 1, D) f32
        k_r = kv_gathered[:, :, :, :D].float().reshape(total_Sq * H, TopK, D).transpose(1, 2)  # (SH, D, TopK) f32
        scores = torch.bmm(q_r, k_r).squeeze(1).reshape(total_Sq, H, TopK) * softmax_scale
        del q_r, k_r
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        # LSE with optional sink
        if attn_sink is not None:
            sink_expanded = attn_sink.unsqueeze(0).unsqueeze(-1).expand(total_Sq, -1, -1)
            scores_with_sink = torch.cat([scores, sink_expanded], dim=-1)
            lse = torch.logsumexp(scores_with_sink, dim=-1)  # (total_Sq, H)
        else:
            lse = torch.logsumexp(scores, dim=-1)  # (total_Sq, H)

        # Attention weights (f32 for precision)
        P = torch.exp(scores - lse.unsqueeze(-1))  # (total_Sq, H, TopK)
        P = P.masked_fill(~valid_mask, 0.0)

        # Output via bmm in bf16: (SH, 1, TopK) @ (SH, TopK, d_v) -> (SH, 1, d_v)
        # Kept in bf16 for performance; linear combination error is bounded.
        P_bf16 = P.to(torch.bfloat16).reshape(total_Sq * H, 1, TopK)
        v_r = kv_gathered[:, :, :, :d_v].reshape(total_Sq * H, TopK, d_v)  # bf16
        out = torch.bmm(P_bf16, v_r).squeeze(1).reshape(total_Sq, H, d_v)  # bf16

        # Indexer LSE (partial LSE over first indexer_topk positions)
        lse_indexer = None
        if indexer_topk > 0:
            if indexer_topk >= TopK:
                lse_indexer = lse.clone()
            else:
                idx_scores = scores[:, :, :indexer_topk]
                lse_indexer = torch.logsumexp(idx_scores, dim=-1)

        return out.to(torch.bfloat16), lse, lse_indexer


def _triton_sparse_attn_fwd_2d(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """2D-tiled Triton sparse attention forward.

    Uses BLOCK_Q queries per program for better throughput. Each program
    handles multiple queries simultaneously, amortizing kernel launch overhead
    and improving instruction scheduling.
    """
    total_Sq, H, D = q.shape
    total_Skv = kv.shape[0]
    TopK = topk_idxs.shape[-1]

    BLOCK_Q = 16

    # Allocate outputs
    out = torch.empty((total_Sq, H, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((total_Sq, H), dtype=torch.float32, device=q.device)
    lse_indexer = None
    if indexer_topk > 0:
        lse_indexer = torch.empty((total_Sq, H), dtype=torch.float32, device=q.device)

    # Grid: one program per (query_block, head)
    num_q_blocks = (total_Sq + BLOCK_Q - 1) // BLOCK_Q
    grid = (num_q_blocks, H)

    _sparse_attn_fwd_2d_kernel[grid](
        q, kv, topk_idxs, out, lse,
        attn_sink if attn_sink is not None else torch.empty(0, device=q.device),
        lse_indexer if lse_indexer is not None else torch.empty(0, device=q.device),
        softmax_scale,
        total_Sq, total_Skv, TopK, D, d_v,
        H,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(-1) if kv.dim() > 1 else 1,
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_SINK=(attn_sink is not None),
        HAS_LSE_IDX=(indexer_topk > 0),
        INDEXER_TOPK=indexer_topk if indexer_topk > 0 else 0,
        BLOCK_Q=BLOCK_Q,
    )

    # Handle edge case
    if indexer_topk > 0 and indexer_topk >= TopK:
        lse_indexer = lse.clone()

    return out, lse, lse_indexer


# Threshold: use PyTorch-native path for TopK <= this value.
# Below this, gather + cuBLAS bmm outperforms the Triton tiled kernel.
# Above this, the gathered KV tensor becomes too large and Triton's
# streaming approach is more memory-efficient.
_PYTORCH_FWD_TOPK_THRESHOLD = 512

# Maximum total elements in gathered KV before falling back to Triton.
# This guards against the case where total_Sq * H * TopK is large enough
# that the random gather saturates memory bandwidth.
# 2M elements ensures the gather stays efficient for L2/HBM bandwidth.
_PYTORCH_FWD_MAX_GATHER_ELEMENTS = 2 * 1024 * 1024


def triton_sparse_attn_fwd(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Sparse attention forward — dispatches to PyTorch-native or Triton kernel.

    For TopK <= _PYTORCH_FWD_TOPK_THRESHOLD, uses gather + cuBLAS batched matmul
    which is significantly faster. For very large TopK, falls back to the Triton
    tiled kernel to avoid excessive memory from materializing gathered KV.

    Args:
        q: Query tensor ``(total_S_q, H, D)`` bf16.
        kv: KV tensor ``(total_S_kv, D_full)`` bf16 where D_full >= D.
        topk_idxs: ``(total_S_q, H, TopK)`` int32 — global KV indices, -1 for invalid.
        softmax_scale: attention scale factor.
        d_v: value dimension (may differ from D for MLA).
        attn_sink: ``(H,)`` f32 — per-head bias-only sink.
        topk_length: unused (kept for API compatibility).
        indexer_topk: if > 0, compute separate LSE for first positions.

    Returns:
        out: ``(total_S_q, H, d_v)`` bf16.
        lse: ``(total_S_q, H)`` f32.
        lse_indexer: ``(total_S_q, H)`` f32 if indexer_topk > 0, else None.
    """
    TopK = topk_idxs.shape[-1]
    total_Sq, H = topk_idxs.shape[0], topk_idxs.shape[1]
    # For shared indices (MLA), actual gather is total_Sq * TopK (not * H)
    shared = (topk_idxs.stride(1) == 0)
    gather_elements = total_Sq * TopK if shared else total_Sq * H * TopK

    if TopK <= _PYTORCH_FWD_TOPK_THRESHOLD and gather_elements <= _PYTORCH_FWD_MAX_GATHER_ELEMENTS:
        return _pytorch_sparse_attn_fwd(
            q, kv, topk_idxs, softmax_scale, d_v, attn_sink, indexer_topk
        )

    # Use 2D-tiled Triton kernel for better throughput
    return _triton_sparse_attn_fwd_2d(
        q, kv, topk_idxs, softmax_scale, d_v, attn_sink, indexer_topk
    )


def triton_sparse_attn_bwd(
    dO: Tensor,
    q: Tensor,
    kv: Tensor,
    out: Tensor,
    lse: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
) -> dict:
    """Triton sparse attention backward (replaces cudnn.DSA.sparse_attention_backward).

    Args:
        dO: gradient of output ``(total_S_q, H, d_v)`` bf16.
        q: query ``(total_S_q, H, D)`` bf16.
        kv: KV ``(total_S_kv, D_full)`` bf16.
        out: forward output ``(total_S_q, H, d_v)`` bf16.
        lse: log-sum-exp ``(total_S_q, H)`` f32.
        topk_idxs: ``(total_S_q, H, TopK)`` int32.
        softmax_scale: attention scale factor.
        d_v: value dimension.
        attn_sink: per-head sink bias or None.

    Returns:
        dict with keys: ``dq`` (total_S_q, H, D), ``dkv`` (total_S_kv, D_full),
        ``d_sink`` (H,) or None.
    """
    total_Sq, H, D = q.shape
    total_Skv = kv.shape[0]
    D_full = kv.shape[-1] if kv.dim() > 1 else D
    TopK = topk_idxs.shape[-1]

    # Ensure topk_idxs is contiguous — expanded tensors (stride=0 from
    # .expand()) can cause illegal memory access in Triton kernels due to
    # pointer arithmetic with zero strides.
    if not topk_idxs.is_contiguous():
        topk_idxs = topk_idxs.contiguous()

    # Choose BLOCK_K: controls the inner-loop unroll factor.
    # Smaller BLOCK_K reduces code size / register pressure at the cost of
    # more outer-loop iterations (runtime overhead is negligible since the
    # outer loop is a simple counter).
    if TopK <= 32:
        BLOCK_K = 16
    elif TopK <= 128:
        BLOCK_K = 32
    else:
        BLOCK_K = 64

    # Allocate gradient outputs — use f32 for dkv to ensure atomic_add
    # compatibility and avoid potential bf16 atomic issues.
    dq = torch.zeros_like(q)
    dkv = torch.zeros((total_Skv, D_full), dtype=torch.float32, device=q.device)

    grid = (total_Sq, H)

    _sparse_attn_bwd_kernel[grid](
        q, kv, topk_idxs, out, lse, dO,
        dq, dkv,
        torch.empty(0, device=q.device),  # DSINK placeholder
        attn_sink if attn_sink is not None else torch.empty(0, device=q.device),
        softmax_scale,
        total_Sq, total_Skv, TopK, D, d_v,
        H,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(-1) if kv.dim() > 1 else 1,
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_SINK=(attn_sink is not None),
        BLOCK_K=BLOCK_K,
    )

    # Compute d_sink if needed (gradient w.r.t. bias-only sink)
    d_sink = None
    if attn_sink is not None:
        # Bias-only sink: s_sink = attn_sink[h], p_sink = exp(s_sink - lse)
        # The sink contributes no value, so output is weighted sum of TopK values only,
        # divided by (l_topk + exp(sink - m)). The gradient w.r.t. sink_bias is:
        # d_sink[h] = sum_q [ p_sink * (-Di) ]  where Di = sum(dO * O)
        p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, H)
        dO_f = dO.float()
        out_f = out.float()
        Di = (dO_f * out_f).sum(-1)  # (total_Sq, H)
        # Since sink has value=0, dS_sink = p_sink * (0 - Di) = -p_sink * Di
        ds_sink = -p_sink * Di  # (total_Sq, H)
        d_sink = ds_sink.sum(0)  # (H,)

    return {"dq": dq, "dkv": dkv.to(torch.bfloat16), "d_sink": d_sink}


# ---------------------------------------------------------------------------
# Fused backward utilities (Triton epilogue kernels for bf16 BMM path)
# ---------------------------------------------------------------------------

from megatron.plugin.dsa_kernel.triton_sparse_attn_bwd import (
    fused_mask_scatter_add,
    fused_exp_mask,
    should_use_triton_bwd,
)


# ---------------------------------------------------------------------------
# Aliases for backward-compatible import names
# ---------------------------------------------------------------------------

triton_sparse_attn_forward = triton_sparse_attn_fwd
triton_sparse_attn_backward = triton_sparse_attn_bwd
