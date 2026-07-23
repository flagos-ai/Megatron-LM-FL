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
# Head-Parallel Forward kernel (WGMMA, shared indices)
# ---------------------------------------------------------------------------


def _hp_fwd_configs():
    """Autotune configs for HP forward kernel.

    Varies BLOCK_K (tile size over TopK) and num_warps.
    - BLOCK_K=16: lower register pressure, more loop iterations
    - BLOCK_K=32: balanced
    - BLOCK_K=64: fewer iterations, higher register pressure
    - num_warps=4: better for memory-bound (D=512 with small TopK)
    - num_warps=8: better for compute-bound (large TopK, occupancy limited)
    """
    configs = []
    for block_k in [16, 32, 64]:
        for nw in [4, 8]:
            configs.append(
                triton.Config({"BLOCK_K": block_k}, num_warps=nw, num_stages=2)
            )
    return configs


@triton.autotune(
    configs=_hp_fwd_configs(),
    key=["TopK", "D", "DV", "H"],
)
@triton.jit
def _sparse_attn_fwd_hp_kernel(
    Q_ptr, KV_ptr, IDX_ptr, OUT_ptr, LSE_ptr,
    SINK_ptr, LSE_IDX_ptr,
    softmax_scale,
    total_Sq, total_Skv, TopK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_kv_s, stride_kv_d,
    stride_idx_s, stride_idx_k,
    stride_out_s, stride_out_h, stride_out_d,
    stride_lse_s, stride_lse_h,
    HAS_SINK: tl.constexpr,
    HAS_LSE_IDX: tl.constexpr,
    INDEXER_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Head-parallel sparse attention forward using tl.dot (WGMMA on Hopper).

    Exploits MLA shared-index structure: all heads share the same TopK indices
    for each query position. This allows a single KV gather to be amortized
    across BLOCK_H heads, and both score computation and output accumulation
    use tl.dot() which maps to WGMMA (Tensor Core matrix multiply).

    Program mapping: (pid_q, pid_hb) where pid_q is the query position and
    pid_hb selects the head block [pid_hb * BLOCK_H : (pid_hb+1) * BLOCK_H].

    GEMM dimensions:
      Score:  Q_tile(BLOCK_H, D) @ K_tile^T(D, BLOCK_K) → (BLOCK_H, BLOCK_K)
      Output: P_tile(BLOCK_H, BLOCK_K) @ V_tile(BLOCK_K, DV) → (BLOCK_H, DV)

    Both satisfy WGMMA alignment: M=BLOCK_H≥16, K=D or BLOCK_K≥16, N≥16.
    """
    pid_q = tl.program_id(0)
    pid_hb = tl.program_id(1)

    if pid_q >= total_Sq:
        return

    h_start = pid_hb * BLOCK_H
    h_range = h_start + tl.arange(0, BLOCK_H)  # (BLOCK_H,)
    d_range = tl.arange(0, D)                   # (D,)
    dv_range = tl.arange(0, DV)                 # (DV,)

    # =========================================================================
    # Load Q tile: (BLOCK_H, D) — loaded once, reused across all TopK tiles
    # =========================================================================
    # Q layout: (total_Sq, H, D) — load BLOCK_H heads for this query
    q_base = pid_q * stride_q_s
    Q_tile = tl.load(
        Q_ptr + q_base + h_range[:, None] * stride_q_h + d_range[None, :] * stride_q_d,
        mask=(h_range[:, None] < H) & (d_range[None, :] < D),
        other=0.0,
    ).to(tl.float16)  # (BLOCK_H, D) f16 for tl.dot

    # =========================================================================
    # Online softmax state: per-head running max and sum
    # =========================================================================
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)                # running sum(exp)
    acc = tl.zeros([BLOCK_H, DV], dtype=tl.float32)            # output accumulator

    # Handle attention sink FIRST — initializes running max from sink bias.
    # Processing sink before TopK ensures the running max starts from a meaningful
    # baseline, reducing accumulator rescale magnitude during the TopK loop.
    if HAS_SINK:
        sink_vals = tl.load(SINK_ptr + h_range, mask=h_range < H, other=0.0)  # (BLOCK_H,)
        m_i = tl.maximum(m_i, sink_vals)
        p_sink = tl.exp(sink_vals - m_i)
        l_i = l_i + p_sink
        # Sink contributes no value (bias-only), so acc stays zero

    # Indexer LSE tracking (optional)
    if HAS_LSE_IDX:
        m_idx = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_idx = tl.zeros([BLOCK_H], dtype=tl.float32)

    # =========================================================================
    # Tiled iteration over TopK positions in groups of BLOCK_K
    # =========================================================================
    # Indices layout: (total_Sq, 1, TopK) with stride_idx_h=0 (shared)
    # We use head-0 slice: IDX_ptr + pid_q * stride_idx_s + k * stride_idx_k
    idx_base = pid_q * stride_idx_s

    for tile_start in range(0, TopK, BLOCK_K):
        k_range = tile_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
        k_valid = k_range < TopK

        # --- Load BLOCK_K indices (shared across heads) ---
        kv_indices = tl.load(
            IDX_ptr + idx_base + k_range * stride_idx_k,
            mask=k_valid,
            other=-1,
        )  # (BLOCK_K,) int32
        valid_mask = (kv_indices >= 0) & k_valid  # (BLOCK_K,)
        safe_indices = tl.where(valid_mask, kv_indices, 0)  # clamp for safe load

        # --- Gather K tile: (BLOCK_K, D) ---
        kv_bases = safe_indices * stride_kv_s  # (BLOCK_K,)
        K_tile = tl.load(
            KV_ptr + kv_bases[:, None] + d_range[None, :] * stride_kv_d,
            mask=valid_mask[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_K, D) f16

        # --- Score GEMM: (BLOCK_H, D) @ (D, BLOCK_K) → (BLOCK_H, BLOCK_K) ---
        # tl.dot uses f16 inputs with f32 accumulator (WGMMA on Hopper)
        scores = tl.dot(Q_tile, tl.trans(K_tile))  # (BLOCK_H, BLOCK_K) f32
        scores = scores * softmax_scale

        # Mask invalid positions
        scores = tl.where(
            valid_mask[None, :],  # broadcast (1, BLOCK_K) over (BLOCK_H, BLOCK_K)
            scores,
            float("-inf"),
        )

        # --- Tiled online softmax update ---
        tile_max = tl.max(scores, axis=1)  # (BLOCK_H,)
        m_new = tl.maximum(m_i, tile_max)

        # Rescale existing accumulator
        exp_old = tl.exp(m_i - m_new)  # (BLOCK_H,)
        acc = acc * exp_old[:, None]
        l_i = l_i * exp_old

        # Compute attention weights for this tile
        P_tile = tl.exp(scores - m_new[:, None])  # (BLOCK_H, BLOCK_K) f32
        P_tile = tl.where(valid_mask[None, :], P_tile, 0.0)
        tile_sum = tl.sum(P_tile, axis=1)  # (BLOCK_H,)
        l_i = l_i + tile_sum
        m_i = m_new

        # --- Gather V tile: (BLOCK_K, DV) ---
        V_tile = tl.load(
            KV_ptr + kv_bases[:, None] + dv_range[None, :] * stride_kv_d,
            mask=valid_mask[:, None] & (dv_range[None, :] < DV),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_K, DV) f16

        # --- Output GEMM: (BLOCK_H, BLOCK_K) @ (BLOCK_K, DV) → (BLOCK_H, DV) ---
        acc += tl.dot(P_tile.to(tl.float16), V_tile)  # WGMMA, f32 accumulator

        # --- Indexer LSE tracking (first INDEXER_TOPK positions) ---
        if HAS_LSE_IDX:
            if tile_start < INDEXER_TOPK:
                # Only count k-positions within [0, INDEXER_TOPK)
                idx_valid = valid_mask & (k_range < INDEXER_TOPK)  # (BLOCK_K,)
                idx_scores = tl.where(
                    idx_valid[None, :],
                    scores,
                    float("-inf"),
                )  # (BLOCK_H, BLOCK_K)
                idx_tile_max = tl.max(idx_scores, axis=1)  # (BLOCK_H,)
                m_idx_new = tl.maximum(m_idx, idx_tile_max)
                exp_old_idx = tl.exp(m_idx - m_idx_new)
                l_idx = l_idx * exp_old_idx
                P_idx = tl.exp(idx_scores - m_idx_new[:, None])
                P_idx = tl.where(idx_valid[None, :], P_idx, 0.0)
                l_idx = l_idx + tl.sum(P_idx, axis=1)
                m_idx = m_idx_new

    # =========================================================================
    # Finalize: normalize output and compute LSE
    # =========================================================================

    # Normalize: out = acc / l_i
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out_tile = acc / safe_l[:, None]  # (BLOCK_H, DV) f32

    # LSE = m_i + log(l_i)
    safe_l_for_log = tl.where(l_i > 0.0, l_i, 1.0)
    lse_vals = m_i + tl.log(safe_l_for_log)  # (BLOCK_H,)

    # =========================================================================
    # Store outputs
    # =========================================================================
    out_base = pid_q * stride_out_s
    tl.store(
        OUT_ptr + out_base + h_range[:, None] * stride_out_h + dv_range[None, :] * stride_out_d,
        out_tile.to(tl.bfloat16),
        mask=(h_range[:, None] < H) & (dv_range[None, :] < DV),
    )

    # Store LSE
    lse_base = pid_q * stride_lse_s
    tl.store(
        LSE_ptr + lse_base + h_range * stride_lse_h,
        lse_vals,
        mask=h_range < H,
    )

    # Store indexer LSE
    if HAS_LSE_IDX:
        safe_l_idx = tl.where(l_idx > 0.0, l_idx, 1.0)
        lse_idx_vals = m_idx + tl.log(safe_l_idx)
        tl.store(
            LSE_IDX_ptr + lse_base + h_range * stride_lse_h,
            lse_idx_vals,
            mask=h_range < H,
        )


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
    BLOCK_K: tl.constexpr,
):
    """2D-tiled sparse attention forward with tiled online softmax.

    Each program handles BLOCK_Q queries for one head. KV positions are
    processed in tiles of BLOCK_K. Within each tile, scores are gathered into
    a (BLOCK_Q, BLOCK_K) matrix and a single tiled online-softmax update is
    performed, reducing the number of expensive accumulator rescales from TopK
    to TopK/BLOCK_K.

    P0 Hopper optimization: tiled softmax reduces rescale overhead by BLOCK_K×.
    The tiled structure also enables better instruction scheduling and provides
    the foundation for future tl.dot()-based WGMMA when shared indices are
    available.
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

    # Tiled iteration over TopK positions in groups of BLOCK_K.
    # Each tile: gather K+V, compute scores, find tile max, compute weighted V.
    # Uses single-pass with deferred V accumulation to enable software pipelining.
    #
    # P1 optimization: The loop structure is designed for Triton's software
    # pipelining (num_stages). The loads at the start of each k_off iteration
    # can overlap with the prior iteration's compute.
    for tile_start in range(0, TopK, BLOCK_K):

        # === Phase 1: Gather K, compute scores, find tile max ===
        # We need tile_max before computing exp weights. First pass over
        # BLOCK_K positions to find the max score.
        tile_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for k_off in range(BLOCK_K):
            k_pos = tile_start + k_off
            if k_pos < TopK:
                # Load index for each query
                idx_offsets = q_ids * stride_idx_s + pid_h * stride_idx_h + k_pos * stride_idx_k
                kv_indices = tl.load(IDX_ptr + idx_offsets, mask=q_valid, other=-1)
                valid_q = (kv_indices >= 0) & q_valid
                safe_idx = tl.where(valid_q, kv_indices, 0)

                # Gather K: (BLOCK_Q, D)
                kv_bases = safe_idx * stride_kv_s
                k_vec = tl.load(
                    KV_ptr + kv_bases[:, None] + d_range[None, :] * stride_kv_d,
                    mask=valid_q[:, None] & (d_range[None, :] < D),
                    other=0.0,
                ).to(tl.float32)

                # Score: dot product per query
                score = tl.sum(q_block * k_vec, axis=1)  # (BLOCK_Q,)
                score = tl.where(valid_q, score, float("-inf"))
                tile_max = tl.maximum(tile_max, score)

        # === Softmax rescale: ONE per tile ===
        m_new = tl.maximum(m_i, tile_max)
        exp_old = tl.exp(m_i - m_new)  # (BLOCK_Q,)
        # Rescale accumulator once for entire tile
        acc = acc * exp_old[:, None]
        l_i = l_i * exp_old

        # === Phase 2: Recompute scores, gather V, accumulate ===
        # Second pass: compute exp weights and weighted V sum.
        # Loads here can be pipelined by Triton's num_stages with Phase 1
        # loads of the *next* tile iteration.
        tile_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

        for k_off in range(BLOCK_K):
            k_pos = tile_start + k_off
            if k_pos < TopK:
                # Re-load index and gather K+V
                idx_offsets = q_ids * stride_idx_s + pid_h * stride_idx_h + k_pos * stride_idx_k
                kv_indices = tl.load(IDX_ptr + idx_offsets, mask=q_valid, other=-1)
                valid_q = (kv_indices >= 0) & q_valid
                safe_idx = tl.where(valid_q, kv_indices, 0)
                kv_bases = safe_idx * stride_kv_s

                # Gather K and recompute score
                k_vec = tl.load(
                    KV_ptr + kv_bases[:, None] + d_range[None, :] * stride_kv_d,
                    mask=valid_q[:, None] & (d_range[None, :] < D),
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(q_block * k_vec, axis=1)
                score = tl.where(valid_q, score, float("-inf"))

                # Exp weight relative to m_new
                w = tl.exp(score - m_new)  # (BLOCK_Q,)

                # Gather V: (BLOCK_Q, DV)
                v_vec = tl.load(
                    KV_ptr + kv_bases[:, None] + dv_range[None, :] * stride_kv_d,
                    mask=valid_q[:, None] & (dv_range[None, :] < DV),
                    other=0.0,
                ).to(tl.float32)

                # Accumulate weighted V
                acc += w[:, None] * v_vec
                tile_sum += w

                # Indexer LSE update
                if HAS_LSE_IDX:
                    if k_pos < INDEXER_TOPK:
                        m_idx_new = tl.maximum(m_idx, score)
                        exp_idx = tl.exp(score - m_idx_new)
                        l_idx = l_idx * tl.exp(m_idx - m_idx_new) + exp_idx
                        m_idx = m_idx_new

        # Update softmax denominator
        l_i = l_i + tile_sum
        m_i = m_new

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
# Head-Parallel Backward kernel (WGMMA, shared indices)
# ---------------------------------------------------------------------------


def _hp_bwd_configs():
    """Autotune configs for HP backward kernel.

    Backward has higher register pressure than forward (holds dQ_acc + Q + dO),
    so larger BLOCK_K or more warps can help depending on TopK.
    """
    configs = []
    for block_k in [16, 32, 64]:
        for nw in [4, 8]:
            configs.append(
                triton.Config({"BLOCK_K": block_k}, num_warps=nw, num_stages=2)
            )
    return configs


@triton.autotune(
    configs=_hp_bwd_configs(),
    key=["TopK", "D", "DV", "H"],
)
@triton.jit
def _sparse_attn_bwd_hp_kernel(
    Q_ptr, KV_ptr, IDX_ptr, OUT_ptr, LSE_ptr, DO_ptr,
    DQ_ptr, DKV_ptr,
    softmax_scale,
    total_Sq, total_Skv, TopK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_kv_s, stride_kv_d,
    stride_idx_s, stride_idx_k,
    stride_out_s, stride_out_h, stride_out_d,
    stride_lse_s, stride_lse_h,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Head-parallel sparse attention backward using tl.dot (WGMMA on Hopper).

    Exploits MLA shared-index structure (same as forward HP kernel). Computes
    gradients dQ, dK, dV using WGMMA matrix multiplies:

    Per tile of BLOCK_K KV positions:
      Score recompute: Q(BLOCK_H, D) @ K^T(D, BLOCK_K) → (BLOCK_H, BLOCK_K) WGMMA
      dOV:            dO(BLOCK_H, DV) @ V^T(DV, BLOCK_K) → (BLOCK_H, BLOCK_K) WGMMA
      dQ accumulate:  dS(BLOCK_H, BLOCK_K) @ K(BLOCK_K, D) → (BLOCK_H, D)    WGMMA
      dK per tile:    dS^T(BLOCK_K, BLOCK_H) @ Q(BLOCK_H, D) → (BLOCK_K, D)  WGMMA
      dV per tile:    P^T(BLOCK_K, BLOCK_H) @ dO(BLOCK_H, DV) → (BLOCK_K, DV) WGMMA

    dQ is stored directly (per-query, no conflicts).
    dK and dV use f32 atomic_add (multiple queries write to same KV position).

    P0 optimization: Tiled score recomputation — one tl.dot per tile vs per-position.
    P1 optimization: num_stages=2 for software pipelining of KV gather.
    """
    pid_q = tl.program_id(0)
    pid_hb = tl.program_id(1)

    if pid_q >= total_Sq:
        return

    h_start = pid_hb * BLOCK_H
    h_range = h_start + tl.arange(0, BLOCK_H)  # (BLOCK_H,)
    d_range = tl.arange(0, D)                   # (D,)
    dv_range = tl.arange(0, DV)                 # (DV,)

    # =========================================================================
    # Load Q, dO, O, LSE — all reused across TopK tiles
    # =========================================================================
    q_base = pid_q * stride_q_s
    out_base = pid_q * stride_out_s

    # Q_tile: (BLOCK_H, D) f16 — for score recompute and dK
    Q_tile = tl.load(
        Q_ptr + q_base + h_range[:, None] * stride_q_h + d_range[None, :] * stride_q_d,
        mask=(h_range[:, None] < H) & (d_range[None, :] < D),
        other=0.0,
    ).to(tl.float16)  # (BLOCK_H, D)

    # dO_tile: (BLOCK_H, DV) f16 — for dOV and dV
    dO_tile = tl.load(
        DO_ptr + out_base + h_range[:, None] * stride_out_h + dv_range[None, :] * stride_out_d,
        mask=(h_range[:, None] < H) & (dv_range[None, :] < DV),
        other=0.0,
    ).to(tl.float16)  # (BLOCK_H, DV)

    # O_tile: (BLOCK_H, DV) f32 — for Di computation
    O_tile = tl.load(
        OUT_ptr + out_base + h_range[:, None] * stride_out_h + dv_range[None, :] * stride_out_d,
        mask=(h_range[:, None] < H) & (dv_range[None, :] < DV),
        other=0.0,
    ).to(tl.float32)  # (BLOCK_H, DV)

    # LSE: (BLOCK_H,) f32
    lse_base = pid_q * stride_lse_s
    lse_vals = tl.load(
        LSE_ptr + lse_base + h_range * stride_lse_h,
        mask=h_range < H,
        other=0.0,
    )  # (BLOCK_H,) f32

    # Di = sum(dO * O, axis=-1) per head: (BLOCK_H,) f32
    Di = tl.sum(dO_tile.to(tl.float32) * O_tile, axis=1)  # (BLOCK_H,)

    # dQ accumulator: (BLOCK_H, D) f32 — stored once at end
    dq_acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    # =========================================================================
    # Tiled iteration over TopK positions
    # =========================================================================
    idx_base = pid_q * stride_idx_s

    for tile_start in range(0, TopK, BLOCK_K):
        k_range = tile_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
        k_valid = k_range < TopK

        # --- Load BLOCK_K indices (shared across heads) ---
        kv_indices = tl.load(
            IDX_ptr + idx_base + k_range * stride_idx_k,
            mask=k_valid,
            other=-1,
        )  # (BLOCK_K,) int32
        valid_mask = (kv_indices >= 0) & k_valid  # (BLOCK_K,)
        safe_indices = tl.where(valid_mask, kv_indices, 0)

        # --- Gather K tile: (BLOCK_K, D) f16 ---
        kv_bases = safe_indices * stride_kv_s  # (BLOCK_K,)
        K_tile = tl.load(
            KV_ptr + kv_bases[:, None] + d_range[None, :] * stride_kv_d,
            mask=valid_mask[:, None] & (d_range[None, :] < D),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_K, D)

        # --- Gather V tile: (BLOCK_K, DV) f16 ---
        V_tile = tl.load(
            KV_ptr + kv_bases[:, None] + dv_range[None, :] * stride_kv_d,
            mask=valid_mask[:, None] & (dv_range[None, :] < DV),
            other=0.0,
        ).to(tl.float16)  # (BLOCK_K, DV)

        # --- Score GEMM: (BLOCK_H, D) @ (D, BLOCK_K) → (BLOCK_H, BLOCK_K) ---
        scores = tl.dot(Q_tile, tl.trans(K_tile))  # (BLOCK_H, BLOCK_K) f32
        scores = scores * softmax_scale

        # --- Recompute P = exp(scores - lse) * valid_mask ---
        P = tl.exp(scores - lse_vals[:, None])  # (BLOCK_H, BLOCK_K) f32
        P = tl.where(valid_mask[None, :], P, 0.0)

        # --- dOV GEMM: (BLOCK_H, DV) @ (DV, BLOCK_K) → (BLOCK_H, BLOCK_K) ---
        # dOV[h, k] = sum_d(dO[h, d] * V[k, d])
        dOV = tl.dot(dO_tile, tl.trans(V_tile))  # (BLOCK_H, BLOCK_K) f32

        # --- dS = P * (dOV - Di) * softmax_scale ---
        dS = P * (dOV - Di[:, None]) * softmax_scale  # (BLOCK_H, BLOCK_K) f32

        # --- dQ GEMM: (BLOCK_H, BLOCK_K) @ (BLOCK_K, D) → (BLOCK_H, D) ---
        dq_acc += tl.dot(dS.to(tl.float16), K_tile)  # WGMMA, f32 accumulator

        # --- dK and dV via per-position reduction ---
        # We have P(BLOCK_H, BLOCK_K) and dS(BLOCK_H, BLOCK_K) computed from
        # tl.dot scores (consistent with forward). For each position k:
        #   dK[k] = sum_h(dS[h,k] * Q[h,:])  → (D,)
        #   dV[k] = sum_h(P[h,k] * dO[h,:])  → (DV,)
        # Extract column k from P/dS using a compile-time mask (k_off is unrolled).
        for k_off in range(BLOCK_K):
            k_pos = tile_start + k_off
            if k_pos < TopK:
                kv_idx = tl.load(IDX_ptr + idx_base + k_pos * stride_idx_k)
                is_valid = kv_idx >= 0
                if is_valid:
                    kv_offset = kv_idx * stride_kv_s

                    # Extract column k_off from dS and P using compile-time mask.
                    # Since k_off is a compile-time constant (range unrolled),
                    # the mask `tl.arange(0, BLOCK_K) == k_off` is resolved at
                    # compile time and the tl.sum reduces to selecting one column.
                    col_mask = tl.arange(0, BLOCK_K) == k_off  # (BLOCK_K,) compile-time
                    # dS_col[h] = dS[h, k_off]
                    dS_col = tl.sum(tl.where(col_mask[None, :], dS, 0.0), axis=1)  # (BLOCK_H,)
                    # P_col[h] = P[h, k_off]
                    P_col = tl.sum(tl.where(col_mask[None, :], P, 0.0), axis=1)  # (BLOCK_H,)

                    # dK = sum_h(dS_col[h] * Q[h,:]) → (D,)
                    dK_vec = tl.sum(dS_col[:, None] * Q_tile.to(tl.float32), axis=0)  # (D,)
                    # dV = sum_h(P_col[h] * dO[h,:]) → (DV,)
                    dV_vec = tl.sum(P_col[:, None] * dO_tile.to(tl.float32), axis=0)  # (DV,)

                    tl.atomic_add(
                        DKV_ptr + kv_offset + d_range * stride_kv_d,
                        dK_vec,
                        mask=d_range < D,
                    )
                    tl.atomic_add(
                        DKV_ptr + kv_offset + dv_range * stride_kv_d,
                        dV_vec,
                        mask=dv_range < DV,
                    )

    # =========================================================================
    # Store dQ (non-atomic, per-query exclusive)
    # =========================================================================
    tl.store(
        DQ_ptr + q_base + h_range[:, None] * stride_q_h + d_range[None, :] * stride_q_d,
        dq_acc.to(tl.bfloat16),
        mask=(h_range[:, None] < H) & (d_range[None, :] < D),
    )


# ---------------------------------------------------------------------------
# Python wrapper functions
# ---------------------------------------------------------------------------


def _triton_sparse_attn_bwd_hp(
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
    """Head-parallel WGMMA backward for sparse attention.

    Uses tl.dot() for all five GEMMs in the backward pass:
    - Score recompute: Q @ K^T
    - dOV: dO @ V^T
    - dQ: dS @ K
    - dK: dS^T @ Q
    - dV: P^T @ dO

    Requirements (same as HP forward):
    - Shared indices: topk_idxs.stride(1) == 0
    - H >= 16, divisible by BLOCK_H(=16)
    - D, DV multiples of 16

    BLOCK_K and num_warps are selected by @triton.autotune keyed on (TopK, D, DV, H).
    """
    total_Sq, H, D = q.shape
    total_Skv = kv.shape[0]
    D_full = kv.shape[-1] if kv.dim() > 1 else D
    TopK = topk_idxs.shape[-1]

    BLOCK_H = 16

    # Ensure topk_idxs is contiguous (expanded tensors with stride=0 cause issues)
    if not topk_idxs.is_contiguous():
        topk_idxs = topk_idxs.contiguous()

    # Allocate gradient outputs
    dq = torch.zeros_like(q)  # bf16, per-query exclusive write
    dkv = torch.zeros((total_Skv, D_full), dtype=torch.float32, device=q.device)

    num_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    grid = (total_Sq, num_head_blocks)

    # BLOCK_K, num_warps, num_stages chosen by @triton.autotune
    _sparse_attn_bwd_hp_kernel[grid](
        q, kv, topk_idxs, out, lse, dO,
        dq, dkv,
        softmax_scale,
        total_Sq, total_Skv, TopK, D, d_v,
        H,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(-1) if kv.dim() > 1 else 1,
        topk_idxs.stride(0), topk_idxs.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        BLOCK_H=BLOCK_H,
    )

    # Compute d_sink in Python (same as existing backward)
    d_sink = None
    if attn_sink is not None:
        p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, H)
        Di = (dO.float() * out.float()).sum(-1)  # (total_Sq, H)
        ds_sink = -p_sink * Di  # (total_Sq, H)
        d_sink = ds_sink.sum(0)  # (H,)

    return {"dq": dq, "dkv": dkv.to(torch.bfloat16), "d_sink": d_sink}


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
        # P1: Software pipelining for 1D forward kernel.
        num_stages=2,
    )

    # Handle edge case: if indexer_topk >= TopK, lse_indexer should equal lse
    if indexer_topk > 0 and indexer_topk >= TopK:
        lse_indexer = lse.clone()

    return out, lse, lse_indexer


def _triton_sparse_attn_fwd_hp(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Head-parallel Triton sparse attention forward using WGMMA.

    Exploits MLA shared-index structure: all H heads share the same TopK
    indices per query position. The KV gather is amortized across heads,
    and both score and output accumulation use tl.dot() → WGMMA on Hopper.

    This kernel is faster than cuBLAS BMM for all TopK values because:
    - Zero intermediate GMEM allocation (scores/P never materialize)
    - Single kernel launch (vs 4 for gather + bmm + softmax + bmm)
    - KV gather shared across all heads (same memory traffic, more compute reuse)

    Requirements:
    - Shared indices: topk_idxs.stride(1) == 0
    - H must be >= 16 and divisible by BLOCK_H(=16)
    - D and DV must be multiples of 16 (WGMMA alignment)

    BLOCK_K and num_warps are selected by @triton.autotune keyed on (TopK, D, DV, H).
    """
    total_Sq, H, D = q.shape
    total_Skv = kv.shape[0]
    TopK = topk_idxs.shape[-1]

    BLOCK_H = 16

    # Allocate outputs
    out = torch.empty((total_Sq, H, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((total_Sq, H), dtype=torch.float32, device=q.device)
    lse_indexer = None
    if indexer_topk > 0:
        lse_indexer = torch.empty((total_Sq, H), dtype=torch.float32, device=q.device)

    # Grid: one program per (query_position, head_block)
    num_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    grid = (total_Sq, num_head_blocks)

    # BLOCK_K, num_warps, num_stages chosen by @triton.autotune
    _sparse_attn_fwd_hp_kernel[grid](
        q, kv, topk_idxs, out, lse,
        attn_sink if attn_sink is not None else torch.empty(0, device=q.device),
        lse_indexer if lse_indexer is not None else torch.empty(0, device=q.device),
        softmax_scale,
        total_Sq, total_Skv, TopK, D, d_v,
        H,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(-1) if kv.dim() > 1 else 1,
        topk_idxs.stride(0), topk_idxs.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_SINK=(attn_sink is not None),
        HAS_LSE_IDX=(indexer_topk > 0),
        INDEXER_TOPK=indexer_topk if indexer_topk > 0 else 0,
        BLOCK_H=BLOCK_H,
    )

    # Handle edge case
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
    """2D-tiled Triton sparse attention forward with tiled softmax.

    Uses BLOCK_Q queries per program and processes KV positions in tiles of
    BLOCK_K. The tiled approach reduces online softmax rescale operations from
    TopK to TopK/BLOCK_K, providing significant speedup for large TopK values.
    """
    total_Sq, H, D = q.shape
    total_Skv = kv.shape[0]
    TopK = topk_idxs.shape[-1]

    BLOCK_Q = 16
    # Choose BLOCK_K: balance rescale reduction vs register pressure.
    # Must be >= 16 for WGMMA alignment, and divide TopK evenly is preferred
    # (non-divisible case handled by the kernel's `if k_pos < TopK` guard).
    if TopK <= 32:
        BLOCK_K = 16
    elif TopK <= 128:
        BLOCK_K = 16
    else:
        BLOCK_K = 32

    # P1: Choose num_warps for Hopper — more warps help hide memory latency
    # for the large D=512 gather pattern. 4 warps is optimal for BLOCK_Q=16
    # with D=512 (each warp handles 4 queries' worth of 512-wide vectors).
    num_warps = 4 if D <= 512 else 8

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
        BLOCK_K=BLOCK_K,
        # P1: Software pipelining — Triton inserts async copy + double buffering
        # for loads within the tiled loop. 2 stages = double buffer (current +
        # prefetch next), which overlaps memory latency with compute.
        num_stages=2,
        num_warps=num_warps,
    )

    # Handle edge case
    if indexer_topk > 0 and indexer_topk >= TopK:
        lse_indexer = lse.clone()

    return out, lse, lse_indexer


# Threshold: use PyTorch-native path for TopK <= this value.
# The HP WGMMA kernel handles all shared-index cases (checked first in dispatch),
# so this threshold only applies to non-shared or non-aligned fallback scenarios.
# After P0 (tiled softmax) and P1 (software pipelining) optimizations, the 2D
# Triton kernel is competitive with PyTorch BMM at lower TopK values. Lowered
# from 512 to 256: beyond TopK=256, the gathered KV tensor becomes large enough
# that Triton's streaming (zero intermediate allocation) wins clearly.
_PYTORCH_FWD_TOPK_THRESHOLD = 256

# Maximum total elements in gathered KV before falling back to Triton.
# Lowered from 2M to 1M: with optimized Triton kernels, the crossover point
# where Triton's streaming beats cuBLAS BMM shifts earlier.
_PYTORCH_FWD_MAX_GATHER_ELEMENTS = 1 * 1024 * 1024


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
    """Sparse attention forward — dispatches to optimal kernel.

    Dispatch priority:
    1. Head-parallel WGMMA kernel (shared indices, H>=16, dims aligned to 16)
       — uses tl.dot for both score and output, beats cuBLAS for all TopK.
    2. PyTorch BMM fallback (non-shared indices with small TopK).
    3. 2D-tiled Triton kernel (non-shared indices with large TopK).

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
    D = q.shape[-1]
    shared = (topk_idxs.stride(1) == 0)

    # Priority 1: Head-parallel WGMMA kernel for shared-index MLA
    # Requirements: shared indices, H>=16, D and d_v are multiples of 16
    if shared and H >= 16 and (H % 16 == 0) and (D % 16 == 0) and (d_v % 16 == 0):
        return _triton_sparse_attn_fwd_hp(
            q, kv, topk_idxs, softmax_scale, d_v, attn_sink, indexer_topk
        )

    # Priority 2: PyTorch BMM for small TopK (non-shared or non-aligned case)
    gather_elements = total_Sq * TopK if shared else total_Sq * H * TopK
    if TopK <= _PYTORCH_FWD_TOPK_THRESHOLD and gather_elements <= _PYTORCH_FWD_MAX_GATHER_ELEMENTS:
        return _pytorch_sparse_attn_fwd(
            q, kv, topk_idxs, softmax_scale, d_v, attn_sink, indexer_topk
        )

    # Priority 3: 2D-tiled Triton kernel for non-shared large TopK
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
    """Triton sparse attention backward — per-position kernel.

    NOTE: The HP backward kernel (_triton_sparse_attn_bwd_hp) is disabled because
    extracting columns from tl.dot outputs produces NaN gradients. When the HP
    forward was used, the caller should use _bmm_backward with score_dtype=f16
    (cuBLAS f16×f16→f32 matches tl.dot accumulation). This Triton kernel is used
    only when the forward also used per-position tl.sum(q*k) score computation.

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
    TopK = topk_idxs.shape[-1]

    # Per-position Triton backward (original kernel)
    total_Skv = kv.shape[0]
    D_full = kv.shape[-1] if kv.dim() > 1 else D

    # Ensure topk_idxs is contiguous
    if not topk_idxs.is_contiguous():
        topk_idxs = topk_idxs.contiguous()

    if TopK <= 32:
        BLOCK_K = 16
    elif TopK <= 128:
        BLOCK_K = 32
    else:
        BLOCK_K = 64

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
        num_stages=2,
        num_warps=4,
    )

    # Compute d_sink if needed
    d_sink = None
    if attn_sink is not None:
        p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, H)
        dO_f = dO.float()
        out_f = out.float()
        Di = (dO_f * out_f).sum(-1)  # (total_Sq, H)
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
