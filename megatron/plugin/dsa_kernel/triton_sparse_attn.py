# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton sparse attention HP (Head-Parallel) WGMMA forward kernel for DSA training.

Replaces the dependency on ``flash_mla.flash_mla_sparse_fwd`` with a pure Triton
implementation optimized for Hopper (SM90) WGMMA tensor core operations.

The kernel operates on "flat" (unbatched) tensors where Q and KV are concatenated
across the batch dimension, and topk_idxs provides global indices into KV.

Training requirements (always satisfied):
  - Shared indices: all heads use the same TopK indices (MLA structure)
  - H >= 16, H % 16 == 0, D % 16 == 0, d_v % 16 == 0

Non-HP fallback (forward/backward) uses PyTorch BMM in
``megatron.plugin.dsa_kernel.legacy.pytorch_sparse_attn``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl


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
    ).to(tl.bfloat16)  # (BLOCK_H, D) bf16 for tl.dot

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
        ).to(tl.bfloat16)  # (BLOCK_K, D) bf16

        # --- Score GEMM: (BLOCK_H, D) @ (D, BLOCK_K) → (BLOCK_H, BLOCK_K) ---
        # tl.dot uses bf16 inputs with f32 accumulator (WGMMA on Hopper)
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
        ).to(tl.bfloat16)  # (BLOCK_K, DV) bf16

        # --- Output GEMM: (BLOCK_H, BLOCK_K) @ (BLOCK_K, DV) → (BLOCK_H, DV) ---
        acc += tl.dot(P_tile.to(tl.bfloat16), V_tile)  # WGMMA, f32 accumulator

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
# HP Forward Wrapper
# ---------------------------------------------------------------------------


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



# ---------------------------------------------------------------------------
# Forward Dispatch
# ---------------------------------------------------------------------------


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
    """Sparse attention forward — HP WGMMA kernel for training.

    Training always uses the head-parallel WGMMA kernel (shared indices,
    H>=16, dims aligned to 16). For non-HP scenarios (decoding, testing),
    falls back to legacy implementations.

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
    D = q.shape[-1]
    H = topk_idxs.shape[1]
    shared = (topk_idxs.stride(1) == 0)

    # HP kernel conditions (always true in training)
    hp_eligible = (
        shared and H >= 16 and (H % 16 == 0)
        and (D % 16 == 0) and (d_v % 16 == 0)
    )

    if hp_eligible:
        return _triton_sparse_attn_fwd_hp(
            q, kv, topk_idxs, softmax_scale, d_v, attn_sink, indexer_topk
        )

    # Fallback to legacy (should not happen in training)
    import warnings
    warnings.warn(
        "triton_sparse_attn_fwd: HP kernel conditions not met "
        f"(shared={shared}, H={H}, D={D}, d_v={d_v}). "
        "Falling back to legacy PyTorch BMM implementation.",
        RuntimeWarning,
        stacklevel=2,
    )
    from megatron.plugin.dsa_kernel.legacy.pytorch_sparse_attn import (
        pytorch_sparse_attn_fwd,
    )
    return pytorch_sparse_attn_fwd(
        q, kv, topk_idxs, softmax_scale, d_v, attn_sink, indexer_topk
    )


# ---------------------------------------------------------------------------
# Backward (delegated to legacy or BMM — training uses _hp_bmm_backward)
# ---------------------------------------------------------------------------


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
    """Sparse attention backward — PyTorch BMM fallback.

    In training, the HP forward path uses _hp_bmm_backward (cuBLAS bf16 BMM)
    instead of this function. This entry point exists for API compatibility
    and for non-HP scenarios (testing/decoding).

    Returns:
        dict with keys: ``dq``, ``dkv``, ``d_sink``.
    """
    from megatron.plugin.dsa_kernel.legacy.pytorch_sparse_attn import (
        pytorch_sparse_attn_bwd,
    )
    dq, dkv, d_sink = pytorch_sparse_attn_bwd(
        dO, q, kv, topk_idxs, out, lse, attn_sink, softmax_scale, d_v
    )
    return {"dq": dq, "dkv": dkv, "d_sink": d_sink}


# ---------------------------------------------------------------------------
# Fused backward utilities (Triton epilogue kernels for bf16 BMM path)
# ---------------------------------------------------------------------------

from megatron.plugin.dsa_kernel.triton_sparse_attn_bwd import (
    fused_mask_scatter_add,
    fused_exp_mask,
    fused_dq,
    fused_dkv,
    sorted_scatter_add,
    should_use_triton_bwd,
)


# ---------------------------------------------------------------------------
# Aliases for backward-compatible import names
# ---------------------------------------------------------------------------

triton_sparse_attn_forward = triton_sparse_attn_fwd
triton_sparse_attn_backward = triton_sparse_attn_bwd
