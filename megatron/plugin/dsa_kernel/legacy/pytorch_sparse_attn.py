# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Legacy PyTorch BMM sparse attention implementations.

These implementations use cuBLAS batched GEMM (torch.bmm) for score computation.
They are retained for:
  1. Low-load / decoding scenarios (small TopK, short sequences)
  2. Numerical reference testing against Triton kernels
  3. Fallback when Triton compilation is unavailable

Enable via environment variable: DSA_USE_LEGACY=1

For training (seq>=2048, TopK>=256), use the pure Triton implementations
in the parent package (triton_sparse_attn.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Forward: PyTorch BMM sparse attention
# ---------------------------------------------------------------------------


def pytorch_sparse_attn_fwd(
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

    Best suited for small TopK (<=256) and short sequences where cuBLAS
    kernel launch overhead is amortized.

    Args:
        q: Query tensor ``(total_S_q, H, D)`` bf16.
        kv: KV tensor ``(total_S_kv, D_full)`` bf16.
        topk_idxs: ``(total_S_q, H, TopK)`` int32.
        softmax_scale: attention scale factor.
        d_v: value dimension.
        attn_sink: ``(H,)`` f32 — per-head bias-only sink.
        indexer_topk: if > 0, compute separate LSE for first positions.

    Returns:
        out: ``(total_S_q, H, d_v)`` bf16.
        lse: ``(total_S_q, H)`` f32.
        lse_indexer: ``(total_S_q, H)`` f32 if indexer_topk > 0, else None.
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

        # Compute scores in f32 via bmm for precision.
        q_r = q.float().reshape(total_Sq * H, 1, D)  # (SH, 1, D) f32
        k_r = kv_gathered[:, :, :, :D].float().reshape(total_Sq * H, TopK, D).transpose(1, 2)
        scores = torch.bmm(q_r, k_r).squeeze(1).reshape(total_Sq, H, TopK) * softmax_scale
        del q_r, k_r
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        # LSE with optional sink
        if attn_sink is not None:
            sink_expanded = attn_sink.unsqueeze(0).unsqueeze(-1).expand(total_Sq, -1, -1)
            scores_with_sink = torch.cat([scores, sink_expanded], dim=-1)
            lse = torch.logsumexp(scores_with_sink, dim=-1)
        else:
            lse = torch.logsumexp(scores, dim=-1)

        # Attention weights (f32 for precision)
        P = torch.exp(scores - lse.unsqueeze(-1))
        P = P.masked_fill(~valid_mask, 0.0)

        # Output via bmm in bf16
        P_bf16 = P.to(torch.bfloat16).reshape(total_Sq * H, 1, TopK)
        v_r = kv_gathered[:, :, :, :d_v].reshape(total_Sq * H, TopK, d_v)
        out = torch.bmm(P_bf16, v_r).squeeze(1).reshape(total_Sq, H, d_v)

        # Indexer LSE
        lse_indexer = None
        if indexer_topk > 0:
            if indexer_topk >= TopK:
                lse_indexer = lse.clone()
            else:
                idx_scores = scores[:, :, :indexer_topk]
                lse_indexer = torch.logsumexp(idx_scores, dim=-1)

        return out.to(torch.bfloat16), lse, lse_indexer


# ---------------------------------------------------------------------------
# Backward: PyTorch BMM sparse attention (f32 score recomputation)
# ---------------------------------------------------------------------------


def pytorch_sparse_attn_bwd(
    grad_out: Tensor,   # (total_Sq, H, d_v) bf16
    query: Tensor,      # (total_Sq, H, D) bf16
    kv: Tensor,         # (total_Skv, D_full) bf16
    topk_idxs: Tensor,  # (total_Sq, H, TopK) int32, shared (stride(1)==0)
    out: Tensor,        # (total_Sq, H, d_v) bf16
    lse: Tensor,        # (total_Sq, H) f32
    attn_sink: Optional[Tensor],  # (H,) f32 or None
    softmax_scale: float,
    d_v: int,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """BMM-based backward for shared-index sparse attention (f32 path).

    Recomputes scores via cuBLAS BMM (f32 inputs, same accumulation as
    the PyTorch BMM forward) so that exp(scores - lse) is numerically consistent.

    Returns:
        (dq, dkv, d_sink) where dq is (total_Sq, H, D) and dkv is (total_Skv, D_full).
    """
    total_Sq, H, D = query.shape
    total_Skv = kv.shape[0]
    D_full = kv.shape[-1] if kv.dim() > 1 else D
    TopK = topk_idxs.shape[-1]

    # Shared indices: use head 0
    idxs_shared = topk_idxs[:, 0, :]  # (total_Sq, TopK)
    valid_shared = idxs_shared >= 0    # (total_Sq, TopK)
    safe_shared = idxs_shared.clamp(min=0).long()

    # Gather KV
    flat_idxs = safe_shared.reshape(-1)
    kv_gathered = kv[flat_idxs].reshape(total_Sq, TopK, D_full).float()

    kv_is_shared = (D == d_v == D_full)

    # Di = sum(dO * O) per (query, head)
    Di = (grad_out.float() * out.float()).sum(dim=-1)  # (total_Sq, H)

    # Recompute scores (f32 × f32 → f32)
    q_f32 = query.float()
    scores = torch.bmm(q_f32, kv_gathered[:, :, :D].transpose(1, 2)) * softmax_scale

    # P = exp(scores - lse) * valid_mask
    valid_exp = valid_shared.unsqueeze(1).expand(-1, H, -1)
    P = torch.exp(scores - lse.unsqueeze(-1))
    del scores
    P.masked_fill_(~valid_exp, 0.0)

    # dov = dO @ V^T (f32)
    dO_f32 = grad_out.float()
    dov = torch.bmm(dO_f32, kv_gathered[:, :, :d_v].transpose(1, 2))

    # dS = P * (dov - Di) * scale
    dS = P * (dov - Di.unsqueeze(-1)) * softmax_scale
    del dov

    # dQ = dS @ K (f32)
    dq = torch.bmm(dS, kv_gathered[:, :, :D])

    # dKV: dK = dS^T @ Q, dV = P^T @ dO
    dkv_gathered = torch.bmm(dS.transpose(1, 2), q_f32)  # (S, TopK, D) f32
    del dS
    if kv_is_shared:
        torch.baddbmm(dkv_gathered, P.transpose(1, 2), dO_f32, out=dkv_gathered)
    else:
        dv = torch.bmm(P.transpose(1, 2), dO_f32)
        dkv_tmp = torch.zeros(
            total_Sq, TopK, D_full, dtype=torch.float32, device=query.device
        )
        dkv_tmp[:, :, :D] = dkv_gathered
        dkv_tmp[:, :, :d_v] += dv
        dkv_gathered = dkv_tmp
    del P, q_f32, dO_f32

    # Scatter dkv_gathered back to full KV positions
    dkv_gathered.masked_fill_(~valid_shared.unsqueeze(-1), 0.0)
    dkv = torch.zeros(total_Skv, D_full, dtype=torch.float32, device=query.device)
    dkv.scatter_add_(
        0,
        flat_idxs.unsqueeze(-1).expand(-1, D_full),
        dkv_gathered.reshape(-1, D_full),
    )

    dq_out = dq.to(query.dtype)
    dkv_out = dkv.to(kv.dtype)

    # d_sink
    d_sink = None
    if attn_sink is not None:
        p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, H)
        ds_sink = -p_sink * Di
        d_sink = ds_sink.sum(0)  # (H,)

    return dq_out, dkv_out, d_sink


# ---------------------------------------------------------------------------
# Backward: PyTorch BMM for FusedIndexerSparseAttn (baseline f32 path)
# ---------------------------------------------------------------------------


def pytorch_fused_bwd(
    grad_output: Tensor,  # (sq, b, np * d_v) or flat
    q_flat: Tensor,       # (total_Sq, np, d) bf16
    kv_flat: Tensor,      # (skv * b, d_kv) bf16
    global_idxs: Tensor,  # (total_Sq, 1, TopK) int32
    out_flat: Tensor,     # (total_Sq, np, d_v) bf16
    lse: Tensor,          # (total_Sq, np) f32
    attn_sink: Optional[Tensor],  # (np,) f32 or None
    softmax_scale: float,
    total_Sq: int,
    np_: int,
    d: int,
    d_v: int,
    d_kv: int,
    skv_b: int,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """PyTorch BMM backward for FusedIndexerSparseAttn (baseline f32 path).

    Fast for small configs (small TopK, short sequences). Uses full f32
    for all intermediate computations.

    Returns:
        (dq, dkv, d_sink) where shapes match the flat layout.
    """
    TopK = global_idxs.shape[-1]

    dO_flat = grad_output.reshape(total_Sq, np_, d_v).float()
    out_f = out_flat.float()
    q_f = q_flat.float()

    # Di = sum(dO * O) per (query, head)
    Di = (dO_flat * out_f).sum(dim=-1)  # (total_Sq, np)

    # Shared indices: (total_Sq, 1, TopK) -> squeeze to (total_Sq, TopK)
    idxs_shared = global_idxs.squeeze(1)  # (total_Sq, TopK)
    valid_shared = idxs_shared >= 0       # (total_Sq, TopK)
    safe_shared = idxs_shared.clamp(min=0).long()

    # Gather KV once: (total_Sq * TopK) -> (total_Sq, TopK, d_kv)
    flat_idxs = safe_shared.reshape(-1)
    kv_gathered = kv_flat[flat_idxs].reshape(total_Sq, TopK, d_kv)

    kv_is_shared = (d == d_v == d_kv)

    # Upcast kv_gathered to f32 for the full backward.
    kv_f = kv_gathered.float()
    del kv_gathered

    # Loop-free backward via batched BMM.
    scores = torch.bmm(q_f, kv_f[:, :, :d].transpose(1, 2)) * softmax_scale
    valid_exp = valid_shared.unsqueeze(1).expand(-1, np_, -1)
    P = torch.exp(scores - lse.unsqueeze(-1))
    del scores
    P.masked_fill_(~valid_exp, 0.0)

    dov = torch.bmm(dO_flat, kv_f[:, :, :d_v].transpose(1, 2))
    dS = P * (dov - Di.unsqueeze(-1)) * softmax_scale
    del dov

    dq = torch.bmm(dS, kv_f[:, :, :d])

    dkv_gathered = torch.bmm(dS.transpose(1, 2), q_f)  # dk: (S, T, d)
    del dS
    if kv_is_shared:
        torch.baddbmm(dkv_gathered, P.transpose(1, 2), dO_flat, out=dkv_gathered)
    else:
        dv = torch.bmm(P.transpose(1, 2), dO_flat)
        dkv_tmp = torch.zeros(total_Sq, TopK, d_kv, dtype=torch.float32, device=q_flat.device)
        dkv_tmp[:, :, :d] = dkv_gathered
        dkv_tmp[:, :, :d_v] += dv
        dkv_gathered = dkv_tmp
    del P, kv_f

    dkv_gathered.masked_fill_(~valid_shared.unsqueeze(-1), 0.0)
    dkv = torch.zeros(skv_b, d_kv, dtype=torch.float32, device=q_flat.device)
    dkv.scatter_add_(
        0,
        flat_idxs.unsqueeze(-1).expand(-1, d_kv),
        dkv_gathered.reshape(-1, d_kv),
    )

    dq_out = dq.to(q_flat.dtype)
    dkv_out = dkv.to(kv_flat.dtype)

    # d_sink
    d_sink = None
    if attn_sink is not None:
        p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, np)
        ds_sink = -p_sink * Di
        d_sink = ds_sink.sum(0)  # (np,)

    return dq_out, dkv_out, d_sink


__all__ = [
    "pytorch_sparse_attn_fwd",
    "pytorch_sparse_attn_bwd",
    "pytorch_fused_bwd",
]
