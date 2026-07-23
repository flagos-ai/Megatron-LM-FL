# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton-based DSA kernel wrappers — drop-in replacement for ``dsa_kernels.py``.

Provides the same high-level API as ``dsa_kernels.py`` but uses Triton kernels
instead of cuDNN DSA namespace and FlashMLA. No external CUDA kernel
dependencies required.

Public API:

* ``build_flat_topk_idxs`` / ``local_to_global_flat`` — index helpers.
* ``dsa_sparse_attn`` — differentiable sparse attention, flat layout (Path A / Path C step 2).
* ``dsa_sparse_attn_sbhd`` — sparse attention with SBHD interface (used by csa.py).
* ``indexer_topk`` — indexer scoring + top-K selection (Path C inference).
* ``fused_indexer_sparse_attn`` — fused indexer loss + sparse attention (Path B training).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.plugin.dsa_kernel.triton_sparse_attn import (
    triton_sparse_attn_forward,
    triton_sparse_attn_backward,
)
from megatron.plugin.dsa_kernel.triton_sparse_attn_bwd import (
    fused_mask_scatter_add,
    fused_exp_mask,
)
from megatron.plugin.dsa_kernel.triton_indexer_kernels import (
    sparse_indexer_score_recompute,
    sparse_attn_score_recompute,
    dense_indexer_score_recompute,
    dense_attn_score_recompute,
    indexer_topk_selection,
    fused_sparse_indexer_loss_and_backward,
    fused_dense_indexer_loss_and_backward,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profiling utilities (enabled via DSA_PROFILE=1 env var)
# ---------------------------------------------------------------------------

_DSA_PROFILE = os.environ.get("DSA_PROFILE", "0") == "1"


class _CudaProfiler:
    """Lightweight CUDA event profiler for forward pass breakdown."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._events = []  # list of (name, start_event, end_event)
        self._current_start = None
        self._current_name = None

    def start(self, name: str):
        if not self.enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._current_start = start
        self._current_name = name

    def stop(self):
        if not self.enabled or self._current_start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append((self._current_name, self._current_start, end))
        self._current_start = None
        self._current_name = None

    def report(self, prefix: str = ""):
        if not self.enabled or not self._events:
            return
        torch.cuda.synchronize()
        total = 0.0
        parts = []
        for name, start, end in self._events:
            elapsed = start.elapsed_time(end)
            total += elapsed
            parts.append(f"    {name}: {elapsed:.3f} ms")
        print(f"{prefix}FusedIndexerSparseAttn forward breakdown (total={total:.3f} ms):")
        for p in parts:
            print(p)
        self._events.clear()


# ---------------------------------------------------------------------------
# Index helpers (pure PyTorch, no kernel dependency)
# ---------------------------------------------------------------------------


def local_to_global_flat(local_idxs: Tensor, batch_size: int, seqlen_kv: int) -> Tensor:
    """Convert local per-batch indices to global flat indices (SB layout).

    Follows the same convention as ``dsa_kernels.local_to_global_flat``:
    flat row order is SB (seq-major); global index is ``local * B + b``
    for valid entries and ``-1`` otherwise.

    Args:
        local_idxs: ``(b, sq, topk)`` int, values in ``[0, seqlen_kv)`` or -1.
        batch_size: ``B``.
        seqlen_kv: KV sequence length per batch.

    Returns:
        ``(sq*b, topk)`` int32.
    """
    b, sq, topk = local_idxs.shape
    assert b == batch_size

    # Permute to SB order: (b, sq, topk) -> (sq, b, topk) -> (sq*b, topk)
    idxs_sb = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
    valid = idxs_sb >= 0
    batch_ids = torch.arange(sq * b, device=local_idxs.device) % b
    batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
    idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
    return idxs_sb.int()


def build_flat_topk_idxs(
    *idx_groups: Tensor, batch_size: int, seqlen_kv: int, compact: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    """Combine local per-batch index groups and convert to flat global form.

    Drop-in replacement for ``dsa_kernels.build_flat_topk_idxs`` that uses
    PyTorch argsort for compact instead of cuDNN's ``compactify_wrapper``.

    Each *idx_group* is ``(b, sq, topk_i)`` with local per-batch KV indices.
    ``-1`` marks invalid positions.

    Args:
        *idx_groups: one or more ``(b, sq, topk_i)`` int tensors.
        batch_size: ``B``.
        seqlen_kv: total KV sequence length per batch.
        compact: if True, pack valid entries to the front of each row and
            additionally return ``topk_length``; if False, leave as-is.

    Returns:
        ``(topk_idxs, topk_length)`` where
        ``topk_idxs`` is ``(sq*b, total_topk)`` int32 (flat global) and
        ``topk_length`` is ``(sq*b,)`` int32 when ``compact``, else ``None``.
    """
    combined = torch.cat(idx_groups, dim=-1)  # (b, sq, total_topk)
    global_idxs = local_to_global_flat(combined, batch_size, seqlen_kv)

    topk_length_flat = None
    if compact:
        valid_mask = global_idxs >= 0
        sorted_indices = valid_mask.int().argsort(dim=-1, descending=True, stable=True)
        global_idxs = global_idxs.gather(-1, sorted_indices)
        topk_length_flat = valid_mask.sum(dim=-1).int()

    return global_idxs, topk_length_flat


# ---------------------------------------------------------------------------
# Helper: SBHD <-> BSHD conversions (matches dsa_kernels.py layout conventions)
# ---------------------------------------------------------------------------


def _sbhd_to_bshd_indexer_inputs(
    q_indexer: Tensor,  # (sq, b, idx_nh, idx_hd)
    k_indexer: Tensor,  # (sk, b, idx_hd)
    weights: Tensor,    # (sq, b, idx_nh)
    indexer_softmax_scale: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Transpose indexer inputs from SBHD to BSHD layout.

    Note: .contiguous() is omitted — downstream consumers (einsum, topk,
    elementwise ops) all handle strided tensors correctly.
    """
    q_bshd = q_indexer.permute(1, 0, 2, 3)   # (b, sq, nh, hd)
    k_bsd = k_indexer.permute(1, 0, 2)       # (b, sk, hd)
    w_bsh = weights.permute(1, 0, 2)         # (b, sq, nh)
    # Scale weights
    w_bsh_scaled = w_bsh * indexer_softmax_scale
    return q_bshd, k_bsd, w_bsh, w_bsh_scaled


def _indexer_topk_bshd(
    q_bshd: Tensor,    # (B, S_q, H_q, D)
    k_bsd: Tensor,     # (B, S_k, D)
    w_bsh: Tensor,     # (B, S_q, H_q) — already scaled
    topk: int,
    ratio: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute indexer scores and top-K selection.

    Returns:
        topk_indices: (B, S_q, topk) int32
        topk_length: (B, S_q) int32
        full_scores: (B, S_q, S_k) fp32
    """
    scores = indexer_topk_selection(q_bshd, k_bsd, w_bsh, topk, ratio)
    topk_indices = scores["topk_indices"]
    topk_length = scores["topk_length"]
    full_scores = scores["full_scores"]
    return topk_indices, topk_length, full_scores


# ---------------------------------------------------------------------------
# Path A / Path C: dsa_sparse_attn
# ---------------------------------------------------------------------------


class _DSASparseAttnFunc(torch.autograd.Function):
    """Differentiable sparse attention using pure Triton kernels.

    Forward dispatches to HP WGMMA or 2D-tiled Triton kernel (no PyTorch BMM).

    The backward adapts to match the forward path's dot-product method:
    - When forward used the HP WGMMA kernel (tl.dot f16×f16→f32), backward uses
      cuBLAS BMM with f16 inputs for numerically consistent score recomputation.
    - When forward used the 2D Triton kernel (tl.sum(q*k)), backward uses the
      Triton per-position kernel which shares the same accumulation order.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,     # (total_Sq, H, D)
        kv: Tensor,        # (total_Skv, D_kv) where D_kv >= D
        topk_idxs: Tensor, # (total_Sq, H, TopK) or (total_Sq, 1, TopK)
        softmax_scale: float,
        d_v: int,
        attn_sink: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        out, lse, _ = triton_sparse_attn_forward(
            query, kv, topk_idxs, softmax_scale, d_v, attn_sink
        )
        ctx.has_attn_sink = attn_sink is not None

        # Determine if the HP WGMMA kernel was used (shared + aligned dims).
        # Backward needs this to select numerically consistent score recomputation.
        D = query.shape[-1]
        H = topk_idxs.shape[1]
        shared = (topk_idxs.stride(1) == 0)

        ctx.used_hp_fwd = (
            shared and H >= 16 and (H % 16 == 0)
            and (D % 16 == 0) and (d_v % 16 == 0)
        )

        logger.debug(
            "_DSASparseAttnFunc.forward: total_Sq=%d, H=%d, TopK=%d, shared=%s, "
            "fwd_path=%s",
            topk_idxs.shape[0], H, topk_idxs.shape[-1], shared,
            "hp_wgmma" if ctx.used_hp_fwd else "triton_2d",
        )

        if attn_sink is not None:
            ctx.save_for_backward(query, kv, topk_idxs, out, lse, attn_sink)
        else:
            ctx.save_for_backward(query, kv, topk_idxs, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.d_v = d_v
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        if ctx.has_attn_sink:
            query, kv, topk_idxs, out, lse, attn_sink = ctx.saved_tensors
        else:
            query, kv, topk_idxs, out, lse = ctx.saved_tensors
            attn_sink = None

        logger.debug(
            "_DSASparseAttnFunc.backward: used_hp_fwd=%s, bwd_path=%s",
            ctx.used_hp_fwd,
            "bmm_f16" if ctx.used_hp_fwd else "triton",
        )

        if ctx.used_hp_fwd:
            # --- BMM backward with f16 score recomputation ---
            # HP forward used tl.dot(Q_f16, K_f16^T) → f32 accumulator.
            # cuBLAS BMM with f16 inputs also does f16×f16→f32 accumulation,
            # so exp(scores_bwd - lse_fwd) is numerically consistent.
            dq, dkv, d_sink = _DSASparseAttnFunc._hp_bmm_backward(
                grad_out, query, kv, topk_idxs, out, lse, attn_sink,
                ctx.softmax_scale, ctx.d_v,
            )
        else:
            # --- Triton backward: forward used 2D Triton kernel ---
            # Both fwd and bwd use tl.sum(q * k) → same dot product, consistent.
            bwd_result = triton_sparse_attn_backward(
                grad_out, query, kv, out, lse, topk_idxs,
                ctx.softmax_scale, ctx.d_v, attn_sink
            )
            dq, dkv, d_sink = bwd_result["dq"], bwd_result["dkv"], bwd_result["d_sink"]

        return dq, dkv, None, None, None, d_sink

    @staticmethod
    def _hp_bmm_backward(
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
        """BMM-based backward for HP WGMMA forward path.

        Uses cuBLAS BMM with f16 inputs for score recomputation, matching the
        HP forward kernel's tl.dot(Q_f16, K_f16^T) → f32 accumulation.
        This ensures exp(scores_bwd - lse_fwd) is numerically consistent.
        """
        total_Sq, H, D = query.shape
        total_Skv = kv.shape[0]
        D_full = kv.shape[-1] if kv.dim() > 1 else D
        TopK = topk_idxs.shape[-1]

        # Shared indices: (total_Sq, H, TopK) with stride(1)==0 → use head 0
        idxs_shared = topk_idxs[:, 0, :]  # (total_Sq, TopK)
        valid_shared = idxs_shared >= 0    # (total_Sq, TopK)
        safe_shared = idxs_shared.clamp(min=0).long()

        # Gather KV once in bf16
        flat_idxs = safe_shared.reshape(-1)  # (total_Sq * TopK)
        kv_gathered = kv[flat_idxs].reshape(total_Sq, TopK, D_full)  # bf16

        kv_is_shared = (D == d_v == D_full)

        # Di = sum(dO * O) per (query, head)
        Di = (grad_out.float() * out.float()).sum(dim=-1)  # (total_Sq, H)

        # Recompute scores via BMM in f16 — matches HP WGMMA forward
        # (tl.dot(Q_f16, K_f16^T) → f32 accumulator)
        q_score = query.to(torch.float16)  # (S, H, D)
        k_score = kv_gathered[:, :, :D].to(torch.float16)  # (S, TopK, D)
        scores = torch.bmm(
            q_score, k_score.transpose(1, 2)
        ).float() * softmax_scale  # always f32 output
        del q_score, k_score

        # P = exp(scores - lse) * valid_mask
        P = fused_exp_mask(scores, lse, valid_shared, H)
        del scores

        # dov = dO @ V^T (f32 for precision in dS computation)
        dov = torch.bmm(
            grad_out.float(), kv_gathered[:, :, :d_v].float().transpose(1, 2)
        )  # (S, H, TopK) f32

        # dS = P * (dov - Di) * scale
        dS = P * (dov - Di.unsqueeze(-1)) * softmax_scale
        del dov

        # dQ = dS @ K (f32 for precision)
        dq = torch.bmm(dS, kv_gathered[:, :, :D].float())  # (S, H, D) f32

        # dKV: dK = dS^T @ Q, dV = P^T @ dO
        q_f32 = query.float()
        dO_f32 = grad_out.float()

        dkv_gathered = torch.bmm(dS.transpose(1, 2), q_f32)  # (S, TopK, D) f32
        del dS
        if kv_is_shared:
            # dV = P^T @ dO: in-place add (dK + dV into same buffer)
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
        valid_flat = valid_shared.reshape(-1)
        dkv = torch.zeros(total_Skv, D_full, dtype=torch.float32, device=query.device)
        fused_mask_scatter_add(dkv_gathered, flat_idxs, valid_flat, dkv)

        dq_out = dq.to(query.dtype)
        dkv_out = dkv.to(kv.dtype)

        # d_sink: gradient of the bias-only attention sink
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, H)
            ds_sink = -p_sink * Di  # (total_Sq, H)
            d_sink = ds_sink.sum(0)  # (H,)

        return dq_out, dkv_out, d_sink


def dsa_sparse_attn(
    query: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Sparse attention forward (differentiable).

    Drop-in replacement for the FlashMLA-based ``dsa_sparse_attn`` in
    ``dsa_kernels.py``.

    Args:
        query: ``(total_S_q, H, D)`` bf16 — flat queries.
        kv: ``(total_S_kv, D_kv)`` bf16 — flat KV (single-head).
        topk_idxs: ``(total_S_q, H_kv, TopK)`` int32 — global KV indices.
            H_kv is typically 1 for MQA.
        softmax_scale: scaling applied to Q @ K^T.
        d_v: value dimension (typically 512 for DSA).
        attn_sink: ``(H,)`` f32 — per-head sink bias (optional).
        topk_length: ``(total_S_q, H_kv)`` int32 — valid count per query (optional,
            for compact mode). If None, -1 entries in topk_idxs are used as mask.
        indexer_topk: if > 0, compute separate LSE for first ``indexer_topk``
            positions (used by fused indexer path).

    Returns:
        ``(out, lse, lse_indexer)``
        - out: ``(total_S_q, H, d_v)`` bf16.
        - lse: ``(total_S_q, H)`` fp32.
        - lse_indexer: ``(total_S_q, H)`` fp32 or None.
    """
    # Expand topk_idxs to per-head if needed (H_kv=1 -> broadcast)
    total_Sq, H, D = query.shape
    if topk_idxs.shape[1] == 1 and H > 1:
        topk_idxs = topk_idxs.expand(-1, H, -1)

    if indexer_topk > 0:
        # Split computation: full attention + indexer-only LSE
        out, lse = _DSASparseAttnFunc.apply(
            query, kv, topk_idxs, softmax_scale, d_v, attn_sink
        )
        # Compute LSE for first indexer_topk positions
        TopK = topk_idxs.shape[-1]
        if indexer_topk >= TopK:
            lse_indexer = lse.clone()
        else:
            idx_subset = topk_idxs[:, :, :indexer_topk].contiguous()
            _, lse_indexer, _ = triton_sparse_attn_forward(
                query, kv, idx_subset, softmax_scale, d_v, attn_sink
            )
        return out, lse, lse_indexer
    else:
        out, lse = _DSASparseAttnFunc.apply(
            query, kv, topk_idxs, softmax_scale, d_v, attn_sink
        )
        return out, lse, None


def dsa_sparse_attn_sbhd(
    query: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tensor:
    """Sparse attention with SBHD interface (matches dsa_kernels.dsa_sparse_attn).

    Reshapes from ``(sq, b, np, d)`` layout to flat ``(total_Sq, H, D)`` and
    delegates to :func:`dsa_sparse_attn`.

    Args:
        query: ``(sq, b, np, d)`` bf16 SBHD.
        kv:    ``(skv, b, d)`` bf16 SBD (K=V).
        attn_sink: ``(np,)`` f32.
        topk_idxs: ``(sq*b, topk)`` int32 — flat global indices.
        softmax_scale: scalar float.
        topk_length: ``(sq*b,)`` int32 — optional compact fast-path.
        indexer_topk: int; 0 for Paths A/C, positive for Path B.

    Returns:
        ``(sq, b, np * d_v)`` bf16 output.
    """
    sq, b, np_, d = query.shape
    skv = kv.shape[0]
    q_flat = query.reshape(sq * b, np_, d)
    kv_flat = kv.reshape(skv * b, d)
    # dsa_sparse_attn expects (total_Sq, H_kv, TopK); core produces (total_Sq, TopK)
    idxs = topk_idxs.unsqueeze(1) if topk_idxs.dim() == 2 else topk_idxs
    tlen = topk_length
    if tlen is not None and tlen.dim() == 1:
        tlen = tlen.unsqueeze(1)
    out_flat, _lse, _ = dsa_sparse_attn(
        q_flat, kv_flat, idxs, softmax_scale, d, attn_sink, tlen, indexer_topk
    )
    d_v = out_flat.shape[-1]
    return out_flat.reshape(sq, b, np_ * d_v)


# ---------------------------------------------------------------------------
# Path C inference: indexer_topk
# ---------------------------------------------------------------------------


def indexer_topk(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk: int,
    ratio: int,
    indexer_softmax_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Indexer scoring + top-K selection (inference path).

    Drop-in replacement for the cuDNN/TRT-LLM-based ``indexer_topk``.

    Args:
        q_indexer: ``(sq, b, idx_nh, idx_hd)`` bf16 SBHD.
        k_indexer: ``(sk, b, idx_hd)`` bf16 SBD.
        weights:   ``(sq, b, idx_nh)`` bf16 SBH — raw (unscaled) weights.
        topk: number of top-K indices to select.
        ratio: compression ratio for the causal mask.
        indexer_softmax_scale: scale applied to indexer scores.

    Returns:
        topk_indices: ``(b, sq, topk)`` int32.
        topk_length:  ``(b, sq)`` int32.
    """
    q_bshd, k_bsd, _w_bsh_raw, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
        q_indexer, k_indexer, weights, indexer_softmax_scale
    )
    topk_indices, topk_length, _ = _indexer_topk_bshd(q_bshd, k_bsd, w_bsh_scaled, topk, ratio)
    return topk_indices, topk_length


# ---------------------------------------------------------------------------
# Path B training: fused_indexer_sparse_attn
# ---------------------------------------------------------------------------


_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny


def _kl_loss_from_target_predict(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) reduced and scaled by loss_coeff."""
    eps = _CLIP_PROB_MIN
    t = target.clamp(min=eps)
    p = predict.clamp(min=eps)
    kl_per_row = (t * (torch.log(t) - torch.log(p))).sum(dim=-1)  # (B, S_q)

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


def _kl_loss_from_dense_scores(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL loss from dense scores (full-KV path)."""
    eps = _CLIP_PROB_MIN
    B, S_q, S_k = attn_score.shape

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    safe_l1 = attn_l1norm.clamp(min=eps)
    safe_lse = index_lse.clone()
    safe_lse[~row_valid] = 0.0

    target = attn_score / safe_l1.unsqueeze(-1)
    target_clamped = target.clamp(min=eps)
    position_valid = torch.isfinite(index_score)
    safe_index_score = torch.where(position_valid, index_score, torch.zeros_like(index_score))
    log_predict = safe_index_score - safe_lse.unsqueeze(-1)

    kl_terms = target_clamped * (torch.log(target_clamped) - log_predict)
    kl_terms = torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms))
    kl_per_row = kl_terms.sum(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


class FusedIndexerSparseAttnFunc(torch.autograd.Function):
    """Path B: fused indexer (+KL loss) + sparse attention.

    Differentiable w.r.t. ``query``, ``kv_full``, ``attn_sink``,
    ``q_indexer``, ``k_indexer``, ``weights``.

    Two indexer-loss variants selected by ``sparse_loss``:
    - Sparse: KL over top-K positions only.
    - Dense: KL over all causally valid KV positions.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,       # (sq, b, np, d)
        kv_full: Tensor,     # (skv, b, d)
        attn_sink: Tensor,   # (np,) f32
        window_idxs: Tensor, # (b, sq, win_topk) int32
        q_indexer: Tensor,   # (sq, b, idx_nh, idx_hd)
        k_indexer: Tensor,   # (n_comp, b, idx_hd)
        weights: Tensor,     # (sq, b, idx_nh) — raw
        indexer_topk: int,
        ratio: int,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        kv_offset: int,
        calculate_per_token_loss: bool,
    ) -> Tuple[Tensor, Tensor]:
        sq, b, np_, d = query.shape
        skv = kv_full.shape[0]
        n_comp = k_indexer.shape[0]
        idx_nh, idx_hd = q_indexer.shape[2], q_indexer.shape[3]

        effective_topk = min(indexer_topk, n_comp)

        logger.debug(
            "FusedIndexerSparseAttnFunc.forward: sq=%d, b=%d, np=%d, d=%d, "
            "skv=%d, n_comp=%d, effective_topk=%d, sparse_loss=%s, "
            "loss_coeff=%.4g",
            sq, b, np_, d, skv, n_comp, effective_topk, sparse_loss, loss_coeff,
        )

        prof = _CudaProfiler(enabled=_DSA_PROFILE)

        # 1. Transpose indexer inputs SBHD -> BSHD
        prof.start("step1_sbhd_to_bshd")
        q_idx_bshd, k_idx_bsd, w_bsh, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
            q_indexer, k_indexer, weights, indexer_softmax_scale
        )
        prof.stop()

        # 2. Indexer scoring + top-K
        prof.start("step2_indexer_topk")
        topk_indices_cmp, _, indexer_scores = _indexer_topk_bshd(
            q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, ratio
        )
        prof.stop()

        # 3. Combine indices (compressed + window)
        prof.start("step3_4_combine_flatten")
        # Add kv_offset to compressed indices
        topk_indices_global = topk_indices_cmp.clone()
        valid_cmp = topk_indices_global >= 0
        topk_indices_global[valid_cmp] += kv_offset

        # Combine: compressed first, then window
        combined_idxs = torch.cat([topk_indices_global, window_idxs], dim=-1)  # (b, sq, total_topk)
        total_topk = combined_idxs.shape[-1]

        # 4. Flatten for sparse attention
        # Use SB (seq-major) flat layout: flat[s * b + batch_idx] = orig[s, batch_idx]
        # query: (sq, b, np, d) -> (sq*b, np, d)  — already SB order via reshape
        # kv_full: (skv, b, d) -> (skv*b, d)      — already SB order via reshape
        q_flat = query.reshape(sq * b, np_, d)
        kv_flat = kv_full.reshape(skv * b, -1)

        # Convert local per-batch indices to global flat indices (SB layout).
        # For SB flat KV: global_idx = local_kv_idx * b + batch_idx
        # combined_idxs: (b, sq, total_topk) with local values in [0, skv)
        batch_ids = torch.arange(b, device=query.device, dtype=combined_idxs.dtype)
        global_idxs = combined_idxs.clone()
        valid_mask = global_idxs >= 0
        global_idxs = torch.where(
            valid_mask,
            global_idxs * b + batch_ids.view(b, 1, 1),
            global_idxs,
        )  # (b, sq, total_topk)
        # Permute to SB order then flatten: (b, sq, topk) -> (sq, b, topk) -> (sq*b, topk)
        global_idxs = global_idxs.permute(1, 0, 2).reshape(sq * b, total_topk)
        # MLA: all heads share KV indices. Keep as (sq*b, 1, TopK) to avoid
        # redundant np_ copies in save_for_backward and backward gather.
        global_idxs = global_idxs.unsqueeze(1)  # (sq*b, 1, total_topk)
        # Expand for forward (uses stride trick, no memory allocation)
        global_idxs_expanded = global_idxs.expand(-1, np_, -1)  # (sq*b, np, total_topk)
        prof.stop()

        # 5. Sparse attention forward
        prof.start("step5_sparse_attn_fwd")
        # When sparse_loss is enabled, compute partial LSE for the first
        # effective_topk positions (compressed indices) in a single pass,
        # avoiding a redundant second forward call.
        _indexer_topk_for_lse = effective_topk if sparse_loss else 0
        out_flat, lse, lse_indexer_raw = triton_sparse_attn_forward(
            q_flat, kv_flat, global_idxs_expanded, softmax_scale, d, attn_sink,
            indexer_topk=_indexer_topk_for_lse,
        )
        prof.stop()

        # 6. Compute indexer loss
        # P3 optimization: skip step 6+7 entirely when loss_coeff == 0
        if loss_coeff == 0:
            indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)
            precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
            precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
            precomputed_grad_weights = torch.zeros_like(weights)

            # Save for backward
            ctx.save_for_backward(
                q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse,
                precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights,
            )
            ctx.softmax_scale = softmax_scale
            ctx.sq = sq
            ctx.b = b
            ctx.np_ = np_
            ctx.d = d
            ctx.skv = skv

            d_v = out_flat.shape[-1]
            output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)
            prof.report(f"  [sq={sq}, b={b}, np={np_}, topk={total_topk}, loss_coeff=0] ")
            return output, indexer_loss

        # 6+7. Fused: compute indexer loss AND pre-compute indexer backward in one pass.
        # This replaces the separate step 6 (score_recompute + KL) and step 7
        # (indexer backward), eliminating redundant gather/einsum operations.
        prof.start("step6_7_indexer_loss_bwd")
        needs_grad = any(
            t.requires_grad for t in (query, kv_full, attn_sink, q_indexer, k_indexer, weights)
        )

        # Prepare attention tensors in BSHD layout (shared by both paths).
        # .contiguous() omitted: downstream einsum/indexing handle strided tensors.
        q_attn_bshd = query.permute(1, 0, 2, 3)  # (b, sq, np, d)
        lse_indexer_bsh = lse_indexer_raw.reshape(sq, b, np_).permute(1, 0, 2) if sparse_loss else None

        if sparse_loss:
            k_attn_bsd = kv_full[:, :, :d].permute(1, 0, 2)  # (b, skv, d)

            if needs_grad:
                # Fused: loss + backward in one pass (single K gather + einsum)
                indexer_loss, precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights = (
                    fused_sparse_indexer_loss_and_backward(
                        q_idx_bshd, k_idx_bsd, w_bsh_scaled,
                        topk_indices_cmp,
                        q_attn_bshd, k_attn_bsd, lse_indexer_bsh,
                        indexer_softmax_scale=indexer_softmax_scale,
                        softmax_scale=softmax_scale,
                        loss_coeff=loss_coeff,
                        calculate_per_token_loss=calculate_per_token_loss,
                        idx_nh=idx_nh,
                        kv_offset=kv_offset,
                    )
                )
                # BSHD -> SBHD (match input layout)
                precomputed_grad_q_indexer = precomputed_grad_q_indexer.permute(1, 0, 2, 3).contiguous()
                precomputed_grad_k_indexer = precomputed_grad_k_indexer.permute(1, 0, 2).contiguous()
                precomputed_grad_weights = precomputed_grad_weights.permute(1, 0, 2).contiguous()
            else:
                # Inference: only compute loss, no backward
                predict_result = sparse_indexer_score_recompute(
                    q_idx_bshd, k_idx_bsd, w_bsh_scaled, topk_indices_cmp,
                    qhead_per_kv_head=idx_nh,
                )
                # Shift valid compressed indices by kv_offset, keep -1 as-is
                topk_for_target = topk_indices_cmp.clone()
                valid_cmp_mask = topk_for_target >= 0
                topk_for_target[valid_cmp_mask] += kv_offset
                target_result = sparse_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_indexer_bsh, topk_for_target,
                    softmax_scale, qhead_per_kv_head=np_,
                )
                indexer_loss = _kl_loss_from_target_predict(
                    target_result["target"], predict_result["predict"],
                    topk_indices_cmp, loss_coeff, calculate_per_token_loss
                )
                precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
                precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
                precomputed_grad_weights = torch.zeros_like(weights)
        else:
            # Dense path
            k_attn_bsd = kv_full[kv_offset:kv_offset + n_comp, :, :d].permute(1, 0, 2)
            lse_bsh = lse.reshape(sq, b, np_).permute(1, 0, 2)

            if needs_grad:
                # Fused: loss + backward in one pass
                indexer_loss, precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights = (
                    fused_dense_indexer_loss_and_backward(
                        q_idx_bshd, k_idx_bsd, w_bsh,
                        topk_indices_cmp,
                        q_attn_bshd, k_attn_bsd, lse_bsh,
                        indexer_softmax_scale=indexer_softmax_scale,
                        softmax_scale=softmax_scale,
                        loss_coeff=loss_coeff,
                        ratio=ratio,
                        calculate_per_token_loss=calculate_per_token_loss,
                        idx_nh=idx_nh,
                    )
                )
                # BSHD -> SBHD (match input layout)
                precomputed_grad_q_indexer = precomputed_grad_q_indexer.permute(1, 0, 2, 3).contiguous()
                precomputed_grad_k_indexer = precomputed_grad_k_indexer.permute(1, 0, 2).contiguous()
                precomputed_grad_weights = precomputed_grad_weights.permute(1, 0, 2).contiguous()
            else:
                # Inference: only compute loss
                dense_idx_result = dense_indexer_score_recompute(
                    q_idx_bshd, k_idx_bsd, w_bsh_scaled,
                    qhead_per_kv_head=idx_nh, sm_scale=1.0, ratio=ratio,
                )
                # Pass lse=None so dense_attn_score_recompute uses self-contained
                # softmax over compressed keys only (matching unfused reference).
                # Using the full LSE (which includes window tokens in the
                # denominator) would make compressed-token probabilities too small.
                dense_attn_result = dense_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, None,
                    qhead_per_kv_head=np_, softmax_scale=softmax_scale, ratio=ratio,
                )
                indexer_loss = _kl_loss_from_dense_scores(
                    dense_attn_result["out"], dense_attn_result["denom"],
                    dense_idx_result["out"], dense_idx_result["denom"],
                    topk_indices_cmp, loss_coeff, calculate_per_token_loss,
                )
                precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
                precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
                precomputed_grad_weights = torch.zeros_like(weights)

        # Save for backward
        prof.stop()
        ctx.save_for_backward(
            q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse,
            precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights,
        )
        ctx.softmax_scale = softmax_scale
        ctx.sq = sq
        ctx.b = b
        ctx.np_ = np_
        ctx.d = d
        ctx.skv = skv

        # Return
        d_v = out_flat.shape[-1]
        output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)

        prof.report(f"  [sq={sq}, b={b}, np={np_}, topk={total_topk}] ")
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        (
            q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse,
            precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights,
        ) = ctx.saved_tensors

        sq, b, np_, d = ctx.sq, ctx.b, ctx.np_, ctx.d
        skv = ctx.skv

        d_v = out_flat.shape[-1]
        total_Sq = sq * b
        TopK = global_idxs.shape[-1]
        d_kv = kv_flat.shape[-1]

        prof = _CudaProfiler(enabled=_DSA_PROFILE)

        logger.debug(
            "FusedIndexerSparseAttnFunc.backward: sq=%d, b=%d, np=%d, "
            "TopK=%d, d_kv=%d, bwd_path=optimized_bf16_bmm",
            sq, b, np_, TopK, d_kv,
        )

        # --- Optimized path: bf16 BMM + Triton fused epilogues ---
        # Training always uses this path (shared indices, large TopK).
        # Key savings vs legacy f32 path:
        #   1. No f32 upcast of kv_gathered (saves 768MB alloc + bandwidth)
        #   2. Fused exp+mask (saves 1 kernel launch + 24MB read/write)
        #   3. Fused mask+scatter (saves 384MB read pass)
        prof.start("bwd_prepare_gather")
        dO_flat = grad_output.reshape(total_Sq, np_, d_v)

        # Shared indices: (total_Sq, 1, TopK) -> squeeze to (total_Sq, TopK)
        idxs_shared = global_idxs.squeeze(1)  # (total_Sq, TopK)
        valid_shared = idxs_shared >= 0       # (total_Sq, TopK)
        safe_shared = idxs_shared.clamp(min=0).long()

        # Gather KV once in bf16 — no f32 upcast! cuBLAS bf16 BMM does f32 accumulation.
        flat_idxs = safe_shared.reshape(-1)   # (total_Sq * TopK)
        kv_gathered = kv_flat[flat_idxs].reshape(total_Sq, TopK, d_kv)  # bf16, 384MB

        kv_is_shared = (d == d_v == d_kv)

        # Di = sum(dO * O) per (query, head) — needed for dS
        # Use bf16 inputs, f32 reduction (accurate enough for Di)
        Di = (dO_flat.float() * out_flat.float()).sum(dim=-1)  # (total_Sq, np)
        prof.stop()

        # --- BMM in bf16 (Tensor Core, f32 internal accumulation) ---
        prof.start("bwd_scores_P")
        scores = torch.bmm(
            q_flat.reshape(total_Sq, np_, d),
            kv_gathered[:, :, :d].transpose(1, 2)  # (S, D, T)
        ).float() * ctx.softmax_scale  # bf16 bmm → bf16, then f32 cast + scale

        # Fused exp + mask → P (single Triton kernel, replaces 3 ops)
        P = fused_exp_mask(scores, lse, valid_shared, np_)
        del scores
        prof.stop()

        prof.start("bwd_dS")
        # dov = dO @ V^T: bf16 bmm → f32
        dov = torch.bmm(
            dO_flat,
            kv_gathered[:, :, :d_v].transpose(1, 2)
        ).float()  # (S, np, TopK) f32
        dS = P * (dov - Di.unsqueeze(-1)) * ctx.softmax_scale
        del dov
        prof.stop()

        prof.start("bwd_dQ")
        # dQ = dS @ K: cast dS to bf16 for Tensor Core, keep kv in bf16
        dq = torch.bmm(dS.to(kv_gathered.dtype), kv_gathered[:, :, :d]).float()
        prof.stop()

        prof.start("bwd_dKV")
        # dK = dS^T @ Q and dV = P^T @ dO
        q_f32 = q_flat.reshape(total_Sq, np_, d).float()
        dO_f32 = dO_flat.float()

        dkv_gathered = torch.bmm(dS.transpose(1, 2), q_f32)  # (S, T, D) f32
        del dS
        if kv_is_shared:
            # dV = P^T @ dO: in-place add via baddbmm (1 kernel, 0 extra alloc)
            torch.baddbmm(dkv_gathered, P.transpose(1, 2), dO_f32, out=dkv_gathered)
        else:
            dv = torch.bmm(P.transpose(1, 2), dO_f32)
            dkv_tmp = torch.zeros(total_Sq, TopK, d_kv, dtype=torch.float32, device=q_flat.device)
            dkv_tmp[:, :, :d] = dkv_gathered
            dkv_tmp[:, :, :d_v] += dv
            dkv_gathered = dkv_tmp
        del P, q_f32, dO_f32
        prof.stop()

        prof.start("bwd_scatter")
        # Fused mask + scatter_add (single Triton kernel, replaces 2 ops)
        valid_flat = valid_shared.reshape(-1)
        dkv = torch.zeros(skv * b, d_kv, dtype=torch.float32, device=q_flat.device)
        fused_mask_scatter_add(dkv_gathered, flat_idxs, valid_flat, dkv)

        grad_query = dq.to(q_flat.dtype).reshape(sq, b, np_, d)
        grad_kv_full = dkv.to(kv_flat.dtype).reshape(skv, b, -1)
        prof.stop()

        # d_sink
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, np)
            ds_sink = -p_sink * Di  # (total_Sq, np)
            d_sink = ds_sink.sum(0)  # (np,)

        prof.start("bwd_indexer_grads")
        # Scale pre-computed indexer grads by actual grad_loss
        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss
        prof.stop()

        prof.report(f"  [sq={sq}, b={b}, np={np_}, topk={TopK}] ")

        return (
            grad_query,
            grad_kv_full,
            d_sink,
            None,  # window_idxs
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
            None, None, None, None, None, None, None, None,  # scalar args
        )


def fused_indexer_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    window_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    ratio: int,
    softmax_scale: float,
    indexer_softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    kv_offset: int,
    calculate_per_token_loss: bool,
) -> Tuple[Tensor, Tensor]:
    """Fused indexer loss + sparse attention (Path B training).

    Drop-in replacement for ``dsa_kernels.fused_indexer_sparse_attn``.

    Returns:
        ``(output, indexer_loss)`` where output is ``(sq, b, np * d_v)`` bf16
        and indexer_loss is a scalar f32.
    """
    return FusedIndexerSparseAttnFunc.apply(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
        sparse_loss,
        kv_offset,
        calculate_per_token_loss,
    )


__all__ = [
    "build_flat_topk_idxs",
    "local_to_global_flat",
    "dsa_sparse_attn",
    "dsa_sparse_attn_sbhd",
    "indexer_topk",
    "fused_indexer_sparse_attn",
]
