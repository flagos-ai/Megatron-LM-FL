# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton DSA kernels: local-computation replacements for the fused CSA path.

The fused CSA training path in
``megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention``
calls two kernel families:

* :func:`_csa_fwd_flash_mla` — sparse-attention forward (FlashMLA);
* the ``_DSA`` namespace — indexer scoring/top-K, teacher target/predict
  recomputes and sparse/dense indexer backward (cuDNN Frontend DSA).

This module implements both families with Triton kernels for the attention
forward/backward (WGMMA head-parallel kernels) and eager PyTorch math for the
indexer entry points, adapted to the current dev contracts:

* compact ``topk_idxs`` with ``-1`` suffix and ``topk_length`` valid prefix;
* sanitised (``>= 0``) indices in backward with validity derived from
  ``topk_length``;
* THD flat-global indices, ``cu_seqlens_q/k`` and ``q_causal_offsets``;
* query padding rows (zeroed ``dO``/``lse``);
* TP-local attention heads (head-summed teacher mass aggregated by the
  caller — this module performs no collectives);
* full-denominator sparse indexer loss (no partial ``lse_indexer``).

These kernels only replace local computation; they never hide NCCL
collectives or process groups.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.plugin.dsa_kernel.triton_sparse_attn import triton_csa_fwd_flash_mla
from megatron.plugin.dsa_kernel.triton_sparse_attn_bwd import (
    fused_dkv,
    fused_dq,
    sorted_scatter_add,
)

__all__ = [
    "build_triton_dsa_namespace",
    "triton_csa_fwd_flash_mla",
    "triton_csa_sparse_attn_backward",
]

_DENSE_BLOCK_Q = 64


# ---------------------------------------------------------------------------
# Sparse-attention backward (``sparse_attention_backward_wrapper`` contract)
# ---------------------------------------------------------------------------


def _hp_backward(
    q_flat: Tensor,
    kv_flat: Tensor,
    out_flat: Tensor,
    dO_flat: Tensor,
    lse_full: Tensor,
    attn_sink: Optional[Tensor],
    global_idxs: Tensor,
    topk_length: Tensor,
    softmax_scale: float,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Hopper head-parallel backward via the fused Triton dQ/dKV kernels."""
    total_Sq, H, D = q_flat.shape
    total_Skv = kv_flat.shape[0]
    TopK = global_idxs.shape[-1]

    valid_shared = torch.arange(TopK, device=global_idxs.device)[None, :] < topk_length[:, None]
    safe_shared = global_idxs  # already sanitised (>= 0)

    flat_idxs = safe_shared.reshape(-1)
    kv_gathered = kv_flat[flat_idxs].reshape(total_Sq, TopK, kv_flat.shape[-1])

    Di = (dO_flat.float() * out_flat.float()).sum(dim=-1)  # (Sq, H)

    scores = torch.bmm(q_flat, kv_gathered[:, :, :D].transpose(1, 2)).float() * softmax_scale

    dq = fused_dq(scores, lse_full, Di, dO_flat, kv_gathered[:, :, :D], valid_shared, softmax_scale)
    dkv_gathered = fused_dkv(
        scores, lse_full, Di, dO_flat, q_flat, kv_gathered[:, :, :D], valid_shared, softmax_scale
    )
    del scores

    dkv = torch.zeros(total_Skv, kv_flat.shape[-1], dtype=torch.float32, device=q_flat.device)
    sorted_scatter_add(dkv_gathered, flat_idxs, valid_shared.reshape(-1), dkv)

    d_sink = None
    if attn_sink is not None:
        p_sink = torch.exp(attn_sink.unsqueeze(0) - lse_full)
        d_sink = (-p_sink * Di).sum(0)

    return dq.to(q_flat.dtype), dkv.to(kv_flat.dtype), d_sink


def _legacy_backward(
    q_flat: Tensor,
    kv_flat: Tensor,
    out_flat: Tensor,
    dO_flat: Tensor,
    lse_full: Tensor,
    attn_sink: Optional[Tensor],
    global_idxs: Tensor,
    topk_length: Tensor,
    softmax_scale: float,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Non-HP fallback: BMM backward with topk_length-derived validity."""
    from megatron.plugin.dsa_kernel.legacy.pytorch_sparse_attn import pytorch_sparse_attn_bwd

    total_Sq, H, D = q_flat.shape
    TopK = global_idxs.shape[-1]
    valid_shared = torch.arange(TopK, device=global_idxs.device)[None, :] < topk_length[:, None]
    masked_idxs = global_idxs.masked_fill(~valid_shared, -1)
    masked_idxs_3d = torch.as_strided(
        masked_idxs, (total_Sq, 1, TopK), (masked_idxs.stride(0), 0, masked_idxs.stride(1))
    )
    dq, dkv, d_sink = pytorch_sparse_attn_bwd(
        dO_flat,
        q_flat,
        kv_flat,
        masked_idxs_3d,
        out_flat,
        lse_full,
        attn_sink,
        softmax_scale,
        out_flat.shape[-1],
    )
    return dq, dkv, d_sink


def triton_csa_sparse_attn_backward(
    q_flat: Tensor,
    kv_flat: Tensor,
    out_flat: Tensor,
    dO_flat: Tensor,
    lse: Tensor,
    attn_sink: Optional[Tensor],
    global_idxs: Tensor,
    softmax_scale: float,
    topk_length: Tensor,
) -> dict:
    """Sparse-attention backward, matching the cuDNN ``sparse_attention_backward_wrapper``.

    ``global_idxs`` are the sanitised (``>= 0``) compacted indices; validity is
    derived from ``topk_length``. ``lse`` follows the FlashMLA convention
    (excludes the sink), so the full softmax denominator is rebuilt here via
    ``logaddexp`` — the forward kernel's probabilities included the sink bias.
    """
    if attn_sink is not None:
        lse_full = torch.logaddexp(lse.float(), attn_sink.float().view(1, -1))
    else:
        lse_full = lse.float()

    H = q_flat.shape[1]
    D = q_flat.shape[-1]
    d_v = out_flat.shape[-1]
    total_Sq = q_flat.shape[0]
    TopK = global_idxs.shape[-1]
    hp_eligible = (
        global_idxs.ndim == 2 and H >= 16 and (H % 16 == 0) and (D % 16 == 0) and (d_v % 16 == 0)
    )
    if hp_eligible:
        dq, dkv, d_sink = _hp_backward(
            q_flat,
            kv_flat,
            out_flat,
            dO_flat,
            lse_full,
            attn_sink,
            global_idxs,
            topk_length,
            softmax_scale,
        )
    else:
        dq, dkv, d_sink = _legacy_backward(
            q_flat,
            kv_flat,
            out_flat,
            dO_flat,
            lse_full,
            attn_sink,
            global_idxs,
            topk_length,
            softmax_scale,
        )
    return {"dq": dq, "dkv": dkv, "d_sink": d_sink}


# ---------------------------------------------------------------------------
# Indexer entry points (eager PyTorch, adapted to the cuDNN wrapper contracts)
# ---------------------------------------------------------------------------


def _ratio_causal_valid(sq: int, sk: int, ratio: int, device: torch.device) -> Tensor:
    """``(sq, sk)`` bool mask: ``k < floor((q + 1) / ratio)``."""
    q_idx = torch.arange(sq, device=device)
    k_idx = torch.arange(sk, device=device)
    return k_idx.unsqueeze(0) < ((q_idx + 1) // ratio).unsqueeze(1)


def _indexer_scores_bshd(
    q: Tensor,  # (b, sq, nh, hd)
    k: Tensor,  # (b, sk, hd)
    w: Tensor,  # (b, sq, nh), already sm_scale-scaled
    ratio: int,
) -> Tensor:
    """Full indexer scores ``(b, sq, sk)`` f32, ``-inf`` at ratio-masked positions."""
    b, sq, nh, hd = q.shape
    sk = k.shape[1]
    q_f = q.float()
    k_f = k.float()
    w_f = w.float()
    scores = torch.empty(b, sq, sk, dtype=torch.float32, device=q.device)
    valid = _ratio_causal_valid(sq, sk, ratio, q.device)
    for q0 in range(0, sq, _DENSE_BLOCK_Q):
        q1 = min(q0 + _DENSE_BLOCK_Q, sq)
        per_head = torch.einsum("bqhd,bkd->bqhk", q_f[:, q0:q1], k_f)
        block = (torch.relu(per_head) * w_f[:, q0:q1].unsqueeze(-1)).sum(dim=2)
        scores[:, q0:q1] = torch.where(valid[q0:q1], block, torch.full_like(block, float("-inf")))
    return scores


def _indexer_scores_thd(
    q: Tensor,  # (total_q, nh, hd)
    k: Tensor,  # (total_k, hd)
    w: Tensor,  # (total_q, nh), already sm_scale-scaled
    ratio: int,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_k: int,
    q_causal_offsets: Optional[Tensor],
) -> Tensor:
    """Full indexer scores ``(total_q, max_seqlen_k)`` f32, ``-inf`` at masked positions."""
    total_q, nh, hd = q.shape
    k_f = k.float()
    w_f = w.float()
    scores = torch.full(
        (total_q, max_seqlen_k), float("-inf"), dtype=torch.float32, device=q.device
    )
    row_idx = torch.arange(total_q, device=q.device, dtype=torch.int32)
    row_batch_ids = torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp_max(
        cu_seqlens_q.shape[0] - 2
    )
    row_valid = row_idx < cu_seqlens_q[-1]
    pos_in_seq = row_idx - cu_seqlens_q[row_batch_ids]
    if q_causal_offsets is not None:
        pos_in_seq = pos_in_seq + q_causal_offsets[row_batch_ids]
    pos_in_seq = torch.where(row_valid, pos_in_seq, torch.zeros_like(pos_in_seq))
    seqlen_kv_per_row = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    seq_lens = ((pos_in_seq + 1) // ratio).clamp(max=seqlen_kv_per_row[row_batch_ids])
    seq_lens = torch.where(row_valid, seq_lens, torch.zeros_like(seq_lens)).to(torch.int64)

    q_f = q.float()
    full_scores = torch.empty(total_q, k_f.shape[0], dtype=torch.float32, device=q.device)
    for q0 in range(0, total_q, _DENSE_BLOCK_Q):
        q1 = min(q0 + _DENSE_BLOCK_Q, total_q)
        per_head = torch.einsum("qhd,kd->qhk", q_f[q0:q1], k_f)
        full_scores[q0:q1] = (torch.relu(per_head) * w_f[q0:q1].unsqueeze(-1)).sum(dim=1)
    # Per-row local compressed positions 0..seq_lens[row) map to global
    # ``cu_seqlens_k[batch] + k`` in the full K buffer; positions beyond the
    # row's visible prefix stay -inf (matches the cuDNN THD kernel output
    # ``(total_q, max_seqlen_k)``).
    k_idx = torch.arange(max_seqlen_k, device=q.device)
    offsets = cu_seqlens_k[row_batch_ids]
    col = k_idx.unsqueeze(0) + offsets.unsqueeze(1)
    valid_k = k_idx.unsqueeze(0) < seq_lens.unsqueeze(1)
    gathered = full_scores.gather(1, col.clamp(0, k_f.shape[0] - 1))
    return torch.where(valid_k, gathered, torch.full_like(gathered, float("-inf")))


def _indexer_top_k(scores_flat: Tensor, seq_lens: Tensor, top_k: int) -> Tensor:
    """Per-row top-K over valid prefixes; ``-1`` for invalid slots."""
    n, sk = scores_flat.shape
    valid = torch.arange(sk, device=scores_flat.device).unsqueeze(0) < seq_lens.unsqueeze(1)
    masked = scores_flat.masked_fill(~valid, float("-inf"))
    topk_k = min(top_k, sk)
    vals, idx = torch.topk(masked, k=topk_k, dim=-1)
    out = torch.where(torch.isfinite(vals), idx, torch.full_like(idx, -1))
    if topk_k < top_k:
        out = torch.cat(
            [out, torch.full((n, top_k - topk_k), -1, dtype=torch.int32, device=out.device)], dim=-1
        )
    return out.int()


def _indexer_backward_common(
    q: Tensor,  # (B, Sq, nh, hd) fake-BSHD for THD
    w: Tensor,  # (B, Sq, nh) raw (unscaled)
    k_gathered: Tensor,  # (B, Sq, K, hd) K-gathered at the scored positions
    grad_combined: Tensor,  # (B, Sq, K) K = topk or full Sk
    scatter_idx: Tensor,  # (B, Sq, K) int64 target k-ids for the dK scatter
    sm_scale: float,
    k_shape: Tuple[int, int],  # (B, Sk) of the full K buffer
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backprop ``grad_combined`` through ``combined = sm_scale * sum_h relu(q@k) * w``.

    Returns ``(d_q, d_k, d_w)`` in the input dtypes (``d_k`` scattered to the
    full ``(B, Sk, hd)`` buffer via ``scatter_idx``).
    """
    q_f = q.float()
    w_f = w.float()
    per_head = torch.einsum("bshd,bstd->bsht", q_f, k_gathered.float())
    relu_mask = per_head > 0
    relu_scores = per_head * relu_mask

    grad_relu = grad_combined.unsqueeze(2) * w_f.unsqueeze(-1) * sm_scale
    grad_w = (grad_combined.unsqueeze(2) * relu_scores * sm_scale).sum(dim=-1)
    grad_pre_relu = grad_relu * relu_mask

    grad_q = torch.einsum("bsht,bstd->bshd", grad_pre_relu, k_gathered.float())

    grad_k_gathered = torch.einsum("bsht,bshd->bstd", grad_pre_relu, q_f)
    B, Sq = q_f.shape[:2]
    K = grad_combined.shape[-1]
    hd = k_gathered.shape[-1]
    grad_k = torch.zeros(*k_shape, hd, dtype=torch.float32, device=q.device)
    flat_idx = scatter_idx.clamp_min(0).reshape(B, Sq * K)
    flat_grad = grad_k_gathered.reshape(B, Sq * K, hd)
    grad_k.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, hd), flat_grad)

    return grad_q.to(q.dtype), grad_k.to(q.dtype), grad_w.to(w.dtype)


def _sparse_kl_grad_logits(predict: Tensor, target: Tensor) -> Tensor:
    """Gradient of the eps-clamped KL ``sum target * log(target / predict)``
    w.r.t. the logits of ``predict = softmax(logits)`` (predict/target already
    eps-clamped by the caller's loss function)."""
    eps = 1e-10
    scaled_target = target / (predict + eps)
    correction = (scaled_target * predict).sum(dim=-1, keepdim=True)
    return predict * (correction - scaled_target)


def _indexer_backward_wrapper(
    q, w, k, target, predict, topk_indices, sm_scale, loss_coeff, grad_loss
):
    """Sparse KL backward (``indexer_backward_wrapper`` contract).

    The cuDNN wrapper returns gradients for a KL mean over all physical query
    rows. CSA compensates for that mean when per-token loss is requested, so
    the Triton provider must preserve the same normalization contract.
    """
    invalid = topk_indices < 0
    num_query_rows = max(q.shape[0] * q.shape[1], 1)
    grad_combined = (
        _sparse_kl_grad_logits(predict, target)
        * (loss_coeff / num_query_rows)
        * grad_loss
    )
    grad_combined = grad_combined.masked_fill(invalid, 0.0)
    scatter_idx = topk_indices.long()
    batch_ids = torch.arange(q.shape[0], device=q.device)[:, None, None]
    k_gathered = k[batch_ids, scatter_idx.clamp_min(0)]
    d_q, d_k, d_w = _indexer_backward_common(
        q, w, k_gathered, grad_combined, scatter_idx, sm_scale, k.shape[:2]
    )
    return {"d_index_q": d_q, "d_index_k": d_k, "d_weights": d_w}


def _dense_indexer_backward_wrapper(
    q,
    w,
    k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    sm_scale,
    loss_coeff,
    grad_loss,
    ratio=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q_causal_offsets=None,
):
    """Dense KL backward (``dense_indexer_backward_wrapper`` contract).

    Accepts both fake-BSHD (B=1) and native THD (3-D) inputs; the output rank
    mirrors the input rank. THD ``grad_k`` scatters through the same per-row
    ``cu_seqlens_k[batch] + local`` offset mapping the forward uses.
    """
    is_thd = q.ndim == 3
    num_query_rows = max(q.shape[0] if is_thd else q.shape[0] * q.shape[1], 1)
    if is_thd:
        q = q.unsqueeze(0)
        w = w.unsqueeze(0)
        k = k.unsqueeze(0)
        attn_score = attn_score.unsqueeze(0)
        attn_l1norm = attn_l1norm.unsqueeze(0)
        index_score = index_score.unsqueeze(0)
        index_lse = index_lse.unsqueeze(0)
    eps = torch.finfo(torch.float32).tiny
    row_valid = (attn_l1norm > eps) & torch.isfinite(index_lse)
    safe_l1 = attn_l1norm.clamp(min=eps)
    safe_lse = torch.where(row_valid, index_lse, torch.zeros_like(index_lse))
    target = attn_score / safe_l1.unsqueeze(-1)
    target_clamped = target.clamp(min=eps)
    position_valid = torch.isfinite(index_score)
    log_predict = index_score - safe_lse.unsqueeze(-1)
    predict = torch.exp(log_predict)
    grad_combined = (
        _sparse_kl_grad_logits(predict, target_clamped)
        * (loss_coeff / num_query_rows)
        * grad_loss
    )
    grad_combined = torch.where(position_valid, grad_combined, torch.zeros_like(grad_combined))
    grad_combined = torch.where(
        row_valid.unsqueeze(-1), grad_combined, torch.zeros_like(grad_combined)
    )
    B, Sq, Sk = grad_combined.shape
    if is_thd:
        row_idx = torch.arange(Sq, device=q.device, dtype=torch.int32)
        row_batch_ids = torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp_max(
            cu_seqlens_q.shape[0] - 2
        )
        offsets = cu_seqlens_k[row_batch_ids]
        col = torch.arange(Sk, device=q.device).unsqueeze(0) + offsets.unsqueeze(1)
        scatter_idx = col.clamp(0, k.shape[1] - 1).unsqueeze(0).expand(B, Sq, Sk)
        k_gathered = k[0][scatter_idx[0].clamp(0, k.shape[1] - 1)].unsqueeze(0)
    else:
        k_gathered = k.unsqueeze(1).expand(B, Sq, Sk, k.shape[-1])
        scatter_idx = torch.arange(Sk, device=q.device).view(1, 1, Sk).expand(B, Sq, Sk)
    d_q, d_k, d_w = _indexer_backward_common(
        q, w, k_gathered, grad_combined, scatter_idx, sm_scale, k.shape[:2]
    )
    if is_thd:
        d_q = d_q.squeeze(0)
        d_k = d_k.squeeze(0)
        d_w = d_w.squeeze(0)
    return {"d_index_q": d_q, "d_index_k": d_k, "d_weights": d_w}


def _compactify_wrapper(global_idxs: Tensor) -> dict:
    """Compact valid indices into a per-row prefix (``compactify_wrapper`` contract)."""
    valid_mask = global_idxs >= 0
    sorted_indices = valid_mask.int().argsort(dim=-1, descending=True, stable=True)
    compact_idxs = global_idxs.gather(-1, sorted_indices)
    topk_length = valid_mask.sum(dim=-1).int()
    return {
        "indices": compact_idxs.int().contiguous(),
        "topk_length": topk_length.int().contiguous(),
    }


def build_triton_dsa_namespace() -> SimpleNamespace:
    """Build a ``_DSA``-namespace-compatible object for the fused CSA path."""

    def indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        **kwargs,
    ):
        k = k.squeeze(-2) if k.ndim > 2 else k
        if q.ndim == 4:
            return {"scores": _indexer_scores_bshd(q, k, w, ratio)}
        return {
            "scores": _indexer_scores_thd(
                q, k, w, ratio, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, q_causal_offsets
            )
        }

    def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n=1, return_val=False, **kwargs):
        return {"indices": _indexer_top_k(scores_flat, seq_lens, top_k)}

    def sparse_indexer_score_recompute_wrapper(
        q_bshd, k_bsd, w_bsh, topk_bst, qhead_per_kv_head=1, topk_indices_global=False, **kwargs
    ):
        valid = topk_bst >= 0
        safe = topk_bst.long().clamp_min(0)
        batch_ids = torch.arange(q_bshd.shape[0], device=q_bshd.device)[:, None, None]
        k_gathered = k_bsd[batch_ids, safe]
        per_head = torch.einsum("bshd,bstd->bsht", q_bshd.float(), k_gathered.float())
        s = (torch.relu(per_head) * w_bsh.float().unsqueeze(-1)).sum(dim=2)
        s = torch.where(valid, s, torch.full_like(s, float("-inf")))
        predict = torch.softmax(s, dim=-1)
        return {"predict": predict}

    def sparse_attn_score_recompute_wrapper(
        q_bshd,
        k_bsd,
        lse_bsh,
        topk_bst,
        softmax_scale,
        qhead_per_kv_head=1,
        topk_indices_global=False,
        **kwargs,
    ):
        valid = topk_bst >= 0
        safe = topk_bst.long().clamp_min(0)
        batch_ids = torch.arange(q_bshd.shape[0], device=q_bshd.device)[:, None, None]
        k_gathered = k_bsd[batch_ids, safe]
        scores = torch.einsum("bshd,bstd->bsht", q_bshd.float(), k_gathered.float()) * softmax_scale
        head_sum = torch.exp(scores - lse_bsh.float().unsqueeze(-1)).sum(dim=2)
        head_sum = head_sum.masked_fill(~valid, 0.0)
        denom = head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return {"target": head_sum / denom}

    def dense_indexer_score_recompute_wrapper(
        q_indexer,
        k_indexer,
        weights,
        qhead_per_kv_head=1,
        sm_scale=1.0,
        ratio=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        **kwargs,
    ):
        k = k_indexer.squeeze(-2) if k_indexer.ndim > 2 else k_indexer
        if q_indexer.ndim == 4:
            b, sq, nh, hd = q_indexer.shape
            sk = k.shape[1]
            q_f = q_indexer.float()
            k_f = k.float()
            w_f = weights.float()
            scores = torch.empty(b, sq, sk, dtype=torch.float32, device=q_indexer.device)
            valid = _ratio_causal_valid(sq, sk, ratio, q_indexer.device)
            for q0 in range(0, sq, _DENSE_BLOCK_Q):
                q1 = min(q0 + _DENSE_BLOCK_Q, sq)
                per_head = torch.einsum("bqhd,bkd->bqhk", q_f[:, q0:q1], k_f)
                block = (torch.relu(per_head) * w_f[:, q0:q1].unsqueeze(-1)).sum(dim=2) * sm_scale
                scores[:, q0:q1] = torch.where(
                    valid[q0:q1], block, torch.full_like(block, float("-inf"))
                )
            out = scores
        else:
            out = (
                _indexer_scores_thd(
                    q_indexer,
                    k,
                    weights,
                    ratio,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_k,
                    q_causal_offsets,
                )
                * sm_scale
            )
        denom = torch.logsumexp(out, dim=-1)
        return {"out": out, "denom": denom}

    def dense_attn_score_recompute_wrapper(
        q_attn,
        k_attn,
        lse,
        softmax_scale,
        qhead_per_kv_head=1,
        ratio=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        **kwargs,
    ):
        k = k_attn.squeeze(-2) if k_attn.ndim > 2 else k_attn
        if q_attn.ndim == 4:
            b, sq, nh, hd = q_attn.shape
            sk = k.shape[1]
            q_f = q_attn.float()
            k_f = k.float()
            scores = torch.empty(b, sq, sk, dtype=torch.float32, device=q_attn.device)
            valid = _ratio_causal_valid(sq, sk, ratio, q_attn.device)
            for q0 in range(0, sq, _DENSE_BLOCK_Q):
                q1 = min(q0 + _DENSE_BLOCK_Q, sq)
                per_head = torch.einsum("bqhd,bkd->bqhk", q_f[:, q0:q1], k_f) * softmax_scale
                block = torch.exp(per_head - lse[:, q0:q1].float().unsqueeze(-1)).sum(dim=2)
                scores[:, q0:q1] = torch.where(valid[q0:q1], block, torch.zeros_like(block))
            out = scores
        else:
            total_q, nh, hd = q_attn.shape
            max_seqlen_kv = max_seqlen_k
            k_f = k.float()
            q_f = q_attn.float()
            out = torch.zeros(total_q, max_seqlen_kv, dtype=torch.float32, device=q_attn.device)
            row_idx = torch.arange(total_q, device=q_attn.device, dtype=torch.int32)
            row_batch_ids = torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp_max(
                cu_seqlens_q.shape[0] - 2
            )
            row_valid = row_idx < cu_seqlens_q[-1]
            pos_in_seq = row_idx - cu_seqlens_q[row_batch_ids]
            if q_causal_offsets is not None:
                pos_in_seq = pos_in_seq + q_causal_offsets[row_batch_ids]
            pos_in_seq = torch.where(row_valid, pos_in_seq, torch.zeros_like(pos_in_seq))
            seqlen_kv_per_row = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            seq_lens = ((pos_in_seq + 1) // ratio).clamp(max=seqlen_kv_per_row[row_batch_ids])
            seq_lens = torch.where(row_valid, seq_lens, torch.zeros_like(seq_lens)).to(torch.int64)
            full_scores = torch.empty(
                total_q, k_f.shape[0], dtype=torch.float32, device=q_attn.device
            )
            for q0 in range(0, total_q, _DENSE_BLOCK_Q):
                q1 = min(q0 + _DENSE_BLOCK_Q, total_q)
                per_head = torch.einsum("qhd,kd->qhk", q_f[q0:q1], k_f) * softmax_scale
                full_scores[q0:q1] = torch.exp(per_head - lse[q0:q1].float().unsqueeze(-1)).sum(
                    dim=1
                )
            k_idx = torch.arange(max_seqlen_kv, device=q_attn.device)
            offsets = cu_seqlens_k[row_batch_ids]
            col = k_idx.unsqueeze(0) + offsets.unsqueeze(1)
            valid_k = k_idx.unsqueeze(0) < seq_lens.unsqueeze(1)
            gathered = full_scores.gather(1, col.clamp(0, k_f.shape[0] - 1))
            out = torch.where(valid_k, gathered, torch.zeros_like(gathered))
        denom = out.sum(dim=-1)
        return {"out": out, "denom": denom}

    def indexer_backward_wrapper(
        q, w, k, attn, idx, topk, sm_scale=1.0, loss_coeff=1.0, grad_loss=1.0, block_I=128, **kwargs
    ):
        return _indexer_backward_wrapper(q, w, k, attn, idx, topk, sm_scale, loss_coeff, grad_loss)

    def dense_indexer_backward_wrapper(
        q,
        w,
        k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=1.0,
        block_I=128,
        ratio=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        **kwargs,
    ):
        return _dense_indexer_backward_wrapper(
            q,
            w,
            k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale,
            loss_coeff,
            grad_loss,
            ratio,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q_causal_offsets,
        )

    return SimpleNamespace(
        compactify_wrapper=_compactify_wrapper,
        indexer_forward_wrapper=indexer_forward_wrapper,
        indexer_top_k_wrapper=indexer_top_k_wrapper,
        sparse_indexer_score_recompute_wrapper=sparse_indexer_score_recompute_wrapper,
        sparse_attn_score_recompute_wrapper=sparse_attn_score_recompute_wrapper,
        dense_indexer_score_recompute_wrapper=dense_indexer_score_recompute_wrapper,
        dense_attn_score_recompute_wrapper=dense_attn_score_recompute_wrapper,
        indexer_backward_wrapper=indexer_backward_wrapper,
        dense_indexer_backward_wrapper=dense_indexer_backward_wrapper,
        sparse_attention_backward_wrapper=triton_csa_sparse_attn_backward,
    )
