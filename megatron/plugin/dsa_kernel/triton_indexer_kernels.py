# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Triton indexer scoring and backward kernels for DSA.

Replaces the dependency on ``cudnn.DSA`` indexer-related wrappers:
- sparse_indexer_score_recompute_wrapper
- sparse_attn_score_recompute_wrapper
- dense_indexer_score_recompute_wrapper
- dense_attn_score_recompute_wrapper
- sparse_indexer_backward
- dense_indexer_backward

Also provides the indexer top-K selection (replacing TRT-LLM radix top-K).

Optimized to avoid materializing O(S_q × S_k) intermediates where possible:
- Sparse functions use direct batch advanced indexing instead of expand+gather.
- Dense functions tile over the Q dimension in blocks of _DENSE_BLOCK_Q.
- Sparse backward uses vectorized scatter_add_ instead of a Python for-loop.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl

from megatron.plugin.dsa_kernel.triton_dsa_utils import (
    compute_ratio_causal_mask,
    topk_with_causal_mask,
)


# Block size for dense Q-tiling. Trades peak memory for kernel-launch overhead.
# 64 is a good default; lower to 32 if D >= 512 and GPU memory is very tight.
_DENSE_BLOCK_Q = 512


# ---------------------------------------------------------------------------
# Sparse indexer score: compute ``predict`` distribution
# ---------------------------------------------------------------------------


def sparse_indexer_score_recompute(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk_indices: Tensor,
    qhead_per_kv_head: int = 1,
) -> Dict[str, Tensor]:
    """Compute ``predict`` distribution (softmax over top-K of indexer scores).

    Pure PyTorch implementation matching cudnn.DSA.sparse_indexer_score_recompute_wrapper.

    Args:
        q_indexer: ``(B, S_q, H_q, D)`` bf16.
        k_indexer: ``(B, S_k, D)`` bf16.
        weights:   ``(B, S_q, H_q)`` bf16.
        topk_indices: ``(B, S_q, topk)`` int32.
        qhead_per_kv_head: number of Q heads per KV head (MQA ratio).

    Returns:
        dict with "predict": ``(B, S_q, topk)`` fp32 softmax distribution.
    """
    B, S_q, H_q, D = q_indexer.shape
    topk = topk_indices.shape[-1]

    q = q_indexer.float()  # (B, S_q, H_q, D)
    w = weights.float()  # (B, S_q, H_q)

    # Gather K at topk positions using batch advanced indexing — O(B*S_q*topk*D)
    # No S_q×S_k expansion needed.
    idx_expanded = topk_indices.long().clamp(min=0)  # (B, S_q, topk)
    batch_idx = torch.arange(B, device=k_indexer.device)[:, None, None]  # (B, 1, 1)
    k_gathered = k_indexer.float()[batch_idx, idx_expanded]  # (B, S_q, topk, D)

    # Compute scores: (B, S_q, H_q, topk) = Q @ K^T
    scores = torch.einsum("bqhd,bqtd->bqht", q, k_gathered)  # (B, S_q, H_q, topk)

    # ReLU activation
    scores = torch.relu(scores)

    # Weight by per-head weights: (B, S_q, H_q, topk) * (B, S_q, H_q, 1)
    weighted_scores = scores * w.unsqueeze(-1)

    # Sum over heads: (B, S_q, topk)
    combined_scores = weighted_scores.sum(dim=2)

    # Mask invalid positions (topk_indices == -1)
    invalid_mask = topk_indices == -1  # (B, S_q, topk)
    combined_scores = combined_scores.masked_fill(invalid_mask, float("-inf"))

    # Softmax over topk dimension -> predict
    predict = torch.softmax(combined_scores, dim=-1)
    # Zero out invalid positions (softmax of all -inf gives nan -> 0)
    predict = predict.masked_fill(invalid_mask, 0.0)

    return {"predict": predict}


# ---------------------------------------------------------------------------
# Sparse attention score recompute: compute ``target`` distribution
# ---------------------------------------------------------------------------


def sparse_attn_score_recompute(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    qhead_per_kv_head: int = 1,
) -> Dict[str, Tensor]:
    """Compute ``target`` distribution (L1-normalised head-sum softmax).

    Pure PyTorch implementation matching cudnn.DSA.sparse_attn_score_recompute_wrapper.

    Args:
        q_attn: ``(B, S_q, H_q, D)`` bf16.
        k_attn: ``(B, S_k, D)`` bf16.
        lse:    ``(B, S_q, H_q)`` fp32 — log-sum-exp from attention forward.
        topk_indices: ``(B, S_q, topk)`` int32.
        softmax_scale: attention scale factor.
        qhead_per_kv_head: MQA ratio.

    Returns:
        dict with "target": ``(B, S_q, topk)`` fp32 L1-normalized distribution.
    """
    B, S_q, H_q, D = q_attn.shape
    topk = topk_indices.shape[-1]

    q = q_attn.float()  # (B, S_q, H_q, D)

    # Gather K at topk positions — O(B*S_q*topk*D), no S_k expansion
    idx_expanded = topk_indices.long().clamp(min=0)  # (B, S_q, topk)
    batch_idx = torch.arange(B, device=k_attn.device)[:, None, None]  # (B, 1, 1)
    k_gathered = k_attn.float()[batch_idx, idx_expanded]  # (B, S_q, topk, D)

    # Compute attention scores: (B, S_q, H_q, topk)
    scores = torch.einsum("bqhd,bqtd->bqht", q, k_gathered) * softmax_scale

    # Subtract LSE and exp: exp(score - lse) per head
    # lse: (B, S_q, H_q) -> (B, S_q, H_q, 1)
    attn_probs = torch.exp(scores - lse.unsqueeze(-1))  # (B, S_q, H_q, topk)

    # Sum over heads: (B, S_q, topk)
    head_sum = attn_probs.sum(dim=2)

    # Mask invalid
    invalid_mask = topk_indices == -1
    head_sum = head_sum.masked_fill(invalid_mask, 0.0)

    # L1 normalize over topk dimension
    denom = head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    target = head_sum / denom

    return {"target": target}


# ---------------------------------------------------------------------------
# Dense indexer score: full S_k scoring — tiled over Q blocks
# ---------------------------------------------------------------------------


def dense_indexer_score_recompute(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    qhead_per_kv_head: int = 1,
    sm_scale: float = 1.0,
    ratio: int = 1,
) -> Dict[str, Tensor]:
    """Dense indexer score forward over the full S_k axis.

    Pure PyTorch implementation matching cudnn.DSA.dense_indexer_score_recompute_wrapper.
    Tiles over the Q dimension to avoid materializing (B, S_q, H_q, S_k) at once.

    Args:
        q_indexer: ``(B, S_q, H_q, D)`` bf16.
        k_indexer: ``(B, S_k, D)`` bf16.
        weights:   ``(B, S_q, H_q)`` bf16.
        qhead_per_kv_head: MQA ratio.
        sm_scale: indexer softmax scale.
        ratio: compression ratio for causal mask.

    Returns:
        dict with "out": ``(B, S_q, S_k)`` fp32 and "denom": ``(B, S_q)`` fp32.
    """
    B, S_q, H_q, D = q_indexer.shape
    S_k = k_indexer.shape[1]

    k = k_indexer.float()  # (B, S_k, D)

    # Pre-compute ratio causal mask (only S_q × S_k, typically small fp32)
    mask = compute_ratio_causal_mask(S_q, S_k, ratio, q_indexer.device)  # (S_q, S_k)

    # Output buffer
    out = torch.empty(B, S_q, S_k, dtype=torch.float32, device=q_indexer.device)

    BLOCK_Q = _DENSE_BLOCK_Q
    for q_start in range(0, S_q, BLOCK_Q):
        q_end = min(q_start + BLOCK_Q, S_q)
        q_block = q_indexer[:, q_start:q_end].float()  # (B, block, H_q, D)
        w_block = weights[:, q_start:q_end].float()  # (B, block, H_q)

        # Compute scores: (B, block, H_q, S_k)
        scores_block = torch.einsum("bqhd,bkd->bqhk", q_block, k) * sm_scale

        # ReLU
        scores_block = torch.relu(scores_block)

        # Weight and sum over heads: (B, block, S_k)
        out_block = (scores_block * w_block.unsqueeze(-1)).sum(dim=2)

        # Apply ratio causal mask
        out_block = out_block + mask[q_start:q_end].unsqueeze(0)

        out[:, q_start:q_end] = out_block

    # Compute LSE (denom)
    denom = torch.logsumexp(out, dim=-1)  # (B, S_q)

    return {"out": out, "denom": denom}


# ---------------------------------------------------------------------------
# Dense attention score: full S_k scoring — tiled over Q blocks
# ---------------------------------------------------------------------------


def dense_attn_score_recompute(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    softmax_scale: float,
    qhead_per_kv_head: int = 1,
    ratio: int = 1,
) -> Dict[str, Tensor]:
    """Dense attention score forward over the full S_k axis.

    Pure PyTorch implementation matching cudnn.DSA.dense_attn_score_recompute_wrapper.
    Tiles over the Q dimension to avoid materializing (B, S_q, H_q, S_k) at once.

    Args:
        q_attn: ``(B, S_q, H_q, D)`` bf16.
        k_attn: ``(B, S_k, D)`` bf16.
        lse:    ``(B, S_q, H_q)`` fp32.
        softmax_scale: attention scale.
        qhead_per_kv_head: MQA ratio.
        ratio: compression ratio for causal mask.

    Returns:
        dict with "out": ``(B, S_q, S_k)`` fp32 and "denom": ``(B, S_q)`` fp32.
    """
    B, S_q, H_q, D = q_attn.shape
    S_k = k_attn.shape[1]

    k = k_attn.float()  # (B, S_k, D)

    # Pre-compute ratio causal mask
    mask = compute_ratio_causal_mask(S_q, S_k, ratio, q_attn.device)  # (S_q, S_k)

    # Output buffer
    out = torch.empty(B, S_q, S_k, dtype=torch.float32, device=q_attn.device)

    BLOCK_Q = _DENSE_BLOCK_Q
    for q_start in range(0, S_q, BLOCK_Q):
        q_end = min(q_start + BLOCK_Q, S_q)
        q_block = q_attn[:, q_start:q_end].float()  # (B, block, H_q, D)
        lse_block = lse[:, q_start:q_end]  # (B, block, H_q)
        mask_block = mask[q_start:q_end]  # (block, S_k)

        # Compute scores: (B, block, H_q, S_k)
        scores_block = torch.einsum("bqhd,bkd->bqhk", q_block, k) * softmax_scale

        # Apply ratio causal mask to scores (before exp)
        scores_block = scores_block + mask_block.unsqueeze(0).unsqueeze(2)

        # exp(score - LSE): (B, block, H_q, S_k)
        attn_probs_block = torch.exp(scores_block - lse_block.unsqueeze(-1))

        # Sum over heads: (B, block, S_k)
        out[:, q_start:q_end] = attn_probs_block.sum(dim=2)

    # L1 norm denom
    denom = out.sum(dim=-1)  # (B, S_q)

    return {"out": out, "denom": denom}


# ---------------------------------------------------------------------------
# Indexer top-K selection — tiled over Q blocks
# ---------------------------------------------------------------------------


def indexer_topk_selection(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk: int,
    ratio: int,
    indexer_softmax_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """Compute indexer scores and select top-K with ratio causal masking.

    Replaces TRT-LLM's radix top-K kernel.
    Tiles over the Q dimension to avoid materializing (B, S_q, H_q, S_k) at once.

    Args:
        q_indexer: ``(B, S_q, H_q, D)`` bf16.
        k_indexer: ``(B, S_k, D)`` bf16.
        weights:   ``(B, S_q, H_q)`` bf16 — scaled weights.
        topk: number of top-K indices.
        ratio: compression ratio for causal mask.
        indexer_softmax_scale: scale for indexer scores.

    Returns:
        dict with:
            "topk_indices": ``(B, S_q, topk)`` int32.
            "topk_length": ``(B, S_q)`` int32.
            "full_scores": ``(B, S_q, S_k)`` fp32 — full indexer scores (for backward).
    """
    B, S_q, H_q, D = q_indexer.shape
    S_k = k_indexer.shape[1]

    k = k_indexer.float()  # (B, S_k, D)

    # Output buffer for full scores
    scores = torch.empty(B, S_q, S_k, dtype=torch.float32, device=q_indexer.device)

    BLOCK_Q = _DENSE_BLOCK_Q
    for q_start in range(0, S_q, BLOCK_Q):
        q_end = min(q_start + BLOCK_Q, S_q)
        q_block = q_indexer[:, q_start:q_end].float()  # (B, block, H_q, D)
        w_block = weights[:, q_start:q_end].float()  # (B, block, H_q)

        # Compute per-head scores: (B, block, H_q, S_k)
        per_head_block = torch.einsum("bqhd,bkd->bqhk", q_block, k) * indexer_softmax_scale
        per_head_block = torch.relu(per_head_block)

        # Weight and sum over heads: (B, block, S_k)
        scores[:, q_start:q_end] = (per_head_block * w_block.unsqueeze(-1)).sum(dim=2)

    # Top-K with causal mask (operates on the full (B, S_q, S_k) scores)
    topk_indices, topk_length = topk_with_causal_mask(scores, topk, ratio)

    return {"topk_indices": topk_indices, "topk_length": topk_length, "full_scores": scores}


# Keep the old name as an alias for backward compatibility with imports
indexer_topk_select = indexer_topk_selection


# ---------------------------------------------------------------------------
# Indexer backward — sparse path
# ---------------------------------------------------------------------------


def sparse_indexer_backward(
    grad_loss: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk_indices: Tensor,
    predict: Tensor,
    target: Tensor,
    qhead_per_kv_head: int = 1,
    indexer_softmax_scale: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward through the sparse indexer path.

    Computes gradients w.r.t. q_indexer, k_indexer, and weights given the KL loss.

    The forward computes:
        scores[h,t] = ReLU(Q[h] · K[t]^T) * scale
        combined[t] = sum_h(scores[h,t] * W[h])
        predict[t] = softmax(combined)[t]
        loss = KL(target || predict)

    Uses batch advanced indexing for gather (no S_k expansion) and vectorized
    scatter_add_ (no Python for-loop over S_q).

    Args:
        grad_loss: scalar gradient from loss.
        q_indexer: ``(B, S_q, H_q, D)`` bf16.
        k_indexer: ``(B, S_k, D)`` bf16.
        weights: ``(B, S_q, H_q)`` bf16.
        topk_indices: ``(B, S_q, topk)`` int32.
        predict: ``(B, S_q, topk)`` fp32.
        target: ``(B, S_q, topk)`` fp32.
        qhead_per_kv_head: MQA ratio.
        indexer_softmax_scale: scale factor.

    Returns:
        (grad_q_indexer, grad_k_indexer, grad_weights)
    """
    B, S_q, H_q, D = q_indexer.shape
    topk = topk_indices.shape[-1]

    q = q_indexer.float()
    w = weights.float()

    # Gather K at topk positions — batch advanced indexing, no S_k expansion
    idx_expanded = topk_indices.long().clamp(min=0)  # (B, S_q, topk)
    batch_idx = torch.arange(B, device=k_indexer.device)[:, None, None]
    k_gathered = k_indexer.float()[batch_idx, idx_expanded]  # (B, S_q, topk, D)

    # Recompute forward scores
    per_head_scores = torch.einsum("bqhd,bqtd->bqht", q, k_gathered) * indexer_softmax_scale
    relu_mask = per_head_scores > 0
    per_head_scores = torch.relu(per_head_scores)  # (B, S_q, H_q, topk)
    combined = (per_head_scores * w.unsqueeze(-1)).sum(dim=2)  # (B, S_q, topk)

    # Grad through KL + softmax: d_loss/d_combined
    invalid_mask = topk_indices == -1
    grad_combined = (predict - target) * grad_loss  # (B, S_q, topk)
    grad_combined = grad_combined.masked_fill(invalid_mask, 0.0)

    # Grad through head-sum + weight multiply
    # combined[t] = sum_h(relu_score[h,t] * w[h])
    # grad_relu_score[h,t] = grad_combined[t] * w[h]
    grad_relu_scores = grad_combined.unsqueeze(2) * w.unsqueeze(-1)  # (B, S_q, H_q, topk)

    # grad_weights[h] = sum_t(grad_combined[t] * relu_score[h,t])
    grad_w = (grad_combined.unsqueeze(2) * per_head_scores).sum(dim=-1)  # (B, S_q, H_q)

    # Grad through ReLU
    grad_pre_relu = grad_relu_scores * relu_mask.float()  # (B, S_q, H_q, topk)
    grad_pre_relu = grad_pre_relu * indexer_softmax_scale

    # Grad through Q @ K^T
    grad_q = torch.einsum("bqht,bqtd->bqhd", grad_pre_relu, k_gathered)

    # grad_k_gathered = grad_pre_relu^T @ q: (B, S_q, topk, D)
    grad_k_gathered = torch.einsum("bqht,bqhd->bqtd", grad_pre_relu, q)

    # Scatter grad_k_gathered back to full k_indexer — vectorized, no Python loop
    grad_k = torch.zeros_like(k_indexer, dtype=torch.float32)  # (B, S_k, D)
    # Flatten S_q and topk dims for a single scatter_add_ call
    flat_idx = idx_expanded.reshape(B, S_q * topk, 1).expand(-1, -1, D)  # (B, S_q*topk, D)
    flat_grad = grad_k_gathered.reshape(B, S_q * topk, D)  # (B, S_q*topk, D)
    grad_k.scatter_add_(1, flat_idx, flat_grad)

    return (
        grad_q.to(q_indexer.dtype),
        grad_k.to(k_indexer.dtype),
        grad_w.to(weights.dtype),
    )


# ---------------------------------------------------------------------------
# Indexer backward — dense path (tiled over Q blocks)
# ---------------------------------------------------------------------------


def dense_indexer_backward(
    grad_loss: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_out: Tensor,
    indexer_denom: Tensor,
    attn_out: Tensor,
    attn_denom: Tensor,
    qhead_per_kv_head: int = 1,
    indexer_softmax_scale: float = 1.0,
    ratio: int = 1,
    calculate_per_token_loss: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward through the dense indexer loss path.

    Tiles over S_q to avoid materializing (B, S_q, H_q, S_k) per-head scores.

    The dense path computes KL(target || predict) over all S_k positions where:
        predict = softmax(indexer_out)  [over S_k]
        target = attn_out / attn_denom  [L1 normalized]

    Args:
        grad_loss: scalar gradient.
        q_indexer: ``(B, S_q, H_q, D)`` bf16.
        k_indexer: ``(B, S_k, D)`` bf16.
        weights: ``(B, S_q, H_q)`` bf16.
        indexer_out: ``(B, S_q, S_k)`` fp32 — raw indexer scores.
        indexer_denom: ``(B, S_q)`` fp32 — LSE of indexer_out.
        attn_out: ``(B, S_q, S_k)`` fp32 — head-summed attention weights.
        attn_denom: ``(B, S_q)`` fp32 — L1 norm of attn_out.
        qhead_per_kv_head: MQA ratio.
        indexer_softmax_scale: scale factor.
        ratio: compression ratio.
        calculate_per_token_loss: whether loss is per-token sum or mean.

    Returns:
        (grad_q_indexer, grad_k_indexer, grad_weights)
    """
    B, S_q, H_q, D = q_indexer.shape
    S_k = k_indexer.shape[1]

    # Recompute predict and target (these are (B, S_q, S_k) — already stored)
    predict = torch.softmax(indexer_out, dim=-1)  # (B, S_q, S_k)
    safe_denom = attn_denom.clamp(min=1e-12).unsqueeze(-1)
    target = attn_out / safe_denom  # (B, S_q, S_k)

    # Grad through KL: d_KL/d_indexer_logit = (predict - target) * grad_loss
    mask = compute_ratio_causal_mask(S_q, S_k, ratio, q_indexer.device)
    valid = mask.unsqueeze(0) == 0  # (1, S_q, S_k)

    grad_logits = (predict - target) * grad_loss  # (B, S_q, S_k)
    grad_logits = grad_logits.masked_fill(~valid, 0.0)

    if not calculate_per_token_loss:
        grad_logits = grad_logits / (B * S_q)

    # Backward through indexer score computation — tiled over Q blocks
    # indexer_out[b, q, k] = sum_h(ReLU(Q[b,q,h] · K[b,k]^T * scale) * W[b,q,h])
    k = k_indexer.float()  # (B, S_k, D)

    grad_q = torch.empty(B, S_q, H_q, D, dtype=torch.float32, device=q_indexer.device)
    grad_k = torch.zeros(B, S_k, D, dtype=torch.float32, device=k_indexer.device)
    grad_w = torch.empty(B, S_q, H_q, dtype=torch.float32, device=weights.device)

    BLOCK_Q = _DENSE_BLOCK_Q
    for q_start in range(0, S_q, BLOCK_Q):
        q_end = min(q_start + BLOCK_Q, S_q)

        q_block = q_indexer[:, q_start:q_end].float()  # (B, block, H_q, D)
        w_block = weights[:, q_start:q_end].float()  # (B, block, H_q)
        grad_logits_block = grad_logits[:, q_start:q_end]  # (B, block, S_k)

        # Recompute per-head scores for this Q block
        per_head_scores = torch.einsum("bqhd,bkd->bqhk", q_block, k) * indexer_softmax_scale
        relu_mask = per_head_scores > 0
        per_head_scores_relu = torch.relu(per_head_scores)  # (B, block, H_q, S_k)

        # grad_relu[b,q,h,k] = grad_logits[b,q,k] * W[b,q,h]
        grad_relu = grad_logits_block.unsqueeze(2) * w_block.unsqueeze(-1)  # (B, block, H_q, S_k)

        # grad_weights[b,q,h] = sum_k(grad_logits[b,q,k] * relu_score[b,q,h,k])
        grad_w[:, q_start:q_end] = (grad_logits_block.unsqueeze(2) * per_head_scores_relu).sum(dim=-1)

        # Grad through ReLU
        grad_pre_relu = grad_relu * relu_mask.float() * indexer_softmax_scale  # (B, block, H_q, S_k)

        # Grad through matmul
        grad_q[:, q_start:q_end] = torch.einsum("bqhk,bkd->bqhd", grad_pre_relu, k)
        # Accumulate grad_k from this block
        grad_k += torch.einsum("bqhk,bqhd->bkd", grad_pre_relu, q_block)

    return (
        grad_q.to(q_indexer.dtype),
        grad_k.to(k_indexer.dtype),
        grad_w.to(weights.dtype),
    )


# ---------------------------------------------------------------------------
# Fused sparse indexer loss + backward (P0 optimization)
# ---------------------------------------------------------------------------


def fused_sparse_indexer_loss_and_backward(
    q_idx_bshd: Tensor,
    k_idx_bsd: Tensor,
    w_bsh: Tensor,
    topk_indices_cmp: Tensor,
    q_attn_bshd: Tensor,
    k_attn_bsd: Tensor,
    lse_bsh: Tensor,
    indexer_softmax_scale: float,
    softmax_scale: float,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
    idx_nh: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """One-pass fused: predict + target + KL loss + indexer backward.

    Merges ``sparse_indexer_score_recompute`` + ``sparse_attn_score_recompute`` +
    ``_kl_loss_from_target_predict`` + ``sparse_indexer_backward`` into a single
    pass, sharing the K gather and intermediate computations.

    Args:
        q_idx_bshd: ``(B, S_q, H_q, D_idx)`` bf16 — indexer queries.
        k_idx_bsd: ``(B, S_k, D_idx)`` bf16 — indexer keys.
        w_bsh: ``(B, S_q, H_q)`` bf16 — raw (unscaled) weights.
        topk_indices_cmp: ``(B, S_q, topk)`` int32.
        q_attn_bshd: ``(B, S_q, np, D_attn)`` bf16 — attention queries.
        k_attn_bsd: ``(B, S_k, D_attn)`` bf16 — attention keys.
        lse_bsh: ``(B, S_q, np)`` fp32 — LSE from attention forward.
        indexer_softmax_scale: scale for indexer scores.
        softmax_scale: scale for attention scores.
        loss_coeff: KL loss coefficient.
        calculate_per_token_loss: if True, use sum instead of mean.
        idx_nh: number of indexer heads (qhead_per_kv_head).

    Returns:
        (indexer_loss, grad_q_indexer, grad_k_indexer, grad_weights)
    """
    B, S_q, H_q, D_idx = q_idx_bshd.shape
    topk = topk_indices_cmp.shape[-1]
    eps = torch.finfo(torch.float32).tiny

    # --- Single gather (shared between indexer and attn) ---
    idx_expanded = topk_indices_cmp.long().clamp(min=0)  # (B, S_q, topk)
    batch_idx = torch.arange(B, device=k_idx_bsd.device)[:, None, None]  # (B, 1, 1)
    k_idx_gathered = k_idx_bsd.float()[batch_idx, idx_expanded]  # (B, S_q, topk, D_idx)
    k_attn_gathered = k_attn_bsd.float()[batch_idx, idx_expanded]  # (B, S_q, topk, D_attn)

    invalid_mask = topk_indices_cmp == -1  # (B, S_q, topk)

    # --- Compute predict (indexer distribution) ---
    q_idx = q_idx_bshd.float()
    w = w_bsh.float()
    per_head_scores = torch.einsum("bqhd,bqtd->bqht", q_idx, k_idx_gathered) * indexer_softmax_scale
    relu_mask = per_head_scores > 0
    per_head_scores_relu = torch.relu(per_head_scores)  # (B, S_q, H_q, topk)
    combined = (per_head_scores_relu * w.unsqueeze(-1)).sum(dim=2)  # (B, S_q, topk)
    combined = combined.masked_fill(invalid_mask, float("-inf"))
    predict = torch.softmax(combined, dim=-1).masked_fill(invalid_mask, 0.0)

    # --- Compute target (attention distribution) ---
    q_attn = q_attn_bshd.float()
    attn_scores = torch.einsum("bqhd,bqtd->bqht", q_attn, k_attn_gathered) * softmax_scale
    attn_probs = torch.exp(attn_scores - lse_bsh.unsqueeze(-1))  # (B, S_q, np, topk)
    head_sum = attn_probs.sum(dim=2)  # (B, S_q, topk)
    head_sum = head_sum.masked_fill(invalid_mask, 0.0)
    denom = head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    target = head_sum / denom  # (B, S_q, topk)

    # --- KL loss ---
    t = target.clamp(min=eps)
    p = predict.clamp(min=eps)
    kl_per_row = (t * (torch.log(t) - torch.log(p))).sum(dim=-1)  # (B, S_q)
    row_valid = (~invalid_mask).any(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    indexer_loss = loss_coeff * loss

    # --- Indexer backward (grad_loss=1, will be scaled later) ---
    grad_combined = (predict - target)  # (B, S_q, topk)
    grad_combined = grad_combined.masked_fill(invalid_mask, 0.0)

    # grad_relu_scores[h,t] = grad_combined[t] * w[h]
    grad_relu_scores = grad_combined.unsqueeze(2) * w.unsqueeze(-1)  # (B, S_q, H_q, topk)

    # grad_weights[h] = sum_t(grad_combined[t] * relu_score[h,t])
    grad_w = (grad_combined.unsqueeze(2) * per_head_scores_relu).sum(dim=-1)  # (B, S_q, H_q)

    # Grad through ReLU
    grad_pre_relu = grad_relu_scores * relu_mask.float() * indexer_softmax_scale

    # Grad through Q @ K^T
    grad_q = torch.einsum("bqht,bqtd->bqhd", grad_pre_relu, k_idx_gathered)
    grad_k_gathered = torch.einsum("bqht,bqhd->bqtd", grad_pre_relu, q_idx)

    # Scatter grad_k_gathered back to full k_indexer
    S_k = k_idx_bsd.shape[1]
    grad_k = torch.zeros(B, S_k, D_idx, dtype=torch.float32, device=k_idx_bsd.device)
    flat_idx = idx_expanded.reshape(B, S_q * topk, 1).expand(-1, -1, D_idx)
    flat_grad = grad_k_gathered.reshape(B, S_q * topk, D_idx)
    grad_k.scatter_add_(1, flat_idx, flat_grad)

    return (
        indexer_loss,
        grad_q.to(q_idx_bshd.dtype),
        grad_k.to(k_idx_bsd.dtype),
        grad_w.to(w_bsh.dtype),
    )


# ---------------------------------------------------------------------------
# Fused dense indexer loss + backward (P0 optimization)
# ---------------------------------------------------------------------------


def fused_dense_indexer_loss_and_backward(
    q_idx_bshd: Tensor,
    k_idx_bsd: Tensor,
    w_bsh: Tensor,
    topk_indices_cmp: Tensor,
    q_attn_bshd: Tensor,
    k_attn_bsd: Tensor,
    lse_bsh: Tensor,
    indexer_softmax_scale: float,
    softmax_scale: float,
    loss_coeff: float,
    ratio: int = 1,
    calculate_per_token_loss: bool = False,
    idx_nh: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """One-pass fused dense: indexer score + attn score + KL loss + backward.

    Merges ``dense_indexer_score_recompute`` + ``dense_attn_score_recompute`` +
    ``_kl_loss_from_dense_scores`` + ``dense_indexer_backward`` into a single
    tiled pass over Q blocks.

    Args:
        q_idx_bshd: ``(B, S_q, H_q, D_idx)`` bf16 — indexer queries.
        k_idx_bsd: ``(B, S_k, D_idx)`` bf16 — indexer keys (compressed range).
        w_bsh: ``(B, S_q, H_q)`` bf16 — raw weights.
        topk_indices_cmp: ``(B, S_q, topk)`` int32 — for row validity.
        q_attn_bshd: ``(B, S_q, np, D_attn)`` bf16 — attention queries.
        k_attn_bsd: ``(B, S_k, D_attn)`` bf16 — attention keys (compressed range).
        lse_bsh: ``(B, S_q, np)`` fp32 — LSE from attention forward.
        indexer_softmax_scale: scale for indexer scores.
        softmax_scale: scale for attention scores.
        loss_coeff: KL loss coefficient.
        ratio: compression ratio for causal mask.
        calculate_per_token_loss: if True, use sum instead of mean.
        idx_nh: number of indexer heads.

    Returns:
        (indexer_loss, grad_q_indexer, grad_k_indexer, grad_weights)
    """
    B, S_q, H_q, D_idx = q_idx_bshd.shape
    S_k = k_idx_bsd.shape[1]
    np_ = q_attn_bshd.shape[2]
    D_attn = q_attn_bshd.shape[3]
    eps = torch.finfo(torch.float32).tiny

    k_idx = k_idx_bsd.float()  # (B, S_k, D_idx)
    k_attn = k_attn_bsd.float()  # (B, S_k, D_attn)

    # Causal mask: which KV positions each query can attend to
    causal_mask = compute_ratio_causal_mask(S_q, S_k, ratio, q_idx_bshd.device)  # (S_q, S_k)

    # Row validity from topk_indices
    row_valid = (topk_indices_cmp >= 0).any(dim=-1)  # (B, S_q)

    # Accumulators
    grad_q = torch.empty(B, S_q, H_q, D_idx, dtype=torch.float32, device=q_idx_bshd.device)
    grad_k = torch.zeros(B, S_k, D_idx, dtype=torch.float32, device=k_idx_bsd.device)
    grad_w = torch.empty(B, S_q, H_q, dtype=torch.float32, device=w_bsh.device)
    kl_acc = torch.zeros(B, S_q, dtype=torch.float32, device=q_idx_bshd.device)

    BLOCK_Q = _DENSE_BLOCK_Q
    for q_start in range(0, S_q, BLOCK_Q):
        q_end = min(q_start + BLOCK_Q, S_q)
        block_len = q_end - q_start

        # Slice inputs for this Q block
        q_idx_block = q_idx_bshd[:, q_start:q_end].float()  # (B, block, H_q, D_idx)
        w_block = w_bsh[:, q_start:q_end].float()  # (B, block, H_q)
        q_attn_block = q_attn_bshd[:, q_start:q_end].float()  # (B, block, np, D_attn)
        lse_block = lse_bsh[:, q_start:q_end]  # (B, block, np)
        mask_block = causal_mask[q_start:q_end]  # (block, S_k)
        row_valid_block = row_valid[:, q_start:q_end]  # (B, block)

        # --- Indexer scores (predict) ---
        per_head_scores = torch.einsum("bqhd,bkd->bqhk", q_idx_block, k_idx) * indexer_softmax_scale
        relu_mask = per_head_scores > 0
        per_head_scores_relu = torch.relu(per_head_scores)  # (B, block, H_q, S_k)
        combined = (per_head_scores_relu * w_block.unsqueeze(-1)).sum(dim=2)  # (B, block, S_k)
        # Apply causal mask
        combined = combined.masked_fill(~mask_block.unsqueeze(0), float("-inf"))
        index_lse = torch.logsumexp(combined, dim=-1)  # (B, block)
        index_score = combined  # keep for KL

        # --- Attention scores (target) ---
        attn_per_head = torch.einsum("bqhd,bkd->bqhk", q_attn_block, k_attn) * softmax_scale
        # (B, block, np, S_k)
        attn_probs = torch.exp(attn_per_head - lse_block.unsqueeze(-1))
        attn_score = attn_probs.sum(dim=2)  # (B, block, S_k)
        attn_score = attn_score.masked_fill(~mask_block.unsqueeze(0), 0.0)
        attn_l1norm = attn_score.sum(dim=-1)  # (B, block)

        # --- KL loss for this block ---
        safe_l1 = attn_l1norm.clamp(min=eps)
        safe_lse = index_lse.clone()
        safe_lse[~row_valid_block] = 0.0
        target_block = attn_score / safe_l1.unsqueeze(-1)
        target_clamped = target_block.clamp(min=eps)
        position_valid = torch.isfinite(index_score)
        safe_index_score = torch.where(position_valid, index_score, torch.zeros_like(index_score))
        log_predict = safe_index_score - safe_lse.unsqueeze(-1)
        kl_terms = target_clamped * (torch.log(target_clamped) - log_predict)
        kl_terms = torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms))
        kl_per_row_block = kl_terms.sum(dim=-1)  # (B, block)
        kl_per_row_block = torch.where(row_valid_block, kl_per_row_block, torch.zeros_like(kl_per_row_block))
        kl_acc[:, q_start:q_end] = kl_per_row_block

        # --- Dense indexer backward for this block ---
        # grad through KL + logsumexp: predict_prob - target_prob
        predict_prob = torch.softmax(combined.masked_fill(~mask_block.unsqueeze(0), float("-inf")), dim=-1)
        predict_prob = predict_prob.masked_fill(~mask_block.unsqueeze(0), 0.0)
        target_prob = target_block.masked_fill(~mask_block.unsqueeze(0), 0.0)

        grad_logits = predict_prob - target_prob  # (B, block, S_k)
        grad_logits = grad_logits.masked_fill(~row_valid_block.unsqueeze(-1), 0.0)

        # grad_relu[b,q,h,k] = grad_logits[b,q,k] * W[b,q,h]
        grad_relu = grad_logits.unsqueeze(2) * w_block.unsqueeze(-1)  # (B, block, H_q, S_k)

        # grad_weights[b,q,h] = sum_k(grad_logits[b,q,k] * relu_score[b,q,h,k])
        grad_w[:, q_start:q_end] = (grad_logits.unsqueeze(2) * per_head_scores_relu).sum(dim=-1)

        # Grad through ReLU
        grad_pre_relu = grad_relu * relu_mask.float() * indexer_softmax_scale

        # Grad through matmul
        grad_q[:, q_start:q_end] = torch.einsum("bqhk,bkd->bqhd", grad_pre_relu, k_idx)
        grad_k += torch.einsum("bqhk,bqhd->bkd", grad_pre_relu, q_idx_block)

    # Final loss
    loss = kl_acc.sum() if calculate_per_token_loss else kl_acc.mean()
    indexer_loss = loss_coeff * loss

    return (
        indexer_loss,
        grad_q.to(q_idx_bshd.dtype),
        grad_k.to(k_idx_bsd.dtype),
        grad_w.to(w_bsh.dtype),
    )
