# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton fused DSA indexer KL-loss kernels (dense variant).

The unfused reference (``dsa.py``: ``fwd_fused_indexer_loss_naive`` /
``bwd_fused_indexer_loss_naive``) materialises two tensors that carry an
attention/indexer *head* dimension:

* per-head indexer scores ``[sq, b, index_n_heads, sk]``
* per-head attention scores ``[b, np, sq, sk]``

At the GLM5.2 744B shape (``sq = sk = 4096``, ``np = 16`` after TP=4,
``index_n_heads = 32``, ``b = 1``) those are ~2 GiB and ~1 GiB of fp32 *per
DSA ``full`` layer*, and the backward pass recomputes both. The head-free
``[b, sq, sk]`` tensors that actually feed the KL divergence are only 64 MiB.

These kernels perform the head reduction *inside* Triton so that no tensor
carrying a head dimension is ever written to global memory. Everything that
remains ``[b, sq, sk]``-shaped (the KL itself, the softmax, the row
reductions, the tensor-parallel all-reduce) stays in PyTorch, where it is
cheap and trivially comparable against the reference.

The maths is unchanged — this is an implementation optimisation only, so loss
curves stay directly comparable with runs made before it. Only the *dense*
indexer loss (``dsa_indexer_use_sparse_loss=False``, the default and what
GLM5.2 trains with) is implemented here; callers must fall back to the
unfused path for the sparse variant.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl


NEG_INF = float("-inf")


@lru_cache(maxsize=16)
def _causal_mask(sq: int, sk: int, device: torch.device) -> Tensor:
    """Cached ``[sq, sk]`` additive causal mask (``-inf`` above the diagonal).

    A local copy of ``dsa._cached_causal_neg_inf_mask``; duplicated rather than
    imported because ``dsa`` imports this module. Shared and read-only.
    """
    return torch.triu(
        torch.full((sq, sk), NEG_INF, dtype=torch.float32, device=device), diagonal=1
    )


# ---------------------------------------------------------------------------
# Block-size selection
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (>= 16, the tl.dot minimum)."""
    return max(16, 1 << (n - 1).bit_length())


def _select_blocks(head_dim_padded: int) -> Tuple[int, int, int]:
    """Pick ``(BLOCK_Q, BLOCK_K, num_warps)`` for a padded head dim.

    The score tile is ``[BLOCK_Q, BLOCK_K]`` fp32 and each operand tile is
    ``[BLOCK, head_dim_padded]``, so the budget shrinks as the head dim grows.
    """
    if head_dim_padded >= 256:
        return 32, 64, 4
    return 64, 64, 4


# ---------------------------------------------------------------------------
# Kernel 1: indexer index scores (head reduction fused)
# ---------------------------------------------------------------------------


@triton.jit
def _index_score_kernel(
    Q_ptr, K_ptr, W_ptr, OUT_ptr,
    SQ, SK, H,
    stride_q_s, stride_q_b, stride_q_h, stride_q_d,
    stride_k_s, stride_k_b, stride_k_d,
    stride_w_s, stride_w_b, stride_w_h,
    stride_o_b, stride_o_s, stride_o_k,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``out[b, s, t] = sum_h relu(q[s, b, h, :] . k[t, b, :]) * w[s, b, h]``.

    Mirrors ``dsa._compute_index_scores`` but reduces over ``h`` in registers,
    so the ``[sq, b, h, sk]`` intermediate never exists. ``w`` already carries
    the ``index_n_heads**-0.5 * softmax_scale`` factor (applied in
    ``DSAIndexer.forward_before_topk``), hence no scaling of ``q . k`` here.

    ``k`` is single-head and therefore shared by all indexer heads, so its tile
    is loaded once outside the head loop.
    """
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D_PAD)
    mask_q = offs_q < SQ
    mask_k = offs_k < SK
    mask_d = offs_d < D

    # K tile [BLOCK_K, D] — shared across every indexer head.
    k_tile = tl.load(
        K_ptr + offs_k[:, None] * stride_k_s + pid_b * stride_k_b
        + offs_d[None, :] * stride_k_d,
        mask=mask_k[:, None] & mask_d[None, :],
        other=0.0,
    )
    k_t = tl.trans(k_tile)

    acc = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

    for h in range(H):
        q_tile = tl.load(
            Q_ptr + offs_q[:, None] * stride_q_s + pid_b * stride_q_b
            + h * stride_q_h + offs_d[None, :] * stride_q_d,
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0,
        )
        # bf16 x bf16 -> fp32 accumulate: products are exact, matching the
        # reference's fp32 matmul up to summation order. input_precision only
        # binds for fp32 operands, where it forbids a silent TF32 downgrade.
        scores = tl.dot(q_tile, k_t, input_precision="ieee")
        scores = tl.maximum(scores, 0.0)
        w = tl.load(
            W_ptr + offs_q * stride_w_s + pid_b * stride_w_b + h * stride_w_h,
            mask=mask_q,
            other=0.0,
        ).to(tl.float32)
        acc += scores * w[:, None]

    tl.store(
        OUT_ptr + pid_b * stride_o_b + offs_q[:, None] * stride_o_s
        + offs_k[None, :] * stride_o_k,
        acc,
        mask=mask_q[:, None] & mask_k[None, :],
    )


# ---------------------------------------------------------------------------
# Kernel 2: per-head attention log-sum-exp
# ---------------------------------------------------------------------------


@triton.jit
def _attn_lse_kernel(
    Q_ptr, K_ptr, M_ptr, LSE_ptr,
    SQ, SK,
    softmax_scale,
    stride_q_s, stride_q_b, stride_q_h, stride_q_d,
    stride_k_s, stride_k_b, stride_k_h, stride_k_d,
    stride_m_b, stride_m_s, stride_m_k,
    stride_l_b, stride_l_h, stride_l_s,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``lse[b, h, s] = logsumexp_t(q . k * scale + mask)`` via online softmax.

    Output is only ``[b, np, sq]`` (~256 KiB at the 744B shape), so this pass
    costs a full Q@K sweep but writes almost nothing. Kernel 3 then reuses the
    LSE to emit the head-summed probabilities directly.

    Rows that are fully masked (every KV position ``-inf``, which the
    compressed-KV paths can produce) would give ``-inf + log(0)``; they are
    stored as ``0.0`` instead, and kernel 3 yields an all-zero row for them —
    matching the reference, which zeroes such rows after the softmax.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D_PAD)
    mask_q = offs_q < SQ
    mask_d = offs_d < D

    q_tile = tl.load(
        Q_ptr + offs_q[:, None] * stride_q_s + pid_b * stride_q_b
        + pid_h * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_q[:, None] & mask_d[None, :],
        other=0.0,
    )

    run_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    run_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for kt in range(0, SK, BLOCK_K):
        offs_k = kt + tl.arange(0, BLOCK_K)
        mask_k = offs_k < SK
        k_tile = tl.load(
            K_ptr + offs_k[:, None] * stride_k_s + pid_b * stride_k_b
            + pid_h * stride_k_h + offs_d[None, :] * stride_k_d,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale
        if HAS_MASK:
            m_tile = tl.load(
                M_ptr + pid_b * stride_m_b + offs_q[:, None] * stride_m_s
                + offs_k[None, :] * stride_m_k,
                mask=mask_q[:, None] & mask_k[None, :],
                other=float("-inf"),
            )
            scores = scores + m_tile
        scores = tl.where(mask_k[None, :], scores, float("-inf"))

        new_max = tl.maximum(run_max, tl.max(scores, axis=1))
        # Keep the exponent argument finite so a fully-masked row (new_max still
        # -inf) yields exp(-inf - 0) = 0 rather than exp(-inf + inf) = NaN.
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.exp(run_max - safe_max)
        probs = tl.exp(scores - safe_max[:, None])
        run_sum = run_sum * alpha + tl.sum(probs, axis=1)
        run_max = new_max

    lse = tl.where(run_sum > 0.0, run_max + tl.log(run_sum), 0.0)
    tl.store(
        LSE_ptr + pid_b * stride_l_b + pid_h * stride_l_h + offs_q * stride_l_s,
        lse,
        mask=mask_q,
    )


# ---------------------------------------------------------------------------
# Kernel 3: head-summed attention probabilities
# ---------------------------------------------------------------------------


@triton.jit
def _attn_head_sum_kernel(
    Q_ptr, K_ptr, M_ptr, LSE_ptr, OUT_ptr,
    SQ, SK, NP,
    softmax_scale,
    stride_q_s, stride_q_b, stride_q_h, stride_q_d,
    stride_k_s, stride_k_b, stride_k_h, stride_k_d,
    stride_m_b, stride_m_s, stride_m_k,
    stride_l_b, stride_l_h, stride_l_s,
    stride_o_b, stride_o_s, stride_o_k,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``out[b, s, t] = sum_h exp(q_h . k_h * scale + mask - lse[b, h, s])``.

    This is exactly ``softmax(attention_scores, dim=-1).sum(dim=1)`` from
    ``dsa.compute_dsa_indexer_loss``, but the ``[b, np, sq, sk]`` fp32 tensor
    is never written out: each program owns one ``[BLOCK_Q, BLOCK_K]`` output
    tile and accumulates over heads in registers.
    """
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D_PAD)
    mask_q = offs_q < SQ
    mask_k = offs_k < SK
    mask_d = offs_d < D

    if HAS_MASK:
        m_tile = tl.load(
            M_ptr + pid_b * stride_m_b + offs_q[:, None] * stride_m_s
            + offs_k[None, :] * stride_m_k,
            mask=mask_q[:, None] & mask_k[None, :],
            other=float("-inf"),
        )

    acc = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

    for h in range(NP):
        q_tile = tl.load(
            Q_ptr + offs_q[:, None] * stride_q_s + pid_b * stride_q_b
            + h * stride_q_h + offs_d[None, :] * stride_q_d,
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0,
        )
        k_tile = tl.load(
            K_ptr + offs_k[:, None] * stride_k_s + pid_b * stride_k_b
            + h * stride_k_h + offs_d[None, :] * stride_k_d,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale
        if HAS_MASK:
            scores = scores + m_tile
        lse = tl.load(
            LSE_ptr + pid_b * stride_l_b + h * stride_l_h + offs_q * stride_l_s,
            mask=mask_q,
            other=0.0,
        )
        probs = tl.exp(scores - lse[:, None])
        acc += tl.where(mask_k[None, :], probs, 0.0)

    tl.store(
        OUT_ptr + pid_b * stride_o_b + offs_q[:, None] * stride_o_s
        + offs_k[None, :] * stride_o_k,
        acc,
        mask=mask_q[:, None] & mask_k[None, :],
    )


# ---------------------------------------------------------------------------
# Kernel 4a: indexer backward — grad_q and grad_weights
# ---------------------------------------------------------------------------


@triton.jit
def _index_score_bwd_qw_kernel(
    Q_ptr, K_ptr, W_ptr, G_ptr, DQ_ptr, DW_ptr,
    SQ, SK,
    stride_q_s, stride_q_b, stride_q_h, stride_q_d,
    stride_k_s, stride_k_b, stride_k_d,
    stride_w_s, stride_w_b, stride_w_h,
    stride_g_b, stride_g_s, stride_g_k,
    stride_dq_s, stride_dq_b, stride_dq_h, stride_dq_d,
    stride_dw_s, stride_dw_b, stride_dw_h,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gradients that reduce over the KV axis, given ``g = dL/d(index_score)``.

    ``dw[s, h] = sum_t g[s, t] * relu(q_h . k_t)``
    ``dq[s, h, :] = sum_t g[s, t] * w[s, h] * 1[q_h . k_t > 0] * k_t``

    The per-head score is recomputed tile-by-tile instead of being reloaded
    from a saved ``[sq, b, h, sk]`` tensor.
    """
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D_PAD)
    mask_q = offs_q < SQ
    mask_d = offs_d < D

    q_tile = tl.load(
        Q_ptr + offs_q[:, None] * stride_q_s + pid_b * stride_q_b
        + pid_h * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_q[:, None] & mask_d[None, :],
        other=0.0,
    )
    w = tl.load(
        W_ptr + offs_q * stride_w_s + pid_b * stride_w_b + pid_h * stride_w_h,
        mask=mask_q,
        other=0.0,
    ).to(tl.float32)

    dq_acc = tl.zeros([BLOCK_Q, D_PAD], dtype=tl.float32)
    dw_acc = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for kt in range(0, SK, BLOCK_K):
        offs_k = kt + tl.arange(0, BLOCK_K)
        mask_k = offs_k < SK
        k_tile = tl.load(
            K_ptr + offs_k[:, None] * stride_k_s + pid_b * stride_k_b
            + offs_d[None, :] * stride_k_d,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q_tile, tl.trans(k_tile))
        g = tl.load(
            G_ptr + pid_b * stride_g_b + offs_q[:, None] * stride_g_s
            + offs_k[None, :] * stride_g_k,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        )

        # d/d(weights): flows through relu(score), not through the mask.
        dw_acc += tl.sum(g * tl.maximum(scores, 0.0), axis=1)

        # d/d(pre-relu score), then propagate into q.
        g_pre = tl.where(scores > 0.0, g * w[:, None], 0.0)
        dq_acc += tl.dot(g_pre, k_tile.to(tl.float32), input_precision="ieee")

    tl.store(
        DQ_ptr + offs_q[:, None] * stride_dq_s + pid_b * stride_dq_b
        + pid_h * stride_dq_h + offs_d[None, :] * stride_dq_d,
        dq_acc,
        mask=mask_q[:, None] & mask_d[None, :],
    )
    tl.store(
        DW_ptr + offs_q * stride_dw_s + pid_b * stride_dw_b + pid_h * stride_dw_h,
        dw_acc,
        mask=mask_q,
    )


# ---------------------------------------------------------------------------
# Kernel 4b: indexer backward — grad_k
# ---------------------------------------------------------------------------


@triton.jit
def _index_score_bwd_k_kernel(
    Q_ptr, K_ptr, W_ptr, G_ptr, DK_ptr,
    SQ, SK,
    stride_q_s, stride_q_b, stride_q_h, stride_q_d,
    stride_k_s, stride_k_b, stride_k_d,
    stride_w_s, stride_w_b, stride_w_h,
    stride_g_b, stride_g_s, stride_g_k,
    stride_dk_h, stride_dk_s, stride_dk_b, stride_dk_d,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``dk[t, :] = sum_s sum_h g[s, t] * w[s, h] * 1[q_h . k_t > 0] * q[s, h, :]``.

    Reduces over the query axis, so it needs its own grid orientation (one
    program per KV tile). The head axis is *also* reduced, but carrying it in the
    grid instead of looping inside keeps the GPU busy: ``k`` is single-head, so
    with the head loop inline the grid was only ``(b, sk/BLOCK_K)`` — 16 programs
    at sq=1024, i.e. ~12% occupancy on a 132-SM H200, which is what made the
    fused backward slower than the reference at short sequences.

    Each program writes ``dk_partial[pid_h]``, and the caller sums over that axis.
    That keeps the reduction deterministic (no atomics) at the cost of an
    ``[H, sk, b, D_PAD]`` fp32 buffer.
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D_PAD)
    mask_k = offs_k < SK
    mask_d = offs_d < D

    k_tile = tl.load(
        K_ptr + offs_k[:, None] * stride_k_s + pid_b * stride_k_b
        + offs_d[None, :] * stride_k_d,
        mask=mask_k[:, None] & mask_d[None, :],
        other=0.0,
    )
    k_t = tl.trans(k_tile)

    dk_acc = tl.zeros([BLOCK_K, D_PAD], dtype=tl.float32)

    for qt in range(0, SQ, BLOCK_Q):
        offs_q = qt + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < SQ
        g = tl.load(
            G_ptr + pid_b * stride_g_b + offs_q[:, None] * stride_g_s
            + offs_k[None, :] * stride_g_k,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        )
        q_tile = tl.load(
            Q_ptr + offs_q[:, None] * stride_q_s + pid_b * stride_q_b
            + pid_h * stride_q_h + offs_d[None, :] * stride_q_d,
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0,
        )
        w = tl.load(
            W_ptr + offs_q * stride_w_s + pid_b * stride_w_b + pid_h * stride_w_h,
            mask=mask_q,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(q_tile, k_t)
        g_pre = tl.where(scores > 0.0, g * w[:, None], 0.0)
        dk_acc += tl.dot(tl.trans(g_pre), q_tile.to(tl.float32), input_precision="ieee")

    tl.store(
        DK_ptr + pid_h * stride_dk_h + offs_k[:, None] * stride_dk_s
        + pid_b * stride_dk_b + offs_d[None, :] * stride_dk_d,
        dk_acc,
        mask=mask_k[:, None] & mask_d[None, :],
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _mask_strides(mask: Optional[Tensor], b: int) -> Tuple[int, int, int]:
    """Strides for a ``[sq, sk]`` or ``[b, sq, sk]`` mask.

    A 2-D (batch-shared) mask is addressed with a zero batch stride, so both
    layouts share one kernel.
    """
    if mask is None:
        return 0, 0, 0
    if mask.dim() == 2:
        return 0, mask.stride(0), mask.stride(1)
    assert mask.shape[0] in (1, b), f"unexpected mask batch dim: {tuple(mask.shape)}"
    batch_stride = mask.stride(0) if mask.shape[0] == b else 0
    return batch_stride, mask.stride(1), mask.stride(2)


def indexer_index_score(q: Tensor, k: Tensor, weights: Tensor) -> Tensor:
    """Head-reduced indexer scores, without any mask applied.

    Args:
        q: ``[sq, b, index_n_heads, d]`` indexer queries.
        k: ``[sk, b, d]`` indexer keys (single head, shared across heads).
        weights: ``[sq, b, index_n_heads]`` per-head weights, already scaled.

    Returns:
        ``[b, sq, sk]`` fp32 — equivalent to ``dsa._compute_index_scores``.
    """
    sq, b, h, d = q.shape
    sk = k.shape[0]
    d_pad = _next_pow2(d)
    block_q, block_k, num_warps = _select_blocks(d_pad)

    out = torch.empty((b, sq, sk), dtype=torch.float32, device=q.device)
    grid = (b, triton.cdiv(sq, block_q), triton.cdiv(sk, block_k))
    _index_score_kernel[grid](
        q, k, weights, out,
        sq, sk, h,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        D=d, D_PAD=d_pad,
        BLOCK_Q=block_q, BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def attn_head_sum(
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    mask: Optional[Tensor],
) -> Tensor:
    """Head-summed attention softmax probabilities.

    Args:
        query: ``[sq, b, np, hn]`` attention queries.
        key: ``[sk, b, np, hn]`` attention keys.
        softmax_scale: scale applied to ``q . k``.
        mask: additive ``-inf``/``0`` mask, ``[sq, sk]`` or ``[b, sq, sk]``.

    Returns:
        ``[b, sq, sk]`` fp32 — equivalent to
        ``softmax(q @ k^T * scale + mask, dim=-1).sum(dim=1)``, with
        fully-masked rows returned as zero.
    """
    sq, b, np_, hn = query.shape
    sk = key.shape[0]
    d_pad = _next_pow2(hn)
    block_q, block_k, num_warps = _select_blocks(d_pad)
    has_mask = mask is not None
    sm_b, sm_s, sm_k = _mask_strides(mask, b)
    mask_arg = mask if has_mask else query  # unused pointer when HAS_MASK=False

    lse = torch.empty((b, np_, sq), dtype=torch.float32, device=query.device)
    _attn_lse_kernel[(b, np_, triton.cdiv(sq, block_q))](
        query, key, mask_arg, lse,
        sq, sk,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        sm_b, sm_s, sm_k,
        lse.stride(0), lse.stride(1), lse.stride(2),
        D=hn, D_PAD=d_pad, HAS_MASK=has_mask,
        BLOCK_Q=block_q, BLOCK_K=block_k,
        num_warps=num_warps,
    )

    head_sum = torch.empty((b, sq, sk), dtype=torch.float32, device=query.device)
    _attn_head_sum_kernel[(b, triton.cdiv(sq, block_q), triton.cdiv(sk, block_k))](
        query, key, mask_arg, lse, head_sum,
        sq, sk, np_,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        sm_b, sm_s, sm_k,
        lse.stride(0), lse.stride(1), lse.stride(2),
        head_sum.stride(0), head_sum.stride(1), head_sum.stride(2),
        D=hn, D_PAD=d_pad, HAS_MASK=has_mask,
        BLOCK_Q=block_q, BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return head_sum


def indexer_index_score_backward(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    grad_index_score: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward of :func:`indexer_index_score`.

    Args:
        q: ``[sq, b, h, d]`` indexer queries.
        k: ``[sk, b, d]`` indexer keys.
        weights: ``[sq, b, h]`` per-head weights.
        grad_index_score: ``[b, sq, sk]`` fp32 gradient w.r.t. the scores.

    Returns:
        ``(grad_q, grad_weights, grad_k)`` in the input layouts and dtypes.
    """
    sq, b, h, d = q.shape
    sk = k.shape[0]
    d_pad = _next_pow2(d)
    block_q, block_k, num_warps = _select_blocks(d_pad)

    grad_index_score = grad_index_score.contiguous()
    grad_q = torch.empty((sq, b, h, d_pad), dtype=torch.float32, device=q.device)
    grad_w = torch.empty((sq, b, h), dtype=torch.float32, device=q.device)
    # dk reduces over heads. Parallelising that axis instead of looping over it
    # inside the kernel needs a per-head buffer, summed below; see the kernel
    # docstring for why the extra allocation is worth it.
    grad_k_partial = torch.empty((h, sk, b, d_pad), dtype=torch.float32, device=q.device)

    _index_score_bwd_qw_kernel[(b, triton.cdiv(sq, block_q), h)](
        q, k, weights, grad_index_score, grad_q, grad_w,
        sq, sk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        grad_index_score.stride(0), grad_index_score.stride(1),
        grad_index_score.stride(2),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
        grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
        D=d, D_PAD=d_pad,
        BLOCK_Q=block_q, BLOCK_K=block_k,
        num_warps=num_warps,
    )

    _index_score_bwd_k_kernel[(b, triton.cdiv(sk, block_k), h)](
        q, k, weights, grad_index_score, grad_k_partial,
        sq, sk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        grad_index_score.stride(0), grad_index_score.stride(1),
        grad_index_score.stride(2),
        grad_k_partial.stride(0), grad_k_partial.stride(1),
        grad_k_partial.stride(2), grad_k_partial.stride(3),
        D=d, D_PAD=d_pad,
        BLOCK_Q=block_q, BLOCK_K=block_k,
        num_warps=num_warps,
    )

    return (
        grad_q[..., :d].to(q.dtype),
        grad_w.to(weights.dtype),
        grad_k_partial.sum(dim=0)[..., :d].to(k.dtype),
    )


# ---------------------------------------------------------------------------
# Shared PyTorch glue: mask -> row validity, head_sum -> target, predict
# ---------------------------------------------------------------------------


def _row_masks(mask: Tensor, b: int) -> Tuple[Tensor, Tensor]:
    """Return ``(row_valid, causal_valid)`` for a ``-inf``/``0`` additive mask.

    ``row_valid`` broadcasts against ``[b, sq, sk]`` and marks rows with at
    least one unmasked position; ``causal_valid`` marks individual valid
    positions. Mirrors ``dsa.compute_dsa_indexer_loss``.
    """
    causal_valid = mask > NEG_INF
    if causal_valid.dim() == 2:
        causal_valid = causal_valid.unsqueeze(0)
    row_valid = causal_valid.any(dim=-1, keepdim=True)
    return row_valid, causal_valid


def _target_and_predict(
    index_score: Tensor,
    head_sum: Tensor,
    row_valid: Tensor,
    pg_collection,
) -> Tuple[Tensor, Tensor]:
    """Turn head-summed probabilities and masked scores into KL distributions.

    ``target`` is the L1-normalised attention distribution (all-reduced across
    tensor-parallel ranks, which hold different heads); ``predict`` is the
    indexer softmax. Both are zero on fully-masked rows.

    ``head_sum`` always arrives fresh from :func:`attn_head_sum`, so it is
    masked in place.
    """
    head_sum.mul_(row_valid)
    if pg_collection is not None and pg_collection.tp.size() > 1:
        # Heads are sharded across TP ranks, so the head sum must be completed
        # before normalising — same point as the reference all-reduce.
        torch.distributed.all_reduce(head_sum.contiguous(), group=pg_collection.tp)
    target = head_sum / head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    predict = torch.softmax(
        index_score.masked_fill(~row_valid, 0.0), dim=-1, dtype=torch.float32
    )
    predict = predict * row_valid
    return target, predict


# ---------------------------------------------------------------------------
# Drop-in forward / backward replacements
# ---------------------------------------------------------------------------


def fwd_indexer_loss_triton(
    q: Tensor,
    weights: Tensor,
    k: Tensor,
    query: Tensor,
    key: Tensor,
    topk: int,
    softmax_scale: float,
    loss_coeff: float,
    mask: Optional[Tensor],
    pg_collection,
    calculate_per_token_loss: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Triton counterpart of ``dsa.fwd_fused_indexer_loss_naive`` (dense only).

    Returns ``(topk_indices, indexer_loss, index_score)``. ``index_score`` is
    the ``[b, sq, sk]`` score tensor; it is returned so callers can hand it to
    the backward pass, though the backward recomputes it by default.

    ``mask`` is threaded exactly as the reference threads it, including the
    asymmetric ``mask=None`` case: ``fused_qk_topk_naive`` only adds a mask when
    one is given, so the index scores stay non-causal, while
    ``compute_dsa_indexer_loss`` falls back to an upper-triangular mask for the
    *attention* scores. Note the reference's own backward is not consistent with
    this -- it masks both -- so :func:`bwd_indexer_loss_triton` mirrors that
    instead. ``DSAttention`` always passes a real mask, so neither quirk is
    reachable in training.
    """
    sq, b, _, _ = q.shape
    sk = k.shape[0]
    # Attention always gets a causal mask; the index scores only get one if the
    # caller supplied it (see docstring).
    attn_mask = _causal_mask(sq, sk, q.device) if mask is None else mask

    index_score = indexer_index_score(q, k, weights)
    if mask is not None:
        index_score = index_score + mask
    topk_indices = index_score.topk(min(topk, sk), dim=-1)[1]

    row_valid, _ = _row_masks(attn_mask, b)
    head_sum = attn_head_sum(query, key, softmax_scale, attn_mask)
    target, predict = _target_and_predict(index_score, head_sum, row_valid, pg_collection)

    kl_per_row = (
        target * (torch.log(target + 1e-10) - torch.log(predict + 1e-10))
    ).sum(dim=-1)
    kl_div = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return topk_indices, kl_div * loss_coeff, index_score


def bwd_indexer_loss_triton(
    q: Tensor,
    weights: Tensor,
    k: Tensor,
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    grad_loss: Tensor,
    pg_collection,
    mask: Optional[Tensor],
    calculate_per_token_loss: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Triton counterpart of ``dsa.bwd_fused_indexer_loss_naive`` (dense only).

    Recomputes the two head-dimension reductions rather than saving them, and
    returns ``(grad_q, grad_weights, grad_k)``.
    """
    sq, b, _, _ = q.shape
    sk = k.shape[0]
    # The reference is asymmetric between its own passes, and this mirrors it:
    # forward reaches the index scores through fused_qk_topk_naive, which only adds a
    # mask when one is given, whereas bwd_fused_indexer_loss_naive adds the causal mask
    # to *both* attention and index scores unconditionally. DSAttention always supplies
    # a mask, so the difference only surfaces in direct unit tests.
    attn_mask = _causal_mask(sq, sk, q.device) if mask is None else mask

    index_score = indexer_index_score(q, k, weights) + attn_mask

    row_valid, causal_valid = _row_masks(attn_mask, b)
    head_sum = attn_head_sum(query, key, softmax_scale, attn_mask)
    target, predict = _target_and_predict(index_score, head_sum, row_valid, pg_collection)

    # d(loss)/d(kl_per_element), constant across the [b, sq, sk] grid.
    grad_kl = grad_loss * loss_coeff
    if not calculate_per_token_loss:
        grad_kl = grad_kl / (b * sq)

    # KL = target * (log target - log predict)  =>  d/d(predict) = -target / predict
    grad_predict = -target / (predict + 1e-10) * grad_kl
    # Softmax backward.
    sum_grad = (grad_predict * predict).sum(dim=-1, keepdim=True)
    grad_index_score = predict * (grad_predict - sum_grad)
    grad_index_score = grad_index_score * causal_valid

    return indexer_index_score_backward(q, k, weights, grad_index_score)


__all__ = [
    "attn_head_sum",
    "bwd_indexer_loss_triton",
    "fwd_indexer_loss_triton",
    "indexer_index_score",
    "indexer_index_score_backward",
]
