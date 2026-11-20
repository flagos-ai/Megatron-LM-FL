# Copyright 2026 FlagOS Contributors
"""Tiled weighted SwiGLU backward for the Ascend Triton backend."""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _weighted_swiglu_backward(
    G,
    Y,
    W,
    DY,
    DW,
    ROWS,
    WIDTH: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    Q: tl.constexpr,
    NATIVE_GRAD_ROUNDING: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute input gradients and accumulate each weight gradient across column tiles."""
    for start in range(tl.program_id(0) * BLOCK_R, ROWS, tl.num_programs(0) * BLOCK_R):
        r = start.to(tl.int64) + tl.arange(0, BLOCK_R)
        w = tl.load(W + r, r < ROWS, 0).to(tl.float32)
        total = tl.full((BLOCK_R,), 0, tl.float32)
        for column in range(0, WIDTH, BLOCK_C):
            c = column + tl.arange(0, BLOCK_C)
            mask = (r[:, None] < ROWS) & (c[None, :] < WIDTH)
            a = tl.load(Y + r[:, None] * (2 * WIDTH) + c[None, :], mask, 0)
            b = tl.load(Y + r[:, None] * (2 * WIDTH) + WIDTH + c[None, :], mask, 0)
            g = tl.load(G + r[:, None] * WIDTH + c[None, :], mask, 0)
            af = a.to(tl.float32)
            bf = b.to(tl.float32)
            sigmoid = 1.0 / (1.0 + tl.exp(-af))
            weighted = (g.to(tl.float32) * w[:, None]).to(P).to(tl.float32)
            if NATIVE_GRAD_ROUNDING:
                # aclnnSwiGluGrad evaluates derivatives before the output dtype cast.
                da = weighted * sigmoid * (1.0 + af * (1.0 - sigmoid)) * bf
                db = weighted * af * sigmoid
            else:
                # Core rounds sigmoid and each arithmetic stage to its promoted dtype.
                s = sigmoid.to(a.dtype).to(tl.float32)
                complement = (1.0 - s).to(a.dtype).to(tl.float32)
                product = (af * complement).to(a.dtype).to(tl.float32)
                factor = (1.0 + product).to(a.dtype).to(tl.float32)
                da = (weighted * s).to(D).to(tl.float32)
                da = (da * factor).to(D).to(tl.float32)
                da = da * bf
                silu = (af * sigmoid).to(a.dtype).to(tl.float32)
                db = weighted * silu
            # Match native forward's single output rounding when recomputing dw.
            output = (af * sigmoid * bf).to(a.dtype).to(tl.float32)
            wg = g.to(W.dtype.element_ty).to(tl.float32)
            dw = (output * wg).to(Q).to(tl.float32)
            total += tl.sum(tl.where(c[None, :] < WIDTH, dw, 0.0), 1)
            tl.store(DY + r[:, None] * (2 * WIDTH) + c[None, :], da, mask)
            tl.store(DY + r[:, None] * (2 * WIDTH) + WIDTH + c[None, :], db, mask)
        tl.store(DW + r, total, r < ROWS)


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    """Read the Ascend backend's required vector-core count."""
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return int(props["num_vectorcore"])


def weighted_swiglu_backward(g, y, weights, native_grad_rounding):
    """Launch a single kernel, preserving promoted products and FP32 row reductions."""
    width = y.shape[-1] // 2
    shape = (*y.shape[:-1], width)
    g = g.expand(shape).contiguous()
    y = y.contiguous()
    weights = weights.expand((*y.shape[:-1], 1)).contiguous()
    dy = torch.empty_like(y)
    dw = torch.empty_like(weights)
    types = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    p = torch.promote_types(g.dtype, weights.dtype)
    rows = y.numel() // y.shape[-1]
    # Bound UB usage: width 4352 overflows with a full power-of-two row tile.
    block_c = min(1024, triton.next_power_of_2(width))
    block_r = min(4, max(1, 2048 // block_c))
    grid = min(triton.cdiv(rows, block_r), _vector_cores(y.device.index))
    _weighted_swiglu_backward[(grid,)](
        g,
        y,
        weights,
        dy,
        dw,
        rows,
        width,
        types[p],
        types[torch.promote_types(p, y.dtype)],
        types[torch.promote_types(y.dtype, weights.dtype)],
        native_grad_rounding,
        block_r,
        block_c,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dy, dw


@triton.jit
def _clamped_values(a, b, limit):
    """Compute in FP32, then round the activation once as Core does."""
    af, bf = a.to(tl.float32), b.to(tl.float32)
    # Selects preserve NaNs, matching torch.clamp rather than minimum's minNum semantics.
    ac = tl.where(af > limit, limit, af)
    bc = tl.where(bf < -limit, -limit, tl.where(bf > limit, limit, bf))
    sigmoid = 1.0 / (1.0 + tl.exp(-ac))
    silu = ac * sigmoid
    activation = (silu * bc).to(a.dtype).to(tl.float32)
    return ac, bc, sigmoid, silu, activation


@triton.jit(do_not_specialize=["ROWS"])
def _clamped_swiglu_forward(
    X,
    W,
    Y,
    ROWS,
    WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    WEIGHTED: tl.constexpr,
    PRODUCT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    for start in range(tl.program_id(0) * BLOCK_R, ROWS, tl.num_programs(0) * BLOCK_R):
        r = start.to(tl.int64) + tl.arange(0, BLOCK_R)
        if WEIGHTED:
            weight = tl.load(W + r, r < ROWS, 0).to(PRODUCT).to(tl.float32)
        for column in range(0, WIDTH, BLOCK_C):
            c = column + tl.arange(0, BLOCK_C)
            mask = (r[:, None] < ROWS) & (c[None, :] < WIDTH)
            offset = r[:, None] * (2 * WIDTH) + c[None, :]
            a = tl.load(X + offset, mask, 0)
            b = tl.load(X + offset + WIDTH, mask, 0)
            _, _, _, _, value = _clamped_values(a, b, LIMIT)
            if WEIGHTED:
                value = (value * weight[:, None]).to(PRODUCT)
            tl.store(Y + r[:, None] * WIDTH + c[None, :], value, mask)


@triton.jit(do_not_specialize=["ROWS"])
def _clamped_swiglu_backward(
    X,
    W,
    G,
    DX,
    DW,
    ROWS,
    WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    WEIGHTED: tl.constexpr,
    PRODUCT: tl.constexpr,
    GRAD_PRODUCT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    for start in range(tl.program_id(0) * BLOCK_R, ROWS, tl.num_programs(0) * BLOCK_R):
        r = start.to(tl.int64) + tl.arange(0, BLOCK_R)
        if WEIGHTED:
            weight = tl.load(W + r, r < ROWS, 0).to(PRODUCT).to(tl.float32)
            total = tl.full((BLOCK_R,), 0, tl.float32)
        for column in range(0, WIDTH, BLOCK_C):
            c = column + tl.arange(0, BLOCK_C)
            mask = (r[:, None] < ROWS) & (c[None, :] < WIDTH)
            offset = r[:, None] * (2 * WIDTH) + c[None, :]
            a = tl.load(X + offset, mask, 0)
            b = tl.load(X + offset + WIDTH, mask, 0)
            grad = tl.load(G + r[:, None] * WIDTH + c[None, :], mask, 0).to(tl.float32)
            ac, bc, sigmoid, silu, activation = _clamped_values(a, b, LIMIT)
            if WEIGHTED:
                scaled = (grad * weight[:, None]).to(GRAD_PRODUCT).to(tl.float32)
                # Core casts g to the probability dtype before recomputing its gradient.
                wg = grad.to(W.dtype.element_ty).to(tl.float32)
                dw = (activation * wg).to(PRODUCT).to(tl.float32)
                total += tl.sum(tl.where(c[None, :] < WIDTH, dw, 0.0), axis=1)
            else:
                scaled = grad
            da = scaled * sigmoid * (1.0 + ac * (1.0 - sigmoid)) * bc
            db = scaled * silu
            # Multiply masks as Core does, including its NaN/Inf propagation.
            da *= (a.to(tl.float32) <= LIMIT).to(tl.float32)
            db *= ((b.to(tl.float32) >= -LIMIT) & (b.to(tl.float32) <= LIMIT)).to(tl.float32)
            tl.store(DX + offset, da, mask)
            tl.store(DX + offset + WIDTH, db, mask)
        if WEIGHTED:
            tl.store(DW + r, total, r < ROWS)


@lru_cache(maxsize=2)
def _log_clamped(weighted):
    logging.getLogger(__name__).info("Ascend clamped SwiGLU fusion weighted=%s", weighted)


def _clamped_launch(x, weights, limit, grad=None):
    """Bound vector-core and UB resources without materializing FP32 activation buffers."""
    width = x.shape[-1] // 2
    rows = x.numel() // x.shape[-1]
    weighted = weights is not None
    output = torch.empty_like(x) if grad is not None else x.new_empty((*x.shape[:-1], width))
    dw = (
        x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)
        if weighted and grad is not None
        else None
    )
    if rows == 0:
        if dw is not None:
            dw.zero_()
        return (output, dw) if grad is not None else output
    types = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    product = types[torch.promote_types(x.dtype, weights.dtype) if weighted else x.dtype]
    grad_product = types[torch.result_type(x, weights) if weighted else x.dtype]
    if weighted:
        weights = weights.expand((*x.shape[:-1], 1)).contiguous()
    block_c = min(512, triton.next_power_of_2(width))
    block_r = min(4, max(1, 1024 // block_c))
    grid = (min(triton.cdiv(rows, block_r), _vector_cores(x.device.index)),)
    args = dict(
        ROWS=rows,
        WIDTH=width,
        LIMIT=float(limit),
        WEIGHTED=weighted,
        PRODUCT=product,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if grad is None:
        _clamped_swiglu_forward[grid](x, weights, output, **args)
        _log_clamped(weighted)
        return output
    _clamped_swiglu_backward[grid](
        x, weights, grad.contiguous(), output, dw, GRAD_PRODUCT=grad_product, **args
    )
    return output, dw


def clamped_swiglu_forward(
    x: torch.Tensor, weights: torch.Tensor | None, limit: float
) -> torch.Tensor:
    """Compute clamped SwiGLU with Core's FP32 stages and optional per-token weights."""
    return _clamped_launch(x, weights, limit)


def clamped_swiglu_backward(
    grad: torch.Tensor, x: torch.Tensor, weights: torch.Tensor | None, limit: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return independent input storage and numerical per-token probability gradients."""
    return _clamped_launch(x, weights, limit, grad)
