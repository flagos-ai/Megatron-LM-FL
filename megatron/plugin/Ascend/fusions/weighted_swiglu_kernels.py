# Copyright 2026 FlagOS Contributors
"""Tiled weighted SwiGLU backward for the Ascend Triton backend."""

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
