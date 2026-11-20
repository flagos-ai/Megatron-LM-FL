# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(
    X,
    W,
    BIAS,
    O,
    LINEAR,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    N,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BS: tl.constexpr,
    SILU: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    nt: tl.constexpr = triton.cdiv(T, BT)
    nd: tl.constexpr = triton.cdiv(D, BD)
    for task in range(tl.program_id(0), N // (T * D) * nt * nd, tl.num_programs(0)):
        c = task % nd * BD + tl.arange(0, BD)[None, :]
        t = task // nd % nt * BT + tl.arange(0, BT)[:, None]
        b = task // (nd * nt)
        mask = (t < T) & (c < D)
        y = tl.full((BT, BD), 0, tl.float32)
        for k in range(K):
            s = t + k - K + 1
            x = tl.load(
                X + b * XS0 + tl.minimum(tl.maximum(s, 0), T - 1) * XS1 + c * XS2,
                mask & (s >= 0),
                other=0,
            ).to(tl.float32)
            w = tl.load(W + c * WS0 + k * WS1, c < D, other=0).to(tl.float32)
            y = y + x * w
        if BIAS is not None:
            y += tl.load(BIAS + c * BS, c < D, other=0).to(tl.float32)
        y = y.to(X.dtype.element_ty).to(tl.float32)
        if LINEAR is not None:
            tl.store(LINEAR + (b * T + t) * D + c, y, mask)
        if SILU:
            sigmoid = 1.0 / (1.0 + tl.exp(-y))
            y = y * sigmoid
        tl.store(O + (b * T + t) * D + c, y, mask)


@triton.jit
def _silu_bwd(
    Y,
    G,
    O,
    N,
    T: tl.constexpr,
    D: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    nt: tl.constexpr = triton.cdiv(T, BT)
    nd: tl.constexpr = triton.cdiv(D, BD)
    for task in range(tl.program_id(0), N // (T * D) * nt * nd, tl.num_programs(0)):
        c = task % nd * BD + tl.arange(0, BD)[None, :]
        t = task // nd % nt * BT + tl.arange(0, BT)[:, None]
        b = task // (nd * nt)
        mask = (t < T) & (c < D)
        i = (b * T + t) * D + c
        y = tl.load(Y + i, mask, other=0).to(tl.float32)
        g = tl.load(G + b * GS0 + t * GS1 + c * GS2, mask, other=0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-y))
        tl.store(O + i, g * (sigmoid * (1.0 + y * (1.0 - sigmoid))), mask)


@triton.jit
def _dx(
    G,
    W,
    DX,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    N,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    nt: tl.constexpr = triton.cdiv(T, BT)
    nd: tl.constexpr = triton.cdiv(D, BD)
    for task in range(tl.program_id(0), N // (T * D) * nt * nd, tl.num_programs(0)):
        c = task % nd * BD + tl.arange(0, BD)[None, :]
        t = task // nd % nt * BT + tl.arange(0, BT)[:, None]
        b = task // (nd * nt)
        mask = (t < T) & (c < D)
        dx = tl.full((BT, BD), 0, tl.float32)
        for k in range(K):
            s = t + K - 1 - k
            g = tl.load(G + (b * T + tl.minimum(s, T - 1)) * D + c, mask & (s < T), other=0).to(
                tl.float32
            )
            w = tl.load(W + c * WS0 + k * WS1, c < D, other=0).to(tl.float32)
            dx = dx + g * w
        tl.store(DX + (b * T + t) * D + c, dx, mask)


@triton.jit
def _dw(
    X,
    G,
    P,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    ROWS,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    SPLITS: tl.constexpr,
    DB: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(D, BD)
    length = tl.cdiv(ROWS, SPLITS * BT) * BT
    for task in range(tl.program_id(0), SPLITS * tiles, tl.num_programs(0)):
        part = task // tiles
        c = task % tiles * BD + tl.arange(0, BD)
        for k in range(K + DB):
            acc = tl.full((BT, BD), 0, tl.float32)
            for start in range(part * length, tl.minimum((part + 1) * length, ROWS), BT):
                r = start + tl.arange(0, BT)
                t = r % T
                b = r // T
                g = tl.load(
                    G + r[:, None] * D + c[None, :], (r[:, None] < ROWS) & (c[None, :] < D), other=0
                ).to(tl.float32)
                if k < K:
                    s = t + k - K + 1
                    x = tl.load(
                        X
                        + b[:, None] * XS0
                        + tl.minimum(tl.maximum(s[:, None], 0), T - 1) * XS1
                        + c[None, :] * XS2,
                        (r[:, None] < ROWS) & (s[:, None] >= 0) & (c[None, :] < D),
                        other=0,
                    ).to(tl.float32)
                    g = g * x
                acc = acc + g
            tl.store(P + (part * (K + DB) + k) * D + c, tl.sum(acc, 0), c < D)


@triton.jit
def _reduce(
    P,
    DW,
    DBIAS,
    D: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    DB: tl.constexpr,
    BD: tl.constexpr,
):
    c = tl.program_id(0) * BD + tl.arange(0, BD)
    p = tl.arange(0, triton.next_power_of_2(SPLITS))
    for k in range(K + DB):
        v = tl.load(
            P + (p[:, None] * (K + DB) + k) * D + c[None, :],
            (p[:, None] < SPLITS) & (c[None, :] < D),
            other=0,
        )
        v = tl.sum(v, 0)
        if k < K:
            tl.store(DW + c * K + k, v, c < D)
        else:
            if DB:
                tl.store(DBIAS + c, v, c < D)


@lru_cache(maxsize=None)
def _cores(device):
    return int(triton.runtime.driver.active.utils.get_device_properties(device)["num_vectorcore"])


def _grid(x, tasks):
    return (max(1, min(tasks, _cores(x.device.index))),)


def _channel_block(tensor, preferred):
    return min(
        preferred if tensor.stride(-1) == 1 else 128,
        max(32, triton.next_power_of_2(tensor.shape[-1])),
    )


def forward(x, weight, bias, silu, save_linear=False):
    weight = weight.t().contiguous().t()
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    linear = torch.empty_like(output) if save_linear else None
    _conv[_grid(x, triton.cdiv(x.numel(), 1024))](
        x,
        weight,
        bias,
        output,
        linear,
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        x.numel(),
        *x.stride(),
        *weight.stride(),
        bias.stride(0) if bias is not None else 0,
        silu,
        min(8, triton.next_power_of_2(x.shape[1])),
        _channel_block(x, 512),
        enable_fp_fusion=False,
    )
    return output, linear


def silu_backward(linear, grad):
    result = torch.empty_like(linear)
    _silu_bwd[_grid(linear, triton.cdiv(linear.numel(), 2048))](
        linear,
        grad,
        result,
        linear.numel(),
        linear.shape[1],
        linear.shape[2],
        *grad.stride(),
        min(8, triton.next_power_of_2(linear.shape[1])),
        _channel_block(grad, 512),
        enable_fp_fusion=False,
    )
    return result


def backward(x, weight, grad, need_x, need_weight, need_bias):
    weight = weight.t().contiguous().t()
    dx = dw = db = None
    if need_x:
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _dx[_grid(x, triton.cdiv(x.numel(), 1024))](
            grad,
            weight,
            dx,
            x.shape[1],
            x.shape[2],
            weight.shape[1],
            x.numel(),
            *weight.stride(),
            min(8, triton.next_power_of_2(x.shape[1])),
            _channel_block(grad, 512),
            enable_fp_fusion=False,
        )
    if need_weight or need_bias:
        rows, d, k = x.shape[0] * x.shape[1], x.shape[2], weight.shape[1]
        splits = min(32, triton.cdiv(rows, 128))
        partial = torch.empty((splits, k + need_bias, d), device=x.device, dtype=torch.float32)
        dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
        db = torch.empty((d,), device=x.device, dtype=x.dtype) if need_bias else None
        bd = min(128 if x.stride(-1) == 1 else 32, triton.next_power_of_2(d))
        _dw[_grid(x, splits * triton.cdiv(d, bd))](
            x,
            grad,
            partial,
            x.shape[1],
            d,
            k,
            rows,
            *x.stride(),
            splits,
            need_bias,
            32,
            bd,
            enable_fp_fusion=False,
        )
        _reduce[(triton.cdiv(d, 32),)](partial, dw, db, d, k, splits, need_bias, 32)
    return dx, dw if need_weight else None, db
