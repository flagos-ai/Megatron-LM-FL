# Copyright 2026 FlagOS Contributors
"""Shared sigmoid/SiLU gate kernels with FP32 arithmetic and bounded launches."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel"])
def _gate_fwd(Y, G, O, numel, BLOCK: tl.constexpr, SILU_GATE: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
        g = tl.load(G + offsets, mask, other=0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-g))
        activation = g * sigmoid if SILU_GATE else sigmoid
        tl.store(O + offsets, y * activation, mask)


@triton.jit(do_not_specialize=["numel"])
def _gate_bwd(
    Y,
    G,
    D,
    DY,
    DG,
    numel,
    BLOCK: tl.constexpr,
    SILU_GATE: tl.constexpr,
    NEEDS_DY: tl.constexpr,
    NEEDS_DG: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
        g = tl.load(G + offsets, mask, other=0).to(tl.float32)
        d = tl.load(D + offsets, mask, other=0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-g))
        if NEEDS_DY:
            activation = g * sigmoid if SILU_GATE else sigmoid
            tl.store(DY + offsets, d * activation, mask)
        if NEEDS_DG:
            if SILU_GATE:
                derivative = sigmoid * (1.0 + g * (1.0 - sigmoid))
                dg = (d * y) * derivative
            else:
                # Native sigmoid backward multiplies by (1 - output) before output.
                dg = ((d * y) * (1.0 - sigmoid)) * sigmoid
            tl.store(DG + offsets, dg, mask)


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return max(1, min(int(props["num_vectorcore"]), 65535))


def _launch_options(tensor):
    block = 2048
    grid = min(triton.cdiv(tensor.numel(), block), _vector_cores(tensor.device.index))
    return grid, block


def gate_forward(normalized, gate, output_dtype, silu=True):
    output = torch.empty_like(normalized, dtype=output_dtype)
    if not normalized.numel():
        return output
    grid, block = _launch_options(normalized)
    _gate_fwd[(grid,)](
        normalized,
        gate,
        output,
        normalized.numel(),
        block,
        silu,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def gate_backward(normalized, gate, grad_output, silu=True, needs_input_grad=(True, True)):
    dy = torch.empty_like(normalized) if needs_input_grad[0] else None
    dg = torch.empty_like(gate) if needs_input_grad[1] else None
    if not normalized.numel():
        return dy, dg
    grid, block = _launch_options(normalized)
    _gate_bwd[(grid,)](
        normalized,
        gate,
        grad_output,
        dy,
        dg,
        normalized.numel(),
        block,
        silu,
        *needs_input_grad,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dy, dg
