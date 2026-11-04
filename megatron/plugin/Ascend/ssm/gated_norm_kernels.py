# Copyright 2026 FlagOS Contributors
"""Pointwise GDN gate kernels with FP32 arithmetic and bounded launches."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel"])
def _gate_fwd(Y, G, O, numel, BLOCK: tl.constexpr):
    for start in range(tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < numel
        y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
        g = tl.load(G + offsets, mask, other=0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sigmoid
        tl.store(O + offsets, y * silu, mask)


@triton.jit(do_not_specialize=["numel"])
def _gate_bwd(Y, G, D, DY, DG, numel, BLOCK: tl.constexpr):
    for start in range(tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < numel
        y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
        g = tl.load(G + offsets, mask, other=0).to(tl.float32)
        d = tl.load(D + offsets, mask, other=0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sigmoid
        derivative = sigmoid * (1.0 + g * (1.0 - sigmoid))
        tl.store(DY + offsets, d * silu, mask)
        tl.store(DG + offsets, (d * y) * derivative, mask)


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return max(1, min(int(props["num_vectorcore"]), 65535))


def _launch_options(tensor):
    block = 2048
    grid = min(triton.cdiv(tensor.numel(), block), _vector_cores(tensor.device.index))
    return grid, block


def gate_forward(normalized, gate, output_dtype):
    output = torch.empty_like(normalized, dtype=output_dtype)
    if not normalized.numel():
        return output
    grid, block = _launch_options(normalized)
    _gate_fwd[(grid,)](
        normalized, gate, output, normalized.numel(), block, num_warps=4, enable_fp_fusion=False
    )
    return output


def gate_backward(normalized, gate, grad_output):
    dy = torch.empty_like(normalized)
    dg = torch.empty_like(gate)
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
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dy, dg
