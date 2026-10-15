# Copyright 2026 FlagOS Contributors
"""Native NPU SwiGLU primitives with the Core implementations as fallbacks.

Supported floating-point NPU inputs use the native operators when the
NPU vendor is selected. Other inputs keep the original Core implementations.
"""
from typing import Callable

import torch


def _native_op(y: torch.Tensor, name: str) -> Callable[..., torch.Tensor] | None:
    """Return public native operators only for the supported input contract."""
    if not (
        y.device.type == "npu"
        and y.layout == torch.strided
        and y.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 1 <= y.ndim <= 8
        and y.numel() > 0
        and y.shape[-1] % 2 == 0
    ):
        return None
    import torch_npu

    op = getattr(torch_npu, name, None)
    return op if callable(op) else None


def swiglu(y: torch.Tensor) -> torch.Tensor:
    """Apply native SiLU(gate) * value for supported 1D to 8D NPU inputs."""
    op = _native_op(y, "npu_swiglu")
    if op is None:
        from megatron.core.fusions.fused_bias_swiglu import swiglu as original

        return original.__wrapped__(y)
    # The public torch_npu API requires contiguous input; this is a no-op
    # for the usual contiguous activation and preserves strided-input values.
    return op(y.contiguous(), dim=-1)


def swiglu_back(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Keep the existing SwiGLUFunction saved-input and gradient contract."""
    op = _native_op(y, "npu_swiglu_backward")
    if op is None or not (
        g.device == y.device
        and g.layout == torch.strided
        and g.dtype == y.dtype
        and tuple(g.shape) == (*y.shape[:-1], y.shape[-1] // 2)
        # aclnnSwiGluGrad requires 64-byte aligned input rows for dim=-1.
        and y.shape[-1] * y.element_size() % 64 == 0
    ):
        from megatron.core.fusions.fused_bias_swiglu import swiglu_back as original

        return original.__wrapped__(g, y)
    # Preserve promoted-gradient semantics by falling back for mixed dtypes
    # rather than casting a FP32 weighted gradient down to the input dtype.
    return op(g.contiguous(), y.contiguous(), dim=-1)
