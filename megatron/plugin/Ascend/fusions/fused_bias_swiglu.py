# Copyright 2026 FlagOS Contributors
"""Ascend SwiGLU overrides with Core fallbacks for unsupported inputs."""

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _native_op(y, name):
    """Return an available native operator for its supported input contract."""
    if not (
        y.device.type == "npu"
        and y.layout == torch.strided
        and y.dtype in _FLOAT_DTYPES
        and 1 <= y.ndim <= 8
        and y.numel() > 0
        and y.shape[-1] % 2 == 0
    ):
        return None
    import torch_npu

    op = getattr(torch_npu, name, None)
    return op if callable(op) else None


def swiglu(y):
    """Compute SwiGLU using the native operator when supported."""
    op = _native_op(y, "npu_swiglu")
    if op is None:
        from megatron.core.fusions.fused_bias_swiglu import swiglu as original

        # Bypass override dispatch; calling original here would recurse.
        return original.__wrapped__(y)
    return op(y.contiguous(), dim=-1)


def swiglu_back(g, y):
    """Use native input gradients for matching, aligned inputs."""
    op = _native_op(y, "npu_swiglu_backward")
    if op is None or not (
        g.device == y.device
        and g.layout == torch.strided
        and g.dtype == y.dtype
        and tuple(g.shape) == (*y.shape[:-1], y.shape[-1] // 2)
        # aclnnSwiGluGrad requires input rows aligned to 64 bytes.
        and y.shape[-1] * y.element_size() % 64 == 0
    ):
        from megatron.core.fusions.fused_bias_swiglu import swiglu_back as original

        return original.__wrapped__(g, y)
    return op(g.contiguous(), y.contiguous(), dim=-1)


def weighted_swiglu_back(g, y, weights):
    """Fuse weighted input gradients and the per-row weight-gradient reduction."""
    from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu_back as original

    if (
        # dw recomputes the dispatched native forward, so it must be available.
        _native_op(y, "npu_swiglu") is None
        or any(
            t.device != y.device or t.layout != torch.strided or t.dtype not in _FLOAT_DTYPES
            for t in (g, weights)
        )
        or (torch.is_grad_enabled() and any(t.requires_grad for t in (g, y, weights)))
        or (weights.ndim > 0 and weights.shape[-1] != 1)
    ):
        return original.__wrapped__(g, y, weights)
    shape = (*y.shape[:-1], y.shape[-1] // 2)
    weighted_shape = torch.broadcast_shapes(g.shape, weights.shape)
    if torch.broadcast_shapes(weighted_shape, shape) != shape:
        return original.__wrapped__(g, y, weights)
    try:
        from .weighted_swiglu_kernels import weighted_swiglu_backward
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        return original.__wrapped__(g, y, weights)

    import torch_npu

    # Match native backward when selectable; otherwise match Core's staged rounding.
    native_grad_rounding = (
        torch.promote_types(g.dtype, weights.dtype) == y.dtype
        and weighted_shape == shape
        and y.shape[-1] * y.element_size() % 64 == 0
        and callable(getattr(torch_npu, "npu_swiglu_backward", None))
    )
    return weighted_swiglu_backward(g, y, weights, native_grad_rounding)


def _triton_available():
    try:
        from . import weighted_swiglu_kernels  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        return False
    return True


def _clamped_supported(input, bias, fp8_input_store, clamp_value):
    return (
        input.device.type == "npu"
        and input.layout == torch.strided
        and input.dtype in _FLOAT_DTYPES
        and input.ndim in (2, 3)
        and input.shape[-1] > 0
        and input.shape[-1] % 2 == 0
        and bias is None
        and not fp8_input_store
        and clamp_value is not None
        and clamp_value > 0
        and _triton_available()
    )


class _ClampedSwiGLU(torch.autograd.Function):
    """Match Core's FP32 clamped activation and weighted backward."""

    @staticmethod
    def forward(ctx, input, weights, limit):
        """Save input references and produce the fused activation."""
        from .weighted_swiglu_kernels import clamped_swiglu_forward

        ctx.save_for_backward(input, weights)
        ctx.limit = limit
        return clamped_swiglu_forward(input, weights, limit)

    @staticmethod
    def backward(ctx, grad):
        """Recompute activation tiles and reduce broadcast probability gradients."""
        input, weights = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Reuse Core's differentiable analytic backward for higher-order gradients.
            from megatron.core.fusions.fused_bias_swiglu import (
                clamped_swiglu_back,
                clamped_weighted_swiglu_back,
            )

            if weights is None:
                return clamped_swiglu_back(grad, input, ctx.limit), None, None
            dx, dw = clamped_weighted_swiglu_back(grad, input, weights, ctx.limit)
        else:
            from .weighted_swiglu_kernels import clamped_swiglu_backward

            dx, dw = clamped_swiglu_backward(grad, input, weights, ctx.limit)
            if dw is not None:
                # Core rounds each row reduction before Autograd reduces broadcast dimensions.
                dw = dw.to(torch.promote_types(input.dtype, weights.dtype)).to(weights.dtype)
        if dw is not None:
            dw = dw.sum_to_size(weights.shape)
        return dx, dw, None


def bias_swiglu_impl(
    input: torch.Tensor,
    bias: torch.Tensor | None,
    fp8_input_store: bool = False,
    cpu_offload_input: bool = False,
    clamp_value: float | None = None,
) -> torch.Tensor:
    """Fuse Core clamped SwiGLU without a full FP32 input copy."""
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl as original

    if cpu_offload_input or not _clamped_supported(input, bias, fp8_input_store, clamp_value):
        return original.__wrapped__(input, bias, fp8_input_store, cpu_offload_input, clamp_value)
    return _ClampedSwiGLU.apply(input.contiguous(), None, clamp_value)


def weighted_bias_swiglu_impl(
    input: torch.Tensor,
    bias: torch.Tensor | None,
    weights: torch.Tensor,
    fp8_input_store: bool = False,
    clamp_value: float | None = None,
) -> torch.Tensor:
    """Fuse Core clamped SwiGLU and per-token probability multiplication/reduction."""
    from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl as original

    row_shape = (input.shape[:-1].numel(), 1)
    if not _clamped_supported(input, bias, fp8_input_store, clamp_value) or not (
        weights.device == input.device
        and weights.layout == torch.strided
        and weights.dtype in _FLOAT_DTYPES
        and weights.ndim <= len(row_shape)
        and all(
            w == 1 or w == size for w, size in zip(reversed(weights.shape), reversed(row_shape))
        )
    ):
        return original.__wrapped__(input, bias, weights, fp8_input_store, clamp_value)
    # Core flattens token dimensions before applying per-token probabilities.
    output = _ClampedSwiGLU.apply(
        input.contiguous().view(-1, input.shape[-1]), weights, clamp_value
    )
    return output.view(*input.shape[:-1], input.shape[-1] // 2)
