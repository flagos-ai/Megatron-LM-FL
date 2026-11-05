# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

from importlib.util import find_spec

import torch
import torch.nn.functional as F

_HAS_TRITON = find_spec("triton") is not None


def _reference(x, weight, bias, activation):
    output = F.conv1d(
        x.transpose(1, 2).contiguous(),
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[-1] - 1,
        groups=x.shape[-1],
    )
    output = output[..., : x.shape[1]]
    if activation:
        output = F.silu(output)
    return output.transpose(1, 2)


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, silu):
        from .causal_conv1d_kernels import forward

        output, linear = forward(x, weight, bias, silu, save_linear=silu)
        ctx.save_for_backward(x, weight, bias, linear)
        ctx.silu = silu
        return output

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias, linear = ctx.saved_tensors
        if torch.is_grad_enabled():
            inputs = [v for v in (x, weight, bias) if v is not None and v.requires_grad]
            output = _reference(x, weight, bias, ctx.silu)
            grads = iter(torch.autograd.grad(output, inputs, grad, create_graph=True))
            return tuple(
                next(grads) if v is not None and v.requires_grad else None
                for v in (x, weight, bias)
            ) + (None,)
        from .causal_conv1d_kernels import backward, silu_backward

        dconv = silu_backward(linear, grad) if ctx.silu else grad.contiguous()
        dx, dw, db = backward(x, weight, dconv, *ctx.needs_input_grad[:3])
        return dx, dw, db, None


def causal_conv1d(
    x,
    weight=None,
    bias=None,
    residual=None,
    initial_state=None,
    output_final_state=False,
    activation=None,
    backend="triton",
    cu_seqlens=None,
    **kwargs,
):
    if (
        backend != "triton"
        or kwargs
        or x.device.type != "npu"
        or x.ndim != 3
        or weight is None
        or initial_state is not None
        or output_final_state
        or cu_seqlens is not None
        or activation not in (None, "silu", "swish")
    ):
        from fla.modules.convolution import causal_conv1d as original

        return original(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            backend=backend,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
    silu = activation in ("silu", "swish")
    if torch.is_autocast_enabled(x.device.type) and x.dtype != torch.float64:
        dtype = torch.get_autocast_dtype(x.device.type)
        x, weight, bias = (v.to(dtype) if v is not None else None for v in (x, weight, bias))
    if x.stride(-1) != 1:
        x = x.contiguous()
    if (
        not _HAS_TRITON
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not x.numel()
        or weight.ndim != 2
        or weight.shape[0] != x.shape[-1]
        or weight.shape[1] == 0
        or (bias is not None and bias.shape != (x.shape[-1],))
        or any(
            v is not None and (v.dtype != x.dtype or v.device != x.device) for v in (weight, bias)
        )
    ):
        output = _reference(x, weight, bias, silu)
    elif torch.is_grad_enabled() and any(
        v is not None and v.requires_grad for v in (x, weight, bias)
    ):
        output = _CausalConv1d.apply(x, weight, bias, silu)
    else:
        from .causal_conv1d_kernels import forward

        output = forward(x, weight, bias, silu)[0]
    if residual is not None:
        output = output + residual
    return output, None


def gated_delta_net_conv1d(self, **kwargs):
    return causal_conv1d(**kwargs)
