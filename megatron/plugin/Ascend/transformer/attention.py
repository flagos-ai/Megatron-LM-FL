# Copyright 2026 FlagOS Contributors
"""Fuse attention output gating without materializing full FP32 intermediates."""

from importlib.util import find_spec

import torch

_HAS_TRITON = find_spec("triton") is not None
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class _SigmoidGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        from ..ssm.gated_norm_kernels import gate_forward

        with torch.npu.device(x.device):
            output = gate_forward(x, gate, x.dtype, silu=False)
        ctx.save_for_backward(x, gate)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Recompute with differentiable operations for higher-order requests.
            sigmoid = torch.sigmoid(gate.float())
            grad = grad_output.float()
            dx = (grad * sigmoid).to(x.dtype) if ctx.needs_input_grad[0] else None
            dg = (
                (((grad * x.float()) * (1 - sigmoid)) * sigmoid).to(gate.dtype)
                if ctx.needs_input_grad[1]
                else None
            )
            return dx, dg

        from ..ssm.gated_norm_kernels import gate_backward

        with torch.npu.device(x.device):
            return gate_backward(
                x, gate, grad_output.contiguous(), silu=False, needs_input_grad=ctx.needs_input_grad
            )


def apply_output_gate(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply FP32 sigmoid gating and return the original activation dtype."""
    if (
        not _HAS_TRITON
        or x.device.type != "npu"
        or gate.device != x.device
        or not x.is_contiguous()
        or x.dtype not in _DTYPES
        or gate.dtype not in _DTYPES
    ):
        from megatron.core.transformer.attention import Attention

        return Attention._apply_output_gate.__wrapped__(self, x, gate)
    return _SigmoidGate.apply(x, gate.contiguous().view(x.shape))
