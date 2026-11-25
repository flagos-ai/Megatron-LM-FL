# Copyright 2026 FlagOS Contributors
"""Fuse GDN's post-normalization gate without materializing FP32 activations.

The existing normalization module still owns its hooks, saved statistics and
parameter gradients. The gate uses FP32 arithmetic with the original output dtype.
"""
from importlib.util import find_spec

import torch
import torch.nn.functional as F

_HAS_TRITON = find_spec("triton") is not None


class _GatedSiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized, gate, output_dtype):
        from .gated_norm_kernels import gate_forward

        output = gate_forward(normalized, gate, output_dtype)
        ctx.save_for_backward(normalized, gate)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        normalized, gate = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Keep higher-order requests differentiable; ordinary training uses
            # the fused first-order backward below.
            g = gate.float()
            sigmoid = torch.sigmoid(g)
            dy = grad_output.float() * F.silu(g)
            dg = (grad_output.float() * normalized.float()) * (sigmoid * (1 + g * (1 - sigmoid)))
            return dy.to(normalized.dtype), dg.to(gate.dtype), None

        from .gated_norm_kernels import gate_backward

        dy, dg = gate_backward(normalized, gate, grad_output.contiguous())
        return dy, dg, None


def apply_gated_norm(self, x, gate):
    """Fuse SiLU gating; the override registry selects the NPU implementation."""
    if x.device.type != "npu" or not _HAS_TRITON or self.act_fn is not F.silu:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet

        return GatedDeltaNet._apply_gated_norm.__wrapped__(self, x, gate)

    normalized = self.out_norm(x.reshape(-1, x.shape[-1]))
    gate = gate.reshape(-1, gate.shape[-1])
    # These kernels compute in FP32; retain native FP64 promotion semantics.
    if normalized.dtype == torch.float64 or gate.dtype == torch.float64:
        return (normalized * self.act_fn(gate.float())).to(x.dtype)
    if normalized.shape != gate.shape:
        # Broadcast gradients must reduce in FP32 before casting to input dtype.
        normalized, gate = normalized.float(), gate.float()
    normalized, gate = torch.broadcast_tensors(normalized, gate)
    return _GatedSiLUMul.apply(normalized.contiguous(), gate.contiguous(), x.dtype)
