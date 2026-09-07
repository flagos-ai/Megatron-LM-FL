# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2024 LinkedIn Corporation
# SPDX-License-Identifier: BSD-2-Clause
#
# Modified by the SlideFormer Authors.

"""The pre-0.7 Liger Linear+CE accumulation path used by SlideFormer.

This module deliberately exposes an operation, not a model monkey patch.  The
Megatron integration supplies MCore hidden states, output weight and labels at
the stable GPTModel post-processing boundary.
"""

from __future__ import annotations

import inspect

import torch
import triton
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd, element_mul_kernel, is_hip

MAX_FUSED_SIZE = 65536 // 2
_CE_KERNEL_HAS_PREDICTED_TOKENS = (
    "predicted_tokens_ptr" in inspect.signature(liger_cross_entropy_kernel.fn).parameters
)


def _forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if reduction not in {"mean", "sum"}:
        raise ValueError("SlideFormer Legacy LCE supports reduction='mean' or 'sum'")

    token_count, hidden_size = inputs.shape
    vocab_size = weight.shape[0]
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
    increase_factor = triton.cdiv(vocab_size, hidden_size)
    chunk_size = triton.next_power_of_2(triton.cdiv(token_count, increase_factor))
    num_chunks = triton.cdiv(token_count, chunk_size)

    grad_input = torch.zeros_like(inputs)
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    losses = torch.zeros(token_count, dtype=torch.float32, device=inputs.device)

    non_ignored = (target != ignore_index).sum().item()
    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, token_count)
        input_chunk = inputs[start:end]
        target_chunk = target[start:end].contiguous()
        logits = input_chunk @ weight.t()
        if bias is not None:
            logits = logits + bias
        logits = logits.contiguous()
        loss_slice = losses[start:end]

        kernel_args = dict(
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=None,
            loss_ptr=loss_slice,
            z_loss_ptr=None,
            loss_stride=loss_slice.stride(-1),
            token_accuracy_ptr=None,
            token_accuracy_stride=0,
            n_cols=vocab_size,
            n_non_ignore=non_ignored,
            sum_non_ignore_weight=non_ignored,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=0.0,
            label_smoothing=0.0,
            reduction=reduction,
            softcap=None,
            RETURN_Z_LOSS=False,
            RETURN_TOKEN_ACCURACY=False,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=False,
            HAS_GRADIENTS=True,
            BLOCK_SIZE=block_size,
            num_warps=32 if not is_hip() else 16,
        )
        if _CE_KERNEL_HAS_PREDICTED_TOKENS:
            kernel_args.update(
                predicted_tokens_ptr=None, predicted_tokens_stride=0, RETURN_PREDICTED_TOKENS=False
            )
        liger_cross_entropy_kernel[(end - start,)](**kernel_args)

        losses[start:end] = loss_slice
        grad_input[start:end] = logits @ weight
        if grad_weight is not None:
            # Liger >=0.7 changed this to a temporary FP32 mm followed by add.
            # The original in-place addmm is materially faster for Qwen's head.
            torch.addmm(
                input=grad_weight,
                mat1=logits.t().to(input_chunk.dtype),
                mat2=input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )
        if grad_bias is not None:
            torch.add(input=grad_bias, other=logits.sum(dim=0), out=grad_bias, alpha=1.0)

    return losses.sum(), grad_input, grad_weight, grad_bias


def _scale_saved_gradient(gradient: torch.Tensor, grad_output: torch.Tensor) -> None:
    rows, columns = gradient.shape
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(columns))
    element_mul_kernel[(rows,)](
        gradient,
        gradient.stride(-2),
        grad_output,
        columns,
        BLOCK_SIZE=block_size,
        num_warps=32 if not is_hip() else 16,
    )


class LegacyFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        loss, grad_input, grad_weight, grad_bias = _forward(
            inputs, weight, target, bias=bias, ignore_index=ignore_index, reduction=reduction
        )
        ctx.save_for_backward(grad_input.detach(), grad_weight, grad_bias)
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if not torch.equal(grad_output, grad_output.new_tensor(1.0)):
            _scale_saved_gradient(grad_input, grad_output)
            if grad_weight is not None:
                _scale_saved_gradient(grad_weight, grad_output)
            if grad_bias is not None:
                _scale_saved_gradient(grad_bias.unsqueeze(-1), grad_output)
        return grad_input, grad_weight, None, grad_bias, None, None


class LegacyFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, *, ignore_index: int = -100, reduction: str = "sum") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("SlideFormer Legacy LCE supports reduction='mean' or 'sum'")
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        weight: torch.Tensor,
        inputs: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return LegacyFusedLinearCrossEntropyFunction.apply(
            inputs, weight, target, bias, self.ignore_index, self.reduction
        )
