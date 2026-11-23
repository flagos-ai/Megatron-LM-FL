# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Terminal recompute retains both Jacobians and fused accumulation semantics."""
import pytest
import torch
from megatron.plugin.slideformer.kernels import _LinearBackwardOnly


@pytest.mark.parametrize('fused', [False, True])
def test_terminal_projection_gradients_and_accumulation(fused):
    torch.manual_seed(614)
    weight = torch.nn.Parameter(torch.randn(13, 17, dtype=torch.float64))
    ref_weight = torch.nn.Parameter(weight.detach().clone())
    if fused:
        weight.main_grad = torch.full_like(weight, float('nan'))
        weight.overwrite_main_grad = True
    for _ in range(2):
        x = torch.randn(3, 5, 17, dtype=torch.float64, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        incoming = torch.randn(3, 5, 13, dtype=torch.float64)
        # This is the incoming gradient from the original forward's residual;
        # the terminal value is deliberately unused in checkpoint recompute.
        _LinearBackwardOnly.apply(x, weight).backward(incoming)
        torch.nn.functional.linear(ref_x, ref_weight).backward(incoming)
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(weight.main_grad if fused else weight.grad, ref_weight.grad)
    if fused:
        assert weight.grad is None
        assert weight.grad_added_to_main_grad
        assert weight.overwrite_main_grad is False
