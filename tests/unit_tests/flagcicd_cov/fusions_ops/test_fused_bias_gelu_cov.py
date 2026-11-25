# Copyright (c) 2026, BAAI. All rights reserved.
"""Tanh-approximation bias-GELU: forward values, backward gradient, and the autograd wrapper."""

import pytest
import torch

from megatron.core.fusions import fused_bias_gelu


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_bias_gelu_matches_tanh_gelu_reference_when_bias_is_added(dtype):
    torch.manual_seed(1234)
    bias = torch.randn(16, dtype=dtype, device=DEVICE)
    y = torch.randn(8, 16, dtype=dtype, device=DEVICE)

    out = _eager(fused_bias_gelu.bias_gelu)(bias, y)

    expected = torch.nn.functional.gelu(bias.double() + y.double(), approximate="tanh").to(dtype)
    tols = (
        dict(rtol=1.0e-5, atol=1.0e-5) if dtype == torch.float32 else dict(rtol=1.6e-2, atol=1.0e-2)
    )
    torch.testing.assert_close(out.double(), expected.double(), **tols)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_bias_gelu_back_matches_autograd_reference_for_the_tanh_gelu(dtype):
    torch.manual_seed(2345)
    g = torch.randn(8, 16, dtype=dtype, device=DEVICE)
    bias = torch.randn(16, dtype=dtype, device=DEVICE)
    y = torch.randn(8, 16, dtype=dtype, device=DEVICE)

    grad = _eager(fused_bias_gelu.bias_gelu_back)(g, bias, y)

    x = (bias.double() + y.double()).requires_grad_(True)
    torch.nn.functional.gelu(x, approximate="tanh").backward(g.double())
    tols = (
        dict(rtol=1.0e-5, atol=1.0e-5) if dtype == torch.float32 else dict(rtol=1.6e-2, atol=1.0e-2)
    )
    torch.testing.assert_close(grad.double(), x.grad, **tols)


def test_bias_gelu_impl_forward_and_backward_grads_match_reference_when_inputs_require_grad():
    torch.manual_seed(3456)
    input = torch.randn(8, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    bias = torch.randn(8, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_gelu.bias_gelu_impl(input, bias)
    out.backward(grad_output)

    x_ref = input.detach().requires_grad_(True)
    b_ref = bias.detach().requires_grad_(True)
    ref_out = torch.nn.functional.gelu(b_ref + x_ref, approximate="tanh")
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias.grad, b_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, bias.grad, rtol=0.0, atol=0.0)


def test_bias_gelu_public_call_matches_tanh_gelu_reference():
    torch.manual_seed(3579)
    bias = torch.randn(16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_gelu.bias_gelu(bias, y)

    expected = torch.nn.functional.gelu(bias + y, approximate="tanh")
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)
