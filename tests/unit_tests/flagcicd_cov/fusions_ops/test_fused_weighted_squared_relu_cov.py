# Copyright (c) 2026, BAAI. All rights reserved.
"""Weighted squared-ReLU: per-token weighting of ReLU(x)^2 and grads w.r.t. input and weights."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_weighted_squared_relu


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _squared_relu_ref(x):
    return torch.pow(F.relu(x), 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_weighted_squared_relu_scales_activation_and_restores_input_dtype(dtype):
    torch.manual_seed(1234)
    x = torch.randn(8, 16, dtype=dtype, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_weighted_squared_relu.weighted_squared_relu)(x, weights)

    assert out.dtype == dtype
    expected = (_squared_relu_ref(x.double()) * weights.double()).to(dtype)
    tols = (
        dict(rtol=1.0e-5, atol=1.0e-5) if dtype == torch.float32 else dict(rtol=1.6e-2, atol=1.0e-2)
    )
    torch.testing.assert_close(out.double(), expected.double(), **tols)


def test_squared_relu_back_returns_two_relu_times_upstream_grad():
    torch.manual_seed(2345)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    x = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_weighted_squared_relu._squared_relu_back)(g, x)

    torch.testing.assert_close(grad, g * 2 * F.relu(x), rtol=1.0e-5, atol=1.0e-5)


def test_weighted_squared_relu_back_returns_grads_for_input_and_weights():
    torch.manual_seed(3456)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    x = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    input_grad, weights_grad = _eager(fused_weighted_squared_relu.weighted_squared_relu_back)(
        g, x, weights
    )

    x_ref = x.detach().requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    (_squared_relu_ref(x_ref) * w_ref).backward(g)
    torch.testing.assert_close(input_grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights_grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    assert weights_grad.shape == (8, 1)


def test_weighted_squared_relu_impl_forward_and_backward_match_reference_for_batched_input():
    torch.manual_seed(4567)
    input = torch.randn(2, 3, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    weights = torch.rand(6, 1, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(2, 3, 16, dtype=torch.float32, device=DEVICE)

    out = fused_weighted_squared_relu.weighted_squared_relu_impl(input, weights)
    out.backward(grad_output)

    assert out.shape == (2, 3, 16)
    x_ref = input.detach().reshape(6, 16).requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    ref_out = (_squared_relu_ref(x_ref) * w_ref).reshape(2, 3, 16)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad.reshape(6, 16), x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights.grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_squared_relu_public_call_matches_squared_relu_reference():
    torch.manual_seed(5679)
    x = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    out = fused_weighted_squared_relu.weighted_squared_relu(x, weights)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _squared_relu_ref(x) * weights, rtol=1.0e-5, atol=1.0e-5)
