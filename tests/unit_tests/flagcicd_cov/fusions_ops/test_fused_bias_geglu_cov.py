# Copyright (c) 2026, BAAI. All rights reserved.
"""GEGLU: tanh-GELU gating of the first half, bias add, backward, and 2D/3D shape handling."""

import pytest
import torch

from megatron.core.fusions import fused_bias_geglu


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tanh_gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_geglu_gates_first_half_with_tanh_gelu_when_input_is_split(dtype):
    torch.manual_seed(1234)
    y = torch.randn(8, 32, dtype=dtype, device=DEVICE)

    out = _eager(fused_bias_geglu.geglu)(y)

    y_1, y_2 = torch.chunk(y.double(), 2, -1)
    expected = (_tanh_gelu(y_1) * y_2).to(dtype)
    tols = (
        dict(rtol=1.0e-5, atol=1.0e-5) if dtype == torch.float32 else dict(rtol=1.6e-2, atol=1.0e-2)
    )
    torch.testing.assert_close(out.double(), expected.double(), **tols)


def test_bias_geglu_adds_bias_before_gating_when_bias_is_provided():
    torch.manual_seed(2345)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_geglu.bias_geglu)(bias, y)

    y_1, y_2 = torch.chunk(bias + y, 2, -1)
    expected = _tanh_gelu(y_1) * y_2
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


def test_geglu_back_matches_autograd_reference_for_both_halves():
    torch.manual_seed(3456)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_bias_geglu.geglu_back)(g, y)

    y_ref = y.detach().requires_grad_(True)
    y_1, y_2 = torch.chunk(y_ref, 2, -1)
    (_tanh_gelu(y_1) * y_2).backward(g)
    torch.testing.assert_close(grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_geglu_back_matches_autograd_reference_after_bias_addition():
    torch.manual_seed(4567)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_bias_geglu.bias_geglu_back)(g, y, bias)

    y_ref = (y.detach() + bias.detach()).requires_grad_(True)
    y_1, y_2 = torch.chunk(y_ref, 2, -1)
    (_tanh_gelu(y_1) * y_2).backward(g)
    torch.testing.assert_close(grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_geglu_impl_forward_and_backward_match_reference_for_flat_input_with_bias():
    torch.manual_seed(5678)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_geglu.bias_geglu_impl(input, bias)
    out.backward(grad_output)

    y_ref = (input.detach() + bias.detach()).requires_grad_(True)
    y_1, y_2 = torch.chunk(y_ref, 2, -1)
    ref_out = _tanh_gelu(y_1) * y_2
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias.grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    assert torch.equal(input.grad, bias.grad)


@pytest.mark.parametrize("has_bias", [True, False], ids=["with_bias", "without_bias"])
def test_bias_geglu_impl_restores_three_dimensional_shape_when_input_is_batched(has_bias):
    torch.manual_seed(6789)
    input = torch.randn(2, 4, 32, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE) if has_bias else None

    out = fused_bias_geglu.bias_geglu_impl(input, bias)

    assert out.shape == (2, 4, 16)
    y_1, y_2 = torch.chunk(input.reshape(8, 32) + (bias if has_bias else 0.0), 2, -1)
    expected = (_tanh_gelu(y_1) * y_2).reshape(2, 4, 16)
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


def test_geglu_impl_forward_and_backward_match_reference_without_bias():
    torch.manual_seed(7890)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_geglu.bias_geglu_impl(input, None)
    out.backward(grad_output)

    y_ref = input.detach().requires_grad_(True)
    y_1, y_2 = torch.chunk(y_ref, 2, -1)
    ref_out = _tanh_gelu(y_1) * y_2
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_geglu_public_call_matches_tanh_gelu_reference():
    torch.manual_seed(2468)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = fused_bias_geglu.geglu(y)

    y_1, y_2 = torch.chunk(y, 2, -1)
    expected = _tanh_gelu(y_1) * y_2
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)
