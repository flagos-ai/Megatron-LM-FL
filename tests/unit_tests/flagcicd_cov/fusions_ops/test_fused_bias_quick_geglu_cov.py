# Copyright (c) 2026, BAAI. All rights reserved.
"""Quick-GEGLU: sigmoid-GELU gating with linear offset, token weighting, FP8 store, clamping."""

import pytest
import torch

from megatron.core.fusions import fused_bias_geglu


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def _quick_geglu(y, linear_offset):
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return _quick_gelu(y_1) * (y_2 + linear_offset)


def test_quick_gelu_matches_sigmoid_approximation_reference():
    torch.manual_seed(1234)
    y = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_geglu.quick_gelu)(y)

    expected = y * torch.sigmoid(1.702 * y)
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("linear_offset", [0.0, 0.25], ids=["no_offset", "offset"])
def test_quick_geglu_gates_first_half_with_quick_gelu_when_input_is_split(linear_offset):
    torch.manual_seed(2345)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_geglu.quick_geglu)(y, linear_offset)

    torch.testing.assert_close(out, _quick_geglu(y, linear_offset), rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("linear_offset", [0.0, 0.25], ids=["no_offset", "offset"])
def test_quick_geglu_back_matches_autograd_reference_for_both_halves(linear_offset):
    torch.manual_seed(3456)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_bias_geglu.quick_geglu_back)(g, y, linear_offset)

    y_ref = y.detach().requires_grad_(True)
    _quick_geglu(y_ref, linear_offset).backward(g)
    torch.testing.assert_close(grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_quick_geglu_scales_tokens_and_restores_input_dtype():
    torch.manual_seed(4567)
    y = torch.randn(8, 32, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_geglu.weighted_quick_geglu)(y, weights)

    assert out.dtype == torch.bfloat16
    expected = (_quick_geglu(y.double(), 0.0) * weights.double()).to(torch.bfloat16)
    torch.testing.assert_close(out.double(), expected.double(), rtol=1.6e-2, atol=1.0e-2)


def test_weighted_quick_geglu_back_returns_grads_for_input_and_weights():
    torch.manual_seed(5678)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    input_grad, weights_grad = _eager(fused_bias_geglu.weighted_quick_geglu_back)(g, y, weights)

    y_ref = y.detach().requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    (_quick_geglu(y_ref, 0.0) * w_ref).backward(g)
    torch.testing.assert_close(input_grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights_grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    assert weights_grad.shape == (8, 1)


def test_weighted_bias_quick_geglu_adds_bias_before_weighted_gating():
    torch.manual_seed(6789)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_geglu.weighted_bias_quick_geglu)(y, bias, weights)

    expected = _quick_geglu(y + bias, 0.0) * weights
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_quick_geglu_back_returns_grads_for_input_bias_and_weights():
    torch.manual_seed(7890)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    input_grad, bias_grad, weights_grad = _eager(fused_bias_geglu.weighted_bias_quick_geglu_back)(
        g, y, bias, weights
    )

    y_ref = (y.detach() + bias.detach()).requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    (_quick_geglu(y_ref, 0.0) * w_ref).backward(g)
    torch.testing.assert_close(input_grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias_grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights_grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_quick_geglu_impl_forward_and_backward_match_reference_without_bias():
    torch.manual_seed(8901)
    input = torch.randn(2, 3, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    weights = torch.rand(6, 1, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(2, 3, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_geglu.weighted_bias_quick_geglu_impl(input, None, weights, linear_offset=0.5)
    out.backward(grad_output)

    assert out.shape == (2, 3, 16)
    y_ref = input.detach().reshape(6, 32).requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    ref_out = (_quick_geglu(y_ref, 0.5) * w_ref).reshape(2, 3, 16)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad.reshape(6, 32), y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights.grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_quick_geglu_impl_with_bias_and_fp8_store_matches_quantized_reference():
    torch.manual_seed(9012)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    input_fused = input.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True)

    out = fused_bias_geglu.weighted_bias_quick_geglu_impl(
        input_fused, bias_fused, weights_fused, fp8_input_store=True, linear_offset=0.25
    )
    out.backward(grad_output)

    stored = input.to(torch.float8_e4m3fn).to(torch.float32)
    x_ref = stored.requires_grad_(True)
    w_ref = weights.clone().requires_grad_(True)
    b_ref = bias.clone().requires_grad_(True)
    ref_out = _quick_geglu(x_ref + b_ref, 0.25) * w_ref
    ref_out.backward(grad_output)

    expected_out = _quick_geglu(input + bias, 0.25) * weights
    torch.testing.assert_close(out, expected_out, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input_fused.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias_fused.grad, b_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights_fused.grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_quick_geglu_impl_clamps_input_when_clamp_value_is_set():
    torch.manual_seed(1122)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)
    clamp_value = 1.0
    input_fused = input.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)

    out = fused_bias_geglu.weighted_bias_quick_geglu_impl(
        input_fused, None, weights_fused, linear_offset=0.0, clamp_value=clamp_value
    )

    x_glu, x_linear = input.chunk(2, -1)
    clamped = torch.cat(
        (x_glu.clamp(min=None, max=clamp_value), x_linear.clamp(min=-clamp_value, max=clamp_value)),
        -1,
    )
    expected = (_quick_geglu(clamped, 0.0) * weights).to(torch.float32)
    assert input.abs().max().item() > clamp_value
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


def test_quick_geglu_public_call_matches_quick_gelu_reference_with_offset():
    torch.manual_seed(2024)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = fused_bias_geglu.quick_geglu(y, 0.25)

    torch.testing.assert_close(out, _quick_geglu(y, 0.25), rtol=1.0e-5, atol=1.0e-5)
