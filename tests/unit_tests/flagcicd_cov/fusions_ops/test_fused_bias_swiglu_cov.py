# Copyright (c) 2026, BAAI. All rights reserved.
"""SwiGLU: SiLU gating, bias add, token weighting, clamped variants, and their backward grads."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_bias_swiglu


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _swiglu_ref(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


def _clamped_swiglu_ref(y, clamp_value):
    y_1, y_2 = torch.chunk(y.to(torch.float32), 2, -1)
    y_1 = y_1.clamp(min=None, max=clamp_value)
    y_2 = y_2.clamp(min=-clamp_value, max=clamp_value)
    return (F.silu(y_1) * y_2).to(y.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_swiglu_gates_first_half_with_silu_when_input_is_split(dtype):
    torch.manual_seed(1234)
    y = torch.randn(8, 32, dtype=dtype, device=DEVICE)

    out = _eager(fused_bias_swiglu.swiglu)(y)

    tols = (
        dict(rtol=1.0e-5, atol=1.0e-5) if dtype == torch.float32 else dict(rtol=1.6e-2, atol=1.0e-2)
    )
    torch.testing.assert_close(out.double(), _swiglu_ref(y.double()), **tols)


def test_bias_swiglu_adds_bias_before_gating_when_bias_is_provided():
    torch.manual_seed(2345)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_swiglu.bias_swiglu)(y, bias)

    torch.testing.assert_close(out, _swiglu_ref(y + bias), rtol=1.0e-5, atol=1.0e-5)


def test_weighted_swiglu_scales_tokens_and_restores_input_dtype():
    torch.manual_seed(3456)
    y = torch.randn(8, 32, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_swiglu.weighted_swiglu)(y, weights)

    assert out.dtype == torch.bfloat16
    expected = (_swiglu_ref(y.double()) * weights.double()).to(torch.bfloat16)
    torch.testing.assert_close(out.double(), expected.double(), rtol=1.6e-2, atol=1.0e-2)


def test_clamped_swiglu_clamps_gate_and_linear_halves_when_clamp_value_is_set():
    torch.manual_seed(4567)
    clamp_value = 1.0
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0

    out = _eager(fused_bias_swiglu.clamped_swiglu)(y, clamp_value)

    assert y.abs().max().item() > clamp_value
    torch.testing.assert_close(out, _clamped_swiglu_ref(y, clamp_value), rtol=1.0e-5, atol=1.0e-5)


def test_clamped_swiglu_restores_bfloat16_dtype_after_float32_compute():
    torch.manual_seed(5678)
    clamp_value = 1.0
    y = torch.randn(8, 32, dtype=torch.bfloat16, device=DEVICE)

    out = _eager(fused_bias_swiglu.clamped_swiglu)(y, clamp_value)

    assert out.dtype == torch.bfloat16
    expected = _clamped_swiglu_ref(y, clamp_value)
    torch.testing.assert_close(out.double(), expected.double(), rtol=1.6e-2, atol=1.0e-2)


def test_bias_clamped_swiglu_adds_bias_before_clamping():
    torch.manual_seed(6789)
    clamp_value = 1.0
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_swiglu.bias_clamped_swiglu)(y, bias, clamp_value)

    torch.testing.assert_close(
        out, _clamped_swiglu_ref(y + bias, clamp_value), rtol=1.0e-5, atol=1.0e-5
    )


def test_clamped_weighted_swiglu_scales_clamped_tokens_and_restores_dtype():
    torch.manual_seed(7890)
    clamp_value = 1.0
    y = torch.randn(8, 32, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_swiglu.clamped_weighted_swiglu)(y, weights, clamp_value)

    assert out.dtype == torch.bfloat16
    expected = (_clamped_swiglu_ref(y, clamp_value).double() * weights.double()).to(torch.bfloat16)
    torch.testing.assert_close(out.double(), expected.double(), rtol=1.6e-2, atol=1.0e-2)


def test_swiglu_back_matches_autograd_reference_for_both_halves():
    torch.manual_seed(8901)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_bias_swiglu.swiglu_back)(g, y)

    y_ref = y.detach().requires_grad_(True)
    _swiglu_ref(y_ref).backward(g)
    torch.testing.assert_close(grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_swiglu_back_matches_autograd_reference_after_bias_addition():
    torch.manual_seed(9012)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_bias_swiglu.bias_swiglu_back)(g, y, bias)

    x_ref = (y.detach() + bias.detach()).requires_grad_(True)
    _swiglu_ref(x_ref).backward(g)
    torch.testing.assert_close(grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_swiglu_back_returns_grads_for_input_and_weights():
    torch.manual_seed(1122)
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    input_grad, weights_grad = _eager(fused_bias_swiglu.weighted_swiglu_back)(g, y, weights)

    y_ref = y.detach().requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    (_swiglu_ref(y_ref) * w_ref).backward(g)
    torch.testing.assert_close(input_grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights_grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    assert weights_grad.shape == (8, 1)


def test_clamped_swiglu_back_masks_grads_where_clamp_is_active():
    torch.manual_seed(2233)
    clamp_value = 1.0
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0

    grad = _eager(fused_bias_swiglu.clamped_swiglu_back)(g, y, clamp_value)

    y_ref = y.detach().requires_grad_(True)
    _clamped_swiglu_ref(y_ref, clamp_value).backward(g)
    torch.testing.assert_close(grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    assert (y[:, :16] > clamp_value).any()
    assert (y[:, 16:].abs() > clamp_value).any()


def test_bias_clamped_swiglu_back_matches_reference_after_bias_addition():
    torch.manual_seed(3344)
    clamp_value = 1.0
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    grad = _eager(fused_bias_swiglu.bias_clamped_swiglu_back)(g, y, bias, clamp_value)

    x_ref = (y.detach() + bias.detach()).requires_grad_(True)
    _clamped_swiglu_ref(x_ref, clamp_value).backward(g)
    torch.testing.assert_close(grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_clamped_weighted_swiglu_back_returns_grads_for_input_and_weights():
    torch.manual_seed(4455)
    clamp_value = 1.0
    g = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    input_grad, weights_grad = _eager(fused_bias_swiglu.clamped_weighted_swiglu_back)(
        g, y, weights, clamp_value
    )

    y_ref = y.detach().requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    (_clamped_swiglu_ref(y_ref, clamp_value) * w_ref).backward(g)
    torch.testing.assert_close(input_grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights_grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_swiglu_impl_forward_and_backward_match_reference_with_bias():
    torch.manual_seed(5566)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.bias_swiglu_impl(input, bias)
    out.backward(grad_output)

    x_ref = input.detach().requires_grad_(True)
    b_ref = bias.detach().requires_grad_(True)
    ref_out = _swiglu_ref(x_ref + b_ref)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias.grad, b_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, bias.grad, rtol=0.0, atol=0.0)


def test_bias_swiglu_impl_with_clamp_matches_reference_for_forward_and_backward():
    torch.manual_seed(6677)
    clamp_value = 1.0
    input = (torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0).requires_grad_(True)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.bias_swiglu_impl(input, bias, clamp_value=clamp_value)
    out.backward(grad_output)

    x_ref = input.detach().requires_grad_(True)
    b_ref = bias.detach().requires_grad_(True)
    ref_out = _clamped_swiglu_ref(x_ref + b_ref, clamp_value)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias.grad, b_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_swiglu_impl_without_bias_matches_reference_for_forward_and_backward():
    torch.manual_seed(7788)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.bias_swiglu_impl(input, None)
    out.backward(grad_output)

    x_ref = input.detach().requires_grad_(True)
    ref_out = _swiglu_ref(x_ref)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_swiglu_impl_without_bias_and_with_clamp_matches_reference():
    torch.manual_seed(8899)
    clamp_value = 1.0
    input = (torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0).requires_grad_(True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.bias_swiglu_impl(input, None, clamp_value=clamp_value)
    out.backward(grad_output)

    x_ref = input.detach().requires_grad_(True)
    ref_out = _clamped_swiglu_ref(x_ref, clamp_value)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("has_bias", [True, False], ids=["with_bias", "without_bias"])
def test_bias_swiglu_impl_offload_flag_keeps_numerics_unchanged(has_bias):
    torch.manual_seed(9900)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE) if has_bias else None
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.bias_swiglu_impl(input, bias, cpu_offload_input=True)
    out.backward(grad_output)

    x_ref = input.detach().requires_grad_(True)
    ref_out = _swiglu_ref(x_ref + (bias if has_bias else 0.0))
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_bias_swiglu_impl_fp8_input_store_backward_uses_dequantized_input():
    torch.manual_seed(1011)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
    input_fused = input.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True)

    out = fused_bias_swiglu.bias_swiglu_impl(input_fused, bias_fused, fp8_input_store=True)
    out.backward(grad_output)

    stored = input.to(torch.float8_e4m3fn).to(torch.float32)
    x_ref = stored.requires_grad_(True)
    b_ref = bias.detach().requires_grad_(True)
    ref_out = _swiglu_ref(x_ref + b_ref)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, _swiglu_ref(input + bias), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input_fused.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(bias_fused.grad, b_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_swiglu_impl_matches_reference_for_forward_and_backward():
    torch.manual_seed(1213)
    input = torch.randn(2, 3, 32, dtype=torch.float32, device=DEVICE, requires_grad=True)
    weights = torch.rand(6, 1, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(2, 3, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.weighted_bias_swiglu_impl(input, None, weights)
    out.backward(grad_output)

    assert out.shape == (2, 3, 16)
    y_ref = input.detach().reshape(6, 32).requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    ref_out = (_swiglu_ref(y_ref) * w_ref).reshape(2, 3, 16)
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad.reshape(6, 32), y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights.grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_swiglu_impl_with_clamp_matches_reference_for_backward():
    torch.manual_seed(1415)
    clamp_value = 1.0
    input = (torch.randn(8, 32, dtype=torch.float32, device=DEVICE) * 3.0).requires_grad_(True)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.weighted_bias_swiglu_impl(input, None, weights, clamp_value=clamp_value)
    out.backward(grad_output)

    y_ref = input.detach().requires_grad_(True)
    w_ref = weights.detach().requires_grad_(True)
    ref_out = _clamped_swiglu_ref(y_ref, clamp_value) * w_ref
    ref_out.backward(grad_output)

    torch.testing.assert_close(out, ref_out.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(input.grad, y_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(weights.grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_weighted_bias_swiglu_impl_raises_not_implemented_error_when_bias_is_provided():
    torch.manual_seed(1617)
    input = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)
    weights = torch.rand(8, 1, dtype=torch.float32, device=DEVICE)

    with pytest.raises(NotImplementedError, match="Bias is not supported for weighted swiglu"):
        fused_bias_swiglu.weighted_bias_swiglu_impl(input, bias, weights)


def test_swiglu_public_call_matches_silu_gated_reference():
    torch.manual_seed(1819)
    y = torch.randn(8, 32, dtype=torch.float32, device=DEVICE)

    out = fused_bias_swiglu.swiglu(y)

    torch.testing.assert_close(out, _swiglu_ref(y), rtol=1.0e-5, atol=1.0e-5)
