# Copyright (c) 2026, BAAI. All rights reserved.
"""mHC fused kernels runnable on SM80: forward/backward parity against unfused references."""

import math

import pytest
import torch

import megatron.core.fusions.fused_mhc_kernels as mhc

BF16_RTOL = 1.6e-2
BF16_ATOL = 1e-2
FP32_RTOL = 1e-5
FP32_ATOL = 1e-5


def _uniform(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="cuda").uniform_(-1.0, 1.0)


def _sinkhorn_reference(logits, num_iterations, eps):
    matrix = logits.softmax(dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def _h_aggregate_reference(x, h_pre):
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _h_post_bda_reference(h_res, original_residual, h_post, x, bias):
    mixed = torch.einsum("sbji,sbjc->sbic", h_res, original_residual)
    output = h_post.unsqueeze(-1) * x.unsqueeze(2) + mixed
    if bias is not None:
        output = output + h_post.unsqueeze(-1) * bias
    return output


def _proj_rms_compute_h_reference(
    x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, compute_h_eps
):
    projection = torch.matmul(x, weight.t())
    r = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    alpha = torch.cat(
        [alpha_pre.expand(n), alpha_post.expand(n), alpha_res.expand(weight.shape[0] - 2 * n)]
    )
    h = projection * alpha.unsqueeze(0) / (r + eps) + bias
    return (
        h[..., :n].sigmoid() + compute_h_eps,
        h[..., n : 2 * n].sigmoid() * 2,
        h[..., 2 * n :],
        r,
    )


@pytest.mark.parametrize("num_iterations", [1, 5], ids=["one-iteration", "five-iterations"])
def test_triton_sinkhorn_backward_matches_reference_gradient_when_iterations_vary(num_iterations):
    torch.manual_seed(1234)
    logits = _uniform(2, 3, 4, 4, dtype=torch.float32)
    grad_output = _uniform(2, 3, 4, 4, dtype=torch.float32)
    eps = 1e-6
    _, m_init = mhc._triton_sinkhorn_fwd(logits, num_iterations, eps)
    grad_input = mhc._triton_sinkhorn_bwd(grad_output, m_init, num_iterations, eps)

    reference_input = logits.double().requires_grad_(True)
    _sinkhorn_reference(reference_input, num_iterations, eps).backward(grad_output.double())
    torch.testing.assert_close(
        grad_input.double(), reference_input.grad, rtol=FP32_RTOL, atol=FP32_ATOL
    )


def test_fused_sinkhorn_auto_policy_autograd_matches_reference_with_large_eps():
    torch.manual_seed(1235)
    logits = _uniform(1, 2, 4, 4).requires_grad_(True)
    grad_output = _uniform(1, 2, 4, 4)
    num_iterations = 5
    eps = 5e-2
    output = mhc.fused_sinkhorn(logits, num_iterations, eps)
    output.backward(grad_output)

    reference_input = logits.detach().double().requires_grad_(True)
    reference_output = _sinkhorn_reference(reference_input, num_iterations, eps)
    reference_output.backward(grad_output.double())
    torch.testing.assert_close(output.double(), reference_output, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(
        logits.grad.double(), reference_input.grad, rtol=BF16_RTOL, atol=BF16_ATOL
    )


def test_fused_sinkhorn_native_policy_matches_reference_forward_and_backward():
    torch.manual_seed(1240)
    logits = _uniform(2, 3, 4, 4, dtype=torch.float32).requires_grad_(True)
    grad_output = _uniform(2, 3, 4, 4, dtype=torch.float32)
    output = mhc.fused_sinkhorn(logits, 5, 1e-6, backend="native")
    output.backward(grad_output)

    reference_input = logits.detach().double().requires_grad_(True)
    reference_output = _sinkhorn_reference(reference_input, 5, 1e-6)
    reference_output.backward(grad_output.double())
    torch.testing.assert_close(output.double(), reference_output, rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(
        logits.grad.double(), reference_input.grad, rtol=FP32_RTOL, atol=FP32_ATOL
    )


def test_fused_h_aggregate_auto_policy_autograd_matches_reference():
    torch.manual_seed(1236)
    x = _uniform(2, 3, 4, 64).requires_grad_(True)
    h_pre = _uniform(2, 3, 4).requires_grad_(True)
    grad_output = _uniform(2, 3, 64)
    output = mhc.fused_h_aggregate(x, h_pre)
    output.backward(grad_output)

    x_reference = x.detach().double().requires_grad_(True)
    h_reference = h_pre.detach().double().requires_grad_(True)
    reference_output = _h_aggregate_reference(x_reference, h_reference)
    reference_output.backward(grad_output.double())
    torch.testing.assert_close(output.double(), reference_output, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(x.grad.double(), x_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(
        h_pre.grad.double(), h_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL
    )


def test_torch_h_aggregate_bwd_matches_reference_and_preserves_input_dtypes():
    torch.manual_seed(1237)
    grad_output = _uniform(2, 3, 64)
    x = _uniform(2, 3, 4, 64)
    h_pre = _uniform(2, 3, 4)
    grad_x, grad_h = mhc._torch_h_aggregate_bwd(grad_output, x, h_pre)

    x_reference = x.double().requires_grad_(True)
    h_reference = h_pre.double().requires_grad_(True)
    _h_aggregate_reference(x_reference, h_reference).backward(grad_output.double())
    assert grad_x.dtype == torch.bfloat16
    assert grad_h.dtype == torch.bfloat16
    torch.testing.assert_close(grad_x.double(), x_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(grad_h.double(), h_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_fused_h_aggregate_native_policy_matches_reference_forward_and_backward():
    torch.manual_seed(1241)
    x = _uniform(2, 3, 4, 64).requires_grad_(True)
    h_pre = _uniform(2, 3, 4).requires_grad_(True)
    grad_output = _uniform(2, 3, 64)
    output = mhc.fused_h_aggregate(x, h_pre, backend="native")
    output.backward(grad_output)

    x_reference = x.detach().double().requires_grad_(True)
    h_reference = h_pre.detach().double().requires_grad_(True)
    reference_output = _h_aggregate_reference(x_reference, h_reference)
    reference_output.backward(grad_output.double())
    torch.testing.assert_close(output.double(), reference_output, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(x.grad.double(), x_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(
        h_pre.grad.double(), h_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL
    )


@pytest.mark.parametrize("has_bias", [True, False], ids=["with-bias", "without-bias"])
def test_fused_h_post_bda_auto_policy_autograd_matches_reference_forward_and_backward(has_bias):
    torch.manual_seed(1238)
    s, b, n, c = 2, 2, 4, 64
    h_res = _uniform(s, b, n, n).requires_grad_(True)
    original_residual = _uniform(s, b, n, c).requires_grad_(True)
    h_post = _uniform(s, b, n).requires_grad_(True)
    x = _uniform(s, b, c).requires_grad_(True)
    bias = _uniform(c).requires_grad_(True) if has_bias else None
    grad_output = _uniform(s, b, n, c)
    output = mhc.fused_h_post_bda(h_res, original_residual, h_post, x, bias)
    output.backward(grad_output)

    h_res_reference = h_res.detach().double().requires_grad_(True)
    residual_reference = original_residual.detach().double().requires_grad_(True)
    h_post_reference = h_post.detach().double().requires_grad_(True)
    x_reference = x.detach().double().requires_grad_(True)
    bias_reference = bias.detach().double().requires_grad_(True) if has_bias else None
    reference_output = _h_post_bda_reference(
        h_res_reference, residual_reference, h_post_reference, x_reference, bias_reference
    )
    reference_output.backward(grad_output.double())
    torch.testing.assert_close(output.double(), reference_output, rtol=BF16_RTOL, atol=BF16_ATOL)
    for actual, expected in (
        (h_res.grad, h_res_reference.grad),
        (original_residual.grad, residual_reference.grad),
        (h_post.grad, h_post_reference.grad),
        (x.grad, x_reference.grad),
    ):
        torch.testing.assert_close(actual.double(), expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    if has_bias:
        torch.testing.assert_close(
            bias.grad.double(), bias_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )


@pytest.mark.parametrize("has_bias", [True, False], ids=["with-bias", "without-bias"])
def test_torch_h_post_bda_bwd_matches_reference_for_every_gradient(has_bias):
    torch.manual_seed(1239)
    s, b, n, c = 2, 2, 4, 64
    grad_output = _uniform(s, b, n, c)
    h_res = _uniform(s, b, n, n)
    original_residual = _uniform(s, b, n, c)
    h_post = _uniform(s, b, n)
    x = _uniform(s, b, c)
    bias = _uniform(c) if has_bias else None
    g_hr, g_res, g_hp, g_x, g_bias = mhc._torch_h_post_bda_bwd(
        grad_output, h_res, original_residual, h_post, x, bias
    )

    h_res_reference = h_res.double().requires_grad_(True)
    residual_reference = original_residual.double().requires_grad_(True)
    h_post_reference = h_post.double().requires_grad_(True)
    x_reference = x.double().requires_grad_(True)
    bias_reference = bias.double().requires_grad_(True) if has_bias else None
    _h_post_bda_reference(
        h_res_reference, residual_reference, h_post_reference, x_reference, bias_reference
    ).backward(grad_output.double())
    assert g_hr.dtype == h_res.dtype
    assert g_res.dtype == original_residual.dtype
    assert g_hp.dtype == h_post.dtype
    assert g_x.dtype == x.dtype
    for actual, expected in (
        (g_hr, h_res_reference.grad),
        (g_res, residual_reference.grad),
        (g_hp, h_post_reference.grad),
        (g_x, x_reference.grad),
    ):
        torch.testing.assert_close(actual.double(), expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    if has_bias:
        assert g_bias.dtype == bias.dtype
        torch.testing.assert_close(
            g_bias.double(), bias_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )
    else:
        assert g_bias is None


def test_fused_h_post_bda_native_policy_matches_reference_forward_and_backward():
    torch.manual_seed(1242)
    s, b, n, c = 2, 2, 4, 64
    h_res = _uniform(s, b, n, n).requires_grad_(True)
    original_residual = _uniform(s, b, n, c).requires_grad_(True)
    h_post = _uniform(s, b, n).requires_grad_(True)
    x = _uniform(s, b, c).requires_grad_(True)
    bias = _uniform(c).requires_grad_(True)
    grad_output = _uniform(s, b, n, c)
    output = mhc.fused_h_post_bda(h_res, original_residual, h_post, x, bias, backend="native")
    output.backward(grad_output)

    h_res_reference = h_res.detach().double().requires_grad_(True)
    residual_reference = original_residual.detach().double().requires_grad_(True)
    h_post_reference = h_post.detach().double().requires_grad_(True)
    x_reference = x.detach().double().requires_grad_(True)
    bias_reference = bias.detach().double().requires_grad_(True)
    reference_output = _h_post_bda_reference(
        h_res_reference, residual_reference, h_post_reference, x_reference, bias_reference
    )
    reference_output.backward(grad_output.double())
    torch.testing.assert_close(output.double(), reference_output, rtol=BF16_RTOL, atol=BF16_ATOL)
    for actual, expected in (
        (h_res.grad, h_res_reference.grad),
        (original_residual.grad, residual_reference.grad),
        (h_post.grad, h_post_reference.grad),
        (x.grad, x_reference.grad),
        (bias.grad, bias_reference.grad),
    ):
        torch.testing.assert_close(actual.double(), expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_fused_proj_rms_compute_h_native_policy_matches_reference_forward_and_backward():
    torch.manual_seed(1244)
    m, n, k = 16, 4, 256
    n_out = n * n + 2 * n
    eps = 1e-6
    compute_h_eps = 1e-6
    x = _uniform(m, k)
    weight = _uniform(n_out, k, dtype=torch.float32)
    alpha_pre = _uniform(1, dtype=torch.float32)
    alpha_post = _uniform(1, dtype=torch.float32)
    alpha_res = _uniform(1, dtype=torch.float32)
    bias = _uniform(n_out, dtype=torch.float32)
    grad_h_pre = _uniform(m, n, dtype=torch.float32)
    grad_h_post = _uniform(m, n, dtype=torch.float32)
    grad_h_res = _uniform(m, n_out - 2 * n, dtype=torch.float32)
    grad_r = _uniform(m, 1, dtype=torch.float32)

    x_fused = x.detach().clone().requires_grad_(True)
    weight_fused = weight.detach().clone().requires_grad_(True)
    alpha_pre_fused = alpha_pre.detach().clone().requires_grad_(True)
    alpha_post_fused = alpha_post.detach().clone().requires_grad_(True)
    alpha_res_fused = alpha_res.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True)
    h_pre, h_post, h_res, r = mhc.fused_proj_rms_compute_h(
        x_fused,
        weight_fused,
        alpha_pre_fused,
        alpha_post_fused,
        alpha_res_fused,
        bias_fused,
        n,
        eps,
        compute_h_eps,
        backend="native",
    )
    loss = (
        (h_pre * grad_h_pre).sum()
        + (h_post * grad_h_post).sum()
        + (h_res * grad_h_res).sum()
        + (r * grad_r).sum()
    )
    loss.backward()

    x_reference = x.double().requires_grad_(True)
    weight_reference = weight.double().requires_grad_(True)
    alpha_pre_reference = alpha_pre.double().requires_grad_(True)
    alpha_post_reference = alpha_post.double().requires_grad_(True)
    alpha_res_reference = alpha_res.double().requires_grad_(True)
    bias_reference = bias.double().requires_grad_(True)
    reference_h_pre, reference_h_post, reference_h_res, reference_r = _proj_rms_compute_h_reference(
        x_reference,
        weight_reference,
        alpha_pre_reference,
        alpha_post_reference,
        alpha_res_reference,
        bias_reference,
        n,
        eps,
        compute_h_eps,
    )
    reference_loss = (
        (reference_h_pre * grad_h_pre.double()).sum()
        + (reference_h_post * grad_h_post.double()).sum()
        + (reference_h_res * grad_h_res.double()).sum()
        + (reference_r * grad_r.double()).sum()
    )
    reference_loss.backward()
    assert h_pre.dtype == torch.float32
    for actual, expected in (
        (h_pre, reference_h_pre),
        (h_post, reference_h_post),
        (h_res, reference_h_res),
        (r, reference_r),
    ):
        torch.testing.assert_close(actual.double(), expected, rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(
        x_fused.grad.double(), x_reference.grad, rtol=BF16_RTOL, atol=BF16_ATOL
    )
    for actual, expected in (
        (weight_fused.grad, weight_reference.grad),
        (alpha_pre_fused.grad, alpha_pre_reference.grad),
        (alpha_post_fused.grad, alpha_post_reference.grad),
        (alpha_res_fused.grad, alpha_res_reference.grad),
        (bias_fused.grad, bias_reference.grad),
    ):
        torch.testing.assert_close(actual.double(), expected, rtol=FP32_RTOL, atol=FP32_ATOL)


def test_fused_add_3_sums_three_tensors_and_passes_gradients_through():
    torch.manual_seed(1243)
    a = _uniform(4, 32).requires_grad_(True)
    b = _uniform(4, 32).requires_grad_(True)
    c = _uniform(4, 32).requires_grad_(True)
    grad_output = _uniform(4, 32)
    result = mhc.fused_add_3(a, b, c)
    result.backward(grad_output)

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result.double(), a.double() + b.double() + c.double(), rtol=BF16_RTOL, atol=BF16_ATOL
    )
    for tensor in (a, b, c):
        assert torch.equal(tensor.grad, grad_output)
