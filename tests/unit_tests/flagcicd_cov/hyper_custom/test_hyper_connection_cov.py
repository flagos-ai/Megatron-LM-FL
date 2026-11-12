# Copyright (c) 2026, BAAI. All rights reserved.
"""mHC native ops: sinkhorn projection, aggregation, residual mixing, RMS, output gating."""

import math

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.hyper_connection as hyper_connection
from megatron.core.transformer.hyper_connection import (
    BroadcastTensorFused,
    learned_output_contract,
    native_fused_add_3,
    native_h_aggregate,
    native_h_post_bda,
    native_proj_rms,
    native_sinkhorn,
)


def _eager(fn):
    # torch.compile hides the decorated body from coverage tools; run the original bytecode.
    return torch.compiler.disable(fn)


def _sinkhorn_reference(logits, num_iterations, eps):
    reference = logits.softmax(dim=-1) + eps
    reference = reference / (reference.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        reference = reference / (reference.sum(dim=-1, keepdim=True) + eps)
        reference = reference / (reference.sum(dim=-2, keepdim=True) + eps)
    return reference


def test_sinkhorn_iterations_projects_to_doubly_stochastic_when_run_eagerly():
    torch.manual_seed(1234)
    logits = torch.randn(2, 3, 4, 4)
    matrix = _eager(hyper_connection._sinkhorn_iterations)(logits, 4, 1e-6)
    torch.testing.assert_close(matrix, _sinkhorn_reference(logits, 4, 1e-6), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(2, 3, 4), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(2, 3, 4), rtol=1e-3, atol=1e-3)


def test_native_sinkhorn_matches_iteration_reference_when_compiled_on_gpu():
    torch.manual_seed(1234)
    logits = torch.randn(2, 3, 4, 4, device="cuda")
    matrix = native_sinkhorn(logits, 4, 1e-6)
    torch.testing.assert_close(matrix, _sinkhorn_reference(logits, 4, 1e-6), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        matrix.sum(dim=-2), torch.ones_like(matrix.sum(dim=-2)), rtol=1e-3, atol=1e-3
    )


def test_native_sinkhorn_backward_matches_reference_when_backpropagated():
    torch.manual_seed(1234)
    logits = torch.randn(2, 3, 4, 4)
    grad_output = torch.randn(2, 3, 4, 4)
    num_iterations, eps = 3, 1e-6

    logits_bik = logits.clone().requires_grad_(True)
    native_sinkhorn(logits_bik, num_iterations, eps).backward(grad_output)

    logits_ref = logits.clone().requires_grad_(True)
    _sinkhorn_reference(logits_ref, num_iterations, eps).backward(grad_output)

    torch.testing.assert_close(logits_bik.grad, logits_ref.grad, rtol=1e-5, atol=1e-8)


def test_native_h_aggregate_weights_streams_when_run_eagerly():
    torch.manual_seed(1234)
    s, b, n, C = 3, 2, 4, 5
    x = torch.randn(s, b, n, C, requires_grad=True)
    h_pre = torch.rand(s, b, n, requires_grad=True)

    aggregated = _eager(native_h_aggregate)(x, h_pre)

    torch.testing.assert_close(aggregated, torch.einsum("sbnj,sbn->sbj", x, h_pre))
    grad = torch.randn_like(aggregated)
    aggregated.backward(grad)
    x_ref = x.detach().clone().requires_grad_(True)
    h_pre_ref = h_pre.detach().clone().requires_grad_(True)
    torch.einsum("sbnj,sbn->sbj", x_ref, h_pre_ref).backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(h_pre.grad, h_pre_ref.grad, rtol=1e-6, atol=1e-7)


def test_native_h_aggregate_matches_einsum_reference_when_compiled_on_gpu():
    torch.manual_seed(1234)
    s, b, n, C = 3, 2, 4, 5
    x = torch.randn(s, b, n, C, device="cuda")
    h_pre = torch.rand(s, b, n, device="cuda")
    torch.testing.assert_close(
        native_h_aggregate(x, h_pre), torch.einsum("sbnj,sbn->sbj", x, h_pre), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_native_h_post_bda_mixes_residual_and_output_when_run_eagerly(with_bias):
    torch.manual_seed(1234)
    s, b, n, C = 3, 2, 4, 5
    h_res = torch.randn(s, b, n, n, requires_grad=True)
    residual = torch.randn(s, b, n, C, requires_grad=True)
    h_post = torch.rand(s, b, n, requires_grad=True)
    x = torch.randn(s, b, C, requires_grad=True)
    bias = torch.randn(C, requires_grad=True) if with_bias else None

    output = _eager(native_h_post_bda)(h_res, residual, h_post, x, bias)

    reference = torch.einsum("sbji,sbjc->sbic", h_res, residual) + h_post.unsqueeze(
        -1
    ) * x.unsqueeze(2)
    if with_bias:
        reference = reference + h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
    torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    grad = torch.randn_like(output)
    output.backward(grad)
    (h_res_ref, residual_ref, h_post_ref, x_ref) = (
        h_res.detach().clone().requires_grad_(True),
        residual.detach().clone().requires_grad_(True),
        h_post.detach().clone().requires_grad_(True),
        x.detach().clone().requires_grad_(True),
    )
    bias_ref = bias.detach().clone().requires_grad_(True) if with_bias else None
    reference_ref = torch.einsum("sbji,sbjc->sbic", h_res_ref, residual_ref) + h_post_ref.unsqueeze(
        -1
    ) * x_ref.unsqueeze(2)
    if with_bias:
        reference_ref = reference_ref + h_post_ref.unsqueeze(-1) * bias_ref.view(1, 1, 1, C)
    reference_ref.backward(grad)
    torch.testing.assert_close(h_res.grad, h_res_ref.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(residual.grad, residual_ref.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(h_post.grad, h_post_ref.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-7)
    if with_bias:
        torch.testing.assert_close(bias.grad, bias_ref.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_native_h_post_bda_matches_reference_when_compiled_on_gpu(with_bias):
    torch.manual_seed(1234)
    s, b, n, C = 3, 2, 4, 5
    h_res = torch.randn(s, b, n, n, device="cuda")
    residual = torch.randn(s, b, n, C, device="cuda")
    h_post = torch.rand(s, b, n, device="cuda")
    x = torch.randn(s, b, C, device="cuda")
    bias = torch.randn(C, device="cuda") if with_bias else None

    output = native_h_post_bda(h_res, residual, h_post, x, bias)

    reference = torch.einsum("sbji,sbjc->sbic", h_res, residual) + h_post.unsqueeze(
        -1
    ) * x.unsqueeze(2)
    if with_bias:
        reference = reference + h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)


def test_native_proj_rms_returns_projection_and_inv_rms_when_run_eagerly():
    torch.manual_seed(1234)
    x = torch.randn(9, 12, requires_grad=True)
    weight = torch.randn(7, 12, requires_grad=True)
    eps = 1e-6

    proj, r = _eager(native_proj_rms)(x, weight, eps)

    torch.testing.assert_close(proj, x @ weight.t(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        r, 1.0 / (x.norm(dim=-1, keepdim=True) / math.sqrt(12) + eps), rtol=1e-6, atol=1e-7
    )

    grad_proj = torch.randn_like(proj)
    grad_r = torch.randn_like(r)
    (proj * grad_proj + r * grad_r).sum().backward()
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    proj_ref = x_ref @ weight_ref.t()
    r_ref = 1.0 / (x_ref.norm(dim=-1, keepdim=True) / math.sqrt(12) + eps)
    (proj_ref * grad_proj + r_ref * grad_r).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-5, atol=1e-7)


def test_native_proj_rms_returns_projection_and_inv_rms_when_compiled_on_gpu():
    torch.manual_seed(1234)
    x = torch.randn(9, 12, device="cuda")
    weight = torch.randn(7, 12, device="cuda")
    eps = 1e-6

    proj, r = native_proj_rms(x, weight, eps)

    assert proj.shape == (9, 7)
    assert r.shape == (9, 1)
    torch.testing.assert_close(proj, x @ weight.t(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        r, 1.0 / (x.norm(dim=-1, keepdim=True) / math.sqrt(12) + eps), rtol=1e-5, atol=1e-6
    )


def test_native_fused_add_3_sums_three_tensors_when_run_eagerly():
    torch.manual_seed(1234)
    a, b, c = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    assert torch.equal(_eager(native_fused_add_3)(a, b, c), a + b + c)


def test_native_fused_add_3_sums_three_tensors_when_compiled_on_gpu():
    torch.manual_seed(1234)
    a = torch.randn(4, 3, device="cuda")
    b = torch.randn(4, 3, device="cuda")
    c = torch.randn(4, 3, device="cuda")
    torch.testing.assert_close(native_fused_add_3(a, b, c), a + b + c, rtol=1e-5, atol=1e-6)


def test_learned_output_contract_gates_stream_mixture_when_run_eagerly():
    torch.manual_seed(1234)
    s, b, n, C = 2, 4, 3, 5
    eps = 1e-6
    hidden_states = torch.randn(s, b, n * C, dtype=torch.bfloat16)
    head_fn = torch.randn(n, n * C) * 0.1
    base = torch.randn(n) * 0.1
    scale = torch.randn(1) * 0.5

    output = _eager(learned_output_contract)(hidden_states, head_fn, base, scale, n, eps)

    hidden_32 = hidden_states.float()
    rsqrt = torch.rsqrt(hidden_32.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(hidden_32, head_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + eps
    reference = (pre.unsqueeze(-1) * hidden_32.view(s, b, n, C)).sum(dim=-2).to(torch.bfloat16)
    assert output.dtype is torch.bfloat16
    assert output.shape == (s, b, C)
    torch.testing.assert_close(output, reference, rtol=1.6e-2, atol=1e-2)


def test_learned_output_contract_gates_stream_mixture_when_compiled_on_gpu():
    torch.manual_seed(1234)
    s, b, n, C = 2, 4, 3, 5
    eps = 1e-6
    hidden_states = torch.randn(s, b, n * C, device="cuda", dtype=torch.bfloat16)
    head_fn = torch.randn(n, n * C, device="cuda") * 0.1
    base = torch.randn(n, device="cuda") * 0.1
    scale = torch.randn(1, device="cuda") * 0.5

    output = learned_output_contract(hidden_states, head_fn, base, scale, n, eps)

    hidden_32 = hidden_states.float()
    rsqrt = torch.rsqrt(hidden_32.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(hidden_32, head_fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + eps
    reference = (pre.unsqueeze(-1) * hidden_32.view(s, b, n, C)).sum(dim=-2).to(torch.bfloat16)
    assert output.dtype is torch.bfloat16
    assert output.shape == (s, b, C)
    torch.testing.assert_close(output, reference, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("used_outputs", [1, 2, 3], ids=["one_alias", "two_aliases", "all"])
def test_broadcast_tensor_fused_sums_gradients_per_used_alias_when_backward(used_outputs):
    torch.manual_seed(1234)

    def _sum3(a, b, c):
        return a + b + c

    weights = (1.5, -2.0, 3.25)
    x = torch.randn(5, 3, requires_grad=True)
    alias1, alias2, alias3 = BroadcastTensorFused.apply(x, _sum3)
    loss = sum(
        (weight * alias).sum()
        for weight, alias in zip(weights[:used_outputs], (alias1, alias2, alias3))
    )
    loss.backward()
    expected = torch.full_like(x, sum(weights[:used_outputs]))
    torch.testing.assert_close(x.grad, expected, rtol=0.0, atol=0.0)
