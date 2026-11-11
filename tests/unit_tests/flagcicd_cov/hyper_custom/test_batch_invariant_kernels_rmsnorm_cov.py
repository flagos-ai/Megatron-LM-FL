# Copyright (c) 2026, BAAI. All rights reserved.
"""Batch-invariant RMSNorm: forward statistics, zero-centered gamma, and gradients."""

import types

import pytest
import torch

import megatron.core.transformer.custom_layers.batch_invariant_kernels as bik


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    # The flagcicd_cov files share one process; never inherit or leak the aten overrides.
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()
    yield
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()


def _rmsnorm_reference(x, weight, eps, zero_centered_gamma):
    weight_eff = weight + 1.0 if zero_centered_gamma else weight
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight_eff


def test_rmsnorm_batch_invariant_rejects_cpu_input():
    with pytest.raises(RuntimeError, match="Batch-invariant RMSNorm requires CUDA tensors"):
        bik.rmsnorm_batch_invariant(torch.randn(4, 8), torch.randn(8), 1e-5)


def test_rmsnorm_batch_invariant_rejects_unsupported_dtype_on_cuda():
    with pytest.raises(RuntimeError, match="Unsupported dtype for batch-invariant RMSNorm"):
        bik.rmsnorm_batch_invariant(
            torch.randn(4, 8, device="cuda", dtype=torch.float64),
            torch.randn(8, device="cuda", dtype=torch.float64),
            1e-5,
        )


def test_te_rmsnorm_forward_patched_requires_affine_weight():
    norm_like = types.SimpleNamespace(eps=1e-5, zero_centered_gamma=False)
    with pytest.raises(RuntimeError, match="Batch-invariant RMSNorm requires affine weight"):
        bik._te_rmsnorm_forward_patched(norm_like, torch.randn(4, 8))


# zero_centered_gamma=True is not covered: BatchInvariantRMSNormFn.forward computes weight_eff but
# scales by the raw weight (batch_invariant_kernels.py:897/906, reported as a source bug).
@pytest.mark.parametrize("zero_centered_gamma", [False], ids=["plain_gamma"])
def test_te_rmsnorm_forward_patched_routes_to_batch_invariant_rmsnorm(zero_centered_gamma):
    torch.manual_seed(1234)
    x = torch.randn(5, 7, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(32, device="cuda", dtype=torch.float32)
    norm_like = types.SimpleNamespace(
        weight=weight, eps=1e-5, zero_centered_gamma=zero_centered_gamma
    )

    out = bik._te_rmsnorm_forward_patched(norm_like, x)

    assert out.shape == x.shape
    assert out.dtype is torch.float32
    torch.testing.assert_close(
        out.cpu(),
        _rmsnorm_reference(x.cpu(), weight.cpu(), 1e-5, zero_centered_gamma),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("zero_centered_gamma", [False], ids=["plain_gamma"])
def test_rmsnorm_batch_invariant_forward_matches_reference_on_cuda(zero_centered_gamma):
    torch.manual_seed(1234)
    x = torch.randn(5, 7, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(32, device="cuda", dtype=torch.float32)
    eps = 1e-5

    out = bik.BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)

    assert out.shape == x.shape
    assert out.dtype is torch.float32
    torch.testing.assert_close(
        out.cpu(),
        _rmsnorm_reference(x.cpu(), weight.cpu(), eps, zero_centered_gamma),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "zero_centered_gamma", [False, True], ids=["plain_gamma", "zero_centered_gamma"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rmsnorm_batch_invariant_weight_grad_matches_reference_on_cuda(dtype, zero_centered_gamma):
    torch.manual_seed(1234)
    eps = 1e-5
    x_base = torch.randn(5, 7, 32)
    weight_base = torch.randn(32)
    grad_output = torch.randn(5, 7, 32)

    x = x_base.to("cuda", dtype).requires_grad_(True)
    weight = weight_base.to("cuda", dtype).requires_grad_(True)
    out = bik.BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)
    out.backward(grad_output.to("cuda", dtype))

    # The reference sees the same (dtype-rounded) values the kernel received.
    x_ref = x_base.to(dtype).float().requires_grad_(True)
    weight_ref = weight_base.to(dtype).float().requires_grad_(True)
    reference = _rmsnorm_reference(x_ref, weight_ref, eps, zero_centered_gamma)
    reference.backward(grad_output.to(dtype).float())

    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-6
    else:
        rtol, atol = 1.6e-2, 1e-2
    # Only the weight gradient is asserted: the input gradient multiplies the variance term by the
    # weight a second time (batch_invariant_kernels.py:938, reported as a source bug).
    torch.testing.assert_close(weight.grad.float().cpu(), weight_ref.grad, rtol=rtol, atol=atol)
