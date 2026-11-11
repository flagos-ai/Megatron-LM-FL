# Copyright (c) 2026, BAAI. All rights reserved.
"""Batch-invariant TE GEMM: layout flattening, bias, gradients, and dispatcher guardrails."""

import pytest
import torch

import megatron.core.transformer.custom_layers.batch_invariant_kernels as bik

_LAYOUTS = ["TN", "NN", "NT", "TT"]


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    # The flagcicd_cov files share one process; never inherit or leak the aten overrides.
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()
    yield
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()


def _layout_refs(layout):
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    K, O, N, S = 8, 5, 7, 2
    a = torch.randn(O, K) if transa else torch.randn(K, O)
    b = torch.randn(K, N) if transb else torch.randn(S, N, K)
    return a, b, transa, transb, (K, O, N, S)


def _gemm_reference(a, b, bias, transa, transb):
    op_a = a.transpose(0, 1) if transa else a
    op_b = b.transpose(0, 1) if transb else b
    out = op_b.reshape(-1, op_b.shape[-1]).float() @ op_a.float()
    leading_shape = op_b.shape[:-1]
    out = out.reshape(*leading_shape, out.shape[-1])
    if bias is not None:
        out = out + bias.float()
    return out


def test_te_gemm_fn_forward_rejects_one_dimensional_a():
    with pytest.raises(ValueError, match="opA has insufficient dimensions"):
        bik.BatchInvariantTEGemmFn.apply(torch.randn(8), torch.randn(4, 8), None, None, "NN")


def test_te_gemm_fn_forward_flattens_three_dimensional_a():
    torch.manual_seed(1234)
    K, O, N = 8, 5, 7
    a = torch.randn(1, K, O, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    out = bik.BatchInvariantTEGemmFn.apply(a, b, None, None, "TN")

    assert out.shape == (N, O)
    torch.testing.assert_close(out, b.cpu().float() @ a[0].cpu().float(), rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_te_gemm_fn_forward_and_backward_match_reference_for_layouts(layout, with_bias):
    torch.manual_seed(1234)
    a, b, transa, transb, (K, O, N, S) = _layout_refs(layout)
    bias = torch.randn(O) if with_bias else None

    a_cuda = a.to("cuda", torch.bfloat16).requires_grad_(True)
    b_cuda = b.to("cuda", torch.bfloat16).requires_grad_(True)
    bias_cuda = bias.to("cuda", torch.bfloat16).requires_grad_(True) if with_bias else None

    out = bik.BatchInvariantTEGemmFn.apply(a_cuda, b_cuda, bias_cuda, None, layout)
    expected_shape = (N, O) if b.dim() == 2 else (S, N, O)
    assert out.shape == expected_shape
    reference = _gemm_reference(a, b, bias, transa, transb)
    torch.testing.assert_close(out, reference.to("cuda"), rtol=1.6e-2, atol=1e-2)

    grad_output = torch.randn(expected_shape).to("cuda", torch.bfloat16)
    out.backward(grad_output)

    a_ref = a.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    bias_ref = bias.clone().requires_grad_(True) if with_bias else None
    reference = _gemm_reference(a_ref, b_ref, bias_ref, transa, transb)
    reference.backward(grad_output.cpu().float())

    torch.testing.assert_close(a_cuda.grad, a_ref.grad.to("cuda"), rtol=1.6e-2, atol=1e-2)
    torch.testing.assert_close(b_cuda.grad, b_ref.grad.to("cuda"), rtol=1.6e-2, atol=1e-2)
    if with_bias:
        torch.testing.assert_close(bias_cuda.grad, bias_ref.grad.to("cuda"), rtol=1.6e-2, atol=1e-2)


def test_te_general_gemm_patched_raises_when_original_not_captured():
    with pytest.raises(RuntimeError, match="TE general_gemm original not captured"):
        bik._te_general_gemm_patched(torch.empty(2, 2), torch.empty(2, 2))


@pytest.mark.parametrize(
    "case,error_type,match",
    [
        pytest.param("missing_a", ValueError, "requires A and B tensors", id="missing_a"),
        pytest.param("cpu_tensors", RuntimeError, "requires CUDA tensors", id="cpu_tensors"),
        pytest.param(
            "float64_tensors",
            RuntimeError,
            "Unsupported dtype for batch-invariant GEMM",
            id="float64_tensors",
        ),
        pytest.param(
            "extra_output_overlap",
            RuntimeError,
            "does not support Userbuffers",
            id="extra_output_overlap",
        ),
    ],
)
def test_te_general_gemm_patched_rejects_unsupported_inputs_when_mode_enabled(
    case, error_type, match
):
    if case == "missing_a":
        kwargs = dict(A=None, B=torch.empty(2, 2, device="cuda"))
    elif case == "cpu_tensors":
        kwargs = dict(A=torch.empty(2, 2), B=torch.empty(2, 2))
    elif case == "float64_tensors":
        kwargs = dict(
            A=torch.empty(2, 2, device="cuda", dtype=torch.float64),
            B=torch.empty(2, 2, device="cuda", dtype=torch.float64),
        )
    else:
        kwargs = dict(
            A=torch.empty(2, 2, device="cuda"),
            B=torch.empty(2, 2, device="cuda"),
            extra_output=torch.empty(2, 2, device="cuda"),
        )
    with bik.set_batch_invariant_mode(True):
        with pytest.raises(error_type, match=match):
            bik._te_general_gemm_patched(**kwargs)


@pytest.mark.parametrize("use_out_buffer", [False, True], ids=["returned", "out_buffer"])
def test_te_general_gemm_patched_returns_bias_grad_when_grad_requested(use_out_buffer):
    torch.manual_seed(1234)
    T, K, O = 6, 8, 4
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(T, O, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(O, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(O, K, device="cuda", dtype=torch.bfloat16) if use_out_buffer else None

    with bik.set_batch_invariant_mode(True):
        result = bik._te_general_gemm_patched(
            x, dy, bias=bias, out=out, out_dtype=None, layout="NT", grad=True
        )

    gemm_out, bias_grad, gelu_input, extra_output = result
    assert len(result) == 4
    assert gelu_input is None
    assert extra_output is None
    if use_out_buffer:
        assert gemm_out is out
    torch.testing.assert_close(bias_grad, dy.cpu().float().sum(dim=0), rtol=1.6e-2, atol=1e-2)
    torch.testing.assert_close(
        gemm_out, dy.cpu().float().t() @ x.cpu().float(), rtol=1.6e-2, atol=1e-2
    )
