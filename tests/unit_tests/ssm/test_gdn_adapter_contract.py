# Adapted from flagos-ai/Megatron-LM-FL PR #153, head 9d4788a41d91fe8a13a2e1563eb39ea43184b674.
# See the repository LICENSE for license information.

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

P = importlib.import_module("megatron.plugin.Ascend.ssm.chunk_gated_delta_rule")


def test_fallback_preserves_all_arguments():
    values = [object() for _ in range(5)]
    opts = dict(
        scale=0.2,
        initial_state=object(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=object(),
        head_first=True,
    )
    sentinel = object()
    with patch.object(P, "_fla_original_chunk_gated_delta_rule") as getter:
        getter.return_value.return_value = (sentinel, None)
        assert P.chunk_gated_delta_rule(*values, **opts)[0] is sentinel
        getter.return_value.assert_called_once_with(
            **dict(zip(("q", "k", "v", "g", "beta"), values)), **opts
        )


def test_missing_runtime_delegates_before_kernel_import():
    values = [object() for _ in range(5)]
    with (
        patch.object(P, "_npu_kernels_support", return_value=True),
        patch.object(P, "_runtime_available", return_value=False),
        patch.object(P, "_fla_original_chunk_gated_delta_rule") as getter,
    ):
        P.chunk_gated_delta_rule(*values)
        getter.return_value.assert_called_once()


def test_l2norm_execution_importerror_is_not_swallowed():
    x = SimpleNamespace(device=SimpleNamespace(type="npu"))
    import fla_npu.ops.triton as t

    with (
        patch.object(t, "l2norm", side_effect=ImportError("kernel execution failure")),
        patch.object(P, "_fla_original_l2norm") as fallback,
    ):
        with pytest.raises(ImportError, match="kernel execution failure"):
            P.l2norm(x)
        fallback.assert_not_called()


def test_cross_device_is_unsupported():
    q = SimpleNamespace(device=SimpleNamespace(type="npu"))
    k = SimpleNamespace(device=object())
    assert not P._npu_kernels_support(q, k, k, k, k, None, False, None)


def test_module_import_does_not_require_fla_npu(monkeypatch):
    import builtins
    import importlib.util

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("fla_npu"):
            raise ImportError("optional dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("gdn_without_optional_kernels", P.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._load_kernels.cache_info().currsize == 0
    assert module._get_gdn_function.cache_info().currsize == 0


def test_failed_kernel_load_is_retryable_and_success_is_cached():
    P._load_kernels.cache_clear()
    try:
        with patch.object(P.importlib, "import_module", side_effect=ImportError("missing kernels")):
            with pytest.raises(ImportError, match="missing kernels"):
                P._load_kernels()
        assert P._load_kernels.cache_info().currsize == 0
        kernels = P._load_kernels()
        with patch.object(P.importlib, "import_module", side_effect=AssertionError("loaded twice")):
            assert P._load_kernels() is kernels
    finally:
        P._load_kernels.cache_clear()


def _fp32_recurrent_reference(q, k, v, g, beta):
    """Independent dense GDN recurrence, accumulated in FP32 on CPU."""
    import torch

    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = []
    for index in range(q.shape[1]):
        state = state * g[:, index].exp()[..., None, None]
        key = k[:, index]
        prediction = torch.einsum("bhd,bhdv->bhv", key, state)
        delta = (v[:, index] - prediction) * beta[:, index, :, None]
        state = state + key[..., :, None] * delta[..., None, :]
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, index], state) * q.shape[-1] ** -0.5)
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("length", [65, 128])
def test_npu_output_and_gradients_match_fp32_reference(dtype_name, length, record_property):
    """Check output and all five input gradients, including a partial chunk."""
    import torch

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required for numerical validation")
    if not P._runtime_available():
        pytest.skip("FLA-NPU optional kernels unavailable")

    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(42 + length)
    shape = (1, length, 2, 128)
    values = [
        (torch.randn(shape, generator=generator) / 128**0.5).to(dtype),
        (torch.randn(shape, generator=generator) / 128**0.5).to(dtype),
        (torch.randn(shape, generator=generator) * 0.3).to(dtype),
        -torch.rand(shape[:3], generator=generator) * 0.1,
        torch.sigmoid(torch.randn(shape[:3], generator=generator)).to(dtype),
    ]
    reference_inputs = [value.float().detach().requires_grad_(True) for value in values]
    device_inputs = [value.to("npu").detach().requires_grad_(True) for value in values]
    upstream_gradient = torch.randn(shape, generator=generator) * 0.1

    # Assert the real kernel path is used; this must never pass by falling back.
    with patch.object(
        P, "_fla_original_chunk_gated_delta_rule", side_effect=AssertionError("unexpected fallback")
    ):
        actual, final_state = P.chunk_gated_delta_rule(*device_inputs)
        assert final_state is None
        actual_gradients = torch.autograd.grad(
            actual, device_inputs, upstream_gradient.to("npu").to(dtype)
        )
        torch.npu.synchronize()
    expected = _fp32_recurrent_reference(*reference_inputs)
    # Quantize the upstream gradient equally in both paths.
    expected_gradients = torch.autograd.grad(
        expected, reference_inputs, upstream_gradient.to(dtype).float()
    )

    # Tolerances are declared before observing results. BF16 has lower mantissa precision.
    atol, rtol, relative_rms_limit = (
        (0.01, 0.03, 0.03) if dtype == torch.float16 else (0.05, 0.08, 0.10)
    )
    for name, got, wanted in zip(
        ("output", "dq", "dk", "dv", "dg", "dbeta"),
        (actual, *actual_gradients),
        (expected, *expected_gradients),
    ):
        got = got.detach().float().cpu()
        wanted = wanted.detach()
        assert torch.isfinite(got).all()
        error = got - wanted
        relative_rms = (
            error.square().mean().sqrt() / wanted.square().mean().sqrt().clamp_min(1e-12)
        ).item()
        record_property(name + "_max_abs", error.abs().max().item())
        record_property(name + "_relative_rms", relative_rms)
        torch.testing.assert_close(got, wanted, atol=atol, rtol=rtol)
        assert relative_rms <= relative_rms_limit, (name, relative_rms)
