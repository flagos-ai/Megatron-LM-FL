# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

"""GDN dispatch, fallback, and Ascend training regression tests."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.plugin import decorators

P = importlib.import_module("megatron.plugin.Ascend.ssm.chunk_gated_delta_rule")
C = importlib.import_module("megatron.plugin.Ascend.ssm.causal_conv1d")


@pytest.fixture(autouse=True)
def isolate_dispatch(monkeypatch):
    monkeypatch.setenv("MG_FL_PREFER", "npu")
    for name in ("_plugin_registry", "_lazy_registry", "_plugin_impl_cache"):
        monkeypatch.setattr(decorators, name, {})
    monkeypatch.setattr(decorators, "_original_impl_cache", set())
    for method, module, impl in (
        ("_normalize_qk", "chunk_gated_delta_rule", "normalize_qk"),
        ("_gated_delta_rule", "chunk_gated_delta_rule", "gated_delta_rule"),
        ("_causal_conv1d", "causal_conv1d", "gated_delta_net_conv1d"),
    ):
        decorators.register(
            target="megatron.core.ssm.gated_delta_net.GatedDeltaNet." + method,
            impl="megatron.plugin.Ascend.ssm." + module + "." + impl,
            vendor="npu",
        )
    P._load_kernels.cache_clear()
    P._get_gdn_function.cache_clear()
    yield
    P._load_kernels.cache_clear()
    P._get_gdn_function.cache_clear()


@pytest.mark.parametrize("vendor,deterministic", [("npu", False), ("npu", True), ("cpu", False)])
def test_recurrence_dispatch(monkeypatch, vendor, deterministic):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    monkeypatch.setenv("MG_FL_PREFER", vendor)
    optimized, native = Mock(), Mock()
    monkeypatch.setattr(P, "chunk_gated_delta_rule", optimized)
    layer = SimpleNamespace(
        config=SimpleNamespace(deterministic_mode=deterministic), gated_delta_rule=native
    )
    args = (object(), object(), object())
    options = dict(cu_seqlens=object(), initial_state=object(), output_final_state=True)
    selected = native if deterministic or vendor == "cpu" else optimized
    assert GatedDeltaNet._gated_delta_rule(layer, *args, **options) is selected.return_value
    selected.assert_called_once_with(*args, **options)
    (optimized if selected is native else native).assert_not_called()


@pytest.mark.parametrize("vendor", ["npu", "cpu"])
@pytest.mark.parametrize("method", ["_normalize_qk", "_causal_conv1d"])
def test_helper_dispatch(monkeypatch, vendor, method):
    core = importlib.import_module("megatron.core.ssm.gated_delta_net")
    monkeypatch.setenv("MG_FL_PREFER", vendor)
    name = "l2norm" if method == "_normalize_qk" else "causal_conv1d"
    target = (P if name == "l2norm" else C) if vendor == "npu" else core
    replacement = Mock()
    monkeypatch.setattr(target, name, replacement, raising=False)
    x = object()
    if name == "l2norm":
        result = core.GatedDeltaNet._normalize_qk(object(), x)
        replacement.assert_called_once_with(x)
    else:
        result = core.GatedDeltaNet._causal_conv1d(object(), x=x, activation="silu")
        replacement.assert_called_once_with(x=x, activation="silu")
    assert result is replacement.return_value


def test_fallback_preserves_arguments(monkeypatch):
    fallback = Mock()
    monkeypatch.setitem(
        sys.modules, "fla.ops.gated_delta_rule", SimpleNamespace(chunk_gated_delta_rule=fallback)
    )
    values = [object() for _ in range(5)]
    options = dict(
        scale=0.2,
        initial_state=object(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=object(),
        head_first=True,
    )
    assert P.chunk_gated_delta_rule(*values, **options) is fallback.return_value
    fallback.assert_called_once_with(**dict(zip(("q", "k", "v", "g", "beta"), values)), **options)


@pytest.mark.parametrize("missing", ["fla_npu", "fla_npu.ops.triton", "unrelated_dependency"])
def test_optional_import(monkeypatch, missing):
    def fail(name):
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(P.importlib, "import_module", fail)
    if missing == "fla_npu":
        assert P._optional_module("fla_npu.ops.triton") is None
    else:
        with pytest.raises(ModuleNotFoundError):
            P._optional_module("fla_npu.ops.triton")


def test_kernel_execution_error_propagates(monkeypatch):
    kernel = Mock(side_effect=ImportError("kernel execution failed"))
    monkeypatch.setattr(P, "_optional_module", lambda name: SimpleNamespace(l2norm=kernel))
    with pytest.raises(ImportError, match="kernel execution failed"):
        P.l2norm(
            SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(16, 128), numel=lambda: 2048)
        )


def test_kernel_loading_retry_and_cache(monkeypatch):
    importer = Mock(side_effect=ImportError("broken installation"))
    monkeypatch.setattr(P.importlib, "import_module", importer)
    with pytest.raises(ImportError, match="broken installation"):
        P._load_kernels()
    assert P._load_kernels.cache_info().currsize == 0
    importer.side_effect = None
    first = P._load_kernels()
    importer.reset_mock()
    assert P._load_kernels() is first
    importer.assert_not_called()


@pytest.mark.parametrize("rows", [7, 16, 17])
def test_normalization_backend(monkeypatch, rows):
    native, fallback = Mock(), Mock()
    monkeypatch.setattr(P, "_optional_module", lambda name: SimpleNamespace(l2norm=native))
    monkeypatch.setitem(sys.modules, "fla.modules.l2norm", SimpleNamespace(l2norm=fallback))
    x = SimpleNamespace(
        device=SimpleNamespace(type="npu"), shape=(rows, 128), numel=lambda: rows * 128
    )
    selected = fallback if rows % 16 else native
    assert P.l2norm(x, eps=1e-5, output_dtype=torch.float32) is selected.return_value
    selected.assert_called_once_with(x, eps=1e-5, output_dtype=torch.float32)


@pytest.mark.parametrize(
    "options",
    [
        dict(backend="cuda"),
        dict(custom_option=3),
        dict(cu_seqlens=object()),
        dict(initial_state=object()),
        dict(output_final_state=True),
    ],
)
def test_convolution_fallback(monkeypatch, options):
    fallback = Mock()
    monkeypatch.setitem(
        sys.modules, "fla.modules.convolution", SimpleNamespace(causal_conv1d=fallback)
    )
    x = SimpleNamespace(device=SimpleNamespace(type="npu"), ndim=3)
    assert C.causal_conv1d(x, weight=object(), **options) is fallback.return_value
    for key, value in options.items():
        assert fallback.call_args.kwargs[key] is value


@pytest.fixture
def supported_inputs():
    device = SimpleNamespace(type="npu")
    tensors = [
        SimpleNamespace(device=device, dtype=torch.bfloat16, shape=(1, 65, 2, 128), ndim=4)
        for _ in range(3)
    ]
    tensors.extend(
        SimpleNamespace(device=device, dtype=dtype, shape=(1, 65, 2), ndim=3)
        for dtype in (torch.float32, torch.bfloat16)
    )
    return dict(
        zip(("q", "k", "v", "g", "beta"), tensors),
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )


def test_supported_input(supported_inputs):
    assert P._npu_kernels_support(**supported_inputs)


@pytest.mark.parametrize(
    "target,change",
    [
        ("q", dict(device=SimpleNamespace(type="cpu"))),
        ("k", dict(device=SimpleNamespace(type="npu", index=1))),
        ("k", dict(dtype=torch.float32)),
        ("v", dict(shape=(1, 66, 2, 128))),
        ("g", dict(ndim=4)),
        ("g", dict(shape=(1, 66, 2))),
        ("g", dict(dtype=torch.bfloat16)),
        ("beta", dict(dtype=torch.float32)),
        ("qkv", dict(shape=(1, 65, 2, 64))),
        ("all", dict(shape=(1, 0, 2))),
        ("initial_state", object()),
        ("output_final_state", True),
        ("cu_seqlens", object()),
    ],
)
def test_unsupported_input(supported_inputs, target, change):
    assert P._npu_kernels_support(**supported_inputs)
    if target in ("initial_state", "output_final_state", "cu_seqlens"):
        supported_inputs[target] = change
    else:
        names = (
            ("q", "k", "v")
            if target == "qkv"
            else ("q", "k", "v", "g", "beta") if target == "all" else (target,)
        )
        for name in names:
            attrs = dict(change)
            if target == "all" and name in ("q", "k", "v"):
                attrs["shape"] += (128,)
            vars(supported_inputs[name]).update(attrs)
    assert not P._npu_kernels_support(**supported_inputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [7, 16, 17])
def test_npu_normalization_gradients(dtype, rows, record_property):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    pytest.importorskip("fla_npu")
    torch.manual_seed(153)
    x = torch.randn((rows, 128), device="npu", dtype=dtype, requires_grad=True)
    reference = x.detach().float().requires_grad_()
    expected = reference * (reference.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    actual = P.l2norm(x)
    gradient = torch.randn_like(actual)
    dx = torch.autograd.grad(actual, x, gradient)[0]
    expected_dx = torch.autograd.grad(expected, reference, gradient.float())[0]
    record_property("dtype", str(dtype))
    record_property("rows", rows)
    for name, result, ref in (("output", actual, expected), ("dx", dx, expected_dx)):
        error = ((result.float() - ref).norm() / ref.norm()).item()
        record_property(name + "_relative_l2", error)
        assert torch.isfinite(result).all()
        assert error < 0.006


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("residual", [False, True])
def test_npu_convolution_gradients(activation, residual, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    monkeypatch.setattr(torch.npu.conv, "allow_hf32", False)
    torch.manual_seed(153)
    x = torch.randn((1, 17, 8), device="npu", requires_grad=True)
    weight = torch.randn((8, 4), device="npu", requires_grad=True)
    bias = torch.randn((8,), device="npu", requires_grad=True)
    skip = torch.randn_like(x) if residual else None
    actual, state = C.causal_conv1d(x, weight, bias, residual=skip, activation=activation)
    outputs = []
    for i in range(x.shape[1]):
        value = bias.expand(x.shape[0], -1)
        for j in range(weight.shape[1]):
            position = i + j - weight.shape[1] + 1
            if position >= 0:
                value = value + x[:, position] * weight[:, j]
        outputs.append(value)
    expected = torch.stack(outputs, dim=1)
    if activation == "silu":
        expected = torch.nn.functional.silu(expected)
    if skip is not None:
        expected = expected + skip
    assert state is None
    gradient = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, weight, bias), gradient)
    ref_grads = torch.autograd.grad(expected, (x, weight, bias), gradient)
    for result, ref in zip((actual, *grads), (expected, *ref_grads)):
        torch.testing.assert_close(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [65, 128])
@pytest.mark.parametrize("normalize", [False, True])
def test_npu_recurrence_gradients(dtype, length, normalize, record_property, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    pytest.importorskip("fla_npu")
    forward = Mock(wraps=P.flash_chunk_gated_delta_rule_fwd)
    backward = Mock(wraps=P.flash_chunk_gated_delta_rule_bwd)
    monkeypatch.setattr(P, "flash_chunk_gated_delta_rule_fwd", forward)
    monkeypatch.setattr(P, "flash_chunk_gated_delta_rule_bwd", backward)
    torch.manual_seed(153)
    shape = (1, length, 2, 128)
    q, k, v = [torch.randn(shape, device="npu", dtype=dtype) * 0.1 for _ in range(3)]
    g = -torch.rand(shape[:3], device="npu", dtype=torch.float32) * 0.1
    beta = torch.rand(shape[:3], device="npu", dtype=dtype)
    inputs = [t.detach().requires_grad_() for t in (q, k, v, g, beta)]
    refs = [t.detach().float().requires_grad_() for t in inputs]
    qr, kr, vr, gr, br = refs
    if normalize:
        qr = qr * (qr.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        kr = kr * (kr.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    state = torch.zeros((1, 2, 128, 128), device="npu")
    outputs = []
    for i in range(length):
        state = state * gr[:, i].exp()[..., None, None]
        delta = (vr[:, i] - (state * kr[:, i, :, :, None]).sum(-2)) * br[:, i, :, None]
        state = state + kr[:, i, :, :, None] * delta[..., None, :]
        outputs.append((state * qr[:, i, :, :, None]).sum(-2) * 128**-0.5)
    reference = torch.stack(outputs, dim=1)
    output, final = P.chunk_gated_delta_rule(*inputs, use_qk_l2norm_in_kernel=normalize)
    assert final is None
    gradient = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, inputs, gradient)
    forward.assert_called_once()
    backward.assert_called_once()
    ref_grads = torch.autograd.grad(reference, refs, gradient.float())
    tolerance = 0.015 if dtype == torch.bfloat16 else 0.004
    record_property("dtype", str(dtype))
    record_property("length", length)
    record_property("normalize", normalize)
    for name, actual, expected in zip(
        ("output", "dq", "dk", "dv", "dg", "dbeta"),
        (output, *actual_grads),
        (reference, *ref_grads),
    ):
        error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        record_property(name + "_relative_l2", error.item())
        assert torch.isfinite(actual).all()
        assert error.item() < tolerance, error.item()
