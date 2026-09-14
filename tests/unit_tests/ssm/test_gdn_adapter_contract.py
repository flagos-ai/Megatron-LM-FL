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
