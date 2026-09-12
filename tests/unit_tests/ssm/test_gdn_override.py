import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.plugin import decorators

P = importlib.import_module('megatron.plugin.Ascend.ssm.chunk_gated_delta_rule')
C = importlib.import_module('megatron.plugin.Ascend.ssm.causal_conv1d')
CORE = importlib.import_module('megatron.core.ssm.gated_delta_net')


@pytest.fixture(autouse=True)
def clear_dispatch(monkeypatch):
    monkeypatch.setenv('MG_FL_PREFER', 'npu')
    decorators._plugin_impl_cache.clear()
    decorators._original_impl_cache.clear()
    yield
    decorators._plugin_impl_cache.clear()
    decorators._original_impl_cache.clear()


def test_method_registration_and_dispatch(monkeypatch):
    replacement = Mock(return_value=('optimized', None))
    monkeypatch.setattr(P, 'chunk_gated_delta_rule', replacement)
    layer = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    q, k, v, cu = (object() for _ in range(4))
    assert GatedDeltaNet._gated_delta_rule(layer, q, k, v, cu_seqlens=cu)[0] == 'optimized'
    replacement.assert_called_once_with(q, k, v, cu_seqlens=cu)
    assert 'GatedDeltaNet._gated_delta_rule' in decorators._lazy_registry
    assert 'chunk.chunk_gated_delta_rule' not in decorators._lazy_registry


def test_deterministic_keeps_native_selection(monkeypatch):
    native = Mock(return_value=('reference', None))
    optimized = Mock(side_effect=AssertionError('must not enter optimized GDN'))
    monkeypatch.setattr(P, 'chunk_gated_delta_rule', optimized)
    layer = SimpleNamespace(
        config=SimpleNamespace(deterministic_mode=True), gated_delta_rule=native
    )
    q, k, v, cu = (object() for _ in range(4))
    assert GatedDeltaNet._gated_delta_rule(layer, q, k, v, cu_seqlens=cu)[0] == 'reference'
    native.assert_called_once_with(q, k, v, cu_seqlens=cu)


def test_non_npu_keeps_native_selection(monkeypatch):
    monkeypatch.setenv('MG_FL_PREFER', 'cpu')
    native = Mock(return_value=('native', None))
    layer = SimpleNamespace(gated_delta_rule=native)
    args = (object(), object(), object())
    opts = dict(cu_seqlens=object(), initial_state=object(), output_final_state=True)
    assert GatedDeltaNet._gated_delta_rule(layer, *args, **opts)[0] == 'native'
    native.assert_called_once_with(*args, **opts)


@pytest.mark.parametrize('vendor,target,helper', [('npu', P, 'l2norm'), ('cpu', CORE, 'l2norm')])
def test_normalization_dispatch(monkeypatch, vendor, target, helper):
    monkeypatch.setenv('MG_FL_PREFER', vendor)
    fn = Mock(return_value='normalized')
    monkeypatch.setattr(target, helper, fn)
    x = object()
    assert GatedDeltaNet._normalize_qk(object(), x) == 'normalized'
    fn.assert_called_once_with(x)


@pytest.mark.parametrize('vendor,target', [('npu', C), ('cpu', CORE)])
def test_convolution_dispatch_preserves_arguments(monkeypatch, vendor, target):
    monkeypatch.setenv('MG_FL_PREFER', vendor)
    fn = Mock(return_value=('conv', None))
    monkeypatch.setattr(target, 'causal_conv1d', fn)
    options = dict(x=object(), weight=object(), cu_seqlens=object(), activation='silu')
    assert GatedDeltaNet._causal_conv1d(object(), **options)[0] == 'conv'
    fn.assert_called_once_with(**options)


def test_kernel_error_propagates(monkeypatch):
    monkeypatch.setattr(
        P, 'chunk_gated_delta_rule', Mock(side_effect=RuntimeError('kernel failed'))
    )
    layer = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    with pytest.raises(RuntimeError, match='kernel failed'):
        GatedDeltaNet._gated_delta_rule(layer, None, None, None)


def test_core_decorator_placement():
    import ast
    from pathlib import Path

    module = ast.parse(Path(CORE.__file__).read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "GatedDeltaNet")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in ("_normalize_qk", "_gated_delta_rule", "_causal_conv1d"):
        assert [ast.unparse(d) for d in methods[name].decorator_list] == ["overridable"]
    assert [
        ast.unparse(d) for d in methods["_prepare_qkv_for_gated_delta_rule"].decorator_list
    ] == ["jit_fuser"]
