"""GDN hook registration, shared-manager use and deterministic fallback."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ssm import gated_delta_net as core
from megatron.plugin.Ascend.ssm import gated_delta_net as plugin


@pytest.fixture(autouse=True)
def select_npu_override(monkeypatch):
    from megatron.plugin import decorators

    monkeypatch.setenv("MG_FL_PREFER", "npu")
    monkeypatch.setattr(decorators, "_plugin_impl_cache", {})
    monkeypatch.setattr(decorators, "_original_impl_cache", set())


def test_registered_hooks_reach_shared_manager(monkeypatch):
    from transformer_engine.plugin.core import manager

    calls = []

    class Recorder:
        registry = SimpleNamespace(get_implementations=lambda name: [object()])

        def ensure_initialized(self):
            pass

        def call(self, name, **kwargs):
            calls.append((name, kwargs))
            return kwargs["value"], None

    monkeypatch.setattr(manager, "get_default_manager", lambda: Recorder())
    obj = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    q = torch.randn(1, 3, 2, 4)
    gate = torch.zeros(1, 3, 2)
    out, state = core.GatedDeltaNet._gated_delta_rule(obj, q, q, q, g=gate, beta=gate)
    assert out is q and state is None
    assert calls[0][0] == "gated_delta_net_forward"
    assert calls[0][1]["use_qk_l2norm"] is False
    torch.testing.assert_close(
        core.GatedDeltaNet._normalize_qk(obj, q), plugin.normalize_qk(obj, q)
    )


def test_deterministic_mode_uses_native_torch(monkeypatch):
    from transformer_engine.plugin.core import manager

    def forbidden():
        raise AssertionError("Deterministic mode must not access the TE manager")

    monkeypatch.setattr(manager, "get_default_manager", forbidden)
    obj = SimpleNamespace(config=SimpleNamespace(deterministic_mode=True), gated_delta_rule=core.torch_chunk_gated_delta_rule)
    q = torch.randn(1, 3, 2, 4) * 0.1
    gate = torch.zeros(1, 3, 2)
    actual = core.GatedDeltaNet._gated_delta_rule(obj, q, q, q, g=gate, beta=gate.sigmoid())[0]
    expected = core.torch_chunk_gated_delta_rule(q, q, q, gate, gate.sigmoid())[0]
    torch.testing.assert_close(actual, expected)


def test_declined_call_matches_native_forward_backward(monkeypatch):
    from transformer_engine.plugin.core import manager

    class Decline:
        registry = SimpleNamespace(get_implementations=lambda name: [object()])

        def ensure_initialized(self):
            pass

        def call(self, *args, **kwargs):
            return NotImplemented

    monkeypatch.setattr(manager, "get_default_manager", Decline)
    torch.manual_seed(42)
    xs = [torch.randn(1, 65, 2, d) * 0.1 for d in (4, 4, 6)]
    xs += [-torch.rand(1, 65, 2), torch.rand(1, 65, 2)]
    a = [x.clone().requires_grad_() for x in xs]
    b = [x.clone().requires_grad_() for x in xs]
    obj = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    actual = plugin.gated_delta_rule(obj, *a[:3], g=a[3], beta=a[4])[0]
    expected = core.torch_chunk_gated_delta_rule(*b)[0]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad)


@pytest.mark.parametrize("selector", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("have_fla", [False, True])
def test_constructor_preserves_fla_requirement(monkeypatch, selector, have_fla):
    monkeypatch.setenv("MG_FL_PREFER", selector)
    monkeypatch.setattr(core, "HAVE_FLA", have_fla)

    class ConstructionReached(Exception):
        pass

    def reached(self, config):
        raise ConstructionReached

    monkeypatch.setattr(core.MegatronModule, "__init__", reached)
    obj = object.__new__(core.GatedDeltaNet)
    if have_fla:
        with pytest.raises(ConstructionReached):
            core.GatedDeltaNet.__init__(obj, config=None, submodules=None)
    else:
        with pytest.raises(ImportError, match="FLA is not installed"):
            core.GatedDeltaNet.__init__(obj, config=None, submodules=None)






def test_absent_te_operator_uses_native_torch(monkeypatch):
    from transformer_engine.plugin.core import manager

    fake = SimpleNamespace(
        ensure_initialized=lambda: None,
        registry=SimpleNamespace(get_implementations=lambda name: []),
        call=lambda *a, **k: pytest.fail('Absent operator must not be called'),
    )
    monkeypatch.setattr(manager, 'get_default_manager', lambda: fake)
    q = torch.randn(1, 3, 2, 4) * .1
    g = -torch.rand(1, 3, 2)
    obj = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    actual = plugin.gated_delta_rule(obj, q, q, q, g=g, beta=g.sigmoid())
    expected = core.torch_chunk_gated_delta_rule(q, q, q, g, g.sigmoid())
    torch.testing.assert_close(actual, expected)


def test_missing_te_import_uses_native_torch(monkeypatch):
    import builtins
    original = builtins.__import__

    def without_te(name, *args, **kwargs):
        if name == 'transformer_engine.plugin.core.manager':
            raise ImportError('TE unavailable')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_te)
    q = torch.randn(1, 3, 2, 4) * .1
    g = -torch.rand(1, 3, 2)
    obj = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    actual = plugin.gated_delta_rule(obj, q, q, q, g=g, beta=g.sigmoid())
    expected = core.torch_chunk_gated_delta_rule(q, q, q, g, g.sigmoid())
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('error', [ValueError, RuntimeError, torch.OutOfMemoryError])
def test_manager_execution_error_never_uses_native(monkeypatch, error):
    from transformer_engine.plugin.core import manager

    def fail(*a, **k):
        raise error('kernel failure')

    fake = SimpleNamespace(
        ensure_initialized=lambda: None,
        registry=SimpleNamespace(get_implementations=lambda name: [object()]),
        call=fail,
    )
    monkeypatch.setattr(manager, 'get_default_manager', lambda: fake)
    monkeypatch.setattr(plugin, '_torch_fallback', lambda *a, **k: pytest.fail('Do not hide execution errors'))
    q = torch.randn(1, 3, 2, 4)
    g = torch.zeros(1, 3, 2)
    obj = SimpleNamespace(config=SimpleNamespace(deterministic_mode=False))
    with pytest.raises(error, match='kernel failure'):
        plugin.gated_delta_rule(obj, q, q, q, g=g, beta=g.sigmoid())


def test_torch_fallback_normalization_and_state_gradients():
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(3)]
    state = torch.zeros(1, 2, 4, 4, requires_grad=True)
    g = -torch.rand(1, 3, 2)
    out, final = plugin._torch_fallback(q, k, v, g=g, beta=g.sigmoid(),
        initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    expected = core.torch_chunk_gated_delta_rule(plugin.normalize_qk(None, q),
        plugin.normalize_qk(None, k), v, g, g.sigmoid(), initial_state=state,
        output_final_state=True)
    torch.testing.assert_close((out, final), expected)
    (out.sum() + final.sum()).backward()
    for x in (q, k, v, state):
        assert x.grad is not None and torch.isfinite(x.grad).all()
