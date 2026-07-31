from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import megatron.megalens.core_adapter as core_adapter
from megatron.core.observability import reset_trace_sink, scoped_forward, trace_scope
from megatron.core.transformer.attention import Attention, CoreAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.megalens.core_adapter import MegaLensTraceSink
from megatron.megalens.trace import Tracer


@pytest.fixture(autouse=True)
def _reset_core_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def test_attention_probe_wraps_runtime_implementation_not_protocol() -> None:
    assert getattr(Attention.forward, "__megatron_trace_event__", None) == "attention"
    assert getattr(CoreAttention.forward, "__megatron_trace_event__", None) is None


def test_mlp_and_moe_probe_markers_cover_the_runtime_methods() -> None:
    expected_events = {
        MLP.forward: "MLP.forward",
        TopKRouter.forward: "moe-router",
        MoELayer.dispatch: "moe-dispatch",
        MoELayer.shared_experts_compute: "moe-shared-expert",
        MoELayer.routed_experts_compute: "moe-experts",
        MoELayer._combine_with_scope: "moe-combine",
    }

    for method, event_name in expected_events.items():
        assert getattr(method, "__megatron_trace_event__", None) == event_name


def test_adapter_flattens_attrs_and_copies_caller_mappings() -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="full")
    tracer._pendings = []
    tracer._iteration_open = True
    ticks: list[tuple[str, str, dict[str, Any]]] = []
    tracer._tick = lambda name, phase, attrs: ticks.append((name, phase, dict(attrs)))
    sink = MegaLensTraceSink(tracer)
    ctx = {"iteration": 137}
    attrs = {"kind": "runtime"}

    with sink.scope("attention", ctx=ctx, slots=("tokens",), attrs=attrs) as scope:
        assert scope.get("iteration") == 137
        assert scope.set("tokens", 16)

    assert ctx == {"iteration": 137}
    assert attrs == {"kind": "runtime"}
    assert ticks == [
        ("attention", "B", {"iteration": 137}),
        ("attention", "E", {"kind": "runtime", "tokens": 16}),
    ]


def test_adapter_respects_tracer_window_and_granularity() -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="base")
    sink = MegaLensTraceSink(tracer)

    assert not sink.is_enabled("forward")
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    assert sink.is_enabled("forward")
    assert sink.is_enabled("forward-step")
    assert sink.is_enabled("combined-forward-backward-step")
    assert sink.is_enabled("backward-step")
    assert sink.is_enabled("grad-sync")
    assert sink.is_enabled("all-grads-sync")
    assert sink.is_enabled("p2p-launch")
    assert sink.is_enabled("p2p-batch-complete")
    assert sink.is_enabled("p2p-batch-device-sync")
    assert sink.is_enabled("bridge-send-forward")
    assert sink.is_enabled("bridge-recv-forward")
    assert sink.is_enabled("bridge-send-backward")
    assert sink.is_enabled("bridge-recv-backward")
    assert not sink.is_enabled("attention")


def test_adapter_can_drive_decorated_core_contract() -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="full")
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    ticks: list[tuple[str, str]] = []
    tracer._tick = lambda name, phase, attrs: ticks.append((name, phase))

    from megatron.core.observability import install_trace_sink

    install_trace_sink(MegaLensTraceSink(tracer))

    @scoped_forward("attention")
    def forward(value: int) -> int:
        return value + 1

    assert forward(3) == 4
    assert ticks == [("attention", "B"), ("attention", "E")]
    with trace_scope("manual", attrs={"kind": "test"}):
        pass
    assert ticks[-2:] == [("manual", "B"), ("manual", "E")]


def test_capture_query_tolerates_unsupported_torch_compatible_backend(monkeypatch) -> None:
    monkeypatch.setattr(core_adapter, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(core_adapter, "is_graph_warmup", lambda: False)

    def unsupported() -> bool:
        raise RuntimeError("backend does not implement capture query")

    monkeypatch.setattr(core_adapter, "_CUDA_CAPTURE_QUERY", unsupported)
    assert not core_adapter.should_suppress_core_scope()

    monkeypatch.setattr(core_adapter, "_CUDA_CAPTURE_QUERY", lambda: True)
    assert core_adapter.should_suppress_core_scope()
