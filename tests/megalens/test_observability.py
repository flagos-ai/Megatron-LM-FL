# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
from typing import Any

import pytest

from megatron.core.observability import (
    install_trace_sink,
    open_trace_scope,
    prepare_trace_scope,
    reset_trace_sink,
    scoped_forward,
    trace_is_enabled,
    trace_scope,
)


class _FakeScope:
    def __init__(self, events: list[tuple[Any, ...]], name: str, ctx: Any) -> None:
        self.events = events
        self.name = name
        self.ctx = ctx
        self.values: dict[str, Any] = {}

    def __enter__(self) -> "_FakeScope":
        self.events.append(("enter", self.name, self.ctx))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.events.append(("exit", self.name, exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> bool:
        self.values[key] = value
        return True


class _FakeSink:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.events: list[tuple[Any, ...]] = []

    def is_enabled(self, name: str) -> bool:
        self.events.append(("enabled", name))
        return self.enabled

    def scope(self, name: str, *, ctx=None, slots=None, attrs=None) -> _FakeScope:
        self.events.append(("scope", name, slots, attrs))
        return _FakeScope(self.events, name, ctx)


class _SuppressingScope(_FakeScope):
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        super().__exit__(exc_type, exc_value, traceback)
        return True


class _SuppressingSink(_FakeSink):
    def scope(self, name: str, *, ctx=None, slots=None, attrs=None) -> _FakeScope:
        return _SuppressingScope(self.events, name, ctx)


class _CleanupFailingScope(_FakeScope):
    def __init__(
        self, events: list[tuple[Any, ...]], name: str, ctx: Any, cleanup_error: Exception
    ) -> None:
        super().__init__(events, name, ctx)
        self.cleanup_error = cleanup_error

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        super().__exit__(exc_type, exc_value, traceback)
        raise self.cleanup_error


class _CleanupFailingSink(_FakeSink):
    def __init__(self, cleanup_error: Exception) -> None:
        super().__init__()
        self.cleanup_error = cleanup_error

    def scope(self, name: str, *, ctx=None, slots=None, attrs=None) -> _FakeScope:
        return _CleanupFailingScope(self.events, name, ctx, self.cleanup_error)


class _SingleAdmissionSink(_FakeSink):
    def __init__(self) -> None:
        super().__init__()
        self.gate_calls = 0

    def is_enabled(self, name: str) -> bool:
        self.gate_calls += 1
        self.events.append(("enabled", name))
        return self.gate_calls == 1


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def test_null_sink_preserves_result_signature_and_exception() -> None:
    @scoped_forward("add")
    def add(left: int, right: int = 1) -> int:
        return left + right

    signature = inspect.signature(add)
    assert add(2, right=3) == 5
    assert str(signature) == "(left: 'int', right: 'int' = 1) -> 'int'"

    @scoped_forward("raise")
    def fail() -> None:
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        fail()

    with trace_scope("null") as scope:
        assert scope.get("missing") is None
        assert scope.set("ignored", 1)


def test_active_sink_enters_and_exits_on_success_and_error() -> None:
    sink = _FakeSink()
    install_trace_sink(sink)

    @scoped_forward("work")
    def work(value: int) -> int:
        return value * 2

    assert work(4) == 8
    assert sink.events == [
        ("enabled", "work"),
        ("scope", "work", None, None),
        ("enter", "work", None),
        ("exit", "work", None),
    ]

    sink.events.clear()

    @scoped_forward("broken")
    def broken() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        broken()
    assert sink.events[-1] == ("exit", "broken", RuntimeError)


def test_sink_cannot_suppress_model_exception() -> None:
    install_trace_sink(_SuppressingSink())

    @scoped_forward("broken")
    def broken() -> None:
        raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        broken()


def test_direct_scope_cannot_suppress_model_exception() -> None:
    install_trace_sink(_SuppressingSink())

    with pytest.raises(RuntimeError, match="model failure"):
        with trace_scope("broken-direct"):
            raise RuntimeError("model failure")


def test_direct_scope_preserves_model_exception_when_trace_cleanup_fails() -> None:
    cleanup_error = OSError("trace cleanup failed")
    sink = _CleanupFailingSink(cleanup_error)
    install_trace_sink(sink)
    model_error = RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure") as raised:
        with trace_scope("broken-direct"):
            raise model_error

    assert raised.value is model_error
    assert raised.value.__notes__ == [
        "trace scope cleanup also failed: OSError('trace cleanup failed')"
    ]
    assert sink.events[-1] == ("exit", "broken-direct", RuntimeError)


def test_direct_scope_propagates_trace_cleanup_error_after_success() -> None:
    cleanup_error = OSError("trace cleanup failed")
    sink = _CleanupFailingSink(cleanup_error)
    install_trace_sink(sink)

    with pytest.raises(OSError, match="trace cleanup failed") as raised:
        with trace_scope("cleanup-fails"):
            pass

    assert raised.value is cleanup_error
    assert sink.events[-1] == ("exit", "cleanup-fails", None)


def test_suppression_executes_function_without_opening_scope() -> None:
    sink = _FakeSink()
    install_trace_sink(sink, suppress_scope=lambda: True)

    @scoped_forward("captured")
    def captured() -> str:
        return "result"

    assert captured() == "result"
    assert sink.events == [("enabled", "captured")]


def test_disabled_sink_skips_suppression_query() -> None:
    suppress_calls = 0

    def suppress() -> bool:
        nonlocal suppress_calls
        suppress_calls += 1
        return False

    sink = _FakeSink(enabled=False)
    install_trace_sink(sink, suppress_scope=suppress)

    @scoped_forward("inactive")
    def inactive() -> int:
        return 1

    assert inactive() == 1
    assert sink.events == [("enabled", "inactive")]
    assert suppress_calls == 0


def test_capture_safe_scope_bypasses_suppression() -> None:
    sink = _FakeSink()
    install_trace_sink(sink, suppress_scope=lambda: True)

    @scoped_forward("capture-safe", capture_safe=True)
    def captured() -> str:
        return "result"

    assert captured() == "result"
    assert sink.events[0] == ("enabled", "capture-safe")


def test_trace_is_enabled_uses_the_same_sink_and_suppression_gates() -> None:
    assert trace_is_enabled("metadata") is False

    sink = _FakeSink()
    install_trace_sink(sink, suppress_scope=lambda: True)

    assert trace_is_enabled("metadata") is False
    assert trace_is_enabled("metadata", capture_safe=True) is True


def test_prepared_scope_consumes_one_gate_snapshot() -> None:
    sink = _SingleAdmissionSink()
    install_trace_sink(sink)

    gate = prepare_trace_scope("single-admission")
    assert gate is not None
    with open_trace_scope(gate, "single-admission", ctx={"field": "complete"}):
        pass

    assert sink.gate_calls == 1
    assert sink.events == [
        ("enabled", "single-admission"),
        ("scope", "single-admission", None, None),
        ("enter", "single-admission", {"field": "complete"}),
        ("exit", "single-admission", None),
    ]


def test_context_result_hook_and_direct_scope() -> None:
    sink = _FakeSink()
    install_trace_sink(sink)

    @scoped_forward(
        "metadata",
        ctx_factory=lambda value: {"input": value},
        result_hook=lambda result, scope: scope.set("output", result),
    )
    def metadata(value: int) -> int:
        return value + 1

    assert metadata(2) == 3
    assert ("enter", "metadata", {"input": 2}) in sink.events

    sink.events.clear()
    with trace_scope("direct", ctx={"iteration": 7}) as scope:
        assert scope.set("tokens", 16)
    assert ("enter", "direct", {"iteration": 7}) in sink.events


def test_reset_restores_null_fast_path() -> None:
    sink = _FakeSink()
    install_trace_sink(sink)
    reset_trace_sink()

    @scoped_forward("after-reset")
    def after_reset() -> int:
        return 7

    assert after_reset() == 7
    assert sink.events == []


@pytest.mark.parametrize(
    "callable_source",
    [
        "async def target():\n    return 1",
        "def target():\n    yield 1",
        "async def target():\n    yield 1",
    ],
)
def test_scoped_forward_rejects_deferred_callables(callable_source: str) -> None:
    namespace: dict[str, Any] = {}
    exec(callable_source, namespace)

    with pytest.raises(TypeError, match="synchronous callables only"):
        scoped_forward("deferred")(namespace["target"])
