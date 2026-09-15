# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-light observability hooks for Megatron Core.

Core modules use this facade without importing a concrete tracing package. The
default sink is a process-wide no-op. A runtime integration may install a sink
after argument parsing and provide a predicate that suppresses unsafe regions
such as CUDA Graph capture or compiler tracing.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal, Mapping, ParamSpec, Protocol, Sequence, TypeVar


class TraceScope(Protocol):
    """Context manager returned by a trace sink."""

    def __enter__(self) -> "TraceScope": ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> Literal[False] | None: ...

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> bool: ...


class TraceSink(Protocol):
    """Runtime-provided trace implementation.

    ``is_enabled`` is called from hot paths and during graph capture. It must
    remain a capture-safe, allocation-light state check with no device work.
    """

    def is_enabled(self, name: str) -> bool: ...

    def scope(
        self,
        name: str,
        *,
        ctx: Mapping[str, Any] | None = None,
        slots: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> TraceScope: ...


class _NoopTraceScope:
    __slots__ = ()

    def __enter__(self) -> "_NoopTraceScope":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> None:
        return None

    def get(self, key: str) -> None:
        return None

    def set(self, key: str, value: Any) -> bool:
        return True


_NOOP_SCOPE = _NoopTraceScope()


class _NonSuppressingTraceScope:
    """Delegate a sink scope while preserving exceptions from the traced code."""

    __slots__ = ("_manager", "_active")

    def __init__(self, manager: TraceScope) -> None:
        self._manager = manager
        self._active: TraceScope | None = None

    def __enter__(self) -> "_NonSuppressingTraceScope":
        self._active = self._manager.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> Literal[False]:
        if exc_value is None:
            self._manager.__exit__(exc_type, exc_value, traceback)
            return False

        try:
            self._manager.__exit__(exc_type, exc_value, traceback)
        except Exception as trace_error:
            exc_value.add_note(f"trace scope cleanup also failed: {trace_error!r}")
        return False

    def get(self, key: str) -> Any | None:
        assert self._active is not None, "trace scope must be entered before use"
        return self._active.get(key)

    def set(self, key: str, value: Any) -> bool:
        assert self._active is not None, "trace scope must be entered before use"
        return self._active.set(key, value)


class NullTraceSink:
    """No-op sink used until a runtime integration explicitly installs one."""

    __slots__ = ()

    def is_enabled(self, name: str) -> bool:
        return False

    def scope(
        self,
        name: str,
        *,
        ctx: Mapping[str, Any] | None = None,
        slots: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> TraceScope:
        return _NOOP_SCOPE


NULL_TRACE_SINK = NullTraceSink()


def _never_suppress() -> bool:
    return False


@dataclass(frozen=True, slots=True)
class TraceGate:
    """Accepted tracing-state snapshot consumed immediately by a probe call site."""

    sink: TraceSink
    suppress_scope: Callable[[], bool]


_NULL_STATE = TraceGate(NULL_TRACE_SINK, _never_suppress)
_STATE = _NULL_STATE


def _active_trace_state(name: str, *, capture_safe: bool) -> TraceGate | None:
    """Return the accepting state for an event without allocating a scope."""
    state = _STATE
    if state is _NULL_STATE:
        return None
    if not state.sink.is_enabled(name):
        return None
    if not capture_safe and state.suppress_scope():
        return None
    return state


def prepare_trace_scope(name: str, *, capture_safe: bool = False) -> TraceGate | None:
    """Evaluate a probe gate once before its caller allocates event metadata."""
    return _active_trace_state(name, capture_safe=capture_safe)


def trace_is_enabled(name: str, *, capture_safe: bool = False) -> bool:
    """Check whether an event would open a scope at this call site."""
    return prepare_trace_scope(name, capture_safe=capture_safe) is not None


def install_trace_sink(
    sink: TraceSink, *, suppress_scope: Callable[[], bool] | None = None
) -> None:
    """Atomically replace the process-wide sink used by Core probes."""

    if sink is None:
        raise TypeError("sink must implement TraceSink")

    global _STATE
    if sink is NULL_TRACE_SINK and suppress_scope is None:
        _STATE = _NULL_STATE
        return
    _STATE = TraceGate(sink, suppress_scope or _never_suppress)


def reset_trace_sink() -> None:
    """Restore the process-wide no-op sink."""

    global _STATE
    _STATE = _NULL_STATE


def trace_scope(
    name: str,
    *,
    ctx: Mapping[str, Any] | None = None,
    slots: Sequence[str] | None = None,
    attrs: Mapping[str, Any] | None = None,
    capture_safe: bool = False,
) -> TraceScope:
    """Return an active scope when the installed sink accepts this event."""

    gate = prepare_trace_scope(name, capture_safe=capture_safe)
    return open_trace_scope(gate, name, ctx=ctx, slots=slots, attrs=attrs)


def open_trace_scope(
    gate: TraceGate | None,
    name: str,
    *,
    ctx: Mapping[str, Any] | None = None,
    slots: Sequence[str] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> TraceScope:
    """Open a scope from a previously accepted gate without evaluating it again."""

    if gate is None:
        return _NOOP_SCOPE
    return _NonSuppressingTraceScope(gate.sink.scope(name, ctx=ctx, slots=slots, attrs=attrs))


_P = ParamSpec("_P")
_R = TypeVar("_R")
_ContextFactory = Callable[..., Mapping[str, Any] | None]
_ResultHook = Callable[[Any, TraceScope], None]


def scoped_forward(
    name: str,
    *,
    capture_safe: bool = False,
    ctx_factory: _ContextFactory | None = None,
    result_hook: _ResultHook | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap a callable in a lazily enabled trace scope."""

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if (
            inspect.iscoroutinefunction(fn)
            or inspect.isasyncgenfunction(fn)
            or inspect.isgeneratorfunction(fn)
        ):
            raise TypeError("scoped_forward supports synchronous callables only")

        @wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            gate = prepare_trace_scope(name, capture_safe=capture_safe)
            if gate is None:
                return fn(*args, **kwargs)

            ctx = ctx_factory(*args, **kwargs) if ctx_factory is not None else None
            scope_manager = gate.sink.scope(name, ctx=ctx)
            active_scope = scope_manager.__enter__()
            try:
                result = fn(*args, **kwargs)
                if result_hook is not None:
                    result_hook(result, active_scope)
            except BaseException as model_error:
                try:
                    scope_manager.__exit__(
                        type(model_error), model_error, model_error.__traceback__
                    )
                except Exception as trace_error:
                    model_error.add_note(f"trace scope cleanup also failed: {trace_error!r}")
                raise
            else:
                scope_manager.__exit__(None, None, None)
                return result

        setattr(wrapper, "__megatron_trace_event__", name)
        return wrapper

    return decorator


__all__ = [
    "NULL_TRACE_SINK",
    "NullTraceSink",
    "TraceGate",
    "TraceScope",
    "TraceSink",
    "install_trace_sink",
    "open_trace_scope",
    "prepare_trace_scope",
    "reset_trace_sink",
    "scoped_forward",
    "trace_is_enabled",
    "trace_scope",
]
