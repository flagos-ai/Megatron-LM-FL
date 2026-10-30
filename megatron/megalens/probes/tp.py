# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-light trace helpers for direct tensor-parallel Linear collectives."""

# BEGIN MEGALENS OBSERVABILITY
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import count
from typing import Any, ContextManager, Iterator

from megatron.core.observability import (
    TraceGate,
    open_trace_scope,
    prepare_trace_scope,
    trace_is_enabled,
)
from megatron.core.utils import get_process_group_peer_ranks

_NOOP_SCOPE = nullcontext()
_NOOP_ASYNC_SCOPE = nullcontext(None)

_ASYNC_LAUNCH_EVENT = "tp-linear-async-launch"
_ASYNC_COMPLETE_EVENT = "tp-linear-async-complete"
_ASYNC_OPERATION_SEQUENCE = count(1)


def _tp_allreduce_context(input_, group_size: int) -> dict[str, object]:
    """Build TP all-reduce metadata only after the probe gate accepts it."""
    return {
        "data_bytes": int(input_.numel() * input_.element_size()),
        "group_size": group_size,
        "op": "all_reduce",
        "timing_phase": "collective_call",
        "payload_role": "inplace_input_output",
    }


def _tp_all_gather_context(
    input_, group_size: int, dim: str, split_sizes=None
) -> dict[str, object]:
    """Build the source-compatible TP all-gather metadata after its gate accepts."""
    ctx = {
        "data_bytes": int(input_.numel() * input_.element_size()),
        "group_size": group_size,
        "op": "all-gather",
        "dim": dim,
    }
    if split_sizes is not None:
        ctx["split_sizes"] = split_sizes
    return ctx


def _tp_reduce_scatter_context(
    input_, group_size: int, dim: str, split_sizes=None
) -> dict[str, object]:
    """Build source-compatible TP reduce-scatter metadata after its gate accepts."""
    ctx = {
        "data_bytes": int(input_.numel() * input_.element_size()),
        "group_size": group_size,
        "op": "reduce-scatter",
        "dim": dim,
    }
    if split_sizes is not None:
        ctx["split_sizes"] = split_sizes
    return ctx


@dataclass(frozen=True, slots=True)
class _AsyncLinearOperation:
    """Scalar-only identity for one direct TP Linear asynchronous request."""

    operation_id: str
    collective_op: str
    data_bytes: int | None
    group_size: int | None
    dim: str | None
    launch_site: str
    payload_role: str

    def trace_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "operation_id": self.operation_id,
            "operation_id_scope": "rank_local",
            "execution_route": "local_linear_direct_async",
            "collective_op": self.collective_op,
            "data_bytes": self.data_bytes,
            "group_size": self.group_size,
            "launch_site": self.launch_site,
            "pass_direction": "backward",
            "payload_role": self.payload_role,
        }
        if self.dim is not None:
            fields["dim"] = self.dim
        return fields


@dataclass(frozen=True, slots=True)
class _AsyncLinearObservation:
    """Invocation-local correlation token; it never owns a tensor or Work."""

    launch_observed: bool
    operation: _AsyncLinearOperation


def _best_effort_tensor_bytes(tensor: Any) -> int | None:
    try:
        return int(tensor.numel() * tensor.element_size())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _best_effort_group_size(group: Any) -> int | None:
    try:
        return int(group.size())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _record_scope_error(scope: Any, outcome_key: str, error: BaseException) -> None:
    """Best-effort error metadata that cannot replace the model/backend error."""

    try:
        scope.set(outcome_key, False)
        scope.set("error_type", type(error).__name__)
    except Exception as trace_error:
        error.add_note(f"trace outcome recording also failed: {trace_error!r}")


def _build_async_linear_observation(
    input_tensor: Any,
    group: Any,
    *,
    collective_op: str,
    dim: str | None,
    launch_site: str,
    payload_role: str,
) -> tuple[TraceGate | None, _AsyncLinearObservation] | None:
    launch_gate = prepare_trace_scope(_ASYNC_LAUNCH_EVENT)
    completion_enabled = trace_is_enabled(_ASYNC_COMPLETE_EVENT)
    if launch_gate is None and not completion_enabled:
        return None

    operation = _AsyncLinearOperation(
        operation_id=f"tp-linear:{next(_ASYNC_OPERATION_SEQUENCE)}",
        collective_op=collective_op,
        data_bytes=_best_effort_tensor_bytes(input_tensor),
        group_size=_best_effort_group_size(group),
        dim=dim,
        launch_site=launch_site,
        payload_role=payload_role,
    )
    observation = _AsyncLinearObservation(
        launch_observed=launch_gate is not None, operation=operation
    )
    return launch_gate, observation


@contextmanager
def _active_async_linear_launch_scope(
    launch_gate: TraceGate, observation: _AsyncLinearObservation
) -> Iterator[_AsyncLinearObservation]:
    context = {
        **observation.operation.trace_fields(),
        "async_op": True,
        "completion_included": False,
        "timing_phase": "launch_attempt",
    }
    with open_trace_scope(
        launch_gate, "tp-linear-async-launch", ctx=context, slots=("api_returned", "error_type")
    ) as scope:
        try:
            yield observation
        except BaseException as launch_error:
            _record_scope_error(scope, "api_returned", launch_error)
            raise
        else:
            scope.set("api_returned", True)


def async_linear_collective_launch_scope(
    input_tensor: Any,
    group: Any,
    *,
    collective_op: str,
    dim: str | None,
    launch_site: str,
    payload_role: str,
) -> ContextManager[_AsyncLinearObservation | None]:
    """Observe one existing async TP Linear API launch without owning its Work.

    If both lifecycle gates are closed here, a gate enabled later in the same
    Work lifetime does not retroactively create a correlation token.
    """

    prepared = _build_async_linear_observation(
        input_tensor,
        group,
        collective_op=collective_op,
        dim=dim,
        launch_site=launch_site,
        payload_role=payload_role,
    )
    if prepared is None:
        return _NOOP_ASYNC_SCOPE
    launch_gate, observation = prepared
    if launch_gate is None:
        return nullcontext(observation)
    return _active_async_linear_launch_scope(launch_gate, observation)


def wait_async_linear_collective(
    request: Any,
    observation: _AsyncLinearObservation | None,
    *,
    completion_site: str,
    wait_role: str,
    terminal: bool = True,
):
    """Call one existing Work.wait and optionally emit its correlated boundary."""

    if observation is None:
        return request.wait()

    completion_gate = prepare_trace_scope(_ASYNC_COMPLETE_EVENT)
    if completion_gate is None:
        return request.wait()

    context = {
        **observation.operation.trace_fields(),
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "work_wait",
        "completion_site": completion_site,
        "duration_attribution": "per_request",
        "global_device_completion_guaranteed": False,
        "host_blocking_guaranteed": False,
        "launch_observed": observation.launch_observed,
        "op": "wait",
        "terminal": bool(terminal),
        "timing_phase": "stream_dependency",
        "wait_role": wait_role,
    }
    with open_trace_scope(
        completion_gate, "tp-linear-async-complete", ctx=context, slots=("completed", "error_type")
    ) as scope:
        try:
            result = request.wait()
        except BaseException as wait_error:
            _record_scope_error(scope, "completed", wait_error)
            raise
        scope.set("completed", result is not False)
    return result


@contextmanager
def _active_sync_linear_all_gather_scope(
    gate: TraceGate, trace_context: dict[str, object], group
) -> Iterator[None]:
    with open_trace_scope(
        gate, "tp-all-gather-first", ctx=trace_context, slots=("group",)
    ) as scope:
        yield
        if scope.get("op") == "all-gather":
            scope.set("group", get_process_group_peer_ranks(group))


def sync_linear_all_gather_scope(input_tensor, group) -> ContextManager[None]:
    """Observe one direct synchronous first-dimension AllGather physical leaf."""

    gate = prepare_trace_scope("tp-all-gather-first")
    if gate is None:
        return _NOOP_SCOPE
    trace_context = {
        "data_bytes": int(input_tensor.numel() * input_tensor.element_size()),
        "group_size": int(group.size()),
        "op": "all-gather",
        "dim": "first",
    }
    return _active_sync_linear_all_gather_scope(gate, trace_context, group)


@contextmanager
def _active_mapping_scope(
    gate: TraceGate,
    name: str,
    context: dict[str, object],
    group: Any,
    *,
    group_before: bool = False,
) -> Iterator[None]:
    with open_trace_scope(gate, name, ctx=context, slots=("group",)) as scope:
        if group_before and scope.get("op") == "all_reduce":
            scope.set("group", get_process_group_peer_ranks(group))
        yield
        if name != "tp-allreduce":
            scope.set("group", get_process_group_peer_ranks(group))


def all_reduce_scope(name: str, input_: Any, group: Any, group_size: int) -> ContextManager[None]:
    """Observe the native all-reduce, including its source single-rank no-op."""
    gate = prepare_trace_scope(name)
    if gate is None:
        return _NOOP_SCOPE
    return _active_mapping_scope(
        gate, name, _tp_allreduce_context(input_, group_size), group, group_before=group_size > 1
    )


def all_gather_scope(
    name: str, input_: Any, group: Any, group_size: int, dim: str, split_sizes: Any = None
) -> ContextManager[None]:
    """Observe one native mapping all-gather with lazy metadata and peer slots."""
    gate = prepare_trace_scope(name)
    if gate is None:
        return _NOOP_SCOPE
    return _active_mapping_scope(
        gate, name, _tp_all_gather_context(input_, group_size, dim, split_sizes), group
    )


def reduce_scatter_scope(
    name: str, input_: Any, group: Any, group_size: int, dim: str, split_sizes: Any = None
) -> ContextManager[None]:
    """Observe one native mapping reduce-scatter without changing its branches."""
    gate = prepare_trace_scope(name)
    if gate is None:
        return _NOOP_SCOPE
    return _active_mapping_scope(
        gate, name, _tp_reduce_scatter_context(input_, group_size, dim, split_sizes), group
    )


__all__ = [
    "async_linear_collective_launch_scope",
    "sync_linear_all_gather_scope",
    "wait_async_linear_collective",
]
# END MEGALENS OBSERVABILITY
