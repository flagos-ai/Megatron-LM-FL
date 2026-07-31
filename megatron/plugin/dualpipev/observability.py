# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace metadata for DualPipeV schedule phases."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from itertools import count
from threading import RLock
from typing import Any

from megatron.core.observability import open_trace_scope, prepare_trace_scope, trace_is_enabled

WARMUP = "warmup"
STEADY = "steady"
COOLDOWN = "cooldown"

A2A_LAUNCH_EVENT = "ep-alltoall-async-launch"
A2A_COMPLETE_EVENT = "ep-alltoall-async-complete"

_A2A_SEQUENCE = count(1)
_A2A_WORK_OBSERVATIONS: dict[int, "_A2AWorkObservation"] = {}
_A2A_WORK_OBSERVATION_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class _A2AOperation:
    """Rank-local identity and payload metadata for one DualPipeV A2A Work."""

    operation_id: str
    layer: Any
    ep_size: Any
    tp_size: Any
    data_bytes: Any
    group_size: int
    pass_direction: str
    logical_phase: str
    payload_role: str

    def trace_fields(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "request_id": self.operation_id,
            "operation_id_scope": "rank_local",
            "layer": self.layer,
            "comm_type": "ep-alltoall",
            "dispatcher": "alltoall",
            "data_bytes": self.data_bytes,
            "group_size": self.group_size,
            "ep_size": self.ep_size,
            "tp_size": self.tp_size,
            "execution_route": "dualpipev_fb_overlap",
            "pass_direction": self.pass_direction,
            "logical_phase": self.logical_phase,
            "payload_role": self.payload_role,
            "transport_api": "all_to_all_single",
        }


@dataclass(frozen=True, slots=True)
class _A2AObservation:
    launch_gate: Any
    operation: _A2AOperation
    completion_enabled: bool


@dataclass(slots=True)
class _A2AWorkObservation:
    request_ref: Any
    operation: _A2AOperation | None


def _best_effort_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _next_a2a_operation_id() -> str:
    return f"dualpipev-a2a:{next(_A2A_SEQUENCE)}"


def _build_a2a_operation(
    input_tensor, *, trace_owner, group_size, pass_direction, logical_phase, payload_role
):
    dispatcher = getattr(getattr(trace_owner, "mlp", None), "token_dispatcher", None)
    try:
        data_bytes = int(input_tensor.numel() * input_tensor.element_size())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        data_bytes = None
    return _A2AOperation(
        operation_id=_next_a2a_operation_id(),
        layer=getattr(trace_owner, "layer_number", None),
        ep_size=_best_effort_int(getattr(dispatcher, "ep_size", None)),
        tp_size=_best_effort_int(getattr(dispatcher, "tp_size", None)),
        data_bytes=data_bytes,
        group_size=int(group_size),
        pass_direction=pass_direction,
        logical_phase=logical_phase,
        payload_role=payload_role,
    )


def prepare_async_all_to_all_observation(
    input_tensor, *, trace_owner, group_size, pass_direction, logical_phase, payload_role
):
    """Prepare one lazy launch/completion observation without touching the Work."""
    if (
        trace_owner is None
        or pass_direction is None
        or logical_phase is None
        or payload_role is None
    ):
        return None
    launch_gate = prepare_trace_scope("ep-alltoall-async-launch")
    completion_enabled = trace_is_enabled("ep-alltoall-async-complete")
    if launch_gate is None and not completion_enabled:
        return None
    operation = _build_a2a_operation(
        input_tensor,
        trace_owner=trace_owner,
        group_size=group_size,
        pass_direction=pass_direction,
        logical_phase=logical_phase,
        payload_role=payload_role,
    )
    return _A2AObservation(
        launch_gate=launch_gate, operation=operation, completion_enabled=completion_enabled
    )


def async_all_to_all_launch_scope(observation: _A2AObservation):
    """Open the API-launch interval for one prepared DualPipeV A2A operation."""
    context = {
        **observation.operation.trace_fields(),
        "async_op": True,
        "completion_included": False,
        "stage": "ep_request_launch",
        "timing_phase": "async_dispatch",
    }
    return open_trace_scope(observation.launch_gate, "ep-alltoall-async-launch", ctx=context)


def register_async_all_to_all_work(request: Any, observation: _A2AObservation) -> None:
    """Best-effort weak registration that preserves the exact backend Work."""
    if request is None or not observation.completion_enabled:
        return

    request_key = id(request)

    def remove_observation(request_ref: Any) -> None:
        with _A2A_WORK_OBSERVATION_LOCK:
            current = _A2A_WORK_OBSERVATIONS.get(request_key)
            if current is not None and current.request_ref is request_ref:
                _A2A_WORK_OBSERVATIONS.pop(request_key, None)

    try:
        request_ref = weakref.ref(request, remove_observation)
    except TypeError:
        return

    with _A2A_WORK_OBSERVATION_LOCK:
        current = _A2A_WORK_OBSERVATIONS.get(request_key)
        if current is not None and current.request_ref() is request:
            if (
                current.operation is not None
                and current.operation.operation_id != observation.operation.operation_id
            ):
                current.operation = None
            return
        _A2A_WORK_OBSERVATIONS[request_key] = _A2AWorkObservation(
            request_ref=request_ref, operation=observation.operation
        )


def _get_async_all_to_all_operation(request: Any) -> _A2AOperation | None:
    request_key = id(request)
    with _A2A_WORK_OBSERVATION_LOCK:
        observation = _A2A_WORK_OBSERVATIONS.get(request_key)
        if observation is None or observation.request_ref() is not request:
            _A2A_WORK_OBSERVATIONS.pop(request_key, None)
            return None
        if observation.operation is None:
            _A2A_WORK_OBSERVATIONS.pop(request_key, None)
            return None
        return observation.operation


def _discard_async_all_to_all_operation(request: Any, operation: _A2AOperation) -> None:
    request_key = id(request)
    with _A2A_WORK_OBSERVATION_LOCK:
        observation = _A2A_WORK_OBSERVATIONS.get(request_key)
        if (
            observation is not None
            and observation.request_ref() is request
            and observation.operation is operation
        ):
            _A2A_WORK_OBSERVATIONS.pop(request_key, None)


def _async_all_to_all_wait_context(
    operation: _A2AOperation, *, terminal: bool, completion_site: str, timeout_supplied: bool
) -> dict[str, Any]:
    return {
        **operation.trace_fields(),
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "work_wait",
        "completion_site": completion_site,
        "duration_attribution": "per_request",
        "host_blocking_guaranteed": False,
        "op": "wait",
        "stage": "ep_request_completion",
        "terminal": terminal,
        "timeout_supplied": timeout_supplied,
        "timing_phase": "stream_dependency",
        "wait_role": "terminal" if terminal else "dependency",
    }


def wait_async_all_to_all(
    request: Any,
    *args: Any,
    terminal: bool = True,
    completion_site: str = "dualpipev_overlap_caller",
    **kwargs: Any,
):
    """Wait on a raw A2A Work and emit a correlated lifecycle completion."""
    if not _A2A_WORK_OBSERVATIONS:
        return request.wait(*args, **kwargs)

    operation = _get_async_all_to_all_operation(request)
    if operation is None:
        return request.wait(*args, **kwargs)

    completion_gate = prepare_trace_scope("ep-alltoall-async-complete")
    if completion_gate is None:
        result = request.wait(*args, **kwargs)
        if terminal and result is not False:
            _discard_async_all_to_all_operation(request, operation)
        return result

    completion_scope = open_trace_scope(
        completion_gate,
        "ep-alltoall-async-complete",
        ctx=_async_all_to_all_wait_context(
            operation,
            terminal=terminal,
            completion_site=completion_site,
            timeout_supplied=bool(args or "timeout" in kwargs),
        ),
        slots=("completed", "error_type"),
    )
    with completion_scope as completion:
        try:
            result = request.wait(*args, **kwargs)
        except BaseException as wait_error:
            completion.set("completed", False)
            completion.set("error_type", type(wait_error).__name__)
            raise
        succeeded = result is not False
        completion.set("completed", succeeded)
    if terminal and succeeded:
        _discard_async_all_to_all_operation(request, operation)
    return result


def _operation_id(microbatch, dualpipev_stage):
    if microbatch is None or dualpipev_stage is None:
        return None
    return f"pp:microbatch={microbatch}:dualpipev_stage={dualpipev_stage}"


def _phase_context(
    *,
    current_microbatch,
    dualpipev_stage,
    schedule_phase,
    is_first_microbatch,
    is_last_stage,
    uses_model_graph,
):
    return {
        "current_microbatch": current_microbatch,
        "vp_stage": None,
        "dualpipev_stage": dualpipev_stage,
        "is_first_microbatch": is_first_microbatch,
        "is_last_stage": is_last_stage,
        "operation_id": _operation_id(current_microbatch, dualpipev_stage),
        "schedule": "dualpipev",
        "schedule_phase": schedule_phase,
        "uses_model_graph": uses_model_graph,
        "timing_phase": "framework_phase",
    }


def phase_scope(
    name,
    *,
    current_microbatch,
    dualpipev_stage,
    schedule_phase,
    is_first_microbatch,
    is_last_stage,
    uses_model_graph,
    slots=(),
):
    """Open one standalone DualPipeV phase after a single lazy gate."""
    gate = prepare_trace_scope(name)
    ctx = None
    if gate is not None:
        ctx = _phase_context(
            current_microbatch=current_microbatch,
            dualpipev_stage=dualpipev_stage,
            schedule_phase=schedule_phase,
            is_first_microbatch=is_first_microbatch,
            is_last_stage=is_last_stage,
            uses_model_graph=uses_model_graph,
        )
    return open_trace_scope(gate, name, ctx=ctx, slots=slots)


def _combined_context(
    *,
    forward_microbatch,
    backward_microbatch,
    forward_dualpipev_stage,
    backward_dualpipev_stage,
    schedule_phase,
):
    forward_operation_id = _operation_id(forward_microbatch, forward_dualpipev_stage)
    backward_operation_id = _operation_id(backward_microbatch, backward_dualpipev_stage)
    return {
        "operation_id": (
            "pp-combined:" f"forward={forward_operation_id}:backward={backward_operation_id}"
        ),
        "forward_operation_id": forward_operation_id,
        "backward_operation_id": backward_operation_id,
        "forward_microbatch": forward_microbatch,
        "backward_microbatch": backward_microbatch,
        "forward_vp_stage": None,
        "backward_vp_stage": None,
        "forward_dualpipev_stage": forward_dualpipev_stage,
        "backward_dualpipev_stage": backward_dualpipev_stage,
        "execution_mode": "combined",
        "overlap_active": True,
        "schedule": "dualpipev",
        "schedule_phase": schedule_phase,
        "uses_model_graph": True,
        "timing_phase": "framework_phase",
    }


def combined_scope(
    *,
    forward_microbatch,
    backward_microbatch,
    forward_dualpipev_stage,
    backward_dualpipev_stage,
    schedule_phase,
):
    """Open one fused forward/model-graph-backward phase."""
    name = "combined-forward-backward-step"
    gate = prepare_trace_scope(name)
    ctx = None
    if gate is not None:
        ctx = _combined_context(
            forward_microbatch=forward_microbatch,
            backward_microbatch=backward_microbatch,
            forward_dualpipev_stage=forward_dualpipev_stage,
            backward_dualpipev_stage=backward_dualpipev_stage,
            schedule_phase=schedule_phase,
        )
    return open_trace_scope(gate, name, ctx=ctx)


def grad_sync_scope():
    """Open the source-equivalent final gradient synchronization phase."""
    name = "grad-sync"
    gate = prepare_trace_scope(name)
    ctx = {"schedule": "dualpipev", "timing_phase": "framework_phase"} if gate is not None else None
    return open_trace_scope(gate, name, ctx=ctx)


__all__ = [
    "A2A_COMPLETE_EVENT",
    "A2A_LAUNCH_EVENT",
    "COOLDOWN",
    "STEADY",
    "WARMUP",
    "async_all_to_all_launch_scope",
    "combined_scope",
    "grad_sync_scope",
    "phase_scope",
    "prepare_async_all_to_all_observation",
    "register_async_all_to_all_work",
    "wait_async_all_to_all",
]
