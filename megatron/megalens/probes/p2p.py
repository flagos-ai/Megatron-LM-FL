# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Implicit P2P launch, native-wait observation, and weak Work correlation."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from itertools import count
from threading import RLock
from typing import Any, Callable, Optional, Union

import torch

from megatron.core.observability import open_trace_scope, prepare_trace_scope, trace_is_enabled

# BEGIN MEGALENS OBSERVABILITY
_P2P_BATCH_SEQUENCE = count(1)
_P2P_LAUNCH_GATE_UNSET = object()


def _p2p_observation_enabled(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    transport_api: str,
    launch_enabled: bool = False,
) -> bool:
    """Check relevant launch/wait events before allocating operation metadata."""
    if (
        tensor_send_prev is None
        and tensor_recv_prev is None
        and tensor_send_next is None
        and tensor_recv_next is None
    ):
        return False
    if launch_enabled:
        return True
    if transport_api == "ring_exchange":
        return False
    if transport_api == "batch_isend_irecv" and trace_is_enabled("p2p-batch-complete"):
        return True
    if tensor_send_prev is not None and trace_is_enabled("send-backward"):
        return True
    if tensor_recv_prev is not None and trace_is_enabled("recv-forward"):
        return True
    if tensor_send_next is not None and trace_is_enabled("send-forward"):
        return True
    if tensor_recv_next is not None and trace_is_enabled("recv-backward"):
        return True
    return False


def _get_distributed_backend(group) -> Optional[str]:
    """Best-effort backend metadata that never controls communication."""
    try:
        return str(torch.distributed.get_backend(group))
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class _P2POperation:
    """Rank-local identity and payload metadata for one logical P2P Work."""

    batch_id: str
    key: str
    event_name: str
    direction: str
    pipeline_direction: str
    peer_rank: int
    data_bytes: int
    backend: Optional[str]
    transport_api: str
    completion_mode: str

    @property
    def operation_id(self) -> str:
        return f"{self.batch_id}:{self.key}"

    @property
    def request_id(self) -> Optional[str]:
        if self.transport_api == "ring_exchange":
            return None
        return self.operation_id

    def trace_fields(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "request_id": self.request_id,
            "direction": self.direction,
            "pipeline_direction": self.pipeline_direction,
            "peer_rank": self.peer_rank,
            "data_bytes": self.data_bytes,
            "microbatch": None,
            "comm_type": "p2p",
            "backend": self.backend,
            "transport_api": self.transport_api,
            "completion_mode": self.completion_mode,
        }


@dataclass(slots=True)
class _P2PWorkObservation:
    """Weakly correlate one raw backend Work with its logical P2P operation."""

    request_ref: Any
    operation: Optional[_P2POperation]


def _build_p2p_operations(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
    backend: Optional[str],
    transport_api: str,
    wait_on_reqs: bool,
    backends_by_key: Optional[dict[str, Optional[str]]] = None,
) -> list[_P2POperation]:
    entries = []
    if tensor_send_prev is not None:
        entries.append(
            (
                "send_prev",
                "send-backward",
                "send",
                "backward",
                prev_pipeline_rank,
                tensor_send_prev.numel() * tensor_send_prev.element_size(),
            )
        )
    if tensor_recv_prev is not None:
        entries.append(
            (
                "recv_prev",
                "recv-forward",
                "recv",
                "forward",
                prev_pipeline_rank,
                tensor_recv_prev.numel() * tensor_recv_prev.element_size(),
            )
        )
    if tensor_send_next is not None:
        entries.append(
            (
                "send_next",
                "send-forward",
                "send",
                "forward",
                next_pipeline_rank,
                tensor_send_next.numel() * tensor_send_next.element_size(),
            )
        )
    if tensor_recv_next is not None:
        entries.append(
            (
                "recv_next",
                "recv-backward",
                "recv",
                "backward",
                next_pipeline_rank,
                tensor_recv_next.numel() * tensor_recv_next.element_size(),
            )
        )
    if not entries:
        return []

    batch_id = f"p2p:{next(_P2P_BATCH_SEQUENCE)}"
    completion_mode = (
        "inline"
        if transport_api == "ring_exchange"
        else "internal_wait" if wait_on_reqs else "external_wait"
    )
    return [
        _P2POperation(
            batch_id=batch_id,
            key=key,
            event_name=event_name,
            direction=direction,
            pipeline_direction=pipeline_direction,
            peer_rank=peer_rank,
            data_bytes=int(data_bytes),
            backend=(backends_by_key or {}).get(key, backend),
            transport_api=transport_api,
            completion_mode=completion_mode,
        )
        for key, event_name, direction, pipeline_direction, peer_rank, data_bytes in entries
    ]


def _p2p_backend_fields(operations: list[_P2POperation]) -> dict[str, Any]:
    """Summarize operation backends without treating missing metadata as a concrete value."""
    backends = sorted(
        {operation.backend for operation in operations if operation.backend is not None}
    )
    backend_complete = all(operation.backend is not None for operation in operations)
    if not backend_complete:
        backend = None
    elif len(backends) == 1:
        backend = backends[0]
    else:
        backend = "mixed"
    return {"backend": backend, "backends": backends, "backend_complete": backend_complete}


def _p2p_launch_context(
    p2p_func: Callable[..., Any], operations: list[_P2POperation], call_kwargs: dict[str, Any]
) -> dict[str, Any]:
    del p2p_func, call_kwargs
    first = operations[0]
    request_pairing = {
        "isend_irecv": "key",
        "batch_isend_irecv": "backend_dependent",
        "ring_exchange": "none",
    }.get(first.transport_api, "unknown")
    context = {
        "batch_id": first.batch_id,
        "comm_type": "p2p-launch",
        "timing_phase": ("inline_api_call" if first.transport_api == "ring_exchange" else "launch"),
        **_p2p_backend_fields(operations),
        "transport_api": first.transport_api,
        "request_pairing": request_pairing,
        "completion_mode": first.completion_mode,
        "completion_included": False,
        "operation_count": len(operations),
        "operations": [operation.trace_fields() for operation in operations],
    }
    if first.transport_api == "ring_exchange":
        context.update(
            {
                "api_return_included": True,
                "completion_guarantee": "api_return_observed",
                "completion_kind": "inline_api_return",
                "device_completion_guaranteed": False,
                "duration_attribution": "shared_nonexclusive",
                "host_blocking_guaranteed": False,
                "operation_ids": [operation.operation_id for operation in operations],
                "operation_id_scope": "rank_local",
                "physical_request_count": 0,
                "stage": "p2p_inline_api",
            }
        )
    return context


def _launch_p2p(
    p2p_func: Callable[..., Any],
    operations: list[_P2POperation],
    call_kwargs: dict[str, Any],
    *,
    launch_gate: Any = _P2P_LAUNCH_GATE_UNSET,
):
    if launch_gate is _P2P_LAUNCH_GATE_UNSET:
        launch_gate = prepare_trace_scope("p2p-launch")
    if launch_gate is None:
        return p2p_func(**call_kwargs)

    inline_completion = operations[0].transport_api == "ring_exchange"
    launch_scope = open_trace_scope(
        launch_gate,
        "p2p-launch",
        ctx=_p2p_launch_context(p2p_func, operations, call_kwargs),
        slots=("completed", "error_type") if inline_completion else None,
    )
    with launch_scope as launch:
        try:
            result = p2p_func(**call_kwargs)
        except BaseException as launch_error:
            if inline_completion:
                launch.set("completed", False)
                launch.set("error_type", type(launch_error).__name__)
            raise
        if inline_completion:
            launch.set("completed", True)
        return result


setattr(_launch_p2p, "__megatron_trace_event__", "p2p-launch")


def _p2p_wait_context(
    request: Any, operation: _P2POperation, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    del request
    return {
        **operation.trace_fields(),
        "batch_id": operation.batch_id,
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "work_wait",
        "completion_site": (
            "communicate_internal_wait"
            if operation.completion_mode == "internal_wait"
            else "exposed_request_wait"
        ),
        "duration_attribution": "per_request",
        "host_blocking_guaranteed": False,
        "op": "wait",
        "operation_count": 1,
        "operation_ids": [operation.operation_id],
        "operation_id_scope": "rank_local",
        "request_pairing": (
            "position" if operation.transport_api == "batch_isend_irecv" else "key"
        ),
        "stage": "p2p_request_completion",
        "timeout_supplied": bool(args or "timeout" in kwargs),
        "timing_phase": "stream_dependency",
    }


def _wait_p2p_request_in_scope(request: Any, completion_scope: Any, *args: Any, **kwargs: Any):
    with completion_scope as completion:
        try:
            result = request.wait(*args, **kwargs)
        except BaseException as wait_error:
            completion.set("completed", False)
            completion.set("error_type", type(wait_error).__name__)
            raise
        completion.set("completed", result is not False)
        return result


def _wait_send_forward(request: Any, operation: _P2POperation, *args: Any, **kwargs: Any):
    completion_gate = prepare_trace_scope("send-forward")
    if completion_gate is None:
        return request.wait(*args, **kwargs)
    completion_scope = open_trace_scope(
        completion_gate,
        "send-forward",
        ctx=_p2p_wait_context(request, operation, *args, **kwargs),
        slots=("completed", "error_type"),
    )
    return _wait_p2p_request_in_scope(request, completion_scope, *args, **kwargs)


def _wait_recv_forward(request: Any, operation: _P2POperation, *args: Any, **kwargs: Any):
    completion_gate = prepare_trace_scope("recv-forward")
    if completion_gate is None:
        return request.wait(*args, **kwargs)
    completion_scope = open_trace_scope(
        completion_gate,
        "recv-forward",
        ctx=_p2p_wait_context(request, operation, *args, **kwargs),
        slots=("completed", "error_type"),
    )
    return _wait_p2p_request_in_scope(request, completion_scope, *args, **kwargs)


def _wait_send_backward(request: Any, operation: _P2POperation, *args: Any, **kwargs: Any):
    completion_gate = prepare_trace_scope("send-backward")
    if completion_gate is None:
        return request.wait(*args, **kwargs)
    completion_scope = open_trace_scope(
        completion_gate,
        "send-backward",
        ctx=_p2p_wait_context(request, operation, *args, **kwargs),
        slots=("completed", "error_type"),
    )
    return _wait_p2p_request_in_scope(request, completion_scope, *args, **kwargs)


def _wait_recv_backward(request: Any, operation: _P2POperation, *args: Any, **kwargs: Any):
    completion_gate = prepare_trace_scope("recv-backward")
    if completion_gate is None:
        return request.wait(*args, **kwargs)
    completion_scope = open_trace_scope(
        completion_gate,
        "recv-backward",
        ctx=_p2p_wait_context(request, operation, *args, **kwargs),
        slots=("completed", "error_type"),
    )
    return _wait_p2p_request_in_scope(request, completion_scope, *args, **kwargs)


def _p2p_batch_wait_context(request: Any, operations: list[_P2POperation]) -> dict[str, Any]:
    del request
    first = operations[0]
    return {
        "batch_id": first.batch_id,
        "comm_type": "p2p",
        **_p2p_backend_fields(operations),
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "aggregate_work_wait",
        "completion_mode": first.completion_mode,
        "duration_attribution": "shared_nonexclusive",
        "host_blocking_guaranteed": False,
        "op": "wait",
        "operation_count": len(operations),
        "operation_ids": [operation.operation_id for operation in operations],
        "operation_id_scope": "rank_local",
        "operations": [operation.trace_fields() for operation in operations],
        "physical_request_count": 1,
        "request_id": f"{first.batch_id}:aggregate",
        "request_pairing": "aggregate",
        "stage": "batch_p2p_completion",
        "timing_phase": "stream_dependency",
        "transport_api": first.transport_api,
    }


def _wait_p2p_batch_request(request: Any, operations: list[_P2POperation]):
    completion_gate = prepare_trace_scope("p2p-batch-complete")
    if completion_gate is None:
        return request.wait()
    completion_scope = open_trace_scope(
        completion_gate,
        "p2p-batch-complete",
        ctx=_p2p_batch_wait_context(request, operations),
        slots=("completed", "error_type"),
    )
    with completion_scope as completion:
        try:
            result = request.wait()
        except BaseException as wait_error:
            completion.set("completed", False)
            completion.set("error_type", type(wait_error).__name__)
            raise
        completion.set("completed", result is not False)
        return result


def _p2p_batch_device_sync_context(
    operations: list[_P2POperation], physical_request_count: int, transport_api: str
) -> dict[str, Any]:
    operation_count = len(operations)
    if operation_count == 0:
        backend_fields = {"backend": None, "backends": [], "backend_complete": False}
        request_pairing = "none"
    else:
        backend_fields = _p2p_backend_fields(operations)
        if transport_api == "ring_exchange":
            request_pairing = "none"
        elif physical_request_count == operation_count:
            request_pairing = "position"
        elif physical_request_count == 1 and operation_count > 1:
            request_pairing = "aggregate"
        else:
            request_pairing = "unknown"
    return {
        "backend": backend_fields["backend"],
        "backends": backend_fields["backends"],
        "backend_complete": backend_fields["backend_complete"],
        "batch_id": operations[0].batch_id if operations else None,
        "comm_type": "p2p",
        "completion_guarantee": "host_after_current_device_synchronize",
        "completion_included": True,
        "completion_kind": "device_synchronize",
        "completion_site": "batch_p2p_sync_workaround",
        "device_scope": "current_device",
        "duration_attribution": "device_wide_nonexclusive",
        "has_p2p_operations": operation_count > 0,
        "op": "synchronize",
        "operation_count": operation_count,
        "operation_ids": [operation.operation_id for operation in operations],
        "operation_id_scope": "rank_local",
        "operations": [operation.trace_fields() for operation in operations],
        "physical_request_count": physical_request_count,
        "request_pairing": request_pairing,
        "stage": "batch_p2p_device_sync",
        "timing_phase": "device_synchronize",
        "transport_api": transport_api,
    }


def _synchronize_p2p_batch(
    sync_gate: Any,
    operations: list[_P2POperation],
    physical_request_count: int,
    transport_api: str,
    synchronize: Callable[[], Any],
):
    if sync_gate is None:
        return synchronize()
    sync_scope = open_trace_scope(
        sync_gate,
        "p2p-batch-device-sync",
        ctx=_p2p_batch_device_sync_context(operations, physical_request_count, transport_api),
        slots=(
            "completed",
            "device_completion_guaranteed",
            "error_type",
            "host_blocking_guaranteed",
        ),
    )
    with sync_scope as completion:
        try:
            result = synchronize()
        except BaseException as sync_error:
            completion.set("completed", False)
            completion.set("device_completion_guaranteed", False)
            completion.set("error_type", type(sync_error).__name__)
            completion.set("host_blocking_guaranteed", False)
            raise
        completion.set("completed", True)
        completion.set("device_completion_guaranteed", True)
        completion.set("host_blocking_guaranteed", True)
        return result


def _wait_p2p_request(request: Any, operation: Optional[_P2POperation], *args: Any, **kwargs: Any):
    if operation is not None:
        if operation.event_name == "send-forward":
            return _wait_send_forward(request, operation, *args, **kwargs)
        if operation.event_name == "recv-forward":
            return _wait_recv_forward(request, operation, *args, **kwargs)
        if operation.event_name == "send-backward":
            return _wait_send_backward(request, operation, *args, **kwargs)
        if operation.event_name == "recv-backward":
            return _wait_recv_backward(request, operation, *args, **kwargs)
    return request.wait(*args, **kwargs)


def wait_p2p_request(communicator: Any, request: Any, *args: Any, **kwargs: Any):
    """Wait through an optional communicator observation capability.

    Custom communicators without that capability keep their historical raw
    ``Work.wait`` behavior. Calling ``request.wait`` directly remains valid,
    but it bypasses directional completion observation.
    """
    observed_wait = getattr(communicator, "wait_p2p_request", None)
    if callable(observed_wait):
        return observed_wait(request, *args, **kwargs)
    return request.wait(*args, **kwargs)


def _p2p_batch_request_pairing(requests: Any, operation_count: int) -> str:
    """Classify batch Work cardinality without changing backend behavior."""
    request_count = len(requests)
    if request_count == operation_count:
        return "position"
    if request_count == 1 and operation_count > 1:
        return "aggregate"
    return "unknown"


# END MEGALENS OBSERVABILITY


def _register_p2p_request(self, request: Any, operation: _P2POperation) -> None:
    """Best-effort registration that never changes communication behavior."""
    request_key = id(request)
    owner_ref = weakref.ref(self)

    def remove_observation(request_ref: Any) -> None:
        owner = owner_ref()
        if owner is None:
            return
        with owner._p2p_work_observation_lock:
            current = owner._p2p_work_observations.get(request_key)
            if current is not None and current.request_ref is request_ref:
                owner._p2p_work_observations.pop(request_key, None)

    try:
        request_ref = weakref.ref(request, remove_observation)
    except TypeError:
        return

    with self._p2p_work_observation_lock:
        current = self._p2p_work_observations.get(request_key)
        if current is not None and current.request_ref() is request:
            if (
                current.operation is not None
                and current.operation.operation_id != operation.operation_id
            ):
                # One physical Work cannot be attributed to two individual operations.
                current.operation = None
            return
        self._p2p_work_observations[request_key] = _P2PWorkObservation(
            request_ref=request_ref, operation=operation
        )


def _register_p2p_requests(
    self, requests: Union[list[Any], dict[str, Any]], operations: list[_P2POperation]
) -> None:
    if isinstance(requests, list):
        if len(requests) != len(operations):
            return
        for request, operation in zip(requests, operations):
            if trace_is_enabled(operation.event_name):
                _register_p2p_request(self, request, operation)
        return

    operations_by_key = {operation.key: operation for operation in operations}
    for key, request in requests.items():
        operation = operations_by_key.get(key)
        if operation is not None and trace_is_enabled(operation.event_name):
            _register_p2p_request(self, request, operation)


def _next_p2p_wait_observation(self, request: Any) -> Optional[_P2POperation]:
    request_key = id(request)
    with self._p2p_work_observation_lock:
        observation = self._p2p_work_observations.get(request_key)
        if observation is None or observation.request_ref() is not request:
            self._p2p_work_observations.pop(request_key, None)
            return None
        if observation.operation is None:
            self._p2p_work_observations.pop(request_key, None)
            return None
        return observation.operation


def wait_registered_request(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
    """Wait on a raw P2P Work and emit a correlated completion when registered."""
    if not self._p2p_work_observations:
        return request.wait(*args, **kwargs)
    operation = _next_p2p_wait_observation(self, request)
    return _wait_p2p_request(request, operation, *args, **kwargs)


def initialize_communicator(owner: Any) -> None:
    """Initialize the existing weak correlation state without retaining Works."""
    owner._p2p_work_observations = {}
    owner._p2p_work_observation_lock = RLock()


class _P2PBatch:
    """One call-local observation; input references never escape the communicator call."""

    def __init__(
        self,
        transport_api: str,
        wait_on_reqs: bool,
        group_plan: list[tuple[str, Any]] | None,
        call_kwargs: dict[str, Any],
    ) -> None:
        self.transport_api = transport_api
        self.wait_on_reqs = wait_on_reqs
        self.group_plan = group_plan
        self.call_kwargs = call_kwargs
        has_operations = any(
            call_kwargs[key] is not None
            for key in (
                'tensor_send_prev',
                'tensor_recv_prev',
                'tensor_send_next',
                'tensor_recv_next',
            )
        )
        self.launch_gate = prepare_trace_scope('p2p-launch') if has_operations else None
        self.observe = _p2p_observation_enabled(
            tensor_send_prev=call_kwargs['tensor_send_prev'],
            tensor_recv_prev=call_kwargs['tensor_recv_prev'],
            tensor_send_next=call_kwargs['tensor_send_next'],
            tensor_recv_next=call_kwargs['tensor_recv_next'],
            transport_api=transport_api,
            launch_enabled=self.launch_gate is not None,
        )
        self.operations = []
        self._operations_cache = None
        self.physical_request_count = 0

    def ensure_operations(self) -> list[_P2POperation]:
        if self._operations_cache is not None:
            return self._operations_cache
        values = self.call_kwargs
        active_tensors = {
            'send_prev': values['tensor_send_prev'],
            'recv_prev': values['tensor_recv_prev'],
            'send_next': values['tensor_send_next'],
            'recv_next': values['tensor_recv_next'],
        }
        backends_by_key = (
            {
                key: _get_distributed_backend(group)
                for key, group in self.group_plan
                if active_tensors[key] is not None
            }
            if self.group_plan is not None
            else None
        )
        self._operations_cache = _build_p2p_operations(
            tensor_send_prev=values['tensor_send_prev'],
            tensor_recv_prev=values['tensor_recv_prev'],
            tensor_send_next=values['tensor_send_next'],
            tensor_recv_next=values['tensor_recv_next'],
            prev_pipeline_rank=values['prev_pipeline_rank'],
            next_pipeline_rank=values['next_pipeline_rank'],
            backend=_get_distributed_backend(values['group']),
            transport_api=self.transport_api,
            wait_on_reqs=self.wait_on_reqs,
            backends_by_key=backends_by_key,
        )
        return self._operations_cache

    def launch(self, function: Callable[..., Any]) -> Any:
        if self.observe:
            self.operations = self.ensure_operations()
            requests = _launch_p2p(
                function, self.operations, self.call_kwargs, launch_gate=self.launch_gate
            )
        else:
            requests = function(**self.call_kwargs)
        self.physical_request_count = len(requests)
        return requests

    def wait_requests(self, requests: Any) -> None:
        operations = self.operations
        if not operations:
            for request in requests if isinstance(requests, list) else requests.values():
                request.wait()
        elif isinstance(requests, list):
            pairing = _p2p_batch_request_pairing(requests, len(operations))
            if pairing == 'aggregate':
                _wait_p2p_batch_request(requests[0], operations)
            elif pairing == 'position':
                for request, operation in zip(requests, operations):
                    _wait_p2p_request(request, operation)
            else:
                for request in requests:
                    request.wait()
        else:
            by_key = {operation.key: operation for operation in operations}
            for key, request in requests.items():
                _wait_p2p_request(request, by_key.get(key))

    def register_requests(self, owner: Any, requests: Any) -> None:
        if self.operations:
            _register_p2p_requests(owner, requests, self.operations)

    def synchronize(self, synchronize: Callable[[], Any]) -> Any:
        gate = prepare_trace_scope('p2p-batch-device-sync')
        operations = self.ensure_operations() if gate is not None else []
        return _synchronize_p2p_batch(
            gate, operations, self.physical_request_count, self.transport_api, synchronize
        )


def launch_p2p(
    function: Callable[..., Any],
    *,
    transport_api: str,
    wait_on_reqs: bool,
    group_plan: list[tuple[str, Any]] | None = None,
    **call_kwargs: Any,
) -> tuple[Any, _P2PBatch]:
    """Invoke the existing launch and return its raw Works plus call-local observation."""
    observation = _P2PBatch(transport_api, wait_on_reqs, group_plan, call_kwargs)
    return observation.launch(function), observation
