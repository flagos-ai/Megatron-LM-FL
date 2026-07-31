# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import weakref
from dataclasses import dataclass
from functools import partial  # FlagScale Add
from itertools import count
from threading import RLock
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.observability import open_trace_scope, prepare_trace_scope, trace_is_enabled
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

# FlagScale Begin
from megatron.core.utils import get_pg_rank, get_pg_size, nvtx_decorator
from megatron.plugin.hetero.p2p_communication import (
    recv_backward_hetero,
    recv_forward_hetero,
    send_backward_hetero,
    send_backward_recv_forward_hetero,
    send_forward_hetero,
    send_forward_recv_backward_hetero,
)

# FlagScale End

# Types
Shape = Union[List[int], torch.Size]

# FlagScale Begin
from megatron.plugin.platform import get_platform

cur_platform = get_platform()
# FlagScale End


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
    operations: list[_P2POperation], physical_request_count: int
) -> dict[str, Any]:
    operation_count = len(operations)
    if operation_count == 0:
        backend_fields = {"backend": None, "backends": [], "backend_complete": False}
        request_pairing = "none"
    else:
        backend_fields = _p2p_backend_fields(operations)
        if physical_request_count == operation_count:
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
        "transport_api": "batch_isend_irecv",
    }


def _synchronize_p2p_batch(
    sync_gate: Any, operations: list[_P2POperation], physical_request_count: int
):
    if sync_gate is None:
        return cur_platform.synchronize()
    sync_scope = open_trace_scope(
        sync_gate,
        "p2p-batch-device-sync",
        ctx=_p2p_batch_device_sync_context(operations, physical_request_count),
        slots=(
            "completed",
            "device_completion_guaranteed",
            "error_type",
            "host_blocking_guaranteed",
        ),
    )
    with sync_scope as completion:
        try:
            result = cur_platform.synchronize()
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


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_group_plan(group: torch.distributed.ProcessGroup) -> list[tuple[str, Any]]:
    """Resolve the physical process group and launch order for individual P2P operations."""
    primary_group = group
    if group.size() == 2 and torch.distributed.get_backend(group) != 'ucc':
        # A second communicator lets opposite rank pairs overlap when PP size is two.
        alternate_group = torch.distributed.group.WORLD
    else:
        alternate_group = group

    if group.rank() % 2 == 0:
        return [
            ("send_next", primary_group),
            ("recv_prev", alternate_group),
            ("send_prev", primary_group),
            ("recv_next", alternate_group),
        ]
    return [
        ("recv_prev", primary_group),
        ("send_next", alternate_group),
        ("recv_next", primary_group),
        ("send_prev", alternate_group),
    ]


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
    group_plan: Optional[list[tuple[str, Any]]] = None,
):
    reqs = {}
    tensors = {
        "send_prev": tensor_send_prev,
        "recv_prev": tensor_recv_prev,
        "send_next": tensor_send_next,
        "recv_next": tensor_recv_next,
    }
    peer_ranks = {
        "send_prev": prev_pipeline_rank,
        "recv_prev": prev_pipeline_rank,
        "send_next": next_pipeline_rank,
        "recv_next": next_pipeline_rank,
    }
    resolved_group_plan = group_plan if group_plan is not None else _p2p_group_plan(group)
    for key, request_group in resolved_group_plan:
        tensor = tensors[key]
        if tensor is None:
            continue
        if key.startswith("send"):
            reqs[key] = torch.distributed.isend(
                tensor=tensor, dst=peer_ranks[key], group=request_group
            )
        else:
            reqs[key] = torch.distributed.irecv(
                tensor=tensor, src=peer_ranks[key], group=request_group
            )
    return reqs


def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


class P2PCommunicator:
    """P2P (Point-to-Point) Communicator for pipeline parallelism.

    This class handles communication between pipeline stages by managing
    tensor exchanges between consecutive stages in the pipeline.
    """

    def __init__(self, pp_group: dist.ProcessGroup, config: ModelParallelConfig):
        # Basic attrs
        self.pp_group = pp_group
        self.config = config
        self._p2p_work_observations: dict[int, _P2PWorkObservation] = {}
        self._p2p_work_observation_lock = RLock()
        # FlagScale Begin
        if not isinstance(self.pp_group, list):
            world_size = self.pp_group.size()
            curr_rank_in_pg = self.pp_group.rank()

            next_rank_pg = (curr_rank_in_pg + 1) % world_size
            prev_rank_pg = (curr_rank_in_pg - 1) % world_size

            self.next_rank: int | None = dist.get_global_rank(self.pp_group, next_rank_pg)
            self.prev_rank: int | None = dist.get_global_rank(self.pp_group, prev_rank_pg)
            self.virtual_pipeline_model_parallel_size = (
                config.virtual_pipeline_model_parallel_size
                if config.virtual_pipeline_model_parallel_size is not None
                else None
            )
        # FlagScale End

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
                    self._register_p2p_request(request, operation)
            return

        operations_by_key = {operation.key: operation for operation in operations}
        for key, request in requests.items():
            operation = operations_by_key.get(key)
            if operation is not None and trace_is_enabled(operation.event_name):
                self._register_p2p_request(request, operation)

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

    def wait_p2p_request(self, request: Any, *args: Any, **kwargs: Any):
        """Wait on a raw P2P Work and emit a correlated completion when registered."""
        if not self._p2p_work_observations:
            return request.wait(*args, **kwargs)
        operation = self._next_p2p_wait_observation(request)
        return _wait_p2p_request(request, operation, *args, **kwargs)

    @property
    def is_pp_first_stage(self) -> bool:
        """Return True if pp first stage."""
        return is_pp_first_stage(self.pp_group)

    @property
    def is_pp_last_stage(self) -> bool:
        """Return True if pp last stage."""
        return is_pp_last_stage(self.pp_group)

    @property
    def total_stages(self) -> int:
        """Return total number of pipeline stages."""
        return get_pg_size(self.pp_group)  # FlagScale Add

    @property
    def current_stage(self) -> int:
        """Return current pipeline stage index (0-indexed)."""
        return get_pg_rank(self.pp_group)  # FlagScale Add

    def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        """Communicate tensor shapes between stages. Used to communicate
        tensor shapes before the actual tensor communication happens.
        This is required when the sequence lengths across micro batches
        are not uniform.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if
                            set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if
                            set to None).
            recv_prev: boolean for whether tensor should be received from
                    previous rank.
            recv_next: boolean for whether tensor should be received from
                    next rank.
        Returns:
            (recv_prev_shape, recv_next_shape)
        """
        config = self.config
        recv_prev_shape_tensor = None
        recv_next_shape_tensor = None
        send_prev_shape_tensor = None
        send_next_shape_tensor = None
        if recv_prev:
            recv_prev_shape_tensor = torch.empty(
                (3,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
        if recv_next:
            recv_next_shape_tensor = torch.empty(
                (3,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
        if tensor_send_prev is not None:
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev.size(),
                device=cur_platform.current_device(),
                dtype=torch.int64,  # FlagScale Add
            )
        if tensor_send_next is not None:
            send_next_shape_tensor = torch.tensor(
                tensor_send_next.size(),
                device=cur_platform.current_device(),
                dtype=torch.int64,  # FlagScale Add
            )

        if config.use_ring_exchange_p2p:
            torch.distributed.ring_exchange(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
            )
        else:
            ops = []
            if send_prev_shape_tensor is not None:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_prev_shape_tensor, self.prev_rank, self.pp_group
                )
                ops.append(send_prev_op)
            if recv_prev_shape_tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_prev_shape_tensor, self.prev_rank, self.pp_group
                )
                ops.append(recv_prev_op)
            if send_next_shape_tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_next_shape_tensor, self.next_rank, self.pp_group
                )
                ops.append(send_next_op)
            if recv_next_shape_tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_next_shape_tensor, self.next_rank, self.pp_group
                )
                ops.append(recv_next_op)
            if len(ops) > 0:
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

            # To protect against race condition when using batch_isend_irecv().
            # should take this out once the bug with batch_isend_irecv is resolved.
            cur_platform.synchronize()  # FlagScale Add

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor.tolist()

        recv_next_shape = [0, 0, 0]
        if recv_next_shape_tensor is not None:
            recv_next_shape = recv_next_shape_tensor.tolist()

        return recv_prev_shape, recv_next_shape

    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        wait_on_reqs: bool = True,
        group=None,  ######## FlagScale Add ########
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Communicate tensors between stages. Used as helper method in other
        communication methods that are used in megatron/schedules.py.

        Args:
            tensor_send_next (torch.Tensor, optional):
                Tensor to send to next rank (no tensor sent if None)

            tensor_send_prev (torch.Tensor, optional):
                Tensor to send to prev rank (no tensor sent if None)

            recv_prev (boolean, required):
                whether tensor should be received from previous rank.

            recv_next (boolean, required):
                whether tensor should be received from next rank.

            tensor_shape (List[int] or torch.Size, required):
                shape of tensor to receive (this method assumes that all
                tensors sent and received in a single function call are
                the same shape).

            wait_on_reqs (boolean, optional, default=False):
                For non-batched p2p communication, wait on each request
                before returning.

        Returns:
            tuple containing

            - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
            - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

        """

        config = self.config
        tensor_recv_prev_func = None
        tensor_recv_next_func = None

        if config.variable_seq_lengths or config.mtp_standalone:
            recv_prev_shape, recv_next_shape = self._communicate_shapes(
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape

        def create_tensor_recv_prev():
            return torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=cur_platform.current_device(),  # FlagScale Add
                dtype=config.pipeline_dtype,
            )

        def create_tensor_recv_next():
            return torch.empty(
                recv_next_shape,
                requires_grad=True,
                device=cur_platform.current_device(),  # FlagScale Add
                dtype=config.pipeline_dtype,
            )

        if recv_prev:
            if config.pipeline_dtype is None:
                raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_prev is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_prev_func = create_tensor_recv_prev

        if recv_next:
            if config.pipeline_dtype is None:
                raise RuntimeError("dtype must be provided if recv_next is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_next is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_next_func = create_tensor_recv_next

        # Send tensors in both the forward and backward directions as appropriate.
        if config.use_ring_exchange_p2p:

            def _ring_exchange_wrapper(**kwargs):
                torch.distributed.ring_exchange(**kwargs)
                return []

            p2p_func = _ring_exchange_wrapper
            transport_api = "ring_exchange"
        elif config.batch_p2p_comm:
            assert wait_on_reqs
            p2p_func = _batched_p2p_ops
            transport_api = "batch_isend_irecv"
        else:
            p2p_func = _p2p_ops
            transport_api = "isend_irecv"

        ######### FlagScale Begin #########
        if group is not None:
            pp_group = group
            curr_rank_in_pg = pp_group.rank()
            world_size = pp_group.size()
            next_rank_pg = (curr_rank_in_pg + 1) % world_size
            prev_rank_pg = (curr_rank_in_pg - 1) % world_size
            next_rank: int | None = dist.get_global_rank(pp_group, next_rank_pg)
            prev_rank: int | None = dist.get_global_rank(pp_group, prev_rank_pg)
        ######### FlagScale End #########
        # FlagScale Begin
        else:
            pp_group = self.pp_group
            next_rank = self.next_rank
            prev_rank = self.prev_rank
        # FlagScale End

        if config.use_ring_exchange_p2p or config.batch_p2p_comm:
            reqs = []
        else:
            reqs = {}

        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()

        p2p_group_plan = None
        if transport_api == "isend_irecv":
            p2p_group_plan = _p2p_group_plan(pp_group)
            p2p_func = partial(_p2p_ops, group_plan=p2p_group_plan)

        has_p2p_operations = any(
            tensor is not None
            for tensor in (tensor_send_prev, tensor_recv_prev, tensor_send_next, tensor_recv_next)
        )
        launch_gate = prepare_trace_scope("p2p-launch") if has_p2p_operations else None
        observe_p2p = _p2p_observation_enabled(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            transport_api=transport_api,
            launch_enabled=launch_gate is not None,
        )

        operations_cache: Optional[list[_P2POperation]] = None

        def ensure_operations() -> list[_P2POperation]:
            nonlocal operations_cache
            if operations_cache is not None:
                return operations_cache
            active_tensors = {
                "send_prev": tensor_send_prev,
                "recv_prev": tensor_recv_prev,
                "send_next": tensor_send_next,
                "recv_next": tensor_recv_next,
            }
            backends_by_key = (
                {
                    key: _get_distributed_backend(request_group)
                    for key, request_group in p2p_group_plan
                    if active_tensors[key] is not None
                }
                if p2p_group_plan is not None
                else None
            )
            operations_cache = _build_p2p_operations(
                tensor_send_prev=tensor_send_prev,
                tensor_recv_prev=tensor_recv_prev,
                tensor_send_next=tensor_send_next,
                tensor_recv_next=tensor_recv_next,
                prev_pipeline_rank=prev_rank,
                next_pipeline_rank=next_rank,
                backend=_get_distributed_backend(pp_group),
                transport_api=transport_api,
                wait_on_reqs=wait_on_reqs,
                backends_by_key=backends_by_key,
            )
            return operations_cache

        if observe_p2p:
            operations = ensure_operations()
            p2p_reqs = _launch_p2p(
                p2p_func,
                operations,
                {
                    "tensor_send_prev": tensor_send_prev,
                    "tensor_recv_prev": tensor_recv_prev,
                    "tensor_send_next": tensor_send_next,
                    "tensor_recv_next": tensor_recv_next,
                    "group": pp_group,
                    "prev_pipeline_rank": prev_rank,
                    "next_pipeline_rank": next_rank,
                },
                launch_gate=launch_gate,
            )
        else:
            operations = []
            p2p_reqs = p2p_func(
                tensor_send_prev=tensor_send_prev,
                tensor_recv_prev=tensor_recv_prev,
                tensor_send_next=tensor_send_next,
                tensor_recv_next=tensor_recv_next,
                group=pp_group,
                prev_pipeline_rank=prev_rank,
                next_pipeline_rank=next_rank,
            )
        physical_request_count = len(p2p_reqs)
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

        if wait_on_reqs and len(reqs) > 0:
            if not operations:
                for req in reqs if isinstance(reqs, list) else reqs.values():
                    req.wait()
            elif isinstance(reqs, list):
                request_pairing = _p2p_batch_request_pairing(reqs, len(operations))
                if request_pairing == "aggregate":
                    _wait_p2p_batch_request(reqs[0], operations)
                elif request_pairing == "position":
                    for req, operation in zip(reqs, operations):
                        _wait_p2p_request(req, operation)
                else:
                    for req in reqs:
                        req.wait()
            else:
                operations_by_key = {operation.key: operation for operation in operations}
                for key, req in reqs.items():
                    _wait_p2p_request(req, operations_by_key.get(key))
            reqs = None
        elif len(reqs) > 0 and operations:
            self._register_p2p_requests(reqs, operations)

        if config.batch_p2p_comm and config.batch_p2p_sync:
            # To protect against race condition when using batch_isend_irecv().
            # User should assert that we have a modern enough PyTorch to not need this
            sync_gate = prepare_trace_scope("p2p-batch-device-sync")
            sync_operations = ensure_operations() if sync_gate is not None else []
            _synchronize_p2p_batch(sync_gate, sync_operations, physical_request_count)

        return tensor_recv_prev, tensor_recv_next, reqs

    @nvtx_decorator()
    def recv_forward(
        self, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        # FlagScale Begin
        if self.config.enable_hetero:
            return recv_forward_hetero(
                tensor_shapes, is_first_stage, self.config, partial(self._communicate)
            )
        # FlagScale End
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        config = self.config
        for tensor_shape in tensor_shapes:
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_tensor_shapes:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def recv_backward(
        self, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from next rank in pipeline (backward receive)."""
        # FlagScale Begin
        if self.config.enable_hetero:
            return recv_backward_hetero(
                tensor_shapes, is_last_stage, self.config, partial(self._communicate)
            )
        # FlagScale End
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        config = self.config
        output_tensor_grads = []
        for tensor_shape in tensor_shapes:
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
        if unwrap_tensor_shapes:
            return output_tensor_grads[0]
        return output_tensor_grads

    @nvtx_decorator()
    def send_forward(self, output_tensors, is_last_stage: bool) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        # FlagScale Begin
        if self.config.enable_hetero:
            return send_forward_hetero(
                output_tensors, is_last_stage, self.config, partial(self._communicate)
            )
        # FlagScale End
        config = self.config
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]

        for output_tensor in output_tensors:
            if not is_last_stage:
                if config.timers is not None:
                    config.timers('forward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('forward-send').stop()

    @nvtx_decorator()
    def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
        """Send tensor to previous rank in pipeline (backward send)."""
        # FlagScale Begin
        if self.config.enable_hetero:
            return send_backward_hetero(
                input_tensor_grads, is_first_stage, self.config, partial(self._communicate)
            )
        # FlagScale End
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        config = self.config
        for input_tensor_grad in input_tensor_grads:
            if not is_first_stage:
                if config.timers is not None:
                    config.timers('backward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('backward-send').stop()

    @nvtx_decorator()
    def send_forward_recv_backward(
        self, output_tensors, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with next rank in pipeline."""
        # FlagScale Begin
        if self.config.enable_hetero:
            return send_forward_recv_backward_hetero(
                output_tensors,
                tensor_shapes,
                is_last_stage,
                self.config,
                partial(self._communicate),
            )
        # FlagScale End
        config = self.config
        unwrap_output_tensors = False
        if not isinstance(output_tensors, list):
            unwrap_output_tensors = True
            output_tensors = [output_tensors]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        output_tensor_grads = []
        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('forward-send-backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-send-backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
        if unwrap_output_tensors:
            return output_tensor_grads[0]
        return output_tensor_grads

    @nvtx_decorator()
    def send_backward_recv_forward(
        self, input_tensor_grads, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with previous rank in pipeline."""
        # FlagScale Begin
        if self.config.enable_hetero:
            return send_backward_recv_forward_hetero(
                input_tensor_grads,
                tensor_shapes,
                is_first_stage,
                self.config,
                partial(self._communicate),
            )
        # FlagScale End
        config = self.config
        unwrap_input_tensor_grads = False
        if not isinstance(input_tensor_grads, list):
            unwrap_input_tensor_grads = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('backward-send-forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-send-forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_input_tensor_grads:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def send_forward_recv_forward(
        self,
        output_tensor: torch.Tensor,
        recv_prev: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('forward-send-forward-recv', log_level=2).start()
        input_tensor, _, wait_handles = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('forward-send-forward-recv').stop()
        if overlap_p2p_comm:
            return input_tensor, wait_handles
        return input_tensor

    @nvtx_decorator()
    def send_backward_recv_backward(
        self,
        input_tensor_grad: torch.Tensor,
        recv_next: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> torch.Tensor:
        """Batched recv from next rank and send to previous rank in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('backward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, wait_handles = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('backward-send-backward-recv').stop()
        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    @nvtx_decorator()
    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
    ) -> torch.Tensor:
        """Batched send and recv with previous and next ranks in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
        input_tensor, output_tensor_grad, _ = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv').stop()
        return input_tensor, output_tensor_grad

    ########## FlagScale Begin ##########
    def warm_up_comm_group(self):
        """Warm up the communication group by performing a dummy send and recv."""
        if self.config.enable_hetero:
            self.warm_up_comm_group_hetero()
            return
        # NOTE(lizhiyu): For enbale config.variable_seq_lengths and pp_size > 2
        if not self.config.variable_seq_lengths or self.pp_group.size() <= 2:
            return
        self.config.variable_seq_lengths = False
        rank = torch.distributed.get_rank()
        # This is arbitrary because the shape of the recv tensor needs
        # to be specified when communicating.
        # It can be changed into any other shape.
        tensor_shape = [1]
        to_send_tensor = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=cur_platform.current_device(),
            dtype=self.config.pipeline_dtype,
        )
        to_recv_tensor = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=cur_platform.current_device(),
            dtype=self.config.pipeline_dtype,
        )

        group_ranks = torch.distributed.get_process_group_ranks(self.pp_group)
        pipeline_rank = self.pp_group.rank()
        if pipeline_rank == 0:
            self._communicate(
                tensor_send_next=to_send_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=to_recv_tensor.shape,
                group=self.pp_group,
            )
        elif pipeline_rank == len(group_ranks) - 1:
            self._communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=to_recv_tensor.shape,
                group=self.pp_group,
            )
        elif rank in group_ranks:
            self._communicate(
                tensor_send_next=to_send_tensor,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=to_recv_tensor.shape,
                group=self.pp_group,
            )
        self.config.variable_seq_lengths = True

    def warm_up_comm_group_hetero(self):
        """Warm up the communication for all PP groups, to avoid the hang issue.

        P2P comm would call batch_isend_irecv API, which requires
        all ranks of the group to participate if this API is the
        first collective call in the group passed to `dist.P2POp`.

        See batch_isend_irecv for more details.
        """
        rank = torch.distributed.get_rank()
        # This is arbitrary because the shape of the recv tensor needs
        # to be specified when communicating.
        # It can be changed into any other shape.
        tensor_shape = [1]
        to_send_tensor = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=(
                cur_platform.current_device()
                if "cpu:gloo" != torch.distributed.get_backend(self.pp_group[0])
                else torch.device("cpu")
            ),
            dtype=self.config.pipeline_dtype,
        )
        to_recv_tensor = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=(
                cur_platform.current_device()
                if "cpu:gloo" != torch.distributed.get_backend(self.pp_group[0])
                else torch.device("cpu")
            ),
            dtype=self.config.pipeline_dtype,
        )

        for pp_g in self.pp_group:
            group_ranks = torch.distributed.get_process_group_ranks(pp_g)
            pipeline_rank = pp_g.rank()
            if pipeline_rank == 0:
                self._communicate(
                    tensor_send_next=to_send_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=to_recv_tensor.shape,
                    group=pp_g,
                )
            elif pipeline_rank == len(group_ranks) - 1:
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=to_recv_tensor.shape,
                    group=pp_g,
                )
            elif rank in group_ranks:
                self._communicate(
                    tensor_send_next=to_send_tensor,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=to_recv_tensor.shape,
                    group=pp_g,
                )

    ########## FlagScale End ##########
