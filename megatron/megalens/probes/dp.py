# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Implicit DP metadata and native wait/completion observation."""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Any, ContextManager, List

from megatron.core.observability import open_trace_scope, prepare_trace_scope
from megatron.core.utils import get_process_group_peer_ranks

if TYPE_CHECKING:
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucket

# BEGIN MEGALENS OBSERVABILITY
_DP_OPERATION_SEQUENCE = count(1)


def _next_dp_operation_id(kind: str) -> str:
    """Return a rank-local identity for an accepted DP communication probe."""
    return f"dp:{kind}:{next(_DP_OPERATION_SEQUENCE)}"


def _dp_allreduce_context(
    *,
    async_op: bool,
    data_bytes: int,
    group_size: int,
    group_role: str,
    n_buckets: int,
    operation_id: str,
    overlap_enabled: bool,
) -> dict[str, object]:
    """Build DP all-reduce metadata only after the probe gate accepts it."""
    return {
        "api_async_op": bool(async_op),
        "async_op": bool(async_op),
        "completion_included": False,
        "data_bytes": data_bytes,
        "group_size": group_size,
        "group_role": group_role,
        "n_buckets": n_buckets,
        "op": "all_reduce",
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "overlap_enabled": bool(overlap_enabled),
        "payload_role": "gradient_bucket",
        "stage": "main_bucket_allreduce",
        "timing_phase": "async_dispatch" if async_op else "collective_call",
    }


def _dp_reduce_scatter_context(
    *,
    async_op: bool,
    data_bytes: int,
    group_size: int,
    n_buckets: int,
    operation_id: str,
    overlap_enabled: bool,
) -> dict[str, object]:
    """Build DistOpt reduce-scatter metadata only after the probe gate accepts it."""
    return {
        "api_async_op": bool(async_op),
        "async_op": bool(async_op),
        "completion_included": False,
        "data_bytes": data_bytes,
        "group_size": group_size,
        "group_role": "intra_optimizer_instance",
        "n_buckets": n_buckets,
        "op": "reduce_scatter",
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "overlap_enabled": bool(overlap_enabled),
        "payload_role": "gradient_bucket",
        "stage": "intra_instance_reduce_scatter",
        "timing_phase": "async_dispatch" if async_op else "collective_call",
    }


def _dp_inter_instance_allreduce_context(
    *, data_bytes: int, group_size: int, n_buckets: int, operation_id: str, overlap_enabled: bool
) -> dict[str, object]:
    """Build inter-DistOpt all-reduce metadata only after the probe gate accepts it."""
    return {
        "api_async_op": False,
        "async_op": False,
        "completion_included": False,
        "data_bytes": data_bytes,
        "group_size": group_size,
        "group_role": "inter_optimizer_instance",
        "n_buckets": n_buckets,
        "op": "all_reduce",
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "overlap_enabled": bool(overlap_enabled),
        "payload_role": "gradient_shard",
        "stage": "inter_instance_shard_allreduce",
        "timing_phase": "collective_call",
    }


def _dp_param_allgather_data_bytes(
    buckets: List["_ParamAndGradBucket"], *, use_distributed_optimizer: bool
) -> int:
    """Return receive-buffer bytes for the accepted parameter all-gather probe."""
    if use_distributed_optimizer:
        return sum(
            int(bucket.param_data.numel() * bucket.param_data.element_size())
            for bucket in buckets
            if bucket.param_data is not None
        )

    data_bytes = 0
    for bucket in buckets:
        flat_sizes = bucket.layerwise_param_flat_sizes
        if not flat_sizes:
            continue
        data_bytes += int(sum(flat_sizes) * bucket.params_list[0].element_size())
    return data_bytes


def _dp_param_allgather_context(
    *,
    async_op: bool,
    data_bytes: int,
    group_size: int,
    n_buckets: int,
    operation_id: str,
    overlap_enabled: bool,
    use_distributed_optimizer: bool,
) -> dict[str, object]:
    """Build parameter all-gather metadata only after the probe gate accepts it."""
    optimizer_kind = "distributed" if use_distributed_optimizer else "layerwise"
    return {
        "api_async_op": bool(async_op),
        "async_op": bool(async_op),
        "completion_included": False,
        "data_bytes": data_bytes,
        "group_size": group_size,
        "group_role": "intra_optimizer_instance",
        "n_buckets": n_buckets,
        "op": "all_gather",
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "optimizer_kind": optimizer_kind,
        "overlap_enabled": bool(overlap_enabled),
        "payload_role": "parameter_bucket",
        "stage": f"{optimizer_kind}_optimizer_param_allgather",
        "timing_phase": "async_dispatch" if async_op else "collective_call",
    }


def _dp_param_sync_completion_context(
    *, completion_site: str, operation_id: Optional[str]
) -> dict[str, object]:
    """Build parameter-sync stream-dependency metadata after its gate accepts it."""
    return {
        "completion_included": True,
        "completion_guarantee": "current_stream_after_wait",
        "completion_kind": "work_wait",
        "completion_site": completion_site,
        "host_blocking_guaranteed": False,
        "launch_observed": operation_id is not None,
        "op": "wait",
        "operation_count": int(operation_id is not None),
        "operation_id": operation_id,
        "operation_ids": [operation_id] if operation_id is not None else [],
        "operation_id_scope": "rank_local",
        "stage": "parameter_allgather_completion",
        "timing_phase": "stream_dependency",
    }


def _dp_grad_sync_completion_context(
    *,
    completion_kind: str,
    completion_site: str,
    force_all_reduce: bool,
    num_distributed_optimizer_instances: int,
    operations: Tuple[Dict[str, str], ...],
    use_distributed_optimizer: bool,
) -> dict[str, object]:
    """Build gradient-sync completion metadata after its gate accepts it."""
    operation_ids = [operation["operation_id"] for operation in operations]
    is_stream_join = completion_kind == "stream_join"
    return {
        "completion_included": True,
        "completion_guarantee": (
            "current_stream_after_join" if is_stream_join else "current_stream_after_wait"
        ),
        "completion_kind": completion_kind,
        "completion_site": completion_site,
        "force_all_reduce": bool(force_all_reduce),
        "launch_observed": bool(operations),
        "num_distributed_optimizer_instances": num_distributed_optimizer_instances,
        "host_blocking_guaranteed": False,
        "op": "wait_stream" if is_stream_join else "wait",
        "operation_count": len(operation_ids),
        "operation_ids": operation_ids,
        "operation_id_scope": "rank_local",
        "operations": [dict(operation) for operation in operations],
        "stage": "gradient_collective_completion",
        "timing_phase": "stream_dependency",
        "use_distributed_optimizer": bool(use_distributed_optimizer),
    }


# END MEGALENS OBSERVABILITY


def wait_param_gather(self: Any, *, completion_site: str) -> None:
    """Wait for a pending parameter gather and expose its completion boundary."""
    handle = self.param_gather_handle
    assert handle is not None
    completion_gate = prepare_trace_scope("dp-param-sync-complete")
    completion_context = None
    if completion_gate is not None:
        completion_context = _dp_param_sync_completion_context(
            completion_site=completion_site,
            operation_id=getattr(self, "_param_gather_trace_operation_id", None),
        )
    completion_scope = open_trace_scope(
        completion_gate,
        "dp-param-sync-complete",
        ctx=completion_context,
        slots=("completed", "error_type"),
    )
    wait_completed = False
    try:
        with completion_scope as completion:
            try:
                handle.wait()
            except BaseException as wait_error:
                completion.set("completed", False)
                completion.set("error_type", type(wait_error).__name__)
                raise
            completion.set("completed", True)
            wait_completed = True
    finally:
        if wait_completed:
            self.param_gather_handle = None
            self._param_gather_trace_operation_id = None


def grad_sync_completion_scope(
    self: Any, *, completion_kind: str, completion_site: str, force_all_reduce: bool
) -> ContextManager[Any]:
    """Open one completion event for all gradient collectives joined at this boundary."""
    completion_gate = prepare_trace_scope("dp-grad-sync-complete")
    completion_context = None
    if completion_gate is not None:
        completion_context = _dp_grad_sync_completion_context(
            completion_kind=completion_kind,
            completion_site=completion_site,
            force_all_reduce=force_all_reduce,
            num_distributed_optimizer_instances=(
                self.ddp_config.num_distributed_optimizer_instances
            ),
            operations=getattr(self, "_grad_sync_trace_operations", ()),
            use_distributed_optimizer=self.ddp_config.use_distributed_optimizer,
        )
    return open_trace_scope(
        completion_gate,
        "dp-grad-sync-complete",
        ctx=completion_context,
        slots=("completed", "error_type"),
    )


class _CollectiveScope:
    """Keep metadata gating inside the probe, including peer-rank queries."""

    __slots__ = ("manager", "enabled")

    def __init__(self, manager: Any, enabled: bool = True) -> None:
        self.manager = manager
        self.enabled = enabled

    def __enter__(self) -> _CollectiveScope:
        if self.enabled:
            self.manager.__enter__()
        return self

    def __exit__(self, *exception: Any) -> bool | None:
        if self.enabled:
            return self.manager.__exit__(*exception)
        return None

    def observe_group(self, group: Any) -> None:
        if self.enabled:
            self.manager.set('group', get_process_group_peer_ranks(group))


_NOOP_COLLECTIVE_SCOPE = _CollectiveScope(None, enabled=False)


def _collective_scope(gate: Any, manager: Any) -> _CollectiveScope:
    if gate is None:
        return _NOOP_COLLECTIVE_SCOPE
    return _CollectiveScope(manager)


def param_gather_scope(owner: Any, *, async_op: bool) -> tuple[_CollectiveScope, str | None]:
    """Prepare the existing parameter-gather event without caller-side gating."""
    gate = prepare_trace_scope('dp-param-all-gather')
    context = None
    operation_id = None
    if gate is not None:
        data_bytes = _dp_param_allgather_data_bytes(
            owner.buckets, use_distributed_optimizer=owner.ddp_config.use_distributed_optimizer
        )
        if owner.ddp_config.use_distributed_optimizer or data_bytes > 0:
            operation_id = _next_dp_operation_id('param-all-gather')
            context = _dp_param_allgather_context(
                async_op=async_op,
                data_bytes=data_bytes,
                group_size=owner.intra_distributed_optimizer_instance_size,
                n_buckets=len(owner.buckets),
                operation_id=operation_id,
                overlap_enabled=owner.ddp_config.overlap_param_gather,
                use_distributed_optimizer=owner.ddp_config.use_distributed_optimizer,
            )
        else:
            gate = None
    return (
        _collective_scope(
            gate, open_trace_scope(gate, 'dp-param-all-gather', ctx=context, slots=('group',))
        ),
        operation_id,
    )


def grad_sync_scope(
    owner: Any, *, async_op: bool, force_all_reduce: bool, operations: list[dict[str, str]]
) -> _CollectiveScope:
    """Prepare the existing gradient event and append only observed identities."""
    context = None
    reduce_scatter = owner.ddp_config.use_distributed_optimizer and not force_all_reduce
    event_name = 'dp-reduce-scatter' if reduce_scatter else 'dp-allreduce'
    gate = prepare_trace_scope(event_name)
    if gate is not None:
        data_bytes = sum(
            int(bucket.grad_data.numel() * bucket.grad_data.element_size())
            for bucket in owner.buckets
        )
        if reduce_scatter:
            operation_id = _next_dp_operation_id('reduce-scatter')
            context = _dp_reduce_scatter_context(
                async_op=async_op,
                data_bytes=data_bytes,
                group_size=owner.collective_group_size,
                n_buckets=len(owner.buckets),
                operation_id=operation_id,
                overlap_enabled=owner.ddp_config.overlap_grad_reduce,
            )
            stage = 'intra_instance_reduce_scatter'
        else:
            operation_id = _next_dp_operation_id('allreduce')
            context = _dp_allreduce_context(
                async_op=async_op,
                data_bytes=data_bytes,
                group_size=owner.collective_group_size,
                group_role=(
                    'intra_optimizer_instance'
                    if owner.ddp_config.use_distributed_optimizer
                    else 'data_parallel'
                ),
                n_buckets=len(owner.buckets),
                operation_id=operation_id,
                overlap_enabled=owner.ddp_config.overlap_grad_reduce,
            )
            stage = 'main_bucket_allreduce'
        operations.append({'event_name': event_name, 'operation_id': operation_id, 'stage': stage})
    # Literal event calls remain visible to the source/target probe scanner.
    if reduce_scatter:
        return _collective_scope(
            gate, open_trace_scope(gate, 'dp-reduce-scatter', ctx=context, slots=('group',))
        )
    return _collective_scope(
        gate, open_trace_scope(gate, 'dp-allreduce', ctx=context, slots=('group',))
    )


def inter_instance_scope(owner: Any, operations: list[dict[str, str]]) -> _CollectiveScope:
    """Prepare the existing inter-instance collective observation."""
    gate = prepare_trace_scope('dp-allreduce')
    context = None
    if gate is not None:
        operation_id = _next_dp_operation_id('inter-instance-allreduce')
        data_bytes = sum(
            int(
                bucket.grad_data.numel()
                // owner.intra_distributed_optimizer_instance_size
                * bucket.grad_data.element_size()
            )
            for bucket in owner.buckets
        )
        context = _dp_inter_instance_allreduce_context(
            data_bytes=data_bytes,
            group_size=owner.ddp_config.num_distributed_optimizer_instances,
            n_buckets=len(owner.buckets),
            operation_id=operation_id,
            overlap_enabled=owner.ddp_config.overlap_grad_reduce,
        )
        operations.append(
            {
                'event_name': 'dp-allreduce',
                'operation_id': operation_id,
                'stage': 'inter_instance_shard_allreduce',
            }
        )
    return _collective_scope(
        gate, open_trace_scope(gate, 'dp-allreduce', ctx=context, slots=('group',))
    )


def wait_grad_sync(owner: Any, *, force_all_reduce: bool) -> None:
    """Observe one native Work wait; clear its handle only after successful return."""
    scope = grad_sync_completion_scope(
        owner,
        completion_kind='work_wait',
        completion_site='finish_grad_sync',
        force_all_reduce=force_all_reduce,
    )
    completed = False
    try:
        with scope as completion:
            try:
                owner.grad_reduce_handle.wait()
            except BaseException as error:
                completion.set('completed', False)
                completion.set('error_type', type(error).__name__)
                raise
            completion.set('completed', True)
            completed = True
    finally:
        if completed:
            owner.grad_reduce_handle = None
            owner._grad_sync_trace_operations = ()


def join_grad_sync(owner: Any, platform: Any, *, force_all_reduce: bool) -> None:
    """Observe the existing communication-stream join without adding synchronization."""
    scope = grad_sync_completion_scope(
        owner,
        completion_kind='stream_join',
        completion_site='finish_grad_sync',
        force_all_reduce=force_all_reduce,
    )
    completed = False
    try:
        with scope as completion:
            try:
                platform.current_stream().wait_stream(owner.communication_stream)
            except BaseException as error:
                completion.set('completed', False)
                completion.set('error_type', type(error).__name__)
                raise
            completion.set('completed', True)
            completed = True
    finally:
        if completed:
            owner._grad_sync_trace_operations = ()
