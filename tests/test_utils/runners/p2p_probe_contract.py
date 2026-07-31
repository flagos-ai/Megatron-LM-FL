# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P2P route contracts for the controlled PP2 training profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_DIRECTIONAL_EVENTS = frozenset(
    ("send-forward", "recv-forward", "send-backward", "recv-backward")
)
_BATCH_DIRECTIONS = {
    0: ("send-forward", "recv-backward"),
    1: ("recv-forward", "send-backward"),
}


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    by_rank: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None:
            raise ValueError(f"trace shard {rank} has no global rank")
        if rank.global_rank in by_rank:
            raise ValueError(f"duplicate trace shard for global rank {rank.global_rank}")
        by_rank[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return by_rank


def _failure(code: str, message: str, *, rank: int, iteration: int) -> Failure:
    return Failure(code, message, f"rank={rank} iteration={iteration}")


def _events(iteration: Iteration, name: str, phase: str) -> list[Event]:
    return [
        event
        for event in iteration.events
        if event.name == name and event.ph == phase
    ]


def _route_events(iteration: Iteration, phase: str) -> list[Event]:
    return [
        event
        for event in iteration.events
        if event.name in _DIRECTIONAL_EVENTS and event.ph == phase
    ]


def _check_fields(
    event: Event,
    expected: Mapping[str, Any],
    *,
    rank: int,
    iteration: int,
) -> list[Failure]:
    failures = []
    for field, value in expected.items():
        if field not in event.attrs or event.attrs[field] != value:
            observed = event.attrs[field] if field in event.attrs else "<missing>"
            failures.append(
                _failure(
                    "trace.p2p.field",
                    f"event {event.name!r} has {field}={observed!r}, "
                    f"expected {value!r}",
                    rank=rank,
                    iteration=iteration,
                )
            )
    return failures


def _operation_event_name(operation: Mapping[str, Any]) -> str:
    return f"{operation.get('direction')}-{operation.get('pipeline_direction')}"


def _validate_iteration(
    iteration: Iteration,
    *,
    rank: int,
    batched: bool,
) -> tuple[list[Failure], set[str]]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    transport_api = "batch_isend_irecv" if batched else "isend_irecv"
    launch_pairing = "backend_dependent" if batched else "key"
    completion_pairing = "position" if batched else "key"

    launches = _events(iteration, "p2p-launch", "B")
    launch_ends = _events(iteration, "p2p-launch", "E")
    completions = _route_events(iteration, "B")
    completion_ends = _route_events(iteration, "E")
    syncs = _events(iteration, "p2p-batch-device-sync", "B")
    sync_ends = _events(iteration, "p2p-batch-device-sync", "E")

    expected_launches = 2 if batched else None
    valid_launch_count = (
        bool(launches)
        and len(launches) == len(launch_ends)
        and (expected_launches is None or len(launches) == expected_launches)
    )
    if not valid_launch_count:
        expected = "at least one pair" if expected_launches is None else str(expected_launches)
        failures.append(
            _failure(
                "trace.p2p.launch_count",
                f"p2p-launch B/E={len(launches)}/{len(launch_ends)}, "
                f"expected {expected}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    completion_counts_match = len(completions) == len(completion_ends) and all(
        len(_events(iteration, name, "B")) == len(_events(iteration, name, "E"))
        for name in _DIRECTIONAL_EVENTS
    )
    if not completions or not completion_counts_match:
        failures.append(
            _failure(
                "trace.p2p.completion_count",
                f"directional B/E={len(completions)}/{len(completion_ends)}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    expected_syncs = len(launches) if batched else 0
    if len(syncs) != expected_syncs or len(sync_ends) != expected_syncs:
        failures.append(
            _failure(
                "trace.p2p.sync_count",
                f"p2p-batch-device-sync B/E={len(syncs)}/{len(sync_ends)}, "
                f"expected {expected_syncs}/{expected_syncs}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    launched: dict[str, tuple[str, str, str]] = {}
    operations_by_batch: dict[str, set[str]] = {}
    completion_modes: set[str] = set()
    for launch in launches:
        failures.extend(
            _check_fields(
                launch,
                {
                    "transport_api": transport_api,
                    "request_pairing": launch_pairing,
                    "completion_included": False,
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        batch_id = launch.attrs.get("batch_id")
        operations = launch.attrs.get("operations")
        mode = launch.attrs.get("completion_mode")
        allowed_modes = {"internal_wait"} if batched else {"internal_wait", "external_wait"}
        if (
            not isinstance(batch_id, str)
            or batch_id in operations_by_batch
            or not isinstance(operations, list)
            or not operations
            or launch.attrs.get("operation_count") != len(operations)
            or mode not in allowed_modes
        ):
            failures.append(
                _failure(
                    "trace.p2p.launch",
                    "p2p-launch has an invalid batch, operation list, or completion mode",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue

        completion_modes.add(mode)
        operation_ids: set[str] = set()
        for operation in operations:
            if not isinstance(operation, Mapping):
                failures.append(
                    _failure(
                        "trace.p2p.operation",
                        "p2p-launch contains a non-mapping operation",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            operation_id = operation.get("operation_id")
            event_name = _operation_event_name(operation)
            if (
                not isinstance(operation_id, str)
                or operation_id in launched
                or operation.get("request_id") != operation_id
                or operation.get("transport_api") != transport_api
                or operation.get("completion_mode") != mode
                or event_name not in _DIRECTIONAL_EVENTS
            ):
                failures.append(
                    _failure(
                        "trace.p2p.operation",
                        f"p2p-launch contains an invalid operation {operation!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            operation_ids.add(operation_id)
            launched[operation_id] = (batch_id, event_name, mode)
        operations_by_batch[batch_id] = operation_ids

    if len(completions) != len(launched):
        failures.append(
            _failure(
                "trace.p2p.completion_count",
                f"directional completions={len(completions)}, "
                f"launched operations={len(launched)}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    completed_ids: list[str] = []
    for completion in completions:
        operation_id = completion.attrs.get("operation_id")
        launched_operation = launched.get(operation_id)
        if launched_operation is None:
            failures.append(
                _failure(
                    "trace.p2p.identity",
                    f"completion {completion.name!r} has unknown "
                    f"operation_id={operation_id!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        batch_id, event_name, mode = launched_operation
        completion_site = (
            "communicate_internal_wait"
            if mode == "internal_wait"
            else "exposed_request_wait"
        )
        failures.extend(
            _check_fields(
                completion,
                {
                    "batch_id": batch_id,
                    "transport_api": transport_api,
                    "request_pairing": completion_pairing,
                    "completion_mode": mode,
                    "completion_site": completion_site,
                    "completion_included": True,
                    "completion_kind": "work_wait",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        if completion.name != event_name:
            failures.append(
                _failure(
                    "trace.p2p.identity",
                    f"operation_id={operation_id!r} completed as "
                    f"{completion.name!r}, expected {event_name!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if isinstance(operation_id, str):
            completed_ids.append(operation_id)

    if len(completed_ids) != len(set(completed_ids)) or set(completed_ids) != set(launched):
        failures.append(
            _failure(
                "trace.p2p.identity",
                "launch and completion operation IDs do not form a one-to-one mapping",
                rank=rank,
                iteration=iteration_id,
            )
        )

    if batched:
        synced_batches = set()
        for sync in syncs:
            failures.extend(
                _check_fields(
                    sync,
                    {
                        "transport_api": "batch_isend_irecv",
                        "completion_kind": "device_synchronize",
                        "completion_site": "batch_p2p_sync_workaround",
                        "completion_included": True,
                    },
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            batch_id = sync.attrs.get("batch_id")
            operation_ids = sync.attrs.get("operation_ids")
            expected_ids = operations_by_batch.get(batch_id)
            if (
                expected_ids is None
                or not isinstance(operation_ids, list)
                or set(operation_ids) != expected_ids
            ):
                failures.append(
                    _failure(
                        "trace.p2p.sync_identity",
                        f"device sync cannot be paired with batch_id={batch_id!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            elif isinstance(batch_id, str):
                synced_batches.add(batch_id)
        if synced_batches != set(operations_by_batch):
            failures.append(
                _failure(
                    "trace.p2p.sync_identity",
                    "launch and device-sync batch IDs differ",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

        expected_sequence = tuple(
            item
            for direction in _BATCH_DIRECTIONS[rank]
            for item in (
                ("p2p-launch", "B"),
                ("p2p-launch", "E"),
                (direction, "B"),
                (direction, "E"),
                ("p2p-batch-device-sync", "B"),
                ("p2p-batch-device-sync", "E"),
            )
        )
        observed_sequence = tuple(
            (event.name, event.ph)
            for event in iteration.events
            if event.name in _DIRECTIONAL_EVENTS
            or event.name in {"p2p-launch", "p2p-batch-device-sync"}
        )
        if observed_sequence != expected_sequence:
            failures.append(
                _failure(
                    "trace.p2p.sequence",
                    "batched P2P launch, wait, and device-sync order differs",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    return failures, completion_modes


def _validate_pp2_route(trace_root: Path, *, batched: bool) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    observed_completion_modes: set[str] = set()
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            Failure(
                "trace.p2p.ranks",
                f"PP2 P2P contract expects ranks [0, 1], observed {sorted(by_rank)}",
                "pp2-p2p",
            )
        )

    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.p2p.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        for iteration in iterations:
            pipeline_ranks = {event.rank.pipeline for event in iteration.events}
            if pipeline_ranks != {rank}:
                failures.append(
                    _failure(
                        "trace.p2p.pipeline_rank",
                        f"events use pipeline ranks {sorted(pipeline_ranks)}, "
                        f"expected [{rank}]",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
            iteration_failures, completion_modes = _validate_iteration(
                iteration,
                rank=rank,
                batched=batched,
            )
            failures.extend(iteration_failures)
            observed_completion_modes.update(completion_modes)

    if not batched and "external_wait" not in observed_completion_modes:
        failures.append(
            Failure(
                "trace.p2p.external_wait",
                "PP2 VPP route contains no scheduler-observed external wait",
                "pp2-unbatched",
            )
        )
    return tuple(failures)


def validate_pp2_batched_route(trace_root: Path) -> tuple[Failure, ...]:
    """Validate batch transport, internal waits, and device synchronization."""

    return _validate_pp2_route(trace_root, batched=True)


def validate_pp2_unbatched_route(trace_root: Path) -> tuple[Failure, ...]:
    """Validate unbatched key pairing and at least one external wait."""

    return _validate_pp2_route(trace_root, batched=False)


def _validate_ring_iteration(
    iteration: Iteration,
    *,
    rank: int,
) -> tuple[list[Failure], set[str], set[str]]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    launches = _events(iteration, "p2p-launch", "B")
    launch_ends = _events(iteration, "p2p-launch", "E")
    syncs = _events(iteration, "p2p-batch-device-sync", "B")
    sync_ends = _events(iteration, "p2p-batch-device-sync", "E")
    forbidden = [
        event
        for event in iteration.events
        if event.name in _DIRECTIONAL_EVENTS
        or event.name == "p2p-batch-complete"
    ]
    if not launches or len(launches) != len(launch_ends):
        failures.append(
            _failure(
                "trace.p2p.launch_count",
                f"ring p2p-launch B/E={len(launches)}/{len(launch_ends)}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    if forbidden:
        failures.append(
            _failure(
                "trace.p2p.ring_completion",
                "ring exchange must not emit Work wait events",
                rank=rank,
                iteration=iteration_id,
            )
        )
    if len(syncs) != len(launches) or len(sync_ends) != len(launches):
        failures.append(
            _failure(
                "trace.p2p.sync_count",
                f"ring device-sync B/E={len(syncs)}/{len(sync_ends)}, "
                f"expected {len(launches)}/{len(launches)}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    relevant_sequence = tuple(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in {"p2p-launch", "p2p-batch-device-sync"}
    )
    expected_sequence = (
        (
            ("p2p-launch", "B"),
            ("p2p-launch", "E"),
            ("p2p-batch-device-sync", "B"),
            ("p2p-batch-device-sync", "E"),
        )
        * len(launches)
    )
    if relevant_sequence != expected_sequence:
        failures.append(
            _failure(
                "trace.p2p.sequence",
                "ring launch and configured device-sync order differs",
                rank=rank,
                iteration=iteration_id,
            )
        )

    batch_ids: set[str] = set()
    operation_ids: set[str] = set()
    observed_directions: set[str] = set()
    expected_directions = set(_BATCH_DIRECTIONS[rank])
    for launch, launch_end in zip(launches, launch_ends):
        failures.extend(
            _check_fields(
                launch,
                {
                    "comm_type": "p2p-launch",
                    "timing_phase": "inline_api_call",
                    "backend": "nccl",
                    "backends": ["nccl"],
                    "backend_complete": True,
                    "transport_api": "ring_exchange",
                    "request_pairing": "none",
                    "completion_mode": "inline",
                    "completion_included": False,
                    "api_return_included": True,
                    "completion_guarantee": "api_return_observed",
                    "completion_kind": "inline_api_return",
                    "device_completion_guaranteed": False,
                    "duration_attribution": "shared_nonexclusive",
                    "host_blocking_guaranteed": False,
                    "operation_id_scope": "rank_local",
                    "physical_request_count": 0,
                    "stage": "p2p_inline_api",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        failures.extend(
            _check_fields(
                launch_end,
                {"completed": True},
                rank=rank,
                iteration=iteration_id,
            )
        )
        batch_id = launch.attrs.get("batch_id")
        operations = launch.attrs.get("operations")
        recorded_ids = launch.attrs.get("operation_ids")
        if (
            not isinstance(batch_id, str)
            or batch_id in batch_ids
            or not isinstance(operations, list)
            or not operations
            or launch.attrs.get("operation_count") != len(operations)
            or not isinstance(recorded_ids, list)
        ):
            failures.append(
                _failure(
                    "trace.p2p.launch",
                    "ring p2p-launch has an invalid batch or operation list",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        batch_ids.add(batch_id)
        expected_ids: list[str] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                failures.append(
                    _failure(
                        "trace.p2p.operation",
                        "ring p2p-launch contains a non-mapping operation",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            operation_id = operation.get("operation_id")
            event_name = _operation_event_name(operation)
            data_bytes = operation.get("data_bytes")
            if (
                not isinstance(operation_id, str)
                or operation_id in operation_ids
                or not operation_id.startswith(f"{batch_id}:")
                or operation.get("request_id") is not None
                or operation.get("transport_api") != "ring_exchange"
                or operation.get("completion_mode") != "inline"
                or operation.get("backend") != "nccl"
                or operation.get("peer_rank") != 1 - rank
                or event_name not in expected_directions
                or not isinstance(data_bytes, int)
                or isinstance(data_bytes, bool)
                or data_bytes <= 0
            ):
                failures.append(
                    _failure(
                        "trace.p2p.operation",
                        f"ring p2p-launch contains an invalid operation {operation!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            operation_ids.add(operation_id)
            expected_ids.append(operation_id)
            observed_directions.add(event_name)
        if recorded_ids != expected_ids:
            failures.append(
                _failure(
                    "trace.p2p.identity",
                    f"ring operation_ids={recorded_ids!r}, expected {expected_ids!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    synced_batch_ids: set[str] = set()
    for sync, sync_end in zip(syncs, sync_ends):
        failures.extend(
            _check_fields(
                sync,
                {
                    "comm_type": "p2p",
                    "backend": "nccl",
                    "backends": ["nccl"],
                    "backend_complete": True,
                    "transport_api": "ring_exchange",
                    "request_pairing": "none",
                    "completion_site": "batch_p2p_sync_workaround",
                    "completion_included": True,
                    "completion_kind": "device_synchronize",
                    "physical_request_count": 0,
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        failures.extend(
            _check_fields(
                sync_end,
                {
                    "completed": True,
                    "device_completion_guaranteed": True,
                    "host_blocking_guaranteed": True,
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        batch_id = sync.attrs.get("batch_id")
        sync_operations = sync.attrs.get("operations")
        sync_operation_ids = sync.attrs.get("operation_ids")
        if (
            not isinstance(batch_id, str)
            or batch_id not in batch_ids
            or batch_id in synced_batch_ids
            or not isinstance(sync_operations, list)
            or not isinstance(sync_operation_ids, list)
            or sync.attrs.get("operation_count") != len(sync_operations)
            or sync_operation_ids
            != [
                operation.get("operation_id")
                for operation in sync_operations
                if isinstance(operation, Mapping)
            ]
            or set(sync_operation_ids) - operation_ids
        ):
            failures.append(
                _failure(
                    "trace.p2p.sync_identity",
                    f"ring device sync cannot be paired with batch_id={batch_id!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            synced_batch_ids.add(batch_id)
    if synced_batch_ids != batch_ids:
        failures.append(
            _failure(
                "trace.p2p.sync_identity",
                "ring launch and device-sync batch IDs differ",
                rank=rank,
                iteration=iteration_id,
            )
        )

    if observed_directions != expected_directions:
        failures.append(
            _failure(
                "trace.p2p.direction",
                f"ring directions are {sorted(observed_directions)}, "
                f"expected {sorted(expected_directions)}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return failures, batch_ids, operation_ids


def validate_pp2_ring_route(trace_root: Path) -> tuple[Failure, ...]:
    """Validate ring API return followed by the configured device sync."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            Failure(
                "trace.p2p.ranks",
                f"PP2 ring contract expects ranks [0, 1], observed {sorted(by_rank)}",
                "pp2-ring",
            )
        )

    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.p2p.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        rank_batch_ids: set[str] = set()
        rank_operation_ids: set[str] = set()
        for iteration in iterations:
            pipeline_ranks = {event.rank.pipeline for event in iteration.events}
            if pipeline_ranks != {rank}:
                failures.append(
                    _failure(
                        "trace.p2p.pipeline_rank",
                        f"events use pipeline ranks {sorted(pipeline_ranks)}, "
                        f"expected [{rank}]",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
            iteration_failures, batch_ids, operation_ids = _validate_ring_iteration(
                iteration,
                rank=rank,
            )
            failures.extend(iteration_failures)
            if rank_batch_ids & batch_ids or rank_operation_ids & operation_ids:
                failures.append(
                    _failure(
                        "trace.p2p.identity",
                        "ring batch or operation identity repeats across iterations",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
            rank_batch_ids.update(batch_ids)
            rank_operation_ids.update(operation_ids)
    return tuple(failures)
