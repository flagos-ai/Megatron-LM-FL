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

_DIRECTIONAL_EVENTS = frozenset(("send-forward", "recv-forward", "send-backward", "recv-backward"))

_BATCH_DIRECTIONS = {0: ("send-forward", "recv-backward"), 1: ("recv-forward", "send-backward")}


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
    return [event for event in iteration.events if event.name == name and event.ph == phase]


def _route_events(iteration: Iteration, phase: str) -> list[Event]:
    return [
        event
        for event in iteration.events
        if event.name in _DIRECTIONAL_EVENTS and event.ph == phase
    ]


def _check_fields(
    event: Event, expected: Mapping[str, Any], *, rank: int, iteration: int
) -> list[Failure]:
    failures = []
    for field, value in expected.items():
        if field not in event.attrs or event.attrs[field] != value:
            observed = event.attrs[field] if field in event.attrs else "<missing>"
            failures.append(
                _failure(
                    "trace.p2p.field",
                    f"event {event.name!r} has {field}={observed!r}, " f"expected {value!r}",
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
    expected_batched_operation_groups: Sequence[Sequence[str]] | None = None,
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
    batch_completions = _events(iteration, "p2p-batch-complete", "B")
    batch_completion_ends = _events(iteration, "p2p-batch-complete", "E")
    syncs = _events(iteration, "p2p-batch-device-sync", "B")
    sync_ends = _events(iteration, "p2p-batch-device-sync", "E")

    if batched:
        operation_groups = tuple(
            tuple(group)
            for group in (
                expected_batched_operation_groups
                if expected_batched_operation_groups is not None
                else ((direction,) for direction in _BATCH_DIRECTIONS[rank])
            )
        )
        expected_launches = len(operation_groups)
    else:
        operation_groups = ()
        expected_launches = None
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
                f"p2p-launch B/E={len(launches)}/{len(launch_ends)}, " f"expected {expected}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    directional_counts_match = len(completions) == len(completion_ends) and all(
        len(_events(iteration, name, "B")) == len(_events(iteration, name, "E"))
        for name in _DIRECTIONAL_EVENTS
    )
    if batched:
        expected_directional_completions = sum(len(group) == 1 for group in operation_groups)
        expected_batch_completions = sum(len(group) > 1 for group in operation_groups)
        if (
            not directional_counts_match
            or len(completions) != expected_directional_completions
            or len(batch_completions) != expected_batch_completions
            or len(batch_completion_ends) != expected_batch_completions
        ):
            failures.append(
                _failure(
                    "trace.p2p.completion_count",
                    "completion B/E counts differ from singleton/aggregate "
                    f"batches: directional={len(completions)}/{len(completion_ends)} "
                    f"expected={expected_directional_completions}; "
                    f"aggregate={len(batch_completions)}/"
                    f"{len(batch_completion_ends)} "
                    f"expected={expected_batch_completions}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    elif not completions or not directional_counts_match:
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
    operations_by_batch: dict[str, tuple[str, ...]] = {}
    observed_operation_groups: list[tuple[str, ...]] = []
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
        operation_ids: list[str] = []
        operation_names: list[str] = []
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
            operation_names.append(event_name)
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
            operation_ids.append(operation_id)
            launched[operation_id] = (batch_id, event_name, mode)
        operations_by_batch[batch_id] = tuple(operation_ids)
        observed_operation_groups.append(tuple(operation_names))

    if batched and tuple(observed_operation_groups) != operation_groups:
        failures.append(
            _failure(
                "trace.p2p.batch_structure",
                "batched P2P operation groups "
                f"{observed_operation_groups!r}, expected {operation_groups!r}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    if not batched and len(completions) != len(launched):
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
                    f"completion {completion.name!r} has unknown " f"operation_id={operation_id!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        batch_id, event_name, mode = launched_operation
        completion_site = (
            "communicate_internal_wait" if mode == "internal_wait" else "exposed_request_wait"
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
                    "operation_count": 1,
                    "operation_ids": [operation_id],
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
        expected_batch_ids = operations_by_batch.get(batch_id)
        if batched and expected_batch_ids != (operation_id,):
            failures.append(
                _failure(
                    "trace.p2p.identity",
                    f"directional completion for operation_id={operation_id!r} "
                    "does not belong to a singleton batch",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if isinstance(operation_id, str):
            completed_ids.append(operation_id)

    for completion, completion_end in zip(completions, completion_ends):
        if completion.name != completion_end.name:
            failures.append(
                _failure(
                    "trace.p2p.identity",
                    f"directional completion begins as {completion.name!r} and "
                    f"ends as {completion_end.name!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        failures.extend(
            _check_fields(completion_end, {"completed": True}, rank=rank, iteration=iteration_id)
        )

    if batched:
        completed_batches: list[str] = []
        for completion in batch_completions:
            batch_id = completion.attrs.get("batch_id")
            operation_ids = completion.attrs.get("operation_ids")
            expected_ids = operations_by_batch.get(batch_id)
            failures.extend(
                _check_fields(
                    completion,
                    {
                        "transport_api": "batch_isend_irecv",
                        "request_pairing": "aggregate",
                        "stage": "batch_p2p_completion",
                        "timing_phase": "stream_dependency",
                        "completion_mode": "internal_wait",
                        "completion_included": True,
                        "completion_kind": "aggregate_work_wait",
                        "completion_guarantee": "current_stream_after_wait",
                        "duration_attribution": "shared_nonexclusive",
                        "host_blocking_guaranteed": False,
                        "physical_request_count": 1,
                    },
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            # The producer has no completion_site for an aggregate Work. Its
            # stage and timing_phase fields identify the observed wait boundary.
            if (
                expected_ids is None
                or len(expected_ids) <= 1
                or not isinstance(operation_ids, list)
                or tuple(operation_ids) != expected_ids
                or completion.attrs.get("operation_count") != len(expected_ids)
            ):
                failures.append(
                    _failure(
                        "trace.p2p.identity",
                        "aggregate completion does not preserve its launch "
                        f"batch or ordered operation IDs for batch_id={batch_id!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            if isinstance(batch_id, str):
                completed_batches.append(batch_id)
            if isinstance(operation_ids, list):
                completed_ids.extend(
                    operation_id for operation_id in operation_ids if isinstance(operation_id, str)
                )

        for completion_end in batch_completion_ends:
            failures.extend(
                _check_fields(
                    completion_end, {"completed": True}, rank=rank, iteration=iteration_id
                )
            )

        expected_aggregate_batches = {
            batch_id
            for batch_id, operation_ids in operations_by_batch.items()
            if len(operation_ids) > 1
        }
        if (
            len(completed_batches) != len(set(completed_batches))
            or set(completed_batches) != expected_aggregate_batches
        ):
            failures.append(
                _failure(
                    "trace.p2p.identity",
                    "multi-operation launch and aggregate-completion batch IDs differ",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

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
        for sync, sync_end in zip(syncs, sync_ends):
            batch_id = sync.attrs.get("batch_id")
            expected_ids = operations_by_batch.get(batch_id)
            expected_pairing = (
                "aggregate" if expected_ids is not None and len(expected_ids) > 1 else "position"
            )
            failures.extend(
                _check_fields(
                    sync,
                    {
                        "transport_api": "batch_isend_irecv",
                        "completion_kind": "device_synchronize",
                        "completion_site": "batch_p2p_sync_workaround",
                        "completion_included": True,
                        "request_pairing": expected_pairing,
                        "physical_request_count": 1,
                    },
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            operation_ids = sync.attrs.get("operation_ids")
            if (
                expected_ids is None
                or not isinstance(operation_ids, list)
                or tuple(operation_ids) != expected_ids
                or sync.attrs.get("operation_count") != len(expected_ids)
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
            failures.extend(
                _check_fields(sync_end, {"completed": True}, rank=rank, iteration=iteration_id)
            )
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
            for group in operation_groups
            for item in (
                (("p2p-launch", "B"), ("p2p-launch", "E"))
                + (
                    (group[0] if len(group) == 1 else "p2p-batch-complete", "B"),
                    (group[0] if len(group) == 1 else "p2p-batch-complete", "E"),
                )
                + (("p2p-batch-device-sync", "B"), ("p2p-batch-device-sync", "E"))
            )
        )
        observed_sequence = tuple(
            (event.name, event.ph)
            for event in iteration.events
            if event.name in _DIRECTIONAL_EVENTS
            or event.name in {"p2p-launch", "p2p-batch-complete", "p2p-batch-device-sync"}
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


def _validate_pp2_route(
    trace_root: Path,
    *,
    batched: bool,
    expected_batched_operation_groups: Mapping[int, Sequence[Sequence[str]]] | None = None,
) -> tuple[Failure, ...]:
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
                    f"rank {rank} expects iterations [1, 2], " f"observed {list(iteration_ids)}",
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
                expected_batched_operation_groups=(
                    expected_batched_operation_groups[rank]
                    if expected_batched_operation_groups is not None
                    else None
                ),
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
