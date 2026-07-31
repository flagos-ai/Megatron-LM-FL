# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DP overlap lifecycle contracts for the controlled DP2 profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from megatron.megalens.trace_aggregate import (
    Event,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_COMMON_DISPATCH = {
    "api_async_op": True,
    "async_op": True,
    "completion_included": False,
    "operation_id_scope": "rank_local",
    "overlap_enabled": True,
    "timing_phase": "async_dispatch",
}
_COMMON_COMPLETION = {
    "completion_guarantee": "current_stream_after_wait",
    "completion_included": True,
    "completion_kind": "work_wait",
    "host_blocking_guaranteed": False,
    "launch_observed": True,
    "operation_id_scope": "rank_local",
    "timing_phase": "stream_dependency",
}
_STANDARD_DISPATCHES = {
    "dp-allreduce": {
        "op": "all_reduce",
        "group_role": "data_parallel",
        "payload_role": "gradient_bucket",
        "stage": "main_bucket_allreduce",
    }
}
_DISTOPT_DISPATCHES = {
    "dp-reduce-scatter": {
        "op": "reduce_scatter",
        "group_role": "intra_optimizer_instance",
        "payload_role": "gradient_bucket",
        "stage": "intra_instance_reduce_scatter",
    },
    "dp-param-all-gather": {
        "op": "all_gather",
        "group_role": "intra_optimizer_instance",
        "optimizer_kind": "distributed",
        "payload_role": "parameter_bucket",
        "stage": "distributed_optimizer_param_allgather",
    },
}
_ALL_ROUTE_EVENTS = frozenset(
    (
        *_STANDARD_DISPATCHES,
        *_DISTOPT_DISPATCHES,
        "dp-grad-sync-complete",
        "dp-param-sync-complete",
    )
)


def _load_rank_events(trace_root: Path) -> Mapping[int, list[tuple[int, Event]]]:
    result: dict[int, list[tuple[int, Event]]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("DP contract requires one shard per global rank")
        result[rank.global_rank] = [
            (int(iteration.iteration_id), event)
            for iteration in read_benchmark_file(rank, content)
            if iteration.iteration_id is not None
            for event in iteration.events
            if event.name in _ALL_ROUTE_EVENTS
        ]
    return result


def _failure(code: str, message: str, rank: int, iteration: int | None = None) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(code, message, evidence)


def _field_failures(
    event: Event,
    expected: Mapping[str, Any],
    rank: int,
    iteration: int,
) -> list[Failure]:
    return [
        _failure(
            "trace.dp.field",
            f"{event.name!r} has {field}={event.attrs.get(field, '<missing>')!r}, "
            f"expected {value!r}",
            rank,
            iteration,
        )
        for field, value in expected.items()
        if event.attrs.get(field, "<missing>") != value
    ]


def _validate_rank(
    events: list[tuple[int, Event]],
    dispatch_specs: Mapping[str, Mapping[str, Any]],
    *,
    distopt: bool,
    rank: int,
) -> list[Failure]:
    completion_specs = {
        "dp-grad-sync-complete": {
            "allowed": frozenset(("dp-reduce-scatter",) if distopt else ("dp-allreduce",)),
            "fields": {
                "completion_site": "finish_grad_sync",
                "force_all_reduce": False,
                "num_distributed_optimizer_instances": 1,
                "op": "wait",
                "stage": "gradient_collective_completion",
                "use_distributed_optimizer": distopt,
            },
        }
    }
    if distopt:
        completion_specs["dp-param-sync-complete"] = {
            "allowed": frozenset(("dp-param-all-gather",)),
            "fields": {
                "completion_site": "finish_param_sync",
                "op": "wait",
                "stage": "parameter_allgather_completion",
            },
        }

    expected_names = set(dispatch_specs) | set(completion_specs)
    failures = [
        _failure("trace.dp.route", f"unexpected DP route event {name!r}", rank)
        for name in sorted({event.name for _, event in events} - expected_names)
    ]
    for name in sorted(expected_names):
        begins = [event for _, event in events if event.name == name and event.ph == "B"]
        ends = [event for _, event in events if event.name == name and event.ph == "E"]
        if not begins or len(begins) != len(ends):
            failures.append(
                _failure(
                    "trace.dp.event_count",
                    f"{name!r} has B/E={len(begins)}/{len(ends)}",
                    rank,
                )
            )
        if name in completion_specs and any(
            event.attrs.get("completed") is not True for event in ends
        ):
            failures.append(
                _failure("trace.dp.completion", f"{name!r} did not complete", rank)
            )

    launches: dict[str, tuple[str, str, int]] = {}
    completed: set[str] = set()
    for position, (iteration, event) in enumerate(events):
        if event.ph != "B":
            continue
        if event.name in dispatch_specs:
            failures.extend(
                _field_failures(
                    event,
                    {**_COMMON_DISPATCH, **dispatch_specs[event.name]},
                    rank,
                    iteration,
                )
            )
            operation_id = event.attrs.get("operation_id")
            if not isinstance(operation_id, str) or operation_id in launches:
                failures.append(
                    _failure(
                        "trace.dp.operation_id",
                        f"invalid or duplicate operation_id={operation_id!r}",
                        rank,
                        iteration,
                    )
                )
            else:
                launches[operation_id] = (
                    event.name,
                    str(event.attrs.get("stage")),
                    position,
                )
            continue
        if event.name not in completion_specs:
            continue

        spec = completion_specs[event.name]
        failures.extend(
            _field_failures(
                event,
                {**_COMMON_COMPLETION, **spec["fields"]},
                rank,
                iteration,
            )
        )
        operation_ids = event.attrs.get("operation_ids")
        if (
            not isinstance(operation_ids, list)
            or not operation_ids
            or event.attrs.get("operation_count") != len(operation_ids)
        ):
            failures.append(
                _failure(
                    "trace.dp.completion",
                    f"{event.name!r} has an invalid operation list",
                    rank,
                    iteration,
                )
            )
            continue
        descriptors = event.attrs.get("operations", [])
        for operation_id in operation_ids:
            launch = launches.get(operation_id)
            if (
                not isinstance(operation_id, str)
                or launch is None
                or operation_id in completed
                or launch[0] not in spec["allowed"]
                or launch[2] >= position
            ):
                failures.append(
                    _failure(
                        "trace.dp.operation_id",
                        f"{event.name!r} cannot pair operation_id={operation_id!r}",
                        rank,
                        iteration,
                    )
                )
                continue
            if event.name == "dp-grad-sync-complete":
                expected_descriptor = {
                    "event_name": launch[0],
                    "operation_id": operation_id,
                    "stage": launch[1],
                }
                if expected_descriptor not in descriptors:
                    failures.append(
                        _failure(
                            "trace.dp.completion",
                            f"missing descriptor for operation_id={operation_id!r}",
                            rank,
                            iteration,
                        )
                    )
            elif event.attrs.get("operation_id") != operation_id or len(operation_ids) != 1:
                failures.append(
                    _failure(
                        "trace.dp.completion",
                        "parameter completion must identify one dispatch",
                        rank,
                        iteration,
                    )
                )
            completed.add(operation_id)

    if completed != set(launches):
        failures.append(
            _failure(
                "trace.dp.operation_id",
                "dispatch and completion operation IDs differ",
                rank,
            )
        )
    return failures


def _validate(
    trace_root: Path,
    dispatch_specs: Mapping[str, Mapping[str, Any]],
    *,
    distopt: bool,
) -> tuple[Failure, ...]:
    return tuple(
        failure
        for rank, events in _load_rank_events(trace_root).items()
        for failure in _validate_rank(events, dispatch_specs, distopt=distopt, rank=rank)
    )


def validate_dp_standard_overlap(trace_root: Path) -> tuple[Failure, ...]:
    return _validate(trace_root, _STANDARD_DISPATCHES, distopt=False)


def validate_dp_distopt_overlap(trace_root: Path) -> tuple[Failure, ...]:
    return _validate(trace_root, _DISTOPT_DISPATCHES, distopt=True)
