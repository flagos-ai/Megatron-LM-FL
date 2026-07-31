# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TP/SP trace contracts for the controlled local Transformer profile."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_COLLECTIVE_SPECS = {
    "tp-all-gather-first": {"op": "all-gather", "dim": "first"},
    "tp-all-gather-last": {"op": "all-gather", "dim": "last"},
    "tp-reduce-scatter": {"op": "reduce-scatter", "dim": "first"},
    "tp-reduce-scatter-last": {"op": "reduce-scatter", "dim": "last"},
}
_SP_GQA_COLLECTIVES = frozenset(_COLLECTIVE_SPECS)
_NO_SP_GQA_COLLECTIVES = frozenset(
    (
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
    )
)
_LINEAR_EVENTS = frozenset(
    ("tp-linear-async-launch", "tp-linear-async-complete")
)
_LINEAR_ROUTE_SPECS = {
    "all-gather": {
        "dim": "first",
        "launch_site": "linear_backward_wgrad_input_all_gather",
        "payload_role": "weight_gradient_input",
        "completion_site": "linear_backward_wgrad_input_ready",
        "wait_role": "dependency",
    },
    "reduce-scatter": {
        "dim": "first",
        "launch_site": "linear_backward_dgrad_reduce_scatter",
        "payload_role": "input_gradient",
        "completion_site": "linear_backward_dgrad_reduce_scatter_return",
        "wait_role": "return",
    },
    "all-reduce": {
        "launch_site": "linear_backward_dgrad_all_reduce",
        "payload_role": "input_gradient",
        "completion_site": "linear_backward_dgrad_all_reduce_return",
        "wait_role": "return",
    },
}
_SP_LINEAR_ROUTES = frozenset(("all-gather", "reduce-scatter"))
_ALLREDUCE_LINEAR_ROUTES = frozenset(("all-reduce",))
_LINEAR_LAUNCH_FIELDS = {
    "operation_id_scope": "rank_local",
    "execution_route": "local_linear_direct_async",
    "pass_direction": "backward",
    "async_op": True,
    "completion_included": False,
    "timing_phase": "launch_attempt",
}
_LINEAR_COMPLETION_FIELDS = {
    "operation_id_scope": "rank_local",
    "execution_route": "local_linear_direct_async",
    "pass_direction": "backward",
    "completion_guarantee": "current_stream_after_wait",
    "completion_included": True,
    "completion_kind": "work_wait",
    "duration_attribution": "per_request",
    "global_device_completion_guaranteed": False,
    "host_blocking_guaranteed": False,
    "launch_observed": True,
    "op": "wait",
    "terminal": True,
    "timing_phase": "stream_dependency",
}
_LINEAR_MATCH_FIELDS = (
    "collective_op",
    "data_bytes",
    "group_size",
    "dim",
    "launch_site",
    "payload_role",
)
_FINAL_SYNC_EVENTS = frozenset(
    (
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
        "embedding-grads-allreduce",
    )
)


@dataclass(frozen=True)
class _Span:
    begin: Event
    end: Event
    begin_position: int
    end_position: int
    parent_begin_position: int | None


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


def _field_failures(
    event: Event,
    expected: Mapping[str, object],
    *,
    code: str,
    rank: int,
    iteration: int,
) -> list[Failure]:
    return [
        _failure(
            code,
            f"event {event.name!r} has {field}="
            f"{event.attrs.get(field, '<missing>')!r}, expected {value!r}",
            rank=rank,
            iteration=iteration,
        )
        for field, value in expected.items()
        if event.attrs.get(field, "<missing>") != value
    ]


def _pair_spans(
    iteration: Iteration,
    names: Iterable[str],
    *,
    rank: int,
) -> tuple[Mapping[str, Sequence[_Span]], list[Failure]]:
    selected = frozenset(names)
    pending: list[tuple[str, int, Event, int | None]] = []
    spans: dict[str, list[_Span]] = defaultdict(list)
    failures: list[Failure] = []
    iteration_id = int(iteration.iteration_id)

    for position, event in enumerate(iteration.events):
        if event.name not in selected:
            continue
        if event.ph == "B":
            parent_position = pending[-1][1] if pending else None
            pending.append((event.name, position, event, parent_position))
        elif event.ph == "E":
            if not pending or pending[-1][0] != event.name:
                failures.append(
                    _failure(
                        "trace.tp.nesting",
                        f"event {event.name!r} does not close the active TP scope",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            _, begin_position, begin, parent_position = pending.pop()
            spans[event.name].append(
                _Span(begin, event, begin_position, position, parent_position)
            )
        else:
            failures.append(
                _failure(
                    "trace.tp.phase",
                    f"event {event.name!r} uses unsupported phase {event.ph!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    if pending:
        failures.append(
            _failure(
                "trace.tp.unmatched_begin",
                f"TP collective scopes have {len(pending)} unmatched begin record(s)",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return spans, failures


def _validate_collective_span(
    span: _Span,
    *,
    rank: int,
    iteration: int,
) -> list[Failure]:
    expected = _COLLECTIVE_SPECS[span.begin.name]
    failures = [
        _failure(
            "trace.tp.collective_field",
            f"event {span.begin.name!r} has {field}="
            f"{span.begin.attrs.get(field, '<missing>')!r}, expected {value!r}",
            rank=rank,
            iteration=iteration,
        )
        for field, value in expected.items()
        if span.begin.attrs.get(field) != value
    ]
    data_bytes = span.begin.attrs.get("data_bytes")
    group_size = span.begin.attrs.get("group_size")
    group = span.end.attrs.get("group")
    if "group" in span.begin.attrs:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} records peer group on begin",
                rank=rank,
                iteration=iteration,
            )
        )
    if not isinstance(data_bytes, int) or isinstance(data_bytes, bool) or data_bytes <= 0:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} has invalid data_bytes={data_bytes!r}",
                rank=rank,
                iteration=iteration,
            )
        )
    if group_size != 2:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} has group_size={group_size!r}, expected 2",
                rank=rank,
                iteration=iteration,
            )
        )
    if group != [1 - rank]:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} has peer group={group!r}, "
                f"expected {[1 - rank]!r}",
                rank=rank,
                iteration=iteration,
            )
        )
    begin_only_fields = {"op", "dim", "data_bytes", "group_size", "split_sizes"}
    duplicated = sorted(begin_only_fields & span.end.attrs.keys())
    if duplicated:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} repeats begin fields on end: {duplicated}",
                rank=rank,
                iteration=iteration,
            )
        )
    if "split_sizes" in span.begin.attrs and span.begin.attrs.get("dim") != "first":
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} uses split_sizes outside first dimension",
                rank=rank,
                iteration=iteration,
            )
        )
    return failures


def _validate_collective_hierarchy(
    iteration: Iteration,
    *,
    rank: int,
    required_names: frozenset[str],
    forbidden_names: frozenset[str] = frozenset(),
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_spans(iteration, _COLLECTIVE_SPECS, rank=rank)
    for name in required_names:
        if not spans.get(name):
            failures.append(
                _failure(
                    "trace.tp.collective_count",
                    f"event {name!r} has no complete span",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    for name in forbidden_names:
        if spans.get(name):
            failures.append(
                _failure(
                    "trace.tp.collective_count",
                    f"event {name!r} must be absent from this profile",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    for name in _COLLECTIVE_SPECS:
        for span in spans.get(name, ()):
            failures.extend(
                _validate_collective_span(
                    span,
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    physical_spans = spans.get("tp-reduce-scatter", ())
    for outer in spans.get("tp-reduce-scatter-last", ()):
        nested = [
            inner
            for inner in physical_spans
            if inner.parent_begin_position == outer.begin_position
        ]
        if len(nested) != 1:
            failures.append(
                _failure(
                    "trace.tp.collective_hierarchy",
                    "tp-reduce-scatter-last must contain exactly one "
                    "tp-reduce-scatter physical span",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        inner = nested[0]
        for field in ("data_bytes", "group_size"):
            if outer.begin.attrs.get(field) != inner.begin.attrs.get(field):
                failures.append(
                    _failure(
                        "trace.tp.collective_hierarchy",
                        f"nested ReduceScatter spans disagree on {field}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        if outer.end.attrs.get("group") != inner.end.attrs.get("group"):
            failures.append(
                _failure(
                    "trace.tp.collective_hierarchy",
                    "nested ReduceScatter spans disagree on peer group",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    return failures


def _validate_linear_lifecycle(
    iteration: Iteration,
    *,
    rank: int,
    expected_routes: frozenset[str],
) -> tuple[list[Failure], set[str]]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_spans(iteration, _LINEAR_EVENTS, rank=rank)
    launches = spans.get("tp-linear-async-launch", ())
    completions = spans.get("tp-linear-async-complete", ())
    if not launches or len(launches) != len(completions):
        failures.append(
            _failure(
                "trace.tp_linear.event_count",
                f"linear launch/completion spans={len(launches)}/{len(completions)}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    launched: dict[str, _Span] = {}
    observed_routes: set[str] = set()
    for span in launches:
        begin = span.begin
        operation_id = begin.attrs.get("operation_id")
        route = begin.attrs.get("collective_op")
        failures.extend(
            _field_failures(
                begin,
                _LINEAR_LAUNCH_FIELDS,
                code="trace.tp_linear.field",
                rank=rank,
                iteration=iteration_id,
            )
        )
        if route not in _LINEAR_ROUTE_SPECS:
            failures.append(
                _failure(
                    "trace.tp_linear.route",
                    f"local Linear profile observed unsupported route {route!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            observed_routes.add(str(route))
            route_fields = {
                field: value
                for field, value in _LINEAR_ROUTE_SPECS[str(route)].items()
                if field not in {"completion_site", "wait_role"}
            }
            failures.extend(
                _field_failures(
                    begin,
                    route_fields,
                    code="trace.tp_linear.field",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            if "dim" not in _LINEAR_ROUTE_SPECS[str(route)] and "dim" in begin.attrs:
                failures.append(
                    _failure(
                        "trace.tp_linear.field",
                        f"route {route!r} records unsupported dim={begin.attrs['dim']!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        data_bytes = begin.attrs.get("data_bytes")
        if (
            not isinstance(data_bytes, int)
            or isinstance(data_bytes, bool)
            or data_bytes <= 0
        ):
            failures.append(
                _failure(
                    "trace.tp_linear.field",
                    f"linear launch has invalid data_bytes={data_bytes!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if begin.attrs.get("group_size") != 2:
            failures.append(
                _failure(
                    "trace.tp_linear.field",
                    f"linear launch has group_size={begin.attrs.get('group_size')!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if (
            not isinstance(operation_id, str)
            or not operation_id.startswith("tp-linear:")
            or operation_id in launched
        ):
            failures.append(
                _failure(
                    "trace.tp_linear.operation_id",
                    f"invalid or duplicate operation_id={operation_id!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            launched[operation_id] = span
        failures.extend(
            _field_failures(
                span.end,
                {"api_returned": True, "error_type": None},
                code="trace.tp_linear.completion",
                rank=rank,
                iteration=iteration_id,
            )
        )

    if observed_routes != set(expected_routes):
        failures.append(
            _failure(
                "trace.tp_linear.route",
                f"local Linear routes are {sorted(observed_routes)}, "
                f"expected {sorted(expected_routes)}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    completed: set[str] = set()
    for span in completions:
        begin = span.begin
        operation_id = begin.attrs.get("operation_id")
        route = begin.attrs.get("collective_op")
        failures.extend(
            _field_failures(
                begin,
                _LINEAR_COMPLETION_FIELDS,
                code="trace.tp_linear.field",
                rank=rank,
                iteration=iteration_id,
            )
        )
        if route in _LINEAR_ROUTE_SPECS:
            route_spec = _LINEAR_ROUTE_SPECS[str(route)]
            route_fields = {
                "launch_site": route_spec["launch_site"],
                "payload_role": route_spec["payload_role"],
                "completion_site": route_spec["completion_site"],
                "wait_role": route_spec["wait_role"],
            }
            if "dim" in route_spec:
                route_fields["dim"] = route_spec["dim"]
            failures.extend(
                _field_failures(
                    begin,
                    route_fields,
                    code="trace.tp_linear.field",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            if "dim" not in route_spec and "dim" in begin.attrs:
                failures.append(
                    _failure(
                        "trace.tp_linear.field",
                        f"route {route!r} records unsupported dim={begin.attrs['dim']!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        launch = launched.get(operation_id) if isinstance(operation_id, str) else None
        if (
            launch is None
            or operation_id in completed
            or launch.end_position >= span.begin_position
        ):
            failures.append(
                _failure(
                    "trace.tp_linear.operation_id",
                    f"completion cannot pair operation_id={operation_id!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            for field in _LINEAR_MATCH_FIELDS:
                if launch.begin.attrs.get(field) != begin.attrs.get(field):
                    failures.append(
                        _failure(
                            "trace.tp_linear.field",
                            f"operation_id={operation_id!r} disagrees on {field}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
            completed.add(operation_id)
        failures.extend(
            _field_failures(
                span.end,
                {"completed": True, "error_type": None},
                code="trace.tp_linear.completion",
                rank=rank,
                iteration=iteration_id,
            )
        )

    if completed != set(launched):
        failures.append(
            _failure(
                "trace.tp_linear.operation_id",
                "linear launch and completion operation IDs differ",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return failures, set(launched)


def _validate_final_grad_sync(
    iteration: Iteration,
    *,
    rank: int,
    schedule: str,
    expect_sp_layernorm: bool,
) -> tuple[list[Failure], int | None]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_spans(iteration, _FINAL_SYNC_EVENTS, rank=rank)
    expected_counts = {
        "grad-sync": 1,
        "all-grads-sync": 1,
        "sp-layernorm-allreduce": int(expect_sp_layernorm),
        "embedding-grads-allreduce": 0,
    }
    for name, expected in expected_counts.items():
        observed = len(spans.get(name, ()))
        if observed != expected:
            failures.append(
                _failure(
                    "trace.tp.final_sync_count",
                    f"event {name!r} has {observed} span(s), expected {expected}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    grad_spans = spans.get("grad-sync", ())
    all_grads_spans = spans.get("all-grads-sync", ())
    layernorm_spans = spans.get("sp-layernorm-allreduce", ())
    grad = grad_spans[0] if len(grad_spans) == 1 else None
    all_grads = all_grads_spans[0] if len(all_grads_spans) == 1 else None
    layernorm = layernorm_spans[0] if len(layernorm_spans) == 1 else None
    if grad is not None:
        failures.extend(
            _field_failures(
                grad.begin,
                {
                    "schedule": schedule,
                    "timing_phase": "framework_phase",
                },
                code="trace.tp.final_sync_field",
                rank=rank,
                iteration=iteration_id,
            )
        )
        repeated = {"schedule", "timing_phase"} & grad.end.attrs.keys()
        if repeated:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    f"grad-sync repeats begin fields on end: {sorted(repeated)}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    if grad is not None:
        for child in (all_grads, layernorm):
            if child is not None and child.parent_begin_position != grad.begin_position:
                failures.append(
                    _failure(
                        "trace.tp.final_sync_parent",
                        f"event {child.begin.name!r} is not a direct child of grad-sync",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    if (
        all_grads is not None
        and layernorm is not None
        and all_grads.end_position >= layernorm.begin_position
    ):
        failures.append(
            _failure(
                "trace.tp.final_sync_order",
                "sp-layernorm-allreduce does not follow all-grads-sync",
                rank=rank,
                iteration=iteration_id,
            )
        )

    data_bytes: int | None = None
    if layernorm is not None:
        failures.extend(
            _field_failures(
                layernorm.begin,
                {
                    "group_size": 2,
                    "reduce_op": "SUM",
                    "grad_bucket": "sum",
                },
                code="trace.tp.final_sync_field",
                rank=rank,
                iteration=iteration_id,
            )
        )
        observed_bytes = layernorm.begin.attrs.get("data_bytes")
        if (
            not isinstance(observed_bytes, int)
            or isinstance(observed_bytes, bool)
            or observed_bytes <= 0
        ):
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    f"sp-layernorm-allreduce has invalid data_bytes={observed_bytes!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            data_bytes = observed_bytes
        if "group" in layernorm.begin.attrs:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "sp-layernorm-allreduce records peer group on begin",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if layernorm.end.attrs.get("group") != [1 - rank]:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    f"sp-layernorm-allreduce has peer group="
                    f"{layernorm.end.attrs.get('group')!r}, expected {[1 - rank]!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        repeated = {
            "data_bytes",
            "group_size",
            "reduce_op",
            "grad_bucket",
            "timing_phase",
        } & layernorm.end.attrs.keys()
        if repeated:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "sp-layernorm-allreduce repeats begin fields on end: "
                    f"{sorted(repeated)}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if "timing_phase" in layernorm.begin.attrs:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "sp-layernorm-allreduce records an unsupported timing_phase",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    return failures, data_bytes


def _validate_tp2_gqa_collective_hierarchy(
    trace_root: Path,
    *,
    required_names: frozenset[str],
    forbidden_names: frozenset[str],
    profile_name: str,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            Failure(
                "trace.tp.ranks",
                f"TP2 contract expects ranks [0, 1], observed {sorted(by_rank)}",
                profile_name,
            )
        )

    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.tp.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        for iteration in iterations:
            observed_coordinates = {
                (event.rank.data, event.rank.pipeline, event.rank.tensor)
                for event in iteration.events
            }
            if observed_coordinates != {(0, 0, rank)}:
                failures.append(
                    _failure(
                        "trace.tp.coordinates",
                        f"events use coordinates {sorted(observed_coordinates)}, "
                        f"expected [(0, 0, {rank})]",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
            failures.extend(
                _validate_collective_hierarchy(
                    iteration,
                    rank=rank,
                    required_names=required_names,
                    forbidden_names=forbidden_names,
                )
            )
    return tuple(failures)


def validate_tp2_gqa_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    return _validate_tp2_gqa_collective_hierarchy(
        trace_root,
        required_names=_SP_GQA_COLLECTIVES,
        forbidden_names=frozenset(),
        profile_name="tp2-gqa-sp",
    )


def validate_tp2_gqa_no_sp_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    return _validate_tp2_gqa_collective_hierarchy(
        trace_root,
        required_names=_NO_SP_GQA_COLLECTIVES,
        forbidden_names=frozenset(("tp-all-gather-first",)),
        profile_name="tp2-gqa-no-sp",
    )


def _validate_tp2_linear_lifecycle(
    trace_root: Path,
    *,
    expected_routes: frozenset[str],
    profile_name: str,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            Failure(
                "trace.tp_linear.ranks",
                f"TP2 linear contract expects ranks [0, 1], observed {sorted(by_rank)}",
                profile_name,
            )
        )

    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.tp_linear.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        rank_operation_ids: set[str] = set()
        for iteration in iterations:
            iteration_failures, operation_ids = _validate_linear_lifecycle(
                iteration,
                rank=rank,
                expected_routes=expected_routes,
            )
            failures.extend(iteration_failures)
            duplicates = rank_operation_ids & operation_ids
            if duplicates:
                failures.append(
                    _failure(
                        "trace.tp_linear.operation_id",
                        f"operation IDs repeat across iterations: {sorted(duplicates)}",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
            rank_operation_ids.update(operation_ids)
    return tuple(failures)


def validate_tp2_sp_linear_lifecycle(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp2_linear_lifecycle(
        trace_root,
        expected_routes=_SP_LINEAR_ROUTES,
        profile_name="tp2-local-sp",
    )


def validate_tp2_local_allreduce_lifecycle(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp2_linear_lifecycle(
        trace_root,
        expected_routes=_ALLREDUCE_LINEAR_ROUTES,
        profile_name="tp2-local-allreduce",
    )


def _validate_tp2_final_grad_sync(
    trace_root: Path,
    *,
    schedule: str,
    expect_sp_layernorm: bool,
    profile_name: str,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            Failure(
                "trace.tp.final_sync_ranks",
                f"TP2 final-sync contract expects ranks [0, 1], "
                f"observed {sorted(by_rank)}",
                profile_name,
            )
        )

    bytes_by_iteration: dict[int, dict[int, int]] = defaultdict(dict)
    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.tp.final_sync_iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        for iteration in iterations:
            iteration_failures, data_bytes = _validate_final_grad_sync(
                iteration,
                rank=rank,
                schedule=schedule,
                expect_sp_layernorm=expect_sp_layernorm,
            )
            failures.extend(iteration_failures)
            if data_bytes is not None:
                bytes_by_iteration[int(iteration.iteration_id)][rank] = data_bytes

    for iteration, rank_bytes in sorted(bytes_by_iteration.items()):
        if set(rank_bytes) == {0, 1} and len(set(rank_bytes.values())) != 1:
            failures.append(
                Failure(
                    "trace.tp.final_sync_field",
                    f"iteration {iteration} has unequal SP payload bytes {rank_bytes}",
                    f"iteration={iteration}",
                )
            )
    return tuple(failures)


def validate_tp2_sp_final_grad_sync(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp2_final_grad_sync(
        trace_root,
        schedule="no-pipelining",
        expect_sp_layernorm=True,
        profile_name="tp2-local-sp",
    )


def validate_tp2_no_sp_final_grad_sync(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp2_final_grad_sync(
        trace_root,
        schedule="no-pipelining",
        expect_sp_layernorm=False,
        profile_name="tp2-local-allreduce",
    )


def validate_tp2_sp_profile(trace_root: Path) -> tuple[Failure, ...]:
    validators = (
        validate_tp2_gqa_collective_hierarchy,
        validate_tp2_sp_linear_lifecycle,
        validate_tp2_sp_final_grad_sync,
    )
    return tuple(
        failure
        for validator in validators
        for failure in validator(trace_root)
    )
