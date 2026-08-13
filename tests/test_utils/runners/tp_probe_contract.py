# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TP/SP trace contracts for controlled Transformer and MoE profiles."""

from __future__ import annotations

from collections import Counter, defaultdict
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
_QWEN3_SP_COLLECTIVES = frozenset(
    ("tp-all-gather-first", "tp-reduce-scatter")
)
_QWEN3_FORBIDDEN_LAST_DIM_COLLECTIVES = frozenset(
    ("tp-all-gather-last", "tp-reduce-scatter-last")
)
_LINEAR_EVENTS = frozenset(
    ("tp-linear-async-launch", "tp-linear-async-complete")
)
_TE_LINEAR_BOUNDARY_EVENTS = _LINEAR_EVENTS | frozenset(
    ("transformer_layer", "attention", "MLP.forward")
)
_TE_OP_FUSER_BOUNDARY_EVENTS = _LINEAR_EVENTS | frozenset(
    (
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "MLP.forward",
    )
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
_TP2_EP4_FLEX_DIRECT_SPECS = {
    "tp-reduce-scatter": (
        1,
        {"op": "reduce-scatter", "dim": "first"},
    ),
    "tp-allreduce": (
        2,
        {
            "op": "all_reduce",
            "timing_phase": "collective_call",
            "payload_role": "inplace_input_output",
        },
    ),
    "tp-all-gather-first": (
        2,
        {"op": "all-gather", "dim": "first"},
    ),
}


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
    tensor_parallel_size: int,
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
    if group_size != tensor_parallel_size:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} has group_size={group_size!r}, "
                f"expected {tensor_parallel_size}",
                rank=rank,
                iteration=iteration,
            )
        )
    expected_peers = [
        peer for peer in range(tensor_parallel_size) if peer != rank
    ]
    if group != expected_peers:
        failures.append(
            _failure(
                "trace.tp.collective_field",
                f"event {span.begin.name!r} has peer group={group!r}, "
                f"expected {expected_peers!r}",
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
    expected_counts: Mapping[str, int] | None = None,
    tensor_parallel_size: int = 2,
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
    if expected_counts is not None:
        for name, expected_count in expected_counts.items():
            observed_count = len(spans.get(name, ()))
            if observed_count != expected_count:
                failures.append(
                    _failure(
                        "trace.tp.collective_count",
                        f"event {name!r} has {observed_count} complete span(s), "
                        f"expected {expected_count}",
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
                    tensor_parallel_size=tensor_parallel_size,
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
    tensor_parallel_size: int,
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
        if begin.attrs.get("group_size") != tensor_parallel_size:
            failures.append(
                _failure(
                    "trace.tp_linear.field",
                    f"linear launch has group_size={begin.attrs.get('group_size')!r}, "
                    f"expected {tensor_parallel_size}",
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
    tp_peers: Sequence[int],
    embedding_peer: int | None,
) -> tuple[list[Failure], int | None, int | None]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_spans(iteration, _FINAL_SYNC_EVENTS, rank=rank)
    expected_counts = {
        "grad-sync": 1,
        "all-grads-sync": 1,
        "sp-layernorm-allreduce": int(expect_sp_layernorm),
        "embedding-grads-allreduce": int(embedding_peer is not None),
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
    embedding_spans = spans.get("embedding-grads-allreduce", ())
    grad = grad_spans[0] if len(grad_spans) == 1 else None
    all_grads = all_grads_spans[0] if len(all_grads_spans) == 1 else None
    layernorm = layernorm_spans[0] if len(layernorm_spans) == 1 else None
    embedding = embedding_spans[0] if len(embedding_spans) == 1 else None
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
        for child in (all_grads, layernorm, embedding):
            if child is not None and child.parent_begin_position != grad.begin_position:
                failures.append(
                    _failure(
                        "trace.tp.final_sync_parent",
                        f"event {child.begin.name!r} is not a direct child of grad-sync",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    ordered_children = tuple(
        child for child in (all_grads, layernorm, embedding) if child is not None
    )
    for previous, current in zip(ordered_children, ordered_children[1:]):
        if previous.end_position >= current.begin_position:
            failures.append(
                _failure(
                    "trace.tp.final_sync_order",
                    f"event {current.begin.name!r} does not follow "
                    f"{previous.begin.name!r}",
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
                    "group_size": len(tp_peers) + 1,
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
        if layernorm.end.attrs.get("group") != list(tp_peers):
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    f"sp-layernorm-allreduce has peer group="
                    f"{layernorm.end.attrs.get('group')!r}, "
                    f"expected {list(tp_peers)!r}",
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
    embedding_bytes: int | None = None
    if embedding is not None:
        failures.extend(
            _field_failures(
                embedding.begin,
                {
                    "group_size": 2,
                    "embedding_kind": "word",
                },
                code="trace.tp.final_sync_field",
                rank=rank,
                iteration=iteration_id,
            )
        )
        observed_bytes = embedding.begin.attrs.get("data_bytes")
        if (
            not isinstance(observed_bytes, int)
            or isinstance(observed_bytes, bool)
            or observed_bytes <= 0
        ):
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "embedding-grads-allreduce has invalid "
                    f"data_bytes={observed_bytes!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            embedding_bytes = observed_bytes
        unsupported_begin = {
            "group",
            "reduce_op",
            "grad_bucket",
            "timing_phase",
            "operation_id",
        } & embedding.begin.attrs.keys()
        if unsupported_begin:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "embedding-grads-allreduce has unsupported begin fields: "
                    f"{sorted(unsupported_begin)}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if embedding_peer is not None and embedding.end.attrs.get("group") != [
            embedding_peer
        ]:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "embedding-grads-allreduce has peer group="
                    f"{embedding.end.attrs.get('group')!r}, "
                    f"expected {[embedding_peer]!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        repeated = {
            "data_bytes",
            "group_size",
            "embedding_kind",
            "reduce_op",
            "grad_bucket",
            "timing_phase",
            "operation_id",
        } & embedding.end.attrs.keys()
        if repeated:
            failures.append(
                _failure(
                    "trace.tp.final_sync_field",
                    "embedding-grads-allreduce repeats begin fields on end: "
                    f"{sorted(repeated)}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    return failures, data_bytes, embedding_bytes


def _validate_tp_gqa_collective_hierarchy(
    trace_root: Path,
    *,
    tensor_parallel_size: int,
    required_names: frozenset[str],
    forbidden_names: frozenset[str],
    profile_name: str,
    expected_counts: Mapping[str, int] | None = None,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(tensor_parallel_size))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp.ranks",
                f"TP{tensor_parallel_size} contract expects ranks "
                f"{list(expected_ranks)}, observed {sorted(by_rank)}",
                profile_name,
            )
        )

    for rank in expected_ranks:
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
                    expected_counts=expected_counts,
                    tensor_parallel_size=tensor_parallel_size,
                )
            )
    return tuple(failures)


def validate_tp2_gqa_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    return _validate_tp_gqa_collective_hierarchy(
        trace_root,
        tensor_parallel_size=2,
        required_names=_SP_GQA_COLLECTIVES,
        forbidden_names=frozenset(),
        profile_name="tp2-gqa-sp",
    )


def validate_tp2_gqa_no_sp_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    return _validate_tp_gqa_collective_hierarchy(
        trace_root,
        tensor_parallel_size=2,
        required_names=_NO_SP_GQA_COLLECTIVES,
        forbidden_names=frozenset(("tp-all-gather-first",)),
        profile_name="tp2-gqa-no-sp",
    )


def _validate_qwen3_tp_sp_collective_hierarchy(
    trace_root: Path, *, tensor_parallel_size: int
) -> tuple[Failure, ...]:
    return _validate_tp_gqa_collective_hierarchy(
        trace_root,
        tensor_parallel_size=tensor_parallel_size,
        required_names=_QWEN3_SP_COLLECTIVES,
        forbidden_names=_QWEN3_FORBIDDEN_LAST_DIM_COLLECTIVES,
        profile_name=f"qwen3-tp{tensor_parallel_size}-sp",
        expected_counts={
            "tp-all-gather-first": 2,
            "tp-all-gather-last": 0,
            "tp-reduce-scatter": 1,
            "tp-reduce-scatter-last": 0,
        },
    )


def validate_qwen3_tp2_sp_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the first-dimension TP/SP route used by Qwen3 GQA8 on TP2."""

    return _validate_qwen3_tp_sp_collective_hierarchy(
        trace_root, tensor_parallel_size=2
    )


def validate_qwen3_tp4_sp_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the first-dimension TP/SP route used by Qwen3 GQA8 on TP4."""

    return _validate_qwen3_tp_sp_collective_hierarchy(
        trace_root, tensor_parallel_size=4
    )


def validate_qwen3_tp8_sp_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the first-dimension TP/SP route used by Qwen3 GQA8 on TP8."""

    return _validate_qwen3_tp_sp_collective_hierarchy(
        trace_root, tensor_parallel_size=8
    )


def _validate_tp_linear_lifecycle(
    trace_root: Path,
    *,
    tensor_parallel_size: int,
    expected_routes: frozenset[str],
    profile_name: str,
    expected_operation_count: int | None = None,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(tensor_parallel_size))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp_linear.ranks",
                f"TP{tensor_parallel_size} linear contract expects ranks "
                f"{list(expected_ranks)}, observed {sorted(by_rank)}",
                profile_name,
            )
        )

    for rank in expected_ranks:
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
                tensor_parallel_size=tensor_parallel_size,
            )
            failures.extend(iteration_failures)
            if (
                expected_operation_count is not None
                and len(operation_ids) != expected_operation_count
            ):
                failures.append(
                    _failure(
                        "trace.tp_linear.event_count",
                        f"linear lifecycle has {len(operation_ids)} operation(s), "
                        f"expected {expected_operation_count}",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
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
    return _validate_tp_linear_lifecycle(
        trace_root,
        tensor_parallel_size=2,
        expected_routes=_SP_LINEAR_ROUTES,
        profile_name="tp2-local-sp",
    )


def validate_tp2_local_allreduce_lifecycle(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp_linear_lifecycle(
        trace_root,
        tensor_parallel_size=2,
        expected_routes=_ALLREDUCE_LINEAR_ROUTES,
        profile_name="tp2-local-allreduce",
    )


def _validate_qwen3_tp_sp_linear_lifecycle(
    trace_root: Path, *, tensor_parallel_size: int
) -> tuple[Failure, ...]:
    return _validate_tp_linear_lifecycle(
        trace_root,
        tensor_parallel_size=tensor_parallel_size,
        expected_routes=_SP_LINEAR_ROUTES,
        profile_name=f"qwen3-tp{tensor_parallel_size}-sp",
        expected_operation_count=2,
    )


def validate_qwen3_tp2_sp_linear_lifecycle(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the two MCore Local Linear AG/RS operations in Qwen3 TP2/SP."""

    return _validate_qwen3_tp_sp_linear_lifecycle(
        trace_root, tensor_parallel_size=2
    )


def validate_qwen3_tp4_sp_linear_lifecycle(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the two MCore Local Linear AG/RS operations in Qwen3 TP4/SP."""

    return _validate_qwen3_tp_sp_linear_lifecycle(
        trace_root, tensor_parallel_size=4
    )


def validate_qwen3_tp8_sp_linear_lifecycle(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the two MCore Local Linear AG/RS operations in Qwen3 TP8/SP."""

    return _validate_qwen3_tp_sp_linear_lifecycle(
        trace_root, tensor_parallel_size=8
    )


def _validate_qwen3_tp_sp_absences(
    trace_root: Path, *, tensor_parallel_size: int
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    for rank, iterations in _load_iterations(trace_root).items():
        for iteration in iterations:
            if any(event.name == "tp-allreduce" for event in iteration.events):
                failures.append(
                    _failure(
                        "trace.tp.collective_count",
                        f"Qwen3 TP{tensor_parallel_size}/SP must not use the "
                        "non-SP TP all-reduce route",
                        rank=rank,
                        iteration=int(iteration.iteration_id),
                    )
                )
    return tuple(failures)


def _validate_tp_final_grad_sync(
    trace_root: Path,
    *,
    tensor_parallel_size: int,
    schedule: str,
    expect_sp_layernorm: bool,
    profile_name: str,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(tensor_parallel_size))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp.final_sync_ranks",
                f"TP{tensor_parallel_size} final-sync contract expects ranks "
                f"{list(expected_ranks)}, "
                f"observed {sorted(by_rank)}",
                profile_name,
            )
        )

    bytes_by_iteration: dict[int, dict[int, int]] = defaultdict(dict)
    for rank in expected_ranks:
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
            iteration_failures, data_bytes, _ = _validate_final_grad_sync(
                iteration,
                rank=rank,
                schedule=schedule,
                expect_sp_layernorm=expect_sp_layernorm,
                tp_peers=tuple(peer for peer in expected_ranks if peer != rank),
                embedding_peer=None,
            )
            failures.extend(iteration_failures)
            if data_bytes is not None:
                bytes_by_iteration[int(iteration.iteration_id)][rank] = data_bytes

    for iteration, rank_bytes in sorted(bytes_by_iteration.items()):
        if set(rank_bytes) == set(expected_ranks) and len(set(rank_bytes.values())) != 1:
            failures.append(
                Failure(
                    "trace.tp.final_sync_field",
                    f"iteration {iteration} has unequal SP payload bytes {rank_bytes}",
                    f"iteration={iteration}",
                )
            )
    return tuple(failures)


def validate_tp2_sp_final_grad_sync(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp_final_grad_sync(
        trace_root,
        tensor_parallel_size=2,
        schedule="no-pipelining",
        expect_sp_layernorm=True,
        profile_name="tp2-local-sp",
    )


def validate_tp2_no_sp_final_grad_sync(trace_root: Path) -> tuple[Failure, ...]:
    return _validate_tp_final_grad_sync(
        trace_root,
        tensor_parallel_size=2,
        schedule="no-pipelining",
        expect_sp_layernorm=False,
        profile_name="tp2-local-allreduce",
    )


def validate_tp2_pp2_embedding_final_grad_sync(
    trace_root: Path,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = (0, 1, 2, 3)
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp.final_sync_ranks",
                "TP2xPP2 embedding contract expects ranks [0, 1, 2, 3], "
                f"observed {sorted(by_rank)}",
                "tp2-pp2-embedding",
            )
        )

    rank_coordinates: dict[int, tuple[int, int]] = {}
    for rank in expected_ranks:
        iterations = by_rank.get(rank, ())
        coordinates = {
            (event.rank.data, event.rank.pipeline, event.rank.tensor)
            for iteration in iterations
            for event in iteration.events
        }
        coordinate = next(iter(coordinates)) if len(coordinates) == 1 else None
        if (
            coordinate is not None
            and coordinate[0] == 0
            and coordinate[1] in (0, 1)
            and coordinate[2] in (0, 1)
        ):
            _, pipeline_rank, tensor_rank = coordinate
            rank_coordinates[rank] = (int(pipeline_rank), int(tensor_rank))
        else:
            failures.append(
                Failure(
                    "trace.tp.final_sync_coordinates",
                    f"rank {rank} has invalid coordinates {sorted(map(str, coordinates))}",
                    f"rank={rank}",
                )
            )

    coordinate_to_rank = {
        coordinate: rank for rank, coordinate in rank_coordinates.items()
    }
    expected_coordinates = {
        (pipeline, tensor)
        for pipeline in (0, 1)
        for tensor in (0, 1)
    }
    if set(coordinate_to_rank) != expected_coordinates:
        failures.append(
            Failure(
                "trace.tp.final_sync_coordinates",
                "TP2xPP2 coordinates are "
                f"{sorted(coordinate_to_rank)}, expected {sorted(expected_coordinates)}",
                "tp2-pp2-embedding",
            )
        )

    sp_bytes: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    embedding_bytes: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for rank in expected_ranks:
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
        coordinate = rank_coordinates.get(rank)
        if coordinate is None:
            continue
        pipeline_rank, tensor_rank = coordinate
        tp_peer = coordinate_to_rank.get((pipeline_rank, 1 - tensor_rank))
        embedding_peer = coordinate_to_rank.get((1 - pipeline_rank, tensor_rank))
        if tp_peer is None or embedding_peer is None:
            continue
        for iteration in iterations:
            iteration_id = int(iteration.iteration_id)
            iteration_failures, sp_payload, embedding_payload = (
                _validate_final_grad_sync(
                    iteration,
                    rank=rank,
                    schedule="non-interleaved-1f1b",
                    expect_sp_layernorm=True,
                    tp_peers=(tp_peer,),
                    embedding_peer=embedding_peer,
                )
            )
            failures.extend(iteration_failures)
            if sp_payload is not None:
                sp_bytes[(iteration_id, pipeline_rank)][rank] = sp_payload
            if embedding_payload is not None:
                embedding_bytes[(iteration_id, tensor_rank)][rank] = embedding_payload

    for (iteration, pipeline_rank), rank_bytes in sorted(sp_bytes.items()):
        if len(rank_bytes) == 2 and len(set(rank_bytes.values())) != 1:
            failures.append(
                Failure(
                    "trace.tp.final_sync_field",
                    f"iteration {iteration} PP rank {pipeline_rank} has unequal "
                    f"SP payload bytes {rank_bytes}",
                    f"iteration={iteration} pp_rank={pipeline_rank}",
                )
            )
    for (iteration, tensor_rank), rank_bytes in sorted(embedding_bytes.items()):
        if len(rank_bytes) == 2 and len(set(rank_bytes.values())) != 1:
            failures.append(
                Failure(
                    "trace.tp.final_sync_field",
                    f"iteration {iteration} TP rank {tensor_rank} has unequal "
                    f"embedding payload bytes {rank_bytes}",
                    f"iteration={iteration} tp_rank={tensor_rank}",
                )
            )
    return tuple(failures)


def validate_tp2_pp4_multimicrobatch(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the focused eight-rank TP2xPP4 integration smoke."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(8))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp_pp.ranks",
                f"expected ranks {list(expected_ranks)}, observed {sorted(by_rank)}",
                "tp2-pp4-multimicrobatch",
            )
        )

    coordinates: dict[tuple[int, int], int] = {}
    for rank, iterations in by_rank.items():
        observed = {
            (event.rank.data, event.rank.pipeline, event.rank.tensor)
            for iteration in iterations
            for event in iteration.events
        }
        if len(observed) == 1:
            data, pipeline, tensor = next(iter(observed))
            if data == 0 and pipeline in range(4) and tensor in range(2):
                coordinates[(int(pipeline), int(tensor))] = rank
                continue
        failures.append(
            Failure(
                "trace.tp_pp.coordinates",
                f"rank {rank} has invalid coordinates {sorted(map(str, observed))}",
                f"rank={rank}",
            )
        )
    expected_coordinates = {(pp, tp) for pp in range(4) for tp in range(2)}
    if set(coordinates) != expected_coordinates:
        failures.append(
            Failure(
                "trace.tp_pp.coordinates",
                f"expected {sorted(expected_coordinates)}, observed {sorted(coordinates)}",
                "tp2-pp4-multimicrobatch",
            )
        )

    peer_stage = {
        "send-forward": 1,
        "recv-backward": 1,
        "recv-forward": -1,
        "send-backward": -1,
    }
    sent_payloads: dict[tuple[int, int, int, str], Counter[int]] = defaultdict(Counter)
    received_payloads: dict[tuple[int, int, int, str], Counter[int]] = defaultdict(Counter)
    for (pipeline_rank, tensor_rank), rank in coordinates.items():
        iterations = by_rank[rank]
        iteration_ids = tuple(item.iteration_id for item in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.tp_pp.iterations",
                    f"expected [1, 2], observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        expected_directions = {
            name
            for name, delta in peer_stage.items()
            if 0 <= pipeline_rank + delta < 4
        }
        for iteration in iterations:
            iteration_id = int(iteration.iteration_id)
            compute_spans, pairing_failures = _pair_spans(
                iteration, ("forward-step", "backward-step"), rank=rank
            )
            failures.extend(pairing_failures)
            for name in ("forward-step", "backward-step"):
                spans = compute_spans.get(name, ())
                microbatches = [
                    span.begin.attrs.get("current_microbatch") for span in spans
                ]
                operation_ids = [span.end.attrs.get("operation_id") for span in spans]
                expected_ids = [
                    f"pp:microbatch={microbatch}:vp=none" for microbatch in range(4)
                ]
                if microbatches != list(range(4)) or operation_ids != expected_ids:
                    failures.append(
                        _failure(
                            "trace.tp_pp.microbatches",
                            f"{name} microbatches={microbatches!r}, "
                            f"operation_ids={operation_ids!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )

            p2p_spans, pairing_failures = _pair_spans(
                iteration,
                (
                    "p2p-launch",
                    "p2p-batch-complete",
                    "p2p-batch-device-sync",
                    *peer_stage,
                ),
                rank=rank,
            )
            failures.extend(pairing_failures)
            launches = [span.begin for span in p2p_spans.get("p2p-launch", ())]
            operations = [
                operation
                for launch in launches
                for operation in launch.attrs.get("operations", ())
                if isinstance(operation, Mapping)
            ]
            launched_ids = [operation.get("operation_id") for operation in operations]
            completed_ids = [
                span.begin.attrs.get("operation_id")
                for name in peer_stage
                for span in p2p_spans.get(name, ())
            ]
            completed_ids.extend(
                operation_id
                for span in p2p_spans.get("p2p-batch-complete", ())
                for operation_id in span.begin.attrs.get("operation_ids", ())
            )
            if (
                any(not isinstance(operation_id, str) for operation_id in launched_ids)
                or len(set(launched_ids)) != len(launched_ids)
                or Counter(launched_ids) != Counter(completed_ids)
            ):
                failures.append(
                    _failure(
                        "trace.tp_pp.p2p_completion",
                        "P2P launch and completion operation IDs differ",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            directions = Counter(
                f"{operation.get('direction')}-{operation.get('pipeline_direction')}"
                for operation in operations
            )
            if set(directions) != expected_directions or any(
                count != 4 for count in directions.values()
            ):
                failures.append(
                    _failure(
                        "trace.tp_pp.p2p_directions",
                        f"P2P operation counts are {dict(directions)!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            for operation in operations:
                direction = (
                    f"{operation.get('direction')}-"
                    f"{operation.get('pipeline_direction')}"
                )
                expected_peer = coordinates.get(
                    (pipeline_rank + peer_stage.get(direction, 99), tensor_rank)
                )
                data_bytes = operation.get("data_bytes")
                if (
                    operation.get("peer_rank") != expected_peer
                    or not isinstance(data_bytes, int)
                    or isinstance(data_bytes, bool)
                    or data_bytes <= 0
                ):
                    failures.append(
                        _failure(
                            "trace.tp_pp.p2p_operation",
                            f"invalid adjacent-stage operation {operation!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                    continue
                key = (
                    iteration_id,
                    rank if operation.get("direction") == "send" else expected_peer,
                    expected_peer if operation.get("direction") == "send" else rank,
                    str(operation.get("pipeline_direction")),
                )
                payloads = (
                    sent_payloads
                    if operation.get("direction") == "send"
                    else received_payloads
                )
                payloads[key][data_bytes] += 1

            tp_spans, pairing_failures = _pair_spans(
                iteration, ("sp-layernorm-allreduce",), rank=rank
            )
            failures.extend(pairing_failures)
            sp_spans = tp_spans.get("sp-layernorm-allreduce", ())
            if len(sp_spans) != 1 or sp_spans[0].begin.attrs.get("group_size") != 2:
                failures.append(
                    _failure(
                        "trace.tp_pp.tp_group",
                        "expected one closed SP LayerNorm AllReduce with group_size=2",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )

    if sent_payloads != received_payloads:
        failures.append(
            Failure(
                "trace.tp_pp.p2p_payload",
                "adjacent-stage send and receive payload multisets differ",
                "tp2-pp4-multimicrobatch",
            )
        )
    return tuple(failures)


def validate_tp2_local_allreduce_profile(
    trace_root: Path,
) -> tuple[Failure, ...]:
    validators = (
        validate_tp2_gqa_no_sp_collective_hierarchy,
        validate_tp2_local_allreduce_lifecycle,
        validate_tp2_no_sp_final_grad_sync,
    )
    return tuple(
        failure
        for validator in validators
        for failure in validator(trace_root)
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


def _validate_qwen3_tp_sp_profile(
    trace_root: Path, *, tensor_parallel_size: int
) -> tuple[Failure, ...]:
    return (
        *_validate_qwen3_tp_sp_collective_hierarchy(
            trace_root, tensor_parallel_size=tensor_parallel_size
        ),
        *_validate_qwen3_tp_sp_linear_lifecycle(
            trace_root, tensor_parallel_size=tensor_parallel_size
        ),
        *_validate_tp_final_grad_sync(
            trace_root,
            tensor_parallel_size=tensor_parallel_size,
            schedule="no-pipelining",
            expect_sp_layernorm=True,
            profile_name=f"qwen3-tp{tensor_parallel_size}-sp",
        ),
        *_validate_qwen3_tp_sp_absences(
            trace_root, tensor_parallel_size=tensor_parallel_size
        ),
    )


def validate_qwen3_tp2_sp_profile(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the fixed Qwen3-0.6B TP2/SP communication boundary."""

    return _validate_qwen3_tp_sp_profile(trace_root, tensor_parallel_size=2)


def validate_qwen3_tp4_sp_profile(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the fixed Qwen3-0.6B TP4/SP communication boundary."""

    return _validate_qwen3_tp_sp_profile(trace_root, tensor_parallel_size=4)


def validate_qwen3_tp8_sp_profile(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the fixed Qwen3-0.6B TP8/SP communication boundary."""

    return _validate_qwen3_tp_sp_profile(trace_root, tensor_parallel_size=8)


def _validate_qwen3_tp4_local_no_sp_collectives(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the fixed collective counts of the local control."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(4))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp.ranks",
                "Qwen3 TP4 local/no-SP contract expects ranks [0, 1, 2, 3], "
                f"observed {sorted(by_rank)}",
                "qwen3-tp4-local-no-sp",
            )
        )

    selected_names = ("tp-allreduce", *_COLLECTIVE_SPECS)
    for rank in expected_ranks:
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
            iteration_id = int(iteration.iteration_id)
            spans, pairing_failures = _pair_spans(
                iteration,
                selected_names,
                rank=rank,
            )
            failures.extend(pairing_failures)
            observed_allreduces = len(spans.get("tp-allreduce", ()))
            if observed_allreduces != 57:
                failures.append(
                    _failure(
                        "trace.tp.allreduce_count",
                        f"event 'tp-allreduce' has {observed_allreduces} span(s), "
                        "expected 57",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            for name in _COLLECTIVE_SPECS:
                observed = len(spans.get(name, ()))
                if observed:
                    failures.append(
                        _failure(
                            "trace.tp.collective_count",
                            f"event {name!r} has {observed} span(s), expected 0",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
    return tuple(failures)


def validate_qwen3_tp4_local_no_sp_profile(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate Qwen3 TP4 with MCore local layers and sequence parallel off."""

    return (
        *_validate_qwen3_tp4_local_no_sp_collectives(trace_root),
        *_validate_tp_linear_lifecycle(
            trace_root,
            tensor_parallel_size=4,
            expected_routes=_ALLREDUCE_LINEAR_ROUTES,
            profile_name="qwen3-tp4-local-no-sp",
            expected_operation_count=57,
        ),
        *_validate_tp_final_grad_sync(
            trace_root,
            tensor_parallel_size=4,
            schedule="no-pipelining",
            expect_sp_layernorm=True,
            profile_name="qwen3-tp4-local-no-sp",
        ),
    )


def _validate_tp2_ep4_flex_iteration(
    iteration: Iteration,
    *,
    rank: int,
) -> tuple[list[Failure], set[str]]:
    """Validate the model-TP domain in the fixed TP2/ETP1/EP4 workload."""

    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    spans, pairing_failures = _pair_spans(
        iteration,
        _TP2_EP4_FLEX_DIRECT_SPECS,
        rank=rank,
    )
    failures.extend(pairing_failures)
    expected_coordinates = (rank // 2, 0, rank % 2)
    expected_peer = [rank ^ 1]
    for name, (expected_count, expected_fields) in (
        _TP2_EP4_FLEX_DIRECT_SPECS.items()
    ):
        direct_spans = spans.get(name, ())
        if len(direct_spans) != expected_count:
            failures.append(
                _failure(
                    "trace.tp_ep.collective_count",
                    f"event {name!r} has {len(direct_spans)} span(s), "
                    f"expected {expected_count}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        for span in direct_spans:
            failures.extend(
                _field_failures(
                    span.begin,
                    {**expected_fields, "group_size": 2},
                    code="trace.tp_ep.collective_field",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            data_bytes = span.begin.attrs.get("data_bytes")
            if (
                not isinstance(data_bytes, int)
                or isinstance(data_bytes, bool)
                or data_bytes <= 0
            ):
                failures.append(
                    _failure(
                        "trace.tp_ep.collective_field",
                        f"event {name!r} has invalid data_bytes={data_bytes!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            coordinates = (
                span.begin.rank.data,
                span.begin.rank.pipeline,
                span.begin.rank.tensor,
            )
            if coordinates != expected_coordinates:
                failures.append(
                    _failure(
                        "trace.tp_ep.coordinates",
                        f"event {name!r} uses coordinates {coordinates}, "
                        f"expected {expected_coordinates}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            if span.end.attrs.get("group") != expected_peer:
                failures.append(
                    _failure(
                        "trace.tp_ep.collective_group",
                        f"event {name!r} has peer group="
                        f"{span.end.attrs.get('group')!r}, expected {expected_peer!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )

    linear_failures, operation_ids = _validate_linear_lifecycle(
        iteration,
        rank=rank,
        expected_routes=_SP_LINEAR_ROUTES,
        tensor_parallel_size=2,
    )
    failures.extend(linear_failures)
    if len(operation_ids) != 2:
        failures.append(
            _failure(
                "trace.tp_ep.linear_count",
                f"model-TP Linear lifecycle has {len(operation_ids)} operation(s), "
                "expected 2",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return failures, operation_ids


def validate_tp2_ep4_flex_tp_domain(trace_root: Path) -> tuple[Failure, ...]:
    """Validate model TP2 for the fixed ETP1×EP4 Flex profile."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(8))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.tp_ep.ranks",
                f"expected ranks {expected_ranks}, observed {tuple(sorted(by_rank))}",
                "tp2-etp1-ep4-flex",
            )
        )

    for rank in expected_ranks:
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.tp_ep.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        rank_operation_ids: set[str] = set()
        for iteration in iterations:
            iteration_failures, operation_ids = _validate_tp2_ep4_flex_iteration(
                iteration,
                rank=rank,
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


def _validate_tp2_sp_te_linear_boundary(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the MCore-visible boundary around the controlled TE Linear route."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_counts = {
        "transformer_layer": 2,
        "attention": 2,
        "MLP.forward": 2,
        "tp-linear-async-launch": 2,
        "tp-linear-async-complete": 2,
    }
    expected_routes = Counter(("all-gather", "reduce-scatter"))

    for rank in (0, 1):
        for iteration in by_rank.get(rank, ()):
            iteration_id = int(iteration.iteration_id)
            spans, pairing_failures = _pair_spans(
                iteration,
                _TE_LINEAR_BOUNDARY_EVENTS,
                rank=rank,
            )
            failures.extend(pairing_failures)
            for name, expected in expected_counts.items():
                observed = len(spans.get(name, ()))
                if observed != expected:
                    failures.append(
                        _failure(
                            "trace.tp_te.scope_count",
                            f"event {name!r} has {observed} span(s), expected {expected}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )

            layers = spans.get("transformer_layer", ())
            attention = spans.get("attention", ())
            mlp = spans.get("MLP.forward", ())
            for layer in layers:
                layer_attention = tuple(
                    span
                    for span in attention
                    if span.parent_begin_position == layer.begin_position
                )
                layer_mlp = tuple(
                    span
                    for span in mlp
                    if span.parent_begin_position == layer.begin_position
                )
                if len(layer_attention) != 1 or len(layer_mlp) != 1:
                    failures.append(
                        _failure(
                            "trace.tp_te.scope_hierarchy",
                            "each transformer layer must directly contain one "
                            "attention and one MLP scope",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                    continue
                attention_span = layer_attention[0]
                mlp_span = layer_mlp[0]
                if not (
                    layer.begin_position
                    < attention_span.begin_position
                    < attention_span.end_position
                    < mlp_span.begin_position
                    < mlp_span.end_position
                    < layer.end_position
                ):
                    failures.append(
                        _failure(
                            "trace.tp_te.scope_hierarchy",
                            "transformer layer scopes must follow attention then MLP",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )

            routes = Counter(
                str(span.begin.attrs.get("collective_op"))
                for span in spans.get("tp-linear-async-launch", ())
            )
            if routes != expected_routes:
                failures.append(
                    _failure(
                        "trace.tp_te.linear_routes",
                        f"MCore-visible Linear routes are {dict(routes)}, "
                        f"expected {dict(expected_routes)}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    return tuple(failures)


def validate_tp2_sp_te_linear_profile(trace_root: Path) -> tuple[Failure, ...]:
    """Validate TE model scopes plus the TP/SP operations visible at MCore boundaries."""

    validators = (
        validate_tp2_sp_profile,
        _validate_tp2_sp_te_linear_boundary,
    )
    return tuple(
        failure
        for validator in validators
        for failure in validator(trace_root)
    )


def _validate_tp2_sp_te_op_fuser_boundary(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the outer scopes retained when TEFusedMLP replaces MCore MLP."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_counts = {
        "transformer_layer": 2,
        "_forward_attention": 2,
        "attention": 2,
        "_forward_mlp": 2,
        "MLP.forward": 0,
        "tp-linear-async-launch": 2,
        "tp-linear-async-complete": 2,
    }
    expected_routes = Counter(("all-gather", "reduce-scatter"))

    for rank in (0, 1):
        for iteration in by_rank.get(rank, ()):
            iteration_id = int(iteration.iteration_id)
            spans, pairing_failures = _pair_spans(
                iteration,
                _TE_OP_FUSER_BOUNDARY_EVENTS,
                rank=rank,
            )
            failures.extend(pairing_failures)
            for name, expected in expected_counts.items():
                observed = len(spans.get(name, ()))
                if observed != expected:
                    failures.append(
                        _failure(
                            "trace.tp_te_op_fuser.scope_count",
                            f"event {name!r} has {observed} span(s), expected {expected}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )

            layers = spans.get("transformer_layer", ())
            attention_outer = spans.get("_forward_attention", ())
            attention = spans.get("attention", ())
            mlp = spans.get("_forward_mlp", ())
            for layer in layers:
                layer_attention_outer = tuple(
                    span
                    for span in attention_outer
                    if span.parent_begin_position == layer.begin_position
                )
                layer_mlp = tuple(
                    span
                    for span in mlp
                    if span.parent_begin_position == layer.begin_position
                )
                if len(layer_attention_outer) != 1 or len(layer_mlp) != 1:
                    failures.append(
                        _failure(
                            "trace.tp_te_op_fuser.scope_hierarchy",
                            "each transformer layer must directly contain one outer "
                            "attention scope and one outer MLP scope",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                    continue
                attention_outer_span = layer_attention_outer[0]
                inner_attention = tuple(
                    span
                    for span in attention
                    if span.parent_begin_position == attention_outer_span.begin_position
                )
                if len(inner_attention) != 1:
                    failures.append(
                        _failure(
                            "trace.tp_te_op_fuser.scope_hierarchy",
                            "each outer attention scope must directly contain one attention scope",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                    continue
                attention_span = inner_attention[0]
                mlp_span = layer_mlp[0]
                if not (
                    layer.begin_position
                    < attention_outer_span.begin_position
                    < attention_span.begin_position
                    < attention_span.end_position
                    < attention_outer_span.end_position
                    < mlp_span.begin_position
                    < mlp_span.end_position
                    < layer.end_position
                ):
                    failures.append(
                        _failure(
                            "trace.tp_te_op_fuser.scope_hierarchy",
                            "transformer layer scopes must follow outer attention then outer MLP",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )

            routes = Counter(
                str(span.begin.attrs.get("collective_op"))
                for span in spans.get("tp-linear-async-launch", ())
            )
            if routes != expected_routes:
                failures.append(
                    _failure(
                        "trace.tp_te_op_fuser.linear_routes",
                        f"MCore-visible Linear routes are {dict(routes)}, "
                        f"expected {dict(expected_routes)}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    return tuple(failures)


def validate_tp2_sp_te_op_fuser_profile(trace_root: Path) -> tuple[Failure, ...]:
    """Validate TE op-fuser outer scopes plus MCore-visible TP/SP operations."""

    validators = (
        validate_tp2_sp_profile,
        _validate_tp2_sp_te_op_fuser_boundary,
    )
    return tuple(
        failure
        for validator in validators
        for failure in validator(trace_root)
    )
