# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CP coexistence contracts for controlled GPT training profiles."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners import dp_probe_contract
from tests.test_utils.runners import gpt_probe_contract
from tests.test_utils.runners.megalens_run_manifest import Failure


@dataclass(frozen=True)
class _Span:
    begin: Event
    end: Event
    begin_position: int
    end_position: int
    parent_begin_position: int | None


_DP_SCOPE_NAMES = frozenset(("grad-sync", "all-grads-sync", "dp-allreduce"))
_FORBIDDEN_DP_EVENTS = frozenset(
    ("dp-grad-sync-complete", "dp-reduce-scatter", "dp-param-all-gather")
)
_INFRASTRUCTURE_FIELDS = frozenset(
    ("dev", "iteration", "g_rk", "dp_rk", "pp_rk", "tp_rk")
)
_OPERATION_ID_PATTERN = re.compile(r"^dp:allreduce:[1-9]\d*$")
_DISTOPT_ROUTE_NAMES = frozenset(
    (
        "dp-reduce-scatter",
        "dp-grad-sync-complete",
        "dp-param-all-gather",
        "dp-param-sync-complete",
    )
)


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


def _pair_dp_scopes(
    iteration: Iteration,
    *,
    rank: int,
) -> tuple[Mapping[str, Sequence[_Span]], list[Failure]]:
    pending: list[tuple[str, int, Event, int | None]] = []
    spans: dict[str, list[_Span]] = defaultdict(list)
    failures: list[Failure] = []
    iteration_id = int(iteration.iteration_id)

    for position, event in enumerate(iteration.events):
        if event.name not in _DP_SCOPE_NAMES:
            continue
        if event.ph == "B":
            parent_position = pending[-1][1] if pending else None
            pending.append((event.name, position, event, parent_position))
        elif event.ph == "E":
            if not pending or pending[-1][0] != event.name:
                failures.append(
                    _failure(
                        "trace.cp.dp_sync_pairing",
                        f"event {event.name!r} does not close the active DP scope",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            name, begin_position, begin, parent_position = pending.pop()
            spans[name].append(
                _Span(begin, event, begin_position, position, parent_position)
            )
        else:
            failures.append(
                _failure(
                    "trace.cp.dp_sync_phase",
                    f"event {event.name!r} uses unsupported phase {event.ph!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    if pending:
        failures.append(
            _failure(
                "trace.cp.dp_sync_pairing",
                f"DP sync scopes have {len(pending)} unmatched begin record(s)",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return spans, failures


def _validate_dp_cp_group(
    iteration: Iteration,
    *,
    rank: int,
    context_parallel_size: int,
) -> tuple[list[Failure], str | None, int | None]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_dp_scopes(iteration, rank=rank)
    for name in _DP_SCOPE_NAMES:
        observed = len(spans.get(name, ()))
        if observed != 1:
            failures.append(
                _failure(
                    "trace.cp.dp_sync_count",
                    f"event {name!r} has {observed} span(s), expected 1",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    forbidden = sorted(
        {event.name for event in iteration.events} & _FORBIDDEN_DP_EVENTS
    )
    if forbidden:
        failures.append(
            _failure(
                "trace.cp.dp_route",
                f"CP{context_parallel_size} standard DDP observed "
                f"forbidden events {forbidden}",
                rank=rank,
                iteration=iteration_id,
            )
        )

    grad_sync = spans.get("grad-sync", ())
    all_grads_sync = spans.get("all-grads-sync", ())
    allreduces = spans.get("dp-allreduce", ())
    if len(grad_sync) == len(all_grads_sync) == len(allreduces) == 1:
        if all_grads_sync[0].parent_begin_position != grad_sync[0].begin_position:
            failures.append(
                _failure(
                    "trace.cp.dp_sync_hierarchy",
                    "all-grads-sync must be a direct child of grad-sync",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        if allreduces[0].parent_begin_position != all_grads_sync[0].begin_position:
            failures.append(
                _failure(
                    "trace.cp.dp_sync_hierarchy",
                    "dp-allreduce must be a direct child of all-grads-sync",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    if len(allreduces) != 1:
        return failures, None, None

    expected_begin = {
        "api_async_op": False,
        "async_op": False,
        "completion_included": False,
        "op": "all_reduce",
        "group_role": "data_parallel",
        "group_size": context_parallel_size,
        "n_buckets": 1,
        "operation_id_scope": "rank_local",
        "overlap_enabled": False,
        "payload_role": "gradient_bucket",
        "stage": "main_bucket_allreduce",
        "timing_phase": "collective_call",
    }
    expected_peer = [
        peer for peer in range(context_parallel_size) if peer != rank
    ]
    span = allreduces[0]
    for field, expected in expected_begin.items():
        observed = span.begin.attrs.get(field, "<missing>")
        if observed != expected:
            failures.append(
                _failure(
                    "trace.cp.dp_route",
                    f"dp-allreduce has {field}={observed!r}, expected {expected!r}",
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
                "trace.cp.dp_allreduce_payload",
                f"dp-allreduce has invalid data_bytes={data_bytes!r}",
                rank=rank,
                iteration=iteration_id,
            )
        )
        data_bytes = None

    operation_id = span.begin.attrs.get("operation_id")
    if not isinstance(operation_id, str) or _OPERATION_ID_PATTERN.fullmatch(
        operation_id
    ) is None:
        failures.append(
            _failure(
                "trace.cp.dp_operation_id",
                f"dp-allreduce has invalid rank-local operation_id={operation_id!r}",
                rank=rank,
                iteration=iteration_id,
            )
        )
        operation_id = None

    end_fields = set(span.end.attrs) - _INFRASTRUCTURE_FIELDS
    if "group" in span.begin.attrs or end_fields != {"group"}:
        failures.append(
            _failure(
                "trace.cp.dp_cp_group",
                "dp-allreduce must record group only on its end; "
                f"observed begin={span.begin.attrs.get('group', '<absent>')!r}, "
                f"end fields={sorted(end_fields)}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    if span.end.attrs.get("group") != expected_peer:
        failures.append(
            _failure(
                "trace.cp.dp_cp_group",
                f"dp-allreduce peer group is {span.end.attrs.get('group')!r}, "
                f"expected {expected_peer!r}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return failures, operation_id, data_bytes


def _validate_tp1_pp1_dp1_cp_identity(
    iteration: Iteration,
    *,
    rank: int,
) -> list[Failure]:
    """Validate the rank identity of the controlled TP1/PP1/DP1 CP profiles."""

    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    cp_events = sorted(
        {
            event.name
            for event in iteration.events
            if event.name.startswith(("cp-", "cp_"))
        }
    )
    if cp_events:
        failures.append(
            _failure(
                "trace.cp.unexpected_event",
                f"trace contains CP-specific events {cp_events}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    cp_fields = sorted(
        {
            field
            for event in iteration.events
            for field in event.attrs
            if field.startswith("cp_")
        }
    )
    if cp_fields:
        failures.append(
            _failure(
                "trace.cp.unexpected_field",
                f"trace contains CP-specific fields {cp_fields}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    observed_coordinates = {
        (
            event.rank.global_rank,
            event.rank.data,
            event.rank.pipeline,
            event.rank.tensor,
        )
        for event in iteration.events
    }
    expected_coordinates = {(rank, 0, 0, 0)}
    if observed_coordinates != expected_coordinates:
        failures.append(
            _failure(
                "trace.cp.coordinates",
                f"events use rank coordinates {sorted(observed_coordinates)}, "
                f"expected {sorted(expected_coordinates)}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return failures


def _validate_te_dp1_coexistence(
    trace_root: Path, *, context_parallel_size: int
) -> tuple[Failure, ...]:
    failures = list(
        gpt_probe_contract.validate_gpt_cp_dp1_eager_phases(
            trace_root, context_parallel_size=context_parallel_size
        )
    )
    by_rank = _load_iterations(trace_root)
    payload_sizes: dict[tuple[int, int], int] = {}
    expected_ranks = tuple(range(context_parallel_size))
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != expected_ranks:
        failures.append(
            Failure(
                "trace.cp.ranks",
                f"CP{context_parallel_size} contract expects ranks "
                f"{list(expected_ranks)}, observed {list(observed_ranks)}",
                f"cp{context_parallel_size}-te-coexistence",
            )
        )

    for rank in expected_ranks:
        iterations = by_rank.get(rank, ())
        operation_ids: set[str] = set()
        iteration_ids = tuple(item.iteration_id for item in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.cp.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        for iteration in iterations:
            iteration_id = int(iteration.iteration_id)
            failures.extend(
                _validate_tp1_pp1_dp1_cp_identity(
                    iteration,
                    rank=rank,
                )
            )
            iteration_failures, operation_id, data_bytes = _validate_dp_cp_group(
                iteration,
                rank=rank,
                context_parallel_size=context_parallel_size,
            )
            failures.extend(iteration_failures)
            if data_bytes is not None:
                payload_sizes[(rank, iteration_id)] = data_bytes
            if operation_id is not None:
                if operation_id in operation_ids:
                    failures.append(
                        _failure(
                            "trace.cp.dp_operation_id",
                            f"operation_id={operation_id!r} repeats across iterations",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                operation_ids.add(operation_id)
    if len(set(payload_sizes.values())) > 1:
        failures.append(
            Failure(
                "trace.cp.dp_payload_consistency",
                "dp-allreduce data_bytes differ across CP ranks or iterations",
                str(dict(sorted(payload_sizes.items()))),
            )
        )
    return tuple(failures)


def validate_cp2_te_coexistence(trace_root: Path) -> tuple[Failure, ...]:
    """Validate existing GPT and DP probes for TP1/PP1/CP2/DP1."""

    return _validate_te_dp1_coexistence(trace_root, context_parallel_size=2)


def validate_cp4_te_coexistence(trace_root: Path) -> tuple[Failure, ...]:
    """Validate existing GPT and DP probes for TP1/PP1/CP4/DP1."""

    return _validate_te_dp1_coexistence(trace_root, context_parallel_size=4)


def _validate_qwen3_distopt_iteration(
    iteration: Iteration,
    *,
    rank: int,
    context_parallel_size: int,
) -> tuple[list[Failure], dict[str, tuple[int, ...]]]:
    """Validate the DP×CP group carried by existing DistOpt events."""

    iteration_id = int(iteration.iteration_id)
    failures = _validate_tp1_pp1_dp1_cp_identity(
        iteration,
        rank=rank,
    )
    selected = [
        event for event in iteration.events if event.name in _DISTOPT_ROUTE_NAMES
    ]
    phase_counts = Counter((event.name, event.ph) for event in selected)
    reduce_scatter_count = phase_counts[("dp-reduce-scatter", "B")]
    if reduce_scatter_count <= 0:
        failures.append(
            _failure(
                "trace.cp.distopt_count",
                "Qwen3 CP DistOpt has no gradient ReduceScatter",
                rank=rank,
                iteration=iteration_id,
            )
        )
    expected_counts = {
        "dp-reduce-scatter": reduce_scatter_count,
        "dp-grad-sync-complete": reduce_scatter_count,
        "dp-param-all-gather": phase_counts[("dp-param-all-gather", "B")],
        "dp-param-sync-complete": phase_counts[("dp-param-all-gather", "B")],
    }
    for name, expected in expected_counts.items():
        observed = (phase_counts[(name, "B")], phase_counts[(name, "E")])
        if observed != (expected, expected):
            failures.append(
                _failure(
                    "trace.cp.distopt_count",
                    f"event {name!r} has B/E={observed}, "
                    f"expected {expected}/{expected}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    expected_peers = [
        peer for peer in range(context_parallel_size) if peer != rank
    ]
    payloads: dict[str, list[int]] = defaultdict(list)
    for event in selected:
        if event.name not in {"dp-reduce-scatter", "dp-param-all-gather"}:
            continue
        if event.ph == "B":
            for field, expected in {
                "group_size": context_parallel_size,
                "group_role": "intra_optimizer_instance",
            }.items():
                if event.attrs.get(field, "<missing>") != expected:
                    failures.append(
                        _failure(
                            "trace.cp.distopt_group",
                            f"{event.name!r} has {field}="
                            f"{event.attrs.get(field, '<missing>')!r}, "
                            f"expected {expected!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
            data_bytes = event.attrs.get("data_bytes")
            if (
                not isinstance(data_bytes, int)
                or isinstance(data_bytes, bool)
                or data_bytes <= 0
            ):
                failures.append(
                    _failure(
                        "trace.cp.distopt_payload",
                        f"{event.name!r} has invalid data_bytes={data_bytes!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            else:
                payloads[event.name].append(data_bytes)
        elif event.ph == "E" and event.attrs.get("group") != expected_peers:
            failures.append(
                _failure(
                    "trace.cp.distopt_group",
                    f"{event.name!r} peer group is {event.attrs.get('group')!r}, "
                    f"expected {expected_peers!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    return failures, {name: tuple(values) for name, values in payloads.items()}


def _validate_qwen3_distopt_coexistence(
    trace_root: Path, *, context_parallel_size: int
) -> tuple[Failure, ...]:
    failures = list(
        gpt_probe_contract.validate_gpt_cp_dp1_eager_phases(
            trace_root,
            context_parallel_size=context_parallel_size,
            expected_layers=28,
        )
    )
    failures.extend(dp_probe_contract.validate_dp_distopt_overlap(trace_root))
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(range(context_parallel_size))
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != expected_ranks:
        failures.append(
            Failure(
                "trace.cp.ranks",
                f"Qwen3 CP{context_parallel_size} contract expects ranks "
                f"{list(expected_ranks)}, observed {list(observed_ranks)}",
                f"qwen3-cp{context_parallel_size}-distopt",
            )
        )

    signatures: dict[tuple[int, str], dict[int, tuple[int, ...]]] = defaultdict(dict)
    for rank in expected_ranks:
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(item.iteration_id for item in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.cp.iterations",
                    f"rank {rank} expects iterations [1, 2], "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        for iteration in iterations:
            iteration_id = int(iteration.iteration_id)
            iteration_failures, payloads = _validate_qwen3_distopt_iteration(
                iteration,
                rank=rank,
                context_parallel_size=context_parallel_size,
            )
            failures.extend(iteration_failures)
            for name, values in payloads.items():
                signatures[(iteration_id, name)][rank] = values

    for (iteration_id, name), rank_payloads in sorted(signatures.items()):
        if set(rank_payloads) == set(expected_ranks) and len(
            set(rank_payloads.values())
        ) != 1:
            failures.append(
                Failure(
                    "trace.cp.distopt_payload",
                    f"iteration {iteration_id} {name!r} payload sequences differ "
                    f"across CP ranks: {rank_payloads}",
                    f"iteration={iteration_id}",
                )
            )
    return tuple(failures)


def validate_qwen3_cp2_distopt_coexistence(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate Qwen3-0.6B with TP1/PP1/CP2/DP1 DistOpt overlap."""

    return _validate_qwen3_distopt_coexistence(
        trace_root, context_parallel_size=2
    )


def validate_qwen3_cp4_distopt_coexistence(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate Qwen3-0.6B with TP1/PP1/CP4/DP1 DistOpt overlap."""

    return _validate_qwen3_distopt_coexistence(
        trace_root, context_parallel_size=4
    )
