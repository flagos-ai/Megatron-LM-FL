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
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_spans(iteration, _COLLECTIVE_SPECS, rank=rank)
    for name in _COLLECTIVE_SPECS:
        if not spans.get(name):
            failures.append(
                _failure(
                    "trace.tp.collective_count",
                    f"event {name!r} has no complete span",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
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


def validate_tp2_gqa_collective_hierarchy(
    trace_root: Path,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            Failure(
                "trace.tp.ranks",
                f"TP2 contract expects ranks [0, 1], observed {sorted(by_rank)}",
                "tp2-local",
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
            failures.extend(_validate_collective_hierarchy(iteration, rank=rank))
    return tuple(failures)
