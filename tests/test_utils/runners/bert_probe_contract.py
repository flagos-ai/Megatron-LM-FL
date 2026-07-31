# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact runtime contract for the source-compatible BERT encoder probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from megatron.megalens.trace_aggregate import Iteration

_METADATA_FIELDS = frozenset(
    ("iteration", "dev", "g_rk", "dp_rk", "pp_rk", "tp_rk")
)
_RAW_FIELDS = frozenset(
    ("name", "ph", "rel_ts", "dev", "g_rk", "dp_rk", "pp_rk", "tp_rk")
)


def validate_bert_encoder_trace_contract(
    parsed_contents: Sequence[Sequence[Iteration]],
    *,
    expected_ranks: Sequence[int],
    expected_iterations: Sequence[int],
    raw_rows_by_rank: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Validate exact counts, fields, coordinates, and direct parent scopes."""

    phase_counts: Counter[tuple[int, int, str]] = Counter()
    parent_counts: Counter[tuple[int, int, str]] = Counter()
    business_fields: dict[str, dict[str, dict[str, list[str]]]] = {}
    issues: list[str] = []

    if tuple(expected_ranks) != (0,):
        issues.append(f"bert-encoder expects rank [0], got {list(expected_ranks)}")
    if tuple(expected_iterations) != (1, 2):
        issues.append(
            f"bert-encoder expects iterations [1, 2], got {list(expected_iterations)}"
        )

    for content in parsed_contents:
        for iteration in content:
            rank = iteration.ranks[0].global_rank
            iteration_id = iteration.iteration_id
            if rank is None or iteration_id is None:
                issues.append("BERT encoder trace lacks rank or iteration identity")
                continue

            open_by_name: dict[str, list[Any]] = defaultdict(list)
            spans: list[tuple[Any, Any]] = []
            for event in iteration.events:
                if event.ph == "B":
                    open_by_name[event.name].append(event)
                elif event.ph == "E" and open_by_name[event.name]:
                    spans.append((open_by_name[event.name].pop(), event))

                if event.name != "encoder":
                    continue
                if event.ph not in {"B", "E"}:
                    issues.append(
                        f"rank {rank} iteration {iteration_id} encoder "
                        f"has phase {event.ph!r}"
                    )
                    continue
                phase_counts[(rank, iteration_id, event.ph)] += 1
                fields = sorted(set(event.attrs) - _METADATA_FIELDS)
                business_fields.setdefault(str(rank), {}).setdefault(
                    str(iteration_id), {}
                )[event.ph] = fields
                if fields:
                    issues.append(
                        f"rank {rank} iteration {iteration_id} encoder "
                        f"{event.ph} has business fields {fields}"
                    )

            encoder_spans = [
                (begin, end) for begin, end in spans if begin.name == "encoder"
            ]
            if len(encoder_spans) != 1:
                issues.append(
                    f"rank {rank} iteration {iteration_id} has "
                    f"{len(encoder_spans)} complete encoder spans, expected 1"
                )
            for begin, end in encoder_spans:
                if begin.rel_ts > end.rel_ts:
                    issues.append(
                        f"rank {rank} iteration {iteration_id} encoder end "
                        "precedes its begin"
                    )
                enclosing = [
                    span
                    for span in spans
                    if span[0] is not begin
                    and span[0].rel_ts <= begin.rel_ts
                    and end.rel_ts <= span[1].rel_ts
                ]
                parent = (
                    max(enclosing, key=lambda span: (span[0].rel_ts, -span[1].rel_ts))[0]
                    if enclosing
                    else None
                )
                parent_name = parent.name if parent is not None else "<none>"
                parent_counts[(rank, iteration_id, parent_name)] += 1
                if parent_name != "forward-step":
                    issues.append(
                        f"rank {rank} iteration {iteration_id} encoder has "
                        f"direct parent {parent_name!r}, expected 'forward-step'"
                    )

    for rank in expected_ranks:
        for iteration_id in expected_iterations:
            observed = tuple(
                phase_counts[(rank, iteration_id, phase)] for phase in ("B", "E")
            )
            if observed != (1, 1):
                issues.append(
                    f"rank {rank} iteration {iteration_id} encoder has "
                    f"B/E={observed[0]}/{observed[1]}, expected 1/1"
                )

    raw_phase_counts: Counter[tuple[int, str]] = Counter()
    raw_field_sets: dict[str, list[list[str]]] = {}
    for rank, rows in raw_rows_by_rank.items():
        for row in rows:
            phase = str(row.get("ph"))
            raw_phase_counts[(rank, phase)] += 1
            if phase not in {"B", "E"}:
                issues.append(
                    f"rank {rank} raw encoder has unsupported phase {phase!r}"
                )
            fields = frozenset(row)
            raw_field_sets.setdefault(str(rank), []).append(sorted(fields))
            if fields != _RAW_FIELDS:
                issues.append(
                    f"rank {rank} raw encoder {phase} fields are "
                    f"{sorted(fields)}, expected {sorted(_RAW_FIELDS)}"
                )
            expected_coordinates = {
                "dev": 0,
                "g_rk": rank,
                "dp_rk": 0,
                "pp_rk": 0,
                "tp_rk": 0,
            }
            mismatched = {
                field: row.get(field)
                for field, expected in expected_coordinates.items()
                if type(row.get(field)) is not int or row.get(field) != expected
            }
            if mismatched:
                issues.append(
                    f"rank {rank} raw encoder coordinates differ: {mismatched}"
                )

    for rank in expected_ranks:
        observed = {
            phase: raw_phase_counts[(rank, phase)] for phase in ("B", "E")
        }
        if observed != {"B": 2, "E": 2}:
            issues.append(
                f"rank {rank} raw encoder phases are {observed}, "
                "expected {'B': 2, 'E': 2}"
            )

    observations: Mapping[str, Any] = {
        "profile": "bert-encoder",
        "event_phase_counts": {
            str(rank): {
                str(iteration_id): {
                    phase: phase_counts[(rank, iteration_id, phase)]
                    for phase in ("B", "E")
                }
                for iteration_id in expected_iterations
            }
            for rank in expected_ranks
        },
        "raw_event_phase_counts": {
            str(rank): {
                phase: raw_phase_counts[(rank, phase)] for phase in ("B", "E")
            }
            for rank in expected_ranks
        },
        "direct_parent_counts": {
            str(rank): {
                str(iteration_id): parent_counts[
                    (rank, iteration_id, "forward-step")
                ]
                for iteration_id in expected_iterations
            }
            for rank in expected_ranks
        },
        "business_fields": business_fields,
        "raw_field_sets": raw_field_sets,
    }
    return observations, tuple(issues)
