# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace contract for the controlled TP2/EP4 Flex+DeepEP profile."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners import tp_probe_contract
from tests.test_utils.runners.megalens_run_manifest import Failure

_RANKS = tuple(range(8))
_ITERATIONS = (1, 2)
_PRIMITIVES = ("ep-alltoall-dispatch", "ep-alltoall-combine")
_PHASES = ("B", "E", "B", "E")
_TOPOLOGY = {
    "comm_type": "ep-deepep",
    "dispatcher": "flex",
    # Flex reports the expert TP domain. This profile keeps expert TP at 1
    # while model TP is 2, so the DeepEP group contains the four EP ranks.
    "group_size": 4,
    "ep_size": 4,
    "tp_size": 1,
}
_ASYNC_EVENTS = frozenset(
    {
        "ep-alltoall-async-launch",
        "ep-alltoall-async-complete",
    }
)
_ASYNC_FIELDS = frozenset(
    {
        "operation_id",
        "request_id",
        "async_op",
        "completion_included",
        "completion_site",
        "terminal",
        "wait_role",
    }
)
_MOE_ROUTE_SEQUENCE = (
    ("moe-dispatch", "B"),
    ("ep-alltoall-dispatch", "B"),
    ("ep-alltoall-dispatch", "E"),
    ("moe-dispatch", "E"),
    ("moe-combine", "B"),
    ("ep-alltoall-combine", "B"),
    ("ep-alltoall-combine", "E"),
    ("moe-combine", "E"),
) * 2


def _failure(code: str, message: str, rank: int, iteration: int | None) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.moe_flex_deepep.{code}", message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("Flex+DeepEP contract requires one shard per global rank")
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _validate_primitive(
    iteration: Iteration,
    *,
    rank: int,
    iteration_id: int,
    name: str,
) -> list[Failure]:
    events = [event for event in iteration.events if event.name == name]
    failures: list[Failure] = []
    phases = tuple(event.ph for event in events)
    if phases != _PHASES:
        failures.append(
            _failure(
                "phases",
                f"{name} expected phases {_PHASES}, observed {phases}",
                rank,
                iteration_id,
            )
        )

    for occurrence, event in enumerate(events):
        unexpected_fields = sorted(_ASYNC_FIELDS.intersection(event.attrs))
        if unexpected_fields:
            failures.append(
                _failure(
                    "lifecycle",
                    f"{name} record={occurrence} exposes async fields {unexpected_fields}",
                    rank,
                    iteration_id,
                )
            )

    for occurrence, event in enumerate(event for event in events if event.ph == "E"):
        for field, expected in _TOPOLOGY.items():
            observed = event.attrs.get(field)
            if observed != expected:
                failures.append(
                    _failure(
                        "metadata",
                        f"{name} occurrence={occurrence} has {field}={observed!r}, "
                        f"expected {expected!r}",
                        rank,
                        iteration_id,
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
                    "data_bytes",
                    f"{name} occurrence={occurrence} has invalid data_bytes={data_bytes!r}",
                    rank,
                    iteration_id,
                )
            )
    return failures


def _validate_iteration(iteration: Iteration, rank: int) -> list[Failure]:
    if iteration.iteration_id is None:
        return [_failure("iteration", "scope is outside a numbered iteration", rank, None)]

    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    async_events: list[Event] = [
        event for event in iteration.events if event.name in _ASYNC_EVENTS
    ]
    if async_events:
        failures.append(
            _failure(
                "lifecycle",
                f"observed unsupported async lifecycle events "
                f"{tuple(event.name for event in async_events)}",
                rank,
                iteration_id,
            )
        )
    primitive_sequence = tuple(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in _PRIMITIVES
    )
    expected_sequence = (
        ("ep-alltoall-dispatch", "B"),
        ("ep-alltoall-dispatch", "E"),
        ("ep-alltoall-combine", "B"),
        ("ep-alltoall-combine", "E"),
    ) * 2
    if primitive_sequence != expected_sequence:
        failures.append(
            _failure(
                "order",
                f"expected fused primitive sequence {expected_sequence}, "
                f"observed {primitive_sequence}",
                rank,
                iteration_id,
            )
        )
    moe_route_sequence = tuple(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in {
            "moe-dispatch",
            "moe-combine",
            "ep-alltoall-dispatch",
            "ep-alltoall-combine",
        }
    )
    if moe_route_sequence != _MOE_ROUTE_SEQUENCE:
        failures.append(
            _failure(
                "scope_order",
                f"expected MoE route sequence {_MOE_ROUTE_SEQUENCE}, "
                f"observed {moe_route_sequence}",
                rank,
                iteration_id,
            )
        )
    for name in _PRIMITIVES:
        failures.extend(
            _validate_primitive(
                iteration,
                rank=rank,
                iteration_id=iteration_id,
                name=name,
            )
        )
    return failures


def validate_tp2_ep4_flex_deepep(trace_root: Path) -> tuple[Failure, ...]:
    """Validate fused DeepEP plus the separate model-TP domain."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.moe_flex_deepep.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "tp2-ep4-flex-deepep",
            )
        )
    for rank, iterations in sorted(by_rank.items()):
        observed_iterations = tuple(iteration.iteration_id for iteration in iterations)
        if observed_iterations != _ITERATIONS:
            failures.append(
                _failure(
                    "iterations",
                    f"expected iteration IDs {_ITERATIONS}, observed {observed_iterations}",
                    rank,
                    None,
                )
            )
        for iteration in iterations:
            failures.extend(_validate_iteration(iteration, rank))
    failures.extend(tp_probe_contract.validate_tp2_ep4_flex_tp_domain(trace_root))
    return tuple(failures)


__all__ = ["validate_tp2_ep4_flex_deepep"]
