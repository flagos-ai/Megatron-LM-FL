# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace contract for the controlled non-overlap shared-expert profile."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_RANKS = (0, 1)
_ITERATIONS = (1, 2)
_PHASES = ("B", "E", "B", "E")


def _failure(code: str, message: str, rank: int, iteration: int | None) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.moe_shared_expert.{code}", message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("shared-expert contract requires one shard per global rank")
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _validate_iteration(iteration: Iteration, rank: int) -> list[Failure]:
    if iteration.iteration_id is None:
        return [_failure("iteration", "scope is outside a numbered iteration", rank, None)]

    iteration_id = int(iteration.iteration_id)
    events = [event for event in iteration.events if event.name == "moe-shared-expert"]
    failures: list[Failure] = []
    phases = tuple(event.ph for event in events)
    if phases != _PHASES:
        failures.append(
            _failure(
                "phases",
                f"expected shared-expert phases {_PHASES}, observed {phases}",
                rank,
                iteration_id,
            )
        )

    end_events: list[Event] = [event for event in events if event.ph == "E"]
    layers = tuple(event.attrs.get("layer") for event in end_events)
    if layers != (1, 2):
        failures.append(
            _failure(
                "layers",
                f"expected shared-expert layers (1, 2), observed {layers}",
                rank,
                iteration_id,
            )
        )
    for event in end_events:
        if event.attrs.get("ep_size") != 2:
            failures.append(
                _failure(
                    "ep_size",
                    f"layer={event.attrs.get('layer')!r} has "
                    f"ep_size={event.attrs.get('ep_size')!r}, expected 2",
                    rank,
                    iteration_id,
                )
            )
    return failures


def validate_ep2_shared_expert(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one completed non-overlap shared-expert call per layer."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.moe_shared_expert.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "ep2-alltoall-shared-expert",
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
    return tuple(failures)


__all__ = ["validate_ep2_shared_expert"]
