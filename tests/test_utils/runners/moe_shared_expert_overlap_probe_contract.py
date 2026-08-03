# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace contract for the controlled EP2 shared-expert overlap profile."""

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
_LAYERS = (1, 2)
_STAGES = (
    "pre_forward_comm",
    "linear_fc1_forward_and_act",
    "linear_fc2_forward",
    "post_forward_comm",
    "get_output",
)
_SCOPE_COUNT = len(_LAYERS) * len(_STAGES)
_PHASES = ("B", "E") * _SCOPE_COUNT
_EP_NAMES = ("ep-alltoall-dispatch", "ep-alltoall-combine")
_EP_PHASES = ("B", "E") * len(_LAYERS)
_EXPECTED_LAYERS = tuple(layer for layer in _LAYERS for _stage in _STAGES)
_EXPECTED_STAGES = _STAGES * len(_LAYERS)


def _failure(code: str, message: str, rank: int, iteration: int | None) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.moe_shared_expert_overlap.{code}", message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError(
                "shared-expert overlap contract requires one shard per global rank"
            )
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _paired_intervals(events: Sequence[Event]) -> tuple[tuple[int, int, Event], ...]:
    return tuple(
        (begin.rel_ts, end.rel_ts, end)
        for begin, end in zip(events[::2], events[1::2])
        if begin.ph == "B" and end.ph == "E"
    )


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
                f"expected shared-expert overlap phases {_PHASES}, observed {phases}",
                rank,
                iteration_id,
            )
        )

    end_events: list[Event] = [event for event in events if event.ph == "E"]
    layers = tuple(event.attrs.get("layer") for event in end_events)
    if layers != _EXPECTED_LAYERS:
        failures.append(
            _failure(
                "layers",
                f"expected shared-expert overlap layers {_EXPECTED_LAYERS}, "
                f"observed {layers}",
                rank,
                iteration_id,
            )
        )

    stages = tuple(event.attrs.get("stage") for event in end_events)
    if stages != _EXPECTED_STAGES:
        failures.append(
            _failure(
                "stages",
                f"expected shared-expert overlap stages {_EXPECTED_STAGES}, "
                f"observed {stages}",
                rank,
                iteration_id,
            )
        )

    for event in end_events:
        if event.attrs.get("ep_size") != 2:
            failures.append(
                _failure(
                    "ep_size",
                    f"layer={event.attrs.get('layer')!r} "
                    f"stage={event.attrs.get('stage')!r} has "
                    f"ep_size={event.attrs.get('ep_size')!r}, expected 2",
                    rank,
                    iteration_id,
                )
            )

    shared_intervals = _paired_intervals(events) if phases == _PHASES else ()
    for start, end, event in shared_intervals:
        if end <= start:
            failures.append(
                _failure(
                    "interval",
                    f"layer={event.attrs.get('layer')!r} "
                    f"stage={event.attrs.get('stage')!r} has interval [{start}, {end}]",
                    rank,
                    iteration_id,
                )
            )

    for name in _EP_NAMES:
        ep_events = [event for event in iteration.events if event.name == name]
        ep_phases = tuple(event.ph for event in ep_events)
        if ep_phases != _EP_PHASES:
            failures.append(
                _failure(
                    "ep_phases",
                    f"{name} expected phases {_EP_PHASES}, observed {ep_phases}",
                    rank,
                    iteration_id,
                )
            )
            continue

        for start, end, _event in _paired_intervals(ep_events):
            if end <= start:
                failures.append(
                    _failure(
                        "interval",
                        f"{name} has interval [{start}, {end}]",
                        rank,
                        iteration_id,
                    )
                )
    return failures


def validate_ep2_shared_expert_overlap(trace_root: Path) -> tuple[Failure, ...]:
    """Validate all five shared-stream stages for two layers, ranks and iterations."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.moe_shared_expert_overlap.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "ep2-alltoall-shared-expert-overlap",
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


__all__ = ["validate_ep2_shared_expert_overlap"]
