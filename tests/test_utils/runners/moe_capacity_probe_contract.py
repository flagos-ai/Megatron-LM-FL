# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace contract for the controlled EP2 capacity/drop profile."""

from __future__ import annotations

import math
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
_PAIR_COUNT = 2
_CAPACITY_FACTOR = 0.5
_ROUTER_PROFILE_FIELDS = {
    "ep_size": 2,
    "num_experts": 4,
    "num_local_experts": 2,
    "router_topk": 2,
    "num_tokens": 128,
}
_PAIR_FIELDS = (
    "layer",
    "ep_size",
    "num_experts",
    "num_local_experts",
    "router_topk",
)
_HANDOFF_FIELDS = (
    "dropped_tokens",
    "drop_rate",
    "expert_cv",
    "top1_expert_share",
    "aux_loss",
    "z_loss",
)


def _failure(code: str, message: str, rank: int, iteration: int | None) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.moe_capacity.{code}", message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("capacity/drop contract requires one shard per global rank")
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _phase_events(iteration: Iteration, name: str) -> list[tuple[int, Event]]:
    return [
        (position, event)
        for position, event in enumerate(iteration.events)
        if event.name == name and event.ph == "E"
    ]


def _validate_pair(
    router: Event,
    dispatch: Event,
    *,
    rank: int,
    iteration: int,
    occurrence: int,
) -> list[Failure]:
    failures: list[Failure] = []
    evidence = f"occurrence={occurrence}"
    for field in _PAIR_FIELDS:
        router_value = router.attrs.get(field)
        dispatch_value = dispatch.attrs.get(field)
        if router_value != dispatch_value:
            failures.append(
                _failure(
                    "pair",
                    f"{evidence} field {field!r} differs: "
                    f"{router_value!r} != {dispatch_value!r}",
                    rank,
                    iteration,
                )
            )

    for field, expected in _ROUTER_PROFILE_FIELDS.items():
        observed = router.attrs.get(field)
        if observed != expected:
            failures.append(
                _failure(
                    "profile",
                    f"{evidence} Router field {field!r}={observed!r}, expected {expected!r}",
                    rank,
                    iteration,
                )
            )

    for field in _HANDOFF_FIELDS:
        router_value = router.attrs.get(field)
        dispatch_value = dispatch.attrs.get(field)
        if router_value != dispatch_value:
            failures.append(
                _failure(
                    "handoff",
                    f"{evidence} field {field!r} differs: "
                    f"{router_value!r} != {dispatch_value!r}",
                    rank,
                    iteration,
                )
            )

    num_tokens = router.attrs.get("num_tokens")
    router_topk = router.attrs.get("router_topk")
    routed_tokens = router.attrs.get("routed_tokens")
    dropped_tokens = router.attrs.get("dropped_tokens")
    drop_rate = router.attrs.get("drop_rate")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (num_tokens, router_topk, routed_tokens, dropped_tokens)
    ):
        failures.append(
            _failure(
                "type",
                f"{evidence} assignment counts must be integers",
                rank,
                iteration,
            )
        )
        return failures

    assignments_before_drop = num_tokens * router_topk
    expected_dropped = max(0, assignments_before_drop - routed_tokens)
    expected_rate = expected_dropped / assignments_before_drop if assignments_before_drop else 0.0
    max_routed = math.ceil(
        assignments_before_drop / _ROUTER_PROFILE_FIELDS["num_experts"] * _CAPACITY_FACTOR
    ) * _ROUTER_PROFILE_FIELDS["num_experts"]
    if routed_tokens > max_routed:
        failures.append(
            _failure(
                "capacity_bound",
                f"{evidence} routed_tokens={routed_tokens}, capacity upper bound is {max_routed}",
                rank,
                iteration,
            )
        )
    if dropped_tokens != expected_dropped or not 0 < dropped_tokens < assignments_before_drop:
        failures.append(
            _failure(
                "dropped_tokens",
                f"{evidence} dropped_tokens={dropped_tokens!r}, expected positive "
                f"{expected_dropped}",
                rank,
                iteration,
            )
        )
    if not isinstance(drop_rate, (int, float)) or isinstance(drop_rate, bool):
        failures.append(
            _failure(
                "drop_rate",
                f"{evidence} drop_rate must be numeric, observed {drop_rate!r}",
                rank,
                iteration,
            )
        )
    elif not math.isfinite(float(drop_rate)) or not math.isclose(
        float(drop_rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append(
            _failure(
                "drop_rate",
                f"{evidence} drop_rate={drop_rate!r}, expected {expected_rate}",
                rank,
                iteration,
            )
        )
    if dispatch.attrs.get("dispatcher") != "alltoall":
        failures.append(
            _failure(
                "dispatcher",
                f"{evidence} dispatcher={dispatch.attrs.get('dispatcher')!r}, "
                "expected 'alltoall'",
                rank,
                iteration,
            )
        )
    if dispatch.attrs.get("capacity_factor") != _CAPACITY_FACTOR:
        failures.append(
            _failure(
                "capacity_factor",
                f"{evidence} capacity_factor={dispatch.attrs.get('capacity_factor')!r}, "
                f"expected {_CAPACITY_FACTOR}",
                rank,
                iteration,
            )
        )
    if dispatch.attrs.get("num_tokens") != routed_tokens:
        failures.append(
            _failure(
                "dispatch_tokens",
                f"{evidence} dispatch num_tokens={dispatch.attrs.get('num_tokens')!r}, "
                f"expected routed_tokens={routed_tokens}",
                rank,
                iteration,
            )
        )
    return failures


def _validate_iteration(iteration: Iteration, rank: int) -> list[Failure]:
    if iteration.iteration_id is None:
        return [_failure("iteration", "scope is outside a numbered iteration", rank, None)]
    iteration_id = int(iteration.iteration_id)
    routers = _phase_events(iteration, "moe-router")
    dispatches = _phase_events(iteration, "moe-dispatch")
    failures: list[Failure] = []
    if len(routers) != _PAIR_COUNT or len(dispatches) != _PAIR_COUNT:
        failures.append(
            _failure(
                "count",
                f"expected {_PAIR_COUNT} Router/Dispatch pairs, observed "
                f"{len(routers)}/{len(dispatches)}",
                rank,
                iteration_id,
            )
        )
    observed_layers = tuple(router.attrs.get("layer") for _, router in routers)
    if observed_layers != (1, 2):
        failures.append(
            _failure(
                "layers",
                f"expected Router layers (1, 2), observed {observed_layers}",
                rank,
                iteration_id,
            )
        )
    for occurrence, ((router_position, router), (dispatch_position, dispatch)) in enumerate(
        zip(routers, dispatches, strict=False)
    ):
        next_router_position = (
            routers[occurrence + 1][0] if occurrence + 1 < len(routers) else None
        )
        if router_position >= dispatch_position or (
            next_router_position is not None and dispatch_position >= next_router_position
        ):
            failures.append(
                _failure(
                    "order",
                    f"occurrence={occurrence} does not form Router then Dispatch",
                    rank,
                    iteration_id,
                )
            )
        failures.extend(
            _validate_pair(
                router,
                dispatch,
                rank=rank,
                iteration=iteration_id,
                occurrence=occurrence,
            )
        )
    return failures


def validate_ep2_capacity_drop(trace_root: Path) -> tuple[Failure, ...]:
    """Validate nonzero drop fields and the matching Router-to-Dispatch handoff."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.moe_capacity.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "ep2-alltoall-capacity-drop",
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


__all__ = ["validate_ep2_capacity_drop"]
