# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace contracts for the controlled EP2 MoE recompute and FP8 profiles."""

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
_PHASE_NAMES = ("moe-router", "moe-dispatch", "moe-experts", "moe-combine")
_ROUTER_DISPATCH_FIELDS = (
    "dropped_tokens",
    "drop_rate",
    "expert_cv",
    "top1_expert_share",
    "aux_loss",
    "z_loss",
)
_PROFILE_FIELDS = {
    "ep_size": 2,
    "num_experts": 4,
    "num_local_experts": 2,
}
_COLLECTIVE_BY_PHASE = {
    "moe-dispatch": "ep-alltoall-dispatch",
    "moe-combine": "ep-alltoall-combine",
}


def _failure(
    code: str,
    message: str,
    rank: int,
    iteration: int | None,
    profile: str,
) -> Failure:
    evidence = f"profile={profile} rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.moe_recompute_fp8.{code}", message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("recompute/FP8 contract requires one shard per global rank")
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _phase_records(iteration: Iteration) -> list[tuple[int, Event]]:
    return [
        (position, event)
        for position, event in enumerate(iteration.events)
        if event.name in _PHASE_NAMES
    ]


def _expected_name_phases(layers: Sequence[int]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, phase)
        for _layer in layers
        for name in _PHASE_NAMES
        for phase in ("B", "E")
    )


def _validate_collective_window(
    iteration: Iteration,
    *,
    start: int,
    end: int,
    collective: str,
    layer: int,
    rank: int,
    iteration_id: int,
    profile: str,
) -> list[Failure]:
    phases = tuple(
        event.ph
        for position, event in enumerate(iteration.events)
        if start < position < end and event.name == collective
    )
    if phases == ("B", "E"):
        return []
    return [
        _failure(
            "collective_phases",
            f"layer={layer} {collective} expected phases ('B', 'E'), observed {phases}",
            rank,
            iteration_id,
            profile,
        )
    ]


def _validate_call(
    iteration: Iteration,
    records: Sequence[tuple[int, Event]],
    *,
    layer: int,
    occurrence: int,
    rank: int,
    iteration_id: int,
    profile: str,
    recompute: bool,
) -> list[Failure]:
    failures: list[Failure] = []
    end_events = {event.name: event for _position, event in records if event.ph == "E"}
    if tuple(end_events) != _PHASE_NAMES:
        return failures

    for name, event in end_events.items():
        if event.attrs.get("layer") != layer:
            failures.append(
                _failure(
                    "layers",
                    f"occurrence={occurrence} {name} has layer="
                    f"{event.attrs.get('layer')!r}, expected {layer}",
                    rank,
                    iteration_id,
                    profile,
                )
            )
        for field, expected in _PROFILE_FIELDS.items():
            if event.attrs.get(field) != expected:
                failures.append(
                    _failure(
                        "profile",
                        f"occurrence={occurrence} {name} field {field!r}="
                        f"{event.attrs.get(field)!r}, expected {expected!r}",
                        rank,
                        iteration_id,
                        profile,
                    )
                )

    router = end_events["moe-router"]
    dispatch = end_events["moe-dispatch"]
    for field in _ROUTER_DISPATCH_FIELDS:
        router_value = router.attrs.get(field)
        dispatch_value = dispatch.attrs.get(field)
        if router_value != dispatch_value:
            failures.append(
                _failure(
                    "handoff",
                    f"occurrence={occurrence} field {field!r} differs: "
                    f"{router_value!r} != {dispatch_value!r}",
                    rank,
                    iteration_id,
                    profile,
                )
            )

    aux_loss = router.attrs.get("aux_loss")
    original_forward = recompute and occurrence < 2
    if original_forward:
        if aux_loss is not None:
            failures.append(
                _failure(
                    "aux_loss_mode",
                    f"occurrence={occurrence} no-grad forward has "
                    f"aux_loss={aux_loss!r}, expected None",
                    rank,
                    iteration_id,
                    profile,
                )
            )
    elif (
        isinstance(aux_loss, bool)
        or not isinstance(aux_loss, (int, float))
        or not math.isfinite(float(aux_loss))
    ):
        failures.append(
            _failure(
                "aux_loss_mode",
                f"occurrence={occurrence} grad-enabled call has "
                f"aux_loss={aux_loss!r}, expected a finite number",
                rank,
                iteration_id,
                profile,
            )
        )
    if router.attrs.get("router_topk") != 2 or dispatch.attrs.get("router_topk") != 2:
        failures.append(
            _failure(
                "router_topk",
                f"occurrence={occurrence} Router/Dispatch router_topk="
                f"{router.attrs.get('router_topk')!r}/"
                f"{dispatch.attrs.get('router_topk')!r}, expected 2/2",
                rank,
                iteration_id,
                profile,
            )
        )
    if dispatch.attrs.get("dispatcher") != "alltoall":
        failures.append(
            _failure(
                "dispatcher",
                f"occurrence={occurrence} dispatcher="
                f"{dispatch.attrs.get('dispatcher')!r}, expected 'alltoall'",
                rank,
                iteration_id,
                profile,
            )
        )

    positions = {
        (event.name, event.ph): position for position, event in records
    }
    for phase_name, collective in _COLLECTIVE_BY_PHASE.items():
        failures.extend(
            _validate_collective_window(
                iteration,
                start=positions[(phase_name, "B")],
                end=positions[(phase_name, "E")],
                collective=collective,
                layer=layer,
                rank=rank,
                iteration_id=iteration_id,
                profile=profile,
            )
        )
    return failures


def _validate_iteration(
    iteration: Iteration,
    rank: int,
    *,
    expected_layers: Sequence[int],
    profile: str,
    recompute: bool,
) -> list[Failure]:
    if iteration.iteration_id is None:
        return [
            _failure(
                "iteration",
                "MoE scopes are outside a numbered iteration",
                rank,
                None,
                profile,
            )
        ]

    iteration_id = int(iteration.iteration_id)
    records = _phase_records(iteration)
    observed_name_phases = tuple((event.name, event.ph) for _position, event in records)
    expected_name_phases = _expected_name_phases(expected_layers)
    failures: list[Failure] = []
    if observed_name_phases != expected_name_phases:
        failures.append(
            _failure(
                "sequence",
                f"expected MoE name/phase sequence {expected_name_phases}, "
                f"observed {observed_name_phases}",
                rank,
                iteration_id,
                profile,
            )
        )
        return failures

    records_per_call = len(_PHASE_NAMES) * 2
    for occurrence, layer in enumerate(expected_layers):
        start = occurrence * records_per_call
        failures.extend(
            _validate_call(
                iteration,
                records[start : start + records_per_call],
                layer=layer,
                occurrence=occurrence,
                rank=rank,
                iteration_id=iteration_id,
                profile=profile,
                recompute=recompute,
            )
        )
    return failures


def _validate_profile(
    trace_root: Path,
    *,
    expected_layers: Sequence[int],
    profile: str,
    recompute: bool,
) -> tuple[Failure, ...]:
    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.moe_recompute_fp8.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                profile,
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
                    profile,
                )
            )
        for iteration in iterations:
            failures.extend(
                _validate_iteration(
                    iteration,
                    rank,
                    expected_layers=expected_layers,
                    profile=profile,
                    recompute=recompute,
                )
            )
    return tuple(failures)


def validate_ep2_recompute(trace_root: Path) -> tuple[Failure, ...]:
    """Validate forward and reverse-order backward recomputation of both MoE layers."""

    return _validate_profile(
        trace_root,
        expected_layers=(1, 2, 2, 1),
        profile="ep2-recompute",
        recompute=True,
    )


def validate_ep2_fp8(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the source MoE phase contract while FP8 delayed scaling is enabled."""

    return _validate_profile(
        trace_root,
        expected_layers=(1, 2),
        profile="ep2-fp8",
        recompute=False,
    )


def validate_ep2_fp8_recompute(trace_root: Path) -> tuple[Failure, ...]:
    """Validate MoE recomputation while FP8 delayed scaling is enabled."""

    return _validate_profile(
        trace_root,
        expected_layers=(1, 2, 2, 1),
        profile="ep2-fp8-recompute",
        recompute=True,
    )


__all__ = [
    "validate_ep2_fp8",
    "validate_ep2_fp8_recompute",
    "validate_ep2_recompute",
]
