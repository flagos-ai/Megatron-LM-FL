# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Combined 1F1B trace contract for the controlled PP2/VPP2/EP2 profile."""

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

_EVENT = "combined-forward-backward-step"
_RANKS = (0, 1, 2, 3)
_ITERATIONS = (1, 2)
_MODES = frozenset(("forward", "combined", "backward"))
_OPERATION_KEYS = frozenset((microbatch, vp) for microbatch in range(4) for vp in range(2))
_OPERATION_IDS = frozenset(f"pp:microbatch={mb}:vp={vp}" for mb, vp in _OPERATION_KEYS)
_BEGIN_FIELDS = frozenset(
    (
        "operation_id",
        "forward_operation_id",
        "backward_operation_id",
        "forward_microbatch",
        "backward_microbatch",
        "forward_vp_stage",
        "backward_vp_stage",
        "execution_mode",
        "overlap_active",
        "schedule",
        "timing_phase",
    )
)


def _failure(code: str, message: str, rank: int, iteration: int | None) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.combined_1f1b.{code}", message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("combined 1F1B contract requires one shard per global rank")
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _paired_begins(iteration: Iteration, rank: int) -> tuple[list[Event], list[Failure]]:
    iteration_id = iteration.iteration_id
    # scoped_forward writes context on B and an attribute-free E, so nesting order pairs spans.
    active: Event | None = None
    begins: list[Event] = []
    failures: list[Failure] = []
    for event in iteration.events:
        if event.name != _EVENT:
            continue
        if event.ph == "B":
            if active is not None:
                failures.append(
                    _failure(
                        "pairing",
                        f"{_EVENT!r} begins while another span is active",
                        rank,
                        iteration_id,
                    )
                )
            active = event
        elif event.ph == "E":
            if active is None:
                failures.append(
                    _failure(
                        "pairing",
                        f"{_EVENT!r} ends without a matching begin",
                        rank,
                        iteration_id,
                    )
                )
            else:
                begins.append(active)
                active = None
        else:
            failures.append(
                _failure(
                    "phase",
                    f"{_EVENT!r} uses unsupported phase {event.ph!r}",
                    rank,
                    iteration_id,
                )
            )
    if active is not None:
        failures.append(
            _failure("pairing", f"{_EVENT!r} begin has no matching end", rank, iteration_id)
        )
    if not begins:
        failures.append(_failure("missing", f"no paired {_EVENT!r} span", rank, iteration_id))
    return begins, failures


def _validate_iteration(iteration: Iteration, rank: int) -> list[Failure]:
    if iteration.iteration_id is None:
        return [_failure("iteration", "scope is outside a numbered iteration", rank, None)]
    iteration_id = int(iteration.iteration_id)
    begins, failures = _paired_begins(iteration, rank)
    modes: set[str] = set()
    operations: dict[str, list[tuple[tuple[int, int], str]]] = {
        "forward": [],
        "backward": [],
    }
    outer_ids: list[str] = []

    for begin in begins:
        attrs = begin.attrs
        missing = sorted(_BEGIN_FIELDS - attrs.keys())
        if missing:
            failures.append(_failure("field", f"begin lacks fields {missing}", rank, iteration_id))

        mode = attrs.get("execution_mode")
        if mode not in _MODES:
            failures.append(
                _failure(
                    "mode",
                    f"invalid execution_mode={mode!r}",
                    rank,
                    iteration_id,
                )
            )
            continue
        assert isinstance(mode, str)
        modes.add(mode)
        expected = {
            "overlap_active": mode == "combined",
            "schedule": "combined-1f1b",
            "timing_phase": "framework_phase",
        }
        for field, value in expected.items():
            observed = attrs.get(field, "<missing>")
            matches = observed is value if field == "overlap_active" else observed == value
            if not matches:
                failures.append(
                    _failure(
                        "field",
                        f"mode {mode!r} has {field}={observed!r}, expected {value!r}",
                        rank,
                        iteration_id,
                    )
                )

        active_ids: dict[str, str | None] = {}
        for direction in ("forward", "backward"):
            is_active = direction == mode or mode == "combined"
            microbatch = attrs.get(f"{direction}_microbatch")
            vp_stage = attrs.get(f"{direction}_vp_stage")
            operation_id = attrs.get(f"{direction}_operation_id")
            active_ids[direction] = operation_id if isinstance(operation_id, str) else None
            if not is_active:
                if any(value is not None for value in (microbatch, vp_stage, operation_id)):
                    failures.append(
                        _failure(
                            "inactive_direction",
                            f"inactive {direction} fields must all be None",
                            rank,
                            iteration_id,
                        )
                    )
                continue
            if (
                not isinstance(microbatch, int)
                or isinstance(microbatch, bool)
                or not isinstance(vp_stage, int)
                or isinstance(vp_stage, bool)
            ):
                failures.append(
                    _failure(
                        "identity",
                        f"active {direction} has invalid identity {(microbatch, vp_stage)!r}",
                        rank,
                        iteration_id,
                    )
                )
                continue
            expected_id = f"pp:microbatch={microbatch}:vp={vp_stage}"
            if operation_id != expected_id:
                failures.append(
                    _failure(
                        "identity",
                        f"active {direction} has operation_id={operation_id!r}, "
                        f"expected {expected_id!r}",
                        rank,
                        iteration_id,
                    )
                )
                continue
            operations[direction].append(((microbatch, vp_stage), expected_id))

        expected_outer_id = (
            f"pp-combined:forward={active_ids['forward']}:backward={active_ids['backward']}"
        )
        outer_id = attrs.get("operation_id")
        if outer_id != expected_outer_id:
            failures.append(
                _failure(
                    "identity",
                    f"outer operation_id={outer_id!r}, expected {expected_outer_id!r}",
                    rank,
                    iteration_id,
                )
            )
        elif isinstance(outer_id, str):
            outer_ids.append(outer_id)

    missing_modes = sorted(_MODES - modes)
    if missing_modes:
        failures.append(
            _failure(
                "mode_coverage",
                f"missing modes {missing_modes}",
                rank,
                iteration_id,
            )
        )

    for direction, observed in operations.items():
        keys = [key for key, _ in observed]
        operation_ids = [operation_id for _, operation_id in observed]
        missing = sorted(_OPERATION_KEYS - set(keys))
        unexpected = sorted(set(keys) - _OPERATION_KEYS)
        if missing or unexpected:
            failures.append(
                _failure(
                    "coverage",
                    f"{direction} coverage differs: missing={missing}, unexpected={unexpected}",
                    rank,
                    iteration_id,
                )
            )
        if len(keys) != len(set(keys)) or len(operation_ids) != len(set(operation_ids)):
            failures.append(
                _failure(
                    "duplicate",
                    f"{direction} reuses an identity",
                    rank,
                    iteration_id,
                )
            )
        if set(operation_ids) != _OPERATION_IDS:
            failures.append(
                _failure(
                    "operation_id_coverage",
                    f"{direction} operation IDs differ",
                    rank,
                    iteration_id,
                )
            )
    if len(outer_ids) != len(set(outer_ids)):
        failures.append(_failure("duplicate", "outer operation ID is reused", rank, iteration_id))
    return failures


def validate_ep2_fine_grained_combined(trace_root: Path) -> tuple[Failure, ...]:
    """Validate fused scopes by structure and identity coverage, not mode counts."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.combined_1f1b.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "ep2-fine-grained",
            )
        )
    for rank, iterations in by_rank.items():
        ids = tuple(iteration.iteration_id for iteration in iterations)
        if ids != _ITERATIONS:
            failures.append(
                _failure(
                    "iterations",
                    f"expected iteration IDs {_ITERATIONS}, observed {ids}",
                    rank,
                    None,
                )
            )
        for iteration in iterations:
            failures.extend(_validate_iteration(iteration, rank))
    return tuple(failures)


__all__ = ["validate_ep2_fine_grained_combined"]
