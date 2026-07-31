# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime contracts for controlled heterogeneous MiMo training."""

from __future__ import annotations

import json
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

_BRIDGE_EVENTS = frozenset(
    (
        "bridge-p2p-launch",
        "bridge-grid-broadcast",
        "bridge-send-forward",
        "bridge-recv-forward",
        "bridge-send-backward",
        "bridge-recv-backward",
    )
)
_TWO_RANK_ROLES = {0: "encoder", 1: "llm"}
_EIGHT_RANK_ROLES = {
    0: "encoder",
    1: "encoder",
    2: "encoder",
    3: "encoder",
    4: "llm",
    5: "llm",
    6: "llm",
    7: "llm",
}


def _failure(code: str, message: str, evidence: str) -> Failure:
    return Failure(code, message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    by_rank: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None:
            raise ValueError(f"trace shard {rank} has no global rank")
        if rank.global_rank in by_rank:
            raise ValueError(f"duplicate trace shard for rank {rank.global_rank}")
        by_rank[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return by_rank


def _event_span(events: Sequence[Event], name: str) -> tuple[Event, Event] | None:
    matching = [event for event in events if event.name == name]
    if len(matching) != 2 or tuple(event.ph for event in matching) != ("B", "E"):
        return None
    return matching[0], matching[1]


def _validate_mimo_training_trace(
    trace_root: Path, expected_ranks: Sequence[int], *, required_bridge_ranks: frozenset[int]
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    observed_bridge_ranks: set[int] = set()
    for rank in expected_ranks:
        iterations = by_rank.get(rank, ())
        evidence = f"rank={rank}"
        if len(iterations) != 1:
            failures.append(
                _failure(
                    "trace.mimo.iteration_count",
                    f"rank {rank} has {len(iterations)} iterations, expected 1",
                    evidence,
                )
            )
            continue
        events = iterations[0].events
        optimizer = _event_span(events, "optimizer-step")
        if optimizer is None:
            failures.append(
                _failure(
                    "trace.mimo.optimizer_step",
                    "optimizer-step must contain exactly one B/E pair",
                    evidence,
                )
            )
            continue
        bridge_events = [event for event in events if event.name in _BRIDGE_EVENTS]
        bridge_ends = [event.rel_ts for event in bridge_events if event.ph == "E"]
        if bridge_ends:
            observed_bridge_ranks.add(rank)
        if bridge_events and rank not in required_bridge_ranks:
            failures.append(
                _failure(
                    "trace.mimo.bridge_rank",
                    "MiMo Bridge event appeared on an inactive topology rank",
                    evidence,
                )
            )
        elif not bridge_ends and rank in required_bridge_ranks:
            failures.append(
                _failure(
                    "trace.mimo.bridge",
                    "MiMo training iteration contains no completed Bridge event",
                    evidence,
                )
            )
        elif bridge_ends and max(bridge_ends) > optimizer[0].rel_ts:
            failures.append(
                _failure(
                    "trace.mimo.order",
                    "optimizer-step begins before the Bridge schedule completes",
                    evidence,
                )
            )
    if not observed_bridge_ranks:
        failures.append(
            _failure(
                "trace.mimo.bridge",
                "MiMo training iteration contains no completed Bridge event",
                "all-ranks",
            )
        )
    return tuple(failures)


def validate_mimo_training_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Check the two-rank optimizer step relative to each local Bridge schedule."""

    return _validate_mimo_training_trace(
        trace_root, tuple(_TWO_RANK_ROLES), required_bridge_ranks=frozenset(_TWO_RANK_ROLES)
    )


def validate_mimo_training_fanin_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Check the DP4-to-TP2/PP2 Bridge and optimizer boundaries."""

    return _validate_mimo_training_trace(
        trace_root, tuple(_EIGHT_RANK_ROLES), required_bridge_ranks=frozenset((0, 1, 2, 3, 4, 5))
    )


def validate_mimo_training_fanout_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Check the TP2/PP2-to-DP4 Bridge and optimizer boundaries."""

    return _validate_mimo_training_trace(
        trace_root, tuple(_EIGHT_RANK_ROLES), required_bridge_ranks=frozenset((2, 3, 4, 5, 6, 7))
    )


def _validate_mimo_training_run(
    run_root: Path,
    trace_enabled: bool,
    *,
    expected_roles: Mapping[int, str],
    loss_ranks: frozenset[int],
) -> tuple[Failure, ...]:
    """Validate real F/B, optimizer update, and checkpoint reload."""

    failures: list[Failure] = []
    checkpoint_paths: set[str] = set()
    for rank, role in expected_roles.items():
        path = run_root / f"training-result-rank-{rank}.json"
        if not path.is_file():
            failures.append(
                _failure(
                    "run.mimo.result_missing",
                    f"rank {rank} did not write a training result",
                    str(path),
                )
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "backward_completed": True,
            "checkpoint_format": "torch_dist",
            "checkpoint_model_reloaded": True,
            "checkpoint_optimizer_reloaded": True,
            "completed": True,
            "global_rank": rank,
            "loss_finite": True,
            "module_role": role,
            "optimizer_step": "completed",
            "optimizer_success": True,
            "trace_enabled": trace_enabled,
            "use_distributed_optimizer": False,
            "world_size": len(expected_roles),
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                failures.append(
                    _failure(
                        "run.mimo.result",
                        f"rank {rank} has {field}={payload.get(field)!r}, expected {value!r}",
                        str(path),
                    )
                )
        grad_norm = payload.get("grad_norm")
        if (
            not isinstance(grad_norm, (int, float))
            or not math.isfinite(grad_norm)
            or grad_norm <= 0
        ):
            failures.append(
                _failure(
                    "run.mimo.grad_norm",
                    f"rank {rank} has invalid grad_norm={grad_norm!r}",
                    str(path),
                )
            )
        parameter_count = payload.get("parameter_count")
        changed = payload.get("parameters_changed")
        if (
            not isinstance(parameter_count, int)
            or parameter_count <= 0
            or not isinstance(changed, int)
            or changed <= 0
            or changed > parameter_count
        ):
            failures.append(
                _failure(
                    "run.mimo.parameters",
                    f"rank {rank} has invalid parameter update counts",
                    str(path),
                )
            )
        expected_loss_count = 4 if rank in loss_ranks else 0
        if payload.get("loss_count") != expected_loss_count:
            failures.append(
                _failure(
                    "run.mimo.loss",
                    f"rank {rank} has loss_count={payload.get('loss_count')!r}, "
                    f"expected {expected_loss_count}",
                    str(path),
                )
            )
        checkpoint_path = payload.get("checkpoint_path")
        if isinstance(checkpoint_path, str):
            checkpoint_paths.add(checkpoint_path)
        else:
            failures.append(
                _failure(
                    "run.mimo.checkpoint_path",
                    f"rank {rank} did not record a checkpoint path",
                    str(path),
                )
            )
        checkpoint_file_count = payload.get("checkpoint_file_count")
        if not isinstance(checkpoint_file_count, int) or checkpoint_file_count <= 0:
            failures.append(
                _failure(
                    "run.mimo.checkpoint_files",
                    f"rank {rank} recorded no checkpoint files",
                    str(path),
                )
            )

    if len(checkpoint_paths) == 1:
        checkpoint_root = run_root / next(iter(checkpoint_paths))
        for child in ("model", "optimizer"):
            child_path = checkpoint_root / child
            if not child_path.is_dir() or not any(child_path.rglob("*")):
                failures.append(
                    _failure(
                        "run.mimo.checkpoint_missing",
                        f"checkpoint component {child!r} is absent",
                        str(child_path),
                    )
                )
    elif checkpoint_paths:
        failures.append(
            _failure(
                "run.mimo.checkpoint_disagreement",
                "ranks recorded different checkpoint paths",
                ",".join(sorted(checkpoint_paths)),
            )
        )
    return tuple(failures)


def validate_mimo_training_run(run_root: Path, trace_enabled: bool) -> tuple[Failure, ...]:
    """Validate the two-rank MiMo terminal state."""

    return _validate_mimo_training_run(
        run_root, trace_enabled, expected_roles=_TWO_RANK_ROLES, loss_ranks=frozenset((1,))
    )


def validate_mimo_training_fanin_run(run_root: Path, trace_enabled: bool) -> tuple[Failure, ...]:
    """Validate the DP4 encoder to TP2/PP2 language-model terminal state."""

    return _validate_mimo_training_run(
        run_root, trace_enabled, expected_roles=_EIGHT_RANK_ROLES, loss_ranks=frozenset((6, 7))
    )


def validate_mimo_training_fanout_run(run_root: Path, trace_enabled: bool) -> tuple[Failure, ...]:
    """Validate the TP2/PP2 encoder to DP4 language-model terminal state."""

    return _validate_mimo_training_run(
        run_root,
        trace_enabled,
        expected_roles=_EIGHT_RANK_ROLES,
        loss_ranks=frozenset((4, 5, 6, 7)),
    )
