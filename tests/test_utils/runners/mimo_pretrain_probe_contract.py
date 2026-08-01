# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact contracts for colocated MiMo production pretraining."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_OPTIMIZER_NAMES = frozenset(("optimizer", "optimizer-step", "optimizer-postprocess"))
_OPTIMIZER_SEQUENCE = (
    ("optimizer", "B"),
    ("optimizer-step", "B"),
    ("optimizer-step", "E"),
    ("optimizer", "E"),
    ("optimizer-postprocess", "B"),
    ("optimizer-postprocess", "E"),
)
_BRIDGE_NAMES = frozenset(
    (
        "bridge-p2p-launch",
        "bridge-grid-broadcast",
        "bridge-send-forward",
        "bridge-recv-forward",
        "bridge-send-backward",
        "bridge-recv-backward",
    )
)


def _failure(code: str, message: str, evidence: str) -> Failure:
    return Failure(code, message, evidence)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    by_rank: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None:
            raise ValueError(f"trace shard {rank} has no global rank")
        if rank.global_rank in by_rank:
            raise ValueError(f"duplicate trace shard for global rank {rank.global_rank}")
        by_rank[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return by_rank


def validate_mimo_pretrain_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Validate two production iterations on both colocated DP ranks."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if set(by_rank) != {0, 1}:
        failures.append(
            _failure(
                "trace.mimo_pretrain.ranks",
                f"expected ranks [0, 1], observed {sorted(by_rank)}",
                str(trace_root),
            )
        )

    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(int(item.iteration_id) for item in iterations)
        if iteration_ids != (1, 2):
            failures.append(
                _failure(
                    "trace.mimo_pretrain.iterations",
                    f"rank {rank} has iterations {iteration_ids}, expected (1, 2)",
                    f"rank={rank}",
                )
            )
            continue

        for iteration in iterations:
            evidence = f"rank={rank} iteration={iteration.iteration_id}"
            optimizer_sequence = tuple(
                (event.name, event.ph)
                for event in iteration.events
                if event.name in _OPTIMIZER_NAMES
            )
            if optimizer_sequence != _OPTIMIZER_SEQUENCE:
                failures.append(
                    _failure(
                        "trace.mimo_pretrain.optimizer_sequence",
                        f"optimizer sequence is {optimizer_sequence!r}",
                        evidence,
                    )
                )

            phase_counts = Counter(
                (event.name, event.ph)
                for event in iteration.events
                if event.name in {"forward-step", "loss"}
            )
            for name in ("forward-step", "loss"):
                observed = (phase_counts[(name, "B")], phase_counts[(name, "E")])
                if observed != (2, 2):
                    failures.append(
                        _failure(
                            "trace.mimo_pretrain.microbatches",
                            f"event {name!r} has B/E={observed}, expected (2, 2)",
                            evidence,
                        )
                    )

            if any(event.name in _BRIDGE_NAMES for event in iteration.events):
                failures.append(
                    _failure(
                        "trace.mimo_pretrain.bridge",
                        "colocated production pretrain emitted a Bridge event",
                        evidence,
                    )
                )

    return tuple(failures)


def validate_mimo_pretrain_run(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Validate terminal counters and the standard torch_dist checkpoint."""

    failures: list[Failure] = []
    expected = {
        "checkpoint_tracker_iteration": 2,
        "completed": True,
        "consumed_train_samples": 8,
        "final_iteration": 2,
        "loaded_iteration": 0,
        "trace_enabled": trace_enabled,
        "train_iters": 2,
        "world_size": 2,
    }
    for rank in (0, 1):
        path = run_root / f"mimo-pretrain-result-rank-{rank}.json"
        if not path.is_file():
            failures.append(
                _failure(
                    "run.mimo_pretrain.result_missing",
                    f"rank {rank} did not write a production pretrain result",
                    str(path),
                )
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field, value in {**expected, "global_rank": rank}.items():
            if payload.get(field) != value:
                failures.append(
                    _failure(
                        "run.mimo_pretrain.result",
                        f"rank {rank} has {field}={payload.get(field)!r}, expected {value!r}",
                        str(path),
                    )
                )

    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "2":
        failures.append(
            _failure(
                "run.mimo_pretrain.tracker",
                "checkpoint tracker does not point to iteration 2",
                str(tracker),
            )
        )
    iteration_root = checkpoint_root / "iter_0000002"
    files = tuple(path for path in iteration_root.rglob("*") if path.is_file())
    if not iteration_root.is_dir() or not files:
        failures.append(
            _failure(
                "run.mimo_pretrain.checkpoint",
                "iteration 2 torch_dist checkpoint is absent or empty",
                str(iteration_root),
            )
        )
    elif not (iteration_root / "common.pt").is_file():
        failures.append(
            _failure(
                "run.mimo_pretrain.common_state",
                "iteration 2 checkpoint has no common state",
                str(iteration_root),
            )
        )
    return tuple(failures)
