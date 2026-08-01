# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Terminal artifact contract for controlled two-iteration training profiles."""

from __future__ import annotations

from pathlib import Path

from tests.test_utils.runners.megalens_run_manifest import Failure


def validate_two_iteration_checkpoint(
    run_root: Path, _trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal torch_dist checkpoint from a two-iteration run."""

    failures: list[Failure] = []
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "2":
        failures.append(
            Failure(
                "run.training.tracker",
                "training checkpoint tracker does not point to iteration 2",
                str(tracker),
            )
        )

    common_state = checkpoint_root / "iter_0000002" / "common.pt"
    if not common_state.is_file():
        failures.append(
            Failure(
                "run.training.checkpoint",
                "training iteration 2 torch_dist checkpoint has no common state",
                str(common_state),
            )
        )
    return tuple(failures)


def validate_two_iteration_legacy_pp2_checkpoint(
    run_root: Path, _trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal legacy checkpoint for a PP2 distributed optimizer run."""

    failures: list[Failure] = []
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "2":
        failures.append(
            Failure(
                "run.training.tracker",
                "training checkpoint tracker does not point to iteration 2",
                str(tracker),
            )
        )

    iteration_root = checkpoint_root / "iter_0000002"
    for pipeline_rank in range(2):
        rank_root = iteration_root / f"mp_rank_00_{pipeline_rank:03d}"
        for name in ("model_optim_rng.pt", "distrib_optim.pt"):
            artifact = rank_root / name
            if not artifact.is_file():
                failures.append(
                    Failure(
                        "run.training.checkpoint",
                        "training iteration 2 legacy checkpoint is incomplete",
                        str(artifact),
                    )
                )
    return tuple(failures)
