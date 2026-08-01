# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run the colocated MiMo example through Megatron's production pretrain loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch.distributed as dist

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY_ROOT_TEXT = str(_REPOSITORY_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _REPOSITORY_ROOT_TEXT]
sys.path.insert(0, _REPOSITORY_ROOT_TEXT)


def add_flagscale_mimo_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the MiMo example arguments and consume FlagScale's log directory."""

    from examples.mimo.train import add_mimo_args

    parser = add_mimo_args(parser)
    parser.add_argument("--straggler-log-dir", type=str, default=None)
    return parser


def _read_tracker(checkpoint_root: Path) -> int | None:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    value = tracker.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None


def _write_result() -> None:
    from megatron.training import get_args

    args = get_args()
    rank = dist.get_rank()
    checkpoint_root = Path(args.save)
    run_root = Path(os.environ["MEGALENS_GATE_CONTAINER_RUN_DIR"])
    payload = {
        "checkpoint_tracker_iteration": _read_tracker(checkpoint_root),
        "completed": True,
        "consumed_train_samples": int(args.consumed_train_samples),
        "final_iteration": int(args.curr_iteration),
        "global_rank": rank,
        "loaded_iteration": int(args.iteration),
        "trace_enabled": bool(args.trace),
        "train_iters": int(args.train_iters),
        "world_size": dist.get_world_size(),
    }
    result = run_root / f"mimo-pretrain-result-rank-{rank}.json"
    result.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Delegate to the existing MiMo providers and production pretrain loop."""

    from examples.mimo.train import (
        forward_step,
        model_provider,
        train_valid_test_datasets_provider,
    )
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={},
        extra_args_provider=add_flagscale_mimo_arguments,
    )
    _write_result()


if __name__ == "__main__":
    main()
