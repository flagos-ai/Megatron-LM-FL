# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Execute a controlled MultiModule schedule under one optional MegaLens iteration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY_ROOT_TEXT = str(_REPOSITORY_ROOT)
sys.path[:] = [
    entry for entry in sys.path if entry != _REPOSITORY_ROOT_TEXT
]
sys.path.insert(0, _REPOSITORY_ROOT_TEXT)

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from megatron.megalens.runtime import MegaLensRuntime
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import (
    destroy_all_grids,
    run_multimodule_schedule_test,
)
from tests.unit_tests.test_utilities import Utils


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=Path)
    return parser


def _runtime_args(system) -> SimpleNamespace:
    return SimpleNamespace(
        trace=bool(system.get("trace", False)),
        trace_mode=int(system.get("trace_mode", 1)),
        trace_dir=str(system.trace_dir),
        trace_interval=int(system.get("trace_interval", 1)),
        continuous_trace_iterations=int(
            system.get("continuous_trace_iterations", 1)
        ),
        trace_granularity=str(system.get("trace_granularity", "base")),
        trace_cupti_kernels=str(system.get("trace_cupti_kernels", "off")),
        trace_gather_to_rank0=False,
        hardware_monitor=False,
        sentinel_hw_sample_ms=100.0,
        sentinel_flush_interval=100,
        cuda_graph_impl="none",
        profile=False,
        use_pytorch_profiler=False,
    )


def _losses_are_finite(losses: Sequence[object]) -> bool:
    return all(
        isinstance(loss, dict)
        and isinstance(loss.get("loss_reduced"), torch.Tensor)
        and bool(torch.isfinite(loss["loss_reduced"]).all().item())
        for loss in losses
    )


def _grid_config(grid, *, name: str | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "tp": int(grid["tp"]),
        "pp": int(grid["pp"]),
        "dp": int(grid["dp"]),
        "grid_offset": int(grid["grid_offset"]),
    }
    if name is not None:
        config["name"] = name
    return config


def _bridge_grids(bridge) -> tuple[list[dict[str, object]], dict[str, object]]:
    encoder_grid = bridge.get("encoder_grid")
    llm_grid = bridge.get("llm_grid")
    if encoder_grid is None:
        encoder_grid = {
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "grid_offset": 0,
        }
    if llm_grid is None:
        llm_grid = {
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "grid_offset": 1,
        }
    return [_grid_config(encoder_grid, name="encoder")], _grid_config(llm_grid)


def _grid_size(grid: dict[str, object]) -> int:
    return int(grid["tp"]) * int(grid["pp"]) * int(grid["dp"])


def _module_role(
    rank: int,
    encoder_configs: Sequence[dict[str, object]],
    llm_config: dict[str, object],
) -> str:
    for encoder_config in encoder_configs:
        offset = int(encoder_config["grid_offset"])
        if offset <= rank < offset + _grid_size(encoder_config):
            return "encoder"
    llm_offset = int(llm_config["grid_offset"])
    if llm_offset <= rank < llm_offset + _grid_size(llm_config):
        return "llm"
    raise ValueError(f"rank {rank} is outside the configured Bridge grids")


def _result_payload(
    *,
    rank: int,
    world_size: int,
    losses: Sequence[object],
    run_state: dict[str, object],
    module_role: str,
    trace_enabled: bool,
) -> dict[str, object]:
    return {
        "backward_completed": True,
        "completed": True,
        "global_rank": rank,
        "gradient_count": run_state["gradient_count"],
        "gradient_finite": run_state["gradient_finite"],
        "gradient_norm": run_state["gradient_norm"],
        "loss_count": len(losses),
        "loss_finite": _losses_are_finite(losses),
        "module_role": module_role,
        "optimizer_step": "not_run",
        "trace_enabled": trace_enabled,
        "world_size": world_size,
    }


def _write_result(
    run_dir: Path,
    *,
    rank: int,
    payload: dict[str, object],
) -> None:
    path = run_dir / f"training-result-rank-{rank}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = OmegaConf.load(args.config_file)
    system = config.train.system
    bridge = config.train.bridge
    encoder_configs, llm_config = _bridge_grids(bridge)
    run_dir = Path(os.environ["MEGALENS_GATE_CONTAINER_RUN_DIR"])
    runtime_args = _runtime_args(system)
    runtime = None
    completed = False
    result_payload = None

    Utils.initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        configured_world_size = sum(
            _grid_size(grid) for grid in (*encoder_configs, llm_config)
        )
        if configured_world_size != world_size:
            raise ValueError(
                "configured Bridge grids contain "
                f"{configured_world_size} ranks, but the distributed world has "
                f"{world_size}"
            )
        if runtime_args.trace:
            runtime = MegaLensRuntime(runtime_args)
        iteration = runtime.iteration(1) if runtime is not None else nullcontext()
        losses, run_state = run_multimodule_schedule_test(
            encoder_configs,
            llm_config,
            hidden_size=int(bridge.hidden_size),
            seq_length=int(bridge.seq_length),
            micro_batch_size=int(bridge.micro_batch_size),
            num_microbatches=int(bridge.num_microbatches),
            iteration_context=iteration,
            return_run_state=True,
        )
        result_payload = _result_payload(
            rank=rank,
            world_size=world_size,
            losses=losses,
            run_state=run_state,
            module_role=_module_role(rank, encoder_configs, llm_config),
            trace_enabled=runtime_args.trace,
        )
        completed = True
    finally:
        if runtime is not None:
            runtime.shutdown(graceful=completed)
        Utils.destroy_model_parallel()
        destroy_all_grids()
        if dist.is_initialized():
            dist.destroy_process_group()
    assert result_payload is not None
    _write_result(run_dir, rank=rank, payload=result_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
