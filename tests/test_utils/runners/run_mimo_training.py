# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run one controlled heterogeneous MiMo update and checkpoint round trip."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY_ROOT_TEXT = str(_REPOSITORY_ROOT)
sys.path[:] = [
    entry for entry in sys.path if entry != _REPOSITORY_ROOT_TEXT
]
sys.path.insert(0, _REPOSITORY_ROOT_TEXT)

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.megalens.runtime import MegaLensRuntime
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    MimoTrainingState,
    destroy_all_grids,
    get_mimo_model,
    run_mimo_1f1b_test,
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


def _grid_config(grid) -> dict[str, int]:
    return {
        "tp": int(grid["tp"]),
        "pp": int(grid["pp"]),
        "dp": int(grid["dp"]),
        "grid_offset": int(grid["grid_offset"]),
    }


def _grid_size(grid: Mapping[str, int]) -> int:
    return grid["tp"] * grid["pp"] * grid["dp"]


def _module_role(
    rank: int,
    encoder_grid: Mapping[str, int],
    llm_grid: Mapping[str, int],
) -> str:
    encoder_offset = encoder_grid["grid_offset"]
    if encoder_offset <= rank < encoder_offset + _grid_size(encoder_grid):
        return "encoder"
    llm_offset = llm_grid["grid_offset"]
    if llm_offset <= rank < llm_offset + _grid_size(llm_grid):
        return "llm"
    raise ValueError(f"rank {rank} is outside the configured MiMo grids")


def _losses_are_finite(losses: Sequence[object]) -> bool:
    for loss in losses:
        if not isinstance(loss, Mapping) or "loss_reduced" not in loss:
            return False
        value = loss["loss_reduced"]
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all().item()):
                return False
        elif (
            not isinstance(value, (int, float))
            or not bool(torch.isfinite(torch.tensor(value)).item())
        ):
            return False
    return True


def _assert_nested_equal(expected, observed, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(observed, torch.Tensor) or not torch.equal(expected, observed):
            raise AssertionError(f"checkpoint mismatch at {path}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            raise AssertionError(f"checkpoint keys differ at {path}")
        for key, value in expected.items():
            _assert_nested_equal(value, observed[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, type(expected)) or len(expected) != len(observed):
            raise AssertionError(f"checkpoint sequence differs at {path}")
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed)
        ):
            _assert_nested_equal(
                expected_item,
                observed_item,
                f"{path}[{index}]",
            )
        return
    if expected != observed:
        raise AssertionError(f"checkpoint value differs at {path}")


def _checkpoint_round_trip(
    run_dir: Path,
    *,
    state: MimoTrainingState,
    mimo,
) -> dict[str, object]:
    checkpoint_root = run_dir / "checkpoints" / "iteration-1"
    model_checkpoint = checkpoint_root / "model"
    optimizer_checkpoint = checkpoint_root / "optimizer"
    if dist.get_rank() == 0:
        model_checkpoint.mkdir(parents=True)
        optimizer_checkpoint.mkdir()
    dist.barrier()

    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in state.model.named_parameters()
    }
    expected_optimizer = copy.deepcopy(state.optimizer.state_dict())

    save(state.model.sharded_state_dict(), str(model_checkpoint))
    optimizer_state = state.optimizer.sharded_state_dict(
        state.model.sharded_state_dict(),
        is_loading=False,
    )
    save(
        optimizer_state,
        str(optimizer_checkpoint),
        validate_access_integrity=False,
    )
    dist.barrier()

    restored_model, _, _, _, _ = get_mimo_model(
        encoder_name="images",
        encoder_grid=state.encoder_grid,
        llm_grid=state.llm_grid,
        hidden_size=int(mimo.hidden_size),
        num_layers=int(mimo.num_layers),
        vocab_size=int(mimo.vocab_size),
        seq_len=int(mimo.seq_length),
        use_distributed_optimizer=bool(mimo.use_distributed_optimizer),
    )
    restored_optimizer = get_mimo_optimizer(
        restored_model,
        copy.deepcopy(state.optimizer_config),
    )

    model_template = restored_model.sharded_state_dict()
    loaded_model, missing, unexpected = load(
        model_template,
        str(model_checkpoint),
        strict=StrictHandling.RETURN_ALL,
    )
    real_missing = [key for key in missing if "_extra_state" not in key]
    real_unexpected = [key for key in unexpected if "_extra_state" not in key]
    if real_missing or real_unexpected:
        raise AssertionError(
            "checkpoint model keys differ: "
            f"missing={real_missing}, unexpected={real_unexpected}"
        )
    restored_model.load_state_dict(loaded_model)

    optimizer_template = restored_optimizer.sharded_state_dict(
        restored_model.sharded_state_dict(),
        is_loading=True,
    )
    loaded_optimizer = load(
        optimizer_template,
        str(optimizer_checkpoint),
        validate_access_integrity=False,
    )
    restored_optimizer.load_state_dict(loaded_optimizer)

    observed_parameters = dict(restored_model.named_parameters())
    if set(expected_parameters) != set(observed_parameters):
        raise AssertionError("checkpoint model parameter names differ after reload")
    for name, expected in expected_parameters.items():
        if not torch.equal(expected, observed_parameters[name]):
            raise AssertionError(
                f"checkpoint model parameter {name!r} differs after reload"
            )
    _assert_nested_equal(
        expected_optimizer,
        restored_optimizer.state_dict(),
        "optimizer",
    )
    dist.barrier()

    return {
        "checkpoint_file_count": sum(
            path.is_file() for path in checkpoint_root.rglob("*")
        ),
        "checkpoint_format": "torch_dist",
        "checkpoint_model_reloaded": True,
        "checkpoint_optimizer_reloaded": True,
        "checkpoint_path": checkpoint_root.relative_to(run_dir).as_posix(),
    }


def _write_result(run_dir: Path, rank: int, payload: Mapping[str, object]) -> None:
    path = run_dir / f"training-result-rank-{rank}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = OmegaConf.load(args.config_file)
    system = config.train.system
    mimo = config.train.mimo
    encoder_grid = _grid_config(mimo.encoder_grid)
    llm_grid = _grid_config(mimo.llm_grid)
    run_dir = Path(os.environ["MEGALENS_GATE_CONTAINER_RUN_DIR"])
    runtime_args = _runtime_args(system)
    runtime = None
    completed = False
    result_payload = None

    Utils.initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        configured_world_size = _grid_size(encoder_grid) + _grid_size(llm_grid)
        if configured_world_size != world_size:
            raise ValueError(
                "configured MiMo grids contain "
                f"{configured_world_size} ranks, but the distributed world has "
                f"{world_size}"
            )
        if runtime_args.trace:
            runtime = MegaLensRuntime(runtime_args)
        iteration = runtime.iteration(1) if runtime is not None else nullcontext()
        losses, state = run_mimo_1f1b_test(
            encoder_tp=encoder_grid["tp"],
            encoder_pp=encoder_grid["pp"],
            encoder_dp=encoder_grid["dp"],
            encoder_offset=encoder_grid["grid_offset"],
            llm_tp=llm_grid["tp"],
            llm_pp=llm_grid["pp"],
            llm_dp=llm_grid["dp"],
            llm_offset=llm_grid["grid_offset"],
            hidden_size=int(mimo.hidden_size),
            num_layers=int(mimo.num_layers),
            vocab_size=int(mimo.vocab_size),
            seq_length=int(mimo.seq_length),
            micro_batch_size=int(mimo.micro_batch_size),
            num_microbatches=int(mimo.num_microbatches),
            use_distributed_optimizer=bool(mimo.use_distributed_optimizer),
            iteration_context=iteration,
            return_run_state=True,
        )
        checkpoint = _checkpoint_round_trip(
            run_dir,
            state=state,
            mimo=mimo,
        )
        result_payload = {
            "backward_completed": True,
            **checkpoint,
            "completed": True,
            "global_rank": rank,
            "grad_norm": state.grad_norm,
            "loss_count": len(losses),
            "loss_finite": _losses_are_finite(losses),
            "module_role": _module_role(rank, encoder_grid, llm_grid),
            "optimizer_step": "completed",
            "optimizer_success": state.optimizer_success,
            "parameter_count": state.parameter_count,
            "parameters_changed": state.changed_parameter_count,
            "trace_enabled": runtime_args.trace,
            "use_distributed_optimizer": bool(mimo.use_distributed_optimizer),
            "world_size": world_size,
        }
        completed = True
    finally:
        if runtime is not None:
            runtime.shutdown(graceful=completed)
        if dist.is_initialized():
            dist.barrier()
        Utils.destroy_model_parallel()
        destroy_all_grids()
        if dist.is_initialized():
            dist.destroy_process_group()
    assert result_payload is not None
    _write_result(run_dir, rank, result_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
