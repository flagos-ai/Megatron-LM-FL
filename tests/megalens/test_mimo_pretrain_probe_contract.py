# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import mimo_pretrain_probe_contract
from tests.test_utils.runners import run_flagscale_megalens as gate

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "flagscale_single_node_mimo_pretrain_smoke.yaml"
)
_SAVE_FIXTURE = _FIXTURE.with_name(
    "flagscale_single_node_mimo_pretrain_save_smoke.yaml"
)
_RESUME_FIXTURE = _FIXTURE.with_name(
    "flagscale_single_node_mimo_pretrain_resume_smoke.yaml"
)


def _write_trace(
    trace_root: Path,
    *,
    include_bridge: bool = False,
    iterations: tuple[int, ...] = (1, 2),
) -> None:
    trace_root.mkdir(parents=True)
    for rank in (0, 1):
        rows: list[dict[str, object]] = []
        timestamp = 0

        def event(name: str, phase: str, **fields: object) -> None:
            nonlocal timestamp
            timestamp += 1
            rows.append(
                {
                    "name": name,
                    "ph": phase,
                    "rel_ts": timestamp,
                    "dev": rank,
                    "g_rk": rank,
                    "dp_rk": rank,
                    "pp_rk": 0,
                    "tp_rk": 0,
                    **fields,
                }
            )

        for iteration in iterations:
            rows.append(
                {
                    "name": "iteration",
                    "ph": "B",
                    "iteration": iteration,
                    "pad_before": 0,
                }
            )
            for _ in range(2):
                event("forward-step", "B")
                event("loss", "B")
                event("loss", "E")
                event("forward-step", "E")
            if include_bridge:
                event("bridge-p2p-launch", "B")
                event("bridge-p2p-launch", "E")
            event("optimizer", "B")
            event("optimizer-step", "B")
            event("optimizer-step", "E")
            event("optimizer", "E")
            event("optimizer-postprocess", "B")
            event("optimizer-postprocess", "E")
            rows.append(
                {
                    "name": "iteration",
                    "ph": "E",
                    "iteration": iteration,
                    "duration_wall": timestamp,
                }
            )

        path = trace_root / f"benchmark-global-{rank}-data-{rank}-pipeline-0-tensor-0.json"
        path.write_text(json.dumps(rows), encoding="utf-8")


def _write_run_artifacts(
    run_root: Path,
    *,
    consumed_train_samples: int = 8,
    final_iteration: int = 2,
    loaded_iteration: int = 0,
    trace_enabled: bool,
) -> None:
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        str(final_iteration), encoding="utf-8"
    )
    iteration_root = checkpoint_root / f"iter_{final_iteration:07d}"
    iteration_root.mkdir()
    (iteration_root / "common.pt").write_bytes(b"common")
    for rank in (0, 1):
        payload = {
            "checkpoint_tracker_iteration": final_iteration,
            "completed": True,
            "consumed_train_samples": consumed_train_samples,
            "final_iteration": final_iteration,
            "global_rank": rank,
            "loaded_iteration": loaded_iteration,
            "trace_enabled": trace_enabled,
            "train_iters": 2,
            "world_size": 2,
        }
        (run_root / f"mimo-pretrain-result-rank-{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_mimo_pretrain_profile_uses_megatron_production_entry() -> None:
    config = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))

    assert config["experiment"]["task"] == {
        "type": "train",
        "backend": "megatron",
        "entrypoint": (
            "/workspace/Megatron-LM-FL/"
            "tests/test_utils/runners/run_flagscale_mimo_pretrain.py"
        ),
    }
    assert config["experiment"]["runner"]["nproc_per_node"] == 2
    assert config["train"]["system"]["tensor_model_parallel_size"] == 1
    assert config["train"]["system"]["pipeline_model_parallel_size"] == 1
    assert config["train"]["system"]["use_distributed_optimizer"] is False
    assert config["train"]["system"]["trace_granularity"] == "full"
    assert config["train"]["system"]["trace_interval"] == 2
    assert config["train"]["system"]["continuous_trace_iterations"] == 2
    assert config["train"]["system"]["checkpoint"]["save_interval"] == 100
    assert config["train"]["model"]["global_batch_size"] == 4
    assert config["train"]["model"]["micro_batch_size"] == 1
    assert config["train"]["model"]["train_iters"] == 2
    assert gate._CONFIG_PROFILES[_FIXTURE.stem] == "mimo-pretrain2"


def test_mimo_pretrain_resume_profiles_preserve_the_two_iteration_schedule() -> None:
    save_config = yaml.safe_load(_SAVE_FIXTURE.read_text(encoding="utf-8"))
    resume_config = yaml.safe_load(_RESUME_FIXTURE.read_text(encoding="utf-8"))

    assert "exit_interval" not in save_config["train"]["system"]
    assert save_config["train"]["system"]["logging"]["log_energy"] is True
    assert save_config["train"]["system"]["checkpoint"]["save_interval"] == 1
    assert save_config["train"]["model"]["train_iters"] == 2
    assert resume_config["train"]["system"]["checkpoint"]["load"] == (
        gate.CONTAINER_CHECKPOINT_LOAD_ROOT
    )
    assert resume_config["train"]["system"]["checkpoint"]["ckpt_step"] == 1
    assert resume_config["train"]["model"]["train_iters"] == 2
    assert gate._CONFIG_PROFILES[_SAVE_FIXTURE.stem] == "mimo-pretrain-save2"
    assert gate._CONFIG_PROFILES[_RESUME_FIXTURE.stem] == "mimo-pretrain-resume2"


def test_mimo_pretrain_trace_contract_accepts_production_sequence(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root)

    report = manifest.validate_trace(
        trace_root,
        gate.PROFILES["mimo-pretrain2"],
        trace_enabled=True,
    )

    assert report.passed


def test_mimo_pretrain_trace_contract_rejects_bridge_events(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root, include_bridge=True)

    failures = mimo_pretrain_probe_contract.validate_mimo_pretrain_trace(trace_root)

    assert {failure.code for failure in failures} == {"trace.mimo_pretrain.bridge"}


def test_mimo_pretrain_run_contract_accepts_checkpoint_and_counters(
    tmp_path: Path,
) -> None:
    _write_run_artifacts(tmp_path, trace_enabled=True)

    assert mimo_pretrain_probe_contract.validate_mimo_pretrain_run(tmp_path, True) == ()


def test_mimo_pretrain_resume_contract_accepts_only_the_restored_iteration(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root, iterations=(2,))
    _write_run_artifacts(
        tmp_path,
        consumed_train_samples=8,
        final_iteration=2,
        loaded_iteration=1,
        trace_enabled=True,
    )

    report = manifest.validate_trace(
        trace_root,
        gate.PROFILES["mimo-pretrain-resume2"],
        trace_enabled=True,
    )
    failures = mimo_pretrain_probe_contract.validate_mimo_pretrain_resume_run(
        tmp_path, True
    )

    assert report.passed
    assert failures == ()


def test_mimo_pretrain_save_contract_accepts_the_restart_checkpoint(
    tmp_path: Path,
) -> None:
    _write_run_artifacts(
        tmp_path,
        consumed_train_samples=8,
        final_iteration=2,
        loaded_iteration=0,
        trace_enabled=False,
    )
    restart = tmp_path / "checkpoints" / "iter_0000001"
    restart.mkdir()
    (restart / "common.pt").write_bytes(b"common")

    assert (
        mimo_pretrain_probe_contract.validate_mimo_pretrain_save_run(
            tmp_path, False
        )
        == ()
    )
