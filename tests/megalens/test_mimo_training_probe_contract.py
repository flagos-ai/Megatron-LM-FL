# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from pathlib import Path

import pytest
import yaml

from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import mimo_probe_contract
from tests.test_utils.runners import run_flagscale_megalens as gate

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flagscale_single_node_mimo_smoke.yaml"
_FANIN_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "flagscale_single_node_mimo_fanin.yaml"
)
_FANOUT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "flagscale_single_node_mimo_fanout.yaml"
)


def _write_results(
    run_root: Path,
    *,
    trace_enabled: bool,
    roles: tuple[tuple[int, str], ...] = ((0, "encoder"), (1, "llm")),
    loss_ranks: frozenset[int] = frozenset((1,)),
) -> None:
    checkpoint_root = run_root / "checkpoints" / "iteration-1"
    for component in ("model", "optimizer"):
        component_root = checkpoint_root / component
        component_root.mkdir(parents=True, exist_ok=True)
        (component_root / "metadata.json").write_text("{}", encoding="utf-8")
    for rank, role in roles:
        payload = {
            "backward_completed": True,
            "checkpoint_file_count": 2,
            "checkpoint_format": "torch_dist",
            "checkpoint_model_reloaded": True,
            "checkpoint_optimizer_reloaded": True,
            "checkpoint_path": "checkpoints/iteration-1",
            "completed": True,
            "global_rank": rank,
            "grad_norm": 1.25,
            "loss_count": 4 if rank in loss_ranks else 0,
            "loss_finite": True,
            "module_role": role,
            "optimizer_step": "completed",
            "optimizer_success": True,
            "parameter_count": 12,
            "parameters_changed": 10,
            "trace_enabled": trace_enabled,
            "use_distributed_optimizer": False,
            "world_size": len(roles),
        }
        (run_root / f"training-result-rank-{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _write_trace(
    trace_root: Path,
    *,
    optimizer_before_bridge: bool = False,
    world_size: int = 2,
    bridge_ranks: frozenset[int] | None = None,
) -> None:
    trace_root.mkdir(parents=True)
    direction_attributes = {
        "bridge-send-forward": ("send", "forward", 1),
        "bridge-recv-forward": ("recv", "forward", 0),
        "bridge-send-backward": ("send", "backward", 0),
        "bridge-recv-backward": ("recv", "backward", 1),
    }
    for rank in range(world_size):
        rows: list[dict[str, object]] = [
            {"name": "iteration", "ph": "B", "iteration": 1, "pad_before": 0}
        ]
        timestamp = 0

        def event(name: str, phase: str, **attributes: object) -> None:
            nonlocal timestamp
            timestamp += 1
            rows.append(
                {
                    "name": name,
                    "ph": phase,
                    "rel_ts": timestamp,
                    "dev": rank,
                    "g_rk": rank,
                    "dp_rk": 0,
                    "pp_rk": 0,
                    "tp_rk": 0,
                    **attributes,
                }
            )

        def optimizer() -> None:
            event("optimizer-step", "B")
            event("optimizer-step", "E")

        def bridge() -> None:
            event(
                "bridge-p2p-launch",
                "B",
                batch_id=f"bridge-p2p:{rank}:1",
                message_kind="shape",
                operation_count=1,
                operations=[],
                src_module="images",
                dest_module="language",
                transport_api="batch_isend_irecv",
            )
            event("bridge-p2p-launch", "E")
            event(
                "bridge-grid-broadcast",
                "B",
                collective_role="source",
                grid_side="encoder",
                message_kind="shape",
                pipeline_direction="forward",
                source_rank=0,
                src_module="images",
                dest_module="language",
                transport_api="broadcast",
            )
            event("bridge-grid-broadcast", "E")
            for name, (direction, pipeline_direction, peer_rank) in direction_attributes.items():
                event(
                    name,
                    "B",
                    communicator_kind="bridge",
                    data_bytes=16,
                    direction=direction,
                    message_kind="shape",
                    peer_rank=peer_rank,
                    pipeline_direction=pipeline_direction,
                    src_module="images",
                    dest_module="language",
                    transport_api="batch_isend_irecv",
                )
                event(name, "E")

        if optimizer_before_bridge:
            optimizer()
        if bridge_ranks is None or rank in bridge_ranks:
            bridge()
        if not optimizer_before_bridge:
            optimizer()
        rows.append({"name": "iteration", "ph": "E", "iteration": 1, "duration_wall": timestamp})
        path = trace_root / f"benchmark-global-{rank}-data-0-pipeline-0-tensor-0.json"
        path.write_text(json.dumps(rows), encoding="utf-8")


def test_mimo_profile_uses_controlled_native_training_entry() -> None:
    config = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))

    assert config["experiment"]["task"] == {
        "type": "train",
        "backend": "native",
        "entrypoint": (
            "/workspace/Megatron-LM-FL/" "tests/test_utils/runners/run_mimo_training.py"
        ),
    }
    assert config["experiment"]["runner"]["nproc_per_node"] == 2
    assert config["train"]["system"]["trace_granularity"] == "full"
    assert config["train"]["mimo"]["use_distributed_optimizer"] is False
    assert gate._CONFIG_PROFILES[_FIXTURE.stem] == "mimo-train2"


@pytest.mark.parametrize(
    ("fixture", "profile", "micro_batch_size", "encoder_grid", "llm_grid"),
    (
        (
            _FANIN_FIXTURE,
            "mimo-train8-fanin",
            4,
            {"tp": 1, "pp": 1, "dp": 4, "grid_offset": 0},
            {"tp": 2, "pp": 2, "dp": 1, "grid_offset": 4},
        ),
        (
            _FANOUT_FIXTURE,
            "mimo-train8-fanout",
            1,
            {"tp": 2, "pp": 2, "dp": 1, "grid_offset": 0},
            {"tp": 1, "pp": 1, "dp": 4, "grid_offset": 4},
        ),
    ),
)
def test_mimo_eight_rank_profiles_reuse_proven_asymmetric_grids(
    fixture: Path,
    profile: str,
    micro_batch_size: int,
    encoder_grid: dict[str, int],
    llm_grid: dict[str, int],
) -> None:
    config = yaml.safe_load(fixture.read_text(encoding="utf-8"))

    assert config["experiment"]["task"]["entrypoint"].endswith("/run_mimo_training.py")
    assert config["experiment"]["runner"]["nproc_per_node"] == 8
    assert config["train"]["system"]["trace_granularity"] == "full"
    assert config["train"]["mimo"]["micro_batch_size"] == micro_batch_size
    assert config["train"]["mimo"]["encoder_grid"] == encoder_grid
    assert config["train"]["mimo"]["llm_grid"] == llm_grid
    assert gate._CONFIG_PROFILES[fixture.stem] == profile


def test_mimo_run_contract_accepts_real_update_and_checkpoint_reload(tmp_path: Path) -> None:
    _write_results(tmp_path, trace_enabled=True)

    assert mimo_probe_contract.validate_mimo_training_run(tmp_path, True) == ()


@pytest.mark.parametrize(
    ("validator", "loss_ranks"),
    (
        (mimo_probe_contract.validate_mimo_training_fanin_run, frozenset((6, 7))),
        (mimo_probe_contract.validate_mimo_training_fanout_run, frozenset((4, 5, 6, 7))),
    ),
)
def test_mimo_eight_rank_run_contract_accepts_terminal_state(
    tmp_path: Path, validator, loss_ranks: frozenset[int]
) -> None:
    roles = tuple((rank, "encoder" if rank < 4 else "llm") for rank in range(8))
    _write_results(tmp_path, trace_enabled=True, roles=roles, loss_ranks=loss_ranks)

    assert validator(tmp_path, True) == ()


def test_mimo_trace_contract_accepts_bridge_then_optimizer(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root)

    report = manifest.validate_trace(trace_root, gate.PROFILES["mimo-train2"], trace_enabled=True)

    assert report.passed


@pytest.mark.parametrize(
    ("profile", "bridge_ranks"),
    (
        ("mimo-train8-fanin", frozenset((0, 1, 2, 3, 4, 5))),
        ("mimo-train8-fanout", frozenset((2, 3, 4, 5, 6, 7))),
    ),
)
def test_mimo_eight_rank_trace_contract_accepts_bridge_then_optimizer(
    tmp_path: Path, profile: str, bridge_ranks: frozenset[int]
) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root, world_size=8, bridge_ranks=bridge_ranks)

    report = manifest.validate_trace(trace_root, gate.PROFILES[profile], trace_enabled=True)

    assert report.passed


def test_mimo_trace_contract_rejects_optimizer_before_bridge(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root, optimizer_before_bridge=True)

    failures = mimo_probe_contract.validate_mimo_training_trace(trace_root)

    assert {failure.code for failure in failures} == {"trace.mimo.order"}
