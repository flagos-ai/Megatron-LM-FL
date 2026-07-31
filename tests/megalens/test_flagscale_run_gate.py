# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import run_flagscale_megalens as gate

_FIXTURES = Path(__file__).parent / "fixtures"
_CONFIG_PROFILE_CASES = {
    "flagscale_single_node_smoke.yaml": "pp1",
    "flagscale_single_node_pp2_smoke.yaml": "pp2",
    "flagscale_single_node_ep2_smoke.yaml": "ep2-alltoall",
    "flagscale_single_node_ep2_fine_grained_smoke.yaml": "ep2-fine-grained",
    "flagscale_single_node_dp2_standard_smoke.yaml": "dp2-standard-ddp",
    "flagscale_single_node_dp2_distopt_smoke.yaml": "dp2-distopt",
    "flagscale_single_node_dp8_standard_smoke.yaml": "dp8-standard-ddp",
    "flagscale_single_node_dp8_distopt_smoke.yaml": "dp8-distopt",
    "flagscale_single_node_te_cuda_graph_attn_smoke.yaml": ("te-attn-cuda-graph"),
    "flagscale_single_node_te_cuda_graph_moe_router_smoke.yaml": (
        "te-moe-router-cuda-graph"
    ),
    "flagscale_single_node_bert_smoke.yaml": "bert-encoder",
}


def _write_rank_trace(run_dir: Path, rank: int, event: str = "forward") -> None:
    trace_root = run_dir / "traces"
    trace_root.mkdir(exist_ok=True)
    fields = {
        "iteration": 2,
        "g_rk": rank,
        "dp_rk": rank,
        "pp_rk": 0,
        "tp_rk": 0,
    }
    path = trace_root / f"benchmark-global-{rank}-data-{rank}-pipeline-0-tensor-0.json"
    path.write_text(
        json.dumps(
            [
                {"name": event, "ph": "B", **fields},
                {"name": event, "ph": "E", **fields},
            ]
        ),
        encoding="utf-8",
    )


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "trace-on",
    image: str = "example/flagscale:dev",
    child_returncode: int = 0,
    config: str = "flagscale_single_node_smoke.yaml",
) -> tuple[int, Path, tuple[str, ...]]:
    run_dir = tmp_path / "run"
    observed_command: tuple[str, ...] = ()

    def fake_run_foreground(
        argv: tuple[str, ...],
        *,
        env: dict[str, str],
        launcher_log: Path,
        **_: object,
    ) -> gate.ExecutionResult:
        nonlocal observed_command
        observed_command = tuple(argv)
        launcher_log.write_text("official FlagScale entrypoint\n", encoding="utf-8")
        if mode == "trace-on" and child_returncode == 0:
            _write_rank_trace(Path(env["MEGALENS_GATE_RUN_DIR"]), 0)
        return gate.ExecutionResult(child_returncode)

    monkeypatch.setattr(gate, "run_foreground", fake_run_foreground)
    result = gate.main(
        (
            "--run-dir",
            str(run_dir),
            "--input-config",
            str(_FIXTURES / config),
            "--mode",
            mode,
            "--image",
            image,
            "--controller-revision",
            "legacy-value-is-accepted",
        )
    )
    return result, run_dir, observed_command


@pytest.mark.parametrize(
    ("config", "profile"),
    tuple(_CONFIG_PROFILE_CASES.items()),
)
def test_existing_yaml_profiles_remain_selectable(config: str, profile: str) -> None:
    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            str(_FIXTURES / config),
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
        )
    )

    assert gate._profile_from_arguments(args).name == profile


def test_runner_uses_requested_image_current_source_and_flagscale_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, command = _invoke(tmp_path, monkeypatch)
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["status"] == "completed"
    assert payload["image"] == "example/flagscale:dev"
    assert payload["source"]["path"] == str(gate._REPOSITORY_ROOT.resolve())
    assert payload["source"]["head"] == gate._source_head(gate._REPOSITORY_ROOT)
    source_mount = f"{gate._REPOSITORY_ROOT.resolve()}:{gate.CONTAINER_SOURCE_ROOT}:ro"
    assert source_mount in command
    assert command.index(source_mount) < command.index("example/flagscale:dev")
    assert any("conda activate flagscale-train" in argument for argument in command)
    assert ("flagscale", "run") == command[-5:-3]
    assert command[-1] == "--action=test"


def test_runner_records_config_returncode_log_and_trace_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, command = _invoke(tmp_path, monkeypatch)
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["command"] == list(command)
    assert payload["config"]["path"].endswith("inputs/flagscale_single_node_smoke.yaml")
    assert len(payload["config"]["sha256"]) == 64
    assert payload["execution"]["returncode"] == 0
    assert Path(payload["execution"]["log"]).read_text(encoding="utf-8") == (
        "official FlagScale entrypoint\n"
    )
    assert payload["validation"]["trace"]["ranks"] == [0]
    assert len(payload["validation"]["trace"]["shards"]) == 1


def test_trace_off_succeeds_without_trace_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(tmp_path, monkeypatch, mode="trace-off")
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["mode"] == "trace-off"
    assert payload["validation"]["trace"]["shards"] == []


def test_child_failure_is_preserved_in_simple_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path,
        monkeypatch,
        child_returncode=23,
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 23
    assert payload["status"] == "failed"
    assert payload["execution"]["returncode"] == 23
    assert payload["validation"]["passed"] is False


def test_profile_failure_changes_zero_child_exit_to_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path,
        monkeypatch,
        config="flagscale_single_node_pp2_smoke.yaml",
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 1
    assert payload["execution"]["returncode"] == 0
    assert {failure["code"] for failure in payload["validation"]["failures"]} >= {
        "trace.rank_count",
        "trace.event_missing",
    }


def test_docker_command_keeps_optional_flagscale_overlay_without_hash_lock(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "training.py"
    overlay.write_text("# compatibility overlay\n", encoding="utf-8")
    command = gate._docker_command(
        run_dir=tmp_path,
        config_name="smoke",
        mode="trace-on",
        image="example/flagscale:dev",
        source_root=gate._REPOSITORY_ROOT,
        rdzv_port=12345,
        ep_dispatcher=None,
        flagscale_training_overlay=overlay,
    )

    assert (
        f"{overlay.resolve()}:"
        "/workspace/FlagScale/flagscale/train/megatron/training/training.py:ro"
    ) in command
    assert not any("sha256:" in argument for argument in command)
