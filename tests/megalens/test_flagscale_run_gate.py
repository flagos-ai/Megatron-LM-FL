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
    "flagscale_single_node_gpt_eager_full_smoke.yaml": "gpt-eager-full",
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


def _write_gpt_phase_trace(
    trace_root: Path,
    *,
    rank: int,
    pipeline_rank: int,
    include_postprocess: bool,
    eager_layers: int = 0,
) -> None:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    timestamp = 0

    def event(name: str, phase: str) -> None:
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
                "pp_rk": pipeline_rank,
                "tp_rk": 0,
            }
        )

    for iteration in (1, 2):
        rows.append(
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        event("forward-step", "B")
        event("decoder", "B")
        for _ in range(eager_layers):
            event("transformer_layer", "B")
            event("_forward_attention", "B")
            event("attention", "B")
            event("attention", "E")
            event("_forward_attention", "E")
            event("_forward_mlp", "B")
            event("MLP.forward", "B")
            event("MLP.forward", "E")
            event("_forward_mlp", "E")
            event("transformer_layer", "E")
        event("decoder", "E")
        event("decoder-postprocess", "B")
        if include_postprocess:
            event("output_layer", "B")
            event("output_layer", "E")
            event("loss", "B")
            event("loss", "E")
        event("decoder-postprocess", "E")
        event("forward-step", "E")
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    path = (
        trace_root
        / f"benchmark-global-{rank}-data-0-pipeline-{pipeline_rank}-tensor-0.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


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


def test_raw_framework_event_requirements_use_the_enclosing_iteration() -> None:
    required_fields = {
        field
        for profile in gate.PROFILES.values()
        for requirement in profile.events
        for field in requirement.fields
    }

    assert "iteration" not in required_fields
    assert {"g_rk", "dp_rk", "pp_rk", "tp_rk"} <= required_fields


def test_gpt_pp1_and_pp2_profiles_enforce_stage_specific_model_phases(
    tmp_path: Path,
) -> None:
    pp1_root = tmp_path / "pp1"
    _write_gpt_phase_trace(
        pp1_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
    )
    assert gate.PROFILES["gpt-eager-full"].contract(pp1_root) == ()

    pp2_root = tmp_path / "pp2"
    _write_gpt_phase_trace(
        pp2_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
    )
    _write_gpt_phase_trace(
        pp2_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
    )
    assert gate.PROFILES["pp2"].contract(pp2_root) == ()

    invalid_root = tmp_path / "invalid-pp2"
    _write_gpt_phase_trace(
        invalid_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
    )
    _write_gpt_phase_trace(
        invalid_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
    )
    failures = gate.PROFILES["pp2"].contract(invalid_root)
    assert failures
    assert {failure.code for failure in failures} == {"trace.gpt.count"}


def test_gpt_eager_profile_rejects_an_incomplete_layer_sequence(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "incomplete-eager"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=1,
    )

    failures = gate.PROFILES["gpt-eager-full"].contract(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.eager_layers",
        "trace.gpt.eager_layers",
    ]
    assert [failure.evidence for failure in failures] == [
        "rank=0 iteration=1",
        "rank=0 iteration=2",
    ]


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
