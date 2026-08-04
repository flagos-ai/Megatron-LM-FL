# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tests.test_utils.runners import cp_probe_contract
from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import run_flagscale_megalens as gate
from tests.test_utils.runners import training_run_contract

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE = _FIXTURES / "flagscale_single_node_cp2_te_smoke.yaml"
_TP2_TE_FIXTURE = _FIXTURES / "flagscale_single_node_tp2_sp_te_linear_smoke.yaml"


def _write_cp2_trace(
    trace_root: Path,
    *,
    rank: int,
    data_rank: int = 0,
    wrong_peer: bool = False,
    omit_attention: bool = False,
) -> Path:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    timestamp = 0

    def event(name: str, phase: str, **attrs: object) -> None:
        nonlocal timestamp
        timestamp += 1
        rows.append(
            {
                "name": name,
                "ph": phase,
                "rel_ts": timestamp,
                "dev": rank,
                "g_rk": rank,
                "dp_rk": data_rank,
                "pp_rk": 0,
                "tp_rk": 0,
                **attrs,
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
        for _ in range(2):
            event("transformer_layer", "B")
            event("_forward_attention", "B")
            if not omit_attention:
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
        event("output_layer", "B")
        event("output_layer", "E")
        event("loss", "B")
        event("loss", "E")
        event("decoder-postprocess", "E")
        event("forward-step", "E")
        event(
            "grad-sync",
            "B",
            schedule="no-pipelining",
            timing_phase="framework_phase",
        )
        event("all-grads-sync", "B")
        event(
            "dp-allreduce",
            "B",
            api_async_op=False,
            async_op=False,
            completion_included=False,
            data_bytes=32768,
            group_role="data_parallel",
            group_size=2,
            n_buckets=1,
            op="all_reduce",
            operation_id=f"dp:allreduce:{iteration}",
            operation_id_scope="rank_local",
            overlap_enabled=False,
            payload_role="gradient_bucket",
            stage="main_bucket_allreduce",
            timing_phase="collective_call",
        )
        event("dp-allreduce", "E", group=[rank if wrong_peer else 1 - rank])
        event("all-grads-sync", "E")
        event("grad-sync", "E")
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
        / f"benchmark-global-{rank}-data-{data_rank}-pipeline-0-tensor-0.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _write_terminal_checkpoint(run_root: Path) -> None:
    checkpoint_root = run_root / "checkpoints"
    iteration_root = checkpoint_root / "iter_0000002"
    iteration_root.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        "2", encoding="utf-8"
    )
    (iteration_root / "common.pt").write_bytes(b"checkpoint")


def test_cp2_fixture_is_the_tp1_cp2_derivative_of_the_te_baseline() -> None:
    baseline = yaml.safe_load(_TP2_TE_FIXTURE.read_text(encoding="utf-8"))
    cp2 = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    baseline["experiment"]["exp_name"] = cp2["experiment"]["exp_name"]
    baseline_system = baseline["train"]["system"]
    baseline_system["tensor_model_parallel_size"] = 1
    baseline_system["context_parallel_size"] = 2
    baseline_system["sequence_parallel"] = False

    assert cp2 == baseline

    runner = cp2["experiment"]["runner"]
    system = cp2["train"]["system"]
    model = cp2["train"]["model"]
    world_size = runner["nproc_per_node"]
    model_parallel_size = (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    assert world_size // model_parallel_size == 1
    assert system["sequence_parallel"] is False
    assert system["precision"]["bf16"] is True
    assert model["transformer_impl"] == "transformer_engine"
    assert model["num_layers"] == 2
    assert model["seq_length"] == 128
    assert model["seq_length"] % (2 * system["context_parallel_size"]) == 0
    assert model["train_iters"] == 2
    assert cp2["train"]["data"]["mock_data"] is True


def test_cp2_runner_profile_requires_only_existing_probe_events() -> None:
    profile = gate.PROFILES["cp2-te"]
    event_names = {requirement.name for requirement in profile.events}

    assert profile.rank_count == 2
    assert event_names == {
        "forward-step",
        "decoder",
        "decoder-postprocess",
        "output_layer",
        "loss",
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "MLP.forward",
        "grad-sync",
        "all-grads-sync",
        "dp-allreduce",
    }
    assert not any(name.startswith("cp-") for name in event_names)
    assert profile.contract is cp_probe_contract.validate_cp2_te_coexistence
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_transformer_engine_cp2_checkpoint
    )
    assert gate._CONFIG_PROFILES[_FIXTURE.stem] == "cp2-te"


def test_cp2_contract_accepts_gpt_hierarchy_and_dp_cp_group(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_cp2_trace(tmp_path, rank=rank)

    profile = gate.PROFILES["cp2-te"]
    report = manifest.validate_trace(tmp_path, profile, trace_enabled=True)

    assert report.passed
    assert report.ranks == (0, 1)
    assert report.failures == ()


def test_cp2_contract_rejects_a_non_cp_peer_group(tmp_path: Path) -> None:
    _write_cp2_trace(tmp_path, rank=0, wrong_peer=True)
    _write_cp2_trace(tmp_path, rank=1)

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_cp_group" in {failure.code for failure in failures}


def test_cp2_contract_requires_exactly_one_nested_dp_allreduce(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    begin = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "B"
    )
    end = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "E"
    )
    duplicate_begin = {**begin, "operation_id": "dp:allreduce:0:duplicate"}
    insertion = next(
        index
        for index, row in enumerate(rows)
        if row.get("name") == "all-grads-sync" and row.get("ph") == "E"
    )
    rows[insertion:insertion] = [duplicate_begin, dict(end)]
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_sync_count" in {failure.code for failure in failures}


def test_cp2_contract_requires_grad_sync_nesting(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    all_grads_end_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("name") == "all-grads-sync" and row.get("ph") == "E"
    )
    all_grads_end = rows.pop(all_grads_end_index)
    dp_begin_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("name") == "dp-allreduce" and row.get("ph") == "B"
    )
    rows.insert(dp_begin_index, all_grads_end)
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_sync_hierarchy" in {failure.code for failure in failures}


def test_cp2_contract_rejects_async_dp_allreduce(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    begin = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "B"
    )
    begin["async_op"] = True
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_route" in {failure.code for failure in failures}


def test_cp2_contract_forbids_alternative_dp_events(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    begin = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "B"
    )
    end = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "E"
    )
    insertion = rows.index(begin)
    rows[insertion:insertion] = [
        dict(begin, name="dp-reduce-scatter"),
        dict(end, name="dp-reduce-scatter"),
    ]
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_route" in {failure.code for failure in failures}


def test_cp2_contract_requires_peer_only_end_fields(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    end = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "E"
    )
    end["completed"] = True
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_cp_group" in {failure.code for failure in failures}


def test_cp2_contract_requires_equal_dp_payload_sizes(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    begin = next(
        row
        for row in rows
        if row.get("name") == "dp-allreduce" and row.get("ph") == "B"
    )
    begin["data_bytes"] = 16384
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_payload_consistency" in {
        failure.code for failure in failures
    }


def test_cp2_contract_rejects_dp_rank_coordinates(tmp_path: Path) -> None:
    _write_cp2_trace(tmp_path, rank=0, data_rank=1)
    _write_cp2_trace(tmp_path, rank=1)

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.coordinates" in {failure.code for failure in failures}


def test_cp2_contract_requires_the_two_layer_gpt_hierarchy(tmp_path: Path) -> None:
    _write_cp2_trace(tmp_path, rank=0, omit_attention=True)
    _write_cp2_trace(tmp_path, rank=1)

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.gpt.eager_layers" in {failure.code for failure in failures}


def test_cp2_profile_requires_rank_fields_on_existing_events(tmp_path: Path) -> None:
    first_path = _write_cp2_trace(tmp_path, rank=0)
    _write_cp2_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    first_attention = next(row for row in rows if row.get("name") == "attention")
    first_attention.pop("dp_rk")
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    report = manifest.validate_trace(
        tmp_path,
        gate.PROFILES["cp2-te"],
        trace_enabled=True,
    )

    assert "trace.field_missing" in {failure.code for failure in report.failures}


def test_cp2_training_contract_requires_the_launcher_cp_size(tmp_path: Path) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  context_parallel_size ............................ 1\n",
        encoding="utf-8",
    )

    failures = (
        training_run_contract.validate_two_iteration_transformer_engine_cp2_checkpoint(
            tmp_path, False
        )
    )
    assert {failure.code for failure in failures} == {
        "run.training.context_parallel_size"
    }

    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  context_parallel_size ............................ 2\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_transformer_engine_cp2_checkpoint(
            tmp_path, False
        )
        == ()
    )
