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
_CP4_FIXTURE = _FIXTURES / "flagscale_single_node_cp4_te_smoke.yaml"
_TP2_TE_FIXTURE = _FIXTURES / "flagscale_single_node_tp2_sp_te_linear_smoke.yaml"
_QWEN3_TP2_FIXTURE = _FIXTURES / "flagscale_single_node_qwen3_enron_tp2_sp.yaml"
_QWEN3_CP2_FIXTURE = _FIXTURES / "flagscale_single_node_qwen3_enron_cp2.yaml"
_QWEN3_CP4_FIXTURE = _FIXTURES / "flagscale_single_node_qwen3_enron_cp4.yaml"


def _write_cp_trace(
    trace_root: Path,
    *,
    rank: int,
    context_parallel_size: int = 2,
    data_rank: int = 0,
    wrong_peer: bool = False,
    omit_attention: bool = False,
    num_layers: int = 2,
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
        for _ in range(num_layers):
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
            group_size=context_parallel_size,
            n_buckets=1,
            op="all_reduce",
            operation_id=f"dp:allreduce:{iteration}",
            operation_id_scope="rank_local",
            overlap_enabled=False,
            payload_role="gradient_bucket",
            stage="main_bucket_allreduce",
            timing_phase="collective_call",
        )
        peers = [
            peer for peer in range(context_parallel_size) if peer != rank
        ]
        event("dp-allreduce", "E", group=[rank] if wrong_peer else peers)
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


def _write_qwen3_cp_distopt_trace(
    trace_root: Path,
    *,
    rank: int,
    context_parallel_size: int,
    num_layers: int = 28,
    wrong_group_size: bool = False,
    param_gather_iteration: int = 2,
    omit_sync_iteration: int | None = None,
) -> Path:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    timestamp = 0
    next_operation = 0
    peers = [peer for peer in range(context_parallel_size) if peer != rank]

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
                "dp_rk": 0,
                "pp_rk": 0,
                "tp_rk": 0,
                **attrs,
            }
        )

    def distopt_scope(name: str, *, data_bytes: int) -> None:
        nonlocal next_operation
        next_operation += 1
        operation_kind = (
            "reduce-scatter" if name == "dp-reduce-scatter" else "param-all-gather"
        )
        operation_id = f"dp:{operation_kind}:{next_operation}"
        route = (
            {
                "op": "reduce_scatter",
                "payload_role": "gradient_bucket",
                "stage": "intra_instance_reduce_scatter",
            }
            if name == "dp-reduce-scatter"
            else {
                "op": "all_gather",
                "optimizer_kind": "distributed",
                "payload_role": "parameter_bucket",
                "stage": "distributed_optimizer_param_allgather",
            }
        )
        group_size = context_parallel_size
        if wrong_group_size and next_operation == 1:
            group_size = 1
        event(
            name,
            "B",
            api_async_op=True,
            async_op=True,
            completion_included=False,
            data_bytes=data_bytes,
            group_role="intra_optimizer_instance",
            group_size=group_size,
            n_buckets=1,
            operation_id=operation_id,
            operation_id_scope="rank_local",
            overlap_enabled=True,
            timing_phase="async_dispatch",
            **route,
        )
        event(name, "E", group=peers)

        if name == "dp-reduce-scatter":
            completion_name = "dp-grad-sync-complete"
            completion_fields = {
                "completion_site": "finish_grad_sync",
                "force_all_reduce": False,
                "num_distributed_optimizer_instances": 1,
                "op": "wait",
                "operation_count": 1,
                "operation_ids": [operation_id],
                "operations": [
                    {
                        "event_name": name,
                        "operation_id": operation_id,
                        "stage": route["stage"],
                    }
                ],
                "stage": "gradient_collective_completion",
                "use_distributed_optimizer": True,
            }
        else:
            completion_name = "dp-param-sync-complete"
            completion_fields = {
                "completion_site": "finish_param_sync",
                "op": "wait",
                "operation_count": 1,
                "operation_id": operation_id,
                "operation_ids": [operation_id],
                "stage": "parameter_allgather_completion",
            }
        event(
            completion_name,
            "B",
            completion_guarantee="current_stream_after_wait",
            completion_included=True,
            completion_kind="work_wait",
            host_blocking_guaranteed=False,
            launch_observed=True,
            operation_id_scope="rank_local",
            timing_phase="stream_dependency",
            **completion_fields,
        )
        event(completion_name, "E", completed=True, error_type=None)

    for iteration in (1, 2):
        rows.append(
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        if iteration == param_gather_iteration:
            for payload in (800, 400):
                distopt_scope("dp-param-all-gather", data_bytes=payload)
        event("forward-step", "B")
        event("decoder", "B")
        for _ in range(num_layers):
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
        event("output_layer", "B")
        event("output_layer", "E")
        event("loss", "B")
        event("loss", "E")
        event("decoder-postprocess", "E")
        event("forward-step", "E")
        if iteration != omit_sync_iteration:
            event(
                "grad-sync",
                "B",
                schedule="no-pipelining",
                timing_phase="framework_phase",
            )
            event("all-grads-sync", "B")
        for payload in (1600, 1200, 800):
            distopt_scope("dp-reduce-scatter", data_bytes=payload)
        if iteration != omit_sync_iteration:
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

    path = trace_root / f"benchmark-global-{rank}-data-0-pipeline-0-tensor-0.json"
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


def test_cp4_fixture_is_the_cp4_derivative_of_the_cp2_profile() -> None:
    cp2 = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    cp4 = yaml.safe_load(_CP4_FIXTURE.read_text(encoding="utf-8"))
    cp2["experiment"]["exp_name"] = cp4["experiment"]["exp_name"]
    cp2["experiment"]["runner"]["nproc_per_node"] = 4
    cp2["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    cp2["train"]["system"]["context_parallel_size"] = 4

    assert cp4 == cp2

    runner = cp4["experiment"]["runner"]
    system = cp4["train"]["system"]
    model = cp4["train"]["model"]
    world_size = runner["nproc_per_node"]
    model_parallel_size = (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    assert world_size // model_parallel_size == 1
    assert model["seq_length"] % (2 * system["context_parallel_size"]) == 0


def test_qwen3_cp2_fixture_only_replaces_tp_sp_with_cp() -> None:
    baseline = yaml.safe_load(_QWEN3_TP2_FIXTURE.read_text(encoding="utf-8"))
    cp2 = yaml.safe_load(_QWEN3_CP2_FIXTURE.read_text(encoding="utf-8"))

    baseline["experiment"]["exp_name"] = cp2["experiment"]["exp_name"]
    baseline["train"]["system"]["tensor_model_parallel_size"] = 1
    baseline["train"]["system"]["context_parallel_size"] = 2
    baseline["train"]["system"]["sequence_parallel"] = False
    assert cp2 == baseline

    system = cp2["train"]["system"]
    model = cp2["train"]["model"]
    data = cp2["train"]["data"]
    assert system["use_distributed_optimizer"] is True
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert model["transformer_impl"] == "transformer_engine"
    assert model["attention_backend"] == "flash"
    assert model["num_layers"] == 28
    assert model["seq_length"] % (2 * system["context_parallel_size"]) == 0
    assert data["data_path"] == "${oc.env:MEGALENS_QWEN3_DATA_PATH}"
    assert "mock_data" not in data


def test_qwen3_cp4_fixture_only_scales_context_parallelism() -> None:
    cp2 = yaml.safe_load(_QWEN3_CP2_FIXTURE.read_text(encoding="utf-8"))
    cp4 = yaml.safe_load(_QWEN3_CP4_FIXTURE.read_text(encoding="utf-8"))

    cp4["experiment"]["exp_name"] = cp2["experiment"]["exp_name"]
    cp4["experiment"]["runner"]["nproc_per_node"] = 2
    cp4["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    cp4["train"]["system"]["context_parallel_size"] = 2
    assert cp4 == cp2


def test_qwen3_cp_profiles_require_existing_gpt_and_distopt_events() -> None:
    cp2 = gate.PROFILES["qwen3-enron-cp2"]
    cp4 = gate.PROFILES["qwen3-enron-cp4"]
    event_names = {requirement.name for requirement in cp2.events}

    assert cp2.rank_count == 2
    assert cp4.rank_count == 4
    assert cp4.events == cp2.events
    assert cp2.contract is cp_probe_contract.validate_qwen3_cp2_distopt_coexistence
    assert cp4.contract is cp_probe_contract.validate_qwen3_cp4_distopt_coexistence
    assert (
        cp2.run_contract
        is training_run_contract.validate_two_iteration_qwen3_cp2_checkpoint
    )
    assert (
        cp4.run_contract
        is training_run_contract.validate_two_iteration_qwen3_cp4_checkpoint
    )
    assert {"dp-reduce-scatter", "dp-param-all-gather"} <= event_names
    assert "dp-allreduce" not in event_names
    assert not any(name.startswith(("cp-", "cp_")) for name in event_names)
    assert gate._CONFIG_PROFILES[_QWEN3_CP2_FIXTURE.stem] == "qwen3-enron-cp2"
    assert gate._CONFIG_PROFILES[_QWEN3_CP4_FIXTURE.stem] == "qwen3-enron-cp4"


def test_qwen3_cp2_contract_accepts_qwen3_28_layer_synthetic_trace(
    tmp_path: Path,
) -> None:
    for rank in range(2):
        _write_qwen3_cp_distopt_trace(
            tmp_path, rank=rank, context_parallel_size=2
        )

    report = manifest.validate_trace(
        tmp_path,
        gate.PROFILES["qwen3-enron-cp2"],
        trace_enabled=True,
    )

    assert report.passed
    assert report.ranks == (0, 1)
    assert report.failures == ()


def test_qwen3_cp4_contract_accepts_qwen3_28_layer_synthetic_trace(
    tmp_path: Path,
) -> None:
    for rank in range(4):
        _write_qwen3_cp_distopt_trace(
            tmp_path, rank=rank, context_parallel_size=4
        )

    report = manifest.validate_trace(
        tmp_path,
        gate.PROFILES["qwen3-enron-cp4"],
        trace_enabled=True,
    )

    assert report.passed
    assert report.ranks == (0, 1, 2, 3)
    assert report.failures == ()


def test_qwen3_cp_contract_rejects_the_tiny_model_and_wrong_group(
    tmp_path: Path,
) -> None:
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=0,
        context_parallel_size=2,
        num_layers=2,
        wrong_group_size=True,
    )
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=1,
        context_parallel_size=2,
        num_layers=2,
    )

    failures = cp_probe_contract.validate_qwen3_cp2_distopt_coexistence(tmp_path)
    codes = {failure.code for failure in failures}

    assert "trace.gpt.eager_layers" in codes
    assert "trace.cp.distopt_group" in codes


def test_qwen3_cp_contract_rejects_cross_rank_param_gather_iteration_mismatch(
    tmp_path: Path,
) -> None:
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=0,
        context_parallel_size=2,
        param_gather_iteration=1,
    )
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=1,
        context_parallel_size=2,
        param_gather_iteration=2,
    )

    failures = cp_probe_contract.validate_qwen3_cp2_distopt_coexistence(tmp_path)

    assert "trace.cp.distopt_payload" in {failure.code for failure in failures}


def test_qwen3_cp_contract_requires_each_iteration_sync_hierarchy(
    tmp_path: Path,
) -> None:
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=0,
        context_parallel_size=2,
        omit_sync_iteration=1,
    )
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=1,
        context_parallel_size=2,
    )

    failures = cp_probe_contract.validate_qwen3_cp2_distopt_coexistence(tmp_path)

    assert "trace.cp.distopt_sync" in {failure.code for failure in failures}


def test_qwen3_cp_contract_rejects_reversed_distopt_scope(
    tmp_path: Path,
) -> None:
    path = _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=0,
        context_parallel_size=2,
    )
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=1,
        context_parallel_size=2,
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    positions = [
        position
        for position, row in enumerate(rows)
        if row.get("name") == "dp-reduce-scatter"
    ]
    rows[positions[0]], rows[positions[1]] = rows[positions[1]], rows[positions[0]]
    path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_qwen3_cp2_distopt_coexistence(tmp_path)

    assert "trace.cp.distopt_pairing" in {failure.code for failure in failures}


def test_qwen3_cp_contract_requires_dispatch_end_before_completion(
    tmp_path: Path,
) -> None:
    path = _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=0,
        context_parallel_size=2,
    )
    _write_qwen3_cp_distopt_trace(
        tmp_path,
        rank=1,
        context_parallel_size=2,
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    begin = next(
        position
        for position, row in enumerate(rows)
        if row.get("name") == "dp-reduce-scatter" and row.get("ph") == "B"
    )
    assert [row["name"] for row in rows[begin : begin + 4]] == [
        "dp-reduce-scatter",
        "dp-reduce-scatter",
        "dp-grad-sync-complete",
        "dp-grad-sync-complete",
    ]
    rows[begin : begin + 4] = [
        rows[begin],
        rows[begin + 2],
        rows[begin + 3],
        rows[begin + 1],
    ]
    rows[begin]["n_buckets"] = 0
    path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_qwen3_cp2_distopt_coexistence(tmp_path)
    codes = {failure.code for failure in failures}

    assert "trace.cp.distopt_pairing" in codes
    assert "trace.cp.distopt_payload" in codes


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


def test_cp4_runner_profile_reuses_only_existing_probe_events() -> None:
    cp2 = gate.PROFILES["cp2-te"]
    cp4 = gate.PROFILES["cp4-te"]

    assert cp4.rank_count == 4
    assert cp4.events == cp2.events
    assert cp4.contract is cp_probe_contract.validate_cp4_te_coexistence
    assert (
        cp4.run_contract
        is training_run_contract.validate_two_iteration_transformer_engine_cp4_checkpoint
    )
    assert gate._CONFIG_PROFILES[_CP4_FIXTURE.stem] == "cp4-te"


def test_cp2_contract_accepts_gpt_hierarchy_and_dp_cp_group(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_cp_trace(tmp_path, rank=rank)

    profile = gate.PROFILES["cp2-te"]
    report = manifest.validate_trace(tmp_path, profile, trace_enabled=True)

    assert report.passed
    assert report.ranks == (0, 1)
    assert report.failures == ()


def test_cp4_contract_accepts_gpt_hierarchy_and_dp_cp_group(tmp_path: Path) -> None:
    expected_peers = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
    }
    for rank in range(4):
        path = _write_cp_trace(
            tmp_path, rank=rank, context_parallel_size=4
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        observed_peers = [
            row["group"]
            for row in rows
            if row.get("name") == "dp-allreduce" and row.get("ph") == "E"
        ]
        assert observed_peers == [expected_peers[rank], expected_peers[rank]]

    report = manifest.validate_trace(
        tmp_path,
        gate.PROFILES["cp4-te"],
        trace_enabled=True,
    )

    assert report.passed
    assert report.ranks == (0, 1, 2, 3)
    assert report.failures == ()


def test_cp4_contract_rejects_cp2_group_size(tmp_path: Path) -> None:
    first_path = _write_cp_trace(
        tmp_path, rank=0, context_parallel_size=4
    )
    for rank in (1, 2, 3):
        _write_cp_trace(tmp_path, rank=rank, context_parallel_size=4)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("name") == "dp-allreduce" and row.get("ph") == "B":
            row["group_size"] = 2
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp4_te_coexistence(tmp_path)

    assert "trace.cp.dp_route" in {failure.code for failure in failures}


def test_cp2_contract_rejects_a_non_cp_peer_group(tmp_path: Path) -> None:
    _write_cp_trace(tmp_path, rank=0, wrong_peer=True)
    _write_cp_trace(tmp_path, rank=1)

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.dp_cp_group" in {failure.code for failure in failures}


def test_cp2_contract_requires_exactly_one_nested_dp_allreduce(tmp_path: Path) -> None:
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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


def test_cp_contract_rejects_speculative_cp_events_and_fields(tmp_path: Path) -> None:
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
    rows = json.loads(first_path.read_text(encoding="utf-8"))
    forward_begin = next(
        row
        for row in rows
        if row.get("name") == "forward-step" and row.get("ph") == "B"
    )
    forward_begin["cp_rk"] = 0
    insertion = rows.index(forward_begin) + 1
    rows[insertion:insertion] = [
        {**forward_begin, "name": "cp-kv-exchange"},
        {**forward_begin, "name": "cp-kv-exchange", "ph": "E"},
    ]
    first_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)
    codes = {failure.code for failure in failures}

    assert "trace.cp.unexpected_event" in codes
    assert "trace.cp.unexpected_field" in codes


def test_cp2_contract_requires_peer_only_end_fields(tmp_path: Path) -> None:
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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
    _write_cp_trace(tmp_path, rank=0, data_rank=1)
    _write_cp_trace(tmp_path, rank=1)

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.cp.coordinates" in {failure.code for failure in failures}


def test_cp2_contract_requires_the_two_layer_gpt_hierarchy(tmp_path: Path) -> None:
    _write_cp_trace(tmp_path, rank=0, omit_attention=True)
    _write_cp_trace(tmp_path, rank=1)

    failures = cp_probe_contract.validate_cp2_te_coexistence(tmp_path)

    assert "trace.gpt.eager_layers" in {failure.code for failure in failures}


def test_cp2_profile_requires_rank_fields_on_existing_events(tmp_path: Path) -> None:
    first_path = _write_cp_trace(tmp_path, rank=0)
    _write_cp_trace(tmp_path, rank=1)
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


def test_cp4_training_contract_requires_the_launcher_cp_size(tmp_path: Path) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  context_parallel_size ............................ 2\n",
        encoding="utf-8",
    )

    failures = (
        training_run_contract.validate_two_iteration_transformer_engine_cp4_checkpoint(
            tmp_path, False
        )
    )
    assert {failure.code for failure in failures} == {
        "run.training.context_parallel_size"
    }

    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  context_parallel_size ............................ 4\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_transformer_engine_cp4_checkpoint(
            tmp_path, False
        )
        == ()
    )
