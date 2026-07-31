# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.test_utils.runners import dp_probe_contract
from tests.test_utils.runners import gpt_probe_contract
from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import p2p_probe_contract
from tests.test_utils.runners import run_flagscale_megalens as gate
from tests.test_utils.runners import tp_probe_contract

_FIXTURES = Path(__file__).parent / "fixtures"
_CONFIG_PROFILE_CASES = {
    "flagscale_single_node_smoke.yaml": "pp1",
    "flagscale_single_node_gpt_eager_full_smoke.yaml": "gpt-eager-full",
    "flagscale_single_node_tp2_sp_local_smoke.yaml": "tp2-sp-local",
    "flagscale_single_node_pp2_smoke.yaml": "pp2",
    "flagscale_single_node_pp2_unbatched_smoke.yaml": "pp2-unbatched",
    "flagscale_single_node_ep2_smoke.yaml": "ep2-alltoall",
    "flagscale_single_node_ep2_fine_grained_smoke.yaml": "ep2-fine-grained",
    "flagscale_single_node_dp2_standard_smoke.yaml": "dp2-standard-ddp",
    "flagscale_single_node_dp2_standard_overlap_smoke.yaml": (
        "dp2-standard-ddp-overlap"
    ),
    "flagscale_single_node_dp2_distopt_smoke.yaml": "dp2-distopt",
    "flagscale_single_node_dp2_distopt_overlap_smoke.yaml": (
        "dp2-distopt-overlap"
    ),
    "flagscale_single_node_dp2_layerwise_overlap_smoke.yaml": (
        "dp2-layerwise-overlap"
    ),
    "flagscale_single_node_dp4_distopt_multi_instance_overlap_smoke.yaml": (
        "dp4-distopt-multi-instance-overlap"
    ),
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
    include_optimizer: bool = False,
    include_optimizer_postprocess: bool = True,
    p2p_route: str | None = None,
) -> None:
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
                "dp_rk": 0,
                "pp_rk": pipeline_rank,
                "tp_rk": 0,
                **attrs,
            }
        )

    def p2p_events(iteration: int) -> None:
        if p2p_route is None:
            return
        if p2p_route not in {"batch", "unbatched"}:
            raise ValueError(f"unknown P2P route: {p2p_route}")

        transport_api = (
            "batch_isend_irecv" if p2p_route == "batch" else "isend_irecv"
        )
        launch_pairing = "backend_dependent" if p2p_route == "batch" else "key"
        completion_pairing = "position" if p2p_route == "batch" else "key"
        if p2p_route == "batch":
            launch_groups = (
                ("internal_wait", (("send-forward", "send_next"),)),
                ("internal_wait", (("recv-backward", "recv_next"),)),
            )
            if rank == 1:
                launch_groups = (
                    ("internal_wait", (("recv-forward", "recv_prev"),)),
                    ("internal_wait", (("send-backward", "send_prev"),)),
                )
        else:
            backward_mode = "internal_wait" if rank == 0 else "external_wait"
            launch_groups = (
                (
                    "internal_wait",
                    (
                        ("recv-forward", "recv_prev"),
                        ("send-forward", "send_next"),
                    ),
                ),
                (
                    backward_mode,
                    (
                        ("send-backward", "send_prev"),
                        ("recv-backward", "recv_next"),
                    ),
                ),
            )

        for index, (completion_mode, directional_events) in enumerate(launch_groups):
            batch_id = f"p2p:{iteration}:{index}"
            operations = []
            for event_name, key in directional_events:
                direction, pipeline_direction = event_name.split("-", 1)
                operation_id = f"{batch_id}:{key}"
                operations.append(
                    {
                        "operation_id": operation_id,
                        "request_id": operation_id,
                        "direction": direction,
                        "pipeline_direction": pipeline_direction,
                        "peer_rank": 1 - rank,
                        "data_bytes": 32768,
                        "microbatch": None,
                        "comm_type": "p2p",
                        "backend": "nccl",
                        "transport_api": transport_api,
                        "completion_mode": completion_mode,
                    }
                )
            event(
                "p2p-launch",
                "B",
                batch_id=batch_id,
                comm_type="p2p-launch",
                backend="nccl",
                backends=["nccl"],
                backend_complete=True,
                transport_api=transport_api,
                request_pairing=launch_pairing,
                completion_mode=completion_mode,
                completion_included=False,
                operation_count=len(operations),
                operations=operations,
            )
            event("p2p-launch", "E")
            completion_site = (
                "communicate_internal_wait"
                if completion_mode == "internal_wait"
                else "exposed_request_wait"
            )
            for operation in operations:
                event_name = (
                    f"{operation['direction']}-{operation['pipeline_direction']}"
                )
                operation_id = operation["operation_id"]
                event(
                    event_name,
                    "B",
                    **operation,
                    batch_id=batch_id,
                    request_pairing=completion_pairing,
                    completion_site=completion_site,
                    completion_included=True,
                    completion_kind="work_wait",
                    operation_count=1,
                    operation_ids=[operation_id],
                )
                event(event_name, "E", completed=True, error_type=None)
            if p2p_route == "batch":
                operation_ids = [
                    operation["operation_id"] for operation in operations
                ]
                event(
                    "p2p-batch-device-sync",
                    "B",
                    batch_id=batch_id,
                    comm_type="p2p",
                    backend="nccl",
                    backends=["nccl"],
                    backend_complete=True,
                    transport_api=transport_api,
                    request_pairing="position",
                    completion_site="batch_p2p_sync_workaround",
                    completion_included=True,
                    completion_kind="device_synchronize",
                    operation_count=len(operations),
                    operation_ids=operation_ids,
                    operations=operations,
                    physical_request_count=len(operations),
                )
                event(
                    "p2p-batch-device-sync",
                    "E",
                    completed=True,
                    device_completion_guaranteed=True,
                    error_type=None,
                    host_blocking_guaranteed=True,
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
        p2p_events(iteration)
        if include_optimizer:
            event("optimizer", "B")
            event("optimizer-step", "B")
            event("optimizer-step", "E")
            event("optimizer", "E")
            if include_optimizer_postprocess:
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


def test_unbatched_pp2_profile_uses_vpp_without_unsupported_p2p_cli_keys() -> None:
    config = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_pp2_unbatched_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    system = config["train"]["system"]
    model = config["train"]["model"]

    assert system["pipeline_model_parallel_size"] == 2
    assert system["num_layers_per_virtual_pipeline_stage"] == 1
    assert system["microbatch_group_size_per_virtual_pipeline_stage"] == 2
    assert {
        "batch_p2p_comm",
        "batch_p2p_sync",
        "no_overlap_p2p_communication",
    }.isdisjoint(system)
    assert model["num_layers"] == 4
    assert model["micro_batch_size"] == 1
    assert model["global_batch_size"] == 2


@pytest.mark.parametrize(
    ("baseline_name", "overlap_name", "profile_name", "changes", "contract"),
    (
        (
            "flagscale_single_node_dp2_standard_smoke.yaml",
            "flagscale_single_node_dp2_standard_overlap_smoke.yaml",
            "dp2-standard-ddp-overlap",
            {"overlap_grad_reduce": True},
            dp_probe_contract.validate_dp_standard_overlap,
        ),
        (
            "flagscale_single_node_dp2_distopt_smoke.yaml",
            "flagscale_single_node_dp2_distopt_overlap_smoke.yaml",
            "dp2-distopt-overlap",
            {"overlap_grad_reduce": True, "overlap_param_gather": True},
            dp_probe_contract.validate_dp_distopt_overlap,
        ),
    ),
)
def test_dp2_overlap_profiles_only_enable_the_selected_overlap_route(
    baseline_name: str,
    overlap_name: str,
    profile_name: str,
    changes: dict[str, bool],
    contract,
) -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / baseline_name).read_text(encoding="utf-8")
    )
    overlap = yaml.safe_load(
        (_FIXTURES / overlap_name).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = overlap["experiment"]["exp_name"]
    baseline["train"]["system"].update(changes)

    assert overlap == baseline
    assert gate.PROFILES[profile_name].contract is contract


def test_dp4_multi_instance_profile_only_expands_the_dp_overlap_topology() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_dp2_distopt_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    multi_instance = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_dp4_distopt_multi_instance_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = multi_instance["experiment"]["exp_name"]
    baseline["experiment"]["runner"]["nproc_per_node"] = 4
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    baseline["train"]["system"]["num_distributed_optimizer_instances"] = 2
    baseline["train"]["model"]["global_batch_size"] = 4

    assert multi_instance == baseline
    assert (
        gate.PROFILES["dp4-distopt-multi-instance-overlap"].contract
        is dp_probe_contract.validate_dp_multi_instance_distopt_overlap
    )

    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            "unknown.yaml",
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
            "--topology",
            "dp4",
        )
    )
    assert gate._profile_from_arguments(args).name == (
        "dp4-distopt-multi-instance-overlap"
    )


def test_dp2_layerwise_profile_only_selects_dist_muon_parameter_overlap() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_dp2_standard_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    layerwise = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_dp2_layerwise_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = layerwise["experiment"]["exp_name"]
    baseline["train"]["system"]["overlap_param_gather"] = True
    baseline["train"]["model"]["optimizer"]["optimizer"] = "dist_muon"

    assert layerwise == baseline
    system = layerwise["train"]["system"]
    optimizer = layerwise["train"]["model"]["optimizer"]
    assert system["use_distributed_optimizer"] is False
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert optimizer["optimizer"] == "dist_muon"
    assert "muon_tp_mode" not in optimizer
    assert "use_padded_layerwise_optimizer" not in system

    profile = gate.PROFILES["dp2-layerwise-overlap"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "dp-allreduce",
        "dp-param-all-gather",
        "dp-grad-sync-complete",
        "dp-param-sync-complete",
    }
    assert profile.contract is dp_probe_contract.validate_dp_layerwise_overlap


def test_tp2_sp_profile_only_selects_the_local_tp_routes() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_gpt_eager_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    tp2_sp = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_local_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = tp2_sp["experiment"]["exp_name"]
    baseline["experiment"]["runner"]["nproc_per_node"] = 2
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    baseline["train"]["system"]["tensor_model_parallel_size"] = 2
    baseline["train"]["system"]["sequence_parallel"] = True
    model = baseline["train"]["model"]
    model["no_gradient_accumulation_fusion"] = True
    model["group_query_attention"] = True
    model["num_query_groups"] = 1
    model["normalization"] = "LayerNorm"

    assert tp2_sp == baseline
    model = tp2_sp["train"]["model"]
    system = tp2_sp["train"]["system"]
    assert model["transformer_impl"] == "local"
    assert model["global_batch_size"] == 1
    assert model["untie_embeddings_and_output_weights"] is True
    assert system["pipeline_model_parallel_size"] == 1
    assert system["context_parallel_size"] == 1

    profile = gate.PROFILES["tp2-sp-local"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "tp-all-gather-first",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    }
    assert profile.contract is tp_probe_contract.validate_tp2_sp_profile

    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            "unknown.yaml",
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
            "--topology",
            "tp2",
        )
    )
    assert gate._profile_from_arguments(args).name == "tp2-sp-local"


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
        include_optimizer=True,
    )
    _write_gpt_phase_trace(
        pp2_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        include_optimizer=True,
    )
    assert gpt_probe_contract.validate_gpt_pp2_training_phases(pp2_root) == ()

    invalid_root = tmp_path / "invalid-pp2"
    _write_gpt_phase_trace(
        invalid_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        include_optimizer=True,
    )
    _write_gpt_phase_trace(
        invalid_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        include_optimizer=True,
    )
    failures = gpt_probe_contract.validate_gpt_pp2_training_phases(invalid_root)
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


def test_gpt_pp2_profile_rejects_missing_optimizer_postprocess(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "missing-optimizer-postprocess"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        include_optimizer=True,
        include_optimizer_postprocess=False,
    )
    _write_gpt_phase_trace(
        trace_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        include_optimizer=True,
        include_optimizer_postprocess=False,
    )

    failures = gpt_probe_contract.validate_gpt_pp2_training_phases(trace_root)

    assert {failure.code for failure in failures} == {"trace.optimizer.sequence"}
    assert len(failures) == 4


def test_pp2_batched_profile_accepts_internal_wait_route(tmp_path: Path) -> None:
    trace_root = tmp_path / "pp2-batched"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            include_optimizer=True,
            p2p_route="batch",
        )

    assert gate.PROFILES["pp2"].contract(trace_root) == ()


def test_pp2_unbatched_profile_accepts_mixed_wait_route(tmp_path: Path) -> None:
    trace_root = tmp_path / "pp2-unbatched"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        p2p_route="unbatched",
    )
    _write_gpt_phase_trace(
        trace_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        p2p_route="unbatched",
    )

    assert gate.PROFILES["pp2-unbatched"].contract(trace_root) == ()


def test_pp2_unbatched_contract_rejects_batched_route(tmp_path: Path) -> None:
    trace_root = tmp_path / "wrong-pp2-route"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        p2p_route="batch",
    )
    _write_gpt_phase_trace(
        trace_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        p2p_route="batch",
    )

    failures = p2p_probe_contract.validate_pp2_unbatched_route(trace_root)

    codes = {failure.code for failure in failures}
    assert "trace.p2p.field" in codes
    assert "trace.p2p.sync_count" in codes
    assert "trace.p2p.external_wait" in codes


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
