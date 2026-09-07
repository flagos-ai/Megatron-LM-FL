# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_utils.runners import check_ring_exchange, gpt_probe_contract
from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import run_flagscale_megalens as gate
from tests.test_utils.runners import tp_probe_contract, training_run_contract

_FIXTURES = Path(__file__).parent / "fixtures"

_CONFIG_PROFILE_CASES = {
    'flagscale_single_node_smoke.yaml': 'pp1',
    'flagscale_single_node_gpt_eager_continuous_cupti_smoke.yaml': 'gpt-eager-continuous-cupti',
    'flagscale_single_node_cp2_te_smoke.yaml': 'cp2-te',
    'flagscale_single_node_tp2_sp_local_smoke.yaml': 'tp2-sp-local',
    'flagscale_single_node_tp2_pp4_multimicrobatch_smoke.yaml': 'tp2-pp4-multimicrobatch',
    'flagscale_single_node_pp2_smoke.yaml': 'pp2',
    'flagscale_single_node_ep2_smoke.yaml': 'ep2-alltoall',
    'flagscale_single_node_dp2_standard_overlap_smoke.yaml': 'dp2-standard-ddp-overlap',
    'flagscale_single_node_dp2_distopt_overlap_smoke.yaml': 'dp2-distopt-overlap',
    'flagscale_single_node_te_cuda_graph_full_cupti_smoke.yaml': 'te-full-cuda-kernels',
}


def _write_rank_trace(run_dir: Path, rank: int, event: str = "forward") -> None:
    trace_root = run_dir / "traces"
    trace_root.mkdir(exist_ok=True)
    fields = {"iteration": 2, "g_rk": rank, "dp_rk": rank, "pp_rk": 0, "tp_rk": 0}
    path = trace_root / f"benchmark-global-{rank}-data-{rank}-pipeline-0-tensor-0.json"
    path.write_text(
        json.dumps([{"name": event, "ph": "B", **fields}, {"name": event, "ph": "E", **fields}]),
        encoding="utf-8",
    )


def _write_terminal_checkpoint(run_dir: Path, *, iteration: int = 2) -> None:
    checkpoint_root = run_dir / "checkpoints"
    iteration_root = checkpoint_root / f"iter_{iteration:07d}"
    iteration_root.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        str(iteration), encoding="utf-8"
    )
    (iteration_root / "common.pt").write_bytes(b"checkpoint")


def _write_gpt_phase_trace(
    trace_root: Path,
    *,
    rank: int,
    pipeline_rank: int,
    include_postprocess: bool,
    tensor_rank: int = 0,
    eager_layers: int = 0,
    external_mlp_layers: int = 0,
    recompute_mlp_layers: int = 0,
    include_optimizer: bool = False,
    include_backward: bool = False,
    eager_iterations: frozenset[int] | None = None,
    model_iterations: frozenset[int] | None = None,
    iteration_ids: tuple[int, ...] = (1, 2),
    microbatches: int = 1,
    include_schedule_finalize: bool = False,
    include_optimizer_postprocess: bool = True,
    p2p_route: str | None = None,
    kernel_iterations: frozenset[int] | None = None,
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
                "tp_rk": tensor_rank,
                **attrs,
            }
        )

    def p2p_events(iteration: int) -> None:
        if p2p_route is None:
            return
        if p2p_route != "batch":
            raise ValueError(f"unknown P2P route: {p2p_route}")
        directions = (
            (("send-forward", "send_next"), ("recv-backward", "recv_next"))
            if rank == 0
            else (("recv-forward", "recv_prev"), ("send-backward", "send_prev"))
        )
        for index, (name, key) in enumerate(directions):
            batch_id = f"p2p:{iteration}:{index}"
            operation_id = f"{batch_id}:{key}"
            direction, pipeline_direction = name.split("-", 1)
            operation = {
                "operation_id": operation_id,
                "request_id": operation_id,
                "direction": direction,
                "pipeline_direction": pipeline_direction,
                "peer_rank": 1 - rank,
                "data_bytes": 32768,
                "microbatch": None,
                "comm_type": "p2p",
                "backend": "nccl",
                "transport_api": "batch_isend_irecv",
                "completion_mode": "internal_wait",
            }
            event(
                "p2p-launch",
                "B",
                batch_id=batch_id,
                comm_type="p2p-launch",
                backend="nccl",
                backends=["nccl"],
                backend_complete=True,
                transport_api="batch_isend_irecv",
                request_pairing="backend_dependent",
                completion_mode="internal_wait",
                completion_included=False,
                operation_count=1,
                operations=[operation],
            )
            event("p2p-launch", "E")
            event(
                name,
                "B",
                **operation,
                batch_id=batch_id,
                request_pairing="position",
                completion_site="communicate_internal_wait",
                completion_included=True,
                completion_kind="work_wait",
                operation_count=1,
                operation_ids=[operation_id],
            )
            event(name, "E", completed=True, error_type=None)
            event(
                "p2p-batch-device-sync",
                "B",
                batch_id=batch_id,
                comm_type="p2p",
                backend="nccl",
                backends=["nccl"],
                backend_complete=True,
                transport_api="batch_isend_irecv",
                request_pairing="position",
                completion_site="batch_p2p_sync_workaround",
                completion_included=True,
                completion_kind="device_synchronize",
                operation_count=1,
                operation_ids=[operation_id],
                operations=[operation],
                physical_request_count=1,
            )
            event(
                "p2p-batch-device-sync",
                "E",
                completed=True,
                device_completion_guaranteed=True,
                error_type=None,
                host_blocking_guaranteed=True,
            )

    for iteration in iteration_ids:
        rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration})
        if model_iterations is None or iteration in model_iterations:
            layer_count = (
                eager_layers if eager_iterations is None or iteration in eager_iterations else 0
            )
            for microbatch in range(microbatches):
                event("forward-step", "B", current_microbatch=microbatch)
                event("decoder", "B")
                for _ in range(layer_count):
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
                for _ in range(external_mlp_layers):
                    event("MLP.forward", "B")
                    event("MLP.forward", "E")
                event("decoder", "E")
                event("decoder-postprocess", "B")
                if include_postprocess:
                    event("output_layer", "B")
                    event("output_layer", "E")
                    event("loss", "B")
                    event("loss", "E")
                event("decoder-postprocess", "E")
                if include_schedule_finalize:
                    event("forward-step-calc-loss", "B", current_microbatch=microbatch)
                    event("forward-step-calc-loss", "E")
                event("forward-step", "E")
                if include_backward:
                    event("backward-step", "B", current_microbatch=microbatch)
                    for _ in range(recompute_mlp_layers):
                        event("MLP.forward", "B")
                        event("MLP.forward", "E")
                    event("backward-step", "E")
            if include_schedule_finalize and include_backward:
                event("grad-sync", "B")
                event("all-grads-sync", "B")
                event("all-grads-sync", "E")
                event("grad-sync", "E")
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
            {"name": "iteration", "ph": "E", "iteration": iteration, "duration_wall": timestamp}
        )

    for iteration in sorted(kernel_iterations or ()):
        start_us = iteration * 100
        end_us = start_us + 20
        iter_rel_start_us = 50
        iter_rel_end_us = iter_rel_start_us + 20
        rows.append(
            {
                "record_type": "cuda_kernel",
                "name": "transformer_engine::model_gemm",
                "ph": "X",
                "start_us": start_us,
                "end_us": end_us,
                "wall_start_us": 10_000 + start_us,
                "wall_end_us": 10_000 + end_us,
                "iter_rel_start_us": iter_rel_start_us,
                "iter_rel_end_us": iter_rel_end_us,
                "duration_us": end_us - start_us,
                "device": rank,
                "iteration": iteration,
                "g_rk": rank,
                "dp_rk": 0,
                "pp_rk": pipeline_rank,
                "tp_rk": tensor_rank,
            }
        )

    path = (
        trace_root / f"benchmark-global-{rank}-data-0-pipeline-{pipeline_rank}-"
        f"tensor-{tensor_rank}.json"
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
    write_checkpoint: bool = True,
) -> tuple[int, Path, tuple[str, ...]]:
    run_dir = tmp_path / "run"
    observed_command: tuple[str, ...] = ()

    def fake_run_foreground(
        argv: tuple[str, ...], *, env: dict[str, str], launcher_log: Path, **_: object
    ) -> gate.ExecutionResult:
        nonlocal observed_command
        observed_command = tuple(argv)
        launcher_log.write_text("official FlagScale entrypoint\n", encoding="utf-8")
        if child_returncode == 0:
            run_root = Path(env["MEGALENS_GATE_RUN_DIR"])
            if write_checkpoint:
                _write_terminal_checkpoint(run_root)
            if mode == "trace-on":
                _write_rank_trace(run_root, 0)
        return gate.ExecutionResult(child_returncode)

    monkeypatch.setattr(gate, "run_foreground", fake_run_foreground)
    monkeypatch.setattr(gate, "_source_head", lambda _source_root: "a" * 40)
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


@pytest.mark.parametrize(("config", "profile"), tuple(_CONFIG_PROFILE_CASES.items()))
def test_existing_yaml_profiles_remain_selectable(config: str, profile: str) -> None:
    configuration = yaml.safe_load((_FIXTURES / config).read_text(encoding="utf-8"))
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
    assert (
        configuration["experiment"]["runner"]["nproc_per_node"] == gate.PROFILES[profile].rank_count
    )
    assert configuration["train"]["model"]["train_iters"] == 2
    assert configuration["train"]["data"]["mock_data"] is True


@pytest.mark.parametrize("config", ("flagscale_single_node_mimo_smoke.yaml", "custom.yaml"))
def test_unknown_config_requires_an_explicit_profile_before_launch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], config: str
) -> None:
    config_path = tmp_path / config
    config_path.write_text("train: {}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        gate.main(
            (
                "--run-dir",
                str(run_dir),
                "--input-config",
                str(config_path),
                "--mode",
                "trace-on",
                "--image",
                "example/flagscale:dev",
            )
        )
    assert error.value.code == 2
    assert "select a supported --profile explicitly" in capsys.readouterr().err
    assert not run_dir.exists()


def test_custom_config_can_select_a_supported_profile() -> None:
    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            "custom.yaml",
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
            "--profile",
            "tp2-sp-local",
        )
    )
    assert gate._profile_from_arguments(args).name == "tp2-sp-local"


def test_ep2_fixture_supports_allgather_dispatch() -> None:
    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            str(_FIXTURES / "flagscale_single_node_ep2_smoke.yaml"),
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
            "--ep-dispatcher",
            "allgather",
        )
    )
    profile = gate._profile_from_arguments(args)
    assert profile.name == "ep2-allgather"
    assert gate._ep_dispatcher(profile, args.ep_dispatcher) == "allgather"


def test_training_contract_requires_the_terminal_checkpoint(tmp_path: Path) -> None:
    failures = training_run_contract.validate_two_iteration_checkpoint(tmp_path, False)

    assert {failure.code for failure in failures} == {
        "run.training.tracker",
        "run.training.checkpoint",
    }

    _write_terminal_checkpoint(tmp_path)

    assert training_run_contract.validate_two_iteration_checkpoint(tmp_path, False) == ()


def test_continuous_cupti_training_contract_requires_one_window_extraction(tmp_path: Path) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    arguments = "\n".join(
        (
            "[default0]:  trace_interval ...................... 2",
            "[default0]:  continuous_trace_iterations ........ 2",
            "[default0]:  trace_cupti_kernels ................. on",
        )
    )
    launcher_log.write_text(arguments, encoding="utf-8")
    contract = training_run_contract.validate_two_iteration_continuous_cuda_kernel_checkpoint
    assert contract(tmp_path, False) == ()

    launcher_log.write_text(
        arguments + "\n[default0]:[trace] extracted 200 cuda kernel events at iter 2\n",
        encoding="utf-8",
    )
    assert contract(tmp_path, True) == ()

    launcher_log.write_text(
        arguments + "\n[default0]:[trace] extracted 200 cuda kernel events at iter 1\n",
        encoding="utf-8",
    )
    failures = contract(tmp_path, True)
    assert [failure.code for failure in failures] == ["run.training.cuda_kernel_capture"]


def test_transformer_engine_training_contract_requires_the_parsed_model_route(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ local\n", encoding="utf-8"
    )

    failures = training_run_contract.validate_two_iteration_transformer_engine_checkpoint(
        tmp_path, False
    )
    assert {failure.code for failure in failures} == {"run.training.transformer_impl"}

    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ " "transformer_engine\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_transformer_engine_checkpoint(tmp_path, False)
        == ()
    )


def test_te_full_cuda_graph_training_contract_requires_capture_and_replay(tmp_path: Path) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    valid_log = "\n".join(
        (
            "[default0]:  transformer_impl ................ transformer_engine",
            "[default0]:  cuda_graph_impl ................. transformer_engine",
            "[default0]:  cuda_graph_scope ................ []",
            "[default0]:  cuda_graph_warmup_steps ......... 1",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:" "Rank 0: 2 graphable layers.",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:" "Start CUDA Graphs capture...",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Time spent in CUDA Graphs capture on rank 0: 1.25s",
            "[default0]: iteration 1/ 2 | lm loss: 1.0 |",
            "[default0]: iteration 2/ 2 | lm loss: 0.9 |",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Rank 0: 2 graphs deleted with explicit reset, "
            "0 graphs deleted without explicit reset.",
        )
    )
    launcher_log.write_text(valid_log, encoding="utf-8")

    assert (
        training_run_contract.validate_two_iteration_te_full_cuda_graph_checkpoint(tmp_path, True)
        == ()
    )

    kernel_log = "\n".join(
        (
            valid_log,
            "[default0]:  trace_cupti_kernels ............. on",
            "[default0]:[trace] extracted 120 cuda kernel events at iter 1",
            "[default0]:[trace] extracted 80 cuda kernel events at iter 2",
        )
    )
    launcher_log.write_text(kernel_log, encoding="utf-8")
    assert (
        training_run_contract.validate_two_iteration_te_full_cuda_graph_kernel_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        kernel_log.replace("[default0]:[trace] extracted 80 cuda kernel events at iter 2", ""),
        encoding="utf-8",
    )
    failures = training_run_contract.validate_two_iteration_te_full_cuda_graph_kernel_checkpoint(
        tmp_path, True
    )
    assert [failure.code for failure in failures] == ["run.training.cuda_kernel_capture"]

    launcher_log.write_text(
        valid_log.replace(
            "cuda_graph_scope ................ []",
            "cuda_graph_scope ................ [<CudaGraphScope.attn: 2>]",
        ),
        encoding="utf-8",
    )
    failures = training_run_contract.validate_two_iteration_te_full_cuda_graph_checkpoint(
        tmp_path, True
    )
    assert [failure.code for failure in failures] == ["run.training.cuda_graph_scope"]


def test_raw_framework_event_requirements_use_the_enclosing_iteration() -> None:
    required_fields = {
        field
        for profile in gate.PROFILES.values()
        for requirement in profile.events
        for field in requirement.fields
    }

    assert "iteration" not in required_fields
    assert {"g_rk", "dp_rk", "pp_rk", "tp_rk"} <= required_fields


def test_tp2_pp4_multimicrobatch_profile_selects_the_focused_contract() -> None:
    config = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_tp2_pp4_multimicrobatch_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment"]["runner"]["nproc_per_node"] == 8
    assert config["train"]["system"]["tensor_model_parallel_size"] == 2
    assert config["train"]["system"]["pipeline_model_parallel_size"] == 4
    assert config["train"]["model"]["micro_batch_size"] == 1
    assert config["train"]["model"]["global_batch_size"] == 4
    assert config["train"]["model"]["train_iters"] == 2

    profile = gate.PROFILES["tp2-pp4-multimicrobatch"]

    assert profile.rank_count == 8
    assert profile.contract is tp_probe_contract.validate_tp2_pp4_multimicrobatch
    assert profile.run_contract is training_run_contract.validate_two_iteration_checkpoint
    assert (
        gate._CONFIG_PROFILES["flagscale_single_node_tp2_pp4_multimicrobatch_smoke"]
        == "tp2-pp4-multimicrobatch"
    )
    assert {requirement.name for requirement in profile.events} == {
        "forward-step",
        "backward-step",
        "p2p-launch",
        "send-forward",
        "recv-forward",
        "send-backward",
        "recv-backward",
        "optimizer",
        "optimizer-step",
        "optimizer-postprocess",
        "sp-layernorm-allreduce",
    }


def test_gpt_pp1_and_pp2_profiles_enforce_stage_specific_model_phases(tmp_path: Path) -> None:
    pp1_root = tmp_path / "pp1"
    _write_gpt_phase_trace(
        pp1_root, rank=0, pipeline_rank=0, include_postprocess=True, eager_layers=2
    )
    assert gpt_probe_contract.validate_gpt_pp1_eager_phases(pp1_root) == ()

    pp2_root = tmp_path / "pp2"
    _write_gpt_phase_trace(
        pp2_root, rank=0, pipeline_rank=0, include_postprocess=False, include_optimizer=True
    )
    _write_gpt_phase_trace(
        pp2_root, rank=1, pipeline_rank=1, include_postprocess=True, include_optimizer=True
    )
    assert gpt_probe_contract.validate_gpt_pp2_training_phases(pp2_root) == ()

    invalid_root = tmp_path / "invalid-pp2"
    _write_gpt_phase_trace(
        invalid_root, rank=0, pipeline_rank=0, include_postprocess=True, include_optimizer=True
    )
    _write_gpt_phase_trace(
        invalid_root, rank=1, pipeline_rank=1, include_postprocess=True, include_optimizer=True
    )
    failures = gpt_probe_contract.validate_gpt_pp2_training_phases(invalid_root)
    assert failures
    assert {failure.code for failure in failures} == {"trace.gpt.count"}


def test_gpt_eager_profile_rejects_an_incomplete_layer_sequence(tmp_path: Path) -> None:
    trace_root = tmp_path / "incomplete-eager"
    _write_gpt_phase_trace(
        trace_root, rank=0, pipeline_rank=0, include_postprocess=True, eager_layers=1
    )

    failures = gpt_probe_contract.validate_gpt_pp1_eager_phases(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.eager_layers",
        "trace.gpt.eager_layers",
    ]
    assert [failure.evidence for failure in failures] == [
        "rank=0 iteration=1",
        "rank=0 iteration=2",
    ]


def test_gpt_eager_continuous_cupti_profile_uses_one_profiler_window(tmp_path: Path) -> None:
    trace_root = tmp_path / "continuous-cupti"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        kernel_iterations=frozenset((1, 2)),
    )
    profile = gate.PROFILES["gpt-eager-continuous-cupti"]
    assert profile.contract(trace_root) == ()

    trace_path = next(trace_root.glob("*.json"))
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("record_type") == "cuda_kernel" and row.get("iteration") == 2:
            row["wall_start_us"] += 100
            row["wall_end_us"] += 100
    trace_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = profile.contract(trace_root)
    assert [failure.code for failure in failures] == ["trace.gpt.continuous_kernel_window"]


def test_te_full_cuda_graph_profile_separates_eager_and_replay_phases(tmp_path: Path) -> None:
    trace_root = tmp_path / "te-full-cuda-graph"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
    )

    assert gpt_probe_contract.validate_te_full_cuda_graph_phases(trace_root) == ()


def test_te_full_cuda_graph_profile_rejects_inner_replay_events(tmp_path: Path) -> None:
    trace_root = tmp_path / "te-full-cuda-graph-inner-replay"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1, 2)),
    )

    failures = gpt_probe_contract.validate_te_full_cuda_graph_phases(trace_root)

    assert [failure.code for failure in failures] == ["trace.gpt.cuda_graph_replay_inner"]
    assert failures[0].evidence == "rank=0 iteration=2"


@pytest.mark.parametrize("profile_name", ("te-full-cuda-kernels",))
def test_cuda_graph_kernel_profiles_capture_eager_and_replay_device_work(
    tmp_path: Path, profile_name: str
) -> None:
    trace_root = tmp_path / profile_name
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        kernel_iterations=frozenset((1, 2)),
    )

    assert gate.PROFILES[profile_name].contract(trace_root) == ()


@pytest.mark.parametrize("profile_name", ("te-full-cuda-kernels",))
def test_cuda_graph_kernel_profiles_require_replay_device_work(
    tmp_path: Path, profile_name: str
) -> None:
    trace_root = tmp_path / f"{profile_name}-missing-replay-kernels"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        kernel_iterations=frozenset((1,)),
    )

    failures = gate.PROFILES[profile_name].contract(trace_root)

    assert [failure.code for failure in failures] == ["trace.gpt.cuda_graph_kernel_capture"]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_gpt_pp2_profile_rejects_missing_optimizer_postprocess(tmp_path: Path) -> None:
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


def test_ring_exchange_preflight_reports_selected_torch_build() -> None:
    fake_torch = SimpleNamespace(
        __file__=__file__,
        __version__="2.9.0+cu128",
        distributed=SimpleNamespace(ring_exchange=lambda **kwargs: None),
    )

    available, detail = check_ring_exchange.describe_capability(fake_torch)

    assert available is True
    assert detail == (
        f"ring_exchange=available torch=2.9.0+cu128 " f"torch_path={Path(__file__).resolve()}"
    )


def test_ring_exchange_preflight_rejects_a_standard_torch_build() -> None:
    fake_torch = SimpleNamespace(
        __file__=__file__, __version__="2.8.0+cpu", distributed=SimpleNamespace()
    )

    available, detail = check_ring_exchange.describe_capability(fake_torch)

    assert available is False
    assert detail == (
        f"ring_exchange=unavailable torch=2.8.0+cpu " f"torch_path={Path(__file__).resolve()}"
    )


def test_runner_uses_requested_image_current_source_and_flagscale_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returncode, run_dir, command = _invoke(tmp_path, monkeypatch)
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["status"] == "completed"
    assert payload["image"] == "example/flagscale:dev"
    assert payload["source"]["path"] == str(gate._REPOSITORY_ROOT.resolve())
    assert payload["source"]["head"] == gate._source_head(gate._REPOSITORY_ROOT)
    source_mount = f"{gate._REPOSITORY_ROOT.resolve()}:{gate.CONTAINER_SOURCE_ROOT}:ro"
    dataset_overlay = (
        f"{run_dir / 'build' / 'megatron-core-datasets'}:"
        f"{gate.CONTAINER_SOURCE_ROOT}/megatron/core/datasets"
    )
    assert source_mount in command
    assert dataset_overlay in command
    assert command.index(source_mount) < command.index("example/flagscale:dev")
    assert command.index(source_mount) < command.index(dataset_overlay)
    assert (run_dir / "build" / "megatron-core-datasets" / "Makefile").is_file()
    assert not tuple((run_dir / "build" / "megatron-core-datasets").glob("helpers_cpp*.so"))
    assert any("conda activate flagscale-train" in argument for argument in command)
    assert ("flagscale", "run") == command[-5:-3]
    assert command[-1] == "--action=test"


def test_source_head_rejects_a_dirty_mounted_checkout(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=source_root, check=True)
    (source_root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.py"), cwd=source_root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=MegaLens Test",
            "-c",
            "user.email=megalens@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ),
        cwd=source_root,
        check=True,
    )

    assert (
        gate._source_head(source_root)
        == subprocess.check_output(
            ("git", "-C", str(source_root), "rev-parse", "HEAD"), text=True
        ).strip()
    )

    (source_root / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit, match="2"):
        gate.main(
            (
                "--run-dir",
                str(run_dir),
                "--input-config",
                str(_FIXTURES / "flagscale_single_node_smoke.yaml"),
                "--mode",
                "trace-off",
                "--image",
                "example/flagscale:dev",
                "--megatron-source-root",
                str(source_root),
            )
        )
    assert not run_dir.exists()


def test_runner_records_config_returncode_log_and_trace_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returncode, run_dir, _ = _invoke(tmp_path, monkeypatch, mode="trace-off")
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["mode"] == "trace-off"
    assert payload["validation"]["trace"]["shards"] == []


def test_trace_off_rejects_a_missing_terminal_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path, monkeypatch, mode="trace-off", write_checkpoint=False
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 1
    assert payload["execution"]["returncode"] == 0
    assert payload["status"] == "failed"
    assert {failure["code"] for failure in payload["validation"]["failures"]} == {
        "run.training.tracker",
        "run.training.checkpoint",
    }


def test_child_failure_is_preserved_in_simple_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returncode, run_dir, _ = _invoke(tmp_path, monkeypatch, child_returncode=23)
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 23
    assert payload["status"] == "failed"
    assert payload["execution"]["returncode"] == 23
    assert payload["validation"]["passed"] is False


def test_profile_failure_changes_zero_child_exit_to_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path, monkeypatch, config="flagscale_single_node_pp2_smoke.yaml"
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 1
    assert payload["execution"]["returncode"] == 0
    assert {failure["code"] for failure in payload["validation"]["failures"]} >= {
        "trace.rank_count",
        "trace.event_missing",
    }


def test_docker_command_keeps_optional_flagscale_overlay_without_hash_lock(tmp_path: Path) -> None:
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
