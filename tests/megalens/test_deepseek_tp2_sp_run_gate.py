# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.test_utils.runners import deepseek_tp2_sp_probe_contract as contract
from tests.test_utils.runners import run_flagscale_megalens as gate
from tests.test_utils.runners import training_run_contract

_FIXTURES = Path(__file__).parent / "fixtures"
_BASELINE = _FIXTURES / "flagscale_dual_node_deepseek_d0_dp4_mock.yaml"
_DERIVED = _FIXTURES / "flagscale_single_node_deepseek_tp2_sp_mock.yaml"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_deepseek_tp2_sp_fixture_only_applies_the_reviewed_single_node_derivation() -> None:
    expected = deepcopy(_load(_BASELINE))
    derived = _load(_DERIVED)
    expected["experiment"]["exp_name"] = "megalens-g5-5-deepseek-tp2-sp-mock"
    expected["experiment"]["runner"] = {
        "per_node_task": False,
        "no_shared_fs": False,
        "rdzv_backend": "static",
        "rdzv_endpoint": "${oc.env:MEGALENS_GATE_RDZV_ENDPOINT}",
        "hostfile": None,
        "nproc_per_node": 8,
        "redirects": 0,
        "tee": 3,
    }
    expected["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = (
        "0,1,2,3,4,5,6,7"
    )
    expected["train"]["system"]["tensor_model_parallel_size"] = 2
    expected["hydra"]["run"]["dir"] = (
        "${oc.env:MEGALENS_GATE_CONTAINER_RUN_DIR}/hydra"
    )

    assert derived == expected

    system = derived["train"]["system"]
    model = derived["train"]["model"]
    world_size = derived["experiment"]["runner"]["nproc_per_node"]
    model_dp = world_size // (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    expert_dp = world_size // (
        system["expert_tensor_parallel_size"]
        * system["expert_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
    )
    microbatches = model["global_batch_size"] // (
        model["micro_batch_size"] * model_dp
    )

    assert (world_size, model_dp, expert_dp, microbatches) == (8, 2, 1, 2)
    assert (
        system["tensor_model_parallel_size"],
        system["pipeline_model_parallel_size"],
        system["expert_model_parallel_size"],
        system["expert_tensor_parallel_size"],
        system["context_parallel_size"],
    ) == (2, 2, 4, 1, 1)
    assert system["sequence_parallel"] is True
    assert (
        model["num_layers"],
        model["hidden_size"],
        model["ffn_hidden_size"],
        model["moe_ffn_hidden_size"],
    ) == (27, 2048, 11264, 1408)
    assert model["multi_latent_attention"] is True
    assert model["moe_shared_expert_intermediate_size"] == 2816
    assert model["moe_token_dispatcher_type"] == "alltoall"
    assert (model["num_experts"], model["moe_router_topk"]) == (64, 6)
    assert model["mtp_num_layers"] == 1
    assert (model["micro_batch_size"], model["global_batch_size"]) == (1, 4)
    assert model["train_iters"] == 2
    assert derived["train"]["data"]["mock_data"] is True


def test_deepseek_tp2_sp_profile_is_selected_by_the_fixture() -> None:
    profile = gate.PROFILES["deepseek-tp2-sp-mock"]

    assert profile.rank_count == 8
    assert profile.contract is contract.validate_deepseek_tp2_sp_trace
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_deepseek_tp2_sp_checkpoint
    )
    assert gate._CONFIG_PROFILES[_DERIVED.stem] == "deepseek-tp2-sp-mock"
    names = {requirement.name for requirement in profile.events}
    assert {
        "forward-step",
        "decoder",
        "decoder-postprocess",
        "output_layer",
        "loss",
        "moe-shared-expert",
        "moe-router",
        "moe-dispatch",
        "moe-experts",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
        "tp-all-gather-first",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-allreduce",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    } <= names
    assert "tp-reduce-scatter-last" not in names


def test_docker_command_mounts_deepseek_tokenizer_read_only(tmp_path: Path) -> None:
    tokenizer_root = tmp_path / "qwentokenizer"
    tokenizer_root.mkdir()

    command = gate._docker_command(
        run_dir=tmp_path,
        config_name="deepseek-tp2-sp-mock",
        mode="trace-on",
        image="example/flagscale:dev",
        source_root=gate._REPOSITORY_ROOT,
        rdzv_port=12345,
        ep_dispatcher=None,
        flagscale_training_overlay=None,
        deepseek_tokenizer_root=tokenizer_root,
    )

    assert (
        f"{tokenizer_root.resolve()}:{gate.CONTAINER_DEEPSEEK_TOKENIZER_ROOT}:ro"
        in command
    )
    assert (
        "MEGALENS_DEEPSEEK_TOKENIZER_PATH="
        f"{gate.CONTAINER_DEEPSEEK_TOKENIZER_ROOT}"
        in command
    )


def test_deepseek_profile_requires_a_host_tokenizer_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="2"):
        gate.main(
            (
                "--run-dir",
                str(tmp_path / "run"),
                "--input-config",
                str(_DERIVED),
                "--mode",
                "trace-off",
                "--image",
                "example/flagscale:dev",
            )
        )


def test_deepseek_run_contract_accepts_the_fixed_terminal_configuration(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "iter_0000002"
    checkpoint.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(
        "2\n", encoding="utf-8"
    )
    (checkpoint / "common.pt").write_bytes(b"checkpoint")
    lines = ["[default]: transformer_impl .... transformer_engine"]
    lines.extend(
        f"[default]: {name} .... {value}"
        for name, value in training_run_contract._DEEPSEEK_TP2_SP_ARGUMENTS
    )
    (tmp_path / "launcher.log").write_text("\n".join(lines), encoding="utf-8")

    assert (
        training_run_contract.validate_two_iteration_deepseek_tp2_sp_checkpoint(
            tmp_path, True
        )
        == ()
    )


def _write_rank_trace(
    trace_root: Path,
    rank: int,
    *,
    model_data_parallel_size: int = 2,
    microbatches_per_iteration: int = 2,
    include_dp_groups: bool = False,
    expert_tensor_parallel_size: int = 1,
) -> None:
    stage_size = 2 * model_data_parallel_size
    pipeline_rank = rank // stage_size
    stage_rank = rank % stage_size
    tensor_rank = stage_rank % 2
    data_rank = stage_rank // 2
    expert_tensor_rank = stage_rank % expert_tensor_parallel_size
    expert_data_parallel_size = stage_size // (
        contract.EXPERT_MODEL_PARALLEL_SIZE * expert_tensor_parallel_size
    )
    expert_data_rank = stage_rank // (
        contract.EXPERT_MODEL_PARALLEL_SIZE * expert_tensor_parallel_size
    )
    tp_ep_group_base = (
        pipeline_rank * stage_size
        + expert_data_rank
        * contract.EXPERT_MODEL_PARALLEL_SIZE
        * expert_tensor_parallel_size
    )
    tp_ep_group = tuple(
        range(
            tp_ep_group_base,
            tp_ep_group_base
            + contract.EXPERT_MODEL_PARALLEL_SIZE
            * expert_tensor_parallel_size,
        )
    )
    main_layers = tuple(range(2, 14)) if pipeline_rank == 0 else tuple(range(14, 28))
    rows: list[dict[str, object]] = []
    timestamp = 0

    def event(name: str, phase: str, iteration: int, **attrs: object) -> None:
        nonlocal timestamp
        timestamp += 1
        rows.append(
            {
                "name": name,
                "ph": phase,
                "rel_ts": timestamp,
                "iteration": iteration,
                "g_rk": rank,
                "dp_rk": data_rank,
                "pp_rk": pipeline_rank,
                "tp_rk": tensor_rank,
                **attrs,
            }
        )

    def moe_call(iteration: int, layer: int) -> None:
        topology = {
            "layer": layer,
            "ep_size": 4,
            "num_experts": 64,
            "num_local_experts": 16,
        }
        handoff = {
            "dropped_tokens": 0,
            "drop_rate": 0.0,
            "expert_cv": 0.125,
            "top1_expert_share": 0.03125,
            "aux_loss": 0.02,
            "z_loss": None,
        }
        ep_collective = {
            "comm_type": "ep-alltoall",
            "dispatcher": "alltoall",
            "data_bytes": 1048576,
            "group_size": 4,
            "ep_size": 4,
            "tp_size": expert_tensor_parallel_size,
        }
        event("moe-shared-expert", "B", iteration)
        event("moe-shared-expert", "E", iteration, layer=layer, ep_size=4)
        event("moe-router", "B", iteration)
        event(
            "tp-allreduce",
            "B",
            iteration,
            op="all_reduce",
            data_bytes=512,
            group_size=2,
            timing_phase="collective_call",
            payload_role="inplace_input_output",
        )
        event("tp-allreduce", "E", iteration, group=[rank ^ 1])
        event(
            "moe-router",
            "E",
            iteration,
            router_topk=6,
            num_tokens=2048,
            routed_tokens=12288,
            routing_entropy=1.5,
            **topology,
            **handoff,
        )
        collective(
            iteration,
            "tp-all-gather-first",
            "all-gather",
            group_size=len(tp_ep_group),
            peers=[peer for peer in tp_ep_group if peer != rank],
            data_bytes=512,
        )
        event("moe-dispatch", "B", iteration)
        event("ep-alltoall-dispatch", "B", iteration)
        event("ep-alltoall-dispatch", "E", iteration, **ep_collective)
        event(
            "moe-dispatch",
            "E",
            iteration,
            router_topk=6,
            dispatcher="alltoall",
            num_tokens=12288,
            capacity_factor=None,
            **topology,
            **handoff,
        )
        if expert_tensor_parallel_size > 1:
            for _ in range(2):
                expert_tp_collective(iteration, "tp-all-gather-first", "all-gather")
        event("moe-experts", "B", iteration)
        event(
            "moe-experts",
            "E",
            iteration,
            routed_tokens=12288 * expert_tensor_parallel_size,
            expert_cv=0.0,
            top1_expert_share=0.0625,
            expert_max_over_mean=1.0,
            tokens_per_expert=[768 * expert_tensor_parallel_size] * 16,
            **topology,
        )
        if expert_tensor_parallel_size > 1:
            expert_tp_collective(iteration, "tp-reduce-scatter", "reduce-scatter")
        event("moe-combine", "B", iteration)
        event("ep-alltoall-combine", "B", iteration)
        event("ep-alltoall-combine", "E", iteration, **ep_collective)
        event(
            "moe-combine",
            "E",
            iteration,
            dispatcher="alltoall",
            num_tokens=12288,
            **topology,
        )

    def collective(
        iteration: int,
        name: str,
        op: str,
        *,
        dim: str = "first",
        group_size: int = 2,
        peers: list[int] | None = None,
        data_bytes: int = 32768,
        split_sizes: list[int] | None = None,
    ) -> None:
        split_fields = {} if split_sizes is None else {"split_sizes": split_sizes}
        event(
            name,
            "B",
            iteration,
            op=op,
            dim=dim,
            data_bytes=data_bytes,
            group_size=group_size,
            **split_fields,
        )
        event(name, "E", iteration, group=peers if peers is not None else [rank ^ 1])

    def expert_tp_collective(iteration: int, name: str, op: str) -> None:
        collective(
            iteration,
            name,
            op,
            group_size=expert_tensor_parallel_size,
            peers=[rank ^ 1],
            split_sizes=[12288, 12288],
        )

    def linear(iteration: int, route: str, occurrence: int) -> None:
        operation_id = f"tp-linear:{rank}:{iteration}:{route}:{occurrence}"
        is_gather = route == "all-gather"
        attrs = {
            "operation_id": operation_id,
            "operation_id_scope": "rank_local",
            "execution_route": "local_linear_direct_async",
            "collective_op": route,
            "data_bytes": 32768,
            "group_size": 2,
            "launch_site": (
                "linear_backward_wgrad_input_all_gather"
                if is_gather
                else "linear_backward_dgrad_reduce_scatter"
            ),
            "pass_direction": "backward",
            "payload_role": "weight_gradient_input" if is_gather else "input_gradient",
            "dim": "first",
        }
        event(
            "tp-linear-async-launch",
            "B",
            iteration,
            **attrs,
            async_op=True,
            completion_included=False,
            timing_phase="launch_attempt",
        )
        event(
            "tp-linear-async-launch",
            "E",
            iteration,
            api_returned=True,
            error_type=None,
        )
        event(
            "tp-linear-async-complete",
            "B",
            iteration,
            **attrs,
            completion_guarantee="current_stream_after_wait",
            completion_included=True,
            completion_kind="work_wait",
            completion_site=(
                "linear_backward_wgrad_input_ready"
                if is_gather
                else "linear_backward_dgrad_reduce_scatter_return"
            ),
            duration_attribution="per_request",
            global_device_completion_guaranteed=False,
            host_blocking_guaranteed=False,
            launch_observed=True,
            op="wait",
            terminal=True,
            timing_phase="stream_dependency",
            wait_role="dependency" if is_gather else "return",
        )
        event(
            "tp-linear-async-complete",
            "E",
            iteration,
            completed=True,
            error_type=None,
        )

    def dp_collective(
        iteration: int,
        name: str,
        role: str,
        group: tuple[int, ...],
    ) -> tuple[str, str]:
        is_reduce_scatter = name == "dp-reduce-scatter"
        operation_id = f"dp:{name}:{role}:{rank}:{iteration}"
        route_fields = (
            {}
            if is_reduce_scatter
            else {"optimizer_kind": "distributed"}
        )
        event(
            name,
            "B",
            iteration,
            api_async_op=True,
            async_op=True,
            completion_included=False,
            data_bytes=32768,
            group_role="intra_optimizer_instance",
            group_size=len(group),
            n_buckets=1,
            op="reduce_scatter" if is_reduce_scatter else "all_gather",
            operation_id=operation_id,
            operation_id_scope="rank_local",
            overlap_enabled=True,
            payload_role=(
                "gradient_bucket" if is_reduce_scatter else "parameter_bucket"
            ),
            stage=(
                "intra_instance_reduce_scatter"
                if is_reduce_scatter
                else "distributed_optimizer_param_allgather"
            ),
            timing_phase="async_dispatch",
            **route_fields,
        )
        event(
            name,
            "E",
            iteration,
            group=[peer for peer in group if peer != rank],
        )
        return operation_id, (
            "intra_instance_reduce_scatter"
            if is_reduce_scatter
            else "distributed_optimizer_param_allgather"
        )

    def dp_completion(
        iteration: int,
        name: str,
        operation_id: str,
        stage: str,
    ) -> None:
        is_grad = name == "dp-reduce-scatter"
        completion_name = (
            "dp-grad-sync-complete" if is_grad else "dp-param-sync-complete"
        )
        completion_fields = (
            {
                "force_all_reduce": False,
                "num_distributed_optimizer_instances": 1,
                "operations": [
                    {
                        "event_name": name,
                        "operation_id": operation_id,
                        "stage": stage,
                    }
                ],
                "use_distributed_optimizer": True,
            }
            if is_grad
            else {"operation_id": operation_id}
        )
        event(
            completion_name,
            "B",
            iteration,
            completion_guarantee="current_stream_after_wait",
            completion_included=True,
            completion_kind="work_wait",
            completion_site=(
                "finish_grad_sync" if is_grad else "finish_param_sync"
            ),
            host_blocking_guaranteed=False,
            launch_observed=True,
            op="wait",
            operation_count=1,
            operation_ids=[operation_id],
            operation_id_scope="rank_local",
            stage=(
                "gradient_collective_completion"
                if is_grad
                else "parameter_allgather_completion"
            ),
            timing_phase="stream_dependency",
            **completion_fields,
        )
        event(
            completion_name,
            "E",
            iteration,
            completed=True,
            error_type=None,
        )

    for iteration in (1, 2):
        rows.append(
            {"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration}
        )
        for _microbatch in range(microbatches_per_iteration):
            event("forward-step", "B", iteration)
            event("decoder", "B", iteration)
            for layer in main_layers:
                moe_call(iteration, layer)
            event("decoder", "E", iteration)
            event("decoder-postprocess", "B", iteration)
            if pipeline_rank == 1:
                moe_call(iteration, 1)
                event("output_layer", "B", iteration)
                event("output_layer", "E", iteration)
                event("loss", "B", iteration)
                event("loss", "E", iteration)
            event("decoder-postprocess", "E", iteration)
            event("forward-step", "E", iteration)
        stage_base = pipeline_rank * stage_size
        if expert_tensor_parallel_size > 1:
            moe_calls = microbatches_per_iteration * (
                len(main_layers) + (1 if pipeline_rank == 1 else 0)
            )
            for _ in range(moe_calls):
                expert_tp_collective(
                    iteration, "tp-all-gather-first", "all-gather"
                )
                for _ in range(2):
                    expert_tp_collective(
                        iteration, "tp-reduce-scatter", "reduce-scatter"
                    )
        collective(iteration, "tp-all-gather-first", "all-gather")
        if pipeline_rank == 1:
            collective(
                iteration,
                "tp-all-gather-last",
                "all-gather",
                dim="last",
            )
        collective(iteration, "tp-reduce-scatter", "reduce-scatter")
        if pipeline_rank == 1:
            for occurrence in range(2 * microbatches_per_iteration):
                linear(iteration, "all-gather", occurrence)
                linear(iteration, "reduce-scatter", occurrence)
        event(
            "grad-sync",
            "B",
            iteration,
            schedule="non-interleaved-1f1b",
            timing_phase="framework_phase",
        )
        event("all-grads-sync", "B", iteration)
        event("all-grads-sync", "E", iteration)
        event(
            "sp-layernorm-allreduce",
            "B",
            iteration,
            data_bytes=512,
            group_size=2,
            reduce_op="SUM",
            grad_bucket="sum",
        )
        event("sp-layernorm-allreduce", "E", iteration, group=[rank ^ 1])
        event(
            "embedding-grads-allreduce",
            "B",
            iteration,
            data_bytes=622329856,
            group_size=2,
            embedding_kind="word",
        )
        embedding_peer = rank + stage_size if pipeline_rank == 0 else rank - stage_size
        event("embedding-grads-allreduce", "E", iteration, group=[embedding_peer])
        if include_dp_groups:
            model_group = tuple(
                stage_base + tensor_rank + data * 2
                for data in range(model_data_parallel_size)
            )
            expert_rank = (
                stage_rank // expert_tensor_parallel_size
            ) % contract.EXPERT_MODEL_PARALLEL_SIZE
            expert_group = tuple(
                stage_base
                + expert_tensor_rank
                + expert_tensor_parallel_size * expert_rank
                + data
                * expert_tensor_parallel_size
                * contract.EXPERT_MODEL_PARALLEL_SIZE
                for data in range(expert_data_parallel_size)
            )
            for name in ("dp-reduce-scatter", "dp-param-all-gather"):
                for role, group in (
                    ("model-dp", model_group),
                    ("expert-dp", expert_group),
                ):
                    operation_id, stage = dp_collective(
                        iteration,
                        name,
                        role,
                        group,
                    )
                    dp_completion(iteration, name, operation_id, stage)
        event("grad-sync", "E", iteration)
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    trace_root.mkdir(parents=True, exist_ok=True)
    path = trace_root / (
        f"benchmark-global-{rank}-data-{data_rank}-pipeline-{pipeline_rank}-"
        f"tensor-{tensor_rank}.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_deepseek_tp2_sp_contract_accepts_the_fixed_l3_trace(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    for rank in range(8):
        _write_rank_trace(trace_root, rank)

    assert contract.validate_deepseek_tp2_sp_trace(trace_root) == ()


def test_deepseek_tp2_sp_contract_rejects_wrong_sp_tokens_and_tp_peer(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    for rank in range(8):
        _write_rank_trace(trace_root, rank)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    router = next(
        row
        for row in rows
        if row.get("name") == "moe-router" and row.get("ph") == "E"
    )
    router["num_tokens"] = 4096
    gather_end = next(
        row
        for row in rows
        if row.get("name") == "tp-all-gather-first" and row.get("ph") == "E"
    )
    gather_end["group"] = [2]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_tp2_sp_trace(trace_root)

    assert {
        "trace.deepseek_tp2_sp.field",
        "trace.deepseek_tp2_sp.tp_collective_group",
    } <= {failure.code for failure in failures}


def test_deepseek_tp2_sp_contract_requires_mtp_last_gather_on_the_last_stage(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    for rank in range(8):
        _write_rank_trace(trace_root, rank)
    rank_four = trace_root / (
        "benchmark-global-4-data-0-pipeline-1-tensor-0.json"
    )
    rows = json.loads(rank_four.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("name") != "tp-all-gather-last"]
    rank_four.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_tp2_sp_trace(trace_root)

    assert "trace.deepseek_tp2_sp.tp_collective_count" in {
        failure.code for failure in failures
    }


def _write_deepseek_d1_etp1_trace(trace_root: Path) -> None:
    for rank in range(contract.D1_WORLD_SIZE):
        _write_rank_trace(
            trace_root,
            rank,
            model_data_parallel_size=contract.D1_MODEL_DATA_PARALLEL_SIZE,
            microbatches_per_iteration=contract.D1_MICROBATCHES_PER_ITERATION,
            include_dp_groups=True,
        )


def _write_deepseek_d1_etp2_trace(trace_root: Path) -> None:
    for rank in range(contract.D1_ETP2_WORLD_SIZE):
        _write_rank_trace(
            trace_root,
            rank,
            model_data_parallel_size=contract.D1_ETP2_MODEL_DATA_PARALLEL_SIZE,
            microbatches_per_iteration=(
                contract.D1_ETP2_MICROBATCHES_PER_ITERATION
            ),
            include_dp_groups=True,
            expert_tensor_parallel_size=2,
        )


def test_deepseek_d1_etp1_contract_accepts_the_fixed_l3_trace(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp1_trace(trace_root)

    assert contract.validate_deepseek_d1_tp2_sp_etp1_trace(trace_root) == ()


def test_deepseek_d1_etp2_contract_accepts_the_fixed_l3_trace(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)

    assert contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root) == ()


def test_deepseek_d1_etp2_topology_uses_the_reviewed_rank_groups() -> None:
    topology = contract._D1_ETP2_TOPOLOGY
    ranks = range(contract.D1_ETP2_WORLD_SIZE)

    assert {topology.tp_ep_group(rank) for rank in ranks} == {
        tuple(range(8)),
        tuple(range(8, 16)),
    }
    assert {topology.expert_tensor_parallel_group(rank) for rank in ranks} == {
        (rank, rank + 1) for rank in range(0, 16, 2)
    }
    assert {topology.expert_model_parallel_group(rank) for rank in ranks} == {
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (8, 10, 12, 14),
        (9, 11, 13, 15),
    }
    assert {topology.model_data_parallel_group(rank) for rank in ranks} == {
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (8, 10, 12, 14),
        (9, 11, 13, 15),
    }
    assert {topology.expert_data_parallel_group(rank) for rank in ranks} == {
        (rank,) for rank in ranks
    }


def test_deepseek_d1_etp2_contract_requires_etp2_dispatch_and_metadata(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    dispatch = next(
        row
        for row in rows
        if row.get("name") == "ep-alltoall-dispatch" and row.get("ph") == "E"
    )
    dispatch["tp_size"] = 1
    metadata_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("name") == "tp-all-gather-first"
        and row.get("ph") == "B"
        and row.get("group_size") == 8
        and "split_sizes" not in row
    )
    del rows[metadata_index : metadata_index + 2]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)
    codes = {failure.code for failure in failures}

    assert "trace.deepseek_tp2_sp.field" in codes
    assert "trace.deepseek_tp2_sp.tp_collective_count" in codes


def test_deepseek_d1_etp2_contract_requires_dispatcher_tp_work(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    dispatcher_collective = next(
        row
        for row in rows
        if row.get("name") == "tp-all-gather-first"
        and row.get("ph") == "B"
        and "split_sizes" in row
    )
    dispatcher_collective.pop("split_sizes")
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dispatcher_tp_count" in {
        failure.code for failure in failures
    }


@pytest.mark.parametrize("invalid_split_sizes", ["invalid", [], [1], [1, True]])
def test_deepseek_d1_etp2_contract_rejects_invalid_dispatcher_split_sizes(
    tmp_path: Path,
    invalid_split_sizes: object,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    dispatcher_collective = next(
        row
        for row in rows
        if row.get("name") == "tp-all-gather-first"
        and row.get("ph") == "B"
        and "split_sizes" in row
    )
    dispatcher_collective["split_sizes"] = invalid_split_sizes
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dispatcher_tp_field" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp2_contract_rejects_zero_dispatcher_token_splits(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    first_call_collectives = [
        row
        for row in rows
        if row.get("ph") == "B" and "split_sizes" in row
    ][:3]
    assert len(first_call_collectives) == 3
    for row in first_call_collectives:
        row["split_sizes"] = [0, 0]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dispatcher_tp_split" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp2_contract_requires_one_split_list_per_moe_call(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    dispatcher_collective = next(
        row
        for row in rows
        if row.get("ph") == "B" and "split_sizes" in row
    )
    dispatcher_collective["split_sizes"] = [12287, 12289]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dispatcher_tp_split" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp2_contract_keeps_dispatcher_tp_at_moe_boundaries(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    dispatcher_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("name") == "tp-all-gather-first"
        and row.get("ph") == "B"
        and "split_sizes" in row
    )
    split_sizes = rows[dispatcher_index].pop("split_sizes")
    model_collective = next(
        row
        for index, row in reversed(tuple(enumerate(rows)))
        if index != dispatcher_index
        and row.get("name") == "tp-all-gather-first"
        and row.get("ph") == "B"
        and row.get("group_size") == 2
        and "split_sizes" not in row
    )
    model_collective["split_sizes"] = split_sizes
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dispatcher_tp_order" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp2_contract_requires_equal_etp_peer_workloads(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_one = trace_root / (
        "benchmark-global-1-data-0-pipeline-0-tensor-1.json"
    )
    rows = json.loads(rank_one.read_text(encoding="utf-8"))
    experts = next(
        row
        for row in rows
        if row.get("name") == "moe-experts" and row.get("ph") == "E"
    )
    experts["tokens_per_expert"][0] += 1
    experts["tokens_per_expert"][1] -= 1
    rank_one.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.etp_workload" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp2_contract_requires_combine_conservation(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_one = trace_root / (
        "benchmark-global-1-data-0-pipeline-0-tensor-1.json"
    )
    rows = json.loads(rank_one.read_text(encoding="utf-8"))
    combine = next(
        row
        for row in rows
        if row.get("name") == "moe-combine" and row.get("ph") == "E"
    )
    combine["num_tokens"] += 1
    rank_one.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)
    codes = {failure.code for failure in failures}

    assert "trace.deepseek_tp2_sp.combine_conservation" in codes
    assert "trace.deepseek_tp2_sp.etp_combine_conservation" in codes


def test_deepseek_d1_etp2_contract_rejects_wrong_expert_dp_singleton_peer(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    expert_reduce_scatter = next(
        row
        for row in rows
        if row.get("name") == "dp-reduce-scatter"
        and row.get("ph") == "E"
        and row.get("group") == []
    )
    expert_reduce_scatter["group"] = [1]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dp_group" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp2_contract_requires_dp_work_completion(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("name") != "dp-grad-sync-complete"]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.dp.event_count" in {failure.code for failure in failures}


def test_deepseek_d1_etp2_contract_requires_expert_dp_rs_each_iteration(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp2_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    dispatch_end = next(
        index
        for index, row in enumerate(rows)
        if row.get("iteration") == 2
        and row.get("name") == "dp-reduce-scatter"
        and row.get("ph") == "E"
        and row.get("group") == []
    )
    dispatch_begin = dispatch_end - 1
    operation_id = rows[dispatch_begin]["operation_id"]
    completion_begin = next(
        index
        for index, row in enumerate(rows)
        if row.get("iteration") == 2
        and row.get("name") == "dp-grad-sync-complete"
        and row.get("ph") == "B"
        and operation_id in row.get("operation_ids", [])
    )
    removed = {
        dispatch_begin,
        dispatch_end,
        completion_begin,
        completion_begin + 1,
    }
    rows = [row for index, row in enumerate(rows) if index not in removed]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp2_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dp_group_count" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp1_contract_keeps_ep4_replicas_independent(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp1_trace(trace_root)

    for rank, data_rank, delta in ((0, 0, -1), (4, 2, 1)):
        path = trace_root / (
            f"benchmark-global-{rank}-data-{data_rank}-pipeline-0-tensor-0.json"
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        experts = next(
            row
            for row in rows
            if row.get("name") == "moe-experts" and row.get("ph") == "E"
        )
        experts["tokens_per_expert"][0] += delta
        experts["routed_tokens"] += delta
        combine = next(
            row
            for row in rows
            if row.get("name") == "moe-combine" and row.get("ph") == "E"
        )
        combine["num_tokens"] += delta
        path.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp1_trace(trace_root)
    conservation_failures = [
        failure
        for failure in failures
        if failure.code == "trace.deepseek_tp2_sp.ep_conservation"
    ]

    assert len(conservation_failures) == 2


def test_deepseek_d1_etp1_contract_rejects_wrong_expert_dp_peer(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp1_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    expert_reduce_scatter = next(
        row
        for row in rows
        if row.get("name") == "dp-reduce-scatter"
        and row.get("ph") == "E"
        and row.get("group") == [4]
    )
    expert_reduce_scatter["group"] = [2]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp1_trace(trace_root)

    assert "trace.deepseek_tp2_sp.dp_group" in {
        failure.code for failure in failures
    }


def test_deepseek_d1_etp1_contract_requires_dp_work_completion(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_deepseek_d1_etp1_trace(trace_root)
    rank_zero = trace_root / (
        "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("name") != "dp-grad-sync-complete"]
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = contract.validate_deepseek_d1_tp2_sp_etp1_trace(trace_root)

    assert "trace.dp.event_count" in {failure.code for failure in failures}
