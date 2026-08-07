# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from megatron.core import parallel_state
from tests.test_utils.runners import deepseek_d0_probe_contract as contract
from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_DP8_FIXTURE = (
    Path(__file__).parent / "fixtures" / "flagscale_dual_node_deepseek_d0_bf16.yaml"
)
_DP4_FIXTURE = (
    Path(__file__).parent / "fixtures" / "flagscale_dual_node_deepseek_d0_dp4_mock.yaml"
)
_D3_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "flagscale_dual_node_deepseek_d3_no_pp_mock.yaml"
)
_TEST_MICROBATCHES = 1
_D0_DP8_DATA_PARALLEL_SIZE = 8
_D0_DP4_DATA_PARALLEL_SIZE = 4
_D0_EXPERT_MODEL_PARALLEL_SIZE = 4
_D2_DATA_PARALLEL_SIZE = 8
_D2_EXPERT_MODEL_PARALLEL_SIZE = 8
_D3_PIPELINE_MODEL_PARALLEL_SIZE = 1


def _expert_metrics(tokens_per_expert: list[int]) -> dict[str, float]:
    routed_tokens = sum(tokens_per_expert)
    if routed_tokens == 0:
        return {
            "expert_cv": 0.0,
            "top1_expert_share": 0.0,
            "expert_max_over_mean": 0.0,
        }
    mean_tokens = routed_tokens / len(tokens_per_expert)
    variance = sum((value - mean_tokens) ** 2 for value in tokens_per_expert) / len(
        tokens_per_expert
    )
    max_tokens = max(tokens_per_expert)
    return {
        "expert_cv": math.sqrt(variance) / mean_tokens,
        "top1_expert_share": max_tokens / routed_tokens,
        "expert_max_over_mean": max_tokens / mean_tokens,
    }


def _load_config(path: Path = _DP8_FIXTURE) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_rank_trace(
    trace_root: Path,
    rank: int,
    *,
    data_parallel_size: int = _D0_DP8_DATA_PARALLEL_SIZE,
    expert_model_parallel_size: int = _D0_EXPERT_MODEL_PARALLEL_SIZE,
    pipeline_model_parallel_size: int = 2,
    include_dp_groups: bool = False,
    include_etp1_metadata: bool = False,
) -> None:
    pipeline_rank = rank // data_parallel_size
    data_rank = rank % data_parallel_size
    if pipeline_model_parallel_size == 1:
        main_layers = tuple(range(2, 28))
    else:
        main_layers = (
            tuple(range(2, 14))
            if pipeline_rank == 0
            else tuple(range(14, 28))
        )
    timestamp = 0
    rows: list[dict[str, object]] = []

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
                "tp_rk": 0,
                **attrs,
            }
        )

    def moe_call(iteration: int, layer: int) -> None:
        num_local_experts = 64 // expert_model_parallel_size
        routed_tokens = 24576
        tokens_per_expert = [routed_tokens // num_local_experts] * num_local_experts
        topology = {
            "layer": layer,
            "ep_size": expert_model_parallel_size,
            "num_experts": 64,
            "num_local_experts": num_local_experts,
        }
        router_workload = {
            "num_tokens": 4096,
            "routed_tokens": 24576,
            "dropped_tokens": 0,
            "drop_rate": 0.0,
            "expert_cv": 0.125,
            "top1_expert_share": 0.03125,
            "routing_entropy": 1.5,
            "aux_loss": 0.02,
            "z_loss": None,
        }
        handoff = {
            field: router_workload[field]
            for field in (
                "dropped_tokens",
                "drop_rate",
                "expert_cv",
                "top1_expert_share",
                "aux_loss",
                "z_loss",
            )
        }
        collective = {
            "comm_type": "ep-alltoall",
            "dispatcher": "alltoall",
            "data_bytes": 1048576,
            "group_size": expert_model_parallel_size,
            "ep_size": expert_model_parallel_size,
            "tp_size": 1,
        }

        event("moe-shared-expert", "B", iteration)
        event(
            "moe-shared-expert",
            "E",
            iteration,
            layer=layer,
            ep_size=expert_model_parallel_size,
        )
        event("moe-router", "B", iteration)
        event(
            "moe-router",
            "E",
            iteration,
            router_topk=6,
            **topology,
            **router_workload,
        )
        if include_etp1_metadata:
            stage_base = pipeline_rank * data_parallel_size
            expert_data_rank = data_rank // expert_model_parallel_size
            group_base = stage_base + expert_data_rank * expert_model_parallel_size
            metadata_group = tuple(
                range(group_base, group_base + expert_model_parallel_size)
            )
            event(
                "tp-all-gather-first",
                "B",
                iteration,
                op="all-gather",
                dim="first",
                data_bytes=512,
                group_size=expert_model_parallel_size,
            )
            event(
                "tp-all-gather-first",
                "E",
                iteration,
                group=[peer for peer in metadata_group if peer != rank],
            )
        event("moe-dispatch", "B", iteration)
        event("ep-alltoall-dispatch", "B", iteration)
        event("ep-alltoall-dispatch", "E", iteration, **collective)
        event(
            "moe-dispatch",
            "E",
            iteration,
            router_topk=6,
            dispatcher="alltoall",
            num_tokens=24576,
            capacity_factor=None,
            **topology,
            **handoff,
        )
        event("moe-experts", "B", iteration)
        event(
            "moe-experts",
            "E",
            iteration,
            routed_tokens=routed_tokens,
            tokens_per_expert=tokens_per_expert,
            **topology,
            **_expert_metrics(tokens_per_expert),
        )
        event("moe-combine", "B", iteration)
        event("ep-alltoall-combine", "B", iteration)
        event("ep-alltoall-combine", "E", iteration, **collective)
        event(
            "moe-combine",
            "E",
            iteration,
            dispatcher="alltoall",
            num_tokens=24576,
            **topology,
        )

    def dp_collective(
        iteration: int,
        name: str,
        role: str,
        group: tuple[int, ...],
    ) -> tuple[str, str]:
        is_reduce_scatter = name == "dp-reduce-scatter"
        operation_id = f"dp:{name}:{role}:{rank}:{iteration}"
        stage = (
            "intra_instance_reduce_scatter"
            if is_reduce_scatter
            else "distributed_optimizer_param_allgather"
        )
        optimizer_fields = (
            {} if is_reduce_scatter else {"optimizer_kind": "distributed"}
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
            stage=stage,
            timing_phase="async_dispatch",
            **optimizer_fields,
        )
        event(
            name,
            "E",
            iteration,
            group=[peer for peer in group if peer != rank],
        )
        return operation_id, stage

    def dp_completion(
        iteration: int,
        name: str,
        operation_id: str,
        stage: str,
    ) -> None:
        is_gradient = name == "dp-reduce-scatter"
        completion_name = (
            "dp-grad-sync-complete" if is_gradient else "dp-param-sync-complete"
        )
        route_fields = (
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
            if is_gradient
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
                "finish_grad_sync" if is_gradient else "finish_param_sync"
            ),
            host_blocking_guaranteed=False,
            launch_observed=True,
            op="wait",
            operation_count=1,
            operation_ids=[operation_id],
            operation_id_scope="rank_local",
            stage=(
                "gradient_collective_completion"
                if is_gradient
                else "parameter_allgather_completion"
            ),
            timing_phase="stream_dependency",
            **route_fields,
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
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        for _microbatch in range(_TEST_MICROBATCHES):
            event("forward-step", "B", iteration)
            event("decoder", "B", iteration)
            for layer in main_layers:
                moe_call(iteration, layer)
            event("decoder", "E", iteration)
            event("decoder-postprocess", "B", iteration)
            if pipeline_rank == pipeline_model_parallel_size - 1:
                moe_call(iteration, 1)
                event("output_layer", "B", iteration)
                event("output_layer", "E", iteration)
                event("loss", "B", iteration)
                event("loss", "E", iteration)
            event("decoder-postprocess", "E", iteration)
            event("forward-step", "E", iteration)
        if include_dp_groups:
            stage_base = pipeline_rank * data_parallel_size
            model_group = tuple(range(stage_base, stage_base + data_parallel_size))
            expert_rank = data_rank % expert_model_parallel_size
            expert_data_parallel_size = data_parallel_size // expert_model_parallel_size
            expert_group = tuple(
                stage_base + expert_rank + replica * expert_model_parallel_size
                for replica in range(expert_data_parallel_size)
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
        f"benchmark-global-{rank}-data-{data_rank}-"
        f"pipeline-{pipeline_rank}-tensor-0.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_profile(
    trace_root: Path,
    *,
    data_parallel_size: int = _D0_DP8_DATA_PARALLEL_SIZE,
    expert_model_parallel_size: int = _D0_EXPERT_MODEL_PARALLEL_SIZE,
    pipeline_model_parallel_size: int = 2,
    include_dp_groups: bool = False,
    include_etp1_metadata: bool = False,
) -> None:
    for rank in range(pipeline_model_parallel_size * data_parallel_size):
        _write_rank_trace(
            trace_root,
            rank,
            data_parallel_size=data_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            include_dp_groups=include_dp_groups,
            include_etp1_metadata=include_etp1_metadata,
        )


def _mutate_rank(
    trace_root: Path,
    rank: int,
    mutate: Callable[[list[dict[str, object]]], None],
    *,
    data_parallel_size: int = _D0_DP8_DATA_PARALLEL_SIZE,
) -> None:
    pipeline_rank = rank // data_parallel_size
    data_rank = rank % data_parallel_size
    path = trace_root / (
        f"benchmark-global-{rank}-data-{data_rank}-"
        f"pipeline-{pipeline_rank}-tensor-0.json"
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    mutate(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_d2_profile(trace_root: Path) -> None:
    _write_profile(
        trace_root,
        data_parallel_size=_D2_DATA_PARALLEL_SIZE,
        expert_model_parallel_size=_D2_EXPERT_MODEL_PARALLEL_SIZE,
        include_dp_groups=True,
        include_etp1_metadata=True,
    )


def _write_d3_profile(trace_root: Path) -> None:
    _write_profile(
        trace_root,
        data_parallel_size=contract.D3_DATA_PARALLEL_SIZE,
        expert_model_parallel_size=contract.D3_EXPERT_MODEL_PARALLEL_SIZE,
        pipeline_model_parallel_size=_D3_PIPELINE_MODEL_PARALLEL_SIZE,
        include_dp_groups=True,
        include_etp1_metadata=True,
    )


def _adjust_first_local_expert_workload(
    rows: list[dict[str, object]],
    delta: int,
) -> None:
    experts = next(
        row
        for row in rows
        if row.get("name") == "moe-experts"
        and row.get("ph") == "E"
        and row.get("iteration") == 1
    )
    tokens_per_expert = list(experts["tokens_per_expert"])
    tokens_per_expert[0] += delta
    experts["tokens_per_expert"] = tokens_per_expert
    experts["routed_tokens"] = int(experts["routed_tokens"]) + delta
    experts.update(_expert_metrics(tokens_per_expert))
    combine = next(
        row
        for row in rows
        if row.get("name") == "moe-combine"
        and row.get("ph") == "E"
        and row.get("iteration") == 1
    )
    combine["num_tokens"] = int(combine["num_tokens"]) + delta


def _remove_dp_route(
    rows: list[dict[str, object]],
    *,
    name: str,
    role: str,
    iterations: tuple[int, ...],
) -> None:
    completion_name = (
        "dp-grad-sync-complete"
        if name == "dp-reduce-scatter"
        else "dp-param-sync-complete"
    )
    for iteration in iterations:
        begin_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == name
            and row.get("ph") == "B"
            and row.get("iteration") == iteration
            and f":{role}:" in str(row.get("operation_id"))
        )
        operation_id = rows[begin_index]["operation_id"]
        route_end_index = next(
            index
            for index in range(begin_index + 1, len(rows))
            if rows[index].get("name") == name and rows[index].get("ph") == "E"
        )
        completion_begin_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == completion_name
            and row.get("ph") == "B"
            and operation_id in row.get("operation_ids", [])
        )
        completion_end_index = next(
            index
            for index in range(completion_begin_index + 1, len(rows))
            if rows[index].get("name") == completion_name
            and rows[index].get("ph") == "E"
        )
        for index in sorted(
            (
                begin_index,
                route_end_index,
                completion_begin_index,
                completion_end_index,
            ),
            reverse=True,
        ):
            rows.pop(index)


def test_deepseek_d0_probe_profile_preserves_the_guide_model_and_parallel_contract() -> (
    None
):
    config = _load_config()
    experiment = config["experiment"]
    runner = experiment["runner"]
    system = config["train"]["system"]
    model = config["train"]["model"]
    data = config["train"]["data"]

    assert config["action"] == "test"
    assert runner["type"] == "cloud"
    assert runner["nnodes"] == 2
    assert runner["nproc_per_node"] == 8
    assert runner["node_rank"] == "${oc.decode:${oc.env:NODE_RANK}}"
    assert runner["master_addr"] == "${oc.env:MASTER_ADDR}"
    assert runner["master_port"] == "${oc.decode:${oc.env:MASTER_PORT}}"
    assert experiment["envs"]["NCCL_NVLS_ENABLE"] == 0

    assert system["tensor_model_parallel_size"] == 1
    assert system["pipeline_model_parallel_size"] == 2
    assert system["decoder_first_pipeline_num_layers"] == 13
    assert system["context_parallel_size"] == 1
    assert system["expert_model_parallel_size"] == 4
    assert system["expert_tensor_parallel_size"] == 1
    data_parallel_size = 16 // (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    assert data_parallel_size == 8
    assert data_parallel_size // system["expert_model_parallel_size"] == 2
    assert system["sequence_parallel"] is True
    assert system["use_distributed_optimizer"] is True
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert system["precision"] == {
        "bf16": True,
        "attention_softmax_in_fp32": True,
        "accumulate_allreduce_grads_in_fp32": True,
    }
    assert system["trace"] == "${oc.decode:${oc.env:MEGALENS_GATE_TRACE,false}}"
    assert system["trace_mode"] == 1
    assert system["trace_interval"] == 1
    assert system["continuous_trace_iterations"] == 1
    assert system["trace_granularity"] == "full"
    assert system["trace_cupti_kernels"] == "off"
    assert system["checkpoint"] == {
        "save_interval": 2,
        "load": None,
        "ckpt_format": "torch_dist",
    }
    assert experiment["save_steps"] == 2
    assert experiment["load"] is None
    assert experiment["ckpt_format"] == "torch_dist"
    assert "legacy_tokenizer" not in data["tokenizer"]
    assert contract.D0_RANK_ORDER == "tp-cp-ep-dp-pp"
    assert (
        "expert_model_parallel_size"
        not in inspect.signature(contract.validate_deepseek_d0_trace).parameters
    )
    assert (
        inspect.signature(parallel_state.initialize_model_parallel)
        .parameters["order"]
        .default
        == contract.D0_RANK_ORDER
    )

    assert model["num_layers"] == 27
    assert model["hidden_size"] == 2048
    assert model["multi_latent_attention"] is True
    assert model["attention_backend"] == "unfused"
    assert model["kv_lora_rank"] == 512
    assert model["moe_layer_freq"] == "[0]+[1]*26"
    assert model["num_experts"] == 64
    assert model["moe_router_topk"] == 6
    assert model["moe_shared_expert_intermediate_size"] == 2816
    assert model["moe_token_dispatcher_type"] == "alltoall"
    assert model["mtp_num_layers"] == 1
    assert model["seq_length"] == 4096
    assert model["micro_batch_size"] == 1
    assert model["global_batch_size"] == 512
    assert model["train_iters"] == 2
    assert (
        model["global_batch_size"] // (model["micro_batch_size"] * data_parallel_size)
        == contract.DEFAULT_MICROBATCHES_PER_ITERATION
    )
    # This automatic Probe profile isolates the TE/FlagGems/FlagCX environment gate.
    # V3.2 real guide training restores flagos/true/flagcx from the supplied script.
    assert (
        model["te_fl_prefer"],
        model["enable_flag_gems"],
        system["distributed_backend"],
    ) == (
        "vendor",
        False,
        "nccl",
    )

    assert data["data_path"] == "${oc.env:MEGALENS_DEEPSEEK_DATA_PATH}"
    assert data["tokenizer"]["tokenizer_path"] == (
        "${oc.env:MEGALENS_DEEPSEEK_TOKENIZER_PATH}"
    )
    assert data["tokenizer"]["tokenizer_type"] == "QwenTokenizerFS"
    assert "mock_data" not in data


def test_deepseek_d0_dp4_mock_profile_has_only_the_reviewed_derivation() -> None:
    guide = _load_config()
    derived = _load_config(_DP4_FIXTURE)
    expected = deepcopy(guide)
    expected_experiment = expected["experiment"]
    expected_experiment["exp_name"] = "megalens-g7-9-deepseek-d0-dp4-mock"
    expected_experiment["runner"]["nproc_per_node"] = 4
    expected_experiment["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    expected["train"]["system"]["num_workers"] = 0
    expected["train"]["model"]["global_batch_size"] = 4
    expected_data = expected["train"]["data"]
    expected_data.pop("data_path")
    expected_data["mock_data"] = True
    expected_data["data_cache_path"] = (
        "${oc.env:MEGALENS_GATE_CONTAINER_RUN_DIR}/data-cache"
    )

    assert derived == expected

    derived_runner = derived["experiment"]["runner"]
    derived_system = derived["train"]["system"]
    derived_model = derived["train"]["model"]

    world_size = derived_runner["nnodes"] * derived_runner["nproc_per_node"]
    data_parallel_size = world_size // (
        derived_system["tensor_model_parallel_size"]
        * derived_system["pipeline_model_parallel_size"]
        * derived_system["context_parallel_size"]
    )
    assert data_parallel_size == _D0_DP4_DATA_PARALLEL_SIZE
    assert data_parallel_size // derived_system["expert_model_parallel_size"] == 1
    assert (
        derived_model["global_batch_size"]
        // (derived_model["micro_batch_size"] * data_parallel_size)
        == _TEST_MICROBATCHES
    )


def test_deepseek_d3_no_pp_profile_only_applies_the_reviewed_derivation() -> None:
    baseline = _load_config(_DP4_FIXTURE)
    derived = _load_config(_D3_FIXTURE)
    expected = deepcopy(baseline)
    expected["experiment"]["exp_name"] = (
        "megalens-g7-9-deepseek-d3-no-pp-mock"
    )
    expected["experiment"]["runner"]["nproc_per_node"] = 8
    expected["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = (
        "0,1,2,3,4,5,6,7"
    )
    system = expected["train"]["system"]
    system["pipeline_model_parallel_size"] = 1
    system.pop("decoder_first_pipeline_num_layers")
    expected["train"]["model"]["global_batch_size"] = 16

    assert derived == expected

    runner = derived["experiment"]["runner"]
    system = derived["train"]["system"]
    model = derived["train"]["model"]
    world_size = runner["nnodes"] * runner["nproc_per_node"]
    model_dp = world_size // (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    expert_dp = world_size // (
        system["expert_tensor_parallel_size"]
        * system["expert_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    microbatches = model["global_batch_size"] // (
        model["micro_batch_size"] * model_dp
    )

    assert (world_size, model_dp, expert_dp, microbatches) == (16, 16, 4, 1)
    assert (
        system["tensor_model_parallel_size"],
        system["pipeline_model_parallel_size"],
        system["context_parallel_size"],
        system["expert_model_parallel_size"],
        system["expert_tensor_parallel_size"],
    ) == (1, 1, 1, 4, 1)
    assert "decoder_first_pipeline_num_layers" not in system
    assert model["num_layers"] == 27
    assert model["mtp_num_layers"] == 1
    assert derived["train"]["data"]["mock_data"] is True


def test_deepseek_d0_profile_is_not_registered_with_the_single_node_docker_runner() -> (
    None
):
    assert _DP8_FIXTURE.stem not in single_node_gate._CONFIG_PROFILES
    assert _DP4_FIXTURE.stem not in single_node_gate._CONFIG_PROFILES
    assert _D3_FIXTURE.stem not in single_node_gate._CONFIG_PROFILES


def test_deepseek_d0_contract_accepts_the_exact_pp2_ep4_trace(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    assert (
        contract.validate_deepseek_d0_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
        )
        == ()
    )


def test_deepseek_d0_contract_accepts_the_dp4_automatic_derivative(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(
        trace_root,
        data_parallel_size=_D0_DP4_DATA_PARALLEL_SIZE,
    )

    assert (
        contract.validate_deepseek_d0_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
            data_parallel_size=_D0_DP4_DATA_PARALLEL_SIZE,
        )
        == ()
    )


def test_deepseek_d2_contract_accepts_nonuniform_workload_across_one_ep8_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)
    _mutate_rank(
        trace_root,
        0,
        lambda rows: _adjust_first_local_expert_workload(rows, -1),
    )
    _mutate_rank(
        trace_root,
        4,
        lambda rows: _adjust_first_local_expert_workload(rows, 1),
    )

    assert (
        contract.validate_deepseek_d2_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
        )
        == ()
    )


def test_deepseek_d3_contract_accepts_pp1_dp16_ep4_expert_dp4(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d3_profile(trace_root)

    assert (
        contract.validate_deepseek_d3_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
        )
        == ()
    )


def test_deepseek_d3_contract_requires_expert_dp4_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d3_profile(trace_root)

    def break_expert_dp_group(rows: list[dict[str, object]]) -> None:
        begin_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and ":expert-dp:" in str(row.get("operation_id"))
        )
        end = rows[begin_index + 1]
        assert end.get("name") == "dp-reduce-scatter"
        assert end.get("ph") == "E"
        end["group"] = []

    _mutate_rank(
        trace_root,
        0,
        break_expert_dp_group,
        data_parallel_size=contract.D3_DATA_PARALLEL_SIZE,
    )
    failures = contract.validate_deepseek_d3_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d3.dp_group" in {
        failure.code for failure in failures
    }


def test_deepseek_d3_contract_requires_ep4_metadata_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d3_profile(trace_root)

    def break_metadata_group(rows: list[dict[str, object]]) -> None:
        end = next(
            row
            for row in rows
            if row.get("name") == "tp-all-gather-first"
            and row.get("ph") == "E"
        )
        end["group"] = [4, 5, 6]

    _mutate_rank(
        trace_root,
        0,
        break_metadata_group,
        data_parallel_size=contract.D3_DATA_PARALLEL_SIZE,
    )
    failures = contract.validate_deepseek_d3_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d3.metadata_collective" in {
        failure.code for failure in failures
    }


def test_deepseek_d2_contract_requires_ep8_topology_fields(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def restore_ep4_router_topology(rows: list[dict[str, object]]) -> None:
        router = next(
            row
            for row in rows
            if row.get("name") == "moe-router"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        router["ep_size"] = 4

    _mutate_rank(trace_root, 0, restore_ep4_router_topology)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.field" in {failure.code for failure in failures}
    assert any("expected 8" in failure.message for failure in failures)


def test_deepseek_d2_contract_requires_ep8_etp1_alltoall_metadata(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def restore_ep4_collective_size(rows: list[dict[str, object]]) -> None:
        collective = next(
            row
            for row in rows
            if row.get("name") == "ep-alltoall-dispatch"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        collective["group_size"] = 4

    _mutate_rank(trace_root, 0, restore_ep4_collective_size)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.field" in {failure.code for failure in failures}
    assert any("ep-alltoall-dispatch" in failure.message for failure in failures)


def test_deepseek_d2_contract_requires_eight_local_expert_counts(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def restore_sixteen_local_counts(rows: list[dict[str, object]]) -> None:
        experts = next(
            row
            for row in rows
            if row.get("name") == "moe-experts"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        experts["tokens_per_expert"] = [1536] * 16

    _mutate_rank(trace_root, 0, restore_sixteen_local_counts)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.expert_workload" in {failure.code for failure in failures}
    assert any("8 non-negative counts" in failure.message for failure in failures)


def test_deepseek_d2_contract_derives_expert_metrics_from_local_counts(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def restore_ep4_top1_share(rows: list[dict[str, object]]) -> None:
        experts = next(
            row
            for row in rows
            if row.get("name") == "moe-experts"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        experts["top1_expert_share"] = 0.0625

    _mutate_rank(trace_root, 0, restore_ep4_top1_share)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.expert_metric" in {failure.code for failure in failures}
    assert any("expected 0.125" in failure.message for failure in failures)


def test_deepseek_d2_contract_conserves_assignments_within_each_ep8_stage(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)
    _mutate_rank(
        trace_root,
        0,
        lambda rows: _adjust_first_local_expert_workload(rows, -1),
    )
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.ep_conservation" in {failure.code for failure in failures}
    assert any(
        "EP group (0, 1, 2, 3, 4, 5, 6, 7)" in failure.message for failure in failures
    )


def test_deepseek_d2_contract_does_not_conserve_across_pipeline_stages(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)
    _mutate_rank(
        trace_root,
        0,
        lambda rows: _adjust_first_local_expert_workload(rows, -1),
    )
    _mutate_rank(
        trace_root,
        8,
        lambda rows: _adjust_first_local_expert_workload(rows, 1),
    )
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    ep_failures = [
        failure
        for failure in failures
        if failure.code == "trace.deepseek_d2.ep_conservation"
    ]
    assert len(ep_failures) == 2
    assert any(
        "EP group (0, 1, 2, 3, 4, 5, 6, 7)" in failure.message
        for failure in ep_failures
    )
    assert any(
        "EP group (8, 9, 10, 11, 12, 13, 14, 15)" in failure.message
        for failure in ep_failures
    )


def test_deepseek_d2_contract_requires_metadata_gather_stage_peers(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def remove_one_metadata_peer(rows: list[dict[str, object]]) -> None:
        gather = next(
            row
            for row in rows
            if row.get("name") == "tp-all-gather-first"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        gather["group"] = list(gather["group"])[1:]

    _mutate_rank(trace_root, 0, remove_one_metadata_peer)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.metadata_collective" in {
        failure.code for failure in failures
    }
    assert any(
        "expected [1, 2, 3, 4, 5, 6, 7]" in failure.message for failure in failures
    )


def test_deepseek_d2_contract_rejects_dispatcher_splits_on_etp1_metadata(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def add_dispatcher_splits(rows: list[dict[str, object]]) -> None:
        gather = next(
            row
            for row in rows
            if row.get("name") == "tp-all-gather-first"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        gather["split_sizes"] = [3072] * 8

    _mutate_rank(trace_root, 0, add_dispatcher_splits)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.etp_collective" in {failure.code for failure in failures}


def test_deepseek_d2_contract_rejects_etp_reduce_scatter(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def add_reduce_scatter(rows: list[dict[str, object]]) -> None:
        combine_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-combine"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        template = rows[combine_index]
        rows[combine_index:combine_index] = [
            {
                **template,
                "name": "tp-reduce-scatter",
                "ph": "B",
                "op": "reduce-scatter",
                "dim": "first",
                "data_bytes": 512,
                "group_size": 8,
            },
            {
                **template,
                "name": "tp-reduce-scatter",
                "ph": "E",
                "group": [1, 2, 3, 4, 5, 6, 7],
            },
        ]

    _mutate_rank(trace_root, 0, add_reduce_scatter)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.etp_collective" in {failure.code for failure in failures}


def test_deepseek_d2_contract_requires_model_dp8_and_expert_dp1_groups(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def corrupt_model_dp_peers(rows: list[dict[str, object]]) -> None:
        begin_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 8
        )
        end = next(
            row
            for row in rows[begin_index + 1 :]
            if row.get("name") == "dp-reduce-scatter" and row.get("ph") == "E"
        )
        end["group"] = list(end["group"])[1:]

    _mutate_rank(trace_root, 0, corrupt_model_dp_peers)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group" in {failure.code for failure in failures}


def test_deepseek_d2_contract_requires_completion_after_dispatch_end(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def move_completion_inside_dispatch(rows: list[dict[str, object]]) -> None:
        dispatch_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-param-all-gather"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 8
        )
        operation_id = rows[dispatch_begin]["operation_id"]
        dispatch_end = next(
            index
            for index in range(dispatch_begin + 1, len(rows))
            if rows[index].get("name") == "dp-param-all-gather"
            and rows[index].get("ph") == "E"
        )
        completion_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-param-sync-complete"
            and row.get("ph") == "B"
            and operation_id in row.get("operation_ids", [])
        )
        completion_end = next(
            index
            for index in range(completion_begin + 1, len(rows))
            if rows[index].get("name") == "dp-param-sync-complete"
            and rows[index].get("ph") == "E"
        )
        completion = rows[completion_begin : completion_end + 1]
        del rows[completion_begin : completion_end + 1]
        rows[dispatch_end:dispatch_end] = completion

    _mutate_rank(trace_root, 0, move_completion_inside_dispatch)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_completion_order" in {
        failure.code for failure in failures
    }


def test_deepseek_d2_contract_accepts_parameter_completion_in_next_iteration(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def defer_parameter_completion(rows: list[dict[str, object]]) -> None:
        launch = next(
            row
            for row in rows
            if row.get("name") == "dp-param-all-gather"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 8
        )
        operation_id = launch["operation_id"]
        completion_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-param-sync-complete"
            and row.get("ph") == "B"
            and operation_id in row.get("operation_ids", [])
        )
        completion_end = next(
            index
            for index in range(completion_begin + 1, len(rows))
            if rows[index].get("name") == "dp-param-sync-complete"
            and rows[index].get("ph") == "E"
        )
        completion = rows[completion_begin : completion_end + 1]
        del rows[completion_begin : completion_end + 1]
        for row in completion:
            row["iteration"] = 2
        iteration_two_start = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "forward-step"
            and row.get("ph") == "B"
            and row.get("iteration") == 2
        )
        rows[iteration_two_start:iteration_two_start] = completion

    _mutate_rank(trace_root, 0, defer_parameter_completion)

    assert (
        contract.validate_deepseek_d2_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
        )
        == ()
    )


def test_deepseek_d2_contract_rejects_nested_model_and_expert_dp_routes(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def nest_expert_route_inside_model_route(rows: list[dict[str, object]]) -> None:
        model_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 8
        )
        expert_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 1
        )
        expert_end = next(
            index
            for index in range(expert_begin + 1, len(rows))
            if rows[index].get("name") == "dp-reduce-scatter"
            and rows[index].get("ph") == "E"
        )
        expert_route = [rows[expert_begin], rows[expert_end]]
        rows.pop(expert_end)
        rows.pop(expert_begin)
        model_end = next(
            index
            for index in range(model_begin + 1, len(rows))
            if rows[index].get("name") == "dp-reduce-scatter"
            and rows[index].get("ph") == "E"
        )
        rows[model_end:model_end] = expert_route

    _mutate_rank(trace_root, 0, nest_expert_route_inside_model_route)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group_nesting" in {
        failure.code for failure in failures
    }


def test_deepseek_d2_contract_rejects_completion_inside_another_dispatch(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def move_model_completion_inside_expert_dispatch(
        rows: list[dict[str, object]],
    ) -> None:
        model_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 8
        )
        operation_id = rows[model_begin]["operation_id"]
        completion_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-grad-sync-complete"
            and row.get("ph") == "B"
            and operation_id in row.get("operation_ids", [])
        )
        completion_end = next(
            index
            for index in range(completion_begin + 1, len(rows))
            if rows[index].get("name") == "dp-grad-sync-complete"
            and rows[index].get("ph") == "E"
        )
        completion = rows[completion_begin : completion_end + 1]
        del rows[completion_begin : completion_end + 1]
        expert_begin = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 1
        )
        rows[expert_begin + 1 : expert_begin + 1] = completion

    _mutate_rank(trace_root, 0, move_model_completion_inside_expert_dispatch)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group_nesting" in {
        failure.code for failure in failures
    }


def test_deepseek_d2_contract_rejects_nested_completion_scopes(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def nest_expert_completion_inside_model_completion(
        rows: list[dict[str, object]],
    ) -> None:
        operation_ids = {
            int(row["group_size"]): row["operation_id"]
            for row in rows
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") in (1, 8)
        }
        completion_pairs: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
        indexes: list[int] = []
        for group_size, operation_id in operation_ids.items():
            begin = next(
                index
                for index, row in enumerate(rows)
                if row.get("name") == "dp-grad-sync-complete"
                and row.get("ph") == "B"
                and operation_id in row.get("operation_ids", [])
            )
            end = next(
                index
                for index in range(begin + 1, len(rows))
                if rows[index].get("name") == "dp-grad-sync-complete"
                and rows[index].get("ph") == "E"
            )
            completion_pairs[group_size] = (rows[begin], rows[end])
            indexes.extend((begin, end))
        for index in sorted(indexes, reverse=True):
            rows.pop(index)
        expert_dispatch_end = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
            and index > 0
            and rows[index - 1].get("group_size") == 1
        )
        model_begin, model_end = completion_pairs[8]
        expert_begin, expert_end = completion_pairs[1]
        rows[expert_dispatch_end + 1 : expert_dispatch_end + 1] = [
            model_begin,
            expert_begin,
            expert_end,
            model_end,
        ]

    _mutate_rank(trace_root, 0, nest_expert_completion_inside_model_completion)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group_nesting" in {
        failure.code for failure in failures
    }


def test_deepseek_d2_contract_rejects_expert_dp1_peer(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def add_singleton_expert_dp_peer(rows: list[dict[str, object]]) -> None:
        begin_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-reduce-scatter"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
            and row.get("group_size") == 1
        )
        end = next(
            row
            for row in rows[begin_index + 1 :]
            if row.get("name") == "dp-reduce-scatter" and row.get("ph") == "E"
        )
        end["group"] = [1]

    _mutate_rank(trace_root, 0, add_singleton_expert_dp_peer)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group" in {failure.code for failure in failures}


def test_deepseek_d2_contract_requires_expert_dp1_reduce_scatter_each_iteration(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)
    _mutate_rank(
        trace_root,
        0,
        lambda rows: _remove_dp_route(
            rows,
            name="dp-reduce-scatter",
            role="expert-dp",
            iterations=(2,),
        ),
    )
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group_count" in {failure.code for failure in failures}
    assert any("iteration=2" in failure.evidence for failure in failures)


def test_deepseek_d2_contract_requires_expert_dp1_parameter_gather_window(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)
    _mutate_rank(
        trace_root,
        0,
        lambda rows: _remove_dp_route(
            rows,
            name="dp-param-all-gather",
            role="expert-dp",
            iterations=(1, 2),
        ),
    )
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d2.dp_group_count" in {failure.code for failure in failures}
    assert any("dp-param-all-gather" in failure.message for failure in failures)


def test_deepseek_d2_contract_requires_distopt_work_completion(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_d2_profile(trace_root)

    def remove_completion(rows: list[dict[str, object]]) -> None:
        launch = next(
            row
            for row in rows
            if row.get("name") == "dp-param-all-gather"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        operation_id = launch["operation_id"]
        begin_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "dp-param-sync-complete"
            and row.get("ph") == "B"
            and operation_id in row.get("operation_ids", [])
        )
        end_index = next(
            index
            for index in range(begin_index + 1, len(rows))
            if rows[index].get("name") == "dp-param-sync-complete"
            and rows[index].get("ph") == "E"
        )
        rows.pop(end_index)
        rows.pop(begin_index)

    _mutate_rank(trace_root, 0, remove_completion)
    failures = contract.validate_deepseek_d2_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.dp.operation_id" in {failure.code for failure in failures}


def test_deepseek_d0_contract_rejects_unreviewed_data_parallel_sizes(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be 4 or 8"):
        contract.validate_deepseek_d0_trace(
            tmp_path,
            data_parallel_size=12,
        )


def test_deepseek_d0_contract_requires_shared_expert_per_moe_call(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def remove_first_shared_begin(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-shared-expert"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        rows.pop(index)

    _mutate_rank(trace_root, 0, remove_first_shared_begin)
    failures = contract.validate_deepseek_d0_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d0.moe_sequence" in {failure.code for failure in failures}


def test_deepseek_d0_contract_keeps_target_mtp_layer_one_in_postprocess(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    # MixedPara baseline builds the inner MTP Transformer layer without the target's
    # explicit MTP numbering path. Under this D0 PP2 split it inherits the last-stage
    # offset and reports layer 14. The target architecture reports MTP layer 1, and its
    # decoder-postprocess parent distinguishes it from the main decoder layers.
    def apply_mixedpara_pp_offset(rows: list[dict[str, object]]) -> None:
        in_postprocess = False
        for row in rows:
            if row.get("name") == "decoder-postprocess":
                in_postprocess = row.get("ph") == "B"
                continue
            if in_postprocess and row.get("layer") == 1:
                row["layer"] = 14

    _mutate_rank(trace_root, 8, apply_mixedpara_pp_offset)
    failures = contract.validate_deepseek_d0_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d0.field" in {failure.code for failure in failures}
    assert any("region=decoder-postprocess" in failure.evidence for failure in failures)


def test_deepseek_d0_contract_requires_ep4_router_dispatch_handoff(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def change_dispatch_drop_rate(rows: list[dict[str, object]]) -> None:
        dispatch = next(
            row
            for row in rows
            if row.get("name") == "moe-dispatch"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        dispatch["drop_rate"] = 0.5

    _mutate_rank(trace_root, 0, change_dispatch_drop_rate)
    failures = contract.validate_deepseek_d0_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d0.router_handoff" in {failure.code for failure in failures}


def test_deepseek_d0_contract_requires_routed_assignment_count_at_dispatch(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def restore_pre_route_token_count(rows: list[dict[str, object]]) -> None:
        dispatch = next(
            row
            for row in rows
            if row.get("name") == "moe-dispatch"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        dispatch["num_tokens"] = 4096

    _mutate_rank(trace_root, 0, restore_pre_route_token_count)
    failures = contract.validate_deepseek_d0_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d0.field" in {failure.code for failure in failures}


def test_deepseek_d0_contract_relates_combine_to_the_local_expert_workload(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def change_combine_num_tokens(rows: list[dict[str, object]]) -> None:
        combine = next(
            row
            for row in rows
            if row.get("name") == "moe-combine"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        combine["num_tokens"] = 24575

    _mutate_rank(trace_root, 0, change_combine_num_tokens)
    failures = contract.validate_deepseek_d0_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d0.combine_workload" in {
        failure.code for failure in failures
    }


def test_deepseek_d0_contract_accepts_finite_expert_metrics_without_derivation(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def change_finite_expert_metrics(rows: list[dict[str, object]]) -> None:
        experts = next(
            row
            for row in rows
            if row.get("name") == "moe-experts"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        experts.update(
            {
                "expert_cv": 7.0,
                "top1_expert_share": 0.9,
                "expert_max_over_mean": 3.0,
            }
        )

    _mutate_rank(trace_root, 0, change_finite_expert_metrics)

    assert (
        contract.validate_deepseek_d0_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
        )
        == ()
    )


def test_deepseek_d0_contract_conserves_assignments_across_each_ep4_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def reduce_consistent_local_workload(rows: list[dict[str, object]]) -> None:
        experts = next(
            row
            for row in rows
            if row.get("name") == "moe-experts"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        experts["routed_tokens"] = 24575
        experts["tokens_per_expert"] = [1535] + [1536] * 15
        combine = next(
            row
            for row in rows
            if row.get("name") == "moe-combine"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        combine["num_tokens"] = 24575

    _mutate_rank(trace_root, 0, reduce_consistent_local_workload)
    failures = contract.validate_deepseek_d0_trace(
        trace_root,
        microbatches_per_iteration=_TEST_MICROBATCHES,
    )

    assert "trace.deepseek_d0.ep_conservation" in {failure.code for failure in failures}


def test_deepseek_d0_contract_accepts_a_zero_token_local_expert_rank(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def set_local_workload(
        rows: list[dict[str, object]],
        *,
        routed_tokens: int,
        tokens_per_expert: list[int],
        combine_data_bytes: int,
    ) -> None:
        experts = next(
            row
            for row in rows
            if row.get("name") == "moe-experts"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        experts["routed_tokens"] = routed_tokens
        experts["tokens_per_expert"] = tokens_per_expert
        combine = next(
            row
            for row in rows
            if row.get("name") == "moe-combine"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        combine["num_tokens"] = routed_tokens
        collective = next(
            row
            for row in rows
            if row.get("name") == "ep-alltoall-combine"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        collective["data_bytes"] = combine_data_bytes

    _mutate_rank(
        trace_root,
        0,
        lambda rows: set_local_workload(
            rows,
            routed_tokens=0,
            tokens_per_expert=[0] * 16,
            combine_data_bytes=0,
        ),
    )
    _mutate_rank(
        trace_root,
        1,
        lambda rows: set_local_workload(
            rows,
            routed_tokens=49152,
            tokens_per_expert=[3072] * 16,
            combine_data_bytes=1048576,
        ),
    )

    assert (
        contract.validate_deepseek_d0_trace(
            trace_root,
            microbatches_per_iteration=_TEST_MICROBATCHES,
        )
        == ()
    )
