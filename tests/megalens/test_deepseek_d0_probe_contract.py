# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from megatron.core import parallel_state
from tests.test_utils.runners import deepseek_d0_probe_contract as contract
from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_DP8_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "flagscale_dual_node_deepseek_d0_bf16.yaml"
)
_DP4_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "flagscale_dual_node_deepseek_d0_dp4_mock.yaml"
)
_TEST_MICROBATCHES = 1
_D0_DP8_DATA_PARALLEL_SIZE = 8
_D0_DP4_DATA_PARALLEL_SIZE = 4


def _load_config(path: Path = _DP8_FIXTURE) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_rank_trace(
    trace_root: Path,
    rank: int,
    *,
    data_parallel_size: int = _D0_DP8_DATA_PARALLEL_SIZE,
) -> None:
    pipeline_rank = rank // data_parallel_size
    data_rank = rank % data_parallel_size
    main_layers = tuple(range(2, 14)) if pipeline_rank == 0 else tuple(range(14, 28))
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
        topology = {
            "layer": layer,
            "ep_size": 4,
            "num_experts": 64,
            "num_local_experts": 16,
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
            "group_size": 4,
            "ep_size": 4,
            "tp_size": 1,
        }

        event("moe-shared-expert", "B", iteration)
        event("moe-shared-expert", "E", iteration, layer=layer, ep_size=4)
        event("moe-router", "B", iteration)
        event(
            "moe-router",
            "E",
            iteration,
            router_topk=6,
            **topology,
            **router_workload,
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
            num_tokens=4096,
            capacity_factor=None,
            **topology,
            **handoff,
        )
        event("moe-experts", "B", iteration)
        event(
            "moe-experts",
            "E",
            iteration,
            routed_tokens=24576,
            expert_cv=0.0,
            top1_expert_share=0.0625,
            expert_max_over_mean=1.0,
            tokens_per_expert=[1536] * 16,
            **topology,
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
            if pipeline_rank == 1:
                moe_call(iteration, 1)
                event("output_layer", "B", iteration)
                event("output_layer", "E", iteration)
                event("loss", "B", iteration)
                event("loss", "E", iteration)
            event("decoder-postprocess", "E", iteration)
            event("forward-step", "E", iteration)
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
) -> None:
    for rank in range(2 * data_parallel_size):
        _write_rank_trace(
            trace_root,
            rank,
            data_parallel_size=data_parallel_size,
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


def test_deepseek_d0_probe_profile_preserves_the_guide_model_and_parallel_contract() -> None:
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
        model["global_batch_size"]
        // (model["micro_batch_size"] * data_parallel_size)
        == contract.DEFAULT_MICROBATCHES_PER_ITERATION
    )
    # This automatic Probe profile isolates the TE/FlagGems/FlagCX environment gate.
    # V3.2 real guide training restores flagos/true/flagcx from the supplied script.
    assert (model["te_fl_prefer"], model["enable_flag_gems"], system["distributed_backend"]) == (
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


def test_deepseek_d0_profile_is_not_registered_with_the_single_node_docker_runner() -> None:
    assert _DP8_FIXTURE.stem not in single_node_gate._CONFIG_PROFILES
    assert _DP4_FIXTURE.stem not in single_node_gate._CONFIG_PROFILES


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

    assert "trace.deepseek_d0.router_handoff" in {
        failure.code for failure in failures
    }


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

    assert "trace.deepseek_d0.ep_conservation" in {
        failure.code for failure in failures
    }


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
