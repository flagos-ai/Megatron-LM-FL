# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from tests.test_utils.runners import deepseek_d0_probe_contract as contract
from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "flagscale_dual_node_deepseek_d0_bf16.yaml"
)
_TEST_MICROBATCHES = 1


def _load_config() -> dict[str, Any]:
    with _FIXTURE.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_rank_trace(trace_root: Path, rank: int) -> None:
    pipeline_rank = rank // 8
    data_rank = rank % 8
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
            routed_tokens=1536,
            expert_cv=0.0,
            top1_expert_share=0.0625,
            expert_max_over_mean=1.0,
            tokens_per_expert=[96] * 16,
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
            num_tokens=4096,
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


def _write_profile(trace_root: Path) -> None:
    for rank in range(16):
        _write_rank_trace(trace_root, rank)


def _mutate_rank(
    trace_root: Path,
    rank: int,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    pipeline_rank = rank // 8
    data_rank = rank % 8
    path = trace_root / (
        f"benchmark-global-{rank}-data-{data_rank}-"
        f"pipeline-{pipeline_rank}-tensor-0.json"
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    mutate(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_deepseek_d0_probe_derivative_preserves_the_guide_model_and_parallel_contract() -> None:
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
    assert system["trace_granularity"] == "full"
    assert system["trace_cupti_kernels"] == "off"

    assert model["num_layers"] == 27
    assert model["hidden_size"] == 2048
    assert model["multi_latent_attention"] is True
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


def test_deepseek_d0_profile_is_not_registered_with_the_single_node_docker_runner() -> None:
    assert _FIXTURE.stem not in single_node_gate._CONFIG_PROFILES


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
    # explicit MTP numbering path. Under this PP2 split it can inherit the last-stage
    # offset and report layer 14. The target architecture reports MTP layer 1, and its
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
