# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_FIXTURES = Path(__file__).parent / "fixtures"
_BASELINE = _FIXTURES / "flagscale_dual_node_deepseek_d0_dp4_mock.yaml"
_D1_ETP1 = (
    _FIXTURES / "flagscale_dual_node_deepseek_d1_tp2_sp_etp1_mock.yaml"
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_deepseek_d1_etp1_fixture_only_applies_the_reviewed_tp2_derivation() -> None:
    expected = deepcopy(_load(_BASELINE))
    derived = _load(_D1_ETP1)
    expected["experiment"]["exp_name"] = (
        "megalens-g7-9-deepseek-d1-tp2-sp-etp1-mock"
    )
    expected["experiment"]["runner"]["nproc_per_node"] = 8
    expected["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = (
        "0,1,2,3,4,5,6,7"
    )
    expected["train"]["system"]["tensor_model_parallel_size"] = 2

    assert derived == expected


def test_deepseek_d1_etp1_fixture_derives_the_reviewed_parallel_topology() -> None:
    config = _load(_D1_ETP1)
    runner = config["experiment"]["runner"]
    system = config["train"]["system"]
    model = config["train"]["model"]

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
    )
    microbatches = model["global_batch_size"] // (
        model["micro_batch_size"] * model_dp
    )

    assert (world_size, model_dp, expert_dp, microbatches) == (16, 4, 2, 1)
    assert (
        system["tensor_model_parallel_size"],
        system["pipeline_model_parallel_size"],
        system["context_parallel_size"],
        system["expert_model_parallel_size"],
        system["expert_tensor_parallel_size"],
    ) == (2, 2, 1, 4, 1)
    assert system["sequence_parallel"] is True
    assert system["use_distributed_optimizer"] is True
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert config["experiment"]["envs"]["NCCL_NVLS_ENABLE"] == 0
    assert (
        model["num_layers"],
        model["hidden_size"],
        model["ffn_hidden_size"],
        model["moe_ffn_hidden_size"],
    ) == (27, 2048, 11264, 1408)
    assert model["multi_latent_attention"] is True
    assert (model["num_experts"], model["moe_router_topk"]) == (64, 6)
    assert model["moe_shared_expert_intermediate_size"] == 2816
    assert model["moe_token_dispatcher_type"] == "alltoall"
    assert model["mtp_num_layers"] == 1
    assert (model["micro_batch_size"], model["global_batch_size"]) == (1, 4)
    assert model["train_iters"] == 2
    assert config["train"]["data"]["mock_data"] is True


def test_deepseek_d1_etp1_fixture_stays_out_of_the_single_node_runner() -> None:
    assert _D1_ETP1.stem not in single_node_gate._CONFIG_PROFILES
