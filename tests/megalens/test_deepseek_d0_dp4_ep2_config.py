# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_FIXTURES = Path(__file__).parent / "fixtures"
_BASELINE = _FIXTURES / "flagscale_dual_node_deepseek_d0_dp4_mock.yaml"
_D0_DP4_EP2 = (
    _FIXTURES / "flagscale_dual_node_deepseek_d0_dp4_ep2_mock.yaml"
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_deepseek_d0_dp4_ep2_fixture_only_reduces_the_ep_group() -> None:
    expected = deepcopy(_load(_BASELINE))
    derived = _load(_D0_DP4_EP2)
    expected["experiment"]["exp_name"] = (
        "megalens-g7-9-deepseek-d0-dp4-ep2-mock"
    )
    expected["train"]["system"]["expert_model_parallel_size"] = 2

    assert derived == expected


def test_deepseek_d0_dp4_ep2_fixture_derives_the_reviewed_topology() -> None:
    config = _load(_D0_DP4_EP2)
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
        * system["context_parallel_size"]
    )
    microbatches = model["global_batch_size"] // (
        model["micro_batch_size"] * model_dp
    )

    assert (world_size, model_dp, expert_dp, microbatches) == (8, 4, 2, 1)
    assert (
        system["tensor_model_parallel_size"],
        system["pipeline_model_parallel_size"],
        system["context_parallel_size"],
        system["expert_model_parallel_size"],
        system["expert_tensor_parallel_size"],
    ) == (1, 2, 1, 2, 1)
    assert system["decoder_first_pipeline_num_layers"] == 13
    assert model["num_experts"] // system["expert_model_parallel_size"] == 32
    assert (model["micro_batch_size"], model["global_batch_size"]) == (1, 4)
    assert model["train_iters"] == 2
    assert config["train"]["data"]["mock_data"] is True


def test_deepseek_d0_dp4_ep2_fixture_stays_out_of_the_single_node_runner() -> None:
    assert _D0_DP4_EP2.stem not in single_node_gate._CONFIG_PROFILES
