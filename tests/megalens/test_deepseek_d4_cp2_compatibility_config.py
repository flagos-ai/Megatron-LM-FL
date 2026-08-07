# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_FIXTURES = Path(__file__).parent / "fixtures"
_BASELINE = _FIXTURES / "flagscale_dual_node_deepseek_d0_dp8_mock.yaml"
_D4_CP2_NEGATIVE = (
    _FIXTURES / "flagscale_dual_node_deepseek_d4_cp2_unfused_negative.yaml"
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_deepseek_d4_fixture_only_applies_the_reviewed_cp2_derivation() -> None:
    expected = deepcopy(_load(_BASELINE))
    derived = _load(_D4_CP2_NEGATIVE)
    expected["experiment"]["exp_name"] = (
        "megalens-g7-9-deepseek-d4-cp2-unfused-negative"
    )
    expected["train"]["system"]["context_parallel_size"] = 2
    expected["train"]["model"]["global_batch_size"] = 4

    assert derived == expected


def test_deepseek_d4_fixture_derives_the_target_runtime_groups() -> None:
    config = _load(_D4_CP2_NEGATIVE)
    runner = config["experiment"]["runner"]
    system = config["train"]["system"]
    model = config["train"]["model"]

    world_size = runner["nnodes"] * runner["nproc_per_node"]
    model_dp = world_size // (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )
    runtime_expert_dp = world_size // (
        system["expert_tensor_parallel_size"]
        * system["expert_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
    )
    dp_cp_group_size = model_dp * system["context_parallel_size"]
    microbatches = model["global_batch_size"] // (
        model["micro_batch_size"] * model_dp
    )

    assert (
        world_size,
        model_dp,
        runtime_expert_dp,
        dp_cp_group_size,
        microbatches,
    ) == (16, 4, 2, 8, 1)
    assert (
        system["tensor_model_parallel_size"],
        system["pipeline_model_parallel_size"],
        system["context_parallel_size"],
        system["expert_model_parallel_size"],
        system["expert_tensor_parallel_size"],
    ) == (1, 2, 2, 4, 1)
    assert system["decoder_first_pipeline_num_layers"] == 13
    assert model["seq_length"] % (2 * system["context_parallel_size"]) == 0
    assert (model["micro_batch_size"], model["global_batch_size"]) == (1, 4)
    assert model["train_iters"] == 2
    assert config["train"]["data"]["mock_data"] is True


def test_deepseek_d4_fixture_preserves_the_negative_preflight_boundary() -> None:
    config = _load(_D4_CP2_NEGATIVE)
    model = config["train"]["model"]
    data = config["train"]["data"]

    assert model["transformer_impl"] == "transformer_engine"
    assert model["multi_latent_attention"] is True
    assert model["attention_backend"] == "unfused"
    assert data["reset_position_ids"] is True
    assert data["reset_attention_mask"] is True
    assert _D4_CP2_NEGATIVE.stem not in single_node_gate._CONFIG_PROFILES
    assert _D4_CP2_NEGATIVE.stem not in single_node_gate._OFFLINE_CONFIG_PROFILES
