# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

_FIXTURES = Path(__file__).parent / "fixtures"
_PROFILES = {
    ("dp2", "standard"): (
        "flagscale_dual_node_dp2_standard_smoke.yaml",
        "flagscale_single_node_dp2_standard_smoke.yaml",
        1,
        2,
        "0",
    ),
    ("dp2", "distopt"): (
        "flagscale_dual_node_dp2_distopt_smoke.yaml",
        "flagscale_single_node_dp2_distopt_smoke.yaml",
        1,
        2,
        "0",
    ),
    ("dp16", "standard"): (
        "flagscale_dual_node_dp16_standard_smoke.yaml",
        "flagscale_single_node_dp8_standard_smoke.yaml",
        8,
        16,
        "0,1,2,3,4,5,6,7",
    ),
    ("dp16", "distopt"): (
        "flagscale_dual_node_dp16_distopt_smoke.yaml",
        "flagscale_single_node_dp8_distopt_smoke.yaml",
        8,
        16,
        "0,1,2,3,4,5,6,7",
    ),
}


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURES / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize(("topology", "optimizer"), _PROFILES)
def test_dual_node_probe_profile_contract(topology: str, optimizer: str) -> None:
    filename, source_filename, nproc, global_batch, visible_devices = _PROFILES[
        (topology, optimizer)
    ]
    profile = _load(filename)
    runner = profile["experiment"]["runner"]
    environment = profile["experiment"]["envs"]
    system = profile["train"]["system"]
    model = profile["train"]["model"]

    assert profile["action"] == "test"
    assert runner["type"] == "cloud"
    assert runner["nnodes"] == 2
    assert runner["node_rank"] == "${oc.decode:${oc.env:NODE_RANK}}"
    assert runner["nproc_per_node"] == nproc
    assert runner["master_addr"] == "${oc.env:MASTER_ADDR}"
    assert runner["master_port"] == "${oc.decode:${oc.env:MASTER_PORT}}"
    assert runner["rdzv_backend"] == "static"
    assert runner["rdzv_endpoint"] == "${oc.env:MASTER_ADDR}:${oc.env:MASTER_PORT}"
    assert environment["CUDA_VISIBLE_DEVICES"] == visible_devices
    if topology == "dp16":
        assert environment["NCCL_NVLS_ENABLE"] == 0
    else:
        assert "NCCL_NVLS_ENABLE" not in environment
    assert profile["hydra"]["run"]["dir"].endswith(
        "/hydra/node_${oc.env:NODE_RANK}"
    )

    assert system["tensor_model_parallel_size"] == 1
    assert system["pipeline_model_parallel_size"] == 1
    assert system["context_parallel_size"] == 1
    assert system["overlap_grad_reduce"] is False
    assert system["overlap_param_gather"] is False
    assert system["trace"] == "${oc.decode:${oc.env:MEGALENS_GATE_TRACE,false}}"
    assert system["trace_mode"] == 1
    assert system["trace_granularity"] == "base"
    assert system["trace_cupti_kernels"] == "off"
    assert model["micro_batch_size"] == 1
    assert model["global_batch_size"] == global_batch

    is_distopt = optimizer == "distopt"
    assert system["use_distributed_optimizer"] is is_distopt
    if is_distopt:
        assert system["num_distributed_optimizer_instances"] == 1
    else:
        assert "num_distributed_optimizer_instances" not in system

    expected_train = deepcopy(_load(source_filename)["train"])
    expected_train["model"]["global_batch_size"] = global_batch
    assert profile["train"] == expected_train


@pytest.mark.parametrize("topology", ("dp2", "dp16"))
def test_dual_node_optimizer_profiles_only_change_optimizer_contract(
    topology: str,
) -> None:
    standard = _load(_PROFILES[(topology, "standard")][0])
    distopt = _load(_PROFILES[(topology, "distopt")][0])

    standard["experiment"]["exp_name"] = distopt["experiment"]["exp_name"]
    standard_system = standard["train"]["system"]
    standard_system["use_distributed_optimizer"] = True
    standard_system["num_distributed_optimizer_instances"] = 1

    assert standard == distopt
