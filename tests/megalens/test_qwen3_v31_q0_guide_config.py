# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "flagscale_dual_node_qwen3_v31_q0_guide.yaml"
)


def _load() -> dict[str, Any]:
    return yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))


def test_v31_q0_preserves_the_supplied_model_and_parallel_contract() -> None:
    config = _load()
    experiment = config["experiment"]
    runner = experiment["runner"]
    system = config["train"]["system"]
    model = config["train"]["model"]

    world_size = runner["nnodes"] * runner["nproc_per_node"]
    data_parallel_size = world_size // (
        system["tensor_model_parallel_size"]
        * system["pipeline_model_parallel_size"]
        * system["context_parallel_size"]
    )

    assert config["action"] == "test"
    assert experiment["task"]["entrypoint"] == (
        "flagscale/train/megatron/train_gpt.py"
    )
    assert (runner["type"], runner["nnodes"], runner["nproc_per_node"]) == (
        "cloud",
        2,
        8,
    )
    assert (
        system["tensor_model_parallel_size"],
        system["pipeline_model_parallel_size"],
        system["context_parallel_size"],
        data_parallel_size,
    ) == (1, 1, 1, 16)
    assert system["sequence_parallel"] is True
    # Preserve the guide input. Megatron normalizes sequence parallelism to
    # false when TP is one, so Q0 has no cross-rank SP communication.
    assert not (
        system["sequence_parallel"]
        and system["tensor_model_parallel_size"] > 1
    )
    assert system["use_distributed_optimizer"] is True
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert system["disable_bias_linear"] is True
    assert system["reset_position_ids"] is True
    assert system["reset_attention_mask"] is True
    assert system["qk_layernorm"] is True
    assert system["precision"] == {
        "bf16": True,
        "attention_softmax_in_fp32": True,
        "accumulate_allreduce_grads_in_fp32": True,
    }

    assert (
        model["num_layers"],
        model["hidden_size"],
        model["ffn_hidden_size"],
        model["num_attention_heads"],
        model["num_query_groups"],
    ) == (28, 1024, 3072, 16, 8)
    assert model["kv_channels"] == 128
    assert model["group_query_attention"] is True
    assert (model["seq_length"], model["max_position_embeddings"]) == (
        4096,
        40960,
    )
    assert model["init_method_std"] == 0.006
    assert model["no_rope_fusion"] is True
    assert model["untie_embeddings_and_output_weights"] is False
    assert (model["micro_batch_size"], model["global_batch_size"]) == (4, 2048)
    assert model["global_batch_size"] // (
        model["micro_batch_size"] * data_parallel_size
    ) == 32
    assert model["train_samples"] == 29297664
    assert "train_iters" not in model
    assert model["optimizer"]["lr_scheduler"] == {
        "lr": 3.0e-3,
        "min_lr": 3.0e-4,
        "lr_warmup_samples": 2048000,
        "lr_decay_style": "cosine",
    }


def test_v31_q0_requires_the_guide_inputs_and_flagos_backends() -> None:
    config = _load()
    experiment = config["experiment"]
    system = config["train"]["system"]
    model = config["train"]["model"]
    data = config["train"]["data"]

    assert (
        experiment["save_steps"],
        experiment["load"],
        experiment["ckpt_format"],
    ) == (2000, None, "torch")
    assert system["checkpoint"] == {
        "save_interval": "${experiment.save_steps}",
        "load": "${experiment.load}",
        "ckpt_format": "${experiment.ckpt_format}",
    }
    assert (
        model["transformer_impl"],
        model["te_fl_prefer"],
        model["enable_flag_gems"],
        system["distributed_backend"],
    ) == ("transformer_engine", "flagos", True, "flagcx")
    assert model["attention_backend"] == "flash"
    assert "flag_gems_unused" not in model
    assert "te_fl_per_op" not in model
    assert "te_fl_allow_vendors" not in model
    assert "te_fl_deny_vendors" not in model
    assert "distributed_backend" not in model
    assert model["profile"] is True
    assert (model["profile_step_start"], model["profile_step_end"]) == (3, 5)
    assert model["profile_ranks"] == [0, 7]
    assert model["use_pytorch_profiler"] is True

    assert data["data_path"] == "${oc.env:MEGALENS_V31_QWEN3_DATA_PATH}"
    assert data["tokenizer"]["tokenizer_path"] == (
        "${oc.env:MEGALENS_V31_QWEN3_TOKENIZER_PATH}"
    )
    assert data["tokenizer"]["tokenizer_type"] == "QwenTokenizerFS"
    assert data["tokenizer"]["vocab_size"] == 151851
    assert data["tokenizer"]["make_vocab_size_divisible_by"] == 64
    assert data["split"] == 1
    assert data["no_mmap_bin_files"] is True
    assert "mock_data" not in data
    assert "legacy_tokenizer" not in data["tokenizer"]


def test_v31_q0_limits_adaptations_to_the_reviewed_runtime_boundary() -> None:
    config = _load()
    experiment = config["experiment"]
    runner = experiment["runner"]
    envs = experiment["envs"]
    system = config["train"]["system"]

    assert runner["rdzv_endpoint"] == (
        "${oc.env:MASTER_ADDR}:${oc.env:MASTER_PORT}"
    )
    assert runner["node_rank"] == "${oc.decode:${oc.env:NODE_RANK}}"
    assert experiment["exp_dir"] == "${oc.env:MEGALENS_V31_RUN_DIR}"
    assert envs["GLOO_SOCKET_IFNAME"] == "bond0.2208"
    assert envs["NCCL_SOCKET_IFNAME"] == "bond0.2208"
    assert envs["NCCL_IB_HCA"] == (
        "mlx5_101,mlx5_102,mlx5_103,mlx5_104,"
        "mlx5_105,mlx5_106,mlx5_107,mlx5_108"
    )
    assert envs["FLAGCX_SOCKET_IFNAME"] == "=bond0.2208"
    assert envs["FLAGCX_IB_DISABLE"] == 0
    assert envs["FLAGCX_IB_HCA"] == envs["NCCL_IB_HCA"]
    assert envs["FLAGCX_TOPO_DETECTION_DISABLE"] == 0
    assert envs["FLAGCX_DEBUG"] == "INFO"
    assert envs["FLAGCX_DEBUG_SUBSYS"] == "INIT,ENV,NET"
    assert envs["NCCL_NVLS_ENABLE"] == 0
    assert envs["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"

    assert system["trace"] == "${oc.decode:${oc.env:MEGALENS_V31_TRACE,false}}"
    assert system["trace_mode"] == 1
    assert system["trace_interval"] == 1000
    assert system["continuous_trace_iterations"] == 1
    assert system["trace_granularity"] == "full"
    assert system["trace_cupti_kernels"] == "off"
    assert system["hardware_monitor"] is False
    assert config["hydra"]["run"]["dir"] == (
        "${oc.env:MEGALENS_V31_RUN_DIR}/hydra/node_${oc.env:NODE_RANK}"
    )
    assert _FIXTURE.stem not in single_node_gate._CONFIG_PROFILES
    assert _FIXTURE.stem not in single_node_gate._OFFLINE_CONFIG_PROFILES
