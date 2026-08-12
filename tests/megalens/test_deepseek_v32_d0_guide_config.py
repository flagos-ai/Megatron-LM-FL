# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tests.test_utils.runners import run_flagscale_megalens as single_node_gate

_FIXTURES = Path(__file__).parent / "fixtures"
_AUTOMATIC_D0 = _FIXTURES / "flagscale_dual_node_deepseek_d0_bf16.yaml"
_V32_D0 = _FIXTURES / "flagscale_dual_node_deepseek_v32_d0_guide.yaml"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _changed_paths(
    baseline: object,
    derived: object,
    prefix: tuple[str, ...] = (),
) -> set[str]:
    if isinstance(baseline, dict) and isinstance(derived, dict):
        paths: set[str] = set()
        for key in baseline.keys() | derived.keys():
            if key not in baseline or key not in derived:
                paths.add(".".join((*prefix, key)))
                continue
            paths.update(
                _changed_paths(baseline[key], derived[key], (*prefix, key))
            )
        return paths
    if baseline != derived:
        return {".".join(prefix)}
    return set()


def test_v32_d0_only_restores_the_reviewed_guide_runtime_fields() -> None:
    baseline = _load(_AUTOMATIC_D0)
    guide = _load(_V32_D0)

    assert _changed_paths(baseline, guide) == {
        "experiment.ckpt_format",
        "experiment.envs.CUDA_LAUNCH_BLOCKING",
        "experiment.envs.GLOO_SOCKET_IFNAME",
        "experiment.envs.LOGLEVEL",
        "experiment.envs.NCCL_IB_DISABLE",
        "experiment.envs.NCCL_IB_HCA",
        "experiment.envs.NCCL_SOCKET_IFNAME",
        "experiment.exp_dir",
        "experiment.exp_name",
        "experiment.save_steps",
        "hydra.run.dir",
        "train.data.data_path",
        "train.data.tokenizer.tokenizer_path",
        "train.model.enable_flag_gems",
        "train.model.flag_gems_log_path",
        "train.model.flag_gems_unused",
        "train.model.profile",
        "train.model.profile_ranks",
        "train.model.profile_step_end",
        "train.model.profile_step_start",
        "train.model.te_fl_prefer",
        "train.model.tensorboard_dir",
        "train.model.train_iters",
        "train.model.use_pytorch_profiler",
        "train.system.checkpoint.ckpt_format",
        "train.system.checkpoint.load",
        "train.system.checkpoint.save_interval",
        "train.system.distributed_backend",
        "train.system.logging.log_memory_to_tensorboard",
        "train.system.logging.log_timers_to_tensorboard",
        "train.system.logging.log_validation_ppl_to_tensorboard",
        "train.system.logging.wandb_exp_name",
        "train.system.logging.wandb_project",
        "train.system.trace",
        "train.system.trace_dir",
        "train.system.trace_interval",
    }


def test_v32_d0_preserves_the_supplied_model_and_parallel_contract() -> None:
    config = _load(_V32_D0)
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
    expert_data_parallel_size = world_size // (
        system["pipeline_model_parallel_size"]
        * system["expert_model_parallel_size"]
        * system["expert_tensor_parallel_size"]
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
        system["expert_model_parallel_size"],
        system["expert_tensor_parallel_size"],
        expert_data_parallel_size,
    ) == (1, 2, 1, 8, 4, 1, 2)
    assert system["decoder_first_pipeline_num_layers"] == 13
    assert system["sequence_parallel"] is True
    assert system["use_distributed_optimizer"] is True
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True

    assert model["num_layers"] == 27
    assert model["hidden_size"] == 2048
    assert model["multi_latent_attention"] is True
    assert model["attention_backend"] == "unfused"
    assert (model["qk_head_dim"], model["qk_pos_emb_head_dim"], model["v_head_dim"]) == (
        128,
        64,
        128,
    )
    assert model["num_experts"] == 64
    assert model["moe_router_topk"] == 6
    assert model["moe_shared_expert_intermediate_size"] == 2816
    assert model["mtp_num_layers"] == 1
    assert (model["micro_batch_size"], model["global_batch_size"]) == (1, 512)
    assert model["global_batch_size"] // (
        model["micro_batch_size"] * data_parallel_size
    ) == 64
    assert model["train_iters"] == 102400


def test_v32_d0_requires_the_real_guide_inputs_and_flagos_backends() -> None:
    config = _load(_V32_D0)
    experiment = config["experiment"]
    system = config["train"]["system"]
    model = config["train"]["model"]
    data = config["train"]["data"]

    assert (experiment["save_steps"], experiment["ckpt_format"]) == (500, "torch")
    assert (
        model["te_fl_prefer"],
        model["enable_flag_gems"],
        system["distributed_backend"],
    ) == ("flagos", True, "flagcx")
    assert model["flag_gems_unused"] == [
        "baddbmm",
        "mm",
        "normal_",
        "sum_dim",
        "gather_backward",
        "index_add_",
        "slice_backward",
        "mul",
    ]
    assert model["profile"] is True
    assert (model["profile_step_start"], model["profile_step_end"]) == (3, 5)
    assert model["profile_ranks"] == [0, 7]
    assert model["use_pytorch_profiler"] is True

    assert data["data_path"] == "${oc.env:MEGALENS_V32_DEEPSEEK_DATA_PATH}"
    assert data["tokenizer"]["tokenizer_path"] == (
        "${oc.env:MEGALENS_V32_DEEPSEEK_TOKENIZER_PATH}"
    )
    assert data["tokenizer"]["tokenizer_type"] == "QwenTokenizerFS"
    assert "mock_data" not in data
    assert "legacy_tokenizer" not in data["tokenizer"]

    envs = experiment["envs"]
    assert envs["GLOO_SOCKET_IFNAME"] == "bond0.2208"
    assert envs["NCCL_SOCKET_IFNAME"] == "bond0.2208"
    assert envs["NCCL_IB_HCA"] == (
        "mlx5_101,mlx5_102,mlx5_103,mlx5_104,"
        "mlx5_105,mlx5_106,mlx5_107,mlx5_108"
    )
    assert envs["NCCL_NVLS_ENABLE"] == 0
    assert system["trace"] == "${oc.decode:${oc.env:MEGALENS_V32_TRACE,false}}"
    assert system["trace_interval"] == 1000
    assert system["continuous_trace_iterations"] == 1
    assert _V32_D0.stem not in single_node_gate._CONFIG_PROFILES
    assert _V32_D0.stem not in single_node_gate._OFFLINE_CONFIG_PROFILES
