# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import dataclasses
import json
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from megatron.training import arguments


def _minimal_training_argv(extra_args=None):
    argv = [
        "program",
        "--num-layers",
        "2",
        "--hidden-size",
        "16",
        "--num-attention-heads",
        "4",
        "--seq-length",
        "8",
        "--max-position-embeddings",
        "8",
        "--micro-batch-size",
        "2",
        "--train-iters",
        "4",
        "--lr",
        "0.001",
        "--min-lr",
        "0.0",
        "--bf16",
        "--mock-data",
    ]
    if extra_args:
        argv.extend(extra_args)
    return argv


def _parse_minimal_training_args(monkeypatch, extra_args=None):
    monkeypatch.setattr("sys.argv", _minimal_training_argv(extra_args))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return arguments.parse_args()


def _patch_validate_environment(monkeypatch):
    monkeypatch.setattr(arguments, "get_device_arch_version", lambda: 10)
    monkeypatch.setattr(arguments, "is_flashinfer_min_version", lambda version: True)
    monkeypatch.setattr(arguments, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(arguments, "is_torch_min_version", lambda version: True)


def test_add_megatron_arguments_registers_training_parser_groups():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    returned_parser = arguments.add_megatron_arguments(parser)
    group_titles = {group.title for group in parser._action_groups}

    assert returned_parser is parser
    assert "network size" in group_titles
    assert "training" in group_titles
    assert "learning rate and weight decay" in group_titles
    assert "checkpointing" in group_titles
    assert "distributed init" in group_titles
    assert "validation" in group_titles
    assert "data and dataloader" in group_titles
    assert "tokenizer" in group_titles


def test_parser_accepts_representative_training_arguments():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments.add_megatron_arguments(parser)

    parsed = parser.parse_args(
        [
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
            "--lr",
            "0.001",
            "--min-lr",
            "0.0001",
            "--lr-decay-style",
            "cosine",
            "--dataloader-type",
            "single",
            "--tokenizer-type",
            "NullTokenizer",
            "--bf16",
            "--use-distributed-optimizer",
        ]
    )

    assert parsed.num_layers == 2
    assert parsed.hidden_size == 16
    assert parsed.num_attention_heads == 4
    assert parsed.micro_batch_size == 1
    assert parsed.global_batch_size == 1
    assert parsed.lr == 0.001
    assert parsed.min_lr == 0.0001
    assert parsed.lr_decay_style == "cosine"
    assert parsed.dataloader_type == "single"
    assert parsed.tokenizer_type == "NullTokenizer"
    assert parsed.bf16
    assert parsed.use_distributed_optimizer


def test_parse_args_sets_rank_and_world_size_from_environment(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
        ],
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    parsed = arguments.parse_args()

    assert parsed.rank == 3
    assert parsed.world_size == 8
    assert parsed.enable_msc


def test_parse_args_allows_extra_provider_and_unknown_args(monkeypatch):
    def extra_provider(parser):
        parser.add_argument("--custom-flag", type=int, default=0)
        return parser

    monkeypatch.setattr("sys.argv", ["program", "--custom-flag", "7", "--unknown-flag"])

    parsed = arguments.parse_args(extra_args_provider=extra_provider, ignore_unknown_args=True)

    assert parsed.custom_flag == 7


def test_parse_args_yaml_and_disable_msc_paths(monkeypatch):
    loaded_args = SimpleNamespace(yaml_loaded=True, enable_msc=True)
    yaml_module = ModuleType("megatron.training.yaml_arguments")
    yaml_module.__spec__ = ModuleSpec("megatron.training.yaml_arguments", loader=None)
    yaml_module.load_yaml = lambda path: loaded_args
    monkeypatch.setitem(sys.modules, "megatron.training.yaml_arguments", yaml_module)
    monkeypatch.setattr(
        "sys.argv",
        _minimal_training_argv(["--yaml-cfg", "config.yaml"]),
    )
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")

    parsed = arguments.parse_args()

    assert parsed is loaded_args
    assert parsed.rank == 2
    assert parsed.world_size == 4

    calls = []
    monkeypatch.setattr("sys.argv", _minimal_training_argv(["--disable-msc"]))
    monkeypatch.setattr(arguments.MultiStorageClientFeature, "disable", lambda: calls.append("disable"))
    monkeypatch.setattr(arguments.MultiStorageClientFeature, "is_enabled", lambda: False)
    monkeypatch.setattr(arguments, "warn_rank_0", lambda message, *args: calls.append(message))

    parsed = arguments.parse_args()

    assert parsed.enable_msc is False
    assert "disable" in calls
    assert any("MSC feature is disabled" in item for item in calls if isinstance(item, str))


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("[0,1]*2", [0, 1, 0, 1]),
        ("([1]+[0])*2", [1, 0, 1, 0]),
        ("[1,0,0]", [1, 0, 0]),
    ],
)
def test_eval_pattern_accepts_safe_list_expressions(pattern, expected):
    assert arguments._eval_pattern(pattern) == expected


def test_eval_pattern_rejects_unsafe_expression():
    with pytest.raises(ValueError, match="Invalid pattern"):
        arguments._eval_pattern("[import('os').system('echo unsafe')]")


def test_frequency_and_tuple_helpers():
    assert arguments.no_rope_freq_type(None) is None
    assert arguments.no_rope_freq_type(2) == 2
    assert arguments.no_rope_freq_type("2") == 2
    assert arguments.no_rope_freq_type("[1,0]") == [1, 0]

    assert arguments.moe_freq_type(3) == 3
    assert arguments.moe_freq_type("3") == 3
    assert arguments.moe_freq_type("[1,0,1]") == [1, 0, 1]

    assert arguments.la_freq_type(None) is None
    assert arguments.la_freq_type(4) == 4
    assert arguments.la_freq_type("4") == 4
    assert arguments.la_freq_type("[1,1,0]") == [1, 1, 0]

    assert arguments.tuple_type(None) is None
    assert arguments.tuple_type((1, 2)) == (1, 2)
    assert arguments.tuple_type("1,2,3") == (1, 2, 3)
    assert arguments.tuple_type("(4,5)") == (4, 5)


def test_validate_args_derives_basic_training_defaults(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)

    validated = arguments.validate_args(args)

    assert validated is args
    assert args.data_parallel_size == 1
    assert args.global_batch_size == 2
    assert args.dataloader_type == "single"
    assert args.encoder_num_layers == 2
    assert args.encoder_seq_length == 8
    assert args.ffn_hidden_size == 64
    assert args.kv_channels == 4
    assert args.params_dtype == torch.bfloat16
    assert args.accumulate_allreduce_grads_in_fp32
    assert args.use_dist_ckpt == (args.ckpt_format != "torch")
    assert args.start_weight_decay == args.weight_decay
    assert args.end_weight_decay == args.weight_decay


def test_validate_args_handles_data_path_split_and_phase_transitions(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(
        monkeypatch,
        [
            "--data-path",
            "1.0",
            "train",
            "--phase-transition-iterations",
            "8, 2, 5",
        ],
    )
    args.mock_data = False

    arguments.validate_args(args)

    assert args.split == "969, 30, 1"
    assert args.phase_transition_iterations == [2, 5, 8]


def test_validate_args_updates_deprecated_cuda_graph_flag(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--enable-cuda-graph"])

    arguments.validate_args(args)

    assert args.cuda_graph_impl == "local"
    assert not hasattr(args, "enable_cuda_graph")


def test_validate_args_resolves_rl_parallel_generation_alias(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--global-batch-size", "6"])
    args.perform_rl_step = True
    args.grpo_group_size = 3
    args.grpo_prompts_per_step = 2
    args.grpo_iterations = 1
    args.rl_parallel_generation_tasks = 2
    args.rl_num_parallel_generations = None
    args.rl_num_parallel_generation_batches = None
    args.rl_generation_batch_size = None
    args.rl_partial_rollouts = True
    args.rl_use_sequence_packing = False

    arguments.validate_args(args)

    assert args.rl_num_parallel_generations == 6
    assert args.rl_parallel_generation_tasks == 2
    assert args.rl_generation_batch_size == 1
    assert args.grpo_samples_per_iteration == 6
    assert not args.rl_enforce_generation_order


def test_validate_args_resolves_rl_generation_batches(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--global-batch-size", "4"])
    args.perform_rl_step = True
    args.grpo_group_size = 2
    args.grpo_prompts_per_step = 2
    args.grpo_iterations = 1
    args.rl_parallel_generation_tasks = None
    args.rl_num_parallel_generations = None
    args.rl_num_parallel_generation_batches = 2
    args.rl_generation_batch_size = None
    args.rl_partial_rollouts = True
    args.rl_use_sequence_packing = False

    arguments.validate_args(args)

    assert args.rl_generation_batch_size == 2
    assert args.rl_parallel_generation_tasks == 4
    assert args.rl_enforce_generation_order
    assert args.grpo_samples_per_iteration == 4


def test_validate_args_moe_deprecated_and_tokenizer_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)
    args.num_experts = 2
    args.moe_ffn_hidden_size = None
    args.moe_router_load_balancing_type = ["aux_loss"]
    args.moe_aux_loss_coeff = [0.1]
    args.no_weight_decay_cond_type = "apply_wd_to_qk_layernorm"
    args.apply_wd_to_qk_layernorm = False
    args.tiktoken_special_tokens = {"<extra>": 1}
    args.tokenizer_special_tokens = None
    args.tokenizer_hf_use_fast = True
    args.tokenizer_hf_include_special_tokens = True

    arguments.validate_args(args)

    assert args.moe_ffn_hidden_size == args.ffn_hidden_size
    assert args.moe_router_load_balancing_type == "aux_loss"
    assert args.moe_aux_loss_coeff == 0.1
    assert args.apply_wd_to_qk_layernorm
    assert args.no_weight_decay_cond_type is None
    assert args.tokenizer_special_tokens == {"<extra>": 1}


def test_validate_args_skip_train_and_async_save_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--ckpt-format", "torch_dist"])
    args.skip_train = True
    args.perform_rl_step = False
    args.no_load_optim = False
    args.async_save = True
    args.use_persistent_ckpt_worker = False

    arguments.validate_args(args)

    assert args.no_load_optim
    assert not args.async_save


def test_validate_args_dtype_and_precision_guard_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)
    args.sequence_parallel = True
    args.tensor_model_parallel_size = 1
    args.main_params_dtype = "bf16"
    args.exp_avg_dtype = "fp32"
    args.exp_avg_sq_dtype = "fp16"
    args.mamba_inference_conv_states_dtype = "auto"
    args.mamba_inference_ssm_states_dtype = None
    args.grad_reduce_in_bf16 = True
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.bias_gelu_fusion = True

    arguments.validate_args(args)

    assert args.sequence_parallel is False
    assert args.main_params_dtype is torch.bfloat16
    assert args.exp_avg_dtype is torch.float32
    assert args.exp_avg_sq_dtype is torch.float16
    assert args.mamba_inference_conv_states_dtype is None
    assert args.accumulate_allreduce_grads_in_fp32 is False
    assert args.bias_gelu_fusion is False


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda args: setattr(args, "non_persistent_ckpt_type", "local"),
            "nvidia_resiliency_ext is required",
        ),
        (
            lambda args: (
                setattr(args, "use_legacy_models", True),
                setattr(args, "ckpt_format", "torch_dist"),
            ),
            "legacy model format",
        ),
        (
            lambda args: (
                setattr(args, "attention_backend", arguments.AttnBackend.local),
                setattr(args, "spec", ["transformer_engine"]),
            ),
            "attention-backend local",
        ),
        (
            lambda args: (
                setattr(args, "perform_rl_step", True),
                setattr(args, "rl_persist_cuda_graphs", True),
                setattr(args, "cuda_graph_impl", "none"),
            ),
            "rl-persist-cuda-graphs",
        ),
        (
            lambda args: (
                setattr(args, "perform_rl_step", True),
                setattr(args, "rl_training_cuda_graphs", True),
                setattr(args, "cuda_graph_impl", "none"),
            ),
            "rl-training-cuda-graphs",
        ),
        (
            lambda args: (
                setattr(args, "perform_rl_step", True),
                setattr(args, "rl_num_parallel_generations", 2),
                setattr(args, "rl_parallel_generation_tasks", 1),
            ),
            "Cannot specify both",
        ),
        (
            lambda args: (
                setattr(args, "perform_rl_step", True),
                setattr(args, "rl_num_parallel_generation_batches", 2),
                setattr(args, "rl_partial_rollouts", False),
            ),
            "rl-num-parallel-generation-batches requires",
        ),
        (
            lambda args: (
                setattr(args, "perform_rl_step", True),
                setattr(args, "rl_use_sequence_packing", True),
                setattr(args, "micro_batch_size", 2),
            ),
            "micro_batch_size must be 1",
        ),
        (
            lambda args: setattr(args, "num_dataset_builder_threads", 0),
            "",
        ),
        (
            lambda args: (
                setattr(args, "train_samples", 10),
                setattr(args, "train_iters", 4),
            ),
            "expected iteration-based training",
        ),
        (
            lambda args: (
                setattr(args, "fp16_lm_cross_entropy", True),
                setattr(args, "fp16", False),
            ),
            "lm cross entropy",
        ),
        (
            lambda args: (
                setattr(args, "fp32_residual_connection", True),
                setattr(args, "bf16", False),
            ),
            "residual connection",
        ),
        (
            lambda args: setattr(args, "no_weight_decay_cond_type", "bad"),
            "Invalid no_weight_decay_cond_type",
        ),
        (
            lambda args: (
                setattr(args, "distribute_saved_activations", True),
                setattr(args, "tensor_model_parallel_size", 1),
            ),
            "distribute",
        ),
        (
            lambda args: (
                setattr(args, "recompute_granularity", "selective"),
                setattr(args, "recompute_method", "block"),
            ),
            "selective",
        ),
        (
            lambda args: (
                setattr(args, "overlap_param_gather", True),
                setattr(args, "use_distributed_optimizer", False),
                setattr(args, "use_megatron_fsdp", False),
                setattr(args, "optimizer", "adam"),
            ),
            "overlap-param-gather",
        ),
        (
            lambda args: (
                setattr(args, "fp4", True),
                setattr(args, "fp8", True),
            ),
            "cannot be used simultaneously",
        ),
        (
            lambda args: (
                setattr(args, "fp4", False),
                setattr(args, "fp4_param_gather", True),
            ),
            "fp4-param-gather",
        ),
        (
            lambda args: (
                setattr(args, "use_rotary_position_embeddings", False),
                setattr(args, "rotary_interleaved", True),
                setattr(args, "use_legacy_models", True),
            ),
            "rotary-interleaved",
        ),
        (
            lambda args: (
                setattr(args, "add_position_embedding", False),
                setattr(args, "position_embedding_type", "learned_absolute"),
            ),
            "no-position-embedding",
        ),
        (
            lambda args: (
                setattr(args, "position_embedding_type", "mrope"),
                setattr(args, "mrope_section", None),
            ),
            "mrope-section",
        ),
        (
            lambda args: (
                setattr(args, "expert_model_parallel_size", 2),
                setattr(args, "num_experts", None),
            ),
            "num_experts",
        ),
        (
            lambda args: (
                setattr(args, "ckpt_format", "torch_dcp"),
                setattr(args, "use_torch_fsdp2", False),
            ),
            "torch_dcp",
        ),
        (
            lambda args: (
                setattr(args, "ckpt_format", "fsdp_dtensor"),
                setattr(args, "use_megatron_fsdp", False),
            ),
            "fsdp_dtensor",
        ),
        (
            lambda args: (
                setattr(args, "mock_data", True),
                setattr(args, "data_path", ["train"]),
            ),
            "single data source",
        ),
        (
            lambda args: (
                setattr(args, "fim_data", True),
                setattr(args, "mock_data", False),
                setattr(args, "fim_rate", None),
            ),
            "fim-rate",
        ),
        (
            lambda args: (
                setattr(args, "deterministic_mode", True),
                setattr(args, "use_flash_attn", True),
            ),
            "Flash attention",
        ),
        (
            lambda args: (
                setattr(args, "load_main_params_from_ckpt", True),
                setattr(args, "no_load_optim", False),
            ),
            "load-main-params",
        ),
        (
            lambda args: (
                setattr(args, "inference_batch_times_seqlen_threshold", 1),
                setattr(args, "pipeline_model_parallel_size", 1),
            ),
            "inference-batch-times-seqlen-threshold",
        ),
        (
            lambda args: (
                setattr(args, "inference_dynamic_batching", True),
                setattr(args, "inference_dynamic_batching_buffer_size_gb", None),
            ),
            "",
        ),
        (
            lambda args: (
                setattr(args, "moe_use_upcycling", True),
                setattr(args, "save", None),
            ),
            "upcycling",
        ),
        (
            lambda args: (
                setattr(args, "skip_train", True),
                setattr(args, "perform_rl_step", True),
                setattr(args, "no_load_optim", True),
                setattr(args, "rl_offload_optimizer_during_inference", True),
            ),
            "rl-offload-optimizer",
        ),
        (
            lambda args: (
                setattr(args, "optimizer", "muon"),
                setattr(args, "overlap_grad_reduce", True),
            ),
            "Muon optimizer",
        ),
        (
            lambda args: (
                setattr(args, "optimizer_cpu_offload", True),
                setattr(args, "use_precision_aware_optimizer", False),
            ),
            "optimizer cpu offload",
        ),
        (
            lambda args: (
                setattr(args, "replication", True),
                setattr(args, "replication_jump", None),
            ),
            "replication-jump",
        ),
        (
            lambda args: (
                setattr(args, "delay_wgrad_compute", True),
                setattr(args, "transformer_impl", "local"),
            ),
            "Delaying wgrad",
        ),
        (
            lambda args: (
                setattr(args, "fine_grained_activation_offloading", True),
                setattr(args, "transformer_impl", "local"),
            ),
            "Fine-grained activation",
        ),
        (
            lambda args: (
                setattr(args, "mtp_num_layers", 1),
                setattr(args, "position_embedding_type", "relative"),
            ),
            "Multi-Token Prediction",
        ),
        (
            lambda args: (
                setattr(args, "multi_latent_attention", True),
                setattr(args, "group_query_attention", True),
            ),
            "mutually exclusive",
        ),
        (
            lambda args: (
                setattr(args, "mla_down_proj_fusion", True),
                setattr(args, "multi_latent_attention", False),
            ),
            "mla-down-proj-fusion",
        ),
        (
            lambda args: (
                setattr(args, "moe_latent_size", 0),
                setattr(args, "num_experts", 2),
            ),
            "greater than zero",
        ),
    ],
)
def test_validate_args_rejects_training_configuration_guard_paths(monkeypatch, mutator, match):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch)
    mutator(args)

    with pytest.raises((AssertionError, RuntimeError, ValueError), match=match):
        arguments.validate_args(args)


def test_validate_args_accepts_pipeline_layout_and_warning_mutation_paths(monkeypatch):
    _patch_validate_environment(monkeypatch)
    messages = []
    monkeypatch.setattr(arguments, "warn_rank_0", lambda message, *args: messages.append(message))
    monkeypatch.setattr(arguments, "print_rank_0", lambda message, *args: messages.append(message))
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: messages.append(("det", enabled)))
    args = _parse_minimal_training_args(
        monkeypatch,
        [
            "--pipeline-model-parallel-size",
            "2",
            "--num-layers-per-virtual-pipeline-stage",
            "1",
            "--overlap-p2p-comm",
            "--ckpt-format",
            "torch_dist",
        ],
    )
    args.world_size = 2
    args.rank = 0
    args.overlap_param_gather = True
    args.overlap_grad_reduce = True
    args.use_distributed_optimizer = True
    args.use_gloo_process_groups = True
    args.ckpt_fully_parallel_save = False
    args.use_dist_ckpt = True
    args.use_dist_ckpt_deprecated = True
    args.dist_ckpt_format_deprecated = True
    args.ckpt_fully_parallel_save_deprecated = True
    args.ckpt_fully_parallel_load = True
    args.ckpt_fully_parallel_load_exchange_algo = "ring"
    args.async_save = True
    args.use_persistent_ckpt_worker = False
    args.fake_process_group = True
    args.moe_token_dispatcher_type = "allgather"
    args.replication = False
    args.replication_jump = 2
    args.apply_query_key_layer_scaling = True
    args.result_rejected_tracker_filename = "tracker.txt"
    args.iterations_to_skip = [1]
    args.fim_data = False
    args.deterministic_mode = True
    args.use_flash_attn = False
    args.cross_entropy_loss_fusion = False
    args.cuda_graph_scope = ["full"]
    args.cuda_graph_impl = "none"

    class _FakeRerun:
        @staticmethod
        def get_skipped_iterations_from_tracker_file(path):
            assert path == "tracker.txt"
            return [3, 5]

    monkeypatch.setattr(arguments, "RerunStateMachine", _FakeRerun)

    arguments.validate_args(args)

    assert args.virtual_pipeline_model_parallel_size is None
    assert args.no_load_optim is False
    assert args.async_save is False
    assert args.replication_jump is None
    assert args.attention_softmax_in_fp32 is True
    assert args.iterations_to_skip == [1, 3, 5]
    assert args.cuda_graph_scope == []
    assert ("det", True) in messages


def test_validate_model_config_args_from_heterogeneous_config_accepts_matching_args():
    config = {
        "hidden_act": "silu",
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "tie_word_embeddings": False,
        "rope_theta": 10000,
        "rope_scaling": {"factor": 2.0},
        "block_configs": [
            {"attention": {"n_heads_in_group": 2}},
            {"attention": {"n_heads_in_group": 2}},
        ],
    }
    args = SimpleNamespace(
        heterogeneous_layers_config_path=None,
        heterogeneous_layers_config_encoded_json=json.dumps(config),
        swiglu=True,
        normalization="RMSNorm",
        group_query_attention=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        use_rope_scaling=True,
        use_rotary_position_embeddings=True,
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        untie_embeddings_and_output_weights=True,
        rotary_base=10000,
        rope_scaling_factor=2.0,
        num_query_groups=2,
    )

    arguments.validate_model_config_args_from_heterogeneous_config(args)


def test_validate_model_config_args_from_heterogeneous_config_rejects_mismatch():
    config = {
        "hidden_act": "silu",
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "tie_word_embeddings": True,
        "rope_theta": 10000,
        "rope_scaling": {"factor": 1.0},
        "block_configs": [{"attention": {"n_heads_in_group": 2}}],
    }
    args = SimpleNamespace(
        heterogeneous_layers_config_path=None,
        heterogeneous_layers_config_encoded_json=json.dumps(config),
        swiglu=False,
        normalization="LayerNorm",
        group_query_attention=False,
        position_embedding_type="learned_absolute",
        rotary_percent=0.5,
        use_rope_scaling=False,
        use_rotary_position_embeddings=False,
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        untie_embeddings_and_output_weights=True,
        rotary_base=5000,
        rope_scaling_factor=2.0,
        num_query_groups=1,
    )

    with pytest.raises(ValueError, match="Arguments differ from heterogeneous config"):
        arguments.validate_model_config_args_from_heterogeneous_config(args)


@dataclasses.dataclass(init=False)
class _FakeTransformerConfig:
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_core_transformer_config_from_args_maps_validated_args(monkeypatch):
    _patch_validate_environment(monkeypatch)
    args = _parse_minimal_training_args(monkeypatch, ["--swiglu", "--group-query-attention"])
    arguments.validate_args(args)

    config = arguments.core_transformer_config_from_args(args, config_class=_FakeTransformerConfig)

    assert config.kwargs["num_layers"] == 2
    assert config.kwargs["hidden_size"] == 16
    assert config.kwargs["num_attention_heads"] == 4
    assert config.kwargs["pipeline_dtype"] == torch.bfloat16
    assert config.kwargs["deallocate_pipeline_outputs"] is True
    assert config.kwargs["batch_p2p_comm"] is True
    assert config.kwargs["gated_linear_unit"] is True
    assert config.kwargs["activation_func"] is torch.nn.functional.silu
    assert config.kwargs["num_query_groups"] == args.num_query_groups
