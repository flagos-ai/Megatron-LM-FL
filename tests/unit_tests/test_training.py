# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training import checkpointing
from megatron.training import training
from megatron.training.training import (
    build_train_valid_test_data_loaders,
    build_train_valid_test_data_iterators,
    checkpoint_and_decide_exit,
    compute_throughputs_and_append_to_progress_log,
    destroy_global_state,
    evaluate,
    evaluate_and_print_results,
    get_model,
    get_start_time_from_progress_log,
    get_train_valid_test_num_samples,
    get_megatron_optimizer_config,
    get_optimizer_param_scheduler,
    dummy_train_step,
    num_floating_point_operations,
    post_training_step_callbacks,
    pretrain,
    preprocess_common_state_dict,
    save_checkpoint_and_time,
    set_startup_timestamps,
    should_disable_forward_pre_hook,
    setup_model_and_optimizer,
    train,
    train_step,
    training_log,
    update_train_iters,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def create_flop_args(**overrides):
    args = SimpleNamespace(
        group_query_attention=True,
        num_query_groups=2,
        num_attention_heads=4,
        num_layers=4,
        num_experts=None,
        moe_layer_freq=None,
        moe_router_topk=1,
        mtp_num_layers=None,
        moe_ffn_hidden_size=None,
        ffn_hidden_size=32,
        moe_latent_size=None,
        moe_shared_expert_intermediate_size=None,
        swiglu=False,
        multi_latent_attention=False,
        q_lora_rank=None,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        kv_lora_rank=4,
        v_head_dim=8,
        hidden_size=16,
        kv_channels=4,
        seq_length=8,
        attention_output_gate=False,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        padded_vocab_size=128,
        hybrid_layer_pattern=None,
        mamba_state_dim=8,
        mamba_head_dim=4,
        mamba_num_groups=2,
        mamba_num_heads=4,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_num_floating_point_operations_dense_transformer_paths():
    dense = create_flop_args(num_experts=None, group_query_attention=False)
    gated = create_flop_args(attention_output_gate=True, swiglu=True, mtp_num_layers=1)

    assert num_floating_point_operations(dense, batch_size=2) > 0
    assert dense.num_query_groups == dense.num_attention_heads
    assert num_floating_point_operations(gated, batch_size=2) > num_floating_point_operations(
        create_flop_args(), batch_size=2
    )


def test_num_floating_point_operations_moe_and_linear_attention_paths():
    moe = create_flop_args(
        num_experts=4,
        moe_layer_freq=[1, 0, 1, 0],
        moe_router_topk=2,
        moe_ffn_hidden_size=24,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=6,
        mtp_num_layers=1,
    )
    linear = create_flop_args(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=2,
    )

    assert num_floating_point_operations(moe, batch_size=1) > 0
    assert num_floating_point_operations(linear, batch_size=1) > 0


def test_num_floating_point_operations_mla_moe_frequency_and_hybrid_paths():
    mla_without_q_lora = create_flop_args(
        group_query_attention=False,
        multi_latent_attention=True,
        q_lora_rank=None,
        kv_lora_rank=4,
    )
    mla_with_q_lora = create_flop_args(
        group_query_attention=False,
        multi_latent_attention=True,
        q_lora_rank=4,
        kv_lora_rank=4,
    )
    moe_frequency = create_flop_args(
        num_experts=4,
        moe_layer_freq=2,
        moe_router_topk=2,
        moe_ffn_hidden_size=24,
        moe_shared_expert_intermediate_size=8,
        mtp_num_layers=2,
    )
    matching_moe_pattern = create_flop_args(
        num_experts=4,
        moe_layer_freq=[1, 0, 1, 0],
        moe_router_topk=2,
        moe_ffn_hidden_size=24,
        moe_shared_expert_intermediate_size=8,
        mtp_num_layers=2,
    )
    hybrid = create_flop_args(
        hybrid_layer_pattern="MG*-E/MM",
        moe_ffn_hidden_size=24,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=6,
        mtp_num_layers=2,
        swiglu=True,
    )

    assert num_floating_point_operations(mla_without_q_lora, batch_size=1) > 0
    assert num_floating_point_operations(mla_without_q_lora, batch_size=1) > num_floating_point_operations(
        mla_with_q_lora, batch_size=1
    )
    assert num_floating_point_operations(moe_frequency, batch_size=1) == num_floating_point_operations(
        matching_moe_pattern, batch_size=1
    )
    assert num_floating_point_operations(hybrid, batch_size=2) > 0


def test_num_floating_point_operations_validates_attention_patterns():
    with pytest.raises(RuntimeError, match="moe-layer-freq"):
        num_floating_point_operations(create_flop_args(num_experts=2, moe_layer_freq="bad"), 1)

    with pytest.raises(AssertionError, match="moe_layer_pattern"):
        num_floating_point_operations(
            create_flop_args(num_experts=2, moe_layer_freq=[1, 0]),
            1,
        )

    with pytest.raises(AssertionError):
        num_floating_point_operations(
            create_flop_args(group_query_attention=True, multi_latent_attention=True),
            1,
        )

    with pytest.raises(ValueError, match="linear_attention_freq is None"):
        num_floating_point_operations(
            create_flop_args(experimental_attention_variant="gated_delta_net"),
            1,
        )

    with pytest.raises(AssertionError, match="linear_attention_pattern"):
        num_floating_point_operations(
            create_flop_args(
                experimental_attention_variant="gated_delta_net",
                linear_attention_freq=[1, 0],
            ),
            1,
        )

    with pytest.raises(ValueError, match="Invalid linear_attention_freq"):
        num_floating_point_operations(
            create_flop_args(
                experimental_attention_variant="gated_delta_net",
                linear_attention_freq=object(),
            ),
            1,
        )


def test_startup_timestamp_and_datetime_helpers(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "_TRAIN_START_TIME", training._TRAIN_START_TIME)
    monkeypatch.setattr(training, "_STARTUP_TIMESTAMPS", dict(training._STARTUP_TIMESTAMPS))
    monkeypatch.setattr(training.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))

    set_startup_timestamps(program_start=10.0, main_entry=11.0)
    training.print_datetime("fixed", override_timestamp=0)

    assert training._STARTUP_TIMESTAMPS["program_start"] == 10.0
    assert training._STARTUP_TIMESTAMPS["main_entry"] == 11.0
    assert "barrier" in calls
    assert any("fixed" in item[1] and "1970" in item[1] for item in calls if isinstance(item, tuple))


def test_print_datetime_uses_current_time_when_no_override(monkeypatch):
    calls = []
    monkeypatch.setattr(training.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))

    training.print_datetime("current")

    assert calls[0] == "barrier"
    assert len(calls) == 2
    assert calls[1][0] == "print"
    assert calls[1][1].startswith("[current] datetime: ")


def test_destroy_global_state_delegates_to_subsystems(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "destroy_global_vars", lambda: calls.append("global-vars"))
    monkeypatch.setattr(training, "destroy_num_microbatches_calculator", lambda: calls.append("microbatches"))
    monkeypatch.setattr(training, "destroy_global_memory_buffer", lambda: calls.append("memory-buffer"))
    monkeypatch.setattr(training.SymmetricMemoryManager, "destroy", lambda: calls.append("symmetric-memory"))
    monkeypatch.setattr(training, "destroy_model_parallel", lambda: calls.append("model-parallel"))
    monkeypatch.setattr(training, "destroy_rerun_state_machine", lambda: calls.append("rerun"))

    destroy_global_state()

    assert calls == [
        "global-vars",
        "microbatches",
        "memory-buffer",
        "symmetric-memory",
        "model-parallel",
        "rerun",
    ]


def test_get_start_time_from_progress_log_handles_async_and_world_size_reset(monkeypatch, tmp_path):
    progress = tmp_path / "progress.txt"
    progress.write_text(
        "\n".join(
            [
                "2026-05-25 10:00:00\tJob ID: old\t# GPUs: 2\tStarting job",
                "2026-05-25 10:01:00\tJob ID: old\t# GPUs: 2\tSaved checkpoint\t"
                "Iteration: 1\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 1.00e+03\tTokens: 1",
                "2026-05-25 10:02:00\tJob ID: new\t# GPUs: 4\tStarting job",
                "2026-05-25 10:03:00\tJob ID: new\t# GPUs: 4\tSaving async checkpoint\t"
                "Iteration: 2\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 2.00e+03\tTokens: 1",
                "2026-05-25 10:04:00\tJob ID: new\t# GPUs: 4\tSaved async checkpoint\t"
                "Iteration: 2\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 0.00e+00\tTokens: 1",
                "2026-05-25 10:05:00\tJob ID: same\t# GPUs: 4\tStarting job",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(training, "get_args", lambda: SimpleNamespace(save=str(tmp_path), world_size=4))
    monkeypatch.setattr(training, "print_rank_0", lambda *args, **kwargs: None)

    start_time, start_flops = get_start_time_from_progress_log()

    assert start_time.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-25 10:02:00"
    assert start_flops == 1000.0


def test_get_start_time_from_progress_log_handles_direct_async_commit_and_missing_match(
    monkeypatch, tmp_path
):
    progress = tmp_path / "progress.txt"
    progress.write_text(
        "\n".join(
            [
                "2026-05-25 10:00:00\tJob ID: base\t# GPUs: 8\tStarting job",
                "2026-05-25 10:01:00\tJob ID: base\t# GPUs: 8\tSaved checkpoint\t"
                "Iteration: 1\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 5.00e+03\tTokens: 1",
                "2026-05-25 10:02:00\tJob ID: base\t# GPUs: 8\tSaved async checkpoint\t"
                "Iteration: 2\tJob throughput: 1\tCumulative throughput: 1\t"
                "Floating-point operations: 0.00e+00\tTokens: 1",
                "2026-05-25 10:03:00\tJob ID: next\t# GPUs: 4\tStarting job",
                "2026-05-25 10:04:00\tJob ID: next\t# GPUs: 8\tStarting job",
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(save=str(tmp_path), world_size=8)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "print_rank_0", lambda *args, **kwargs: None)

    start_time, start_flops = get_start_time_from_progress_log()
    assert start_time.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-25 10:04:00"
    assert start_flops == 5000.0

    args.world_size = 16
    with pytest.raises(AssertionError, match="Starting job"):
        get_start_time_from_progress_log()


def test_preprocess_common_state_dict_strips_rank_and_sorts_optimizer_groups():
    common_state = {
        "args": SimpleNamespace(
            rank=3,
            local_rank=1,
            use_distributed_optimizer=True,
        ),
        "optimizer": {
            "param_state": {},
            "optimizer": {
                "param_groups": [
                    {
                        "wd_mult": 1.0,
                        "lr_mult": 2.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                    {
                        "wd_mult": 1.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                ]
            },
        },
    }

    processed = preprocess_common_state_dict(common_state)

    assert "rank" not in processed["args"]
    assert "local_rank" not in processed["args"]
    assert "param_state" not in processed["optimizer"]
    assert [group["lr_mult"] for group in processed["optimizer"]["optimizer"]["param_groups"]] == [1.0, 2.0]
    assert "rank" in vars(common_state["args"])


def test_preprocess_common_state_dict_handles_chained_optimizers():
    optimizer_state = {
        0: {
            "param_state": {},
            "optimizer": {
                "param_groups": [
                    {
                        "wd_mult": 2.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                    {
                        "wd_mult": 1.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                        "is_vision_model_param": False,
                        "is_engram_parallel": False,
                    },
                ]
            },
        },
        2: {"optimizer": {"param_groups": []}},
    }
    common_state = {
        "args": SimpleNamespace(use_distributed_optimizer=True, rank=7),
        "optimizer": optimizer_state,
    }

    processed = preprocess_common_state_dict(common_state)

    assert "rank" not in processed["args"]
    assert "param_state" not in processed["optimizer"][0]
    assert [group["wd_mult"] for group in processed["optimizer"][0]["optimizer"]["param_groups"]] == [
        1.0,
        2.0,
    ]


def test_preprocess_common_state_dict_handles_optimizer_early_return_paths():
    common_state = {
        "args": SimpleNamespace(use_distributed_optimizer=True, rank=4, local_rank=0),
        "optimizer": {
            0: {
                "param_state": {},
            },
            1: {
                "param_state": {},
                "optimizer": {
                    "state": "kept",
                },
            },
        },
    }

    processed = preprocess_common_state_dict(common_state)

    assert "rank" not in processed["args"]
    assert "local_rank" not in processed["args"]
    assert processed["optimizer"][0] == {}
    assert processed["optimizer"][1] == {"optimizer": {"state": "kept"}}
    assert "param_state" in common_state["optimizer"][0]


def test_get_train_valid_test_num_samples_iteration_sample_and_phase_paths(monkeypatch):
    args = SimpleNamespace(
        train_samples=None,
        train_iters=9,
        global_batch_size=4,
        full_validation=False,
        skip_train=False,
        eval_interval=3,
        eval_iters=2,
        phase_transition_iterations=None,
        iteration=0,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)

    assert get_train_valid_test_num_samples() == (36, 32, 8)

    args.train_samples = 100
    args.full_validation = True
    assert get_train_valid_test_num_samples() == (100, None, 8)

    args.full_validation = False
    args.skip_train = True
    assert get_train_valid_test_num_samples() == (100, 8, 8)

    args.skip_train = False
    args.phase_transition_iterations = [3, 6]
    args.iteration = 4
    assert get_train_valid_test_num_samples()[0] == 12

    args.iteration = 0
    assert get_train_valid_test_num_samples()[0] == 12

    args.phase_transition_iterations = None
    args.train_samples = 1
    args.train_iters = None
    args.full_validation = False
    args.skip_train = False
    with pytest.raises(AssertionError):
        get_train_valid_test_num_samples()


def test_build_train_valid_test_datasets_prints_targets_and_delegates(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))

    result = training.build_train_valid_test_datasets(
        lambda samples: calls.append(("provider", samples)) or ("train", "valid", "test"),
        train_valid_test_num_samples=(10, None, 4),
    )

    assert result == ("train", "valid", "test")
    assert ("provider", (10, None, 4)) in calls
    assert ("print", "    validation: None") in calls


def test_cyclic_iter_restarts_after_exhausting_iterable():
    iterator = training.cyclic_iter([1, 2])

    assert [next(iterator), next(iterator), next(iterator), next(iterator)] == [1, 2, 1, 2]


def test_update_train_iters_constant_and_rampup(monkeypatch):
    printed = []
    monkeypatch.setattr(training, "print_rank_0", lambda message: printed.append(message))

    already_iteration_based = SimpleNamespace(
        train_iters=7,
        rampup_batch_size=None,
        train_samples=100,
        global_batch_size=8,
    )
    update_train_iters(already_iteration_based)
    assert already_iteration_based.train_iters == 7
    assert printed == []

    constant = SimpleNamespace(
        train_iters=None,
        rampup_batch_size=None,
        train_samples=100,
        global_batch_size=8,
    )
    update_train_iters(constant)
    assert constant.train_iters == 12
    assert printed == ["setting training iterations to 12"]

    calls = []
    rampup = SimpleNamespace(
        train_iters=None,
        rampup_batch_size=(2, 2, 4),
        train_samples=12,
        global_batch_size=4,
    )
    monkeypatch.setattr(training, "update_num_microbatches", lambda consumed, consistency_check=False: calls.append(consumed))
    monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 2)
    update_train_iters(rampup)

    assert rampup.train_iters == 4
    assert calls == [0, 2, 4, 0]
    assert printed[-1] == "setting training iterations to 4"


def test_pretrain_skip_train_runs_validation_test_and_shutdown(monkeypatch):
    calls = []
    real_tensor = torch.tensor
    args = SimpleNamespace(
        fine_grained_activation_offloading=False,
        log_progress=False,
        non_persistent_ckpt_type=None,
        perform_rl_step=False,
        virtual_pipeline_model_parallel_size=None,
        skip_train=True,
        iteration=4,
        world_size=1,
        train_iters=None,
        do_train=True,
        do_valid=True,
        do_test=True,
        dataloader_type=None,
        save=None,
    )

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    class FakeTimer:
        def start(self, barrier=False):
            calls.append(("timer-start", barrier))

        def stop(self):
            calls.append("timer-stop")

        def set_elapsed(self, value):
            calls.append(("timer-set", value))

    class FakeTimers:
        def __call__(self, name, log_level=None):
            calls.append(("timer", name, log_level))
            return FakeTimer()

        def log(self, names, barrier=False):
            calls.append(("timer-log", tuple(names), barrier))

    class FakeWandb:
        def __init__(self):
            self.config = SimpleNamespace(
                update=lambda values: calls.append(("wandb-config", values))
            )

        def finish(self):
            calls.append("wandb-finish")

    model = [SimpleNamespace()]
    config = SimpleNamespace()
    monkeypatch.setitem(training._STARTUP_TIMESTAMPS, "program_start", None)
    monkeypatch.setitem(training._STARTUP_TIMESTAMPS, "main_entry", None)
    monkeypatch.setitem(training._STARTUP_TIMESTAMPS, "pretrain_entry", None)
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training.torch.distributed, "all_reduce", lambda tensor, op=None: calls.append(("all-reduce", tensor.item())))
    monkeypatch.setattr(training, "initialize_megatron", lambda **kwargs: calls.append(("init", kwargs["store"])))
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "set_jit_fusion_options", lambda: calls.append("jit"))
    monkeypatch.setattr(training, "set_startup_timestamps", lambda **kwargs: calls.append(("startup", kwargs.get("program_start"))))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print0", message)))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("datetime", items[0])))
    monkeypatch.setattr(training.one_logger_utils, "get_timestamp_in_ms", lambda: 123)
    monkeypatch.setattr(training.one_logger_utils, "on_pretrain_start", lambda: calls.append("pretrain-start"))
    monkeypatch.setattr(training.one_logger_utils, "track_config_flags", lambda *items: calls.append(("flags", items)))
    monkeypatch.setattr(training.one_logger_utils, "finish", lambda: calls.append("one-finish"))
    monkeypatch.setattr(training, "setup_model_and_optimizer", lambda *items, **kwargs: (model, "optimizer", "scheduler"))
    monkeypatch.setattr(training, "get_model_config", lambda model: config)
    monkeypatch.setattr(training, "build_train_valid_test_data_iterators", lambda provider: ("train-iter", "valid-iter", "test-iter"))
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_wandb_writer", lambda: FakeWandb())
    monkeypatch.setattr(training.ft_integration, "setup", lambda: calls.append("ft-setup"))
    monkeypatch.setattr(training.ft_integration, "on_checkpointing_start", lambda: calls.append("ft-ckpt-start"))
    monkeypatch.setattr(training.ft_integration, "on_checkpointing_end", lambda **kwargs: calls.append(("ft-ckpt-end", kwargs.get("is_async_finalization"))))
    monkeypatch.setattr(training.ft_integration, "shutdown", lambda: calls.append("ft-shutdown"))
    monkeypatch.setattr(training, "maybe_finalize_async_save", lambda **kwargs: calls.append(("finalize", kwargs)))
    monkeypatch.setattr(training, "evaluate_and_print_results", lambda *items, **kwargs: calls.append(("eval-print", items[0], kwargs["write_to_tensorboard"])))

    pretrain(
        lambda samples: None,
        lambda: None,
        training.ModelType.encoder_or_decoder,
        lambda *_: None,
        store="store",
    )

    assert ("init", "store") in calls
    assert "jit" in calls
    assert ("flags", (None, True, True, True, True, None)) in calls
    assert ("eval-print", "iteration 4 on validation set", False) in calls
    assert ("eval-print", "iteration 4 on test set", False) in calls
    assert "wandb-finish" in calls
    assert "ft-shutdown" in calls
    assert "one-finish" in calls


def test_pretrain_rejects_rl_inference_weight_offload_without_separate_model(monkeypatch):
    calls = []
    real_tensor = torch.tensor
    args = SimpleNamespace(
        fine_grained_activation_offloading=False,
        log_progress=False,
        non_persistent_ckpt_type=None,
        perform_rl_step=True,
        rl_inference_tensor_model_parallel_size=None,
        rl_inference_pipeline_model_parallel_size=None,
        rl_inference_expert_model_parallel_size=None,
        rl_inference_expert_tensor_model_parallel_size=None,
        rl_offload_inference_model_weights_when_idle=True,
        virtual_pipeline_model_parallel_size=None,
    )

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    class FakeTimer:
        def start(self, barrier=False):
            calls.append(("timer-start", barrier))

        def stop(self):
            calls.append("timer-stop")

        def set_elapsed(self, value):
            calls.append(("timer-set", value))

    class FakeTimers:
        def __call__(self, name, log_level=None):
            calls.append(("timer", name, log_level))
            return FakeTimer()

        def log(self, names, barrier=False):
            calls.append(("timer-log", tuple(names), barrier))

    monkeypatch.setitem(training._STARTUP_TIMESTAMPS, "program_start", None)
    monkeypatch.setitem(training._STARTUP_TIMESTAMPS, "main_entry", None)
    monkeypatch.setitem(training._STARTUP_TIMESTAMPS, "pretrain_entry", None)
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(
        training.torch.distributed,
        "all_reduce",
        lambda tensor, op=None: calls.append(("all-reduce", tensor.item())),
    )
    monkeypatch.setattr(training, "initialize_megatron", lambda **kwargs: calls.append("init"))
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "set_jit_fusion_options", lambda: calls.append("jit"))
    monkeypatch.setattr(training, "set_startup_timestamps", lambda **kwargs: calls.append("startup"))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print0", message)))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("datetime", items[0])))
    monkeypatch.setattr(training.one_logger_utils, "get_timestamp_in_ms", lambda: 123)
    monkeypatch.setattr(training.one_logger_utils, "on_pretrain_start", lambda: calls.append("pretrain-start"))
    monkeypatch.setattr(training.ft_integration, "setup", lambda: calls.append("ft-setup"))
    monkeypatch.setattr(training, "setup_model_and_optimizer", lambda *items, **kwargs: ([SimpleNamespace()], None, None))
    monkeypatch.setattr(training, "get_model_config", lambda model: SimpleNamespace())

    with pytest.raises(ValueError, match="requires a separate inference model"):
        pretrain(
            lambda samples: None,
            lambda: None,
            training.ModelType.encoder_or_decoder,
            lambda *_: None,
        )

    assert "init" in calls
    assert "ft-setup" in calls
    assert "jit" in calls
    assert "pretrain-start" in calls
    assert ("datetime", "after model, optimizer, and learning rate scheduler are built") in calls


def test_checkpoint_and_decide_exit_save_and_iteration_paths(monkeypatch):
    calls = []
    args = SimpleNamespace(
        exit_signal_handler=False,
        save="/tmp/checkpoints",
        save_interval=5,
        non_persistent_save_interval=None,
        exit_duration_in_mins=None,
        exit_interval=None,
        phase_transition_iterations=None,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(training, "save_checkpoint_and_time", lambda *items, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("print", items)))

    assert checkpoint_and_decide_exit(None, None, None, 10, 0.0, {}, None) is False
    assert calls == [{"train_data_iterator": None}]

    args.save_interval = None
    args.exit_interval = 3
    assert checkpoint_and_decide_exit(None, None, None, 6, 0.0, {}, None) is True
    assert ("print", ("exiting program at iteration 6",)) in calls


def test_checkpoint_and_decide_exit_signal_and_non_persistent_paths(monkeypatch):
    calls = []
    args = SimpleNamespace(
        exit_signal_handler=True,
        save="/tmp/checkpoints",
        save_interval=None,
        non_persistent_save_interval=4,
        exit_duration_in_mins=None,
        exit_interval=None,
        phase_transition_iterations=None,
    )
    signal_handler = SimpleNamespace(signals_received=lambda: [15])
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(training, "get_signal_handler", lambda: signal_handler)
    monkeypatch.setattr(training, "save_checkpoint_and_time", lambda *items, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("print", items)))

    assert checkpoint_and_decide_exit("m", "o", "s", 4, 123.0, {"ctx": True}, "iter") is True
    assert calls[0] == {"train_data_iterator": "iter"}
    assert ("print", ("exiting program after receiving SIGTERM.",)) in calls

    calls.clear()
    args.exit_signal_handler = False
    assert checkpoint_and_decide_exit("m", "o", "s", 8, 123.0, {}, "iter") is False
    assert calls == [{"non_persistent_ckpt": True, "train_data_iterator": "iter"}]


def test_checkpoint_and_decide_exit_duration_and_phase_transition_paths(monkeypatch):
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    args = SimpleNamespace(
        exit_signal_handler=False,
        save="/tmp/checkpoints",
        save_interval=5,
        non_persistent_save_interval=None,
        exit_duration_in_mins=1,
        exit_interval=None,
        phase_transition_iterations=None,
    )
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training.torch.distributed, "all_reduce", lambda tensor, op=None: calls.append(("all-reduce", tensor.item())))
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(training, "_TRAIN_START_TIME", 0.0)
    monkeypatch.setattr(training.time, "time", lambda: 120.0)
    monkeypatch.setattr(training, "save_checkpoint_and_time", lambda *items, **kwargs: calls.append(("save", kwargs)))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("print", items)))

    assert checkpoint_and_decide_exit("m", "o", "s", 10, 123.0, {}, "iter") is True
    assert len([item for item in calls if item[0] == "save"]) == 1
    assert any(item[0] == "print" and "exiting program after" in item[1][0] for item in calls)

    calls.clear()
    args.save_interval = None
    args.exit_duration_in_mins = None
    args.phase_transition_iterations = [6]
    assert checkpoint_and_decide_exit("m", "o", "s", 6, 123.0, {}, "iter") is True
    assert calls == [("save", {"train_data_iterator": "iter"}), ("print", ("exiting program at iteration 6",))]


def test_compute_throughputs_and_append_to_progress_log_formats_checkpoint_line(monkeypatch):
    calls = []
    args = SimpleNamespace(
        save="/tmp/checkpoints",
        num_floating_point_operations_so_far=1.0e12,
        world_size=2,
        consumed_train_samples=500,
        seq_length=1024,
        async_save=True,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "_TRAIN_START_TIME", 90.0)
    monkeypatch.setattr(training.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        training,
        "get_start_time_from_progress_log",
        lambda: (training.datetime.fromtimestamp(80), 0.5e12),
    )
    monkeypatch.setattr(training, "append_to_progress_log", lambda message: calls.append(message))

    compute_throughputs_and_append_to_progress_log(7, 3.0e12)

    assert len(calls) == 1
    assert calls[0].startswith("Saving async checkpoint\tIteration: 7")
    assert "Floating-point operations: 3.00e+12" in calls[0]
    assert "Tokens (in billions): 0.00" in calls[0]

    args.save = None
    calls.clear()
    compute_throughputs_and_append_to_progress_log(8, 4.0e12)
    assert calls == []


def test_compute_throughputs_and_append_to_progress_log_formats_sync_checkpoint(monkeypatch):
    calls = []
    args = SimpleNamespace(
        save="/tmp/checkpoints",
        num_floating_point_operations_so_far=1.0e12,
        world_size=4,
        consumed_train_samples=2_000_000,
        seq_length=2048,
        async_save=False,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "_TRAIN_START_TIME", 50.0)
    monkeypatch.setattr(training.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        training,
        "get_start_time_from_progress_log",
        lambda: (training.datetime.fromtimestamp(75), 0.25e12),
    )
    monkeypatch.setattr(training, "append_to_progress_log", lambda message: calls.append(message))

    compute_throughputs_and_append_to_progress_log(11, 5.0e12)

    assert len(calls) == 1
    assert calls[0].startswith("Saved checkpoint\tIteration: 11")
    assert "Job throughput:" in calls[0]
    assert "Cumulative throughput:" in calls[0]
    assert "Tokens (in billions): 4.10" in calls[0]


def test_save_checkpoint_and_time_covers_persistent_progress_and_cleanup(monkeypatch):
    calls = []
    args = SimpleNamespace(
        use_megatron_fsdp=False,
        use_distributed_optimizer=True,
        optimizer="adam",
        overlap_param_gather=True,
        fp8=True,
        log_progress=True,
        log_energy=True,
        async_save=False,
    )

    class FakeTimer:
        def __init__(self, name):
            self.name = name

        def start(self, barrier=False):
            calls.append(("timer-start", self.name, barrier))

        def stop(self, barrier=False):
            calls.append(("timer-stop", self.name, barrier))

        def elapsed(self):
            calls.append(("timer-elapsed", self.name))
            return 1.25

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer(name)

        def log(self, names):
            calls.append(("timer-log", tuple(names)))

    class FakeEnergyMonitor:
        def pause(self):
            calls.append("energy-pause")

        def resume(self):
            calls.append("energy-resume")

    class FakeModelChunk:
        def free_overlap_buffers(self):
            calls.append("free-overlap")

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_energy_monitor", lambda: FakeEnergyMonitor())
    monkeypatch.setattr(training, "force_param_sync", lambda model: calls.append(("force-sync", len(model))))
    monkeypatch.setattr(training.torch.cuda, "empty_cache", lambda: calls.append("empty-cache"))
    monkeypatch.setattr(training, "report_memory", lambda message: calls.append(("memory", message)))
    monkeypatch.setattr(
        training,
        "save_checkpoint",
        lambda *items, **kwargs: calls.append(("save", items[0], kwargs["non_persistent_ckpt"], kwargs["train_data_iterator"])),
    )
    monkeypatch.setattr(training.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: calls.append(("e2e", items)))
    monkeypatch.setattr(
        training.one_logger_utils,
        "on_save_checkpoint_end",
        lambda duration, iteration, async_save: calls.append(("save-end", duration, iteration, async_save)),
    )
    monkeypatch.setattr(
        training,
        "compute_throughputs_and_append_to_progress_log",
        lambda iteration, flops: calls.append(("progress", iteration, flops)),
    )
    monkeypatch.setattr(training, "num_checkpoints_memory_reported", 0)

    save_checkpoint_and_time(
        12,
        [FakeModelChunk()],
        "optimizer",
        "scheduler",
        123.0,
        {"ctx": True},
        train_data_iterator="train-iter",
    )

    assert ("force-sync", 1) in calls
    assert "free-overlap" in calls
    assert ("save", 12, False, "train-iter") in calls
    assert ("progress", 12, 123.0) in calls
    assert "energy-pause" in calls and "energy-resume" in calls
    assert "gc" in calls


def test_save_checkpoint_and_time_non_persistent_skips_progress(monkeypatch):
    calls = []
    args = SimpleNamespace(
        use_megatron_fsdp=False,
        use_distributed_optimizer=False,
        optimizer="adam",
        overlap_param_gather=False,
        fp8=False,
        log_progress=True,
        log_energy=False,
        async_save=True,
    )

    class FakeTimer:
        def __init__(self, name):
            self.name = name

        def start(self, barrier=False):
            calls.append(("start", self.name, barrier))

        def stop(self, barrier=False):
            calls.append(("stop", self.name, barrier))

        def elapsed(self):
            return 0.5

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer(name)

        def log(self, names):
            calls.append(("log", tuple(names)))

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_energy_monitor", lambda: SimpleNamespace(pause=lambda: None, resume=lambda: None))
    monkeypatch.setattr(training.torch.cuda, "empty_cache", lambda: calls.append("empty-cache"))
    monkeypatch.setattr(training, "report_memory", lambda message: calls.append(("memory", message)))
    monkeypatch.setattr(training, "save_checkpoint", lambda *items, **kwargs: calls.append(("save", kwargs["non_persistent_ckpt"])))
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: None)
    monkeypatch.setattr(training.one_logger_utils, "on_save_checkpoint_end", lambda *items: calls.append(("save-end", items)))
    monkeypatch.setattr(training, "compute_throughputs_and_append_to_progress_log", lambda *items: calls.append("progress"))
    monkeypatch.setattr(training, "num_checkpoints_memory_reported", training.MAX_NUM_CHECKPOINTS_MEMORY_REPORTED)

    save_checkpoint_and_time(3, [object()], None, None, 10.0, {}, non_persistent_ckpt=True)

    assert ("save", True) in calls
    assert ("start", "save-checkpoint-non-persistent", True) in calls
    assert "progress" not in calls
    assert not any(item[0] == "memory" for item in calls if isinstance(item, tuple))


def test_evaluate_reduces_losses_and_restores_state(monkeypatch):
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    class FakeTimer:
        def start(self, barrier=False):
            calls.append(("timer-start", barrier))

        def stop(self):
            calls.append("timer-stop")

    class FakeTimers:
        def __call__(self, name, log_level=None):
            calls.append(("timer", name, log_level))
            return FakeTimer()

        def log(self, names):
            calls.append(("timer-log", tuple(names)))

    class FakeModel:
        def eval(self):
            calls.append("eval")

        def train(self):
            calls.append("train")

    class FakeRerunStateMachine:
        def get_mode(self):
            calls.append("get-mode")
            return "original-mode"

        def set_mode(self, mode):
            calls.append(("set-mode", mode))

    args = SimpleNamespace(
        vision_pretraining=False,
        vision_pretraining_type=None,
        global_batch_size=4,
        micro_batch_size=2,
        data_parallel_size=1,
        cuda_graph_impl=None,
        cuda_graph_scope=[],
        cuda_graph_warmup_steps=0,
        seq_length=8,
        decoder_seq_length=None,
        eval_iters=2,
        empty_unused_memory_level=0,
        sft=False,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
    )

    def fake_forward_backward_func(**kwargs):
        calls.append(("forward", kwargs["num_microbatches"], kwargs["forward_only"], kwargs.get("collect_non_loss_data", False)))
        return [
            {"lm loss": torch.tensor([2.0, 4.0])},
            {"lm loss": torch.tensor([4.0, 4.0])},
        ]

    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunStateMachine())
    monkeypatch.setattr(training, "get_forward_backward_func", lambda: fake_forward_backward_func)
    monkeypatch.setattr(training.ft_integration, "on_eval_step_start", lambda: calls.append("eval-start"))
    monkeypatch.setattr(training.ft_integration, "on_eval_step_end", lambda: calls.append("eval-end"))
    monkeypatch.setattr(training.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(training.mpu, "get_data_parallel_group", lambda with_context_parallel=True: "dp")
    monkeypatch.setattr(training.torch.distributed, "all_reduce", lambda tensor, group=None, op=None: calls.append(("all-reduce", group)))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))

    loss_dict, collected, timelimit = evaluate(
        lambda *_: None,
        "data-iter",
        [FakeModel()],
        process_non_loss_data_func=None,
        config=SimpleNamespace(timers="set"),
        verbose=True,
        non_loss_data_func=lambda model: ("non-loss", len(model)),
        eval_iters=2,
    )

    assert timelimit is False
    assert collected == ("non-loss", 1)
    assert loss_dict["lm loss"].item() == pytest.approx(0.75)
    assert args.consumed_valid_samples == 8
    assert ("set-mode", training.RerunMode.DISABLED) in calls
    assert ("set-mode", "original-mode") in calls
    assert "eval" in calls and "train" in calls


def test_evaluate_handles_sft_and_timelimit_paths(monkeypatch):
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    class FakeRerunStateMachine:
        def get_mode(self):
            return "original-mode"

        def set_mode(self, mode):
            calls.append(("mode", mode))

    class FakeTimer:
        def start(self, barrier=False):
            pass

        def stop(self):
            pass

    class FakeTimers:
        def __call__(self, *items, **kwargs):
            return FakeTimer()

        def log(self, names):
            calls.append(("log", tuple(names)))

    args = SimpleNamespace(
        vision_pretraining=False,
        vision_pretraining_type=None,
        global_batch_size=2,
        micro_batch_size=1,
        data_parallel_size=1,
        cuda_graph_impl=None,
        cuda_graph_scope=[],
        cuda_graph_warmup_steps=0,
        seq_length=8,
        decoder_seq_length=None,
        eval_iters=1,
        empty_unused_memory_level=1,
        sft=True,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
    )

    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunStateMachine())
    monkeypatch.setattr(
        training,
        "get_forward_backward_func",
        lambda: lambda **kwargs: [{"lm loss": torch.tensor([4.0, 2.0])}],
    )
    monkeypatch.setattr(training.ft_integration, "on_eval_step_start", lambda: None)
    monkeypatch.setattr(training.ft_integration, "on_eval_step_end", lambda: None)
    monkeypatch.setattr(training.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(training.mpu, "get_data_parallel_group", lambda with_context_parallel=True: "dp")
    monkeypatch.setattr(training.torch.distributed, "all_reduce", lambda tensor, group=None, op=None: None)
    monkeypatch.setattr(training.torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(training.torch.cuda, "empty_cache", lambda: calls.append("empty-cache"))

    loss_dict, _, timelimit = evaluate(lambda *_: None, None, [SimpleNamespace(eval=lambda: None, train=lambda: None)], None, SimpleNamespace())

    assert timelimit is False
    assert loss_dict["lm loss"].item() == pytest.approx(2.0)
    assert "empty-cache" in calls

    args.exit_duration_in_mins = -1
    monkeypatch.setattr(training, "_TRAIN_START_TIME", 0.0)
    loss_dict, collected, timelimit = evaluate(lambda *_: None, None, [SimpleNamespace(eval=lambda: None, train=lambda: None)], None, SimpleNamespace())

    assert loss_dict is None
    assert collected is None
    assert timelimit is True
    assert ("mode", "original-mode") in calls


def test_evaluate_and_print_results_full_validation_multiple_sets(monkeypatch):
    calls = []
    real_tensor = torch.tensor
    args = SimpleNamespace(
        multiple_validation_sets=True,
        full_validation=True,
        eval_iters=[2, 3],
        consumed_train_samples=64,
        log_validation_ppl_to_tensorboard=True,
    )

    class FakeWriter:
        def add_scalar(self, name, value, iteration):
            calls.append(("scalar", name, value, iteration))

    class FakeWandb:
        def log(self, payload, iteration=None):
            calls.append(("wandb", tuple(sorted(payload)), iteration))

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    def fake_evaluate(*items, **kwargs):
        calls.append(("evaluate", items[2], kwargs["eval_iters"]))
        return {"lm loss": torch.tensor(0.5 + kwargs["eval_iters"])}, "collected", False

    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: FakeWriter())
    monkeypatch.setattr(training, "get_wandb_writer", lambda: FakeWandb())
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(training.torch.distributed, "broadcast", lambda tensor, src: calls.append(("broadcast", tensor.tolist(), src)))
    monkeypatch.setattr(training, "evaluate", fake_evaluate)
    monkeypatch.setattr(training, "is_last_rank", lambda: True)
    monkeypatch.setattr(training, "print_rank_last", lambda message: calls.append(("print", message)))

    processed = []
    evaluate_and_print_results(
        "iteration 5",
        lambda *_: None,
        ["valid-a", "valid-b"],
        "model",
        5,
        lambda data, iteration, writer: processed.append((data, iteration, writer is not None)),
        SimpleNamespace(),
        verbose=True,
    )

    assert args.eval_iters == [2, 3]
    assert ("evaluate", "model", 2) in calls
    assert ("evaluate", "model", 3) in calls
    assert processed == [("collected", 5, True), ("collected", 5, True)]
    assert any(item[:2] == ("scalar", "lm loss validation-0") for item in calls)
    assert any(item[0] == "print" and "validation-1 loss" in item[1] for item in calls)


def test_evaluate_and_print_results_timelimit_returns_early(monkeypatch):
    calls = []
    args = SimpleNamespace(
        multiple_validation_sets=False,
        full_validation=False,
        eval_iters=1,
        consumed_train_samples=0,
        log_validation_ppl_to_tensorboard=False,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: object())
    monkeypatch.setattr(training, "get_wandb_writer", lambda: None)
    monkeypatch.setattr(training, "evaluate", lambda *items, **kwargs: ({}, None, True))
    monkeypatch.setattr(training, "print_rank_last", lambda message: calls.append(message))

    evaluate_and_print_results(
        "iteration 1",
        lambda *_: None,
        "valid",
        "model",
        1,
        lambda *items: calls.append("processed"),
        SimpleNamespace(),
    )

    assert calls == []


def test_build_train_valid_test_data_loaders_regular_and_rl_paths(monkeypatch):
    real_tensor = torch.tensor
    args = SimpleNamespace(
        iteration=2,
        train_samples=64,
        train_iters=4,
        eval_interval=2,
        eval_iters=1,
        global_batch_size=8,
        consumed_train_samples=16,
        consumed_valid_samples=0,
        phase_transition_iterations=[1],
        perform_rl_step=False,
        skip_train=False,
        full_validation=True,
        multiple_validation_sets=True,
    )
    loader_calls = []
    provider_calls = []

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    def provider(samples):
        provider_calls.append(samples)
        return "train", ["valid-a", "valid-b"], "test"

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "print_rank_0", lambda *items, **kwargs: None)
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training.torch.distributed, "broadcast", lambda tensor, src: None)
    monkeypatch.setattr(
        training,
        "build_pretraining_data_loader",
        lambda dataset, consumed: loader_calls.append((dataset, consumed)) or (dataset, consumed),
    )

    train_loader, valid_loaders, test_loader = build_train_valid_test_data_loaders(provider)

    assert provider_calls == [(56, None, 8)]
    assert train_loader == ("train", 8)
    assert valid_loaders == [("valid-a", 0), ("valid-b", 0)]
    assert test_loader == ("test", 0)
    assert args.do_train and args.do_valid and args.do_test

    args.perform_rl_step = True
    args.train_iters = 0
    args.eval_iters = 0
    args.full_validation = False
    args.do_train = args.do_valid = args.do_test = False
    train_loader, valid_loaders, test_loader = build_train_valid_test_data_loaders(provider)
    assert (train_loader, valid_loaders, test_loader) == (None, None, None)
    assert not args.do_train and not args.do_valid and not args.do_test


def test_build_train_valid_test_data_iterators_wraps_multiple_validation_loaders(monkeypatch):
    args = SimpleNamespace(
        dataloader_type="external",
        full_validation=True,
        multiple_validation_sets=True,
        eval_iters=1,
    )
    wrapped = []
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(
        training,
        "build_train_valid_test_data_loaders",
        lambda provider: ("train-loader", [["valid-a"], ["valid-b"]], "test-loader"),
    )
    monkeypatch.setattr(training, "RerunDataIterator", lambda iterator: wrapped.append(iterator) or ("wrapped", iterator))

    train_iter, valid_iters, test_iter = build_train_valid_test_data_iterators(lambda _: None)

    assert train_iter == ("wrapped", "train-loader")
    assert [valid_iter[0] for valid_iter in valid_iters] == ["wrapped", "wrapped"]
    assert [next(valid_iter[1]) for valid_iter in valid_iters] == ["valid-a", "valid-b"]
    assert test_iter == ("wrapped", "test-loader")
    assert args.eval_iters == [1, 1]


def test_build_train_valid_test_data_loaders_rejects_partial_multiple_validation(monkeypatch):
    real_tensor = torch.tensor
    args = SimpleNamespace(
        iteration=0,
        train_samples=None,
        train_iters=4,
        eval_interval=2,
        eval_iters=1,
        global_batch_size=8,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        phase_transition_iterations=None,
        perform_rl_step=False,
        skip_train=False,
        full_validation=False,
        multiple_validation_sets=True,
    )

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "print_rank_0", lambda *items, **kwargs: None)
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training.torch.distributed, "broadcast", lambda tensor, src: None)
    monkeypatch.setattr(
        training,
        "build_pretraining_data_loader",
        lambda dataset, consumed: (dataset, consumed),
    )

    with pytest.raises(NotImplementedError, match="multiple-validation-sets"):
        build_train_valid_test_data_loaders(lambda samples: ("train", ["valid-a", "valid-b"], "test"))


def test_build_train_valid_test_data_iterators_single_and_cyclic_modes(monkeypatch):
    wrapped = []
    args = SimpleNamespace(
        dataloader_type="single",
        full_validation=False,
        multiple_validation_sets=False,
        eval_iters=1,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(
        training,
        "build_train_valid_test_data_loaders",
        lambda provider: ([1], [[2]], [3]),
    )
    monkeypatch.setattr(training, "RerunDataIterator", lambda iterator: wrapped.append(iterator) or iterator)

    train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(lambda _: None)

    assert next(train_iter) == 1
    assert next(valid_iter) == 2
    assert next(test_iter) == 3

    args.dataloader_type = "cyclic"
    train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(lambda _: None)
    assert [next(train_iter), next(train_iter)] == [1, 1]
    assert [next(valid_iter), next(valid_iter)] == [2, 2]
    assert [next(test_iter), next(test_iter)] == [3, 3]


def test_should_disable_forward_pre_hook_requires_dist_optimizer_and_overlap():
    args = SimpleNamespace(
        use_megatron_fsdp=False,
        use_distributed_optimizer=True,
        optimizer="adam",
        overlap_param_gather=True,
    )
    assert should_disable_forward_pre_hook(args) is True

    args.use_megatron_fsdp = True
    assert should_disable_forward_pre_hook(args) is False

    args.use_megatron_fsdp = False
    args.use_distributed_optimizer = False
    args.optimizer = "adam"
    assert should_disable_forward_pre_hook(args) is False

    args.optimizer = "distributed_adam"
    args.overlap_param_gather = False
    assert should_disable_forward_pre_hook(args) is False


def test_dummy_train_step_consumes_microbatches_until_rerun_stops(monkeypatch):
    calls = []

    class FakeRerunStateMachine:
        def __init__(self):
            self.should_run = True

        def should_run_forward_backward(self, data_iterator):
            should_run = self.should_run
            self.should_run = False
            return should_run

    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunStateMachine())
    monkeypatch.setattr(
        training,
        "get_batch_on_this_tp_rank",
        lambda iterator: calls.append(("tp", iterator)) or "batch",
    )
    monkeypatch.setattr(
        training,
        "get_batch_on_this_cp_rank",
        lambda batch: calls.append(("cp", batch)) or "cp-batch",
    )

    dummy_train_step("iterator")

    assert calls == [
        ("tp", "iterator"),
        ("cp", "batch"),
        ("tp", "iterator"),
        ("cp", "batch"),
    ]


def test_train_step_success_path_averages_losses_and_steps_scheduler(monkeypatch):
    calls = []
    args = SimpleNamespace(
        save_dgrads_interval=None,
        save_wgrads_interval=None,
        seq_length=8,
        micro_batch_size=2,
        decoder_seq_length=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        save=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        vision_pretraining_type=None,
        barrier_with_L1_time=False,
        qk_clip=False,
        log_max_attention_logit=False,
        log_num_zeros_in_grad=True,
        curr_iteration=0,
        data_parallel_size=2,
    )

    class FakeTimer:
        def __init__(self, name):
            self.name = name

        def start(self, barrier=False):
            calls.append(("timer-start", self.name, barrier))

        def stop(self):
            calls.append(("timer-stop", self.name))

    class FakeTimers:
        def __call__(self, name, log_level=None):
            calls.append(("timer", name, log_level))
            return FakeTimer(name)

    class FakeRerunStateMachine:
        def __init__(self):
            self.runs = 0

        def should_run_forward_backward(self, iterator):
            self.runs += 1
            return self.runs == 1

        def should_checkpoint_and_exit(self):
            return False, False, 0

    class FakeModelChunk:
        force_all_reduce = None

        def zero_grad_buffer(self):
            calls.append("zero-grad-buffer")

    class FakeOptimizer:
        def zero_grad(self):
            calls.append("optimizer-zero-grad")

        def step(self):
            calls.append("optimizer-step")
            return True, 1.5, 2

    class FakeScheduler:
        def step(self, increment):
            calls.append(("scheduler-step", increment))

    def fake_forward_backward(**kwargs):
        calls.append(("forward-backward", kwargs["num_microbatches"], kwargs["force_all_reduce"]))
        return [
            {"lm loss": torch.tensor([2.0])},
            {"lm loss": torch.tensor([4.0])},
        ]

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunStateMachine())
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "logical_and_across_model_parallel_group", lambda value: value)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda value: value)
    monkeypatch.setattr(training.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)

    loss_dict, skipped, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros, max_logit = train_step(
        lambda *_: None,
        "data-iter",
        [FakeModelChunk()],
        FakeOptimizer(),
        FakeScheduler(),
        SimpleNamespace(),
        fake_forward_backward,
        iteration=0,
    )

    assert loss_dict["lm loss"].item() == pytest.approx(3.0)
    assert skipped == 0
    assert not should_checkpoint
    assert not should_exit
    assert exit_code == 0
    assert grad_norm == 1.5
    assert num_zeros == 2
    assert max_logit == 0
    assert "zero-grad-buffer" in calls
    assert "optimizer-zero-grad" in calls
    assert "optimizer-step" in calls
    assert ("scheduler-step", 8) in calls
    assert ("forward-backward", 2, False) in calls


def test_train_single_iteration_control_flow(monkeypatch):
    calls = []
    args = SimpleNamespace(
        perform_rl_step=False,
        hybrid_context_parallel=False,
        run_workload_inspector_server=False,
        iteration=0,
        world_size=1,
        consumed_train_samples=0,
        train_samples=None,
        train_iters=1,
        save=None,
        async_save=False,
        log_throughput=True,
        num_floating_point_operations_so_far=0.0,
        seq_length=8,
        overlap_grad_reduce=False,
        align_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        log_energy=False,
        manual_gc=False,
        log_straggler=False,
        cuda_graph_impl=None,
        cuda_graph_scope=[],
        cuda_graph_warmup_steps=0,
        optimizer_cuda_graph=False,
        profile=False,
        profile_ranks=[],
        use_pytorch_profiler=False,
        check_weight_hash_across_dp_replicas_interval=None,
        distributed_timeout_seconds_after_init=None,
        rl_use_sequence_packing=False,
        iterations_to_skip=[],
        skip_train=False,
        micro_batch_size=2,
        decrease_batch_size_if_needed=False,
        skipped_train_samples=0,
        log_params_norm=False,
        eval_interval=1,
        eval_iters=1,
        do_valid=True,
        manual_gc_eval=False,
        num_experts=None,
        save_interval=None,
        exit_signal_handler=False,
        non_persistent_save_interval=None,
        exit_duration_in_mins=None,
        exit_interval=None,
        phase_transition_iterations=None,
    )

    class FakeTimer:
        def __init__(self, name):
            self.name = name

        def start(self, barrier=False):
            calls.append(("timer-start", self.name, barrier))

        def stop(self):
            calls.append(("timer-stop", self.name))

        def elapsed(self):
            calls.append(("timer-elapsed", self.name))
            return 0.25

        def active_time(self):
            return 1.0

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer(name)

    class FakeModel:
        def train(self):
            calls.append("model-train")

    class FakeRerunStateMachine:
        current_iteration = 0

    class FakeOptimizer:
        is_stub_optimizer = False
        param_groups = [{"lr": 0.01}]

        def scale_loss(self, loss):
            return loss

        def get_loss_scale(self):
            return torch.tensor(1.0)

    config = SimpleNamespace(
        grad_scale_func=None,
        timers=None,
        no_sync_func=None,
        param_sync_func="param-sync",
        finalize_model_grads_func=None,
    )

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_energy_monitor", lambda: SimpleNamespace())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "write_args_to_tensorboard", lambda: calls.append("write-args"))
    monkeypatch.setattr(training, "get_attr_wrapped_model", lambda model, name: "pg")
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunStateMachine())
    monkeypatch.setattr(training.one_logger_utils, "on_train_start", lambda **kwargs: calls.append(("train-start", kwargs["train_iters"])))
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: calls.append(("e2e", items)))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_forward_backward_func", lambda: "forward-backward")
    monkeypatch.setattr(training, "should_disable_forward_pre_hook", lambda args: False)
    monkeypatch.setattr(training, "update_num_microbatches", lambda *items, **kwargs: calls.append(("update-mbs", kwargs.get("consistency_check"))))
    monkeypatch.setattr(training.ft_integration, "on_checkpointing_start", lambda: calls.append("ckpt-start"))
    monkeypatch.setattr(training.ft_integration, "on_checkpointing_end", lambda **kwargs: calls.append(("ckpt-end", kwargs.get("is_async_finalization"))))
    monkeypatch.setattr(training.ft_integration, "on_training_step_start", lambda: calls.append("step-start"))
    monkeypatch.setattr(training.ft_integration, "on_training_step_end", lambda: calls.append("step-end"))
    monkeypatch.setattr(training, "maybe_finalize_async_save", lambda **kwargs: calls.append(("finalize", kwargs)))
    monkeypatch.setattr(
        training,
        "train_step",
        lambda *items, **kwargs: ({"lm loss": torch.tensor([1.0])}, 0, False, False, 0, 1.0, 0, 2.0),
    )
    monkeypatch.setattr(training.mpu, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 2)
    monkeypatch.setattr(training, "get_current_running_global_batch_size", lambda: 2)
    monkeypatch.setattr(training, "num_floating_point_operations", lambda args, batch_size: 100.0)
    monkeypatch.setattr(training, "get_canonical_lr_for_logging", lambda param_groups: 0.01)
    monkeypatch.setattr(training, "training_log", lambda *items, **kwargs: calls.append(("training-log", items[3])) or False)
    monkeypatch.setattr(training, "evaluate_and_print_results", lambda *items, **kwargs: calls.append(("evaluate", items[0], items[4])))
    monkeypatch.setattr(training, "post_training_step_callbacks", lambda *items, **kwargs: calls.append(("post", items[3])) or items[5])
    monkeypatch.setattr(training, "checkpoint_and_decide_exit", lambda *items, **kwargs: calls.append(("checkpoint", items[3])) or False)
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: SimpleNamespace(flush=lambda: calls.append("flush")))
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: calls.append(("print-datetime", items[0])))

    iteration, flops = train(
        lambda *_: None,
        [FakeModel()],
        FakeOptimizer(),
        SimpleNamespace(step=lambda increment: calls.append(("scheduler-step", increment))),
        "train-iter",
        "valid-iter",
        None,
        config,
        {},
        None,
    )

    assert iteration == 1
    assert flops == 100.0
    assert "model-train" in calls
    assert ("train-start", 1) in calls
    assert ("training-log", 1) in calls
    assert ("evaluate", "iteration 1", 1) in calls
    assert ("checkpoint", 1) in calls
    assert "flush" in calls


def test_train_skip_iteration_and_exit_path(monkeypatch):
    calls = []
    args = SimpleNamespace(
        perform_rl_step=False,
        hybrid_context_parallel=False,
        run_workload_inspector_server=False,
        iteration=0,
        world_size=1,
        consumed_train_samples=0,
        train_samples=None,
        train_iters=1,
        save=None,
        async_save=False,
        log_throughput=False,
        num_floating_point_operations_so_far=0.0,
        seq_length=8,
        overlap_grad_reduce=False,
        align_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        log_energy=False,
        manual_gc=False,
        log_straggler=False,
        cuda_graph_impl=None,
        cuda_graph_scope=[],
        cuda_graph_warmup_steps=0,
        optimizer_cuda_graph=False,
        profile=False,
        profile_ranks=[],
        use_pytorch_profiler=False,
        check_weight_hash_across_dp_replicas_interval=None,
        distributed_timeout_seconds_after_init=None,
        rl_use_sequence_packing=False,
        iterations_to_skip=[1],
        skip_train=False,
        micro_batch_size=2,
        decrease_batch_size_if_needed=False,
        skipped_train_samples=0,
        log_params_norm=False,
        eval_interval=None,
        do_valid=False,
        manual_gc_eval=False,
        num_experts=None,
        save_interval=None,
        exit_signal_handler=False,
        non_persistent_save_interval=None,
        exit_duration_in_mins=None,
        exit_interval=None,
        phase_transition_iterations=None,
    )

    class FakeTimer:
        def start(self, barrier=False):
            pass

        def stop(self):
            pass

        def active_time(self):
            return 0.0

    class FakeTimers:
        def __call__(self, *items, **kwargs):
            return FakeTimer()

    class FakeModel:
        def train(self):
            calls.append("train-mode")

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_energy_monitor", lambda: SimpleNamespace())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "write_args_to_tensorboard", lambda: None)
    monkeypatch.setattr(training, "get_attr_wrapped_model", lambda model, name: "pg")
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: SimpleNamespace(current_iteration=0))
    monkeypatch.setattr(training.one_logger_utils, "on_train_start", lambda **kwargs: None)
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: calls.append("e2e"))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_forward_backward_func", lambda: "forward-backward")
    monkeypatch.setattr(training, "should_disable_forward_pre_hook", lambda args: False)
    monkeypatch.setattr(training, "update_num_microbatches", lambda *items, **kwargs: None)
    monkeypatch.setattr(training.ft_integration, "on_checkpointing_start", lambda: None)
    monkeypatch.setattr(training.ft_integration, "on_checkpointing_end", lambda **kwargs: None)
    monkeypatch.setattr(training, "maybe_finalize_async_save", lambda **kwargs: None)
    monkeypatch.setattr(training, "dummy_train_step", lambda iterator: calls.append(("dummy", iterator)))
    monkeypatch.setattr(training.mpu, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: None)
    monkeypatch.setattr(training, "print_datetime", lambda *items, **kwargs: None)

    iteration, flops = train(
        lambda *_: None,
        [FakeModel()],
        None,
        None,
        "train-iter",
        None,
        None,
        SimpleNamespace(grad_scale_func=None, timers=None, no_sync_func=None, param_sync_func=None, finalize_model_grads_func=None),
        {},
        None,
    )

    assert iteration == 1
    assert flops == 0.0
    expected_skipped_batch_size = 2 * args.micro_batch_size * 1
    assert args.consumed_train_samples == expected_skipped_batch_size
    assert args.skipped_train_samples == expected_skipped_batch_size
    assert ("dummy", "train-iter") in calls


def test_training_log_updates_accumulators_and_writers(monkeypatch):
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)

    class FakeTimer:
        def __init__(self, name):
            self.name = name

        def elapsed(self, barrier=False, reset=True):
            calls.append(("elapsed", self.name, barrier, reset))
            return 2.0

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer(name)

        def write(self, names, writer, iteration, normalizer=None, reset=False):
            calls.append(("write", tuple(names), iteration, normalizer, reset, writer is not None))

        def log(self, names, normalizer=None, reset=True):
            calls.append(("log", tuple(names), normalizer, reset))

    class FakeWriter:
        def add_scalar(self, name, value, iteration):
            calls.append(("scalar", name, iteration))

    class FakeWandb:
        def log(self, payload, iteration=None):
            calls.append(("wandb", tuple(sorted(payload)), iteration))

    args = SimpleNamespace(
        timing_log_level=2,
        perform_rl_step=False,
        rl_use_sequence_packing=False,
        micro_batch_size=2,
        data_parallel_size=3,
        world_size=6,
        seq_length=8,
        tensorboard_log_interval=1,
        consumed_train_samples=48,
        skipped_train_samples=0,
        log_loss_scale_to_tensorboard=True,
        log_world_size_to_tensorboard=True,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=True,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=None,
        log_interval=1,
        train_iters=10,
        log_throughput=True,
        log_timers_to_tensorboard=True,
        log_energy=False,
        record_memory_history=False,
        memory_snapshot_path="unused",
        log_memory_interval=None,
    )
    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: FakeWriter())
    monkeypatch.setattr(training, "get_wandb_writer", lambda: FakeWandb())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_energy_monitor", lambda: SimpleNamespace(lap=lambda: 0.0))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda value: value)
    monkeypatch.setattr(training.one_logger_utils, "track_app_tag", lambda *items: calls.append(("tag", items)))
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: calls.append(("e2e", items)))
    monkeypatch.setattr(training, "num_floating_point_operations", lambda args, batch_size: 12e12)
    monkeypatch.setattr(training, "print_rank_last", lambda message: calls.append(("print_last", message)))

    total_loss = {}
    report_memory = training_log(
        {"lm loss": torch.tensor([2.0])},
        total_loss,
        learning_rate=0.001,
        iteration=2,
        loss_scale=128.0,
        report_memory_flag=False,
        skipped_iter=0,
        grad_norm=1.5,
        params_norm=2.5,
        num_zeros_in_grad=3,
        max_attention_logit=4.0,
    )

    assert report_memory is False
    assert total_loss["advanced iterations"] == 0
    assert total_loss["skipped iterations"] == 0
    assert total_loss["nan iterations"] == 0
    assert any(item[:2] == ("scalar", "learning-rate") for item in calls)
    assert any(item[:2] == ("scalar", "throughput") for item in calls)
    assert any(item[0] == "print_last" and "lm loss" in item[1] for item in calls)


def test_training_log_skipped_nan_memory_and_auxiliary_metrics(monkeypatch):
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    class FakeTimer:
        def elapsed(self, barrier=False, reset=True):
            calls.append(("elapsed", barrier, reset))
            return 4.0

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer()

        def write(self, *items, **kwargs):
            calls.append(("write", items, kwargs))

        def log(self, names, normalizer=None, reset=True):
            calls.append(("log", tuple(names), normalizer, reset))

    args = SimpleNamespace(
        timing_log_level=1,
        perform_rl_step=False,
        rl_use_sequence_packing=False,
        micro_batch_size=2,
        data_parallel_size=2,
        world_size=4,
        seq_length=8,
        tensorboard_log_interval=10,
        consumed_train_samples=16,
        skipped_train_samples=3,
        log_loss_scale_to_tensorboard=False,
        log_world_size_to_tensorboard=False,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=False,
        num_experts=2,
        moe_router_load_balancing_type="aux_loss seq_aux_loss global_aux_loss",
        moe_z_loss_coeff=0.1,
        num_layers=3,
        moe_per_layer_logging=True,
        moe_layer_freq=[1, 0, 1],
        mtp_num_layers=1,
        dsa_indexer_loss_coeff=0.2,
        log_interval=2,
        train_iters=12,
        log_throughput=True,
        log_timers_to_tensorboard=False,
        log_energy=True,
        record_memory_history=False,
        memory_snapshot_path="unused",
        log_memory_interval=2,
    )

    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: None)
    monkeypatch.setattr(training, "get_wandb_writer", lambda: None)
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_energy_monitor", lambda: SimpleNamespace(lap=lambda: 40.0))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda value: value)
    monkeypatch.setattr(training.one_logger_utils, "track_app_tag", lambda *items: calls.append(("tag", items)))
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", lambda *items: calls.append(("e2e", items)))
    monkeypatch.setattr(training, "num_floating_point_operations", lambda args, batch_size: 8e12)
    monkeypatch.setattr(training, "print_rank_last", lambda message: calls.append(("print_last", message)))
    monkeypatch.setattr(training, "is_hybrid_model", lambda args: False)
    monkeypatch.setattr(training, "track_moe_metrics", lambda **kwargs: calls.append(("moe", tuple(kwargs["track_names"]), kwargs["num_layers"])))
    monkeypatch.setattr(
        training.MTPLossLoggingHelper,
        "track_mtp_metrics",
        lambda *items: calls.append(("mtp", items[1])),
    )
    monkeypatch.setattr(
        training.DSAIndexerLossLoggingHelper,
        "track_indexer_metrics",
        lambda **kwargs: calls.append(("dsa", kwargs["iteration"])),
    )
    monkeypatch.setattr(training.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(training, "report_theoretical_memory", lambda *items, **kwargs: calls.append(("theoretical", kwargs.get("verbose"))))
    monkeypatch.setattr(training, "report_memory", lambda message: calls.append(("memory", message)))
    monkeypatch.setattr(training, "get_loaded_iteration", lambda: 0)

    total_loss = {}
    report_memory = training_log(
        {"lm loss": torch.tensor([float("inf")])},
        total_loss,
        learning_rate=None,
        iteration=2,
        loss_scale=64.0,
        report_memory_flag=True,
        skipped_iter=1,
        grad_norm=None,
        params_norm=None,
        num_zeros_in_grad=None,
        max_attention_logit=0.0,
    )

    assert report_memory is False
    assert total_loss["advanced iterations"] == 0
    assert total_loss["skipped iterations"] == 0
    assert total_loss["nan iterations"] == 0
    assert ("moe", ("load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss", "z_loss"), 3) in calls
    assert ("mtp", 2) in calls
    assert ("dsa", 2) in calls
    assert any(item[0] == "print_last" and "number of nan iterations" in item[1] for item in calls)
    assert ("theoretical", True) in calls
    assert ("memory", "(after 2 iterations)") in calls


def test_post_training_step_callbacks_runs_optional_hooks(monkeypatch):
    calls = []

    class FakeCudaRuntime:
        def cudaProfilerStop(self):
            calls.append("profiler-stop")
            return 0

    class FakeNvtxContext:
        def __exit__(self, exc_type, exc, tb):
            calls.append("nvtx-exit")

    args = SimpleNamespace(
        train_sync_interval=2,
        log_interval=2,
        log_straggler=True,
        check_weight_hash_across_dp_replicas_interval=2,
        adlr_autoresume=True,
        adlr_autoresume_interval=2,
        profile=True,
        profile_step_end=2,
        profile_ranks=[],
        use_pytorch_profiler=False,
        manual_gc=True,
        manual_gc_interval=2,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(training.stimer, "report", lambda flops, interval: calls.append(("straggler", flops, interval)))
    monkeypatch.setattr(training, "should_disable_forward_pre_hook", lambda args: True)
    monkeypatch.setattr(training, "disable_forward_pre_hook", lambda model: calls.append("disable-hook"))
    monkeypatch.setattr(training, "enable_forward_pre_hook", lambda model: calls.append("enable-hook"))
    monkeypatch.setattr(training, "check_param_hashes_across_dp_replicas", lambda model, cross_check=True: True)
    monkeypatch.setattr(training.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(training, "check_adlr_autoresume_termination", lambda *items: calls.append("autoresume"))
    monkeypatch.setattr(training.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(training.torch.cuda, "cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(training.torch.cuda, "check_error", lambda value: calls.append(("check-error", value)))
    monkeypatch.setattr(training.gc, "collect", lambda: calls.append("gc"))

    remaining_flops = post_training_step_callbacks(
        model=[object()],
        optimizer=object(),
        opt_param_scheduler=object(),
        iteration=2,
        prof=None,
        num_floating_point_operations_since_last_log_event=123.0,
        nsys_nvtx_context=FakeNvtxContext(),
    )

    assert remaining_flops == 0.0
    assert "sync" in calls
    assert ("straggler", 123.0, 2) in calls
    assert "disable-hook" in calls and "enable-hook" in calls
    assert "autoresume" in calls
    assert "nvtx-exit" in calls
    assert "gc" in calls


def test_get_optimizer_param_scheduler_iteration_and_sample_modes(monkeypatch):
    created = []

    class FakeScheduler:
        def __init__(self, optimizer, **kwargs):
            self.optimizer = optimizer
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(training, "OptimizerParamScheduler", FakeScheduler)
    iteration_args = SimpleNamespace(
        train_iters=10,
        lr_decay_iters=None,
        global_batch_size=4,
        lr_wsd_decay_iters=2,
        lr_warmup_fraction=0.1,
        lr_warmup_iters=0,
        train_samples=None,
        lr_decay_samples=None,
        lr_wsd_decay_samples=None,
        lr_warmup_samples=0,
        lr_warmup_init=0.0,
        lr=0.01,
        min_lr=0.001,
        lr_decay_style="linear",
        start_weight_decay=0.0,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    monkeypatch.setattr(training, "get_args", lambda: iteration_args)

    scheduler = get_optimizer_param_scheduler("optimizer")

    assert scheduler.kwargs["lr_decay_steps"] == 40
    assert scheduler.kwargs["lr_warmup_steps"] == 4
    assert scheduler.kwargs["wsd_decay_steps"] == 8

    sample_args = SimpleNamespace(**vars(iteration_args))
    sample_args.train_iters = None
    sample_args.train_samples = 100
    sample_args.lr_decay_samples = None
    sample_args.lr_wsd_decay_samples = 20
    sample_args.lr_warmup_fraction = None
    sample_args.lr_warmup_samples = 5
    monkeypatch.setattr(training, "get_args", lambda: sample_args)
    monkeypatch.setattr(training, "update_train_iters", lambda args: setattr(args, "train_iters", 25))

    scheduler = get_optimizer_param_scheduler("optimizer")

    assert scheduler.kwargs["lr_decay_steps"] == 100
    assert scheduler.kwargs["lr_warmup_steps"] == 5
    assert scheduler.kwargs["wsd_decay_steps"] == 20


def test_get_optimizer_param_scheduler_rejects_missing_training_horizon(monkeypatch):
    args = SimpleNamespace(train_iters=None, train_samples=None)
    monkeypatch.setattr(training, "get_args", lambda: args)

    with pytest.raises(Exception, match="either train-iters or train-samples"):
        get_optimizer_param_scheduler("optimizer")


def test_get_megatron_optimizer_config_selects_supported_optimizers():
    adam_config, adam_overrides = get_megatron_optimizer_config(SimpleNamespace(optimizer="adam"))
    muon_config, _ = get_megatron_optimizer_config(SimpleNamespace(optimizer="muon"))
    sgd_config, sgd_overrides = get_megatron_optimizer_config(SimpleNamespace(optimizer="sgd"))

    assert adam_config.optimizer == "adam"
    assert muon_config.optimizer == "muon"
    assert sgd_config.optimizer == "sgd"
    assert isinstance(adam_overrides, dict)
    assert isinstance(sgd_overrides, dict)
    with pytest.raises(ValueError, match="Invalid optimizer type"):
        get_megatron_optimizer_config(SimpleNamespace(optimizer="rmsprop"))


def test_setup_model_and_optimizer_builds_optimizer_and_loads_checkpoint(monkeypatch):
    calls = []
    args = SimpleNamespace(
        skip_train=False,
        perform_rl_step=False,
        no_load_optim=False,
        use_mup=True,
        use_gloo_process_groups=False,
        dump_param_to_param_group_map=False,
        moe_use_upcycling=False,
        load="/tmp/ckpt",
        pretrained_checkpoint=None,
        ckpt_convert_format=None,
        use_torch_fsdp2=False,
        ckpt_format="torch",
        fp16=False,
        bf16=False,
    )

    class FakeTimer:
        def start(self, barrier=False):
            calls.append(("timer-start", barrier))

        def stop(self, barrier=False):
            calls.append(("timer-stop", barrier))

        def active_time(self):
            return 1.5

    class FakeTimers:
        def __call__(self, name, log_level=None):
            calls.append(("timer", name, log_level))
            return FakeTimer()

        def log(self, names):
            calls.append(("timer-log", tuple(names)))

    class FakeOneLogger:
        def log_metrics(self, metrics):
            calls.append(("metrics", tuple(sorted(metrics))))

    model = [SimpleNamespace()]
    config = SimpleNamespace(mup_width_mult=2)
    optimizer_config = SimpleNamespace(optimizer="adam", timers=None)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_one_logger", lambda: FakeOneLogger())
    monkeypatch.setattr(training.one_logger_utils, "get_timestamp_in_ms", lambda: 1000)
    monkeypatch.setattr(training, "get_model", lambda *items, **kwargs: calls.append(("get-model", kwargs["wrap_with_ddp"])) or model)
    monkeypatch.setattr(training, "unwrap_model", lambda item: item)
    monkeypatch.setattr(training, "get_megatron_optimizer_config", lambda item: (optimizer_config, {"base": True}))
    monkeypatch.setattr(training, "get_model_config", lambda item: config)
    monkeypatch.setattr(
        training,
        "get_mup_config_overrides",
        lambda **kwargs: calls.append(("mup", kwargs["mup_width_mult"], kwargs["optimizer_type"])) or {"mup": True},
    )
    monkeypatch.setattr(
        training,
        "get_megatron_optimizer",
        lambda config, model, **kwargs: calls.append(("optimizer", kwargs["config_overrides"])) or "optimizer",
    )
    monkeypatch.setattr(training, "get_optimizer_param_scheduler", lambda optimizer: calls.append(("scheduler", optimizer)) or "scheduler")
    monkeypatch.setattr(
        training,
        "load_checkpoint",
        lambda model, optimizer, scheduler, **kwargs: calls.append(("load", kwargs["checkpointing_context"], kwargs["skip_load_to_model_and_opt"])) or (9, 99.0),
    )

    returned_model, optimizer, scheduler = setup_model_and_optimizer(
        lambda: None,
        training.ModelType.encoder_or_decoder,
        checkpointing_context={"ctx": True},
    )

    assert returned_model is model
    assert optimizer == "optimizer"
    assert scheduler == "scheduler"
    assert args.iteration == 9
    assert args.num_floating_point_operations_so_far == 99.0
    assert optimizer_config.timers is not None
    assert ("get-model", True) in calls
    assert ("mup", 2, "adam") in calls
    assert ("optimizer", {"base": True, "mup": True}) in calls
    assert ("load", {"ctx": True}, False) in calls


def test_setup_model_and_optimizer_uses_muon_and_bert_initialization(monkeypatch):
    calls = []
    args = SimpleNamespace(
        skip_train=False,
        perform_rl_step=False,
        no_load_optim=False,
        use_mup=False,
        use_gloo_process_groups=True,
        dump_param_to_param_group_map=False,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        ckpt_convert_format=None,
        use_torch_fsdp2=False,
        ckpt_format="torch",
        fp16=True,
        bf16=False,
    )

    class FakeUnwrappedModel:
        def init_state_dict_from_bert(self):
            calls.append("bert-init")

    class FakeOptimizer:
        def reload_model_params(self):
            calls.append("reload")

    model = [SimpleNamespace()]
    unwrapped = [FakeUnwrappedModel()]
    optimizer_config = SimpleNamespace(optimizer="muon", timers=None)
    optimizer = FakeOptimizer()
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_model", lambda *items, **kwargs: model)
    monkeypatch.setattr(training, "unwrap_model", lambda item: unwrapped)
    monkeypatch.setattr(training, "get_megatron_optimizer_config", lambda item: (optimizer_config, {"base": True}))
    monkeypatch.setattr(
        training,
        "get_megatron_muon_optimizer",
        lambda config, model, **kwargs: calls.append(("muon", kwargs["use_gloo_process_groups"], kwargs["layer_wise_distributed_optimizer"])) or optimizer,
    )
    monkeypatch.setattr(training, "get_optimizer_param_scheduler", lambda optimizer: "scheduler")
    monkeypatch.setattr(training, "print_rank_0", lambda message: calls.append(("print", message)))

    returned_model, returned_optimizer, scheduler = setup_model_and_optimizer(
        lambda: None,
        training.ModelType.encoder_or_decoder,
    )

    assert returned_model is model
    assert returned_optimizer is optimizer
    assert scheduler == "scheduler"
    assert args.iteration == 0
    assert args.num_floating_point_operations_so_far == 0
    assert ("muon", True, False) in calls
    assert "bert-init" in calls
    assert "reload" in calls


def test_get_model_builds_single_chunk_without_cuda_or_ddp(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self, **metadata):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
            self.metadata = metadata

    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        init_model_with_meta_device=False,
        load=None,
        export_kd_teacher_load=None,
        use_torch_fsdp2=True,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
    )
    pg_collection = SimpleNamespace(pp="pp", dp="dp", cp="cp", tp="tp")
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.ProcessGroupCollection, "use_mpu_process_groups", lambda: pg_collection)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda pp: True)
    monkeypatch.setattr(
        training.tensor_parallel,
        "set_defaults_if_not_set_tensor_model_parallel_attributes",
        lambda param: calls.append(("set-defaults", tuple(param.shape))),
    )
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda model: calls.append(("amax", len(model))))

    model = get_model(lambda **kwargs: FakeModel(**kwargs), wrap_with_ddp=False)

    assert len(model) == 1
    assert model[0].metadata["pre_process"] is True
    assert model[0].metadata["post_process"] is True
    assert model[0].model_type is training.ModelType.encoder_or_decoder
    assert ("set-defaults", (1,)) in calls
    assert ("amax", 1) in calls


def test_get_model_builds_virtual_pipeline_chunks(monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self, **metadata):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
            self.metadata = metadata

    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=3,
        init_model_with_meta_device=False,
        load=None,
        export_kd_teacher_load=None,
        use_torch_fsdp2=True,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
    )
    pg_collection = SimpleNamespace(pp="pp", dp="dp", cp="cp", tp="tp")
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.ProcessGroupCollection, "use_mpu_process_groups", lambda: pg_collection)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 2 if group == "pp" else 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_vp_first_stage", lambda vp_stage, vp_size: vp_stage == 0)
    monkeypatch.setattr(training, "is_vp_last_stage", lambda vp_stage, vp_size: vp_stage == vp_size - 1)
    monkeypatch.setattr(
        training.tensor_parallel,
        "set_defaults_if_not_set_tensor_model_parallel_attributes",
        lambda param: None,
    )
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda model: None)

    model = get_model(lambda **kwargs: FakeModel(**kwargs), wrap_with_ddp=False)

    assert [chunk.vp_stage for chunk in model] == [0, 1, 2]
    assert [chunk.metadata["pre_process"] for chunk in model] == [True, False, False]
    assert [chunk.metadata["post_process"] for chunk in model] == [False, False, True]


def test_get_model_wraps_with_fsdp2_and_broadcasts_params(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self, **metadata):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(2))
            self.metadata = metadata

    class FakeWrappedModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(("wrap", kwargs["disable_bucketing"]))

        def broadcast_params(self):
            calls.append("broadcast")

    class FakeStream:
        def wait_stream(self, stream):
            calls.append(("wait", stream))

    class FakeStreamContext:
        def __enter__(self):
            calls.append("stream-enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("stream-exit")

    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        init_model_with_meta_device=False,
        load=None,
        export_kd_teacher_load=None,
        use_torch_fsdp2=True,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
        torch_fsdp2_reshard_after_forward=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=True,
    )
    pg_collection = SimpleNamespace(pp="pp", dp="dp", cp="cp", tp="tp")
    monkeypatch.setattr(training, "HAVE_FSDP2", True)
    monkeypatch.setattr(training, "torch_FSDP", FakeWrappedModel)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.ProcessGroupCollection, "use_mpu_process_groups", lambda: pg_collection)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda pp: True)
    monkeypatch.setattr(training.tensor_parallel, "set_defaults_if_not_set_tensor_model_parallel_attributes", lambda param: calls.append(("defaults", tuple(param.shape))))
    monkeypatch.setattr(training, "get_model_config", lambda model: SimpleNamespace())
    monkeypatch.setattr(training.torch.cuda, "Stream", lambda: FakeStream())
    monkeypatch.setattr(training.torch.cuda, "current_stream", lambda: FakeStream())
    monkeypatch.setattr(training.torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda model: calls.append(("amax", len(model))))

    model = get_model(lambda **kwargs: FakeModel(**kwargs), wrap_with_ddp=True)

    assert len(model) == 1
    assert isinstance(model[0], FakeWrappedModel)
    assert model[0].kwargs["ddp_config"].reshard_after_forward is False
    assert model[0].kwargs["module"].metadata["pre_process"] is True
    assert ("defaults", (2,)) in calls
    assert ("wrap", False) in calls
    assert "broadcast" in calls
    assert "stream-enter" in calls and "stream-exit" in calls


def test_get_model_wraps_virtual_chunks_with_ddp_config_and_side_stream(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self, **metadata):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(4))
            self.metadata = metadata

    class FakeDDP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(
                (
                    "ddp",
                    kwargs["module"].metadata["vp_stage"],
                    kwargs["disable_bucketing"],
                    kwargs["ddp_config"].bucket_size,
                )
            )

    class FakeStream:
        def wait_stream(self, stream):
            calls.append(("wait", stream))

    class FakeStreamContext:
        def __enter__(self):
            calls.append("stream-enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("stream-exit")

    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=2,
        init_model_with_meta_device=True,
        load=None,
        export_kd_teacher_load=None,
        use_torch_fsdp2=False,
        use_cpu_initialization=False,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
        accumulate_allreduce_grads_in_fp32=False,
        check_for_nan_in_loss_and_grad=True,
        check_for_large_grads=False,
        ddp_num_buckets=2,
        ddp_bucket_size=None,
        ddp_pad_buckets_for_high_nccl_busbw=True,
        ddp_reduce_scatter_with_fp32_accumulation=False,
        ddp_param_name_patterns_for_fp32_local_accumulation=[".*weight"],
        ddp_average_in_collective=True,
        overlap_grad_reduce=True,
        megatron_fsdp_main_params_dtype=torch.float32,
        megatron_fsdp_main_grads_dtype=torch.float32,
        megatron_fsdp_grad_comm_dtype=torch.float32,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
    )
    pg_collection = SimpleNamespace(pp="pp", dp="dp", cp="cp", tp="tp")
    monkeypatch.setattr(training, "DDP", FakeDDP)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training.ProcessGroupCollection, "use_mpu_process_groups", lambda: pg_collection)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 2 if group == "pp" else 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda pp: True)
    monkeypatch.setattr(training, "is_vp_first_stage", lambda vp_stage, vp_size: vp_stage == 0)
    monkeypatch.setattr(training, "is_vp_last_stage", lambda vp_stage, vp_size: vp_stage == vp_size - 1)
    monkeypatch.setattr(
        training.tensor_parallel,
        "set_defaults_if_not_set_tensor_model_parallel_attributes",
        lambda param: calls.append(("defaults", param.nelement())),
    )
    monkeypatch.setattr(training, "to_empty_if_meta_device", lambda model, device: model)
    monkeypatch.setattr(training, "get_model_config", lambda model: SimpleNamespace())
    monkeypatch.setattr(training.torch.cuda, "Stream", lambda: FakeStream())
    monkeypatch.setattr(training.torch.cuda, "current_stream", lambda: FakeStream())
    monkeypatch.setattr(training.torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda model: calls.append(("amax", len(model))))

    model = get_model(lambda **kwargs: FakeModel(**kwargs), wrap_with_ddp=True)

    assert len(model) == 2
    assert [chunk.kwargs["module"].metadata["pre_process"] for chunk in model] == [True, False]
    assert [chunk.kwargs["module"].metadata["post_process"] for chunk in model] == [False, True]
    assert ("ddp", 0, False, 4) in calls
    assert ("ddp", 1, True, 4) in calls
    assert model[0].kwargs["ddp_config"].param_name_patterns_for_fp32_local_accumulation == (
        ".*weight",
    )
    assert ("defaults", 4) in calls
    assert ("amax", 2) in calls
    assert "stream-enter" in calls and "stream-exit" in calls


def test_setup_model_and_optimizer_skips_optimizer_for_inference_only_rl(monkeypatch):
    calls = []
    args = SimpleNamespace(
        skip_train=True,
        perform_rl_step=True,
        no_load_optim=True,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        ckpt_convert_format=None,
        fp16=False,
        bf16=False,
    )
    model = [SimpleNamespace()]
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: object())
    monkeypatch.setattr(
        training,
        "get_one_logger",
        lambda: SimpleNamespace(log_metrics=lambda metrics: calls.append(("metrics", tuple(sorted(metrics))))),
    )
    monkeypatch.setattr(training.one_logger_utils, "get_timestamp_in_ms", lambda: 123)
    monkeypatch.setattr(training, "get_model", lambda *items, **kwargs: model)
    monkeypatch.setattr(training, "unwrap_model", lambda model: model)
    monkeypatch.setattr(training, "update_train_iters", lambda item: calls.append("update-iters"))

    returned_model, optimizer, scheduler = setup_model_and_optimizer(
        lambda: None,
        training.ModelType.encoder_or_decoder,
    )

    assert returned_model is model
    assert optimizer is None
    assert scheduler is None
    assert args.iteration == 0
    assert args.num_floating_point_operations_so_far == 0
    assert "update-iters" in calls


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )

def test_setup_model_and_optimizer_ckpt_convert_format(monkeypatch):
    calls = []
    args = SimpleNamespace(
        skip_train=False,
        perform_rl_step=False,
        no_load_optim=False,
        use_mup=False,
        use_gloo_process_groups=False,
        dump_param_to_param_group_map=False,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        ckpt_convert_format="torch_dist",
        ckpt_format="torch",
        ckpt_convert_save="/tmp/converted",
        save="/tmp/original",
        fp16=False,
        bf16=False,
        use_torch_fsdp2=False,
        iteration=0,
        num_floating_point_operations_so_far=0,
    )

    class FakeTimerObj:
        def start(self, barrier=False):
            pass

        def stop(self, barrier=False):
            pass

        def active_time(self):
            return 1.5

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimerObj()

        def log(self, names):
            pass

    model = [SimpleNamespace()]
    optimizer, scheduler = "optimizer", "scheduler"
    optimizer_config = SimpleNamespace(optimizer="adam", timers=None)

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_model", lambda *a, **kw: model)
    monkeypatch.setattr(training, "unwrap_model", lambda m: m)
    monkeypatch.setattr(training, "get_megatron_optimizer_config", lambda a: (optimizer_config, {}))
    monkeypatch.setattr(
        training,
        "get_megatron_optimizer",
        lambda c, m, **kw: calls.append("build-optimizer") or optimizer,
    )
    monkeypatch.setattr(training, "get_optimizer_param_scheduler", lambda o: scheduler)
    monkeypatch.setattr(
        training, "update_use_dist_ckpt", lambda a: calls.append("update-dist-ckpt")
    )
    monkeypatch.setattr(
        training,
        "save_checkpoint",
        lambda it, m, o, s, f, **kw: calls.append(("save", it, f is not None)),
    )
    monkeypatch.setattr(training, "print_rank_0", lambda msg: calls.append(("print", msg)))
    monkeypatch.setattr(training.torch.distributed, "barrier", lambda **kw: calls.append("barrier"))

    with pytest.raises(SystemExit) as exc_info:
        training.setup_model_and_optimizer(lambda: None, training.ModelType.encoder_or_decoder)

    assert exc_info.value.code is None
    assert "update-dist-ckpt" in calls
    assert ("save", 0, True) in calls
    assert calls[-1] == "barrier"
    assert any("converted checkpoint" in str(c) for c in calls)


def test_setup_model_and_optimizer_moe_upcycling(monkeypatch):
    calls = []
    args = SimpleNamespace(
        skip_train=False,
        perform_rl_step=False,
        no_load_optim=False,
        use_mup=False,
        use_gloo_process_groups=False,
        dump_param_to_param_group_map=False,
        moe_use_upcycling=True,
        num_experts=8,
        expert_model_parallel_size=2,
        ffn_hidden_size=4096,
        moe_upcycling_granularity=2,
        load=None,
        pretrained_checkpoint=None,
        ckpt_convert_format=None,
        save="/tmp/upcycled",
        fp16=True,
        bf16=False,
        use_torch_fsdp2=False,
        ckpt_format="torch",
        iteration=0,
        num_floating_point_operations_so_far=0,
    )

    class FakeOptimizer:
        def reload_model_params(self):
            calls.append("reload-model-params")

    class FakeTimerObj:
        def start(self, barrier=False):
            pass

        def stop(self, barrier=False):
            pass

        def active_time(self):
            return 1.5

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimerObj()

        def log(self, names):
            pass

    model = [SimpleNamespace()]
    dense_model = [SimpleNamespace()]
    optimizer = FakeOptimizer()
    scheduler = "scheduler"
    optimizer_config = SimpleNamespace(optimizer="adam", timers=None)

    # first call returns model, second returns dense_model
    model_calls = [model, dense_model]
    call_idx = [0]

    def fake_get_model(*a, **kw):
        result = model_calls[call_idx[0]]
        call_idx[0] += 1
        return result

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_model", fake_get_model)
    monkeypatch.setattr(training, "unwrap_model", lambda m: m)
    monkeypatch.setattr(
        training,
        "get_megatron_optimizer_config",
        lambda a: (optimizer_config, {}),
    )
    monkeypatch.setattr(
        training,
        "get_megatron_optimizer",
        lambda c, m, **kw: calls.append("build-optimizer") or optimizer,
    )
    monkeypatch.setattr(training, "get_optimizer_param_scheduler", lambda o: scheduler)
    monkeypatch.setattr(
        training,
        "checkpoint_exists",
        lambda path: calls.append(("checkpoint-exists", path)) or False,
    )
    monkeypatch.setattr(
        training.torch.distributed,
        "barrier",
        lambda **kw: calls.append("barrier"),
    )
    monkeypatch.setattr(
        training,
        "save_checkpoint",
        lambda it, m, o, s, f, **kw: calls.append(("save", it)),
    )
    monkeypatch.setattr(
        training, "print_rank_0", lambda msg: calls.append(("print", msg))
    )
    monkeypatch.setattr(
        training,
        "load_checkpoint",
        lambda m, o, s, **kw: calls.append("load-checkpoint") or (0, 0.0),
    )
    monkeypatch.setattr(
        training.upcycling_utils,
        "load_and_upcycle_model",
        lambda lc, uw, dm, **kw: calls.append("upcycle") or (None, 999.0),
    )

    training.setup_model_and_optimizer(
        lambda: None, training.ModelType.encoder_or_decoder
    )

    assert "barrier" in calls
    assert ("checkpoint-exists", "/tmp/upcycled") in calls
    assert "build-optimizer" in calls
    assert "upcycle" in calls
    assert "reload-model-params" in calls
    assert any("Upcycled checkpoint" in str(c) for c in calls)

    assert args.iteration == 1
    assert args.num_experts == 8
    assert args.expert_model_parallel_size == 2
    assert args.ffn_hidden_size == 4096


def test_train_step_save_dgrads_and_wgrads_paths(monkeypatch):
    calls = []
    args = SimpleNamespace(
        seq_length=1024, micro_batch_size=4, global_batch_size=32,
        save_dgrads_interval=1,
        save_wgrads_interval=1,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        vision_pretraining_type="",
        barrier_with_L1_time=False,
        log_num_zeros_in_grad=False,
        qk_clip=False,
        log_max_attention_logit=False,
    )
    data_iterator = iter([None])

    class FakeOptimizer:
        def zero_grad(self):
            pass

        def step(self):
            calls.append("optimizer-step")
            return True, torch.tensor(1.0), torch.tensor(0)

    optimizer = FakeOptimizer()
    opt_param_scheduler = SimpleNamespace()

    model = [SimpleNamespace()]
    model[0].force_all_reduce = False
    model[0].zero_grad_buffer = lambda: None

    class FakeParam:
        main_grad = None

    def fake_named_parameters():
        p = FakeParam()
        p.main_grad = torch.ones(4, 4)
        yield ("layer.weight", p)

    monkeypatch.setattr(training, "unwrap_model",
                        lambda m: SimpleNamespace(named_parameters=fake_named_parameters))

    loop_counter = [0]

    class FakeRerunMachine:
        def should_run_forward_backward(self, di):
            loop_counter[0] += 1
            if loop_counter[0] == 1:
                return True
            return False

        def should_checkpoint_and_exit(self):
            return False, False, None

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: SimpleNamespace())
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: FakeRerunMachine())
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "clip_qk", lambda m, **kw: 0.0)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda v: v)
    monkeypatch.setattr(training, "logical_and_across_model_parallel_group", lambda v: v)
    monkeypatch.setattr(training, "enable_dgrad_logging",
                        lambda m, s: calls.append("enable-dgrad"))
    monkeypatch.setattr(training, "disable_dgrad_logging",
                        lambda: calls.append("disable-dgrad"))
    monkeypatch.setattr(training, "save_dgrads",
                        lambda it: calls.append(("save-dgrads", it)))
    monkeypatch.setattr(training.checkpointing, "save_grads",
                        lambda save_dir, sd, it, label: calls.append(("save-grads", label, it)))

    
    def fake_forward_backward(**kw):
        calls.append("fbw")
        return [{"lm_loss": torch.tensor(1.0)}]

    training.train_step(
        forward_step_func=lambda *a, **kw: None,
        data_iterator=data_iterator,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        config=SimpleNamespace(timers=None),
        forward_backward_func=fake_forward_backward,
        iteration=0,
    )

    # dgrads path（while 循环内执行）
    assert "enable-dgrad" in calls
    assert "fbw" in calls
    assert "disable-dgrad" in calls
    assert ("save-dgrads", 1) in calls
    # wgrads path（while 循环外执行）
    assert ("save-grads", "wgrads", 1) in calls


# ============================================================================
# 覆盖率提升测试：training_log - GRPO/world_size/显存/memory 指标路径
# ============================================================================
def test_training_log_grpo_and_auxiliary_writer_paths(monkeypatch):
    """覆盖 training_log 中 writer/scalar 写入分支中尚未被覆盖的路径。

    现有测试 test_training_log_skipped_nan_memory_and_auxiliary_metrics
    设了 writer=None，因此 if writer and ... 块内的所有代码都未执行过。
    本测试提供 FakeWriter 和对应的 args flag，互补覆盖。

    新覆盖的路径（均在 if writer 块内）：
      - skipped-train-samples > 0 → writer/wandb 写入
      - log_world_size_to_tensorboard=True
      - perform_rl_step=True → GRPO collection iteration
      - log_memory_to_tensorboard=True → 显存指标
      - log_loss_scale_to_tensorboard=True
      - log_max_attention_logit=True
      - rl_use_sequence_packing + has_rl_utils → RL seq packing 指标
      - grad_norm 非空 → 正常日志路径
    """
    calls = []
    real_tensor = torch.tensor

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    class FakeWriter:
        def add_scalar(self, name, value, iteration):
            calls.append(("scalar", name, value, iteration))

    class FakeWandb:
        def log(self, payload, iteration=None):
            calls.append(("wandb", tuple(sorted(payload)), iteration))

    class FakeTimer:
        def elapsed(self, barrier=False, reset=True):
            return 4.0

    class FakeTimers:
        def __call__(self, name, log_level=None):
            return FakeTimer()

        def write(self, *items, **kwargs):
            pass

        def log(self, names, normalizer=None, reset=True):
            pass

    class FakeMemStats:
        def __getitem__(self, key):
            return 1024

    args = SimpleNamespace(
        timing_log_level=1,
        perform_rl_step=True,                          # ★ GRPO
        rl_use_sequence_packing=True,                 # ★ RL seq packing
        grpo_iterations=1,
        grpo_samples_per_iteration=128,
        global_batch_size=32,
        micro_batch_size=2,
        data_parallel_size=2,
        world_size=8,
        seq_length=8,
        tensorboard_log_interval=10,                   # iteration=10 时触发
        consumed_train_samples=16,
        skipped_train_samples=10,                      # ★ >0 触发 skipped 标量
        log_loss_scale_to_tensorboard=True,            # ★
        log_world_size_to_tensorboard=True,            # ★
        log_memory_to_tensorboard=True,                # ★
        log_max_attention_logit=True,                  # ★
        num_experts=None,                              # 不触发 MoE，简化测试
        moe_router_load_balancing_type="",
        moe_z_loss_coeff=None,
        num_layers=3,
        moe_per_layer_logging=False,
        moe_layer_freq=[1, 0, 1],
        mtp_num_layers=1,
        dsa_indexer_loss_coeff=0.2,
        log_interval=2,
        train_iters=12,
        log_throughput=True,
        log_timers_to_tensorboard=False,
        log_energy=True,
        record_memory_history=False,
        memory_snapshot_path="unused",
        log_memory_interval=2,
    )

    # ★ rl_utils mock（需同时设 has_rl_utils 和 rl_utils 模块）
    class FakeRlUtils:
        @staticmethod
        def get_sequence_packing_tensorboard_metrics(args):
            calls.append("rl-packing-metrics")
            return {"packed_bins": 5}

    monkeypatch.setattr(training.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: FakeTimers())
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: FakeWriter())
    monkeypatch.setattr(training, "get_wandb_writer", lambda: FakeWandb())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_energy_monitor",
                        lambda: SimpleNamespace(lap=lambda: 40.0))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group",
                        lambda value: value)
    monkeypatch.setattr(training.one_logger_utils, "track_app_tag",
                        lambda *items: None)
    monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics",
                        lambda *items: None)
    monkeypatch.setattr(training, "num_floating_point_operations",
                        lambda args, batch_size: 8e12)
    monkeypatch.setattr(training, "print_rank_last", lambda message: None)
    monkeypatch.setattr(training, "is_hybrid_model", lambda args: False)
    monkeypatch.setattr(training, "track_moe_metrics", lambda **kwargs: None)
    monkeypatch.setattr(training.MTPLossLoggingHelper, "track_mtp_metrics",
                        lambda *items: None)
    monkeypatch.setattr(training.DSAIndexerLossLoggingHelper,
                        "track_indexer_metrics", lambda **kwargs: None)
    monkeypatch.setattr(training.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(training, "report_theoretical_memory",
                        lambda *items, **kwargs: None)
    monkeypatch.setattr(training, "report_memory", lambda message: calls.append(("memory", message)))
    monkeypatch.setattr(training, "get_loaded_iteration", lambda: 0)
    # ★ RL 相关 mock
    monkeypatch.setattr(training, "has_rl_utils", True)
    monkeypatch.setattr(training, "rl_utils", FakeRlUtils())
    # ★ cuda memory_stats mock
    monkeypatch.setattr(training.torch.cuda, "memory_stats", lambda: FakeMemStats())

    training_log(
        {"lm loss": torch.tensor([2.0])},
        total_loss_dict={},
        learning_rate=0.0001,
        iteration=10,                                   # ★ 整除 interval=10
        loss_scale=64.0,
        report_memory_flag=True,
        skipped_iter=0,
        grad_norm=torch.tensor(1.0),                   # 非空 → 覆盖 grad-norm 路径
        params_norm=None,
        num_zeros_in_grad=None,
        max_attention_logit=0.5,
    )

    # 验证 writer/scalar 被正确调用
    scalar_names = {name for (_, name, _, _) in calls if isinstance(name, str)}
    assert "skipped-train-samples" in scalar_names       # skipped_train_samples=10
    assert "grpo_collection_iteration" in scalar_names   # perform_rl_step=True
    assert "world-size" in scalar_names                  # log_world_size=True
    assert "mem-reserved-bytes" in scalar_names          # log_memory=True
    assert "mem-allocated-bytes" in scalar_names
    assert "mem-max-allocated-bytes" in scalar_names
    assert "loss-scale" in scalar_names                  # log_loss_scale=True
    assert "max_attention_logit" in scalar_names         # log_max_attention_logit=True
    assert "grad-norm" in scalar_names                   # grad_norm 非空
    assert "batch-size" in scalar_names                  # 基本标量

    # 验证 wandb 也写入了
    assert any(item[0] == "wandb" for item in calls)

    # 验证 GRPO 值计算正确
    # iteration=10, grpo_iterations=1, grpo_samples_per_iteration=128, global_batch_size=32
    # grpo_collection_iteration = 10 // (1 * (128//32)) = 10 // 4 = 2
    grpo_calls = [item for item in calls if item[0] == "scalar" and item[1] == "grpo_collection_iteration"]
    assert grpo_calls[0][2] == 2

    # 验证 rl_utils 被调用
    assert "rl-packing-metrics" in calls
