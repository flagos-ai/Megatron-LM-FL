# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

import megatron.training.async_utils as training_async
import megatron.training.one_logger_utils as one_logger_utils
import megatron.training.theoretical_memory_usage as memory_usage
import megatron.training.utils as training_utils


class _OneLogger:
    def __init__(self):
        self.store = {}
        self.metrics = []
        self.tags = []
        self.finished = False

    def get_context_manager(self):
        return nullcontext()

    def store_set(self, key, value):
        self.store[key] = value

    def store_get(self, key):
        return self.store[key]

    def store_has_key(self, key):
        return key in self.store

    def store_pop(self, key):
        return self.store.pop(key)

    def log_metrics(self, metrics):
        self.metrics.append(dict(metrics))

    def log_app_tag(self, tag):
        self.tags.append(tag)

    def finish(self):
        self.finished = True


def _args(**overrides):
    defaults = dict(
        app_tag_run_name=None,
        app_tag_run_version="v1",
        data_parallel_size=2,
        context_parallel_size=1,
        global_batch_size=8,
        micro_batch_size=2,
        pipeline_model_parallel_size=2,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=1,
        world_size=4,
        seq_length=16,
        log_throughput=True,
        hidden_size=32,
        kv_channels=8,
        num_attention_heads=4,
        group_query_attention=False,
        num_query_groups=None,
        num_experts=None,
        swiglu=False,
        moe_shared_expert_intermediate_size=None,
        num_layers=4,
        moe_layer_freq=1,
        moe_ffn_hidden_size=64,
        mtp_num_layers=None,
        normalization="LayerNorm",
        multi_latent_attention=False,
        q_lora_rank=None,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        kv_lora_rank=4,
        v_head_dim=8,
        ffn_hidden_size=64,
        moe_router_topk=2,
        padded_vocab_size=128,
        untie_embeddings_and_output_weights=False,
        use_distributed_optimizer=False,
        recompute_granularity="selective",
        sequence_parallel=True,
        virtual_pipeline_model_parallel_size=None,
        hybrid_layer_pattern=None,
        data_path=None,
        data_args_path=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
        save=None,
        bf16=False,
        log_device_memory_used=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _seed_one_logger(logger):
    logger.store_set(
        "get_e2e_base_metrics",
        lambda: dict(
            iteration=7,
            train_duration=4.0,
            eval_duration=2.0,
            eval_iterations=2,
            total_flops_since_current_train_start=8 * 10**12,
            num_floating_point_operations_so_far=12 * 10**12,
            consumed_train_samples=72,
            world_size=4,
            seq_length=16,
        ),
    )
    logger.store_set("iteration_start", 2)
    logger.store_set("train_samples_start", 8)
    logger.store_set("train_iterations_time_msecs_total", 1000.0)
    logger.store_set("tracked_train_iterations", 3)
    logger.store_set("validation_iterations_time_msecs_total", 500.0)
    logger.store_set("tracked_validation_iterations", 1)
    logger.store_set("train_throughput_per_gpu_max", 0.5)
    logger.store_set("save_checkpoint_count", 0)
    logger.store_set("save_checkpoint_sync_time_total", 0.0)
    logger.store_set("app_train_loop_start_time", 1234)


def test_one_logger_e2e_training_checkpoint_and_tag_paths(monkeypatch):
    logger = _OneLogger()
    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: logger)
    monkeypatch.setattr(one_logger_utils, "get_args", lambda: _args(app_tag_run_name="run"))
    monkeypatch.setattr(one_logger_utils, "get_timestamp_in_ms", lambda: 1000 + len(logger.metrics))

    one_logger_utils.on_pretrain_start()
    assert logger.store_get("app_tag_run_name") == "run"
    assert logger.metrics[-1]["app_run_type"] == "training"

    one_logger_utils.on_train_start(
        iteration=2,
        consumed_train_samples=8,
        train_samples=None,
        seq_length=16,
        train_iters=10,
        save="/tmp/ckpt",
        async_save=True,
        log_throughput=True,
        num_floating_point_operations_so_far=3 * 10**12,
    )
    assert logger.metrics[-1]["train_samples_target"] == 80
    assert logger.metrics[-1]["save_checkpoint_strategy"] == "async"

    _seed_one_logger(logger)
    metrics = one_logger_utils._produce_e2e_metrics(log_throughput=True, throughput=1.5)
    assert metrics["train_iterations"] == 5
    assert metrics["train_throughput_per_gpu_max"] == 1.5
    assert "train_iterations_time_msecs_min" in metrics
    one_logger_utils.track_e2e_metrics(log_throughput=True, throughput=1.6)
    assert logger.metrics[-1]["train_throughput_per_gpu_max"] == 1.6

    productive = one_logger_utils.on_save_checkpoint_start(async_save=True)
    assert productive["save_checkpoint_async_count"] == 1
    one_logger_utils.on_save_checkpoint_end(
        save_checkpoint_duration=0.25, current_iteration=7, async_save=True
    )
    one_logger_utils.on_save_checkpoint_success(productive, async_save=True)
    assert logger.store_get("iters_prod_max") == 7

    productive2 = one_logger_utils.on_save_checkpoint_start(async_save=False)
    productive2["train_iterations_productive_end"] = 8
    one_logger_utils.on_save_checkpoint_end(
        save_checkpoint_duration=0.5, current_iteration=8, async_save=False
    )
    one_logger_utils.on_save_checkpoint_success(productive2, async_save=False)
    assert logger.metrics[-1]["train_iterations_productive_end"] == 8
    assert logger.store_get("iters_prod_max") == 8

    one_logger_utils.track_config_flags(
        train_iters=10,
        skip_train=False,
        do_train=True,
        do_valid=False,
        do_test=True,
        dataloader_type="single",
    )
    assert logger.metrics[-1]["is_test_iterations_enabled"] is True
    logger.store_set("app_tag_run_name", "run")
    logger.store_set("app_tag_run_version", "v1")
    one_logger_utils.track_app_tag(batch_size=8, world_size=4, seq_length=16)
    assert logger.tags == ["run_v1_8_4_16"]
    one_logger_utils.finish()
    assert logger.finished is True

    monkeypatch.setattr(one_logger_utils, "get_one_logger", lambda: None)
    assert one_logger_utils._produce_e2e_metrics() is None
    assert one_logger_utils.on_save_checkpoint_start(async_save=False) is None


def test_theoretical_memory_dense_moe_mla_and_report_paths(monkeypatch):
    printed = []
    monkeypatch.setattr(memory_usage, "print_rank_0", lambda msg: printed.append(msg))
    monkeypatch.setattr(memory_usage, "is_hybrid_model", lambda args: False)

    dense = _args(pipeline_model_parallel_size=1, tensor_model_parallel_size=2)
    dense_weight = memory_usage.compute_weight_and_optimizer_memory(dense, verbose=True)
    dense_act = memory_usage.compute_activation_memory(dense, num_microbatches=1, verbose=True)
    dense_act_no_sp = memory_usage.compute_activation_memory_without_sp(
        dense, num_microbatches=1, verbose=True
    )
    assert dense_weight > 0
    assert dense_act > 0
    assert dense_act_no_sp > 0

    moe = _args(
        num_experts=4,
        moe_layer_freq=[1, 0, 1, 0],
        swiglu=True,
        moe_shared_expert_intermediate_size=16,
        mtp_num_layers=2,
        untie_embeddings_and_output_weights=True,
        use_distributed_optimizer=True,
        data_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    moe_weight = memory_usage.compute_weight_and_optimizer_memory(moe, verbose=True)
    assert moe_weight > 0
    assert moe_weight != dense_weight
    assert memory_usage.compute_activation_memory(moe, num_microbatches=4, verbose=True) > 0

    mla = _args(
        multi_latent_attention=True,
        group_query_attention=False,
        q_lora_rank=None,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
    )
    assert memory_usage.compute_weight_and_optimizer_memory(mla) > 0
    mla.q_lora_rank = 4
    assert memory_usage.compute_weight_and_optimizer_memory(mla) > 0

    totals = memory_usage.report_theoretical_memory(dense, num_microbatches=1, verbose=True)
    assert len(totals) == 3
    assert printed[-1] == "compute_activation_memory with SP"
    dense.sequence_parallel = False
    assert memory_usage.report_theoretical_memory(dense, num_microbatches=1)[2] > 0
    assert printed[-1] == "compute_activation_memory_without_sp"
    monkeypatch.setattr(memory_usage, "is_hybrid_model", lambda args: True)
    assert memory_usage.report_theoretical_memory(dense) is None

    with pytest.raises(AssertionError, match="Invalid length"):
        memory_usage.compute_weight_and_optimizer_memory(
            _args(num_experts=2, moe_layer_freq=[1, 0])
        )


def test_training_utils_cpu_masks_blend_and_rank_helpers(monkeypatch, tmp_path):
    data = torch.tensor([[1, 2, 0, 9], [4, 0, 5, 9]])
    attention_mask, loss_mask, position_ids = training_utils.get_ltor_masks_and_position_ids(
        data,
        eod_token=0,
        pad_token=9,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        pad_mask_loss=True,
    )
    assert attention_mask.shape == (2, 1, 4, 4)
    assert loss_mask[0, 2].item() == 0.0
    assert loss_mask[0, 3].item() == 0.0
    assert loss_mask[1, 3].item() == 0.0
    assert position_ids.shape == data.shape

    monkeypatch.setattr(training_utils, "_safe_get_rank", lambda: 0)
    assert training_utils.is_rank0() is True
    monkeypatch.setattr(training_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(training_utils.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(training_utils, "_safe_get_rank", lambda: 3)
    assert training_utils.is_last_rank() is True
    monkeypatch.setattr(training_utils.torch.distributed, "get_backend", lambda: "fake")
    training_utils.print_rank_last("visible on fake backend")
    assert training_utils.is_hybrid_model(_args(hybrid_layer_pattern="M-M*-")) is True

    monkeypatch.setattr(training_utils.mpu, "is_pipeline_first_stage", lambda **kwargs: kwargs["vp_stage"] == 0)
    monkeypatch.setattr(training_utils.mpu, "is_pipeline_last_stage", lambda **kwargs: kwargs["vp_stage"] == 1)
    assert training_utils.is_first_or_last_pipeline_stage(0) is True
    assert training_utils.is_first_or_last_pipeline_stage(1) is True

    monkeypatch.setattr(
        training_utils,
        "get_blend_from_list",
        lambda items: ("blend", tuple(items or [])),
    )
    data_args = tmp_path / "data.txt"
    data_args.write_text("0.7 train 0.3 valid")
    per_split = tmp_path / "per_split.json"
    per_split.write_text(
        json.dumps({"train": "1 train", "valid": ["1", "valid"], "test": None})
    )
    blend, blend_per_split = training_utils.get_blend_and_blend_per_split(
        _args(data_args_path=str(data_args))
    )
    assert blend[1] == ("0.7", "train", "0.3", "valid")
    assert blend_per_split is None
    blend, blend_per_split = training_utils.get_blend_and_blend_per_split(
        _args(per_split_data_args_path=str(per_split))
    )
    assert blend is None
    assert [item[0] for item in blend_per_split] == ["blend", "blend", "blend"]
    assert training_utils.get_blend_and_blend_per_split(_args()) == (None, None)

    writes = []
    monkeypatch.setattr(training_utils, "get_args", lambda: _args(save=str(tmp_path), world_size=8))
    monkeypatch.setattr(training_utils.torch.distributed, "barrier", lambda: writes.append("barrier"))
    monkeypatch.setattr(training_utils.torch.distributed, "get_rank", lambda: 0)
    training_utils.append_to_progress_log("step done", barrier=True)
    assert writes == ["barrier"]
    assert "step done" in (tmp_path / "progress.txt").read_text()


def test_training_async_queue_lifecycle_and_callbacks(monkeypatch):
    events = []

    class _Queue:
        warmed = []

        def __init__(self, persistent=False):
            self.persistent = persistent
            self.requests = []
            self.closed = []

        @classmethod
        def warmup_persistent_caller(cls, rank, **kwargs):
            cls.warmed.append((rank, kwargs))

        def schedule_async_request(self, request):
            self.requests.append(request)
            events.append(("schedule", request))

        def maybe_finalize_async_calls(self, blocking, no_dist=False):
            events.append(("finalize", blocking, no_dist))

        def get_num_unfinalized_calls(self):
            return len(self.requests)

        def close(self, abort=False):
            self.closed.append(abort)
            events.append(("close", abort))

    class _ResultQueue:
        def __init__(self):
            self._manager = SimpleNamespace(shutdown=lambda: events.append(("shutdown", None)))

    def _strategy(name, reader=None):
        module = {
            "AsyncCallsQueue": _Queue,
            "get_write_results_queue": lambda method: events.append(("results", method)),
        }
        if reader is not None:
            return None, SimpleNamespace(clear_metadata_cache=lambda: events.append(("clear", reader)))
        return name, module

    monkeypatch.setattr(training_async, "get_args", lambda: SimpleNamespace(
        async_strategy="mcore",
        use_persistent_ckpt_worker=True,
        async_ckpt_cpu_priority=3,
        async_ckpt_io_priority=4,
        async_save=True,
    ))
    monkeypatch.setattr(training_async, "get_async_strategy", _strategy)
    monkeypatch.setattr(training_async, "print_rank_0", lambda msg: events.append(("print", msg)))
    monkeypatch.setattr(training_async, "_async_calls_queue", None)
    monkeypatch.setattr(training_async, "_results_queue", _ResultQueue())
    checkpointing_stub = ModuleType("megatron.training.checkpointing")
    checkpointing_stub.finalize_deletion_processes = lambda blocking=False: events.append(
        ("delete", blocking)
    )
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing_stub)

    queue = training_async._get_async_calls_queue()
    assert queue.persistent is True
    training_async.schedule_async_save("request")
    assert events[-1] == ("schedule", "request")
    assert training_async.is_empty_async_queue() is False
    training_async.maybe_finalize_async_save(blocking=True, terminate=True)
    assert ("finalize", True, False) in events
    assert ("close", False) in events

    training_async.init_persistent_async_worker(rank=0, mp_mode="fork")
    assert _Queue.warmed[-1][1]["mp_mode"] == "fork"
    assert ("results", "fork") in events

    training_async.reset_persistent_async_worker("mcore")
    assert ("shutdown", None) in events
    assert ("clear", "CachedMetadataFileSystemReader") in events

    class _Writer:
        def get_save_function_and_args(self):
            return (
                lambda *args: events.append(("save", args)),
                lambda: events.append(("preload", None)),
                ("payload",),
            )

    class _Request:
        def __init__(self, save_fn, save_args, finalize_fns, async_fn_kwargs=None, preload_fn=None):
            self.save_fn = save_fn
            self.save_args = save_args
            self.finalize_fns = finalize_fns
            self.async_fn_kwargs = async_fn_kwargs
            self.preload_fn = preload_fn

    monkeypatch.setattr(training_async, "NVRxAsyncRequest", _Request)
    monkeypatch.setattr(
        training_async,
        "save_state_dict_async_finalize",
        lambda *args: events.append(("finalize-state", args)),
    )
    request = training_async.get_save_and_finalize_callbacks(_Writer(), ("state", "dict"))
    request.preload_fn()
    request.save_fn(*request.save_args)
    request.finalize_fns[0]()
    assert ("preload", None) in events
    assert ("save", ("payload",)) in events
    assert ("finalize-state", ("state", "dict")) in events

    monkeypatch.setattr(training_async, "get_args", lambda: SimpleNamespace(async_save=False))
    training_async.maybe_finalize_async_save(blocking=True, terminate=True)
