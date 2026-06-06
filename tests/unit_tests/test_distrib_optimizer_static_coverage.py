# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.optimizer.distrib_optimizer as distrib_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig


class _Group:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _Bucket:
    def __init__(self, numel, offset=0, unpadded=None):
        self.grad_data = torch.zeros(numel)
        self.offset = offset
        self.numel_unpadded = numel if unpadded is None else unpadded


class _Buffer:
    def __init__(self, params, bucket, rank=0, world_size=2):
        self.param_dtype = torch.float32
        self.grad_dtype = torch.float32
        self.data_parallel_group = _Group(size=world_size, rank=rank)
        self.buckets = [bucket]
        self._params = params

    def get_unpacked_index_map(self):
        return self._params


def _range_tuple(range_obj):
    return (range_obj.start, range_obj.end, range_obj.size)


def _optimizer_config_with_precision_aware_mode(enabled):
    config = OptimizerConfig()
    config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = enabled
    return config


def test_distributed_optimizer_range_and_gbuf_mapping_static_paths(monkeypatch):
    monkeypatch.setattr(distrib_optimizer.cur_platform, "device_name", lambda: "cpu")
    p0 = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
    p1 = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
    param_world_map = {
        p0: (10, 16, 6),
        p1: (16, 22, 6),
        p2: (22, 28, 6),
    }

    base = distrib_optimizer.Range(5, 11)
    assert _range_tuple(base) == (5, 11, 6)
    assert _range_tuple(base.normalize(0)) == (0, 6, 6)
    assert str(base) == "5,11 [6]"
    assert repr(base) == str(base)
    assert len(base) == 6

    param_range_map = distrib_optimizer.DistributedOptimizer._build_model_gbuf_param_range_map(
        param_world_map,
        distrib_optimizer.Range(16, 22),
        bucket_offset=10,
    )
    assert list(param_range_map) == [p1]
    assert _range_tuple(param_range_map[p1]["gbuf_world"]) == (16, 22, 6)
    assert _range_tuple(param_range_map[p1]["gbuf_world_in_bucket"]) == (6, 12, 6)
    assert _range_tuple(param_range_map[p1]["gbuf_local"]) == (0, 6, 6)
    assert _range_tuple(param_range_map[p1]["param"]) == (0, 6, 6)

    bucket = _Bucket(numel=18, offset=10)
    buffer = _Buffer(param_world_map, bucket, rank=1, world_size=3)
    model_range = distrib_optimizer.DistributedOptimizer._build_model_gbuf_range(buffer, 0)
    assert list(model_range["param_map"]) == [p1]
    gbuf_range_map = distrib_optimizer.DistributedOptimizer._build_gbuf_range_map(buffer)
    assert (torch.float32, torch.float32) in gbuf_range_map

    model_param_gbuf_map = distrib_optimizer.DistributedOptimizer._build_model_param_gbuf_map(
        [gbuf_range_map]
    )
    assert model_param_gbuf_map[p1] == (0, (torch.float32, torch.float32), 0)

    with pytest.raises(AssertionError, match="single bucket"):
        distrib_optimizer.DistributedOptimizer._build_model_param_gbuf_map(
            [gbuf_range_map, gbuf_range_map]
        )


def test_distributed_optimizer_group_ranges_and_main_param_shards_cpu_paths(monkeypatch):
    monkeypatch.setattr(distrib_optimizer.cur_platform, "device_name", lambda: "cpu")
    monkeypatch.setattr(distrib_optimizer.cur_platform, "current_device", lambda: "cpu")
    copy_calls = []
    monkeypatch.setattr(
        distrib_optimizer.tensor_parallel,
        "copy_tensor_model_parallel_attributes",
        lambda dst, src: copy_calls.append((dst.shape, src.shape)),
    )
    monkeypatch.setattr(distrib_optimizer, "is_float8tensor", lambda param: False)
    monkeypatch.setattr(distrib_optimizer, "is_nvfp4tensor", lambda param: False)

    fp32_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    fp32_param.shared = True
    bf16_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).to(torch.bfloat16))

    fp32_ranges = {
        "param_map": {
            fp32_param: {
                "gbuf_world": distrib_optimizer.Range(4, 8),
                "gbuf_world_in_bucket": distrib_optimizer.Range(4, 8),
                "gbuf_local": distrib_optimizer.Range(0, 4),
                "param": distrib_optimizer.Range(4, 8),
            }
        }
    }
    bf16_ranges = {
        "param_map": {
            bf16_param: {
                "gbuf_world": distrib_optimizer.Range(0, 4),
                "gbuf_world_in_bucket": distrib_optimizer.Range(0, 4),
                "gbuf_local": distrib_optimizer.Range(0, 4),
                "param": distrib_optimizer.Range(0, 4),
            }
        }
    }
    gbuf_ranges = [
        {(torch.float32, torch.float32): [fp32_ranges]},
        {(torch.bfloat16, torch.float32): [bf16_ranges]},
    ]
    param_gbuf_map = {
        fp32_param: (0, (torch.float32, torch.float32), 0),
        bf16_param: (1, (torch.bfloat16, torch.float32), 0),
    }
    param_groups = [
        {
            "params": [fp32_param],
            "lr_mult": 1.0,
            "wd_mult": 1.0,
            "is_decoupled_lr": False,
            "is_expert_parallel": False,
        },
        {
            "params": [bf16_param],
            "lr_mult": 2.0,
            "wd_mult": 0.0,
            "is_decoupled_lr": True,
            "is_expert_parallel": False,
        },
    ]

    local_map, group_ranges = distrib_optimizer.DistributedOptimizer._build_optimizer_group_ranges(
        param_groups,
        gbuf_ranges,
    )
    assert local_map[fp32_param] == (0, 0)
    assert local_map[bf16_param] == (1, 0)
    assert group_ranges[0]["orig_group"] is param_groups[0]

    groups = distrib_optimizer.DistributedOptimizer._build_model_and_main_param_groups(
        gbuf_ranges,
        param_gbuf_map,
        group_ranges,
        OptimizerConfig(use_precision_aware_optimizer=False),
    )
    model_float16, model_fp32, shard_float16, shard_fp32, shard_fp32_from_float16 = groups
    assert model_float16 == [[], [bf16_param]]
    assert model_fp32 == [[fp32_param], []]
    assert torch.equal(shard_fp32[0][0], fp32_param.view(-1)[4:8])
    assert shard_float16[1][0].dtype == torch.bfloat16
    assert shard_fp32_from_float16[1][0].dtype == torch.float32
    assert bf16_param.main_param_sharded is True
    assert copy_calls
    assert param_groups[0]["params"] == [shard_fp32[0][0]]
    assert param_groups[1]["params"] == [shard_fp32_from_float16[1][0]]

    pa_group_ranges = [
        {"params": [bf16_param], "orig_group": {"params": [bf16_param]}},
    ]
    pa_groups = distrib_optimizer.DistributedOptimizer._build_model_and_main_param_groups(
        [gbuf_ranges[1]],
        {bf16_param: (0, (torch.bfloat16, torch.float32), 0)},
        pa_group_ranges,
        _optimizer_config_with_precision_aware_mode(True),
    )
    assert pa_groups[4] == [[None]]
    assert pa_group_ranges[0]["orig_group"]["params"][0].dtype == torch.bfloat16

    bad_param = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    bad_ranges = [
        {
            (torch.float64, torch.float32): [
                {
                    "param_map": {
                        bad_param: {
                            "param": distrib_optimizer.Range(0, 1),
                            "gbuf_world": distrib_optimizer.Range(0, 1),
                            "gbuf_world_in_bucket": distrib_optimizer.Range(0, 1),
                            "gbuf_local": distrib_optimizer.Range(0, 1),
                        }
                    }
                }
            ]
        }
    ]
    with pytest.raises(TypeError, match="Wrapped parameters"):
        distrib_optimizer.DistributedOptimizer._build_model_and_main_param_groups(
            bad_ranges,
            {bad_param: (0, (torch.float64, torch.float32), 0)},
            [{"params": [bad_param], "orig_group": {"params": [bad_param]}}],
            OptimizerConfig(),
        )


def test_distributed_optimizer_lightweight_instance_state_helpers(monkeypatch):
    opt = object.__new__(distrib_optimizer.DistributedOptimizer)
    model_param = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    main_param = torch.nn.Parameter(torch.arange(2, dtype=torch.float32))
    opt.model_param_group_index_map = {model_param: (0, 0)}
    opt.optimizer = SimpleNamespace(
        param_groups=[{"params": [main_param]}],
        state={
            main_param: {
                "exp_avg": torch.ones(2),
                "exp_avg_sq": torch.ones(2) * 2,
                "step": 3,
            }
        },
    )
    opt.config = _optimizer_config_with_precision_aware_mode(False)

    tensors = opt._get_main_param_and_optimizer_states(model_param)
    assert set(tensors) == {"param", "exp_avg", "exp_avg_sq"}
    assert tensors["param"] is main_param
    opt._set_main_param_and_optimizer_states(
        model_param,
        {
            "param": torch.ones_like(main_param) * 7,
            "exp_avg": torch.ones(2) * 8,
            "exp_avg_sq": torch.ones(2) * 9,
        },
    )
    assert torch.equal(main_param, torch.ones(2) * 7)
    assert torch.equal(opt.optimizer.state[main_param]["exp_avg"], torch.ones(2) * 8)

    sharded_model_param = torch.nn.Parameter(torch.ones(2))

    class _PrecisionAwareOptimizer:
        def __init__(self):
            self.param_groups = [{"params": [sharded_model_param]}]
            self.state = {
                sharded_model_param: {
                    "master_param": torch.ones(2),
                    "exp_avg": torch.ones(2) * 2,
                    "meta": "skip",
                }
            }
            self.scaled = {}

        def get_unscaled_state(self, param, key):
            return self.state[param][key] + 1

        def set_scaled_state(self, param, key, value):
            self.scaled[(param, key)] = value

    opt.optimizer = _PrecisionAwareOptimizer()
    opt.config = _optimizer_config_with_precision_aware_mode(True)
    tensors = opt._get_main_param_and_optimizer_states(model_param)
    assert set(tensors) == {"param", "exp_avg"}
    assert torch.equal(tensors["param"], torch.ones(2) + 1)
    opt._set_main_param_and_optimizer_states(
        model_param, {"param": torch.ones(2) * 3, "exp_avg": torch.ones(2) * 4}
    )
    assert (sharded_model_param, "master_param") in opt.optimizer.scaled
    assert (sharded_model_param, "exp_avg") in opt.optimizer.scaled

    opt.gbuf_ranges = [
        {
            (torch.float32, torch.float32): [
                {
                    "param_map": {
                        model_param: {
                            "gbuf_local": distrib_optimizer.Range(0, 2),
                            "gbuf_world": distrib_optimizer.Range(4, 6),
                        }
                    }
                }
            ]
        }
    ]
    opt.per_bucket_numel = [{(torch.float32, torch.float32): [8]}]
    opt.per_bucket_numel_unpadded = [{(torch.float32, torch.float32): [6]}]
    state = opt.get_parameter_state_dp_reshardable()
    assert state["per_bucket_numel"] == opt.per_bucket_numel
    assert state[0][(torch.float32, torch.float32)][0][0]["gbuf_local_start"] == 0
    assert state[0][(torch.float32, torch.float32)][0][0]["gbuf_local_end"] == 2
