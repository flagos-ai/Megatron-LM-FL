# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
import megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer as fsdp_buffer


class _FakeGlobalMemoryBuffer:
    def __init__(self):
        self.calls = []

    def get_tensor(self, shape, dtype, name, mem_alloc_context=None):
        self.calls.append((tuple(shape), dtype, name, mem_alloc_context))
        return torch.empty(*shape, dtype=dtype)


def _patch_dist(monkeypatch, rank=0, world_size=2):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)


def _ddp_config(strategy="optim_grads_params"):
    return DistributedDataParallelConfig(data_parallel_sharding_strategy=strategy)


def test_fsdp_buffer_padding_indexes_and_assertion_helpers(monkeypatch):
    _patch_dist(monkeypatch, rank=1, world_size=2)
    cfg = _ddp_config()

    assert fsdp_buffer._pad(5, 4) == 8
    fsdp_buffer._p_assert(True, "ok")
    fsdp_buffer._p_assert(False, "logged only", raise_assertion_error=False)
    with pytest.raises(AssertionError, match="boom"):
        fsdp_buffer._p_assert(False, "boom")

    elements = [torch.Size([3]), torch.Size([5]), torch.Size([1]), torch.Size([4])]
    item_map, bucket_index, shard_index = fsdp_buffer.build_data_parallel_buffer_index(
        elements,
        data_parallel_rank=1,
        data_parallel_world_size=2,
        is_data_distributed=True,
        ddp_config=cfg,
        bucket_id=7,
        chunk_size_factor=4,
    )
    assert bucket_index.bucket_id == 7
    assert bucket_index.size % 8 == 0
    assert shard_index.local_data_index == 0
    assert shard_index.bucket_data_index == shard_index.size
    assert sorted(item_map) == [0, 1, 2, 3]

    unsharded = fsdp_buffer._get_dp_buffer_shard_bucket_index(
        bucket_index,
        is_data_distributed=False,
        data_parallel_world_size=2,
        data_parallel_rank=1,
    )
    assert unsharded.local_data_index == unsharded.global_data_index

    no_shard_cfg = _ddp_config("no_shard")
    _, no_shard_bucket, no_shard_local = fsdp_buffer.build_data_parallel_buffer_index(
        [torch.Size([2]), torch.Size([3])],
        data_parallel_rank=0,
        data_parallel_world_size=1,
        is_data_distributed=False,
        ddp_config=no_shard_cfg,
    )
    assert no_shard_bucket.size == 5
    assert no_shard_local.local_data_index == 0


def test_fsdp_temporary_storage_resize_and_rotary_allocators(monkeypatch):
    _patch_dist(monkeypatch)
    fake_global = _FakeGlobalMemoryBuffer()
    monkeypatch.setattr(fsdp_buffer, "get_global_memory_buffer", lambda: fake_global)
    monkeypatch.setattr(fsdp_buffer.cur_platform, "synchronize", lambda: None)

    temporary = fsdp_buffer.TemporaryBucketAllocator()
    first = temporary.allocate(0, 4, torch.float32, torch.device("cpu"))
    second = temporary.allocate(0, 8, torch.float32, torch.device("cpu"))
    assert first is second
    temporary.free(0)
    assert temporary.buckets == {}

    resize = fsdp_buffer.StorageResizeBasedBucketAllocator()
    resized = resize.allocate(3, 6, torch.float32, torch.device("cpu"))
    assert resized.data.numel() == 6
    resize.free(3)
    assert 3 in resize.buckets

    rotary = fsdp_buffer.RotaryBucketAllocator("rot")
    bucket0 = rotary.allocate(10, 5, torch.float32, torch.device("cpu"))
    assert bucket0.data.numel() == 5
    assert rotary.using_buffer[10] == 0
    assert rotary._get_gbuf_name(0) == "rot_0"
    same_bucket = rotary.allocate(10, 5, torch.float32, torch.device("cpu"))
    assert same_bucket.data.numel() == 5
    rotary.free(10)
    assert rotary.idle_buffer == [0]
    bucket1 = rotary.allocate(11, 7, torch.float32, torch.device("cpu"))
    assert bucket1.data.numel() == 7
    assert fake_global.calls[-1][2] == "rot_0"


def test_fsdp_fixed_pool_allocator_reuse_fallback_and_persistent_paths(monkeypatch):
    _patch_dist(monkeypatch)
    fake_global = _FakeGlobalMemoryBuffer()
    monkeypatch.setattr(fsdp_buffer, "get_global_memory_buffer", lambda: fake_global)
    monkeypatch.setattr(fsdp_buffer.cur_platform, "synchronize", lambda: None)

    p0 = torch.nn.Parameter(torch.ones(2, 2))
    p1 = torch.nn.Parameter(torch.ones(2, 2))
    p2 = torch.nn.Parameter(torch.ones(3))
    groups = [
        fsdp_buffer.ParameterGroup([p0], dtype=torch.float32, fsdp_unit_id=0),
        fsdp_buffer.ParameterGroup([p1], dtype=torch.float32, fsdp_unit_id=1),
        fsdp_buffer.ParameterGroup([p2], dtype=torch.float32, fsdp_unit_id=99),
    ]
    allocator = fsdp_buffer.FixedPoolAllocator("pool", groups, size=2)
    assert allocator.fsdp_double_buffer_units == [0, 1]
    assert allocator._is_two_bucket_group_equal([0], [1]) is True
    assert allocator._is_two_bucket_group_equal([0], [2]) is False

    bucket = allocator.allocate(0, 4, torch.float32, torch.device("cpu"))
    assert bucket.data.numel() == 4
    assert allocator.using_buffer[0] == (0, 0)
    same = allocator.allocate(0, 4, torch.float32, torch.device("cpu"))
    assert same.data.numel() == 4
    allocator.free(0)
    assert (0, 0) in allocator.idle_buffer

    fallback = allocator.allocate(2, 3, torch.float32, torch.device("cpu"))
    assert fallback.data.numel() == 3
    allocator.free(2)

    persistent = fsdp_buffer.FixedPoolAllocator(
        "persist", groups, size=1, fallback_to_persistent_buffer=True
    )
    persisted = persistent.allocate(2, 3, torch.float32, torch.device("cpu"))
    assert persisted.data.numel() == 3
    assert "persist_not_fit_in_fixed_pool_2" in fake_global.calls[-1][2]

    with pytest.raises(AssertionError, match="Found no FSDP units"):
        fsdp_buffer.FixedPoolAllocator(
            "bad",
            [fsdp_buffer.ParameterGroup([torch.nn.Parameter(torch.ones(1))], fsdp_unit_id=None)],
        )


def test_fsdp_data_parallel_buffer_unsharded_fetch_set_get_and_param_binding(monkeypatch):
    _patch_dist(monkeypatch, rank=0, world_size=2)
    p0 = torch.nn.Parameter(torch.zeros(2, 2))
    p1 = torch.nn.Parameter(torch.zeros(3))
    allocator = fsdp_buffer.TemporaryBucketAllocator()
    buffer = fsdp_buffer.DataParallelBuffer(
        _ddp_config("no_shard"),
        [p0, p1],
        is_data_distributed=False,
        bucket_id=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        temporary_bucket_allocator=allocator,
    )
    buffer.init_data(torch.zeros(buffer.data_size, dtype=torch.float32))
    buffer.set_item(0, torch.arange(4, dtype=torch.float32).view(2, 2))
    buffer.set_item(1, torch.tensor([4.0, 5.0, 6.0]))

    assert torch.equal(buffer.get_item(0), torch.arange(4, dtype=torch.float32))
    assert torch.equal(buffer.get_item(1), torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(buffer.get_shard_from_local_buffer(), buffer.data[: buffer.shard_bucket_index.size])

    fetched = buffer.fetch_bucket(set_param_data=True)
    assert fetched.data.data_ptr() == buffer.data.data_ptr()
    assert torch.equal(p0.data.flatten(), torch.arange(4, dtype=torch.float32))
    assert torch.equal(buffer.get_item_from_bucket(fetched, 1), torch.tensor([4.0, 5.0, 6.0]))

    temp = buffer.allocate_bucket_storage(dtype=torch.float64, init_values=torch.arange(7.0))
    assert temp.data.dtype == torch.float64
    assert torch.equal(temp.data, torch.arange(7.0, dtype=torch.float64))
    assert torch.equal(buffer.get_shard_from_bucket(temp), temp.data[: buffer.shard_bucket_index.size])
    buffer.free_bucket_storage()

    p0.main_grad = torch.ones_like(p0)
    p1.main_grad = torch.ones_like(p1)
    buffer.reset_param_main_grad()
    assert p0.main_grad is None
    assert p1.main_grad is None

    with pytest.raises(AssertionError, match="Data type mismatch"):
        buffer.init_data(torch.zeros(buffer.data_size, dtype=torch.float64))
    with pytest.raises(AssertionError, match="Data size mismatch"):
        buffer.init_data(torch.zeros(buffer.data_size + 1, dtype=torch.float32))


def test_fsdp_data_parallel_buffer_sharded_slices_and_empty_intersections(monkeypatch):
    _patch_dist(monkeypatch, rank=1, world_size=2)
    params = [
        torch.nn.Parameter(torch.zeros(4)),
        torch.nn.Parameter(torch.zeros(4)),
        torch.nn.Parameter(torch.zeros(2)),
    ]
    buffer = fsdp_buffer.DataParallelBuffer(
        _ddp_config(),
        params,
        is_data_distributed=True,
        bucket_id=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        dp_rank=1,
        temporary_bucket_allocator=fsdp_buffer.TemporaryBucketAllocator(),
    )
    buffer.init_data(torch.zeros(buffer.data_size, dtype=torch.float32))
    for item_id, param in enumerate(params):
        buffer.set_item(item_id, torch.arange(param.numel(), dtype=torch.float32) + 10 * item_id)

    shard = buffer.get_shard_from_local_buffer()
    assert shard.numel() == buffer.shard_bucket_index.size
    assert buffer.locate_item_in_global_item(0)[1] >= buffer.locate_item_in_global_item(0)[0]
    assert buffer._get_item_local_shard_index(0)[1] >= buffer._get_item_local_shard_index(0)[0]
    assert buffer._get_item_local_index(1)[1] >= buffer._get_item_local_index(1)[0]

    not_in_shard = None
    for item_id in buffer.item_index_map:
        if buffer.locate_item_in_global_item(item_id) == (0, 0):
            not_in_shard = item_id
            break
    if not_in_shard is not None:
        assert buffer.get_item(not_in_shard).numel() == 0

    only_shard = buffer.get_item(1, only_shard=True)
    assert only_shard.numel() <= params[1].numel()
    full_bucket = buffer.allocate_bucket_storage(shard=False)
    assert full_bucket.data.numel() == buffer.bucket_index.size
    assert buffer.get_shard_from_bucket(full_bucket).numel() == buffer.shard_bucket_index.size
    buffer.free_bucket_storage()

    shard_bucket = buffer.allocate_bucket_storage(shard=True)
    assert shard_bucket.data.numel() == buffer.shard_bucket_index.size
    buffer.free_bucket_storage()


def test_fsdp_parameter_grouping_policy_shared_expert_and_bucket_maps(monkeypatch):
    _patch_dist(monkeypatch)

    class _Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False)
            self.second = torch.nn.Linear(2, 2, bias=False)
            self.moe = torch.nn.Module()
            self.moe.experts = torch.nn.ModuleList([_Expert()])
            self.shared = torch.nn.Parameter(torch.ones(3))
            self.shared.shared_embedding = True

    model = _Model()
    expert_params = {
        param for name, param in model.named_parameters() if ".experts." in name
    }
    assert expert_params
    policy = fsdp_buffer.BucketingPolicy(
        suggested_bucket_size=4,
        fsdp_unit_modules=[torch.nn.Linear],
        data_parallel_sharding_strategy="optim_grads_params",
    )
    groups, param_to_group, bucket_to_group = fsdp_buffer._get_parameter_groups(
        model,
        policy,
        meta_device_init_fp8_params={},
    )
    assert groups
    assert set(param_to_group) == set(model.parameters())
    assert all(bucket_id in group_ids for bucket_id, group_ids in bucket_to_group.items())
    assert {
        param for group in groups if group.is_expert_param for param in group.params
    } == expert_params
    assert any(group.fsdp_unit_id is not None for group in groups)

    no_aggregate = fsdp_buffer._get_parameter_groups(
        model,
        fsdp_buffer.BucketingPolicy(suggested_bucket_size=None),
        meta_device_init_fp8_params={},
        bucket_group_by_fsdp_unit=False,
    )
    assert all(ids == [bucket_id] for bucket_id, ids in no_aggregate[2].items())


def test_fsdp_gradient_preprocessing_nan_memory_dtype_and_context_paths(monkeypatch):
    _patch_dist(monkeypatch)
    monkeypatch.setattr(fsdp_buffer.cur_platform, "device_name", lambda: "cpu")
    monkeypatch.setattr(fsdp_buffer.cur_platform, "synchronize", lambda: None)
    monkeypatch.setattr(fsdp_buffer.cur_platform, "is_available", lambda: True)
    monkeypatch.setattr(fsdp_buffer.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(fsdp_buffer.cur_platform, "memory_allocated", lambda device: 5)
    monkeypatch.setattr(fsdp_buffer.cur_platform, "memory_reserved", lambda device: 10)
    monkeypatch.setattr(
        fsdp_buffer.cur_platform,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=100),
    )

    grad = torch.ones(4)
    cfg = DistributedDataParallelConfig(gradient_reduce_div_fusion=True)
    monkeypatch.setattr(
        torch.distributed,
        "_make_nccl_premul_sum",
        lambda scale: ("premul", scale),
        raising=False,
    )
    assert fsdp_buffer.gradient_reduce_preprocessing(grad.clone(), 2.0, cfg) == ("premul", 2.0)
    cfg.gradient_reduce_div_fusion = False
    scaled = grad.clone()
    assert fsdp_buffer.gradient_reduce_preprocessing(scaled, 2.0, cfg) == torch.distributed.ReduceOp.SUM
    assert torch.equal(scaled, grad * 2.0)
    cfg.average_in_collective = True
    assert fsdp_buffer.gradient_reduce_preprocessing(grad.clone(), 2.0, cfg) == torch.distributed.ReduceOp.AVG
    assert fsdp_buffer.gradient_reduce_preprocessing(grad.clone(), None, cfg) == torch.distributed.ReduceOp.SUM

    fsdp_buffer._check_nan_in_grad(torch.ones(2))
    with pytest.raises(ValueError, match="Detected NaN or Inf"):
        fsdp_buffer._check_nan_in_grad(torch.tensor([float("nan")]))

    assert fsdp_buffer.check_gpu_memory(threshold=0.9) is False
    assert fsdp_buffer.check_gpu_memory(threshold=0.05) is True
    assert fsdp_buffer._dtype_size(torch.float32) == 4
    assert fsdp_buffer._dtype_size(torch.bfloat16) == 2
    assert fsdp_buffer.to_local_if_dtensor(torch.ones(1)).shape == torch.Size([1])

    ctx = fsdp_buffer.ResetParametersContext(init_param_with_fp8=False, with_cuda_rng_tracker=False)
    assert ctx.__enter__() is ctx
    assert ctx.__exit__(None, None, None) is None

    holder = SimpleNamespace(ddp_config=DistributedDataParallelConfig(nccl_ub=False))
    assert fsdp_buffer.ParamAndGradBuffer.get_mem_alloc_context(holder) is nullcontext
    with pytest.raises(AssertionError, match="NCCL UBR"):
        fsdp_buffer.ParamAndGradBuffer.manual_buffer_registration(holder)
