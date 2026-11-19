# Copyright (c) 2026, BAAI. All rights reserved.
"""param_and_grad_buffer bucket groups dispatch scaled grad reductions and reset microbatch state."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import param_and_grad_buffer as pgb
from megatron.core.distributed.param_and_grad_buffer import (
    _LayerwiseAllGatherHandle,
    _ParamAndGradBucket,
    _ParamAndGradBucketGroup,
    group_params_for_buffers,
    partition_buckets,
    shard_buffer,
)
from megatron.core.optimizer.param_layout import BufferKey


class _StubGroup:
    def __init__(self, rank=0, size=2):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _StubHandle:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _StubCoalescingManager:
    # Stand-in for torch.distributed._coalescing_manager, the collective boundary
    # param_and_grad_buffer imports; records the instances the module under test created.
    created = []

    def __init__(self, group=None, async_ops=False):
        self.waited = False
        _StubCoalescingManager.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def wait(self):
        self.waited = True


def _make_bucket(scaling=0.5, values=(5.0, 6.0, 7.0, 8.0), offset=0):
    params = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    grad_data = torch.tensor(list(values))
    param_index_map = {params[0]: (offset, offset + 2, 0), params[1]: (offset + 2, offset + 4, 0)}
    bucket = _ParamAndGradBucket(
        params=params,
        param_data=torch.arange(offset, offset + 4.0),
        grad_data=grad_data,
        offset=offset,
        numel_unpadded=4,
        gradient_scaling_factor=scaling,
        bucket_id=0,
        param_index_map=param_index_map,
        params_with_extra_main_grads=[],
    )
    return bucket, params, grad_data


def test_shard_buffer_splits_into_equal_chunks():
    buffer = torch.arange(6.0)
    shards = shard_buffer(buffer, 3)
    assert [shard.tolist() for shard in shards] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    with pytest.raises(AssertionError):
        shard_buffer(torch.arange(5.0), 2)


def test_bucket_param_to_index_uses_global_offsets():
    bucket, params, _ = _make_bucket(offset=0)
    assert bucket.param_to_index[params[0]] == (0, 2)
    assert bucket.param_to_index[params[1]] == (2, 4)

    offset_bucket, offset_params, _ = _make_bucket(offset=8)
    assert offset_bucket.offset == 8
    assert offset_bucket.param_to_index[offset_params[0]] == (0, 2)
    assert offset_bucket.param_to_index[offset_params[1]] == (2, 4)


def test_bucket_set_layerwise_params_list_records_flat_sizes():
    bucket, params, _ = _make_bucket()
    bucket.set_layerwise_params_list([[params[0]], [params[1]]])
    assert bucket.layerwise_param_flat_sizes == [2, 2]
    assert bucket.layerwise_params_list == [[params[0]], [params[1]]]


def test_layerwise_all_gather_handle_waits_only_on_last_handle():
    handles = [_StubHandle(), _StubHandle()]
    wrapper = _LayerwiseAllGatherHandle(handles)
    wrapper.wait()
    assert handles[0].waited is False
    assert handles[1].waited is True
    assert wrapper.handles is None


def test_group_params_for_buffers_separates_dtype_expert_and_engram_groups():
    plain = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    plain_second = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    expert = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    expert.allreduce = False
    engram = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    engram.is_engram_embedding = True
    layerwise = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    layerwise.is_managed_by_layer_wise_optimizer = True

    groups = group_params_for_buffers(
        [plain, plain_second, bf16, expert, engram, layerwise], grad_reduce_in_fp32=False
    )

    expected = {
        BufferKey(torch.float32, torch.float32, False, False, False): (
            [plain, plain_second],
            [0, 1],
        ),
        BufferKey(torch.bfloat16, torch.bfloat16, False, False, False): ([bf16], [0]),
        BufferKey(torch.float32, torch.float32, True, False, False): ([expert], [0]),
        BufferKey(torch.float32, torch.float32, False, False, True): ([engram], [0]),
        BufferKey(torch.float32, torch.float32, False, True, False): ([layerwise], [0]),
    }
    assert groups == expected


def test_group_params_for_buffers_fp32_reduces_fp32_params_in_fp32():
    param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    groups = group_params_for_buffers([param], grad_reduce_in_fp32=True)
    assert groups == {BufferKey(torch.bfloat16, torch.float32, False, False, False): ([param], [0])}


def test_partition_buckets_empty_list_returns_empty():
    assert partition_buckets([]) == []


def _make_fake_buffer(param_dtype, buckets, ddp_config=None):
    return SimpleNamespace(
        param_dtype=param_dtype,
        buckets=buckets,
        ddp_config=ddp_config or DistributedDataParallelConfig(),
        data_parallel_group=_StubGroup(),
        data_parallel_world_size=2,
    )


def test_partition_buckets_without_fp8_puts_each_bucket_in_own_group():
    bucket_a, _, _ = _make_bucket()
    bucket_b, _, _ = _make_bucket()
    buffer = _make_fake_buffer(torch.bfloat16, [bucket_a, bucket_b])

    groups = partition_buckets([buffer])

    assert [group.buckets for group in groups] == [[bucket_a], [bucket_b]]


def test_partition_buckets_merges_non_fp8_into_last_fp8_group():
    fp8_bucket_a = SimpleNamespace(params_list=[])
    fp8_bucket_b = SimpleNamespace(params_list=[])
    plain_bucket, _, _ = _make_bucket()
    fp8_buffer = _make_fake_buffer(torch.uint8, [fp8_bucket_a, fp8_bucket_b])
    plain_buffer = _make_fake_buffer(torch.bfloat16, [plain_bucket])

    groups = partition_buckets([plain_buffer, fp8_buffer])

    assert [group.buckets for group in groups] == [[fp8_bucket_a], [fp8_bucket_b, plain_bucket]]


def test_partition_buckets_with_fp32_accumulation_keeps_single_bucket_groups():
    fp8_bucket_a = SimpleNamespace(params_list=[])
    fp8_bucket_b = SimpleNamespace(params_list=[])
    plain_bucket, _, _ = _make_bucket()
    fp8_buffer = _make_fake_buffer(torch.uint8, [fp8_bucket_a, fp8_bucket_b])
    plain_buffer = _make_fake_buffer(torch.bfloat16, [plain_bucket])

    groups = partition_buckets(
        [plain_buffer, fp8_buffer], reduce_scatter_with_fp32_accumulation=True
    )

    assert [group.buckets for group in groups] == [[fp8_bucket_a], [fp8_bucket_b], [plain_bucket]]


def test_partition_buckets_force_single_bucket_group():
    bucket_a, _, _ = _make_bucket()
    bucket_b, _, _ = _make_bucket()
    buffer = _make_fake_buffer(torch.bfloat16, [bucket_a, bucket_b])

    groups = partition_buckets([buffer], force_single_bucket_group=True)

    assert len(groups) == 1
    assert groups[0].buckets == [bucket_a, bucket_b]


def test_register_grad_ready_asserts_for_foreign_param():
    bucket, params, _ = _make_bucket()
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, overlap_param_gather=True)
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _StubGroup(), 2)
    group.golden_per_param_grad_ready_counts = {params[0]: 1, params[1]: 1}
    group.is_first_batch = False

    group.register_grad_ready(params[0])
    assert group.per_param_grad_ready_counts == {params[0]: 1}
    with pytest.raises(AssertionError, match="Param is not in the bucket group"):
        group.register_grad_ready(torch.nn.Parameter(torch.ones(1)))


def test_reset_records_golden_counts_after_first_batch():
    bucket, params, _ = _make_bucket()
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True)
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _StubGroup(), 2)
    group.per_param_grad_ready_counts = {params[0]: 1, params[1]: 2}
    assert group.is_first_batch is True

    group.reset()

    assert group.golden_per_param_grad_ready_counts == {params[0]: 1, params[1]: 2}
    assert group.is_first_batch is False
    assert group.per_param_grad_ready_counts == {}
    assert group.is_last_microbatch is True

    group.per_param_grad_ready_counts = {params[0]: 3}
    group.reset()
    assert group.golden_per_param_grad_ready_counts == {params[0]: 1, params[1]: 2}


def test_check_grads_accepts_finite_grads():
    bucket, _, grad_data = _make_bucket()
    group = _ParamAndGradBucketGroup(
        [bucket], DistributedDataParallelConfig(overlap_grad_reduce=False), _StubGroup(), 2
    )
    grad_data.fill_(1.5)

    group.check_grads(check_for_nan_or_inf=True, check_for_large=True)

    assert torch.all(grad_data == 1.5)


def test_finish_grad_sync_scales_and_reduces_synchronously(monkeypatch):
    bucket, _, grad_data = _make_bucket(scaling=0.5, values=(5.0, 6.0, 7.0, 8.0))
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=False)
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _StubGroup(), 2)
    reductions = []

    def fake_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        reductions.append(op)
        if op == torch.distributed.ReduceOp.AVG:
            tensor.div_(2)
        else:
            tensor.add_(tensor)
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(pgb, "_coalescing_manager", _StubCoalescingManager)

    group.finish_grad_sync()

    assert reductions == [torch.distributed.ReduceOp.SUM]
    torch.testing.assert_close(grad_data, torch.tensor([5.0, 6.0, 7.0, 8.0]), rtol=0, atol=0)
    assert group.grad_reduce_handle is None


def test_finish_grad_sync_averages_when_average_in_collective(monkeypatch):
    bucket, _, grad_data = _make_bucket(scaling=1.0, values=(4.0, 4.0, 4.0, 4.0))
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False, average_in_collective=True
    )
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _StubGroup(), 2)

    def fake_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        if op == torch.distributed.ReduceOp.AVG:
            tensor.div_(2)
        else:
            tensor.add_(tensor)
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(pgb, "_coalescing_manager", _StubCoalescingManager)

    group.finish_grad_sync()

    torch.testing.assert_close(grad_data, torch.tensor([2.0, 2.0, 2.0, 2.0]), rtol=0, atol=0)


def test_finish_grad_sync_waits_for_async_dispatch(monkeypatch):
    bucket, params, grad_data = _make_bucket(scaling=1.0, values=(2.0, 2.0, 2.0, 2.0))
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, overlap_param_gather=True)
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _StubGroup(), 2)
    group.golden_per_param_grad_ready_counts = {params[0]: 1, params[1]: 1}
    group.is_first_batch = False
    handle = _StubHandle()

    def fake_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        tensor.add_(tensor)
        return handle

    _StubCoalescingManager.created.clear()
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(pgb, "_coalescing_manager", _StubCoalescingManager)

    group.register_grad_ready(params[0])
    assert group.grad_reduce_handle is None
    group.register_grad_ready(params[1])
    # The bucket group tracks the coalescing manager, not the collective's return value.
    dispatch_manager = _StubCoalescingManager.created[-1]
    assert group.grad_reduce_handle is dispatch_manager
    assert dispatch_manager.waited is False

    grad_data_after_sync = grad_data.clone()
    group.finish_grad_sync()
    assert dispatch_manager.waited is True
    assert group.grad_reduce_handle is None
    torch.testing.assert_close(grad_data, grad_data_after_sync, rtol=0, atol=0)


def test_free_overlap_buffers_waits_on_pending_handle():
    bucket, _, _ = _make_bucket()
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, overlap_param_gather=True)
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _StubGroup(), 2)
    pending = _StubHandle()
    group.param_gather_handle = pending
    bucket.layerwise_gather_list = [torch.empty(2), torch.empty(2)]

    group.free_overlap_buffers()

    assert pending.waited is True
    assert group.param_gather_handle is None
    assert bucket.layerwise_gather_list is None
