# Copyright (c) 2026, BAAI. All rights reserved.
"""Tensor pool alloc/free/reuse accounting and per-pool status; group commit passes tensors through unchanged before any chunk is initialized."""

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    GPUTensorPool,
    OffloadTensorGroup,
    PipelineOffloadManager,
    fine_grained_offloading_group_commit,
)


@pytest.fixture
def offload_manager():
    PipelineOffloadManager.reset_instance()
    yield PipelineOffloadManager.get_instance()
    PipelineOffloadManager.reset_instance()


def test_pool_free_returns_tensor_to_free_queue_for_reuse():
    pool = GPUTensorPool(device="cpu")
    tensor = pool.allocate((2, 3), dtype=torch.float32)
    status = pool.get_pool_status((2, 3), torch.float32)
    assert status["total_count"] == 1
    assert status["allocated_count"] == 1
    assert status["free_count"] == 0
    assert status["utilization"] == 100.0
    pool.free(tensor)
    status = pool.get_pool_status((2, 3), torch.float32)
    assert status["allocated_count"] == 0
    assert status["free_count"] == 1
    reused = pool.allocate((2, 3), dtype=torch.float32)
    assert reused is tensor
    assert pool.get_pool_status()["global_stats"]["pool_hits"] == 1


def test_pool_free_raises_when_no_pool_exists_for_shape():
    pool = GPUTensorPool(device="cpu")
    with pytest.raises(ValueError, match="No pool exists for shape="):
        pool.free(torch.zeros(2, 3))


def test_pool_free_raises_when_tensor_does_not_belong_to_pool():
    pool = GPUTensorPool(device="cpu")
    pool.allocate((2, 3), dtype=torch.float32)
    with pytest.raises(
        ValueError, match="Attempting to free a tensor that doesn't belong to this pool"
    ):
        pool.free(torch.zeros(2, 3))


def test_pool_get_pool_status_requires_dtype_when_shape_is_given():
    pool = GPUTensorPool(device="cpu")
    with pytest.raises(ValueError, match="dtype must be specified when shape is provided"):
        pool.get_pool_status(shape=(2, 3))


def test_pool_get_pool_status_raises_for_unknown_pool():
    pool = GPUTensorPool(device="cpu")
    with pytest.raises(ValueError, match="No pool exists for shape="):
        pool.get_pool_status((2, 3), torch.float32)


def test_pool_get_pool_status_reports_per_pool_counts():
    pool = GPUTensorPool(device="cpu")
    first = pool.allocate((2, 3), dtype=torch.float32)
    second = pool.allocate((2, 3), dtype=torch.float32)
    pool.free(first)
    status = pool.get_pool_status((2, 3), torch.float32)
    assert status["shape"] == (2, 3)
    assert status["dtype"] == torch.float32
    assert status["total_count"] == 2
    assert status["allocated_count"] == 1
    assert status["free_count"] == 1
    assert status["utilization"] == 50.0
    assert second is not first


def test_pool_get_pool_status_reports_all_pools_and_global_stats():
    pool = GPUTensorPool(device="cpu")
    pool.allocate((2, 3), dtype=torch.float32)
    pool.allocate((4,), dtype=torch.int64)
    status = pool.get_pool_status()
    assert set(status["pools"]) == {
        (torch.Size([2, 3]), torch.float32),
        (torch.Size([4]), torch.int64),
    }
    assert status["global_stats"]["total_allocated"] == 2
    assert status["global_stats"]["current_in_use"] == 2
    assert status["global_stats"]["allocation_requests"] == 2
    assert status["global_stats"]["pool_misses"] == 2
    assert status["pools"][(torch.Size([2, 3]), torch.float32)]["total_count"] == 1
    assert status["pools"][(torch.Size([4]), torch.int64)]["total_count"] == 1


def test_offload_tensor_group_pop_tensor_returns_pushed_tensor_once():
    group = OffloadTensorGroup("linear")
    tensor = torch.randn(2, 4, device="cuda")
    group.push_tensor((1, 0), tensor)
    assert group.pop_tensor((1, 0)) is tensor
    with pytest.raises(KeyError, match="1, 0"):
        group.pop_tensor((1, 0))


def test_group_commit_returns_tensor_unchanged_before_chunk_initialization(offload_manager):
    tensor = torch.randn(2, 4, device="cuda")
    assert fine_grained_offloading_group_commit(tensor, "any_group") is tensor
