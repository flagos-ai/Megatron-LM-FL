# Copyright (c) 2026, BAAI. All rights reserved.
"""Megatron-FSDP bucket allocators, storage helpers, and gradient safety checks."""

import logging

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    FixedPoolAllocator,
    ParameterGroup,
    RotaryBucketAllocator,
    TemporaryBucketAllocator,
    _alloc_storage,
    _check_nan_in_grad,
    _dtype_size,
    _free_storage,
    _get_communication_stream,
    _p_assert,
    check_gpu_memory,
    gradient_reduce_preprocessing,
    override_sharded_param_methods_with_safety_checks,
    to_local_if_dtensor,
)
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils

BUFFER_LOGGER = "megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer"


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_p_assert_raises_assertion_error_with_message():
    with pytest.raises(AssertionError, match="cov failure message"):
        _p_assert(False, "cov failure message")


def test_p_assert_logs_error_without_raising_when_disabled(caplog):
    with caplog.at_level(logging.ERROR, logger=BUFFER_LOGGER):
        _p_assert(False, "cov logged message", raise_assertion_error=False)
    assert "cov logged message" in caplog.text


def test_get_communication_stream_prefers_given_stream():
    sentinel = object()
    assert _get_communication_stream(sentinel, DistributedDataParallelConfig()) is sentinel


def test_get_communication_stream_defaults_to_platform_current_stream():
    # torch.cuda.current_stream() returns a new Python wrapper on every call,
    # so streams must be compared by value, not by identity.
    assert (
        _get_communication_stream(None, DistributedDataParallelConfig())
        == get_platform().current_stream()
    )


def test_alloc_and_free_storage_resize_tensor_storage():
    tensor = torch.empty(0, dtype=torch.float32)
    _alloc_storage(tensor, torch.Size([4]))
    assert tensor._typed_storage()._size() == 4

    _free_storage(tensor)
    assert tensor._typed_storage()._size() == 0

    _alloc_storage(tensor, torch.Size([2]))
    assert tensor._typed_storage()._size() == 2


def test_rotary_bucket_allocator_recycles_global_buffers():
    allocator = RotaryBucketAllocator("cov-rotary")

    first = allocator.allocate(0, 4, torch.float32, torch.device("cpu"))
    second = allocator.allocate(1, 4, torch.float32, torch.device("cpu"))
    assert first.data.data_ptr() != second.data.data_ptr()
    assert allocator.num_global_buffer == 2

    # Re-allocating a bucket id that is still in use returns the same buffer.
    first_again = allocator.allocate(0, 4, torch.float32, torch.device("cpu"))
    assert first_again.data.data_ptr() == first.data.data_ptr()

    allocator.free(0)
    assert allocator.using_buffer == {1: 1}

    recycled = allocator.allocate(2, 4, torch.float32, torch.device("cpu"))
    assert recycled.data.data_ptr() == first.data.data_ptr()
    assert allocator.using_buffer == {1: 1, 2: 0}


def test_temporary_bucket_allocator_allocates_and_frees():
    allocator = TemporaryBucketAllocator()

    bucket = allocator.allocate(0, 4, torch.float32, torch.device("cpu"))
    assert bucket.data.numel() == 4
    assert allocator.allocate(0, 4, torch.float32, torch.device("cpu")).data.data_ptr() == (
        bucket.data.data_ptr()
    )

    allocator.free(0)
    assert 0 not in allocator.buckets
    allocator.free(0)


def _fsdp_group(params, fsdp_unit_id, dtype=torch.float32):
    return ParameterGroup(params=params, dtype=dtype, fsdp_unit_id=fsdp_unit_id)


def test_fixed_pool_allocator_rejects_pool_without_double_buffer_units():
    param = torch.nn.Parameter(torch.zeros(4))
    groups = [_fsdp_group([param], fsdp_unit_id=None)]

    with pytest.raises(AssertionError, match="Found no FSDP units to use fixed-size buffering"):
        FixedPoolAllocator(name="cov-fixed", fsdp_param_groups=groups, size=2)


def test_fixed_pool_allocator_recycles_buffers_across_equal_units(model_parallel):
    model_parallel()
    groups = [
        _fsdp_group(
            [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))], fsdp_unit_id=0
        ),
        _fsdp_group(
            [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))], fsdp_unit_id=1
        ),
        _fsdp_group([torch.nn.Parameter(torch.zeros(4))], fsdp_unit_id=None),
    ]
    allocator = FixedPoolAllocator(name="cov-fixed-pool", fsdp_param_groups=groups, size=2)
    assert allocator.fsdp_double_buffer_units == [0, 1]
    assert allocator.idle_buffer == [(0, 0), (1, 0)]

    bucket_zero = allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    bucket_none = allocator.allocate(2, 4, torch.float32, torch.device("cpu"))
    # The non-double-buffer unit falls back to the plain temporary allocator,
    # which owns its own tensor storage.
    assert bucket_none.data.data_ptr() != bucket_zero.data.data_ptr()

    bucket_one = allocator.allocate(1, 8, torch.float32, torch.device("cpu"))
    assert bucket_one.data.data_ptr() != bucket_zero.data.data_ptr()

    # Re-allocating a bucket id that is still using a buffer reuses the buffer.
    bucket_zero_again = allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    assert bucket_zero_again.data.data_ptr() == bucket_zero.data.data_ptr()

    allocator.free(0)
    assert allocator.using_buffer == {1: (1, 0)}
    recycled = allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    assert recycled.data.data_ptr() == bucket_zero.data.data_ptr()
    assert allocator.using_buffer == {1: (1, 0), 0: (0, 0)}


def test_fixed_pool_allocator_persistent_fallback_names_buffer_per_bucket(model_parallel):
    model_parallel()
    groups = [
        _fsdp_group(
            [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))], fsdp_unit_id=0
        ),
        _fsdp_group([torch.nn.Parameter(torch.zeros(4))], fsdp_unit_id=None),
    ]
    allocator = FixedPoolAllocator(
        name="cov-fixed-fallback",
        fsdp_param_groups=groups,
        size=2,
        fallback_to_persistent_buffer=True,
    )

    bucket = allocator.allocate(1, 4, torch.float32, torch.device("cpu"))
    assert bucket.data.numel() == 4
    # A persistent-fallback bucket keeps its own name in the global memory
    # buffer, so allocating it twice returns the same storage.
    again = allocator.allocate(1, 4, torch.float32, torch.device("cpu"))
    assert again.data.data_ptr() == bucket.data.data_ptr()

    # Freeing a persistent-fallback bucket is a no-op on the pool bookkeeping.
    allocator.free(1)
    assert allocator.using_buffer == {}


def test_dtype_size_reports_bytes_for_common_dtypes():
    assert _dtype_size(torch.float16) == 2
    assert _dtype_size(torch.bfloat16) == 2
    assert _dtype_size(torch.int16) == 2
    assert _dtype_size(torch.float32) == 4
    assert _dtype_size(torch.int32) == 4


def test_to_local_if_dtensor_passes_plain_tensors_through():
    tensor = torch.arange(4.0)
    assert to_local_if_dtensor(tensor) is tensor


def test_gradient_reduce_preprocessing_picks_reduce_op_and_scales():
    grad = torch.full((4,), 2.0)

    reduce_op = gradient_reduce_preprocessing(grad, None, DistributedDataParallelConfig())
    assert reduce_op == torch.distributed.ReduceOp.SUM
    torch.testing.assert_close(grad, torch.full((4,), 2.0), rtol=0, atol=0)

    avg_config = DistributedDataParallelConfig(average_in_collective=True)
    reduce_op = gradient_reduce_preprocessing(grad, 0.5, avg_config)
    assert reduce_op == torch.distributed.ReduceOp.AVG
    torch.testing.assert_close(grad, torch.full((4,), 2.0), rtol=0, atol=0)

    scaled_config = DistributedDataParallelConfig(gradient_reduce_div_fusion=False)
    reduce_op = gradient_reduce_preprocessing(grad, 0.5, scaled_config)
    assert reduce_op == torch.distributed.ReduceOp.SUM
    torch.testing.assert_close(grad, torch.full((4,), 1.0), rtol=0, atol=0)

    bf16_grad = torch.full((4,), 2.0, dtype=torch.bfloat16)
    div_fusion_config = DistributedDataParallelConfig(gradient_reduce_div_fusion=True)
    reduce_op = gradient_reduce_preprocessing(bf16_grad, 0.25, div_fusion_config)
    assert reduce_op == torch.distributed.ReduceOp.SUM
    torch.testing.assert_close(
        bf16_grad, torch.full((4,), 0.5, dtype=torch.bfloat16), rtol=0, atol=0
    )


def test_check_nan_in_grad_accepts_finite_and_rejects_nonfinite():
    _check_nan_in_grad(torch.randn(8))
    with pytest.raises(ValueError, match="Detected NaN or Inf in wgrad"):
        _check_nan_in_grad(torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="Detected NaN or Inf in wgrad"):
        _check_nan_in_grad(torch.tensor([1.0, float("inf")]))


def test_check_gpu_memory_threshold_bounds_allocated_ratio():
    # The check reports whether allocated or reserved memory is over the
    # threshold; both ratios are at most 1, so an unsatisfiable threshold can
    # never report near-full, and a zero threshold always does.
    assert check_gpu_memory(threshold=2.0) is False
    if get_platform().is_available():
        assert check_gpu_memory(threshold=0.0) is True


def test_override_sharded_param_methods_warn_for_freed_storage_params():
    live = torch.nn.Parameter(torch.randn(2, 2))
    freed = torch.nn.Parameter(torch.randn(2, 2))
    _free_storage(freed)

    override_sharded_param_methods_with_safety_checks([live, freed], None)

    moved = live.to(torch.float64)
    assert moved.dtype == torch.float64
    torch.testing.assert_close(moved, live.data.to(torch.float64), rtol=0, atol=0)
    assert live.cpu().device.type == "cpu"

    with pytest.warns(UserWarning, match="The parameter may be sharded by Megatron-FSDP"):
        empty_to = freed.to(torch.float64)
    assert empty_to.shape == ()
    with pytest.warns(UserWarning, match="The parameter may be sharded by Megatron-FSDP"):
        empty_cpu = freed.cpu()
    assert empty_cpu.shape == ()
    assert empty_cpu.device.type == "cpu"
