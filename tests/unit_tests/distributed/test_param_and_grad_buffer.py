# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
import math
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed import param_and_grad_buffer as pgb
from megatron.core.distributed.param_and_grad_buffer import (
    BufferType,
    _LayerwiseAllGatherHandle,
    _ParamAndGradBucket,
    _ParamAndGradBucketGroup,
    partition_buckets,
    shard_buffer,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import TestModel, Utils


def get_model_and_buffers(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    bias: bool,
    shared_embedding: bool,
    bucket_size: int,
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int = 1,
    grad_reduce_in_fp32: bool = True,
    param_name_patterns_for_fp32_local_accumulation: tuple = (),
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=bucket_size,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
        param_name_patterns_for_fp32_local_accumulation=param_name_patterns_for_fp32_local_accumulation,
    )
    model = TestModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=bias,
        shared_embedding=shared_embedding,
    ).bfloat16()

    # Wrap with DistributedDataParallel, and get underlying buffer.
    # Use dummy TransformerConfig with mostly default values. Avoid divide-by-zero
    # errors for num_attention_heads and num_layers.
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config=ddp_config, module=model
    )
    assert len(model.buffers) == 1
    param_and_grad_buffer = model.buffers[0]
    bucket_groups = model.bucket_groups

    return model, param_and_grad_buffer, bucket_groups


@pytest.mark.parametrize("bucket_size", [None, 9000, 9025, 9050, 18000, 18050, 20000])
@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shared_embedding", [False, True])
def test_bucket_sizes(
    bucket_size: Optional[int], use_distributed_optimizer: bool, bias: bool, shared_embedding: bool
):
    Utils.initialize_model_parallel()

    if shared_embedding and bias:
        # Don't bother running shared_embedding + bias since gold values are trickier to compute.
        return

    input_dim = 95
    output_dim = 95
    num_layers = 10
    _, param_and_grad_buffer, _ = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=bias,
        shared_embedding=shared_embedding,
        bucket_size=bucket_size,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=True,
        average_in_collective=False,
    )

    actual_numel_in_each_bucket = [
        bucket.numel_unpadded for bucket in param_and_grad_buffer.buckets
    ]
    actual_numel_padded_in_each_bucket = [
        bucket.grad_data.numel() for bucket in param_and_grad_buffer.buckets
    ]

    def _pad_if_needed(numel_unpadded, divisor):
        if use_distributed_optimizer:
            return math.ceil(numel_unpadded / divisor) * divisor
        return numel_unpadded

    def _pad_bucket_if_needed(numel_unpadded):
        # Want 128-byte alignment for distributed optimizer.
        divisor = math.lcm(parallel_state.get_data_parallel_world_size(), 128)
        return _pad_if_needed(numel_unpadded, divisor)

    def _pad_param_if_needed(numel_unpadded):
        # Want 64-byte alignment for params.
        return _pad_if_needed(numel_unpadded, 64)

    if bucket_size is None:
        # If bucket_size is infinite (None), number of buckets should be 1.
        if shared_embedding and use_distributed_optimizer:
            assert len(param_and_grad_buffer.buckets) == 2
        else:
            assert len(param_and_grad_buffer.buckets) == 1
    else:
        # Else, compute number of buckets.
        numel_in_each_bucket = []
        numel_padded_in_each_bucket = []
        numel_in_last_bucket = 0
        param_sizes = []
        for _ in range(num_layers):
            param_sizes.append(input_dim * output_dim)
            if bias:  # Include bias term.
                param_sizes.append(output_dim)
        # Create separate bucket for first parameter from reverse direction.
        if shared_embedding and use_distributed_optimizer:
            numel_in_each_bucket.append(param_sizes[-1])
            numel_padded_in_each_bucket.append(_pad_bucket_if_needed(param_sizes[-1]))
            param_sizes = param_sizes[:-1]
        # Iterate through params in backward direction.
        for param_size in param_sizes[::-1]:
            numel_in_last_bucket = _pad_param_if_needed(numel_in_last_bucket)
            numel_in_last_bucket += param_size
            if numel_in_last_bucket >= bucket_size:
                numel_in_each_bucket.append(numel_in_last_bucket)
                numel_padded_in_each_bucket.append(_pad_bucket_if_needed(numel_in_last_bucket))
                numel_in_last_bucket = 0
        if numel_in_last_bucket > 0:
            numel_in_each_bucket.append(numel_in_last_bucket)
            numel_padded_in_each_bucket.append(_pad_bucket_if_needed(numel_in_last_bucket))

        assert len(param_and_grad_buffer.buckets) == len(
            numel_in_each_bucket
        ), f"Buckets don't match (got {actual_numel_in_each_bucket} but should be {numel_in_each_bucket})"
        assert actual_numel_in_each_bucket == numel_in_each_bucket, (
            f"Number of parameters in each bucket should be {numel_in_each_bucket}, "
            f"but is {actual_numel_in_each_bucket}"
        )
        if use_distributed_optimizer:
            assert all(
                [
                    x % parallel_state.get_data_parallel_world_size() == 0
                    for x in actual_numel_padded_in_each_bucket
                ]
            ), (
                f"Size of each padded bucket should be divisible by "
                f"{parallel_state.get_data_parallel_world_size()}"
            )
        assert actual_numel_padded_in_each_bucket == numel_padded_in_each_bucket, (
            f"Number of parameters in each padded bucket should be {numel_padded_in_each_bucket}, "
            f"but is {actual_numel_padded_in_each_bucket}"
        )

    Utils.destroy_model_parallel()


def test_param_to_index_alignment_with_padding():
    """Ensure bucket-local param offsets honor padding when DistOpt pads params."""
    Utils.initialize_model_parallel()

    # With input_dim=4, output_dim=4:
    #   - weight: 4*4 = 16 elements
    #   - bias: 4 elements
    # Since 16 % 64 != 0, the bias must be padded away from the weight,
    # making padding observable.
    input_dim = 4
    output_dim = 4
    model, param_and_grad_buffer, _ = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        bias=True,
        shared_embedding=False,
        bucket_size=None,  # single bucket
        use_distributed_optimizer=True,  # enforces 64-element alignment
        overlap_grad_reduce=True,
        average_in_collective=False,
    )

    bucket = param_and_grad_buffer.buckets[0]
    naive_offset = 0
    padding_observed = False

    for param in bucket.params_list:
        global_start, global_end, _ = param_and_grad_buffer.param_index_map[param]
        expected_local_start = global_start - bucket.offset
        expected_local_end = global_end - bucket.offset
        local_start, local_end = bucket.param_to_index[param]

        # param_to_index should match the padded offsets used in the global buffer.
        assert (local_start, local_end) == (expected_local_start, expected_local_end)

        # At least one param should have been padded relative to naive packing.
        if local_start != naive_offset:
            padding_observed = True
        naive_offset = local_end

        # Verify the slice retrieved via param_to_index matches param.data view.
        param_slice = bucket.param_data.view(-1)[local_start:local_end]
        torch.testing.assert_close(param_slice, param.data.view(-1))

    assert padding_observed, (
        "Expected padding to be applied between params. "
        "Ensure model dimensions are chosen such that param sizes are not multiples of 64."
    )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
@pytest.mark.parametrize("average_in_collective", [False, True])
@pytest.mark.parametrize("num_distributed_optimizer_instances", [1, 2])
def test_grad_sync(
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int,
):
    Utils.initialize_model_parallel(
        num_distributed_optimizer_instances=num_distributed_optimizer_instances
    )
    # Skip test if num_distributed_optimizer_instances > 1 and not using distributed optimizer
    if num_distributed_optimizer_instances > 1 and not use_distributed_optimizer:
        pytest.skip("Multiple optimizer instances require distributed optimizer to be enabled")

    input_dim = 100
    output_dim = 100
    num_layers = 10
    model, param_and_grad_buffer, bucket_groups = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=True,
        shared_embedding=False,
        bucket_size=None,  # Group all params into single bucket.
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        average_in_collective=average_in_collective,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )
    param_to_bucket_group = {}
    for bucket_group in bucket_groups:
        for param in bucket_group.params:
            assert param not in param_to_bucket_group
            param_to_bucket_group[param] = bucket_group

    param_and_grad_buffer.grad_data.data.fill_(1.0)
    expected_grad_data_value_after_collective = 1
    # Data in param_and_grad_buffer.grad_data[0] is 1/DP.
    # When average_in_collective=False, the grad data is always first scaled by 1/DP and then
    # summed by AR/RS.
    # When use_distributed_optimizer=True, only rank0's param_and_grad_buffer.grad_data[0] is
    # updated; other ranks update another shard of grad_data while keeping
    # param_and_grad_buffer.grad_data[0] unchanged (=1/DP).
    if (
        use_distributed_optimizer
        and (not average_in_collective)
        and parallel_state.get_data_parallel_rank(
            with_context_parallel=True, partial_data_parallel=True
        )
        != 0
    ):
        expected_grad_data_value_after_collective /= parallel_state.get_data_parallel_world_size()

    register_grad_sync_context = (
        contextlib.nullcontext() if overlap_grad_reduce else pytest.raises(AssertionError)
    )

    # Call register_grad_ready for all params before starting test to seed tracking
    # data structures.
    params = list(model.parameters())
    for param in params:
        with register_grad_sync_context:
            bucket_group = param_to_bucket_group[param]
            bucket_group.register_grad_ready(param)
    # Call reset to set .is_first_batch to False.
    for param in params:
        bucket_group = param_to_bucket_group[param]
        bucket_group.reset()

    for i, param in enumerate(params):
        assert param in param_to_bucket_group
        bucket_group = param_to_bucket_group[param]
        finish_grad_sync_context = contextlib.nullcontext()
        if (
            i < (len(params) - 1)
            and overlap_grad_reduce
            and num_distributed_optimizer_instances == 1
        ):
            # Can't finish grad sync until all params have been registered ready.
            finish_grad_sync_context = pytest.raises(AssertionError)

        with register_grad_sync_context:
            bucket_group.register_grad_ready(param)

        with finish_grad_sync_context:
            # When overlap_grad_reduce is True, this should throw an assertion error until all
            # params in the model have registered their grad above.
            # When overlap_grad_reduce is False, the collective is forced through.
            bucket_group.finish_grad_sync()

        expected_grad_data_value = expected_grad_data_value_after_collective
        if overlap_grad_reduce and i < (len(params) - 1):
            expected_grad_data_value = 1
        assert param_and_grad_buffer.grad_data[0] == expected_grad_data_value

        if not overlap_grad_reduce:
            # Reset grad_data for subsequent collectives.
            param_and_grad_buffer.grad_data.data.fill_(1.0)

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("force_all_reduce", [False, True])
def test_force_all_reduce_uses_correct_collective(force_all_reduce: bool):
    """Test that force_all_reduce=True causes all-reduce to be used instead of reduce-scatter."""
    Utils.initialize_model_parallel()

    input_dim = 100
    output_dim = 100
    num_layers = 2
    model, param_and_grad_buffer, _ = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=True,
        shared_embedding=False,
        bucket_size=None,
        use_distributed_optimizer=True,  # This normally uses reduce-scatter.
        overlap_grad_reduce=False,
        average_in_collective=False,
    )

    # Mock the collective operations to track which one is called.
    with (
        mock.patch('torch.distributed.all_reduce') as mock_all_reduce,
        mock.patch(
            'megatron.core.distributed.param_and_grad_buffer.dist_reduce_scatter_func'
        ) as mock_reduce_scatter,
    ):
        # Set up the mocks to be no-ops.
        mock_all_reduce.return_value = None
        mock_reduce_scatter.return_value = None

        # Trigger the grad sync via the DDP model's finish_grad_sync method.
        model.finish_grad_sync(force_all_reduce=force_all_reduce)

        if force_all_reduce:
            # When force_all_reduce=True, all_reduce should be called.
            assert (
                mock_all_reduce.called
            ), "Expected all_reduce to be called when force_all_reduce=True"
            assert (
                not mock_reduce_scatter.called
            ), "Expected reduce_scatter NOT to be called when force_all_reduce=True"
        else:
            # When force_all_reduce=False with distributed optimizer, reduce_scatter should be called.
            assert (
                mock_reduce_scatter.called
            ), "Expected reduce_scatter to be called when force_all_reduce=False"
            assert (
                not mock_all_reduce.called
            ), "Expected all_reduce NOT to be called when force_all_reduce=False"

    Utils.destroy_model_parallel()


def test_manual_bucket_group_cpu_sync_partition_and_buffer_helpers(monkeypatch):
    class _Group:
        def __init__(self, rank=0, size=2):
            self._rank = rank
            self._size = size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    class _Handle:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    class _Coalescing:
        def __init__(self, group, async_ops=False):
            self.group = group
            self.async_ops = async_ops
            self.waited = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def wait(self):
            self.waited = True

    p0 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p1 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p0.main_grad = torch.full_like(p0, 5.0)
    p0.main_grad_copy_in_grad_buffer = torch.zeros_like(p0)
    param_index_map = {p0: (0, 2, 0), p1: (2, 4, 0)}
    grad_data = torch.arange(4.0)
    bucket = _ParamAndGradBucket(
        params=[p0, p1],
        param_data=torch.arange(4.0),
        grad_data=grad_data,
        offset=0,
        numel_unpadded=4,
        gradient_scaling_factor=0.5,
        bucket_id=0,
        param_index_map=param_index_map,
        params_with_extra_main_grads=[p0],
    )
    assert bucket.param_to_index[p1] == (2, 4)
    bucket.set_layerwise_params_list([[p0], [p1]])
    assert bucket.layerwise_param_flat_sizes == [2, 2]
    assert [shard.tolist() for shard in shard_buffer(torch.arange(6), 3)] == [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(AssertionError):
        shard_buffer(torch.arange(5), 2)

    all_reduce_calls = []

    def _all_reduce(tensor, op=None, group=None, async_op=False):
        all_reduce_calls.append((tensor.clone(), op, group, async_op))
        tensor.add_(1.0)
        return _Handle() if async_op else None

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(pgb, "_coalescing_manager", _Coalescing)

    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=False)
    group = _ParamAndGradBucketGroup([bucket], ddp_config, _Group(), 2)
    group.per_param_grad_ready_counts = {p0: 1, p1: 2}
    group.reset()
    assert group.golden_per_param_grad_ready_counts == {p0: 1, p1: 2}
    assert group.is_first_batch is False

    group.finish_grad_sync(force_all_reduce=True)
    assert all_reduce_calls
    torch.testing.assert_close(p0.main_grad, p0.main_grad_copy_in_grad_buffer)
    assert p0.main_grad_copy_in_grad_buffer.tolist() == [3.5, 3.5]

    overlap_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        overlap_param_gather=True,
    )
    overlap_group = _ParamAndGradBucketGroup([bucket], overlap_config, _Group(), 2)
    overlap_group.golden_per_param_grad_ready_counts = {p0: 1, p1: 1}
    overlap_group.is_first_batch = False
    overlap_group.register_grad_ready(p0)
    assert overlap_group.per_param_grad_ready_counts == {p0: 1}
    with pytest.raises(AssertionError, match="Param is not"):
        overlap_group.register_grad_ready(torch.nn.Parameter(torch.ones(1)))
    overlap_group.register_grad_ready(p1, force_all_reduce=True)
    assert overlap_group.grad_reduce_handle is not None
    overlap_group.finish_grad_sync()
    assert overlap_group.grad_reduce_handle is None

    gathered_remote = torch.tensor([9.0, 10.0])

    def _all_gather(gather_list, tensor, group=None, async_op=False):
        gather_list[1].copy_(gathered_remote)
        return _Handle() if async_op else None

    monkeypatch.setattr(torch.distributed, "all_gather", _all_gather)
    bucket.set_layerwise_params_list([[p0], [p1]])
    overlap_group.start_param_sync(force_sync=True)
    torch.testing.assert_close(p1.detach(), gathered_remote)
    assert bucket.layerwise_gather_list is None
    assert bucket._layerwise_src_buffer is None

    bucket.set_layerwise_params_list([[p0], [p1]])
    overlap_group.param_gather_dispatched = False
    overlap_group.start_param_sync()
    assert overlap_group.param_gather_handle is not None
    overlap_group.finish_param_sync(skip_next_bucket_dispatch=True)
    assert overlap_group.param_gather_handle is None
    torch.testing.assert_close(p1.detach(), gathered_remote)

    pending = _Handle()
    overlap_group.param_gather_handle = pending
    bucket.layerwise_gather_list = [torch.empty(2), torch.empty(2)]
    bucket._layerwise_src_buffer = torch.empty(2)
    overlap_group.free_overlap_buffers()
    assert pending.waited is True
    assert bucket.layerwise_gather_list is None

    wrapper = _LayerwiseAllGatherHandle([_Handle(), _Handle()])
    last = wrapper.handles[-1]
    wrapper.wait()
    assert last.waited is True and wrapper.handles is None

    class _Buffer:
        def __init__(self, dtype, buckets, config=ddp_config):
            self.param_dtype = dtype
            self.buckets = buckets
            self.ddp_config = config
            self.data_parallel_group = _Group()
            self.data_parallel_world_size = 2

    bfloat_buffer = _Buffer(torch.bfloat16, [bucket])
    fp8_bucket_a = SimpleNamespace(params_list=[])
    fp8_bucket_b = SimpleNamespace(params_list=[])
    fp8_buffer = _Buffer(torch.uint8, [fp8_bucket_a, fp8_bucket_b])
    assert partition_buckets([]) == []
    assert len(partition_buckets([bfloat_buffer], force_single_bucket_group=True)) == 1
    assert len(partition_buckets([bfloat_buffer])) == 1
    fp8_groups = partition_buckets([fp8_buffer, bfloat_buffer])
    assert fp8_groups[-1].buckets == [fp8_bucket_b, bucket]
    split_groups = partition_buckets(
        [fp8_buffer, bfloat_buffer], reduce_scatter_with_fp32_accumulation=True
    )
    assert [g.buckets for g in split_groups] == [[fp8_bucket_a], [fp8_bucket_b], [bucket]]
    with pytest.raises(AssertionError):
        partition_buckets([bfloat_buffer, _Buffer(torch.bfloat16, [bucket])])

    tiny_buffer = mock.Mock()
    tiny_buffer.ddp_config = ddp_config
    tiny_buffer.data_parallel_world_size = 1
    tiny_buffer.param_data = torch.arange(4.0)
    tiny_buffer.grad_data = torch.arange(4.0)
    tiny_buffer.param_data_cpu = None
    tiny_buffer.grad_data_size = 0
    tiny_buffer.has_nvfp4_params = False
    tiny_buffer.param_index_map = param_index_map
    tiny_buffer._get = pgb._ParamAndGradBuffer._get.__get__(tiny_buffer, pgb._ParamAndGradBuffer)
    assert tiny_buffer._get(torch.Size([2]), 1, BufferType.PARAM).tolist() == [1.0, 2.0]
    assert tiny_buffer._get(torch.Size([2]), 2, BufferType.GRAD).tolist() == [2.0, 3.0]
    with pytest.raises(Exception, match="Illegal buffer type"):
        tiny_buffer._get(torch.Size([1]), 0, object())
    assert pgb._ParamAndGradBuffer.get_unpacked_index_map(tiny_buffer) is param_index_map


class TestFreeOverlapBuffers:
    """Tests for free_overlap_buffers() which releases GPU memory before async checkpoint saves."""

    @staticmethod
    def _make_model():
        """Create a DDP-wrapped model with overlap_param_gather enabled."""
        Utils.initialize_model_parallel()
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=None,
        )
        module = TestModel(
            input_dim=32, output_dim=32, num_layers=2, bias=False, shared_embedding=False
        ).bfloat16()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config=ddp_config,
            module=module,
        )
        return model

    def test_bucket_group_clears_buffers(self):
        """free_overlap_buffers on a bucket group should None-out per-bucket layerwise buffers."""
        model = self._make_model()

        for bg in model.bucket_groups:
            # Simulate buffers that would be allocated by start_param_sync.
            for bucket in bg.buckets:
                bucket.layerwise_gather_list = [torch.empty(8), torch.empty(8)]
                bucket._layerwise_src_buffer = torch.empty(16)

            bg.free_overlap_buffers()

            for bucket in bg.buckets:
                assert (
                    bucket.layerwise_gather_list is None
                ), "layerwise_gather_list should be None after free_overlap_buffers"
                assert (
                    bucket._layerwise_src_buffer is None
                ), "_layerwise_src_buffer should be None after free_overlap_buffers"

        Utils.destroy_model_parallel()

    def test_bucket_group_waits_on_pending_handle(self):
        """free_overlap_buffers should wait() on any pending param_gather_handle."""
        model = self._make_model()

        for bg in model.bucket_groups:
            mock_handle = mock.MagicMock()
            bg.param_gather_handle = mock_handle

            bg.free_overlap_buffers()

            mock_handle.wait.assert_called_once()
            assert (
                bg.param_gather_handle is None
            ), "param_gather_handle should be None after free_overlap_buffers"

        Utils.destroy_model_parallel()

    def test_bucket_group_noop_when_no_buffers(self):
        """free_overlap_buffers should be safe to call when no buffers are allocated."""
        model = self._make_model()

        for bg in model.bucket_groups:
            assert bg.param_gather_handle is None
            for bucket in bg.buckets:
                assert bucket.layerwise_gather_list is None
                assert bucket._layerwise_src_buffer is None

            # Should not raise.
            bg.free_overlap_buffers()

        Utils.destroy_model_parallel()

    def test_ddp_free_overlap_buffers_delegates(self):
        """DDP.free_overlap_buffers should call free_overlap_buffers on all bucket groups."""
        model = self._make_model()

        with mock.patch.object(type(model.bucket_groups[0]), 'free_overlap_buffers') as mock_free:
            model.free_overlap_buffers()
            assert mock_free.call_count == len(
                model.bucket_groups + model.expert_parallel_bucket_groups
            ), "free_overlap_buffers should be called on every bucket group"

        Utils.destroy_model_parallel()


class TestFP32LocalGradAccumulation:
    """Tests for the FP32 local gradient accumulation feature
    (param_name_patterns_for_fp32_local_accumulation)."""

    @staticmethod
    def _make_model(patterns, bucket_size=None):
        """Create a DDP-wrapped model with FP32 local grad accumulation patterns."""
        return get_model_and_buffers(
            input_dim=100,
            output_dim=100,
            num_layers=3,
            bias=True,
            shared_embedding=False,
            bucket_size=bucket_size,
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            average_in_collective=False,
            grad_reduce_in_fp32=False,
            param_name_patterns_for_fp32_local_accumulation=patterns,
        )

    def test_config_validation_with_grad_reduce_in_fp32(self):
        """param_name_patterns_for_fp32_local_accumulation and grad_reduce_in_fp32 are
        mutually exclusive."""
        with pytest.raises(AssertionError):
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, param_name_patterns_for_fp32_local_accumulation=('all',)
            )

    def test_pattern_matching_creates_fp32_main_grad(self):
        """Params matching patterns should get a float32 main_grad and a
        main_grad_copy_in_grad_buffer; non-matching params should not."""
        Utils.initialize_model_parallel()
        # Match only weight params (not bias).
        model, buf, _ = self._make_model(patterns=('*.weight',))

        for name, param in model.module.named_parameters():
            if 'weight' in name:
                assert param.main_grad.dtype == torch.float32, f"{name} main_grad should be float32"
                assert hasattr(param, 'main_grad_copy_in_grad_buffer')
                assert param.main_grad_copy_in_grad_buffer is not None
                # The copy in grad buffer should be in the buffer's grad dtype (bf16).
                assert param.main_grad_copy_in_grad_buffer.dtype == buf.grad_dtype
            else:
                # Bias params should not be promoted.
                assert (
                    param.main_grad.dtype == buf.grad_dtype
                ), f"{name} main_grad should remain in grad_dtype"
                assert getattr(param, 'main_grad_copy_in_grad_buffer', None) is None

        Utils.destroy_model_parallel()

    def test_all_pattern_matches_every_param(self):
        """The 'all' pattern should match every parameter."""
        Utils.initialize_model_parallel()
        model, buf, _ = self._make_model(patterns=('all',))

        for name, param in model.module.named_parameters():
            assert (
                param.main_grad.dtype == torch.float32
            ), f"{name} main_grad should be float32 with 'all' pattern"
            assert getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None

        Utils.destroy_model_parallel()

    def test_bucket_tracks_params_with_extra_main_grads(self):
        """Each bucket's params_with_extra_main_grads should contain exactly
        the params that matched the patterns."""
        Utils.initialize_model_parallel()
        model, buf, _ = self._make_model(patterns=('*.weight',))

        promoted_params = set()
        for name, param in model.module.named_parameters():
            if 'weight' in name:
                promoted_params.add(param)

        bucket_promoted = set()
        for bucket in buf.buckets:
            for param in bucket.params_with_extra_main_grads:
                bucket_promoted.add(param)
            # Every param in params_with_extra_main_grads should also be in bucket.params.
            assert bucket.params_with_extra_main_grads == [] or set(
                bucket.params_with_extra_main_grads
            ).issubset(bucket.params)

        assert (
            bucket_promoted == promoted_params
        ), "Bucket-tracked promoted params should match the set of pattern-matched params"

        Utils.destroy_model_parallel()

    def test_no_patterns_means_no_extra_main_grads(self):
        """With no patterns, no params should have extra main_grads."""
        Utils.initialize_model_parallel()
        _, buf, _ = self._make_model(patterns=())

        assert len(buf.extra_main_grads) == 0
        for bucket in buf.buckets:
            assert len(bucket.params_with_extra_main_grads) == 0

        Utils.destroy_model_parallel()

    def test_reset_zeros_extra_main_grads(self):
        """reset() should zero out both grad_data and all extra main_grads."""
        Utils.initialize_model_parallel()
        _, buf, _ = self._make_model(patterns=('all',))

        # Fill extra main_grads and grad_data with non-zero values.
        buf.grad_data.fill_(1.0)
        for grad in buf.extra_main_grads:
            grad.fill_(42.0)

        buf.reset()

        assert torch.all(buf.grad_data == 0), "grad_data should be zeroed after reset"
        for grad in buf.extra_main_grads:
            assert torch.all(grad == 0), "extra main_grads should be zeroed after reset"

        Utils.destroy_model_parallel()

    def test_scale_gradients_scales_extra_main_grads(self):
        """scale_gradients() should scale both grad_data and extra main_grads."""
        Utils.initialize_model_parallel()
        _, buf, _ = self._make_model(patterns=('all',))

        buf.grad_data.fill_(2.0)
        for grad in buf.extra_main_grads:
            grad.fill_(4.0)

        buf.scale_gradients(0.5)

        assert torch.allclose(
            buf.grad_data, torch.tensor(1.0, dtype=buf.grad_data.dtype)
        ), "grad_data should be scaled"
        for grad in buf.extra_main_grads:
            assert torch.allclose(
                grad, torch.tensor(2.0, dtype=grad.dtype)
            ), "extra main_grads should be scaled"

        Utils.destroy_model_parallel()

    def test_grad_sync_copies_to_and_from_comm_buffer(self):
        """During grad sync, values in FP32 main_grad should be copied to the comm buffer
        before the collective, and the reduced result should be copied back afterward."""
        Utils.initialize_model_parallel()
        model, buf, bucket_groups = self._make_model(patterns=('all',))

        # Simulate accumulated gradients in FP32 main_grad.
        for param in model.parameters():
            param.main_grad.fill_(1.0)

        # Run grad sync (non-overlapped, so finish_grad_sync triggers start + wait).
        model.finish_grad_sync()

        # After sync, main_grad should contain the reduced result (not the original 1.0,
        # since the collective may have scaled / averaged). The key invariant is that
        # main_grad should equal main_grad_copy_in_grad_buffer (the comm buffer slice)
        # after the copy-back.
        for param in model.parameters():
            if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                torch.testing.assert_close(
                    param.main_grad,
                    param.main_grad_copy_in_grad_buffer.float(),
                    msg="main_grad should equal comm buffer after grad sync copy-back",
                )

        Utils.destroy_model_parallel()
