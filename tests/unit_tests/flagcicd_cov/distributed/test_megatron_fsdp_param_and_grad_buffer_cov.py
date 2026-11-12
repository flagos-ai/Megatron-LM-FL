# Copyright (c) 2026, BAAI. All rights reserved.
"""Megatron-FSDP buffer index packing, parameter grouping, and DataParallelBuffer shard access."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    DataParallelBuffer,
    _get_parameter_groups,
    build_data_parallel_buffer_index,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_build_data_parallel_buffer_index_packs_items_in_order():
    ddp_config = DistributedDataParallelConfig()

    item_map, bucket_index, shard_index = build_data_parallel_buffer_index(
        [torch.Size([4, 2]), torch.Size([3])],
        data_parallel_rank=1,
        data_parallel_world_size=2,
        is_data_distributed=False,
        ddp_config=ddp_config,
    )

    assert item_map[0].global_data_index == 0
    assert item_map[0].size == 8
    assert item_map[1].global_data_index == 8
    assert item_map[1].size == 3
    assert item_map[1].item_id == 1
    assert item_map[1].bucket_id == 0
    assert bucket_index.size == 11
    # Unsharded buffers keep a "virtual shard" aligned with the DP rank.
    assert shard_index.global_data_index == 5
    assert shard_index.local_data_index == 5
    assert shard_index.size == 5


def test_build_data_parallel_buffer_index_pads_bucket_for_sharding():
    ddp_config = DistributedDataParallelConfig(data_parallel_sharding_strategy="optim_grads_params")

    item_map, bucket_index, shard_index = build_data_parallel_buffer_index(
        [torch.Size([4, 2]), torch.Size([3])],
        data_parallel_rank=1,
        data_parallel_world_size=4,
        is_data_distributed=True,
        ddp_config=ddp_config,
    )

    # 11 elements are padded up to a multiple of the data parallel world size.
    assert bucket_index.size == 12
    assert shard_index.size == 3
    assert shard_index.global_data_index == 3
    assert shard_index.local_data_index == 0
    assert item_map[0].global_data_index == 0
    assert item_map[1].global_data_index == 8


def test_build_data_parallel_buffer_index_fills_alignment_gaps_with_fragments():
    ddp_config = DistributedDataParallelConfig()

    item_map, bucket_index, _ = build_data_parallel_buffer_index(
        [torch.Size([6]), torch.Size([2]), torch.Size([4])],
        data_parallel_rank=0,
        data_parallel_world_size=1,
        is_data_distributed=False,
        ddp_config=ddp_config,
        chunk_size_factor=4,
    )

    # The 6-element item leaves a 2-element gap up to the next multiple of 4,
    # which the 2-element fragment exactly fills.
    assert item_map[0].global_data_index == 0
    assert item_map[1].global_data_index == 6
    assert item_map[2].global_data_index == 8
    assert bucket_index.size == 12


def test_build_data_parallel_buffer_index_pairs_unaligned_remainders():
    ddp_config = DistributedDataParallelConfig()

    item_map, bucket_index, _ = build_data_parallel_buffer_index(
        [torch.Size([6]), torch.Size([6])],
        data_parallel_rank=0,
        data_parallel_world_size=1,
        is_data_distributed=False,
        ddp_config=ddp_config,
        chunk_size_factor=4,
    )

    # Both items have a remainder of 2 modulo 4, so their remainders share one
    # grid and the second item starts right after the first.
    assert item_map[0].global_data_index == 0
    assert item_map[1].global_data_index == 6
    assert bucket_index.size == 12


def test_get_parameter_groups_splits_by_unit_dtype_and_bucket_size():
    class ExpertUnit(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(dim, dim))
            self.norm = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.experts = torch.nn.Module()
            self.experts.dense = torch.nn.Parameter(torch.zeros(dim, dim))

    class NestedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unit0 = ExpertUnit(4)
            self.shared = torch.nn.Parameter(torch.zeros(4, 4))
            self.plain = torch.nn.Linear(4, 4)

    module = NestedModel()
    module.shared.shared_embedding = True
    policy = BucketingPolicy(
        suggested_bucket_size=16,
        fsdp_unit_modules=[ExpertUnit],
        data_parallel_sharding_strategy="optim_grads_params",
    )

    groups, param_to_group, bucket_to_bucket_group = _get_parameter_groups(module, policy, {})

    # torch.nn.Module.named_parameters() yields the root module's own parameters
    # (shared) before recursing into its children (unit0, plain).
    assert param_to_group[module.shared] == 0
    assert param_to_group[module.plain.weight] == 1
    assert param_to_group[module.plain.bias] == 2
    assert param_to_group[module.unit0.weight] == 3
    assert param_to_group[module.unit0.norm] == 4
    assert param_to_group[module.unit0.experts.dense] == 5

    # The fp32 and bf16 buckets of the same FSDP unit share one collective bucket group.
    assert bucket_to_bucket_group[3] == [3, 4]
    assert bucket_to_bucket_group[4] == [3, 4]
    # The expert bucket of the unit and the shared-embedding bucket stay separate.
    assert bucket_to_bucket_group[5] == [5]
    assert bucket_to_bucket_group[0] == [0]

    assert groups[0].params == [module.shared]
    assert groups[0].fsdp_unit_id is None
    assert groups[1].params == [module.plain.weight]
    assert groups[2].params == [module.plain.bias]
    assert groups[3].params == [module.unit0.weight]
    assert groups[3].fsdp_unit_id == 0
    assert groups[3].dtype == torch.float32
    assert groups[3].chunk_size_factor == 4
    assert groups[4].dtype == torch.bfloat16
    assert groups[4].chunk_size_factor == 1
    assert groups[5].is_expert_param is True


def test_data_parallel_buffer_slices_items_by_shard(model_parallel):
    model_parallel()
    params = [torch.nn.Parameter(torch.arange(2.0)), torch.nn.Parameter(torch.arange(2.0))]
    ddp_config = DistributedDataParallelConfig(data_parallel_sharding_strategy="optim_grads_params")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    buffer = DataParallelBuffer(
        ddp_config,
        params,
        is_data_distributed=True,
        bucket_id=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # 4 elements padded to a multiple of world_size, one shard per rank.
    assert buffer.bucket_index.size == world_size
    assert buffer.data_size == 1
    assert buffer.shard_bucket_index.global_data_index == rank

    buffer.init_data(torch.full((buffer.data_size,), float(rank)))
    # The first item covers global rows [0, 2); its intersection with this
    # rank's one-element shard is empty from rank 2 onwards (F1).
    if rank < 2:
        assert buffer.locate_item_in_global_item(0) == (rank, rank + 1)
    else:
        assert buffer.locate_item_in_global_item(0) == (0, 0)

    item_slice = buffer.get_item(0)
    if rank < 2:
        torch.testing.assert_close(item_slice, torch.tensor([float(rank)]), rtol=0, atol=0)
    else:
        assert item_slice.numel() == 0


def test_data_parallel_buffer_fetches_and_allocates_buckets(model_parallel):
    model_parallel()
    params = [torch.nn.Parameter(torch.arange(16.0))]
    ddp_config = DistributedDataParallelConfig()
    buffer = DataParallelBuffer(
        ddp_config,
        params,
        is_data_distributed=False,
        bucket_id=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    buffer.init_data(torch.arange(buffer.data_size, dtype=torch.float32))

    bucket = buffer.fetch_bucket()
    assert bucket.data.data_ptr() == buffer.data.data_ptr()

    item = buffer.get_item_from_bucket(bucket, 0)
    torch.testing.assert_close(item, torch.arange(16.0), rtol=0, atol=0)

    values = torch.arange(buffer.bucket_index.size, dtype=torch.float32)
    fresh = buffer.allocate_bucket_storage(dtype=torch.float32, init_values=values)
    torch.testing.assert_close(fresh.data, values, rtol=0, atol=0)

    buffer.set_item(0, torch.full((16,), 7.0))
    torch.testing.assert_close(buffer.get_item(0), torch.full((16,), 7.0), rtol=0, atol=0)

    # get_shard_from_local_buffer returns this rank's virtual shard of the
    # replicated buffer, so seed rank-dependent values first (F1).
    buffer.set_item(0, torch.arange(16.0))
    rank = torch.distributed.get_rank()
    shard = buffer.get_shard_from_local_buffer()
    torch.testing.assert_close(
        shard, torch.tensor([float(2 * rank), float(2 * rank + 1)]), rtol=0, atol=0
    )


def test_data_parallel_buffer_uses_custom_dtype_bucket(model_parallel):
    model_parallel()
    params = [torch.nn.Parameter(torch.arange(2.0))]
    buffer = DataParallelBuffer(
        DistributedDataParallelConfig(),
        params,
        is_data_distributed=False,
        bucket_id=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    buffer.init_data(torch.arange(buffer.data_size, dtype=torch.float32))

    bucket = buffer.fetch_bucket(dtype=torch.bfloat16)

    assert bucket.data.dtype == torch.bfloat16
    assert bucket.data.numel() == buffer.bucket_index.size
