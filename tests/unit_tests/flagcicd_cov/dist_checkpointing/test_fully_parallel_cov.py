# Copyright (c) 2026, BAAI. All rights reserved.

"""Fully parallel save/load wrappers: roundtrips through wrapped torch_dist strategies, shard-distribution caching across saves and loads, and tensors-metadata delegation."""

import shutil

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.exchange_utils import _sharded_tensor_shard_id
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
    _fill_in_deferred_sharded_items,
    distribute_main_replicas_with_precomputed_distribution,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def shared_ckpt_root(tmp_path_factory):
    root = tmp_path_factory.getbasetemp().parent / 'flagcicd_dist_ckpt'
    if Utils.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    yield root
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if Utils.rank == 0:
        shutil.rmtree(root, ignore_errors=True)


def _make_sharded_state_dict():
    """A replicated key with identical values and a rank-unique chunked key."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return {
        'sd_repl': ShardedTensor.from_rank_offsets('key_repl', torch.ones(4), replica_id=rank),
        'sd_chunk': ShardedTensor.from_rank_offsets(
            'key_chunk', torch.ones(3) * (rank + 1), (0, rank, world_size)
        ),
    }


def _make_ckpt_dir(root, name):
    ckpt_dir = root / name
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    return ckpt_dir


def test_fully_parallel_async_save_roundtrips_through_wrappers(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    state_dict = _make_sharded_state_dict()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'fully_parallel_async')

    save_wrapper = FullyParallelSaveStrategyWrapper(
        TorchDistSaveShardedStrategy(), torch.distributed.group.WORLD
    )
    request = save_wrapper.async_save(state_dict, ckpt_dir, async_strategy='mcore')
    assert isinstance(request, AsyncRequest)
    request.execute_sync()
    assert (ckpt_dir / f'__{rank}_0.distcp').is_file()

    load_wrapper = FullyParallelLoadStrategyWrapper(
        TorchDistLoadShardedStrategy(), torch.distributed.group.WORLD
    )
    loaded = load_wrapper.load(_make_sharded_state_dict(), ckpt_dir)

    assert torch.all(loaded['sd_repl'] == 1)
    assert torch.all(loaded['sd_chunk'] == rank + 1)


def test_fully_parallel_save_reuses_cached_distribution(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    world_size = torch.distributed.get_world_size()
    save_wrapper = FullyParallelSaveStrategyWrapper(
        TorchDistSaveShardedStrategy(), torch.distributed.group.WORLD, do_cache_distribution=True
    )
    first_dir = _make_ckpt_dir(shared_ckpt_root, 'fully_parallel_cached_1')
    second_dir = _make_ckpt_dir(shared_ckpt_root, 'fully_parallel_cached_2')

    save_wrapper.save(_make_sharded_state_dict(), first_dir)
    # key_repl is a single replicated shard, key_chunk adds one shard per rank.
    assert len(save_wrapper.cached_distribution.main_rank_for_shard) == world_size + 1
    save_wrapper.save(_make_sharded_state_dict(), second_dir)

    load_wrapper = FullyParallelLoadStrategyWrapper(
        TorchDistLoadShardedStrategy(), torch.distributed.group.WORLD
    )
    first_load = load_wrapper.load(_make_sharded_state_dict(), first_dir)
    second_load = load_wrapper.load(_make_sharded_state_dict(), second_dir)
    assert torch.all(first_load['sd_repl'] == second_load['sd_repl'])
    assert torch.equal(first_load['sd_chunk'], second_load['sd_chunk'])


def test_fully_parallel_load_reuses_cached_distribution(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    save_wrapper = FullyParallelSaveStrategyWrapper(
        TorchDistSaveShardedStrategy(), torch.distributed.group.WORLD
    )
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'fully_parallel_load_cache')
    save_wrapper.save(_make_sharded_state_dict(), ckpt_dir)

    load_wrapper = FullyParallelLoadStrategyWrapper(
        TorchDistLoadShardedStrategy(), torch.distributed.group.WORLD, do_cache_distribution=True
    )
    first_load = load_wrapper.load(_make_sharded_state_dict(), ckpt_dir)
    # key_repl is a single replicated shard, key_chunk adds one shard per rank.
    assert len(load_wrapper.cached_distribution.main_rank_for_shard) == world_size + 1
    second_load = load_wrapper.load(_make_sharded_state_dict(), ckpt_dir)

    assert torch.all(first_load['sd_repl'] == 1) and torch.all(second_load['sd_repl'] == 1)
    assert torch.all(first_load['sd_chunk'] == rank + 1)
    assert torch.equal(first_load['sd_chunk'], second_load['sd_chunk'])


def test_fully_parallel_load_tensors_metadata_delegates_to_base(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    world_size = torch.distributed.get_world_size()
    save_wrapper = FullyParallelSaveStrategyWrapper(
        TorchDistSaveShardedStrategy(), torch.distributed.group.WORLD
    )
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'fully_parallel_metadata')
    save_wrapper.save(_make_sharded_state_dict(), ckpt_dir)

    load_wrapper = FullyParallelLoadStrategyWrapper(
        TorchDistLoadShardedStrategy(), torch.distributed.group.WORLD
    )
    metadata = load_wrapper.load_tensors_metadata(ckpt_dir)

    assert set(metadata.keys()) == {'key_repl', 'key_chunk'}
    chunk = metadata['key_chunk']
    assert chunk.data is None
    assert chunk.global_shape == (3 * world_size,)


def test_distribute_main_replicas_requires_distribution_for_nontrivial_group(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    with pytest.raises(ValueError, match='precomputed_distribution must be not None'):
        distribute_main_replicas_with_precomputed_distribution(
            _make_sharded_state_dict(), None, None
        )


def test_fill_in_deferred_sharded_items_raises_on_missing_shard():
    sh_ten = ShardedTensor.from_rank_offsets('missing_key', torch.ones(2), replica_id=0)

    with pytest.raises(CheckpointingException, match='Missing loaded item shard'):
        _fill_in_deferred_sharded_items({'sd': sh_ten}, {}, ShardedTensor, _sharded_tensor_shard_id)
