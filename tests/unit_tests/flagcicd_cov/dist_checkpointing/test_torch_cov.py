# Copyright (c) 2026, BAAI. All rights reserved.

"""torch_dist strategies: state-dict flattening rules, MCore-to-PyTorch conversion, cached metadata readers, separation-hint file layout, and async save defaults."""

import shutil

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
    _get_filesystem_reader,
    flatten_state_dict,
    get_async_strategy,
    mcore_to_pyt_state_dict,
    sharded_tensor_to_torch_sharded_tensor,
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


def _make_ckpt_dir(root, name):
    ckpt_dir = root / name
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    return ckpt_dir


def test_flatten_state_dict_rejects_duplicated_flatten_keys():
    sh_ten = ShardedTensor.from_rank_offsets('k', torch.ones(2), replica_id=0)
    state_dict = {'a': {'x': sh_ten}, 'a.x': sh_ten}

    with pytest.raises(ValueError, match='duplicated flatten key a.x'):
        flatten_state_dict(state_dict)


def test_sharded_tensor_to_torch_sharded_tensor_defaults_to_current_rank(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    sh_ten = ShardedTensor.from_rank_offsets('stt_key', torch.ones(2, 4).t(), replica_id=0)

    pyt_sh_ten = sharded_tensor_to_torch_sharded_tensor([sh_ten])

    assert pyt_sh_ten.size() == torch.Size([4, 2])
    local_shard = pyt_sh_ten.local_shards()[0]
    assert str(local_shard.metadata.placement).startswith(f'rank:{torch.distributed.get_rank()}/')
    assert local_shard.metadata.shard_offsets == [0, 0]
    assert local_shard.metadata.shard_sizes == [4, 2]
    assert local_shard.tensor.is_contiguous()
    assert torch.equal(local_shard.tensor, torch.ones(2, 4).t().contiguous())
    assert pyt_sh_ten.mcore_sh_ten.key == 'stt_key'


def test_mcore_to_pyt_state_dict_rejects_missing_data_during_save(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    sh_ten = ShardedTensor.from_rank_offsets(
        'no_data_key', torch.ones(4), replica_id=0
    ).without_data()

    with pytest.raises(CheckpointingException, match='`data` attr is None for '):
        mcore_to_pyt_state_dict({'no_data_key': [sh_ten]}, is_loading=False)


def test_get_filesystem_reader_returns_cached_metadata_reader(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'torch_cached_reader')
    state_dict = {
        'sd_chunk': ShardedTensor.from_rank_offsets(
            'reader_key', torch.ones(2), (0, rank, world_size)
        )
    }
    save(state_dict, ckpt_dir)

    reader = _get_filesystem_reader(ckpt_dir, cache_metadata=True, async_strategy='mcore')
    assert isinstance(reader, CachedMetadataFileSystemReader)
    metadata = reader.read_metadata()
    assert set(metadata.state_dict_metadata.keys()) == {'reader_key'}
    assert type(_get_filesystem_reader(ckpt_dir)) is FileSystemReader

    CachedMetadataFileSystemReader.clear_metadata_cache()


def test_get_async_strategy_rejects_unknown_strategy():
    with pytest.raises(TypeError, match='async_strategy bogus is not supported'):
        get_async_strategy('bogus')


def test_separation_hint_save_splits_prefix_files(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'torch_separation_hint')
    strategy = TorchDistSaveShardedStrategy('torch_dist', 1, separation_hint='prefix')
    state_dict = {
        'sd_plain': ShardedTensor.from_rank_offsets(
            'plain_key', torch.ones(2) * (rank + 1), (0, rank, world_size)
        ),
        'sd_hinted': ShardedTensor.from_rank_offsets(
            'prefix_hinted_key', torch.ones(3) * (rank + 2), (0, rank, world_size)
        ),
    }

    save(state_dict, ckpt_dir, strategy)

    prefix_files = [path.name for path in ckpt_dir.iterdir() if path.name.startswith('prefix')]
    assert len(prefix_files) == world_size
    loaded = load(state_dict, ckpt_dir)
    assert torch.all(loaded['sd_plain'] == rank + 1)
    assert torch.all(loaded['sd_hinted'] == rank + 2)


def test_async_save_with_default_nvrx_strategy_roundtrips(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    sync_dir = _make_ckpt_dir(shared_ckpt_root, 'torch_nvrx_sync')
    async_dir = _make_ckpt_dir(shared_ckpt_root, 'torch_nvrx_async')
    state_dict = {
        'sd_chunk': ShardedTensor.from_rank_offsets(
            'nvrx_key', torch.ones(3) * (rank + 1), (0, rank, world_size)
        )
    }

    save(state_dict, sync_dir)
    async_request = save(state_dict, async_dir, async_sharded_save=True)
    async_request.execute_sync()

    loaded_async = load(state_dict, async_dir)
    loaded_sync = load(state_dict, sync_dir)
    assert torch.equal(loaded_async['sd_chunk'], loaded_sync['sd_chunk'])
    assert torch.all(loaded_async['sd_chunk'] == rank + 1)
