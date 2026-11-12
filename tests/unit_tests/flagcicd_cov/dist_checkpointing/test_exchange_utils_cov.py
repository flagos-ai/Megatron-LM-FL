# Copyright (c) 2026, BAAI. All rights reserved.

"""Shard exchange between ranks: gather and broadcast of shards by ShardDistribution, duplicate-shard rejection, and empty-tensor allocation for shards this rank does not hold."""

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.exchange_utils import (
    ShardDistribution,
    _get_empty_tensor_for_exchange,
    _sharded_tensor_shard_id,
    exchange_by_distribution,
    exchange_loaded_objects_gather_object,
    exchange_loaded_tensors_gather_object,
    exchange_loaded_tensors_gather_rounds,
    is_float8tensor,
)
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils

cur_platform = get_platform()


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _shard_id_for_chunk(key, chunk_idx, world_size):
    sh_ten = ShardedTensor.from_rank_offsets(key, torch.ones(2), (0, chunk_idx, world_size))
    return _sharded_tensor_shard_id(sh_ten)


def _uniform_distribution(world_size):
    """Every rank needs every shard, chunk ``idx`` is loaded by rank ``idx``."""
    shard_to_metadata = {}
    main_rank_for_shard = {}
    all_ranks_for_shard = {}
    for idx in range(world_size):
        shard_id = _shard_id_for_chunk('keyA', idx, world_size)
        sh_ten = ShardedTensor.from_rank_offsets(
            'keyA', torch.ones(2), (0, idx, world_size)
        ).without_data()
        shard_to_metadata[shard_id] = sh_ten
        main_rank_for_shard[shard_id] = idx
        all_ranks_for_shard[shard_id] = list(range(world_size))
    return ShardDistribution(
        main_rank_for_shard, set(shard_to_metadata.keys()), shard_to_metadata, all_ranks_for_shard
    )


@pytest.mark.parametrize(
    'tensor',
    [
        pytest.param(torch.ones(2, 4), id='float32'),
        pytest.param(torch.ones(2, 4, dtype=torch.float8_e4m3fn), id='float8-dtype'),
    ],
)
def test_is_float8tensor_returns_false_for_plain_torch_tensors_of_any_dtype(tensor):
    assert is_float8tensor(tensor) is False


def test_get_empty_tensor_for_exchange_allocates_device_copy_for_unneeded_cpu_shard(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    sh_ten = ShardedTensor.from_rank_offsets('unneeded', torch.ones(3, 5), replica_id=0)
    shard_id = _sharded_tensor_shard_id(sh_ten)

    tensor, orig_device = _get_empty_tensor_for_exchange(shard_id, {}, {shard_id: sh_ten}, {})

    assert orig_device is None
    assert tensor.shape == (3, 5)
    assert tensor.device.type == cur_platform.device_name()
    assert sh_ten.data.device.type == 'cpu'


def test_exchange_by_distribution_gathers_all_ranks_shards(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    distribution = _uniform_distribution(world_size)
    loaded_tensors = {_shard_id_for_chunk('keyA', rank, world_size): torch.full((2,), float(rank))}

    result = exchange_by_distribution(
        loaded_tensors, {}, distribution, None, exchange_algo='gather_object'
    )

    assert set(result.keys()) == set(distribution.main_rank_for_shard.keys())
    for shard_id, main_rank in distribution.main_rank_for_shard.items():
        assert torch.all(result[shard_id] == main_rank)


def test_exchange_loaded_tensors_gather_object_rejects_duplicate_shards(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    world_size = torch.distributed.get_world_size()
    distribution = _uniform_distribution(world_size)
    loaded_tensors = {_shard_id_for_chunk('keyA', 0, world_size): torch.zeros(2)}

    with pytest.raises(CheckpointingException, match='Duplicate shard ids'):
        exchange_loaded_tensors_gather_object(loaded_tensors, {}, distribution)


def test_exchange_loaded_objects_gather_object_rejects_duplicate_shards(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    loaded_objects = {('obj', (0,), None): torch.distributed.get_rank()}

    with pytest.raises(CheckpointingException, match='Duplicate shard ids'):
        exchange_loaded_objects_gather_object(loaded_objects)


def test_exchange_loaded_tensors_gather_rounds_gathers_via_world_group(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    distribution = _uniform_distribution(world_size)

    own_shard_id = _shard_id_for_chunk('keyA', rank, world_size)
    loaded_tensors = {own_shard_id: torch.full((3,), float(rank))}
    unloaded_shards = {}
    for shard_id, main_rank in distribution.main_rank_for_shard.items():
        if shard_id != own_shard_id:
            sh_ten = ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(3), (0, main_rank, world_size)
            )
            unloaded_shards[shard_id] = sh_ten.without_data()

    result = exchange_loaded_tensors_gather_rounds(loaded_tensors, unloaded_shards, distribution)

    assert set(result.keys()) == set(distribution.shard_to_metadata.keys())
    for shard_id, main_rank in distribution.main_rank_for_shard.items():
        assert torch.all(result[shard_id].cpu() == main_rank)


def test_exchange_by_distribution_broadcasts_main_rank_shards(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    distribution = _uniform_distribution(world_size)

    own_shard_id = _shard_id_for_chunk('keyA', rank, world_size)
    loaded_tensors = {own_shard_id: torch.full((2,), float(rank))}
    unloaded_shards = {}
    for shard_id, main_rank in distribution.main_rank_for_shard.items():
        if shard_id != own_shard_id:
            sh_ten = ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2), (0, main_rank, world_size)
            )
            unloaded_shards[shard_id] = sh_ten.without_data()

    result = exchange_by_distribution(
        loaded_tensors, unloaded_shards, distribution, None, exchange_algo='broadcast'
    )

    assert set(result.keys()) == set(distribution.main_rank_for_shard.keys())
    for shard_id, main_rank in distribution.main_rank_for_shard.items():
        assert torch.all(result[shard_id].cpu() == main_rank)


def test_exchange_by_distribution_rejects_unknown_exchange_algo(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    distribution = _uniform_distribution(torch.distributed.get_world_size())

    with pytest.raises(NotImplementedError, match='Unrecognized gather algorithm: bogus'):
        exchange_by_distribution({}, {}, distribution, None, exchange_algo='bogus')
