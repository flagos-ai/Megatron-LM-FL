# Copyright (c) 2026, BAAI. All rights reserved.

"""Checkpointable wrappers for DCP: tensor operations rejected on checkpoint-only objects, their reprs, and shard lookup by MetadataIndex offset."""

import pytest
import torch
from torch.distributed.checkpoint.metadata import MetadataIndex

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies.checkpointable import (
    CheckpointableShardedTensor,
    LocalShardsContainer,
)


def _build_container():
    first = ShardedTensor.from_rank_offsets('lsc', torch.ones(2), (0, 0, 2))
    second = ShardedTensor.from_rank_offsets('lsc', torch.ones(2) * 2, (0, 1, 2))
    return LocalShardsContainer(
        [
            CheckpointableShardedTensor.from_sh_ten(first),
            CheckpointableShardedTensor.from_sh_ten(second),
        ]
    )


def test_checkpointable_sharded_tensor_rejects_tensor_operations():
    sh_ten = ShardedTensor.from_rank_offsets('cst', torch.ones(2, 3), replica_id=0)
    cst = CheckpointableShardedTensor.from_sh_ten(sh_ten)

    with pytest.raises(NotImplementedError, match='CheckpointableShardedTensor.__torch_dispatch__'):
        torch.add(cst, cst)


def test_local_shards_container_rejects_tensor_operations():
    container = _build_container()

    with pytest.raises(NotImplementedError, match='LocalShardsContainer.__torch_dispatch__'):
        torch.add(container, container)


def test_checkpointable_repr_contains_underlying_sharded_tensor():
    sh_ten = ShardedTensor.from_rank_offsets('cst', torch.ones(2, 3), replica_id=0)
    cst = CheckpointableShardedTensor.from_sh_ten(sh_ten)

    assert repr(cst).startswith('CheckpointableShardedTensor(')
    assert "key='cst'" in repr(cst)


def test_local_shards_container_repr_contains_local_shards():
    container = _build_container()

    assert repr(container).startswith('LocalShardsContainer(')
    assert "key='lsc'" in repr(container)


def test_get_tensor_shard_requires_an_offset():
    container = _build_container()

    with pytest.raises(ValueError, match='without an offset'):
        container.__get_tensor_shard__(MetadataIndex('lsc', None))


def test_get_tensor_shard_uses_index_hint_when_offset_matches():
    container = _build_container()

    shard = container.__get_tensor_shard__(MetadataIndex('lsc', torch.Size([2]), index=1))

    assert shard.tolist() == [2.0, 2.0]


def test_get_tensor_shard_falls_back_to_linear_search_on_wrong_hint():
    container = _build_container()

    shard = container.__get_tensor_shard__(MetadataIndex('lsc', torch.Size([2]), index=0))

    assert shard.tolist() == [2.0, 2.0]


def test_get_tensor_shard_raises_when_no_shard_matches_offset():
    container = _build_container()

    with pytest.raises(ValueError, match=r"Could not find shard at 'torch.Size\(\[9\]\)'"):
        container.__get_tensor_shard__(MetadataIndex('lsc', torch.Size([9])))
