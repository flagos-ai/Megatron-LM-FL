# Copyright (c) 2026, BAAI. All rights reserved.

"""Serialization entrypoints: argument checks for metadata loading, deprecated default-strategy getters, and async save with integrity verification."""

import logging
import shutil

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
    load_content_metadata,
)
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
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


def _make_ckpt_dir(root, name):
    ckpt_dir = root / name
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    return ckpt_dir


def test_load_content_metadata_requires_checkpoint_dir_or_state_dict():
    with pytest.raises(
        ValueError, match='Both checkpoint_dir and loaded_state_dict cannot be None'
    ):
        load_content_metadata()


def test_get_default_save_sharded_strategy_warns_and_returns_torch_dist(caplog):
    with caplog.at_level(logging.WARNING):
        strategy = get_default_save_sharded_strategy()

    assert isinstance(strategy, TorchDistSaveShardedStrategy)
    assert 'get_default_save_sharded_strategy is deprecated' in caplog.text


def test_get_default_load_sharded_strategy_warns_and_returns_torch_dist(caplog):
    with caplog.at_level(logging.WARNING):
        strategy = get_default_load_sharded_strategy('/unused/checkpoint/dir')

    assert isinstance(strategy, TorchDistLoadShardedStrategy)
    assert 'get_default_load_sharded_strategy is deprecated' in caplog.text


def test_async_save_with_integrity_verification_roundtrips(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'serialization_async_integrity')
    state_dict = {
        'sd_chunk': ShardedTensor.from_rank_offsets(
            'async_integrity_key', torch.ones(3) * (rank + 1), (0, rank, world_size)
        )
    }

    async_request = save(
        state_dict, ckpt_dir, async_sharded_save=True, async_strategy='mcore', verify_integrity=True
    )

    assert isinstance(async_request, AsyncRequest)
    assert len(async_request.finalize_fns) == 3
    async_request.execute_sync()

    assert (ckpt_dir / 'integrity.json').is_file()
    loaded = load(state_dict, ckpt_dir, verify_integrity=True)
    assert torch.all(loaded['sd_chunk'] == rank + 1)
