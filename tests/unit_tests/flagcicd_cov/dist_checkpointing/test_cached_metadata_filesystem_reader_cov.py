# Copyright (c) 2026, BAAI. All rights reserved.

"""CachedMetadataFileSystemReader: one metadata object per checkpoint path, shared across reader instances, cleared on demand, and bypassed when caching is disabled."""

import shutil

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata

from megatron.core.dist_checkpointing import ShardedTensor, save
from megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
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


@pytest.fixture
def saved_ckpt_dir(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = shared_ckpt_root / 'cached_reader_ckpt'
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    save(
        {
            'cached_key': ShardedTensor.from_rank_offsets(
                'cached_key', torch.ones(2) * rank, (0, rank, world_size)
            )
        },
        ckpt_dir,
    )
    torch.distributed.barrier()
    yield ckpt_dir
    CachedMetadataFileSystemReader.clear_metadata_cache()


def test_read_metadata_returns_cached_object_on_second_reader_of_same_checkpoint(saved_ckpt_dir):
    first = CachedMetadataFileSystemReader(saved_ckpt_dir)
    metadata_first = first.read_metadata()
    second = CachedMetadataFileSystemReader(saved_ckpt_dir)
    metadata_second = second.read_metadata()

    assert isinstance(metadata_first, Metadata)
    assert metadata_first is metadata_second
    assert set(metadata_first.state_dict_metadata.keys()) == {'cached_key'}


def test_read_metadata_rereads_after_cache_cleared(saved_ckpt_dir):
    first = CachedMetadataFileSystemReader(saved_ckpt_dir)
    metadata_first = first.read_metadata()
    CachedMetadataFileSystemReader.clear_metadata_cache()

    second = CachedMetadataFileSystemReader(saved_ckpt_dir)
    assert second.read_metadata() is not metadata_first
    assert FileSystemReader(saved_ckpt_dir).read_metadata() is not metadata_first


def test_read_metadata_bypasses_cache_when_cache_disabled(saved_ckpt_dir):
    first = CachedMetadataFileSystemReader(saved_ckpt_dir)
    metadata_first = first.read_metadata()

    no_cache = CachedMetadataFileSystemReader(saved_ckpt_dir, cache_metadata=False)
    fresh = no_cache.read_metadata()

    assert fresh is not metadata_first
    assert isinstance(fresh, Metadata)
    assert set(fresh.state_dict_metadata.keys()) == {'cached_key'}


def test_read_metadata_returns_cached_object_when_checkpoint_passed_as_string(saved_ckpt_dir):
    first = CachedMetadataFileSystemReader(saved_ckpt_dir)
    metadata_first = first.read_metadata()

    same_path = CachedMetadataFileSystemReader(str(saved_ckpt_dir))
    assert same_path.read_metadata() is metadata_first
