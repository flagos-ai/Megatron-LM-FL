# Copyright (c) 2026, BAAI. All rights reserved.

"""Async state-dict saver: SavePlan dataclass diffing, global-metadata reuse verification, planning and finalization with the default planner, and cached-metadata reuse across load and save."""

import logging
import shutil
from dataclasses import replace

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader, SavePlan
from torch.distributed.checkpoint.filesystem import _StoragePrefix
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.utils import _DistWrapper

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    _compare_dataclasses,
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
    verify_global_md_reuse,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
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


def _planner():
    """The planner configuration TorchDistSaveShardedStrategy passes to the async saver."""
    return MCoreSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False)


def _make_ckpt_dir(root, name):
    ckpt_dir = root / name
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    return ckpt_dir


def test_compare_dataclasses_reports_differing_fields():
    first = SavePlan(items=[1], storage_data=None, planner_data={})
    second = SavePlan(items=[2], storage_data=None, planner_data={})

    assert _compare_dataclasses(first, second) == ['items: [1] != [2]']


def test_compare_dataclasses_reports_equal_objects():
    plan = SavePlan(items=[1], storage_data=None, planner_data={'k': 'v'})

    assert _compare_dataclasses(plan, replace(plan)) == 'All fields are equal'


def test_compare_dataclasses_reports_different_types():
    plan = SavePlan(items=[], storage_data=None, planner_data={})

    assert _compare_dataclasses(plan, _StoragePrefix('__0_')) == (
        f'Objects are of different types: {type(plan)} and {type(_StoragePrefix("__0_"))}'
    )


def test_verify_global_md_reuse_accepts_matching_plans(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    dist_wrapper = _DistWrapper(None, True, 0)
    local_plan = SavePlan(items=[f'item_{rank}'], storage_data=None, planner_data={})
    loaded_all_plans = [
        SavePlan(items=[f'item_{other_rank}'], storage_data=None, planner_data={})
        for other_rank in range(world_size)
    ]

    reuse = verify_global_md_reuse(loaded_all_plans, local_plan, rank, dist_wrapper)

    assert reuse is True


def test_verify_global_md_reuse_logs_mismatching_plans(model_parallel, caplog):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    dist_wrapper = _DistWrapper(None, True, 0)
    local_plan = SavePlan(items=[f'other_{rank}'], storage_data=None, planner_data={})
    loaded_all_plans = [
        SavePlan(items=[f'item_{other_rank}'], storage_data=None, planner_data={})
        for other_rank in range(world_size)
    ]

    with caplog.at_level(logging.DEBUG):
        reuse = verify_global_md_reuse(loaded_all_plans, local_plan, rank, dist_wrapper)

    assert reuse is False
    assert 'local_verify_reuse is False: diffs -' in caplog.text


def test_verify_global_md_reuse_rejects_wrong_number_of_plans(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    world_size = torch.distributed.get_world_size()
    dist_wrapper = _DistWrapper(None, True, 0)
    local_plan = SavePlan(items=['item_0'], storage_data=None, planner_data={})
    loaded_all_plans = [
        SavePlan(items=['item_0'], storage_data=None, planner_data={})
        for _ in range(world_size + 1)
    ]

    reuse = verify_global_md_reuse(
        loaded_all_plans, local_plan, torch.distributed.get_rank(), dist_wrapper
    )

    assert reuse is False


def test_save_state_dict_async_plan_defaults_to_default_save_planner(
    model_parallel, shared_ckpt_root
):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'sds_default_planner')
    writer = FileSystemWriterAsync(ckpt_dir)
    state_dict = {f'default_planner_key_{rank}': torch.arange(4) + rank}

    save_ret = save_state_dict_async_plan(state_dict, writer, None, 0, planner=None)

    (planned_writer, global_metadata, dist_wrapper) = save_ret[0]
    assert len(planned_writer.write_buckets) == 1
    save_fn, preload_fn, save_args = planned_writer.get_save_function_and_args()
    AsyncRequest(save_fn, tuple(save_args), [], preload_fn=preload_fn).execute_sync()
    save_state_dict_async_finalize(planned_writer, global_metadata, dist_wrapper)

    metadata = FileSystemReader(ckpt_dir).read_metadata()
    assert set(metadata.state_dict_metadata.keys()) == {
        f'default_planner_key_{other_rank}' for other_rank in range(world_size)
    }
    assert metadata.state_dict_metadata[f'default_planner_key_{rank}'].size == torch.Size([4])
    if rank == 0:
        assert isinstance(global_metadata, Metadata)


def test_save_state_dict_async_plan_reuses_loaded_global_metadata(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    state_dict = {
        'sd_chunk': ShardedTensor.from_rank_offsets(
            'reuse_key', torch.ones(3) * (rank + 1), (0, rank, world_size)
        )
    }
    first_dir = _make_ckpt_dir(shared_ckpt_root, 'sds_reuse_first')
    second_dir = _make_ckpt_dir(shared_ckpt_root, 'sds_reuse_second')

    save(state_dict, first_dir)
    load_strategy = TorchDistLoadShardedStrategy(cache_metadata=True)
    loaded = load(state_dict, first_dir, load_strategy)
    assert torch.all(loaded['sd_chunk'] == rank + 1)

    save_strategy = TorchDistSaveShardedStrategy(cached_metadata=True)
    # Mirrors the load-to-save metadata wiring in megatron/training/checkpointing.py.
    save_strategy.cached_global_metadata = load_strategy.cached_global_metadata
    save(state_dict, second_dir, save_strategy)

    assert save_strategy.validated_loaded_metadata_reuse
    assert save_strategy.cached_global_metadata is load_strategy.cached_global_metadata
    reloaded = load(state_dict, second_dir)
    assert torch.all(reloaded['sd_chunk'] == rank + 1)
