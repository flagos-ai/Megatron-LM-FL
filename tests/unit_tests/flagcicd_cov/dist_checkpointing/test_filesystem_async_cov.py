# Copyright (c) 2026, BAAI. All rights reserved.

"""FileSystemWriterAsync: write-item bucketing and sizing, checkpoint-id validation against the plain writer, threaded writes, and error surfacing when write results or the checkpoint directory are missing."""

import io
import shutil

import pytest
import torch
from torch.distributed.checkpoint import (
    CheckpointException,
    FileSystemReader,
    FileSystemWriter,
    SavePlan,
)
from torch.distributed.checkpoint.planner_helpers import _create_write_items

import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
    _item_size,
    _process_memory,
    _split_by_separation_hint,
    _split_by_size_and_type,
)
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
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


def _tensor_write_item(fqn, tensor):
    items = _create_write_items(fqn, tensor)
    assert len(items) == 1
    return items[0]


def _make_ckpt_dir(root, name):
    ckpt_dir = root / name
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    return ckpt_dir


def test_split_by_size_and_type_returns_single_bin_unchanged():
    items = [_tensor_write_item('single', torch.ones(4))]

    assert _split_by_size_and_type(1, items) == [items]


def test_split_by_size_and_type_balances_tensors_across_bins():
    small = _tensor_write_item('small', torch.ones(4))
    large = _tensor_write_item('large', torch.ones(6))
    byte_item = _create_write_items('bytes', io.BytesIO(b'abc'))[0]

    buckets = _split_by_size_and_type(2, [small, large, byte_item])

    assert buckets[0] == [byte_item, large]
    assert buckets[1] == [small]


def test_split_by_separation_hint_groups_items_by_fqn_prefix():
    hinted = _tensor_write_item('prefix_hinted', torch.ones(2))
    plain = _tensor_write_item('plain', torch.ones(2))

    grouped = _split_by_separation_hint([[hinted, plain]], separation_hint='prefix')

    assert grouped == {'': [[plain]], 'prefix': [[hinted]]}


def test_split_by_separation_hint_without_hint_keeps_buckets():
    item = _tensor_write_item('plain', torch.ones(2))

    assert _split_by_separation_hint([[item]], None) == {'': [[item]]}


def test_item_size_multiplies_chunk_sizes_by_element_size():
    item = _tensor_write_item('sized', torch.ones(3, 4, dtype=torch.float32))

    assert _item_size(item) == 48


def test_process_memory_requires_psutil(monkeypatch):
    monkeypatch.setattr(filesystem_async, 'HAVE_PSUTIL', False)

    with pytest.raises(RuntimeError, match='psutil is not installed'):
        _process_memory()


def test_writer_rejects_single_file_per_rank_disabled(tmp_path):
    with pytest.raises(NotImplementedError, match='single_file_per_rank flag not supported'):
        FileSystemWriterAsync(tmp_path, single_file_per_rank=False)


def test_validate_checkpoint_id_accepts_msc_uri():
    assert FileSystemWriterAsync.validate_checkpoint_id('msc://bucket/path') is True
    assert FileSystemWriter.validate_checkpoint_id('msc://bucket/path') is False


def test_validate_checkpoint_id_delegates_to_plain_filesystem_writer(tmp_path):
    local_path = str(tmp_path)

    assert FileSystemWriterAsync.validate_checkpoint_id(local_path) is True
    assert FileSystemWriter.validate_checkpoint_id(local_path) is True
    assert FileSystemWriterAsync.validate_checkpoint_id('http://bucket/path') is False
    assert FileSystemWriter.validate_checkpoint_id('http://bucket/path') is False


def test_write_data_is_not_implemented_for_async_writer(tmp_path):
    writer = FileSystemWriterAsync(tmp_path)

    with pytest.raises(NotImplementedError, match='write_data not implemented'):
        writer.write_data(SavePlan(items=[], planner_data={}), None)


def test_retrieve_write_results_reports_empty_results_queue(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'fsa_empty_results_queue')
    writer = FileSystemWriterAsync(ckpt_dir)
    state_dict = {f'empty_queue_key_{rank}': torch.ones(4)}

    save_ret = save_state_dict_async_plan(state_dict, writer, None, 0, planner=_planner())

    (planned_writer, _, _) = save_ret[0]
    assert len(planned_writer.write_buckets) == 1
    retrieved = planned_writer.retrieve_write_results()

    assert isinstance(retrieved[0], RuntimeError)
    assert 'results_queue should not be empty' in str(retrieved[0])


def test_write_preloaded_data_multithread_runs_worker_threads(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'fsa_multithread')
    writer = FileSystemWriterAsync(ckpt_dir, thread_count=2)
    state_dict = {f'mt_key_a_{rank}': torch.ones(4), f'mt_key_b_{rank}': torch.ones(6)}

    save_ret = save_state_dict_async_plan(state_dict, writer, None, 0, planner=_planner())

    (planned_writer, global_metadata, dist_wrapper) = save_ret[0]
    assert len(planned_writer.write_buckets) == 2
    save_fn, preload_fn, save_args = planned_writer.get_save_function_and_args()
    AsyncRequest(save_fn, tuple(save_args), [], preload_fn=preload_fn).execute_sync()
    save_state_dict_async_finalize(planned_writer, global_metadata, dist_wrapper)

    distcp_files = sorted(path.name for path in ckpt_dir.glob(f'__{rank}_*.distcp'))
    assert len(distcp_files) == 2
    metadata = FileSystemReader(ckpt_dir).read_metadata()
    assert set(metadata.state_dict_metadata.keys()) == {
        f'mt_key_a_{r}' for r in range(world_size)
    } | {f'mt_key_b_{r}' for r in range(world_size)}


def test_async_save_raises_when_checkpoint_directory_is_removed(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ckpt_dir = _make_ckpt_dir(shared_ckpt_root, 'fsa_failed_write')
    strategy = TorchDistSaveShardedStrategy()
    state_dict = {
        'sd_key': ShardedTensor.from_rank_offsets(
            'failed_write_key', torch.ones(4) * (rank + 1), (0, rank, world_size)
        )
    }

    request = strategy.async_save(state_dict, ckpt_dir, async_strategy='mcore')
    torch.distributed.barrier()
    if rank == 0:
        shutil.rmtree(ckpt_dir)
    torch.distributed.barrier()

    with pytest.raises(CheckpointException, match='CheckpointException') as caught:
        request.execute_sync()

    failures = caught.value.failures
    assert set(failures.keys()) == set(range(world_size))
    for exc, _ in failures.values():
        assert isinstance(exc, RuntimeError)
        assert 'An error is caught while starting worker 0' in str(exc)
