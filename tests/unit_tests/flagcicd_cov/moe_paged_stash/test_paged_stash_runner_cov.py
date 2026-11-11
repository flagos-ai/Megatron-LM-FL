# Copyright (c) 2026, BAAI. All rights reserved.
"""PagedStashRunner: config tracking, microbatch prefetch and cross-rank overflow flags."""

import types

import pytest
import torch

from megatron.core.transformer.moe.paged_stash import (
    PagedStashManager,
    PagedStashRunner,
    get_paged_stash_context,
    paged_stash_group_commit,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def _fresh_stash_manager(monkeypatch):
    """Give every test a pristine singleton and undo the swap afterwards."""
    monkeypatch.setattr(PagedStashManager, 'STASH_MGR', None, raising=False)
    yield


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _runner_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_paged_stash=True,
        moe_expert_rank_capacity_factor=1.5,
        use_transformer_engine_op_fuser=True,
        moe_token_dispatcher_type='flex',
        moe_flex_dispatcher_backend='hybridep',
        moe_paged_stash_page_size=16,
        moe_paged_stash_buffer_size_factor_cuda=2.0,
        moe_paged_stash_buffer_size_factor_cpu=0.0,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _StubCommManager:
    """HybridEP comm-manager surface read by the runner."""

    def __init__(self, capacity_factor):
        self.moe_expert_rank_capacity_factor = capacity_factor


class _StubDispatcher:
    """Token-dispatcher surface read by the runner, with HybridEP-style budget flags."""

    def __init__(self, config, budget_flag=False):
        self.config = config
        self._comm_manager = _StubCommManager(config.moe_expert_rank_capacity_factor)
        self._budget_flag = budget_flag

    def check_over_budget(self):
        if self._budget_flag is None:
            return None
        return torch.tensor(self._budget_flag, device='cuda')

    def reset_over_budget(self):
        self._budget_flag = False


class _StubMLP(torch.nn.Module):
    def __init__(self, dispatcher):
        super().__init__()
        self.token_dispatcher = dispatcher


class _StubLayer(torch.nn.Module):
    def __init__(self, mlp=None):
        super().__init__()
        if mlp is not None:
            self.mlp = mlp


class _StubModelChunk(torch.nn.Module):
    """GPT-model-chunk surface walked by PagedStashRunner: config, decoder layers, MTP."""

    def __init__(self, config, layers=(), mtp_layers=(), mtp_process=False):
        super().__init__()
        self.config = config
        self.decoder = types.SimpleNamespace(layers=list(layers))
        self.mtp = types.SimpleNamespace(layers=list(mtp_layers))
        self.mtp_process = mtp_process
        self.zero_grad_buffer_calls = 0

    def zero_grad_buffer(self):
        self.zero_grad_buffer_calls += 1


def _moe_chunk(config, dispatcher=None):
    if dispatcher is None:
        dispatcher = _StubDispatcher(config)
    return _StubModelChunk(config, layers=[_StubLayer(_StubMLP(dispatcher))])


def _marked_activation(device):
    activation = torch.randn(128, 64, device=device)
    activation.grouped_tensor_scale_inv = False
    return activation


def _stash_manager_after_one_stash(device, *, cuda_factor, cpu_factor):
    """Captured manager that stashed one real activation with the given tier factors."""
    config = _runner_config(
        moe_paged_stash_buffer_size_factor_cuda=cuda_factor,
        moe_paged_stash_buffer_size_factor_cpu=cpu_factor,
    )
    paged_stash_group_commit(torch.zeros(1, device=device))
    manager = PagedStashManager.get_instance()
    paged_stash_reset(True, config=config)
    paged_stash_init_chunk_handler(1, 0)
    num_tokens_tensor = torch.tensor([100], device=device)
    with get_paged_stash_context(
        name='moe', max_num_tokens=128, num_tokens_tensor=num_tokens_tensor, avg_num_tokens=100
    ):
        manager.on_save_for_backward(_marked_activation(device))
    paged_stash_reset(True, config=config)
    manager.on_save_for_backward(_marked_activation(device))
    manager.stash_paged_tensors(1)
    manager.wait_for_stash_to_complete()
    return manager


def test_runner_tracks_train_and_decoder_configs_deduplicating_shared_objects():
    config = _runner_config()
    decoder_config = _runner_config()
    chunk_a = _StubModelChunk(decoder_config)
    chunk_b = _StubModelChunk(decoder_config)
    runner = PagedStashRunner(config, False, [chunk_a, chunk_b], None, None)
    assert runner._configs_to_sync_moe_paged_stash == [config, decoder_config]
    assert runner.moe_layers == []


def test_runner_collects_layers_whose_dispatchers_check_over_budget():
    config = _runner_config()
    moe_mlp = _StubMLP(_StubDispatcher(config))
    dense_mlp = _StubMLP(types.SimpleNamespace())
    chunk = _StubModelChunk(
        config, layers=[_StubLayer(moe_mlp), _StubLayer(dense_mlp), _StubLayer()]
    )
    runner = PagedStashRunner(config, False, [chunk], None, None)
    assert runner.moe_layers == [moe_mlp]


def test_runner_collects_moe_layers_from_mtp_layers_when_processed():
    config = _runner_config()
    mtp_mlp = _StubMLP(_StubDispatcher(config))
    chunk = _StubModelChunk(config, mtp_layers=[_StubLayer(mtp_mlp)], mtp_process=True)
    runner = PagedStashRunner(config, False, [chunk], None, None)
    assert runner.moe_layers == [mtp_mlp]


def test_data_read_buffers_microbatches_for_a_single_model_chunk():
    chunk = _StubModelChunk(_runner_config())
    runner = PagedStashRunner(None, False, [chunk], None, None)
    batches = [{'batch': 0}, {'batch': 1}, {'batch': 2}]

    saved, data_list = runner.data_read(iter(batches), [chunk], True, 2)
    assert len(saved) == 1
    assert len(data_list) == 1
    assert next(data_list[0]) == {'batch': 0}
    assert next(data_list[0]) == {'batch': 1}
    assert next(saved[0]) == {'batch': 0}


def test_data_read_returns_a_none_entry_when_the_iterator_is_missing():
    chunk = _StubModelChunk(_runner_config())
    runner = PagedStashRunner(None, False, [chunk], None, None)

    saved, data_list = runner.data_read(None, [chunk], True, 1)
    assert saved == [None]
    assert data_list == [None]


def test_data_read_buffers_each_model_chunk_iterator_separately():
    config = _runner_config()
    chunks = [_StubModelChunk(config), _StubModelChunk(config)]
    runner = PagedStashRunner(None, False, chunks, None, None)
    first_batches = [{'batch': 10}, {'batch': 11}]
    second_batches = [{'batch': 20}, {'batch': 21}]

    saved, data_list = runner.data_read(
        [iter(first_batches), iter(second_batches)], chunks, True, 2
    )
    assert len(saved) == 2
    assert next(data_list[0]) == {'batch': 10}
    assert next(data_list[0]) == {'batch': 11}
    assert next(data_list[1]) == {'batch': 20}
    assert next(saved[1]) == {'batch': 20}


def test_data_read_returns_none_entries_per_missing_chunk_iterator():
    config = _runner_config()
    chunks = [_StubModelChunk(config), _StubModelChunk(config)]
    runner = PagedStashRunner(None, False, chunks, None, None)

    saved, data_list = runner.data_read([iter([{'batch': 0}]), None], chunks, True, 1)
    assert next(data_list[0]) == {'batch': 0}
    assert data_list[1] is None
    assert saved[1] is None


def test_data_read_rejects_an_iterator_count_that_mismatches_the_chunks():
    config = _runner_config()
    chunks = [_StubModelChunk(config), _StubModelChunk(config)]
    runner = PagedStashRunner(None, False, chunks, None, None)
    with pytest.raises(AssertionError):
        runner.data_read([iter([{'batch': 0}])], chunks, True, 1)


def test_data_read_rejects_multiple_iterators_for_a_single_chunk():
    chunk = _StubModelChunk(_runner_config())
    runner = PagedStashRunner(None, False, [chunk], None, None)
    with pytest.raises(AssertionError):
        runner.data_read([iter([{'batch': 0}]), iter([{'batch': 1}])], [chunk], True, 1)


def test_check_moe_overflow_all_reduces_over_budget_flags_across_ranks(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _runner_config()
    chunk = _moe_chunk(config, _StubDispatcher(config, budget_flag=True))
    runner = PagedStashRunner(config, False, [chunk], None, None)

    world_size = torch.distributed.get_world_size()
    assert runner.check_moe_overflow() == (0, world_size, 0)


def test_check_moe_overflow_ignores_layers_without_a_budget_flag(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _runner_config()
    chunk = _moe_chunk(config, _StubDispatcher(config, budget_flag=None))
    runner = PagedStashRunner(config, False, [chunk], None, None)

    assert runner.check_moe_overflow() == (0, 0, 0)


def test_check_moe_overflow_reports_stash_overflow_across_ranks(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    _stash_manager_after_one_stash('cuda', cuda_factor=0.001, cpu_factor=0.0)
    config = _runner_config()
    runner = PagedStashRunner(config, False, [_moe_chunk(config)], None, None)

    world_size = torch.distributed.get_world_size()
    assert runner.check_moe_overflow() == (world_size, 0, 0)


def test_check_moe_overflow_reports_host_spill_across_ranks(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    _stash_manager_after_one_stash('cuda', cuda_factor=0.001, cpu_factor=2.0)
    config = _runner_config()
    runner = PagedStashRunner(config, False, [_moe_chunk(config)], None, None)

    world_size = torch.distributed.get_world_size()
    assert runner.check_moe_overflow() == (0, 0, world_size)
