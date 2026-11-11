# Copyright (c) 2026, BAAI. All rights reserved.
"""PagedStashRunner fallback: rerun preparation, capacity-factor restore, retry loop."""

import types

import pytest
import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
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


class _RecordingSchedule:
    """Records each forward/backward call the runner makes."""

    def __init__(self, result):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _RecordingOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1


class _RecordingChainedOptimizer(_RecordingOptimizer):
    def __init__(self, *chained):
        super().__init__()
        self.chained_optimizers = list(chained)


class _RecordingDistributedOptimizer(DistributedOptimizer):
    """Boundary stand-in: records param-buffer copies without optimizer state."""

    def __init__(self):
        self.copy_calls = 0
        self.shard_fp32_from_float16_groups = True

    def _copy_main_params_to_param_buffer(self):
        self.copy_calls += 1


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


def test_prepare_for_rerun_clears_capacity_budget_and_stash_flags(monkeypatch):
    config = _runner_config()
    dispatcher = _StubDispatcher(config, budget_flag=True)
    chunk = _moe_chunk(config, dispatcher)
    chained = _RecordingChainedOptimizer()
    runner = PagedStashRunner(
        config, False, [chunk], chained, FullCudaGraphWrapper(lambda **kwargs: None)
    )
    graph_state = {
        'result': {'training': 'stale', 'validation': 'kept'},
        'cuda_graph': {'training': None, 'validation': None},
        'curr_iteration': {'training': 2, 'validation': 1},
    }
    for name, state in graph_state.items():
        monkeypatch.setattr(FullCudaGraphWrapper, name, state)
    _stash_manager_after_one_stash('cuda', cuda_factor=0.001, cpu_factor=0.0)
    manager = PagedStashManager.get_instance()

    runner.prepare_for_rerun(is_training=True)
    assert dispatcher._comm_manager.moe_expert_rank_capacity_factor is None
    assert dispatcher.check_over_budget().item() == 0
    assert manager.overflow.item() == 0
    assert manager.host_spill.item() == 0
    assert config.moe_paged_stash is False
    assert chunk.zero_grad_buffer_calls == 1
    assert chained.zero_grad_calls == 1
    assert graph_state['result']['training'] is None
    assert graph_state['result']['validation'] == 'kept'
    assert graph_state['curr_iteration']['training'] == 0
    assert manager.stash_buffers is None


def test_prepare_for_rerun_keeps_page_buffers_for_validation():
    config = _runner_config()
    chunk = _moe_chunk(config)
    runner = PagedStashRunner(config, False, [chunk], _RecordingOptimizer(), None)
    _stash_manager_after_one_stash('cuda', cuda_factor=2.0, cpu_factor=0.0)
    manager = PagedStashManager.get_instance()

    runner.prepare_for_rerun(is_training=False)
    buffer = manager.stash_buffers[torch.float32][64]
    assert buffer.num_cuda_pages == 13
    assert chunk.zero_grad_buffer_calls == 1


def test_prepare_for_rerun_runs_without_an_optimizer_or_graph_schedule():
    config = _runner_config()
    chunk = _moe_chunk(config)
    runner = PagedStashRunner(config, False, [chunk], None, lambda **kwargs: None)

    runner.prepare_for_rerun(is_training=True)
    assert config.moe_paged_stash is False
    assert chunk.zero_grad_buffer_calls == 1
    assert PagedStashManager.get_instance().stash_buffers is None


def test_prepare_for_rerun_copies_main_params_for_chained_optimizers():
    config = _runner_config()
    chunk = _moe_chunk(config)
    distributed_sub = _RecordingDistributedOptimizer()
    plain_sub = _RecordingOptimizer()
    chained = _RecordingChainedOptimizer(distributed_sub, plain_sub)
    runner = PagedStashRunner(config, True, [chunk], chained, lambda **kwargs: None)

    runner.prepare_for_rerun(is_training=True)
    assert distributed_sub.copy_calls == 1
    assert chained.zero_grad_calls == 1


def test_prepare_for_rerun_copies_main_params_for_a_direct_distributed_optimizer():
    config = _runner_config()
    chunk = _moe_chunk(config)
    distributed = _RecordingDistributedOptimizer()
    runner = PagedStashRunner(config, True, [chunk], distributed, lambda **kwargs: None)

    runner.prepare_for_rerun(is_training=True)
    assert distributed.copy_calls == 1


def test_call_rejects_positional_arguments():
    runner = PagedStashRunner(_runner_config(), False, [], None, None)
    with pytest.raises(AssertionError, match='does not accept positional args'):
        runner(
            1, model=[], data_iterator=None, num_microbatches=1, seq_length=8, forward_only=False
        )


def test_call_requires_the_core_keyword_arguments():
    runner = PagedStashRunner(_runner_config(), False, [], None, None)
    with pytest.raises(AssertionError):
        runner(model=[], data_iterator=None, seq_length=8, forward_only=False)


def test_call_runs_the_schedule_once_when_no_flag_is_set(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _runner_config()
    chunk = _moe_chunk(config)
    result = torch.tensor([1234.0], device='cuda')
    schedule = _RecordingSchedule(result)
    runner = PagedStashRunner(config, False, [chunk], None, schedule)

    returned = runner(
        model=[chunk],
        data_iterator=iter([{'batch': 0}, {'batch': 1}]),
        num_microbatches=1,
        seq_length=8,
        forward_only=False,
    )
    assert len(schedule.calls) == 1
    assert torch.equal(returned, result)
    assert next(schedule.calls[0]['data_iterator'][0]) == {'batch': 0}
    assert config.moe_paged_stash is True


def test_call_reruns_once_after_a_stash_overflow(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _runner_config()
    dispatcher = _StubDispatcher(config, budget_flag=False)
    chunk = _moe_chunk(config, dispatcher)
    result = torch.tensor([7.0], device='cuda')
    schedule = _RecordingSchedule(result)
    runner = PagedStashRunner(config, False, [chunk], None, schedule)
    _stash_manager_after_one_stash('cuda', cuda_factor=0.001, cpu_factor=0.0)
    manager = PagedStashManager.get_instance()

    returned = runner(
        model=[chunk],
        data_iterator=iter([{'batch': 0}, {'batch': 1}]),
        num_microbatches=1,
        seq_length=8,
        forward_only=False,
    )
    assert len(schedule.calls) == 2
    assert next(schedule.calls[0]['data_iterator'][0]) == {'batch': 0}
    assert next(schedule.calls[1]['data_iterator'][0]) == {'batch': 0}
    assert torch.equal(returned, result)
    assert manager.overflow.item() == 0
    assert config.moe_paged_stash is True
    assert dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5


def test_call_reruns_once_after_a_token_dispatch_over_budget(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _runner_config()
    dispatcher = _StubDispatcher(config, budget_flag=True)
    chunk = _moe_chunk(config, dispatcher)
    result = torch.tensor([9.0], device='cuda')
    schedule = _RecordingSchedule(result)
    runner = PagedStashRunner(config, False, [chunk], None, schedule)

    returned = runner(
        model=[chunk],
        data_iterator=iter([{'batch': 0}, {'batch': 1}]),
        num_microbatches=1,
        seq_length=8,
        forward_only=False,
    )
    assert len(schedule.calls) == 2
    assert next(schedule.calls[0]['data_iterator'][0]) == {'batch': 0}
    assert next(schedule.calls[1]['data_iterator'][0]) == {'batch': 0}
    assert torch.equal(returned, result)
    assert dispatcher.check_over_budget().item() == 0
    assert dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert config.moe_paged_stash is True


def test_call_succeeds_without_rerun_when_only_a_host_spill_occurred(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _runner_config()
    chunk = _moe_chunk(config)
    result = torch.tensor([11.0], device='cuda')
    schedule = _RecordingSchedule(result)
    runner = PagedStashRunner(config, False, [chunk], None, schedule)
    _stash_manager_after_one_stash('cuda', cuda_factor=0.001, cpu_factor=2.0)
    manager = PagedStashManager.get_instance()

    returned = runner(
        model=[chunk],
        data_iterator=iter([{'batch': 0}, {'batch': 1}]),
        num_microbatches=1,
        seq_length=8,
        forward_only=False,
    )
    assert len(schedule.calls) == 1
    assert next(schedule.calls[0]['data_iterator'][0]) == {'batch': 0}
    assert torch.equal(returned, result)
    assert manager.host_spill.item() == 1
    assert manager.overflow.item() == 0
    assert config.moe_paged_stash is True
