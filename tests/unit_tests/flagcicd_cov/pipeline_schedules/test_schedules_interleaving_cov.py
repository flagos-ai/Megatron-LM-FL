# Copyright (c) 2026, BAAI. All rights reserved.
"""Interleaved (virtual pipeline) 1F1B schedule runs against a staged reference model."""

from contextlib import contextmanager

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 8
SEQ_LENGTH = 4
MICRO_BATCH_SIZE = 2


class _RecordingTimers:
    def __init__(self):
        self.started = {}
        self.stopped = {}

    def __call__(self, name, log_level=None):
        timers = self

        class _Timer:
            def start(self, barrier=False):
                timers.started[name] = timers.started.get(name, 0) + 1

            def stop(self):
                timers.stopped[name] = timers.stopped.get(name, 0) + 1

        return _Timer()


class _StageModule(torch.nn.Module):
    def __init__(self, config, linear):
        super().__init__()
        self.linear = linear
        self.config = config
        self.model_type = 'unit-test'
        self._input_tensor = None

    def set_input_tensor(self, tensor):
        self._input_tensor = tensor[0] if isinstance(tensor, list) else tensor

    def forward(self, data=None):
        if self._input_tensor is not None:
            data, self._input_tensor = self._input_tensor, None
        return self.linear(data)


def _config(pp_size, vp_size, **overrides):
    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        pipeline_dtype=torch.float32,
    )
    config.hidden_size = HIDDEN_SIZE
    config.calculate_per_token_loss = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_linear(seed, in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE):
    generator = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features, generator=generator))
        linear.bias.copy_(torch.randn(out_features, generator=generator))
    return linear


def _make_input(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE, generator=generator).cuda()


def _forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
    def loss_func(output_tensor):
        loss = output_tensor.float().sum()
        return loss, {'lm loss': loss.detach().clone()}

    output = model(next(data_iterator))
    return output, loss_func


def _build_chunks(config, pp_size, num_chunks, pp_rank, seed):
    chunks = []
    for chunk_id in range(num_chunks):
        linear = _make_linear(seed + chunk_id * pp_size + pp_rank).cuda()
        chunk = _StageModule(config, linear)
        chunk.vp_stage = chunk_id
        chunks.append(chunk)
    return chunks


def _reference(num_stages, seed, inputs, num_microbatches):
    stages = []
    for index in range(num_stages):
        stage = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).cuda()
        donor = _make_linear(seed + index)
        with torch.no_grad():
            stage.weight.copy_(donor.weight)
            stage.bias.copy_(donor.bias)
        stages.append(stage)
    losses = []
    total = torch.zeros((), device='cuda')
    for data in inputs:
        hidden = data
        for stage in stages:
            hidden = stage(hidden)
        losses.append(hidden.sum())
        total = total + hidden.sum()
    (total / num_microbatches).backward()
    return losses, total, stages


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def vp2_world(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    return {
        'first': is_pp_first_stage(pp_group),
        'last': is_pp_last_stage(pp_group),
        'rank': pp_group.rank(),
    }


def test_interleaved_training_reduces_loss_and_grads_per_chunk(vp2_world):
    num_microbatches = 4
    timers = _RecordingTimers()

    @contextmanager
    def no_sync_one():
        yield

    @contextmanager
    def no_sync_two():
        yield

    finalize_calls = []
    config = _config(
        2,
        2,
        timers=timers,
        no_sync_func=[no_sync_one, no_sync_two],
        num_microbatches_with_partial_activation_checkpoints=1,
        fine_grained_activation_offloading=True,
        cuda_graph_impl='local',
        finalize_model_grads_func=lambda *args, **kwargs: finalize_calls.append(args),
    )
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 31)
    inputs = [_make_input(700 + index) for index in range(num_microbatches)]

    forward_backward_func = schedule.get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_pipelining_with_interleaving
    losses = forward_backward_func(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=num_microbatches,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    ref_losses, _, stages = _reference(4, 31, inputs, num_microbatches)
    if vp2_world['last']:
        assert len(losses) == num_microbatches
        for entry, ref_loss in zip(losses, ref_losses):
            assert entry['lm loss'].item() == pytest.approx(ref_loss.item(), rel=1e-5, abs=1e-5)
    else:
        assert losses == []
    for chunk_id, chunk in enumerate(chunks):
        position = chunk_id * 2 + vp2_world['rank']
        torch.testing.assert_close(
            chunk.linear.weight.grad, stages[position].weight.grad, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            chunk.linear.bias.grad, stages[position].bias.grad, rtol=1e-5, atol=1e-5
        )
    assert timers.started['forward-compute'] == num_microbatches * 2
    assert timers.started['backward-compute'] == num_microbatches * 2
    assert timers.started['forward-backward'] == 1
    assert timers.stopped['forward-backward'] == 1
    assert len(finalize_calls) == 1
    assert finalize_calls[0][0] == chunks


def test_interleaved_forward_only_returns_losses_without_grads(vp2_world):
    config = _config(2, 2, moe_paged_stash=True, moe_paged_stash_page_size=128)
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 32)
    inputs = [_make_input(710 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=True,
    )

    ref_losses, _, _ = _reference(4, 32, inputs, 4)
    if vp2_world['last']:
        assert len(losses) == 4
        for entry, ref_loss in zip(losses, ref_losses):
            assert entry['lm loss'].item() == pytest.approx(ref_loss.item(), rel=1e-5, abs=1e-5)
    else:
        assert losses == []
    for chunk in chunks:
        assert chunk.linear.weight.grad is None


def test_interleaved_overlap_p2p_comm_matches_sequential_grads(vp2_world):
    sync_calls = []
    config = _config(
        2,
        2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        grad_sync_func=lambda parameters: sync_calls.append(parameters),
    )
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 33)
    inputs = [_make_input(720 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    _, _, stages = _reference(4, 33, inputs, 4)
    if vp2_world['last']:
        assert len(losses) == 4
    for chunk_id, chunk in enumerate(chunks):
        position = chunk_id * 2 + vp2_world['rank']
        torch.testing.assert_close(
            chunk.linear.weight.grad, stages[position].weight.grad, rtol=1e-5, atol=1e-5
        )
    assert len(sync_calls) == 2


def test_interleaved_warmup_flush_overlaps_comm_with_compute(vp2_world):
    config = _config(
        2,
        2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        overlap_p2p_comm_warmup_flush=True,
        deallocate_pipeline_outputs=True,
    )
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 41)
    inputs = [_make_input(760 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    _, _, stages = _reference(4, 41, inputs, 4)
    if vp2_world['last']:
        assert len(losses) == 4
    for chunk_id, chunk in enumerate(chunks):
        position = chunk_id * 2 + vp2_world['rank']
        torch.testing.assert_close(
            chunk.linear.weight.grad, stages[position].weight.grad, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            chunk.linear.bias.grad, stages[position].bias.grad, rtol=1e-5, atol=1e-5
        )
