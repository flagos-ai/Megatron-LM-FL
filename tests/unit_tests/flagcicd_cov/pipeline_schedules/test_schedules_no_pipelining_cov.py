# Copyright (c) 2026, BAAI. All rights reserved.
"""No-pipelining schedule: loss reduction, per-iteration resets, and epilogue hooks."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.transformer.moe.paged_stash import PagedStashManager
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 8
SEQ_LENGTH = 4
MICRO_BATCH_SIZE = 2


class _RecordingTimers:
    def __init__(self):
        self.started = Counter()
        self.stopped = Counter()

    def __call__(self, name, log_level=None):
        timers = self

        class _Timer:
            def start(self, barrier=False):
                timers.started[name] += 1

            def stop(self):
                timers.stopped[name] += 1

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


class _OffloadStageModule(_StageModule):
    def forward(self, data=None):
        off_interface.init_chunk_handler(
            vp_size=None, vp_stage=None, min_offloaded_tensor_size=1024
        )
        return super().forward(data)


class _FSDPStageModule(_StageModule):
    def __init__(self, config, linear):
        super().__init__(config, linear)
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=True, overlap_param_gather=True)
        self.synchronize_param_gather_calls = 0

    def synchronize_param_gather(self):
        self.synchronize_param_gather_calls += 1


def _config(**overrides):
    config = ModelParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, pipeline_dtype=torch.float32
    )
    config.hidden_size = HIDDEN_SIZE
    config.calculate_per_token_loss = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_linear(seed):
    generator = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator))
        linear.bias.copy_(torch.randn(HIDDEN_SIZE, generator=generator))
    return linear


def _make_input(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE, generator=generator).cuda()


def _forward_step(data_iterator, model):
    def loss_func(output_tensor):
        loss = output_tensor.float().sum()
        return loss, {'lm loss': loss.detach().clone()}

    return model(next(data_iterator)), loss_func


def _reference_losses(linear, inputs):
    with torch.no_grad():
        return sum(linear(data).sum().item() for data in inputs)


def _run(model, inputs, num_microbatches, forward_only):
    return schedule.forward_backward_no_pipelining(
        forward_step_func=_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=forward_only,
    )


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_no_pipelining_training_reduces_loss_and_grads_with_autocast(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    num_microbatches = 2
    timers = _RecordingTimers()
    config = _config(
        timers=timers, enable_autocast=True, autocast_dtype=torch.float32, cuda_graph_impl='local'
    )
    linear = _make_linear(61).cuda()
    model = _StageModule(config, linear)
    inputs = [_make_input(670 + index) for index in range(num_microbatches)]

    losses = _run(model, inputs, num_microbatches, forward_only=False)

    assert len(losses) == num_microbatches
    assert sum(entry['lm loss'].item() for entry in losses) == pytest.approx(
        _reference_losses(linear, inputs), rel=1e-5, abs=1e-5
    )
    reference = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).cuda()
    with torch.no_grad():
        reference.weight.copy_(linear.weight)
        reference.bias.copy_(linear.bias)
    (sum(reference(data).sum() for data in inputs) / num_microbatches).backward()
    torch.testing.assert_close(linear.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(linear.bias.grad, reference.bias.grad, rtol=1e-5, atol=1e-5)
    assert timers.started['forward-compute'] == num_microbatches
    assert timers.stopped['forward-compute'] == num_microbatches
    assert timers.started['backward-compute'] == num_microbatches
    assert timers.started['forward-backward'] == 1
    assert timers.stopped['forward-backward'] == 1


def test_no_pipelining_forward_only_disables_paged_stash_before_step(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _config(moe_paged_stash=True, moe_paged_stash_page_size=128)
    linear = _make_linear(62).cuda()
    model = _StageModule(config, linear)
    inputs = [_make_input(680), _make_input(681)]
    stash = PagedStashManager.get_instance()
    iteration_before = stash.iteration

    losses = _run(model, inputs, 2, forward_only=True)

    assert len(losses) == 2
    assert sum(entry['lm loss'].item() for entry in losses) == pytest.approx(
        _reference_losses(linear, inputs), rel=1e-5, abs=1e-5
    )
    assert stash.enabled is False
    assert stash.iteration == iteration_before + 1
    assert stash.current_schedule_index == 0
    assert stash.page_size == 128
    assert linear.weight.grad is None


def test_no_pipelining_resets_offload_manager_after_step(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    PipelineOffloadManager.reset_instance()
    config = _config(fine_grained_activation_offloading=True)
    linear = _make_linear(63).cuda()
    model = _OffloadStageModule(config, linear)
    inputs = [_make_input(690), _make_input(691)]
    captured_chunks = []

    def recording_forward_step(data_iterator, model):
        output, loss_func = _forward_step(data_iterator, model)
        captured_chunks.append(PipelineOffloadManager.get_instance().cur_forward_chunk())
        return output, loss_func

    losses = schedule.forward_backward_no_pipelining(
        forward_step_func=recording_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=2,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    assert len(captured_chunks) == 2
    assert captured_chunks[0] is not captured_chunks[1]
    assert len(losses) == 2
    assert sum(entry['lm loss'].item() for entry in losses) == pytest.approx(
        _reference_losses(linear, inputs), rel=1e-5, abs=1e-5
    )
    manager = PipelineOffloadManager.get_instance()
    assert captured_chunks[0] in manager._cached_chunks_forward
    assert captured_chunks[1] in manager._cached_chunks_forward
    assert manager._cur_forward_chunk is None
    assert manager._inside_context is False


def test_no_pipelining_forward_only_synchronizes_param_gather_on_fsdp_modules(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config = _config()
    linear = _make_linear(64).cuda()
    model = _FSDPStageModule(config, linear)
    inputs = [_make_input(700), _make_input(701)]

    losses = _run(model, inputs, 2, forward_only=True)

    assert model.synchronize_param_gather_calls == 1
    assert len(losses) == 2
    assert sum(entry['lm loss'].item() for entry in losses) == pytest.approx(
        _reference_losses(linear, inputs), rel=1e-5, abs=1e-5
    )
