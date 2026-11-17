# Copyright (c) 2026, BAAI. All rights reserved.
"""Per-iteration activation state: offload manager resets and deferred embedding wgrads."""

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.utils import is_pp_first_stage
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 8
SEQ_LENGTH = 4
MICRO_BATCH_SIZE = 2


class _StageModule(torch.nn.Module):
    def __init__(self, config, linear, vp_stage):
        super().__init__()
        self.linear = linear
        self.config = config
        self.model_type = 'unit-test'
        self.vp_stage = vp_stage
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


class _DeferredWgradStageModule(_StageModule):
    def __init__(self, config, linear, post_process, vp_stage):
        super().__init__(config, linear, vp_stage)
        self.post_process = post_process
        self.share_embeddings_and_output_weights = True
        self.output_layer = linear
        self.embedding_activation_buffer = []
        self.grad_output_buffer = []

    def forward(self, data=None):
        if self._input_tensor is not None:
            data, self._input_tensor = self._input_tensor, None
        output = self.linear(data)
        if self.post_process:
            self.embedding_activation_buffer.append(data)
            self.grad_output_buffer.append(torch.ones_like(data))
        return output


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


def _forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
    def loss_func(output_tensor):
        loss = output_tensor.float().sum()
        return loss, {'lm loss': loss.detach().clone()}

    output = model(next(data_iterator))
    return output, loss_func


def _reference_two_stage(linear, inputs, num_microbatches):
    stages = []
    for _ in range(2):
        stage = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).cuda()
        with torch.no_grad():
            stage.weight.copy_(linear.weight.detach())
            stage.bias.copy_(linear.bias.detach())
        stages.append(stage)
    total = torch.zeros((), device='cuda')
    for data in inputs:
        hidden = data
        for stage in stages:
            hidden = stage(hidden)
        total = total + hidden.sum()
    (total / num_microbatches).backward()
    return total, stages


def _reference_stages(num_stages, seed, inputs, num_microbatches):
    stages = []
    for index in range(num_stages):
        stage = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).cuda()
        donor = _make_linear(seed + index)
        with torch.no_grad():
            stage.weight.copy_(donor.weight)
            stage.bias.copy_(donor.bias)
        stages.append(stage)
    total = torch.zeros((), device='cuda')
    for data in inputs:
        hidden = data
        for stage in stages:
            hidden = stage(hidden)
        total = total + hidden.sum()
    (total / num_microbatches).backward()
    return stages


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def is_first_stage(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    return is_pp_first_stage(pp_group)


def test_without_interleaving_resets_offload_manager_after_step(is_first_stage):
    PipelineOffloadManager.reset_instance()
    config = _config(2, None, fine_grained_activation_offloading=True)
    linear = _make_linear(30).cuda()
    model = _OffloadStageModule(config, linear, None)
    inputs = [_make_input(655), _make_input(656)]
    captured_chunks = []

    def recording_forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
        output, loss_func = _forward_step(data_iterator, model, checkpoint_activations_microbatch)
        captured_chunks.append(PipelineOffloadManager.get_instance().cur_forward_chunk())
        return output, loss_func

    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=recording_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=2,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    _, stages = _reference_two_stage(linear, inputs, 2)
    assert len(captured_chunks) == 2
    assert captured_chunks[0] is not captured_chunks[1]
    if is_first_stage:
        assert losses == []
        torch.testing.assert_close(linear.weight.grad, stages[0].weight.grad, rtol=1e-5, atol=1e-5)
    else:
        assert len(losses) == 2
        torch.testing.assert_close(linear.weight.grad, stages[1].weight.grad, rtol=1e-5, atol=1e-5)
    manager = PipelineOffloadManager.get_instance()
    assert captured_chunks[0] in manager._cached_chunks_forward
    assert captured_chunks[1] in manager._cached_chunks_forward
    assert manager._cur_forward_chunk is None
    assert manager._inside_context is False


def test_without_interleaving_clears_and_drains_deferred_embedding_wgrad(is_first_stage):
    finalize_calls = []
    config = _config(
        2,
        None,
        defer_embedding_wgrad_compute=True,
        finalize_model_grads_func=lambda *args, **kwargs: finalize_calls.append(args),
    )
    linear = _make_linear(31).cuda()
    model = _DeferredWgradStageModule(
        config, linear, post_process=not is_first_stage, vp_stage=None
    )
    inputs = [_make_input(660), _make_input(661)]
    captured_lengths = []
    if model.post_process:
        leftover = [torch.ones(2, 2, HIDDEN_SIZE, device='cuda') for _ in range(2)]
        model.embedding_activation_buffer.extend(leftover)

    def recording_forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
        output, loss_func = _forward_step(data_iterator, model, checkpoint_activations_microbatch)
        captured_lengths.append(
            (len(model.embedding_activation_buffer), len(model.grad_output_buffer))
        )
        return output, loss_func

    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=recording_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=2,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    assert len(finalize_calls) == 1
    assert finalize_calls[0][0] == [model]
    _, stages = _reference_two_stage(linear, inputs, 2)
    if is_first_stage:
        assert captured_lengths == [(0, 0), (0, 0)]
        assert losses == []
        torch.testing.assert_close(linear.weight.grad, stages[0].weight.grad, rtol=1e-5, atol=1e-5)
    else:
        assert captured_lengths == [(1, 1), (2, 2)]
        assert len(losses) == 2
        torch.testing.assert_close(linear.weight.grad, stages[1].weight.grad, rtol=1e-5, atol=1e-5)
    assert model.embedding_activation_buffer == []
    assert model.grad_output_buffer == []


def test_interleaved_defers_embedding_wgrad_through_last_model_chunk(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = pp_group.rank()
    finalize_calls = []
    config = _config(
        2,
        2,
        defer_embedding_wgrad_compute=True,
        finalize_model_grads_func=lambda *args, **kwargs: finalize_calls.append(args),
    )
    chunks = [
        _StageModule(config, _make_linear(38 + pp_rank).cuda(), 0),
        _DeferredWgradStageModule(config, _make_linear(38 + 2 + pp_rank).cuda(), pp_rank == 1, 1),
    ]
    deferred = chunks[1]
    captured_lengths = []
    if deferred.post_process:
        leftover = [torch.ones(2, 2, HIDDEN_SIZE, device='cuda') for _ in range(2)]
        deferred.embedding_activation_buffer.extend(leftover)
    inputs = [_make_input(750 + index) for index in range(4)]

    def recording_forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
        output, loss_func = _forward_step(data_iterator, model, checkpoint_activations_microbatch)
        if model is deferred:
            captured_lengths.append(len(model.embedding_activation_buffer))
        return output, loss_func

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=recording_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    stages = _reference_stages(4, 38, inputs, 4)
    assert len(finalize_calls) == 1
    if pp_rank == 1:
        assert captured_lengths == [1, 2, 3, 4]
        assert len(losses) == 4
    else:
        assert captured_lengths == [0, 0, 0, 0]
        assert losses == []
    assert deferred.embedding_activation_buffer == []
    assert deferred.grad_output_buffer == []
    for chunk_id, chunk in enumerate(chunks):
        position = chunk_id * 2 + pp_rank
        torch.testing.assert_close(
            chunk.linear.weight.grad, stages[position].weight.grad, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            chunk.linear.bias.grad, stages[position].bias.grad, rtol=1e-5, atol=1e-5
        )
