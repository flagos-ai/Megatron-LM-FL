# Copyright (c) 2026, BAAI. All rights reserved.
"""Non-interleaved pipelined schedule: losses, grads, and argument validation."""

from contextlib import contextmanager

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
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


def _config(**overrides):
    config = ModelParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2, pipeline_dtype=torch.float32
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


def _pipeline_call(model, inputs, num_microbatches, **config_overrides):
    return schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
        **config_overrides,
    )


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


def test_without_interleaving_reduces_loss_and_grads_on_both_stages(is_first_stage):
    num_microbatches = 2
    timers = _RecordingTimers()
    config = _config(timers=timers)
    linear = _make_linear(21).cuda()
    model = _StageModule(config, linear)
    inputs = [_make_input(600 + index) for index in range(num_microbatches)]

    losses = _pipeline_call(model, inputs, num_microbatches)

    total, stages = _reference_two_stage(linear, inputs, num_microbatches)
    if is_first_stage:
        assert losses == []
        torch.testing.assert_close(linear.weight.grad, stages[0].weight.grad, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(linear.bias.grad, stages[0].bias.grad, rtol=1e-5, atol=1e-5)
    else:
        assert len(losses) == num_microbatches
        assert sum(entry['lm loss'].item() for entry in losses) == pytest.approx(
            total.item(), rel=1e-5, abs=1e-5
        )
        torch.testing.assert_close(linear.weight.grad, stages[1].weight.grad, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(linear.bias.grad, stages[1].bias.grad, rtol=1e-5, atol=1e-5)
    assert timers.started['forward-compute'] == timers.stopped['forward-compute']
    assert timers.started['backward-compute'] == timers.stopped['backward-compute']
    assert timers.started['forward-compute'] == num_microbatches
    assert timers.started['backward-compute'] == num_microbatches
    assert timers.started['forward-backward'] == 1
    assert timers.stopped['forward-backward'] == 1


def test_without_interleaving_forward_only_matches_reference_and_skips_backward(is_first_stage):
    config = _config(
        variable_seq_lengths=True,
        cuda_graph_impl='local',
        moe_paged_stash=True,
        moe_paged_stash_page_size=128,
    )
    linear = _make_linear(22).cuda()
    model = _StageModule(config, linear)
    inputs = [_make_input(610 + index) for index in range(2)]

    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=2,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=True,
    )

    total, _ = _reference_two_stage(linear, inputs, 2)
    if is_first_stage:
        assert losses == []
    else:
        assert len(losses) == 2
        assert sum(entry['lm loss'].item() for entry in losses) == pytest.approx(
            total.item(), rel=1e-5, abs=1e-5
        )
    assert linear.weight.grad is None
    assert linear.bias.grad is None


def test_without_interleaving_rejects_overlap_p2p_comm(is_first_stage):
    config = _config(overlap_p2p_comm=True)
    model = _StageModule(config, _make_linear(23).cuda())
    with pytest.raises(ValueError, match='does not support overlapping p2p communication'):
        _pipeline_call(model, [_make_input(620)], 1)


def test_without_interleaving_rejects_partial_process_groups(is_first_stage):
    config = _config()
    model = _StageModule(config, _make_linear(24).cuda())
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    communicator = P2PCommunicator(pp_group=pp_group, config=config)
    with pytest.raises(ValueError, match='Provide both p2p_communicator and pg_collection'):
        _pipeline_call(model, [_make_input(621)], 1, p2p_communicator=communicator)


def test_without_interleaving_rejects_unrelated_pg_collection_type(is_first_stage):
    config = _config()
    model = _StageModule(config, _make_linear(25).cuda())
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    communicator = P2PCommunicator(pp_group=pp_group, config=config)
    with pytest.raises(
        TypeError, match='must be ProcessGroupCollection or MultiModuleProcessGroupCollection'
    ):
        _pipeline_call(
            model, [_make_input(622)], 1, p2p_communicator=communicator, pg_collection=object()
        )


def test_without_interleaving_requires_variable_seq_lengths_for_multimodule(is_first_stage):
    config = _config()
    model = _StageModule(config, _make_linear(26).cuda())
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    communicator = P2PCommunicator(pp_group=pp_group, config=config)
    language_pgs = ProcessGroupCollection()
    language_pgs.cp = parallel_state.get_context_parallel_group()
    pg_collection = MultiModuleProcessGroupCollection(
        module_pgs={'language': language_pgs}, language_model_module_name='language'
    )
    with pytest.raises(ValueError, match='variable_seq_lengths=True required'):
        _pipeline_call(
            model, [_make_input(623)], 1, p2p_communicator=communicator, pg_collection=pg_collection
        )


def test_without_interleaving_passes_partial_checkpoint_ids_per_stage(is_first_stage):
    captured = []

    def recording_forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
        captured.append(checkpoint_activations_microbatch)
        return _forward_step(data_iterator, model, checkpoint_activations_microbatch)

    config = _config(num_microbatches_with_partial_activation_checkpoints=1)
    linear = _make_linear(27).cuda()
    model = _StageModule(config, linear)

    schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=recording_forward_step,
        data_iterator=iter([_make_input(630 + index) for index in range(2)]),
        model=[model],
        num_microbatches=2,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    if is_first_stage:
        assert captured == [False, True]
    else:
        assert captured == [False, False]


def test_without_interleaving_adjusts_tensor_shapes_through_callback(is_first_stage):
    recorded = []

    def adjust_tensor_shapes_fn(recv_tensor_shapes, send_tensor_shapes):
        recorded.append((list(recv_tensor_shapes), list(send_tensor_shapes)))
        return recv_tensor_shapes, send_tensor_shapes

    config = _config()
    linear = _make_linear(28).cuda()
    model = _StageModule(config, linear)
    inputs = [_make_input(640), _make_input(641)]

    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=_forward_step,
        data_iterator=iter(inputs),
        model=[model],
        num_microbatches=2,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
        adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
    )

    assert recorded == [
        (
            [(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE)],
            [(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE)],
        )
    ]
    if not is_first_stage:
        assert len(losses) == 2


def test_without_interleaving_flushes_remaining_grad_sync_outside_no_sync(is_first_stage):
    sync_calls = []

    @contextmanager
    def no_sync_func():
        yield

    def grad_sync_func(parameters):
        sync_calls.append([tuple(parameter.shape) for parameter in parameters])

    config = _config(no_sync_func=no_sync_func, grad_sync_func=grad_sync_func)
    linear = _make_linear(29).cuda()
    model = _StageModule(config, linear)
    inputs = [_make_input(650), _make_input(651)]

    losses = _pipeline_call(model, inputs, 2)

    _, stages = _reference_two_stage(linear, inputs, 2)
    if is_first_stage:
        assert sync_calls == []
        torch.testing.assert_close(linear.weight.grad, stages[0].weight.grad, rtol=1e-5, atol=1e-5)
    else:
        assert sync_calls == [[(HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE,)]]
        torch.testing.assert_close(linear.weight.grad, stages[1].weight.grad, rtol=1e-5, atol=1e-5)
        assert len(losses) == 2


def test_backward_step_multimodule_runs_per_module_backward_and_collects_grads():
    weight = torch.nn.Parameter(torch.randn(3, 3, device='cuda'))
    vision = torch.randn(2, 3, device='cuda', requires_grad=True)
    language = torch.randn(2, 3, device='cuda', requires_grad=True)

    vision_loss = (vision @ weight).sum()
    language_loss = (language @ weight).sum()
    config = _config(
        pipeline_model_parallel_size=1,
        deallocate_pipeline_outputs=True,
        grad_scale_func=lambda tensor: tensor * 2.0,
    )

    input_tensor_grad = schedule.backward_step_multimodule(
        {'vision': vision, 'language': None},
        {'vision': vision_loss, 'language': language_loss},
        None,
        config,
        'language',
    )

    reference = torch.randn(2, 3, device='cuda', requires_grad=True)
    ((reference @ weight).sum() * 2.0).backward()
    assert input_tensor_grad['language'] is None
    torch.testing.assert_close(input_tensor_grad['vision'], reference.grad, rtol=1e-5, atol=1e-5)
    ones = torch.ones(2, 3, device='cuda')
    torch.testing.assert_close(
        weight.grad,
        2.0 * (vision.detach().t() @ ones) + 2.0 * (language.detach().t() @ ones),
        rtol=1e-5,
        atol=1e-5,
    )
