# Copyright (c) 2026, BAAI. All rights reserved.
"""DualPipeV schedule end to end on a V-shaped toy pipeline against an unpipelined reference."""

import collections
import contextlib

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.plugin.dualpipev import dualpipev_schedules
from tests.unit_tests.test_utilities import Utils

SEQ_LENGTH = 4
MICRO_BATCH_SIZE = 2
HIDDEN_SIZE = 8
TOKENS_PER_MICROBATCH = SEQ_LENGTH * MICRO_BATCH_SIZE * HIDDEN_SIZE


class _RecordingTimers:
    """Stands in for megatron Timers: records (kind, name) for every start and stop."""

    def __init__(self):
        self.events = []

    def __call__(self, name, log_level=None):
        events = self.events

        class _Timer:
            def start(self, barrier=False):
                events.append(('start', name))

            def stop(self):
                events.append(('stop', name))

        return _Timer()


class _Stage(torch.nn.Module):
    """One pipeline stage: tanh(x * weight), weight seeded from the stage index."""

    def __init__(self, config, stage):
        super().__init__()
        self.config = config
        self.stage = stage
        self.model_type = ModelType.encoder_or_decoder
        generator = torch.Generator().manual_seed(100 + stage)
        weight = 1.0 + 0.1 * torch.randn(HIDDEN_SIZE, generator=generator)
        self.weight = torch.nn.Parameter(weight.to(torch.cuda.current_device()))
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor[0]

    def forward(self, x):
        source = x if self.input_tensor is None else self.input_tensor
        return torch.tanh(source * self.weight)


def _microbatches(count):
    generator = torch.Generator().manual_seed(7)
    batches = []
    for _ in range(count):
        x = torch.randn(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE, generator=generator)
        target = torch.randn(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE, generator=generator)
        batches.append((x.to(torch.cuda.current_device()), target.to(torch.cuda.current_device())))
    return batches


def _forward_step(loss_kind):
    """forward_step_func; loss_kind 'mean' gives a 2-tuple loss, 'sum' a 3-tuple with token count."""

    def forward_step(data_iterator, model):
        x, target = next(data_iterator)

        def loss_func(output):
            if loss_kind == 'mean':
                loss = ((output - target) ** 2).mean()
                return loss, {'loss': loss.detach().clone()}
            loss = ((output - target) ** 2).sum()
            tokens = torch.tensor(TOKENS_PER_MICROBATCH, dtype=torch.int, device=output.device)
            return loss, tokens, {'loss': loss.detach().clone()}

        return model(x), loss_func

    return forward_step


def _reference(config, pp, batches, loss_kind, per_token):
    """Unpipelined 2*pp-stage chain: per-microbatch losses and every stage's weight gradient."""
    stages = [_Stage(config, stage) for stage in range(2 * pp)]
    losses = []
    for x, target in batches:
        output = x
        for stage in stages:
            output = torch.tanh(output * stage.weight)
        if loss_kind == 'mean':
            loss = ((output - target) ** 2).mean()
            scaled = loss / len(batches)
        else:
            loss = ((output - target) ** 2).sum()
            scaled = loss if per_token else loss / TOKENS_PER_MICROBATCH / len(batches)
        losses.append(loss.detach())
        scaled.backward()
    return torch.stack(losses), [stage.weight.grad for stage in stages]


def _config(pp, **overrides):
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=1,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=pp,
        **overrides,
    )


def _run(config, pp, num_microbatches, loss_kind='mean'):
    rank = parallel_state.get_pipeline_model_parallel_rank()
    chunks = [_Stage(config, rank), _Stage(config, 2 * pp - 1 - rank)]
    batches = _microbatches(num_microbatches)
    store = dualpipev_schedules.forward_backward_pipelining_with_dualpipev(
        forward_step_func=_forward_step(loss_kind),
        data_iterator=[iter(batches), iter(batches)],
        model=chunks,
        num_microbatches=num_microbatches,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
    )
    return chunks, batches, store


def assert_matches_reference(config, pp, chunks, batches, store, loss_kind='mean', per_token=False):
    losses, grads = _reference(config, pp, batches, loss_kind, per_token)
    if parallel_state.get_pipeline_model_parallel_rank() == 0:
        torch.testing.assert_close(torch.stack([entry['loss'] for entry in store]), losses)
    else:
        assert store == []
    for chunk in chunks:
        torch.testing.assert_close(chunk.weight.grad, grads[chunk.stage], rtol=1e-5, atol=1e-6)


@pytest.fixture
def dualpipev(monkeypatch):
    """Initialise model parallel with DualPipeV enabled; reset the module's chunk marker."""
    monkeypatch.setattr(dualpipev_schedules, '_DUALPIPE_CHUNK', None)

    def setup(pp):
        Utils.initialize_model_parallel(
            pipeline_model_parallel_size=pp, create_dualpipev_parallel_size=True
        )

    yield setup
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    ('pp', 'deallocate'),
    [
        pytest.param(2, False, id='pp2'),
        pytest.param(4, False, id='pp4'),
        pytest.param(8, False, id='pp8'),
        pytest.param(4, True, id='pp4-deallocate-outputs'),
    ],
)
def test_schedule_matches_unpipelined_reference(dualpipev, pp, deallocate):
    dualpipev(pp)
    config = _config(pp, deallocate_pipeline_outputs=deallocate)
    chunks, batches, store = _run(config, pp, 2 * pp)
    assert_matches_reference(config, pp, chunks, batches, store)
    assert config.batch_p2p_comm is False


@pytest.mark.parametrize(
    'per_token', [pytest.param(False, id='normalised'), pytest.param(True, id='per-token')]
)
def test_token_count_losses_are_normalised_like_the_reference(dualpipev, per_token):
    dualpipev(4)
    config = _config(4, calculate_per_token_loss=per_token)
    chunks, batches, store = _run(config, 4, 8, loss_kind='sum')
    assert_matches_reference(
        config, 4, chunks, batches, store, loss_kind='sum', per_token=per_token
    )


def test_timers_bracket_every_compute_and_communication_step(dualpipev):
    dualpipev(4)
    config = _config(4)
    config.timers = _RecordingTimers()
    _run(config, 4, 8)
    events = config.timers.events
    counts = collections.Counter(events)
    names = {name for _, name in events}
    assert counts[('start', 'forward-backward')] == 1
    assert counts[('stop', 'forward-backward')] == 1
    assert counts[('start', 'forward-compute')] == 16
    assert counts[('stop', 'forward-compute')] == 16
    assert counts[('start', 'backward-compute')] == 16
    assert counts[('stop', 'backward-compute')] == 16
    assert all(counts[('start', name)] == counts[('stop', name)] for name in names)
    assert {
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-slave-forward-recv',
    } <= names


def test_sync_hooks_wrap_the_schedule_and_receive_each_chunk(dualpipev):
    dualpipev(4)
    calls = {'no_sync': [], 'grad_sync': [], 'finalize': []}

    def no_sync(chunk):
        @contextlib.contextmanager
        def context():
            calls['no_sync'].append(('enter', chunk))
            yield
            calls['no_sync'].append(('exit', chunk))

        return context

    def grad_sync(params):
        calls['grad_sync'].append([parameter.data_ptr() for parameter in params])

    def param_sync(params):
        return None

    def finalize(model, num_tokens, pg_collection=None, force_all_reduce=None):
        calls['finalize'].append(
            (len(model), num_tokens, type(pg_collection).__name__, force_all_reduce)
        )

    config = _config(4)
    config.no_sync_func = [no_sync(0), no_sync(1)]
    config.grad_sync_func = grad_sync
    config.param_sync_func = param_sync
    config.finalize_model_grads_func = finalize
    chunks, _, _ = _run(config, 4, 8)
    assert calls['no_sync'] == [('enter', 0), ('enter', 1), ('exit', 1), ('exit', 0)]
    assert calls['grad_sync'] == [[chunks[0].weight.data_ptr()], [chunks[1].weight.data_ptr()]]
    assert calls['finalize'] == [(2, None, 'ProcessGroupCollection', False)]
    assert config.param_sync_func == [param_sync, param_sync]


def test_finalize_receives_the_local_token_count_for_per_token_loss(dualpipev):
    dualpipev(4)
    seen = []

    def finalize(model, num_tokens, pg_collection=None, force_all_reduce=None):
        seen.append(int(num_tokens))

    config = _config(4, calculate_per_token_loss=True)
    config.finalize_model_grads_func = finalize
    _run(config, 4, 8, loss_kind='sum')
    expected = (
        8 * TOKENS_PER_MICROBATCH if parallel_state.get_pipeline_model_parallel_rank() == 0 else 0
    )
    assert seen == [expected]


@pytest.mark.parametrize(
    'dualpipev_stage', [pytest.param(0, id='first-chunk'), pytest.param(1, id='last-chunk')]
)
def test_stage_is_resolved_from_parallel_state_when_not_given(dualpipev, dualpipev_stage):
    dualpipev(4)
    store = []
    dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(6.0, device=torch.cuda.current_device()),
        lambda t: (t, {'loss': t.detach().clone()}),
        _config(4),
        dualpipev_stage,
        False,
        2,
        store,
    )
    is_loss_stage = dualpipev_stage == 1 and parallel_state.get_pipeline_model_parallel_rank() == 0
    assert len(store) == (1 if is_loss_stage else 0)


@pytest.mark.parametrize(
    ('change', 'error', 'message'),
    [
        pytest.param(
            'communicator_only',
            ValueError,
            'Invalid combination of p2p_communicator, pg_collection',
            id='communicator-without-groups',
        ),
        pytest.param(
            'one_chunk',
            AssertionError,
            'only support chunk model for two consecutive chunks',
            id='one-chunk',
        ),
        pytest.param(
            'one_iterator', AssertionError, 'only support two data_iterators', id='one-iterator'
        ),
        pytest.param(
            'shape_hook',
            AssertionError,
            'adjust_tensor_shapes_fn is not supported',
            id='shape-hook',
        ),
    ],
)
def test_invalid_schedule_arguments_are_rejected(dualpipev, change, error, message):
    dualpipev(2)
    config = _config(2)
    rank = parallel_state.get_pipeline_model_parallel_rank()
    chunks = [_Stage(config, rank), _Stage(config, 3 - rank)]
    iterators = [iter([]), iter([])]
    kwargs = {}
    if change == 'communicator_only':
        kwargs['p2p_communicator'] = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
    if change == 'one_chunk':
        chunks = chunks[:1]
    if change == 'one_iterator':
        iterators = iterators[:1]
    if change == 'shape_hook':
        kwargs['adjust_tensor_shapes_fn'] = lambda *args: args
    with pytest.raises(error, match=message):
        dualpipev_schedules.forward_backward_pipelining_with_dualpipev(
            forward_step_func=_forward_step('mean'),
            data_iterator=iterators,
            model=chunks,
            num_microbatches=4,
            seq_length=SEQ_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            **kwargs,
        )
