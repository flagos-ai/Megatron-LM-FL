# Copyright (c) 2026, BAAI. All rights reserved.
"""DualPipeV step helpers: loss scaling, backward through stage outputs, chunk bookkeeping."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import rerun_state_machine
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.plugin.dualpipev import dualpipev_schedules
from megatron.training import global_vars


def build_config(**overrides):
    return TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=1, **overrides)


@pytest.fixture
def loss_scales(monkeypatch):
    """Reset the global auxiliary-loss scales; monkeypatch restores them afterwards."""
    monkeypatch.setattr(MoEAuxLossAutoScaler, 'main_loss_backward_scale', None)
    monkeypatch.setattr(MTPLossAutoScaler, 'main_loss_backward_scale', torch.tensor(1.0))


@pytest.fixture
def training_args(monkeypatch):
    """Install megatron.training global args with the two loss checks chosen by the test."""

    def install(nan_check=False, spiky_check=False):
        monkeypatch.setattr(
            global_vars,
            '_GLOBAL_ARGS',
            SimpleNamespace(
                check_for_nan_in_loss_and_grad=nan_check, check_for_spiky_loss=spiky_check
            ),
        )
        monkeypatch.setattr(rerun_state_machine, '_GLOBAL_RERUN_STATE_MACHINE', None)

    return install


class _WgradStore:
    def __init__(self):
        self.disabled = 0

    def disable_delay_wgrad_compute(self):
        self.disabled += 1


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


def test_legacy_loss_is_scaled_by_context_parallel_size_and_microbatches():
    store = []
    output, num_tokens = dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(6.0),
        lambda t: (t, {'loss': t.detach().clone()}),
        build_config(),
        None,
        False,
        4,
        store,
        cp_group_size=2,
        is_last_stage=True,
    )
    assert output.item() == 3.0
    assert num_tokens.item() == 0
    assert [entry['loss'].item() for entry in store] == [6.0]


def test_token_count_loss_is_divided_by_tokens_then_microbatches():
    store = []
    output, num_tokens = dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(12.0),
        lambda t: (t, torch.tensor(3, dtype=torch.int), {'loss': t.detach().clone()}),
        build_config(),
        None,
        False,
        2,
        store,
        cp_group_size=1,
        is_last_stage=True,
    )
    assert output.item() == 2.0
    assert num_tokens.item() == 3
    assert [entry['loss'].item() for entry in store] == [12.0]


def test_per_token_loss_is_left_unnormalised():
    store = []
    output, num_tokens = dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(12.0),
        lambda t: (t, torch.tensor(3, dtype=torch.int), {'loss': t.detach().clone()}),
        build_config(calculate_per_token_loss=True),
        None,
        False,
        2,
        store,
        cp_group_size=1,
        is_last_stage=True,
    )
    assert output.item() == 12.0
    assert num_tokens.item() == 3
    assert [entry['loss'].item() for entry in store] == [12.0]


def test_missing_loss_function_stores_the_raw_output():
    store = []
    output, num_tokens = dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(5.0),
        None,
        build_config(),
        None,
        False,
        2,
        store,
        cp_group_size=1,
        is_last_stage=True,
    )
    assert output.item() == 5.0
    assert [t.item() for t in store] == [5.0]


def test_non_loss_data_is_collected_without_scaling():
    store = []
    output, num_tokens = dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(5.0),
        lambda t, non_loss_data=False: ('collected', non_loss_data),
        build_config(),
        None,
        True,
        2,
        store,
        cp_group_size=1,
        is_last_stage=True,
    )
    assert output.item() == 5.0
    assert store == [('collected', True)]


def test_intermediate_stage_neither_calls_nor_stores_the_loss():
    store = []
    output, num_tokens = dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(5.0),
        lambda t: pytest.fail('loss_func must not run off the last stage'),
        build_config(),
        None,
        False,
        2,
        store,
        cp_group_size=None,
        is_last_stage=False,
    )
    assert output.item() == 5.0
    assert store == []


def test_mismatched_dualpipev_stage_is_rejected():
    with pytest.raises(
        AssertionError, match=r"dualpipev_stage \(0\) doesn't match model_dualpipev_stage \(1\)"
    ):
        dualpipev_schedules.forward_step_calc_loss(
            SimpleNamespace(dualpipev_stage=1),
            torch.tensor(5.0),
            None,
            build_config(),
            0,
            False,
            2,
            [],
            cp_group_size=1,
            is_last_stage=True,
        )


def test_last_stage_requires_context_parallel_size():
    with pytest.raises(AssertionError, match='cp_group_size must be provided on last stage'):
        dualpipev_schedules.forward_step_calc_loss(
            None,
            torch.tensor(5.0),
            None,
            build_config(),
            None,
            False,
            2,
            [],
            cp_group_size=None,
            is_last_stage=True,
        )


@pytest.mark.parametrize(
    ('grad_scale', 'per_token', 'expected'),
    [
        pytest.param(None, False, 0.5, id='default-scale'),
        pytest.param(8.0, False, 4.0, id='grad-scale'),
        pytest.param(8.0, True, 8.0, id='per-token'),
    ],
)
def test_moe_aux_loss_scale_follows_grad_scale_and_microbatches(
    loss_scales, grad_scale, per_token, expected
):
    config = build_config(num_moe_experts=2, calculate_per_token_loss=per_token)
    if grad_scale is not None:
        config.grad_scale_func = lambda t: t * grad_scale
    dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(5.0),
        None,
        config,
        None,
        False,
        4,
        [],
        cp_group_size=2,
        is_last_stage=True,
    )
    torch.testing.assert_close(
        MoEAuxLossAutoScaler.main_loss_backward_scale, torch.tensor([expected])
    )


@pytest.mark.parametrize(
    ('per_token', 'expected'),
    [pytest.param(False, 0.25, id='averaged'), pytest.param(True, 1.0, id='per-token')],
)
def test_mtp_loss_scale_is_divided_by_microbatches(loss_scales, per_token, expected):
    config = build_config(mtp_num_layers=1, calculate_per_token_loss=per_token)
    dualpipev_schedules.forward_step_calc_loss(
        None,
        torch.tensor(5.0),
        None,
        config,
        None,
        False,
        4,
        [],
        cp_group_size=1,
        is_last_stage=True,
    )
    torch.testing.assert_close(MTPLossAutoScaler.main_loss_backward_scale, torch.tensor([expected]))


@pytest.mark.parametrize(
    'deallocate', [pytest.param(False, id='autograd'), pytest.param(True, id='custom-backward')]
)
def test_backward_returns_input_gradient_for_given_output_gradient(deallocate):
    config = build_config(deallocate_pipeline_outputs=deallocate)
    stage_input = torch.tensor([1.0, 2.0], requires_grad=True)
    stage_output = stage_input * 3
    deallocate_output_tensor(stage_output, deallocate)
    grad = dualpipev_schedules.backward_step_with_model_graph(
        stage_input, stage_output, torch.tensor([1.0, 10.0]), config
    )
    torch.testing.assert_close(grad, torch.tensor([3.0, 30.0]))


def test_backward_applies_grad_scale_on_the_loss_stage():
    config = build_config()
    config.grad_scale_func = lambda t: t * 2
    stage_input = torch.tensor([1.0, 2.0], requires_grad=True)
    output = (stage_input * 3).sum()
    grad = dualpipev_schedules.backward_step_with_model_graph(stage_input, output, None, config)
    torch.testing.assert_close(grad, torch.tensor([6.0, 6.0]))


def test_backward_keeps_none_for_inputs_without_tensors():
    stage_input = torch.tensor([1.0, 2.0], requires_grad=True)
    grads = dualpipev_schedules.backward_step_with_model_graph(
        [None, stage_input], [stage_input * 3], [torch.tensor([1.0, 1.0])], build_config()
    )
    assert len(grads) == 2
    assert grads[0] is None
    torch.testing.assert_close(grads[1], torch.tensor([3.0, 3.0]))


def test_backward_times_the_compute_when_timers_are_set():
    config = build_config()
    config.timers = _RecordingTimers()
    stage_input = torch.tensor([1.0, 2.0], requires_grad=True)
    stage_output = stage_input * 3
    deallocate_output_tensor(stage_output, False)
    grad = dualpipev_schedules.backward_step_with_model_graph(
        stage_input, stage_output, torch.tensor([1.0, 10.0]), config
    )
    torch.testing.assert_close(grad, torch.tensor([3.0, 30.0]))
    assert config.timers.events == [('start', 'backward-compute'), ('stop', 'backward-compute')]


def test_loss_func_sums_masked_losses_and_reports_token_count(training_args):
    training_args()
    loss, num_tokens, report = dualpipev_schedules.loss_func(
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]), torch.tensor([[2.0, 5.0], [3.0, 4.0]])
    )
    assert loss.item() == 9.0
    assert num_tokens.item() == 3
    assert num_tokens.dtype == torch.int32
    assert report['lm loss'].tolist() == [9.0, 3.0]


def test_loss_func_accepts_finite_loss_under_nan_and_spike_checks(training_args):
    training_args(nan_check=True, spiky_check=True)
    loss, num_tokens, report = dualpipev_schedules.loss_func(
        torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0])
    )
    assert loss.item() == 5.0
    assert report['lm loss'].tolist() == [5.0, 2.0]


def test_loss_func_rejects_nan_loss(training_args):
    training_args(nan_check=True)
    with pytest.raises(RuntimeError, match='found NaN in local forward loss calculation'):
        dualpipev_schedules.loss_func(torch.ones(2), torch.tensor([float('nan'), 3.0]))


def test_dualpipe_chunk_is_unset_until_selected(monkeypatch):
    monkeypatch.setattr(dualpipev_schedules, '_DUALPIPE_CHUNK', None)
    with pytest.raises(AssertionError, match='_DUALPIPE_CHUNK is None'):
        dualpipev_schedules.get_dualpipe_chunk()


def test_selected_dualpipe_chunk_is_returned(monkeypatch):
    monkeypatch.setattr(dualpipev_schedules, '_DUALPIPE_CHUNK', None)
    dualpipev_schedules.set_dualpipe_chunk(1)
    assert dualpipev_schedules.get_dualpipe_chunk() == 1


def test_dw_detach_is_disabled_on_every_module_with_a_wgrad_store():
    first = torch.nn.Linear(2, 2)
    first.wgrad_store = _WgradStore()
    second = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    second[0].wgrad_store = _WgradStore()
    dualpipev_schedules.disable_dw_detach([first, second])
    assert (first.wgrad_store.disabled, second[0].wgrad_store.disabled) == (1, 1)


def test_dw_detach_requires_exactly_two_chunks():
    with pytest.raises(AssertionError, match='only support chunk model for two consecutive chunks'):
        dualpipev_schedules.disable_dw_detach([torch.nn.Linear(2, 2)])
