# Copyright (c) 2026, BAAI. All rights reserved.
"""Loss reduction in forward steps and P2P tensor-shape derivation."""

from collections import Counter

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from tests.unit_tests.test_utilities import Utils

SEQ_LENGTH = 8
MICRO_BATCH_SIZE = 2
HIDDEN_SIZE = 8


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


def _config(**overrides):
    config = ModelParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2, pipeline_dtype=torch.float32
    )
    config.hidden_size = HIDDEN_SIZE
    config.calculate_per_token_loss = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _calc_loss(
    output_tensor, loss_func, config, store, num_microbatches=2, collect_non_loss_data=False
):
    return schedule.forward_step_calc_loss(
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        output_tensor,
        loss_func,
        config,
        None,
        collect_non_loss_data,
        num_microbatches,
        store,
        cp_group_size=1,
        is_last_stage=True,
    )


def test_calc_loss_appends_raw_output_when_loss_func_is_missing():
    output = torch.tensor([1.0, 2.0])
    store = []

    returned, num_tokens = _calc_loss(output, None, _config(), store)

    assert len(store) == 1
    assert store[0] is output
    assert torch.equal(returned, output)
    assert num_tokens.item() == 0


def test_calc_loss_scales_two_tuple_loss_by_microbatches():
    output = torch.tensor(6.0)
    store = []

    def loss_func(out):
        return out, {'lm loss': out.detach().clone()}

    returned, _ = _calc_loss(output, loss_func, _config(), store)

    assert returned.item() == pytest.approx(3.0)
    assert store[0]['lm loss'].item() == pytest.approx(6.0)


def test_calc_loss_divides_three_tuple_loss_by_tokens_and_microbatches():
    output = torch.tensor(6.0)
    store = []

    def loss_func(out):
        return out, torch.tensor(3), {'lm loss': out.detach().clone() * 10}

    returned, num_tokens = _calc_loss(output, loss_func, _config(), store)

    assert returned.item() == pytest.approx(1.0)
    assert num_tokens.item() == 3
    assert store[0]['lm loss'].item() == pytest.approx(60.0)


def test_calc_loss_clamps_zero_token_count_before_dividing():
    output = torch.tensor(6.0)
    store = []

    def loss_func(out):
        return out, torch.tensor(0), {'lm loss': out.detach().clone()}

    returned, _ = _calc_loss(output, loss_func, _config(), store)

    assert returned.item() == pytest.approx(3.0)


def test_calc_loss_collects_non_loss_data_from_loss_func():
    store = []

    def loss_func(out, non_loss_data=False):
        assert non_loss_data is True
        return out * 2.0

    returned, num_tokens = _calc_loss(
        torch.tensor([1.0, 2.0]), loss_func, _config(), store, collect_non_loss_data=True
    )

    assert torch.equal(store[0], torch.tensor([2.0, 4.0]))
    assert torch.equal(returned, torch.tensor([1.0, 2.0]))
    assert num_tokens.item() == 0


def test_calc_loss_stops_forward_compute_timer():
    timers = _RecordingTimers()
    output = torch.tensor(4.0)
    store = []

    def loss_func(out):
        return out, {'lm loss': out.detach().clone()}

    _, _ = _calc_loss(output, loss_func, _config(timers=timers), store)

    assert timers.stopped['forward-compute'] == 1
    assert timers.started['forward-compute'] == 0


def test_calc_loss_sets_per_token_auxiliary_scales_from_grad_scale_func():
    output = torch.tensor(4.0)
    store = []
    config = _config(
        num_moe_experts=4,
        mtp_num_layers=1,
        dsa_indexer_loss_coeff=0.5,
        calculate_per_token_loss=True,
        grad_scale_func=lambda scale: scale * 2.0,
    )

    def loss_func(out):
        return out, {'lm loss': out.detach().clone()}

    _, _ = _calc_loss(output, loss_func, config, store)

    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(2.0)
    assert MTPLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(2.0)
    assert DSAIndexerLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(2.0)


def test_calc_loss_scales_dsa_loss_by_microbatches_when_not_per_token():
    output = torch.tensor(4.0)
    store = []
    config = _config(dsa_indexer_loss_coeff=0.5, calculate_per_token_loss=False)

    def loss_func(out):
        return out, {'lm loss': out.detach().clone()}

    _, _ = _calc_loss(output, loss_func, config, store, num_microbatches=4)

    assert DSAIndexerLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(0.25)


def test_calc_loss_skips_reduction_when_not_last_stage():
    output = torch.tensor(6.0)
    store = []

    def loss_func(out):
        return out, {'lm loss': out.detach().clone()}

    returned, num_tokens = schedule.forward_step_calc_loss(
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        output,
        loss_func,
        _config(),
        None,
        False,
        2,
        store,
        cp_group_size=1,
        is_last_stage=False,
    )

    assert store == []
    assert torch.equal(returned, output)
    assert num_tokens.item() == 0


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'overrides,decoder_seq_length,expected',
    [
        ({'variable_seq_lengths': True}, None, [()]),
        ({}, None, [(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE)]),
        ({}, 6, [(6, MICRO_BATCH_SIZE, HIDDEN_SIZE)]),
        ({'sequence_parallel': True}, None, [(4, MICRO_BATCH_SIZE, HIDDEN_SIZE)]),
        (
            {'enable_hyper_connections': True, 'num_residual_streams': 3},
            None,
            [(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE * 3)],
        ),
    ],
    ids=['variable', 'fixed', 'decoder-seq', 'sequence-parallel', 'hyper-connections'],
)
def test_get_tensor_shapes_scales_sequence_and_hidden_dimensions(
    model_parallel, overrides, decoder_seq_length, expected
):
    model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)

    tensor_shapes = schedule.get_tensor_shapes(
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        decoder_seq_length=decoder_seq_length,
        config=_config(**overrides),
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        cp_group=parallel_state.get_context_parallel_group(),
    )

    assert tensor_shapes == expected
