# Copyright (c) 2026, BAAI. All rights reserved.
"""Learning-rate schedule branches the developer-owned scheduler tests do not reach."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def make_scheduler(lr_decay_style, param_groups=None, **overrides):
    optimizer = MagicMock()
    optimizer.param_groups = param_groups or [{'lr': 0.0, 'weight_decay': 0.0}]
    arguments = dict(
        optimizer=optimizer,
        init_lr=0.0,
        max_lr=1.0,
        min_lr=0.1,
        lr_warmup_steps=10,
        lr_decay_steps=110,
        lr_decay_style=lr_decay_style,
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=110,
        wd_incr_style='constant',
    )
    arguments.update(overrides)
    return OptimizerParamScheduler(**arguments)


def lr_at(scheduler, step):
    scheduler.num_steps = step
    return scheduler.get_lr(scheduler.optimizer.param_groups[0])


def test_constant_style_holds_max_lr_after_warmup_and_past_decay_steps():
    scheduler = make_scheduler('constant')
    assert lr_at(scheduler, 5) == pytest.approx(0.5)
    assert lr_at(scheduler, 11) == 1.0
    assert lr_at(scheduler, 10_000) == 1.0


def test_steps_past_decay_return_min_lr():
    assert lr_at(make_scheduler('linear'), 111) == 0.1


def test_inverse_square_root_decays_and_is_floored_at_min_lr():
    scheduler = make_scheduler('inverse-square-root', lr_decay_steps=100_000)
    assert lr_at(scheduler, 40) == pytest.approx(math.sqrt(10) / math.sqrt(40))
    # 1.0 * sqrt(10) / sqrt(9000) is below min_lr.
    assert lr_at(scheduler, 9000) == 0.1


def test_inverse_square_root_without_warmup_treats_step_zero_as_one():
    scheduler = make_scheduler('inverse-square-root', lr_warmup_steps=0)
    assert lr_at(scheduler, 0) == 1.0
    assert lr_at(scheduler, 4) == pytest.approx(0.5)


def test_warmup_stable_decay_halves_every_decay_window():
    scheduler = make_scheduler('warmup-stable-decay')
    # 100 post-warmup steps split 10:1 -> S = round(100 * 10 / 11) = 91 stable, T = 9 decay.
    stable_steps = 91
    decay_window = 9
    assert lr_at(scheduler, stable_steps - 1) == 1.0
    assert lr_at(scheduler, stable_steps) == 1.0
    assert lr_at(scheduler, stable_steps + decay_window) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ('style', 'coefficient'),
    [
        ('linear', lambda ratio: 1.0 - ratio),
        ('cosine', lambda ratio: 0.5 * (math.cos(math.pi * ratio) + 1.0)),
        ('exponential', lambda ratio: 2.0 * 0.5**ratio - 1.0),
        ('minus_sqrt', lambda ratio: 1.0 - math.sqrt(ratio)),
    ],
)
def test_wsd_anneals_only_inside_the_final_window(style, coefficient):
    scheduler = make_scheduler('WSD', wsd_decay_steps=40, lr_wsd_decay_style=style)
    # The anneal window is the last 40 of 110 steps.
    assert lr_at(scheduler, 70) == pytest.approx(1.0)
    assert lr_at(scheduler, 80) == pytest.approx(0.1 + coefficient(10 / 40) * 0.9)
    assert lr_at(scheduler, 110) == pytest.approx(0.1 + coefficient(1.0) * 0.9)


def test_wsd_requires_decay_steps():
    with pytest.raises(AssertionError):
        make_scheduler('WSD')


def test_unknown_decay_style_is_rejected():
    with pytest.raises(Exception, match='not supported'):
        make_scheduler('bogus', lr_warmup_steps=0)


def test_param_group_bounds_override_scheduler_bounds():
    scheduler = make_scheduler('linear', param_groups=[{'lr': 0.0, 'max_lr': 0.5, 'min_lr': 0.3}])
    assert lr_at(scheduler, 10) == pytest.approx(0.5)
    assert lr_at(scheduler, 60) == pytest.approx(0.4)
    assert lr_at(scheduler, 111) == 0.3


def test_step_fills_tensor_learning_rates_in_place():
    learning_rate = torch.zeros(())
    scheduler = make_scheduler(
        'constant', param_groups=[{'lr': learning_rate, 'weight_decay': 0.0, 'wd_mult': 2.0}]
    )
    scheduler.step(20)
    assert scheduler.optimizer.param_groups[0]['lr'] is learning_rate
    assert learning_rate.item() == 1.0


def stablelm2_config(**overrides):
    values = dict(
        cosine_samples=40,
        cosine_period_samples=0,
        cosine_max_lr=1.0,
        rsqrt_samples=30,
        alpha=2.0,
        beta=1.0,
        global_batch_size=10,
        decay_samples=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stablelm2_normalises_sample_counts_on_construction():
    config = stablelm2_config()
    make_scheduler('linear', stablelm2_scheduler_config=config)
    assert config.rsqrt_samples == 70
    assert config.cosine_period_samples == 110


def test_stablelm2_runs_cosine_then_rsqrt_then_linear_to_zero():
    config = stablelm2_config()
    scheduler = make_scheduler('linear', stablelm2_scheduler_config=config)

    cosine = lr_at(scheduler, 22)
    assert cosine == pytest.approx(0.1 + 0.5 * (math.cos(2 * math.pi * 22 / 110) + 1.0) * 0.9)
    assert config.cosine_lr == cosine

    rsqrt = lr_at(scheduler, 60)
    assert rsqrt == pytest.approx(2.0 / math.sqrt(60 / 10 + 1.0))
    assert config.rsqrt_lr == rsqrt

    # Without decay_samples the tail decays linearly from the last rsqrt value to zero.
    assert lr_at(scheduler, 90) == pytest.approx((1.0 - (90 - 70) / (110 - 70)) * rsqrt)


def test_stablelm2_decay_samples_bound_the_linear_tail_at_min_lr():
    config = stablelm2_config(decay_samples=20, cosine_period_samples=80)
    scheduler = make_scheduler('linear', stablelm2_scheduler_config=config)
    assert config.cosine_period_samples == 80

    rsqrt = lr_at(scheduler, 70)
    # The tail spans steps 70..90 and lands on min_lr.
    assert lr_at(scheduler, 80) == pytest.approx(0.5 * (rsqrt - 0.1) + 0.1)
    assert lr_at(scheduler, 90) == pytest.approx(0.1)
    assert lr_at(scheduler, 100) == 0.1


def legacy_state_dict():
    return {
        'start_lr': 0.8,
        'min_lr': 0.2,
        'warmup_iter': 5,
        'end_iter': 200,
        'decay_style': 'cosine',
        'num_iters': 7,
    }


def test_load_state_dict_accepts_legacy_checkpoint_keys():
    scheduler = make_scheduler('linear')
    scheduler.load_state_dict(legacy_state_dict())
    assert scheduler.max_lr == 0.8
    assert scheduler.min_lr == 0.2
    assert scheduler.lr_warmup_steps == 5
    assert scheduler.lr_decay_steps == 200
    assert scheduler.lr_decay_style == 'cosine'
    assert scheduler.num_steps == 7
    # A checkpoint without weight-decay keys leaves the constructor values alone.
    assert scheduler.wd_incr_steps == 110


def test_load_state_dict_accepts_intermediate_checkpoint_keys():
    scheduler = make_scheduler('linear')
    state = legacy_state_dict()
    state['warmup_steps'] = state.pop('warmup_iter')
    state['decay_steps'] = state.pop('end_iter')
    scheduler.load_state_dict(state)
    assert scheduler.lr_warmup_steps == 5
    assert scheduler.lr_decay_steps == 200


def test_override_keeps_constructor_values_over_the_checkpoint():
    scheduler = make_scheduler(
        'linear', override_opt_param_scheduler=True, use_checkpoint_opt_param_scheduler=False
    )
    scheduler.load_state_dict(legacy_state_dict())
    assert scheduler.max_lr == 1.0
    assert scheduler.lr_decay_steps == 110
    assert scheduler.lr_decay_style == 'linear'


def test_checkpoint_mismatch_is_rejected_when_checkpoint_values_are_not_trusted():
    scheduler = make_scheduler('linear', use_checkpoint_opt_param_scheduler=False)
    with pytest.raises(AssertionError, match='do not match'):
        scheduler.load_state_dict(legacy_state_dict())


def test_override_and_use_checkpoint_are_mutually_exclusive():
    with pytest.raises(AssertionError, match='both override'):
        make_scheduler('linear', override_opt_param_scheduler=True)
