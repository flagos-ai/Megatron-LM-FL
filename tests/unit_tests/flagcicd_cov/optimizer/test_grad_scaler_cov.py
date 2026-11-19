# Copyright (c) 2026, BAAI. All rights reserved.
"""Grad scalers: constant scale under overflow, dynamic growth and backoff with hysteresis."""

import pytest
import torch

from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler


def test_constant_grad_scaler_keeps_scale_when_inf_is_found():
    scaler = ConstantGradScaler(4.0)

    assert float(scaler.scale) == 4.0
    assert float(scaler.inv_scale) == 0.25

    scaler.update(True)
    scaler.update(True)

    assert float(scaler.scale) == 4.0
    assert float(scaler.inv_scale) == 0.25


def test_constant_grad_scaler_state_dict_is_empty_and_load_is_noop():
    scaler = ConstantGradScaler(8.0)

    assert scaler.state_dict() == {}

    scaler.load_state_dict({'scale': torch.tensor([2.0])})

    assert float(scaler.scale) == 8.0


def test_grad_scaler_rejects_non_positive_initial_scale():
    with pytest.raises(AssertionError):
        ConstantGradScaler(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"min_scale": 4.0, "initial_scale": 2.0}, id="min-above-initial"),
        pytest.param({"backoff_factor": 1.0}, id="backoff-not-below-one"),
        pytest.param({"growth_factor": 1.0}, id="growth-not-above-one"),
    ],
)
def test_dynamic_grad_scaler_rejects_invalid_arguments(kwargs):
    defaults = dict(
        initial_scale=2.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
        hysteresis=2,
    )
    with pytest.raises(AssertionError):
        DynamicGradScaler(**{**defaults, **kwargs})


def test_dynamic_grad_scaler_holds_scale_while_infs_stay_within_hysteresis_window():
    scaler = DynamicGradScaler(
        initial_scale=1024.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=4,
        hysteresis=2,
    )

    scaler.update(True)

    assert float(scaler.scale) == 1024.0
    assert scaler.state_dict()['hysteresis_tracker'] == 1
    assert scaler.state_dict()['growth_tracker'] == 0


def test_dynamic_grad_scaler_backs_off_after_hysteresis_infs():
    scaler = DynamicGradScaler(
        initial_scale=1024.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=4,
        hysteresis=2,
    )

    scaler.update(True)
    scaler.update(True)
    assert float(scaler.scale) == 512.0

    scaler.update(True)
    assert float(scaler.scale) == 256.0


def test_dynamic_grad_scaler_floors_scale_at_min_scale():
    scaler = DynamicGradScaler(
        initial_scale=2.0,
        min_scale=2.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=4,
        hysteresis=1,
    )

    scaler.update(True)

    assert float(scaler.scale) == 2.0


def test_dynamic_grad_scaler_grows_after_growth_interval_of_finite_grads():
    scaler = DynamicGradScaler(
        initial_scale=1024.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
        hysteresis=2,
    )

    scaler.update(False)
    assert float(scaler.scale) == 1024.0

    scaler.update(False)
    assert float(scaler.scale) == 2048.0
    assert scaler.state_dict()['growth_tracker'] == 0
    assert scaler.state_dict()['hysteresis_tracker'] == 2


def test_dynamic_grad_scaler_state_dict_round_trip_restores_scale_and_trackers():
    scaler = DynamicGradScaler(
        initial_scale=1024.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
        hysteresis=2,
    )
    scaler.update(False)

    state_dict = scaler.state_dict()
    assert set(state_dict.keys()) == {'scale', 'growth_tracker', 'hysteresis_tracker'}
    assert float(state_dict['scale']) == 1024.0
    assert state_dict['growth_tracker'] == 1

    restored = DynamicGradScaler(
        initial_scale=1.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
        hysteresis=2,
    )
    restored.load_state_dict(state_dict)

    assert float(restored.scale) == 1024.0
    assert restored.state_dict() == {
        'scale': restored.scale,
        'growth_tracker': 1,
        'hysteresis_tracker': 2,
    }
