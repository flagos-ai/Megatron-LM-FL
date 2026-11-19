# Copyright (c) 2026, BAAI. All rights reserved.
"""Float16OptimizerWithFloat16Params: bf16/fp16 Adam parity, clipping, overflow, state dicts."""

import copy

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _param_group(params, lr=0.01, wd_mult=1.0):
    return {
        'params': list(params),
        'is_expert_parallel': False,
        'is_engram_parallel': False,
        'is_vision_model_param': False,
        'default_config': True,
        'wd_mult': wd_mult,
        'lr_mult': 1.0,
        'is_decoupled_lr': False,
        'max_lr': lr,
        'min_lr': None,
    }


def _init_adam_state(opt, config=None):
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)


def _build_optimizer(param, config, grad_scaler):
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    return Float16OptimizerWithFloat16Params(base_optimizer, config, grad_scaler, _init_adam_state)


def _reference_adam_step(initial_value, grad, dtype):
    reference = torch.nn.Parameter(initial_value.detach().clone().float())
    reference.grad = grad.float().clone()
    torch.optim.Adam([reference], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
    return reference.detach().to(dtype)


def test_bf16_optimizer_step_matches_reference_adam_with_fp32_main_params(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    grad = torch.tensor([0.5, 1.0, 2.0, 4.0], device=_device())
    param.main_grad = grad
    config = OptimizerConfig(optimizer='adam', lr=0.01, adam_eps=1e-8, bf16=True, clip_grad=0.0)
    optimizer = _build_optimizer(param, config, None)
    main_param = optimizer.fp32_from_float16_groups[0][0]

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm == 0.0
    assert num_zeros_in_grad == 0
    assert main_param.dtype == torch.float32
    assert param.dtype == torch.bfloat16
    assert param.grad is None
    torch.testing.assert_close(
        param.detach(),
        _reference_adam_step(
            torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()), grad, torch.bfloat16
        ),
        rtol=1.6e-2,
        atol=1e-2,
    )


def test_bf16_optimizer_step_clips_grads_and_counts_zeros(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    grad = torch.tensor([0.0, 4.0, 0.0, 4.0], device=_device())
    param.main_grad = grad
    original_grad = grad.clone()
    config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        bf16=True,
        clip_grad=1.0,
        log_num_zeros_in_grad=True,
    )
    optimizer = _build_optimizer(param, config, None)

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert float(grad_norm) == pytest.approx(32.0**0.5, rel=1e-5)
    assert num_zeros_in_grad == 2
    clipped_grad = original_grad * (1.0 / (32.0**0.5 + 1.0e-6))
    torch.testing.assert_close(
        param.detach(),
        _reference_adam_step(
            torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()),
            clipped_grad,
            torch.bfloat16,
        ),
        rtol=1.6e-2,
        atol=1e-2,
    )


def test_bf16_optimizer_zero_grad_clears_model_and_main_grads(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    param.main_grad = torch.ones(4, device=_device())
    config = OptimizerConfig(optimizer='adam', lr=0.01, bf16=True, clip_grad=0.0)
    optimizer = _build_optimizer(param, config, None)
    optimizer.step()
    main_param = optimizer.fp32_from_float16_groups[0][0]
    param.grad = torch.ones(4, dtype=torch.bfloat16, device=_device())

    optimizer.zero_grad(set_to_none=False)
    assert torch.equal(param.grad, torch.zeros(4, dtype=torch.bfloat16, device=_device()))
    assert torch.equal(main_param.grad, torch.zeros(4, device=_device()))

    optimizer.zero_grad(set_to_none=True)
    assert param.grad is None
    assert main_param.grad is None


def test_bf16_optimizer_state_dict_round_trip_restores_state_and_main_params(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    param.main_grad = torch.tensor([0.5, 1.0, 2.0, 4.0], device=_device())
    config = OptimizerConfig(optimizer='adam', lr=0.01, adam_eps=1e-8, bf16=True, clip_grad=0.0)
    optimizer = _build_optimizer(param, config, None)
    optimizer.step()

    saved_state = copy.deepcopy(optimizer.state_dict())
    assert set(saved_state.keys()) == {'optimizer', 'fp32_from_fp16_params'}

    fresh_param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    fresh_optimizer = _build_optimizer(fresh_param, config, None)
    fresh_optimizer.load_state_dict(saved_state)

    restored_inner = fresh_optimizer.optimizer.state_dict()
    for param_id, saved_values in saved_state['optimizer']['state'].items():
        live_values = restored_inner['state'][param_id]
        assert set(live_values) == set(saved_values)
        for key, saved_value in saved_values.items():
            torch.testing.assert_close(live_values[key], saved_value)
    for live_group, saved_group in zip(
        restored_inner['param_groups'], saved_state['optimizer']['param_groups']
    ):
        assert set(live_group.keys()) == set(saved_group.keys())
        for key, saved_value in saved_group.items():
            if isinstance(saved_value, torch.Tensor):
                torch.testing.assert_close(live_group[key].cpu(), saved_value.cpu())
            else:
                assert live_group[key] == saved_value
    for live_group, saved_group in zip(
        fresh_optimizer.fp32_from_float16_groups, saved_state['fp32_from_fp16_params']
    ):
        for live_param, saved_param in zip(live_group, saved_group):
            torch.testing.assert_close(live_param.detach(), saved_param.detach())


def test_bf16_optimizer_load_state_dict_accepts_legacy_checkpoint_keys(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    param.main_grad = torch.tensor([0.5, 1.0, 2.0, 4.0], device=_device())
    config = OptimizerConfig(optimizer='adam', lr=0.01, adam_eps=1e-8, bf16=True, clip_grad=0.0)
    optimizer = _build_optimizer(param, config, None)
    optimizer.step()

    saved_state = copy.deepcopy(optimizer.state_dict())
    legacy_state = {
        'optimizer_state_dict': saved_state['optimizer'],
        'fp32_from_fp16': saved_state['fp32_from_fp16_params'],
    }

    fresh_param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    fresh_optimizer = _build_optimizer(fresh_param, config, None)
    fresh_optimizer.load_state_dict(legacy_state)

    restored_inner = fresh_optimizer.optimizer.state_dict()
    for param_id, saved_values in legacy_state['optimizer_state_dict']['state'].items():
        live_values = restored_inner['state'][param_id]
        assert set(live_values) == set(saved_values)
        for key, saved_value in saved_values.items():
            torch.testing.assert_close(live_values[key], saved_value)
    for live_group, saved_group in zip(
        fresh_optimizer.fp32_from_float16_groups, legacy_state['fp32_from_fp16']
    ):
        for live_param, saved_param in zip(live_group, saved_group):
            torch.testing.assert_close(live_param.detach(), saved_param.detach())


def test_fp16_optimizer_skips_step_and_leaves_params_when_grads_overflow(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.float16, device=_device()))
    param.main_grad = torch.tensor([float('inf'), 1.0, 2.0, 4.0], device=_device())
    config = OptimizerConfig(
        optimizer='adam', lr=0.01, adam_eps=1e-8, fp16=True, loss_scale=4.0, clip_grad=1.0
    )
    optimizer = _build_optimizer(param, config, ConstantGradScaler(4.0))
    main_param = optimizer.fp32_from_float16_groups[0][0]
    param_before = param.detach().clone()

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is False
    assert grad_norm is None
    assert num_zeros_in_grad is None
    torch.testing.assert_close(param.detach(), param_before)
    torch.testing.assert_close(
        main_param.grad, torch.tensor([float('inf'), 0.25, 0.5, 1.0], device=_device())
    )


def test_fp16_optimizer_holds_dynamic_scale_within_hysteresis_then_backs_off(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.float16, device=_device()))
    config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        fp16=True,
        clip_grad=1.0,
        initial_loss_scale=4.0,
        hysteresis=2,
        loss_scale_window=100,
    )
    grad_scaler = DynamicGradScaler(
        initial_scale=4.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100,
        hysteresis=2,
    )
    optimizer = _build_optimizer(param, config, grad_scaler)
    param_before = param.detach().clone()

    param.main_grad = torch.full((4,), float('inf'), device=_device())
    success, _, _ = optimizer.step()
    assert success is False
    assert float(optimizer.grad_scaler.scale) == 4.0

    optimizer.zero_grad()
    param.main_grad = torch.full((4,), float('inf'), device=_device())
    success, _, _ = optimizer.step()
    assert success is False
    assert float(optimizer.grad_scaler.scale) == 2.0

    torch.testing.assert_close(param.detach(), param_before)


def test_fp16_optimizer_steps_with_unscaled_grads_after_overflow_backoff(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.float16, device=_device()))
    grad = torch.tensor([0.5, 1.0, 2.0, 4.0], device=_device())
    config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        fp16=True,
        clip_grad=0.0,
        initial_loss_scale=4.0,
        hysteresis=1,
        loss_scale_window=100,
    )
    grad_scaler = DynamicGradScaler(
        initial_scale=4.0,
        min_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100,
        hysteresis=1,
    )
    optimizer = _build_optimizer(param, config, grad_scaler)

    param.main_grad = torch.full((4,), float('inf'), device=_device())
    success, _, _ = optimizer.step()
    assert success is False
    assert float(optimizer.grad_scaler.scale) == 2.0

    optimizer.zero_grad()
    param.main_grad = 2.0 * grad
    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm == 0.0
    assert num_zeros_in_grad == 0
    torch.testing.assert_close(
        param.detach(),
        _reference_adam_step(
            torch.full((4,), 2.0, dtype=torch.float16, device=_device()), grad, torch.float16
        ),
        rtol=1e-3,
        atol=1e-3,
    )
