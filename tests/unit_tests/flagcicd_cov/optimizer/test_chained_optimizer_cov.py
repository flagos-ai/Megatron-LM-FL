# Copyright (c) 2026, BAAI. All rights reserved.
"""ChainedOptimizer: stepping all children with a combined grad norm, loss-scale delegation, empty-chain stub, state dict save/load across children, zero-grad, CPU offload and restore."""

import copy

import pytest
import torch

from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
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


def _fp32_optimizer(param, config):
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    return FP32Optimizer(base_optimizer, config, None)


def _init_adam_state(opt, config=None):
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)


def _bf16_optimizer(param, config):
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    return Float16OptimizerWithFloat16Params(base_optimizer, config, None, _init_adam_state)


def _config(**overrides):
    defaults = dict(optimizer='adam', lr=0.01, adam_eps=1e-8)
    return OptimizerConfig(**{**defaults, **overrides})


def _reference_adam_step(param_before, grad, lr=0.01):
    reference = torch.nn.Parameter(param_before.detach().clone())
    reference.grad = grad.clone()
    torch.optim.Adam([reference], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
    return reference.detach()


def test_chained_optimizer_step_updates_params_of_all_children(model_parallel):
    model_parallel()
    param_a = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param_b = torch.nn.Parameter(torch.full((2,), 3.0, device=_device()))
    param_a.main_grad = torch.tensor([0.5, 1.0, 2.0, 4.0], device=_device())
    param_b.main_grad = torch.tensor([1.0, 3.0], device=_device())
    config = _config(clip_grad=0.0)
    chained = ChainedOptimizer([_fp32_optimizer(param_a, config), _fp32_optimizer(param_b, config)])

    success, grad_norm, num_zeros_in_grad = chained.step()

    assert success is True
    assert float(grad_norm) == pytest.approx(31.25**0.5, rel=1e-5)
    assert num_zeros_in_grad is None
    torch.testing.assert_close(
        param_a.detach(),
        _reference_adam_step(torch.full((4,), 2.0, device=_device()), param_a.main_grad),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        param_b.detach(),
        _reference_adam_step(torch.full((2,), 3.0, device=_device()), param_b.main_grad),
        rtol=1e-6,
        atol=1e-7,
    )


def test_chained_optimizer_step_reports_grad_norm_and_zero_count_over_children(model_parallel):
    model_parallel()
    param_a = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param_b = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param_a.main_grad = torch.tensor([0.0, 4.0, 0.0, 4.0], device=_device())
    param_b.main_grad = torch.tensor([4.0, 4.0, 4.0, 4.0], device=_device())
    config = _config(clip_grad=1.0, log_num_zeros_in_grad=True)
    chained = ChainedOptimizer([_fp32_optimizer(param_a, config), _fp32_optimizer(param_b, config)])

    success, grad_norm, num_zeros_in_grad = chained.step()

    assert success is True
    assert float(grad_norm) == pytest.approx(96.0**0.5, rel=1e-5)
    assert num_zeros_in_grad == 2


def test_chained_optimizer_get_loss_scale_delegates_to_first_child(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(2, device=_device()))
    chained = ChainedOptimizer([_fp32_optimizer(param, _config())])

    assert float(chained.get_loss_scale()) == 1.0


def test_chained_optimizer_without_children_is_stub_with_default_loss_scale():
    chained = ChainedOptimizer([])

    assert chained.is_stub_optimizer is True
    assert chained.get_parameters() == []
    assert float(chained.get_loss_scale()) == 1.0


def test_chained_optimizer_step_skips_update_when_grad_norm_exceeds_threshold(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param.main_grad = torch.full((4,), 3.0, device=_device())
    param_before = param.detach().clone()
    config = _config(clip_grad=1.0, grad_norm_skip_threshold=1.0)
    chained = ChainedOptimizer([_fp32_optimizer(param, config)])

    success, grad_norm, num_zeros_in_grad = chained.step()

    assert success is False
    assert float(grad_norm) == pytest.approx(6.0, rel=1e-5)
    assert num_zeros_in_grad is None
    torch.testing.assert_close(param.detach(), param_before)


def test_chained_optimizer_state_dict_round_trip_over_children(model_parallel):
    model_parallel()
    param_a = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    param_b = torch.nn.Parameter(torch.full((2,), 3.0, dtype=torch.bfloat16, device=_device()))
    param_a.main_grad = torch.ones(4, device=_device())
    param_b.main_grad = torch.ones(2, device=_device())
    config = _config(clip_grad=0.0, bf16=True)
    optimizer_a = _bf16_optimizer(param_a, config)
    optimizer_b = _bf16_optimizer(param_b, config)
    chained = ChainedOptimizer([optimizer_a, optimizer_b])
    chained.step()

    saved_state = copy.deepcopy(chained.state_dict())
    assert len(saved_state) == 2
    for state in optimizer_a.optimizer.state.values():
        state['exp_avg'].add_(1.0)
    optimizer_a.fp32_from_float16_groups[0][0].data.add_(1.0)

    chained.load_state_dict({0: saved_state[0], 1: saved_state[1]})

    for child, saved_child in zip(chained.chained_optimizers, saved_state):
        restored_inner = child.optimizer.state_dict()
        for param_id, saved_values in saved_child['optimizer']['state'].items():
            live_values = restored_inner['state'][param_id]
            assert set(live_values) == set(saved_values)
            for key, saved_value in saved_values.items():
                torch.testing.assert_close(live_values[key], saved_value)
        for live_group, saved_group in zip(
            restored_inner['param_groups'], saved_child['optimizer']['param_groups']
        ):
            assert set(live_group.keys()) == set(saved_group.keys())
            for key, saved_value in saved_group.items():
                if isinstance(saved_value, torch.Tensor):
                    torch.testing.assert_close(live_group[key].cpu(), saved_value.cpu())
                else:
                    assert live_group[key] == saved_value
        for live_group, saved_group in zip(
            child.fp32_from_float16_groups, saved_child['fp32_from_fp16_params']
        ):
            for live_param, saved_param in zip(live_group, saved_group):
                torch.testing.assert_close(live_param.detach(), saved_param.detach())


def test_chained_optimizer_load_state_dict_rejects_wrong_number_of_entries(model_parallel):
    model_parallel()
    param_a = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param_b = torch.nn.Parameter(torch.full((2,), 3.0, device=_device()))
    config = _config()
    chained = ChainedOptimizer([_fp32_optimizer(param_a, config), _fp32_optimizer(param_b, config)])

    with pytest.raises(RuntimeError, match='Expected 2 entries'):
        chained.load_state_dict({0: {}})


def test_chained_optimizer_zero_grad_clears_all_children(model_parallel):
    model_parallel()
    param_a = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param_b = torch.nn.Parameter(torch.full((2,), 3.0, device=_device()))
    param_a.main_grad = torch.ones(4, device=_device())
    param_b.main_grad = torch.ones(2, device=_device())
    config = _config(clip_grad=0.0)
    chained = ChainedOptimizer([_fp32_optimizer(param_a, config), _fp32_optimizer(param_b, config)])
    chained.step()
    # FP32Optimizer.prepare_grads aliases each param's main_grad into .grad.
    assert param_a.grad is param_a.main_grad
    assert param_b.grad is param_b.main_grad

    chained.zero_grad(set_to_none=True)

    assert param_a.grad is None
    assert param_b.grad is None


def test_chained_optimizer_offload_and_restore_round_trip(model_parallel):
    model_parallel()
    param_a = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param_b = torch.nn.Parameter(torch.full((2,), 3.0, device=_device()))
    param_a.main_grad = torch.ones(4, device=_device())
    param_b.main_grad = torch.ones(2, device=_device())
    config = _config(clip_grad=0.0)
    chained = ChainedOptimizer([_fp32_optimizer(param_a, config), _fp32_optimizer(param_b, config)])
    chained.step()
    param_a_before = param_a.detach().clone()
    param_b_before = param_b.detach().clone()

    chained.offload_to_cpu()
    assert all(not p.is_cuda for p in chained.get_parameters())

    chained.restore_from_cpu()
    assert all(p.device.type == _device().type for p in chained.get_parameters())
    torch.testing.assert_close(param_a.detach(), param_a_before)
    torch.testing.assert_close(param_b.detach(), param_b_before)
