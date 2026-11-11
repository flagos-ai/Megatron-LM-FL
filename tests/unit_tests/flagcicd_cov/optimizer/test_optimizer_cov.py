# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronOptimizer and FP32Optimizer: grad zeroing, step extraction, Adam parity, offload."""

import copy

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import (
    FP32Optimizer,
    MegatronOptimizer,
    ProxyDict,
    _multi_tensor_copy_this_to_that,
    _zero_grad_group_helper,
)
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


def _reference_adam_step(param_before, grad, lr=0.01):
    reference = param_before.detach().clone()
    reference = torch.nn.Parameter(reference)
    reference.grad = grad.clone()
    torch.optim.Adam([reference], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
    return reference.detach()


def test_zero_grad_group_helper_sets_grads_to_none():
    param = torch.nn.Parameter(torch.ones(4))
    param.grad = torch.ones(4)
    param.decoupled_grad = torch.full((4,), 5.0)

    _zero_grad_group_helper([param], set_to_none=True)

    assert param.grad is None
    assert torch.equal(param.decoupled_grad, torch.full((4,), 5.0))


def test_zero_grad_group_helper_zeros_grads_in_place():
    param = torch.nn.Parameter(torch.ones(4))
    param.grad = torch.ones(4)
    param_with_autograd_grad = torch.nn.Parameter(torch.ones(4))
    param_with_autograd_grad.grad = torch.ones(4, requires_grad=True) * 2.0

    _zero_grad_group_helper([param, param_with_autograd_grad], set_to_none=False)

    assert torch.equal(param.grad, torch.zeros(4))
    assert torch.equal(param_with_autograd_grad.grad, torch.zeros(4))


def test_zero_grad_group_helper_zeros_decoupled_grad():
    param = torch.nn.Parameter(torch.ones(4))
    param.decoupled_grad = torch.ones(4)

    _zero_grad_group_helper([param], set_to_none=False, use_decoupled_grad=True)
    assert torch.equal(param.decoupled_grad, torch.zeros(4))

    _zero_grad_group_helper([param], set_to_none=True, use_decoupled_grad=True)
    assert param.decoupled_grad is None


def test_multi_tensor_copy_this_to_that_copies_values_with_overflow_buf():
    this = [torch.full((4,), 2.0, device=_device()), torch.full((2,), 3.0, device=_device())]
    that = [torch.zeros(4, device=_device()), torch.zeros(2, device=_device())]
    overflow_buf = torch.zeros(1, dtype=torch.int, device=_device())

    _multi_tensor_copy_this_to_that(this, that, overflow_buf=overflow_buf)

    torch.testing.assert_close(that[0], torch.full((4,), 2.0, device=_device()))
    torch.testing.assert_close(that[1], torch.full((2,), 3.0, device=_device()))


def test_multi_tensor_copy_this_to_that_copies_bfloat16_without_overflow_buf():
    this = [torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device())]
    that = [torch.zeros(4, dtype=torch.bfloat16, device=_device())]

    _multi_tensor_copy_this_to_that(this, that, overflow_buf=None)

    torch.testing.assert_close(
        that[0], torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device())
    )


def test_extract_common_per_param_step_returns_shared_step():
    state_dict = {
        'state': {
            0: {'step': torch.tensor(3.0), 'exp_avg': torch.zeros(2)},
            1: {'step': torch.tensor(3.0), 'exp_avg': torch.zeros(2)},
        }
    }

    assert torch.equal(
        MegatronOptimizer._extract_common_per_param_step(state_dict), torch.tensor(3.0)
    )


def test_extract_common_per_param_step_ignores_params_without_step():
    state_dict = {
        'state': {
            0: {'exp_avg': torch.zeros(2)},
            1: {'step': torch.tensor(5.0), 'exp_avg': torch.zeros(2)},
        }
    }

    assert torch.equal(
        MegatronOptimizer._extract_common_per_param_step(state_dict), torch.tensor(5.0)
    )


def test_extract_common_per_param_step_rejects_diverging_steps():
    state_dict = {'state': {0: {'step': torch.tensor(3.0)}, 1: {'step': torch.tensor(4.0)}}}

    with pytest.raises(ValueError, match='step differs per parameter'):
        MegatronOptimizer._extract_common_per_param_step(state_dict)


def test_restore_common_per_param_step_writes_step_to_all_params():
    state_dict = {'state': {0: {'exp_avg': torch.zeros(2)}, 1: {'exp_avg': torch.zeros(2)}}}

    MegatronOptimizer._restore_common_per_param_step(state_dict, torch.tensor(7.0))

    assert torch.equal(state_dict['state'][0]['step'], torch.tensor(7.0))
    assert torch.equal(state_dict['state'][1]['step'], torch.tensor(7.0))


def test_filter_and_reorder_param_groups_reorders_to_current_groups():
    current_groups = [_param_group([], wd_mult=1.0), _param_group([], wd_mult=0.0)]
    saved_groups = [
        {**_param_group(['b'], wd_mult=0.0), 'extra': 1},
        {**_param_group(['a'], wd_mult=1.0), 'extra': 2},
    ]

    final_groups = MegatronOptimizer._filter_and_reorder_param_groups(current_groups, saved_groups)

    assert len(final_groups) == 2
    assert final_groups[0]['wd_mult'] == 1.0
    assert final_groups[1]['wd_mult'] == 0.0
    # Group hyperparameters are matched to the current groups, while the params
    # list keeps the state dict ordering.
    assert final_groups[0]['params'] == ['b']
    assert final_groups[1]['params'] == ['a']
    assert final_groups[0]['extra'] == 2
    assert final_groups[1]['extra'] == 1


def test_filter_and_reorder_param_groups_rejects_unknown_group():
    current_groups = [_param_group([], wd_mult=1.0)]
    saved_groups = [_param_group([], wd_mult=0.0)]

    with pytest.raises(ValueError, match='Could not find parameter group with key'):
        MegatronOptimizer._filter_and_reorder_param_groups(current_groups, saved_groups)


def test_proxy_dict_indexes_iterates_and_sets_inner_dicts():
    proxy = ProxyDict([{'a': 1, 'b': 2}, {'c': 3}])

    assert len(proxy) == 3
    assert proxy[(0, 'a')] == 1
    assert proxy[(1, 'c')] == 3
    assert proxy[(1, 'missing')] is None
    assert list(proxy) == [(0, 'a'), (0, 'b'), (1, 'c')]
    assert list(proxy.items()) == [((0, 'a'), 1), ((0, 'b'), 2), ((1, 'c'), 3)]

    proxy[(0, 'a')] = 10
    assert proxy[(0, 'a')] == 10


def test_fp32_optimizer_step_matches_reference_adam(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4, 4), 2.0, device=_device()))
    grad = torch.arange(16, dtype=torch.float32, device=_device()).reshape(4, 4)
    param.main_grad = grad
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    config = OptimizerConfig(
        optimizer='adam', lr=0.01, adam_eps=1e-8, clip_grad=0.0, log_num_zeros_in_grad=False
    )
    optimizer = FP32Optimizer(base_optimizer, config, _init_adam_state)

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm is None
    assert num_zeros_in_grad is None
    torch.testing.assert_close(
        param.detach(),
        _reference_adam_step(torch.full((4, 4), 2.0, device=_device()), grad),
        rtol=1e-6,
        atol=1e-7,
    )


def test_fp32_optimizer_clips_grads_to_norm_and_counts_zeros(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 3.0, device=_device()))
    grad = torch.tensor([0.0, 4.0, 0.0, 4.0], device=_device())
    param.main_grad = grad
    original_grad = grad.clone()
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    config = OptimizerConfig(optimizer='adam', lr=0.01, clip_grad=1.0, log_num_zeros_in_grad=True)
    optimizer = FP32Optimizer(base_optimizer, config, _init_adam_state)

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm == pytest.approx(32.0**0.5, rel=1e-5)
    assert num_zeros_in_grad == 2
    clipped_grad = original_grad * (1.0 / (32.0**0.5 + 1.0e-6))
    torch.testing.assert_close(
        param.detach(),
        _reference_adam_step(torch.full((4,), 3.0, device=_device()), clipped_grad),
        rtol=1e-6,
        atol=1e-7,
    )


def test_fp32_optimizer_zero_grad_clears_grads(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    base_optimizer = torch.optim.SGD([_param_group([param])], lr=0.01)
    optimizer = FP32Optimizer(base_optimizer, OptimizerConfig(optimizer='sgd', lr=0.01), None)
    param.grad = torch.ones(4, device=_device())

    optimizer.zero_grad(set_to_none=False)
    assert torch.equal(param.grad, torch.zeros(4, device=_device()))

    optimizer.zero_grad(set_to_none=True)
    assert param.grad is None


def test_fp32_optimizer_get_loss_scale_is_one(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(2, device=_device()))
    optimizer = FP32Optimizer(
        torch.optim.Adam([_param_group([param])], lr=0.01),
        OptimizerConfig(optimizer='adam', lr=0.01),
        None,
    )

    assert float(optimizer.get_loss_scale()) == 1.0
    loss = torch.tensor([2.0], device=_device())
    assert torch.equal(optimizer.scale_loss(loss), loss)


def test_fp32_optimizer_state_dict_round_trip(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param.main_grad = torch.ones(4, device=_device())
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    config = OptimizerConfig(optimizer='adam', lr=0.01, clip_grad=0.0)
    optimizer = FP32Optimizer(base_optimizer, config, _init_adam_state)
    optimizer.step()

    saved_state = copy.deepcopy(optimizer.optimizer.state_dict())
    for state in optimizer.optimizer.state.values():
        state['exp_avg'].add_(1.0)
        state['exp_avg_sq'].add_(2.0)

    optimizer.load_state_dict(saved_state)

    restored_inner = optimizer.optimizer.state_dict()
    for param_id, saved_values in saved_state['state'].items():
        live_values = restored_inner['state'][param_id]
        assert set(live_values) == set(saved_values)
        for key, saved_value in saved_values.items():
            torch.testing.assert_close(live_values[key], saved_value)
    for live_group, saved_group in zip(restored_inner['param_groups'], saved_state['param_groups']):
        assert set(live_group.keys()) == set(saved_group.keys())
        for key, saved_value in saved_group.items():
            assert live_group[key] == saved_value


def test_megatron_optimizer_get_grad_stats_parallel_group_migrates_deprecated_attribute(
    model_parallel,
):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(2, device=_device()))
    optimizer = FP32Optimizer(
        torch.optim.Adam([_param_group([param])], lr=0.01),
        OptimizerConfig(optimizer='adam', lr=0.01),
        None,
    )
    mp_group = parallel_state.get_model_parallel_group()
    optimizer.model_parallel_group = mp_group

    with pytest.warns(UserWarning, match='deprecated and renamed'):
        migrated_group = optimizer.get_grad_stats_parallel_group()

    assert migrated_group is mp_group
    assert not hasattr(optimizer, 'model_parallel_group')
    assert optimizer.grad_stats_parallel_group is mp_group


def test_megatron_optimizer_offload_to_cpu_and_restore_round_trip(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.full((4,), 2.0, device=_device()))
    param.main_grad = torch.ones(4, device=_device())
    base_optimizer = torch.optim.Adam(
        [_param_group([param])], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    optimizer = FP32Optimizer(
        base_optimizer, OptimizerConfig(optimizer='adam', lr=0.01, clip_grad=0.0), _init_adam_state
    )
    optimizer.step()
    param_before = param.detach().clone()

    optimizer.offload_to_cpu()
    assert all(not p.is_cuda for p in optimizer.get_parameters())
    for state in optimizer.optimizer.state.values():
        assert all(not v.is_cuda for v in state.values() if isinstance(v, torch.Tensor))

    optimizer.restore_from_cpu()
    assert all(p.device.type == _device().type for p in optimizer.get_parameters())
    for state in optimizer.optimizer.state.values():
        assert all(
            v.device.type == _device().type for v in state.values() if isinstance(v, torch.Tensor)
        )
    torch.testing.assert_close(param.detach(), param_before)
