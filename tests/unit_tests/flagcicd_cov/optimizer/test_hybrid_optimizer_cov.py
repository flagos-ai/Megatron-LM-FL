# Copyright (c) 2026, BAAI. All rights reserved.
"""HybridDeviceOptimizer: CPU sub-optimizer splitting, fp32 master params, stepping, state dicts."""

import copy

import pytest
import torch

from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _param_group(params, lr=0.01):
    return {'params': list(params), 'lr': lr, 'weight_decay': 0.0}


def _build_hdo(params, offload_fraction=0.0, param_update_in_fp32=False):
    return HybridDeviceOptimizer(
        [_param_group(params)],
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=param_update_in_fp32,
        overlap_cpu_optimizer_d2h_h2d=False,
        pin_cpu_grads=False,
    )


def _reference_adamw_step(initial_value, grad, dtype):
    reference = torch.nn.Parameter(initial_value.detach().clone().float())
    reference.grad = grad.float().clone()
    torch.optim.AdamW([reference], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
    return reference.detach().to(dtype)


@torch.no_grad()
def _perturb_state(optimizer):
    # The state also holds the fp32 master parameter, a leaf that requires grad.
    for param, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                value.add_(100.0)


@pytest.mark.parametrize(
    "group_params,expected_num_optimizers",
    [
        pytest.param(
            lambda: [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))],
            2,
            id="one-optimizer-per-param",
        ),
        pytest.param(lambda: torch.nn.Parameter(torch.ones(3)), 1, id="bare-tensor-params"),
        pytest.param(lambda: [], 0, id="empty-groups"),
    ],
)
def test_hybrid_optimizer_build_cpu_optimizer_list_splits_params(
    group_params, expected_num_optimizers
):
    optimizers = HybridDeviceOptimizer.build_cpu_optimizer_list(
        torch.optim.SGD, [{'params': group_params(), 'lr': 0.1}]
    )

    assert len(optimizers) == expected_num_optimizers
    for optimizer in optimizers:
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]['params']) == 1
        assert optimizer.param_groups[0]['lr'] == 0.1


def test_hybrid_optimizer_build_cpu_optimizer_list_returns_empty_without_groups():
    assert HybridDeviceOptimizer.build_cpu_optimizer_list(torch.optim.SGD, []) == []


def test_hybrid_optimizer_update_fp32_param_by_new_param_copies_current_values():
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    optimizer = _build_hdo([param], param_update_in_fp32=True)

    fp32_param = optimizer.param_to_fp32_param[param]
    assert fp32_param.dtype == torch.float32
    torch.testing.assert_close(
        fp32_param.detach(), torch.full((4,), 2.0, device=_device()), rtol=0, atol=0
    )

    param.data.copy_(torch.full((4,), 3.0, dtype=torch.bfloat16, device=_device()))
    optimizer.update_fp32_param_by_new_param()

    torch.testing.assert_close(
        fp32_param.detach(), torch.full((4,), 3.0, device=_device()), rtol=0, atol=0
    )


def test_hybrid_optimizer_zero_grad_clears_param_and_decoupled_grads():
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    optimizer = _build_hdo([param])
    param.grad = torch.full((4,), 2.0, device=_device())
    param.decoupled_grad = torch.full((4,), 3.0, device=_device())

    optimizer.zero_grad(set_to_none=False)
    assert torch.equal(param.grad, torch.zeros(4, device=_device()))
    assert torch.equal(param.decoupled_grad, torch.zeros(4, device=_device()))

    optimizer.zero_grad(set_to_none=True)
    assert param.grad is None
    assert param.decoupled_grad is None


def test_hybrid_optimizer_steps_resident_and_offloaded_params_against_reference():
    param_resident = torch.nn.Parameter(
        torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device())
    )
    param_offloaded = torch.nn.Parameter(
        torch.full((6,), 3.0, dtype=torch.bfloat16, device=_device())
    )
    param_offloaded.is_offloading_candidate = True
    optimizer = _build_hdo(
        [param_resident, param_offloaded], offload_fraction=0.0, param_update_in_fp32=True
    )

    assert set(optimizer.param_to_fp32_param) == {param_resident, param_offloaded}
    assert set(optimizer.gpu_params_map_cpu_copy) == {param_offloaded}
    cpu_copy = optimizer.gpu_params_map_cpu_copy[param_offloaded]
    assert cpu_copy.device.type == 'cpu'
    assert cpu_copy.dtype == torch.float32
    assert len(optimizer.cpu_optimizers) == 1
    assert optimizer.sub_optimizers == [*optimizer.cpu_optimizers, optimizer.gpu_optimizer]

    grad_resident = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.bfloat16, device=_device())
    grad_offloaded = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.bfloat16, device=_device()
    )
    param_resident.grad = grad_resident
    param_offloaded.grad = grad_offloaded

    optimizer.step()

    torch.testing.assert_close(
        param_resident.detach(),
        _reference_adamw_step(
            torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()),
            grad_resident,
            torch.bfloat16,
        ),
        rtol=1.6e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        param_offloaded.detach(),
        _reference_adamw_step(
            torch.full((6,), 3.0, dtype=torch.bfloat16, device=_device()),
            grad_offloaded,
            torch.bfloat16,
        ),
        rtol=1.6e-2,
        atol=1e-2,
    )
    # The optimizer state is keyed by the original parameters and tracks the
    # fp32 master parameter each one was updated through.
    assert set(optimizer.state) == {param_resident, param_offloaded}
    for param in (param_resident, param_offloaded):
        assert float(optimizer.state[param]['step']) == 1.0
        assert optimizer.state[param]['master_param'].dtype == torch.float32


def test_hybrid_optimizer_skips_params_without_grads_when_syncing():
    param_with_grad = torch.nn.Parameter(
        torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device())
    )
    param_resident_frozen = torch.nn.Parameter(
        torch.full((4,), 5.0, dtype=torch.bfloat16, device=_device())
    )
    param_offloaded_frozen = torch.nn.Parameter(
        torch.full((6,), 6.0, dtype=torch.bfloat16, device=_device())
    )
    param_offloaded_frozen.is_offloading_candidate = True
    optimizer = _build_hdo(
        [param_with_grad, param_resident_frozen, param_offloaded_frozen], param_update_in_fp32=True
    )
    grad = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.bfloat16, device=_device())
    param_with_grad.grad = grad

    optimizer.step()

    torch.testing.assert_close(
        param_with_grad.detach(),
        _reference_adamw_step(
            torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()), grad, torch.bfloat16
        ),
        rtol=1.6e-2,
        atol=1e-2,
    )
    # Params without grads are marked as not requiring grad and are skipped by
    # both the resident fp32-copy sync and the CPU grad copy sync.
    assert not optimizer.param_to_fp32_param[param_resident_frozen].requires_grad
    assert not optimizer.gpu_params_map_cpu_copy[param_offloaded_frozen].requires_grad
    torch.testing.assert_close(
        param_resident_frozen.detach(),
        torch.full((4,), 5.0, dtype=torch.bfloat16, device=_device()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        param_offloaded_frozen.detach(),
        torch.full((6,), 6.0, dtype=torch.bfloat16, device=_device()),
        rtol=0,
        atol=0,
    )


def test_hybrid_optimizer_state_dict_round_trip_restores_sub_optimizer_states():
    param_resident = torch.nn.Parameter(
        torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device())
    )
    param_offloaded = torch.nn.Parameter(
        torch.full((6,), 3.0, dtype=torch.bfloat16, device=_device())
    )
    param_offloaded.is_offloading_candidate = True
    optimizer = _build_hdo([param_resident, param_offloaded], param_update_in_fp32=True)
    param_resident.grad = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.bfloat16, device=_device())
    param_offloaded.grad = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.bfloat16, device=_device()
    )
    optimizer.step()

    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_moments = {
        param: {
            key: value.detach().clone()
            for key, value in optimizer.state[param].items()
            if isinstance(value, torch.Tensor)
        }
        for param in (param_resident, param_offloaded)
    }

    _perturb_state(optimizer)
    optimizer.load_state_dict(saved_state)

    for param in (param_resident, param_offloaded):
        for key, saved_value in saved_moments[param].items():
            torch.testing.assert_close(optimizer.state[param][key].cpu(), saved_value.cpu())
    # Sub-optimizers were rebuilt from the loaded state: their inner states are
    # the loaded dicts keyed by the new fp32 master parameters.
    inner_params = [
        p
        for sub_optimizer in optimizer.sub_optimizers
        for group in sub_optimizer.param_groups
        for p in group['params']
    ]
    assert len(inner_params) == 2
    for sub_optimizer in optimizer.sub_optimizers:
        for group in sub_optimizer.param_groups:
            for p in group['params']:
                assert p.dtype == torch.float32
                assert sub_optimizer.state[p]['exp_avg'].shape == p.shape


def test_hybrid_optimizer_dummy_step_initializes_state_and_clears_grads():
    param = torch.nn.Parameter(torch.full((4,), 2.0, dtype=torch.bfloat16, device=_device()))
    optimizer = _build_hdo([param], param_update_in_fp32=True)

    torch.manual_seed(1234)
    expected_grad = torch.randn_like(param).float()
    torch.manual_seed(1234)
    optimizer.dummy_step()

    assert param.grad is None
    assert not hasattr(param, 'decoupled_grad')
    assert set(optimizer.state) == {param}
    state = optimizer.state[param]
    assert float(state['step']) == 1.0
    assert state['master_param'].dtype == torch.float32
    torch.testing.assert_close(state['exp_avg'], 0.1 * expected_grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        state['exp_avg_sq'], 0.001 * expected_grad * expected_grad, rtol=1e-4, atol=1e-6
    )
