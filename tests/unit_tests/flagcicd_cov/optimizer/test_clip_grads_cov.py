# Copyright (c) 2026, BAAI. All rights reserved.
"""Gradient clipping helpers: L2/inf/p-norms of single and multiple tensors reduced over the world group, in-place clipping by total norm (plain and decoupled grads), zero counting over each stats group."""

import pytest
import torch

from megatron.core.optimizer.clip_grads import (
    clip_grad_by_total_norm_fp32,
    count_zeros_fp32,
    get_grad_norm_fp32,
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


def _world_size():
    return torch.distributed.get_world_size()


def test_get_grad_norm_fp32_l2_norm_of_single_tensor_matches_reference(model_parallel):
    model_parallel()
    grad = torch.tensor([[3.0, 4.0]], device=_device())

    total_norm = get_grad_norm_fp32(grad, norm_type=2)

    expected = (25.0 * _world_size()) ** 0.5
    assert total_norm == pytest.approx(expected, rel=1e-5)


def test_get_grad_norm_fp32_l2_norm_of_grad_list_matches_reference(model_parallel):
    model_parallel()
    grads = [torch.tensor([3.0, 4.0], device=_device()), torch.full((2, 2), 5.0, device=_device())]

    total_norm = get_grad_norm_fp32(grads, norm_type=2)

    expected = ((25.0 + 100.0) * _world_size()) ** 0.5
    assert total_norm == pytest.approx(expected, rel=1e-5)


def test_get_grad_norm_fp32_inf_norm_matches_reference(model_parallel):
    model_parallel()
    grads = [torch.tensor([[1.0, -4.0], [2.0, 4.0]], device=_device())]

    total_norm = get_grad_norm_fp32(grads, norm_type=float('inf'))

    assert total_norm == 4.0


def test_get_grad_norm_fp32_third_norm_matches_reference(model_parallel):
    model_parallel()
    grads = [torch.tensor([1.0, 2.0], device=_device())]

    total_norm = get_grad_norm_fp32(grads, norm_type=3)

    expected = (9.0 * _world_size()) ** (1.0 / 3.0)
    assert total_norm == pytest.approx(expected, rel=1e-5)


def test_get_grad_norm_fp32_returns_zero_for_empty_grad_list(model_parallel):
    model_parallel()

    assert get_grad_norm_fp32([], norm_type=2) == 0.0


def test_clip_grad_by_total_norm_fp32_scales_grads_down_to_max_norm(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    param.grad = torch.full((4,), 2.0, device=_device())
    param_without_grad = torch.nn.Parameter(torch.ones(4, device=_device()))

    clip_grad_by_total_norm_fp32([param, param_without_grad], max_norm=1.0, total_norm=100.0)

    clip_coeff = 1.0 / (100.0 + 1.0e-6)
    torch.testing.assert_close(
        param.grad, torch.full((4,), 2.0 * clip_coeff, device=_device()), rtol=1e-5, atol=1e-7
    )


def test_clip_grad_by_total_norm_fp32_keeps_grads_below_max_norm(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    param.grad = torch.full((4,), 2.0, device=_device())

    clip_grad_by_total_norm_fp32([param], max_norm=1.0, total_norm=0.5)

    torch.testing.assert_close(param.grad, torch.full((4,), 2.0, device=_device()))


def test_clip_grad_by_total_norm_fp32_clips_decoupled_grad(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    param.decoupled_grad = torch.full((4,), 4.0, device=_device())

    clip_grad_by_total_norm_fp32([param], max_norm=1.0, total_norm=100.0, use_decoupled_grad=True)

    clip_coeff = 1.0 / (100.0 + 1.0e-6)
    torch.testing.assert_close(
        param.decoupled_grad,
        torch.full((4,), 4.0 * clip_coeff, device=_device()),
        rtol=1e-5,
        atol=1e-7,
    )


def test_count_zeros_fp32_counts_zero_grad_entries_over_world(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    param.grad = torch.tensor([0.0, 1.0, 0.0, 2.0], device=_device())
    param_without_grad = torch.nn.Parameter(torch.ones(4, device=_device()))
    shared_param = torch.nn.Parameter(torch.ones(4, device=_device()))
    shared_param.shared = True
    shared_param.grad = torch.zeros(4, device=_device())

    num_zeros = count_zeros_fp32(
        [param, param_without_grad, shared_param],
        grad_stats_parallel_group=torch.distributed.GroupMember.WORLD,
    )

    assert num_zeros == 2 * _world_size()


def test_count_zeros_fp32_counts_decoupled_grad_zeros(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    param.decoupled_grad = torch.tensor([0.0, 0.0, 1.0, 2.0], device=_device())

    num_zeros = count_zeros_fp32(
        [param],
        grad_stats_parallel_group=torch.distributed.GroupMember.WORLD,
        use_decoupled_grad=True,
    )

    assert num_zeros == 2 * _world_size()


def test_count_zeros_fp32_reduces_over_each_group_in_list(model_parallel):
    model_parallel()
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    param.grad = torch.tensor([0.0, 1.0, 0.0, 2.0], device=_device())

    num_zeros = count_zeros_fp32(
        [param], grad_stats_parallel_group=[torch.distributed.GroupMember.WORLD]
    )

    assert num_zeros == 2 * _world_size()
