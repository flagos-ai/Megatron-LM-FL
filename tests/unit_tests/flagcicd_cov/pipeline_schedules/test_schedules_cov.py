# Copyright (c) 2026, BAAI. All rights reserved.
"""Output-tensor pseudo-freeing, direct C++ autograd backward, and schedule selection."""

import pytest
import torch

import megatron.core.pipeline_parallel.schedules as schedule
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_get_forward_backward_func_returns_dualpipev_schedule_when_enabled(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        create_dualpipev_parallel_size=True,
    )
    from megatron.plugin.dualpipev.dualpipev_schedules import (
        forward_backward_pipelining_with_dualpipev,
    )

    assert schedule.get_forward_backward_func() == forward_backward_pipelining_with_dualpipev


def test_deallocate_output_tensor_frees_nested_dict_and_list_values():
    base = torch.randn(4, requires_grad=True)
    outputs = {'vision': base.sum(), 'language': [base.sum(), base.sum()]}

    schedule.deallocate_output_tensor(outputs, True)

    freed = [outputs['vision'], *outputs['language']]
    for tensor in freed:
        assert tensor.data.shape == (1,)
    for tensor in freed:
        schedule.custom_backward(tensor, None)
    assert torch.equal(base.grad, torch.full_like(base, 3.0))


def test_deallocate_output_tensor_frees_tensor_data_when_enabled():
    out = torch.ones(2, 3, dtype=torch.float64)

    schedule.deallocate_output_tensor(out, True)

    assert out.data.shape == (1,)
    assert out.data.dtype is torch.float64
    assert out.data.device.type == 'cpu'


def test_deallocate_output_tensor_leaves_tensor_untouched_when_disabled():
    out = torch.ones(2, 3)
    data_ptr = out.data_ptr()

    schedule.deallocate_output_tensor(out, False)

    assert torch.equal(out, torch.ones(2, 3))
    assert out.shape == (2, 3)
    assert out.data_ptr() == data_ptr


def test_deallocate_output_tensor_rejects_non_tensor_leaf():
    with pytest.raises(AssertionError, match='expected Tensor, found'):
        schedule.deallocate_output_tensor({'bad': 3}, True)


def test_deallocate_output_tensor_rejects_view_of_another_tensor():
    base = torch.ones(4)
    view = base[:2]
    with pytest.raises(AssertionError, match='counter-productive to free a view'):
        schedule.deallocate_output_tensor(view, True)


def test_custom_backward_uses_implicit_ones_grad_for_scalar_output():
    x = torch.randn(3, requires_grad=True)
    out = (x * 3).sum()

    schedule.custom_backward(out, None)

    assert torch.equal(x.grad, torch.full_like(x, 3.0))


def test_custom_backward_applies_explicit_grad_after_pseudo_free():
    x = torch.randn(3, requires_grad=True)
    out = (x * 3).sum()
    out.data = torch.empty((1,))

    schedule.custom_backward(out, torch.tensor([2.0]))

    assert torch.equal(x.grad, torch.full_like(x, 6.0))


def test_custom_backward_rejects_non_scalar_output():
    x = torch.randn(3, requires_grad=True)
    out = x * 3
    with pytest.raises(AssertionError, match="output should be pseudo-'freed' in schedule"):
        schedule.custom_backward(out, None)


def test_get_tensor_device_reads_first_value_of_dict():
    tensor = torch.ones(2)
    assert schedule.get_tensor_device({'a': tensor, 'b': tensor}) == tensor.device
    assert schedule.get_tensor_device(tensor) == tensor.device


@pytest.mark.parametrize(
    'first_val_step,forward_only,cond,expected',
    [
        (True, True, True, True),
        (True, True, False, False),
        (False, True, True, False),
        (True, False, True, True),
        (None, True, True, True),
    ],
    ids=['and-true', 'and-false', 'first-false', 'not-forward-only', 'no-first-val'],
)
def test_check_first_val_step_combines_first_step_with_condition(
    first_val_step, forward_only, cond, expected
):
    assert schedule.check_first_val_step(first_val_step, forward_only, cond) is expected
