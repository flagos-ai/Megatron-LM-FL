# Copyright (c) 2026, BAAI. All rights reserved.
"""Tensor and module helpers in megatron.core.utils: wrapping, viewless copies, rounding."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.utils import (
    MakeViewlessTensor,
    WrappedTensor,
    accepts_parameter,
    assert_viewless_tensor,
    is_using_quantization_scales,
    make_viewless_tensor,
    round_up_to_nearest_multiple,
)


@pytest.mark.parametrize(
    ('value', 'multiple', 'expected'),
    [(7, 4, 8), (8, 4, 8), (0, 3, 0), (5, 5, 5)],
    ids=['rounds_up', 'stays_on_exact_multiple', 'zero_stays_zero', 'equal_values'],
)
def test_round_up_to_nearest_multiple_rounds_the_value_up(value, multiple, expected):
    assert round_up_to_nearest_multiple(value, multiple) == expected


@pytest.mark.parametrize(
    ('func', 'name', 'expected'),
    [
        pytest.param(lambda first, second: first, 'second', True, id='named_argument'),
        pytest.param(lambda first, second: first, 'missing', False, id='missing_argument'),
        pytest.param(lambda **kwargs: kwargs, 'anything', True, id='var_keyword'),
        pytest.param(lambda *args: args, 'first', False, id='var_positional_only'),
    ],
)
def test_accepts_parameter_matches_named_and_var_keyword_arguments(func, name, expected):
    assert accepts_parameter(func, name) is expected


@pytest.mark.parametrize(
    ('config', 'expected'),
    [
        (SimpleNamespace(fp8='e4m3'), True),
        (SimpleNamespace(fp4='nvfp4'), True),
        (SimpleNamespace(), False),
    ],
    ids=['fp8', 'fp4', 'neither'],
)
def test_is_using_quantization_scales_reads_the_quantization_modes(config, expected):
    assert bool(is_using_quantization_scales(config)) is expected


def test_wrapped_tensor_unwraps_exactly_once():
    tensor = torch.tensor([1.0, 2.0])
    wrapped = WrappedTensor(tensor)
    assert wrapped.unwrap() is tensor
    with pytest.raises(RuntimeError, match='WrappedTensor has already been unwrapped'):
        wrapped.unwrap()


def test_make_viewless_tensor_detaches_a_view_while_keeping_the_graph():
    base = torch.zeros(4)
    view = base[1:3]
    out = make_viewless_tensor(view, requires_grad=False, keep_graph=True)
    assert out._base is None
    assert torch.equal(out, view)


def test_make_viewless_tensor_copies_a_view_without_the_graph():
    base = torch.zeros(6)
    view = base[2:5]
    out = make_viewless_tensor(view, requires_grad=False, keep_graph=False)
    assert out._base is None
    assert torch.equal(out, view)


def test_make_viewless_tensor_backward_returns_the_gradient_unchanged():
    grad_output = torch.ones(3)
    grad_input, grad_requires = MakeViewlessTensor.backward(None, grad_output)
    assert grad_input is grad_output
    assert grad_requires is None


def test_assert_viewless_tensor_passes_non_tensor_entries_through():
    entries = ['padding', torch.zeros(2)]
    assert assert_viewless_tensor(entries) == entries
    assert assert_viewless_tensor(3.14) == 3.14
