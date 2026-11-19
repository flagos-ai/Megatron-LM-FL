# Copyright (c) 2026, BAAI. All rights reserved.
"""Dataset utils: blend-list parsing into prefixes and weights, weight normalisation to unit sum."""

import numpy
import pytest

from megatron.core.datasets.utils import get_blend_from_list, normalize


def test_get_blend_from_list_returns_none_when_the_blend_is_none():
    assert get_blend_from_list(None) is None


def test_get_blend_from_list_treats_odd_length_lists_as_prefixes_only():
    prefixes, weights = get_blend_from_list([" first ", "second", "third"])

    assert prefixes == ["first", "second", "third"]
    assert weights is None


def test_get_blend_from_list_parses_a_numeric_weight_before_each_prefix():
    prefixes, weights = get_blend_from_list(["2", " first ", "3.5", "second"])

    assert prefixes == ["first", "second"]
    assert weights == [2.0, 3.5]


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        pytest.param([1, 3], [0.25, 0.75], id="integers"),
        pytest.param([30, 70], [0.3, 0.7], id="tens"),
        pytest.param([1, 1, 2], [0.25, 0.25, 0.5], id="three-weights"),
    ],
)
def test_normalize_scales_weights_to_unit_sum(weights, expected):
    numpy.testing.assert_allclose(normalize(weights), expected)
