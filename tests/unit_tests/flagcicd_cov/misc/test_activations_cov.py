# Copyright (c) 2026, BAAI. All rights reserved.
"""Squared ReLU, quick GELU, and fast GELU activation values on CPU tensors."""

import pytest
import torch

from megatron.core.activations import fast_gelu, quick_gelu, squared_relu


def test_squared_relu_squares_the_positive_part():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])

    out = squared_relu(x)

    assert torch.equal(out, torch.tensor([0.0, 0.0, 0.0, 0.25, 9.0]))


def test_quick_gelu_scales_input_by_its_sigmoid_gate():
    x = torch.tensor([-1.0, 0.0, 2.0])

    out = quick_gelu(x)

    assert out.shape == x.shape
    assert out[1].item() == pytest.approx(0.0, abs=1e-7)
    assert out[0].item() == pytest.approx(-0.15420423448085785, rel=1e-6)
    assert out[2].item() == pytest.approx(1.935658574104309, rel=1e-6)


def test_fast_gelu_matches_the_tanh_approximation_of_gelu():
    x = torch.tensor([-1.0, 0.0, 2.0])

    out = fast_gelu(x)

    assert out.shape == x.shape
    assert out[1].item() == pytest.approx(0.0, abs=1e-7)
    assert out[0].item() == pytest.approx(-0.15880799293518066, rel=1e-6)
    assert out[2].item() == pytest.approx(1.9545977115631104, rel=1e-6)
