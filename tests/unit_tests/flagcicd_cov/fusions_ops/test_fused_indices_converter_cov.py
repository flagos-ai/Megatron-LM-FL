# Copyright (c) 2026, BAAI. All rights reserved.
"""MoE indices to multihot: scattering of indices and probabilities, gradients back into probs."""

import importlib.util
import sys
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.fusions import fused_indices_converter
from megatron.core.utils import null_decorator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_OF_LOCAL_EXPERTS = 6


@pytest.fixture
def indices():
    # Built per test: a module-level CUDA tensor is created on cuda:0 at import, before each
    # torchrun worker binds its own device.
    return torch.tensor(
        [[0, 2, -1], [5, -1, -1], [3, 1, 4], [-1, -1, -1], [2, 5, 0]],
        dtype=torch.int32,
        device=DEVICE,
    )


def _multihot_reference(indices, probs, num_experts):
    """Scatter ``probs`` into a [tokens, experts] multihot layout; differentiable in probs."""
    valid = indices >= 0
    rows = torch.arange(indices.shape[0], device=indices.device).repeat_interleave(valid.sum(dim=1))
    multihot = torch.zeros(indices.shape[0], num_experts, dtype=torch.bool, device=indices.device)
    probs_multihot = torch.zeros(
        indices.shape[0], num_experts, dtype=probs.dtype, device=probs.device
    ).index_put((rows, indices[valid]), probs[valid])
    multihot[rows, indices[valid]] = True
    return multihot, probs_multihot


def test_fused_indices_to_multihot_scatters_indices_and_probs_into_multihot(indices):
    torch.manual_seed(1234)
    probs = torch.rand(*indices.shape, dtype=torch.float32, device=DEVICE)

    multihot, probs_in_multihot = fused_indices_converter.fused_indices_to_multihot(
        indices, probs, NUM_OF_LOCAL_EXPERTS
    )

    expected_multihot, expected_probs = _multihot_reference(indices, probs, NUM_OF_LOCAL_EXPERTS)
    assert multihot.dtype == torch.bool
    assert probs_in_multihot.dtype == torch.float32
    assert multihot.shape == probs_in_multihot.shape == (5, NUM_OF_LOCAL_EXPERTS)
    assert torch.equal(multihot, expected_multihot)
    assert torch.equal(probs_in_multihot, expected_probs)
    assert not multihot[3].any()


def test_fused_indices_to_multihot_backward_scatters_grads_back_to_indices(indices):
    torch.manual_seed(2345)
    probs = torch.rand(*indices.shape, dtype=torch.float32, device=DEVICE)
    probs_fused = probs.detach().clone().requires_grad_(True)

    _, probs_in_multihot = fused_indices_converter.fused_indices_to_multihot(
        indices, probs_fused, NUM_OF_LOCAL_EXPERTS
    )
    loss = (probs_in_multihot**2).sum()
    loss.backward()

    probs_ref = probs.detach().clone().requires_grad_(True)
    _, ref_multihot = _multihot_reference(indices, probs_ref, NUM_OF_LOCAL_EXPERTS)
    ref_loss = (ref_multihot**2).sum()
    ref_loss.backward()

    assert probs_fused.grad.shape == indices.shape
    assert torch.equal(probs_fused.grad, probs_ref.grad)
    assert torch.equal(probs_fused.grad[3], torch.zeros_like(probs_fused.grad[3]))


def test_fused_indices_converter_module_falls_back_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    spec = importlib.util.spec_from_file_location(
        "fused_indices_converter_no_triton", fused_indices_converter.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.HAVE_TRITON is False
    assert module.triton.jit is null_decorator
    assert module.triton.heuristics is null_decorator
    assert isinstance(module.tl, MagicMock)
