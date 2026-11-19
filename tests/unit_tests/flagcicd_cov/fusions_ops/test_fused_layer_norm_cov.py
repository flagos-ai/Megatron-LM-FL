# Copyright (c) 2026, BAAI. All rights reserved.
"""FusedLayerNorm: parameter reset, persistent-kernel choice by hidden size, apex-missing guard."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_layer_norm
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer import TransformerConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _config(**overrides):
    kwargs = dict(hidden_size=1024, num_layers=2, num_attention_heads=8)
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_fused_layer_norm_resets_weight_to_one_and_bias_to_zero_without_zero_centered_gamma():
    config = _config()

    layer_norm = FusedLayerNorm(config, 64)

    assert layer_norm.persist_layer_norm is False
    assert torch.equal(layer_norm.weight, torch.ones(64, device=layer_norm.weight.device))
    assert torch.equal(layer_norm.bias, torch.zeros(64, device=layer_norm.bias.device))


def test_fused_layer_norm_resets_parameters_to_zero_when_zero_centered_gamma_is_set():
    config = _config(layernorm_zero_centered_gamma=True)

    layer_norm = FusedLayerNorm(config, 64)

    assert layer_norm.zero_centered_gamma is True
    assert torch.equal(layer_norm.weight, torch.zeros(64, device=layer_norm.weight.device))
    assert torch.equal(layer_norm.bias, torch.zeros(64, device=layer_norm.bias.device))


def test_fused_layer_norm_persist_kernel_matches_pytorch_layer_norm_reference():
    torch.manual_seed(1234)
    config = _config(persist_layer_norm=True)
    layer_norm = FusedLayerNorm(config, 1024).to(DEVICE)
    layer_norm.weight.data.copy_(torch.randn(1024, device=DEVICE))
    layer_norm.bias.data.copy_(torch.randn(1024, device=DEVICE))
    x = torch.randn(4, 2, 1024, dtype=torch.float32, device=DEVICE)

    out = layer_norm(x)

    assert layer_norm.persist_layer_norm is True
    expected = F.layer_norm(x, (1024,), layer_norm.weight, layer_norm.bias, layer_norm.eps)
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


def test_fused_layer_norm_persist_kernel_applies_offset_gamma_when_zero_centered():
    torch.manual_seed(2345)
    config = _config(layernorm_zero_centered_gamma=True, persist_layer_norm=True)
    layer_norm = FusedLayerNorm(config, 1024).to(DEVICE)
    layer_norm.weight.data.copy_(torch.randn(1024, device=DEVICE))
    layer_norm.bias.data.copy_(torch.randn(1024, device=DEVICE))
    x = torch.randn(4, 2, 1024, dtype=torch.float32, device=DEVICE)

    out = layer_norm(x)

    expected = F.layer_norm(x, (1024,), layer_norm.weight + 1, layer_norm.bias, layer_norm.eps)
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)


def test_fused_layer_norm_falls_back_to_non_persist_kernel_for_unsupported_hidden_size():
    torch.manual_seed(3456)
    config = _config(persist_layer_norm=True)
    layer_norm = FusedLayerNorm(config, 64).to(DEVICE)
    layer_norm.weight.data.copy_(torch.randn(64, device=DEVICE))
    layer_norm.bias.data.copy_(torch.randn(64, device=DEVICE))
    x = torch.randn(4, 2, 64, dtype=torch.float32, device=DEVICE, requires_grad=True)

    out = layer_norm(x)
    out.backward(torch.ones_like(out))

    assert layer_norm.persist_layer_norm is False
    x_ref = x.detach().requires_grad_(True)
    w_ref = layer_norm.weight.detach().requires_grad_(True)
    b_ref = layer_norm.bias.detach().requires_grad_(True)
    expected = F.layer_norm(x_ref, (64,), w_ref, b_ref, layer_norm.eps)
    expected.backward(torch.ones_like(expected))
    torch.testing.assert_close(out, expected.detach(), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(layer_norm.weight.grad, w_ref.grad, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(layer_norm.bias.grad, b_ref.grad, rtol=1.0e-5, atol=1.0e-5)


def test_fused_layer_norm_raises_value_error_when_fused_kernel_is_unavailable(monkeypatch):
    config = _config()
    monkeypatch.setattr(fused_layer_norm, "HAVE_FUSED_LAYER_NORM", False)

    with pytest.raises(ValueError, match="Apex must be installed to use FusedLayerNorm"):
        FusedLayerNorm(config, 64)
