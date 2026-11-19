# Copyright (c) 2026, BAAI. All rights reserved.
"""Torch-native norm wrappers and the L2Norm module."""

import pytest
import torch

from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
from megatron.core.transformer.torch_norm import L2Norm, WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    defaults = dict(
        num_layers=1, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_torch_layer_norm_module_aliases_wrapped_torch_norm():
    assert WrappedTorchLayerNorm is WrappedTorchNorm


def test_wrapped_torch_norm_builds_torch_layer_norm_for_layernorm_config():
    norm = WrappedTorchNorm(_config(), 32, eps=1e-4)

    assert isinstance(norm, torch.nn.LayerNorm)
    assert norm.normalized_shape == (32,)
    assert norm.eps == 1e-4


def test_wrapped_torch_norm_layernorm_forward_matches_functional_reference():
    norm = WrappedTorchNorm(_config(), 16, eps=1e-5).double()
    x = torch.randn(4, 3, 16, dtype=torch.float64)

    torch.testing.assert_close(
        norm(x), torch.nn.functional.layer_norm(x, (16,), norm.weight, norm.bias, 1e-5)
    )


def test_wrapped_torch_norm_rejects_unknown_normalization():
    config = _config()
    config.normalization = "FrobeniusNorm"

    with pytest.raises(Exception, match="Only LayerNorm, RMSNorm and L2Norm"):
        WrappedTorchNorm(config, 32)


def test_l2_norm_forward_matches_manual_reference():
    norm = L2Norm(16, eps=1e-6)
    x = torch.randn(5, 16)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(norm(x), expected)
    assert norm(x).dtype == torch.float32
