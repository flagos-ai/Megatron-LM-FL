# Copyright (c) 2026, BAAI. All rights reserved.
"""YarnRotaryEmbedding helpers: ramp mask, mscale, concentration factor."""

import math

import torch
from pytest import approx

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
    _yarn_get_concentration_factor,
    _yarn_get_concentration_factor_from_config,
    _yarn_get_mscale,
    _yarn_linear_ramp_mask,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def test_yarn_get_emb_moves_cpu_initialized_freqs_off_cpu_and_returns_mscale():
    yarn = YarnRotaryEmbedding(
        kv_channels=16,
        rotary_percent=1.0,
        use_cpu_initialization=True,
        scaling_factor=2.0,
        original_max_position_embeddings=8,
    )

    emb, mscale = yarn.get_emb(4)

    assert yarn.inv_freq_extra.device.type != "cpu"
    assert yarn.inv_freq_inter.device.type != "cpu"
    assert emb.shape == (4, 1, 1, 16)
    assert mscale == _yarn_get_mscale(2.0)


def test_yarn_linear_ramp_mask_clamps_degenerate_min_max_range():
    ramp = _yarn_linear_ramp_mask(min=3.0, max=3.0, dim=5, device=torch.device("cpu"))

    expected = torch.clamp((torch.arange(5, dtype=torch.float32) - 3.0) / 0.001, 0, 1)
    torch.testing.assert_close(ramp, expected)


def test_yarn_get_mscale_returns_one_for_unit_scale_and_scales_logarithmically():
    assert _yarn_get_mscale(1.0) == 1.0
    assert _yarn_get_mscale(1.0, mscale=2.0) == 1.0
    assert _yarn_get_mscale(4.0, mscale=1.0) == approx(0.1 * math.log(4.0) + 1.0)


def test_yarn_get_concentration_factor_falls_back_to_mscale_when_mscales_missing():
    assert _yarn_get_concentration_factor(4.0, None, None) == _yarn_get_mscale(4.0)
    assert _yarn_get_concentration_factor(4.0, 1.0, 0.5) == approx(
        _yarn_get_mscale(4.0, 1.0) / _yarn_get_mscale(4.0, 0.5)
    )


def test_yarn_concentration_factor_from_config_uses_yarn_attrs_when_present():
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    assert _yarn_get_concentration_factor_from_config(config) == 1.0

    config.yarn_rotary_scaling_factor = 4.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 0.5
    assert _yarn_get_concentration_factor_from_config(config) == approx(
        _yarn_get_mscale(4.0, 1.0) / _yarn_get_mscale(4.0, 0.5)
    )
