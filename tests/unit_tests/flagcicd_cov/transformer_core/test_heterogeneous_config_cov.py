# Copyright (c) 2026, BAAI. All rights reserved.
"""Heterogeneous per-layer config parsing from HF-style block dictionaries."""

import json

import pytest

from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    AttentionConfig,
    HeterogeneousTransformerConfig,
    MLPConfig,
)


def _block(attention=None, ffn=None):
    return {
        "attention": {
            "no_op": False,
            "replace_with_linear": False,
            "n_heads_in_group": 2,
            **(attention or {}),
        },
        "ffn": {"no_op": False, "replace_with_linear": False, "ffn_mult": None, **(ffn or {})},
    }


def _hetero_config(tmp_path, blocks, **overrides):
    config_dict = {"block_configs": blocks}
    path = tmp_path / "hetero_config.json"
    path.write_text(json.dumps(config_dict))
    defaults = dict(
        num_layers=len(blocks),
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        heterogeneous_layers_config_path=str(path),
    )
    defaults.update(overrides)
    return HeterogeneousTransformerConfig(**defaults)


@pytest.mark.parametrize(("n_heads_in_group", "expected_groups"), [(1, 4), (2, 2), (4, 1)])
def test_attention_config_derives_num_query_groups_from_heads_in_group(
    n_heads_in_group, expected_groups
):
    config = AttentionConfig.build_config_from_dict(
        _block(attention={"n_heads_in_group": n_heads_in_group}), num_attention_heads=4
    )

    assert config.num_query_groups == expected_groups
    assert config.no_op is False
    assert config.replace_with_linear is False


def test_attention_config_leaves_num_query_groups_none_for_null_group(tmp_path):
    config = _hetero_config(tmp_path, [_block(attention={"n_heads_in_group": None})])

    assert config.per_block_parameters[0].attention.num_query_groups is None
    assert config.get_config_for_layer(1).num_query_groups == 4


def test_attention_config_keeps_explicit_num_query_groups():
    config = AttentionConfig.build_config_from_dict(
        _block(attention={"num_query_groups": 3}), num_attention_heads=4
    )

    assert config.num_query_groups == 3


def test_attention_config_rejects_non_divisible_heads_in_group():
    with pytest.raises(ValueError, match="must be a multiple of n_heads_in_group"):
        AttentionConfig.build_config_from_dict(
            _block(attention={"n_heads_in_group": 3}), num_attention_heads=4
        )


@pytest.mark.parametrize(
    ("ffn_mult", "hidden_size", "expected_ffn"),
    [(2.625, 64, 256), (2.625, 384, 768), (2.0, 192, 256)],
)
def test_mlp_config_derives_ffn_hidden_size_from_ffn_mult(ffn_mult, hidden_size, expected_ffn):
    config = MLPConfig.build_config_from_dict(
        _block(ffn={"ffn_mult": ffn_mult}), hidden_size=hidden_size
    )

    assert config.ffn_hidden_size == expected_ffn


def test_mlp_config_leaves_ffn_hidden_size_none_for_null_mult(tmp_path):
    config = _hetero_config(tmp_path, [_block(ffn={"ffn_mult": None})])

    assert config.per_block_parameters[0].mlp.ffn_hidden_size is None
    assert config.get_config_for_layer(1).ffn_hidden_size == 256


def test_mlp_config_keeps_explicit_ffn_hidden_size():
    config = MLPConfig.build_config_from_dict(_block(ffn={"ffn_hidden_size": 512}), hidden_size=64)

    assert config.ffn_hidden_size == 512


def test_mlp_config_reads_mlp_key_when_ffn_missing():
    block = {"attention": {"n_heads_in_group": 2}, "mlp": {"ffn_mult": 2.625}}

    config = MLPConfig.build_config_from_dict(block, hidden_size=64)

    assert config.ffn_hidden_size == 256


@pytest.mark.parametrize(("n", "expected"), [(0, 0), (256, 256), (112, 256), (513, 768)])
def test_find_multiple_rounds_up_to_multiple_of_k(n, expected):
    assert MLPConfig.find_multiple(n, 256) == expected


def test_get_config_for_layer_invalid_layer_number_raises(tmp_path):
    config = _hetero_config(tmp_path, [_block()])

    with pytest.raises(ValueError, match="Invalid layer number: 3. Should be in range"):
        config.get_config_for_layer(3)


def test_hetero_config_overrides_main_config_per_layer(tmp_path):
    blocks = [
        _block(attention={"n_heads_in_group": 2}, ffn={"ffn_mult": 2.625}),
        _block(attention={"n_heads_in_group": 1}, ffn={"ffn_hidden_size": 1024}),
    ]
    config = _hetero_config(tmp_path, blocks)

    assert len(config.per_block_parameters) == 2

    layer_one = config.get_config_for_layer(1)
    assert layer_one.num_query_groups == 2
    assert layer_one.ffn_hidden_size == 256
    assert layer_one.hidden_size == 64

    layer_two = config.get_config_for_layer(2)
    assert layer_two.num_query_groups == 4
    assert layer_two.ffn_hidden_size == 1024


def test_hetero_config_decodes_block_parameters_from_encoded_json_when_path_missing(tmp_path):
    config = _hetero_config(tmp_path, [_block()])
    encoded = config.heterogeneous_layers_config_encoded_json

    reloaded = HeterogeneousTransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        heterogeneous_layers_config_path="/nonexistent/hetero.json",
        heterogeneous_layers_config_encoded_json=encoded,
    )

    assert len(reloaded.per_block_parameters) == 1
    assert reloaded.get_config_for_layer(1).num_query_groups == 2
