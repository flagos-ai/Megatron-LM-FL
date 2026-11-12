# Copyright (c) 2026, BAAI. All rights reserved.
"""Experimental attention variant specs: dispatch, linear/moe layer patterns, block spec layout, MLP builders."""

import importlib
import importlib.util
import sys

import pytest

import megatron.core.models.gpt.experimental_attention_variant_module_specs as exp_specs
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _get_dense_mlp_module_spec,
    _get_moe_module_spec,
    _get_self_attention_module_spec,
    get_experimental_attention_variant_module_spec,
    get_linear_attention_pattern,
    get_moe_layer_pattern,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    values = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.fixture
def restore_exp_specs():
    yield
    importlib.reload(exp_specs)


def test_experimental_specs_report_te_unavailable_when_te_import_fails(
    monkeypatch, restore_exp_specs
):
    with monkeypatch.context() as blocker:
        blocker.setitem(sys.modules, 'transformer_engine', None)
        importlib.reload(exp_specs)
        assert exp_specs.HAVE_TE is False

    importlib.reload(exp_specs)
    assert exp_specs.HAVE_TE is (importlib.util.find_spec('transformer_engine') is not None)


def test_experimental_variant_dispatch_builds_gated_delta_net_spec():
    config = _config(experimental_attention_variant='gated_delta_net', linear_attention_freq=2)

    spec = get_experimental_attention_variant_module_spec(config=config)

    assert spec.module is GatedDeltaNet
    assert spec.metainfo['fuse_input_layernorm'] is True


def test_experimental_variant_rejects_unknown_name():
    config = _config(experimental_attention_variant='banana')

    with pytest.raises(ValueError, match='Invalid experimental attention variant'):
        get_experimental_attention_variant_module_spec(config=config)


def test_linear_attention_pattern_marks_every_nth_layer_sdpa():
    config = _config(linear_attention_freq=2)

    assert get_linear_attention_pattern(config) == [1, 0]


def test_linear_attention_pattern_uses_list_directly():
    config = _config(linear_attention_freq=[0, 1])

    assert get_linear_attention_pattern(config) == [0, 1]


def test_linear_attention_pattern_is_all_sdpa_when_freq_none_and_variant_none():
    config = _config()

    assert get_linear_attention_pattern(config) == [0, 0]


def test_linear_attention_pattern_requires_freq_for_linear_variant():
    config = _config()
    config.experimental_attention_variant = 'gated_delta_net'

    with pytest.raises(ValueError, match='but linear_attention_freq is None'):
        get_linear_attention_pattern(config)


def test_linear_attention_pattern_rejects_non_int_list_freq():
    config = _config(linear_attention_freq=1.5)

    with pytest.raises(ValueError, match='Invalid linear_attention_freq'):
        get_linear_attention_pattern(config)


def test_moe_layer_pattern_marks_every_nth_layer_expert():
    config = _config(moe_layer_freq=2)

    assert get_moe_layer_pattern(config) == [1, 0]


def test_moe_layer_pattern_rejects_non_int_list_freq():
    config = _config(moe_layer_freq=1.5)

    with pytest.raises(ValueError, match='Invalid moe_layer_freq'):
        get_moe_layer_pattern(config)


def test_block_spec_builds_standard_attention_when_variant_is_none(model_parallel):
    model_parallel()
    config = _config()

    block_spec = get_transformer_block_with_experimental_attention_variant_spec(config=config)

    assert len(block_spec.layer_specs) == 2
    assert block_spec.layer_specs[0].submodules.self_attention.module is SelfAttention
    assert block_spec.layer_specs[1].submodules.self_attention.module is SelfAttention


def test_block_spec_mixes_experimental_and_standard_layers(model_parallel):
    model_parallel()
    config = _config(experimental_attention_variant='gated_delta_net', linear_attention_freq=2)

    block_spec = get_transformer_block_with_experimental_attention_variant_spec(config=config)

    assert block_spec.layer_specs[0].submodules.self_attention.module is GatedDeltaNet
    assert block_spec.layer_specs[1].submodules.self_attention.module is SelfAttention


def test_block_spec_slices_layers_by_pipeline_layout(model_parallel):
    model_parallel()
    layout = PipelineParallelLayerLayout(
        [[LayerType.embedding, LayerType.decoder, LayerType.decoder, LayerType.loss]], 1
    )
    config = _config(pipeline_model_parallel_layout=layout)

    block_spec = get_transformer_block_with_experimental_attention_variant_spec(config=config)

    assert len(block_spec.layer_specs) == 2
    assert block_spec.layer_specs[0].submodules.self_attention.module is SelfAttention


def test_self_attention_module_spec_fuses_input_layernorm_for_te_backend():
    config = _config()

    attn_spec = _get_self_attention_module_spec(config=config)

    assert attn_spec.module is SelfAttention
    assert attn_spec.metainfo['fuse_input_layernorm'] is True


def test_self_attention_module_spec_keeps_input_layernorm_for_mla():
    config = _config(multi_latent_attention=True)

    attn_spec = _get_self_attention_module_spec(config=config)

    assert attn_spec.metainfo['fuse_input_layernorm'] is False


def test_dense_mlp_module_spec_reports_fused_prenorm():
    builder, fuse_prenorm = _get_dense_mlp_module_spec(config=_config())

    backend = TESpecProvider()
    submodules = builder.keywords['submodules']
    assert builder.func.__self__ is MLP
    assert submodules.linear_fc1 is backend.column_parallel_layer_norm_linear()
    assert submodules.linear_fc2 is backend.row_parallel_linear()
    assert submodules.activation_func is None
    assert fuse_prenorm is True


def test_moe_module_spec_reports_unfused_prenorm():
    builder, fuse_prenorm = _get_moe_module_spec(config=_config(num_moe_experts=4))

    backend = TESpecProvider()
    submodules = builder.keywords['submodules']
    assert builder.func is MoELayer
    assert submodules.experts.func is SequentialMLP
    assert submodules.experts.keywords['submodules'].linear_fc1 is backend.column_parallel_linear()
    assert submodules.shared_experts.func is SharedExpertMLP
    assert fuse_prenorm is False
