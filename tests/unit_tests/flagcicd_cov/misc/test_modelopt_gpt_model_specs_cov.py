# Copyright (c) 2026, BAAI. All rights reserved.
"""get_gpt_modelopt_spec construction for dense, MoE, MLA, and real-quantized GPT configurations."""

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.post_training.modelopt.layers import (
    BlockwiseFP8WeightTransformerLayer,
    FP8WeightTransformerLayer,
    Linear,
    Norm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


@pytest.fixture
def first_pipeline_rank(monkeypatch):
    """Spec construction reads the pipeline rank from parallel_state; pin it to rank zero."""
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda group=None: 0)


def _make_config(**overrides):
    arguments = dict(num_layers=2, hidden_size=16, num_attention_heads=2)
    arguments.update(overrides)
    return TransformerConfig(**arguments)


def _make_moe_config(**overrides):
    arguments = dict(
        num_layers=4,
        hidden_size=16,
        num_attention_heads=2,
        num_moe_experts=2,
        moe_ffn_hidden_size=32,
    )
    arguments.update(overrides)
    return TransformerConfig(**arguments)


def test_dense_spec_builds_local_transformer_layers_with_causal_mask(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_config())

    assert len(spec.layer_specs) == 2
    layer = spec.layer_specs[0]
    assert layer.module is TransformerLayer
    assert layer.submodules.self_attention.module is SelfAttention
    assert layer.submodules.self_attention.params == {'attn_mask_type': AttnMaskType.causal}
    assert layer.submodules.self_attention.submodules.q_layernorm is IdentityOp
    assert layer.submodules.mlp.keywords['submodules'].linear_fc1 is ColumnParallelLinear
    assert spec.layer_norm is Norm


def test_qk_l2_norm_flag_selects_l2_norm_for_q_and_k(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_config(qk_l2_norm=True))

    attention = spec.layer_specs[0].submodules.self_attention.submodules
    assert attention.q_layernorm is L2Norm
    assert attention.k_layernorm is L2Norm


def test_qk_layernorm_config_selects_tenorm_when_l2_norm_is_disabled(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_config(qk_layernorm=True))

    attention = spec.layer_specs[0].submodules.self_attention.submodules
    assert attention.q_layernorm is Norm
    assert attention.k_layernorm is Norm


def test_arbitrary_attention_mask_flag_switches_the_mask_type(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_config(), use_arbitrary_attention_mask=True)

    params = spec.layer_specs[0].submodules.self_attention.params
    assert params == {'attn_mask_type': AttnMaskType.arbitrary}


def test_remap_te_layernorm_adds_fused_key_maps_for_dense_models(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_config(), remap_te_layernorm=True)

    assert spec.layer_specs[0].submodules.sharded_state_dict_keys_map == {
        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
    }


@pytest.mark.parametrize(
    ('real_quant_cfg', 'layer_class'),
    [
        ('None', TransformerLayer),
        ('fp8_real_quant', FP8WeightTransformerLayer),
        ('fp8_blockwise_real_quant', BlockwiseFP8WeightTransformerLayer),
    ],
    ids=['none', 'fp8', 'fp8_blockwise'],
)
def test_real_quant_cfg_selects_the_transformer_layer_class(
    first_pipeline_rank, real_quant_cfg, layer_class
):
    spec = get_gpt_modelopt_spec(_make_config(), real_quant_cfg=real_quant_cfg)

    assert spec.layer_specs[0].module is layer_class


def test_real_quant_cfg_rejects_unknown_values(first_pipeline_rank):
    with pytest.raises(ValueError, match="RealQuantTransformerLayer does not support nope"):
        get_gpt_modelopt_spec(_make_config(), real_quant_cfg='nope')


def test_mla_spec_uses_mla_attention_with_local_projections(first_pipeline_rank):
    config = MLATransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, qk_layernorm=True
    )

    spec = get_gpt_modelopt_spec(config)

    attention = spec.layer_specs[0].submodules.self_attention
    assert attention.module is MLASelfAttention
    assert attention.submodules.linear_q_down_proj is Linear
    assert attention.submodules.q_layernorm is Norm
    assert attention.submodules.core_attention is TEDotProductAttention


def test_mla_remap_te_layernorm_uses_the_qk_layernorm_key_map(first_pipeline_rank):
    config = MLATransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, qk_layernorm=True
    )

    spec = get_gpt_modelopt_spec(config, remap_te_layernorm=True)

    assert spec.layer_specs[0].submodules.sharded_state_dict_keys_map == {
        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        'self_attention.q_layernorm.': 'self_attention.linear_q_up_proj.layer_norm_',
        'self_attention.kv_layernorm.': 'self_attention.linear_kv_up_proj.layer_norm_',
    }


def test_moe_spec_interleaves_expert_layers_at_integer_frequency(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_moe_config(moe_layer_freq=2))

    assert len(spec.layer_specs) == 4
    assert spec.layer_specs[0].submodules.mlp.func is MoELayer
    assert spec.layer_specs[1].submodules.mlp.func == MLP.as_mlp_submodule
    assert spec.layer_specs[2].submodules.mlp.func is MoELayer
    assert spec.layer_specs[3].submodules.mlp.func == MLP.as_mlp_submodule


def test_moe_spec_accepts_an_explicit_pattern_list(first_pipeline_rank):
    spec = get_gpt_modelopt_spec(_make_moe_config(num_layers=2, moe_layer_freq=[0, 1]))

    assert spec.layer_specs[0].submodules.mlp.func == MLP.as_mlp_submodule
    assert spec.layer_specs[1].submodules.mlp.func is MoELayer


def test_moe_spec_rejects_non_int_list_frequencies(first_pipeline_rank):
    with pytest.raises(ValueError, match="Invalid moe_layer_freq: <class 'str'>, all"):
        get_gpt_modelopt_spec(_make_moe_config(moe_layer_freq='all'))


def test_moe_spec_rejects_pattern_values_other_than_zero_and_one(first_pipeline_rank):
    with pytest.raises(ValueError, match=r"Invalid layer pattern: \[2, 0\]"):
        get_gpt_modelopt_spec(_make_moe_config(num_layers=2, moe_layer_freq=[2, 0]))


def test_moe_spec_slices_layers_for_a_non_first_pipeline_rank(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda group=None: 1)

    spec = get_gpt_modelopt_spec(
        _make_moe_config(
            pipeline_model_parallel_size=2, pipeline_dtype=torch.float32, moe_layer_freq=2
        )
    )

    assert len(spec.layer_specs) == 2
    assert spec.layer_specs[0].submodules.mlp.func is MoELayer
    assert spec.layer_specs[1].submodules.mlp.func == MLP.as_mlp_submodule
