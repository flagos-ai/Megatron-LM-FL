# Copyright (c) 2026, BAAI. All rights reserved.
"""Hybrid stack ModelOpt spec selection between the TE default and local module specs."""

import pytest

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.post_training.modelopt.hybrid.model_specs import (
    get_hybrid_stack_modelopt_spec,
    hybrid_stack_spec_no_moe_grouped_gemm,
)
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer


@pytest.mark.parametrize(
    ('moe_grouped_gemm', 'expected_spec'),
    [(True, hybrid_stack_spec), (False, hybrid_stack_spec_no_moe_grouped_gemm)],
    ids=['grouped_gemm', 'per_expert_linears'],
)
def test_use_default_te_spec_selects_the_te_spec_variant_by_grouped_gemm(
    moe_grouped_gemm, expected_spec
):
    spec = get_hybrid_stack_modelopt_spec(
        use_default_te_spec=True, moe_grouped_gemm=moe_grouped_gemm
    )

    assert spec is expected_spec


def test_local_spec_builds_local_mamba_attention_and_mlp_layers():
    spec = get_hybrid_stack_modelopt_spec()

    assert spec.module is HybridStack
    mamba_layer = spec.submodules.mamba_layer
    assert mamba_layer.module.__name__ == 'MambaLayer'
    assert mamba_layer.submodules.mixer.module is MambaMixer
    assert mamba_layer.submodules.sharded_state_dict_keys_map == {}
    attention_layer = spec.submodules.attention_layer
    assert attention_layer.module is TransformerLayer
    assert attention_layer.submodules.self_attention.module is SelfAttention
    assert (
        attention_layer.submodules.self_attention.submodules.core_attention is TEDotProductAttention
    )
    assert spec.submodules.mlp_layer.submodules.mlp.func == MLP.as_mlp_submodule
    assert spec.submodules.moe_layer.submodules.mlp.func is MoELayer


def test_local_spec_with_local_core_attention_uses_dot_product_attention():
    spec = get_hybrid_stack_modelopt_spec(local_core_attention=True)

    core_attention = (
        spec.submodules.attention_layer.submodules.self_attention.submodules.core_attention
    )
    assert core_attention is DotProductAttention


def test_local_spec_remap_te_layernorm_maps_mamba_and_transformer_norms():
    spec = get_hybrid_stack_modelopt_spec(remap_te_layernorm=True)

    assert spec.submodules.mamba_layer.submodules.sharded_state_dict_keys_map == {
        'norm.': 'mixer.in_proj.layer_norm_'
    }
    transformer_map = {
        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
    }
    assert spec.submodules.attention_layer.submodules.sharded_state_dict_keys_map == transformer_map
    assert spec.submodules.mlp_layer.submodules.sharded_state_dict_keys_map == transformer_map
