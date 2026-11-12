# Copyright (c) 2026, BAAI. All rights reserved.
"""LLaVA decoder layer specs: causal attention, norm and MLP submodule wiring for transformer-engine and local builds."""
import pytest

import megatron.core.models.multimodal.llava_spec as llava_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


@pytest.mark.parametrize("qk_layernorm", [False, True], ids=["no_qk_ln", "qk_ln"])
def test_te_default_spec_builds_transformer_layer_with_te_attention(qk_layernorm):
    spec = llava_spec.decoder_model_with_transformer_engine_default_spec(qk_layernorm=qk_layernorm)

    assert isinstance(spec, ModuleSpec)
    assert spec.module is TransformerLayer
    assert isinstance(spec.submodules, TransformerLayerSubmodules)
    attention = spec.submodules.self_attention
    assert attention.module is SelfAttention
    assert isinstance(attention.submodules, SelfAttentionSubmodules)
    assert attention.params["attn_mask_type"] is AttnMaskType.causal
    assert attention.submodules.core_attention is llava_spec.TEDotProductAttention
    assert attention.submodules.linear_qkv is llava_spec.TELayerNormColumnParallelLinear
    assert attention.submodules.linear_proj is llava_spec.TERowParallelLinear
    expected_norm = llava_spec.TENorm if qk_layernorm else IdentityOp
    assert attention.submodules.q_layernorm is expected_norm
    assert attention.submodules.k_layernorm is expected_norm
    mlp = spec.submodules.mlp
    assert mlp.func == MLP.as_mlp_submodule
    mlp_submodules = mlp.keywords['submodules']
    assert mlp_submodules.linear_fc1 is llava_spec.TELayerNormColumnParallelLinear
    assert mlp_submodules.linear_fc2 is llava_spec.TERowParallelLinear


def test_local_default_spec_uses_local_attention_and_norm():
    spec = llava_spec.decoder_model_with_local_default_spec()

    assert spec.module is TransformerLayer
    attention = spec.submodules.self_attention
    assert attention.module is SelfAttention
    assert attention.params["attn_mask_type"] is AttnMaskType.causal
    assert attention.submodules.linear_qkv is llava_spec.ColumnParallelLinear
    assert attention.submodules.core_attention is DotProductAttention
    assert attention.submodules.linear_proj is llava_spec.RowParallelLinear
    assert spec.submodules.input_layernorm is llava_spec.LNImpl
    assert spec.submodules.pre_mlp_layernorm is llava_spec.LNImpl
    mlp = spec.submodules.mlp
    assert mlp.func == MLP.as_mlp_submodule
    mlp_submodules = mlp.keywords['submodules']
    assert mlp_submodules.linear_fc1 is llava_spec.ColumnParallelLinear
    assert mlp_submodules.linear_fc2 is llava_spec.RowParallelLinear
