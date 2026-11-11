# Copyright (c) 2026, BAAI. All rights reserved.
"""ViT layer specs: unmasked attention, local norms and MLP wiring for vision encoders."""
import megatron.core.models.vision.vit_layer_specs as vit_layer_specs
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_local_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def test_local_vit_spec_uses_unmasked_local_attention():
    spec = get_vit_layer_with_local_spec()

    assert isinstance(spec, ModuleSpec)
    assert spec.module is TransformerLayer
    assert isinstance(spec.submodules, TransformerLayerSubmodules)
    attention = spec.submodules.self_attention
    assert attention.module is SelfAttention
    assert isinstance(attention.submodules, SelfAttentionSubmodules)
    assert attention.params["attn_mask_type"] is AttnMaskType.no_mask
    assert attention.submodules.linear_qkv is ColumnParallelLinear
    assert attention.submodules.core_attention is DotProductAttention
    assert attention.submodules.linear_proj is RowParallelLinear
    assert spec.submodules.input_layernorm is vit_layer_specs.LNImpl
    assert spec.submodules.pre_mlp_layernorm is vit_layer_specs.LNImpl
    mlp = spec.submodules.mlp
    assert mlp.func == MLP.as_mlp_submodule
    assert mlp.keywords['submodules'].linear_fc1 is ColumnParallelLinear
    assert mlp.keywords['submodules'].linear_fc2 is RowParallelLinear
