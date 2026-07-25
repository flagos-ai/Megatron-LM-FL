# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mixture-of-Transformers (MoT) model components.

Provides generic N-branch MoT attention and layer spec factories that integrate
with ``megatron.core.models.mot.transformer_layer`` and
``megatron.core.models.mot.routing.MoTPackedSeqParams``.

Key components:
- ``PackedAttentionMoT``: N-branch self-attention with shared core attention.
- ``MoTAttentionBranchSpec``: Per-branch QKV/projection specification.
- ``MoTSelfAttentionSubmodules``: Attention-level N-branch spec container.
- ``get_mot_layer_with_transformer_engine_spec``: Factory for building MoT layer specs.
- ``create_sparse_mask``: flex_attention mask builder for packed sequences.

Usage::

    # Custom N-branch
    layer_spec = get_mot_layer_with_transformer_engine_spec(
        branch_names=["und", "gen", "audio_gen"],
        qk_layernorm=True,
    )
"""

from .attention import (
    MoTAttentionBranchSpec,
    MoTSelfAttentionSubmodules,
    PackedAttentionMoT,
    create_packed_block_mask,
    create_sparse_mask,
)
from .layer_specs import (
    get_mot_layer_with_transformer_engine_spec,
)
from .routing import MoTPackedSeqParams
from .transformer_layer import (
    MoTBranchSpec,
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)

__all__ = [
    # Attention
    'PackedAttentionMoT',
    'MoTAttentionBranchSpec',
    'MoTSelfAttentionSubmodules',
    # Transformer layer
    'MoTBranchSpec',
    'MoTTransformerLayer',
    'MoTTransformerLayerSubmodules',
    # Routing
    'MoTPackedSeqParams',
    # Layer specs
    'get_mot_layer_with_transformer_engine_spec',
    # Utilities
    'create_sparse_mask',
    'create_packed_block_mask',
]
