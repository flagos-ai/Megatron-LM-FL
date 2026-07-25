# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Factory functions for building MoT layer specs with Transformer Engine.

Provides ``get_mot_layer_with_transformer_engine_spec`` which constructs a
``MoTTransformerLayer`` spec with an arbitrary number of branches.
"""

from typing import Dict, List, Optional, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import not_none

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from .transformer_layer import (
    MoTBranchSpec,
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)

from .attention.attention import PackedAttentionMoT
from .attention.submodules import MoTAttentionBranchSpec, MoTSelfAttentionSubmodules


def _get_default_mlp_spec(use_te: bool = True) -> ModuleSpec:
    """Standard dense MLP spec (TE or vanilla)."""
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=not_none(TEColumnParallelLinear) if use_te else ColumnParallelLinear,
            linear_fc2=not_none(TERowParallelLinear) if use_te else RowParallelLinear,
        ),
    )


def get_mot_layer_with_transformer_engine_spec(
    branch_names: List[str],
    qk_layernorm: bool = False,
    use_te: bool = True,
    branch_mlp_specs: Optional[Dict[str, ModuleSpec]] = None,
) -> ModuleSpec:
    """Build a MoTTransformerLayer spec with N branches using TE modules.

    Args:
        branch_names: Ordered list of branch names (e.g. ["und", "gen"]).
            First branch is primary and uses unsuffixed attribute names.
        qk_layernorm: Whether to apply Q/K layernorm in attention.
        use_te: Whether to use Transformer Engine modules.
        branch_mlp_specs: Optional dict mapping branch name to custom MLP spec.
            Branches not in this dict use the default dense MLP.

    Returns:
        A ModuleSpec for ``MoTTransformerLayer`` with all branches configured.

    Example (standard Bagel 2-branch)::

        spec = get_mot_layer_with_transformer_engine_spec(
            branch_names=["und", "gen"],
            qk_layernorm=True,
        )

    Example (3-branch with audio)::

        spec = get_mot_layer_with_transformer_engine_spec(
            branch_names=["und", "gen", "audio_gen"],
            qk_layernorm=True,
        )
    """
    assert len(branch_names) >= 1, "At least one branch is required"
    if branch_mlp_specs is None:
        branch_mlp_specs = {}

    Norm = TENorm if (HAVE_TE and use_te) else IdentityOp
    qkv_linear = ColumnParallelLinear
    proj_linear = TERowParallelLinear if (HAVE_TE and use_te) else RowParallelLinear
    q_ln = Norm if qk_layernorm else IdentityOp
    k_ln = Norm if qk_layernorm else IdentityOp

    # --- Build layer-level branch specs (layernorm + MLP) ---
    layer_branches = []
    for name in branch_names:
        mlp_spec = branch_mlp_specs.get(name, _get_default_mlp_spec(use_te=use_te))
        layer_branches.append(
            MoTBranchSpec(
                name=name,
                input_layernorm=Norm,
                pre_mlp_layernorm=Norm,
                mlp=mlp_spec,
            )
        )

    # --- Build attention-level branch specs (QKV + proj) ---
    attn_branches = []
    for name in branch_names:
        attn_branches.append(
            MoTAttentionBranchSpec(
                name=name,
                linear_qkv=qkv_linear,
                linear_proj=proj_linear,
                q_layernorm=q_ln,
                k_layernorm=k_ln,
            )
        )

    # --- Attention submodules ---
    core_attn = TEDotProductAttention if (HAVE_TE and use_te) else DotProductAttention
    attn_submodules = MoTSelfAttentionSubmodules(
        branches=attn_branches,
        core_attention=core_attn,
    )

    # --- Compose full layer spec ---
    return ModuleSpec(
        module=MoTTransformerLayer,
        submodules=MoTTransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=PackedAttentionMoT,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=attn_submodules,
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp_bda=get_bias_dropout_add,
            branches=layer_branches,
        ),
    )


# Convenience aliases for common configurations


def get_mot_2branch_spec(qk_layernorm: bool = True, use_te: bool = True) -> ModuleSpec:
    """Convenience: standard 2-branch MoT spec (understanding + generation)."""
    return get_mot_layer_with_transformer_engine_spec(
        branch_names=["und", "gen"],
        qk_layernorm=qk_layernorm,
        use_te=use_te,
    )


def get_mot_3branch_spec(qk_layernorm: bool = True, use_te: bool = True) -> ModuleSpec:
    """Convenience: 3-branch MoT spec (understanding + visual_gen + audio_gen)."""
    return get_mot_layer_with_transformer_engine_spec(
        branch_names=["und", "gen", "audio_gen"],
        qk_layernorm=qk_layernorm,
        use_te=use_te,
    )
