# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoT attention submodule specifications.

Defines the per-branch specification for QKV projections and output projection
in a Mixture-of-Transformers self-attention module.
"""

from dataclasses import dataclass, field
from typing import List, Union

from megatron.core.transformer.attention import CoreAttentionBuilder, LinearQkvBuilder
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec


@dataclass
class MoTAttentionBranchSpec:
    """Specification for one branch's QKV + output projection in MoT attention.

    Each branch (e.g. "und", "gen", "audio_gen") has its own:
    - linear_qkv: Column-parallel QKV projection
    - linear_proj: Row-parallel output projection
    - q_layernorm / k_layernorm: Optional query/key layernorms

    The first branch (index 0) uses unsuffixed attribute names for checkpoint
    compatibility with standard Megatron ``SelfAttention``.

    Example::

        MoTAttentionBranchSpec(
            name="und",
            linear_qkv=ColumnParallelLinear,
            linear_proj=TERowParallelLinear,
            q_layernorm=TENorm,
            k_layernorm=TENorm,
        )
    """

    name: str
    linear_qkv: LinearQkvBuilder = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = IdentityOp
    k_layernorm: Union[ModuleSpec, type] = IdentityOp


@dataclass
class MoTSelfAttentionSubmodules:
    """N-branch self-attention submodule specification.

    ``branches`` is an ordered list of branch specs. The first branch is the
    "primary" branch (typically understanding/text) and will use unsuffixed
    attribute names for checkpoint compatibility.

    ``core_attention`` is the shared attention computation kernel used by all
    branches after their individual QKV projections.

    Example (Bagel: understanding + generation)::

        MoTSelfAttentionSubmodules(
            branches=[
                MoTAttentionBranchSpec(name="und", linear_qkv=..., linear_proj=...),
                MoTAttentionBranchSpec(name="gen", linear_qkv=..., linear_proj=...),
            ],
            core_attention=TEDotProductAttention,
        )

    Example (Bagel + audio: 3 branches)::

        MoTSelfAttentionSubmodules(
            branches=[
                MoTAttentionBranchSpec(name="und", ...),
                MoTAttentionBranchSpec(name="gen", ...),
                MoTAttentionBranchSpec(name="audio_gen", ...),
            ],
            core_attention=TEDotProductAttention,
        )
    """

    branches: List[MoTAttentionBranchSpec] = field(default_factory=list)
    core_attention: CoreAttentionBuilder = None

    # --- Compatibility attributes for Attention base class ---
    # Attention.__init__ accesses submodules.linear_proj, etc.
    # We expose the first branch's specs so the base class can initialize.
    # PackedAttentionMoT then creates all per-branch modules itself (including
    # re-creating idx==0), which simply overwrites what the base class built.
    linear_proj: Union[ModuleSpec, type] = None
    linear_qkv: LinearQkvBuilder = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None

    def __post_init__(self):
        """Populate base-class-compat fields from the first branch."""
        if self.branches:
            first = self.branches[0]
            if self.linear_proj is None:
                self.linear_proj = first.linear_proj
            if self.linear_qkv is None:
                self.linear_qkv = first.linear_qkv
            if self.q_layernorm is None:
                self.q_layernorm = first.q_layernorm
            if self.k_layernorm is None:
                self.k_layernorm = first.k_layernorm
