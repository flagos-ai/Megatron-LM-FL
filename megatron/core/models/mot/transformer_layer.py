# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mixture-of-Transformers (MoT) layer implementation.

Implements N-branch attention and MLP where different token subsets (e.g.
understanding vs generation) use separate layernorms/MLPs but share the core
attention computation.  Branches are configured via an ordered list of
``MoTBranchSpec``, making the design generic for any number of modalities.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType, CudaGraphScope
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    get_pg_rank,
    log_single_rank,
    make_viewless_tensor,
)

from .routing import validate_branch_token_indexes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Branch specification
# ---------------------------------------------------------------------------


@dataclass
class MoTBranchSpec:
    """Specification for a single MoT branch (e.g. understanding, generation).

    Each branch has its own input layernorm, pre-MLP layernorm, and MLP.
    Bias-dropout-add is shared after branch outputs are scattered back into the
    full sequence.  The ``name`` field is used as the suffix for
    the corresponding nn.Module attribute names (e.g. name='gen' ->
    ``self.input_layernorm_gen``, ``self.mlp_gen``).

    The first branch (index 0) uses the base attribute names without a suffix
    (``self.input_layernorm``, ``self.mlp``, etc.) to keep checkpoint
    compatibility with the standard ``TransformerLayer``.
    """

    name: str
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp


@dataclass
class MoTTransformerLayerSubmodules:
    """Submodule specifications for the MoT transformer layer.

    ``branches`` is an ordered list of branch specs.  The first branch is the
    "primary" branch (typically understanding/text) and uses unsuffixed attribute
    names.  Additional branches get their ``name`` appended as suffix.

    Example for BAGEL (understanding + generation)::

        MoTTransformerLayerSubmodules(
            branches=[
                MoTBranchSpec(name='und', input_layernorm=RMSNorm, ...),
                MoTBranchSpec(name='gen', input_layernorm=RMSNorm, ...),
            ],
            self_attention=ModuleSpec(...),
            self_attn_bda=ModuleSpec(...),
        )
    """

    # Shared attention (handles multi-branch QKV internally)
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Shared BDA applied after per-branch MLP outputs are scattered.
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Per-branch modules (ordered; first branch uses base attr names)
    branches: List[MoTBranchSpec] = field(default_factory=list)

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MoTTransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    """A single Mixture-of-Transformers layer.

    Implements N-branch processing where different token subsets use separate
    layernorms and MLPs but share the core attention computation.

    Input/Output shape: [seq, batch, hidden].
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoTTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        dualpipev_stage: Optional[int] = None,
    ):
        self.submodules_config = submodules
        super().__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        assert is_mtp_layer is False

        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp), dualpipev_stage
        )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_mtp_layer = is_mtp_layer

        # ============================================================
        # Build per-branch modules
        # ============================================================
        assert len(submodules.branches) >= 1, (
            "MoTTransformerLayer requires at least one branch in submodules.branches"
        )
        self.branch_names: List[str] = []
        self.num_branches = len(submodules.branches)

        # Import here to avoid circular import
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        for idx, branch in enumerate(submodules.branches):
            self.branch_names.append(branch.name)
            # First branch uses base attribute names, others use suffix
            suffix = '' if idx == 0 else f'_{branch.name}'

            # Input layernorm
            ln = branch.input_layernorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            setattr(self, f'input_layernorm{suffix}', ln)

            # Pre-MLP layernorm
            pre_ln = branch.pre_mlp_layernorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            setattr(self, f'pre_mlp_layernorm{suffix}', pre_ln)

            # MLP
            mlp_kwargs = {}
            if isinstance(branch.mlp, ModuleSpec):
                if branch.mlp.module in (MoELayer, TEGroupedMLP, SequentialMLP):
                    mlp_kwargs["pg_collection"] = pg_collection
                    if branch.mlp.module == MoELayer:
                        mlp_kwargs["is_mtp_layer"] = self.is_mtp_layer
                elif branch.mlp.module == MLP:
                    assert hasattr(pg_collection, 'tp'), (
                        f'TP process group is required for MLP in MoTTransformerLayer '
                        f'branch "{branch.name}"'
                    )
                    mlp_kwargs["tp_group"] = pg_collection.tp
                elif TEFusedMLP is not None and branch.mlp.module == TEFusedMLP:
                    assert hasattr(pg_collection, 'tp'), (
                        f'TP process group is required for TEFusedMLP in MoTTransformerLayer '
                        f'branch "{branch.name}"'
                    )
                    mlp_kwargs["tp_group"] = pg_collection.tp
                else:
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f'Unknown MLP type in branch "{branch.name}": {branch.mlp.module}. '
                        f'Using default kwargs.',
                    )
            mlp_module = build_module(branch.mlp, config=self.config, **mlp_kwargs)
            if hasattr(mlp_module, 'set_layer_number'):
                mlp_module.set_layer_number(self.layer_number)
            setattr(self, f'mlp{suffix}', mlp_module)

        # ============================================================
        # Shared attention
        # ============================================================
        attention_optional_kwargs = {"pg_collection": pg_collection}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                # layer_number is 1-indexed, so we need to subtract 1 to get the correct index
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number - 1
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["pg_collection"] = pg_collection
        if pp_layer_offset is not None:
            attention_optional_kwargs["pp_layer_offset"] = pp_layer_offset

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # Shared BiasDropoutFusion for the scattered MLP output.
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        # TODO(zhaoyinglia): recompute

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    # ------------------------------------------------------------------
    # Helper: get branch module by index
    # ------------------------------------------------------------------

    def _get_branch_module(self, attr_base: str, branch_idx: int):
        """Get a branch-specific module by base attribute name and branch index."""
        if branch_idx == 0:
            return getattr(self, attr_base)
        suffix = f'_{self.branch_names[branch_idx]}'
        return getattr(self, f'{attr_base}{suffix}')

    @staticmethod
    def _unpack_norm_output(
        norm_output: Any,
        default_residual: Tensor,
        module_name: str,
    ) -> tuple[Tensor, Tensor]:
        """Normalize Megatron norm outputs to ``(normalized, residual)``."""
        if isinstance(norm_output, tuple):
            if len(norm_output) != 2:
                raise ValueError(
                    f"When {module_name} returns a tuple, it must contain "
                    f"(normalized_output, residual), but got {len(norm_output)} elements"
                )
            normalized_output, residual = norm_output
        else:
            normalized_output = norm_output
            residual = default_residual

        return normalized_output, residual

    # ------------------------------------------------------------------
    # CudaGraph
    # ------------------------------------------------------------------

    def create_mcore_cudagraph_manager(self, config):
        """Register the transformer layer for cudagraphs."""

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if not self.config.cuda_graph_scope:
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.attn in self.config.cuda_graph_scope
            and self.submodules_config.self_attention != IdentityOp
        ):
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.mlp in self.config.cuda_graph_scope
            and self.submodules_config.branches
            and self.submodules_config.branches[0].mlp != IdentityOp
        ):
            assert not self.is_moe_layer
            self.cudagraph_manager = CudaGraphManager(config)

    def __call__(self, *args, **kwargs):
        # Extract mhc_recompute_manager before CUDA graph manager processes kwargs,
        # since CheckpointManager is not a CUDA-graph-supported type.
        kwargs.pop("mhc_recompute_manager", None)
        kwargs.pop("is_last_layer_in_recompute_block", None)

        return super().__call__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ):
        """Forward pass through the MoT transformer layer.

        Token routing is determined by ``packed_seq_params.branch_token_indexes``,
        a dict mapping branch name -> 1-D LongTensor of positions.

        Falls back to single-branch (primary) processing when
        ``branch_token_indexes`` is None or empty.
        """
        # Retrieve branch token indexes from packed_seq_params
        branch_token_indexes: Optional[Dict[str, Tensor]] = None
        if packed_seq_params is not None:
            branch_token_indexes = getattr(packed_seq_params, 'branch_token_indexes', None)

        has_mot = branch_token_indexes is not None and len(branch_token_indexes) > 0

        # if has_mot:
        #     validate_branch_token_indexes(
        #         hidden_states,
        #         branch_token_indexes,
        #         self.branch_names,
        #         packed_seq_params,
        #     )

        # ==============================================================
        # Multi-branch Input LayerNorm
        # ==============================================================
        if has_mot:
            input_layernorm_output = torch.zeros_like(hidden_states)
            residual = torch.zeros_like(hidden_states)
            for idx, name in enumerate(self.branch_names):
                if name not in branch_token_indexes:
                    continue
                indexes = branch_token_indexes[name]
                ln = self._get_branch_module('input_layernorm', idx)
                # Only compute layernorm on the branch's token subset
                branch_input = hidden_states[indexes]
                norm_output = apply_module(ln)(branch_input)
                normed, branch_residual = self._unpack_norm_output(
                    norm_output,
                    branch_input,
                    f'input_layernorm[{name}]',
                )
                input_layernorm_output[indexes] = normed
                residual[indexes] = branch_residual
        else:
            norm_output = apply_module(self.input_layernorm)(hidden_states)
            input_layernorm_output, residual = self._unpack_norm_output(
                norm_output,
                hidden_states,
                'input_layernorm',
            )

        # ==============================================================
        # Shared Self-Attention
        # ==============================================================
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **({"branch_token_indexes": branch_token_indexes} if has_mot else {}),
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        # ==============================================================
        # Multi-branch Pre-MLP LayerNorm + MLPs
        # ==============================================================
        if has_mot:
            # Run each branch MLP only on its token subset (gather → compute → scatter)
            mlp_output = torch.zeros_like(hidden_states)
            residual = torch.zeros_like(hidden_states)
            mlp_bias: Optional[Tensor] = None

            for idx, name in enumerate(self.branch_names):
                if name not in branch_token_indexes:
                    continue
                indexes = branch_token_indexes[name]
                pre_ln = self._get_branch_module('pre_mlp_layernorm', idx)
                mlp_mod = self._get_branch_module('mlp', idx)

                # Only compute on the branch's token subset
                branch_hidden = hidden_states[indexes]
                norm_output = apply_module(pre_ln)(branch_hidden)
                normed, branch_residual = self._unpack_norm_output(
                    norm_output,
                    branch_hidden,
                    f'pre_mlp_layernorm[{name}]',
                )
                branch_mlp_out, branch_mlp_bias = mlp_mod(normed)

                mlp_output[indexes] = branch_mlp_out
                residual[indexes] = branch_residual

                if branch_mlp_bias is not None:
                    if mlp_bias is None:
                        mlp_bias = torch.zeros_like(mlp_output)
                    if branch_mlp_bias.shape == branch_mlp_out.shape:
                        # Bias has same shape as branch output, scatter directly
                        mlp_bias[indexes] = branch_mlp_bias
                    else:
                        # Bias is broadcast-shaped (e.g. [H]), expand to branch shape
                        expanded_bias = branch_mlp_bias.expand_as(branch_mlp_out)
                        mlp_bias[indexes] = expanded_bias

            mlp_output_with_bias = (mlp_output, mlp_bias)
        else:
            norm_output = apply_module(self.pre_mlp_layernorm)(hidden_states)
            pre_mlp_layernorm_output, residual = self._unpack_norm_output(
                norm_output,
                hidden_states,
                'pre_mlp_layernorm',
            )
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        return output, context

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for the MoT transformer layer."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
