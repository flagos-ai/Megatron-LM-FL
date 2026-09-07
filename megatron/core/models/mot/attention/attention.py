# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""N-branch Mixture-of-Transformers Self-Attention using flex_attention.

Each branch (e.g. understanding, generation, audio) has its own QKV and output
projections while the core attention kernel is shared. Token routing uses
index tensors from ``MoTPackedSeqParams.branch_token_indexes``.
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    all_gather_last_dim_from_tensor_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import get_pg_rank

from megatron.core.extensions.transformer_engine import HAVE_TE

from ..routing import MoTPackedSeqParams

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import SplitAlongDim
else:
    SplitAlongDim = None

from .mask import _pad_to_length, create_packed_block_mask
from .submodules import MoTAttentionBranchSpec, MoTSelfAttentionSubmodules

# Increase torch.compile cache limits for flex_attention's generated kernels.
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
_compiled_flex_attention = torch.compile(flex_attention)


class PackedAttentionMoT(Attention):
    """N-branch Mixture-of-Transformers Self-Attention.

    Data layout (training, packed 1D)::

        Input:   [total_seq, 1, hidden]   (Megatron convention: [s, b, h])
        QKV:     [total_seq, num_heads, head_dim]
        flex_attention expects [batch=1, num_heads, total_seq, head_dim]

    Token routing uses index tensors from ``branch_token_indexes`` (a dict
    mapping branch name to 1-D LongTensor of positions).

    Falls back to primary-branch-only when ``branch_token_indexes`` is None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoTSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
        # Pass first branch's submodules as the base for Attention.__init__
        # (it expects linear_qkv, core_attention, linear_proj at minimum)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

        assert len(submodules.branches) >= 1, (
            "PackedAttentionMoT requires at least one branch"
        )

        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.num_kv_heads = self.config.num_query_groups
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        self.branch_names: List[str] = []

        if self.config.attention_output_gate:
            raise NotImplementedError("PackedAttentionMoT does not support attention_output_gate")
        if self.config.context_parallel_size > 1:
            raise NotImplementedError("PackedAttentionMoT does not support context parallelism")

        # Build per-branch QKV, output proj, and optional qk layernorms
        for idx, branch in enumerate(submodules.branches):
            self.branch_names.append(branch.name)
            suffix = '' if idx == 0 else f'_{branch.name}'

            # QKV projection
            qkv = branch.linear_qkv(
                self.config.hidden_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=not_none(self.config.init_method),
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=f'qkv_{branch.name}',
                tp_group=self.pg_collection.tp,
            )
            setattr(self, f'linear_qkv{suffix}', qkv)

            # Output projection
            proj = build_module(
                branch.linear_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=not_none(self.config.output_layer_init_method),
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name=f'proj_{branch.name}',
                tp_group=self.pg_collection.tp,
            )
            setattr(self, f'linear_proj{suffix}', proj)

            # Optional Q/K layernorms
            if branch.q_layernorm is not None and branch.q_layernorm != IdentityOp:
                q_ln = branch.q_layernorm(
                    config=self.config,
                    hidden_size=self.head_dim,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                q_ln = IdentityOp()
            setattr(self, f'q_layernorm{suffix}', q_ln)

            if branch.k_layernorm is not None and branch.k_layernorm != IdentityOp:
                k_ln = branch.k_layernorm(
                    config=self.config,
                    hidden_size=self.head_dim,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                k_ln = IdentityOp()
            setattr(self, f'k_layernorm{suffix}', k_ln)

    def _get_branch_module(self, attr_base: str, branch_idx: int):
        """Get branch-specific module by base name and index."""
        if branch_idx == 0:
            return getattr(self, attr_base)
        suffix = f'_{self.branch_names[branch_idx]}'
        return getattr(self, f'{attr_base}{suffix}')

    def _split_qkv(self, qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split Megatron's query-group-interleaved QKV projection output."""
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )

        if self.config.num_query_groups < self.world_size:
            qkv = all_gather_last_dim_from_tensor_parallel_region(
                qkv, group=self.pg_collection.tp
            )
            group_idx = get_pg_rank(self.pg_collection.tp) // (
                self.world_size // self.config.num_query_groups
            )
            group_size = qkv.size(-1) // self.config.num_query_groups
            qkv = qkv[..., group_idx * group_size : (group_idx + 1) * group_size]

        qkv = qkv.view(
            qkv.size(0),
            self.num_query_groups_per_partition,
            (num_query_heads_per_group + 2) * self.head_dim,
        )
        query, key, value = torch.split(
            qkv,
            [num_query_heads_per_group * self.head_dim, self.head_dim, self.head_dim],
            dim=-1,
        )
        query = query.reshape(
            qkv.size(0), self.num_attention_heads_per_partition, self.head_dim
        )

        if self.config.num_query_groups < self.world_size:
            rank_in_group = get_pg_rank(self.pg_collection.tp) % (
                self.world_size // self.config.num_query_groups
            )
            heads_per_rank = self.num_attention_heads_per_partition // (
                self.world_size // self.config.num_query_groups
            )
            query = query[
                :, rank_in_group * heads_per_rank : (rank_in_group + 1) * heads_per_rank, :
            ]

        return query, key, value

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Not used directly — forward handles multi-branch QKV internally."""
        raise NotImplementedError(
            "PackedAttentionMoT handles QKV computation internally per branch. "
            "Use forward() directly."
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        # MoT-specific
        branch_token_indexes: Optional[Dict[str, Tensor]] = None,
    ):
        """Forward pass with N-branch QKV routing.

        Args:
            hidden_states: [S, B, H] where B=1 for packed sequences.
            branch_token_indexes: Dict mapping branch name to 1-D LongTensor
                of token positions. If None, all tokens use primary branch.
            packed_seq_params: Contains MoTPackedSeqParams with mask info.
        """
        squeeze = hidden_states.dim() == 3
        if squeeze:
            # [S, 1, H] -> [S, H]
            hidden_states = hidden_states.squeeze(1)
        S, H = hidden_states.shape

        tp_size = get_tensor_model_parallel_world_size()
        nH_local = self.num_heads // tp_size
        nKV_local = self.num_kv_heads // tp_size
        D = self.head_dim

        # Determine if multi-branch routing is active
        has_mot = branch_token_indexes is not None and len(branch_token_indexes) > 0

        if not has_mot:
            # Single-branch fallback: use primary branch for all tokens
            branch_token_indexes = {self.branch_names[0]: torch.arange(S, device=hidden_states.device)}

        # ==============================================================
        # 1. Per-branch QKV projections + layernorms
        # ==============================================================
        Q_buf = hidden_states.new_zeros(S, nH_local, D)
        K_buf = hidden_states.new_zeros(S, nKV_local, D)
        V_buf = hidden_states.new_zeros(S, nKV_local, D)

        for idx, name in enumerate(self.branch_names):
            if name not in branch_token_indexes:
                continue
            indexes = branch_token_indexes[name]

            linear_qkv = self._get_branch_module('linear_qkv', idx)
            q_layernorm = self._get_branch_module('q_layernorm', idx)
            k_layernorm = self._get_branch_module('k_layernorm', idx)

            # QKV projection for this branch's tokens
            branch_hidden = hidden_states[indexes]  # [n_tokens, H]
            qkv_out, _ = linear_qkv(branch_hidden)  # [n_tokens, qkv_dim]
            q, k, v = self._split_qkv(qkv_out)

            # Optional QK layernorm
            if not isinstance(q_layernorm, IdentityOp):
                q = apply_module(q_layernorm)(q)
                k = apply_module(k_layernorm)(k)
                if isinstance(q, tuple) or isinstance(k, tuple):
                    raise TypeError(
                        "MoT Q/K layernorms must return Tensors; tuple residual outputs "
                        "are not supported for Q/K normalization"
                    )

            # Scatter into full-sequence buffers
            Q_buf[indexes] = q
            K_buf[indexes] = k
            V_buf[indexes] = v

        # ==============================================================
        # 2. Apply RoPE
        # ==============================================================
        if rotary_pos_emb is not None:
            # Normalize to (q_pos_emb, k_pos_emb) tuple
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb

            # apply_rotary_pos_emb expects [S, B, nH, D] — add batch dim
            Q_buf = apply_rotary_pos_emb(
                Q_buf.unsqueeze(1), q_pos_emb, config=self.config, cu_seqlens=None
            ).squeeze(1)
            K_buf = apply_rotary_pos_emb(
                K_buf.unsqueeze(1), k_pos_emb, config=self.config, cu_seqlens=None
            ).squeeze(1)

        # ==============================================================
        # 3. Core attention (flex_attention with BlockMask)
        # ==============================================================
        # flex_attention expects [B=1, nH, S, D] for Q and [B=1, nKV, S, D] for K/V.
        # With enable_gqa=True, flex_attention handles GQA natively without expanding K/V.
        Q_4d = Q_buf.permute(1, 0, 2).unsqueeze(0)  # [1, nH_local, S, D]
        K_4d = K_buf.permute(1, 0, 2).unsqueeze(0)  # [1, nKV_local, S, D]
        V_4d = V_buf.permute(1, 0, 2).unsqueeze(0)  # [1, nKV_local, S, D]

        use_gqa = nKV_local < nH_local

        # Determine block_mask: prefer pre-built attention_mask from caller,
        # fall back to dynamic construction from packed_seq_params.
        block_mask = None
        if attention_mask is not None:
            block_mask = attention_mask
        elif packed_seq_params is not None and getattr(packed_seq_params, 'block_mask', None) is not None:
            block_mask = packed_seq_params.block_mask
        elif packed_seq_params is not None and hasattr(packed_seq_params, 'sample_lens'):
            mot_params = packed_seq_params
            if (
                mot_params.sample_lens is not None
                and mot_params.split_lens is not None
                and mot_params.attn_modes is not None
            ):
                block_mask = create_packed_block_mask(
                    sample_lens=mot_params.sample_lens,
                    split_lens=mot_params.split_lens,
                    attn_modes=mot_params.attn_modes,
                    device=hidden_states.device,
                )

        # Pad QKV to match BlockMask length (bagel-style: mask decides padded length)
        if block_mask is not None:
            mask_seq_len = block_mask.shape[-1]  # BlockMask Q_LEN
            pad_size = mask_seq_len - S
        else:
            pad_size = 0

        if pad_size > 0:
            Q_4d = _pad_to_length(Q_4d, S + pad_size, dim=2)
            K_4d = _pad_to_length(K_4d, S + pad_size, dim=2)
            V_4d = _pad_to_length(V_4d, S + pad_size, dim=2)

        if block_mask is not None:
            attn_out = _compiled_flex_attention(
                Q_4d, K_4d, V_4d, block_mask=block_mask, enable_gqa=use_gqa
            )
        else:
            # Fallback: scaled dot-product attention (no mask available)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                Q_4d, K_4d, V_4d, is_causal=True, enable_gqa=use_gqa,
            )

        # Remove padding → [1, nH, S, D] -> [S, nH*D]
        if pad_size > 0:
            attn_out = attn_out[:, :, :S, :]
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(S, -1)  # [S, nH_local*D]

        # ==============================================================
        # 5. Per-branch output projections
        # ==============================================================
        # linear_proj (RowParallelLinear) takes sharded input [n, nH_local*D]
        # and outputs full [n, hidden_size] after all-reduce.
        out_buf = hidden_states.new_zeros(S, self.config.hidden_size)

        for idx, name in enumerate(self.branch_names):
            if name not in branch_token_indexes:
                continue
            indexes = branch_token_indexes[name]

            linear_proj = self._get_branch_module('linear_proj', idx)
            branch_attn = attn_out[indexes].unsqueeze(1)  # [n, 1, nH_local*D]
            proj_out, _ = apply_module(linear_proj)(branch_attn)
            out_buf[indexes] = proj_out.squeeze(1)

        # Restore [S, 1, H]
        if squeeze:
            out_buf = out_buf.unsqueeze(1)

        # Bias is typically None for RowParallelLinear without bias
        return out_buf, None
