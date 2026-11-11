# Copyright (c) 2026, BAAI. All rights reserved.
"""AbsorbedMLASelfAttention construction: RoPE type selection, direct vs low-rank query projection parameter sets, unsupported-projection rejection, recompute-input layernorm flagging, and per-head K/V weight interleaving."""

import pytest
import torch

from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils


class _ReferenceMQACoreAttention(torch.nn.Module):
    """Causal MQA core attention standing in for the fused attention backend."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.softmax_scale = kwargs.get('softmax_scale')
        self.k_channels = kwargs.get('k_channels')
        self.v_channels = kwargs.get('v_channels')
        self.pg_collection = kwargs.get('pg_collection')

    def forward(
        self,
        q,
        k,
        v,
        hidden_states=None,
        q_compressed=None,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            return self._forward_thd(q, k, packed_seq_params)
        sq, b, n_heads, _ = q.shape
        kv = k.squeeze(2)
        qh = q.permute(1, 2, 0, 3).float()
        kh = kv.permute(1, 0, 2).float()
        scores = torch.einsum('bnsd,btd->bnst', qh, kh) * self.softmax_scale
        causal = torch.triu(
            torch.full((sq, sq), float('-inf'), dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        probs = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
        v_emb = kv[..., : self.v_channels].permute(1, 0, 2).float()
        out = torch.einsum('bnst,btc->bnsc', probs, v_emb)
        return out.permute(2, 0, 1, 3).to(q.dtype).reshape(sq, b, n_heads * self.v_channels)

    def _forward_thd(self, q, k, packed_seq_params):
        cu_seqlens = packed_seq_params.cu_seqlens_q
        qh = q.float()
        kv = k.squeeze(1).float()
        outputs = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            seq_q = qh[start:end]
            seq_k = kv[start:end]
            scores = torch.einsum('tnd,sd->tns', seq_q, seq_k) * self.softmax_scale
            seq_len = end - start
            causal = torch.triu(
                torch.full(
                    (seq_len, seq_len), float('-inf'), dtype=scores.dtype, device=scores.device
                ),
                diagonal=1,
            ).unsqueeze(1)
            probs = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
            out = torch.einsum('tns,sc->tnc', probs, seq_k[..., : self.v_channels])
            outputs.append(out.to(q.dtype))
        total = torch.cat(outputs, dim=0)
        return total.reshape(total.shape[0], -1)


def _make_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        rope_type='yarn',
        rotary_base=10000,
        rotary_percent=1.0,
        rotary_scaling_factor=40,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=0.0,
        apply_rope_fusion=False,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _make_submodules(core_attention=_ReferenceMQACoreAttention, kv_down_proj=None):
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    backend = TESpecProvider()
    return AbsorbedMLASelfAttentionSubmodules(
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_down_proj=backend.linear(),
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_down_proj=kv_down_proj or backend.linear(),
        linear_k_up_proj=backend.column_parallel_linear(),
        linear_v_up_proj=backend.column_parallel_linear(),
        core_attention=core_attention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
        kv_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
    )


def _build_attention(config, core_attention=_ReferenceMQACoreAttention):
    return AbsorbedMLASelfAttention(
        config=config,
        submodules=_make_submodules(core_attention=core_attention),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=None,
    )


class TestAbsorbedMLAConstruction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    def test_builds_rope_rotary_when_rope_type_is_rope(self):
        config = _make_config(rope_type='rope')
        attn = _build_attention(config)
        assert isinstance(attn.rotary_pos_emb, RotaryEmbedding)
        assert attn.q_head_dim == config.qk_head_dim + config.qk_pos_emb_head_dim
        assert attn.query_projection_size == config.v_head_dim * config.num_attention_heads
        assert attn.softmax_scale == pytest.approx(attn.q_head_dim**-0.5)

    def test_rejects_unsupported_rope_type(self):
        config = _make_config(rope_type='bogus')
        with pytest.raises(ValueError, match='Unsupported RoPE type'):
            _build_attention(config)

    def test_builds_direct_query_projection_when_q_lora_rank_is_none(self):
        config = _make_config(q_lora_rank=None)
        attn = _build_attention(config)
        assert attn.linear_q_proj.weight.shape == (
            config.num_attention_heads * attn.q_head_dim,
            config.hidden_size,
        )
        low_rank = _build_attention(_make_config())
        direct_names = {name for name, _ in attn.named_parameters()}
        low_rank_names = {name for name, _ in low_rank.named_parameters()}
        assert direct_names - low_rank_names == {'linear_q_proj.weight'}
        assert low_rank_names - direct_names == {
            'linear_q_down_proj.weight',
            'linear_q_up_proj.weight',
            'q_layernorm.weight',
        }

    def test_rejects_unsupported_linear_q_down_proj(self):
        config = _make_config()
        submodules = _make_submodules()
        submodules.linear_q_down_proj = torch.nn.Linear
        with pytest.raises(ValueError, match='Unsupported linear_q_down_proj'):
            AbsorbedMLASelfAttention(
                config=config,
                submodules=submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=None,
            )

    def test_rejects_unsupported_linear_kv_down_proj(self):
        config = _make_config()
        submodules = _make_submodules(kv_down_proj=torch.nn.Linear)
        with pytest.raises(ValueError, match='Unsupported linear_kv_down_proj'):
            AbsorbedMLASelfAttention(
                config=config,
                submodules=submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=None,
            )


class TestAbsorbedMLAMisc:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'q_lora_rank', [pytest.param(None, id='direct_q'), pytest.param(32, id='low_rank_q')]
    )
    def test_set_for_recompute_input_layernorm_flags_down_projections_only(self, q_lora_rank):
        attn = _build_attention(_make_config(q_lora_rank=q_lora_rank)).cuda()
        flagged_before = [
            name
            for name, module in attn.named_modules()
            if getattr(module, 'save_original_input', False)
        ]
        assert flagged_before == []
        attn.set_for_recompute_input_layernorm()
        flagged_after = [
            name
            for name, module in attn.named_modules()
            if getattr(module, 'save_original_input', False)
        ]
        if q_lora_rank is None:
            assert flagged_after == ['linear_kv_down_proj']
        else:
            assert flagged_after == ['linear_q_down_proj', 'linear_kv_down_proj']

    def test_clip_qk_is_not_implemented(self):
        attn = _build_attention(_make_config()).cuda()
        with pytest.raises(NotImplementedError, match='clip_qk is not implemented for AbsorbedMLA'):
            attn.clip_qk()

    def test_combine_kv_weights_interleaves_k_and_v_per_head(self):
        config = _make_config()
        attn = _build_attention(config)
        device = torch.cuda.current_device()
        n = attn.num_attention_heads_per_partition
        qk_dim = config.qk_head_dim
        v_dim = config.v_head_dim
        lora = config.kv_lora_rank
        k_weight = torch.arange(n * qk_dim * lora, dtype=torch.float32, device=device).reshape(
            n * qk_dim, lora
        )
        v_weight = (
            torch.arange(n * v_dim * lora, dtype=torch.float32, device=device).reshape(
                n * v_dim, lora
            )
            + 10 * n * qk_dim * lora
        )
        combined = attn._combine_kv_weights(k_weight, v_weight)
        expected = torch.cat(
            [k_weight.view(n, qk_dim, lora), v_weight.view(n, v_dim, lora)], dim=1
        ).view(n * (qk_dim + v_dim), lora)
        assert combined.shape == (n * (qk_dim + v_dim), lora)
        assert torch.equal(combined, expected)
