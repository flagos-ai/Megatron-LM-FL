# Copyright (c) 2026, BAAI. All rights reserved.
"""DSv4 hybrid attention: fused q RMS norm, MTP compress-ratio and yarn rotary selection, activation-offload and selective-recompute forward parity, thd packing, and recompute input-layernorm flags."""

import pytest
import torch

from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import deepseek_v4_hybrid_attention
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_pg_size
from tests.unit_tests.test_utilities import Utils


class _ReferenceThdCoreAttention(torch.nn.Module):
    """Causal MQA core attention standing in for CompressedSparseAttention under thd packing."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs.get('config')
        self.softmax_scale = kwargs.get('softmax_scale') or config.v_head_dim ** -0.5

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        packed_seq_params=None,
        x=None,
        qr=None,
        **kwargs,
    ):
        cu_seqlens = packed_seq_params.cu_seqlens_q
        query_heads = query.float()
        kv = key.squeeze(1).float()
        outputs = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            seq_q = query_heads[start:end]
            seq_k = kv[start:end]
            seq_len = end - start
            scores = torch.einsum('tnd,sd->tns', seq_q, seq_k) * self.softmax_scale
            causal = torch.triu(
                torch.full(
                    (seq_len, seq_len), float('-inf'), dtype=scores.dtype, device=scores.device
                ),
                diagonal=1,
            ).unsqueeze(1)
            probs = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
            outputs.append(torch.einsum('tns,sc->tnc', probs, seq_k).to(query.dtype))
        out = torch.cat(outputs, dim=0)
        return out.reshape(out.shape[0], -1)


def _make_config(**overrides):
    kwargs = dict(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=8,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        o_groups=4,
        o_lora_rank=32,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=8,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=0.0,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _build_attention(config, layer_number, pg_collection, is_mtp_layer=False):
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )
    from megatron.core.transformer.spec_utils import build_module

    spec = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())
    return build_module(
        spec,
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
        is_mtp_layer=is_mtp_layer,
    )


def _make_thd_submodules():
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    backend = TESpecProvider()
    return DSv4HybridSelfAttentionSubmodules(
        linear_q_down_proj=backend.linear(),
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_proj=backend.column_parallel_linear(),
        core_attention=_ReferenceThdCoreAttention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
        kv_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
    )


def _make_hidden(sq, b, config, requires_grad=False):
    device = torch.cuda.current_device()
    hidden = torch.randn(sq, b, config.hidden_size, dtype=torch.bfloat16, device=device)
    if requires_grad:
        hidden.requires_grad_(True)
    return hidden


class TestDsv4QRmsNorm:

    def test_q_rms_norm_matches_eager_reference(self):
        torch.manual_seed(0)
        device = torch.cuda.current_device()
        x = torch.randn(12, 4, 32, dtype=torch.float32, device=device)
        eps = 1e-5
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        compiled = deepseek_v4_hybrid_attention._q_rms_norm(x, eps)
        torch.testing.assert_close(compiled, expected, rtol=1e-5, atol=1e-6)
        eager = deepseek_v4_hybrid_attention._q_rms_norm.__wrapped__(x, eps)
        torch.testing.assert_close(eager, expected, rtol=1e-5, atol=1e-6)


class TestDsv4Construction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        from megatron.core.process_groups_config import ProcessGroupCollection

        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        yield
        Utils.destroy_model_parallel()

    def test_mtp_layer_selects_compress_ratio_after_num_layers(self):
        config = _make_config(num_layers=2, mtp_num_layers=1, csa_compress_ratios=[0, 4, 128])
        attn = _build_attention(config, layer_number=1, pg_collection=self.pg, is_mtp_layer=True)
        assert isinstance(attn.rotary_pos_emb, YarnRotaryEmbedding)
        assert attn.core_attention.compress_ratio == 128
        dim = config.qk_pos_emb_head_dim
        expected_base = config.csa_compress_rotary_base
        recomputed_inv_freq = 1.0 / (
            expected_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        torch.testing.assert_close(
            attn.rotary_pos_emb.inv_freq.cpu(), recomputed_inv_freq, rtol=1e-5, atol=1e-5
        )

    def test_self_attention_resolves_default_process_groups(self):
        config = _make_config()
        attn = _build_attention(config, layer_number=1, pg_collection=None)
        assert isinstance(attn, DSv4HybridSelfAttention)
        assert get_pg_size(attn.pg_collection.tp) == 1

    def test_rejects_non_te_linear_q_down_proj(self):
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        backend = TESpecProvider()
        config = _make_config()
        submodules = _make_thd_submodules()
        submodules.linear_q_down_proj = backend.column_parallel_linear()
        with pytest.raises(ValueError, match='Unsupported linear_q_down_proj'):
            DSv4HybridSelfAttention(
                config=config,
                submodules=submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=None,
            )


class TestDsv4Forward:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        from megatron.core.process_groups_config import ProcessGroupCollection

        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        yield
        Utils.destroy_model_parallel()

    def test_training_forward_with_activation_offload_matches_eager(self):
        offload_config = _make_config(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn', 'attn_proj'],
            cuda_graph_impl='transformer_engine',
        )
        eager_config = _make_config()
        attn_offload = _build_attention(offload_config, 1, self.pg).cuda()
        attn_eager = _build_attention(eager_config, 1, self.pg).cuda()
        attn_eager.load_state_dict(attn_offload.state_dict())
        attn_offload.train()
        attn_eager.train()
        assert attn_offload.offload_core_attention is True
        assert attn_offload.offload_attn_proj is True
        assert attn_eager.offload_core_attention is False
        assert attn_eager.offload_attn_proj is False

        hidden = _make_hidden(32, 2, offload_config)
        out_offload, _ = attn_offload(hidden_states=hidden, attention_mask=None)
        out_eager, _ = attn_eager(hidden_states=hidden.detach().clone(), attention_mask=None)
        assert out_offload.shape == (32, 2, offload_config.hidden_size)
        assert torch.isfinite(out_offload).all()
        torch.testing.assert_close(out_offload, out_eager, rtol=0, atol=0)

    def test_selective_recompute_of_mla_up_proj_matches_eager(self):
        recompute_config = _make_config(
            recompute_granularity='selective', recompute_modules=['mla_up_proj']
        )
        eager_config = _make_config()
        attn_recompute = _build_attention(recompute_config, 1, self.pg).cuda()
        attn_eager = _build_attention(eager_config, 1, self.pg).cuda()
        attn_eager.load_state_dict(attn_recompute.state_dict())
        attn_recompute.train()
        attn_eager.train()
        assert attn_recompute.recompute_up_proj is True
        assert attn_eager.recompute_up_proj is False

        hidden = _make_hidden(32, 2, recompute_config)
        out_recompute, _ = attn_recompute(
            hidden_states=hidden.detach().clone().requires_grad_(True), attention_mask=None
        )
        assert attn_recompute.qkv_up_checkpoint is None
        out_eager, _ = attn_eager(
            hidden_states=hidden.detach().clone().requires_grad_(True), attention_mask=None
        )
        assert out_recompute.shape == (32, 2, recompute_config.hidden_size)
        torch.testing.assert_close(out_recompute, out_eager, rtol=0, atol=0)

        out_recompute.float().square().mean().backward()
        out_eager.float().square().mean().backward()
        for (name, recompute_param), (_, eager_param) in zip(
            attn_recompute.named_parameters(), attn_eager.named_parameters()
        ):
            torch.testing.assert_close(
                recompute_param.grad, eager_param.grad, rtol=1e-5, atol=1e-6, msg=name
            )

    def test_thd_forward_without_padded_cu_seqlens_matches_padded(self):
        config = _make_config()
        attn = DSv4HybridSelfAttention(
            config=config,
            submodules=_make_thd_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=self.pg,
        ).cuda()
        device = torch.cuda.current_device()
        cu_seqlens = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=device)
        plain = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_kv=8,
        )
        padded = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_kv=8,
        )
        hidden = _make_hidden(24, 1, config)
        out_plain, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=plain)
        out_padded, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=padded)
        assert out_plain.shape == (24, 1, config.hidden_size)
        assert torch.isfinite(out_plain).all()
        torch.testing.assert_close(out_plain, out_padded, rtol=0, atol=0)

    def test_get_query_key_value_tensors_rejects_dynamic_context_parallel(self):
        config = _make_config()
        attn = _build_attention(config, 1, self.pg).cuda()
        packed = PackedSeqParams(local_cp_size=2)
        hidden = _make_hidden(32, 2, config)
        with pytest.raises(AssertionError, match='dynamic context parallel'):
            attn.get_query_key_value_tensors(hidden, packed_seq_params=packed)

    def test_set_for_recompute_input_layernorm_flags_q_down_and_kv_projections(self):
        config = _make_config()
        attn = _build_attention(config, 1, self.pg).cuda()
        assert [
            name
            for name, module in attn.named_modules()
            if getattr(module, 'save_original_input', False)
        ] == []
        attn.set_for_recompute_input_layernorm()
        flagged = [
            name
            for name, module in attn.named_modules()
            if getattr(module, 'save_original_input', False)
        ]
        assert flagged == ['linear_q_down_proj', 'linear_kv_proj']
