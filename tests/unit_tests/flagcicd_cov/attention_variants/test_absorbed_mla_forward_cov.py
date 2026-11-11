# Copyright (c) 2026, BAAI. All rights reserved.
"""AbsorbedMLASelfAttention forward and backward: direct-projection query path, fused vs unfused Yarn RoPE parity, packed THD cu_seqlens handling, selective recompute against eager forward and gradients, and deferred weight-gradient flush after backward_dw."""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
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


def _make_hidden(sq, b, config, requires_grad=False):
    device = torch.cuda.current_device()
    hidden = torch.randn(sq, b, config.hidden_size, dtype=torch.bfloat16, device=device)
    if requires_grad:
        hidden.requires_grad_(True)
    return hidden


class TestAbsorbedMLAForward:

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

    def test_rope_forward_without_q_lora_projects_query_directly(self):
        config = _make_config(rope_type='rope', q_lora_rank=None)
        attn = _build_attention(config).cuda()
        attn.train()
        hidden = _make_hidden(32, 2, config, requires_grad=True)
        hidden_probe = hidden.detach()
        q_absorbed, kv_compressed, q_compressed = attn.get_query_key_value_tensors(
            hidden_probe, packed_seq_params=None
        )
        assert q_compressed is hidden_probe
        assert q_absorbed.shape == (
            32,
            2,
            config.num_attention_heads,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
        )
        assert kv_compressed.shape == (32, 2, 1, config.kv_lora_rank + config.qk_pos_emb_head_dim)
        output, bias = attn(hidden, attention_mask=None)
        assert output.shape == (32, 2, config.hidden_size)
        assert bias is None
        output.float().sum().backward()
        assert torch.isfinite(hidden.grad).all()
        for name, param in attn.named_parameters():
            assert torch.isfinite(param.grad).all(), name

    def test_fused_rope_forward_matches_unfused_yarn(self):
        fused_config = _make_config(apply_rope_fusion=True)
        unfused_config = _make_config(apply_rope_fusion=False)
        attn_fused = _build_attention(fused_config).cuda()
        attn_unfused = _build_attention(unfused_config).cuda()
        attn_unfused.load_state_dict(attn_fused.state_dict())
        attn_fused.train()
        attn_unfused.train()
        hidden = _make_hidden(32, 2, fused_config, requires_grad=True)
        out_fused, _ = attn_fused(hidden, attention_mask=None)
        out_unfused, _ = attn_unfused(hidden.detach().clone(), attention_mask=None)
        assert out_fused.shape == (32, 2, fused_config.hidden_size)
        torch.testing.assert_close(out_fused, out_unfused, rtol=3e-2, atol=3e-2)
        out_fused.float().square().mean().backward()
        assert torch.isfinite(hidden.grad).all()
        for name, param in attn_fused.named_parameters():
            assert torch.isfinite(param.grad).all(), name

    def test_thd_forward_without_padded_cu_seqlens_matches_padded(self):
        config = _make_config()
        attn = _build_attention(config).cuda()
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
        out_plain, _ = attn(hidden, attention_mask=None, packed_seq_params=plain)
        out_padded, _ = attn(hidden, attention_mask=None, packed_seq_params=padded)
        assert out_plain.shape == (24, 1, config.hidden_size)
        torch.testing.assert_close(out_plain, out_padded, rtol=0, atol=0)


class TestAbsorbedMLARecompute:

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

    def test_selective_recompute_matches_eager_forward_and_backward(self):
        recompute_config = _make_config(
            recompute_granularity='selective', recompute_modules=['mla_up_proj', 'core_attn']
        )
        eager_config = _make_config()
        attn_recompute = _build_attention(recompute_config).cuda()
        attn_eager = _build_attention(eager_config).cuda()
        attn_eager.load_state_dict(attn_recompute.state_dict())
        attn_recompute.train()
        attn_eager.train()
        assert attn_recompute.recompute_up_proj is True
        assert attn_eager.recompute_up_proj is False

        hidden = _make_hidden(32, 2, recompute_config)
        out_recompute, _ = attn_recompute(
            hidden.detach().clone().requires_grad_(True), attention_mask=None
        )
        assert attn_recompute.qkv_up_checkpoint is None
        out_eager, _ = attn_eager(hidden.detach().clone().requires_grad_(True), attention_mask=None)
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


class TestAbsorbedMLADelayedWgrad:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'q_lora_rank', [pytest.param(None, id='direct_q'), pytest.param(32, id='low_rank_q')]
    )
    def test_backward_dw_flushes_deferred_weight_gradients(self, q_lora_rank):
        def build(delayed):
            config = _make_config(q_lora_rank=q_lora_rank)
            config.delay_wgrad_compute = delayed
            return AbsorbedMLASelfAttention(
                config=config,
                submodules=_make_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=None,
            ).cuda()

        eager = build(False)
        delayed = build(True)
        delayed.load_state_dict(eager.state_dict())
        eager.train()
        delayed.train()

        hidden = _make_hidden(32, 2, eager.config)
        outputs = []
        input_grads = []
        for model in (eager, delayed):
            inputs = hidden.detach().clone().requires_grad_(True)
            output, _ = model(inputs, attention_mask=None)
            output.float().square().mean().backward()
            model.backward_dw()
            outputs.append(output.detach())
            input_grads.append(inputs.grad)

        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(input_grads[0], input_grads[1], rtol=1e-2, atol=1e-5)
        for (name, eager_param), (_, delayed_param) in zip(
            eager.named_parameters(), delayed.named_parameters()
        ):
            torch.testing.assert_close(
                delayed_param.grad, eager_param.grad, rtol=1e-2, atol=1e-5, msg=name
            )
