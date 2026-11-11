# Copyright (c) 2026, BAAI. All rights reserved.
"""MLA construction variants, absorption setup and sharded state dict."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention, MLASelfAttention
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _mla_config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        kv_lora_rank=32,
        qk_head_dim=64,
        v_head_dim=64,
        qk_pos_emb_head_dim=32,
        rope_type="rope",
        rotary_base=10000,
        original_max_position_embeddings=32,
    )
    defaults.update(overrides)
    return MLATransformerConfig(**defaults)


def _mla_submodules(**spec_kwargs):
    return get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            multi_latent_attention=True, **spec_kwargs
        ).self_attention
    )


def test_mla_rejects_unsupported_rope_type(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _mla_config(rope_type="bogus")

    with pytest.raises(ValueError, match="Unsupported RoPE type"):
        MLASelfAttention(
            config, _mla_submodules(), layer_number=1, attn_mask_type=AttnMaskType.causal
        )


def test_mla_without_q_lora_rank_projects_queries_directly(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = MLASelfAttention(
        _mla_config(q_lora_rank=None),
        _mla_submodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda()
    hidden_states = torch.randn(4, 2, 64, device="cuda")

    query, key, value, q_compressed, kv_compressed = attention.get_query_key_value_tensors(
        hidden_states
    )

    assert hasattr(attention, "linear_q_proj")
    assert not hasattr(attention, "linear_q_down_proj")
    assert q_compressed is hidden_states
    assert query.shape == (4, 2, 4, 96)
    assert key.shape == (4, 2, 4, 96)
    assert value.shape == (4, 2, 4, 64)
    with torch.no_grad():
        projected, _ = apply_module(attention.linear_q_proj)(hidden_states)
    torch.testing.assert_close(
        query[..., :64], projected.view(4, 2, 4, 96)[..., :64], rtol=1e-5, atol=1e-6
    )


def test_mla_rejects_unsupported_down_projections(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    submodules = _mla_submodules()
    submodules.linear_q_down_proj = RowParallelLinear
    with pytest.raises(ValueError, match="Unsupported linear_q_down_proj"):
        MLASelfAttention(
            _mla_config(q_lora_rank=32),
            submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    submodules = _mla_submodules()
    submodules.linear_kv_down_proj = RowParallelLinear
    with pytest.raises(ValueError, match="Unsupported linear_kv_down_proj"):
        MLASelfAttention(
            _mla_config(), submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
        )


def test_mla_forward_with_static_context_caches_mla_shaped_kv(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    from megatron.core.inference.contexts import StaticInferenceContext

    attention = (
        MLASelfAttention(
            _mla_config(), _mla_submodules(), layer_number=1, attn_mask_type=AttnMaskType.causal
        )
        .cuda()
        .eval()
    )
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)
    hidden_states = torch.randn(4, 2, 64, device="cuda")
    attention_mask = torch.ones((1, 1, 4, 4), dtype=bool, device="cuda")

    with torch.no_grad():
        output, bias = attention(hidden_states, attention_mask, inference_context=context)
        direct_output, direct_bias = attention(hidden_states, attention_mask)
        _, key, value, _, _ = attention.get_query_key_value_tensors(hidden_states)

    key_memory, value_memory = context.key_value_memory_dict[1]
    assert key_memory.shape == (8, 2, 4, 96)
    assert value_memory.shape == (8, 2, 4, 64)
    torch.testing.assert_close(key_memory[:4], key, rtol=0.0, atol=0.0)
    torch.testing.assert_close(value_memory[:4], value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output, direct_output, rtol=0.0, atol=0.0)
    assert bias.shape == (64,)
    torch.testing.assert_close(bias, direct_bias, rtol=0.0, atol=0.0)


def test_fused_mla_requires_q_lora_rank(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    submodules = _mla_submodules()
    submodules.linear_qkv_down_proj = ColumnParallelLinear

    with pytest.raises(AssertionError, match="FusedMLASelfAttention requires q_lora_rank"):
        FusedMLASelfAttention(
            _mla_config(q_lora_rank=None),
            submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )


def test_fused_mla_rejects_unsupported_qkv_down_proj(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    submodules = _mla_submodules()
    submodules.linear_qkv_down_proj = RowParallelLinear

    with pytest.raises(ValueError, match="Unsupported linear_qkv_down_proj"):
        FusedMLASelfAttention(
            _mla_config(q_lora_rank=32),
            submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )


def test_fused_mla_sharded_state_dict_splits_fused_weight_by_tp(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    from megatron.core import mpu

    submodules = _mla_submodules()
    submodules.linear_qkv_down_proj = ColumnParallelLinear
    attention = FusedMLASelfAttention(
        _mla_config(
            q_lora_rank=16, kv_lora_rank=32, qk_pos_emb_head_dim=8, tensor_model_parallel_size=2
        ),
        submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    )

    sharded = attention.sharded_state_dict(
        metadata={
            "dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True),
            "non_homogeneous_layers": False,
        }
    )

    assert "linear_qkv_down_proj.weight" not in sharded
    q_tensor = sharded["linear_q_down_proj.weight"]
    kv_tensor = sharded["linear_kv_down_proj.weight"]
    assert q_tensor.local_shape == (8, 64)
    assert q_tensor.global_shape == (16, 64)
    assert kv_tensor.local_shape == (20, 64)
    assert kv_tensor.global_shape == (40, 64)


def test_mla_set_for_recompute_marks_kv_down_proj_without_q_lora_rank(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = MLASelfAttention(
        _mla_config(q_lora_rank=None),
        _mla_submodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    )

    attention.set_for_recompute_input_layernorm()

    assert attention.linear_kv_down_proj.save_original_input is True
    assert not hasattr(attention, "linear_q_down_proj")


def test_mla_set_for_recompute_marks_q_and_kv_down_projections(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = MLASelfAttention(
        _mla_config(q_lora_rank=32),
        _mla_submodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    )

    attention.set_for_recompute_input_layernorm()

    assert attention.linear_q_down_proj.save_original_input is True
    assert attention.linear_kv_down_proj.save_original_input is True


def test_mla_clip_qk_without_q_lora_rank_clips_query_projection(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = MLASelfAttention(
        _mla_config(q_lora_rank=None, qk_clip=True, qk_clip_threshold=100.0, qk_clip_alpha=0.5),
        _mla_submodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda()
    original_weight = attention.linear_q_proj.weight.data.clone()
    attention.core_attention.current_max_attn_logits = torch.tensor(
        [150.0, 160.0, 170.0, 180.0], device="cuda"
    )

    attention.clip_qk()

    max_logits = torch.tensor([150.0, 160.0, 170.0, 180.0], device="cuda")
    eta = torch.clamp(100.0 / max_logits, max=1.0).view(4, 1, 1)
    expected = original_weight.clone().view(4, 96, -1)
    expected[:, :64] = expected[:, :64] * eta**0.5
    expected[:, 64:] = expected[:, 64:] * eta
    torch.testing.assert_close(
        attention.linear_q_proj.weight.data, expected.view_as(original_weight), rtol=1e-6, atol=1e-7
    )
    assert attention.core_attention.current_max_attn_logits is None
    assert attention.qk_clip_balancing_eta.shape == (4, 1, 1)
    assert torch.all(attention.qk_clip_balancing_eta <= 1.0)


def test_mla_prepare_for_absorption_splits_kv_up_projection(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = MLASelfAttention(
        _mla_config(q_lora_rank=32, cache_mla_latents=True),
        _mla_submodules(qk_layernorm=True),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda()

    attention.prepare_for_absorption()

    assert not hasattr(attention, "linear_kv_up_proj")
    assert attention.linear_kv_up_proj_linear.weight.shape == (512, 32)
    assert attention.up_k_weight.shape == (4, 64, 32)
    assert attention.up_v_weight.shape == (4, 64, 32)
    assert not isinstance(attention.kv_layernorm, IdentityOp)


def test_mla_uncompress_kv_from_cache_expands_latent_kv(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = MLASelfAttention(
        _mla_config(q_lora_rank=32, cache_mla_latents=True),
        _mla_submodules(qk_layernorm=True),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda()
    attention.prepare_for_absorption()
    kv_cached = torch.randn(4, 2, 64, device="cuda")

    key, value = attention.uncompress_kv_from_cache(kv_cached)

    assert key.shape == (4, 2, 4, 96)
    assert value.shape == (4, 2, 4, 64)
    with torch.no_grad():
        kv, _ = attention.linear_kv_up_proj_linear(kv_cached[..., :32])
        kv = kv.view(4, 2, 4, 128)
        expected_key, expected_value = torch.split(kv, [64, 64], dim=3)
    torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(key[..., :64], expected_key, rtol=1e-5, atol=1e-6)
