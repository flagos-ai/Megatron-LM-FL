# Copyright (c) 2026, BAAI. All rights reserved.
"""Static KV cache bookkeeping and flash decode/prefill attention kernels."""

import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb_with_cos_sin
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _self_attention(config):
    submodules = get_submodules(
        get_gpt_layer_local_submodules(
            qk_layernorm=config.qk_layernorm, qk_l2_norm=config.qk_l2_norm
        ).self_attention
    )
    return SelfAttention(config, submodules, layer_number=1, attn_mask_type=AttnMaskType.padding)


def test_adjust_key_value_reuses_and_slices_static_kv_cache(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = _self_attention(_config()).cuda().eval()
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)
    query = torch.randn(4, 2, 4, 8, device="cuda")
    key = torch.randn(4, 2, 4, 8, device="cuda")
    value = torch.randn(4, 2, 4, 8, device="cuda")
    q_pos_emb = torch.randn(8, 1, 1, 8, device="cuda")
    k_pos_emb = torch.randn(8, 1, 1, 8, device="cuda")

    out = attention._adjust_key_value_for_inference(
        context, query, key, value, (q_pos_emb, k_pos_emb)
    )

    out_query, out_key, out_value, out_rope, mask_type, block_table = out
    assert mask_type == AttnMaskType.padding
    assert block_table is None
    assert out_query.shape == (4, 2, 4, 8)
    assert out_key.shape == (4, 2, 4, 8)
    key_memory, value_memory = context.key_value_memory_dict[1]
    assert key_memory.shape == (8, 2, 4, 8)
    assert torch.equal(key_memory[:4], key)
    assert torch.equal(value_memory[:4], value)
    assert out_rope[0].shape == (4, 1, 1, 8)
    assert torch.equal(out_rope[0], q_pos_emb[:4])
    assert torch.equal(out_rope[1], k_pos_emb[:4])

    context.sequence_len_offset = 4
    context.enable_decode_mode()
    decode_key = torch.randn(1, 2, 4, 8, device="cuda")
    decode_value = torch.randn(1, 2, 4, 8, device="cuda")
    decode_query = torch.randn(1, 2, 4, 8, device="cuda")

    out = attention._adjust_key_value_for_inference(
        context, decode_query, decode_key, decode_value, (q_pos_emb, k_pos_emb)
    )

    out_query, out_key, out_value, out_rope, mask_type, block_table = out
    assert mask_type == AttnMaskType.no_mask
    assert out_key.shape == (5, 2, 4, 8)
    assert torch.equal(key_memory[4:5], decode_key)
    assert torch.equal(out_key, key_memory[:5])
    assert torch.equal(out_rope[0], q_pos_emb[4:5])
    assert torch.equal(out_rope[1], k_pos_emb[:5])


def test_adjust_key_value_applies_rope_for_flash_decode_prefill(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = _self_attention(_config(flash_decode=True)).cuda().eval()
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)
    query = torch.randn(4, 2, 4, 8, device="cuda")
    key = torch.randn(4, 2, 4, 8, device="cuda")
    value = torch.randn(4, 2, 4, 8, device="cuda")
    rotary_cos = torch.randn(8, 4, device="cuda")
    rotary_sin = torch.randn(8, 4, device="cuda")

    out_query, out_key, out_value, out_rope, mask_type, block_table = (
        attention._adjust_key_value_for_inference(
            context, query, key, value, None, rotary_pos_cos=rotary_cos, rotary_pos_sin=rotary_sin
        )
    )

    assert out_rope is None
    expected_key = apply_rotary_pos_emb_with_cos_sin(
        key, rotary_cos[:4], rotary_sin[:4], rotary_interleaved=False
    )
    expected_query = apply_rotary_pos_emb_with_cos_sin(
        query, rotary_cos[:4], rotary_sin[:4], rotary_interleaved=False
    )
    torch.testing.assert_close(out_key, expected_key, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_query, expected_query, rtol=1e-5, atol=1e-6)
    key_memory, _ = context.key_value_memory_dict[1]
    torch.testing.assert_close(key_memory[:4], expected_key, rtol=1e-5, atol=1e-6)


def test_adjust_key_value_applies_rope_for_flash_decode_decode_step(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = _self_attention(_config(flash_decode=True)).cuda().eval()
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)
    context.sequence_len_offset = 4
    context.enable_decode_mode()
    query = torch.randn(1, 2, 4, 8, device="cuda")
    key = torch.randn(1, 2, 4, 8, device="cuda")
    value = torch.randn(1, 2, 4, 8, device="cuda")
    rotary_cos = torch.randn(8, 4, device="cuda")
    rotary_sin = torch.randn(8, 4, device="cuda")

    out_query, out_key, _, _, mask_type, _ = attention._adjust_key_value_for_inference(
        context, query, key, value, None, rotary_pos_cos=rotary_cos, rotary_pos_sin=rotary_sin
    )

    assert mask_type == AttnMaskType.no_mask
    expected_key = apply_rotary_pos_emb_with_cos_sin(
        key, rotary_cos[4:5], rotary_sin[4:5], rotary_interleaved=False
    )
    expected_query = apply_rotary_pos_emb_with_cos_sin(
        query, rotary_cos[4:5], rotary_sin[4:5], rotary_interleaved=False
    )
    torch.testing.assert_close(out_key[4:5], expected_key, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_query, expected_query, rtol=1e-5, atol=1e-6)


def test_flash_decode_returns_new_value_and_caches_rotated_key(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = (
        _self_attention(_config(flash_decode=True, bf16=True, params_dtype=torch.bfloat16))
        .cuda()
        .eval()
    )
    batch_size, heads, head_dim = 2, 4, 8
    query_layer = torch.randn(1, batch_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_layer = torch.randn(1, batch_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_layer = torch.randn(1, batch_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_cache = torch.zeros(4, batch_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    rotary_cos = torch.zeros(1, head_dim // 2, device="cuda", dtype=torch.bfloat16)
    rotary_sin = torch.ones(1, head_dim // 2, device="cuda", dtype=torch.bfloat16)

    out = attention.flash_decode(
        sequence_len_offset=torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
        query_layer=query_layer,
        key_layer=key_layer,
        value_layer=value_layer,
        inference_key_memory=key_cache,
        inference_value_memory=value_cache,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=False,
    )

    assert out.shape == (batch_size, 1, heads, head_dim)
    torch.testing.assert_close(out[:, 0], value_layer[0], rtol=1e-2, atol=1e-2)
    expected_key = torch.cat([-key_layer[..., 4:], key_layer[..., :4]], dim=-1)
    torch.testing.assert_close(key_cache[0], expected_key[0], rtol=1e-2, atol=1e-2)


def test_flash_decode_forward_matches_unpacked_kv_attention(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(bf16=True, params_dtype=torch.bfloat16)
    torch.manual_seed(7)
    plain = _self_attention(config).cuda().eval()
    torch.manual_seed(7)
    flash = (
        _self_attention(_config(bf16=True, params_dtype=torch.bfloat16, flash_decode=True))
        .cuda()
        .eval()
    )
    prefill_hidden = torch.randn(4, 2, 32, device="cuda", dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 2, 32, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        _, key_prefill, value_prefill = plain.get_query_key_value_tensors(prefill_hidden)
    key_memory = torch.zeros(8, 2, 4, 8, device="cuda", dtype=torch.bfloat16)
    value_memory = torch.zeros_like(key_memory)
    key_memory[:4] = key_prefill
    value_memory[:4] = value_prefill

    plain_context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)
    flash_context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)
    plain_context.key_value_memory_dict[1] = (key_memory.clone(), value_memory.clone())
    flash_context.key_value_memory_dict[1] = (key_memory.clone(), value_memory.clone())
    for context in (plain_context, flash_context):
        context.sequence_len_offset = 4
        context.enable_decode_mode()

    with torch.no_grad():
        with InferenceMode.active():
            flash_output, _ = flash(
                decode_hidden,
                None,
                inference_context=flash_context,
                rotary_pos_cos=torch.ones(8, 4, device="cuda", dtype=torch.bfloat16),
                rotary_pos_sin=torch.zeros(8, 4, device="cuda", dtype=torch.bfloat16),
                sequence_len_offset=torch.full((2,), 4, dtype=torch.int32, device="cuda"),
            )
        plain_output, _ = plain(
            decode_hidden, None, inference_context=plain_context, rotary_pos_emb=None
        )

    assert flash_output.shape == (1, 2, 32)
    assert plain_output.shape == (1, 2, 32)
    torch.testing.assert_close(flash_output, plain_output, rtol=1.6e-2, atol=1e-2)


def test_flash_decode_and_prefill_over_paged_kv_returns_shared_value(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = _self_attention(_config(bf16=True, params_dtype=torch.bfloat16)).cuda().eval()
    heads, head_dim, block_size = 2, 8, 4
    shared_value = torch.randn(heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_cache = torch.randn(1, block_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_cache = shared_value.expand(1, block_size, heads, head_dim).contiguous().clone()
    query = torch.randn(4, 1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor([4], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")

    out = attention.flash_decode_and_prefill(
        query,
        key_cache,
        value_cache,
        4,
        4,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_k,
        block_table,
        is_decode_only=False,
    )

    assert out.shape == (4, 1, heads, head_dim)
    torch.testing.assert_close(
        out, shared_value.expand(4, 1, heads, head_dim), rtol=1e-2, atol=1e-2
    )


def test_flash_decode_and_prefill_decode_only_over_paged_kv(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = _self_attention(_config(bf16=True, params_dtype=torch.bfloat16)).cuda().eval()
    heads, head_dim, block_size = 2, 8, 4
    shared_value = torch.randn(heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_cache = torch.randn(2, block_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_cache = shared_value.expand(2, block_size, heads, head_dim).contiguous().clone()
    query = torch.randn(2, 1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor([4], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")

    out = attention.flash_decode_and_prefill(
        query,
        key_cache,
        value_cache,
        2,
        4,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_k,
        block_table,
        is_decode_only=True,
    )

    assert out.shape == (2, 1, heads, head_dim)
    torch.testing.assert_close(
        out, shared_value.expand(2, 1, heads, head_dim), rtol=1e-2, atol=1e-2
    )
