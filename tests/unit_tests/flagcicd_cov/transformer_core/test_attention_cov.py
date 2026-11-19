# Copyright (c) 2026, BAAI. All rights reserved.
"""Attention construction variants, QKV derivation and cross attention."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.parallel_state import get_data_parallel_rank, get_tensor_model_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import (
    Attention,
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
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


def _self_attention(config, layer_number=1, attn_mask_type=AttnMaskType.causal, **kwargs):
    submodules = get_submodules(
        get_gpt_layer_local_submodules(
            qk_layernorm=config.qk_layernorm, qk_l2_norm=config.qk_l2_norm
        ).self_attention
    )
    return SelfAttention(
        config, submodules, layer_number=layer_number, attn_mask_type=attn_mask_type, **kwargs
    )


def test_attention_set_for_recompute_and_clip_qk_are_not_implemented(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    attention = _self_attention(_config())

    with pytest.raises(NotImplementedError, match="set_for_recompute_input_layernorm"):
        Attention.set_for_recompute_input_layernorm(attention)
    with pytest.raises(NotImplementedError, match="clip_qk is not implemented"):
        Attention.clip_qk(attention)


def test_self_attention_spec_k_layernorm_without_qk_norm_raises(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    submodules = get_submodules(get_gpt_layer_local_submodules(qk_l2_norm=True).self_attention)
    submodules.q_layernorm = None

    with pytest.raises(ValueError, match="spec sets k_layernorm="):
        SelfAttention(_config(), submodules, layer_number=1)


def test_self_attention_gated_split_qkv_false_raises(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(attention_output_gate=True)
    attention = _self_attention(config)
    hidden_states = torch.randn(4, 2, 32)

    with pytest.raises(ValueError, match="split_qkv not supported for gated attention"):
        attention.get_query_key_value_tensors(hidden_states, output_gate=True, split_qkv=False)


def test_self_attention_qk_layernorm_hidden_dim_norms_flattened_qkv(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(qk_layernorm=True, qk_layernorm_hidden_dim=True)
    attention = _self_attention(config).cuda()
    hidden_states = torch.randn(4, 2, 32, device="cuda")

    query, key, value = attention.get_query_key_value_tensors(hidden_states)

    assert query.shape == (4, 2, 4, 8)
    assert key.shape == (4, 2, 4, 8)
    with torch.no_grad():
        mixed_qkv, _ = apply_module(attention.linear_qkv)(hidden_states)
        mixed_qkv = mixed_qkv.view(4, 2, 4, 24)
        query_raw, key_raw, value_raw = torch.split(mixed_qkv, [8, 8, 8], dim=3)
        expected_query = apply_module(attention.q_layernorm)(
            query_raw.reshape(4, 2, 1, -1)
        ).reshape(4, 2, 4, 8)
        expected_key = apply_module(attention.k_layernorm)(key_raw.reshape(4, 2, 1, -1)).reshape(
            4, 2, 4, 8
        )
    torch.testing.assert_close(query, expected_query, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(key, expected_key, rtol=1e-5, atol=1e-6)
    assert torch.equal(value, value_raw)
    # FusedLayerNorm stores its normalized shape as hidden_size.
    assert attention.q_layernorm.hidden_size == torch.Size([32])
    assert attention.k_layernorm.hidden_size == torch.Size([32])


def test_self_attention_applies_output_gate_after_core_attention(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attention = _self_attention(_config())
    core_out = torch.randn(4, 2, 32)
    gate = torch.randn(4, 2, 32)

    gated = attention._apply_output_gate(core_out, gate)

    expected = (core_out.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))).to(
        core_out.dtype
    )
    torch.testing.assert_close(gated, expected, rtol=1e-6, atol=1e-6)


def test_attention_returns_explicit_pp_layer_offset_for_inference(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    attention = _self_attention(
        _config(), layer_number=1, attn_mask_type=AttnMaskType.padding, pp_layer_offset=7
    )

    assert attention._get_pp_layer_offset_for_inference() == 7


def test_attention_computes_pp_layer_offset_from_parallel_state(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    from megatron.core.process_groups_config import ProcessGroupCollection

    submodules = get_submodules(get_gpt_layer_local_submodules().self_attention)
    attention = SelfAttention(
        _config(),
        submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.padding,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )

    offset = attention._get_pp_layer_offset_for_inference()

    assert offset == 0
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    assert offset == get_transformer_layer_offset(attention.config, vp_stage=None, pp_rank=0)


def test_cross_attention_rejects_group_query_attention(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    submodules = get_submodules(get_gpt_layer_local_submodules().self_attention)
    cross_submodules = CrossAttentionSubmodules(
        linear_q=submodules.linear_qkv,
        linear_kv=submodules.linear_qkv,
        core_attention=submodules.core_attention,
        linear_proj=submodules.linear_proj,
    )
    config = _config(num_query_groups=2)

    with pytest.raises(ValueError, match="Group query attention is not currently supported"):
        CrossAttention(config, cross_submodules, layer_number=1)


def test_cross_attention_splits_query_and_context_key_value(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    submodules = get_submodules(get_gpt_layer_local_submodules().self_attention)
    cross_submodules = CrossAttentionSubmodules(
        linear_q=submodules.linear_qkv,
        linear_kv=submodules.linear_qkv,
        core_attention=submodules.core_attention,
        linear_proj=submodules.linear_proj,
    )
    cross = CrossAttention(
        _config(), cross_submodules, layer_number=1, attn_mask_type=AttnMaskType.padding
    )
    hidden_states = torch.randn(4, 2, 32)
    key_value_states = torch.randn(6, 2, 32)

    query, key, value = cross.get_query_key_value_tensors(hidden_states, key_value_states)

    assert query.shape == (4, 2, 4, 8)
    assert key.shape == (6, 2, 4, 8)
    assert value.shape == (6, 2, 4, 8)
    with torch.no_grad():
        expected_query, _ = apply_module(cross.linear_q)(hidden_states)
        expected_kv, _ = apply_module(cross.linear_kv)(key_value_states)
        expected_key, expected_value = torch.chunk(expected_kv.view(6, 2, 4, 16), 2, dim=3)
    torch.testing.assert_close(query, expected_query.view(4, 2, 4, 8), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(key, expected_key, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(value, expected_value, rtol=1e-6, atol=1e-6)
    with pytest.raises(AssertionError, match="split_qkv must be True"):
        cross.get_query_key_value_tensors(
            hidden_states, key_value_states, output_gate=True, split_qkv=False
        )


def test_run_realtime_tests_skips_when_qk_layernorm_disabled(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(test_mode=True)
    attention = _self_attention(config)

    assert attention.run_realtime_tests() is None


def test_run_realtime_tests_flags_diverged_norm_weights_across_ranks(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(test_mode=True, qk_layernorm=True)
    attention = _self_attention(config).cuda()
    rank = get_data_parallel_rank()
    with torch.no_grad():
        attention.q_layernorm.weight.mul_(float(rank + 1))

    with pytest.raises(AssertionError, match="Discrepancy between q_w in DP ranks"):
        attention.run_realtime_tests()


def test_run_realtime_tests_accepts_identical_norm_weights(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(test_mode=True, qk_layernorm=True)
    attention = _self_attention(config).cuda()

    assert attention.run_realtime_tests() is None


class TestDotProductAttention:
    def _build(self, config, layer_number=1, **kwargs):
        return DotProductAttention(
            config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.padding,
            attention_type="self",
            **kwargs,
        )

    def test_default_and_explicit_softmax_scale(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        config = _config()
        default = self._build(config, layer_number=2)
        explicit = self._build(config, layer_number=2, softmax_scale=0.3)

        assert default.softmax_scale == 1.0 / (32 / 4) ** 0.5
        assert explicit.softmax_scale == 0.3

    def test_query_key_layer_scaling_divides_softmax_scale(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        config = _config(apply_query_key_layer_scaling=True)
        attention = self._build(config, layer_number=4)

        assert attention.softmax_scale == (1.0 / (32 / 4) ** 0.5) / 4

    def test_window_size_passed_to_scale_mask_softmax(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        config = _config(window_size=(4, 4))
        attention = self._build(config, layer_number=2)

        assert attention.scale_mask_softmax.window_size == (4, 4)

    def test_softmax_offset_off_by_one_is_zeroed(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        config = _config(softmax_type="off-by-one")
        attention = self._build(config)

        assert attention.softmax_offset.shape == (4,)
        assert torch.count_nonzero(attention.softmax_offset) == 0

    def test_softmax_offset_learnable_is_initialized_parameter(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        config = _config(softmax_type="learnable", perform_initialization=True)
        attention = self._build(config)

        assert isinstance(attention.softmax_offset, torch.nn.Parameter)
        assert attention.softmax_offset.shape == (4,)
        assert torch.count_nonzero(attention.softmax_offset) > 0

    def test_softmax_type_unsupported_raises(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        config = _config()
        config.softmax_type = "bogus"

        with pytest.raises(ValueError, match="Softmax type not supported"):
            self._build(config)

    def test_sharded_state_dict_marks_learnable_softmax_offset(self, model_parallel):
        model_parallel(tensor_model_parallel_size=2)
        from megatron.core import mpu

        config = _config(
            softmax_type="learnable", perform_initialization=True, tensor_model_parallel_size=2
        )
        attention = self._build(config)

        sharded = attention.sharded_state_dict(
            metadata={"dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True)}
        )

        assert "softmax_offset" in sharded
        assert sharded["softmax_offset"].local_shape == (2,)
        assert sharded["softmax_offset"].global_shape == (4,)
        assert sharded["softmax_offset"].axis_fragmentations == (2,)

    def test_forward_repeats_grouped_keys_and_values(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        config = _config(num_attention_heads=4, num_query_groups=2)
        attention = self._build(config).cuda().eval()
        sequence_length, batch_size, head_dim = 8, 2, 8
        query = torch.randn(sequence_length, batch_size, 4, head_dim, device="cuda")
        key = torch.randn(sequence_length, batch_size, 2, head_dim, device="cuda")
        value = torch.randn(sequence_length, batch_size, 2, head_dim, device="cuda")

        with torch.no_grad():
            context = attention(query, key, value, None)

        key_rep = key.repeat_interleave(2, dim=2)
        value_rep = value.repeat_interleave(2, dim=2)
        scores = torch.einsum("sbhd,tbhd->bhst", query, key_rep) * attention.softmax_scale
        probs = torch.softmax(scores.float(), dim=-1)
        expected = torch.einsum("bhst,tbhd->sbhd", probs, value_rep).reshape(
            sequence_length, batch_size, -1
        )
        assert context.shape == (sequence_length, batch_size, 4 * head_dim)
        torch.testing.assert_close(
            context, expected, rtol=1e-3, atol=1e-3 * expected.abs().max().item()
        )

    def test_forward_with_sequence_parallel_skips_rng_fork(self, model_parallel):
        model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        config = _config(
            num_attention_heads=4,
            num_query_groups=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        attention = self._build(config).cuda().eval()
        sequence_length, batch_size, head_dim = 4, 2, 8
        query = torch.randn(2, batch_size, 2, head_dim, device="cuda")
        key = torch.randn(2, batch_size, 1, head_dim, device="cuda")
        value = torch.randn(2, batch_size, 1, head_dim, device="cuda")

        with torch.no_grad():
            context = attention(query, key, value, None)

        key_rep = key.repeat_interleave(2, dim=2)
        value_rep = value.repeat_interleave(2, dim=2)
        scores = torch.einsum("sbhd,tbhd->bhst", query, key_rep) * attention.softmax_scale
        probs = torch.softmax(scores.float(), dim=-1)
        expected = torch.einsum("bhst,tbhd->sbhd", probs, value_rep).reshape(
            query.size(0), batch_size, -1
        )
        assert context.shape == (2, batch_size, 2 * head_dim)
        torch.testing.assert_close(
            context, expected, rtol=1e-3, atol=1e-3 * expected.abs().max().item()
        )
