# Copyright (c) 2026, BAAI. All rights reserved.
"""TEDotProductAttention and fused RoPE wrappers: guards, mask translation and BF16 numerics."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

RTOL, ATOL = 1.6e-2, 1e-2


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _make_config(**overrides):
    overrides.setdefault('params_dtype', torch.bfloat16)
    overrides.setdefault('bf16', True)
    # TransformerConfig defaults attention_dropout to 0.1; the references are dropout-free.
    overrides.setdefault('attention_dropout', 0.0)
    return TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)


def _make_attention(config, **kwargs):
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    return TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type='self',
        **kwargs,
    )


def _det_qkv(seq, batch=2, heads=4, dim=8, offset=0):
    values = torch.arange(seq * batch * heads * dim, dtype=torch.float32)
    return (((values % 13) - 6 + offset) / 4.0).to(torch.bfloat16).reshape(seq, batch, heads, dim)


def _attention_reference(q, k, v, bias=None, window_size=None):
    seq_q, seq_k = q.shape[0], k.shape[0]
    qf = q.float().permute(1, 2, 0, 3)
    kf = k.float().permute(1, 2, 0, 3)
    vf = v.float().permute(1, 2, 0, 3)
    scores = qf @ kf.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    if bias is not None:
        scores = scores + bias.float()
    if window_size is None:
        causal = torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float('-inf'))
    else:
        query_pos = torch.arange(seq_q, device=q.device).unsqueeze(1)
        key_pos = torch.arange(seq_k, device=q.device).unsqueeze(0)
        distance = (seq_k - seq_q) + query_pos - key_pos
        # (left, right) window as in flash-attn/TE: key j is visible for i - left <= j <= i + right.
        windowed = (distance >= -window_size[1]) & (distance <= window_size[0])
        scores = scores.masked_fill(~windowed, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = (probs @ vf).permute(2, 0, 1, 3)
    # TE DotProductAttention returns [s, b, heads * head_dim].
    return out.reshape(out.shape[0], out.shape[1], -1).to(q.dtype)


def _make_attention_and_inputs(model_parallel, **config_overrides):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attn = _make_attention(_make_config(**config_overrides))
    attn.cuda()
    q = _det_qkv(16).cuda()
    k = _det_qkv(16, offset=3).cuda()
    v = _det_qkv(16, offset=7).cuda()
    return attn, q, k, v


def test_dpa_rejects_query_key_layer_scaling_mismatch_with_env(monkeypatch):
    monkeypatch.setenv('NVTE_APPLY_QK_LAYER_SCALING', '0')
    config = _make_config(apply_query_key_layer_scaling=True)
    with pytest.raises(ValueError, match='apply_query_key_layer_scaling is True'):
        _make_attention(config)


def test_dpa_rejects_deterministic_mode_when_env_allows_nondeterministic(monkeypatch):
    monkeypatch.delenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', raising=False)
    config = _make_config(deterministic_mode=True)
    with pytest.raises(RuntimeError, match='NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0'):
        _make_attention(config)


def test_dpa_forward_matches_torch_reference(model_parallel):
    attn, q, k, v = _make_attention_and_inputs(model_parallel)
    out = attn(q, k, v, None, AttnMaskType.causal)
    expected = _attention_reference(q, k, v)
    assert out.shape == (16, 2, 32)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_dpa_forward_with_attention_bias_matches_reference(model_parallel):
    attn, q, k, v = _make_attention_and_inputs(model_parallel)
    bias = (((torch.arange(256) % 7) - 3) / 4.0).to(torch.bfloat16).reshape(1, 1, 16, 16).cuda()
    out = attn(q, k, v, None, AttnMaskType.causal, attention_bias=bias)
    expected = _attention_reference(q, k, v, bias=bias)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_dpa_forward_with_num_splits_matches_reference(model_parallel):
    attn, q, k, v = _make_attention_and_inputs(model_parallel)
    out = attn(q, k, v, None, AttnMaskType.causal, num_splits=1)
    expected = _attention_reference(q, k, v)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_dpa_forward_disables_context_parallel_when_local_cp_size_is_one(model_parallel):
    model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    attn = _make_attention(_make_config(context_parallel_size=2))
    attn.cuda()
    assert attn.cp_group is parallel_state.get_context_parallel_group()

    q = _det_qkv(16).cuda()
    k = _det_qkv(16, offset=3).cuda()
    v = _det_qkv(16, offset=7).cuda()
    packed = PackedSeqParams(local_cp_size=1)
    out = attn(q, k, v, None, AttnMaskType.causal, packed_seq_params=packed)
    expected = _attention_reference(q, k, v)
    assert out.shape == (16, 2, 32)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    assert attn.cp_group is None


def test_dpa_forward_switches_to_bottom_right_mask_for_window_decode(model_parallel):
    attn, _, _, _ = _make_attention_and_inputs(model_parallel, window_size=(2, 0))
    q = _det_qkv(1).cuda()
    k = _det_qkv(4, offset=3).cuda()
    v = _det_qkv(4, offset=7).cuda()
    out = attn(q, k, v, None, AttnMaskType.no_mask)
    expected = _attention_reference(q, k, v, window_size=(2, 0))
    assert out.shape == (1, 2, 32)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_dpa_constructs_default_process_groups_when_pg_collection_is_none(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    attn = _make_attention(_make_config(tensor_model_parallel_size=2))
    assert attn._tp_group is parallel_state.get_tensor_model_parallel_group()


def test_dpa_constructs_hierarchical_cp_groups_for_a2a_p2p(model_parallel):
    from megatron.core.process_groups_config import ProcessGroupCollection

    model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(check_initialized=False),
        cp=parallel_state.get_context_parallel_group(check_initialized=False),
        hcp=parallel_state.get_hierarchical_context_parallel_groups(check_initialized=False),
    )
    config = _make_config(tensor_model_parallel_size=2, context_parallel_size=2)
    attn = _make_attention(config, cp_comm_type='a2a+p2p', pg_collection=pg_collection)
    assert attn.cp_comm_type == 'a2a+p2p'
    assert attn.cp_group is pg_collection.hcp


def test_dpa_forwards_softmax_type_to_te(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    attn = _make_attention(_make_config(softmax_type='learnable'))
    assert attn.softmax_type == 'learnable'


def test_dpa_sharded_state_dict_shards_learnable_softmax_offset(model_parallel):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    attn = _make_attention(_make_config(softmax_type='learnable', tensor_model_parallel_size=2))
    state_dict = attn.sharded_state_dict(
        prefix='module.',
        metadata={
            'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)
        },
    )
    # TE modules also carry an _extra_state entry (FP8 metadata), sharded alongside the offset.
    assert set(state_dict) == {'module.softmax_offset', 'module._extra_state'}
    sharded = state_dict['module.softmax_offset']
    assert isinstance(sharded, ShardedTensor)
    assert sharded.key == 'module.softmax_offset'
    assert torch.equal(sharded.data, attn.softmax_offset)


def test_dpa_forward_tracks_max_attention_logits_for_qk_clip(model_parallel):
    attn, q, k, v = _make_attention_and_inputs(model_parallel, qk_clip=True)
    assert attn.current_max_attn_logits is None
    out = attn(q, k, v, None, AttnMaskType.causal)
    expected = _attention_reference(q, k, v)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    first_max = attn.current_max_attn_logits
    attn(q * 0.1, k, v, None, AttnMaskType.causal)
    assert torch.equal(attn.current_max_attn_logits, first_max)


def _rope_angles(seq, dim, interleaved=False):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device='cuda') / dim)
    )
    positions = torch.arange(seq, dtype=torch.float32, device='cuda')
    angles = torch.outer(positions, inv_freq)
    if interleaved:
        angles = torch.repeat_interleave(angles, 2, dim=-1)
    else:
        angles = torch.cat((angles, angles), dim=-1)
    return angles.reshape(seq, 1, 1, dim)


def _rope_reference(t, freqs, interleaved=False):
    cos = torch.cos(freqs).to(t.dtype)
    sin = torch.sin(freqs).to(t.dtype)
    if not interleaved:
        half = t.shape[-1] // 2
        x1, x2 = torch.chunk(t, 2, dim=-1)
        return torch.cat(
            (
                x1 * cos[..., :half] - x2 * sin[..., :half],
                x2 * cos[..., :half] + x1 * sin[..., :half],
            ),
            dim=-1,
        )
    out = torch.empty_like(t)
    x1, x2 = t[..., ::2], t[..., 1::2]
    out[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    out[..., 1::2] = x2 * cos[..., 1::2] + x1 * sin[..., 1::2]
    return out


def test_fused_rope_warns_when_transpose_output_memory_requested(model_parallel):
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb

    model_parallel(tensor_model_parallel_size=1)
    t = _det_qkv(8, dim=16).cuda()
    freqs = _rope_angles(8, 16)
    with pytest.warns(
        UserWarning, match="transpose_output_memory is not supported by TE's fused RoPE"
    ):
        out = fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
    torch.testing.assert_close(out, _rope_reference(t, freqs), rtol=RTOL, atol=ATOL)


def test_fused_rope_thd_interleaved_matches_reference(model_parallel):
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd

    model_parallel(tensor_model_parallel_size=1)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32, device='cuda')
    t = _det_qkv(7, batch=1, dim=16).cuda().reshape(7, 4, 16)
    freqs_full = _rope_angles(4, 16, interleaved=True)
    token_freqs = torch.cat((freqs_full[:4], freqs_full[:3]), dim=0).reshape(7, 1, 16)
    out = fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs_full, interleaved=True)
    expected = _rope_reference(t, token_freqs, interleaved=True)
    assert out.shape == (7, 4, 16)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
