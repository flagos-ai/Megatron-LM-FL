# Copyright (c) 2026, BAAI. All rights reserved.
"""RotaryEmbedding/MultimodalRotaryEmbedding: scaling, interleaving, interpolation, cp slicing."""

import math

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_rotary_dim_is_truncated_by_rotary_percent():
    rope = RotaryEmbedding(kv_channels=8, rotary_percent=0.5, use_cpu_initialization=True)

    expected = 1.0 / (10000 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    assert rope.inv_freq.shape == (2,)
    torch.testing.assert_close(rope.inv_freq, expected.to(rope.inv_freq.device))


def test_apply_scaling_matches_llama_style_interpolation_reference():
    freqs = torch.tensor([1e-4, 1e-3, 1e-2, 1.0])
    rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)

    scaled = rope._apply_scaling(
        freqs,
        factor=4,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=2048,
    )

    expected = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq
        low_wavelen, high_wavelen = 2048 / 1, 2048 / 4
        if wavelen > low_wavelen:
            expected.append(freq / 4)
        elif wavelen < high_wavelen:
            expected.append(freq)
        else:
            smooth = (2048 / wavelen - 1) / (4 - 1)
            expected.append((1 - smooth) * freq / 4 + smooth * freq)
    torch.testing.assert_close(scaled, torch.tensor(expected), rtol=1e-6, atol=1e-8)


def test_rotary_init_applies_scaling_when_rope_scaling_enabled():
    rope = RotaryEmbedding(
        kv_channels=8,
        rotary_percent=1.0,
        use_cpu_initialization=True,
        rope_scaling=True,
        rope_scaling_factor=8.0,
    )

    expected = []
    for freq in RotaryEmbedding(
        kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True
    ).inv_freq.tolist():
        wavelen = 2 * math.pi / freq
        low_wavelen, high_wavelen = 8192 / 1, 8192 / 4
        if wavelen > low_wavelen:
            expected.append(freq / 8)
        elif wavelen < high_wavelen:
            expected.append(freq)
        else:
            smooth = (8192 / wavelen - 1) / (4 - 1)
            expected.append((1 - smooth) * freq / 8 + smooth * freq)
    torch.testing.assert_close(rope.inv_freq, torch.tensor(expected), rtol=1e-6, atol=1e-8)


def test_get_freqs_non_repeated_scales_positions_by_interpolation_factor():
    rope = RotaryEmbedding(
        kv_channels=8,
        rotary_percent=1.0,
        use_cpu_initialization=True,
        seq_len_interpolation_factor=2.0,
    )

    freqs = rope.get_freqs_non_repeated(4)

    expected = torch.outer(
        torch.arange(4, dtype=torch.float32, device=rope.inv_freq.device) / 2, rope.inv_freq
    )
    torch.testing.assert_close(freqs, expected)


def test_get_cos_sin_returns_trig_of_frequency_matrix():
    rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)

    cos, sin = rope.get_cos_sin(4, offset=1)

    expected = rope.get_freqs_non_repeated(4, offset=1)
    torch.testing.assert_close(cos, torch.cos(expected))
    torch.testing.assert_close(sin, torch.sin(expected))


def test_get_emb_interleaves_frequencies_when_rotary_interleaved():
    rope = RotaryEmbedding(
        kv_channels=8, rotary_percent=1.0, rotary_interleaved=True, use_cpu_initialization=True
    )

    emb = rope.get_emb(4)

    freqs = torch.outer(
        torch.arange(4, dtype=torch.float32, device=rope.inv_freq.device), rope.inv_freq
    )
    interleaved = torch.stack((freqs.reshape(-1, 1), freqs.reshape(-1, 1)), dim=-1).reshape(4, -1)
    assert emb.shape == (4, 1, 1, 8)
    torch.testing.assert_close(emb[:, 0, 0, :], interleaved)


def test_get_rotary_seq_len_prefers_max_of_context_and_input_length():
    rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)
    context = StaticInferenceContext(max_batch_size=1, max_sequence_length=4)
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)

    seq_len = rope.get_rotary_seq_len(context, None, torch.zeros(6, 1, 8), config)

    assert seq_len == 6


def test_get_rotary_seq_len_reads_decoder_input_tensor_when_input_missing(model_parallel):
    model_parallel()
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, use_cpu_initialization=True
    )
    decoder = TransformerBlock(
        config=config, spec=get_gpt_layer_local_spec(), pre_process=True, post_process=True
    )
    decoder.set_input_tensor(torch.zeros(5, 1, 8))
    rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)
    context = StaticInferenceContext(max_batch_size=1, max_sequence_length=4)

    seq_len = rope.get_rotary_seq_len(context, decoder, None, config)

    assert seq_len == 5


def test_mrope_dim_is_truncated_by_rotary_percent():
    mrope = MultimodalRotaryEmbedding(kv_channels=8, rotary_percent=0.5)

    expected = 1.0 / (10000 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    assert mrope.inv_freq.shape == (2,)
    torch.testing.assert_close(mrope.inv_freq, expected.to(mrope.inv_freq.device))


def test_apply_interleaved_mrope_lays_out_channels_with_stride_three():
    freqs = torch.zeros(3, 2, 3, 6)
    freqs[0] = 1.0
    freqs[1] = 2.0
    freqs[2] = 3.0

    merged = MultimodalRotaryEmbedding._apply_interleaved_mrope(freqs, [2, 2, 2])

    assert merged.shape == (2, 3, 6)
    assert torch.equal(merged[..., 0], torch.full((2, 3), 1.0))
    assert torch.equal(merged[..., 1], torch.full((2, 3), 2.0))
    assert torch.equal(merged[..., 2], torch.full((2, 3), 3.0))
    assert torch.equal(merged[..., 3], torch.full((2, 3), 1.0))
    assert torch.equal(merged[..., 4], torch.full((2, 3), 2.0))
    assert torch.equal(merged[..., 5], torch.full((2, 3), 3.0))


def test_mrope_forward_interleaved_layout_matches_reference():
    mrope = MultimodalRotaryEmbedding(
        kv_channels=12, rotary_percent=1.0, rotary_interleaved=False, interleaved_mrope=True
    )
    position_ids = torch.stack(
        [torch.arange(4.0), torch.arange(4.0) + 10.0, torch.arange(4.0) + 20.0]
    )[:, None, :]

    emb = mrope(position_ids, [2, 2, 2])

    inv_freq = mrope.inv_freq
    seq = position_ids.to(device=inv_freq.device, dtype=inv_freq.dtype)
    per_channel = [seq[channel][..., None] * inv_freq for channel in range(3)]
    expected = per_channel[0].clone()
    length = 2 * 3
    expected[..., 1:length:3] = per_channel[1][..., 1:length:3]
    expected[..., 2:length:3] = per_channel[2][..., 2:length:3]
    expected = torch.cat((expected, expected), dim=-1)
    expected = expected.transpose(0, 1).unsqueeze(2).contiguous()
    assert emb.shape == (4, 1, 1, 12)
    torch.testing.assert_close(emb, expected)


def test_mrope_forward_scales_positions_by_interpolation_factor():
    position_ids = torch.stack(
        [torch.arange(4.0), torch.arange(4.0) + 10.0, torch.arange(4.0) + 20.0]
    )[:, None, :]

    interpolated = MultimodalRotaryEmbedding(
        kv_channels=12, rotary_percent=1.0, seq_len_interpolation_factor=2.0
    )(position_ids.clone(), [2, 2, 2])
    plain = MultimodalRotaryEmbedding(kv_channels=12, rotary_percent=1.0)

    reference = plain(position_ids / 2, [2, 2, 2])
    torch.testing.assert_close(interpolated, reference)


def test_mrope_forward_slices_sequence_for_context_parallel_rank(model_parallel):
    model_parallel(context_parallel_size=2)
    cp_group = parallel_state.get_context_parallel_group()
    device = get_current_device()
    position_ids = torch.stack(
        [torch.arange(4.0), torch.arange(4.0) + 10.0, torch.arange(4.0) + 20.0]
    )[:, None, :].to(device)
    mrope = MultimodalRotaryEmbedding(kv_channels=12, rotary_percent=1.0)

    emb = mrope(position_ids, [2, 2, 2], cp_group=cp_group)

    full = mrope(position_ids, [2, 2, 2], cp_group=None)
    rank = cp_group.rank()
    expected = full.index_select(0, torch.tensor([rank, 3 - rank], device=full.device))
    assert emb.shape == (2, 1, 1, 12)
    torch.testing.assert_close(emb, expected)
