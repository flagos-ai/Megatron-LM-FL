# Copyright (c) 2026, BAAI. All rights reserved.
"""QK clip: data-parallel max of attention logits and linear_qkv weight clipping."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.optimizer.qk_clip import clip_qk
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _attention_config():
    return TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        qk_clip=True,
        qk_clip_threshold=4.0,
        qk_clip_alpha=0.5,
    )


class _Layer(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attention = attention


class _Decoder(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class _Inner(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder


class _Middle(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = inner


class _Chunk(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = _Middle(inner)


def _build_attention():
    attention = SelfAttention(
        _attention_config(),
        get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
        layer_number=1,
    )
    return attention.to(_device())


def _build_model_chunk(attention):
    return _Chunk(_Inner(_Decoder([_Layer(attention)])))


def _set_rank_logits(attention):
    rank = torch.distributed.get_rank()
    attention.core_attention.current_max_attn_logits = torch.full(
        (attention.num_attention_heads_per_partition,), float(rank + 1), device=_device()
    )


def _qkv_row_layout(attention):
    groups = attention.num_query_groups_per_partition
    query_rows = attention.query_projection_size // groups
    kv_rows = attention.kv_projection_size // groups
    return groups, query_rows, kv_rows


def test_clip_qk_reduces_max_over_data_parallel_and_clips_weights(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    attention = _build_attention()
    _set_rank_logits(attention)
    attention_without_logits = _build_attention()
    attention_without_logits.core_attention.current_max_attn_logits = None
    original_weight = attention.linear_qkv.weight.detach().clone()
    world_size = torch.distributed.get_world_size()

    log_max = clip_qk([_build_model_chunk(attention), _build_model_chunk(attention_without_logits)])

    # Every rank contributed its own rank + 1, so the reduced max over the
    # data-parallel group is the largest contribution.
    assert log_max == float(world_size)
    assert attention.core_attention.current_max_attn_logits is None
    groups, query_rows, kv_rows = _qkv_row_layout(attention)
    total_rows = query_rows + 2 * kv_rows
    original = original_weight.view(groups, total_rows, -1)
    clipped = attention.linear_qkv.weight.detach().view(groups, total_rows, -1)
    # With the reduced max above the threshold, the balancing eta is
    # clamp(threshold / max, max=1.0) and both the q and the k rows of the
    # per-group reshaped weight scale by eta ** alpha.
    expected_scale = (
        torch.clamp(torch.full((groups,), 4.0 / float(world_size), device=original.device), max=1.0)
        ** 0.5
    ).view(groups, 1, 1)
    torch.testing.assert_close(
        clipped[:, :query_rows], original[:, :query_rows] * expected_scale, rtol=0, atol=0
    )
    torch.testing.assert_close(
        clipped[:, query_rows : query_rows + kv_rows],
        original[:, query_rows : query_rows + kv_rows] * expected_scale,
        rtol=0,
        atol=0,
    )
    # The v rows are never part of the clipping.
    torch.testing.assert_close(
        clipped[:, query_rows + kv_rows :], original[:, query_rows + kv_rows :], rtol=0, atol=0
    )
    # The layer without logits was skipped entirely.
    assert attention_without_logits.core_attention.current_max_attn_logits is None


def test_clip_qk_with_log_max_only_reports_max_without_clipping(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    attention = _build_attention()
    _set_rank_logits(attention)
    original_weight = attention.linear_qkv.weight.detach().clone()
    world_size = torch.distributed.get_world_size()

    log_max = clip_qk([_build_model_chunk(attention)], log_max_only=True)

    assert log_max == float(world_size)
    torch.testing.assert_close(
        attention.linear_qkv.weight.detach(), original_weight, rtol=0, atol=0
    )
    # Without clipping the reduced logits stay visible on the attention module.
    torch.testing.assert_close(
        attention.core_attention.current_max_attn_logits,
        torch.full(
            (attention.num_attention_heads_per_partition,), float(world_size), device=_device()
        ),
        rtol=0,
        atol=0,
    )
