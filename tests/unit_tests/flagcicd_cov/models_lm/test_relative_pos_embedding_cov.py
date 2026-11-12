# Copyright (c) 2026, BAAI. All rights reserved.
"""RelativePositionEmbedding: unidirectional bucketing, seq-len resolution."""

import math

import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.common.embeddings.relative_pos_embedding import RelativePositionEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_relative_position_bucket_keeps_only_negative_offsets_when_unidirectional():
    embedding = RelativePositionEmbedding(
        bidirectional=False,
        init_method=init_method_normal(0.02),
        num_attention_heads=2,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
    )
    relative_position = torch.tensor([[0, -1, -5], [1, 0, -20]])

    buckets = embedding._relative_position_bucket(
        relative_position, bidirectional=False, num_buckets=8, max_distance=16
    )

    expected = []
    for value in relative_position.flatten().tolist():
        clamped = max(-value, 0)
        if clamped < 4:
            expected.append(clamped)
        else:
            large = 4 + int(math.log(clamped / 4) / math.log(16 / 4) * (8 - 4))
            expected.append(min(large, 7))
    expected = torch.tensor(expected).reshape(relative_position.shape)
    torch.testing.assert_close(buckets, expected)
    assert buckets.min() >= 0
    assert buckets.max() < 8


def test_get_relative_seq_len_uses_context_max_sequence_length_during_inference():
    context = StaticInferenceContext(max_batch_size=1, max_sequence_length=7)
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)

    seq_len = RelativePositionEmbedding.get_relative_seq_len(context, None, None, config)

    assert seq_len == 7


def test_get_relative_seq_len_reads_transformer_input_tensor(model_parallel):
    model_parallel()
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, use_cpu_initialization=True
    )
    decoder = TransformerBlock(
        config=config, spec=get_gpt_layer_local_spec(), pre_process=True, post_process=True
    )
    decoder.set_input_tensor(torch.zeros(5, 1, 8))

    seq_len = RelativePositionEmbedding.get_relative_seq_len(None, decoder, None, config)

    assert seq_len == 5


def test_get_relative_seq_len_falls_back_to_transformer_input_size(model_parallel):
    model_parallel()
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, use_cpu_initialization=True
    )
    decoder = TransformerBlock(
        config=config, spec=get_gpt_layer_local_spec(), pre_process=True, post_process=True
    )

    seq_len = RelativePositionEmbedding.get_relative_seq_len(
        None, decoder, torch.zeros(3, 1, 8), config
    )

    assert seq_len == 3


def test_get_relative_seq_len_multiplies_by_tensor_parallel_size_when_sequence_parallel(
    model_parallel,
):
    model_parallel()
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, use_cpu_initialization=True
    )
    decoder = TransformerBlock(
        config=config, spec=get_gpt_layer_local_spec(), pre_process=True, post_process=True
    )
    decoder.set_input_tensor(torch.zeros(3, 1, 8))
    config.sequence_parallel = True
    config.tensor_model_parallel_size = 2

    seq_len = RelativePositionEmbedding.get_relative_seq_len(None, decoder, None, config)

    assert seq_len == 6
