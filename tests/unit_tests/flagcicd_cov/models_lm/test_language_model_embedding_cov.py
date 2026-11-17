# Copyright (c) 2026, BAAI. All rights reserved.
"""LanguageModelEmbedding: tokentype tables, zero_parameters, forward with tokentype/mup/fp32 residual."""

import pytest
import torch

from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _bert_style_config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        perform_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_embedding_creates_tokentype_table_when_num_tokentypes_positive(model_parallel):
    model_parallel()
    config = _bert_style_config()

    embedding = LanguageModelEmbedding(
        config=config, vocab_size=32, max_sequence_length=4, num_tokentypes=2
    )

    assert embedding.tokentype_embeddings.num_embeddings == 2
    assert embedding.tokentype_embeddings.embedding_dim == config.hidden_size
    assert embedding.tokentype_embeddings.weight.shape == (2, 8)


def test_zero_parameters_zeroes_word_position_and_tokentype_tables(model_parallel):
    model_parallel()
    config = _bert_style_config()

    embedding = LanguageModelEmbedding(
        config=config, vocab_size=32, max_sequence_length=4, num_tokentypes=2
    )
    embedding.zero_parameters()

    assert torch.count_nonzero(embedding.word_embeddings.weight).item() == 0
    assert torch.count_nonzero(embedding.position_embeddings.weight).item() == 0
    assert torch.count_nonzero(embedding.tokentype_embeddings.weight).item() == 0
    assert embedding.word_embeddings.weight.shared is True
    assert embedding.tokentype_embeddings.weight.shared is True


def test_forward_adds_tokentype_embeddings_permuted_to_sequence_major(model_parallel):
    model_parallel()
    config = _bert_style_config()
    embedding = LanguageModelEmbedding(
        config=config, vocab_size=32, max_sequence_length=4, num_tokentypes=2
    ).eval()
    input_ids = torch.randint(0, 32, (2, 3))
    position_ids = torch.arange(3).unsqueeze(0).expand(2, 3)
    tokentype_ids = torch.randint(0, 2, (2, 3))

    output = embedding(input_ids, position_ids, tokentype_ids)

    expected = (
        embedding.word_embeddings(input_ids) + embedding.position_embeddings(position_ids)
    ).transpose(0, 1) + embedding.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
    expected = expected.contiguous()

    assert output.shape == (3, 2, 8)
    torch.testing.assert_close(output, expected)


def test_forward_scales_embeddings_by_mup_embedding_mult_when_mup_enabled(model_parallel):
    model_parallel()
    config = _bert_style_config(use_mup=True, mup_embedding_mult=2.0)

    embedding = LanguageModelEmbedding(config=config, vocab_size=32, max_sequence_length=4).eval()
    input_ids = torch.randint(0, 32, (2, 3))
    position_ids = torch.arange(3).unsqueeze(0).expand(2, 3)

    output = embedding(input_ids, position_ids)

    expected = (
        embedding.word_embeddings(input_ids) + embedding.position_embeddings(position_ids)
    ).transpose(0, 1).contiguous() * 2.0
    torch.testing.assert_close(output, expected)


def test_forward_casts_embeddings_to_float_when_fp32_residual_connection(model_parallel):
    model_parallel()
    config = _bert_style_config(params_dtype=torch.bfloat16, fp32_residual_connection=True)

    embedding = LanguageModelEmbedding(config=config, vocab_size=32, max_sequence_length=4).eval()
    input_ids = torch.randint(0, 32, (2, 3))
    position_ids = torch.arange(3).unsqueeze(0).expand(2, 3)

    output = embedding(input_ids, position_ids)

    expected = (
        (embedding.word_embeddings(input_ids) + embedding.position_embeddings(position_ids))
        .transpose(0, 1)
        .contiguous()
    )
    assert output.dtype == torch.float32
    assert embedding.word_embeddings.weight.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected.float())
