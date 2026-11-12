# Copyright (c) 2026, BAAI. All rights reserved.
"""QwenHuggingFaceModel: local LM loading, tp-gradient flags, per-token loss and embedding parity."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

import megatron.core.models.huggingface.qwen_model as qwen_model_module
from megatron.core.models.huggingface.qwen_model import QwenHuggingFaceModel
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def qwen_lm_dir(tmp_path):
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    Qwen2ForCausalLM(config).save_pretrained(tmp_path / 'qwen_lm')
    return str(tmp_path / 'qwen_lm')


@pytest.fixture
def restore_qwen_module():
    yield
    importlib.reload(qwen_model_module)


def _megatron_config(qwen_lm_dir):
    config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
    config.language_model_type = 'hf://' + qwen_lm_dir
    return config


def test_qwen_model_loads_local_lm_and_flags_gradients_for_tp_averaging(qwen_lm_dir):
    model = QwenHuggingFaceModel(_megatron_config(qwen_lm_dir)).eval()

    assert type(model.model).__name__ == 'Qwen2ForCausalLM'
    assert all(p.average_gradients_across_tp_domain for p in model.model.parameters())


def test_qwen_forward_returns_per_token_cross_entropy_loss(qwen_lm_dir):
    model = QwenHuggingFaceModel(_megatron_config(qwen_lm_dir)).eval()
    torch.manual_seed(1234)
    decoder_input = torch.randn(4, 2, 32)
    labels = torch.randint(0, 64, (2, 4))

    loss = model(labels=labels, decoder_input=decoder_input, attention_mask=None)

    with torch.no_grad():
        logits = model.model(
            position_ids=None, attention_mask=None, inputs_embeds=decoder_input.permute(1, 0, 2)
        )['logits']
    expected = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
    assert loss.shape == (2, 4)
    torch.testing.assert_close(loss, expected)


def test_qwen_embedding_transposes_input_embeddings_to_sequence_major(qwen_lm_dir):
    model = QwenHuggingFaceModel(_megatron_config(qwen_lm_dir)).eval()
    input_ids = torch.randint(0, 64, (2, 4))

    embeddings = model.embedding(input_ids)

    expected = model.model.get_input_embeddings()(input_ids).transpose(1, 0).contiguous()
    assert embeddings.shape == (4, 2, 32)
    torch.testing.assert_close(embeddings, expected)


def test_qwen_model_reports_missing_transformers_when_import_fails(
    monkeypatch, restore_qwen_module
):
    with monkeypatch.context() as blocker:
        blocker.setitem(sys.modules, 'transformers', None)
        blocker.setitem(sys.modules, 'transformers.models.qwen2', None)
        blocker.setitem(sys.modules, 'transformers.models.qwen2.modeling_qwen2', None)
        importlib.reload(qwen_model_module)
        assert qwen_model_module.HAVE_TRANSFORMERS is False
        assert isinstance(qwen_model_module.Qwen2ForCausalLM, MagicMock)
        with pytest.raises(ImportError, match='transformers is required for QwenHuggingFaceModel'):
            qwen_model_module.QwenHuggingFaceModel(
                TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
            )

    importlib.reload(qwen_model_module)
    assert qwen_model_module.HAVE_TRANSFORMERS is True
    assert issubclass(qwen_model_module.QwenHuggingFaceModel, qwen_model_module.HuggingFaceModule)
