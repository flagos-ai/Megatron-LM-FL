# Copyright (c) 2026, BAAI. All rights reserved.
"""SiglipHuggingFaceModel: local vision encoder loading, forward parity with HF, transformers-absent guard."""

import importlib
import sys

import pytest
import torch
from transformers import SiglipConfig, SiglipVisionModel

import megatron.core.models.huggingface.clip_model as clip_model_module
from megatron.core.models.huggingface.clip_model import SiglipHuggingFaceModel
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def siglip_vision_dir(tmp_path):
    config = SiglipConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4
    )
    vision = config.vision_config
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_hidden_layers = 1
    vision.num_attention_heads = 4
    vision.image_size = 32
    vision.patch_size = 8
    SiglipVisionModel(vision).save_pretrained(tmp_path / 'siglip_vision')
    return str(tmp_path / 'siglip_vision')


@pytest.fixture
def restore_clip_module():
    yield
    importlib.reload(clip_model_module)


def _megatron_config(siglip_vision_dir):
    config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
    config.vision_model_type = 'hf://' + siglip_vision_dir
    return config


def test_siglip_model_loads_local_vision_encoder(siglip_vision_dir):
    model = SiglipHuggingFaceModel(_megatron_config(siglip_vision_dir)).eval()

    assert type(model.model).__name__ == 'SiglipVisionModel'


def test_siglip_forward_returns_last_hidden_state_of_vision_encoder(siglip_vision_dir):
    model = SiglipHuggingFaceModel(_megatron_config(siglip_vision_dir)).eval()
    torch.manual_seed(1234)
    pixel_values = torch.randn(2, 3, 32, 32)

    hidden_states = model(pixel_values=pixel_values)

    with torch.no_grad():
        expected = model.model(pixel_values=pixel_values)['last_hidden_state']
    assert hidden_states.shape == (2, 16, 32)
    torch.testing.assert_close(hidden_states, expected)


def test_siglip_model_reports_missing_transformers_when_import_fails(
    monkeypatch, restore_clip_module
):
    with monkeypatch.context() as blocker:
        blocker.setitem(sys.modules, 'transformers', None)
        blocker.setitem(sys.modules, 'transformers.models.siglip.modeling_siglip', None)
        importlib.reload(clip_model_module)
        assert clip_model_module.HAVE_TRANSFORMERS is False
        with pytest.raises(
            ImportError, match='transformers is required for SiglipHuggingFaceModel'
        ):
            clip_model_module.SiglipHuggingFaceModel(
                TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
            )

    importlib.reload(clip_model_module)
    assert clip_model_module.HAVE_TRANSFORMERS is True
    assert issubclass(clip_model_module.SiglipHuggingFaceModel, clip_model_module.HuggingFaceModule)
