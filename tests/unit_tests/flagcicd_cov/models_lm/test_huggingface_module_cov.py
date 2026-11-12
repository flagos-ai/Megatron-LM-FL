# Copyright (c) 2026, BAAI. All rights reserved.
"""HuggingFaceModule: set_input_tensor, auto model loading, model-type resolution, build_hf_model dispatch."""

import importlib
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model, SiglipConfig, SiglipVisionModel

import megatron.core.models.huggingface.module as hf_module
from megatron.core.models.huggingface.module import (
    AutoHuggingFaceModel,
    HuggingFaceModule,
    build_hf_model,
    get_hf_model_type,
)
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def megatron_config():
    return TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)


@pytest.fixture
def restore_hf_module():
    yield
    importlib.reload(hf_module)


@pytest.fixture
def qwen_base_dir(tmp_path):
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    Qwen2Model(config).save_pretrained(tmp_path / 'qwen_base')
    return str(tmp_path / 'qwen_base')


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


def test_huggingface_module_set_input_tensor_stores_tensor(megatron_config):
    module = HuggingFaceModule(megatron_config)
    tensor = torch.randn(2, 4)

    module.set_input_tensor(tensor)

    assert torch.equal(module.input_tensor, tensor)


def test_auto_huggingface_model_loads_local_model_and_runs_forward(megatron_config, qwen_base_dir):
    megatron_config.huggingface_model_name_or_path = qwen_base_dir

    model = AutoHuggingFaceModel(megatron_config).eval()
    input_ids = torch.randint(0, 64, (2, 4))

    output = model(input_ids=input_ids)
    expected = model.model(input_ids=input_ids)

    assert type(model.model).__name__ == 'Qwen2Model'
    assert 'last_hidden_state' in output
    torch.testing.assert_close(output.last_hidden_state, expected.last_hidden_state)
    assert output.last_hidden_state.shape == (2, 4, 32)


def test_get_hf_model_type_resolves_qwen_and_siglip_architectures(qwen_base_dir, siglip_vision_dir):
    assert get_hf_model_type('hf://' + qwen_base_dir) == 'qwen'
    assert get_hf_model_type('hf://' + siglip_vision_dir) == 'siglip'


def test_build_hf_model_constructs_qwen_and_siglip_wrappers(
    megatron_config, qwen_base_dir, siglip_vision_dir
):
    megatron_config.language_model_type = 'hf://' + qwen_base_dir
    megatron_config.vision_model_type = 'hf://' + siglip_vision_dir

    qwen_model = build_hf_model(megatron_config, 'hf://' + qwen_base_dir)
    siglip_model = build_hf_model(megatron_config, 'hf://' + siglip_vision_dir)

    assert type(qwen_model).__name__ == 'QwenHuggingFaceModel'
    assert type(siglip_model).__name__ == 'SiglipHuggingFaceModel'


def test_huggingface_module_reports_missing_transformers_when_import_fails(
    monkeypatch, megatron_config, restore_hf_module
):
    with monkeypatch.context() as blocker:
        blocker.setitem(sys.modules, 'transformers', None)
        importlib.reload(hf_module)
        assert hf_module.HAVE_TRANSFORMERS is False
        with pytest.raises(ImportError, match='transformers is required for AutoHuggingFaceModel'):
            hf_module.AutoHuggingFaceModel(megatron_config)
        with pytest.raises(ImportError, match='transformers is required for get_hf_model_type'):
            hf_module.get_hf_model_type('hf://anything')

    importlib.reload(hf_module)
    assert hf_module.HAVE_TRANSFORMERS is True
    assert issubclass(hf_module.AutoHuggingFaceModel, hf_module.HuggingFaceModule)
