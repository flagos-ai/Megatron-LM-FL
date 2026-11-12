# Copyright (c) 2026, BAAI. All rights reserved.
"""CLIP ViT vision model: image embedding counts for tile tags and pixel shuffle, and the SigLIP final layer norm over decoder output."""
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.clip_vit_model import CLIPViTModel, get_num_image_embeddings
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device


def test_num_image_embeddings_adds_cradio_class_tokens():
    count = get_num_image_embeddings(
        img_h=224,
        img_w=224,
        patch_dim=16,
        vision_model_type="cradio-g",
        disable_vision_class_token=False,
        class_token_len=1,
        pixel_shuffle=False,
    )
    assert count == 196 + 8


def test_num_image_embeddings_halves_tokens_with_pixel_shuffle():
    count = get_num_image_embeddings(
        img_h=224,
        img_w=224,
        patch_dim=16,
        vision_model_type="clip",
        disable_vision_class_token=False,
        class_token_len=1,
        pixel_shuffle=True,
    )
    assert count == (196 + 1) // 4


@pytest.mark.parametrize(
    "vision_model_type,match",
    [
        ("hf://nvidia/parakeet-tdt-0.6b-v2", "unsupported huggingface vision model"),
        ("vit", "unknown vision model type"),
    ],
    ids=["hf_parakeet", "unknown"],
)
def test_num_image_embeddings_rejects_unsupported_vision_models(vision_model_type, match):
    with pytest.raises(NotImplementedError, match=match):
        get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type=vision_model_type,
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
        )


@pytest.mark.parametrize(
    "tokenizer_type,expected_extra",
    [("llama3p1", 5), ("nemotron5", 6)],
    ids=["llama3p1", "nemotron5"],
)
def test_num_image_embeddings_adds_tile_tags_per_tokenizer(tokenizer_type, expected_extra):
    count = get_num_image_embeddings(
        img_h=224,
        img_w=224,
        patch_dim=16,
        vision_model_type="clip",
        disable_vision_class_token=False,
        class_token_len=1,
        pixel_shuffle=False,
        use_tile_tags=True,
        max_num_tiles=0,
        tokenizer_type=tokenizer_type,
    )
    assert count == 196 + 1 + expected_extra


def test_num_image_embeddings_requires_known_tokenizer_for_tile_tags():
    with pytest.raises(ValueError, match="tokenizer type not defined"):
        get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
            use_tile_tags=True,
            tokenizer_type="whitespace",
        )


def test_num_image_embeddings_pads_qwen_tiles_beyond_ten():
    count = get_num_image_embeddings(
        img_h=224,
        img_w=224,
        patch_dim=16,
        vision_model_type="clip",
        disable_vision_class_token=False,
        class_token_len=1,
        pixel_shuffle=False,
        use_tile_tags=True,
        max_num_tiles=12,
        tokenizer_type="qwen2p5",
    )
    assert count == 196 + 1 + 5 + 1


def test_num_image_embeddings_rejects_more_than_one_hundred_tiles():
    with pytest.raises(ValueError, match="max number of tiles 101 not supported"):
        get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
            use_tile_tags=True,
            max_num_tiles=101,
            tokenizer_type="llama3p1",
        )


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestCLIPViTModelSubtypes:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_internvit_uses_conv_bias_without_padding(self):
        model = CLIPViTModel(
            self.config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=16,
            model_subtype="internvit",
        )
        assert model.conv1.bias.shape == torch.Size([self.config.hidden_size])
        assert model.conv1.padding == (0, 0)

    def test_siglip_forward_applies_ln_post_to_decoder_output(self):
        model = CLIPViTModel(
            self.config,
            self.layer_spec,
            add_class_token=False,
            class_token_len=0,
            img_h=224,
            img_w=224,
            patch_dim=16,
            model_subtype="siglip",
        )
        model.to(get_current_device())
        model.eval()
        images = torch.randn(2, 3, 224, 224, device=get_current_device())

        output = model(images)

        x = model.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = x + model.position_embeddings(model.position_ids)
        x = x.permute(1, 0, 2).contiguous()
        x = model.decoder(x, None)
        pre_ln_output = x.permute(1, 0, 2).contiguous()

        assert output.shape == torch.Size([2, 196, self.config.hidden_size])
        torch.testing.assert_close(output, model.ln_post(pre_ln_output), rtol=1e-5, atol=1e-5)
        assert not torch.allclose(output, pre_ln_output)
