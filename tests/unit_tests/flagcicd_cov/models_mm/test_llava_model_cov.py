# Copyright (c) 2026, BAAI. All rights reserved.
"""LLaVA model: language/vision/projection wiring, image token expansion and label shifting, freezing, checkpoint load hooks, and forward logits with loss."""
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils, get_current_device

LANGUAGE_HIDDEN = 64
VISION_HIDDEN = 16
VOCAB_SIZE = 128
TEXT_SEQ_LEN = 12
IMG_SIZE = 56
PATCH_DIM = 14
IMG_SEQ_LEN = (IMG_SIZE // PATCH_DIM) * (IMG_SIZE // PATCH_DIM) + 1
IMAGE_TOKEN_INDEX = -200


def _language_config(**kwargs):
    config_kwargs = dict(
        num_layers=2,
        hidden_size=LANGUAGE_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    config_kwargs.update(kwargs)
    return TransformerConfig(**config_kwargs)


def _vision_config(**kwargs):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=VISION_HIDDEN,
        num_attention_heads=2,
        use_cpu_initialization=True,
        **kwargs,
    )
    config.vision_model_type = "clip"
    return config


def _projection_config(hidden_size=LANGUAGE_HIDDEN, ffn_hidden_size=32):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=1,
        use_cpu_initialization=True,
    )


def _build_model(**overrides):
    language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
    kwargs = dict(
        language_transformer_config=_language_config(),
        language_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=language_layer_submodules
        ),
        language_vocab_size=VOCAB_SIZE,
        language_max_sequence_length=64,
        vision_transformer_config=_vision_config(),
        vision_transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        drop_vision_class_token=False,
        vision_projection_config=_projection_config(),
        vision_projection_layer_spec=get_submodules(language_layer_submodules.mlp),
        img_h=IMG_SIZE,
        img_w=IMG_SIZE,
        patch_dim=PATCH_DIM,
    )
    kwargs.update(overrides)
    return LLaVAModel(**kwargs)


class TestLLaVAConstruction:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor_wires_clip_vision_gpt_language_and_projection(self):
        model = _build_model()

        assert isinstance(model.vision_model, CLIPViTModel)
        assert isinstance(model.language_model, torch.nn.Module)
        assert model.language_model.__class__.__name__ == 'GPTModel'
        assert model.img_seq_len == IMG_SEQ_LEN
        assert model.image_token_index == IMAGE_TOKEN_INDEX
        assert model.vision_projection.projector_type == 'mlp'

    def test_nemotron_hybrid_language_type_builds_hybrid_model(self):
        language_config = _language_config(hidden_size=256)
        language_config.language_model_type = 'nemotron5-hybrid'
        model = _build_model(
            language_transformer_config=language_config,
            language_vocab_size=VOCAB_SIZE,
            language_max_sequence_length=64,
            vision_projection_config=_projection_config(hidden_size=256, ffn_hidden_size=64),
            hybrid_layer_pattern='M*',
        )

        assert isinstance(model.language_model, HybridModel)
        assert model.language_model.hybrid_layer_pattern == 'M*'
        assert [type(layer).__name__ for layer in model.language_model.decoder.layers] == [
            'MambaLayer',
            'TransformerLayer',
        ]

    def test_radio_g_vision_type_uses_radio_with_ln_post(self):
        vision_config = _vision_config()
        vision_config.vision_model_type = "radio-g"
        model = _build_model(
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        )

        assert isinstance(model.vision_model, RADIOViTModel)
        assert model.vision_model.class_token_len == 5
        assert model.vision_model.ln_post.weight.shape == torch.Size([VISION_HIDDEN])
        assert model.vision_model.embedder.bias.shape == torch.Size([VISION_HIDDEN])

    def test_unsupported_vision_type_is_rejected(self):
        vision_config = _vision_config()
        vision_config.vision_model_type = "whisper"
        with pytest.raises(ValueError, match="Vision model whisper is not supported"):
            _build_model(vision_transformer_config=vision_config)

    def test_shared_embedding_weight_surfaces_language_model_weight(self):
        model = _build_model(share_embeddings_and_output_weights=True)

        shared = model.shared_embedding_or_output_weight()

        expected = model.language_model.shared_embedding_or_output_weight()
        assert shared is expected

    def test_shared_embedding_weight_is_none_without_decoder(self):
        model = _build_model(add_decoder=False)

        assert model.language_model is None
        assert model.shared_embedding_or_output_weight() is None

    def test_set_input_tensor_routes_to_vision_when_encoder_present(self):
        model = _build_model(add_decoder=False)
        input_tensor = torch.randn(IMG_SEQ_LEN, 1, VISION_HIDDEN)

        model.set_input_tensor([input_tensor])

        assert torch.equal(model.vision_model.decoder.input_tensor, input_tensor)

    def test_set_input_tensor_stores_encoder_hidden_state_on_first_stage(self):
        model = _build_model(add_encoder=False)
        input_tensor = torch.randn(TEXT_SEQ_LEN, 1, LANGUAGE_HIDDEN)

        model.set_input_tensor([input_tensor])

        assert model.encoder_hidden_state is input_tensor

    def test_set_input_tensor_routes_to_language_model_without_pre_process(self):
        model = _build_model(add_encoder=False, pre_process=False)
        input_tensor = torch.randn(TEXT_SEQ_LEN, 1, LANGUAGE_HIDDEN)

        model.set_input_tensor([input_tensor])

        assert torch.equal(model.language_model.decoder.input_tensor, input_tensor)

    def test_freeze_flags_stop_gradients_per_module(self):
        model = _build_model()
        language_param = model.language_model.decoder.layers[0].self_attention.linear_qkv.weight
        vision_param = model.vision_model.decoder.layers[0].self_attention.linear_qkv.weight
        projection_param = model.vision_projection.encoder.linear_fc1.weight

        model.freeze(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=False
        )

        assert language_param.requires_grad is False
        assert vision_param.requires_grad is True
        assert projection_param.requires_grad is True

        model.freeze(
            freeze_language_model=False, freeze_vision_model=True, freeze_vision_projection=True
        )

        assert language_param.requires_grad is False
        assert vision_param.requires_grad is False
        assert projection_param.requires_grad is False


class TestLLaVAPreprocessData:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_model().to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _preprocess(
        self, input_ids, labels, loss_mask, image_embeddings, language_embeddings, num_image_tiles
    ):
        return self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=num_image_tiles,
        )

    def test_image_token_expands_into_image_embeddings_with_shifted_labels(self):
        input_ids = torch.tensor([[10, 11, IMAGE_TOKEN_INDEX, 12, 13]])
        labels = torch.tensor([[11, IMAGE_TOKEN_INDEX, 12, 13, 14]])
        loss_mask = torch.ones(1, 5)
        image_embeddings = torch.arange(IMG_SEQ_LEN, dtype=torch.float32)
        image_embeddings = (
            image_embeddings.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, LANGUAGE_HIDDEN) + 100.0
        )
        language_embeddings = torch.arange(5, dtype=torch.float32)
        language_embeddings = (
            language_embeddings.unsqueeze(0).unsqueeze(-1).repeat(1, 1, LANGUAGE_HIDDEN)
        )

        final_embedding, final_labels, final_loss_mask = self._preprocess(
            input_ids,
            labels,
            loss_mask,
            image_embeddings,
            language_embeddings,
            torch.tensor([1], dtype=torch.int),
        )

        assert final_embedding.shape == torch.Size([21, 1, LANGUAGE_HIDDEN])
        expected_embedding = torch.zeros(1, 21, LANGUAGE_HIDDEN)
        expected_embedding[0, 0] = language_embeddings[0, 0, 0]
        expected_embedding[0, 1] = language_embeddings[0, 1, 0]
        expected_embedding[0, 2:19] = image_embeddings[:17, 0]
        expected_embedding[0, 19] = language_embeddings[0, 3, 0]
        expected_embedding[0, 20] = language_embeddings[0, 4, 0]
        torch.testing.assert_close(final_embedding, expected_embedding.transpose(0, 1))

        expected_labels = torch.full((1, 21), -100)
        expected_labels[0, 0] = 11
        expected_labels[0, 18] = 12
        expected_labels[0, 19] = 13
        expected_labels[0, 20] = 14
        assert torch.equal(final_labels, expected_labels)

        expected_loss_mask = torch.zeros(1, 21)
        expected_loss_mask[0, 0] = 1.0
        expected_loss_mask[0, 19] = 1.0
        expected_loss_mask[0, 20] = 1.0
        assert torch.equal(final_loss_mask, expected_loss_mask)

    def test_multiple_tiles_repeat_the_image_embeddings(self):
        input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 10]])
        labels = torch.tensor([[10, 11]])
        loss_mask = torch.ones(1, 2)
        tiles = torch.arange(2 * IMG_SEQ_LEN, dtype=torch.float32).reshape(2, IMG_SEQ_LEN)
        image_embeddings = tiles.unsqueeze(-1).repeat(1, 1, LANGUAGE_HIDDEN).permute(1, 0, 2)
        language_embeddings = torch.zeros(1, 2, LANGUAGE_HIDDEN)

        final_embedding, final_labels, _ = self._preprocess(
            input_ids,
            labels,
            loss_mask,
            image_embeddings,
            language_embeddings,
            torch.tensor([2], dtype=torch.int),
        )

        expected_seq_len = 2 * IMG_SEQ_LEN - 1 + 2
        assert final_embedding.shape == torch.Size([expected_seq_len, 1, LANGUAGE_HIDDEN])
        expected_embedding = torch.zeros(1, expected_seq_len, LANGUAGE_HIDDEN)
        expected_embedding[0, 0:17] = torch.arange(17, dtype=torch.float32).unsqueeze(-1)
        expected_embedding[0, 17:34] = torch.arange(17, 34, dtype=torch.float32).unsqueeze(-1)
        torch.testing.assert_close(final_embedding, expected_embedding.transpose(0, 1))
        expected_labels = torch.full((1, expected_seq_len), -100)
        expected_labels[0, 33] = 10
        expected_labels[0, 34] = 11
        assert torch.equal(final_labels, expected_labels)

    def test_text_only_sample_pads_to_longest_sequence(self):
        input_ids = torch.tensor([[10, IMAGE_TOKEN_INDEX, 11], [20, 21, 22]])
        labels = torch.tensor([[IMAGE_TOKEN_INDEX, 10, 11], [21, 22, 23]])
        loss_mask = torch.ones(2, 3)
        image_embeddings = (
            torch.arange(IMG_SEQ_LEN, dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, LANGUAGE_HIDDEN)
        )
        language_embeddings = torch.zeros(2, 3, LANGUAGE_HIDDEN)

        final_embedding, final_labels, final_loss_mask = self._preprocess(
            input_ids,
            labels,
            loss_mask,
            image_embeddings,
            language_embeddings,
            torch.tensor([1], dtype=torch.int),
        )

        sample0_len = IMG_SEQ_LEN - 1 + 3
        assert final_embedding.shape == torch.Size([sample0_len, 2, LANGUAGE_HIDDEN])
        expected = torch.zeros(2, sample0_len, LANGUAGE_HIDDEN)
        expected[0, 1:18] = image_embeddings[:17, 0]
        assert torch.count_nonzero(final_embedding[19:]) == 0
        torch.testing.assert_close(final_embedding.transpose(0, 1)[1], expected[1])
        expected_labels = torch.full((2, sample0_len), -100)
        expected_labels[0, 17] = 10
        expected_labels[0, 18] = 11
        expected_labels[1, 0:3] = torch.tensor([21, 22, 23])
        assert torch.equal(final_labels, expected_labels)
        expected_loss_mask = torch.zeros(2, sample0_len)
        expected_loss_mask[0, 18] = 1.0
        expected_loss_mask[1, 0:3] = 1.0
        assert torch.equal(final_loss_mask, expected_loss_mask)

    def test_inference_kv_cache_returns_inputs_unchanged(self):
        input_ids = torch.tensor([[10, 11]])
        language_embeddings = torch.randn(1, 2, LANGUAGE_HIDDEN)
        loss_mask = torch.ones(1, 2)

        final_embedding, final_labels, final_loss_mask = self.model._preprocess_data(
            torch.zeros(0, 0, 0),
            language_embeddings,
            input_ids,
            loss_mask,
            None,
            use_inference_kv_cache=True,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.zeros(0, dtype=torch.int),
        )

        assert final_embedding is language_embeddings
        assert final_loss_mask is loss_mask
        assert final_labels is None

    def test_truncation_to_language_max_sequence_length(self):
        truncated_model = _build_model(language_max_sequence_length=24).to(get_current_device())
        input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 10]])
        labels = torch.tensor([[10, 11]])
        loss_mask = torch.ones(1, 2)
        image_embeddings = torch.zeros(IMG_SEQ_LEN, 2, LANGUAGE_HIDDEN)
        language_embeddings = torch.zeros(1, 2, LANGUAGE_HIDDEN)

        final_embedding, final_labels, final_loss_mask = truncated_model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.tensor([2], dtype=torch.int),
        )

        expected_seq_len = 2 * IMG_SEQ_LEN - 1 + 2
        assert expected_seq_len > 24
        assert final_embedding.shape == torch.Size([24, 1, LANGUAGE_HIDDEN])
        assert final_labels.shape == torch.Size([1, 24])
        assert final_loss_mask.shape == torch.Size([1, 24])


class TestLLaVAForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_model().to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _inputs(self, with_image_token):
        input_ids = torch.tensor(
            [[10, 11, IMAGE_TOKEN_INDEX, 12, 13] if with_image_token else [10, 11, 12, 13, 14]]
        ).to(get_current_device())
        position_ids = torch.arange(5).unsqueeze(0).to(get_current_device())
        return input_ids, position_ids

    def test_text_only_forward_matches_language_model_on_combined_embeddings(self):
        input_ids, position_ids = self._inputs(with_image_token=False)
        images = torch.zeros(0, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        logits, _ = self.model(
            images=images, input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )

        input_ids_text = input_ids.clone()
        input_ids_text[input_ids_text == IMAGE_TOKEN_INDEX] = 0
        language_embeddings = self.model.language_model.embedding(
            input_ids=input_ids_text, position_ids=position_ids
        ).transpose(1, 0)
        combined, _, _ = self.model._preprocess_data(
            torch.zeros(0, 0, 0),
            language_embeddings,
            input_ids,
            None,
            None,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.zeros(0, dtype=torch.int),
        )
        expected = self.model.language_model(
            input_ids=None, position_ids=None, attention_mask=None, decoder_input=combined
        )
        assert logits.shape == torch.Size([1, 5, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)

    def test_image_forward_expands_tokens_and_matches_composed_pipeline(self):
        input_ids, position_ids = self._inputs(with_image_token=True)
        images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())
        labels = torch.tensor([[11, IMAGE_TOKEN_INDEX, 12, 13, 14]]).to(get_current_device())
        loss_mask = torch.ones(1, 5, device=get_current_device())

        logits, _ = self.model(
            images=images, input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )

        image_embeddings = self.model.vision_model(images)
        image_embeddings = self.model.vision_projection(image_embeddings.permute(1, 0, 2))
        input_ids_text = input_ids.clone()
        input_ids_text[input_ids_text == IMAGE_TOKEN_INDEX] = 0
        language_embeddings = self.model.language_model.embedding(
            input_ids=input_ids_text, position_ids=position_ids
        ).transpose(1, 0)
        combined, _, _ = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            None,
            None,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.tensor([1], dtype=torch.int),
        )
        expected = self.model.language_model(
            input_ids=None, position_ids=None, attention_mask=None, decoder_input=combined
        )
        assert logits.shape == torch.Size([1, 21, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)

        loss, _ = self.model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )
        shifted_labels = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.tensor([1], dtype=torch.int),
        )[1]
        expected_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), shifted_labels.reshape(-1), reduction='none'
        ).reshape(1, 21)
        masked = shifted_labels.reshape(-1) != -100
        torch.testing.assert_close(
            loss.reshape(-1)[masked], expected_loss[masked], rtol=1e-4, atol=1e-5
        )

    def test_forward_backpropagates_into_all_three_modules(self):
        input_ids, position_ids = self._inputs(with_image_token=True)
        images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())
        labels = torch.tensor([[11, IMAGE_TOKEN_INDEX, 12, 13, 14]]).to(get_current_device())
        loss_mask = torch.ones(1, 5, device=get_current_device())
        self.model.train()

        loss, _ = self.model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )
        loss.sum().backward()

        vision_grad = self.model.vision_model.conv1.weight.grad
        projection_grad = self.model.vision_projection.encoder.linear_fc1.weight.grad
        language_grad = self.model.language_model.decoder.layers[
            0
        ].self_attention.linear_qkv.weight.grad
        assert torch.isfinite(vision_grad).all()
        assert torch.linalg.vector_norm(vision_grad) > 0.0
        assert torch.isfinite(projection_grad).all()
        assert torch.linalg.vector_norm(projection_grad) > 0.0
        assert torch.isfinite(language_grad).all()
        assert torch.linalg.vector_norm(language_grad) > 0.0
