# Copyright (c) 2026, BAAI. All rights reserved.
"""LLaVA vision paths: pixel shuffle, tile tagging, sound embedding insertion, packed dynamic-resolution RADIO inputs, encoder-only outputs, and inference token counts."""
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel, pixel_shuffle
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils, get_current_device

LANGUAGE_HIDDEN = 64
VISION_HIDDEN = 16
VOCAB_SIZE = 128
IMG_SIZE = 56
PATCH_DIM = 14
IMG_SEQ_LEN = (IMG_SIZE // PATCH_DIM) * (IMG_SIZE // PATCH_DIM) + 1
IMAGE_TOKEN_INDEX = -200
SOUND_TOKEN_INDEX = -300


class _EchoSoundModel(torch.nn.Module):
    def forward(self, sound_clips, sound_length):
        return sound_clips, sound_length


def _language_config():
    return TransformerConfig(
        num_layers=2,
        hidden_size=LANGUAGE_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


def _vision_config(vision_model_type="clip"):
    config = TransformerConfig(
        num_layers=2, hidden_size=VISION_HIDDEN, num_attention_heads=2, use_cpu_initialization=True
    )
    config.vision_model_type = vision_model_type
    return config


def _projection_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=LANGUAGE_HIDDEN,
        ffn_hidden_size=32,
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


class TestLLaVAPixelShuffle:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_model(pixel_shuffle=True, drop_vision_class_token=True).to(
            get_current_device()
        )
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_pixel_shuffle_forward_matches_composed_pipeline(self):
        input_ids = torch.tensor([[10, 11, IMAGE_TOKEN_INDEX, 12, 13]]).to(get_current_device())
        position_ids = torch.arange(5).unsqueeze(0).to(get_current_device())
        images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        logits, _ = self.model(
            images=images, input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )

        vision_out = self.model.vision_model(images)[
            :, self.model.vision_model.class_token_len :, :
        ]
        shuffled = pixel_shuffle(vision_out)
        image_embeddings = self.model.vision_projection(shuffled.permute(1, 0, 2))
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
        assert self.model.img_seq_len == (IMG_SEQ_LEN - 1) // 4
        assert logits.shape == torch.Size([1, 5 - 1 + (IMG_SEQ_LEN - 1) // 4, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


class TestLLaVATileTagging:
    TILE_TAGS = [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        # _apply_tile_tagging embeds the tag ids with position_ids=None, which
        # requires a position-embedding-free (rope) language embedding.
        self.model = _build_model(
            tile_tags=self.TILE_TAGS,
            tokenizer_type='llama3p1',
            language_position_embedding_type='rope',
        ).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_tile_tagging_prepends_tag_embeddings_per_tile(self):
        image_embeddings = torch.randn(
            IMG_SEQ_LEN - 5, 2, LANGUAGE_HIDDEN, device=get_current_device()
        )

        tagged = self.model._apply_tile_tagging(
            image_embeddings, torch.tensor([2], device=get_current_device())
        )

        tag_ids = torch.tensor(self.TILE_TAGS, device=image_embeddings.device)
        expected_tags = self.model.language_model.embedding(tag_ids, position_ids=None)
        assert tagged.shape == torch.Size([IMG_SEQ_LEN, 2, LANGUAGE_HIDDEN])
        torch.testing.assert_close(tagged[:5], expected_tags)
        torch.testing.assert_close(tagged[5:], image_embeddings)

    def test_tile_tagging_forward_matches_composed_pipeline(self):
        device = get_current_device()
        input_ids = torch.tensor([[10, IMAGE_TOKEN_INDEX, 12]]).to(device)
        position_ids = torch.arange(3).unsqueeze(0).to(device)
        images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)

        logits, _ = self.model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            num_image_tiles=torch.tensor([2], dtype=torch.int, device=device),
        )

        vision_out = self.model.vision_model(images).permute(1, 0, 2)
        image_embeddings = self.model.vision_projection(vision_out)
        tagged = self.model._apply_tile_tagging(image_embeddings, torch.tensor([2], device=device))
        input_ids_text = input_ids.clone()
        input_ids_text[input_ids_text == IMAGE_TOKEN_INDEX] = 0
        language_embeddings = self.model.language_model.embedding(
            input_ids=input_ids_text, position_ids=position_ids
        ).transpose(1, 0)
        combined, _, _ = self.model._preprocess_data(
            tagged,
            language_embeddings,
            input_ids,
            None,
            None,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.tensor([2], dtype=torch.int, device=device),
        )
        expected = self.model.language_model(
            input_ids=None, position_ids=None, attention_mask=None, decoder_input=combined
        )
        expected_seq_len = 2 * (IMG_SEQ_LEN + 5) - 1 + 3
        assert logits.shape == torch.Size([1, expected_seq_len, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


class TestLLaVASoundPaths:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_model(
            sound_model=_EchoSoundModel(), sound_projection=torch.nn.Linear(8, LANGUAGE_HIDDEN)
        ).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sound_embeddings_replace_sound_token_positions(self):
        input_ids = torch.tensor(
            [[10, SOUND_TOKEN_INDEX, SOUND_TOKEN_INDEX, 11], [20, SOUND_TOKEN_INDEX, 21, 22]]
        ).to(get_current_device())
        language_embeddings = torch.zeros(2, 4, LANGUAGE_HIDDEN, device=get_current_device())
        sound_clips = torch.randn(2, 3, 8, device=get_current_device())
        projected = self.model.sound_projection(sound_clips.permute(1, 0, 2))

        device = get_current_device()
        final_embedding, _, _ = self.model._preprocess_data(
            torch.zeros(0, 0, 0, device=device),
            language_embeddings,
            input_ids,
            torch.ones(2, 4, device=device),
            None,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.zeros(0, dtype=torch.int, device=device),
            sound_embeddings=projected,
            sound_embeddings_len=torch.tensor([2, 1], device=device),
        )

        assert final_embedding.shape == torch.Size([4, 2, LANGUAGE_HIDDEN])
        torch.testing.assert_close(final_embedding[1, 0], projected[0, 0])
        torch.testing.assert_close(final_embedding[2, 0], projected[1, 0])
        torch.testing.assert_close(final_embedding[1, 1], projected[0, 1])
        assert torch.count_nonzero(final_embedding[3, 1]) == 0

    def test_zero_sentinel_sound_clips_are_treated_as_absent(self):
        input_ids = torch.tensor([[10, 11, 12]]).to(get_current_device())
        position_ids = torch.arange(3).unsqueeze(0).to(get_current_device())
        images = torch.zeros(0, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        logits, _ = self.model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            sound_clips=torch.zeros(1, 1, device=get_current_device()),
            sound_length=None,
        )

        assert logits.shape == torch.Size([1, 3, VOCAB_SIZE])


class TestLLaVARadioDynamicResolution:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_model(
            vision_transformer_config=_vision_config("radio"),
            drop_vision_class_token=True,
            dynamic_resolution=True,
        ).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_packed_dynamic_resolution_forward_matches_composed_pipeline(self):
        device = get_current_device()
        patches_per_image = (IMG_SIZE // PATCH_DIM) * (IMG_SIZE // PATCH_DIM)
        global_t = torch.zeros(1, 2 * patches_per_image, 3 * PATCH_DIM * PATCH_DIM, device=device)
        global_t[0, :, :] = torch.arange(2 * patches_per_image, device=device).unsqueeze(1).float()
        imgs_sizes = torch.tensor([[IMG_SIZE, IMG_SIZE], [IMG_SIZE, IMG_SIZE]], device=device)

        def _packed_params():
            cu = torch.tensor(
                [0, patches_per_image, 2 * patches_per_image], dtype=torch.int32, device=device
            )
            return PackedSeqParams(
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=torch.tensor(patches_per_image, dtype=torch.int32, device=device),
                max_seqlen_kv=torch.tensor(patches_per_image, dtype=torch.int32, device=device),
                qkv_format='thd',
            )

        input_ids = torch.tensor([[10, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 11]]).to(device)
        position_ids = torch.arange(4).unsqueeze(0).to(device)

        logits, _ = self.model(
            images=global_t,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            imgs_sizes=imgs_sizes,
            vision_packed_seq_params=_packed_params(),
        )

        # radio.forward extends the caller's cu_seqlens with the class tokens
        # in place, so the reference call needs a fresh patch-only PackedSeqParams.
        vision_out = self.model.vision_model(
            global_t, imgs_sizes=imgs_sizes, packed_seq_params=_packed_params()
        )
        ct_len = self.model.vision_model.class_token_len
        seq_lens = [patches_per_image + ct_len] * 2
        chunks = torch.split(vision_out.squeeze(0), seq_lens, dim=0)
        stripped = [chunk[ct_len:] for chunk in chunks]
        image_embeddings = self.model.vision_projection(
            torch.cat(stripped, dim=0).unsqueeze(0).permute(1, 0, 2)
        )
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
            num_image_tiles=torch.tensor(
                [patches_per_image, patches_per_image], dtype=torch.int, device=device
            ),
            is_packed_dynamic_res=True,
        )
        expected = self.model.language_model(
            input_ids=None, position_ids=None, attention_mask=None, decoder_input=combined
        )
        expected_seq_len = 2 * patches_per_image - 2 + 4
        assert logits.shape == torch.Size([1, expected_seq_len, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


class TestLLaVAEncoderOnlyForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_model(add_decoder=False).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_forward_returns_projected_image_embeddings(self):
        images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        image_embeddings, loss_mask = self.model(
            images=images, input_ids=None, position_ids=None, attention_mask=None
        )

        expected = self.model.vision_projection(self.model.vision_model(images).permute(1, 0, 2))
        assert loss_mask is None
        assert image_embeddings.shape == torch.Size([IMG_SEQ_LEN, 1, LANGUAGE_HIDDEN])
        torch.testing.assert_close(image_embeddings, expected, rtol=1e-5, atol=1e-5)
