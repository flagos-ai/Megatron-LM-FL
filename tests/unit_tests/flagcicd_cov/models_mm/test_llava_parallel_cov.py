# Copyright (c) 2026, BAAI. All rights reserved.
"""LLaVA parallel paths: sequence-parallel embedding sharding with padding, pipeline-parallel sequence padding, and checkpoint load-state hooks."""
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
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


def _vision_config():
    config = TransformerConfig(
        num_layers=2, hidden_size=VISION_HIDDEN, num_attention_heads=2, use_cpu_initialization=True
    )
    config.vision_model_type = "clip"
    return config


def _build_model(language_config, **overrides):
    language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
    kwargs = dict(
        language_transformer_config=language_config,
        language_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=language_layer_submodules
        ),
        language_vocab_size=VOCAB_SIZE,
        language_max_sequence_length=64,
        vision_transformer_config=_vision_config(),
        vision_transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        drop_vision_class_token=False,
        vision_projection_config=TransformerConfig(
            num_layers=1,
            hidden_size=LANGUAGE_HIDDEN,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
        ),
        vision_projection_layer_spec=get_submodules(language_layer_submodules.mlp),
        img_h=IMG_SIZE,
        img_w=IMG_SIZE,
        patch_dim=PATCH_DIM,
    )
    kwargs.update(overrides)
    return LLaVAModel(**kwargs)


class TestLLaVASequenceParallel:
    def setup_method(self, method):
        Utils.initialize_model_parallel(2, 1)
        model_parallel_cuda_manual_seed(123)
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=LANGUAGE_HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        self.model = _build_model(language_config).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_embedding_sharding_pads_to_tp_multiple_and_scatters(self):
        from megatron.core.parallel_state import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
        combined = torch.randn(19, 1, LANGUAGE_HIDDEN, device=get_current_device())
        labels = torch.full((1, 19), -100, device=get_current_device())
        labels[0, 5] = 3
        loss_mask = torch.zeros(1, 19, device=get_current_device())
        loss_mask[0, 5] = 1.0

        sharded, sharded_labels, sharded_loss_mask, _ = (
            self.model._process_embedding_token_parallel(combined, labels, loss_mask, None)
        )

        assert sharded.shape == torch.Size([10, 1, LANGUAGE_HIDDEN])
        padded = torch.nn.functional.pad(combined, (0, 0, 0, 0, 0, 1))
        torch.testing.assert_close(sharded, padded[tp_rank * 10 : (tp_rank + 1) * 10])
        assert sharded_labels.shape == torch.Size([1, 20])
        assert sharded_labels[0, 5].item() == 3
        assert sharded_labels[0, 19].item() == -100
        assert sharded_loss_mask.shape == torch.Size([1, 20])
        assert sharded_loss_mask[0, 5].item() == 1.0
        assert sharded_loss_mask[0, 19].item() == 0.0


class TestLLaVAPipelineParallelPadding:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 2)
        model_parallel_cuda_manual_seed(123)
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=LANGUAGE_HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        self.model = _build_model(language_config).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_pipeline_parallel_pads_sequence_to_max_length(self):
        device = get_current_device()
        input_ids = torch.tensor([[10, 11, IMAGE_TOKEN_INDEX, 12]]).to(device)
        labels = torch.tensor([[11, IMAGE_TOKEN_INDEX, 12, 13]]).to(device)
        loss_mask = torch.ones(1, 4, device=device)
        image_embeddings = torch.zeros(IMG_SEQ_LEN, 1, LANGUAGE_HIDDEN, device=device)
        language_embeddings = torch.zeros(1, 4, LANGUAGE_HIDDEN, device=device)

        final_embedding, final_labels, final_loss_mask = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=IMAGE_TOKEN_INDEX,
            num_image_tiles=torch.tensor([1], dtype=torch.int, device=device),
        )

        expanded_len = IMG_SEQ_LEN - 1 + 4
        assert expanded_len < 64
        assert final_embedding.shape == torch.Size([64, 1, LANGUAGE_HIDDEN])
        assert torch.count_nonzero(final_embedding[expanded_len:]) == 0
        assert final_labels.shape == torch.Size([1, 64])
        assert torch.equal(
            final_labels[0, expanded_len:], torch.full((64 - expanded_len,), -100, device=device)
        )
        assert final_loss_mask.shape == torch.Size([1, 64])
        assert torch.equal(
            final_loss_mask[0, expanded_len:], torch.zeros(64 - expanded_len, device=device)
        )


class TestLLaVALoadStateDictHooks:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_extra_state_keys_are_ignored_on_strict_load(self):
        model = _build_model(
            TransformerConfig(
                num_layers=2,
                hidden_size=LANGUAGE_HIDDEN,
                num_attention_heads=4,
                use_cpu_initialization=True,
            )
        )
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if value is None:
                continue
            if torch.is_floating_point(value):
                state_dict[key] = value + 1.0
        state_dict["language_model.decoder.layers.0.self_attention._extra_state"] = torch.zeros(1)

        model.load_state_dict(state_dict, strict=True)

        restored = model.state_dict()
        # The injected TE extra-state entry is dropped by the load hook, not loaded.
        extra_key = "language_model.decoder.layers.0.self_attention._extra_state"
        assert extra_key not in restored
        for key, value in state_dict.items():
            if value is None or key == extra_key:
                continue
            if torch.is_floating_point(value):
                torch.testing.assert_close(restored[key], value)

    def test_missing_vision_projection_keys_are_ignored_when_allowed(self):
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=LANGUAGE_HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        model = _build_model(language_config, allow_missing_vision_projection_checkpoint=True)
        projection_before = {
            name: value.clone() for name, value in model.vision_projection.state_dict().items()
        }
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if value is None:
                continue
            if torch.is_floating_point(value):
                state_dict[key] = value + 1.0
        for key in list(state_dict.keys()):
            if key.startswith("vision_projection."):
                del state_dict[key]

        result = model.load_state_dict(state_dict, strict=True)

        assert result.missing_keys == []
        assert (
            model.vision_projection.encoder.linear_fc1.weight.shape
            == projection_before["encoder.linear_fc1.weight"].shape
        )
        for name, value in projection_before.items():
            torch.testing.assert_close(model.vision_projection.state_dict()[name], value)
        language_weight = model.language_model.decoder.layers[0].self_attention.linear_qkv.weight
        torch.testing.assert_close(
            language_weight,
            state_dict["language_model.decoder.layers.0.self_attention.linear_qkv.weight"],
        )
