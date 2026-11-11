# Copyright (c) 2026, BAAI. All rights reserved.
"""MimoModel: colocated and non-colocated construction from module grids, text embedding lookup, token-position alignment of modality embeddings, sharded state dicts, and per-rank forward dispatch."""
import pytest
import torch
from torch import nn

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.partition.utils import PartitionAdapter
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device

HIDDEN_SIZE = 64
VOCAB_SIZE = 128
SEQ_LEN = 16
SPECIAL_IMAGE_TOKEN = 50


class _KeywordLinear(nn.Linear):
    """nn.Linear invoked with the ``x=`` keyword MIMO's encode() passes to encoders."""

    def forward(self, x):
        return super().forward(x)


def _language_spec(**config_overrides):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **config_overrides,
    )
    return ModuleSpec(
        module=GPTModel,
        params={
            'config': config,
            'transformer_layer_spec': get_gpt_layer_with_transformer_engine_spec(),
            'vocab_size': VOCAB_SIZE,
            'max_sequence_length': SEQ_LEN,
            'pre_process': True,
            'post_process': True,
        },
    )


def _vision_spec():
    return ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            'encoders': {
                'enc': ModuleSpec(
                    module=_KeywordLinear, params={'in_features': 32, 'out_features': HIDDEN_SIZE}
                )
            },
            'input_projections': [
                ModuleSpec(
                    module=nn.Linear,
                    params={'in_features': HIDDEN_SIZE, 'out_features': HIDDEN_SIZE},
                )
            ],
        },
    )


def _mimo_config(module_to_grid_map=None):
    return MimoModelConfig(
        language_model_spec=_language_spec(),
        modality_submodules_spec={'images': _vision_spec()},
        special_token_ids={'images': SPECIAL_IMAGE_TOKEN},
        module_to_grid_map=module_to_grid_map,
    )


def _image_inputs(num_images):
    return {
        'images': {
            'enc': {
                'x': torch.arange(num_images, dtype=torch.float32, device=get_current_device())
                .unsqueeze(-1)
                .repeat(1, 32)
                + 1.0
            }
        }
    }


def _input_ids_with_images(images_per_sample):
    rows = []
    token = 1
    for count in images_per_sample:
        row = []
        for _ in range(count):
            row.append(SPECIAL_IMAGE_TOKEN)
        while len(row) < SEQ_LEN:
            row.append(token)
            token = (token % 40) + 1
        rows.append(row)
    return torch.tensor(rows, device=get_current_device())


class TestMimoModelColocated:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = MimoModel(_mimo_config()).to(get_current_device())
        self.model.eval()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_colocated_forward_matches_composed_language_model_call(self):
        input_ids = _input_ids_with_images([1, 1])
        position_ids = (
            torch.arange(SEQ_LEN).unsqueeze(0).expand(2, -1).contiguous().to(get_current_device())
        )
        modality_inputs = _image_inputs(2)

        logits, _ = self.model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=modality_inputs
        )

        embeddings = self.model.modality_submodules['images'].forward(
            encoder_inputs=modality_inputs['images']
        )
        text_embeddings = self.model.get_text_embeddings(
            input_ids, position_ids, self.model.special_token_ids
        )
        combined = self.model.align_embeddings_by_token_positions(
            {'images': embeddings, 'text': text_embeddings}, input_ids, self.model.special_token_ids
        )
        expected = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            decoder_input=combined,
            labels=None,
            attention_mask=None,
        )
        assert logits.shape == torch.Size([2, SEQ_LEN, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)

    def test_get_text_embeddings_drops_special_tokens_and_handles_3d_position_ids(self):
        input_ids = _input_ids_with_images([1, 0])
        position_ids = (
            torch.arange(SEQ_LEN).unsqueeze(0).expand(2, -1).contiguous().to(get_current_device())
        )

        text_embeddings = self.model.get_text_embeddings(
            input_ids, position_ids, self.model.special_token_ids
        )

        expected_count = (input_ids != SPECIAL_IMAGE_TOKEN).sum().item()
        assert text_embeddings.shape == torch.Size([expected_count, HIDDEN_SIZE])
        position_ids_3d = position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
        text_embeddings_3d = self.model.get_text_embeddings(
            input_ids, position_ids_3d, self.model.special_token_ids
        )
        torch.testing.assert_close(text_embeddings_3d, text_embeddings)

    def test_align_embeddings_places_modalities_at_token_positions(self):
        image_embeddings = (
            torch.arange(3, dtype=torch.float32).unsqueeze(-1).repeat(1, HIDDEN_SIZE) + 100.0
        )
        text_embeddings = torch.arange(5, dtype=torch.float32).unsqueeze(-1).repeat(1, HIDDEN_SIZE)
        input_ids = torch.tensor(
            [[SPECIAL_IMAGE_TOKEN, 1, SPECIAL_IMAGE_TOKEN, 2], [3, 4, SPECIAL_IMAGE_TOKEN, 5]]
        )

        combined = self.model.align_embeddings_by_token_positions(
            {'images': image_embeddings, 'text': text_embeddings},
            input_ids,
            {'images': SPECIAL_IMAGE_TOKEN},
        )

        assert combined.shape == torch.Size([4, 2, HIDDEN_SIZE])
        expected = torch.zeros(2, 4, HIDDEN_SIZE)
        expected[0, 0] = image_embeddings[0, 0]
        expected[0, 2] = image_embeddings[1, 0]
        expected[0, 1] = text_embeddings[0, 0]
        expected[0, 3] = text_embeddings[1, 0]
        expected[1, 0] = text_embeddings[2, 0]
        expected[1, 1] = text_embeddings[3, 0]
        expected[1, 2] = image_embeddings[2, 0]
        expected[1, 3] = text_embeddings[4, 0]
        torch.testing.assert_close(combined, expected.transpose(0, 1))

    def test_align_embeddings_requires_at_least_one_modality(self):
        with pytest.raises(ValueError, match="At least one modality is required"):
            self.model.align_embeddings_by_token_positions(
                {}, torch.zeros(1, 4, dtype=torch.long), {'images': SPECIAL_IMAGE_TOKEN}
            )

    def test_sharded_state_dict_covers_modality_and_language_entries(self):
        sharded_sd = self.model.sharded_state_dict()

        encoder_weight_key = 'modality_submodules.images.encoders.enc.weight'
        language_embedding_key = 'language_model.embedding.word_embeddings.weight'
        assert encoder_weight_key in sharded_sd
        assert language_embedding_key in sharded_sd
        torch.testing.assert_close(
            sharded_sd[encoder_weight_key].data,
            self.model.modality_submodules['images'].encoders['enc'].weight,
        )


class TestMimoModelPartitionAdapter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_context_parallel_language_config_builds_partition_adapter(self):
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=_language_spec(context_parallel_size=2),
                modality_submodules_spec={'images': _vision_spec()},
                special_token_ids={'images': SPECIAL_IMAGE_TOKEN},
            )
        )

        assert isinstance(model.partition_adapter, PartitionAdapter)
        assert model.partition_adapter.cfg.use_cp is True
        assert model.partition_adapter.cfg.max_seq_len == SEQ_LEN


class TestMimoModelColocatedGridMap:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_grids_without_tp_and_dp_dims_skip_colocated_communicators(self):
        Utils.initialize_model_parallel(1, 1)
        grid = HyperCommGrid(
            shape=[1], dim_names=["pp"], rank_offset=0, backend=Utils.get_backend()
        )
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=_language_spec(),
                modality_submodules_spec={'images': _vision_spec()},
                special_token_ids={'images': SPECIAL_IMAGE_TOKEN},
                module_to_grid_map={'images': grid, MIMO_LANGUAGE_MODULE_KEY: grid},
            )
        )

        assert model.role.mode is ModuleLayout.COLOCATED
        assert model.colocated_comms == {}

    def test_matching_full_grids_build_a_bridge_per_modality(self):
        Utils.initialize_model_parallel(1, 1)
        grid = HyperCommGrid(
            shape=[1, 1, 1, 8],
            dim_names=["tp", "cp", "pp", "dp"],
            rank_offset=0,
            backend=Utils.get_backend(),
        )
        grid.create_pg(["tp"])
        grid.create_pg(["cp"])
        grid.create_pg(["pp"])
        grid.create_pg(["dp"])
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=_language_spec(),
                modality_submodules_spec={'images': _vision_spec()},
                special_token_ids={'images': SPECIAL_IMAGE_TOKEN},
                module_to_grid_map={'images': grid, MIMO_LANGUAGE_MODULE_KEY: grid},
            )
        )

        assert model.role.mode is ModuleLayout.COLOCATED
        assert set(model.colocated_comms.keys()) == {('images', MIMO_LANGUAGE_MODULE_KEY)}

        model.destroy()
        grid.destroy()


class TestMimoModelNonColocated:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        self.rank = torch.distributed.get_rank()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _non_colocated_config(self):
        encoder_grid = HyperCommGrid(
            shape=[1, 1, 1, 4],
            dim_names=["tp", "cp", "pp", "dp"],
            rank_offset=0,
            backend=Utils.get_backend(),
        )
        encoder_grid.create_pg(["pp"])
        encoder_grid.create_pg(["dp"])
        language_grid = HyperCommGrid(
            shape=[1, 1, 1, 4],
            dim_names=["tp", "cp", "pp", "dp"],
            rank_offset=4,
            backend=Utils.get_backend(),
        )
        language_grid.create_pg(["pp"])
        language_grid.create_pg(["dp"])
        return (
            MimoModelConfig(
                language_model_spec=_language_spec(),
                modality_submodules_spec={'images': _vision_spec()},
                special_token_ids={'images': SPECIAL_IMAGE_TOKEN},
                module_to_grid_map={
                    'images': encoder_grid,
                    MIMO_LANGUAGE_MODULE_KEY: language_grid,
                },
            ),
            encoder_grid,
            language_grid,
        )

    def test_role_splits_ranks_between_encoder_and_language_grids(self):
        config, encoder_grid, language_grid = self._non_colocated_config()
        model = MimoModel(config)

        assert model.role.mode is ModuleLayout.NON_COLOCATED
        if self.rank < 4:
            assert set(model.role.modules) == {'images'}
            assert model.language_model is None
            assert 'images' in model.modality_submodules
        else:
            assert set(model.role.modules) == {MIMO_LANGUAGE_MODULE_KEY}
            assert isinstance(model.language_model, GPTModel)
            assert 'images' not in model.modality_submodules

        encoder_grid.destroy()
        language_grid.destroy()

    def test_encoder_ranks_return_projected_embeddings_with_split_sizes(self):
        config, encoder_grid, language_grid = self._non_colocated_config()
        model = MimoModel(config).to(get_current_device())
        model.eval()
        input_ids = _input_ids_with_images([2, 1])
        modality_inputs = _image_inputs(3)

        if self.rank < 4:
            outputs, _ = model(input_ids=input_ids, modality_inputs=modality_inputs)

            expected = model.modality_submodules['images'].forward(
                encoder_inputs=modality_inputs['images']
            )
            assert set(outputs.keys()) == {'images'}
            torch.testing.assert_close(outputs['images'], expected)
            assert outputs['images']._mimo_bridge_split_sizes == [2, 1]

        encoder_grid.destroy()
        language_grid.destroy()

    def test_encoder_ranks_reject_special_tokens_without_inputs(self):
        config, encoder_grid, language_grid = self._non_colocated_config()
        model = MimoModel(config).to(get_current_device())
        input_ids = _input_ids_with_images([1, 0])

        if self.rank < 4:
            with pytest.raises(RuntimeError, match="inputs are missing"):
                model(input_ids=input_ids, modality_inputs=None)

        encoder_grid.destroy()
        language_grid.destroy()

    def test_language_ranks_run_the_language_model_on_text_embeddings(self):
        config, encoder_grid, language_grid = self._non_colocated_config()
        model = MimoModel(config).to(get_current_device())
        model.eval()
        input_ids = _input_ids_with_images([0, 0])
        position_ids = (
            torch.arange(SEQ_LEN).unsqueeze(0).expand(2, -1).contiguous().to(get_current_device())
        )

        if self.rank >= 4:
            logits, _ = model(input_ids=input_ids, position_ids=position_ids, modality_inputs=None)

            text_embeddings = model.get_text_embeddings(
                input_ids, position_ids, model.special_token_ids
            )
            combined = model.align_embeddings_by_token_positions(
                {'text': text_embeddings}, input_ids, model.special_token_ids
            )
            expected = model.language_model(
                input_ids=None,
                position_ids=position_ids,
                decoder_input=combined,
                labels=None,
                attention_mask=None,
            )
            assert logits.shape == torch.Size([2, SEQ_LEN, VOCAB_SIZE])
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)

        encoder_grid.destroy()
        language_grid.destroy()
