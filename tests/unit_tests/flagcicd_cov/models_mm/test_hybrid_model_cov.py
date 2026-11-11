# Copyright (c) 2026, BAAI. All rights reserved.
"""Hybrid language model: deprecated pattern and ratio arguments, yarn rotary embeddings, offloading markers, and forward/inference logits."""
import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.moe.paged_stash import PagedStashManager
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

VOCAB_SIZE = 96
MAX_SEQUENCE_LENGTH = 8
MICRO_BATCH_SIZE = 2


def _hybrid_config(**overrides):
    return TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **overrides,
    )


def _build_hybrid_model(config=None, **kwargs):
    model_config = config if config is not None else _hybrid_config()
    return HybridModel(
        config=model_config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        hybrid_layer_pattern=kwargs.pop('hybrid_layer_pattern', 'M*'),
        position_embedding_type=kwargs.pop('position_embedding_type', 'rope'),
        **kwargs,
    )


def _forward_inputs(sequence_length=None):
    sequence_length = sequence_length or MAX_SEQUENCE_LENGTH
    data = list(range(sequence_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((MICRO_BATCH_SIZE, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((MICRO_BATCH_SIZE, 1)).cuda()
    attention_mask = torch.ones(
        (MICRO_BATCH_SIZE, 1, sequence_length, sequence_length), dtype=bool
    ).cuda()
    return input_ids, position_ids, attention_mask


def _yarn_config():
    config = _hybrid_config()
    config.yarn_rotary_scaling_factor = 2.0
    config.yarn_original_max_position_embeddings = 4
    config.yarn_beta_fast = 32.0
    config.yarn_beta_slow = 1.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 0.0
    config.yarn_correction_range_round_to_int = True
    return config


class TestHybridModelConstruction:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        initialize_rng_tracker(force_reset=True)
        off_interface.reset_instance()

    def test_use_mup_logs_warning_and_builds_model(self, monkeypatch):
        monkeypatch.setattr(HybridModel, 'mup_warning_printed', False, raising=False)
        config = _hybrid_config(use_mup=True)

        model = _build_hybrid_model(config=config)

        assert HybridModel.mup_warning_printed is True
        assert model.decoder.layers[0].layer_number == 1

    def test_deprecated_override_pattern_is_used_when_pattern_is_absent(self):
        model = _build_hybrid_model(hybrid_override_pattern='-M')

        assert model.hybrid_layer_pattern == '-M'

    def test_override_pattern_conflicting_with_layer_pattern_raises(self):
        with pytest.raises(ValueError, match="cannot both be set"):
            _build_hybrid_model(hybrid_layer_pattern='M*', hybrid_override_pattern='-M')

    def test_ratios_conflicting_with_layer_pattern_raises(self):
        with pytest.raises(ValueError, match="cannot be used together with"):
            _build_hybrid_model(hybrid_layer_pattern='M*', hybrid_attention_ratio=0.5)

    def test_deprecated_ratios_generate_the_layer_pattern(self):
        model = _build_hybrid_model(hybrid_attention_ratio=1 / 2)

        assert model.hybrid_layer_pattern == 'M*'
        assert [type(layer).__name__ for layer in model.decoder.layers] == [
            'MambaLayer',
            'TransformerLayer',
        ]

    def test_yarn_position_embedding_builds_yarn_rotary(self):
        model = _build_hybrid_model(config=_yarn_config(), position_embedding_type='yarn')

        assert model.rotary_pos_emb.__class__.__name__ == 'YarnRotaryEmbedding'
        rotary_pos_emb, _ = model.rotary_pos_emb(MAX_SEQUENCE_LENGTH)
        assert rotary_pos_emb.shape == torch.Size(
            [MAX_SEQUENCE_LENGTH, model.config.kv_channels // 2]
        )
        assert torch.isfinite(rotary_pos_emb).all()

    def test_preprocess_for_fine_grained_offloading_marks_params(self):
        model = _build_hybrid_model()

        model.preprocess_for_fine_grained_offloading()

        assert model.disable_param_offloading is False
        assert all(param.offloading_activation is False for param in model.decoder.parameters())
        assert all(
            param.offloading_activation is False for param in model.output_layer.parameters()
        )
        assert model.decoder.final_norm.weight.offloading_activation is False

    def test_preprocess_for_paged_stash_initializes_chunk_handler(self, monkeypatch):
        monkeypatch.setattr(PagedStashManager, 'STASH_MGR', None, raising=False)
        model = _build_hybrid_model()

        model.preprocess_for_paged_stash()

        assert PagedStashManager.get_instance().vp_size == 1
        assert PagedStashManager.get_instance().current_vp_stage == 0

    def test_set_input_tensor_reaches_decoder(self):
        model = _build_hybrid_model()
        input_tensor = torch.ones((MAX_SEQUENCE_LENGTH, MICRO_BATCH_SIZE, 256))

        model.set_input_tensor(input_tensor)

        assert model.decoder.input_tensor.shape == (
            MAX_SEQUENCE_LENGTH,
            MICRO_BATCH_SIZE,
            model.config.hidden_size,
        )

    def test_block_scope_config_creates_cudagraph_manager(self):
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = _hybrid_config(
            cuda_graph_impl='local', inference_cuda_graph_scope=InferenceCudaGraphScope.block
        )
        model = _build_hybrid_model(config=config)

        assert hasattr(model, 'cudagraph_manager')
        assert not hasattr(model.decoder.layers[1], 'cudagraph_manager')

    def test_local_cudagraph_predicate_requires_decode_only_inference(self):
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = _hybrid_config(
            cuda_graph_impl='local', inference_cuda_graph_scope=InferenceCudaGraphScope.block
        )
        model = _build_hybrid_model(config=config).cuda()
        context = StaticInferenceContext(
            max_batch_size=MICRO_BATCH_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH
        )
        input_ids, position_ids, attention_mask = _forward_inputs()

        with InferenceMode.active():
            assert (
                model._should_call_local_cudagraph(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inference_context=context,
                )
                is False
            )
            context.enable_decode_mode()
            assert (
                model._should_call_local_cudagraph(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inference_context=context,
                )
                is True
            )

        assert (
            model._should_call_local_cudagraph(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inference_context=context,
            )
            is False
        )


class TestHybridModelForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _build_hybrid_model()
        self.model.cuda()
        self.model.eval()
        off_interface.reset_instance()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        off_interface.reset_instance()
        PagedStashManager.STASH_MGR = None

    def test_forward_composes_embedding_decoder_and_output_layer(self):
        input_ids, position_ids, attention_mask = _forward_inputs()

        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        decoder_input = self.model.embedding(input_ids=input_ids, position_ids=position_ids)
        rotary_seq_len = self.model.rotary_pos_emb.get_rotary_seq_len(
            None, self.model.decoder, decoder_input, self.model.config, None
        )
        rotary_pos_emb = self.model.rotary_pos_emb(rotary_seq_len)
        hidden_states = self.model.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        expected, _ = self.model.output_layer(
            hidden_states, weight=None, runtime_gather_output=None
        )
        expected = expected.transpose(0, 1).contiguous()
        assert logits.shape == torch.Size([MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH, VOCAB_SIZE])
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)

    def test_forward_with_labels_returns_per_token_cross_entropy(self):
        input_ids, position_ids, attention_mask = _forward_inputs()
        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        labels = torch.randint(0, VOCAB_SIZE, (MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH)).cuda()

        loss = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), reduction='none'
        ).reshape(MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH)
        assert loss.shape == torch.Size([MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH])
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    def test_forward_with_explicit_decoder_input_skips_embedding(self):
        input_ids, position_ids, attention_mask = _forward_inputs()
        decoder_input = self.model.embedding(input_ids=input_ids, position_ids=position_ids)
        logits_from_ids = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        logits = self.model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
        )

        assert logits.shape == torch.Size([MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH, VOCAB_SIZE])
        torch.testing.assert_close(logits, logits_from_ids, rtol=1e-5, atol=1e-5)

    def test_forward_with_yarn_rotary_matches_manual_composition(self):
        model = _build_hybrid_model(config=_yarn_config(), position_embedding_type='yarn').cuda()
        model.eval()
        input_ids, position_ids, attention_mask = _forward_inputs()
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)
        rotary_seq_len = model.rotary_pos_emb.get_rotary_seq_len(
            None, model.decoder, decoder_input, model.config, None
        )
        rotary_pos_emb, _ = model.rotary_pos_emb(rotary_seq_len)
        hidden_states = model.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        expected, _ = model.output_layer(hidden_states, weight=None, runtime_gather_output=None)

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape == torch.Size([MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH, VOCAB_SIZE])
        torch.testing.assert_close(
            logits, expected.transpose(0, 1).contiguous(), rtol=1e-5, atol=1e-5
        )

    def test_forward_runs_fine_grained_offloading_preprocessing(self):
        config = _hybrid_config(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn'],
            cuda_graph_impl='full_iteration',
            cuda_graph_warmup_steps=1,
            fine_grained_offloading_max_inflight_offloads=2,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=1.0,
            use_transformer_engine_op_fuser=True,
            moe_flex_dispatcher_backend='hybridep',
        )
        model = _build_hybrid_model(config=config).cuda()
        model.eval()
        input_ids, position_ids, attention_mask = _forward_inputs()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape == torch.Size([MICRO_BATCH_SIZE, MAX_SEQUENCE_LENGTH, VOCAB_SIZE])
        assert model.disable_param_offloading is False
        assert all(param.offloading_activation is False for param in model.decoder.parameters())
        assert PagedStashManager.get_instance().vp_size == 1

    def test_inference_forward_returns_last_token_logits(self):
        input_ids, position_ids, attention_mask = _forward_inputs()
        full_logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        inference_context = StaticInferenceContext(
            max_batch_size=MICRO_BATCH_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH
        )

        with InferenceMode.active():
            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inference_context=inference_context,
                runtime_gather_output=True,
            )

        assert logits.shape == torch.Size([MICRO_BATCH_SIZE, 1, VOCAB_SIZE])
        torch.testing.assert_close(logits[:, 0, :], full_logits[:, -1, :], rtol=1e-4, atol=1e-4)
