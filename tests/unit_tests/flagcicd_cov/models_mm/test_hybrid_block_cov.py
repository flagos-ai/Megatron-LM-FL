# Copyright (c) 2026, BAAI. All rights reserved.
"""HybridStack: layer-type dispatch from pattern symbols, mamba state shapes, wrapped inference inputs, static-batching contexts, and full-recompute forwards."""
import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import WrappedTensor
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 256
SEQUENCE_LENGTH = 32
MICRO_BATCH_SIZE = 2


def _pg_collection():
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])


def _stack(layer_pattern, **config_kwargs):
    layer_type_list = validate_segment_layers(layer_pattern)
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=len(layer_type_list),
        num_attention_heads=4,
        use_cpu_initialization=True,
        **config_kwargs,
    )
    return HybridStack(
        config,
        hybrid_stack_spec.submodules,
        layer_type_list=layer_type_list,
        pp_layer_offset=0,
        pg_collection=_pg_collection(),
    )


def _forward_inputs():
    hidden_states = torch.randn((SEQUENCE_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE), device='cuda')
    attention_mask = torch.ones(
        (MICRO_BATCH_SIZE, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=bool, device='cuda'
    )
    return hidden_states, attention_mask


class TestHybridStackConstruction:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gdn_symbol_dispatches_to_the_gdn_layer_spec(self):
        submodules = HybridStackSubmodules(
            gdn_layer=hybrid_stack_spec.submodules.mlp_layer,
            attention_layer=hybrid_stack_spec.submodules.attention_layer,
        )
        layer_type_list = validate_segment_layers(Symbols.GDN + Symbols.ATTENTION)
        config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        stack = HybridStack(
            config,
            submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=_pg_collection(),
        )

        assert isinstance(stack.layers[0], MLPLayer)
        assert isinstance(stack.layers[1], TransformerLayer)
        assert isinstance(stack.layers[1].self_attention, SelfAttention)
        assert [layer.layer_number for layer in stack.layers] == [1, 2]

    def test_unknown_layer_symbol_is_rejected(self):
        config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        with pytest.raises(ValueError, match="unexpected layer_type"):
            HybridStack(
                config,
                hybrid_stack_spec.submodules,
                layer_type_list=['X'],
                pg_collection=_pg_collection(),
            )

    def test_mamba_state_shapes_per_request_returns_conv_and_ssm_shapes(self):
        stack = _stack(Symbols.MAMBA + Symbols.ATTENTION)

        conv_shape, ssm_shape = stack.mamba_state_shapes_per_request()

        assert conv_shape == (2 * HIDDEN_SIZE + 2 * 8 * 128, 4)
        assert ssm_shape == (2 * HIDDEN_SIZE // 64, 64, 128)

    def test_mamba_state_shapes_per_request_is_none_without_mamba_layers(self):
        stack = _stack(Symbols.ATTENTION + Symbols.ATTENTION)

        assert stack.mamba_state_shapes_per_request() is None


class TestHybridStackForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_forward_unwraps_wrapped_hidden_states(self):
        stack = _stack(Symbols.ATTENTION).cuda().eval()
        hidden_states, attention_mask = _forward_inputs()

        direct_output = stack(hidden_states, attention_mask=attention_mask)
        wrapped_output = stack(WrappedTensor(hidden_states), attention_mask=attention_mask)

        assert direct_output.shape == torch.Size([SEQUENCE_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE])
        torch.testing.assert_close(wrapped_output, direct_output, rtol=1e-5, atol=1e-5)

    def test_forward_with_static_context_maps_batching_attributes(self):
        stack = _stack(Symbols.ATTENTION).cuda().eval()
        hidden_states, attention_mask = _forward_inputs()
        reference_output = stack(hidden_states, attention_mask=attention_mask)
        context = StaticInferenceContext(
            max_batch_size=MICRO_BATCH_SIZE, max_sequence_length=SEQUENCE_LENGTH
        )

        output = stack(hidden_states, attention_mask=attention_mask, inference_context=context)

        assert context.max_seqlen == SEQUENCE_LENGTH
        assert context.seqlen_offset == 0
        torch.testing.assert_close(output, reference_output, rtol=1e-5, atol=1e-5)

    def test_forward_with_flash_decode_builds_sequence_len_offset_during_inference(self):
        stack = _stack(Symbols.ATTENTION, flash_decode=True).cuda().eval()
        hidden_states, attention_mask = _forward_inputs()
        reference_output = stack(hidden_states, attention_mask=attention_mask)
        context = StaticInferenceContext(
            max_batch_size=MICRO_BATCH_SIZE, max_sequence_length=SEQUENCE_LENGTH
        )

        with InferenceMode.active():
            output = stack(hidden_states, attention_mask=attention_mask, inference_context=context)

        torch.testing.assert_close(output, reference_output, rtol=1e-5, atol=1e-5)

    def test_full_recompute_forward_matches_eager_forward_and_grads(self):
        dropout_kwargs = dict(hidden_dropout=0.0, attention_dropout=0.0)
        torch.manual_seed(123)
        eager_stack = _stack(Symbols.MAMBA + Symbols.ATTENTION, **dropout_kwargs).cuda()
        torch.manual_seed(123)
        recompute_stack = _stack(
            Symbols.MAMBA + Symbols.ATTENTION,
            **dropout_kwargs,
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
        ).cuda()
        # use_cpu_initialization draws weights from the CPU RNG, which
        # model_parallel_cuda_manual_seed does not reset: copy the eager
        # weights so both stacks are identical by construction.
        recompute_stack.load_state_dict(eager_stack.state_dict())
        eager_stack.train()
        recompute_stack.train()
        torch.manual_seed(7)
        hidden_states, attention_mask = _forward_inputs()
        hidden_states.requires_grad_(True)

        model_parallel_cuda_manual_seed(123)
        eager_output = eager_stack(hidden_states, attention_mask=attention_mask)
        model_parallel_cuda_manual_seed(123)
        recompute_output = recompute_stack(hidden_states, attention_mask=attention_mask)
        eager_output.sum().backward()
        recompute_output.sum().backward()

        torch.testing.assert_close(
            recompute_output, eager_output, rtol=1e-3, atol=1e-3 * eager_output.abs().max().item()
        )
        eager_grad = eager_stack.layers[0].mixer.in_proj.weight.grad
        torch.testing.assert_close(
            recompute_stack.layers[0].mixer.in_proj.weight.grad,
            eager_grad,
            rtol=1e-3,
            atol=1e-3 * eager_grad.abs().max().item(),
        )
