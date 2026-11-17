# Copyright (c) 2026, BAAI. All rights reserved.
"""Hyper connection stream mixing around attention and MLP against a submodule composition."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        enable_hyper_connections=True,
        num_residual_streams=2,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _hc_layer(config):
    return build_module(
        get_gpt_layer_local_spec(enable_hyper_connection=True), config=config, layer_number=1
    )


def test_hyper_connection_layer_forward_matches_submodule_composition(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    layer = _hc_layer(_config()).cuda().train()
    torch.manual_seed(7)
    hidden_states = torch.randn(8, 2, 64, device="cuda").requires_grad_(True)
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    output, context = layer(hidden_states=hidden_states, attention_mask=attention_mask)

    reference_input = hidden_states.detach().clone().requires_grad_(True)
    hidden, attn_h_res, attn_h_post, residual = layer.self_attention_hyper_connection(
        reference_input, return_residual=True
    )
    attention_output_with_bias = layer.self_attention(
        layer.input_layernorm(hidden), attention_mask=attention_mask
    )
    hidden = layer.self_attention_hyper_connection.fused_h_res_h_post_bda(
        attn_h_res,
        residual,
        attn_h_post,
        attention_output_with_bias,
        layer.hidden_dropout,
        layer.training,
        layer.config.bias_dropout_fusion,
    )
    hidden, mlp_h_res, mlp_h_post, residual = layer.mlp_hyper_connection(
        hidden, return_residual=True
    )
    mlp_output_with_bias = layer.mlp(layer.pre_mlp_layernorm(hidden))
    reference = layer.mlp_hyper_connection.fused_h_res_h_post_bda(
        mlp_h_res,
        residual,
        mlp_h_post,
        mlp_output_with_bias,
        layer.hidden_dropout,
        layer.training,
        layer.config.bias_dropout_fusion,
    )

    assert context is None
    assert output.shape == (8, 2, 64)
    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
    output.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0.0, atol=0.0)


def test_hyper_connection_layer_get_layer_static_inputs_expands_hidden_streams(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _hc_layer(_config())

    static_inputs = layer.get_layer_static_inputs(seq_length=8, micro_batch_size=2)

    assert set(static_inputs) == {"hidden_states", "attention_mask"}
    hidden_states = static_inputs["hidden_states"]
    assert hidden_states.shape == (8, 2, 64)
    assert hidden_states.dtype == torch.bfloat16
    assert hidden_states.requires_grad is True
    expected_mask = (~torch.tril(torch.ones((8, 8))).bool()).reshape(1, 1, 8, 8).tile(2, 1, 1, 1)
    assert torch.equal(static_inputs["attention_mask"].cpu(), expected_mask)


def test_hyper_connection_layer_lists_whole_layer_without_cudagraph_scope(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _hc_layer(_config())

    assert layer._get_submodules_under_cudagraphs() == [layer]


def test_hyper_connection_block_mhc_recompute_matches_eager_values_and_grads(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    hidden_states = torch.randn(8, 2, 32, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    outputs = {}
    hidden_grads = {}
    for mhc_recompute in (False, True):
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(123)
        overrides = (
            dict(recompute_granularity="selective", recompute_modules=["mhc"])
            if mhc_recompute
            else dict()
        )
        block = (
            TransformerBlock(
                _config(**overrides), get_gpt_layer_local_spec(enable_hyper_connection=True)
            )
            .cuda()
            .train()
        )
        hidden = hidden_states.detach().clone().requires_grad_(True)
        output = block(hidden_states=hidden, attention_mask=attention_mask)
        output.sum().backward()
        outputs[mhc_recompute] = output.detach()
        hidden_grads[mhc_recompute] = hidden.grad.detach().clone()

    torch.testing.assert_close(outputs[True], outputs[False], rtol=0.0, atol=0.0)
    torch.testing.assert_close(hidden_grads[True], hidden_grads[False], rtol=0.0, atol=0.0)
