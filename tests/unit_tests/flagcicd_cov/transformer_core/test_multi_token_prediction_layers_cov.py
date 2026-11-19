# Copyright (c) 2026, BAAI. All rights reserved.
"""MTP layer construction, rolled forward, checkpointed recompute and single-position decode."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_spec,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    defaults = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        mtp_num_layers=2,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _mtp_layer(model_parallel, config):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    spec = get_gpt_mtp_block_spec(
        config=config, spec=get_gpt_layer_local_spec(), use_transformer_engine=False
    )
    block = MultiTokenPredictionBlock(config=config, spec=spec)
    return block.layers[0].cuda().eval()


def _compose_reference(layer, decoder_input, hidden_states, attention_mask):
    with torch.no_grad():
        normed_decoder = apply_module(layer.enorm)(decoder_input)
        normed_hidden = apply_module(layer.hnorm)(hidden_states)
        projected, _ = layer.eh_proj(torch.cat((normed_decoder, normed_hidden), dim=-1))
        layer_output, _ = layer.mtp_model_layer(
            hidden_states=projected, attention_mask=attention_mask
        )
        return apply_module(layer.final_layernorm)(layer_output)


def _embedding(embed):
    def embedding(input_ids, position_ids):
        return embed(input_ids).transpose(0, 1)

    return embedding


def test_mtp_block_builds_layers_with_per_depth_numbering(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config()
    spec = get_gpt_mtp_block_spec(
        config=config, spec=get_gpt_layer_local_spec(), use_transformer_engine=False
    )

    block = MultiTokenPredictionBlock(config=config, spec=spec)

    assert len(block.layers) == 2
    for depth in range(2):
        layer = block.layers[depth]
        assert layer.layer_number == depth + 1
        assert layer.enorm.weight.shape == (32,)
        assert layer.hnorm.weight.shape == (32,)
        assert layer.eh_proj.weight.shape == (32, 64)
        assert layer.final_layernorm.weight.shape == (32,)
    assert block.layers[0].mtp_model_layer is not block.layers[1].mtp_model_layer


def test_mtp_block_repeated_layer_builds_single_shared_layer(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(mtp_use_repeated_layer=True)
    spec = get_gpt_mtp_block_spec(
        config=config, spec=get_gpt_layer_local_spec(), use_transformer_engine=False
    )

    block = MultiTokenPredictionBlock(config=config, spec=spec)

    assert len(block.layers) == 1
    assert block.layers[0].layer_number == 1


def test_mtp_block_repeated_layer_warns_when_given_multiple_specs(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(mtp_use_repeated_layer=True)
    layer_spec = get_mtp_layer_spec(get_gpt_layer_local_spec(), use_transformer_engine=False)
    two_layer_submodules = MultiTokenPredictionBlockSubmodules(layer_specs=[layer_spec] * 2)

    with pytest.warns(UserWarning, match="Repeated MTP mode expects exactly 1 layer spec"):
        block = MultiTokenPredictionBlock(config=config, spec=two_layer_submodules)

    assert len(block.layers) == 1
    assert block.layers[0].layer_number == 1


def test_mtp_layer_forward_composes_norms_projection_and_transformer(model_parallel):
    layer = _mtp_layer(model_parallel, _config())
    torch.manual_seed(7)
    embed = torch.nn.Embedding(16, 32).cuda().eval()
    input_ids = torch.randint(0, 16, (2, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(2, 1)
    hidden_states = torch.randn(8, 2, 32, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    with torch.no_grad():
        output, rolled_ids, rolled_position_ids, rolled_padding_mask = layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            embedding=_embedding(embed),
        )

    expected_rolled_ids = torch.roll(input_ids, shifts=-1, dims=-1)
    expected_rolled_ids[:, -1] = 0
    assert torch.equal(rolled_ids, expected_rolled_ids)
    expected_rolled_positions = torch.roll(position_ids, shifts=-1, dims=-1)
    expected_rolled_positions[:, -1] = 0
    assert torch.equal(rolled_position_ids, expected_rolled_positions)
    assert rolled_padding_mask is None
    expected = _compose_reference(
        layer, embed(expected_rolled_ids).transpose(0, 1), hidden_states, attention_mask
    )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def test_mtp_layer_forward_single_position_uses_given_token_ids(model_parallel):
    layer = _mtp_layer(model_parallel, _config())
    torch.manual_seed(7)
    embed = torch.nn.Embedding(16, 32).cuda().eval()
    next_token_ids = torch.randint(0, 16, (2, 1), device="cuda")
    position_ids = torch.full((2, 1), 3, device="cuda")
    hidden_states = torch.randn(1, 2, 32, device="cuda")
    attention_mask = torch.ones((1, 1, 1, 1), dtype=bool, device="cuda")

    with torch.no_grad():
        output = layer.forward_single_position(
            hidden_states=hidden_states,
            next_token_ids=next_token_ids,
            position_ids=position_ids,
            embedding=_embedding(embed),
            attention_mask=attention_mask,
        )

    expected = _compose_reference(
        layer, embed(next_token_ids).transpose(0, 1), hidden_states, attention_mask
    )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def test_mtp_layer_full_recompute_uniform_matches_eager_values_and_grads(model_parallel):
    layer = _mtp_layer(
        model_parallel,
        _config(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1),
    )
    torch.manual_seed(7)
    embed = torch.nn.Embedding(16, 32).cuda().eval()
    input_ids = torch.randint(0, 16, (2, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(2, 1)
    hidden_states = torch.randn(8, 2, 32, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    outputs = {}
    hidden_grads = {}
    for mode in ("train", "eval"):
        layer.train(mode == "train")
        hidden = hidden_states.detach().clone().requires_grad_(True)
        output, _, _, _ = layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            attention_mask=attention_mask,
            embedding=_embedding(embed),
        )
        output.sum().backward()
        outputs[mode] = output.detach()
        hidden_grads[mode] = hidden.grad.detach().clone()

    torch.testing.assert_close(outputs["train"], outputs["eval"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(hidden_grads["train"], hidden_grads["eval"], rtol=0.0, atol=0.0)


def test_mtp_layer_block_recompute_method_falls_back_to_eager(model_parallel):
    layer = _mtp_layer(
        model_parallel,
        _config(recompute_granularity='full', recompute_method='block', recompute_num_layers=1),
    )
    torch.manual_seed(7)
    embed = torch.nn.Embedding(16, 32).cuda().eval()
    input_ids = torch.randint(0, 16, (2, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(2, 1)
    hidden_states = torch.randn(8, 2, 32, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    layer.train()
    with pytest.warns(UserWarning, match="recompute_method == 'block' is not supported for MTP"):
        output, _, _, _ = layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            embedding=_embedding(embed),
        )

    layer.eval()
    with torch.no_grad():
        expected, _, _, _ = layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            embedding=_embedding(embed),
        )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
