# Copyright (c) 2026, BAAI. All rights reserved.
"""MLP construction variants and activation-fusion forward paths."""

import pytest
import torch

from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import get_submodules
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
        hidden_size=16,
        ffn_hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _mlp_submodules():
    return get_submodules(get_gpt_layer_local_submodules().mlp)


def _tanh_gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def test_expert_mlp_without_ffn_hidden_size_raises(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config()
    config.ffn_hidden_size = None

    with pytest.raises(ValueError, match="MoE MLP requires `ffn_hidden_size`"):
        MLP(config, _mlp_submodules(), is_expert=True)


@pytest.mark.parametrize(
    ("use_kitchen", "expected_stride"), [(True, 1), (False, 2)], ids=["kitchen", "default"]
)
def test_gated_fc1_partition_stride_is_one_for_kitchen_and_two_otherwise(
    model_parallel, use_kitchen, expected_stride
):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(gated_linear_unit=True, use_kitchen=use_kitchen)

    mlp = MLP(config, _mlp_submodules())

    assert mlp.linear_fc1.weight.partition_stride == expected_stride
    assert mlp.linear_fc1.weight.shape == (2 * config.ffn_hidden_size, config.hidden_size)


def test_kitchen_fc1_stride_keeps_gated_mlp_numerics(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    hidden_states = torch.randn(8, 2, 16)

    outputs = []
    for use_kitchen in (False, True):
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = _config(gated_linear_unit=True, use_kitchen=use_kitchen)
        mlp = MLP(config, _mlp_submodules()).eval()
        with torch.no_grad():
            outputs.append(mlp(hidden_states)[0])

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)


def test_use_te_activation_func_adds_bias_before_activation(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(use_te_activation_func=True, add_bias_linear=True)
    config.activation_func = torch.nn.functional.gelu

    mlp = MLP(config, _mlp_submodules()).eval()
    hidden_states = torch.randn(8, 2, 16)

    with torch.no_grad():
        output, output_bias = mlp(hidden_states)

    with torch.no_grad():
        intermediate, bias = mlp.linear_fc1(hidden_states)
        intermediate = intermediate + bias
        intermediate = torch.nn.functional.gelu(intermediate)
        expected, _ = mlp.linear_fc2(intermediate)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert output_bias.shape == (config.hidden_size,)


def test_fused_bias_geglu_matches_tanh_gelu_reference(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(gated_linear_unit=True, bias_activation_fusion=True, add_bias_linear=True)
    config.activation_func = torch.nn.functional.gelu

    mlp = MLP(config, _mlp_submodules()).eval()
    hidden_states = torch.randn(8, 2, 16)

    with torch.no_grad():
        output, output_bias = mlp(hidden_states)

    with torch.no_grad():
        intermediate, bias = mlp.linear_fc1(hidden_states)
        y_1, y_2 = torch.chunk(intermediate + bias, 2, -1)
        intermediate = _tanh_gelu(y_1) * y_2
        expected, _ = mlp.linear_fc2(intermediate)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(output_bias, mlp.linear_fc2.bias)


def test_fused_bias_gelu_matches_tanh_gelu_reference(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(bias_activation_fusion=True, add_bias_linear=True)
    config.activation_func = torch.nn.functional.gelu

    mlp = MLP(config, _mlp_submodules()).eval()
    hidden_states = torch.randn(8, 2, 16)

    with torch.no_grad():
        output, output_bias = mlp(hidden_states)

    with torch.no_grad():
        intermediate, bias = mlp.linear_fc1(hidden_states)
        expected, _ = mlp.linear_fc2(_tanh_gelu(intermediate + bias))
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(output_bias, mlp.linear_fc2.bias)


def test_weighted_quick_geglu_with_per_token_scale_matches_reference(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(
        gated_linear_unit=True,
        bias_activation_fusion=True,
        add_bias_linear=True,
        glu_linear_offset=0.5,
        activation_func=quick_gelu,
    )

    mlp = MLP(config, _mlp_submodules()).eval()
    hidden_states = torch.randn(16, 16)
    per_token_scale = torch.rand(16)

    with torch.no_grad():
        output, output_bias = mlp(hidden_states, per_token_scale=per_token_scale)

    with torch.no_grad():
        intermediate, bias = mlp.linear_fc1(hidden_states)
        y_1, y_2 = torch.chunk(intermediate + bias, 2, -1)
        intermediate = quick_gelu(y_1) * (y_2 + config.glu_linear_offset)
        intermediate = intermediate * per_token_scale.unsqueeze(-1)
        expected, _ = mlp.linear_fc2(intermediate)
        expected = expected + mlp.linear_fc2.bias * per_token_scale.unsqueeze(-1)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert output_bias is None


def test_fused_activation_with_per_token_scale_rejects_unsupported_activation(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(gated_linear_unit=True, bias_activation_fusion=True, add_bias_linear=True)
    config.activation_func = torch.relu

    mlp = MLP(config, _mlp_submodules()).eval()
    hidden_states = torch.randn(8, 2, 16)
    per_token_scale = torch.rand(8, 2)

    with pytest.raises(ValueError, match="Only support fusion of swiglu and quick_gelu"):
        mlp(hidden_states, per_token_scale=per_token_scale)


def test_fused_activation_rejects_unfused_silu_without_gating(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(bias_activation_fusion=True, add_bias_linear=True)
    config.activation_func = torch.nn.functional.silu

    mlp = MLP(config, _mlp_submodules()).eval()
    hidden_states = torch.randn(8, 2, 16)

    with pytest.raises(ValueError, match="Only support fusion of gelu and swiglu"):
        mlp(hidden_states)
