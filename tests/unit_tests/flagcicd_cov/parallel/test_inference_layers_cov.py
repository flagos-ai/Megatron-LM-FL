# Copyright (c) 2026, BAAI. All rights reserved.
'''Tensor-parallel inference linears: eval-mode fused gathers and reduce-scatters, training-mode delegation to the parent layers, and fused layer-norm column-parallel matmuls.'''

import pytest
import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceLinear,
    InferenceRowParallelLinear,
    inference_all_gather_from_tensor_model_parallel_region,
    inference_reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)
        model_parallel_cuda_manual_seed(123)

    yield setup
    SymmetricMemoryManager.destroy()
    Utils.destroy_model_parallel()


def _inference_config(**overrides):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        use_cpu_initialization=False,
        perform_initialization=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _sequence_parallel_config(**overrides):
    return _inference_config(sequence_parallel=True, tensor_model_parallel_size=2, **overrides)


def test_inference_linear_eval_applies_local_matmul(model_parallel):
    model_parallel(1, 1)
    layer = InferenceLinear(
        16,
        16,
        parallel_mode='column',
        config=_inference_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
    )
    layer.eval()
    torch.manual_seed(3)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight)
    x = torch.randn(4, 16, device='cuda')

    output, output_bias = layer(x)
    assert output_bias is None
    torch.testing.assert_close(output, x @ full_weight.t(), rtol=1e-5, atol=1e-5)


def test_inference_linear_training_delegates_to_parent(model_parallel):
    model_parallel(1, 1)
    layer = InferenceLinear(
        16,
        16,
        parallel_mode='column',
        config=_inference_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
    )
    layer.train()
    torch.manual_seed(4)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight)
    x = torch.randn(4, 16, device='cuda')

    output, output_bias = layer(x)
    assert output_bias is None
    torch.testing.assert_close(output, x @ full_weight.t(), rtol=1e-5, atol=1e-5)


def test_inference_column_parallel_linear_tp1_matches_torch_linear(model_parallel):
    model_parallel(1, 1)
    layer = InferenceColumnParallelLinear(
        16,
        32,
        config=_inference_config(),
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(5)
    full_weight = torch.randn(32, 16, device='cuda')
    layer.weight.data.copy_(full_weight)
    x = torch.randn(4, 16, device='cuda')

    output, output_bias = layer(x)
    assert output_bias is None
    assert output.shape == (4, 32)
    torch.testing.assert_close(output, x @ full_weight.t(), rtol=1e-5, atol=1e-5)


def test_inference_column_parallel_linear_tp2_gathers_sequence(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = InferenceColumnParallelLinear(
        16,
        32,
        config=_sequence_parallel_config(),
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(6)
    full_weight = torch.randn(32, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 16 : (rank + 1) * 16])
    x_full = torch.randn(4, 16, device='cuda')
    x = x_full[rank * 2 : (rank + 1) * 2]

    output, output_bias = layer(x)
    assert output_bias is None
    assert output.shape == (4, 16)
    expected = x_full @ full_weight[rank * 16 : (rank + 1) * 16].t()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_inference_layer_norm_column_parallel_linear_tp1_matches_reference(model_parallel):
    model_parallel(1, 1)
    eps = 1e-5
    layer = InferenceLayerNormColumnParallelLinear(
        16,
        32,
        config=_inference_config(),
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(7)
    full_weight = torch.randn(32, 16, device='cuda')
    gamma = torch.randn(16, device='cuda')
    layer.weight.data.copy_(full_weight)
    layer.layer_norm_weight.data.copy_(gamma)
    x = torch.randn(4, 16, device='cuda')

    output, output_bias = layer(x)
    assert output_bias is None
    normalized = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma
    torch.testing.assert_close(output, normalized @ full_weight.t(), rtol=1e-5, atol=1e-5)


def test_inference_layer_norm_column_parallel_linear_tp2_gathers_sequence(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    eps = 1e-5
    layer = InferenceLayerNormColumnParallelLinear(
        16,
        32,
        config=_sequence_parallel_config(),
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(8)
    full_weight = torch.randn(32, 16, device='cuda')
    gamma = torch.randn(16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 16 : (rank + 1) * 16])
    layer.layer_norm_weight.data.copy_(gamma)
    x_full = torch.randn(4, 16, device='cuda')
    x = x_full[rank * 2 : (rank + 1) * 2]

    output, output_bias = layer(x)
    assert output_bias is None
    assert output.shape == (4, 16)
    normalized = x_full / torch.sqrt(x_full.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma
    expected = normalized @ full_weight[rank * 16 : (rank + 1) * 16].t()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_inference_row_parallel_linear_tp1_matches_torch_linear(model_parallel):
    model_parallel(1, 1)
    layer = InferenceRowParallelLinear(
        16,
        16,
        config=_inference_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(9)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight)
    x = torch.randn(4, 16, device='cuda')

    output, output_bias = layer(x)
    assert output_bias is None
    torch.testing.assert_close(output, x @ full_weight.t(), rtol=1e-5, atol=1e-5)


def test_inference_row_parallel_linear_tp2_reduce_scatters_output(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = InferenceRowParallelLinear(
        16,
        16,
        config=_sequence_parallel_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(10)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8])
    x_full = torch.randn(4, 16, device='cuda')
    x = x_full[:, rank * 8 : (rank + 1) * 8]

    output, output_bias = layer(x)
    assert output_bias is None
    assert output.shape == (2, 16)
    expected = (x_full @ full_weight.t())[rank * 2 : (rank + 1) * 2]
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_inference_column_parallel_linear_training_delegates_to_parent(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = InferenceColumnParallelLinear(
        16,
        32,
        config=_sequence_parallel_config(),
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    torch.manual_seed(60)
    full_weight = torch.randn(32, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 16 : (rank + 1) * 16])
    x_full = torch.randn(4, 16, device='cuda')
    x = x_full[rank * 2 : (rank + 1) * 2]

    output, output_bias = layer(x)

    assert output_bias is None
    assert output.shape == (4, 16)
    expected = x_full @ full_weight[rank * 16 : (rank + 1) * 16].t()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_inference_row_parallel_linear_training_delegates_to_parent(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = InferenceRowParallelLinear(
        16,
        16,
        config=_sequence_parallel_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
    )
    torch.manual_seed(61)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8])
    x_full = torch.randn(4, 16, device='cuda')
    x = x_full[:, rank * 8 : (rank + 1) * 8]

    output, output_bias = layer(x)

    assert output_bias is None
    assert output.shape == (2, 16)
    expected = (x_full @ full_weight.t())[rank * 2 : (rank + 1) * 2]
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_inference_row_parallel_linear_fused_setters_store_inputs(model_parallel):
    model_parallel(2, 1)
    layer = InferenceRowParallelLinear(
        16,
        16,
        config=_sequence_parallel_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
    )
    norm_weights = torch.randn(16, device='cuda')
    residual = torch.randn(2, 16, device='cuda')

    assert layer.next_layer_norm_weights is None
    layer._set_next_layer_norm_weights(norm_weights)
    assert layer.next_layer_norm_weights is norm_weights
    layer._set_residual(residual)
    assert layer.residual is residual


def test_inference_all_gather_from_tensor_model_parallel_region(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    config = _sequence_parallel_config()
    layer = InferenceColumnParallelLinear(
        16,
        32,
        config=config,
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(11)
    full_weight = torch.randn(32, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 16 : (rank + 1) * 16])
    x_warmup = torch.randn(4, 16, device='cuda')
    warmup = layer(x_warmup[rank * 2 : (rank + 1) * 2])
    torch.testing.assert_close(
        warmup[0], x_warmup @ full_weight[rank * 16 : (rank + 1) * 16].t(), rtol=1e-5, atol=1e-5
    )

    torch.manual_seed(12)
    full = torch.randn(4, 8, device='cuda')
    x = full[:, rank * 4 : (rank + 1) * 4]
    group = get_tensor_model_parallel_group_if_none(None)

    output = inference_all_gather_from_tensor_model_parallel_region(x, group, config)
    torch.testing.assert_close(output, full, rtol=0, atol=0)


def test_inference_all_gather_from_tensor_model_parallel_region_tp_size_one(model_parallel):
    model_parallel(1, 1)
    x = torch.randn(4, 8, device='cuda')
    group = get_tensor_model_parallel_group_if_none(None)

    output = inference_all_gather_from_tensor_model_parallel_region(x, group, _inference_config())
    assert output is x


def test_inference_reduce_scatter_to_sequence_parallel_region(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    config = _sequence_parallel_config()
    layer = InferenceColumnParallelLinear(
        16,
        32,
        config=config,
        init_method=torch.nn.init.normal_,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    layer.eval()
    torch.manual_seed(13)
    full_weight = torch.randn(32, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 16 : (rank + 1) * 16])
    x_warmup = torch.randn(4, 16, device='cuda')
    warmup = layer(x_warmup[rank * 2 : (rank + 1) * 2])
    torch.testing.assert_close(
        warmup[0], x_warmup @ full_weight[rank * 16 : (rank + 1) * 16].t(), rtol=1e-5, atol=1e-5
    )

    torch.manual_seed(14)
    first = torch.randn(4, 2, 8, device='cuda')
    second = torch.randn(4, 2, 8, device='cuda')
    x = first if rank == 0 else second
    group = get_tensor_model_parallel_group_if_none(None)

    output = inference_reduce_scatter_to_sequence_parallel_region(x, group, config)
    expected = (first + second)[rank * 2 : (rank + 1) * 2]
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_inference_reduce_scatter_to_sequence_parallel_region_tp_size_one(model_parallel):
    model_parallel(1, 1)
    x = torch.randn(4, 2, 8, device='cuda')
    group = get_tensor_model_parallel_group_if_none(None)

    output = inference_reduce_scatter_to_sequence_parallel_region(x, group, _inference_config())
    assert output is x
