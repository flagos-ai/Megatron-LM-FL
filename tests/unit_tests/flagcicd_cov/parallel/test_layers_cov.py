# Copyright (c) 2026, BAAI. All rights reserved.
'''Column/Row/Vocab-parallel layers: master-weight CPU sharding and weight-shard forward results against torch.nn.functional.linear with gather, sequence-parallel and expert variants, plus their gradients.'''

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    _initialize_affine_weight_cpu,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)
        model_parallel_cuda_manual_seed(123)

    yield setup
    Utils.destroy_model_parallel()


def _cpu_init_config():
    return TransformerConfig(
        num_layers=2, hidden_size=16, num_attention_heads=4, use_cpu_initialization=True
    )


def _gpu_config(**overrides):
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


def test_column_parallel_linear_cpu_initialization_matches_master_weight_shard(model_parallel):
    model_parallel(2, 1)
    torch.manual_seed(1234)
    config = _cpu_init_config()
    layer = ColumnParallelLinear(
        16,
        16,
        config=config,
        init_method=config.init_method,
        bias=True,
        skip_bias_add=False,
        keep_master_weight_for_test=True,
    )
    rank = parallel_state.get_tensor_model_parallel_rank()

    assert layer.weight.shape == (8, 16)
    assert layer.master_weight.shape == (16, 16)
    assert torch.equal(layer.weight.data, layer.master_weight[rank * 8 : (rank + 1) * 8])
    assert 'in_features=16, out_features=16, bias=True, TP=2' in repr(layer)


def test_row_parallel_linear_cpu_initialization_matches_master_weight_shard(model_parallel):
    model_parallel(2, 1)
    torch.manual_seed(1234)
    config = _cpu_init_config()
    layer = RowParallelLinear(
        16,
        16,
        config=config,
        init_method=config.init_method,
        bias=True,
        input_is_parallel=False,
        skip_bias_add=False,
        keep_master_weight_for_test=True,
    )
    rank = parallel_state.get_tensor_model_parallel_rank()

    assert layer.weight.shape == (16, 8)
    assert layer.master_weight.shape == (16, 16)
    assert torch.equal(layer.weight.data, layer.master_weight[:, rank * 8 : (rank + 1) * 8])
    assert 'in_features=16, out_features=16, bias=True, TP=2' in repr(layer)


def test_initialize_affine_weight_cpu_shards_with_stride_when_rank_not_given(model_parallel):
    model_parallel(2, 1)
    torch.manual_seed(1234)
    weight = torch.nn.Parameter(torch.empty(8, 16))
    master_weight = _initialize_affine_weight_cpu(
        weight, 16, 16, 8, 0, torch.nn.init.normal_, stride=2, return_master_weight=True
    )
    rank = parallel_state.get_tensor_model_parallel_rank()

    expected = torch.cat(
        [
            master_weight[rank * 4 : (rank + 1) * 4],
            master_weight[8 + rank * 4 : 8 + (rank + 1) * 4],
        ],
        dim=0,
    )
    assert torch.equal(weight.data, expected)


def test_column_parallel_linear_forward_with_gather_output_matches_torch_linear(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=True,
        skip_bias_add=False,
        gather_output=True,
    )
    torch.manual_seed(42)
    full_weight = torch.randn(16, 16, device='cuda', requires_grad=True)
    full_bias = torch.randn(16, device='cuda', requires_grad=True)
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8].detach())
    layer.bias.data.copy_(full_bias[rank * 8 : (rank + 1) * 8].detach())
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full.detach().clone().requires_grad_(True)

    output, output_bias = layer(x)
    expected = F.linear(x_full, full_weight, full_bias)
    assert output_bias is None
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    grad_output = torch.randn(4, 2, 16, device='cuda')
    (output * grad_output).sum().backward()
    (expected * grad_output).sum().backward()
    torch.testing.assert_close(x.grad, x_full.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        layer.weight.grad, full_weight.grad[rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        layer.bias.grad, full_bias.grad[rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )


def test_column_parallel_linear_sequence_parallel_matches_torch_linear(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(sequence_parallel=True, tensor_model_parallel_size=2),
        init_method=torch.nn.init.normal_,
        bias=True,
        skip_bias_add=False,
        gather_output=False,
    )
    torch.manual_seed(43)
    full_weight = torch.randn(16, 16, device='cuda')
    full_bias = torch.randn(16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    layer.bias.data.copy_(full_bias[rank * 8 : (rank + 1) * 8])
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full[rank * 2 : (rank + 1) * 2].detach().clone().requires_grad_(True)

    output, _ = layer(x)
    expected_full = F.linear(x_full, full_weight, full_bias)
    torch.testing.assert_close(
        output, expected_full[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )

    grad_full = torch.randn(4, 2, 16, device='cuda')
    (output * grad_full[:, :, rank * 8 : (rank + 1) * 8]).sum().backward()
    (expected_full * grad_full).sum().backward()
    torch.testing.assert_close(x.grad, x_full.grad[rank * 2 : (rank + 1) * 2], rtol=1e-5, atol=1e-5)


def test_column_parallel_linear_runtime_gather_output_overrides_constructor(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
    )
    torch.manual_seed(44)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    x = torch.randn(4, 2, 16, device='cuda')

    output, _ = layer(x, runtime_gather_output=True)
    expected = F.linear(x, full_weight)
    assert output.shape == (4, 2, 16)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_column_parallel_linear_inference_optimized_all_gather(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(transformer_impl='inference_optimized'),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        gather_output=True,
    )
    layer.eval()
    torch.manual_seed(45)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    x = torch.randn(4, 2, 16, device='cuda')

    output, _ = layer(x)
    expected = F.linear(x, full_weight)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_column_parallel_linear_rejects_missing_weight_when_allocation_skipped(model_parallel):
    model_parallel(2, 1)
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=True,
    )
    with pytest.raises(
        RuntimeError, match='weight was not supplied to ColumnParallelLinear forward pass'
    ):
        layer(torch.randn(2, 2, 16, device='cuda'))


def test_column_parallel_linear_rejects_supplied_weight_of_wrong_shape(model_parallel):
    model_parallel(2, 1)
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=True,
    )
    with pytest.raises(
        RuntimeError, match="supplied weight's shape is \\(4, 16\\), not \\(8, 16\\) as expected"
    ):
        layer(torch.randn(2, 2, 16, device='cuda'), weight=torch.randn(4, 16, device='cuda'))


def test_row_parallel_linear_scatters_input_when_not_parallel(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = RowParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=True,
        input_is_parallel=False,
        skip_bias_add=False,
    )
    torch.manual_seed(46)
    full_weight = torch.randn(16, 16, device='cuda', requires_grad=True)
    full_bias = torch.randn(16, device='cuda')
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8].detach())
    layer.bias.data.copy_(full_bias)
    x = torch.randn(4, 2, 16, device='cuda', requires_grad=True)

    output, output_bias = layer(x)
    assert output_bias is None
    torch.testing.assert_close(output, F.linear(x, full_weight, full_bias), rtol=1e-5, atol=1e-5)

    grad_output = torch.randn(4, 2, 16, device='cuda')
    (output * grad_output).sum().backward()
    expected = F.linear(x, full_weight, full_bias)
    (expected * grad_output).sum().backward()
    torch.testing.assert_close(
        layer.weight.grad, full_weight.grad[:, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )


def test_row_parallel_linear_sequence_parallel_reduce_scatters_output(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = RowParallelLinear(
        16,
        16,
        config=_gpu_config(sequence_parallel=True, tensor_model_parallel_size=2),
        init_method=torch.nn.init.normal_,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
    )
    torch.manual_seed(47)
    full_weight = torch.randn(16, 16, device='cuda')
    full_bias = torch.randn(16, device='cuda')
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8])
    layer.bias.data.copy_(full_bias)
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full[:, :, rank * 8 : (rank + 1) * 8].detach().clone().requires_grad_(True)

    output, output_bias = layer(x)
    assert output_bias is None
    expected_full = F.linear(x_full, full_weight, full_bias)
    torch.testing.assert_close(
        output, expected_full[rank * 2 : (rank + 1) * 2], rtol=1e-5, atol=1e-5
    )

    grad_full = torch.randn(4, 2, 16, device='cuda')
    (output * grad_full[rank * 2 : (rank + 1) * 2]).sum().backward()
    (expected_full * grad_full).sum().backward()
    torch.testing.assert_close(
        x.grad, x_full.grad[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )


def test_row_parallel_linear_expert_bypasses_reduce_and_bias(model_parallel):
    model_parallel(2, 1, expert_model_parallel_size=2)
    rank = parallel_state.get_tensor_model_parallel_rank()
    layer = RowParallelLinear(
        16,
        16,
        config=_gpu_config(expert_model_parallel_size=2),
        init_method=torch.nn.init.normal_,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=True,
    )
    torch.manual_seed(48)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8])
    layer.bias.data.copy_(torch.randn(16, device='cuda'))
    x_full = torch.randn(4, 2, 16, device='cuda')
    x = x_full[:, :, rank * 8 : (rank + 1) * 8]

    output, output_bias = layer(x)
    assert output_bias is layer.bias
    expected = x @ full_weight[:, rank * 8 : (rank + 1) * 8].t()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_vocab_parallel_embedding_inference_optimized_reduce_scatter(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    embedding = VocabParallelEmbedding(
        16,
        8,
        init_method=torch.nn.init.normal_,
        reduce_scatter_embeddings=True,
        config=_gpu_config(transformer_impl='inference_optimized'),
    )
    embedding.eval()
    torch.manual_seed(53)
    full_weight = torch.randn(16, 8, device='cuda')
    with torch.no_grad():
        embedding.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    ids = torch.randint(0, 16, (2, 4), device='cuda')

    output = embedding(ids)
    expected = full_weight[ids].transpose(0, 1)[rank * 2 : (rank + 1) * 2]
    assert output.shape == (2, 2, 8)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
