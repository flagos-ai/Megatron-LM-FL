# Copyright (c) 2026, BAAI. All rights reserved.
'''Tensor-parallel linear backward mechanics: frozen-weight dgrad all-reduce, offload-context annotation gating, deferred embedding wgrad, main-grad accumulation across microbatches and backward_dw no-ops.'''

import pytest
import torch

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
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


def test_frozen_weight_column_parallel_linear_sequence_parallel(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(sequence_parallel=True, tensor_model_parallel_size=2),
        init_method=torch.nn.init.normal_,
        bias=True,
        skip_bias_add=False,
        gather_output=False,
    )
    layer.weight.requires_grad_(False)
    torch.manual_seed(49)
    full_weight = torch.randn(16, 16, device='cuda')
    full_bias = torch.randn(16, device='cuda')
    with torch.no_grad():
        layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
        layer.bias.data.copy_(full_bias[rank * 8 : (rank + 1) * 8])
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full[rank * 2 : (rank + 1) * 2].detach().clone().requires_grad_(True)

    output, _ = layer(x)
    expected_full = torch.nn.functional.linear(x_full, full_weight, full_bias)
    torch.testing.assert_close(
        output, expected_full[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )

    grad_full = torch.randn(4, 2, 16, device='cuda')
    (output * grad_full[:, :, rank * 8 : (rank + 1) * 8]).sum().backward()
    (expected_full * grad_full).sum().backward()
    torch.testing.assert_close(x.grad, x_full.grad[rank * 2 : (rank + 1) * 2], rtol=1e-5, atol=1e-5)
    assert layer.weight.grad is None


def test_frozen_weight_column_parallel_linear_allreduces_dgrad(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
    )
    layer.weight.requires_grad_(False)
    torch.manual_seed(50)
    full_weight = torch.randn(16, 16, device='cuda')
    with torch.no_grad():
        layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full.detach().clone().requires_grad_(True)

    output, _ = layer(x)
    expected_full = torch.nn.functional.linear(x_full, full_weight)
    torch.testing.assert_close(
        output, expected_full[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )

    grad_output = torch.randn(4, 2, 16, device='cuda')
    (output * grad_output[:, :, rank * 8 : (rank + 1) * 8]).sum().backward()
    (expected_full * grad_output).sum().backward()
    torch.testing.assert_close(x.grad, x_full.grad, rtol=1e-5, atol=1e-5)


def test_column_parallel_linear_skips_offload_annotation_outside_cpu_offload_context(
    model_parallel,
):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    from megatron.core.extensions.transformer_engine import get_cpu_offload_context

    config = _gpu_config(cpu_offloading=True)
    layer = ColumnParallelLinear(
        16,
        16,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
    )
    offload_context, _ = get_cpu_offload_context(True, 1, 1, True, False, False, False)
    config._cpu_offloading_context = offload_context
    torch.manual_seed(62)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full.detach().clone().requires_grad_(True)

    output, _ = layer(x)

    assert not hasattr(x, 'activation_offloading')
    expected = torch.nn.functional.linear(x_full, full_weight)[:, :, rank * 8 : (rank + 1) * 8]
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    grad_full = torch.randn(4, 2, 16, device='cuda')
    (output * grad_full[:, :, rank * 8 : (rank + 1) * 8]).sum().backward()
    (torch.nn.functional.linear(x_full, full_weight) * grad_full).sum().backward()
    torch.testing.assert_close(x.grad, x_full.grad, rtol=1e-5, atol=1e-5)


def test_row_parallel_linear_skips_offload_annotation_outside_cpu_offload_context(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    from megatron.core.extensions.transformer_engine import get_cpu_offload_context

    config = _gpu_config(cpu_offloading=True)
    layer = RowParallelLinear(
        16,
        16,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
    )
    offload_context, _ = get_cpu_offload_context(True, 1, 1, True, False, False, False)
    config._cpu_offloading_context = offload_context
    torch.manual_seed(63)
    full_weight = torch.randn(16, 16, device='cuda', requires_grad=True)
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8].detach())
    x_full = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    x = x_full[:, :, rank * 8 : (rank + 1) * 8].detach().clone().requires_grad_(True)

    output, _ = layer(x)

    assert not hasattr(x, 'activation_offloading')
    expected = torch.nn.functional.linear(x_full, full_weight)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    grad_output = torch.randn(4, 2, 16, device='cuda')
    (output * grad_output).sum().backward()
    (expected * grad_output).sum().backward()
    torch.testing.assert_close(
        x.grad, x_full.grad[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-5
    )


def test_column_parallel_linear_defers_embedding_wgrad_compute(model_parallel):
    model_parallel(2, 1)
    config = _gpu_config(
        defer_embedding_wgrad_compute=True,
        gradient_accumulation_fusion=True,
        wgrad_deferral_limit=0,
        te_fl_prefer='flagos',
    )
    layer = ColumnParallelLinear(
        16,
        16,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        embedding_activation_buffer=[],
        grad_output_buffer=[],
    )
    torch.manual_seed(51)
    with torch.no_grad():
        layer.weight.data.copy_(torch.randn(8, 16, device='cuda'))
    x = torch.randn(4, 2, 16, device='cuda', requires_grad=True)

    output, _ = layer(x)
    assert len(layer.embedding_activation_buffer) == 1
    assert torch.equal(layer.embedding_activation_buffer[0], x)

    output.sum().backward()
    assert len(layer.grad_output_buffer) == 1
    assert torch.equal(layer.grad_output_buffer[0], torch.ones(4, 2, 8, device='cuda'))
    assert layer.weight.grad is None


def test_column_parallel_linear_accumulates_main_grad_across_microbatches(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(gradient_accumulation_fusion=True, te_fl_prefer='flagos'),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
    )
    torch.manual_seed(52)
    full_weight = torch.randn(16, 16, device='cuda')
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    layer.weight.main_grad = torch.zeros(8, 16, device='cuda')

    expected_main_grad = torch.zeros(8, 16, device='cuda')
    for _ in range(2):
        x = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
        grad_output = torch.randn(4, 2, 8, device='cuda')
        output, _ = layer(x)
        (output * grad_output).sum().backward()
        expected_main_grad += grad_output.reshape(8, 8).t() @ x.reshape(8, 16)

    torch.testing.assert_close(layer.weight.main_grad, expected_main_grad, rtol=1e-5, atol=1e-5)
    assert layer.weight.grad is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
def test_column_parallel_linear_fused_wgrad_accumulates_main_grad_across_microbatches(
    model_parallel, dtype
):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(gradient_accumulation_fusion=True, params_dtype=dtype),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
    )
    torch.manual_seed(54)
    full_weight = torch.randn(16, 16, device='cuda', dtype=dtype)
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8])
    layer.weight.main_grad = torch.zeros(8, 16, device='cuda', dtype=dtype)

    expected_main_grad = torch.zeros(8, 16, device='cuda', dtype=torch.float32)
    for _ in range(2):
        x = torch.randn(4, 2, 16, device='cuda', dtype=dtype, requires_grad=True)
        grad_output = torch.randn(4, 2, 8, device='cuda', dtype=dtype)
        output, _ = layer(x)
        (output * grad_output).sum().backward()
        expected_main_grad += grad_output.reshape(8, 8).t().float() @ x.reshape(8, 16).float()

    torch.testing.assert_close(
        layer.weight.main_grad.float(), expected_main_grad, rtol=1.6e-2, atol=1e-2
    )
    assert layer.weight.grad is None


def test_column_parallel_linear_rejects_unsupported_main_grad_dtype(model_parallel):
    model_parallel(2, 1)
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(gradient_accumulation_fusion=True),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
    )
    layer.weight.main_grad = torch.zeros(8, 16, device='cuda', dtype=torch.int64)
    x = torch.randn(4, 2, 16, device='cuda', requires_grad=True)

    output, _ = layer(x)
    with pytest.raises(RuntimeError, match='Unsupported gradient type for gradient accumulation'):
        output.sum().backward()


def test_column_parallel_linear_backward_dw_is_noop(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = ColumnParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
    )
    torch.manual_seed(64)
    full_weight = torch.randn(16, 16, device='cuda', requires_grad=True)
    layer.weight.data.copy_(full_weight[rank * 8 : (rank + 1) * 8].detach())
    x = torch.randn(4, 2, 16, device='cuda', requires_grad=True)

    output, _ = layer(x)
    reference = torch.nn.functional.linear(x, full_weight[rank * 8 : (rank + 1) * 8])
    output.sum().backward()
    reference.sum().backward()

    expected_wgrad = full_weight.grad[rank * 8 : (rank + 1) * 8]
    torch.testing.assert_close(layer.weight.grad, expected_wgrad, rtol=1e-5, atol=1e-5)
    layer.backward_dw()
    torch.testing.assert_close(layer.weight.grad, expected_wgrad, rtol=1e-5, atol=1e-5)


def test_row_parallel_linear_backward_dw_is_noop(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    layer = RowParallelLinear(
        16,
        16,
        config=_gpu_config(),
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
    )
    torch.manual_seed(65)
    full_weight = torch.randn(16, 16, device='cuda', requires_grad=True)
    layer.weight.data.copy_(full_weight[:, rank * 8 : (rank + 1) * 8].detach())
    x_full = torch.randn(4, 2, 16, device='cuda')
    x = x_full[:, :, rank * 8 : (rank + 1) * 8].clone().requires_grad_(True)

    output, _ = layer(x)
    reference = torch.nn.functional.linear(x_full, full_weight)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)

    output.sum().backward()
    reference.sum().backward()

    expected_wgrad = full_weight.grad[:, rank * 8 : (rank + 1) * 8]
    torch.testing.assert_close(layer.weight.grad, expected_wgrad, rtol=1e-5, atol=1e-5)
    layer.backward_dw()
    torch.testing.assert_close(layer.weight.grad, expected_wgrad, rtol=1e-5, atol=1e-5)
