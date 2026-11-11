# Copyright (c) 2026, BAAI. All rights reserved.
"""TE grouped expert linear layers: TP partitioning, forward numerics and sharded state dicts."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.quantization.quant_config import GlobMatcher, RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

RTOL, ATOL = 1.6e-2, 1e-2


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _expert_config(**overrides):
    overrides.setdefault('params_dtype', torch.bfloat16)
    overrides.setdefault('bf16', True)
    overrides.setdefault('gradient_accumulation_fusion', False)
    overrides.setdefault('tensor_model_parallel_size', 2)
    return TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)


def _quantized_expert_config(**overrides):
    config = _expert_config(**overrides)
    config.quant_recipe = RecipeConfig(
        matchers=[GlobMatcher(pattern='experts.*', config_key='default')],
        config_dict={
            'default': {
                'transformer_engine_config_type': 'TEQuantizationParams',
                'training_recipe': {
                    'fp8_quantization_recipe': 'tensorwise',
                    'fp8_format': 'e4m3',
                    'override_nonquantized_autocast': True,
                },
            }
        },
    )
    return config


def _det(rows, cols):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((values % 7) - 3).to(torch.bfloat16)


def _det_input(rows, hidden, offset=0):
    values = torch.arange(rows * hidden, dtype=torch.float32)
    return (((values % 11) - 5 + offset) / 4.0).to(torch.bfloat16).reshape(rows, hidden)


def _init_tp2(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    return parallel_state.get_expert_tensor_parallel_group()


def _make_column_grouped(model_parallel, config=None, num_gemms=2, name=None):
    _init_tp2(model_parallel)
    return TEColumnParallelGroupedLinear(
        num_gemms=num_gemms,
        input_size=16,
        output_size=8,
        config=config or _expert_config(),
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=False,
        is_expert=True,
        tp_comm_buffer_name='fc1',
        name=name,
    )


def _make_row_grouped(model_parallel, config=None, num_gemms=2, name=None):
    _init_tp2(model_parallel)
    return TERowParallelGroupedLinear(
        num_gemms=num_gemms,
        input_size=16,
        output_size=8,
        config=config or _expert_config(),
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=True,
        is_expert=True,
        tp_comm_buffer_name='fc2',
        name=name,
    )


def test_te_column_grouped_linear_partitions_weights_across_expert_tp(model_parallel):
    layer = _make_column_grouped(model_parallel)
    assert layer.weight0.shape == (4, 16)
    assert layer.weight1.shape == (4, 16)
    assert layer.explicit_expert_comm is True
    assert layer.tp_size == 1
    assert layer.weight0.partition_dim == 0
    assert layer.weight0.partition_stride == 1
    assert layer.weight0.allreduce is True


def test_te_row_grouped_linear_partitions_weights_across_expert_tp(model_parallel):
    layer = _make_row_grouped(model_parallel)
    assert layer.weight0.shape == (8, 8)
    assert layer.weight1.shape == (8, 8)
    assert layer.explicit_expert_comm is True
    assert layer.weight0.partition_dim == 1
    assert layer.weight0.partition_stride == 1


def test_te_column_grouped_linear_forward_matches_local_shard_reference(model_parallel):
    layer = _make_column_grouped(model_parallel)
    layer.weight0.data.copy_(_det(4, 16).cuda())
    layer.weight1.data.copy_(_det(4, 16).cuda())

    x = _det_input(6, 16).cuda()
    out, bias = layer(x, [3, 3])
    assert bias is None
    expected = torch.cat((x[:3] @ layer.weight0.t(), x[3:] @ layer.weight1.t()), dim=0)
    assert out.shape == (6, 4)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_te_row_grouped_linear_forward_matches_local_shard_reference(model_parallel):
    layer = _make_row_grouped(model_parallel)
    layer.weight0.data.copy_(_det(8, 8).cuda())
    layer.weight1.data.copy_(_det(8, 8).cuda())

    x = _det_input(6, 8).cuda()
    out, bias = layer(x, [3, 3])
    assert bias is None
    expected = torch.cat((x[:3] @ layer.weight0.t(), x[3:] @ layer.weight1.t()), dim=0)
    assert out.shape == (6, 8)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_te_grouped_linear_will_execute_quantized_follows_quant_recipe(model_parallel):
    layer = _make_column_grouped(
        model_parallel, config=_quantized_expert_config(), name='experts.linear_fc1'
    )
    assert layer.will_execute_quantized(False) is True


def test_te_grouped_linear_backward_dw_is_a_noop_without_delayed_wgrad(model_parallel):
    layer = _make_column_grouped(model_parallel)
    weight_before = layer.weight0.detach().clone()
    assert layer.delay_wgrad_compute is False
    layer.backward_dw()
    assert torch.equal(layer.weight0.detach(), weight_before)


def test_te_grouped_linear_load_state_dict_round_trip(model_parallel):
    layer = _make_column_grouped(model_parallel)
    layer.weight0.data.copy_(_det(4, 16).cuda())
    layer.weight1.data.copy_(_det(4, 16).cuda())
    saved = {key: value.clone() for key, value in layer.state_dict().items()}

    layer.weight0.data.zero_()
    layer.weight1.data.zero_()
    layer.load_state_dict(saved)
    assert torch.equal(layer.weight0.detach(), saved['weight0'])
    assert torch.equal(layer.weight1.detach(), saved['weight1'])


def test_te_column_grouped_linear_sharded_state_dict(model_parallel):
    layer = _make_column_grouped(model_parallel)
    state_dict = layer.sharded_state_dict(
        prefix='experts.',
        metadata={
            'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)
        },
    )
    assert set(state_dict) == {
        'experts.weight0',
        'experts.weight1',
        'experts._extra_state',
        'experts._extra_state1',
    }
    for key in ('experts.weight0', 'experts.weight1'):
        sharded = state_dict[key]
        assert isinstance(sharded, ShardedTensor)
        assert sharded.local_shape == (4, 16)
    for key in ('experts._extra_state', 'experts._extra_state1'):
        assert isinstance(state_dict[key], (ShardedTensor, ShardedObject))


def test_te_row_grouped_linear_sharded_state_dict(model_parallel):
    layer = _make_row_grouped(model_parallel)
    state_dict = layer.sharded_state_dict(
        prefix='experts.',
        metadata={
            'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)
        },
    )
    assert set(state_dict) == {
        'experts.weight0',
        'experts.weight1',
        'experts._extra_state',
        'experts._extra_state1',
    }
    for key in ('experts.weight0', 'experts.weight1'):
        sharded = state_dict[key]
        assert isinstance(sharded, ShardedTensor)
        assert sharded.local_shape == (8, 8)
