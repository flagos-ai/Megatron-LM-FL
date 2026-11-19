# Copyright (c) 2026, BAAI. All rights reserved.
"""ModelOpt local Norm and Linear layer construction plus RealQuantTransformerLayer introspection."""

import logging

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.post_training.modelopt import layers as modelopt_layers
from megatron.core.post_training.modelopt.layers import Linear, Norm, RealQuantTransformerLayer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def _layer_norm_builder(config, hidden_size, eps):
    return torch.nn.LayerNorm(hidden_size, eps=eps)


def _make_real_quant_layer():
    config = TransformerConfig(
        num_layers=2, hidden_size=16, num_attention_heads=2, perform_initialization=False
    )
    submodules = TransformerLayerSubmodules(input_layernorm=_layer_norm_builder)
    return RealQuantTransformerLayer(
        config, submodules, pg_collection=ProcessGroupCollection(tp=None, pp=None)
    )


def test_norm_raises_import_error_when_transformer_engine_is_missing(monkeypatch):
    monkeypatch.setattr(modelopt_layers, 'HAVE_TE', False)
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)

    with pytest.raises(ImportError, match='Transformer-Engine is not installed'):
        Norm(config, 8)


def test_norm_rejects_normalization_kinds_other_than_layernorm_and_rmsnorm(monkeypatch):
    monkeypatch.setattr(modelopt_layers, 'HAVE_TE', True)
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, normalization='sandwich'
    )

    with pytest.raises(Exception, match='Only LayerNorm and RMSNorm are curently supported'):
        Norm(config, 8)


@pytest.mark.parametrize(
    'sequence_parallel', [False, True], ids=['sequence_parallel_off', 'sequence_parallel_on']
)
def test_linear_marks_dense_parameters_for_data_parallel_reduction(sequence_parallel):
    config = ModelParallelConfig(sequence_parallel=sequence_parallel, tensor_model_parallel_size=2)

    linear = Linear(4, 8, config=config, init_method=torch.nn.init.normal_)

    assert linear.weight.allreduce is True
    assert linear.bias.allreduce is True
    assert linear.weight.sequence_parallel is sequence_parallel
    assert linear.bias.sequence_parallel is sequence_parallel


@pytest.mark.parametrize(
    ('expert_model_parallel_size', 'allreduce'),
    [(1, True), (2, False)],
    ids=['single_ep_rank', 'multiple_ep_ranks'],
)
def test_linear_marks_expert_parameters_by_expert_parallel_size(
    expert_model_parallel_size, allreduce
):
    config = ModelParallelConfig(expert_model_parallel_size=expert_model_parallel_size)

    linear = Linear(4, 8, config=config, init_method=torch.nn.init.normal_, is_expert=True)

    assert linear.weight.allreduce is allreduce


@pytest.mark.parametrize(
    'params_dtype', [torch.float16, torch.bfloat16], ids=['float16', 'bfloat16']
)
def test_linear_creates_weights_and_bias_in_the_configured_params_dtype(params_dtype):
    config = ModelParallelConfig(params_dtype=params_dtype)

    linear = Linear(4, 8, config=config, init_method=torch.nn.init.normal_)

    assert linear.weight.dtype is params_dtype
    assert linear.bias.dtype is params_dtype
    assert linear.weight.shape == torch.Size([8, 4])
    assert linear.bias.shape == torch.Size([8])


@pytest.mark.parametrize(
    ('unsupported_kwargs', 'message'),
    [
        ({'stride': 2}, 'torch.nn.Linear does not support stride != 1'),
        (
            {'skip_weight_param_allocation': True},
            'torch.nn.Linear layers do not support skip_weight_param_allocation',
        ),
        (
            {'embedding_activation_buffer': [torch.zeros(1)]},
            'torch.nn.Linear does not support embedding_activation_buffer != None',
        ),
        (
            {'grad_output_buffer': [torch.zeros(1)]},
            'torch.nn.Linear does not support grad_output_buffer != None',
        ),
    ],
    ids=['stride', 'skip_weight_alloc', 'embedding_buffer', 'grad_output_buffer'],
)
def test_linear_rejects_arguments_torch_linear_cannot_honor(unsupported_kwargs, message):
    config = ModelParallelConfig()

    with pytest.raises(ValueError, match=message):
        Linear(4, 8, config=config, init_method=torch.nn.init.normal_, **unsupported_kwargs)


def test_linear_forward_returns_output_and_none_bias_by_default():
    config = ModelParallelConfig()
    linear = Linear(4, 8, config=config, init_method=torch.nn.init.normal_, bias=True)
    x = torch.ones(2, 4)

    output, bias = linear(x)

    assert torch.equal(output, torch.nn.functional.linear(x, linear.weight, linear.bias))
    assert bias is None


def test_linear_forward_returns_only_the_output_when_bias_add_is_skipped():
    config = ModelParallelConfig()
    linear = Linear(
        4, 8, config=config, init_method=torch.nn.init.normal_, bias=True, skip_bias_add=True
    )
    x = torch.ones(2, 4)

    output = linear(x)

    assert not isinstance(output, tuple)
    assert torch.equal(output, torch.nn.functional.linear(x, linear.weight, linear.bias))


def test_sharded_state_dict_views_zero_dim_quant_scales_as_one_element(monkeypatch):
    monkeypatch.setattr(
        parallel_state, 'get_tensor_model_parallel_group', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(parallel_state, 'get_data_parallel_group', lambda *args, **kwargs: None)
    config = ModelParallelConfig()
    linear = Linear(4, 8, config=config, init_method=torch.nn.init.normal_)
    linear.register_buffer('linear_weight_scale', torch.tensor(0.5))

    sharded = linear.sharded_state_dict(prefix='module.', metadata={'dp_cp_group': None})

    assert set(sharded) == {'module.weight', 'module.bias', 'module.linear_weight_scale'}
    assert sharded['module.linear_weight_scale'].data.shape == torch.Size([1])
    assert sharded['module.weight'].data.shape == torch.Size([8, 4])


def test_real_quant_layer_constructs_like_a_plain_transformer_layer():
    layer = _make_real_quant_layer()

    assert isinstance(layer, TransformerLayer)
    assert RealQuantTransformerLayer.real_quant_cfg == 'None'
    assert layer.layer_number == 1
    assert layer.input_layernorm.weight.requires_grad


def test_collect_original_tensor_info_records_dtypes_and_shapes():
    layer = _make_real_quant_layer()

    layer._collect_original_tensor_info()

    assert layer._original_tensor_info == {
        'input_layernorm.weight': ('torch.float32', 'torch.Size([16])'),
        'input_layernorm.bias': ('torch.float32', 'torch.Size([16])'),
    }


def test_report_quantize_tensor_info_logs_tensor_summary_on_rank_zero(monkeypatch, caplog):
    layer = _make_real_quant_layer()
    layer._collect_original_tensor_info()
    monkeypatch.setattr(torch.distributed, 'barrier', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda *args, **kwargs: 0)

    caplog.set_level(logging.INFO, logger='megatron.core.post_training.modelopt.layers')
    layer._report_quantize_tensor_info()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        'input_layernorm.weight' in message
        and 'torch.float32' in message
        and 'torch.Size([16])' in message
        for message in messages
    )
