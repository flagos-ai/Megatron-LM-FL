# Copyright (c) 2026, BAAI. All rights reserved.
"""TE layer-norm wrappers: fused LayerNormLinear, residual RMSNorm fusion and activation ops."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEFusedResidualRMSNorm,
    TELayerNormColumnParallelLinear,
    TENorm,
    split_te_layernorm_column_parallel_linear,
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


def _make_config(**overrides):
    overrides.setdefault('params_dtype', torch.bfloat16)
    overrides.setdefault('bf16', True)
    overrides.setdefault('gradient_accumulation_fusion', False)
    overrides.setdefault('tensor_model_parallel_size', 2)
    return TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)


def _det(rows, cols):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((values % 7) - 3).to(torch.bfloat16)


def _det_input(seq, batch, hidden):
    values = torch.arange(seq * batch * hidden, dtype=torch.float32)
    return (((values % 11) - 5) / 4.0).to(torch.bfloat16).reshape(seq, batch, hidden)


def _fill_master(master):
    values = torch.arange(master.numel(), dtype=torch.float32).reshape(master.shape)
    master.copy_((values % 7) - 3)


def _init_tp2(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    return parallel_state.get_tensor_model_parallel_group()


def _quantized_config(**overrides):
    config = _make_config(**overrides)
    config.quant_recipe = RecipeConfig(
        matchers=[GlobMatcher(pattern='layers.*', config_key='default')],
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


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'gather_output': True}, 'do not support gather_output'),
        ({'is_expert': True}, 'do not yet support MoE'),
        ({'skip_weight_param_allocation': True}, 'do not support skip_weight_param_allocation'),
    ],
    ids=['gather-output', 'expert', 'skip-weight'],
)
def test_te_layernorm_column_parallel_linear_rejects_unsupported_arguments(
    model_parallel, kwargs, message
):
    _init_tp2(model_parallel)
    with pytest.raises(ValueError, match=message):
        TELayerNormColumnParallelLinear(
            16,
            8,
            config=_make_config(),
            init_method=lambda w: None,
            **{
                'gather_output': False,
                'bias': False,
                'skip_bias_add': False,
                'is_expert': False,
                **kwargs,
            },
        )


def _make_fused_layernorm_linear(tp_group, normalization='LayerNorm'):
    layer = TELayerNormColumnParallelLinear(
        16,
        8,
        config=_make_config(normalization=normalization, add_bias_linear=True),
        init_method=lambda w: None,
        gather_output=False,
        bias=True,
        skip_bias_add=True,
        is_expert=False,
        tp_group=tp_group,
    )
    layer.weight.data.copy_(_det(4, 16).cuda())
    layer.bias.data.copy_(_det(1, 4).reshape(4).cuda())
    layer.layer_norm_weight.data.fill_(1.0)
    if layer.layer_norm_bias is not None:
        layer.layer_norm_bias.data.zero_()
    return layer


def test_te_layernorm_column_parallel_linear_forward_matches_reference(model_parallel):
    tp_group = _init_tp2(model_parallel)
    layer = _make_fused_layernorm_linear(tp_group)

    x = _det_input(6, 2, 16).cuda()
    out, out_bias = layer(x)
    assert out.shape == (6, 2, 4)
    normalized = torch.nn.functional.layer_norm(
        x.float(), (16,), layer.layer_norm_weight.float(), layer.layer_norm_bias.float()
    )
    expected = normalized.to(torch.bfloat16) @ layer.weight.t()
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out_bias, layer.bias, rtol=RTOL, atol=ATOL)
    assert f'TP={layer.tp_size}' in repr(layer)


def test_te_layernorm_column_parallel_linear_rmsnorm_forward_matches_reference(model_parallel):
    tp_group = _init_tp2(model_parallel)
    layer = _make_fused_layernorm_linear(tp_group, normalization='RMSNorm')

    x = _det_input(6, 2, 16).cuda()
    out, _ = layer(x)
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    normalized = (x_fp32 * inv_rms).to(torch.bfloat16)
    expected = normalized @ layer.weight.t()
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_te_layernorm_column_parallel_linear_cpu_init_zeroes_bias(model_parallel):
    tp_group = _init_tp2(model_parallel)
    tp_rank = torch.distributed.get_rank(group=tp_group)
    layer = TELayerNormColumnParallelLinear(
        16,
        8,
        config=_make_config(use_cpu_initialization=True, add_bias_linear=True),
        init_method=_fill_master,
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    expected_shard = _det(8, 16)[tp_rank * 4 : (tp_rank + 1) * 4]
    assert torch.equal(layer.weight.detach().cpu(), expected_shard)
    assert torch.equal(layer.bias.detach(), torch.zeros(4, dtype=torch.bfloat16))


def test_te_layernorm_column_parallel_linear_sharded_state_dict(model_parallel):
    tp_group = _init_tp2(model_parallel)
    layer = _make_fused_layernorm_linear(tp_group)
    state_dict = layer.sharded_state_dict(
        prefix='module.',
        metadata={
            'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)
        },
    )
    assert set(state_dict) == {
        'module.weight',
        'module.bias',
        'module.layer_norm_weight',
        'module.layer_norm_bias',
        'module._extra_state',
    }
    for name in ('weight', 'bias', 'layer_norm_weight', 'layer_norm_bias'):
        sharded = state_dict[f'module.{name}']
        assert isinstance(sharded, ShardedTensor)
        assert sharded.key == f'module.{name}'
        assert torch.equal(sharded.data, getattr(layer, name))
    extra_state = state_dict['module._extra_state']
    assert isinstance(extra_state, ShardedObject)
    assert extra_state.key == 'module._extra_state'


def test_te_layernorm_column_parallel_linear_will_execute_quantized_follows_quant_recipe(
    model_parallel,
):
    tp_group = _init_tp2(model_parallel)
    layer = TELayerNormColumnParallelLinear(
        16,
        8,
        config=_quantized_config(),
        init_method=lambda w: None,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
        name='layers.0.fc1',
    )
    assert layer.will_execute_quantized(False) is True


def test_split_te_layernorm_column_parallel_linear_matches_fused_output(model_parallel):
    tp_group = _init_tp2(model_parallel)
    fused = _make_fused_layernorm_linear(tp_group)
    norm_layer, linear_layer = split_te_layernorm_column_parallel_linear(
        fused, _make_config(), tp_group=tp_group
    )
    assert linear_layer.sequence_parallel == fused.sequence_parallel
    assert linear_layer.is_first_microbatch == fused.is_first_microbatch
    assert torch.equal(norm_layer.weight.detach(), fused.layer_norm_weight.detach())
    assert torch.equal(norm_layer.bias.detach(), fused.layer_norm_bias.detach())
    assert torch.equal(linear_layer.weight.detach(), fused.weight.detach())
    assert torch.equal(linear_layer.bias.detach(), fused.bias.detach())

    x = _det_input(6, 2, 16).cuda()
    fused_out, fused_bias = fused(x)
    split_out, split_bias = linear_layer(norm_layer(x))
    torch.testing.assert_close(split_out, fused_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(split_bias, fused_bias, rtol=RTOL, atol=ATOL)


def test_split_te_layernorm_column_parallel_linear_copies_fp8_meta_entries(model_parallel):
    tp_group = _init_tp2(model_parallel)
    fused = _make_fused_layernorm_linear(tp_group)
    fused.fp8_meta['fp8_group'] = None
    norm_layer, linear_layer = split_te_layernorm_column_parallel_linear(
        fused, _make_config(), tp_group=tp_group
    )
    assert linear_layer.fp8_meta['fp8_checkpoint'] == fused.fp8_meta['fp8_checkpoint']
    assert linear_layer.fp8_meta['fp8_group'] is None
    assert linear_layer.disable_parameter_transpose_cache == (
        fused.disable_parameter_transpose_cache
    )


def test_te_norm_builds_fused_residual_rmsnorm_when_requested(model_parallel):
    _init_tp2(model_parallel)
    config = _make_config(normalization='RMSNorm', fused_residual_rmsnorm=True)
    norm = TENorm(config=config, hidden_size=16, eps=1e-5, has_residual=True)
    assert isinstance(norm, TEFusedResidualRMSNorm)
    assert norm.weight.shape == (16,)


def test_te_norm_rejects_fused_residual_with_layernorm(model_parallel):
    _init_tp2(model_parallel)
    config = _make_config(normalization='LayerNorm')
    config.fused_residual_rmsnorm = True
    with pytest.raises(ValueError, match='Fused residual is only supported for RMSNorm'):
        TENorm(config=config, hidden_size=16, eps=1e-5, has_residual=True)


def test_te_norm_rejects_unsupported_normalization(model_parallel):
    _init_tp2(model_parallel)
    config = _make_config()
    config.normalization = 'bogus'
    with pytest.raises(Exception, match='Only LayerNorm and RMSNorm are currently supported'):
        TENorm(config=config, hidden_size=16, eps=1e-5)


def _make_fused_residual_norm(model_parallel):
    _init_tp2(model_parallel)
    config = _make_config(normalization='RMSNorm', fused_residual_rmsnorm=True)
    norm = TENorm(config=config, hidden_size=16, eps=1e-5, has_residual=True)
    norm.weight.data.fill_(1.0)
    return norm


def test_fused_residual_rmsnorm_forward_matches_reference_and_runs_post_hook(model_parallel):
    norm = _make_fused_residual_norm(model_parallel)
    hook_calls = []
    norm.register_forward_hook(
        lambda module, args, output: hook_calls.append((module, args, output))
    )

    x = _det_input(6, 2, 16).cuda()
    with pytest.warns(UserWarning, match='TEFusedResidualRMSNorm module has a submodule'):
        out, residual = norm(x)
    # The fused TE op re-runs the module's hooks without exposing tensors (module, (), None); torch
    # then runs the same hook normally on the module's own call with the real input and output.
    assert len(hook_calls) == 2
    assert hook_calls[0] == (norm, (), None)
    assert hook_calls[1][0] is norm and hook_calls[1][1][0] is x
    assert hook_calls[1][2][0] is out

    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = (x_fp32 * inv_rms).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(residual, x, rtol=RTOL, atol=ATOL)


def test_fused_residual_rmsnorm_runs_pre_hook_with_empty_args(model_parallel):
    norm = _make_fused_residual_norm(model_parallel)
    hook_calls = []
    norm.register_forward_pre_hook(lambda module, args: hook_calls.append((module, args)))

    x = _det_input(6, 2, 16).cuda()
    with pytest.warns(UserWarning, match='TEFusedResidualRMSNorm module has a submodule'):
        out, residual = norm(x)
    # torch runs the pre-hook on the module's own call first; the fused TE op then re-runs it
    # without exposing the input tensor.
    assert len(hook_calls) == 2
    assert hook_calls[0][0] is norm and hook_calls[0][1][0] is x
    assert hook_calls[1] == (norm, ())

    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = (x_fp32 * inv_rms).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(residual, x, rtol=RTOL, atol=ATOL)


def test_fused_residual_rmsnorm_raises_when_pre_hook_modifies_input(model_parallel):
    norm = _make_fused_residual_norm(model_parallel)
    norm.register_forward_pre_hook(lambda module, args: torch.zeros(1))

    x = _det_input(6, 2, 16).cuda()
    with pytest.raises(RuntimeError, match='pre-forward hook that modifies input tensor'):
        norm(x)


def test_fused_residual_rmsnorm_raises_when_post_hook_modifies_output(model_parallel):
    norm = _make_fused_residual_norm(model_parallel)
    norm.register_forward_hook(lambda module, args, output: torch.zeros(1))

    x = _det_input(6, 2, 16).cuda()
    with pytest.raises(RuntimeError, match='post-forward hook that modifies output tensor'):
        norm(x)


def test_fused_residual_rmsnorm_raises_when_backward_hooks_present(model_parallel):
    norm = _make_fused_residual_norm(model_parallel)
    norm.register_full_backward_pre_hook(lambda module, grad_output: None)

    x = _det_input(6, 2, 16).cuda()
    with pytest.raises(RuntimeError, match='does not support submodules with pre-backward hooks'):
        norm(x)


@pytest.mark.parametrize(
    ('activation', 'glu', 'expected_op'),
    [
        (torch.nn.functional.silu, True, 'SwiGLU'),
        (torch.nn.functional.gelu, True, 'GEGLU'),
        (torch.nn.functional.gelu, False, 'GELU'),
    ],
    ids=['swiglu', 'geglu', 'gelu'],
)
def test_te_activation_op_selects_te_operation(activation, glu, expected_op):
    config = _make_config(activation_func=activation, gated_linear_unit=glu)
    layer = TEActivationOp(config)
    assert type(layer).__name__ == expected_op


def test_te_activation_op_caches_quantized_input_when_requested():
    config = _make_config(
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        activation_func_fp8_input_store=True,
    )
    layer = TEActivationOp(config)
    assert type(layer).__name__ == 'SwiGLU'
    assert layer.cache_quantized_input is True


def test_te_activation_op_rejects_unsupported_activation():
    config = _make_config(activation_func=torch.tanh, gated_linear_unit=False)
    with pytest.raises(Exception, match='Only SwiGLU, GEGLU, ReGLU, GELU, ReLU are supported'):
        TEActivationOp(config)
