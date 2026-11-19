# Copyright (c) 2026, BAAI. All rights reserved.
"""TE linear wrappers in BF16 on tensor-parallel ranks: forward, guards, sharding and tp-comm."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    TERowParallelLinear,
    set_save_original_input,
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
    # TransformerConfig only accepts delay_wgrad_compute together with MoE expert-parallel overlap;
    # the TE layers read the flag from the built config, so set it after validation.
    delay_wgrad_compute = overrides.pop('delay_wgrad_compute', False)
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)
    config.delay_wgrad_compute = delay_wgrad_compute
    return config


def _det(rows, cols):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((values % 7) - 3).to(torch.bfloat16)


def _det_input(seq, batch, hidden, offset=0):
    values = torch.arange(seq * batch * hidden, dtype=torch.float32)
    return (((values % 11) - 5 + offset) / 4.0).to(torch.bfloat16).reshape(seq, batch, hidden)


def _fill_master(master):
    values = torch.arange(master.numel(), dtype=torch.float32).reshape(master.shape)
    master.copy_((values % 7) - 3)


def _init_tp2(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    return parallel_state.get_tensor_model_parallel_group()


def _gather(tensor, group, dim):
    world = torch.distributed.get_world_size(group=group)
    parts = [torch.empty_like(tensor) for _ in range(world)]
    torch.distributed.all_gather(parts, tensor, group=group)
    return torch.cat(parts, dim=dim)


def _dp_cp_group():
    return parallel_state.get_data_parallel_group(with_context_parallel=True)


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


def _make_te_linear(
    tp_group=None,
    is_expert=False,
    name=None,
    config=None,
    tp_comm_buffer_name=None,
    symmetric_ar_type=None,
    **config_overrides,
):
    return TELinear(
        16,
        8,
        parallel_mode='column' if is_expert else 'duplicated',
        config=config or _make_config(**config_overrides),
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_comm_buffer_name=tp_comm_buffer_name,
        tp_group=tp_group,
        is_expert=is_expert,
        symmetric_ar_type=symmetric_ar_type,
        name=name,
    )


def test_te_linear_duplicated_forward_matches_torch_reference(model_parallel):
    _init_tp2(model_parallel)
    layer = _make_te_linear()
    assert layer.is_first_microbatch is True
    assert layer.weight.allreduce is True
    assert layer.weight.tensor_model_parallel is False
    layer.weight.data.copy_(_det(8, 16).cuda())

    x = _det_input(6, 2, 16).cuda()
    out, bias = layer(x)
    assert bias is None
    assert layer.is_first_microbatch is False
    expected = torch.nn.functional.linear(x, layer.weight)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_te_linear_rejects_skip_weight_param_allocation(model_parallel):
    _init_tp2(model_parallel)
    with pytest.raises(ValueError, match='do not support skip_weight_param_allocation'):
        TELinear(
            16,
            8,
            parallel_mode='duplicated',
            config=_make_config(),
            init_method=lambda w: None,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=True,
        )


def test_te_linear_disables_tp_overlap_for_unsupported_buffer_name(model_parallel):
    _init_tp2(model_parallel)
    config = _make_config(tp_comm_overlap=True)
    with pytest.warns(UserWarning, match='not supported in'):
        _make_te_linear(config=config, name='layers.0.fc2', tp_comm_buffer_name='bad_buffer')
    assert config.tp_comm_overlap is False


def test_te_linear_expert_divides_output_size_across_expert_tp(model_parallel):
    tp_group = _init_tp2(model_parallel)
    layer = _make_te_linear(tp_group=tp_group, is_expert=True, name='experts.fc1')
    assert layer.weight.shape == (4, 16)
    assert layer.weight.allreduce is True
    layer.weight.data.copy_(_det(4, 16).cuda())

    x = _det_input(6, 2, 16).cuda()
    out, bias = layer(x)
    assert bias is None
    expected = torch.nn.functional.linear(x, layer.weight)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_te_linear_delayed_wgrad_computes_weight_grad_on_backward_dw(model_parallel):
    _init_tp2(model_parallel)
    layer = _make_te_linear(delay_wgrad_compute=True)
    layer.weight.data.copy_(_det(8, 16).cuda())

    x = _det_input(6, 2, 16).cuda()
    out, _ = layer(x)
    out.float().sum().backward()
    layer.backward_dw()

    flat_x = x.reshape(-1, 16).float()
    grad_output = torch.ones(12, 8, dtype=torch.float32, device='cuda')
    expected_grad = grad_output.t() @ flat_x
    torch.testing.assert_close(layer.weight.grad.float(), expected_grad, rtol=RTOL, atol=ATOL)


def test_te_linear_backward_dw_leaves_grad_unchanged_when_delay_disabled(model_parallel):
    _init_tp2(model_parallel)
    layer = _make_te_linear()
    layer.weight.data.copy_(_det(8, 16).cuda())

    x = _det_input(6, 2, 16).cuda()
    out, _ = layer(x)
    out.float().sum().backward()
    grad_before = layer.weight.grad.detach().clone()
    layer.backward_dw()
    assert torch.equal(layer.weight.grad.detach(), grad_before)


def test_te_linear_sharded_state_dict_covers_duplicated_weights(model_parallel):
    _init_tp2(model_parallel)
    layer = _make_te_linear()
    state_dict = layer.sharded_state_dict(
        prefix='module.', metadata={'dp_cp_group': _dp_cp_group()}
    )
    assert set(state_dict) == {'module.weight', 'module._extra_state'}
    sharded = state_dict['module.weight']
    assert isinstance(sharded, ShardedTensor)
    assert sharded.key == 'module.weight'
    assert torch.equal(sharded.data, layer.weight)
    extra_state = state_dict['module._extra_state']
    assert isinstance(extra_state, ShardedObject)
    assert extra_state.key == 'module._extra_state'


def test_te_linear_passes_tp_overlap_flags_for_supported_buffer_name(model_parallel, monkeypatch):
    tp_group = _init_tp2(model_parallel)
    real_te_init = te_ext.te.pytorch.Linear.__init__
    te_init_kwargs = {}

    def _recording_init(self, *args, **kwargs):
        te_init_kwargs.update(kwargs)
        real_te_init(self, *args, **kwargs)

    monkeypatch.setattr(te_ext.te.pytorch.Linear, '__init__', _recording_init)
    layer = _make_te_linear(
        tp_group=tp_group,
        is_expert=True,
        name='layers.0.fc2',
        config=_make_config(tp_comm_overlap=True),
        tp_comm_buffer_name='fc2',
    )
    assert layer.weight.shape == (4, 16)
    assert layer.tp_size == 1
    assert layer.expert_parallel is False
    assert te_init_kwargs['ub_overlap_ag'] is False
    assert te_init_kwargs['ub_overlap_rs'] is False
    assert te_init_kwargs['ub_name'] == 'fc2'


def test_te_linear_accepts_symmetric_all_reduce_type(model_parallel, monkeypatch):
    _init_tp2(model_parallel)
    real_te_init = te_ext.te.pytorch.Linear.__init__
    te_init_kwargs = {}

    def _recording_init(self, *args, **kwargs):
        te_init_kwargs.update(kwargs)
        real_te_init(self, *args, **kwargs)

    monkeypatch.setattr(te_ext.te.pytorch.Linear, '__init__', _recording_init)
    layer = _make_te_linear(symmetric_ar_type='two_shot')
    assert layer.symmetric_ar_type == 'two_shot'
    assert layer.weight.shape == (8, 16)
    assert te_init_kwargs['symmetric_ar_type'] == 'two_shot'


def test_te_column_parallel_linear_forward_matches_gathered_reference(model_parallel):
    tp_group = _init_tp2(model_parallel)
    layer = TEColumnParallelLinear(
        16,
        8,
        config=_make_config(),
        init_method=lambda w: None,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    layer.weight.data.copy_(_det(4, 16).cuda())

    x = _det_input(6, 2, 16).cuda()
    out, bias = layer(x)
    assert bias is None
    assert out.shape == (6, 2, 4)
    assert f'in_features={layer.in_features}' in repr(layer)
    full_weight = _gather(layer.weight, tp_group, dim=0)
    expected = torch.nn.functional.linear(x, full_weight)
    gathered_out = _gather(out, tp_group, dim=-1)
    torch.testing.assert_close(gathered_out, expected, rtol=RTOL, atol=ATOL)


def test_te_column_parallel_linear_rejects_gather_output(model_parallel):
    _init_tp2(model_parallel)
    with pytest.raises(ValueError, match='do not support gather_output'):
        TEColumnParallelLinear(
            16,
            8,
            config=_make_config(),
            init_method=lambda w: None,
            gather_output=True,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )


def test_te_column_parallel_linear_cpu_init_partitions_master_weight(model_parallel):
    tp_group = _init_tp2(model_parallel)
    tp_rank = torch.distributed.get_rank(group=tp_group)
    layer = TEColumnParallelLinear(
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
    assert layer.bias.shape == (4,)
    assert torch.equal(layer.bias.detach(), torch.zeros(4, dtype=torch.bfloat16))
    assert layer.bias.allreduce is True
    assert layer.weight.partition_stride == 1


def test_te_row_parallel_linear_forward_matches_gathered_reference(model_parallel):
    tp_group = _init_tp2(model_parallel)
    tp_rank = torch.distributed.get_rank(group=tp_group)
    layer = TERowParallelLinear(
        16,
        8,
        config=_make_config(),
        init_method=lambda w: None,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    layer.weight.data.copy_(_det(8, 8).cuda())

    x_local = _det_input(6, 2, 8, offset=tp_rank).cuda()
    out, bias = layer(x_local)
    assert bias is None
    assert out.shape == (6, 2, 8)
    full_weight = _gather(layer.weight, tp_group, dim=1)
    full_x = _gather(x_local, tp_group, dim=-1)
    expected = torch.nn.functional.linear(full_x, full_weight)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    assert f'in_features={layer.in_features}' in repr(layer)


def test_te_row_parallel_linear_rejects_input_not_parallel(model_parallel):
    _init_tp2(model_parallel)
    with pytest.raises(ValueError, match='do not support input_is_parallel = False'):
        TERowParallelLinear(
            16,
            8,
            config=_make_config(),
            init_method=lambda w: None,
            bias=False,
            input_is_parallel=False,
            skip_bias_add=False,
            is_expert=False,
        )


def test_te_row_parallel_linear_cpu_init_partitions_master_weight(model_parallel):
    tp_group = _init_tp2(model_parallel)
    tp_rank = torch.distributed.get_rank(group=tp_group)
    layer = TERowParallelLinear(
        16,
        8,
        config=_make_config(use_cpu_initialization=True, add_bias_linear=True),
        init_method=_fill_master,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    expected_shard = _det(8, 16)[:, tp_rank * 8 : (tp_rank + 1) * 8]
    assert torch.equal(layer.weight.detach().cpu(), expected_shard)
    assert layer.master_weight is None
    assert layer.bias.shape == (8,)
    assert torch.equal(layer.bias.detach(), torch.zeros(8, dtype=torch.bfloat16))
    assert layer.bias.sequence_parallel is False


def test_te_linear_will_execute_quantized_follows_quant_recipe(model_parallel):
    _init_tp2(model_parallel)
    quantized = _make_te_linear(config=_quantized_config(), name='layers.0.fc1')
    assert quantized.will_execute_quantized(False) is True
    plain = _make_te_linear()
    assert plain.will_execute_quantized(False) is False
    assert plain.will_execute_quantized(True) is True


def test_set_save_original_input_flags_te_linear(model_parallel):
    _init_tp2(model_parallel)
    layer = _make_te_linear()
    set_save_original_input(layer)
    assert layer.save_original_input is True


def test_set_save_original_input_rejects_modules_without_support():
    with pytest.raises(
        ValueError, match='set_save_original_input is only needed on transformer-engine modules'
    ):
        set_save_original_input(torch.nn.Linear(4, 4))
