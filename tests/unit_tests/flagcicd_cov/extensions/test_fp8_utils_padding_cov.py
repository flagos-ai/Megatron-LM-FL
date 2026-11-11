# Copyright (c) 2026, BAAI. All rights reserved.
"""FP8 inference padding wrappers around TE linear layers and layer-type classification."""

import contextlib

import pytest
import torch

from megatron.core import fp8_utils, parallel_state
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.quantization.quant_config import GlobMatcher, RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

RTOL, ATOL = 1.6e-2, 1e-2


class _FakeFp8Padding:
    """Stand-in for TE's FP8 padding kernel; pads the sequence to a multiple of 16."""

    def __init__(self, alignment):
        self.alignment = alignment
        self.seen_shape = None

    def __call__(self, inp, sizes):
        seq_len = sizes[0]
        target_len = ((seq_len + 15) // 16) * 16
        padded = torch.zeros(target_len, inp.shape[1], dtype=inp.dtype, device=inp.device)
        padded[:seq_len].copy_(inp)
        self.seen_shape = tuple(inp.shape)
        return padded, [target_len]


class _FakeFp8Unpadding:
    """Stand-in for TE's FP8 unpadding kernel; drops rows past the original length."""

    def __init__(self, alignment):
        self.alignment = alignment
        self.seen_shape = None

    def __call__(self, inp, sizes):
        self.seen_shape = tuple(inp.shape)
        return inp[: sizes[0]]


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _layer_config(**overrides):
    overrides.setdefault('params_dtype', torch.bfloat16)
    overrides.setdefault('bf16', True)
    overrides.setdefault('gradient_accumulation_fusion', False)
    overrides.setdefault('tensor_model_parallel_size', 2)
    return TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)


def _quantized_layer_config(**overrides):
    config = _layer_config(**overrides)
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


def _deterministic_input(seq, batch, hidden):
    values = torch.arange(seq * batch * hidden, dtype=torch.float32)
    return (((values % 11) - 5) / 4.0).to(torch.bfloat16).reshape(seq, batch, hidden)


def _deterministic_weight(rows, cols):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((values % 7) - 3).to(torch.bfloat16)


def _init_tp2(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    return parallel_state.get_tensor_model_parallel_group()


def _record_inert_quant_context(monkeypatch):
    """Record the fp8_autocast calls of te_ext but return inert contexts.

    SM80 (A100) cannot execute TE FP8 GEMMs, so the wrapper bookkeeping is asserted
    from the recorded call instead of entering the real quantized context.
    """
    calls = []

    def _inert_fp8_autocast(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(te_ext, 'fp8_autocast', _inert_fp8_autocast, raising=False)
    return calls


def test_wrap_te_linear_for_padding_wraps_module_forward_once(model_parallel):
    from megatron.core.extensions.transformer_engine import TELinear

    _init_tp2(model_parallel)
    layer = TELinear(
        16,
        8,
        parallel_mode='duplicated',
        config=_layer_config(),
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
    )
    fp8_utils._wrap_te_linear_for_padding(layer)
    wrapped_forward = layer.forward
    fp8_utils._wrap_te_linear_for_padding(layer)
    assert layer.forward is wrapped_forward


def test_padded_forward_passes_through_when_module_not_quantized(model_parallel):
    from megatron.core.extensions.transformer_engine import TELinear

    _init_tp2(model_parallel)
    layer = TELinear(
        16,
        8,
        parallel_mode='duplicated',
        config=_layer_config(),
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
    )
    layer.weight.data.copy_(_deterministic_weight(8, 16).cuda())
    fp8_utils._wrap_te_linear_for_padding(layer)

    x = _deterministic_input(6, 2, 16).cuda()
    out, bias = layer(x)
    assert bias is None
    expected = torch.nn.functional.linear(x, layer.weight)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_padded_forward_pads_and_unpads_when_module_quantized(model_parallel, monkeypatch):
    from megatron.core.extensions.transformer_engine import TELinear

    _init_tp2(model_parallel)
    pad, unpad = _FakeFp8Padding(1), _FakeFp8Unpadding(1)
    monkeypatch.setattr(fp8_utils, 'Fp8Padding', lambda _: pad)
    monkeypatch.setattr(fp8_utils, 'Fp8Unpadding', lambda _: unpad)
    quant_calls = _record_inert_quant_context(monkeypatch)

    layer = TELinear(
        16,
        8,
        parallel_mode='duplicated',
        config=_quantized_layer_config(),
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        name='layers.0.fc1',
    )
    assert layer.will_execute_quantized(False) is True
    layer.weight.data.copy_(_deterministic_weight(8, 16).cuda())
    fp8_utils._wrap_te_linear_for_padding(layer)

    x = _deterministic_input(6, 2, 16).cuda()
    out, bias = layer(x)
    assert bias is None
    expected = torch.nn.functional.linear(x, layer.weight)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    assert pad.seen_shape == (6, 2 * 16)
    assert unpad.seen_shape == (16, 2 * 8)
    (call,) = quant_calls
    assert call['enabled'] is True
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'
    assert call['fp8_group'] is parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=False
    )


def test_padded_forward_gathers_sequence_parallel_input_for_column_linear(
    model_parallel, monkeypatch
):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

    tp_group = _init_tp2(model_parallel)
    pad, unpad = _FakeFp8Padding(1), _FakeFp8Unpadding(1)
    monkeypatch.setattr(fp8_utils, 'Fp8Padding', lambda _: pad)
    monkeypatch.setattr(fp8_utils, 'Fp8Unpadding', lambda _: unpad)
    quant_calls = _record_inert_quant_context(monkeypatch)
    tp_rank = torch.distributed.get_rank(group=tp_group)

    layer = TEColumnParallelLinear(
        16,
        8,
        config=_quantized_layer_config(sequence_parallel=True),
        init_method=lambda w: None,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
        name='layers.0.fc1',
    )
    assert layer.will_execute_quantized(False) is True
    layer.weight.data.copy_(_deterministic_weight(4, 16).cuda())
    fp8_utils._wrap_te_linear_for_padding(layer)

    full_x = _deterministic_input(6, 2, 16).cuda()
    x_local = full_x[:3].clone() + tp_rank
    out, bias = layer(x_local)
    assert bias is None
    assert layer.sequence_parallel is True
    gathered = torch.empty(6, 2, 16, dtype=torch.bfloat16, device='cuda')
    torch.distributed.all_gather_into_tensor(gathered, x_local.contiguous(), group=tp_group)
    expected = torch.nn.functional.linear(gathered, layer.weight)
    assert out.shape == (6, 2, 4)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    assert pad.seen_shape == (6, 2 * 16)
    assert unpad.seen_shape == (16, 2 * 4)
    (call,) = quant_calls
    assert call['enabled'] is True
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'
    assert call['fp8_group'] is parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=False
    )


def test_padded_forward_reduce_scatters_sequence_parallel_output_for_row_linear(
    model_parallel, monkeypatch
):
    from megatron.core.extensions.transformer_engine import TERowParallelLinear

    tp_group = _init_tp2(model_parallel)
    pad, unpad = _FakeFp8Padding(1), _FakeFp8Unpadding(1)
    monkeypatch.setattr(fp8_utils, 'Fp8Padding', lambda _: pad)
    monkeypatch.setattr(fp8_utils, 'Fp8Unpadding', lambda _: unpad)
    quant_calls = _record_inert_quant_context(monkeypatch)
    tp_rank = torch.distributed.get_rank(group=tp_group)

    layer = TERowParallelLinear(
        16,
        8,
        config=_quantized_layer_config(sequence_parallel=True),
        init_method=lambda w: None,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
        name='layers.0.fc2',
    )
    assert layer.will_execute_quantized(False) is True
    layer.weight.data.copy_(_deterministic_weight(8, 8).cuda())
    fp8_utils._wrap_te_linear_for_padding(layer)

    full_x = _deterministic_input(6, 2, 16).cuda()
    x_local = full_x[:, :, tp_rank * 8 : (tp_rank + 1) * 8].contiguous()
    out, bias = layer(x_local)
    assert bias is None
    assert layer.sequence_parallel is True
    weight_parts = [torch.empty(8, 8, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    torch.distributed.all_gather(weight_parts, layer.weight.contiguous(), group=tp_group)
    full_weight = torch.cat(weight_parts, dim=1)
    full_out = torch.nn.functional.linear(full_x, full_weight)
    expected = full_out[tp_rank * 3 : (tp_rank + 1) * 3]
    assert out.shape == (3, 2, 8)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
    assert pad.seen_shape == (6, 2 * 8)
    assert unpad.seen_shape == (16, 2 * 8)
    (call,) = quant_calls
    assert call['enabled'] is True
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'
    assert call['fp8_group'] is parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=False
    )


def test_is_column_parallel_linear_classifies_layers(model_parallel):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    tp_group = _init_tp2(model_parallel)
    te_layer = TEColumnParallelLinear(
        16,
        8,
        config=_layer_config(),
        init_method=lambda w: None,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    megatron_layer = ColumnParallelLinear(
        16,
        8,
        config=_layer_config(use_cpu_initialization=True),
        init_method=lambda w: None,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    assert fp8_utils.is_column_parallel_linear(te_layer) is True
    assert fp8_utils.is_column_parallel_linear(megatron_layer) is True
    assert fp8_utils.is_column_parallel_linear(torch.nn.Linear(4, 4)) is False


def test_is_row_parallel_linear_classifies_layers(model_parallel):
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.tensor_parallel.layers import RowParallelLinear

    tp_group = _init_tp2(model_parallel)
    te_layer = TERowParallelLinear(
        16,
        8,
        config=_layer_config(),
        init_method=lambda w: None,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    megatron_layer = RowParallelLinear(
        16,
        8,
        config=_layer_config(use_cpu_initialization=True),
        init_method=lambda w: None,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )
    assert fp8_utils.is_row_parallel_linear(te_layer) is True
    assert fp8_utils.is_row_parallel_linear(megatron_layer) is True
    assert fp8_utils.is_row_parallel_linear(torch.nn.Linear(4, 4)) is False
