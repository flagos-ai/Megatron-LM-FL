# Copyright (c) 2026, BAAI. All rights reserved.
"""FP4 utils packed shapes, alignment, shard-cast delegation and context selection."""

import functools
from contextlib import nullcontext

import pytest
import torch

from megatron.core import fp4_utils, parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_FACTORY_PATH = 'megatron.core.utils.get_te_version'


def _make_context_recorder(real_function, calls):
    """Wrap a TE context constructor, recording keyword arguments of every call."""

    @functools.wraps(real_function)
    def recorder(**kwargs):
        calls.append(kwargs)
        return real_function(**kwargs)

    return recorder


def _record_te_pytorch_contexts(monkeypatch):
    """Wrap the fp8_autocast / fp8_model_init functions fp4_utils resolves at call time.

    The wrappers keep the real function's signature (functools.wraps) because
    fp4_utils inspects it before building fp8_model_init arguments.
    """
    import transformer_engine.pytorch as te_pytorch

    recorders = {}
    for name in ('fp8_autocast', 'fp8_model_init'):
        calls = []
        monkeypatch.setattr(
            te_pytorch, name, _make_context_recorder(getattr(te_pytorch, name), calls)
        )
        recorders[name] = calls
    return recorders


def _record_quantize_master_weights(monkeypatch):
    """Record the TE quantize_master_weights boundary fp4_utils imports inside the call."""
    import transformer_engine.pytorch.tensor.utils as te_tensor_utils

    calls = []

    def recording_quantize(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(te_tensor_utils, 'quantize_master_weights', recording_quantize)
    return calls


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    config = TransformerConfig(num_layers=4, hidden_size=32, num_attention_heads=4)
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize(
    ('shape', 'expected'),
    [
        (torch.Size([2, 4]), torch.Size([2, 2])),
        (torch.Size([6]), torch.Size([3])),
        (torch.Size([3, 5, 8]), torch.Size([3, 5, 4])),
        (torch.Size([]), torch.Size([])),
    ],
    ids=['2d', '1d', '3d', 'scalar'],
)
def test_get_nvfp4_rowwise_packed_shape_halves_last_dim(shape, expected):
    assert fp4_utils.get_nvfp4_rowwise_packed_shape(shape) == expected


def test_get_nvfp4_rowwise_packed_shape_rejects_odd_last_dim():
    with pytest.raises(AssertionError, match='NVFP4 requires inner dimension divisible by 2'):
        fp4_utils.get_nvfp4_rowwise_packed_shape(torch.Size([2, 5]))


def test_get_fp4_align_size_returns_128():
    assert fp4_utils.get_fp4_align_size(None) == 128


def test_modify_nvfp4_rowwise_storage_rejects_plain_tensor():
    with pytest.raises(ValueError, match='modify_nvfp4_rowwise_storage expects an NVFP4 tensor'):
        fp4_utils.modify_nvfp4_rowwise_storage(
            torch.zeros(4, 2, dtype=torch.uint8), torch.zeros(4, 2, dtype=torch.uint8)
        )


def test_quantize_nvfp4_param_shard_skips_te_entirely_for_empty_params(monkeypatch):
    quantize_calls = _record_quantize_master_weights(monkeypatch)
    result = fp4_utils.quantize_nvfp4_param_shard([], [], [], None, fsdp_shard_model_params=[])
    assert result is None
    assert quantize_calls == []


@pytest.mark.parametrize(
    ('te_post_all_gather_processing', 'expected_kwargs'),
    [(lambda params: None, {'manual_post_all_gather_processing': True}), (None, {})],
    ids=['with-post-processing', 'without-post-processing'],
)
def test_quantize_nvfp4_param_shard_delegates_shard_casting_to_te(
    monkeypatch, te_post_all_gather_processing, expected_kwargs
):
    quantize_calls = _record_quantize_master_weights(monkeypatch)
    monkeypatch.setattr(fp4_utils, 'te_post_all_gather_processing', te_post_all_gather_processing)
    model_params, main_params, offsets, fsdp_shards = (
        [torch.zeros(2)],
        [torch.ones(2)],
        [0],
        [torch.zeros(1)],
    )
    result = fp4_utils.quantize_nvfp4_param_shard(
        model_params, main_params, offsets, 'dp-group', fsdp_shard_model_params=fsdp_shards
    )
    assert result is None
    (args, kwargs) = quantize_calls[0]
    assert args == (model_params, main_params, offsets, 'dp-group', fsdp_shards)
    assert kwargs == expected_kwargs


def test_get_fp4_recipe_returns_nvfp4_recipe():
    recipe = fp4_utils.get_fp4_recipe(_config(fp4='e2m1', fp4_recipe='nvfp4'))
    assert type(recipe).__name__ == 'NVFP4BlockScaling'
    assert recipe.fp8_dpa is False


def test_get_fp4_recipe_builds_custom_recipe_from_factory_path():
    recipe = fp4_utils.get_fp4_recipe(
        _config(fp4='e2m1', fp4_recipe='custom', fp4_quantizer_factory=_FACTORY_PATH)
    )
    assert type(recipe).__name__ == 'CustomRecipe'


def test_get_fp4_recipe_rejects_unknown_recipe():
    with pytest.raises(ValueError, match='NVFP4BlockScaling and custom are the only supported'):
        fp4_utils.get_fp4_recipe(_config(fp4='e2m1', fp4_recipe='bogus'))


def test_get_fp4_context_returns_nullcontext_when_fp4_disabled(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    context = fp4_utils.get_fp4_context(_config())
    assert recorders['fp8_autocast'] == []
    assert recorders['fp8_model_init'] == []
    assert type(context) is nullcontext


def test_get_fp4_context_returns_nullcontext_for_bf16_boundary_layer(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(
        fp4='e2m1',
        fp4_recipe='nvfp4',
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )
    context = fp4_utils.get_fp4_context(config, layer_no=0)
    assert recorders['fp8_autocast'] == []
    assert recorders['fp8_model_init'] == []
    assert type(context) is nullcontext


def test_get_fp4_context_passes_selected_recipe_to_fp8_autocast(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp4='e2m1', fp4_recipe='nvfp4')
    context = fp4_utils.get_fp4_context(config, layer_no=1)
    assert type(context).__name__ == 'fp8_autocast'
    (call,) = recorders['fp8_autocast']
    assert call['enabled'] is True
    assert call['fp8_group'] is None
    assert type(call['fp8_recipe']).__name__ == 'NVFP4BlockScaling'
    assert recorders['fp8_model_init'] == []


def test_get_fp4_context_passes_selected_recipe_to_fp8_model_init(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp4='e2m1', fp4_recipe='nvfp4', fp4_param=True)
    context = fp4_utils.get_fp4_context(config, is_init=True)
    assert type(context).__name__ == 'fp8_model_init'
    (call,) = recorders['fp8_model_init']
    assert call['enabled'] is True
    assert type(call['recipe']).__name__ == 'NVFP4BlockScaling'
    assert call['preserve_high_precision_init_val'] is True
    assert recorders['fp8_autocast'] == []


def test_get_fp4_context_builds_init_context_without_layer_number(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp4='e2m1', fp4_recipe='nvfp4', fp4_param=True)
    context = fp4_utils.get_fp4_context(config, layer_no=-1, is_init=True)
    assert type(context).__name__ == 'fp8_model_init'
    (call,) = recorders['fp8_model_init']
    assert call['enabled'] is True
    assert type(call['recipe']).__name__ == 'NVFP4BlockScaling'


def test_get_fp4_context_uses_amax_group_when_parallel_initialized(model_parallel, monkeypatch):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp4='e2m1', fp4_recipe='nvfp4')
    context = fp4_utils.get_fp4_context(config, layer_no=1)
    assert type(context).__name__ == 'fp8_autocast'
    (call,) = recorders['fp8_autocast']
    assert call['fp8_group'] is parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=False
    )
    assert type(call['fp8_recipe']).__name__ == 'NVFP4BlockScaling'
