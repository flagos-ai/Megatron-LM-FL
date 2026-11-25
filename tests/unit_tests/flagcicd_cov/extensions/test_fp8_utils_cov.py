# Copyright (c) 2026, BAAI. All rights reserved.
"""FP8 utils recipe/context selection, shard-cast delegation and boundary guards."""

import functools
from contextlib import nullcontext

import pytest
import torch

from megatron.core import fp8_utils, parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import te
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_FACTORY_PATH = 'megatron.core.utils.get_te_version'


def _make_context_recorder(real_function, calls, returned):
    """Wrap a TE context constructor, recording keyword arguments and the context it returns."""

    @functools.wraps(real_function)
    def recorder(**kwargs):
        calls.append(kwargs)
        context = real_function(**kwargs)
        returned.append(context)
        return context

    return recorder


def _record_te_pytorch_contexts(monkeypatch):
    """Wrap the fp8_autocast / fp8_model_init functions fp8_utils resolves at call time.

    The wrappers keep the real function's signature (functools.wraps) because
    fp8_utils inspects it before building fp8_model_init arguments.
    """
    import transformer_engine.pytorch as te_pytorch

    recorders = {}
    for name in ('fp8_autocast', 'fp8_model_init'):
        calls = []
        returned = []
        monkeypatch.setattr(
            te_pytorch, name, _make_context_recorder(getattr(te_pytorch, name), calls, returned)
        )
        recorders[name] = calls
        recorders[f'{name}:returned'] = returned
    return recorders


def _record_cast_master_weights(monkeypatch):
    """Record the TE cast_master_weights_to_fp8 boundary fp8_utils imports inside the call."""
    import transformer_engine.pytorch.tensor.utils as te_tensor_utils

    calls = []

    def recording_cast(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(te_tensor_utils, 'cast_master_weights_to_fp8', recording_cast)
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
    ('dotted_path', 'message'),
    [
        (None, "must be a non-empty string with format 'pkg.mod.func'."),
        ('', "must be a non-empty string with format 'pkg.mod.func'."),
        ('nodot', "Invalid fp8_quantizer_factory 'nodot'. Expected 'pkg.mod.func'."),
        ('flagcicd_cov_missing_module.fn', "Failed to import module 'flagcicd_cov_missing_module'"),
        (
            'megatron.core.fp8_utils.no_such_attr',
            "Attribute 'no_such_attr' not found in module 'megatron.core.fp8_utils'",
        ),
        (
            'megatron.core.fp8_utils.HAVE_TE',
            "Resolved attribute 'megatron.core.fp8_utils.HAVE_TE' is not callable",
        ),
    ],
    ids=['none', 'empty', 'no-dot', 'bad-module', 'missing-attribute', 'not-callable'],
)
def test_resolve_callable_rejects_invalid_import_paths(dotted_path, message):
    with pytest.raises(ValueError, match=message):
        fp8_utils._resolve_callable_from_python_import_path(dotted_path)


def test_resolve_callable_resolves_module_attribute():
    from megatron.core.utils import get_te_version

    assert fp8_utils._resolve_callable_from_python_import_path(_FACTORY_PATH) is get_te_version


def test_get_custom_recipe_wraps_resolved_factory():
    recipe = fp8_utils._get_custom_recipe(_FACTORY_PATH)
    assert type(recipe).__name__ == 'CustomRecipe'


@pytest.mark.parametrize(
    ('recipe', 'expected'), [('mxfp8', 32), ('tensorwise', 16), ('delayed', 16)]
)
def test_fp8_align_size_depends_on_recipe(recipe, expected):
    assert fp8_utils.get_fp8_align_size(Fp8Recipe(recipe)) == expected


@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [
        ({}, [False, False, False, False]),
        (
            {
                'first_last_layers_bf16': True,
                'num_layers_at_start_in_bf16': 1,
                'num_layers_at_end_in_bf16': 1,
            },
            [True, False, False, True],
        ),
        (
            {
                'first_last_layers_bf16': True,
                'num_layers_at_start_in_bf16': 2,
                'num_layers_at_end_in_bf16': 1,
            },
            [True, True, False, True],
        ),
    ],
    ids=['bf16-boundaries-disabled', 'first-and-last-layer', 'two-first-one-last'],
)
def test_is_first_last_bf16_layer_selects_boundary_layers(overrides, expected):
    config = _config(**overrides)
    selected = [fp8_utils.is_first_last_bf16_layer(config, layer) for layer in range(4)]
    assert selected == expected


def test_is_first_last_bf16_layer_treats_negative_layer_number_as_non_boundary():
    config = _config(
        first_last_layers_bf16=True, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1
    )
    assert fp8_utils.is_first_last_bf16_layer(config, -1) is False


def test_quantize_param_shard_skips_te_entirely_for_empty_params(monkeypatch):
    cast_calls = _record_cast_master_weights(monkeypatch)
    result = fp8_utils.quantize_param_shard([], [], [], None, fsdp_shard_model_params=[])
    assert result is None
    assert cast_calls == []


@pytest.mark.parametrize(
    ('fsdp_shard_model_params', 'expect_fsdp_arg'),
    [(None, False), ([], True)],
    ids=['no-fsdp', 'fsdp-shards'],
)
def test_quantize_param_shard_delegates_shard_casting_to_te(
    monkeypatch, fsdp_shard_model_params, expect_fsdp_arg
):
    cast_calls = _record_cast_master_weights(monkeypatch)
    monkeypatch.setattr(fp8_utils, 'te_post_all_gather_processing', lambda params: None)
    model_params, main_params, offsets = [torch.zeros(2)], [torch.ones(2)], [0]
    fp8_utils.quantize_param_shard(
        model_params, main_params, offsets, 'dp-group', fsdp_shard_model_params
    )
    (args, kwargs) = cast_calls[0]
    expected_args = [model_params, main_params, offsets, 'dp-group']
    if expect_fsdp_arg:
        expected_args.append(fsdp_shard_model_params)
    assert args == tuple(expected_args)
    assert kwargs == {'manual_post_all_gather_processing': True}


def test_correct_amax_history_leaves_plain_params_untouched():
    layer = torch.nn.Linear(8, 8)
    before = layer.weight.detach().clone()
    assert fp8_utils.correct_amax_history_if_needed([layer]) is None
    assert torch.equal(layer.weight.detach(), before)


def test_post_all_gather_processing_delegates_params_to_te(monkeypatch):
    delegated = []
    monkeypatch.setattr(
        fp8_utils, 'te_post_all_gather_processing', lambda params: delegated.append(params)
    )
    model_params = [torch.zeros(4, 4)]
    assert fp8_utils.post_all_gather_processing(model_params) is None
    assert delegated == [model_params]


def test_post_all_gather_processing_is_a_noop_when_te_lacks_support(monkeypatch):
    monkeypatch.setattr(fp8_utils, 'te_post_all_gather_processing', None)
    model_params = [torch.zeros(4, 4)]
    assert fp8_utils.post_all_gather_processing(model_params) is None
    assert torch.equal(model_params[0], torch.zeros(4, 4))


@pytest.mark.parametrize(
    ('fp8', 'recipe', 'expected_type'),
    [
        ('e4m3', 'delayed', 'TEDelayedScaling'),
        ('e4m3', 'tensorwise', 'Float8CurrentScaling'),
        ('e4m3', 'blockwise', 'Float8BlockScaling'),
        ('e4m3', 'mxfp8', 'MXFP8BlockScaling'),
        ('hybrid', 'mxfp8', 'MXFP8BlockScaling'),
        ('e4m3', 'custom', 'CustomRecipe'),
    ],
    ids=['delayed', 'tensorwise', 'blockwise', 'mxfp8', 'hybrid', 'custom'],
)
def test_get_fp8_recipe_returns_matching_te_recipe(fp8, recipe, expected_type):
    config = _config(fp8=fp8, fp8_recipe=recipe, fp8_quantizer_factory=_FACTORY_PATH)
    result = fp8_utils.get_fp8_recipe(config)
    assert type(result).__name__ == expected_type
    assert isinstance(result, te.common.recipe.DelayedScaling) is (recipe == 'delayed')


def test_get_fp8_recipe_warns_fp8_interval_is_ignored():
    config = _config(fp8='e4m3', fp8_recipe='delayed', fp8_interval=2)
    with pytest.warns(UserWarning, match='fp8_interval is deprecated and ignored'):
        result = fp8_utils.get_fp8_recipe(config)
    assert type(result).__name__ == 'TEDelayedScaling'


def test_get_fp8_recipe_rejects_unsupported_format():
    with pytest.raises(ValueError, match='E4M3 and HYBRID are the only supported FP8 formats.'):
        fp8_utils.get_fp8_recipe(_config(fp8='bogus', fp8_recipe='delayed'))


def test_get_fp8_recipe_rejects_unsupported_recipe():
    with pytest.raises(ValueError, match='are the only supported FP8 recipes.'):
        fp8_utils.get_fp8_recipe(_config(fp8='e4m3', fp8_recipe='bogus'))


def test_get_fp8_context_returns_nullcontext_for_bf16_training(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    context = fp8_utils.get_fp8_context(_config())
    assert recorders['fp8_autocast'] == []
    assert recorders['fp8_model_init'] == []
    assert type(context) is nullcontext


def test_get_fp8_context_returns_nullcontext_for_bf16_boundary_layer(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(
        fp8='e4m3',
        fp8_recipe='tensorwise',
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )
    context = fp8_utils.get_fp8_context(config, layer_no=0)
    assert recorders['fp8_autocast'] == []
    assert recorders['fp8_model_init'] == []
    assert type(context) is nullcontext


def test_get_fp8_context_passes_selected_recipe_to_fp8_autocast(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp8='e4m3', fp8_recipe='tensorwise')
    context = fp8_utils.get_fp8_context(config, layer_no=1)
    assert context is recorders['fp8_autocast:returned'][-1]
    (call,) = recorders['fp8_autocast']
    assert call['enabled'] is True
    assert call['fp8_group'] is None
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'
    assert recorders['fp8_model_init'] == []


def test_get_fp8_context_passes_selected_recipe_to_fp8_model_init(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp8='e4m3', fp8_recipe='tensorwise', fp8_param=True)
    context = fp8_utils.get_fp8_context(config, is_init=True)
    assert context is recorders['fp8_model_init:returned'][-1]
    (call,) = recorders['fp8_model_init']
    assert call['enabled'] is True
    assert type(call['recipe']).__name__ == 'Float8CurrentScaling'
    assert call['preserve_high_precision_init_val'] is True
    assert recorders['fp8_autocast'] == []


def test_get_fp8_context_rejects_delayed_scaling_with_bf16_boundary_layers():
    config = _config(
        fp8='e4m3',
        fp8_recipe='delayed',
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )
    with pytest.raises(AssertionError, match='Delayed scaling does not support first / last layer'):
        fp8_utils.get_fp8_context(config, layer_no=1)


def test_get_fp8_context_uses_amax_group_when_parallel_initialized(model_parallel, monkeypatch):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp8='e4m3', fp8_recipe='tensorwise')
    context = fp8_utils.get_fp8_context(config, layer_no=1)
    assert context is recorders['fp8_autocast:returned'][-1]
    (call,) = recorders['fp8_autocast']
    assert call['fp8_group'] is parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=False
    )
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'


def test_get_fp8_disabled_context_returns_nullcontext_without_quantization(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    assert type(fp8_utils.get_fp8_disabled_context(_config())) is nullcontext
    assert type(fp8_utils.get_fp8_disabled_context(_config(), is_init=True)) is nullcontext
    assert recorders['fp8_autocast'] == []
    assert recorders['fp8_model_init'] == []


def test_get_fp8_disabled_context_passes_disabled_flag_to_fp8_autocast(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp8='e4m3', fp8_recipe='tensorwise')
    context = fp8_utils.get_fp8_disabled_context(config)
    assert context is recorders['fp8_autocast:returned'][-1]
    assert recorders['fp8_autocast'] == [{'enabled': False}]
    assert recorders['fp8_model_init'] == []


def test_get_fp8_disabled_context_passes_disabled_flag_to_fp8_model_init(monkeypatch):
    recorders = _record_te_pytorch_contexts(monkeypatch)
    config = _config(fp8='e4m3', fp8_recipe='tensorwise', fp8_param=True)
    context = fp8_utils.get_fp8_disabled_context(config, is_init=True)
    assert context is recorders['fp8_model_init:returned'][-1]
    assert recorders['fp8_model_init'] == [{'enabled': False}]
    assert recorders['fp8_autocast'] == []
