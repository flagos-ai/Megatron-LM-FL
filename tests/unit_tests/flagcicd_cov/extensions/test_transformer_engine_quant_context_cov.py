# Copyright (c) 2026, BAAI. All rights reserved.
"""TE per-module quantization: fp8_model_init / fp8_autocast context selection at the TE boundary."""

import functools
from contextlib import nullcontext

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions.transformer_engine import TEQuantizationParams, TEQuantizationRecipe
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_TENSORWISE = {'fp8_quantization_recipe': 'tensorwise', 'fp8_format': 'e4m3'}
_FACTORY_PATH = 'megatron.core.utils.get_te_version'


class _QuantizedGlobalState:
    """Boundary stand-in for TE's global FP8 autocast state, which SM80 cannot enable."""

    @staticmethod
    def is_fp8_enabled():
        return True


def _make_context_recorder(real_function, calls, returned):
    """Wrap a TE context constructor, recording keyword arguments and the context it returns."""

    @functools.wraps(real_function)
    def recorder(**kwargs):
        calls.append(kwargs)
        context = real_function(**kwargs)
        returned.append(context)
        return context

    return recorder


def _record_te_contexts(monkeypatch):
    """Wrap te_ext's fp8_model_init and fp8_autocast with recorders that keep real behavior."""
    recorders = {}
    for name in ('fp8_model_init', 'fp8_autocast'):
        calls = []
        returned = []
        monkeypatch.setattr(
            te_ext, name, _make_context_recorder(getattr(te_ext, name), calls, returned)
        )
        recorders[name] = calls
        recorders[f'{name}:returned'] = returned
    return recorders


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_fp8_model_init_returns_nullcontext_without_params(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    context = te_ext._get_fp8_model_init_for_quant_params(None, True)
    assert recorders['fp8_model_init'] == []
    assert recorders['fp8_autocast'] == []
    assert type(context) is nullcontext


def test_fp8_model_init_selects_evaluation_recipe_when_not_training(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(**{**_TENSORWISE, 'fp8_param': True}),
        evaluation_recipe=TEQuantizationRecipe(
            **{'fp8_quantization_recipe': 'blockwise', 'fp8_format': 'e4m3'}
        ),
    )
    context = te_ext._get_fp8_model_init_for_quant_params(params, training=False)
    assert context is recorders['fp8_model_init:returned'][-1]
    (call,) = recorders['fp8_model_init']
    assert call['enabled'] is False
    assert type(call['recipe']).__name__ == 'Float8BlockScaling'
    assert call['preserve_high_precision_init_val'] is True
    te_ext._get_fp8_model_init_for_quant_params(params, training=True)
    (training_call,) = recorders['fp8_model_init'][1:]
    assert training_call['enabled'] is True
    assert type(training_call['recipe']).__name__ == 'Float8CurrentScaling'


def test_fp8_model_init_falls_back_to_training_recipe_without_evaluation(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(**_TENSORWISE), evaluation_recipe=None
    )
    for training in (True, False):
        context = te_ext._get_fp8_model_init_for_quant_params(params, training)
        assert context is recorders['fp8_model_init:returned'][-1]
    assert len(recorders['fp8_model_init']) == 2
    for call in recorders['fp8_model_init']:
        assert type(call['recipe']).__name__ == 'Float8CurrentScaling'


@pytest.mark.parametrize(
    ('recipe', 'message'),
    [
        ({**_TENSORWISE, 'fp8_format': 'bogus'}, 'Unhandled fp8_format bogus'),
        ({'fp8_quantization_recipe': 'weird', 'fp8_format': 'e4m3'}, 'Unhandled fp8 recipe'),
        ({'fp4_quantization_recipe': 'weird'}, 'Unhandled fp4 recipe'),
    ],
    ids=['bad-fp8-format', 'bad-fp8-recipe', 'bad-fp4-recipe'],
)
def test_fp8_model_init_for_quant_recipe_rejects_unknown_values(recipe, message):
    with pytest.raises(ValueError, match=message):
        te_ext._get_fp8_model_init_for_quant_recipe(TEQuantizationRecipe(**recipe))


def test_fp8_model_init_for_quant_recipe_requires_factory_for_custom_fp8_recipe():
    recipe = TEQuantizationRecipe(fp8_quantization_recipe=Fp8Recipe.custom)
    with pytest.raises(AssertionError):
        te_ext._get_fp8_model_init_for_quant_recipe(recipe)


def test_fp8_model_init_for_quant_recipe_requires_factory_for_custom_fp4_recipe():
    recipe = TEQuantizationRecipe(fp4_quantization_recipe=Fp4Recipe.custom)
    with pytest.raises(AssertionError):
        te_ext._get_fp8_model_init_for_quant_recipe(recipe)


@pytest.mark.parametrize(
    ('recipe', 'expected_recipe_type'),
    [
        ({'fp8_quantization_recipe': 'tensorwise', 'fp8_format': 'e4m3'}, 'Float8CurrentScaling'),
        ({'fp8_quantization_recipe': 'tensorwise', 'fp8_format': 'hybrid'}, 'Float8CurrentScaling'),
        ({'fp8_quantization_recipe': 'blockwise', 'fp8_format': 'e4m3'}, 'Float8BlockScaling'),
        ({'fp8_quantization_recipe': 'mxfp8', 'fp8_format': 'e4m3'}, 'MXFP8BlockScaling'),
        (
            {'fp8_quantization_recipe': 'custom', 'custom_recipe_factory': _FACTORY_PATH},
            'CustomRecipe',
        ),
        ({'fp4_quantization_recipe': 'nvfp4'}, 'NVFP4BlockScaling'),
        (
            {'fp4_quantization_recipe': 'custom', 'custom_recipe_factory': _FACTORY_PATH},
            'CustomRecipe',
        ),
    ],
    ids=[
        'fp8-e4m3',
        'fp8-hybrid',
        'fp8-blockwise',
        'fp8-mxfp8',
        'fp8-custom',
        'fp4-nvfp4',
        'fp4-custom',
    ],
)
def test_fp8_model_init_for_quant_recipe_passes_selected_recipe_to_te(
    monkeypatch, recipe, expected_recipe_type
):
    recorders = _record_te_contexts(monkeypatch)
    context = te_ext._get_fp8_model_init_for_quant_recipe(TEQuantizationRecipe(**recipe))
    assert context is recorders['fp8_model_init:returned'][-1]
    (call,) = recorders['fp8_model_init']
    assert call['enabled'] is False
    assert call['preserve_high_precision_init_val'] is True
    assert type(call['recipe']).__name__ == expected_recipe_type
    assert recorders['fp8_autocast'] == []


def test_fp8_model_init_for_quant_recipe_enables_fp8_params_when_requested(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    recipe = TEQuantizationRecipe(**{**_TENSORWISE, 'fp8_param': True})
    context = te_ext._get_fp8_model_init_for_quant_recipe(recipe)
    assert context is recorders['fp8_model_init:returned'][-1]
    (call,) = recorders['fp8_model_init']
    assert call['enabled'] is True
    assert type(call['recipe']).__name__ == 'Float8CurrentScaling'


def test_fp8_model_init_for_quant_recipe_builds_disabled_context_for_bf16_recipe(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    context = te_ext._get_fp8_model_init_for_quant_recipe(TEQuantizationRecipe())
    assert context is recorders['fp8_model_init:returned'][-1]
    (call,) = recorders['fp8_model_init']
    assert call == {'enabled': False, 'recipe': None, 'preserve_high_precision_init_val': True}


@pytest.mark.parametrize(
    ('recipe', 'expected_selection'),
    [
        ({'override_nonquantized_autocast': False}, 'nullcontext'),
        ({'override_quantized_autocast': False}, 'nullcontext'),
        ({'override_nonquantized_autocast': True}, 'bf16-override'),
        ({**_TENSORWISE, 'override_nonquantized_autocast': True}, 'Float8CurrentScaling'),
        (
            {'fp4_quantization_recipe': 'nvfp4', 'override_nonquantized_autocast': True},
            'NVFP4BlockScaling',
        ),
        (
            {
                'fp8_quantization_recipe': 'mxfp8',
                'fp8_format': 'hybrid',
                'override_nonquantized_autocast': True,
            },
            'MXFP8BlockScaling',
        ),
        (
            {
                'fp8_quantization_recipe': 'blockwise',
                'fp8_format': 'e4m3',
                'override_nonquantized_autocast': True,
            },
            'Float8BlockScaling',
        ),
        (
            {
                'fp8_quantization_recipe': 'custom',
                'custom_recipe_factory': _FACTORY_PATH,
                'override_nonquantized_autocast': True,
            },
            'CustomRecipe',
        ),
        (
            {
                'fp4_quantization_recipe': 'custom',
                'custom_recipe_factory': _FACTORY_PATH,
                'override_nonquantized_autocast': True,
            },
            'CustomRecipe',
        ),
    ],
    ids=[
        'no-override-bf16-context',
        'no-override-quantized-context',
        'force-bf16',
        'force-fp8-tensorwise',
        'force-fp4-nvfp4',
        'force-fp8-mxfp8-hybrid',
        'force-fp8-blockwise',
        'force-fp8-custom',
        'force-fp4-custom',
    ],
)
def test_fp8_autocast_selection_follows_override_flags(monkeypatch, recipe, expected_selection):
    recorders = _record_te_contexts(monkeypatch)
    context = te_ext._get_fp8_autocast_for_quant_recipe(TEQuantizationRecipe(**recipe))
    if expected_selection == 'nullcontext':
        assert recorders['fp8_autocast'] == []
        assert type(context) is nullcontext
    elif expected_selection == 'bf16-override':
        assert context is recorders['fp8_autocast:returned'][-1]
        assert recorders['fp8_autocast'] == [{'enabled': False}]
    else:
        assert context is recorders['fp8_autocast:returned'][-1]
        (call,) = recorders['fp8_autocast']
        assert call['enabled'] is True
        assert call['fp8_group'] is None
        assert type(call['fp8_recipe']).__name__ == expected_selection


def test_fp8_autocast_returns_nullcontext_without_params(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    context = te_ext._get_fp8_autocast_for_quant_params(None, True)
    assert recorders['fp8_autocast'] == []
    assert recorders['fp8_model_init'] == []
    assert type(context) is nullcontext


def test_fp8_autocast_selects_evaluation_recipe_when_not_training(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(
            **{
                'fp8_quantization_recipe': 'mxfp8',
                'fp8_format': 'e4m3',
                'override_nonquantized_autocast': True,
            }
        ),
        evaluation_recipe=TEQuantizationRecipe(
            **{**_TENSORWISE, 'override_nonquantized_autocast': True}
        ),
    )
    for training in (True, False):
        context = te_ext._get_fp8_autocast_for_quant_params(params, training)
        assert context is recorders['fp8_autocast:returned'][-1]
    (training_call, evaluation_call) = recorders['fp8_autocast']
    assert type(training_call['fp8_recipe']).__name__ == 'MXFP8BlockScaling'
    assert type(evaluation_call['fp8_recipe']).__name__ == 'Float8CurrentScaling'


def test_fp8_autocast_falls_back_to_training_recipe_without_evaluation(monkeypatch):
    recorders = _record_te_contexts(monkeypatch)
    params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(
            **{**_TENSORWISE, 'override_nonquantized_autocast': True}
        ),
        evaluation_recipe=None,
    )
    for training in (True, False):
        te_ext._get_fp8_autocast_for_quant_params(params, training)
    assert len(recorders['fp8_autocast']) == 2
    for call in recorders['fp8_autocast']:
        assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'


@pytest.mark.parametrize(
    ('recipe', 'message'),
    [
        (
            {
                'fp8_quantization_recipe': 'weird',
                'fp8_format': 'e4m3',
                'override_nonquantized_autocast': True,
            },
            'Unhandled fp8 recipe',
        ),
        (
            {'fp4_quantization_recipe': 'weird', 'override_nonquantized_autocast': True},
            'Unhandled fp4 recipe',
        ),
    ],
    ids=['bad-fp8-recipe', 'bad-fp4-recipe'],
)
def test_fp8_autocast_for_quant_recipe_rejects_unknown_recipes(recipe, message):
    with pytest.raises(ValueError, match=message):
        te_ext._get_fp8_autocast_for_quant_recipe(TEQuantizationRecipe(**recipe))


def test_fp8_autocast_for_quant_recipe_rejects_bad_format():
    recipe = TEQuantizationRecipe(
        fp8_quantization_recipe=Fp8Recipe.tensorwise,
        fp8_format='bogus',
        override_nonquantized_autocast=True,
    )
    with pytest.raises(ValueError, match='Unhandled fp8_format bogus'):
        te_ext._get_fp8_autocast_for_quant_recipe(recipe)


def test_fp8_autocast_keeps_quantized_context_when_override_disabled(monkeypatch):
    monkeypatch.setattr(te_ext, 'FP8GlobalStateManager', _QuantizedGlobalState, raising=False)
    recorders = _record_te_contexts(monkeypatch)
    recipe = TEQuantizationRecipe(override_quantized_autocast=False)
    context = te_ext._get_fp8_autocast_for_quant_recipe(recipe)
    assert recorders['fp8_autocast'] == []
    assert type(context) is nullcontext


def test_fp8_autocast_overrides_quantized_context_when_requested(monkeypatch):
    monkeypatch.setattr(te_ext, 'FP8GlobalStateManager', _QuantizedGlobalState, raising=False)
    recorders = _record_te_contexts(monkeypatch)
    recipe = TEQuantizationRecipe(**{**_TENSORWISE, 'override_quantized_autocast': True})
    context = te_ext._get_fp8_autocast_for_quant_recipe(recipe)
    assert context is recorders['fp8_autocast:returned'][-1]
    (call,) = recorders['fp8_autocast']
    assert call['enabled'] is True
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'


def test_fp8_autocast_uses_amax_group_when_parallel_initialized(model_parallel, monkeypatch):
    model_parallel(tensor_model_parallel_size=2)
    recorders = _record_te_contexts(monkeypatch)
    recipe = TEQuantizationRecipe(
        fp8_quantization_recipe=Fp8Recipe.tensorwise, override_nonquantized_autocast=True
    )
    context = te_ext._get_fp8_autocast_for_quant_recipe(recipe)
    assert context is recorders['fp8_autocast:returned'][-1]
    (call,) = recorders['fp8_autocast']
    assert call['fp8_group'] is parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=False
    )
    assert type(call['fp8_recipe']).__name__ == 'Float8CurrentScaling'


@pytest.mark.parametrize(
    ('override', 'expected_device'),
    [({'use_cpu_initialization': True}, 'cpu'), ({'init_model_with_meta_device': True}, 'meta')],
    ids=['cpu-init', 'meta-init'],
)
def test_get_extra_te_kwargs_selects_device_from_config(override, expected_device):
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    for name, value in override.items():
        setattr(config, name, value)
    assert te_ext._get_extra_te_kwargs(config) == {
        'params_dtype': torch.float32,
        'device': expected_device,
    }
