# Copyright (c) 2026, BAAI. All rights reserved.
"""TE per-module quantization recipe/params parsing and context-quantization decisions."""

import pytest

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions.transformer_engine import (
    TEDelayedScaling,
    TEQuantizationParams,
    TEQuantizationRecipe,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.quantization.quant_config import MatchContext, QuantizationConfig

_TENSORWISE = {'fp8_quantization_recipe': 'tensorwise', 'fp8_format': 'e4m3'}


def _quant_config(recipe):
    return QuantizationConfig(
        dict(recipe),
        match_input=MatchContext(module_path='layers.0.fc1', layer_number=0),
        config_key='default',
    )


def _params(training, evaluation=None):
    return TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(**training),
        evaluation_recipe=TEQuantizationRecipe(**evaluation) if evaluation is not None else None,
    )


def test_recipe_parse_rejects_unknown_field():
    with pytest.raises(ValueError, match="Field 'bogus' not valid"):
        TEQuantizationRecipe.parse_from_config({**_TENSORWISE, 'bogus': 1})


def test_recipe_parse_rejects_delayed_scaling_recipe():
    with pytest.raises(ValueError, match='Delayed scaling not in scope'):
        TEQuantizationRecipe.parse_from_config({'fp8_quantization_recipe': 'delayed'})


def test_recipe_parse_rejects_fp8_and_fp4_together():
    with pytest.raises(ValueError, match='mutually exclusive'):
        TEQuantizationRecipe.parse_from_config(
            {'fp8_quantization_recipe': 'tensorwise', 'fp4_quantization_recipe': 'nvfp4'}
        )


def test_recipe_parse_requires_factory_for_custom_recipe():
    with pytest.raises(ValueError, match='requires custom_recipe_factory'):
        TEQuantizationRecipe.parse_from_config({'fp8_quantization_recipe': 'custom'})


def test_recipe_parse_copies_known_fields_and_defaults_rest():
    recipe = TEQuantizationRecipe.parse_from_config(
        {
            **_TENSORWISE,
            'override_quantized_autocast': False,
            'custom_recipe_factory': 'pkg.mod.factory',
        }
    )
    assert recipe.fp8_quantization_recipe == Fp8Recipe.tensorwise
    assert recipe.fp4_quantization_recipe is None
    assert recipe.fp8_format == 'e4m3'
    assert recipe.override_quantized_autocast is False
    assert recipe.override_nonquantized_autocast is False
    assert recipe.custom_recipe_factory == 'pkg.mod.factory'
    assert recipe.fp8_param is False
    assert recipe.tp_only_amax_red is False


def test_params_parse_requires_config_type_key():
    with pytest.raises(ValueError, match="must have 'transformer_engine_config_type' key"):
        TEQuantizationParams.parse_from_config(_quant_config(_TENSORWISE))


def test_params_parse_rejects_unknown_config_type():
    with pytest.raises(ValueError, match="Unsupported config type 'bogus'"):
        TEQuantizationParams.parse_from_config(
            _quant_config(
                {'transformer_engine_config_type': 'bogus', 'training_recipe': _TENSORWISE}
            )
        )


def test_params_parse_requires_training_recipe():
    with pytest.raises(ValueError, match="must have 'training_recipe' key"):
        TEQuantizationParams.parse_from_config(
            _quant_config({'transformer_engine_config_type': 'TEQuantizationParams'})
        )


def test_params_parse_uses_training_recipe_only_without_evaluation_recipe():
    params = TEQuantizationParams.parse_from_config(
        _quant_config(
            {
                'transformer_engine_config_type': 'TEQuantizationParams',
                'training_recipe': _TENSORWISE,
            }
        )
    )
    assert params.evaluation_recipe is None
    assert params.training_recipe.fp8_quantization_recipe == Fp8Recipe.tensorwise


def test_params_parse_reads_evaluation_recipe_when_present():
    params = TEQuantizationParams.parse_from_config(
        _quant_config(
            {
                'transformer_engine_config_type': 'TEQuantizationParams',
                'training_recipe': _TENSORWISE,
                'evaluation_recipe': {'fp4_quantization_recipe': 'nvfp4'},
            }
        )
    )
    assert params.training_recipe.fp8_quantization_recipe == Fp8Recipe.tensorwise
    assert params.evaluation_recipe.fp4_quantization_recipe == Fp4Recipe.nvfp4


@pytest.mark.parametrize(
    ('training', 'is_context_quantized'),
    [(True, True), (True, False), (False, True), (False, False)],
    ids=['training-quantized', 'training-bf16', 'evaluation-quantized', 'evaluation-bf16'],
)
def test_should_context_be_quantized_passes_context_flag_through_without_params(
    training, is_context_quantized
):
    assert (
        te_ext._get_should_context_be_quantized_params(None, training, is_context_quantized)
        is is_context_quantized
    )


@pytest.mark.parametrize(
    ('recipe', 'is_context_quantized', 'expected'),
    [
        ({'override_quantized_autocast': False}, True, True),
        ({'override_nonquantized_autocast': False}, False, False),
        ({'override_nonquantized_autocast': True}, False, False),
        ({**_TENSORWISE, 'override_nonquantized_autocast': True}, False, True),
        ({**_TENSORWISE, 'override_quantized_autocast': True}, True, True),
    ],
    ids=['no-override-quantized', 'no-override-bf16', 'force-bf16', 'force-fp8', 'keep-fp8'],
)
def test_should_context_be_quantized_follows_recipe_overrides(
    recipe, is_context_quantized, expected
):
    assert (
        te_ext._get_should_context_be_quantized_recipe(
            TEQuantizationRecipe(**recipe), is_context_quantized
        )
        is expected
    )


@pytest.mark.parametrize(
    ('training', 'is_context_quantized', 'expected'),
    [(False, False, False), (False, True, False), (True, False, True), (True, True, True)],
    ids=['evaluation-bf16', 'evaluation-quantized', 'training-bf16', 'training-quantized'],
)
def test_should_context_be_quantized_selects_recipe_by_training_flag(
    training, is_context_quantized, expected
):
    params = _params(
        training={**_TENSORWISE, 'override_nonquantized_autocast': True},
        evaluation={'override_nonquantized_autocast': True},
    )
    assert (
        te_ext._get_should_context_be_quantized_params(params, training, is_context_quantized)
        is expected
    )


def test_te_delayed_scaling_copies_config_fields_into_recipe():
    config = ModelParallelConfig(
        fp8_margin=3,
        fp8_amax_compute_algo='max',
        fp8_amax_history_len=512,
        fp8_dot_product_attention=True,
        fp8_multi_head_attention=True,
        use_cpu_initialization=True,
    )
    recipe = TEDelayedScaling(
        config, te_ext.te.common.recipe.Format.E4M3, override_linear_precision=(False, False, True)
    )
    assert recipe.margin == 3
    assert recipe.amax_compute_algo == 'max'
    assert recipe.amax_history_len == 512
    assert recipe.override_linear_precision == (False, False, True)
    assert recipe.fp8_dpa is True
    assert recipe.fp8_mha is True


def test_te_delayed_scaling_warns_fp8_interval_is_ignored():
    config = ModelParallelConfig(fp8_interval=2, use_cpu_initialization=True)
    with pytest.warns(UserWarning, match='fp8_interval is deprecated and ignored'):
        recipe = TEDelayedScaling(config, te_ext.te.common.recipe.Format.E4M3)
    assert recipe.margin == 0
    assert recipe.amax_history_len == config.fp8_amax_history_len
