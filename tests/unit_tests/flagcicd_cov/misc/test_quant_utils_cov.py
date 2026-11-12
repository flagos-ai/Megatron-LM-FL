# Copyright (c) 2026, BAAI. All rights reserved.
"""Quantization recipe resolution for layer module paths and YAML recipe loading."""

from megatron.core.quantization.quant_config import MatchContext
from megatron.core.quantization.utils import (
    get_quant_config_or_none,
    kitchen_quantization_recipe_config,
    load_quantization_recipe,
)

RECIPE_YAML = """\
configs:
  nvfp4:
    num_bits: 4
matchers:
  fc1:
    config: "nvfp4"
    type: "glob"
    pattern: "*fc1*"
    enabled: true
"""


def test_get_quant_config_or_none_returns_none_when_the_recipe_is_missing():
    assert get_quant_config_or_none('decoder.layers.0.linear_fc1', None) is None


def test_get_quant_config_or_none_returns_none_when_the_module_path_is_missing():
    recipe = kitchen_quantization_recipe_config(0)

    assert get_quant_config_or_none(None, recipe) is None


def test_get_quant_config_or_none_matches_a_layer_module_path_with_its_layer_number():
    recipe = kitchen_quantization_recipe_config(2)

    matched = get_quant_config_or_none('decoder.layers.7.linear_fc1', recipe)

    assert matched.config == {'kitchen_config_type': 'QLinearParams', 'recipe_idx': 2}
    assert matched.config_key == 'default'
    assert matched.match_input == MatchContext(
        module_path='decoder.layers.7.linear_fc1', layer_number=7
    )


def test_get_quant_config_or_none_matches_a_module_path_without_a_layer_number():
    recipe = kitchen_quantization_recipe_config(2)

    matched = get_quant_config_or_none('decoder.embedding.word_embeddings', recipe)

    assert matched.config == {'kitchen_config_type': 'QLinearParams', 'recipe_idx': 2}
    assert matched.match_input == MatchContext(
        module_path='decoder.embedding.word_embeddings', layer_number=None
    )


def test_load_quantization_recipe_reads_a_yaml_recipe(tmp_path):
    recipe_path = tmp_path / 'recipe.yaml'
    recipe_path.write_text(RECIPE_YAML)

    recipe = load_quantization_recipe(str(recipe_path))

    matched = recipe.match(MatchContext(module_path='decoder.layers.1.linear_fc1', layer_number=1))
    assert matched.config == {'num_bits': 4}
    assert matched.config_key == 'nvfp4'
