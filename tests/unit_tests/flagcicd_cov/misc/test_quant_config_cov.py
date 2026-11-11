# Copyright (c) 2026, BAAI. All rights reserved.
"""Quantization recipe parsing, matching, and repr output."""

import importlib
import sys

import pytest

import megatron.core.quantization.quant_config as quant_config

RECIPE_YAML = """\
configs:
  nvfp4:
    num_bits: 4
  mxfp8:
    num_bits: 8
matchers:
  fc1:
    config: "nvfp4"
    type: "glob"
    pattern: "*fc1*"
    enabled: true
  disabled:
    config: "nvfp4"
    type: "glob"
    pattern: "*never*"
    enabled: false
  fallback:
    config: "mxfp8"
    type: "glob"
    pattern: "*"
    enabled: true
"""


@pytest.fixture
def yaml_absent(monkeypatch):
    """Reload quant_config with the optional yaml import blocked."""
    monkeypatch.setitem(sys.modules, 'yaml', None)
    importlib.reload(quant_config)
    yield
    monkeypatch.undo()
    importlib.reload(quant_config)


def test_from_yaml_file_loads_enabled_matchers_and_configs(tmp_path):
    recipe_path = tmp_path / 'recipe.yaml'
    recipe_path.write_text(RECIPE_YAML)

    recipe = quant_config.RecipeConfig.from_yaml_file(str(recipe_path))

    assert [matcher.pattern for matcher in recipe.matchers] == ['*fc1*', '*']
    assert recipe.configs == {'nvfp4': {'num_bits': 4}, 'mxfp8': {'num_bits': 8}}

    matched = recipe.match(
        quant_config.MatchContext(module_path='decoder.layers.3.linear_fc1', layer_number=3)
    )
    assert matched.config == {'num_bits': 4}
    assert matched.config_key == 'nvfp4'
    assert matched.match_input.layer_number == 3

    fallback = recipe.match(
        quant_config.MatchContext(module_path='decoder.layers.3.linear_qkv', layer_number=3)
    )
    assert fallback.config == {'num_bits': 8}
    assert fallback.config_key == 'mxfp8'


def test_from_config_dict_returns_no_matchers_without_a_matchers_section():
    recipe = quant_config.RecipeConfig.from_config_dict({'configs': {'nvfp4': {'num_bits': 4}}})

    assert recipe.matchers == []
    assert recipe.configs == {'nvfp4': {'num_bits': 4}}


def test_from_config_dict_rejects_unknown_match_types():
    matchers = {'m': {'enabled': True, 'type': 'regex', 'pattern': '*'}}

    with pytest.raises(NotImplementedError, match="Match type 'regex' not implemented"):
        quant_config.RecipeConfig.from_config_dict({'matchers': matchers})


def test_base_matcher_returns_none_when_a_subclass_defers_to_it():
    class DelegatingMatcher(quant_config.Matcher):

        def match(self, context):
            return super().match(context)

    context = quant_config.MatchContext(module_path='decoder.linear_fc1', layer_number=1)

    assert DelegatingMatcher().match(context) is None


def test_recipe_without_a_matching_rule_returns_none():
    recipe = quant_config.RecipeConfig(
        [quant_config.GlobMatcher('*fc1*', 'nvfp4')], {'nvfp4': {'num_bits': 4}}
    )
    context = quant_config.MatchContext(module_path='decoder.linear_qkv', layer_number=1)

    assert recipe.match_to_config_key(context) is None
    assert recipe.match(context) is None


def test_glob_matcher_repr_exposes_pattern_and_config_key():
    matcher = quant_config.GlobMatcher('*fc1*', 'nvfp4')

    assert repr(matcher) == 'GlobMatcher(pattern=*fc1*, config_key=nvfp4)'


def test_quantization_config_repr_exposes_config_and_context():
    context = quant_config.MatchContext(module_path='a.b', layer_number=2)
    config = quant_config.QuantizationConfig({'num_bits': 4}, context, 'nvfp4')

    assert repr(config) == (
        "QuantizationConfig(config={'num_bits': 4}, "
        "match_input=MatchContext(module_path='a.b', layer_number=2), config_key=nvfp4)"
    )


def test_recipe_config_repr_lists_each_matcher():
    recipe = quant_config.RecipeConfig(
        [quant_config.GlobMatcher('*fc1*', 'nvfp4')], {'nvfp4': {'num_bits': 4}}
    )

    text = repr(recipe)

    assert text.startswith('RecipeConfig(\n')
    assert 'GlobMatcher(pattern=*fc1*, config_key=nvfp4)' in text
    assert text.endswith(')')


def test_from_yaml_file_requires_yaml_when_it_is_not_installed(yaml_absent):
    assert quant_config.HAVE_YAML is False

    with pytest.raises(ImportError, match='yaml is not installed. Please install it with'):
        quant_config.RecipeConfig.from_yaml_file('/nonexistent/recipe.yaml')
