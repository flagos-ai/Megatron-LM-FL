# Copyright (c) 2026, BAAI. All rights reserved.
"""Streaming parsers: DeepSeek R1 think-tag extraction and Qwen3 Coder XML tool calls."""

import json

import pytest

from megatron.core.tokenizers.text.parsers import PARSER_MAPPING
from megatron.core.tokenizers.text.parsers.base_parser import BaseParser
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser import Qwen3CoderToolParser

THINK_OPEN = '<' + 'think' + '>'
THINK_CLOSE = '</' + 'think' + '>'


def test_base_parser_returns_input_text_and_empty_info():
    assert BaseParser.parse('hello world') == ('hello world', {})


def test_parser_mapping_registers_the_available_parsers():
    assert PARSER_MAPPING['deepseek-r1-reasoning'] is DeepSeekR1ReasoningParser
    assert PARSER_MAPPING['qwen3-coder-tool'] is Qwen3CoderToolParser


def test_deepseek_parser_extracts_reasoning_between_think_tags():
    text = 'before ' + THINK_OPEN + 'reasoning' + THINK_CLOSE + ' after'
    assert DeepSeekR1ReasoningParser.parse(text) == ('before  after', {'reasoning': 'reasoning'})


def test_deepseek_parser_infers_opening_tag_when_only_closing_tag_is_present():
    text = 'reasoning' + THINK_CLOSE + ' after'
    assert DeepSeekR1ReasoningParser.parse(text) == (' after', {'reasoning': 'reasoning'})


def test_deepseek_parser_returns_text_unchanged_without_closing_tag():
    text = 'answer ' + THINK_OPEN + 'partial reasoning'
    assert DeepSeekR1ReasoningParser.parse(text) == (text, {})


def _tool_call_text(name, parameters):
    rendered = ''.join(f'\n<parameter={param}>{value}</parameter>' for param, value in parameters)
    return f'calling now. <function={name}>{rendered}\n</function>\n</function>'


def _weather_tools(properties):
    return [
        {
            'type': 'function',
            'function': {'name': 'get_weather', 'parameters': {'properties': properties}},
        }
    ]


def test_qwen_parser_returns_text_unchanged_without_function_prefix():
    text = 'just a plain answer'
    assert Qwen3CoderToolParser.parse(text, tools=[]) == (text, {})


def test_qwen_parser_returns_text_unchanged_for_bare_function_prefix():
    text = 'answer <function='
    assert Qwen3CoderToolParser.parse(text, tools=[]) == (text, {})


def test_qwen_parser_extracts_tool_call_with_name_and_typed_arguments():
    text = _tool_call_text('get_weather', [('city', 'Tokyo'), ('unit', 'celsius'), ('days', 3)])
    content, info = Qwen3CoderToolParser.parse(
        text,
        tools=_weather_tools(
            {'city': {'type': 'string'}, 'unit': {'type': 'string'}, 'days': {'type': 'integer'}}
        ),
    )
    assert content == 'calling now. '
    tool_calls = info['tool_calls']
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call['type'] == 'function'
    assert call['function']['name'] == 'get_weather'
    assert json.loads(call['function']['arguments']) == {
        'city': 'Tokyo',
        'unit': 'celsius',
        'days': 3,
    }
    assert call['id'].startswith('call_') and len(call['id']) == len('call_') + 24


def test_qwen_parser_extracts_multiple_tool_calls_with_unique_ids():
    text = _tool_call_text('get_weather', [('city', 'Tokyo')]) + _tool_call_text(
        'get_time', [('zone', 'UTC')]
    )
    content, info = Qwen3CoderToolParser.parse(text, tools=_weather_tools({}))
    assert content == 'calling now. '
    calls = info['tool_calls']
    assert [call['function']['name'] for call in calls] == ['get_weather', 'get_time']
    assert calls[0]['id'] != calls[1]['id']


def test_qwen_parser_extracts_tool_call_without_wrapper_tags():
    text = 'calling now. <function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>'
    content, info = Qwen3CoderToolParser.parse(
        text, tools=_weather_tools({'city': {'type': 'string'}})
    )
    assert content == 'calling now. '
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {'city': 'Tokyo'}


def test_qwen_parser_returns_original_text_for_partial_streaming_call():
    text = 'partial <function=get_weather><parameter=city'
    assert Qwen3CoderToolParser.parse(text, tools=_weather_tools({})) == (text, {})


def test_qwen_parser_keeps_newlines_inside_multiline_parameter_values():
    text = (
        'calling now. <function=get_weather>\n<parameter=report>\nline one\nline two\n'
        '</parameter>\n</function>\n</function>'
    )
    _, info = Qwen3CoderToolParser.parse(text, tools=_weather_tools({}))
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {
        'report': 'line one\nline two'
    }


@pytest.mark.parametrize(
    'spec,value,expected',
    [
        ({'type': 'string'}, 'Tokyo', 'Tokyo'),
        ({'type': 'str'}, 'Tokyo', 'Tokyo'),
        ({'type': 'integer'}, '42', 42),
        ({'type': 'uint8'}, '42', 42),
        ({'type': 'integer'}, 'not-a-number', 'not-a-number'),
        ({'type': 'number'}, '3.5', 3.5),
        ({'type': 'number'}, '4.0', 4),
        ({'type': 'float'}, 'not-a-number', 'not-a-number'),
        ({'type': 'boolean'}, 'true', True),
        ({'type': 'boolean'}, 'false', False),
        ({'type': 'boolean'}, 'maybe', False),
        ({'type': 'object'}, '{"a": 1}', {'a': 1}),
        ({'type': 'array'}, '[1, 2]', [1, 2]),
        ({'anyOf': [{'type': 'object'}]}, '{"a": 1}', {'a': 1}),
        ({'type': 'array'}, '(1, 2)', [1, 2]),
        ({'type': 'array'}, 'not python!', 'not python!'),
    ],
    ids=[
        'string',
        'str-alias',
        'integer',
        'uint8',
        'integer-degenerates',
        'number-fractional',
        'number-integral',
        'float-degenerates',
        'boolean-true',
        'boolean-false',
        'boolean-degenerates',
        'object',
        'array-json',
        'any-of',
        'array-python-literal',
        'array-degenerates',
    ],
)
def test_qwen_parser_converts_parameter_values_to_the_declared_type(spec, value, expected):
    text = _tool_call_text('get_weather', [('p', value)])
    _, info = Qwen3CoderToolParser.parse(text, tools=_weather_tools({'p': spec}))
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {'p': expected}


@pytest.mark.parametrize(
    'schema,value,expected',
    [('integer', 'null', None), ('integer', '42', '42')],
    ids=['null', 'missing-from-schema'],
)
def test_qwen_parser_returns_string_or_none_when_parameter_is_undeclared(schema, value, expected):
    text = _tool_call_text('get_weather', [('p', value)])
    _, info = Qwen3CoderToolParser.parse(text, tools=_weather_tools({'other': {'type': schema}}))
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {'p': expected}


@pytest.mark.parametrize(
    'tools',
    [
        None,
        ['not-a-dict'],
        [{'function': 'not-a-dict'}],
        [{'type': 'other', 'function': {'name': 'get_weather', 'parameters': {}}}],
        [{'type': 'function', 'function': {'name': 'other', 'parameters': {}}}],
        [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': 'raw'}}],
    ],
    ids=[
        'no-tools',
        'config-not-dict',
        'function-not-dict',
        'type-mismatch',
        'name-mismatch',
        'parameters-not-dict',
    ],
)
def test_qwen_parser_leaves_values_as_strings_when_no_schema_matches(tools):
    text = _tool_call_text('get_weather', [('p', '42')])
    _, info = Qwen3CoderToolParser.parse(text, tools=tools)
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {'p': '42'}


def test_qwen_parser_uses_parameters_directly_when_properties_are_missing():
    text = _tool_call_text('get_weather', [('p', '42')])
    _, info = Qwen3CoderToolParser.parse(
        text,
        tools=[
            {
                'type': 'function',
                'function': {'name': 'get_weather', 'parameters': {'p': {'type': 'integer'}}},
            }
        ],
    )
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {'p': 42}


def test_qwen_parser_treats_non_dict_parameter_specs_as_strings():
    text = _tool_call_text('get_weather', [('p', '42')])
    _, info = Qwen3CoderToolParser.parse(
        text,
        tools=[
            {
                'type': 'function',
                'function': {'name': 'get_weather', 'parameters': {'p': 'integer'}},
            }
        ],
    )
    assert json.loads(info['tool_calls'][0]['function']['arguments']) == {'p': '42'}
