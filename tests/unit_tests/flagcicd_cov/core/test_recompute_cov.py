# Copyright (c) 2026, BAAI. All rights reserved.
"""Activation-recompute chunking with stand-in layers and checkpoint functions."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core import recompute


class AttentionLayer(recompute.TransformerLayer):
    """A real TransformerLayer subtype (not initialised): adds `step`, passes the context through."""

    def __init__(self, layer_number, step):
        self.layer_number = layer_number
        self.step = step
        self.seen = None

    def __call__(self, **kwargs):
        self.seen = sorted(kwargs)
        return kwargs['hidden_states'] + self.step, kwargs['context']


class StateSpaceLayer:
    """Stands in for a non-attention layer: accepts fewer arguments, returns a tensor or a tuple."""

    def __init__(self, layer_number, step, as_tuple=False):
        self.layer_number = layer_number
        self.step = step
        self.as_tuple = as_tuple
        self.seen = None

    def __call__(self, **kwargs):
        self.seen = sorted(kwargs)
        output = kwargs['hidden_states'] + self.step
        return (output, 'extra') if self.as_tuple else output


@pytest.fixture
def calls(monkeypatch):
    """Replace checkpointing and quantization contexts with recorders that run the chunk inline."""
    record = SimpleNamespace(checkpoints=[], contexts=[])

    def plain_checkpoint(function, distribute, *args):
        record.checkpoints.append(('tensor_parallel', distribute))
        return function(*args)

    def engine_checkpoint(function, distribute, tracker, group, *args):
        record.checkpoints.append(('transformer_engine', distribute, tracker, group))
        return function(*args)

    @contextmanager
    def quantization(kind, layer_index):
        record.contexts.append((kind, layer_index))
        yield

    monkeypatch.setattr(
        recompute,
        'tensor_parallel',
        SimpleNamespace(
            checkpoint=plain_checkpoint, random=SimpleNamespace(get_cuda_rng_tracker='tracker')
        ),
    )
    monkeypatch.setattr(recompute, 'te_checkpoint', engine_checkpoint)
    monkeypatch.setattr(
        recompute, 'get_fp8_context', lambda config, index: quantization('fp8', index)
    )
    monkeypatch.setattr(
        recompute, 'get_fp4_context', lambda config, index: quantization('fp4', index)
    )
    return record


def make_stack(layers, method='uniform', recompute_num_layers=1, fp8=False, fp4=False):
    return SimpleNamespace(
        layers=layers,
        num_layers_per_pipeline_rank=len(layers),
        pg_collection=SimpleNamespace(tp='tp-group'),
        config=SimpleNamespace(
            fp8=fp8,
            fp4=fp4,
            recompute_method=method,
            recompute_num_layers=recompute_num_layers,
            distribute_saved_activations=False,
        ),
    )


def forward(stack, hidden_states=None, use_inner_quantization_context=False, **kwargs):
    if hidden_states is None:
        hidden_states = torch.zeros(2)
    return recompute.checkpointed_forward(
        stack,
        hidden_states,
        'mask',
        'context',
        'context-mask',
        'rope',
        'bias',
        'packed',
        use_inner_quantization_context,
        **kwargs,
    )


def attention_layers():
    return [AttentionLayer(1, 1.0), AttentionLayer(2, 10.0), AttentionLayer(3, 100.0)]


def test_uniform_checkpoints_each_chunk_and_passes_the_full_layer_arguments(calls):
    layers = attention_layers()
    output = forward(make_stack(layers, recompute_num_layers=2))
    assert output.tolist() == [111.0, 111.0]
    assert calls.checkpoints == [('tensor_parallel', False)] * 2
    assert layers[0].seen == [
        'attention_bias',
        'attention_mask',
        'context',
        'context_mask',
        'hidden_states',
        'inference_context',
        'input_ids',
        'packed_seq_params',
        'padding_mask',
        'rotary_pos_emb',
    ]


def test_uniform_extracts_features_at_chunk_ends_using_the_layer_offset(calls):
    output, features = forward(
        make_stack(attention_layers(), recompute_num_layers=2),
        extract_layer_indices={6, 7},
        layer_offset=5,
    )
    assert output.tolist() == [111.0, 111.0]
    assert [feature.tolist() for feature in features] == [[11.0, 11.0], [111.0, 111.0]]


def test_block_checkpoints_only_the_first_layers_and_extracts_at_chunk_starts(calls):
    output, features = forward(
        make_stack(attention_layers(), 'block', 1), extract_layer_indices={0, 2}
    )
    assert output.tolist() == [111.0, 111.0]
    assert [feature.tolist() for feature in features] == [[1.0, 1.0], [111.0, 111.0]]
    assert calls.checkpoints == [('tensor_parallel', False)]


def test_non_attention_layers_receive_a_reduced_argument_set(calls):
    layers = [
        AttentionLayer(1, 1.0),
        StateSpaceLayer(2, 10.0),
        StateSpaceLayer(3, 100.0, as_tuple=True),
    ]
    output = forward(make_stack(layers, recompute_num_layers=3))
    assert output.tolist() == [111.0, 111.0]
    assert layers[1].seen == [
        'attention_mask',
        'hidden_states',
        'inference_context',
        'packed_seq_params',
        'rotary_pos_emb',
    ]
    assert layers[2].seen == [
        'attention_mask',
        'hidden_states',
        'inference_context',
        'packed_seq_params',
        'rotary_pos_emb',
    ]


@pytest.mark.parametrize(
    ('precision', 'contexts'),
    [
        ({'fp8': True}, [('fp8', 0), ('fp8', 1)]),
        ({'fp4': True}, [('fp4', 0), ('fp4', 1)]),
        ({}, []),
    ],
)
def test_inner_quantization_context_follows_the_precision(calls, precision, contexts):
    layers = [AttentionLayer(1, 1.0), AttentionLayer(2, 1.0)]
    output = forward(
        make_stack(layers, recompute_num_layers=2, **precision), use_inner_quantization_context=True
    )
    assert output.tolist() == [2.0, 2.0]
    assert calls.contexts == contexts


def test_quantized_uniform_uses_the_engine_checkpoint_with_the_tp_group(calls):
    output = forward(make_stack([AttentionLayer(1, 1.0)], fp8=True))
    assert output.tolist() == [1.0, 1.0]
    assert calls.checkpoints == [('transformer_engine', False, 'tracker', 'tp-group')]


def test_quantized_block_skips_checkpointing_until_the_input_requires_grad(calls):
    stack = make_stack([AttentionLayer(1, 1.0), AttentionLayer(2, 1.0)], 'block', 1, fp8=True)
    output = forward(stack)
    assert output.tolist() == [2.0, 2.0]
    assert calls.checkpoints == []
    output = forward(stack, hidden_states=torch.zeros(2, requires_grad=True))
    assert output.tolist() == [2.0, 2.0]
    assert calls.checkpoints == [('transformer_engine', False, 'tracker', 'tp-group')]


def test_unknown_recompute_method_is_rejected(calls):
    with pytest.raises(ValueError, match='Invalid activation recompute method.'):
        forward(make_stack([AttentionLayer(1, 1.0)], 'bogus'))
