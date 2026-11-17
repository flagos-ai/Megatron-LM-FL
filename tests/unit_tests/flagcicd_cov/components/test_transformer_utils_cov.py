# Copyright (c) 2026, BAAI. All rights reserved.
"""Transformer utilities: sliding-window causal masks, GELU variants, checkpoint sharded-object wrapping, sequence-parallel and CUDA-graph attribute caches, and MoE cudagraph scope transitions."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import megatron.core.transformer.utils as transformer_utils
from megatron.core import parallel_state
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from megatron.core.transformer.utils import (
    erf_gelu,
    get_sliding_window_causal_mask,
    make_sharded_object_for_checkpoint,
    openai_gelu,
    transition_moe_cudagraphs,
)


@pytest.fixture
def single_rank_parallel_state(tmp_path):
    dist.init_process_group(
        backend='gloo', init_method=f'file://{tmp_path}/rendezvous', world_size=1, rank=0
    )
    parallel_state.initialize_model_parallel(1, 1)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_get_sliding_window_causal_mask_marks_positions_outside_band():
    sq, skv = 4, 6
    window_size = (2, 1)
    mask = get_sliding_window_causal_mask(sq, skv, window_size)
    expected = torch.zeros(sq, skv, dtype=torch.bool, device=mask.device)
    for i in range(sq):
        for j in range(skv):
            band_start = skv - sq + i - window_size[0]
            band_stop = skv - sq + i + window_size[1]
            expected[i, j] = not (band_start <= j <= band_stop)
    assert torch.equal(mask, expected)


def test_openai_gelu_matches_tanh_formula():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    torch.testing.assert_close(openai_gelu(x), expected)


def test_erf_gelu_matches_error_function_formula():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = x * 0.5 * (torch.erf(x / 1.41421) + 1.0)
    torch.testing.assert_close(erf_gelu(x), expected)


def test_make_sharded_object_defaults_replica_to_parallel_ranks(single_rank_parallel_state):
    obj = object()
    sharded = make_sharded_object_for_checkpoint(obj, 'extra_state')
    assert sharded.data is obj
    assert sharded.replica_id == (0, 0, 0)
    assert sharded.global_shape == (1,)
    assert sharded.global_offset == (0,)


def test_make_sharded_object_turns_prepended_offsets_into_global_shape(single_rank_parallel_state):
    sharded = make_sharded_object_for_checkpoint(
        object(), 'extra_state', sharded_offsets=[(0, 0, 3)]
    )
    assert sharded.global_shape == (3,)
    assert sharded.global_offset == (0,)


def _sequence_parallel_model(position_embedding_type):
    model = nn.Module()
    model.config = SimpleNamespace(sequence_parallel=False)
    model.position_embedding_type = position_embedding_type
    linear = nn.Linear(2, 2)
    linear.sequence_parallel = False
    layer_norm = nn.LayerNorm(2)
    layer_norm.scatter_to_sequence_parallel = False
    embedding = nn.Module()
    embedding.reduce_scatter_embeddings = False
    model.linear = linear
    model.layer_norm = layer_norm
    model.embedding = embedding
    return model


def test_init_sequence_parallel_cache_keeps_all_attributes_for_absolute_embeddings(monkeypatch):
    monkeypatch.setattr(transformer_utils, '_sequence_parallel_attr_cache', None)
    model = _sequence_parallel_model('absolute')
    transformer_utils.set_model_to_sequence_parallel(model, set_to=True)
    assert model.linear.sequence_parallel is True
    assert model.layer_norm.scatter_to_sequence_parallel is True
    assert model.embedding.reduce_scatter_embeddings is True
    assert model.config.sequence_parallel is True
    cache = transformer_utils._sequence_parallel_attr_cache
    assert cache[id(model)]['sequence_parallel'] == [model.linear]
    assert cache[id(model)]['scatter_to_sequence_parallel'] == [model.layer_norm]
    assert cache[id(model)]['reduce_scatter_embeddings'] == [model.embedding]


def test_init_sequence_parallel_cache_drops_embedding_attr_for_learned_absolute(monkeypatch):
    monkeypatch.setattr(transformer_utils, '_sequence_parallel_attr_cache', None)
    model = _sequence_parallel_model('learned_absolute')
    transformer_utils.set_model_to_sequence_parallel(model, set_to=False)
    cache = transformer_utils._sequence_parallel_attr_cache
    assert sorted(cache[id(model)]) == ['scatter_to_sequence_parallel', 'sequence_parallel']
    assert model.embedding.reduce_scatter_embeddings is False


def test_init_sequence_parallel_cache_returns_early_for_cached_model(monkeypatch):
    monkeypatch.setattr(transformer_utils, '_sequence_parallel_attr_cache', None)
    model = _sequence_parallel_model('absolute')
    transformer_utils.set_model_to_sequence_parallel(model, set_to=True)
    cache = transformer_utils._sequence_parallel_attr_cache
    transformer_utils._init_sequence_parallel_cache(model, None)
    assert transformer_utils._sequence_parallel_attr_cache is cache


def _cuda_graph_model():
    model = nn.Module()
    model.config = SimpleNamespace(cuda_graph_impl='none', recompute_granularity='full')
    attention = nn.Module()
    attention.cuda_graph_impl = True
    attention.flash_decode = False
    block = nn.Module()
    block.config = SimpleNamespace(cuda_graph_impl=False, flash_decode=True)
    manager_host = nn.Module()
    manager_host.cudagraph_manager = 'manager-handle'
    manager_host.recompute_granularity = 'full'
    model.attention = attention
    model.block = block
    model.manager_host = manager_host
    return model


def test_toggle_cuda_graphs_switches_cached_attributes(monkeypatch):
    monkeypatch.setattr(transformer_utils, 'cuda_graph_attr_cache', None)
    model = _cuda_graph_model()
    transformer_utils.toggle_cuda_graphs(model, set_to='local')
    assert model.config.cuda_graph_impl == 'local'
    assert model.attention.cuda_graph_impl == 'local'
    assert model.block.config.cuda_graph_impl == 'local'
    assert model.manager_host.recompute_granularity is None
    assert model.manager_host.cudagraph_manager == 'manager-handle'
    transformer_utils.toggle_cuda_graphs(model, set_to='none')
    assert model.config.cuda_graph_impl == 'none'
    assert model.manager_host.recompute_granularity == 'full'
    assert hasattr(model.manager_host, 'cudagraph_manager') is False


def test_toggle_cuda_graphs_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(transformer_utils, 'cuda_graph_attr_cache', None)
    model = _cuda_graph_model()
    with pytest.raises(AssertionError, match='Invalid CUDA graph implementation: bogus'):
        transformer_utils.toggle_cuda_graphs(model, set_to='bogus')


def test_init_cuda_graph_cache_returns_early_for_cached_model(monkeypatch):
    monkeypatch.setattr(transformer_utils, 'cuda_graph_attr_cache', None)
    model = _cuda_graph_model()
    transformer_utils.init_cuda_graph_cache(model)
    cache = transformer_utils.cuda_graph_attr_cache
    transformer_utils.init_cuda_graph_cache(model)
    assert transformer_utils.cuda_graph_attr_cache is cache


class _FullScopeMoELayer(MoETransformerLayer):
    """MoE layer whose real transition_cudagraph_scope('full') runs without CUDA graph managers."""

    def __init__(self):
        nn.Module.__init__(self)
        self.use_partial_cudagraphs = True
        self.mlp = SimpleNamespace(fwd_execution_map=['unused'])
        self.cudagraph_manager = object()


def test_transition_moe_cudagraphs_switches_only_moe_layers():
    model = nn.Module()
    model.moe = _FullScopeMoELayer()
    model.linear = nn.Linear(2, 2)
    transition_moe_cudagraphs(model, 'full')
    assert model.moe.use_partial_cudagraphs is False
    assert model.moe.mlp.fwd_execution_map == ['route', 'expert_compute', 'postprocess']
    assert not hasattr(model.linear, 'use_partial_cudagraphs')
