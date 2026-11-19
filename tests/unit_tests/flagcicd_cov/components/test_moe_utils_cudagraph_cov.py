# Copyright (c) 2026, BAAI. All rights reserved.
"""MoE CUDA-graph control flow: partial-capture signal payloads, tensor-store field rules, and decorator skip, replay, and capture-signal behaviour."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.moe_utils as moe_utils
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    maybe_skip_or_early_return_by_cudagraph,
)


def test_partial_capture_signal_stores_layer_step_and_kwargs():
    layer = object()
    probs = torch.zeros(2)
    signal = MoECudaGraphPartialCaptureSignal(layer, 'route', probs=probs)
    assert signal.moe_layer is layer
    assert signal.return_step == 'route'
    assert signal.kwargs == {'probs': probs}


def test_get_early_return_outputs_route_returns_intermediates():
    probs = torch.zeros(2)
    routing_map = torch.ones(2, dtype=torch.bool)
    signal = MoECudaGraphPartialCaptureSignal(
        object(), 'route', probs=probs, routing_map=routing_map
    )
    hidden_states = torch.zeros(3)
    shared_expert_output = torch.zeros(1)
    outputs = signal.get_early_return_outputs(hidden_states, shared_expert_output)
    assert outputs == [hidden_states, probs, routing_map, shared_expert_output]


def test_get_early_return_outputs_preprocess_collects_dispatcher_tensors():
    permutation = torch.zeros(2)
    dispatcher = SimpleNamespace(
        cudagraph_attrs=['reversible_layout.permutation', 'missing_attr'],
        valid_cudagraph_attrs=None,
        reversible_layout=SimpleNamespace(permutation=permutation),
    )
    layer = SimpleNamespace(token_dispatcher=dispatcher)
    hidden_states = torch.zeros(3)
    probs = torch.zeros(2)
    signal = MoECudaGraphPartialCaptureSignal(
        layer, 'preprocess', hidden_states=hidden_states, probs=probs
    )
    outputs = signal.get_early_return_outputs(hidden_states, None)
    assert outputs == [hidden_states, probs, permutation]
    assert dispatcher.valid_cudagraph_attrs == ['reversible_layout.permutation']


def test_get_early_return_outputs_preprocess_validates_cached_attr_list():
    dispatcher = SimpleNamespace(cudagraph_attrs=['other'], valid_cudagraph_attrs=['first'])
    layer = SimpleNamespace(token_dispatcher=dispatcher)
    signal = MoECudaGraphPartialCaptureSignal(
        layer, 'preprocess', hidden_states=torch.zeros(1), probs=torch.zeros(1)
    )
    with pytest.raises(AssertionError, match='valid_cudagraph_attrs mismatch'):
        signal.get_early_return_outputs(torch.zeros(1), None)


def test_tensor_store_starts_empty_and_reports_set_values():
    store = MoECudaGraphTensorStore()
    assert store.is_empty() is True
    hidden_states = torch.zeros(2)
    store.set(hidden_states=hidden_states)
    assert store.is_empty() is False
    assert store.hidden_states is hidden_states


def test_tensor_store_set_ignores_none_values():
    store = MoECudaGraphTensorStore()
    probs = torch.ones(2)
    store.set(probs=probs, routing_map=None)
    assert store.probs is probs
    assert store.routing_map is None


def test_tensor_store_set_rejects_unknown_field():
    store = MoECudaGraphTensorStore()
    with pytest.raises(AssertionError, match='Invalid field name: bogus'):
        store.set(bogus=torch.zeros(1))


def test_tensor_store_set_rejects_non_tensor_values():
    store = MoECudaGraphTensorStore()
    with pytest.raises(AssertionError, match='Value must be a torch.Tensor'):
        store.set(probs=3)


def test_tensor_store_clear_resets_all_fields():
    store = MoECudaGraphTensorStore()
    store.set(
        hidden_states=torch.zeros(1),
        probs=torch.ones(1),
        routing_map=torch.ones(1, dtype=torch.bool),
        shared_expert_output=torch.zeros(1),
    )
    store.clear()
    assert store.hidden_states is None
    assert store.probs is None
    assert store.routing_map is None
    assert store.shared_expert_output is None


def _moe_layer(
    cuda_graph_modules, tensor_store, cuda_graph_impl='transformer_engine', training=True
):
    return SimpleNamespace(
        config=SimpleNamespace(
            cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
        ),
        training=training,
        cudagraph_tensor_store=tensor_store,
    )


def test_cudagraph_decorator_passes_through_when_impl_disabled():
    layer = _moe_layer([], MoECudaGraphTensorStore(), cuda_graph_impl='none')
    calls = []

    @maybe_skip_or_early_return_by_cudagraph('route')
    def route(_layer, x):
        calls.append(x)
        return x + 1

    assert route(layer, 4) == 5
    assert calls == [4]


def test_cudagraph_decorator_runs_function_when_not_capturing():
    layer = _moe_layer([], MoECudaGraphTensorStore())

    @maybe_skip_or_early_return_by_cudagraph('shared_experts_compute')
    def compute(_layer, x):
        return x * 2

    assert compute(layer, 21) == 42


def test_cudagraph_decorator_skips_shared_experts_when_store_filled():
    store = MoECudaGraphTensorStore()
    shared_expert_output = torch.zeros(2)
    store.set(shared_expert_output=shared_expert_output)
    layer = _moe_layer([CudaGraphModule.moe_router], store)

    @maybe_skip_or_early_return_by_cudagraph('shared_experts_compute')
    def compute(_layer):
        raise AssertionError('shared experts computation must be skipped')

    assert compute(layer) is shared_expert_output


def test_cudagraph_decorator_computes_shared_experts_when_store_has_other_values():
    store = MoECudaGraphTensorStore()
    store.set(probs=torch.ones(1))
    layer = _moe_layer([CudaGraphModule.moe_router], store)

    @maybe_skip_or_early_return_by_cudagraph('shared_experts_compute')
    def compute(_layer, x):
        return x + 1

    assert compute(layer, 1) == 2


def test_cudagraph_decorator_skips_route_when_store_filled():
    store = MoECudaGraphTensorStore()
    probs = torch.full((2,), 0.5)
    routing_map = torch.ones(2, dtype=torch.bool)
    store.set(probs=probs, routing_map=routing_map)
    layer = _moe_layer([CudaGraphModule.moe_router], store)

    @maybe_skip_or_early_return_by_cudagraph('route')
    def route(_layer):
        raise AssertionError('router must be skipped')

    assert route(layer) == (probs, routing_map)


def test_cudagraph_decorator_raises_capture_signal_after_route(monkeypatch):
    # CUDA graph capture cannot run on CPU, so the capture-state probe is pinned to capturing.
    monkeypatch.setattr(moe_utils, 'is_graph_capturing', lambda: True)
    layer = _moe_layer([CudaGraphModule.moe_router], MoECudaGraphTensorStore())
    probs = torch.full((2,), 0.25)
    routing_map = torch.ones(2, dtype=torch.bool)

    @maybe_skip_or_early_return_by_cudagraph('route')
    def route(_layer):
        return probs, routing_map

    with pytest.raises(MoECudaGraphPartialCaptureSignal) as caught:
        route(layer)
    assert caught.value.return_step == 'route'
    assert caught.value.kwargs == {'probs': probs, 'routing_map': routing_map}


def test_cudagraph_decorator_raises_capture_signal_after_preprocess(monkeypatch):
    # CUDA graph capture cannot run on CPU, so the capture-state probe is pinned to capturing.
    monkeypatch.setattr(moe_utils, 'is_graph_capturing', lambda: True)
    layer = _moe_layer([CudaGraphModule.moe_preprocess], MoECudaGraphTensorStore())
    hidden_states = torch.zeros(2)
    probs = torch.zeros(2)

    @maybe_skip_or_early_return_by_cudagraph('preprocess')
    def preprocess(_layer):
        return hidden_states, probs

    with pytest.raises(MoECudaGraphPartialCaptureSignal) as caught:
        preprocess(layer)
    assert caught.value.return_step == 'preprocess'
    assert caught.value.kwargs == {'hidden_states': hidden_states, 'probs': probs}


def test_cudagraph_decorator_skips_preprocess_when_store_filled():
    store = MoECudaGraphTensorStore()
    hidden_states = torch.zeros(2)
    probs = torch.ones(2)
    store.set(hidden_states=hidden_states, probs=probs)
    layer = _moe_layer([CudaGraphModule.moe_preprocess], store)

    @maybe_skip_or_early_return_by_cudagraph('preprocess')
    def preprocess(_layer):
        raise AssertionError('preprocess must be skipped')

    assert preprocess(layer) == (hidden_states, probs)


def test_cudagraph_decorator_runs_preprocess_when_routing_map_present():
    store = MoECudaGraphTensorStore()
    store.set(
        hidden_states=torch.zeros(1),
        probs=torch.zeros(1),
        routing_map=torch.ones(1, dtype=torch.bool),
    )
    layer = _moe_layer([], store)

    @maybe_skip_or_early_return_by_cudagraph('preprocess')
    def preprocess(_layer, x):
        return x - 1, x

    hidden_states, probs = preprocess(layer, 5)
    assert hidden_states == 4
    assert probs == 5
