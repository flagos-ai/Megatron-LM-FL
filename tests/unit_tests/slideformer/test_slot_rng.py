# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint recomputation must preserve MCore's parallel dropout streams."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import (
    _get_all_rng_states,
    _set_all_rng_states,
    get_cuda_rng_tracker,
)
from megatron.plugin.slideformer.checkpoint import slideformer_slot_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint")
@pytest.mark.parametrize("gpu_window", [0, 1])
def test_slot_checkpoint_preserves_parallel_dropout_rng(gpu_window):
    tracker = get_cuda_rng_tracker()
    initial = _get_all_rng_states()
    name = "slide-checkpoint-test"
    try:
        tracker.add(name, 92831)
        x = torch.randn(8, 32, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_()

        def function(value):
            value = F.dropout(value, p=0.25, training=True)
            with tracker.fork(name):
                value = F.dropout(value, p=0.4, training=True)
            return value * torch.rand(()).item()

        before = _get_all_rng_states()
        expected = function(x)
        expected.square().sum().backward()
        after = _get_all_rng_states()
        _set_all_rng_states(*before)
        actual = slideformer_slot_checkpoint(
            function,
            layer_idx=0,
            device=x.device,
            pin_memory=True,
            stream=torch.cuda.Stream(),
            cpu_slots={},
            copy_events={},
            gpu_resident_slots={},
            prefetch_slots={},
            prefetch_events={},
            traffic_counters={},
            lifecycle_records=None,
            prefetch_previous=False,
            total_layers=1,
            gpu_window=gpu_window,
            h2d_scheduler=None,
            args=(y,),
        )
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(y.grad, x.grad, atol=0, rtol=0)
        final = _get_all_rng_states()
        for got, want in zip(final[:2], after[:2]):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        assert final[2].keys() == after[2].keys()
        for key in final[2]:
            torch.testing.assert_close(final[2][key], after[2][key], atol=0, rtol=0)
    finally:
        _set_all_rng_states(*initial)
        tracker.seeds_.discard(92831)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint")
@pytest.mark.parametrize("gpu_window", [0, 1])
@pytest.mark.parametrize("prefetch", [False, True])
def test_slot_checkpoint_keeps_outstanding_microbatches_separate(gpu_window, prefetch):
    torch.manual_seed(314)
    inputs = [torch.randn(4, 16, device="cuda", requires_grad=True) for _ in range(2)]
    references = [value.detach().clone().requires_grad_() for value in inputs]
    weights = [torch.randn(16, 16, device="cuda") for _ in range(2)]
    slots, events, resident, prefetched, prefetch_events = {}, {}, {}, {}, {}
    stream = torch.cuda.Stream()
    outputs, expected = [], []
    for value, reference in zip(inputs, references):
        for layer, weight in enumerate(weights):
            def function(hidden, weight=weight):
                return F.silu(hidden @ weight)

            reference = function(reference)
            value = slideformer_slot_checkpoint(
                function, layer_idx=layer, device=value.device, pin_memory=True,
                stream=stream, cpu_slots=slots, copy_events=events,
                gpu_resident_slots=resident, prefetch_slots=prefetched,
                prefetch_events=prefetch_events, traffic_counters={},
                lifecycle_records=None, prefetch_previous=prefetch,
                total_layers=2, gpu_window=gpu_window, h2d_scheduler=None, args=(value,),
            )
        outputs.append(value)
        expected.append(reference)
    # A pipeline warmup has queued both forwards before the first backward.
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        actual.square().sum().backward()
        reference.square().sum().backward()
    for actual, reference in zip(inputs, references):
        torch.testing.assert_close(actual.grad, reference.grad, atol=0, rtol=0)
