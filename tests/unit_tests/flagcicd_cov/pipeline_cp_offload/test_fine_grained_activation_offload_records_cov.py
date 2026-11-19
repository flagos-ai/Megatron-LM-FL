# Copyright (c) 2026, BAAI. All rights reserved.
"""Group commit wraps only the first tensor of a container; record events tie the compute stream to the transfer streams; the manager context is a no-op before any chunk is initialized."""

import contextlib

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    PipelineOffloadManager,
    fine_grained_offloading_backward_record,
    fine_grained_offloading_forward_record,
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
)


@pytest.fixture
def offload_manager():
    PipelineOffloadManager.reset_instance()
    yield PipelineOffloadManager.get_instance()
    PipelineOffloadManager.reset_instance()


@pytest.mark.parametrize("container_type", [tuple, list], ids=["tuple", "list"])
def test_group_commit_wraps_only_first_tensor_of_container(offload_manager, container_type):
    off_interface.init_chunk_handler(1, 0, 8)
    chunk = offload_manager.cur_forward_chunk()
    first = torch.randn(2, 4, device="cuda")
    second = torch.randn(2, 4, device="cuda")
    started = fine_grained_offloading_group_start(first, "container_group")
    chunk.tensor_push(first)
    committed = fine_grained_offloading_group_commit(
        container_type([started, second]), "container_group"
    )
    assert isinstance(committed, container_type)
    assert len(committed) == 2
    assert torch.equal(committed[0], first)
    assert committed[1] is second
    assert chunk.offload_groups[0].total_offload_bytes == 2 * 4 * 4


@pytest.mark.parametrize("container_type", [tuple, list], ids=["tuple", "list"])
def test_group_commit_returns_empty_container_unchanged(offload_manager, container_type):
    empty = fine_grained_offloading_group_commit(container_type([]), "any_group")
    assert empty == container_type([])


@pytest.mark.parametrize(
    "record_callable",
    [fine_grained_offloading_forward_record, off_interface.forward_record],
    ids=["module_function", "interface_method"],
)
def test_forward_record_records_event_on_current_stream(offload_manager, record_callable):
    event = torch.cuda.Event(enable_timing=True)
    d2h_stream = offload_manager.d2h_stream
    d2h_stream.wait_stream(torch.cuda.current_stream())
    staging = torch.zeros(1024, 1024, device="cuda")
    with torch.cuda.stream(d2h_stream):
        staging.fill_(7.0)
    record_callable(event)
    total = staging.sum()
    marker = torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream().record_event(marker)
    torch.cuda.synchronize()
    assert event.elapsed_time(marker) >= 0.0
    assert total.item() == 7.0 * staging.numel()


def test_backward_record_records_event_during_backward(offload_manager):
    event = torch.cuda.Event(enable_timing=True)
    tensor = torch.randn(3, 4, device="cuda", requires_grad=True)
    recorded = fine_grained_offloading_backward_record(tensor, event)
    assert torch.equal(recorded, tensor)
    assert recorded.requires_grad is True
    recorded.sum().backward()
    marker = torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream().record_event(marker)
    torch.cuda.synchronize()
    assert event.elapsed_time(marker) >= 0.0
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor), rtol=0, atol=0)


def test_get_context_returns_manager_only_when_offload_flag_is_set(offload_manager):
    assert off_interface.get_context(True) is offload_manager
    assert isinstance(off_interface.get_context(False), contextlib.nullcontext)


def test_manager_context_is_noop_before_any_chunk_is_initialized(offload_manager):
    assert offload_manager.cur_forward_chunk() is None
    torch.manual_seed(1234)
    model = torch.nn.Linear(8, 8).cuda()
    torch.manual_seed(1234)
    reference = torch.nn.Linear(8, 8).cuda()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)
    expected = reference(x_reference)
    expected.sum().backward()

    with offload_manager:
        output = model(x)
        output.sum().backward()

    assert offload_manager.cur_forward_chunk() is None
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.bias.grad, reference.bias.grad, rtol=1e-6, atol=1e-6)
