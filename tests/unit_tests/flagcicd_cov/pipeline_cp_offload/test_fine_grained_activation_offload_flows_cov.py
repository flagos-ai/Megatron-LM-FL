# Copyright (c) 2026, BAAI. All rights reserved.
"""Offload manager lifecycle: delayed-group flush, warmup offload/reload with gradients, post-warmup group marking, steady-state chunk reuse, and offload enable/disable."""

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    PipelineOffloadManager,
    fine_grained_offloading_disable_offload,
    fine_grained_offloading_enable_offload,
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_flush_delayed_groups,
    fine_grained_offloading_group_start,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def offload_manager():
    PipelineOffloadManager.reset_instance()
    yield PipelineOffloadManager.get_instance()
    PipelineOffloadManager.reset_instance()


@pytest.fixture
def serial_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _run_forward_group(tensor, chunk, name):
    started = fine_grained_offloading_group_start(tensor, name)
    tag = chunk.tensor_push(tensor)
    committed = fine_grained_offloading_group_commit(started, name)
    return tag, committed


def test_delayed_group_commit_offloads_only_when_delayed_groups_are_flushed(offload_manager):
    off_interface.init_chunk_handler(1, 0, 8)
    chunk = offload_manager.cur_forward_chunk()
    assert chunk.is_empty_chunk() is True
    tensor = torch.randn(4, 16, device="cuda")
    started = fine_grained_offloading_group_start(tensor, "delayed_group")
    chunk.tensor_push(tensor)
    committed = fine_grained_offloading_group_commit(started, "delayed_group", delay_offload=True)
    assert torch.equal(committed, tensor)
    assert offload_manager.cpu_tensor_pool.get_pool_status()["pools"] == {}
    fine_grained_offloading_group_flush_delayed_groups()
    status = offload_manager.cpu_tensor_pool.get_pool_status((4, 16), torch.float32)
    assert status["total_count"] == 1
    assert status["allocated_count"] == 1
    assert chunk.offload_groups[0].total_offload_bytes == 4 * 16 * 4
    fine_grained_offloading_group_flush_delayed_groups()
    status = offload_manager.cpu_tensor_pool.get_pool_status((4, 16), torch.float32)
    assert status["total_count"] == 1
    assert status["allocated_count"] == 1


def test_warmup_group_offload_and_reload_preserves_outputs_and_gradients(offload_manager):
    off_interface.init_chunk_handler(1, 0, 1, max_inflight_offloads=0)
    torch.manual_seed(1234)
    model = torch.nn.Linear(16, 16).cuda()
    torch.manual_seed(1234)
    reference = torch.nn.Linear(16, 16).cuda()
    x = torch.randn(4, 16, device="cuda", requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)
    expected = reference(x_reference)
    expected.sum().backward()

    with off_interface(True, x, "dense_layer") as started:
        hidden = model(started)
    out = off_interface.group_commit(hidden, "dense_layer")
    loss = out.sum()
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(out.detach(), expected.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.bias.grad, reference.bias.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=1e-6, atol=1e-6)
    stats = offload_manager.cpu_tensor_pool.get_pool_status()["global_stats"]
    assert stats["allocation_requests"] == stats["free_requests"]
    assert stats["current_in_use"] == 0
    chunk = offload_manager.cur_forward_chunk()
    assert chunk.finish_all_groups("dense_layer") is True


def test_post_warmup_callback_marks_last_groups_unoffloadable_and_reports_bytes(
    serial_parallel, offload_manager
):
    tensor = torch.randn(4, 32, device="cuda")
    off_interface.init_chunk_handler(1, 0, 8, max_inflight_offloads=0)
    first_chunk = offload_manager.cur_forward_chunk()
    assert offload_manager.cpu_tensor_pool is first_chunk.cpu_tensor_pool
    _run_forward_group(tensor, first_chunk, "a")
    _run_forward_group(tensor, first_chunk, "b")
    off_interface.init_chunk_handler(1, 0, 8, max_inflight_offloads=0)
    second_chunk = offload_manager.cur_forward_chunk()
    _run_forward_group(tensor, second_chunk, "c")
    assert second_chunk is not first_chunk

    offload_manager.reset()

    assert offload_manager.offload_summary_bytes == {"b": 512}
    assert offload_manager.offload_summary_total_bytes == 512
    assert [group.offload for group in first_chunk.offload_groups] == [False, True]
    assert [group.offload for group in second_chunk.offload_groups] == [False]
    status = offload_manager.cpu_tensor_pool.get_pool_status((4, 32), torch.float32)
    assert status["total_count"] == 3
    assert status["allocated_count"] == 0
    assert status["free_count"] == 3
    offload_manager.pop_backward_chunk("b")
    assert offload_manager.cur_backward_chunk() is first_chunk
    assert offload_manager.front_backward_chunk("c") is second_chunk
    assert offload_manager.front_backward_chunk("missing_group") is None


def test_virtual_pipeline_stages_are_pushed_to_backward_queue_in_reverse_order(
    serial_parallel, offload_manager
):
    tensor = torch.randn(4, 32, device="cuda")
    off_interface.init_chunk_handler(2, 0, 8)
    stage_zero_chunk = offload_manager.cur_forward_chunk()
    _run_forward_group(tensor, stage_zero_chunk, "s0")
    off_interface.init_chunk_handler(2, 1, 8)
    stage_one_chunk = offload_manager.cur_forward_chunk()
    assert stage_one_chunk is not stage_zero_chunk
    _run_forward_group(tensor, stage_one_chunk, "s1")

    offload_manager.reset()

    assert offload_manager.front_backward_chunk("s1") is stage_one_chunk
    assert offload_manager.front_backward_chunk("s0") is stage_zero_chunk
    offload_manager.pop_backward_chunk("s0")
    assert offload_manager.cur_backward_chunk() is stage_zero_chunk


def test_virtual_pipeline_backward_preloads_last_group_of_next_chunk(
    serial_parallel, offload_manager
):
    off_interface.init_chunk_handler(2, 0, 8)
    stage_zero_chunk = offload_manager.cur_forward_chunk()
    _run_forward_group(torch.randn(4, 32, device="cuda"), stage_zero_chunk, "shared")
    _run_forward_group(torch.randn(4, 32, device="cuda"), stage_zero_chunk, "shared")
    _run_forward_group(torch.randn(4, 32, device="cuda"), stage_zero_chunk, "tail")
    off_interface.init_chunk_handler(2, 1, 8)
    stage_one_chunk = offload_manager.cur_forward_chunk()
    _run_forward_group(torch.randn(4, 32, device="cuda"), stage_one_chunk, "shared")
    offload_manager.reset()

    steady_shared = torch.randn(4, 32, device="cuda")
    tag, _ = _run_forward_group(steady_shared, stage_zero_chunk, "shared")
    assert tag == (1, 0)
    _run_forward_group(torch.randn(4, 32, device="cuda"), stage_zero_chunk, "tail")
    _run_forward_group(torch.randn(4, 32, device="cuda"), stage_one_chunk, "shared")

    offload_manager.pop_backward_chunk("shared")
    assert offload_manager.cur_backward_chunk() is stage_one_chunk
    stage_one_chunk.on_group_start_backward()
    torch.cuda.synchronize()
    preloaded = stage_zero_chunk.offload_groups[0]._tensors[tag]
    assert preloaded.device.type == "cuda"
    assert torch.equal(preloaded, steady_shared)
    stage_one_chunk.on_group_commit_backward("shared")

    offload_manager.pop_backward_chunk("shared")
    assert offload_manager.cur_backward_chunk() is stage_zero_chunk
    stage_zero_chunk.on_group_commit_backward("shared")
    assert torch.equal(stage_zero_chunk.tensor_pop(tag), steady_shared)


def test_steady_state_forward_reuses_cached_chunk_and_reloads_offloaded_tensors(
    serial_parallel, offload_manager
):
    off_interface.init_chunk_handler(1, 0, 8, max_inflight_offloads=0)
    chunk = offload_manager.cur_forward_chunk()
    _run_forward_group(torch.randn(4, 32, device="cuda"), chunk, "x")
    _run_forward_group(torch.randn(4, 32, device="cuda"), chunk, "y")
    _run_forward_group(torch.randn(4, 32, device="cuda"), chunk, "x")
    offload_manager.reset()

    steady_x = torch.randn(4, 32, device="cuda")
    x_tag, _ = _run_forward_group(steady_x, chunk, "x")
    assert x_tag == (1, 0)
    steady_y = torch.randn(4, 32, device="cuda")
    y_tag, _ = _run_forward_group(steady_y, chunk, "y")
    assert y_tag == (2, 0)

    chunk.on_group_start_backward()
    chunk.on_group_commit_backward("x")
    reloaded = chunk.tensor_pop(x_tag)
    torch.cuda.synchronize()
    assert reloaded.device.type == "cuda"
    assert torch.equal(reloaded, steady_x)

    chunk.on_group_start_backward()
    chunk.on_group_commit_backward("y")
    assert chunk.tensor_pop(y_tag) is steady_y

    status = offload_manager.cpu_tensor_pool.get_pool_status((4, 32), torch.float32)
    assert status["total_count"] == 3
    assert status["allocated_count"] == 0
    assert status["free_count"] == 3


def test_steady_state_forward_returns_input_unchanged_once_chunk_groups_are_exhausted(
    serial_parallel, offload_manager
):
    off_interface.init_chunk_handler(1, 0, 8, max_inflight_offloads=0)
    chunk = offload_manager.cur_forward_chunk()
    _run_forward_group(torch.randn(4, 32, device="cuda"), chunk, "x")
    _run_forward_group(torch.randn(4, 32, device="cuda"), chunk, "x")
    offload_manager.reset()

    steady = torch.randn(4, 32, device="cuda")
    tag, _ = _run_forward_group(steady, chunk, "x")
    chunk.on_group_start_backward()
    chunk.on_group_commit_backward("x")
    torch.cuda.synchronize()
    assert torch.equal(chunk.tensor_pop(tag), steady)

    unplaced_input = torch.randn(2, 2, device="cuda")
    assert fine_grained_offloading_group_start(unplaced_input, "x") is unplaced_input
    status = offload_manager.cpu_tensor_pool.get_pool_status((4, 32), torch.float32)
    assert status["total_count"] == 2
    assert status["allocated_count"] == 0
    assert status["free_count"] == 2


def test_disable_offload_skips_group_hooks_until_re_enabled(offload_manager):
    off_interface.init_chunk_handler(1, 0, 8)
    chunk = offload_manager.cur_forward_chunk()
    fine_grained_offloading_disable_offload()
    assert offload_manager.do_offload is False
    assert chunk.do_offload is False
    x = torch.randn(3, 5, device="cuda", requires_grad=True)
    started = fine_grained_offloading_group_start(x, "disabled_group")
    committed = fine_grained_offloading_group_commit(started, "disabled_group")
    committed.sum().backward()
    assert chunk.offload_groups == []
    assert offload_manager.cpu_tensor_pool.get_pool_status()["pools"] == {}
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
    fine_grained_offloading_enable_offload()
    assert offload_manager.do_offload is True
    assert chunk.do_offload is True
