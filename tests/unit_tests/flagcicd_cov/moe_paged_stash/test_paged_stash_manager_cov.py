# Copyright (c) 2026, BAAI. All rights reserved.
"""PagedStashManager lifecycle: capture stats, buffer allocation, stash/reload queues, hooks."""

import pytest
import torch

from megatron.core.transformer.moe.paged_stash import (
    PagedStashManager,
    PagedTensor,
    get_paged_stash_context,
    paged_stash_group_commit,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.transformer_config import TransformerConfig

_HIDDEN = 64
_MAX_TOKENS = 128
_ACTUAL_TOKENS = 100
_PAGE = 16


@pytest.fixture(autouse=True)
def _fresh_stash_manager(monkeypatch):
    """Give every test a pristine singleton and undo the swap afterwards."""
    monkeypatch.setattr(PagedStashManager, 'STASH_MGR', None, raising=False)
    yield


def _stash_config(*, page_size=_PAGE, cuda_factor=2.0, cpu_factor=0.0):
    return TransformerConfig(
        num_layers=2,
        hidden_size=_HIDDEN,
        num_attention_heads=8,
        moe_paged_stash_page_size=page_size,
        moe_paged_stash_buffer_size_factor_cuda=cuda_factor,
        moe_paged_stash_buffer_size_factor_cpu=cpu_factor,
    )


def _capture_context(manager, device, *, avg_tokens=None, max_tokens=_MAX_TOKENS, tokens=None):
    """Install the layer context that configures num_tokens tracking on the manager."""
    num_tokens_tensor = torch.tensor([tokens or _ACTUAL_TOKENS], device=device)
    return get_paged_stash_context(
        name='moe',
        max_num_tokens=max_tokens,
        num_tokens_tensor=num_tokens_tensor,
        avg_num_tokens=avg_tokens,
    )


def _mark_activation(device, *, max_tokens=_MAX_TOKENS, hidden=_HIDDEN, columnwise=False):
    activation = torch.randn(max_tokens, hidden, device=device)
    activation.grouped_tensor_scale_inv = columnwise
    return activation


def _captured_manager(device, config=None, *, avg_tokens=None, saves=2):
    """Drive the singleton from fresh to 'captured' through real capture saves."""
    paged_stash_group_commit(torch.zeros(1, device=device))
    manager = PagedStashManager.get_instance()
    paged_stash_reset(True, config=config)
    paged_stash_init_chunk_handler(1, 0)
    with _capture_context(manager, device, avg_tokens=avg_tokens):
        for _ in range(saves):
            manager.on_save_for_backward(_mark_activation(device))
    paged_stash_reset(True, config=config)
    return manager


def test_get_instance_returns_the_same_manager_until_reset(monkeypatch):
    first = PagedStashManager.get_instance()
    assert first.enabled is False
    assert first.status == 'begin'
    assert first.page_size == 64
    assert first.vp_size is None
    assert first.pack_stream is first.unpack_stream
    assert PagedStashManager.get_instance() is first

    monkeypatch.setattr(PagedStashManager, 'STASH_MGR', None, raising=False)
    second = PagedStashManager.get_instance()
    assert second is not first
    assert second.status == 'begin'
    assert second.paged_tensors_to_stash == []
    assert PagedStashManager.get_instance() is second


def test_get_schedule_layer_encodes_stage_layer_and_microbatch():
    manager = PagedStashManager.get_instance()
    assert manager.get_schedule_layer(2, 3, 4) == 2 * 1000000 + 3 * 1000 + 4


@pytest.mark.parametrize(
    'layer_no, microbatch_no', [(1000, 0), (0, 1000)], ids=['layer-overflow', 'microbatch-overflow']
)
def test_get_schedule_layer_rejects_codes_beyond_the_encoding_range(layer_no, microbatch_no):
    manager = PagedStashManager.get_instance()
    with pytest.raises(AssertionError, match='Schedule encoding overflow'):
        manager.get_schedule_layer(1, layer_no, microbatch_no)


def test_update_model_chunk_tracks_layers_and_microbatches_per_stage():
    paged_stash_init_chunk_handler(2, 0)
    manager = PagedStashManager.get_instance()
    assert manager.vp_size == 2
    assert manager.current_vp_stage == 0
    assert manager.current_layer == [1, 1]
    assert manager.current_microbatch == [1, 0]

    manager.update_model_chunk(1)
    assert manager.current_layer == [1, 1]
    assert manager.current_microbatch == [1, 1]

    paged_stash_init_chunk_handler(2, 1)
    assert manager.current_vp_stage == 1
    assert manager.current_layer == [1, 1]
    assert manager.current_microbatch == [1, 2]


def test_init_chunk_handler_defaults_to_a_single_stage_when_sizes_are_none():
    paged_stash_init_chunk_handler(None, None)
    manager = PagedStashManager.get_instance()
    assert manager.vp_size == 1
    assert manager.current_vp_stage == 0
    assert manager.current_layer == [1]
    assert manager.current_microbatch == [1]


def test_update_pp_schedule_appends_signed_entries_during_capture():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()

    assert manager.update_pp_schedule(1) == (1, 1)
    manager.current_schedule_index += 1
    assert manager.update_pp_schedule(1) == (2, 1)
    manager.current_schedule_index += 1
    assert manager._pp_schedule == [1001001, 1002001]

    assert manager.update_pp_schedule(-1, -2, -1) == (-2, -1)
    assert manager._pp_schedule == [1001001, 1002001, -1002001]


def test_update_pp_schedule_rejects_replayed_entries_that_do_not_match():
    paged_stash_group_commit(torch.zeros(1, device='cuda'))
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    manager.update_pp_schedule(1, layer_no=2, microbatch_no=1)
    paged_stash_reset(True, config=_stash_config())

    with pytest.raises(AssertionError, match='schedule 1002001 != 1001001'):
        manager.update_pp_schedule(1, layer_no=1, microbatch_no=1)


def test_add_and_remove_paged_tensor_only_apply_while_captured():
    manager = PagedStashManager.get_instance()
    paged = PagedTensor(
        torch.zeros(4, 8, device='cuda'),
        num_tokens_tensor=torch.tensor([4], device='cuda'),
        max_num_tokens=4,
        hidden_size=8,
    )

    manager.add_paged_tensor_to_stash(paged)
    assert manager.paged_tensors_to_stash == []
    manager.remove_paged_tensor_from_stash()
    assert manager.paged_tensors_to_stash == []

    paged_stash_group_commit(torch.zeros(1, device='cuda'))
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    paged_stash_reset(True, config=_stash_config())
    manager.add_paged_tensor_to_stash(paged)
    assert manager.paged_tensors_to_stash == [paged]
    manager.remove_paged_tensor_from_stash()
    assert manager.paged_tensors_to_stash == []


def test_stash_outside_captured_status_is_a_noop():
    manager = PagedStashManager.get_instance()
    manager.stash_paged_tensors(1)
    assert manager.paged_tensors_to_reload == {}
    assert manager.paged_tensors_to_stash == []


def test_stash_moves_pending_paged_tensors_into_the_reload_queue():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    activation = _mark_activation('cuda')
    paged = manager.on_save_for_backward(activation)
    assert manager.paged_tensors_to_stash == [paged]

    manager.stash_paged_tensors(1)
    assert manager.paged_tensors_to_stash == []
    assert manager.paged_tensors_to_reload[1] == [paged]
    assert manager.paged_tensors_stash_in_progress == [paged]
    assert manager._pack_stream_status == 'stashing'
    assert paged._tensor is None
    assert paged._original_tensor.data_ptr() == activation.data_ptr()

    manager.wait_for_stash_to_complete()
    assert manager._pack_stream_status == 'idle'
    assert paged._original_tensor is None
    assert manager.paged_tensors_stash_in_progress == []

    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.free_list_head.tolist() == [7, 0]


def test_stash_rejects_a_reload_queue_with_leftover_entries():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    first = manager.on_save_for_backward(_mark_activation('cuda'))
    manager.stash_paged_tensors(1)
    assert manager.paged_tensors_to_reload[1] == [first]

    manager.on_save_for_backward(_mark_activation('cuda'))
    with pytest.raises(AssertionError, match='is not empty'):
        manager.stash_paged_tensors(1)


@pytest.mark.parametrize('no_wait', [False, True], ids=['wait-stream', 'no-wait'])
def test_reload_restores_paged_tensors_from_the_stash(no_wait):
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    activation = _mark_activation('cuda')
    paged = manager.on_save_for_backward(activation)
    manager.stash_paged_tensors(1)

    manager.reload_paged_tensors(1, no_wait=no_wait)
    assert manager.paged_tensors_to_reload[1] == []
    assert manager._unpack_stream_status == 'reloading'
    assert tuple(paged._tensor.shape) == tuple(paged.original_shape)

    torch.cuda.current_stream().wait_stream(manager.unpack_stream)
    flat = paged._tensor.reshape(-1)
    assert torch.equal(
        flat[: _ACTUAL_TOKENS * _HIDDEN], activation.reshape(-1)[: _ACTUAL_TOKENS * _HIDDEN]
    )
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.free_list_tail.tolist() == [25 + 7, 0]


def test_reset_transitions_begin_to_capture_without_allocating():
    paged_stash_reset(True, config=_stash_config())
    manager = PagedStashManager.get_instance()
    assert manager.enabled is True
    assert manager.status == 'capture'
    assert manager.iteration == 1
    assert manager.current_schedule_index == 0
    assert manager.stash_buffers is None
    assert manager.page_size == _PAGE


def test_reset_transitions_capture_to_captured_and_allocates_buffers():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    assert manager.status == 'captured'
    assert manager.iteration == 2
    assert manager.current_layer == [1]
    assert manager.current_microbatch == [0]
    assert manager.overflow.item() == 0
    assert manager.host_spill.item() == 0

    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 25
    assert buffer.cuda_buffer.shape == (400, _HIDDEN)
    assert buffer.host_buffer is None


def test_reset_skips_state_transitions_when_disabled():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_reset(False, config=_stash_config())
    manager = PagedStashManager.get_instance()
    assert manager.enabled is False
    assert manager.status == 'capture'
    assert manager.iteration == 2
    assert manager.current_schedule_index == 0
    assert manager.stash_buffers is None


def test_reset_reallocates_released_buffers_while_captured():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    manager.release_stash_buffers()
    assert manager.stash_buffers is None

    paged_stash_reset(True, config=_stash_config())
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 25
    assert manager.overflow.item() == 0


def test_reset_applies_default_size_factors_when_config_is_omitted():
    paged_stash_group_commit(torch.zeros(1, device='cuda'))
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', avg_tokens=_ACTUAL_TOKENS):
        for _ in range(2):
            manager.on_save_for_backward(_mark_activation('cuda'))
    paged_stash_reset(True)

    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 14
    assert buffer.cuda_buffer.shape == (224, _HIDDEN)
    assert buffer.host_buffer is None


def test_reset_restores_free_lists_and_flags_between_iterations():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    manager.on_save_for_backward(_mark_activation('cuda'))
    manager.stash_paged_tensors(1)
    manager.wait_for_stash_to_complete()
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.free_list_head.tolist() == [7, 0]

    paged_stash_reset(True, config=_stash_config())
    assert buffer.free_list_head.tolist() == [0, 0]
    assert buffer.free_list_tail.tolist() == [25, 0]
    assert buffer.free_list_cuda.tolist() == list(range(25))
    assert manager.overflow.item() == 0
    assert manager.host_spill.item() == 0


def test_allocate_sizes_cuda_tier_from_avg_tokens_for_positive_factor():
    manager = _captured_manager('cuda', _stash_config(cuda_factor=2.0), avg_tokens=60, saves=2)
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 15
    assert buffer.cuda_buffer.shape == (240, _HIDDEN)


def test_allocate_sizes_cuda_tier_from_actual_tokens_for_negative_factor():
    manager = _captured_manager('cuda', _stash_config(cuda_factor=-2.0), avg_tokens=60, saves=2)
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 25
    assert buffer.cuda_buffer.shape == (400, _HIDDEN)


def test_allocate_sizes_zero_cuda_tier_when_capture_recorded_no_avg_hint():
    manager = _captured_manager('cuda', _stash_config(cuda_factor=2.0), avg_tokens=None, saves=2)
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 0
    assert buffer.cuda_buffer.shape == (0, _HIDDEN)


def test_allocate_adds_pinned_host_tier_for_positive_cpu_factor():
    manager = _captured_manager(
        'cuda', _stash_config(cuda_factor=2.0, cpu_factor=2.0), avg_tokens=_ACTUAL_TOKENS
    )
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 25
    assert buffer.num_host_pages == 25
    assert buffer.host_buffer.shape == (400, _HIDDEN)
    assert buffer.host_buffer.is_pinned()


def test_allocate_sizes_host_tier_from_actual_tokens_for_negative_cpu_factor():
    manager = _captured_manager(
        'cuda', _stash_config(cuda_factor=2.0, cpu_factor=-2.0), avg_tokens=60, saves=2
    )
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.num_cuda_pages == 15
    assert buffer.num_host_pages == 25


def test_allocate_skips_when_capture_recorded_no_token_maximum():
    paged_stash_group_commit(torch.zeros(1, device='cuda'))
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    paged_stash_reset(True, config=_stash_config())
    manager = PagedStashManager.get_instance()
    assert manager.status == 'captured'
    assert manager.stash_buffers == {}
    assert manager.overflow.shape == (1,)
    assert manager.host_spill.shape == (1,)


def test_allocate_zeros_stale_overflow_and_recreates_buffers():
    manager = _captured_manager('cuda', _stash_config(cuda_factor=0.001), avg_tokens=_ACTUAL_TOKENS)
    paged = manager.on_save_for_backward(_mark_activation('cuda'))
    manager.stash_paged_tensors(1)
    manager.wait_for_stash_to_complete()
    assert manager.overflow.item() == 1
    assert manager.stash_buffers[torch.float32][_HIDDEN].num_cuda_pages == 0

    manager.allocate_stash_buffers(
        moe_paged_stash_buffer_size_factor_cuda=2.0, moe_paged_stash_buffer_size_factor_cpu=0.0
    )
    assert manager.overflow.item() == 0
    assert manager.stash_buffers[torch.float32][_HIDDEN].num_cuda_pages == 25
    assert paged._tensor is None


def test_allocate_groups_buffers_by_dtype_and_hidden_size():
    paged_stash_group_commit(torch.zeros(1, device='cuda'))
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', avg_tokens=_ACTUAL_TOKENS):
        manager.on_save_for_backward(_mark_activation('cuda', hidden=_HIDDEN))
        manager.on_save_for_backward(_mark_activation('cuda', hidden=32))
    paged_stash_reset(True, config=_stash_config())

    assert set(manager.stash_buffers[torch.float32]) == {_HIDDEN, 32}
    assert manager.stash_buffers[torch.float32][_HIDDEN].cuda_buffer.shape == (208, _HIDDEN)
    assert manager.stash_buffers[torch.float32][32].cuda_buffer.shape == (208, 32)
