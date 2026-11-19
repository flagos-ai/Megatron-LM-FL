# Copyright (c) 2026, BAAI. All rights reserved.
"""PagedStashManager autograd hooks: capture-time saves, truncation and padded retrieval."""

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


def test_on_save_returns_tensor_unchanged_when_max_num_tokens_unset():
    manager = PagedStashManager.get_instance()
    activation = _mark_activation('cuda')
    assert manager.on_save_for_backward(activation) is activation


@pytest.mark.parametrize('kind', ['zero-dim', 'missing-marker'], ids=['zero-dim', 'missing-marker'])
def test_on_save_returns_tensor_unchanged_for_unpageable_inputs(kind):
    manager = PagedStashManager.get_instance()
    paged_stash_reset(True, config=_stash_config())
    with _capture_context(manager, 'cuda'):
        if kind == 'zero-dim':
            tensor = torch.tensor(3.0, device='cuda')
            tensor.grouped_tensor_scale_inv = False
        else:
            tensor = _mark_activation('cuda')
            del tensor.grouped_tensor_scale_inv
        assert manager.on_save_for_backward(tensor) is tensor


class _MarkedNonTensor:
    """Stands in for a mis-tagged save: passes the attribute probe, fails isinstance."""

    grouped_tensor_scale_inv = False

    def dim(self):
        return 1


def test_on_save_rejects_non_tensor_inputs_that_carry_the_marker():
    manager = PagedStashManager.get_instance()
    paged_stash_reset(True, config=_stash_config())
    with _capture_context(manager, 'cuda'):
        with pytest.raises(AssertionError, match='tensor is not a torch.Tensor'):
            manager.on_save_for_backward(_MarkedNonTensor())


def test_on_save_truncates_to_actual_tokens_and_records_the_token_maximum():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', avg_tokens=_ACTUAL_TOKENS):
        first = _mark_activation('cuda')
        paged_first = manager.on_save_for_backward(first)
        manager.on_save_for_backward(_mark_activation('cuda'))

    assert paged_first._tensor.shape == (_ACTUAL_TOKENS * _HIDDEN,)
    assert torch.equal(paged_first._tensor, first.reshape(-1)[: _ACTUAL_TOKENS * _HIDDEN])
    assert list(paged_first.original_shape) == [_MAX_TOKENS, _HIDDEN]
    assert paged_first.avg_num_tokens == _ACTUAL_TOKENS
    assert paged_first.schedule_layer is None
    assert manager.max_tokens_across_vp_stages == {(torch.float32, _HIDDEN): 2 * _ACTUAL_TOKENS}
    assert manager.max_avg_tokens_across_vp_stages == {(torch.float32, _HIDDEN): 2 * _ACTUAL_TOKENS}
    assert manager.paged_tensors_to_stash == []


def test_on_get_pads_the_truncated_tensor_back_to_padded_shape():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', avg_tokens=_ACTUAL_TOKENS):
        activation = _mark_activation('cuda')
        paged = manager.on_save_for_backward(activation)

        reloaded = manager.on_get_saved_tensor(paged)
    assert reloaded.shape == (_MAX_TOKENS, _HIDDEN)
    assert torch.equal(reloaded[:_ACTUAL_TOKENS], activation[:_ACTUAL_TOKENS])
    assert torch.equal(reloaded[_ACTUAL_TOKENS:], torch.zeros_like(reloaded[_ACTUAL_TOKENS:]))
    assert manager.temp_tokens_across_vp_stages == {(torch.float32, _HIDDEN): 0}
    assert manager.temp_avg_tokens_across_vp_stages == {(torch.float32, _HIDDEN): 0}


def test_on_get_pads_columnwise_scale_inv_by_block_size():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', avg_tokens=64, max_tokens=128, tokens=64):
        scale_inv = torch.randn(128 // 32, 16, device='cuda')
        scale_inv.grouped_tensor_scale_inv = True
        paged = manager.on_save_for_backward(scale_inv)
        assert paged._tensor.shape == (32,)

        reloaded = manager.on_get_saved_tensor(paged)
    assert reloaded.shape == (128 // 32, 16)
    flat = reloaded.reshape(-1)
    assert torch.equal(flat[:32], scale_inv.reshape(-1)[:32])
    assert torch.equal(flat[32:], torch.zeros(32, device=flat.device))
    assert manager.temp_tokens_across_vp_stages == {(torch.float32, 16): 0}
    assert manager.temp_avg_tokens_across_vp_stages == {(torch.float32, 16): 0}


def test_on_get_views_one_byte_elements_as_uint8():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', max_tokens=64, tokens=32):
        payload = torch.arange(64, dtype=torch.uint8, device='cuda').reshape(64, 1)
        payload.grouped_tensor_scale_inv = False
        paged = manager.on_save_for_backward(payload)
        assert paged._tensor.shape == (32,)

        reloaded = manager.on_get_saved_tensor(paged)
    assert reloaded.dtype == torch.uint8
    assert reloaded.shape == (64, 1)
    flat = reloaded.reshape(-1)
    assert torch.equal(flat[:32], payload.reshape(-1)[:32])
    assert flat[32:].sum().item() == 0
    assert manager.temp_tokens_across_vp_stages == {(torch.uint8, 1): 0}


def test_on_get_rejects_multidimensional_stash_tensors():
    paged_stash_reset(True, config=_stash_config())
    paged_stash_init_chunk_handler(1, 0)
    manager = PagedStashManager.get_instance()
    with _capture_context(manager, 'cuda', max_tokens=8, tokens=4):
        manager.on_save_for_backward(_mark_activation('cuda', max_tokens=8, hidden=8))
        paged = PagedTensor(
            torch.zeros(4, 8, device='cuda'),
            num_tokens_tensor=torch.tensor([4], device='cuda'),
            max_num_tokens=8,
            hidden_size=8,
            page_size=4,
        )
        with pytest.raises(AssertionError, match='ndim is not 1'):
            manager.on_get_saved_tensor(paged)


def test_on_get_returns_the_reload_view_without_padding_once_captured():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    payload = torch.arange(_MAX_TOKENS * _HIDDEN, dtype=torch.float32, device='cuda')
    paged = PagedTensor(
        payload,
        num_tokens_tensor=torch.tensor([_ACTUAL_TOKENS], device='cuda'),
        max_num_tokens=_MAX_TOKENS,
        hidden_size=_HIDDEN,
        page_size=_PAGE,
        original_shape=[_MAX_TOKENS, _HIDDEN],
    )
    reloaded = manager.on_get_saved_tensor(paged)
    assert reloaded.shape == (_MAX_TOKENS, _HIDDEN)
    assert torch.equal(reloaded, payload.reshape(_MAX_TOKENS, _HIDDEN))


def test_on_get_raises_when_a_stashed_tensor_was_never_reloaded():
    manager = _captured_manager('cuda', _stash_config(), avg_tokens=_ACTUAL_TOKENS)
    paged = manager.on_save_for_backward(_mark_activation('cuda'))
    manager.stash_paged_tensors(1)
    assert paged._tensor is None
    with pytest.raises(AssertionError, match='_tensor is None'):
        manager.on_get_saved_tensor(paged)


def test_on_get_passes_non_paged_state_through_unchanged():
    manager = PagedStashManager.get_instance()
    value = torch.randn(3, device='cuda')
    assert manager.on_get_saved_tensor(value) is value
