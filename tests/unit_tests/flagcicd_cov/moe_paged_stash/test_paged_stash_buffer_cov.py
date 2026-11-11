# Copyright (c) 2026, BAAI. All rights reserved.
"""PagedStashBuffer page arithmetic and PagedTensor offload/reload across CUDA and pinned host."""

import pytest
import torch

from megatron.core.transformer.moe.paged_stash import (
    GLOBAL_BLOCK_SIZE,
    PagedStashBuffer,
    PagedTensor,
    _paged_stash_pop_kernel,
)


def _flags(device):
    """Fresh shared overflow / host-spill flag pair, as the stash manager allocates them."""
    return (
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
    )


def _paged_activation(device, *, tokens, max_tokens=128, hidden=64, page=16, columnwise=False):
    """A flattened activation plus its clone, ready for one offload/reload round trip."""
    torch.manual_seed(123)
    numel = (max_tokens // 32) * hidden if columnwise else max_tokens * hidden
    original = torch.randn(numel, device=device)
    paged = PagedTensor(
        original.clone(),
        num_tokens_tensor=torch.tensor([tokens], device=device),
        max_num_tokens=max_tokens,
        hidden_size=hidden,
        page_size=page,
        is_columnwise_scale_inv=columnwise,
    )
    return original, paged


def test_cuda_pages_round_up_when_tokens_do_not_fill_the_last_page():
    overflow, host_spill = _flags(torch.device('cpu'))
    buffer = PagedStashBuffer(130, 64, 64, torch.device('cpu'), overflow, host_spill, torch.float32)
    assert buffer.num_cuda_pages == 3
    assert buffer.total_cuda_tokens == 192
    assert buffer.cuda_buffer.shape == (192, 64)
    assert buffer.cuda_buffer.dtype == torch.float32
    assert buffer.host_buffer is None
    assert buffer.num_host_pages == 0
    assert buffer.total_host_tokens == 0
    assert buffer.free_list_head.tolist() == [0, 0]
    assert buffer.free_list_tail.tolist() == [3, 0]
    assert buffer.free_list_capacity.tolist() == [3, 0]
    assert buffer.free_list_cuda.tolist() == [0, 1, 2]
    assert buffer.free_list_host.numel() == 0


def test_host_pages_round_up_and_pin_when_host_tokens_requested():
    overflow, host_spill = _flags('cuda')
    buffer = PagedStashBuffer(
        130, 64, 64, 'cuda', overflow, host_spill, torch.float32, num_tokens_host=70
    )
    assert buffer.num_host_pages == 2
    assert buffer.total_host_tokens == 128
    assert buffer.host_buffer.shape == (128, 64)
    assert buffer.host_buffer.is_pinned()
    assert buffer.free_list_tail.tolist() == [3, 2]
    assert buffer.free_list_capacity.tolist() == [3, 2]
    assert buffer.free_list_host.tolist() == [0, 1]


def test_reset_restores_free_lists_after_pages_have_been_allocated():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(64, 64, 16, device, overflow, host_spill, torch.float32)
    _, paged = _paged_activation(device, tokens=48)
    paged.offload_to_stash(buffer)
    assert buffer.free_list_head.tolist() == [3, 0]
    buffer.reset()
    assert buffer.free_list_head.tolist() == [0, 0]
    assert buffer.free_list_tail.tolist() == [4, 0]
    assert buffer.free_list_cuda.tolist() == [0, 1, 2, 3]
    assert buffer.free_list_host.tolist() == []


def test_reset_restores_host_free_list_after_a_host_spill():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(
        16, 64, 16, device, overflow, host_spill, torch.float32, num_tokens_host=32
    )
    _, cuda_paged = _paged_activation(device, tokens=16)
    cuda_paged.offload_to_stash(buffer)
    _, host_paged = _paged_activation(device, tokens=32)
    host_paged.offload_to_stash(buffer)
    assert host_paged.spilled_to_host.item() == 1
    assert buffer.free_list_head.tolist() == [1, 2]

    buffer.reset()
    assert buffer.free_list_head.tolist() == [0, 0]
    assert buffer.free_list_tail.tolist() == [1, 2]
    assert buffer.free_list_cuda.tolist() == [0]
    assert buffer.free_list_host.tolist() == [0, 1]


def test_repr_lists_page_and_buffer_configuration():
    overflow, host_spill = _flags('cuda')
    buffer = PagedStashBuffer(
        130, 64, 64, 'cuda', overflow, host_spill, torch.float32, num_tokens_host=70
    )
    text = repr(buffer)
    assert text == (
        'PagedStashBuffer(num_cuda_pages=3, num_host_pages=2, '
        'page_size=64, hidden_size=64, device=cuda, dtype=torch.float32)'
    )


def test_paged_tensor_records_layout_and_allocates_page_metadata():
    activation = torch.arange(256, dtype=torch.float32).reshape(8, 32)
    paged = PagedTensor(
        activation,
        num_tokens_tensor=torch.tensor([6]),
        vp_stage=2,
        schedule_layer_no=77,
        max_num_tokens=8,
        hidden_size=32,
        page_size=3,
    )
    assert paged.original_shape == [8, 32]
    assert paged.max_num_pages == 3
    assert paged.page_record.shape == (3,)
    assert paged.page_record.dtype == torch.int64
    assert paged.spilled_to_host.shape == (1,)
    assert paged.element_size == 4
    assert paged.dtype == torch.float32
    assert paged.device == activation.device
    assert paged.vp_stage == 2
    assert paged.schedule_layer == 77
    assert paged.max_num_tokens == 8
    assert paged.hidden_size == 32


@pytest.mark.parametrize(
    'num_tokens_tensor', [None, torch.zeros(2, dtype=torch.int64)], ids=['missing', 'two-elements']
)
def test_paged_tensor_rejects_token_counts_that_are_not_single_elements(num_tokens_tensor):
    with pytest.raises(AssertionError):
        PagedTensor(
            torch.zeros(4, 8), num_tokens_tensor=num_tokens_tensor, max_num_tokens=4, hidden_size=8
        )


def test_offload_then_reload_round_trips_tokens_on_cuda_pages():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(400, 64, 16, device, overflow, host_spill, torch.float32)
    original, paged = _paged_activation(device, tokens=100)
    paged.offload_to_stash(buffer)

    assert paged.page_record.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert paged.spilled_to_host.item() == 0
    assert host_spill.item() == 0
    assert overflow.item() == 0
    assert buffer.free_list_head.tolist() == [7, 0]

    reloaded = torch.empty(paged.original_shape, dtype=torch.float32, device=device)
    paged._tensor = reloaded
    paged.reload_from_stash(buffer)
    assert torch.equal(reloaded[: 100 * 64], original[: 100 * 64])
    assert buffer.free_list_tail.tolist() == [32, 0]
    assert buffer.free_list_head.tolist() == [7, 0]


def test_offload_spills_to_pinned_host_when_cuda_pages_run_out():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(
        16, 64, 16, device, overflow, host_spill, torch.float32, num_tokens_host=64
    )
    _, cuda_paged = _paged_activation(device, tokens=16)
    cuda_paged.offload_to_stash(buffer)
    assert cuda_paged.spilled_to_host.item() == 0
    assert buffer.free_list_head.tolist() == [1, 0]

    host_original, host_paged = _paged_activation(device, tokens=32)
    host_paged.offload_to_stash(buffer)
    assert host_paged.spilled_to_host.item() == 1
    assert host_spill.item() == 1
    assert overflow.item() == 0
    assert host_paged.page_record.tolist() == [0, 1]
    assert buffer.free_list_head.tolist() == [1, 2]

    reloaded = torch.empty(host_paged.original_shape, dtype=torch.float32, device=device)
    host_paged._tensor = reloaded
    host_paged.reload_from_stash(buffer)
    assert torch.equal(reloaded[: 32 * 64], host_original[: 32 * 64])
    assert buffer.free_list_tail.tolist() == [1, 6]


def test_offload_sets_overflow_when_cuda_and_host_tiers_are_both_full():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(16, 64, 16, device, overflow, host_spill, torch.float32)
    original, paged = _paged_activation(device, tokens=32)
    paged.offload_to_stash(buffer)
    assert overflow.item() == 1
    assert paged.spilled_to_host.item() == 1
    assert buffer.free_list_head.tolist() == [0, 0]
    assert paged.page_record.sum().item() == 0


def test_offload_is_noop_for_every_tensor_once_overflow_is_set():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(16, 64, 16, device, overflow, host_spill, torch.float32)
    _, overflowing = _paged_activation(device, tokens=32)
    overflowing.offload_to_stash(buffer)
    assert overflow.item() == 1

    _, following = _paged_activation(device, tokens=16)
    following.offload_to_stash(buffer)
    assert buffer.free_list_head.tolist() == [0, 0]
    assert following.page_record.sum().item() == 0
    assert following.spilled_to_host.item() == 0


def test_reload_preserves_free_list_and_leaves_dst_untouched_when_overflow_set():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(16, 64, 16, device, overflow, host_spill, torch.float32)
    original, stashed = _paged_activation(device, tokens=16)
    stashed.offload_to_stash(buffer)
    assert buffer.free_list_head.tolist() == [1, 0]

    _, overflowing = _paged_activation(device, tokens=32)
    overflowing.offload_to_stash(buffer)
    assert overflow.item() == 1

    reloaded = torch.full(stashed.original_shape, 7.0, dtype=torch.float32, device=device)
    stashed._tensor = reloaded
    stashed.reload_from_stash(buffer)
    assert buffer.free_list_tail.tolist() == [1, 0]
    assert torch.equal(reloaded, torch.full_like(reloaded, 7.0))


def test_pop_kernel_preserves_tails_when_spill_flag_set_but_host_tier_missing():
    device = 'cuda'
    num_tokens = torch.tensor([2], dtype=torch.int64, device=device)
    page_record = torch.tensor([0, 0], dtype=torch.int64, device=device)
    spilled_to_host = torch.tensor([1], dtype=torch.int64, device=device)
    overflow = torch.tensor([0], dtype=torch.int64, device=device)
    free_list_cuda = torch.arange(2, dtype=torch.int64, device=device)
    free_list_host = torch.empty(0, dtype=torch.int64, device=device)
    tail = torch.tensor([2, 0], dtype=torch.int64, device=device)
    capacity = torch.tensor([2, 0], dtype=torch.int64, device=device)
    new_tail = torch.full((2,), -1, dtype=torch.int64, device=device)
    source = torch.zeros(4, 64, device=device)
    destination = torch.full((4, 64), -1.0, device=device)

    _paged_stash_pop_kernel[(1,)](
        source,
        source,
        destination,
        num_tokens,
        page_record,
        spilled_to_host,
        overflow,
        free_list_cuda,
        free_list_host,
        tail,
        capacity,
        new_tail,
        PAGE_SIZE=2,
        HIDDEN_SIZE=64,
        BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
    )
    assert new_tail.tolist() == [2, 0]
    assert destination.min().item() == -1.0


def test_offload_round_trips_hidden_size_equal_to_block_size():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(32, 1024, 4, device, overflow, host_spill, torch.float32)
    original, paged = _paged_activation(device, tokens=6, max_tokens=8, hidden=1024, page=4)
    paged.offload_to_stash(buffer)
    assert buffer.free_list_head.tolist() == [2, 0]

    reloaded = torch.empty(paged.original_shape, dtype=torch.float32, device=device)
    paged._tensor = reloaded
    paged.reload_from_stash(buffer)
    assert torch.equal(reloaded[: 6 * 1024], original[: 6 * 1024])
    assert buffer.free_list_tail.tolist() == [10, 0]


def test_offload_divides_token_count_by_scale_inv_block_for_columnwise_tensors():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(64, 16, 16, device, overflow, host_spill, torch.float32)
    original, paged = _paged_activation(
        device, tokens=64, max_tokens=128, hidden=16, page=16, columnwise=True
    )
    paged.offload_to_stash(buffer)
    assert buffer.free_list_head.tolist() == [1, 0]

    reloaded = torch.empty(paged.original_shape, dtype=torch.float32, device=device)
    paged._tensor = reloaded
    paged.reload_from_stash(buffer)
    assert torch.equal(reloaded[:32], original[:32])
    assert buffer.free_list_tail.tolist() == [5, 0]


def test_offload_accepts_zero_dimensional_token_count_tensor():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(64, 64, 16, device, overflow, host_spill, torch.float32)
    original, paged = _paged_activation(device, tokens=32)
    paged.num_tokens_tensor = torch.tensor(32, device=device)
    assert paged.num_tokens_tensor.dim() == 0

    paged.offload_to_stash(buffer)
    assert paged.num_tokens_tensor.shape == (1,)
    reloaded = torch.empty(paged.original_shape, dtype=torch.float32, device=device)
    paged._tensor = reloaded
    paged.reload_from_stash(buffer)
    assert torch.equal(reloaded[: 32 * 64], original[: 32 * 64])


def test_offload_strides_programs_when_max_blocks_below_token_count():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(64, 64, 16, device, overflow, host_spill, torch.float32)
    original, paged = _paged_activation(device, tokens=48)
    paged.offload_to_stash(buffer, max_blocks=3)
    assert buffer.free_list_head.tolist() == [3, 0]

    reloaded = torch.empty(paged.original_shape, dtype=torch.float32, device=device)
    paged._tensor = reloaded
    paged.reload_from_stash(buffer, max_blocks=3)
    assert torch.equal(reloaded[: 48 * 64], original[: 48 * 64])


def test_offload_recycles_returned_pages_through_the_circular_free_list():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(128, 64, 16, device, overflow, host_spill, torch.float32)
    originals = []
    paged_tensors = []
    for _ in range(3):
        original, paged = _paged_activation(device, tokens=48)
        originals.append(original)
        paged_tensors.append(paged)

    paged_tensors[0].offload_to_stash(buffer)
    paged_tensors[1].offload_to_stash(buffer)
    assert buffer.free_list_head.tolist() == [6, 0]
    assert paged_tensors[1].page_record.tolist() == [3, 4, 5]

    reloaded = torch.empty(paged_tensors[0].original_shape, dtype=torch.float32, device=device)
    paged_tensors[0]._tensor = reloaded
    paged_tensors[0].reload_from_stash(buffer)
    assert torch.equal(reloaded[: 48 * 64], originals[0][: 48 * 64])

    paged_tensors[2].offload_to_stash(buffer)
    assert paged_tensors[2].page_record.tolist() == [6, 7, 0]
    assert buffer.free_list_head.tolist() == [9, 0]

    for index in (1, 2):
        reloaded = torch.empty(
            paged_tensors[index].original_shape, dtype=torch.float32, device=device
        )
        paged_tensors[index]._tensor = reloaded
        paged_tensors[index].reload_from_stash(buffer)
        assert torch.equal(reloaded[: 48 * 64], originals[index][: 48 * 64])
    assert buffer.free_list_tail.tolist() == [17, 0]


def test_reload_raises_when_the_destination_tensor_is_not_preallocated():
    overflow, host_spill = _flags('cuda')
    buffer = PagedStashBuffer(64, 64, 16, 'cuda', overflow, host_spill, torch.float32)
    _, paged = _paged_activation('cuda', tokens=48)
    paged.offload_to_stash(buffer)
    with pytest.raises(AssertionError, match='expects _tensor pre-allocated'):
        paged.reload_from_stash(buffer)


def test_reload_raises_when_the_destination_shape_differs_from_original():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(64, 64, 16, device, overflow, host_spill, torch.float32)
    _, paged = _paged_activation(device, tokens=48)
    paged.offload_to_stash(buffer)
    paged._tensor = torch.empty(4, 8, dtype=torch.float32, device=device)
    with pytest.raises(AssertionError, match='original_shape'):
        paged.reload_from_stash(buffer)


def test_reload_raises_when_the_destination_dtype_differs_from_original():
    device = 'cuda'
    overflow, host_spill = _flags(device)
    buffer = PagedStashBuffer(64, 64, 16, device, overflow, host_spill, torch.float32)
    _, paged = _paged_activation(device, tokens=48)
    paged.offload_to_stash(buffer)
    paged._tensor = torch.empty(paged.original_shape, dtype=torch.float16, device=device)
    with pytest.raises(AssertionError):
        paged.reload_from_stash(buffer)
