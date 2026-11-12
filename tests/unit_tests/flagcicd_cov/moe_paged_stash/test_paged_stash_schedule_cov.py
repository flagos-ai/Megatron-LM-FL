# Copyright (c) 2026, BAAI. All rights reserved.
"""Interleaved forward/backward schedule driven through the paged stash group functions."""

from contextlib import nullcontext

import pytest
import torch

from megatron.core.transformer.moe.paged_stash import (
    PagedStashContext,
    PagedStashManager,
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    get_paged_stash_context,
    paged_stash_group_commit,
    paged_stash_group_start,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.transformer_config import TransformerConfig

_HIDDEN = 64
_MAX_TOKENS = 128
_ACTUAL_TOKENS = 100
_PAGE = 16

_SCHEDULE = [
    ('forward', 1, 1),
    ('forward', 2, 1),
    ('backward', 2, 1),
    ('forward', 1, 2),
    ('backward', 1, 1),
    ('forward', 2, 2),
    ('backward', 2, 2),
    ('backward', 1, 2),
]


@pytest.fixture(autouse=True)
def _fresh_stash_manager(monkeypatch):
    """Give every test a pristine singleton and undo the swap afterwards."""
    monkeypatch.setattr(PagedStashManager, 'STASH_MGR', None, raising=False)
    yield


def _stash_config():
    return TransformerConfig(
        num_layers=2,
        hidden_size=_HIDDEN,
        num_attention_heads=8,
        moe_paged_stash_page_size=_PAGE,
        moe_paged_stash_buffer_size_factor_cuda=2.0,
        moe_paged_stash_buffer_size_factor_cpu=0.0,
    )


class _TokenSquare(torch.autograd.Function):
    """z = t[:num_tokens]^2 whose backward reads the saved (possibly stash-reloaded) rows."""

    @staticmethod
    def forward(ctx, tensor, num_tokens):
        ctx.save_for_backward(tensor)
        ctx.num_tokens = num_tokens
        return tensor[:num_tokens] * tensor[:num_tokens]

    @staticmethod
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        grad_input = torch.zeros_like(tensor)
        grad_input[: ctx.num_tokens] = 2 * grad_output * tensor[: ctx.num_tokens]
        return grad_input, None


def _schedule_inputs(device, base):
    return {
        (layer, microbatch): base[layer, microbatch].clone().requires_grad_(True)
        for layer in (1, 2)
        for microbatch in (1, 2)
    }


def _run_stash_iteration(inputs, device):
    """One interleaved 1F1B iteration through the group_start/commit autograd functions."""
    outputs = {}
    num_tokens_tensor = torch.tensor([_ACTUAL_TOKENS], device=device)
    current_microbatch = None
    for kind, layer, microbatch in _SCHEDULE:
        if kind == 'forward':
            if microbatch != current_microbatch:
                paged_stash_init_chunk_handler(1, 0)
                current_microbatch = microbatch
            leaf = inputs[layer, microbatch]
            with get_paged_stash_context(
                name=f'moe_layer_{layer}',
                max_num_tokens=_MAX_TOKENS,
                num_tokens_tensor=num_tokens_tensor,
                avg_num_tokens=_ACTUAL_TOKENS,
            ):
                staged = paged_stash_group_start(leaf)
                staged.grouped_tensor_scale_inv = False
                activation = _TokenSquare.apply(staged, _ACTUAL_TOKENS)
                outputs[layer, microbatch] = paged_stash_group_commit(activation)
        else:
            outputs[layer, microbatch].sum().backward()
    return outputs


def _run_reference_iteration(inputs):
    outputs = {}
    for kind, layer, microbatch in _SCHEDULE:
        if kind == 'forward':
            leaf = inputs[layer, microbatch]
            outputs[layer, microbatch] = leaf[:_ACTUAL_TOKENS] * leaf[:_ACTUAL_TOKENS]
        else:
            outputs[layer, microbatch].sum().backward()
    return outputs


def _run_stash_training_step(device):
    """Run a reference pass, a capture pass, and a captured pass over the same activations."""
    torch.manual_seed(123)
    base = {
        (layer, microbatch): torch.randn(_MAX_TOKENS, _HIDDEN, device=device)
        for layer in (1, 2)
        for microbatch in (1, 2)
    }
    reference_inputs = _schedule_inputs(device, base)
    reference_outputs = _run_reference_iteration(reference_inputs)

    config = _stash_config()
    captured_inputs = None
    captured_outputs = None
    for _ in range(2):
        paged_stash_reset(True, config=config)
        captured_inputs = _schedule_inputs(device, base)
        captured_outputs = _run_stash_iteration(captured_inputs, device)
    manager = PagedStashManager.get_instance()
    return manager, captured_outputs, captured_inputs, reference_outputs, reference_inputs


def test_paged_stash_round_trips_the_interleaved_schedule_outputs():
    _, outputs, _, reference_outputs, _ = _run_stash_training_step('cuda')
    for key, value in outputs.items():
        torch.testing.assert_close(value, reference_outputs[key], rtol=0.0, atol=0.0)


def test_paged_stash_round_trips_input_gradients_after_reload():
    _, _, inputs, _, reference_inputs = _run_stash_training_step('cuda')
    for key, value in inputs.items():
        torch.testing.assert_close(value.grad, reference_inputs[key].grad, rtol=0.0, atol=0.0)


def test_paged_stash_iteration_leaves_no_pending_work_behind():
    manager, outputs, inputs, reference_outputs, reference_inputs = _run_stash_training_step('cuda')
    assert manager.status == 'captured'
    assert manager.paged_tensors_to_stash == []
    assert manager.paged_tensors_stash_in_progress == []
    assert set(manager.paged_tensors_to_reload) == {1001001, 1001002}
    assert all(len(queue) == 0 for queue in manager.paged_tensors_to_reload.values())
    assert manager._pack_stream_status == 'idle'
    assert manager._unpack_stream_status == 'idle'


def test_paged_stash_iteration_returns_all_pages_to_the_free_list():
    manager, outputs, inputs, reference_outputs, reference_inputs = _run_stash_training_step('cuda')
    buffer = manager.stash_buffers[torch.float32][_HIDDEN]
    assert buffer.free_list_head.tolist() == [14, 0]
    assert buffer.free_list_tail.tolist() == [39, 0]
    assert buffer.free_list_cuda.tolist() == list(range(25))
    assert check_paged_stash_overflow().item() == 0
    assert check_paged_stash_host_spill().item() == 0


def test_paged_stash_iteration_records_the_schedule_in_order():
    manager, *_ = _run_stash_training_step('cuda')
    assert manager._pp_schedule == [
        1001001,
        1002001,
        -1002001,
        1001002,
        -1001001,
        1002002,
        -1002002,
        -1001002,
    ]
    assert manager.current_schedule_index == len(manager._pp_schedule)


def test_group_start_returns_tensor_unchanged_when_disabled():
    activation = torch.randn(4, device='cuda')
    assert paged_stash_group_start(activation) is activation


def test_group_commit_records_device_and_returns_tensor_when_disabled():
    activation = torch.randn(4, device='cuda')
    assert paged_stash_group_commit(activation) is activation
    assert PagedStashManager.get_instance().device == activation.device


def test_group_commit_rejects_use_before_the_chunk_handler_is_initialized():
    paged_stash_reset(True, config=_stash_config())
    activation = torch.randn(4, device='cuda')
    with pytest.raises(AssertionError):
        paged_stash_group_commit(activation)


def test_get_paged_stash_context_returns_nullcontext_when_disabled():
    manager = PagedStashManager.get_instance()
    assert manager.enabled is False
    context = get_paged_stash_context(
        name='moe', max_num_tokens=8, num_tokens_tensor=torch.tensor([4], device='cuda')
    )
    assert type(context) is nullcontext

    retrieved = []

    class _RecordSavedTensor(torch.autograd.Function):
        """y = 2x whose backward records the object autograd hands back from save."""

        @staticmethod
        def forward(ctx, tensor):
            ctx.save_for_backward(tensor)
            return tensor * 2.0

        @staticmethod
        def backward(ctx, grad_output):
            retrieved.append(ctx.saved_tensors[0])
            return grad_output * 2.0

    marked = torch.randn(8, 4, device='cuda', requires_grad=True)
    marked.grouped_tensor_scale_inv = False
    with context as entered:
        assert entered is None
        _RecordSavedTensor.apply(marked).sum().backward()
    assert len(retrieved) == 1
    assert retrieved[0] is marked
    assert manager.max_num_tokens is None
    assert manager.avg_num_tokens is None
    assert manager.num_tokens_tensor is None
    assert manager._current_layer_name is None


def test_get_paged_stash_context_configures_the_manager_when_enabled():
    paged_stash_reset(True, config=_stash_config())
    manager = PagedStashManager.get_instance()
    num_tokens_tensor = torch.tensor([_ACTUAL_TOKENS], device='cuda')
    context = get_paged_stash_context(
        name='moe',
        max_num_tokens=_MAX_TOKENS,
        num_tokens_tensor=num_tokens_tensor,
        avg_num_tokens=90,
    )
    assert isinstance(context, PagedStashContext)
    assert manager.max_num_tokens == _MAX_TOKENS
    assert manager.avg_num_tokens == 90
    assert manager.num_tokens_tensor is not num_tokens_tensor
    assert torch.equal(manager.num_tokens_tensor, num_tokens_tensor)
    assert manager._current_layer_name == 'moe'

    with context:
        pass


def test_get_paged_stash_context_keeps_layer_name_when_name_omitted():
    paged_stash_reset(True, config=_stash_config())
    manager = PagedStashManager.get_instance()
    get_paged_stash_context(name='moe', max_num_tokens=8, num_tokens_tensor=torch.tensor([4]))
    context = get_paged_stash_context(max_num_tokens=8, num_tokens_tensor=torch.tensor([4]))
    assert isinstance(context, PagedStashContext)
    assert manager._current_layer_name == 'moe'


def test_get_paged_stash_context_requires_a_tensor_token_count():
    paged_stash_reset(True, config=_stash_config())
    with pytest.raises(AssertionError):
        get_paged_stash_context(name='moe', max_num_tokens=8, num_tokens_tensor=None)


def test_check_paged_stash_overflow_returns_false_tensor_when_disabled():
    flag = check_paged_stash_overflow()
    assert flag.dtype == torch.bool
    assert flag.shape == (1,)
    assert flag.device.type == 'cuda'
    assert flag.item() == 0


def test_check_paged_stash_host_spill_returns_false_tensor_when_disabled():
    flag = check_paged_stash_host_spill()
    assert flag.dtype == torch.bool
    assert flag.shape == (1,)
    assert flag.device.type == 'cuda'
    assert flag.item() == 0
