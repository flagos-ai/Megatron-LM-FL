# Copyright (c) 2026, BAAI. All rights reserved.
"""DeepEP dispatch/combine autograd functions: hidden-byte sizing, buffer reuse across calls, and layout/dispatch/combine plumbing recorded at the deep_ep Buffer boundary."""

import pytest
import torch

import megatron.core.transformer.moe.fused_a2a as fused_a2a


class _RecordingGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeEventHandle:
    pass


class _FakeEventOverlap:
    def __init__(self, handle=None):
        self.handle = handle
        self.waits = 0

    def current_stream_wait(self):
        self.waits += 1


class _FakeSizeConfig:
    def __init__(self, nvl_scale, rdma_scale):
        self.nvl_scale = nvl_scale
        self.rdma_scale = rdma_scale

    def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
        return hidden_bytes * self.nvl_scale

    def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
        return hidden_bytes * self.rdma_scale


class _FakeDeepEPBuffer:
    constructed = []

    def __init__(self, group, num_nvl_bytes, num_rdma_bytes):
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.layout_call = None
        self.layout_event = None
        self.dispatch_call = None
        self.dispatch_after = None
        self.combine_calls = []
        self.combine_after = None
        self.returned_handle = _FakeEventHandle()
        _FakeDeepEPBuffer.constructed.append(self)

    @classmethod
    def get_dispatch_config(cls, group_size):
        return _FakeSizeConfig(nvl_scale=2, rdma_scale=1)

    @classmethod
    def get_combine_config(cls, group_size):
        return _FakeSizeConfig(nvl_scale=3, rdma_scale=2)

    def get_dispatch_layout(
        self,
        token_indices,
        num_experts,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        self.layout_call = dict(
            token_indices=token_indices,
            num_experts=num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.layout_event = _FakeEventOverlap()
        return 1, 2, 3, 4, self.layout_event

    def dispatch(
        self,
        x,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        previous_event=None,
        handle=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        self.dispatch_call = dict(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            handle=handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.dispatch_after = _FakeEventOverlap()
        if handle is not None:
            return x * 2, None, None, None, None, self.dispatch_after
        return (
            x * 2,
            torch.zeros_like(topk_idx),
            topk_weights * 0.5,
            [3, 5],
            self.returned_handle,
            self.dispatch_after,
        )

    def combine(
        self,
        x,
        handle=None,
        topk_weights=None,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        self.combine_calls.append(
            dict(
                x=x,
                handle=handle,
                topk_weights=topk_weights,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        self.combine_after = _FakeEventOverlap()
        if topk_weights is None:
            return x * 3, None, self.combine_after
        return x * 0.5, topk_weights * 3.0, self.combine_after


@pytest.fixture
def deep_ep_boundary(monkeypatch):
    _FakeDeepEPBuffer.constructed.clear()
    monkeypatch.setattr(fused_a2a, "_buffer", None, raising=False)
    monkeypatch.setattr(fused_a2a, "Buffer", _FakeDeepEPBuffer, raising=False)
    monkeypatch.setattr(fused_a2a, "EventHandle", _FakeEventHandle, raising=False)
    monkeypatch.setattr(fused_a2a, "EventOverlap", _FakeEventOverlap, raising=False)


@pytest.mark.parametrize(
    "dtype,element_bytes,expected_hidden_bytes",
    [
        pytest.param(torch.float32, 4, 32, id="float32-uses-element-size"),
        pytest.param(torch.float16, 2, 16, id="float16-uses-element-size"),
        pytest.param(torch.bool, 1, 16, id="one-byte-dtype-padded-to-two-bytes"),
    ],
)
def test_get_hidden_bytes_counts_two_bytes_per_hidden_element(
    dtype, element_bytes, expected_hidden_bytes
):
    x = torch.zeros(4, 8, dtype=dtype)
    assert x.element_size() == element_bytes
    assert fused_a2a.get_hidden_bytes(x) == expected_hidden_bytes


def test_get_buffer_sizes_buffer_to_max_dispatch_and_combine_hints(deep_ep_boundary):
    group = _RecordingGroup(2)
    buffer = fused_a2a.get_buffer(group, 32)
    assert buffer.num_nvl_bytes == max(32 * 2, 32 * 3)
    assert buffer.num_rdma_bytes == max(32 * 1, 32 * 2)
    assert buffer.group is group


def test_get_buffer_reuses_cached_buffer_until_requested_size_grows(deep_ep_boundary):
    group = _RecordingGroup(2)
    first = fused_a2a.get_buffer(group, 32)
    assert fused_a2a.get_buffer(group, 32) is first
    assert fused_a2a.get_buffer(group, 16) is first
    grown = fused_a2a.get_buffer(group, 128)
    assert grown is not first
    assert grown.num_nvl_bytes == max(128 * 2, 128 * 3)
    assert len(_FakeDeepEPBuffer.constructed) == 2


def test_fused_dispatch_forward_lays_out_tokens_and_dispatches_without_event_wait(deep_ep_boundary):
    x = torch.randn(4, 8)
    token_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    token_probs = torch.rand(4, 2)
    group = _RecordingGroup(2)

    result = fused_a2a.FusedDispatch.apply(x, token_indices, token_probs, 8, group, False, False)

    buffer = fused_a2a._buffer
    assert buffer.layout_call["token_indices"] is token_indices
    assert buffer.layout_call["num_experts"] == 8
    assert buffer.layout_call["previous_event"] is None
    assert buffer.layout_call["async_finish"] is False
    assert buffer.dispatch_call["x"] is x
    assert buffer.dispatch_call["topk_idx"] is token_indices
    assert buffer.dispatch_call["topk_weights"] is token_probs
    assert buffer.dispatch_call["num_tokens_per_rank"] == 1
    assert buffer.dispatch_call["num_tokens_per_rdma_rank"] == 2
    assert buffer.dispatch_call["num_tokens_per_expert"] == 3
    assert buffer.dispatch_call["is_token_in_rank"] == 4
    assert buffer.dispatch_call["previous_event"] is buffer.layout_event
    assert torch.equal(result[0], x * 2)
    assert torch.equal(result[1], torch.zeros_like(token_indices))
    assert torch.equal(result[2], token_probs * 0.5)
    assert torch.equal(result[3], torch.tensor([3, 5]))
    assert result[4] is buffer.returned_handle
    assert buffer.dispatch_after.waits == 0


def test_fused_dispatch_forward_async_creates_overlap_event_and_waits(deep_ep_boundary):
    x = torch.randn(4, 8)
    token_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    token_probs = torch.rand(4, 2)
    group = _RecordingGroup(2)

    result = fused_a2a.FusedDispatch.apply(x, token_indices, token_probs, 8, group, True, True)

    buffer = fused_a2a._buffer
    previous_event = buffer.layout_call["previous_event"]
    assert isinstance(previous_event, _FakeEventOverlap)
    assert isinstance(previous_event.handle, _FakeEventHandle)
    assert buffer.layout_call["async_finish"] is True
    assert buffer.layout_call["allocate_on_comm_stream"] is True
    assert buffer.dispatch_after.waits == 1
    assert result[4] is buffer.returned_handle


def test_fused_dispatch_backward_combines_gradients_through_cached_buffer(deep_ep_boundary):
    x = torch.randn(4, 8, requires_grad=True)
    token_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    token_probs = torch.rand(4, 2, requires_grad=True)
    group = _RecordingGroup(2)

    recv_x, _, recv_probs, _, _ = fused_a2a.FusedDispatch.apply(
        x, token_indices, token_probs, 8, group, False, False
    )
    (recv_x.sum() + recv_probs.sum()).backward()

    buffer = fused_a2a._buffer
    assert len(buffer.combine_calls) == 1
    combine = buffer.combine_calls[0]
    assert torch.equal(combine["x"], torch.ones(4, 8))
    assert torch.equal(combine["topk_weights"], torch.ones(4, 2))
    assert combine["handle"] is buffer.returned_handle
    assert combine["async_finish"] is False
    assert torch.equal(x.grad, torch.full((4, 8), 0.5))
    assert torch.equal(token_probs.grad, torch.full((4, 2), 3.0))


def test_fused_dispatch_backward_async_waits_on_combine_event(deep_ep_boundary):
    x = torch.randn(4, 8, requires_grad=True)
    token_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    token_probs = torch.rand(4, 2, requires_grad=True)
    group = _RecordingGroup(2)

    recv_x, _, recv_probs, _, _ = fused_a2a.FusedDispatch.apply(
        x, token_indices, token_probs, 8, group, True, False
    )
    (recv_x.sum() + recv_probs.sum()).backward()

    buffer = fused_a2a._buffer
    combine = buffer.combine_calls[0]
    assert isinstance(combine["previous_event"], _FakeEventOverlap)
    assert combine["async_finish"] is True
    assert buffer.combine_after.waits == 1
    assert torch.equal(x.grad, torch.full((4, 8), 0.5))


def test_fused_combine_forward_combines_without_event_wait(deep_ep_boundary):
    x = torch.randn(6, 8)
    group = _RecordingGroup(2)
    handle = _FakeEventHandle()

    combined_x, handle_grad = fused_a2a.FusedCombine.apply(x, group, handle, False, False)

    buffer = fused_a2a._buffer
    combine = buffer.combine_calls[0]
    assert combine["x"] is x
    assert combine["handle"] is handle
    assert combine["topk_weights"] is None
    assert combine["previous_event"] is None
    assert combine["async_finish"] is False
    assert torch.equal(combined_x, x * 3)
    assert handle_grad is None
    assert buffer.combine_after.waits == 0


def test_fused_combine_forward_async_waits_on_combine_event(deep_ep_boundary):
    x = torch.randn(6, 8)
    group = _RecordingGroup(2)

    fused_a2a.FusedCombine.apply(x, group, _FakeEventHandle(), True, True)

    buffer = fused_a2a._buffer
    combine = buffer.combine_calls[0]
    assert isinstance(combine["previous_event"], _FakeEventOverlap)
    assert combine["async_finish"] is True
    assert buffer.combine_after.waits == 1


def test_fused_combine_backward_dispatches_gradients_through_cached_buffer(deep_ep_boundary):
    x = torch.randn(6, 8, requires_grad=True)
    group = _RecordingGroup(2)
    handle = _FakeEventHandle()

    combined_x, _ = fused_a2a.FusedCombine.apply(x, group, handle, False, False)
    combined_x.sum().backward()

    buffer = fused_a2a._buffer
    dispatch = buffer.dispatch_call
    assert dispatch["handle"] is handle
    assert dispatch["async_finish"] is False
    assert torch.equal(dispatch["x"], torch.ones(6, 8))
    assert torch.equal(x.grad, torch.full((6, 8), 2.0))


def test_fused_combine_backward_async_waits_on_dispatch_event(deep_ep_boundary):
    x = torch.randn(6, 8, requires_grad=True)
    group = _RecordingGroup(2)

    combined_x, _ = fused_a2a.FusedCombine.apply(x, group, _FakeEventHandle(), True, False)
    combined_x.sum().backward()

    buffer = fused_a2a._buffer
    assert isinstance(buffer.dispatch_call["previous_event"], _FakeEventOverlap)
    assert buffer.dispatch_after.waits == 1
    assert torch.equal(x.grad, torch.full((6, 8), 2.0))
