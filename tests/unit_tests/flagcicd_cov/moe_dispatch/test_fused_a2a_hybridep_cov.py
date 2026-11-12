# Copyright (c) 2026, BAAI. All rights reserved.
"""HybridEP fused dispatch and combine: optional-argument plumbing into HybridEPBuffer, fused-permute fallback warnings, buffer init/reset lifecycle, and gradient round trips."""

import pytest
import torch

import megatron.core.transformer.moe.fused_a2a as fused_a2a


class _RecordingGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeHybridEPBuffer:
    constructed = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.dispatch_calls = []
        self.combine_calls = []
        self.handle = object()
        _FakeHybridEPBuffer.constructed.append(self)

    def dispatch_with_permute(
        self,
        hidden=None,
        routing_map=None,
        probs=None,
        scaling_factor=None,
        num_of_experts_per_rank=None,
        pad_multiple=None,
        num_permuted_tokens=None,
        non_blocking=None,
        handle=None,
        fuse_permute_dispatch=False,
    ):
        self.dispatch_calls.append(
            dict(
                hidden=hidden,
                routing_map=routing_map,
                probs=probs,
                scaling_factor=scaling_factor,
                num_of_experts_per_rank=num_of_experts_per_rank,
                pad_multiple=pad_multiple,
                num_permuted_tokens=num_permuted_tokens,
                non_blocking=non_blocking,
                handle=handle,
                fuse_permute_dispatch=fuse_permute_dispatch,
            )
        )
        return (
            hidden * 2,
            probs * 0.5 if probs is not None else None,
            torch.tensor(1.0),
            torch.tensor([2, 4]),
            self.handle,
        )

    def combine_with_unpermute(
        self, hidden=None, probs=None, handle=None, pad_multiple=None, fuse_unpermute_combine=False
    ):
        self.combine_calls.append(
            dict(
                hidden=hidden,
                probs=probs,
                handle=handle,
                pad_multiple=pad_multiple,
                fuse_unpermute_combine=fuse_unpermute_combine,
            )
        )
        return hidden * 3, probs * 2.0 if probs is not None else None


class _UnfusedHybridEPBuffer(_FakeHybridEPBuffer):

    def dispatch_with_permute(
        self,
        hidden=None,
        routing_map=None,
        probs=None,
        scaling_factor=None,
        num_of_experts_per_rank=None,
        pad_multiple=None,
        num_permuted_tokens=None,
        non_blocking=None,
        handle=None,
    ):
        call = dict(
            hidden=hidden,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=scaling_factor,
            num_of_experts_per_rank=num_of_experts_per_rank,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            handle=handle,
        )
        self.dispatch_calls.append(call)
        return (
            hidden * 2,
            probs * 0.5 if probs is not None else None,
            torch.tensor(1.0),
            torch.tensor([2, 4]),
            self.handle,
        )


@pytest.fixture
def hybridep_buffer(monkeypatch):
    _FakeHybridEPBuffer.constructed.clear()
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", _FakeHybridEPBuffer, raising=False)
    return _FakeHybridEPBuffer


def _routing_inputs():
    hidden = torch.randn(4, 8)
    routing_map = torch.zeros(4, 2, dtype=torch.bool)
    probs = torch.rand(4, 2)
    return hidden, routing_map, probs


@pytest.mark.parametrize(
    "optional_args,expected_extra_kwargs",
    [
        pytest.param({}, {}, id="defaults-pass-only-base-kwargs"),
        pytest.param(
            {"num_sms_dispatch_api": 24},
            {"num_sms_dispatch_api": 24},
            id="dispatch-sm-count-plumbed-through",
        ),
        pytest.param(
            {"num_sms_combine_api": 16, "num_blocks_permute": 3},
            {"num_sms_combine_api": 16, "num_blocks_permute": 3},
            id="combine-sm-and-permute-blocks-plumbed-through",
        ),
        pytest.param(
            {"num_blocks_unpermute": 5, "num_sms_preprocessing_api": 108},
            {"num_blocks_unpermute": 5, "num_sms_preprocessing_api": 108},
            id="unpermute-blocks-and-preprocessing-sm-plumbed-through",
        ),
    ],
)
def test_init_hybrid_ep_buffer_forwards_optional_args_to_hybridep_constructor(
    hybridep_buffer, optional_args, expected_extra_kwargs
):
    group = _RecordingGroup(2)
    fused_a2a.init_hybrid_ep_buffer(
        group, hidden_dim=8, num_tokens=4, num_local_experts=2, **optional_args
    )
    buffer = fused_a2a._hybrid_ep_buffer
    assert isinstance(buffer, _FakeHybridEPBuffer)
    assert buffer.init_kwargs == dict(
        group=group,
        hidden_dim=8,
        max_num_of_tokens_per_rank=4,
        num_local_experts=2,
        use_fp8=False,
        **expected_extra_kwargs,
    )


def test_init_hybrid_ep_buffer_rejects_fp8_dispatch(hybridep_buffer):
    with pytest.raises(AssertionError, match="HybridEP dispatcher does not support fp8 dispatch"):
        fused_a2a.init_hybrid_ep_buffer(
            group=None, hidden_dim=8, num_tokens=4, num_local_experts=2, fp8_dispatch=True
        )


def test_reset_hybrid_ep_buffer_clears_an_initialized_buffer(hybridep_buffer):
    fused_a2a.init_hybrid_ep_buffer(
        _RecordingGroup(2), hidden_dim=8, num_tokens=4, num_local_experts=2
    )
    buffer = fused_a2a._hybrid_ep_buffer
    assert isinstance(buffer, _FakeHybridEPBuffer)
    assert buffer.init_kwargs["max_num_of_tokens_per_rank"] == 4
    fused_a2a.reset_hybrid_ep_buffer()
    assert fused_a2a._hybrid_ep_buffer is None


def test_hybrid_ep_dispatch_forward_initializes_buffer_lazily_and_permutes(hybridep_buffer):
    hidden, routing_map, probs = _routing_inputs()
    group = _RecordingGroup(2)

    result = fused_a2a.HybridEPDispatch.apply(
        hidden, routing_map, probs, group, 2, 24, 16, 3, 5, False, 16, 4, 108
    )

    buffer = fused_a2a._hybrid_ep_buffer
    assert buffer.init_kwargs == dict(
        group=group,
        hidden_dim=8,
        max_num_of_tokens_per_rank=4,
        num_local_experts=2,
        use_fp8=False,
        num_sms_dispatch_api=24,
        num_sms_combine_api=16,
        num_blocks_permute=3,
        num_blocks_unpermute=5,
        num_sms_preprocessing_api=108,
    )
    call = buffer.dispatch_calls[0]
    assert call["hidden"] is hidden
    assert call["routing_map"] is routing_map
    assert call["probs"] is probs
    assert call["scaling_factor"] is None
    assert call["num_of_experts_per_rank"] == 2
    assert call["pad_multiple"] == 4
    assert call["num_permuted_tokens"] == 16
    assert call["non_blocking"] is True
    assert call["fuse_permute_dispatch"] is False
    assert torch.equal(result[0], hidden * 2)
    assert torch.equal(result[1], probs * 0.5)
    assert torch.equal(result[2], torch.tensor(1.0))
    assert torch.equal(result[3], torch.tensor([2, 4]))
    assert result[4] is buffer.handle


def test_hybrid_ep_dispatch_forward_passes_fused_flag_when_supported(hybridep_buffer):
    hidden, routing_map, probs = _routing_inputs()

    fused_a2a.HybridEPDispatch.apply(
        hidden, routing_map, probs, _RecordingGroup(2), 2, None, None, None, None, True, None, None
    )

    buffer = fused_a2a._hybrid_ep_buffer
    assert buffer.dispatch_calls[0]["fuse_permute_dispatch"] is True


def test_hybrid_ep_dispatch_forward_falls_back_when_fused_permute_unsupported(monkeypatch):
    _FakeHybridEPBuffer.constructed.clear()
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", _UnfusedHybridEPBuffer, raising=False)
    hidden, routing_map, probs = _routing_inputs()

    with pytest.warns(UserWarning, match="does not support fused permute dispatch"):
        result = fused_a2a.HybridEPDispatch.apply(
            hidden, routing_map, probs, _RecordingGroup(2), 2, None, None, 3, 5, True, None, None
        )

    buffer = fused_a2a._hybrid_ep_buffer
    assert "num_blocks_permute" not in buffer.init_kwargs
    assert "num_blocks_unpermute" not in buffer.init_kwargs
    call = buffer.dispatch_calls[0]
    assert "fuse_permute_dispatch" not in call
    assert call["non_blocking"] is False
    assert torch.equal(result[0], hidden * 2)


def test_hybrid_ep_dispatch_backward_unpermutes_and_combines_gradients(hybridep_buffer):
    hidden = torch.randn(4, 8, requires_grad=True)
    routing_map = torch.zeros(4, 2, dtype=torch.bool)
    probs = torch.rand(4, 2, requires_grad=True)

    dispatched_hidden, dispatched_probs, _, _, _ = fused_a2a.HybridEPDispatch.apply(
        hidden, routing_map, probs, _RecordingGroup(2), 2, None, None, None, None, False, 16, 4
    )
    (dispatched_hidden.sum() + dispatched_probs.sum()).backward()

    buffer = fused_a2a._hybrid_ep_buffer
    call = buffer.combine_calls[0]
    assert torch.equal(call["hidden"], torch.ones(4, 8))
    assert torch.equal(call["probs"], torch.ones(4, 2))
    assert call["handle"] is buffer.handle
    assert call["pad_multiple"] == 4
    assert call["fuse_unpermute_combine"] is False
    assert torch.equal(hidden.grad, torch.full((4, 8), 3.0))
    assert torch.equal(probs.grad, torch.full((4, 2), 2.0))


def test_hybrid_ep_combine_forward_unpermutes_through_buffer(hybridep_buffer):
    hidden = torch.randn(6, 8)
    fused_a2a.init_hybrid_ep_buffer(
        _RecordingGroup(2), hidden_dim=8, num_tokens=6, num_local_experts=2
    )
    buffer = fused_a2a._hybrid_ep_buffer

    combined = fused_a2a.HybridEPCombine.apply(hidden, buffer.handle, 6, 4, False)

    call = buffer.combine_calls[0]
    assert call["hidden"] is hidden
    assert call["handle"] is buffer.handle
    assert call["pad_multiple"] == 4
    assert call["fuse_unpermute_combine"] is False
    assert torch.equal(combined, hidden * 3)


def test_hybrid_ep_combine_forward_and_backward_pass_fused_flags(hybridep_buffer):
    hidden = torch.randn(6, 8, requires_grad=True)
    fused_a2a.init_hybrid_ep_buffer(
        _RecordingGroup(2), hidden_dim=8, num_tokens=6, num_local_experts=2
    )
    buffer = fused_a2a._hybrid_ep_buffer

    combined = fused_a2a.HybridEPCombine.apply(hidden, buffer.handle, 6, 4, True)
    combined.sum().backward()

    assert buffer.combine_calls[0]["fuse_unpermute_combine"] is True
    assert buffer.dispatch_calls[0]["handle"] is buffer.handle
    assert buffer.dispatch_calls[0]["pad_multiple"] == 4
    assert buffer.dispatch_calls[0]["num_permuted_tokens"] == 6
    assert buffer.dispatch_calls[0]["fuse_permute_dispatch"] is True
    assert torch.equal(hidden.grad, torch.full((6, 8), 2.0))


def test_hybrid_ep_combine_backward_permutes_gradients_through_buffer(hybridep_buffer):
    hidden = torch.randn(6, 8, requires_grad=True)
    fused_a2a.init_hybrid_ep_buffer(
        _RecordingGroup(2), hidden_dim=8, num_tokens=6, num_local_experts=2
    )
    buffer = fused_a2a._hybrid_ep_buffer

    combined = fused_a2a.HybridEPCombine.apply(hidden, buffer.handle, 6, 4, False)
    combined.sum().backward()

    call = buffer.dispatch_calls[0]
    assert torch.equal(call["hidden"], torch.ones(6, 8))
    assert call["scaling_factor"] is None
    assert call["fuse_permute_dispatch"] is False
    assert torch.equal(hidden.grad, torch.full((6, 8), 2.0))
