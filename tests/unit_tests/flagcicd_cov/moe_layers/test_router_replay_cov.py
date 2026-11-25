# Copyright (c) 2026, BAAI. All rights reserved.
"""RouterReplay behaviour: replaying recorded top-k indices from target and backward lists,
static-buffer recording and global instance bookkeeping."""

import pytest
import torch

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


@pytest.fixture
def replay_registry():
    """Empty global RouterReplay registry per test; prior contents are restored afterwards."""
    original = list(RouterReplay.global_router_replay_instances)
    RouterReplay.global_router_replay_instances[:] = []
    yield RouterReplay.global_router_replay_instances
    RouterReplay.global_router_replay_instances[:] = original


def _default_topk(scores, topk, num_groups=None, group_topk=None):
    """Reference top-k that must not be consulted on replay branches."""
    values, indices = torch.topk(scores, topk, dim=-1)
    return values * 100.0, indices + 100


class TestGetReplayTopk:

    def test_get_replay_topk_pops_indices_from_backward_list_when_action_is_replay_backward(
        self, replay_registry
    ):
        replay = RouterReplay()
        first = torch.tensor([[0, 2], [1, 0]])
        second = torch.tensor([[2, 1], [0, 2]])
        replay.set_target_indices(first)
        replay.set_target_indices(second)
        replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        scores = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5]])

        probs, top_indices = replay.get_replay_topk(scores, 2, default_compute_topk=_default_topk)

        assert torch.equal(top_indices, first)
        assert torch.equal(probs, scores.gather(1, first))
        assert replay.replay_backward_list == [second]

        probs, top_indices = replay.get_replay_topk(scores, 2, default_compute_topk=_default_topk)

        assert torch.equal(top_indices, second)
        assert torch.equal(probs, scores.gather(1, second))
        assert replay.replay_backward_list == []

    def test_get_replay_topk_delegates_to_default_when_no_action_is_set(self, replay_registry):
        replay = RouterReplay()
        scores = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5]])

        probs, top_indices = replay.get_replay_topk(scores, 2, default_compute_topk=_default_topk)

        assert torch.equal(top_indices, torch.tensor([[1, 2], [2, 0]]) + 100)
        assert torch.equal(probs, torch.tensor([[0.5, 0.4], [0.5, 0.3]]) * 100.0)


class TestStaticBuffers:

    def test_set_global_static_buffers_assigns_per_layer_slice(self, replay_registry):
        first, second = RouterReplay(), RouterReplay()
        static_buffer = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)

        RouterReplay.set_global_static_buffers(static_buffer)

        assert torch.equal(first.static_buffer, static_buffer[:, 0, :])
        assert torch.equal(second.static_buffer, static_buffer[:, 1, :])
        assert first.static_buffer.data_ptr() == static_buffer[:, 0, :].data_ptr()

    def test_set_global_static_buffers_rejects_wrong_layer_count(self, replay_registry):
        RouterReplay()
        static_buffer = torch.zeros(4, 3, 2)

        with pytest.raises(AssertionError, match='Buffer has 3 layers but there are 1'):
            RouterReplay.set_global_static_buffers(static_buffer)

    def test_clear_global_static_buffers_resets_every_instance(self, replay_registry):
        first, second = RouterReplay(), RouterReplay()
        first_buffer = torch.zeros(4, 2, dtype=torch.long)
        second_buffer = torch.zeros(4, 2, dtype=torch.long)
        first.set_static_buffer(first_buffer)
        second.set_static_buffer(second_buffer)
        first.record_indices(torch.tensor([[1, 2], [3, 0]]))

        RouterReplay.clear_global_static_buffers()

        assert first.static_buffer is None
        assert second.static_buffer is None
        assert torch.equal(first_buffer, torch.tensor([[1, 2], [3, 0], [0, 0], [0, 0]]))
        assert torch.equal(second_buffer, torch.zeros(4, 2, dtype=torch.long))
        fresh_indices = torch.tensor([[2, 0], [1, 3]])
        first.record_indices(fresh_indices)
        assert first.recorded_topk_idx is fresh_indices

    def test_record_indices_copies_into_static_buffer_view(self, replay_registry):
        replay = RouterReplay()
        static_buffer = torch.zeros(6, 2, dtype=torch.long)
        replay.set_static_buffer(static_buffer)
        recorded = torch.tensor([[0, 3], [1, 2], [4, 0], [2, 5]])

        replay.record_indices(recorded)

        assert torch.equal(static_buffer[:4], recorded)
        assert torch.equal(static_buffer[4:], torch.zeros(2, 2, dtype=torch.long))
        assert replay.recorded_topk_idx.data_ptr() == static_buffer.data_ptr()

    def test_clear_static_buffer_drops_the_buffer(self, replay_registry):
        replay = RouterReplay()
        buffer = torch.zeros(6, 2, dtype=torch.long)
        replay.set_static_buffer(buffer)

        replay.clear_static_buffer()

        assert replay.static_buffer is None
        assert torch.equal(buffer, torch.zeros(6, 2, dtype=torch.long))
        recorded = torch.tensor([[0, 1], [2, 3]])
        replay.record_indices(recorded)
        assert replay.recorded_topk_idx is recorded
        assert torch.equal(buffer, torch.zeros(6, 2, dtype=torch.long))


def test_clear_global_router_replay_instances_empties_the_registry(replay_registry):
    RouterReplay()
    RouterReplay()

    RouterReplay.clear_global_router_replay_instances()

    assert RouterReplay.global_router_replay_instances == []
    assert replay_registry == []
