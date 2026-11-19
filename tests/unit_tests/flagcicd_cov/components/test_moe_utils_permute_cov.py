# Copyright (c) 2026, BAAI. All rights reserved.
"""MoE token movement: expert-grouped permutation, drop-and-pad capacity layout, unpermute with probability merging, and chunk reordering."""

import pytest
import torch

from megatron.core.transformer.moe import moe_utils


@pytest.fixture
def deterministic_algorithms():
    """Enable torch deterministic algorithms, restoring the previous mode afterwards."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def test_permute_groups_tokens_by_expert_in_stable_order():
    tokens = torch.arange(8.0).reshape(4, 2)
    routing_map = torch.zeros(4, 2, dtype=torch.bool)
    routing_map[0, 0] = True
    routing_map[1, 1] = True
    routing_map[2, 0] = True
    routing_map[3, 1] = True
    probs = torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.3, 0.0], [0.0, 0.4]])
    permuted, permuted_probs, sorted_indices, pad_offsets, tokens_per_expert = moe_utils.permute(
        tokens, routing_map, probs=probs, num_out_tokens=4
    )
    assert permuted.tolist() == [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0], [6.0, 7.0]]
    torch.testing.assert_close(permuted_probs, torch.tensor([0.1, 0.3, 0.2, 0.4]))
    assert sorted_indices.tolist() == [0, 2, 1, 3]
    assert pad_offsets is None
    assert tokens_per_expert is None


def test_permute_pads_each_expert_to_capacity_with_drop_and_pad():
    tokens = torch.arange(6.0).reshape(3, 2)
    routing_map = torch.zeros(3, 2, dtype=torch.bool)
    routing_map[0, 0] = True
    routing_map[1, 1] = True
    routing_map[2, 1] = True
    probs = torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.0, 0.4]])
    permuted, permuted_probs, sorted_indices, _, _ = moe_utils.permute(
        tokens, routing_map, probs=probs, num_out_tokens=4, drop_and_pad=True
    )
    assert permuted.tolist() == [[0.0, 1.0], [2.0, 3.0], [2.0, 3.0], [4.0, 5.0]]
    torch.testing.assert_close(permuted_probs, torch.tensor([0.1, 0.0, 0.2, 0.4]))
    assert sorted_indices.tolist() == [0, 1, 1, 2]


def test_permute_requires_num_out_tokens_in_argsort_mode():
    tokens = torch.arange(4.0).reshape(2, 2)
    routing_map = torch.ones(2, 1, dtype=torch.bool)
    with pytest.raises(AssertionError, match='num_out_tokens is required'):
        moe_utils.permute(tokens, routing_map)


def test_permute_fused_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_permute', None, raising=False)
    with pytest.raises(ValueError, match='fused_permute is not available'):
        moe_utils.permute(torch.ones(2, 4), torch.ones(2, 2, dtype=torch.bool), fused=True)


def test_permute_fused_with_probs_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_permute_and_pad_with_probs', None, raising=False)
    monkeypatch.setattr(moe_utils, 'fused_permute_with_probs', None, raising=False)
    with pytest.raises(
        ValueError, match=r'Transformer Engine \(TE\) fused kernel is not available'
    ):
        moe_utils.permute(
            torch.ones(2, 4), torch.ones(2, 2, dtype=torch.bool), probs=torch.ones(2, 2), fused=True
        )


def test_unpermute_restores_tokens_and_applies_probs():
    tokens = torch.arange(8.0).reshape(4, 2)
    routing_map = torch.zeros(4, 2, dtype=torch.bool)
    routing_map[0, 0] = True
    routing_map[1, 1] = True
    routing_map[2, 0] = True
    routing_map[3, 1] = True
    probs = torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.3, 0.0], [0.0, 0.4]])
    permuted, permuted_probs, sorted_indices, _, _ = moe_utils.permute(
        tokens, routing_map, probs=probs, num_out_tokens=4
    )
    restored = moe_utils.unpermute(
        permuted, sorted_indices, tokens.shape, probs=probs, routing_map=routing_map
    )
    expected = tokens * probs.max(dim=1).values.unsqueeze(-1)
    torch.testing.assert_close(restored, expected)


def test_unpermute_drop_and_pad_restores_tokens_and_applies_probs():
    tokens = torch.arange(6.0).reshape(3, 2)
    routing_map = torch.zeros(3, 2, dtype=torch.bool)
    routing_map[0, 0] = True
    routing_map[1, 1] = True
    routing_map[2, 1] = True
    probs = torch.tensor([[0.5, 0.0], [0.0, 0.25], [0.0, 0.75]])
    permuted, _, sorted_indices, _, _ = moe_utils.permute(
        tokens, routing_map, probs=probs, num_out_tokens=4, drop_and_pad=True
    )
    restored = moe_utils.unpermute(
        permuted,
        sorted_indices,
        tokens.shape,
        probs=probs,
        routing_map=routing_map,
        drop_and_pad=True,
    )
    expected = tokens * probs.max(dim=1).values.unsqueeze(-1)
    torch.testing.assert_close(restored, expected)


def test_unpermute_requires_routing_map_with_probs():
    permuted = torch.ones(2, 2)
    with pytest.raises(AssertionError, match='Mask must be provided to permute the probs'):
        moe_utils.unpermute(permuted, torch.tensor([0, 1]), (2, 2), probs=torch.ones(2, 2))


def test_unpermute_uses_index_add_when_deterministic_algorithms_enabled(deterministic_algorithms):
    tokens = torch.arange(8.0).reshape(4, 2)
    routing_map = torch.zeros(4, 2, dtype=torch.bool)
    routing_map[0, 0] = True
    routing_map[1, 1] = True
    routing_map[2, 0] = True
    routing_map[3, 1] = True
    permuted, _, sorted_indices, _, _ = moe_utils.permute(tokens, routing_map, num_out_tokens=4)
    restored = moe_utils.unpermute(permuted, sorted_indices, tokens.shape)
    torch.testing.assert_close(restored, tokens)


def test_unpermute_fused_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_unpermute', None, raising=False)
    with pytest.raises(ValueError, match='fused_unpermute is not available'):
        moe_utils.unpermute(torch.ones(2, 2), torch.tensor([0, 1]), (2, 2), fused=True)


def test_sort_chunks_by_idxs_reorders_chunks_and_probs():
    chunks = torch.arange(6.0).reshape(6, 1)
    probs = torch.arange(6.0).reshape(6, 1)
    output, permuted_probs = moe_utils.sort_chunks_by_idxs(
        chunks, torch.tensor([1, 2, 3]), torch.tensor([2, 0, 1]), probs=probs
    )
    assert output.tolist() == [[3.0], [4.0], [5.0], [0.0], [1.0], [2.0]]
    assert permuted_probs.tolist() == [[3.0], [4.0], [5.0], [0.0], [1.0], [2.0]]


def test_sort_chunks_by_idxs_fused_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_sort_chunks_by_index', None, raising=False)
    with pytest.raises(ValueError, match='fused_sort_chunks_by_index is not available'):
        moe_utils.sort_chunks_by_idxs(
            torch.ones(4, 1), torch.tensor([2, 2]), torch.tensor([0, 1]), fused=True
        )


def test_sort_chunks_by_idxs_fused_with_probs_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_sort_chunks_by_index_with_probs', None, raising=False)
    with pytest.raises(ValueError, match='fused_sort_chunks_by_index_with_probs is not available'):
        moe_utils.sort_chunks_by_idxs(
            torch.ones(4, 1),
            torch.tensor([2, 2]),
            torch.tensor([0, 1]),
            probs=torch.ones(4, 1),
            fused=True,
        )
