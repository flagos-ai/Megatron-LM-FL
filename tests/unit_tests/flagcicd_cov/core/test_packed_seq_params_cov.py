# Copyright (c) 2026, BAAI. All rights reserved.
"""Sequence-index precomputation for packed sequences."""

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def test_seq_idx_labels_each_sequence_and_the_trailing_remainder():
    params = PackedSeqParams(cu_seqlens_q=torch.tensor([0, 5, 7, 11]), total_tokens=16)
    assert params.seq_idx.dtype == torch.int32
    assert params.seq_idx.tolist() == [[0] * 5 + [1] * 2 + [2] * 4 + [3] * 5]


def test_padded_lengths_take_precedence_and_exact_fit_adds_no_remainder():
    assert PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 5, 7, 11]),
        cu_seqlens_q_padded=torch.tensor([0, 6, 8, 12]),
        total_tokens=12,
    ).seq_idx.tolist() == [[0] * 6 + [1] * 2 + [2] * 4]


def test_padding_beyond_total_tokens_is_clamped():
    assert PackedSeqParams(
        cu_seqlens_q_padded=torch.tensor([0, 4, 10]), total_tokens=8
    ).seq_idx.tolist() == [[0] * 4 + [1] * 6]


def test_seq_idx_is_left_unset_without_tensor_lengths_or_total_tokens():
    assert PackedSeqParams(cu_seqlens_q=torch.tensor([0, 5])).seq_idx is None
    assert PackedSeqParams(cu_seqlens_q=[0, 5], total_tokens=8).seq_idx is None
    assert PackedSeqParams(total_tokens=8).seq_idx is None
