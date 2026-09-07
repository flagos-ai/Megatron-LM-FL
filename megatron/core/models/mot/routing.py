# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token routing parameters and validation for Mixture-of-Transformers models."""

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch import Tensor


@dataclass
class MoTPackedSeqParams:
    """Packed-sequence metadata and token indexes for MoT branches.

    Empty branch tensors are valid, but all indexes must form a disjoint
    partition of every token position.
    """

    branch_token_indexes: dict = None
    sample_lens: list[int] = None
    split_lens: list[int] = None
    attn_modes: list[str] = None
    block_mask: Any = None


def validate_branch_token_indexes(
    hidden_states: Tensor,
    branch_token_indexes: Dict[str, Tensor],
    branch_names: Sequence[str],
    packed_seq_params: Any,
) -> None:
    """Validate that MoT branch indexes form a disjoint partition of all tokens."""
    if not isinstance(branch_token_indexes, dict):
        raise TypeError(
            "branch_token_indexes must be a dict mapping branch names to index tensors, "
            f"but got {type(branch_token_indexes).__name__}"
        )

    branch_names = tuple(branch_names)
    validation_key = (
        hidden_states.size(0),
        hidden_states.device,
        branch_names,
        tuple(
            (name, indexes.data_ptr(), indexes.numel())
            for name, indexes in branch_token_indexes.items()
            if isinstance(indexes, Tensor)
        ),
    )
    if getattr(packed_seq_params, '_mot_routing_validation_key', None) == validation_key:
        return

    configured_names = set(branch_names)
    actual_names = set(branch_token_indexes)
    unknown_names = actual_names - configured_names
    if unknown_names:
        raise ValueError(
            f"Unknown MoT branches: {sorted(unknown_names, key=str)}; "
            f"configured branches are {list(branch_names)}"
        )

    for name, indexes in branch_token_indexes.items():
        if not isinstance(indexes, Tensor):
            raise TypeError(
                f'MoT branch "{name}" indexes must be a torch.Tensor, '
                f"but got {type(indexes).__name__}"
            )
        if indexes.dim() != 1:
            raise ValueError(
                f'MoT branch "{name}" indexes must be 1-D, '
                f"but got shape {tuple(indexes.shape)}"
            )
        if indexes.dtype != torch.long:
            raise TypeError(
                f'MoT branch "{name}" indexes must have dtype torch.long, '
                f"but got {indexes.dtype}"
            )
        if indexes.device != hidden_states.device:
            raise ValueError(
                f'MoT branch "{name}" indexes are on {indexes.device}, '
                f"but hidden states are on {hidden_states.device}"
            )

    all_indexes = torch.cat(list(branch_token_indexes.values()))
    expected_indexes = torch.arange(
        hidden_states.size(0),
        device=hidden_states.device,
        dtype=torch.long,
    )
    if all_indexes.numel() != expected_indexes.numel() or not torch.equal(
        all_indexes.sort().values,
        expected_indexes,
    ):
        raise ValueError(
            "MoT branch indexes must form a disjoint partition of "
            f"[0, {hidden_states.size(0)})"
        )

    setattr(packed_seq_params, '_mot_routing_validation_key', validation_key)


__all__ = ["MoTPackedSeqParams", "validate_branch_token_indexes"]
