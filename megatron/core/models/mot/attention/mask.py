# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utility functions for Mixture-of-Transformers (MoT) models.

Provides flex_attention mask construction and tensor padding helpers.
"""

from typing import List

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    or_masks,
)


def _pad_to_length(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    """Pad *tensor* with zeros along *dim* so that ``tensor.size(dim) == target_len``."""
    pad_size = target_len - tensor.size(dim)
    if pad_size <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = pad_size
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


def create_sparse_mask(
    document_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    device: torch.device,
):
    """Build a flex_attention mask function for MoT packed sequences.

    Constructs a composite mask that combines:
    - causal masking
    - full/noise bidirectional masking within the same sample
    - cross-noise isolation between different noise sequences
    - per-document (sample) boundary isolation

    Args:
        document_lens: Length of each document in the packed sequence.
        split_lens: Length of each segment (text, image_und, latent, etc.).
        attn_modes: Attention mode for each segment ('causal', 'full', 'noise').
        device: Device to create tensors on.

    Returns:
        A mask function suitable for ``create_block_mask``.
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (
            full_and_noise_seq_id[q_idx] >= 0
        )

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~(
            (noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])
        )

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        value = i if mode in ('full', 'noise') else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if mode == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.tensor(full_and_noise_tmp, device=device)
    noise_seq_id = torch.tensor(noise_tmp, device=device)

    document_id = torch.cat(
        [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
    ).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def create_packed_block_mask(
    sample_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    device: torch.device,
):
    """Build the flex-attention BlockMask for a packed MoT sequence."""
    mask_fn = create_sparse_mask(
        document_lens=sample_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
        device=device,
    )
    mask_len = sum(sample_lens)
    return create_block_mask(
        mask_fn,
        B=1,
        H=None,
        Q_LEN=mask_len,
        KV_LEN=mask_len,
        device=device,
        BLOCK_SIZE=128,
    )
