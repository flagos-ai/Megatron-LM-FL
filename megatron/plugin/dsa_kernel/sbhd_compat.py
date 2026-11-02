# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adapt the retained SBHD Triton facade to the public CSA index ABI."""

from typing import Optional, Tuple

from torch import Tensor


def build_flat_topk_idxs(
    *idx_groups: Tensor, batch_size: int, compact: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    """Globalize SBHD indices without loading the cuDNN compaction provider."""
    from .triton_dsa_kernels import build_flat_topk_idxs as build

    # The legacy argument is retained by the public facade but is not used in
    # its seq-major index conversion. Physical KV indices are already supplied.
    return build(*idx_groups, batch_size=batch_size, seqlen_kv=0, compact=compact)
