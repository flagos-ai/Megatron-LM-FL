"""Ascend implementations for narrow MoE utility override points."""

from typing import Optional, Tuple

import torch


class _IndexedChunkReorder(torch.autograd.Function):
    """Reorder rows with a bijection and apply its inverse in backward."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        probs: Optional[torch.Tensor],
        row_permutation: torch.Tensor,
        inverse_permutation: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ctx.save_for_backward(inverse_permutation)
        output = input.index_select(0, row_permutation)
        permuted_probs = (
            probs.index_select(0, row_permutation) if probs is not None else None
        )
        return output, permuted_probs

    @staticmethod
    def backward(ctx, grad_output, grad_permuted_probs):
        (inverse_permutation,) = ctx.saved_tensors
        grad_input = (
            grad_output.index_select(0, inverse_permutation)
            if grad_output is not None
            else None
        )
        grad_probs = (
            grad_permuted_probs.index_select(0, inverse_permutation)
            if grad_permuted_probs is not None
            else None
        )
        return grad_input, grad_probs, None, None


def _validate_chunk_metadata(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate the bijection required by the indexed NPU implementation."""
    if split_sizes.ndim != 1 or sorted_idxs.ndim != 1:
        raise ValueError("split_sizes and sorted_idxs must be one-dimensional")
    if split_sizes.numel() != sorted_idxs.numel():
        raise ValueError("sorted_idxs must contain exactly one entry for each chunk")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if split_sizes.dtype not in integer_dtypes or sorted_idxs.dtype not in integer_dtypes:
        raise TypeError("split_sizes and sorted_idxs must have integer dtype")

    counts = split_sizes.detach().to(device="cpu", dtype=torch.long)
    chunk_order = sorted_idxs.detach().to(device="cpu", dtype=torch.long)
    if torch.any(counts < 0):
        raise ValueError("split_sizes must be non-negative")
    if int(counts.sum().item()) != input.shape[0]:
        raise ValueError("sum(split_sizes) must equal input.shape[0]")
    if probs is not None and probs.shape[0] != input.shape[0]:
        raise ValueError("probs and input must have the same leading dimension")

    num_chunks = counts.numel()
    expected = torch.arange(num_chunks, dtype=torch.long)
    if num_chunks and not torch.equal(torch.sort(chunk_order).values, expected):
        raise ValueError("sorted_idxs must be a permutation of all chunk indices")
    return counts, chunk_order


def _build_row_permutations(
    counts: torch.Tensor, chunk_order: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build row permutation and inverse from small CPU chunk metadata."""
    num_rows = int(counts.sum().item())
    if num_rows == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    sorted_counts = counts.index_select(0, chunk_order)
    chunk_ids = torch.repeat_interleave(chunk_order, sorted_counts, output_size=num_rows)
    sorted_offsets = sorted_counts.cumsum(0) - sorted_counts
    offsets_within_chunks = torch.arange(num_rows) - torch.repeat_interleave(
        sorted_offsets, sorted_counts, output_size=num_rows
    )
    source_offsets = counts.cumsum(0) - counts
    row_permutation = source_offsets.index_select(0, chunk_ids) + offsets_within_chunks

    inverse_permutation = torch.empty_like(row_permutation)
    inverse_permutation[row_permutation] = torch.arange(num_rows)
    permutation_pair = torch.stack((row_permutation, inverse_permutation)).to(
        device=device, non_blocking=True
    )
    return permutation_pair.unbind(0)


def _indexed_sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Ascend indexed chunk reorder with an explicit bijection contract."""
    counts, chunk_order = _validate_chunk_metadata(input, split_sizes, sorted_idxs, probs)
    row_permutation, inverse_permutation = _build_row_permutations(
        counts, chunk_order, input.device
    )
    return _IndexedChunkReorder.apply(input, probs, row_permutation, inverse_permutation)


def _sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Use indexed reorder for NPU tensors; preserve the original path elsewhere.

    The public core function still owns the explicit TE fused branches. This
    override only replaces its non-fused split/list/cat implementation.
    """
    if input.device.type != "npu":
        from megatron.core.transformer.moe.moe_utils import _sort_chunks_by_idxs as reference

        return reference.__wrapped__(input, split_sizes, sorted_idxs, probs)
    return _indexed_sort_chunks_by_idxs(input, split_sizes, sorted_idxs, probs)
