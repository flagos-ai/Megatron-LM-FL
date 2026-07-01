# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Utility functions for Context Parallel support in CompressedSparseAttention."""

from functools import lru_cache

import torch

# ---------------------------------------------------------------------------
# P2P halo exchange
# ---------------------------------------------------------------------------


def _exchange_halo(
    tensor: torch.Tensor, halo_size: int, cp_group: torch.distributed.ProcessGroup, name: str = ""
) -> torch.Tensor:
    """P2P halo exchange for context parallelism.

    Each rank sends its last `halo_size` tokens to the next rank, and receives
    `halo_size` tokens from the previous rank to prepend.

    Args:
        tensor: [local_seq, ...] local tensor (KV or hidden_states).
        halo_size: number of tokens to exchange.
        cp_group: context parallel process group.
        name: optional name for debug logging.

    Returns:
        tensor_with_halo: [halo_size + local_seq, ...] for rank > 0,
                          [local_seq, ...] for rank 0 (no halo needed).
    """
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()

    if cp_size == 1:
        return tensor

    global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    prev_rank = global_ranks[(cp_rank - 1) % cp_size]
    next_rank = global_ranks[(cp_rank + 1) % cp_size]

    send_buf = tensor[-halo_size:].contiguous()
    recv_buf = torch.empty_like(send_buf)

    ops = []
    if cp_rank < cp_size - 1:
        ops.append(torch.distributed.isend(send_buf, dst=next_rank, group=cp_group))
    if cp_rank > 0:
        ops.append(torch.distributed.irecv(recv_buf, src=prev_rank, group=cp_group))

    for op in ops:
        op.wait()

    if cp_rank == 0:
        return tensor
    else:
        return torch.cat([recv_buf, tensor], dim=0)


# ---------------------------------------------------------------------------
# Differentiable all-gather
# ---------------------------------------------------------------------------


class _AllGatherAlongFirstDim(torch.autograd.Function):
    """Differentiable all-gather along dim 0 for context parallel compressed KV."""

    @staticmethod
    def forward(ctx, input_, cp_group):
        """Forward pass: all-gather input along dim 0."""
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_group.rank()
        ctx.cp_size = cp_group.size()
        ctx.input_size_0 = input_.size(0)

        gathered = [torch.empty_like(input_) for _ in range(ctx.cp_size)]
        torch.distributed.all_gather(gathered, input_.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: slice gradient back to current rank's portion."""
        chunk_size = ctx.input_size_0
        start = ctx.cp_rank * chunk_size
        grad_input = grad_output[start : start + chunk_size].contiguous()
        return grad_input, None


def differentiable_all_gather(tensor, cp_group):
    """All-gather along first dim with gradient support."""
    return _AllGatherAlongFirstDim.apply(tensor, cp_group)


# ---------------------------------------------------------------------------
# Async differentiable all-gather (for compute/communication overlap)
# ---------------------------------------------------------------------------

_ASYNC_AG_HANDLE = None


class _AsyncAllGatherAlongFirstDim(torch.autograd.Function):
    """Async differentiable all-gather: launches non-blocking all-gather in forward.

    The caller must invoke async_differentiable_all_gather_wait() before
    reading the returned output buffer.
    """

    @staticmethod
    def forward(ctx, input_, cp_group):
        global _ASYNC_AG_HANDLE
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_group.rank()
        ctx.cp_size = cp_group.size()
        ctx.input_size_0 = input_.size(0)

        output = torch.empty(
            ctx.cp_size * input_.size(0), *input_.shape[1:],
            dtype=input_.dtype, device=input_.device,
        )
        _ASYNC_AG_HANDLE = torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=cp_group, async_op=True
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        chunk_size = ctx.input_size_0
        start = ctx.cp_rank * chunk_size
        grad_input = grad_output[start : start + chunk_size].contiguous()
        return grad_input, None


def async_differentiable_all_gather_start(tensor, cp_group):
    """Launch async all-gather. Returns output buffer (not yet valid).

    Call async_differentiable_all_gather_wait() before reading the buffer.
    """
    return _AsyncAllGatherAlongFirstDim.apply(tensor, cp_group)


def async_differentiable_all_gather_wait():
    """Wait for the most recent async all-gather to complete."""
    global _ASYNC_AG_HANDLE
    if _ASYNC_AG_HANDLE is not None:
        _ASYNC_AG_HANDLE.wait()
        _ASYNC_AG_HANDLE = None


# ---------------------------------------------------------------------------
# CP-aware sliding window indices
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_window_topk_idxs_cp_cached(
    window_size: int, local_seq_len: int, halo_size: int, device_str: str
) -> torch.Tensor:
    """Compute sliding-window indices with halo offset for CP (cached).

    KV layout after halo exchange: [halo(halo_size) | local(local_seq_len)]
    For local query position i, its position in kv_with_halo is i + halo_size.
    Its window covers [i + halo_size - window_size + 1, i + halo_size].

    Returns:
        indices: [local_seq_len, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(local_seq_len, device=device_str).unsqueeze(1) + halo_size
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs_cp(
    window_size: int, batch_size: int, local_seq_len: int, halo_size: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices with CP halo offset [batch, local_seq_len, window_size]."""
    matrix = _get_window_topk_idxs_cp_cached(window_size, local_seq_len, halo_size, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)
