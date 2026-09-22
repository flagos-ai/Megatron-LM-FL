"""Keep FSDP gradient averaging independent of NCCL-specific reduction operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )


@torch.no_grad()
def gradient_reduce_preprocessing(
    grad_data: torch.Tensor,
    scaling_factor: float | None,
    ddp_config: DistributedDataParallelConfig,
) -> torch.distributed.ReduceOp:
    """Preserve averaging semantics without passing NCCL PREMUL_SUM to MCCL."""
    if scaling_factor is None:
        return torch.distributed.ReduceOp.SUM
    if ddp_config.average_in_collective:
        return torch.distributed.ReduceOp.AVG

    # The torch_musa 2.7.1 backend rejects NCCL PREMUL_SUM with "Unexpected ReduceOp"
    # in both all-reduce and coalesced reduce-scatter. Scaling before SUM keeps
    # rank averaging correct, including the existing unfused and BF16 paths.
    grad_data.mul_(scaling_factor)
    return torch.distributed.ReduceOp.SUM
