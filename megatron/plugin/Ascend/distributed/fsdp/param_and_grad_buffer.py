"""FSDP gradient reduction compatible with HCCL."""

import torch


@torch.no_grad()
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
    """Apply gradient scaling without the NCCL-only PREMUL_SUM operation."""
    if scaling_factor is None:
        return torch.distributed.ReduceOp.SUM
    if ddp_config.average_in_collective:
        return torch.distributed.ReduceOp.AVG

    # HCCL requires explicit scaling even when gradient_reduce_div_fusion is enabled.
    grad_data.mul_(scaling_factor)
    return torch.distributed.ReduceOp.SUM
