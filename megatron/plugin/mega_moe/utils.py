# Copyright (c) FlagOS Team, BAAI Corporation.
# Utility functions for MegaMoE plugin.

import torch
import torch.distributed as dist
from typing import Tuple


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def uneven_all_gather(
    tensor: torch.Tensor, dim: int = 0, group: dist.ProcessGroup = None
) -> torch.Tensor:
    """All-gather tensors that may have different sizes along `dim` across ranks."""
    world_size = dist.get_world_size(group)

    # Exchange sizes
    local_dim_size = torch.tensor(
        [tensor.shape[dim]], device=tensor.device, dtype=torch.long
    )
    all_dim_sizes = [torch.zeros_like(local_dim_size) for _ in range(world_size)]
    dist.all_gather(all_dim_sizes, local_dim_size, group=group)
    all_dim_sizes = [s.item() for s in all_dim_sizes]
    max_dim_size = max(all_dim_sizes)

    # Pad to max size
    if tensor.shape[dim] < max_dim_size:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_dim_size - tensor.shape[dim]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat([tensor, padding], dim=dim)
    else:
        tensor_padded = tensor.contiguous()

    # All-gather
    gathered = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered, tensor_padded, group=group)

    # Remove padding
    trimmed = [
        torch.narrow(gathered[i], dim, 0, all_dim_sizes[i]) for i in range(world_size)
    ]
    return torch.cat(trimmed, dim=dim)
