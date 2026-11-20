import torch

from megatron.core.tensor_parallel import mappings
from megatron.core.utils import get_tensor_model_parallel_group_if_none


def all_gather_last_dim_from_tensor_parallel_region(input_, group=None, *, output_split_count=1):
    """Gather only the shards consumed by each consecutive subset of TP ranks."""
    group = get_tensor_model_parallel_group_if_none(group)
    world_size = group.size()
    # More than two low-precision replica gradients change native RS rounding.
    # Select collectives independently of rank-local autograd state.
    if (
        input_.device.type != "npu"
        or output_split_count <= 1
        or world_size % output_split_count
        or (
            input_.dtype in (torch.float16, torch.bfloat16) and world_size // output_split_count > 2
        )
    ):
        return mappings.all_gather_last_dim_from_tensor_parallel_region.__wrapped__(
            input_, group=group, output_split_count=output_split_count
        )

    replicas = world_size // output_split_count
    partition = group.rank() // replicas
    splits = [input_.shape[0] if rank // replicas == partition else 0 for rank in range(world_size)]
    packed = input_.repeat((replicas,) + (1,) * (input_.ndim - 1))
    received = mappings.all_to_all(group, packed, splits, splits)
    # AllToAll reverses the splits in backward; repeat sums the replica gradients.
    return torch.cat(received.chunk(replicas, dim=0), dim=-1)
