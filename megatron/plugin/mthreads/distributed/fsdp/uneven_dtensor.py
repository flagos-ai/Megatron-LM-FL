# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Avoid unequal-size MCCL gathers when reconstructing uneven DTensor shards.

The validated MCCL runtime rejects the core's unequal-length payload gather
with ``invalid argument``, including layouts with empty ranks. Padding belongs
in this vendor override so other platforms retain the native core behavior.
"""

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    update_uneven_dtensor_chunk_metadata,
)


def _all_gather_uneven(
    tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: dist.ProcessGroup | None = None
) -> None:
    """Use equal payload sizes where native all_gather cannot accept uneven shards.

    MCCL and Gloo need equal-sized buffers on this path. The caller has already
    exchanged every rank's true size, so all ranks can derive the same padded
    length without another collective or a rank-local decision to skip work.
    """
    # Restrict padding to the affected backends; let native code retain its
    # own behavior (including validation of an empty output list) elsewhere.
    if dist.get_backend(group) not in ("mccl", "gloo") or not tensor_list:
        return dist.all_gather(tensor_list, tensor, group=group)

    max_numel = max(output.numel() for output in tensor_list)
    # Only a globally empty payload can skip the collective. A locally empty
    # rank must still participate when another rank has data, or peers can hang.
    if max_numel == 0 and tensor.numel() == 0:
        return None
    # Equal shapes already satisfy the backend contract; padding would only
    # add allocations and copies without fixing a compatibility problem.
    if all(output.shape == tensor.shape for output in tensor_list):
        return dist.all_gather(tensor_list, tensor, group=group)

    padded_input = tensor.reshape(-1)
    if tensor.numel() < max_numel:
        padded_input = tensor.new_zeros(max_numel)
        padded_input[: tensor.numel()].copy_(tensor.reshape(-1))
    padded_outputs = [tensor.new_empty(max_numel) for _ in tensor_list]
    dist.all_gather(padded_outputs, padded_input, group=group)
    # Padding is transport-only: reconstruction uses the original shard sizes.
    # Trim into the caller's buffers so neither padding nor buffer replacement
    # can change the metadata-to-payload correspondence.
    for output, padded in zip(tensor_list, padded_outputs):
        output.copy_(padded[: output.numel()].view_as(output))


def uneven_dtensor_to_full_tensor(dtensor: DTensor) -> torch.Tensor:
    """Preserve core shard reconstruction while adapting its payload collective.

    Uneven shards cannot be reconstructed from an assumed uniform partition.
    Keep the core's true shapes and offsets, changing only the gather transport
    needed by MCCL/Gloo; padding must not alter checkpoint layout or placement.
    """
    if not isinstance(dtensor, DTensor):
        raise TypeError(f"Input must be a DTensor, got {type(dtensor).__name__}.")

    # True offsets are needed for uneven shards; deriving them from a uniform
    # partition would place data incorrectly, especially when ranks are empty.
    if not hasattr(dtensor._local_tensor, "__create_chunk_list__"):
        update_uneven_dtensor_chunk_metadata(dtensor)

    # The local buffer below represents one chunk. Accepting multiple metadata
    # entries would associate its bytes with an ambiguous shape and offset.
    chunk_metadata_list = dtensor.__create_chunk_list__()
    if len(chunk_metadata_list) != 1:
        raise ValueError(
            f"Expected exactly one chunk metadata per rank, got {len(chunk_metadata_list)}."
        )
    local_chunk_metadata = chunk_metadata_list[0]

    local_chunks_info = [
        {
            "shape": dtensor.to_local().shape,
            "offset": getattr(local_chunk_metadata, "offsets", [0] * len(dtensor.shape)),
        }
    ]
    local_buffer = dtensor.to_local().contiguous().view(-1)

    # A multidimensional mesh needs one gather per sharded dimension. Replicate
    # dimensions already contain the same data and must not duplicate chunks.
    for mesh_dim, placement in enumerate(dtensor.placements):
        if isinstance(placement, (Shard, _StridedShard)):
            shard_group = dtensor.device_mesh.get_group(mesh_dim)

            # Previous mesh dimensions may have merged several chunks per rank.
            # Exchange their true shapes before allocating payload buffers so
            # every rank agrees on lengths, including empty contributors.
            group_chunks_info = [None] * shard_group.size()
            dist.all_gather_object(group_chunks_info, local_chunks_info, group=shard_group)

            group_tensors = [
                torch.empty(
                    sum(chunk["shape"].numel() for chunk in chunks_info),
                    dtype=dtensor.dtype,
                    device=dtensor.device,
                )
                for chunks_info in group_chunks_info
            ]

            _all_gather_uneven(group_tensors, local_buffer, group=shard_group)

            # Keep metadata in the same rank/chunk order as the gathered bytes;
            # the final split uses this correspondence to recover each chunk.
            local_chunks_info = [item for sublist in group_chunks_info for item in sublist]
            local_buffer = torch.cat(group_tensors)
        elif not isinstance(placement, Replicate):
            raise ValueError(
                f"Unexpected placement {placement} at mesh dimension {mesh_dim}. "
                f"Expected Shard, _StridedShard, or Replicate."
            )

    # Only true chunk sizes belong in reconstruction; transport padding was
    # discarded before concatenation, so it cannot shift subsequent chunks.
    all_local_chunks = []
    buffer_offset = 0
    for chunk_info in local_chunks_info:
        chunk_shape = chunk_info["shape"]
        chunk_numel = chunk_shape.numel()
        chunk_tensor = local_buffer[buffer_offset : buffer_offset + chunk_numel].view(chunk_shape)
        all_local_chunks.append(chunk_tensor)
        buffer_offset += chunk_numel

    # Rank order alone does not describe uneven or strided placement. Recorded
    # offsets determine where each chunk belongs in the global tensor.
    full_tensor = torch.zeros(dtensor.shape, dtype=dtensor.dtype, device=dtensor.device)
    for chunk_info, local_chunk in zip(local_chunks_info, all_local_chunks):
        offset = chunk_info["offset"]
        slices = tuple(slice(o, o + s) for o, s in zip(offset, local_chunk.shape))
        full_tensor[slices] = local_chunk

    return full_tensor
