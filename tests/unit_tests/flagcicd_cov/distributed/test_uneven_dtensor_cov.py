# Copyright (c) 2026, BAAI. All rights reserved.
"""uneven_dtensor gathers per-rank chunk metadata and redistributes uneven shards into the full tensor."""

import pytest
import torch
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import _StridedShard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    gather_and_compute_chunk_metadata,
    gather_uneven_dtensor_to_full_tensor,
    get_unflattened_state_dict,
    redistribute_uneven_dtensor_to_replicated,
    validate_uneven_dtensor,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _make_global_arange(rows, cols):
    generator = torch.Generator().manual_seed(77)
    tensor = torch.arange(rows * cols, dtype=torch.float32, generator=generator).reshape(rows, cols)
    return tensor.cuda()


def _make_unevenly_sharded_dtensor():
    """Shard rows unevenly (rows % world_size != 0) across the world-size mesh."""
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    mesh = DeviceMesh("cuda", list(range(world_size)))
    rows = world_size * 3 + 2
    global_tensor = _make_global_arange(rows, 4)
    local_list, _ = Shard(0)._split_tensor(
        global_tensor, world_size, with_padding=False, contiguous=True
    )
    local = local_list[rank]
    return (
        DTensor.from_local(
            local, mesh, (Shard(0),), shape=global_tensor.size(), stride=global_tensor.stride()
        ),
        global_tensor,
    )


def test_gather_chunk_metadata_handles_strided_shard_placement(model_parallel):
    model_parallel()
    rank = torch.distributed.get_rank()
    mesh = DeviceMesh("cuda", torch.arange(8).reshape(2, 4))
    local = torch.arange(4.0).reshape(2, 2).cuda()

    dtensor = DTensor.from_local(
        local, mesh, (Shard(0), _StridedShard(1, split_factor=4)), shape=(4, 8), stride=(8, 1)
    )

    chunk_metadata = gather_and_compute_chunk_metadata(dtensor)

    mesh_row, mesh_col = rank // 4, rank % 4
    assert chunk_metadata.sizes == (2, 2)
    assert chunk_metadata.offsets == (mesh_row * 2, mesh_col * 2)


def test_gather_chunk_metadata_rejects_multiple_strided_shards(model_parallel):
    model_parallel()
    mesh = DeviceMesh("cuda", torch.arange(8).reshape(2, 4))
    local = torch.arange(4.0).reshape(2, 2).cuda()

    dtensor = DTensor.from_local(
        local,
        mesh,
        (_StridedShard(0, split_factor=2), _StridedShard(0, split_factor=2)),
        shape=(4, 8),
        stride=(8, 1),
    )

    with pytest.raises(ValueError, match="DTensor has multiple strided shards"):
        gather_and_compute_chunk_metadata(dtensor)


def test_validate_uneven_dtensor_accepts_uneven_shards(model_parallel):
    model_parallel()
    dtensor, _ = _make_unevenly_sharded_dtensor()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    validate_uneven_dtensor(dtensor)

    # The shards are genuinely uneven: 3 * world_size + 2 rows split so that the
    # first two ranks hold one extra row, which the metadata must reflect.
    local_rows = 4 if rank < 2 else 3
    expected_offset = 4 * min(rank, 2) + 3 * max(rank - 2, 0)
    chunk_metadata = gather_and_compute_chunk_metadata(dtensor)
    assert chunk_metadata.sizes == (local_rows, 4)
    assert chunk_metadata.offsets == (expected_offset, 0)
    assert 3 * world_size + 2 == 26


def test_redistribute_uneven_dtensor_to_replicated_restores_full_tensor(model_parallel):
    model_parallel()
    dtensor, global_tensor = _make_unevenly_sharded_dtensor()

    replicated = redistribute_uneven_dtensor_to_replicated(dtensor)

    assert replicated.placements == (Replicate(),)
    torch.testing.assert_close(replicated.to_local(), global_tensor, rtol=0, atol=0)


def test_gather_uneven_dtensor_to_full_tensor_matches_redistributed(model_parallel):
    model_parallel()
    dtensor, global_tensor = _make_unevenly_sharded_dtensor()

    gathered = gather_uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.placements == (Replicate(),)
    torch.testing.assert_close(gathered.to_local(), global_tensor, rtol=0, atol=0)


def test_get_unflattened_state_dict_navigates_nested_keys():
    state_dict = {"model": {"layers": {"0": torch.ones(2)}}}

    assert (
        get_unflattened_state_dict(state_dict, ["model", "layers", "0"])
        is state_dict["model"]["layers"]["0"]
    )
    with pytest.raises(KeyError, match=r"Key \['model', 'missing'\] not found in state_dict"):
        get_unflattened_state_dict(state_dict, ["model", "missing"])
