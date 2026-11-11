# Copyright (c) 2026, BAAI. All rights reserved.
"""fully_shard_model rejects invalid sharding strategies; fully_shard_optimizer requires FSDP-wrapped params."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    ShardingStrategy,
    fully_shard_model,
    fully_shard_optimizer,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module")
def cuda_mesh():
    Utils.initialize_model_parallel()
    yield init_device_mesh(
        "cuda", mesh_shape=(torch.distributed.get_world_size(), 1), mesh_dim_names=("fsdp", "tp")
    )
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "zero_dp_strategy",
    [
        ShardingStrategy.NO_SHARD,
        ShardingStrategy.OPTIM,
        ShardingStrategy.OPTIM_GRADS,
        ShardingStrategy.OPTIM_GRADS_PARAMS,
    ],
    ids=["no_shard", "optim", "optim_grads", "optim_grads_params"],
)
def test_fully_shard_model_converts_strategy_enums_before_outer_validation(
    cuda_mesh, zero_dp_strategy
):
    with pytest.raises(ValueError, match="Invalid Hybrid DP-Outer Sharding Strategy"):
        fully_shard_model(
            torch.nn.Linear(4, 4),
            device_mesh=cuda_mesh,
            zero_dp_strategy=zero_dp_strategy,
            outer_dp_sharding_strategy="invalid_outer",
        )


def test_fully_shard_model_rejects_invalid_inner_strategy(cuda_mesh):
    with pytest.raises(ValueError, match="Invalid FSDP / Inner DP Sharding Strategy"):
        fully_shard_model(
            torch.nn.Linear(4, 4), device_mesh=cuda_mesh, zero_dp_strategy="invalid_inner"
        )


def test_fully_shard_model_rejects_invalid_outer_strategy_enum(cuda_mesh):
    with pytest.raises(ValueError, match="Invalid Hybrid DP-Outer Sharding Strategy"):
        fully_shard_model(
            torch.nn.Linear(4, 4),
            device_mesh=cuda_mesh,
            zero_dp_strategy=ShardingStrategy.OPTIM,
            outer_dp_sharding_strategy=ShardingStrategy.OPTIM_GRADS,
        )


def test_fully_shard_model_outer_sharding_requires_full_inner_sharding(cuda_mesh):
    with pytest.raises(ValueError, match="Sharding with Hybrid (Fully) Sharded Data Parallel"):
        fully_shard_model(
            torch.nn.Linear(4, 4),
            device_mesh=cuda_mesh,
            zero_dp_strategy="optim",
            outer_dp_sharding_strategy=ShardingStrategy.OPTIM,
        )


@pytest.mark.parametrize(
    "dp_outer_dim,hybrid_fsdp_group",
    [("dp_outer", None), (None, torch.distributed.group.WORLD)],
    ids=["missing_hybrid_group", "missing_dp_outer_dim"],
)
def test_fully_shard_model_requires_dp_outer_and_hybrid_group_together(
    cuda_mesh, dp_outer_dim, hybrid_fsdp_group
):
    with pytest.raises(ValueError, match="must be specified together for Hybrid FSDP"):
        fully_shard_model(
            torch.nn.Linear(4, 4),
            device_mesh=cuda_mesh,
            zero_dp_strategy="optim_grads_params",
            dp_outer_dim=dp_outer_dim,
            hybrid_fsdp_group=hybrid_fsdp_group,
        )


def test_fully_shard_model_rejects_meta_device_with_no_shard(cuda_mesh):
    with pytest.raises(ValueError, match="Meta device initialization"):
        fully_shard_model(
            torch.nn.Linear(4, 4),
            device_mesh=cuda_mesh,
            zero_dp_strategy=ShardingStrategy.NO_SHARD,
            init_model_with_meta_device=True,
        )


def test_fully_shard_optimizer_rejects_params_without_fsdp_reference():
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([param], lr=0.1)
    with pytest.raises(ValueError, match="Could not retrieve a reference to MegatronFSDP"):
        fully_shard_optimizer(optimizer)
