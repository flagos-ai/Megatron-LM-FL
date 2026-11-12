# Copyright (c) 2026, BAAI. All rights reserved.
"""FSDP adapter: per-parameter parallelism detection, checkpoint key remapping, and dp/tp mesh construction."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel,
    _get_hsdp_tp_mesh,
)
from tests.unit_tests.test_utilities import Utils


class _CheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(4, 4))


def _make_adapter(config=None):
    adapter = FullyShardedDataParallel.__new__(FullyShardedDataParallel)
    torch.nn.Module.__init__(adapter)
    adapter._MODULE_TYPE_REGISTRY = FullyShardedDataParallel._MODULE_TYPE_REGISTRY
    adapter.config = (
        config if config is not None else SimpleNamespace(fp8=False, gated_linear_unit=False)
    )
    adapter.module = _CheckpointModule()
    return adapter


class RowParallelLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(8, 8))
        self.bias = nn.Parameter(torch.empty(8))


class TELinear(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.parallel_mode = mode
        self.weight = nn.Parameter(torch.empty(8, 8))
        self.bias = nn.Parameter(torch.empty(8))


class TensorParallelModule(nn.Module):
    def __init__(self, partition_dim):
        super().__init__()
        self.tensor_model_parallel = True
        self.partition_dim = partition_dim
        self.weight = nn.Parameter(torch.empty(8, 8))
        self.bias = nn.Parameter(torch.empty(8))


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_detect_parallelism_registry_row_bias_is_replicated():
    adapter = _make_adapter()
    module = RowParallelLinear()
    assert adapter._detect_parallelism_type("weight", module) == "row"
    assert adapter._detect_parallelism_type("bias", module) == "replicated"


def test_detect_parallelism_partition_dim_one_bias_is_replicated():
    adapter = _make_adapter()
    module = TensorParallelModule(partition_dim=1)
    assert adapter._detect_parallelism_type("weight", module) == "row"
    assert adapter._detect_parallelism_type("bias", module) == "replicated"


def test_detect_parallelism_te_linear_row_bias_is_replicated():
    adapter = _make_adapter()
    module = TELinear("row")
    assert adapter._detect_parallelism_type("weight", module) == "row"
    assert adapter._detect_parallelism_type("bias", module) == "replicated"


def test_detect_parallelism_layer_norm_is_replicated():
    adapter = _make_adapter()
    module = nn.LayerNorm(8)
    assert adapter._detect_parallelism_type("weight", module) == "replicated"


def test_annotate_tensor_parallelism_marks_params_and_skips_plain_modules():
    adapter = _make_adapter()
    root = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    row = RowParallelLinear()
    root.add_module("row", row)

    adapter._annotate_tensor_parallelism(root)

    assert row.weight._tensor_parallel_mode == "row"
    assert row.bias._tensor_parallel_mode == "replicated"
    assert root[1].weight._tensor_parallel_mode == "replicated"
    assert not hasattr(root[0].weight, "_tensor_parallel_mode")


def test_load_state_dict_prefixes_keys_with_module():
    adapter = _make_adapter()
    linear = adapter.module.module[0]
    new_state = {
        "0.weight": torch.full_like(linear.weight, 2.5),
        "0.bias": torch.full_like(linear.bias, -0.5),
    }

    adapter.load_state_dict(dict(new_state), strict=True)

    torch.testing.assert_close(linear.weight.data, new_state["0.weight"], rtol=0, atol=0)
    torch.testing.assert_close(linear.bias.data, new_state["0.bias"], rtol=0, atol=0)


def test_load_state_dict_skips_extra_state_and_relaxes_strict_for_fp8():
    config = SimpleNamespace(fp8=True, gated_linear_unit=False)
    adapter = _make_adapter(config=config)
    linear = adapter.module.module[0]
    new_weight = torch.full_like(linear.weight, -1.5)

    adapter.load_state_dict({"0.weight": new_weight, "0._extra_state": object()}, strict=True)

    torch.testing.assert_close(linear.weight.data, new_weight, rtol=0, atol=0)


def test_load_state_dict_relaxes_strict_for_gated_linear_unit():
    config = SimpleNamespace(fp8=False, gated_linear_unit=True)
    adapter = _make_adapter(config=config)
    linear = adapter.module.module[0]
    new_weight = torch.full_like(linear.weight, 3.25)

    adapter.load_state_dict({"0.weight": new_weight}, strict=True)

    torch.testing.assert_close(linear.weight.data, new_weight, rtol=0, atol=0)


def _make_index_adapter(ddp_config):
    adapter = FullyShardedDataParallel.__new__(FullyShardedDataParallel)
    torch.nn.Module.__init__(adapter)
    adapter._MODULE_TYPE_REGISTRY = FullyShardedDataParallel._MODULE_TYPE_REGISTRY
    adapter.ddp_config = ddp_config
    adapter.num_moe_experts = None
    return adapter


def test_init_dist_index_builds_dp_tp_mesh_from_parallel_state(model_parallel):
    model_parallel()
    adapter = _make_index_adapter(DistributedDataParallelConfig())

    dist_index = adapter._init_dist_index(None)

    mesh = dist_index.device_mesh
    assert mesh.mesh_dim_names == ("dp_cp", "tp")
    assert mesh.size() == torch.distributed.get_world_size()
    assert dist_index.get_dp_group() is parallel_state.get_data_parallel_group(
        with_context_parallel=True, partial_data_parallel=False
    )
    assert dist_index.get_outer_fsdp_group() is None
    assert adapter.tp_group is parallel_state.get_tensor_model_parallel_group()


def test_init_dist_index_builds_hsdp_mesh_with_inter_instance_outer_group(model_parallel):
    model_parallel(num_distributed_optimizer_instances=2)
    adapter = _make_index_adapter(
        DistributedDataParallelConfig(num_distributed_optimizer_instances=2)
    )

    dist_index = adapter._init_dist_index(None)

    assert dist_index.use_hybrid_fsdp is True
    assert dist_index.get_outer_fsdp_group() is (
        parallel_state.get_inter_distributed_optimizer_instance_group()
    )
    assert dist_index.get_dp_group() is parallel_state.get_data_parallel_group(
        with_context_parallel=True, partial_data_parallel=False
    )
    assert dist_index.device_mesh.mesh_dim_names == ("outer_fsdp_dp", "dp_cp", "tp")


def test_get_hsdp_tp_mesh_partitions_world_into_outer_and_inner_dp(model_parallel):
    model_parallel(num_distributed_optimizer_instances=2)
    outer_fsdp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
    intra_dp_group = parallel_state.get_data_parallel_group(
        with_context_parallel=True, partial_data_parallel=True
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()

    mesh = _get_hsdp_tp_mesh(outer_fsdp_group, intra_dp_group, tp_group)

    assert mesh.shape == (outer_fsdp_group.size(), intra_dp_group.size(), tp_group.size())
    assert mesh.reshape(-1).tolist() == list(range(torch.distributed.get_world_size()))


def test_sync_rng_states_leaves_rng_untouched_when_tp_group_is_single_rank(model_parallel):
    model_parallel()
    adapter = _make_index_adapter(DistributedDataParallelConfig())
    adapter.tp_group = parallel_state.get_tensor_model_parallel_group()
    assert adapter.tp_group.size() == 1
    torch.manual_seed(123)
    state_before = torch.cuda.get_rng_state()

    adapter.sync_rng_states_across_tp_group()

    assert torch.equal(torch.cuda.get_rng_state(), state_before)
