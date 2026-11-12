# Copyright (c) 2026, BAAI. All rights reserved.
"""Multimodule communicator: 2D/3D tensor adapters, per-rank stage indexing, DAG validation."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
    _prepare_tensor_for_comm,
    _restore_tensor_from_comm,
)
from megatron.plugin.platform import get_platform

_active_grids = []


def _make_grid(offset, tp, cp, pp, dp):
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl",
    )
    for dims in ("tp", "cp", "pp", "dp"):
        grid.create_pg(dims)
    _active_grids.append(grid)
    return grid


def _destroy_grids():
    if dist.is_initialized():
        get_platform().synchronize()
        dist.barrier()
    for grid in _active_grids:
        grid.destroy()
    _active_grids.clear()
    BridgeCommunicator.destroy_broadcast_pgs()
    if dist.is_initialized():
        dist.barrier()


@pytest.fixture
def distributed():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield


def test_prepare_tensor_for_comm_expands_2d_tensors_with_singleton_last_dim():
    assert _prepare_tensor_for_comm(None) is None

    two_d = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expanded = _prepare_tensor_for_comm(two_d)
    assert expanded.shape == (2, 3, 1)
    torch.testing.assert_close(expanded.squeeze(-1), two_d)

    expanded_list = _prepare_tensor_for_comm([two_d, two_d])
    assert [tensor.shape for tensor in expanded_list] == [(2, 3, 1), (2, 3, 1)]

    three_d = torch.zeros(2, 3, 4)
    assert _prepare_tensor_for_comm(three_d) is three_d


def test_restore_tensor_from_comm_squeezes_singleton_last_dim_back_to_2d():
    assert _restore_tensor_from_comm(None) is None

    restored = _restore_tensor_from_comm(torch.arange(6).reshape(2, 3, 1))
    assert restored.shape == (2, 3)
    torch.testing.assert_close(restored, torch.arange(6).reshape(2, 3))

    restored_list = _restore_tensor_from_comm([torch.zeros(2, 3, 1), torch.zeros(2, 3, 4)])
    assert [tensor.shape for tensor in restored_list] == [(2, 3), (2, 3, 4)]


@pytest.fixture
def multimodule_communicator(distributed):
    encoder_grid = _make_grid(offset=0, tp=2, cp=1, pp=2, dp=1)
    decoder_grid = _make_grid(offset=4, tp=1, cp=1, pp=1, dp=2)
    config = ModelParallelConfig(pipeline_dtype=torch.float)
    yield MultiModulePipelineCommunicator(
        {"encoder": encoder_grid, "decoder": decoder_grid},
        {"encoder": ["decoder"], "decoder": []},
        config,
    )
    _destroy_grids()


def test_current_stage_offsets_rank_by_module_position_and_pp_stage(multimodule_communicator):
    assert multimodule_communicator.total_stages == 3
    expected_stage = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    rank = dist.get_rank()
    if rank in expected_stage:
        assert multimodule_communicator.current_stage == expected_stage[rank]
    else:
        assert multimodule_communicator.rank_module_map == {}
        assert multimodule_communicator.current_stage == 0


@pytest.mark.parametrize(
    ("rank_kwargs", "match"),
    [
        ({}, "Topology must be a DAG with at least one terminal"),
        ({"rank": 0, "module_name": "a"}, "Topology contains cycles; expected a DAG"),
    ],
    ids=["no-terminal-module", "cyclic-topology"],
)
def test_compute_total_pipeline_stages_rejects_invalid_dags(rank_kwargs, match, distributed):
    grid_a = HyperCommGrid(shape=[1, 1, 1, 1], dim_names=["tp", "cp", "pp", "dp"], rank_offset=0)
    grid_b = HyperCommGrid(shape=[1, 1, 1, 1], dim_names=["tp", "cp", "pp", "dp"], rank_offset=1)
    cyclic_topology = {"a": ["b"], "b": ["a"]}

    with pytest.raises(ValueError, match=match):
        MultiModulePipelineCommunicator.compute_total_pipeline_stages(
            cyclic_topology, {"a": grid_a, "b": grid_b}, **rank_kwargs
        )
