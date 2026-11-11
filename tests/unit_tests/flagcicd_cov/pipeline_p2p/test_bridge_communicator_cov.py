# Copyright (c) 2026, BAAI. All rights reserved.
"""BridgeCommunicator: per-rank comm-map roles and grid-membership guards on send/recv calls."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator, CommRole
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


@pytest.fixture(scope="module")
def bridge():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    src_grid = _make_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
    dest_grid = _make_grid(offset=4, tp=2, cp=1, pp=1, dp=2)
    yield BridgeCommunicator(src_grid, dest_grid, comm_dtype=torch.float)
    _destroy_grids()


_EXPECTED_COMM_INFO = {
    0: (CommRole.MEMBER, [], []),
    1: (CommRole.SENDER, [4], []),
    2: (CommRole.MEMBER, [], []),
    3: (CommRole.SENDER, [6], []),
    4: (CommRole.RECEIVER, [], [1]),
    5: (CommRole.MEMBER, [], []),
    6: (CommRole.RECEIVER, [], [3]),
    7: (CommRole.MEMBER, [], []),
}


def _assert_current_rank_comm_info(bridge):
    role, send_to_ranks, recv_from_ranks = _EXPECTED_COMM_INFO[dist.get_rank()]
    info = bridge.comm_map[dist.get_rank()]
    assert info.role == role
    assert info.send_to_ranks == send_to_ranks
    assert info.recv_from_ranks == recv_from_ranks


def test_build_comm_map_rejects_leader_counts_without_common_divisor(bridge):
    with pytest.raises(
        ValueError, match="must be evenly divisible. One must be a multiple of the other."
    ):
        bridge.build_comm_map([1, 3], [4, 5, 6])


def test_send_forward_rejects_rank_outside_source_grid(bridge):
    device = torch.device("cuda", torch.cuda.current_device())
    if bridge.is_current_rank_in_grid(bridge.src_grid):
        _assert_current_rank_comm_info(bridge)
    else:
        with pytest.raises(
            ValueError, match=r"\[send_forward\] Rank \d+ is not in the source grid"
        ):
            bridge.send_forward(torch.zeros(2, 3, device=device))


def test_recv_forward_rejects_rank_outside_destination_grid(bridge):
    if bridge.is_current_rank_in_grid(bridge.dest_grid):
        _assert_current_rank_comm_info(bridge)
    else:
        with pytest.raises(
            ValueError, match=r"\[receive_forward\] Rank \d+ is not in the destination grid"
        ):
            bridge.recv_forward()


def test_send_backward_rejects_rank_outside_destination_grid(bridge):
    device = torch.device("cuda", torch.cuda.current_device())
    if bridge.is_current_rank_in_grid(bridge.dest_grid):
        _assert_current_rank_comm_info(bridge)
    else:
        with pytest.raises(
            ValueError, match=r"\[send_backward\] Rank \d+ is not in the destination grid"
        ):
            bridge.send_backward(torch.zeros(2, 3, device=device))


def test_recv_backward_rejects_rank_outside_source_grid(bridge):
    if bridge.is_current_rank_in_grid(bridge.src_grid):
        _assert_current_rank_comm_info(bridge)
    else:
        with pytest.raises(
            ValueError, match=r"\[receive_backward\] Rank \d+ is not in the source grid"
        ):
            bridge.recv_backward()


def test_send_forward_recv_backward_rejects_rank_outside_source_grid(bridge):
    device = torch.device("cuda", torch.cuda.current_device())
    if bridge.is_current_rank_in_grid(bridge.src_grid):
        _assert_current_rank_comm_info(bridge)
    else:
        with pytest.raises(
            ValueError, match="send_forward_recv_backward is only allowed on src grid"
        ):
            bridge.send_forward_recv_backward(torch.zeros(2, 3, device=device))


def test_send_backward_recv_forward_rejects_rank_outside_destination_grid(bridge):
    device = torch.device("cuda", torch.cuda.current_device())
    if bridge.is_current_rank_in_grid(bridge.dest_grid):
        _assert_current_rank_comm_info(bridge)
    else:
        with pytest.raises(
            ValueError, match="send_backward_recv_forward is only allowed on dest grid"
        ):
            bridge.send_backward_recv_forward(torch.zeros(2, 3, device=device))


def test_communicate_shapes_returns_empty_shape_lists_when_nothing_to_exchange(bridge):
    assert bridge._communicate_shapes() == ([], [])


def test_split_tensor_at_batch_dim_rejects_non_positive_split_count(bridge):
    with pytest.raises(ValueError, match="num_splits must be positive, got 0"):
        bridge._split_tensor_at_batch_dim(torch.arange(6).reshape(3, 2), num_splits=0)
