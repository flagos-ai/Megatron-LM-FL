# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.pipeline_parallel.bridge_communicator as bridge_module
import megatron.core.pipeline_parallel.multimodule_communicator as multi_module
from megatron.core.pipeline_parallel.bridge_communicator import (
    BridgeCommunicator,
    CommRole,
    RankCommInfo,
)
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
    RankModuleInfo,
    _prepare_tensor_for_comm,
    _restore_tensor_from_comm,
)


class _Grid:
    def __init__(self, pp_size=2, rank_offset=0, dim_names=None, shape=None):
        self.dim_names = dim_names or ["pp"]
        self.shape = shape or [pp_size]
        self.rank_offset = rank_offset
        self.size = 1
        for dim in self.shape:
            self.size *= dim

    def _coords(self, rank):
        temp = rank - self.rank_offset
        coords = []
        for dim in self.shape:
            coords.append(temp % dim)
            temp //= dim
        return coords

    def _gen_rank_enum(self, dims):
        varying = {self.dim_names.index(dim) for dim in dims}
        groups = {}
        for rank in range(self.rank_offset, self.rank_offset + self.size):
            coords = self._coords(rank)
            key = tuple(coord if idx not in varying else None for idx, coord in enumerate(coords))
            groups.setdefault(key, []).append(rank)
        return list(groups.values())

    def get_pg(self, name):
        return SimpleNamespace(size=lambda: self.shape[self.dim_names.index(name)], rank=lambda: 0)


class _FakeP2P:
    def __init__(self):
        self.calls = []

    def recv_forward(self, tensor_shapes=None, is_first_stage=False):
        self.calls.append(("recv_forward", tensor_shapes, is_first_stage))
        return torch.ones(2, 3, 1)

    def send_forward(self, tensor, is_last_stage=False):
        self.calls.append(("send_forward", tuple(tensor.shape), is_last_stage))

    def send_forward_recv_backward(self, tensor, tensor_shapes=None, is_last_stage=False):
        self.calls.append(("send_forward_recv_backward", tuple(tensor.shape), tensor_shapes))
        return torch.ones(2, 3, 1) * 2

    def send_backward_recv_forward(self, tensor, tensor_shapes=None, is_first_stage=False):
        self.calls.append(("send_backward_recv_forward", tuple(tensor.shape), tensor_shapes))
        return torch.ones(2, 3, 1) * 3

    def recv_backward(self, tensor_shapes=None, is_last_stage=False):
        self.calls.append(("recv_backward", tensor_shapes, is_last_stage))
        return torch.ones(2, 3, 1) * 4

    def send_backward(self, tensor, is_first_stage=False):
        self.calls.append(("send_backward", tuple(tensor.shape), is_first_stage))


class _FakeBridge:
    def __init__(self, src, dest):
        self.src_module_name = src
        self.dest_module_name = dest
        self.calls = []

    def is_current_rank_in_grid(self, grid):
        return True

    def recv_forward(self):
        self.calls.append("recv_forward")
        return torch.ones(2, 3)

    def send_forward(self, tensor):
        self.calls.append(("send_forward", tuple(tensor.shape)))

    def send_forward_recv_backward(self, tensor):
        self.calls.append(("send_forward_recv_backward", tuple(tensor.shape)))
        return torch.ones_like(tensor) * 5

    def send_backward_recv_forward(self, tensor):
        self.calls.append(("send_backward_recv_forward", tuple(tensor.shape)))
        return torch.ones_like(tensor) * 6

    def recv_backward(self):
        self.calls.append("recv_backward")
        return torch.ones(2, 3) * 7

    def send_backward(self, tensor):
        self.calls.append(("send_backward", tuple(tensor.shape)))


def test_multimodule_shape_adapters_and_stage_math_paths():
    two_d = torch.ones(2, 3)
    three_d = torch.ones(2, 3, 4)
    assert _prepare_tensor_for_comm(None) is None
    assert _prepare_tensor_for_comm(two_d).shape == (2, 3, 1)
    assert _prepare_tensor_for_comm(three_d) is three_d
    prepared_list = _prepare_tensor_for_comm([two_d, three_d])
    assert [tuple(t.shape) for t in prepared_list] == [(2, 3, 1), (2, 3, 4)]
    with pytest.raises(AssertionError, match="singleton last dim"):
        _prepare_tensor_for_comm(torch.ones(2, 3, 1))
    assert _restore_tensor_from_comm(None) is None
    assert _restore_tensor_from_comm(torch.ones(2, 3, 1)).shape == (2, 3)
    assert _restore_tensor_from_comm(torch.ones(2, 3, 4)).shape == (2, 3, 4)

    grids = {"a": _Grid(pp_size=2, rank_offset=0), "b": _Grid(pp_size=3, rank_offset=2), "c": _Grid(pp_size=1, rank_offset=5)}
    topology = {"a": ["b"], "b": ["c"], "c": []}
    assert MultiModulePipelineCommunicator.compute_total_pipeline_stages(topology, grids) == 6
    assert (
        MultiModulePipelineCommunicator.compute_total_pipeline_stages(
            topology, grids, rank=3, module_name="b"
        )
        == 4
    )
    with pytest.raises(ValueError, match="cycles"):
        MultiModulePipelineCommunicator.compute_total_pipeline_stages(
            {"a": ["b"], "b": ["a", "c"], "c": []},
            {"a": grids["a"], "b": grids["b"], "c": grids["c"]},
        )
    with pytest.raises(ValueError, match="terminal"):
        MultiModulePipelineCommunicator.compute_total_pipeline_stages({"a": ["a"]}, {"a": grids["a"]})


def test_multimodule_send_recv_wrapper_dispatch_paths():
    p2p_a = _FakeP2P()
    p2p_b = _FakeP2P()
    bridge_in = _FakeBridge("upstream", "a")
    bridge_out = _FakeBridge("b", "downstream")
    communicator = object.__new__(MultiModulePipelineCommunicator)
    communicator.topology = {"a": ["b"], "b": []}
    communicator.module_to_grid_map = {"a": _Grid(2, 0), "b": _Grid(2, 2)}
    communicator.current_rank = 0
    communicator.rank_module_map = {
        "a": RankModuleInfo(
            pp_rank=0,
            pp_size=2,
            p2p_communicator=p2p_a,
            bridge_comms_as_src_module=[],
            bridge_comms_as_dest_module=[bridge_in],
        ),
        "b": RankModuleInfo(
            pp_rank=1,
            pp_size=2,
            p2p_communicator=p2p_b,
            bridge_comms_as_src_module=[bridge_out],
            bridge_comms_as_dest_module=[],
        ),
    }

    assert communicator.is_pp_first_stage is True
    assert communicator.is_pp_last_stage is True
    assert communicator.total_stages == 4
    assert communicator.current_stage == 0
    assert communicator._is_source_module("a") is True
    assert communicator._is_source_module("b") is False
    assert communicator._is_sink_module("b") is True
    assert communicator.is_current_rank_in_grid(_Grid(2, 0)) is True

    received = communicator.recv_forward(tensor_shape=(2, 3, 1))
    assert set(received) == {"upstream", "b"}
    communicator.send_forward({"a": torch.ones(2, 3), "b": torch.ones(2, 3)})
    assert p2p_a.calls[-1][0] == "send_forward"
    assert bridge_out.calls[-1][0] == "send_forward"

    grad = communicator.send_forward_recv_backward(
        {"a": torch.ones(2, 3), "b": torch.ones(2, 3)}, tensor_shape=(2, 3, 1)
    )
    assert set(grad) == {"a", "b"}
    inputs = communicator.send_backward_recv_forward(
        {"upstream": torch.ones(2, 3), "b": torch.ones(2, 3)}, tensor_shape=(2, 3, 1)
    )
    assert set(inputs) == {"upstream", "b"}
    backward = communicator.recv_backward(tensor_shape=(2, 3, 1))
    assert set(backward) == {"a", "b"}
    communicator.send_backward({"upstream": torch.ones(2, 3), "b": torch.ones(2, 3)})
    assert bridge_in.calls[-1][0] == "send_backward"
    assert p2p_b.calls[-1][0] == "send_backward"


def test_bridge_comm_map_grid_helpers_and_tensor_split_paths(monkeypatch):
    destroyed = []
    monkeypatch.setattr(bridge_module.dist, "destroy_process_group", lambda pg: destroyed.append(pg))
    BridgeCommunicator._broadcast_pg_cache = {"a": "pg-a", "b": None}
    BridgeCommunicator.destroy_broadcast_pgs()
    assert destroyed == ["pg-a"]
    assert BridgeCommunicator._broadcast_pg_cache == {}

    created = []
    monkeypatch.setattr(
        bridge_module.dist,
        "new_subgroups_by_enumeration",
        lambda ranks, backend=None: (created.append((ranks, backend)) or f"pg-{len(created)}", None),
    )
    pg1 = BridgeCommunicator._get_or_create_broadcast_pg([[0, 1], [2, 3]])
    pg2 = BridgeCommunicator._get_or_create_broadcast_pg([[2, 3], [0, 1]])
    assert pg1 == pg2
    assert len(created) == 1

    bridge = object.__new__(BridgeCommunicator)
    bridge.src_grid = _Grid(pp_size=2, rank_offset=0)
    bridge.dest_grid = _Grid(pp_size=2, rank_offset=2)
    bridge.current_rank = 1
    bridge.tensor_ndim = 3
    bridge.dim_mapping = {"s": 0, "b": 1, "h": 2}
    bridge.comm_map = {}
    assert bridge._batch_dim == 1
    bridge.tensor_ndim = 2
    assert bridge._batch_dim == 0
    assert bridge.is_current_rank_in_grid(bridge.src_grid) is True
    assert bridge.is_current_rank_in_grid(bridge.dest_grid) is False
    splits = bridge._split_tensor_at_batch_dim(torch.arange(12).reshape(6, 2), 3)
    assert [tuple(split.shape) for split in splits] == [(2, 2), (2, 2), (2, 2)]
    with pytest.raises(ValueError, match="positive"):
        bridge._split_tensor_at_batch_dim(torch.ones(2, 2), 0)

    bridge.src_grid = _Grid(pp_size=4, rank_offset=0)
    bridge.dest_grid = _Grid(pp_size=2, rank_offset=4)
    bridge.build_comm_map([0, 1, 2, 3], [4, 5])
    assert bridge.comm_map[0].role == CommRole.SENDER
    assert bridge.comm_map[4].recv_from_ranks == [0, 1]

    bridge.comm_map = {}
    bridge.build_comm_map([0], [4, 5])
    assert bridge.comm_map[0].send_to_ranks == [4, 5]
    assert bridge.comm_map[4].recv_from_ranks == [0]
    with pytest.raises(ValueError, match="evenly divisible"):
        bridge.build_comm_map([0, 1, 2], [4, 5])

    complex_grid = _Grid(
        rank_offset=10,
        dim_names=["tp", "cp", "pp", "dp"],
        shape=[2, 1, 2, 1],
    )
    bridge.current_rank = 11
    assert bridge.get_leader_rank(complex_grid, is_src=True)[0] == [13]
    assert bridge.get_leader_rank(complex_grid, is_src=False)[0] == [10]
    assert bridge.get_boundary_pp_stage_ranks(complex_grid, is_src=True) == [[12, 13]]
    assert bridge.get_boundary_pp_stage_ranks(complex_grid, is_src=False) == [[10, 11]]


def test_bridge_shape_communication_sender_receiver_member_paths(monkeypatch):
    bridge = object.__new__(BridgeCommunicator)
    bridge.current_rank = 0
    bridge.tensor_ndim = 2
    bridge.comm_map = {
        0: RankCommInfo(role=CommRole.SENDER, send_to_ranks=[2, 3]),
        1: RankCommInfo(role=CommRole.MEMBER),
        2: RankCommInfo(role=CommRole.RECEIVER, recv_from_ranks=[0]),
    }
    monkeypatch.setattr(bridge_module.cur_platform, "current_device", lambda: "cpu")

    class _Req:
        def wait(self):
            return None

    class _Op:
        def __init__(self, fn, tensor, peer):
            self.fn = fn
            self.tensor = tensor
            self.peer = peer

    monkeypatch.setattr(bridge_module.torch.distributed, "P2POp", _Op)
    monkeypatch.setattr(bridge_module.torch.distributed, "isend", object())
    monkeypatch.setattr(bridge_module.torch.distributed, "irecv", object())

    def _batch(ops):
        for op in ops:
            if op.fn is bridge_module.torch.distributed.irecv:
                op.tensor.copy_(torch.tensor([5, 6], dtype=torch.int64))
        return [_Req() for _ in ops]

    monkeypatch.setattr(bridge_module.torch.distributed, "batch_isend_irecv", _batch)
    fwd_shapes, grad_shapes = bridge._communicate_shapes(
        tensor_to_send_next=torch.ones(5, 6), recv_next=True
    )
    assert fwd_shapes == []
    assert grad_shapes == [(5, 6), (5, 6)]

    bridge.current_rank = 2
    fwd_shapes, grad_shapes = bridge._communicate_shapes(
        tensor_to_send_prev=torch.ones(7, 8), recv_prev=True
    )
    assert fwd_shapes == [(5, 6)]
    assert grad_shapes == []

    bridge.current_rank = 1
    assert bridge._communicate_shapes(recv_prev=True, recv_next=True) == ([], [])
