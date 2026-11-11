# Copyright (c) 2026, BAAI. All rights reserved.
"""Colocated bridge communicators: fan-in/fan-out slicing with gradients, and rank roles from module grids."""
import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.comm.colocated_communicator import ColocatedBridgeCommunicator
from megatron.core.models.mimo.config.role import ModuleLayout, RankRole
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def distributed():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _create_grid(tp, dp):
    grid = HyperCommGrid(
        shape=[tp, 1, 1, dp], dim_names=["tp", "cp", "pp", "dp"], backend=Utils.get_backend()
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    return grid


class TestColocatedBridgeDirectionAndSlices:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_equal_dp_bridge_is_pure_passthrough(self, distributed):
        distributed(tensor_model_parallel_size=1)
        src_grid = _create_grid(tp=2, dp=4)
        dest_grid = _create_grid(tp=4, dp=4)
        comm = ColocatedBridgeCommunicator(src_grid, dest_grid)

        assert comm.is_fan_in() is False
        assert comm.is_fan_out() is False
        assert comm.get_slice_info(8).start == 0
        assert comm.get_slice_info(8).size == 8

        tensor = torch.randn(8, 3, device='cuda', requires_grad=True)
        output = comm.communicate(tensor)
        assert torch.equal(output, tensor)
        output.sum().backward()
        assert torch.equal(tensor.grad, torch.ones_like(tensor))

        comm.destroy()
        src_grid.destroy()
        dest_grid.destroy()

    def test_fan_in_bridge_gathers_forward_and_narrows_backward(self, distributed):
        distributed(tensor_model_parallel_size=1)
        src_grid = _create_grid(tp=2, dp=4)
        dest_grid = _create_grid(tp=4, dp=2)
        comm = ColocatedBridgeCommunicator(src_grid, dest_grid)

        assert comm.is_fan_in() is True
        slice_info = comm.get_slice_info(4)
        assert slice_info.size == 2
        assert slice_info.start == [0, 0, 2, 2, 0, 0, 2, 2][dist.get_rank()]

        rank = dist.get_rank()
        sibling = rank ^ 2
        tensor = torch.full((2, 3), float(rank), device='cuda', requires_grad=True)
        output = comm.communicate(tensor)

        assert output.shape == torch.Size([4, 3])
        assert torch.all(output[:2] == float(min(rank, sibling)))
        assert torch.all(output[2:] == float(max(rank, sibling)))

        output.backward(torch.ones_like(output))
        assert torch.equal(tensor.grad, torch.ones(2, 3, device='cuda'))

        comm.destroy()
        src_grid.destroy()
        dest_grid.destroy()

    def test_fan_out_bridge_narrows_forward_and_gathers_backward(self, distributed):
        distributed(tensor_model_parallel_size=1)
        src_grid = _create_grid(tp=4, dp=2)
        dest_grid = _create_grid(tp=2, dp=4)
        comm = ColocatedBridgeCommunicator(src_grid, dest_grid)

        rank = dist.get_rank()
        expected_starts = [0, 0, 2, 2, 0, 0, 2, 2]
        assert comm.is_fan_out() is True
        slice_info = comm.get_slice_info(4)
        assert slice_info.size == 2
        assert slice_info.start == expected_starts[rank]

        full_batch = torch.arange(12, dtype=torch.float32, device='cuda').reshape(4, 3)
        tensor = full_batch.clone().requires_grad_(True)
        output = comm.communicate(tensor)

        assert output.shape == torch.Size([2, 3])
        assert torch.equal(output, full_batch[slice_info.start : slice_info.start + 2])

        output.backward(torch.ones_like(output))
        assert torch.equal(tensor.grad, torch.ones(4, 3, device='cuda'))

        comm.destroy()
        src_grid.destroy()
        dest_grid.destroy()


class TestRankRoleGridMap:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_is_last_stage_returns_false_for_unknown_module(self, distributed):
        distributed(tensor_model_parallel_size=1)
        role = RankRole.build([], None)

        assert role.mode is ModuleLayout.COLOCATED
        assert role.is_last_stage('images') is False

    def test_from_grid_map_raises_when_rank_is_in_no_grid(self, distributed):
        distributed(tensor_model_parallel_size=1)
        rank = dist.get_rank()
        low_grid = HyperCommGrid(
            shape=[2, 1], dim_names=["tp", "dp"], rank_offset=0, backend=Utils.get_backend()
        )
        high_grid = HyperCommGrid(
            shape=[2, 1], dim_names=["tp", "dp"], rank_offset=2, backend=Utils.get_backend()
        )
        grid_map = {'images': low_grid, 'language': high_grid}

        if rank >= 4:
            with pytest.raises(RuntimeError, match="is not in any module grid"):
                RankRole.build(['images'], grid_map)
        else:
            role = RankRole.build(['images'], grid_map)
            assert role.mode is ModuleLayout.NON_COLOCATED
            assert set(role.modules) == ({'images'} if rank < 2 else {'language'})

        low_grid.destroy()
        high_grid.destroy()
