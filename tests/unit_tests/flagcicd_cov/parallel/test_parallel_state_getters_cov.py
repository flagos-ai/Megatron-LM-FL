# Copyright (c) 2026, BAAI. All rights reserved.
'''Model-parallel state getters and setters: context-parallel ranks and groups, amax reduction group scoping, world-size overrides, and the global memory buffer lifecycle.'''

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)

    yield setup
    Utils.destroy_model_parallel()


def test_context_parallel_getters_report_group_and_ranks(model_parallel):
    model_parallel(2, 1, context_parallel_size=2)
    rank = torch.distributed.get_rank()

    assert ps.get_context_parallel_world_size() == 2
    assert ps.get_context_parallel_group().size() == 2
    assert ps.get_context_parallel_rank() == (rank // 2) % 2
    cp_members = [(rank % 2) + 4 * ((rank // 4) % 2) + 2 * k for k in range(2)]
    assert ps.get_context_parallel_global_ranks() == cp_members

    assert ps.get_tensor_and_context_parallel_world_size() == 4
    assert ps.get_tensor_and_context_parallel_rank() == rank % 4
    tensor_and_context_members = [4 * ((rank // 4) % 2) + k for k in range(4)]
    assert (
        torch.distributed.get_process_group_ranks(ps.get_tensor_and_context_parallel_group())
        == tensor_and_context_members
    )

    assert ps.get_data_parallel_world_size(with_context_parallel=True) == 4
    assert ps.get_data_parallel_group(with_context_parallel=True).size() == 4
    assert ps.get_data_parallel_src_rank(with_context_parallel=True) == rank % 2
    assert ps.get_data_parallel_rank(with_context_parallel=True) == rank // 2


def test_get_amax_reduction_group_returns_the_scoped_group(model_parallel):
    model_parallel(2, 1, context_parallel_size=2)

    assert ps.get_amax_reduction_group() is ps.get_tensor_and_data_parallel_group()
    assert (
        ps.get_amax_reduction_group(tp_only_amax_red=True) is ps.get_tensor_model_parallel_group()
    )
    assert ps.get_amax_reduction_group(
        with_context_parallel=True
    ) is ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
    assert (
        ps.get_amax_reduction_group(with_context_parallel=True, tp_only_amax_red=True)
        is ps.get_tensor_and_context_parallel_group()
    )


def test_tensor_and_pipeline_state_setters_override_group_query(model_parallel):
    model_parallel(2, 2)

    ps.set_tensor_model_parallel_world_size(3)
    ps.set_pipeline_model_parallel_world_size(5)
    ps.set_tensor_model_parallel_rank(1)
    ps.set_pipeline_model_parallel_rank(4)

    assert ps.get_tensor_model_parallel_world_size() == 3
    assert ps.get_pipeline_model_parallel_world_size() == 5
    assert ps.get_tensor_model_parallel_rank() == 1
    assert ps.get_pipeline_model_parallel_rank() == 4


def test_expert_state_setters_override_group_query(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()

    ps.set_expert_model_parallel_world_size(3)
    ps.set_expert_model_parallel_rank(2)
    ps.set_expert_tensor_parallel_world_size(4)
    ps.set_expert_tensor_parallel_rank(1)

    assert ps.get_expert_model_parallel_world_size() == 3
    assert ps.get_expert_model_parallel_rank() == 2
    assert ps.get_expert_tensor_parallel_world_size() == 4
    assert ps.get_expert_tensor_parallel_rank() == 1
    assert ps.get_expert_tensor_and_model_parallel_world_size() == 2
    assert ps.get_expert_tensor_and_model_parallel_rank() == rank % 2


def test_expert_tensor_parallel_falls_back_to_tensor_parallel_state_when_group_absent(
    model_parallel,
):
    model_parallel(2, 1)
    ps.destroy_model_parallel()

    ps.set_tensor_model_parallel_world_size(4)
    ps.set_tensor_model_parallel_rank(1)

    assert ps.get_expert_tensor_parallel_world_size() == 4
    assert ps.get_expert_tensor_parallel_rank() == 1


def test_data_parallel_rank_setter_overrides_group_query(model_parallel):
    model_parallel(2, 2)
    rank = torch.distributed.get_rank()
    original = ps.get_data_parallel_rank()
    assert original == (rank - (rank % 2) - 4 * (rank // 4)) // 2

    ps.set_data_parallel_rank(7)
    assert ps.get_data_parallel_rank() == 7
    assert ps.get_data_parallel_rank(with_context_parallel=True) == 7

    ps.set_data_parallel_rank(original)


def test_destroy_global_memory_buffer_requires_reinitialization(model_parallel):
    model_parallel(2, 1)
    buffer = ps.get_global_memory_buffer()
    tensor = buffer.get_tensor([4, 8], torch.float32, 'cov-buffer')
    assert tuple(tensor.shape) == (4, 8)
    assert tensor.dtype == torch.float32

    ps.destroy_global_memory_buffer()
    with pytest.raises(AssertionError, match='global memory buffer is not initialized'):
        ps.get_global_memory_buffer()

    model_parallel(2, 1)
    fresh = ps.get_global_memory_buffer()
    assert tuple(fresh.get_tensor([4, 8], torch.float32, 'cov-buffer').shape) == (4, 8)
