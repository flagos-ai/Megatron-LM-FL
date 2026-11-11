# Copyright (c) 2026, BAAI. All rights reserved.
'''Tensor/sequence-parallel mapping collectives: gather, scatter, reduce-scatter along first and last dim, all-to-all with even and uneven splits, and their backward gradient routes.'''

import pytest
import torch

from megatron.core.tensor_parallel import mappings
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)

    yield setup
    Utils.destroy_model_parallel()


def test_mapping_helpers_bypass_collectives_when_tp_world_size_is_one(model_parallel):
    model_parallel(1, 1)
    x = torch.randn(4, 2, 8, device='cuda')

    assert torch.equal(mappings.scatter_to_tensor_model_parallel_region(x), x)
    assert torch.equal(mappings.scatter_to_sequence_parallel_region(x), x)
    assert torch.equal(mappings.gather_from_sequence_parallel_region(x), x)
    assert torch.equal(mappings.reduce_scatter_to_sequence_parallel_region(x), x)


def test_all_gather_last_dim_from_tensor_parallel_region(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    torch.manual_seed(11)
    full = torch.randn(4, 8, device='cuda')
    x = full[:, rank * 4 : (rank + 1) * 4].clone().requires_grad_(True)

    output = mappings.all_gather_last_dim_from_tensor_parallel_region(x)
    assert torch.equal(output, full)

    base_grad = torch.randn(4, 8, device='cuda')
    grad_output = base_grad * (rank + 1)
    (output * grad_output).sum().backward()
    torch.testing.assert_close(
        x.grad, 3 * base_grad[:, rank * 4 : (rank + 1) * 4], rtol=1e-6, atol=1e-6
    )


def test_reduce_scatter_last_dim_to_tensor_parallel_region(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    group = get_tensor_model_parallel_group_if_none(None)
    torch.manual_seed(12)
    first = torch.randn(4, 8, device='cuda')
    second = torch.randn(4, 8, device='cuda')
    x = (first if rank == 0 else second).clone().requires_grad_(True)

    output = mappings.reduce_scatter_last_dim_to_tensor_parallel_region(x)
    expected = (first + second)[:, rank * 4 : (rank + 1) * 4]
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)

    symbolic = mappings._ReduceScatterToTensorParallelRegion.symbolic(None, x, group)
    torch.testing.assert_close(
        symbolic, (first + second)[:, rank * 4 : (rank + 1) * 4], rtol=1e-6, atol=1e-6
    )

    grad_first = torch.randn(4, 4, device='cuda')
    grad_second = torch.randn(4, 4, device='cuda')
    grad_output = grad_first if rank == 0 else grad_second
    (output * grad_output).sum().backward()
    torch.testing.assert_close(
        x.grad, torch.cat([grad_first, grad_second], dim=-1), rtol=1e-6, atol=1e-6
    )


def test_gather_from_sequence_parallel_region_with_uneven_split_sizes(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    torch.manual_seed(13)
    full = torch.randn(8, 2, 8, device='cuda')
    split_sizes = [3, 5]
    start = sum(split_sizes[:rank])
    x = full[start : start + split_sizes[rank]]

    output = mappings.gather_from_sequence_parallel_region(
        x, output_split_sizes=split_sizes, use_global_buffer=True
    )
    assert torch.equal(output, full)


@pytest.mark.parametrize('use_global_buffer', [True, False], ids=['global-buffer', 'local-buffer'])
def test_reduce_scatter_to_sequence_parallel_region_with_uneven_split_sizes(
    model_parallel, use_global_buffer
):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    torch.manual_seed(14)
    first = torch.randn(8, 2, 8, device='cuda')
    second = torch.randn(8, 2, 8, device='cuda')
    x = first if rank == 0 else second
    split_sizes = [3, 5]

    output = mappings.reduce_scatter_to_sequence_parallel_region(
        x, input_split_sizes=split_sizes, use_global_buffer=use_global_buffer
    )
    total = first + second
    expected = total[:3] if rank == 0 else total[3:]
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_reduce_scatter_to_sequence_parallel_region_with_global_buffer(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    torch.manual_seed(15)
    first = torch.randn(8, 2, 8, device='cuda')
    second = torch.randn(8, 2, 8, device='cuda')
    x = first if rank == 0 else second

    output = mappings.reduce_scatter_to_sequence_parallel_region(x, use_global_buffer=True)
    expected = (first + second)[rank * 4 : (rank + 1) * 4]
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_gather_from_sequence_parallel_region_backward_paths(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    group = get_tensor_model_parallel_group_if_none(None)
    torch.manual_seed(16)
    full = torch.randn(4, 2, 8, device='cuda')
    base_grad = torch.randn(4, 2, 8, device='cuda')
    grad_output = base_grad * (rank + 1)

    gathered = mappings.gather_from_sequence_parallel_region(full[rank * 2 : (rank + 1) * 2])
    assert torch.equal(gathered, full)
    assert torch.equal(
        mappings._GatherFromSequenceParallelRegion.symbolic(
            None, full[rank * 2 : (rank + 1) * 2], group
        ),
        full,
    )

    x = full[rank * 2 : (rank + 1) * 2].clone().requires_grad_(True)
    output = mappings.gather_from_sequence_parallel_region(x)
    (output * grad_output).sum().backward()
    torch.testing.assert_close(
        x.grad, (base_grad * 3)[rank * 2 : (rank + 1) * 2], rtol=1e-6, atol=1e-6
    )

    x = full[rank * 2 : (rank + 1) * 2].clone().requires_grad_(True)
    output = mappings.gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)
    (output * grad_output).sum().backward()
    torch.testing.assert_close(
        x.grad, (base_grad * (rank + 1))[rank * 2 : (rank + 1) * 2], rtol=1e-6, atol=1e-6
    )


def test_scatter_to_sequence_parallel_region_backward_gathers(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    torch.manual_seed(17)
    full = torch.randn(4, 2, 8, device='cuda')
    x = full.clone().requires_grad_(True)

    output = mappings.scatter_to_sequence_parallel_region(x)
    assert torch.equal(output, full[rank * 2 : (rank + 1) * 2])

    grad_output = torch.randn(2, 2, 8, device='cuda')
    (output * grad_output).sum().backward()
    assert torch.equal(x.grad[rank * 2 : (rank + 1) * 2], grad_output)


@pytest.mark.parametrize('use_nccl_stream', [False, True], ids=['default-stream', 'nccl-stream'])
def test_all_to_all_equal_split_permutes_rank_chunks(model_parallel, use_nccl_stream):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    group = get_tensor_model_parallel_group_if_none(None)
    torch.manual_seed(18)
    first_to_first = torch.randn(2, 2, 4, device='cuda')
    second_to_first = torch.randn(2, 2, 4, device='cuda')
    first_to_second = torch.randn(2, 2, 4, device='cuda')
    second_to_second = torch.randn(2, 2, 4, device='cuda')
    x = (
        torch.cat([first_to_first, first_to_second], dim=0)
        if rank == 0
        else torch.cat([second_to_first, second_to_second], dim=0)
    ).requires_grad_(True)

    output = mappings.all_to_all(group, x, use_nccl_stream=use_nccl_stream)
    expected = (
        torch.cat([first_to_first, second_to_first], dim=0)
        if rank == 0
        else torch.cat([first_to_second, second_to_second], dim=0)
    )
    assert torch.equal(output, expected)

    grad_first = torch.randn(4, 2, 4, device='cuda')
    grad_second = torch.randn(4, 2, 4, device='cuda')
    grad_output = grad_first if rank == 0 else grad_second
    (output * grad_output).sum().backward()
    expected_grad = (
        torch.cat([grad_first[:2], grad_second[:2]], dim=0)
        if rank == 0
        else torch.cat([grad_first[2:], grad_second[2:]], dim=0)
    )
    assert torch.equal(x.grad, expected_grad)


def test_all_to_all_uneven_split_sizes(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    group = get_tensor_model_parallel_group_if_none(None)
    torch.manual_seed(19)
    small_to_first = torch.randn(2, 2, 4, device='cuda')
    small_to_second = torch.randn(2, 2, 4, device='cuda')
    large_to_first = torch.randn(6, 2, 4, device='cuda')
    large_to_second = torch.randn(6, 2, 4, device='cuda')
    x = (
        torch.cat([small_to_first, large_to_first], dim=0)
        if rank == 0
        else torch.cat([small_to_second, large_to_second], dim=0)
    )
    output_split_sizes = [2, 2] if rank == 0 else [6, 6]

    output = mappings.all_to_all(
        group, x, output_split_sizes_=output_split_sizes, input_split_sizes=[2, 6]
    )
    expected = (
        torch.cat([small_to_first, small_to_second], dim=0)
        if rank == 0
        else torch.cat([large_to_first, large_to_second], dim=0)
    )
    assert torch.equal(output, expected)


def test_all_to_all_sp2hp_and_hp2sp_roundtrip(model_parallel):
    model_parallel(2, 1)
    rank = torch.distributed.get_rank()
    torch.manual_seed(20)
    sp_full = torch.randn(8, 16, device='cuda')
    x_sp = sp_full[rank * 4 : (rank + 1) * 4]

    hidden_parallel = mappings.all_to_all_sp2hp(x_sp)
    assert hidden_parallel.shape == (8, 8)
    assert torch.equal(hidden_parallel, sp_full[:, rank * 8 : (rank + 1) * 8])

    sequence_parallel = mappings.all_to_all_hp2sp(hidden_parallel)
    assert sequence_parallel.shape == (4, 16)
    assert torch.equal(sequence_parallel, x_sp)
