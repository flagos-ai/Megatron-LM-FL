# Copyright (c) 2026, BAAI. All rights reserved.
"""reduce_scatter_with_fp32_accumulation sums bf16 inputs in fp32 and scatters rank shards."""

import pytest
import torch

from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import (
    _ReduceScatterWithFP32AccumulationWorkHandle,
    reduce_scatter_with_fp32_accumulation,
)
from tests.unit_tests.test_utilities import Utils

NUM_ELEMENTS = 8192


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("async_op", [True, False], ids=["async", "sync"])
@pytest.mark.parametrize("explicit_group", [True, False], ids=["world-group", "default-group"])
def test_reduce_scatter_with_fp32_accumulation_matches_fp32_reference(
    model_parallel, async_op, explicit_group
):
    model_parallel()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    group = torch.distributed.group.WORLD if explicit_group else None

    values = torch.randint(1, 100, (NUM_ELEMENTS,), device="cuda", dtype=torch.int64).float()
    torch.distributed.broadcast(values, src=0)
    input_tensor = values.bfloat16()
    output_tensor = shard_buffer(input_tensor, world_size)[rank].clone()

    handle = reduce_scatter_with_fp32_accumulation(
        output_tensor, input_tensor, torch.distributed.ReduceOp.SUM, group, async_op
    )
    if async_op:
        assert isinstance(handle, _ReduceScatterWithFP32AccumulationWorkHandle)
        assert handle.world_size == world_size
        assert handle.output_tensor is output_tensor
        handle.wait()
    else:
        assert handle is None

    expected = (shard_buffer(values, world_size)[rank] * world_size).bfloat16()
    torch.testing.assert_close(output_tensor, expected, rtol=1.6e-2, atol=1e-2)


def test_reduce_scatter_with_fp32_accumulation_rejects_non_sum_op(model_parallel):
    model_parallel()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    input_tensor = torch.ones(NUM_ELEMENTS, device="cuda")
    output_tensor = shard_buffer(input_tensor, world_size)[rank].clone()
    with pytest.raises(AssertionError):
        reduce_scatter_with_fp32_accumulation(
            output_tensor,
            input_tensor,
            torch.distributed.ReduceOp.MAX,
            torch.distributed.group.WORLD,
            False,
        )
