# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    gradient_reduce_preprocessing,
)
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("collective", ["all_reduce", "reduce_scatter"])
def test_gradient_scaling_matches_rank_average(fusion, collective):
    """Fused and explicit scaling must produce the same distributed average."""
    Utils.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    platform = get_platform()
    group = torch.distributed.new_group(ranks=list(range(world_size)))
    try:
        config = SimpleNamespace(average_in_collective=False, gradient_reduce_div_fusion=fusion)
        gradients = torch.full(
            (world_size * 4,),
            float(rank + 1),
            device=platform.device(platform.current_device()),
            dtype=torch.float32,
        )
        reduce_op = gradient_reduce_preprocessing(gradients, 1.0 / world_size, config)
        if collective == "all_reduce":
            torch.distributed.all_reduce(gradients, op=reduce_op, group=group)
            result = gradients
        else:
            result = torch.empty_like(gradients[:4])
            # Match the coalesced path used by the FSDP gradient pipeline.
            with torch.distributed._coalescing_manager(group):
                torch.distributed.reduce_scatter_tensor(
                    result, gradients, op=reduce_op, group=group
                )
        expected = torch.full_like(result, (world_size + 1) / 2)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    finally:
        torch.distributed.destroy_process_group(group)
