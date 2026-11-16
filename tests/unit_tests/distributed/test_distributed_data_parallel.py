# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import gc
import weakref
from unittest.mock import patch

import pytest
import torch
from packaging import version
from torch import testing

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR, ensure_params_ready
from tests.unit_tests.test_utilities import Utils


# Test model for testing DDP
class TestModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim * 4)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim * 4, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TestDistributedDataParallel:
    @classmethod
    def setup_class(cls):
        cls.grid = None
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        if cls.grid is not None:
            cls.grid.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize("dp_size", [2, 8])  # Test with 2 or 8 GPUs
    def test_ddp_with_dp_process_groups(self, dp_size):
        """Test that DDP works correctly with dp pgs from parallel state and user defined pgs."""

        # Skip test if we don't have enough GPUs
        world_size = torch.distributed.get_world_size()
        if world_size != dp_size:
            pytest.skip(f"This test requires {dp_size} GPUs, but only {world_size} are available")

        # Simple model config
        input_dim = 13
        output_dim = 17

        # Setup DDP config
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)

        # Create two identical models
        model1 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()
        model2 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()

        # Ensure identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)

        # Wrap first model with default process groups
        transformer_config = TransformerConfig(
            num_attention_heads=1, num_layers=1, context_parallel_size=1
        )

        ddp_model1 = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=model1
        )

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimension ep, pp, dp (reversed from device mesh order)
        grid = HyperCommGrid([1, 1, 1, 1, dp_size], ["tp", "cp", "ep", "pp", "dp"])
        self.grid = grid

        # Create process groups config with ONLY dp group
        pg_collection = ProcessGroupCollection()

        pg_collection.dp = grid.create_pg("dp")
        pg_collection.dp_cp = grid.create_pg(["dp", "cp"])
        pg_collection.pp = grid.create_pg("pp")
        pg_collection.tp = grid.create_pg("tp")
        pg_collection.ep = grid.create_pg("ep")

        # Wrap second model with minimal process groups (only dp)
        ddp_model2 = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=model2, pg_collection=pg_collection
        )

        # Create identical inputs with integer values
        batch_size = 2
        input_data = torch.randint(0, 10, (batch_size, input_dim), device='cuda', dtype=torch.long)
        input_data = input_data.float()  # Convert to float for model compatibility

        # Forward pass
        out1 = ddp_model1(input_data)
        out2 = ddp_model2(input_data)

        testing.assert_close(out1, out2, rtol=0, atol=0)

        # Loss and backward
        loss1 = out1.sum()
        loss2 = out2.sum()

        loss1.backward()
        loss2.backward()

        # Check gradients are identical using torch.testing
        for p1, p2 in zip(ddp_model1.parameters(), ddp_model2.parameters()):
            if hasattr(p1, 'main_grad') and hasattr(p2, 'main_grad'):
                testing.assert_close(p1.main_grad, p2.main_grad, rtol=0, atol=0)


class _IndexerProjection(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_weights_proj = torch.nn.Linear(
            128, 128, bias=False, device='cuda', dtype=torch.bfloat16
        )
        self._weights_proj_te_gemm_supported = False

    def forward(self, x):
        return DSAIndexer._project_indexer_weights(self, x)


@pytest.fixture
def wrapped_projection():
    Utils.initialize_model_parallel()
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1)
    config.dsa_indexer_weights_proj_use_quantization = True
    config.dsa_indexer_weights_proj_output_dtype = 'fp32'
    module = _IndexerProjection(config)
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(use_distributed_optimizer=True, overlap_param_gather=True),
        module,
    )
    group = ddp.param_to_bucket_group[module.linear_weights_proj.weight]
    # Model a newly available local parameter shard without constructing or stepping an optimizer.
    # Every non-owned shard remains zero until the real NCCL all-gather publishes it.
    with torch.no_grad():
        for bucket in group.buckets:
            bucket.param_data.zero_()
            chunks = bucket.param_data.chunk(group.intra_distributed_optimizer_instance_size)
            chunks[group.intra_distributed_optimizer_instance_rank].fill_(1)
    try:
        yield ddp, module, group
    finally:
        if ddp.remove_forward_pre_hook_handles:
            ddp.disable_forward_pre_hook(param_sync=False)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('pre_dispatch', [False, True])
def test_fp32_projection_publishes_ddp_parameters(wrapped_projection, pre_dispatch):
    ddp, module, group = wrapped_projection
    assert hasattr(module.linear_weights_proj.weight, PARAM_READY_CALLBACK_ATTR)
    if pre_dispatch:
        group.start_param_sync()
        assert group.param_gather_handle is not None
    x = torch.ones((2, 1, 128), device='cuda', dtype=torch.bfloat16)
    with patch.object(group, 'start_param_sync', wraps=group.start_param_sync) as start:
        actual = ddp(x)
        torch.testing.assert_close(actual, torch.full_like(actual, 128), rtol=0, atol=0)
        assert group.param_gather_dispatched
        assert group.param_gather_handle is None
        assert start.call_count == (0 if pre_dispatch else 1)
        # Repeated reads in the same iteration must not gather again.
        torch.testing.assert_close(ddp(x), actual, rtol=0, atol=0)
        assert start.call_count == (0 if pre_dispatch else 1)


def test_disabled_hooks_preserve_external_param_schedule(wrapped_projection):
    ddp, module, group = wrapped_projection
    ddp.disable_forward_pre_hook(param_sync=False)
    ensure_params_ready([module.linear_weights_proj.weight])
    assert not group.param_gather_dispatched
    group.start_param_sync()
    with patch.object(group, 'finish_param_sync', wraps=group.finish_param_sync) as finish:
        ensure_params_ready([module.linear_weights_proj.weight])
        finish.assert_called_once_with(skip_next_bucket_dispatch=True)
    torch.testing.assert_close(
        module.linear_weights_proj.weight, torch.ones_like(module.linear_weights_proj.weight)
    )


def test_graph_capture_does_not_dispatch_parameter_gather(wrapped_projection):
    _, module, group = wrapped_projection
    with patch(
        'megatron.core.distributed.distributed_data_parallel.is_graph_capturing', return_value=True
    ):
        ensure_params_ready([module.linear_weights_proj.weight])
    assert not group.param_gather_dispatched


def test_rewrap_without_overlap_removes_stale_callback(wrapped_projection):
    ddp, module, _ = wrapped_projection
    ddp.disable_forward_pre_hook(param_sync=False)
    DistributedDataParallel(module.config, DistributedDataParallelConfig(), module)
    assert not hasattr(module.linear_weights_proj.weight, PARAM_READY_CALLBACK_ATTR)


def test_callback_does_not_own_ddp_lifetime(wrapped_projection):
    from megatron.core.distributed.distributed_data_parallel import _BucketParamReadyCallback

    ddp, _, group = wrapped_projection

    # Isolate callback ownership from DDP's separate backward-hook lifetime.
    class Owner:
        pass

    owner = Owner()
    callback = _BucketParamReadyCallback(owner, group)
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None
    callback()
    assert not group.param_gather_dispatched
