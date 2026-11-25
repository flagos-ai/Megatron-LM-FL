# Copyright (c) 2026, BAAI. All rights reserved.
"""DistributedDataParallel param all-gather overlap: pre-hook cycle, bucket-group chaining, and weight unmapping."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed import param_and_grad_buffer as pgb
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

IN_DIM = 16
HIDDEN_DIM = 32
OUT_DIM = 16


class _TinyModel(torch.nn.Module):
    def __init__(self, num_layers=2, in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _transformer_config():
    return TransformerConfig(num_attention_heads=1, num_layers=1)


def _wrap(module, ddp_config, disable_bucketing=False):
    return DistributedDataParallel(
        _transformer_config(),
        ddp_config=ddp_config,
        module=module,
        disable_bucketing=disable_bucketing,
    )


def test_forward_pre_hook_param_gather_cycle(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        bucket_size=512,
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)
    assert set(model.remove_forward_pre_hook_handles) == set(model.module.modules())
    x = torch.randn(4, IN_DIM, device="cuda")

    out_first = model(x)
    model.disable_forward_pre_hook()
    assert len(model.remove_forward_pre_hook_handles) == 0
    model.enable_forward_pre_hook()
    out_second = model(x)
    torch.testing.assert_close(out_first, out_second, rtol=0, atol=0)

    model.disable_forward_pre_hook(param_sync=False)
    assert len(model.remove_forward_pre_hook_handles) == 0


def test_overlap_param_gather_links_next_bucket_groups(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        bucket_size=256,
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)

    bucket_groups = model.bucket_groups
    assert len(bucket_groups) == 3
    for i in range(1, len(bucket_groups)):
        assert (
            bucket_groups[len(bucket_groups) - i].next_param_gather_bucket_group
            is bucket_groups[len(bucket_groups) - i - 1]
        )
    assert bucket_groups[0].next_param_gather_bucket_group is None


def test_use_distributed_optimizer_unmaps_weight_tensor(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    module = _TinyModel(num_layers=1).cuda()
    weights_before = {name: value.detach().clone() for name, value in module.state_dict().items()}
    for layer in module.layers:
        # TransformerEngine keeps a raw-data alias here; with the distributed
        # optimizer the wrapper must drop it so the remapped params stay the
        # only parameter copy.
        layer.weight_tensor = torch.zeros(1, device="cuda")

    model = _wrap(module, DistributedDataParallelConfig(use_distributed_optimizer=True))

    for layer in module.layers:
        assert layer.weight_tensor is None
    for name, value in model.module.state_dict().items():
        torch.testing.assert_close(value.detach(), weights_before[name], rtol=0, atol=0)


def test_weight_tensor_alias_kept_without_distributed_optimizer(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    module = _TinyModel(num_layers=1).cuda()
    weight_tensor_alias = torch.zeros(1, device="cuda")
    module.layers[-1].weight_tensor = weight_tensor_alias

    _wrap(module, DistributedDataParallelConfig(use_distributed_optimizer=False))

    assert module.layers[-1].weight_tensor is weight_tensor_alias


def test_layerwise_param_gather_dispatches_and_clears_handle_in_forward(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    reference = _TinyModel().cuda()
    torch.manual_seed(123)
    ddp_config = DistributedDataParallelConfig(overlap_param_gather=True, overlap_grad_reduce=True)
    model = _wrap(_TinyModel().cuda(), ddp_config)
    x = torch.randn(4, IN_DIM, device="cuda")

    out = model(x)

    # Without layer-wise parameter lists every bucket skips its gather, but the
    # forward pre-hook still dispatches an empty handle per bucket group and
    # waits on it before the first layer consumes parameters.
    for bucket_group in model.bucket_groups:
        assert bucket_group.param_gather_dispatched is True
        assert bucket_group.param_gather_handle is None
        for bucket in bucket_group.buckets:
            assert bucket.layerwise_gather_list is None
    torch.testing.assert_close(out, reference(x), rtol=0, atol=0)


def test_free_overlap_buffers_waits_on_pending_param_gather(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_param_gather=True
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    for bucket_group in model.bucket_groups:
        for bucket in bucket_group.buckets:
            # Each rank overwrites only its own shard of the flat parameter buffer.
            pgb.shard_buffer(bucket.param_data, world_size)[rank].fill_(float(rank + 1))

    model.start_param_sync(force_dispatch=True)

    model.free_overlap_buffers()

    # free_overlap_buffers waited on the outstanding all-gather: every rank's
    # shard value landed in every rank's buffer and no handle is left pending.
    for bucket_group in model.bucket_groups:
        assert bucket_group.param_gather_handle is None
        for bucket in bucket_group.buckets:
            shards = pgb.shard_buffer(bucket.param_data, world_size)
            for shard_rank, shard in enumerate(shards):
                torch.testing.assert_close(
                    shard, torch.full_like(shard, float(shard_rank + 1)), rtol=0, atol=0
                )
