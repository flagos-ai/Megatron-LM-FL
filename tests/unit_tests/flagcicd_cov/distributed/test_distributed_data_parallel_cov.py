# Copyright (c) 2026, BAAI. All rights reserved.
"""DistributedDataParallel grad reduction to the data-parallel average, buffer offload, and hook wiring."""

import pytest
import torch

from megatron.core import parallel_state
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


class _EngramModel(torch.nn.Module):
    def __init__(self, in_dim=8, out_dim=8):
        super().__init__()
        self.dense = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.engram_embedding = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.engram_embedding.weight.is_engram_embedding = True

    def forward(self, x):
        return self.dense(x) + self.engram_embedding(x)


class _DelayedWgradModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.linear.weight.skip_backward_post_hook = True
        self.registered_hooks = []

    def register_wgrad_accumulation_and_reduce_hooks(self, hook):
        self.registered_hooks.append(hook)

    def forward(self, x):
        return self.linear(x)


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


def _reference_grads(model, x):
    reference = _TinyModel().cuda()
    with torch.no_grad():
        for ref_p, ddp_p in zip(reference.parameters(), model.parameters()):
            ref_p.copy_(ddp_p)
    reference.zero_grad()
    reference(x).sum().backward()
    return [param.grad.detach().clone() for param in reference.parameters()]


def _allreduce_average(tensor):
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
    return tensor


@pytest.mark.parametrize("overlap_grad_reduce", [True, False])
def test_grad_reduce_matches_dp_average_of_reference_grads(model_parallel, overlap_grad_reduce):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123 + torch.distributed.get_rank())
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=overlap_grad_reduce,
        check_for_nan_in_grad=True,
        check_for_large_grads=True,
        bucket_size=512,
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)
    x = torch.randn(4, IN_DIM, device="cuda") * (torch.distributed.get_rank() + 1)
    model.zero_grad_buffer()
    model(x).sum().backward()
    model.finish_grad_sync()

    ref_grads = _reference_grads(model, x)
    for param, ref_grad in zip(model.parameters(), ref_grads):
        torch.testing.assert_close(
            param.main_grad, _allreduce_average(ref_grad), rtol=1e-5, atol=1e-5
        )
    if overlap_grad_reduce:
        for param in model.parameters():
            assert param.grad is None


def test_zero_grad_buffer_resets_accumulation_state(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _wrap(_TinyModel().cuda(), DistributedDataParallelConfig(overlap_grad_reduce=True))
    x = torch.randn(4, IN_DIM, device="cuda")
    model(x).sum().backward()
    assert torch.count_nonzero(model.buffers[0].grad_data) > 0
    for param in model.params_with_grad:
        param.grad_added_to_main_grad = True

    model.zero_grad_buffer()

    assert torch.count_nonzero(model.buffers[0].grad_data) == 0
    for param in model.params_with_grad:
        assert param.grad_added_to_main_grad is False
    for bucket_group in model.bucket_groups:
        assert bucket_group.is_last_microbatch is True
        assert bucket_group.grad_reduce_finished is False


def test_frozen_params_are_skipped_from_grad_buffers(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    module = _TinyModel(num_layers=1).cuda()
    module.layers[-1].weight.requires_grad_(False)
    model = _wrap(module, DistributedDataParallelConfig(overlap_grad_reduce=True))
    trainable = [p for p in module.parameters() if p.requires_grad]
    assert model.params_with_grad == trainable
    assert not hasattr(module.layers[-1].weight, "main_grad")
    for buffer in model.buffers:
        assert module.layers[-1].weight not in buffer.param_index_map


def test_broadcast_params_syncs_dense_and_expert_weights(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    module = _TinyModel(num_layers=1).cuda()
    module.layers[-1].weight.allreduce = False
    with torch.no_grad():
        for param in module.parameters():
            param.fill_(float(torch.distributed.get_rank()))
    model = _wrap(module, DistributedDataParallelConfig())

    model.broadcast_params()

    world_size = torch.distributed.get_world_size()
    for param in module.parameters():
        gathered = [torch.empty_like(param) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, param.data)
        for shard in gathered:
            torch.testing.assert_close(shard, gathered[0], rtol=0, atol=0)
        torch.testing.assert_close(param.data, torch.zeros_like(param), rtol=0, atol=0)


@pytest.mark.parametrize("synchronize,empty_cache", [(True, True), (False, False)])
def test_offload_and_restore_grad_buffers_zeroes_grads(model_parallel, synchronize, empty_cache):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _wrap(_TinyModel().cuda(), DistributedDataParallelConfig())
    x = torch.randn(4, IN_DIM, device="cuda")
    model(x).sum().backward()

    model.offload_grad_buffers(synchronize=synchronize, empty_cache=empty_cache)
    model.restore_grad_buffers(synchronize=synchronize)

    for buffer in model.buffers:
        assert buffer.grad_data.storage().size() > 0
        assert torch.count_nonzero(buffer.grad_data) == 0


def test_offload_and_reload_param_buffers_preserves_values(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _wrap(
        _TinyModel().cuda(), DistributedDataParallelConfig(use_distributed_optimizer=True)
    )
    buffer = model.buffers[0]
    param_data_before = buffer.param_data.detach().clone()

    buffer.offload_to_cpu(move_params=True, move_grads=True)
    assert buffer.param_data.storage().size() == 0

    buffer.reload_from_cpu(move_params=True, move_grads=True)
    assert buffer.param_data.storage().size() == param_data_before.numel()
    torch.testing.assert_close(buffer.param_data, param_data_before, rtol=0, atol=0)
    assert torch.count_nonzero(buffer.grad_data) == 0


def test_start_grad_sync_scales_then_sums_across_dp_ranks(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _wrap(_TinyModel().cuda(), DistributedDataParallelConfig(overlap_grad_reduce=False))
    model.zero_grad_buffer()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    for buffer in model.buffers:
        buffer.grad_data.fill_(float(rank + 1))

    model.start_grad_sync()

    expected = sum(range(1, world_size + 1)) / world_size
    for buffer in model.buffers:
        torch.testing.assert_close(
            buffer.grad_data, torch.full_like(buffer.grad_data, expected), rtol=1e-5, atol=1e-5
        )


def test_average_in_collective_averages_grads_in_collective(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123 + torch.distributed.get_rank())
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True, average_in_collective=True, bucket_size=512
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)
    x = torch.randn(4, IN_DIM, device="cuda") * (torch.distributed.get_rank() + 1)
    model.zero_grad_buffer()
    model(x).sum().backward()
    model.finish_grad_sync()

    ref_grads = _reference_grads(model, x)
    for param, ref_grad in zip(model.parameters(), ref_grads):
        torch.testing.assert_close(
            param.main_grad, _allreduce_average(ref_grad), rtol=1e-5, atol=1e-5
        )


def test_engram_params_use_engram_dp_group_and_scaling_factor(model_parallel):
    model_parallel(engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123 + torch.distributed.get_rank())
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, average_in_collective=True)
    module = _EngramModel().cuda()
    model = _wrap(module, ddp_config)

    engram_group = parallel_state.get_engram_data_parallel_group()
    assert len(model.engram_embedding_buffers) == 1
    engram_buffer = model.engram_embedding_buffers[0]
    assert engram_buffer.data_parallel_group is engram_group
    assert len(model.buffers) == 1
    assert model.buffers[0].data_parallel_group is not engram_group
    for bucket in engram_buffer.buckets:
        assert bucket.gradient_scaling_factor == 0.5
    for bucket in model.buffers[0].buckets:
        assert bucket.gradient_scaling_factor == 1.0

    x = torch.randn(4, 8, device="cuda") * (torch.distributed.get_rank() + 1)
    model.zero_grad_buffer()
    model(x).sum().backward()
    model.finish_grad_sync()

    reference = _EngramModel().cuda()
    with torch.no_grad():
        reference.dense.weight.copy_(module.dense.weight)
        reference.engram_embedding.weight.copy_(module.engram_embedding.weight)
    reference.zero_grad()
    reference(x).sum().backward()
    expected_engram = reference.engram_embedding.weight.grad * 0.5
    torch.distributed.all_reduce(
        expected_engram, op=torch.distributed.ReduceOp.AVG, group=engram_group
    )
    torch.testing.assert_close(
        module.engram_embedding.weight.main_grad, expected_engram, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        module.dense.weight.main_grad,
        _allreduce_average(reference.dense.weight.grad),
        rtol=1e-5,
        atol=1e-5,
    )

    with torch.no_grad():
        for param in module.parameters():
            param.fill_(float(torch.distributed.get_rank()))
    model.broadcast_params()
    engram_gathered = [torch.empty_like(module.engram_embedding.weight) for _ in range(4)]
    torch.distributed.all_gather(
        engram_gathered, module.engram_embedding.weight.data, group=engram_group
    )
    for shard in engram_gathered:
        torch.testing.assert_close(shard, engram_gathered[0], rtol=0, atol=0)


def test_multiple_optimizer_instances_assign_inter_group(model_parallel):
    model_parallel(num_distributed_optimizer_instances=2)
    model_parallel_cuda_manual_seed(123)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=2,
        average_in_collective=True,
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)

    inter_group = parallel_state.get_inter_distributed_optimizer_instance_group()
    assert model.inter_dist_opt_group is inter_group
    for bucket_group in model.bucket_groups:
        assert bucket_group.inter_distributed_optimizer_instance_group is inter_group
        assert isinstance(bucket_group.communication_stream, torch.cuda.Stream)
    # All bucket groups from the same partition share one dedicated communication stream.
    assert {group.communication_stream for group in model.bucket_groups} == {
        model.bucket_groups[0].communication_stream
    }
    assert model.bucket_groups[0].intra_distributed_optimizer_instance_size == 4


def test_reduce_scatter_fp32_accumulation_links_predecessor_groups(model_parallel, monkeypatch):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    # Constructing fp32-accumulation bucket groups rebinds the module-level
    # reduce-scatter helper; registering the pre-test value here undoes that
    # rebinding when the test finishes.
    monkeypatch.setattr(pgb, "dist_reduce_scatter_func", pgb.dist_reduce_scatter_func)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        reduce_scatter_with_fp32_accumulation=True,
        bucket_size=256,
    )
    model = _wrap(_TinyModel().cuda(), ddp_config)

    bucket_groups = model.bucket_groups
    assert len(bucket_groups) == 3
    for bucket_group in bucket_groups:
        assert len(bucket_group.buckets) == 1
    for later, earlier in zip(bucket_groups[1:], bucket_groups[:-1]):
        assert later.previous_grad_reduce_bucket_group is earlier


def test_delay_wgrad_compute_registers_module_level_hook(model_parallel):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    module = _DelayedWgradModule().cuda()
    ddp_config = DistributedDataParallelConfig(delay_wgrad_compute=True)
    model = _wrap(module, ddp_config)
    assert len(module.registered_hooks) == 1
    assert callable(module.registered_hooks[0])
    assert model.grad_accs == []
