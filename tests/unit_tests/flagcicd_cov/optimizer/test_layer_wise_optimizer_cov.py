# Copyright (c) 2026, BAAI. All rights reserved.
"""LayerWiseDistributedOptimizer: routing tags, per-rank shards, expert buckets, gather flows."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucket as Bucket
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    _bucket_is_managed_by_layer_wise_optimizer,
    tag_params_for_buffer_routing,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _world_size():
    return torch.distributed.get_world_size()


def _build_raw_chunk(num_matrix_params, num_vector_params, mark_expert=False):
    matrices = torch.nn.ModuleList(
        [
            torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16, device=_device())
            for _ in range(num_matrix_params)
        ]
    )
    vectors = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.full((64,), 0.5, dtype=torch.bfloat16, device=_device()))
            for _ in range(num_vector_params)
        ]
    )
    if mark_expert:
        for vector in vectors:
            vector.allreduce = False

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrices = matrices
            self.vectors = vectors

    return _Chunk()


def _wrap_ddp(chunk):
    return DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        chunk,
    )


def _build_ddp_model(num_matrix_params, num_vector_params, mark_expert=False):
    return _wrap_ddp(_build_raw_chunk(num_matrix_params, num_vector_params, mark_expert))


def _set_main_grads(model, value_fn):
    index = 0
    for param in model.parameters():
        grad = value_fn(index, param)
        with torch.no_grad():
            param.main_grad.copy_(grad)
        index += 1


def _layer_wise_config(**overrides):
    defaults = dict(
        optimizer='muon',
        lr=0.01,
        weight_decay=0.0,
        bf16=True,
        use_layer_wise_distributed_optimizer=True,
        use_distributed_optimizer=False,
        muon_num_ns_steps=5,
        muon_tp_mode='duplicated',
        clip_grad=0.0,
    )
    return OptimizerConfig(**{**defaults, **overrides})


def _all_params_identical_across_ranks(params):
    flat = torch.cat([param.detach().view(-1).float() for param in params])
    gathered = [torch.empty_like(flat) for _ in range(_world_size())]
    torch.distributed.all_gather(gathered, flat)
    for other in gathered:
        torch.testing.assert_close(other, flat, rtol=0, atol=0)


def test_bucket_is_managed_by_layer_wise_optimizer_returns_false_for_empty_bucket():
    empty_bucket = Bucket(
        params=[],
        param_data=None,
        grad_data=torch.zeros(1),
        offset=0,
        numel_unpadded=0,
        gradient_scaling_factor=1.0,
        bucket_id=0,
        param_index_map={},
        params_with_extra_main_grads=[],
    )

    assert (
        _bucket_is_managed_by_layer_wise_optimizer(empty_bucket, default_for_untagged=True) is False
    )
    assert (
        _bucket_is_managed_by_layer_wise_optimizer(empty_bucket, default_for_untagged=False)
        is False
    )


def test_tag_params_for_buffer_routing_marks_only_trainable_matrix_params():
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8, bias=True, dtype=torch.bfloat16)
            self.frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)

    model = _Net()
    tag_params_for_buffer_routing([model])

    assert model.linear.weight.is_managed_by_layer_wise_optimizer is True
    assert model.linear.bias.is_managed_by_layer_wise_optimizer is False
    assert not hasattr(model.frozen, 'is_managed_by_layer_wise_optimizer')


def test_bucket_is_managed_by_layer_wise_optimizer_reads_tag_and_default(model_parallel):
    model_parallel()
    tagged_chunk = _build_raw_chunk(num_matrix_params=1, num_vector_params=1)
    tag_params_for_buffer_routing([tagged_chunk])
    tagged_ddp = _wrap_ddp(tagged_chunk)

    tagged_buckets = [
        bucket
        for group in tagged_ddp.bucket_groups
        for bucket in group.buckets
        if bucket.params_list
    ]
    managed = [
        bucket
        for bucket in tagged_buckets
        if _bucket_is_managed_by_layer_wise_optimizer(bucket, default_for_untagged=False)
    ]
    unmanaged = [bucket for bucket in tagged_buckets if bucket not in managed]
    assert managed and unmanaged
    for bucket in managed:
        assert bucket.params_list[0].is_managed_by_layer_wise_optimizer is True
    for bucket in unmanaged:
        assert bucket.params_list[0].is_managed_by_layer_wise_optimizer is False

    untagged_ddp = _wrap_ddp(_build_raw_chunk(num_matrix_params=1, num_vector_params=1))
    for group in untagged_ddp.bucket_groups:
        for bucket in group.buckets:
            if not bucket.params_list:
                continue
            assert (
                _bucket_is_managed_by_layer_wise_optimizer(bucket, default_for_untagged=True)
                is True
            )
            assert (
                _bucket_is_managed_by_layer_wise_optimizer(bucket, default_for_untagged=False)
                is False
            )


def test_layer_wise_optimizer_steps_shards_and_gathers_params_across_ranks(model_parallel):
    model_parallel()
    ddp_model = _build_ddp_model(num_matrix_params=8, num_vector_params=8)
    optimizer = get_megatron_optimizer(
        _layer_wise_config(), [ddp_model], use_gloo_process_groups=True
    )

    assert isinstance(optimizer, LayerWiseDistributedOptimizer)
    # Ping-pong sharding assigns each of the 16 equal-size params to exactly
    # one rank; every rank owns exactly 2 params.
    params = list(ddp_model.parameters())
    assert len(params) == 16
    assert len(optimizer.dp_cp_params_list) == _world_size()
    assigned = [p for shard in optimizer.dp_cp_params_list for p in shard]
    assert len(assigned) == 16
    assert set(id(p) for p in assigned) == set(id(p) for p in params)
    assert optimizer.expt_dp_params_list is None

    original_values = [param.detach().clone() for param in params]
    _set_main_grads(
        ddp_model, lambda index, param: torch.full_like(param, 0.01 + 0.001 * (index % 5)).float()
    )

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm > 0.0
    assert num_zeros_in_grad is None
    for param, original in zip(params, original_values):
        assert not torch.equal(param.detach(), original)
    _all_params_identical_across_ranks(params)


def test_layer_wise_optimizer_with_expert_params_sets_bucket_lists_and_broadcasts(model_parallel):
    model_parallel()
    ddp_model = _build_ddp_model(num_matrix_params=8, num_vector_params=8, mark_expert=True)
    optimizer = get_megatron_optimizer(
        _layer_wise_config(overlap_param_gather=True), [ddp_model], use_gloo_process_groups=True
    )

    assert isinstance(optimizer, LayerWiseDistributedOptimizer)
    assert optimizer.overlap_param_gather is True
    assert optimizer.use_buffer_param_sync is False
    dense_params = [p for p in ddp_model.parameters() if getattr(p, 'allreduce', True)]
    expert_params = [p for p in ddp_model.parameters() if not getattr(p, 'allreduce', True)]
    assert len(dense_params) == 8
    assert len(expert_params) == 8
    assert len(optimizer.expt_dp_params_list) == _world_size()

    # Every bucket with params received per-rank layer-wise param lists whose
    # union covers exactly the bucket's own params.
    for group in ddp_model.bucket_groups + ddp_model.expert_parallel_bucket_groups:
        for bucket in group.buckets:
            if not bucket.params_list:
                continue
            assert len(bucket.layerwise_params_list) == _world_size()
            bucket_param_ids = set(id(p) for p in bucket.params_list)
            listed_param_ids = set(id(p) for shard in bucket.layerwise_params_list for p in shard)
            assert listed_param_ids == bucket_param_ids

    _set_main_grads(
        ddp_model, lambda index, param: torch.full_like(param, 0.01 + 0.001 * (index % 5)).float()
    )
    success, _, _ = optimizer.step()
    assert success is True

    # With overlap_param_gather the step defers the gather; run it explicitly.
    optimizer.allgather_params()
    _all_params_identical_across_ranks(dense_params + expert_params)

    optimizer.broadcast_params()
    _all_params_identical_across_ranks(dense_params + expert_params)


def test_layer_wise_optimizer_state_dict_file_round_trip(model_parallel, tmp_path):
    model_parallel()
    ddp_model = _build_ddp_model(num_matrix_params=4, num_vector_params=4)
    optimizer = get_megatron_optimizer(
        _layer_wise_config(), [ddp_model], use_gloo_process_groups=True
    )
    assert isinstance(optimizer, LayerWiseDistributedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    _set_main_grads(
        ddp_model, lambda index, param: torch.full_like(param, 0.02 + 0.001 * (index % 4)).float()
    )
    success, _, _ = optimizer.step()
    assert success is True

    path = str(tmp_path / 'layerwise_state.pt')
    optimizer.save_state_dict_to_file(path)
    saved_state = torch.load(path)
    assert len(saved_state) == 2

    fresh_model = _build_ddp_model(num_matrix_params=4, num_vector_params=4)
    fresh_optimizer = get_megatron_optimizer(
        _layer_wise_config(), [fresh_model], use_gloo_process_groups=True
    )
    fresh_optimizer.load_state_dict_from_file(path)

    for fresh_child, saved_child in zip(fresh_optimizer.chained_optimizers, saved_state):
        restored = fresh_child.state_dict()
        for param_id, saved_values in saved_child['optimizer']['state'].items():
            live_values = restored['optimizer']['state'][param_id]
            assert set(live_values) == set(saved_values)
            for key, saved_value in saved_values.items():
                torch.testing.assert_close(live_values[key], saved_value)
        for live_group, saved_group in zip(
            restored['optimizer']['param_groups'], saved_child['optimizer']['param_groups']
        ):
            assert set(live_group.keys()) == set(saved_group.keys())
            for key, saved_value in saved_group.items():
                if isinstance(saved_value, torch.Tensor):
                    torch.testing.assert_close(live_group[key].cpu(), saved_value.cpu())
                else:
                    assert live_group[key] == saved_value
        for live_group, saved_group in zip(
            fresh_child.fp32_from_float16_groups, saved_child['fp32_from_fp16_params']
        ):
            for live_param, saved_param in zip(live_group, saved_group):
                torch.testing.assert_close(live_param.detach(), saved_param.detach())
