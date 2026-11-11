# Copyright (c) 2026, BAAI. All rights reserved.
"""DistributedOptimizer checkpoints: state dict resume, parameter-state files, sharding types."""

import pytest
import torch

from megatron.core.dist_checkpointing import LocalNonpersistentObject, ShardedObject, ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
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


def _build_stepped_optimizer(model_parallel, step=True, **config_overrides):
    model_parallel()
    torch.manual_seed(1234)
    linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=_device())
    linear.weight.data.fill_(1.0)
    ddp_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        linear,
    )
    defaults = dict(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        weight_decay=0.0,
        bf16=True,
        use_distributed_optimizer=True,
        clip_grad=0.0,
    )
    config = OptimizerConfig(**{**defaults, **config_overrides})
    optimizer = get_megatron_optimizer(config, [ddp_model])

    if step:
        grad = torch.arange(256, dtype=torch.float32, device=_device()).reshape(16, 16) / 8.0
        grad.view(-1)[::64] = 0.0
        with torch.no_grad():
            ddp_model.module.weight.main_grad.copy_(grad)
        optimizer.step()
    return ddp_model, optimizer


def _capture_inner_states(distributed_optimizer):
    captured = {}
    for group in distributed_optimizer.optimizer.param_groups:
        for param in group['params']:
            captured[param] = {
                key: value.clone()
                for key, value in distributed_optimizer.optimizer.state[param].items()
                if isinstance(value, torch.Tensor)
            }
    return captured


def _perturb_inner_states(distributed_optimizer):
    for group in distributed_optimizer.optimizer.param_groups:
        for param in group['params']:
            for value in distributed_optimizer.optimizer.state[param].values():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    value.add_(100.0)


def _assert_states_restored(distributed_optimizer, captured):
    for group in distributed_optimizer.optimizer.param_groups:
        for param in group['params']:
            for key, value in distributed_optimizer.optimizer.state[param].items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    torch.testing.assert_close(value, captured[param][key])


def _owned_grad_slice(distributed_optimizer):
    # The deterministic grad is arange(256) / 8 with every 64th entry zeroed
    # over a single 256-element parameter; return the slice of it owned by
    # this DP rank (tp=1/pp=1, so the DP group spans the whole world).
    shard_numel = 256 // torch.distributed.get_world_size()
    dp_rank = distributed_optimizer.data_parallel_group.rank()
    grad = torch.arange(256, dtype=torch.float32) / 8.0
    grad[::64] = 0.0
    return grad[dp_rank * shard_numel : (dp_rank + 1) * shard_numel]


def test_distributed_optimizer_state_dict_round_trip_in_memory(model_parallel):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    distributed_optimizer = optimizer.chained_optimizers[0]
    saved_state = optimizer.state_dict()
    assert set(saved_state.keys()) == {'optimizer'}
    saved_groups = [dict(group) for group in saved_state['optimizer']['param_groups']]

    # Load into a fresh, never-stepped optimizer, exactly like a checkpoint
    # resume. The moments themselves travel via save_parameter_state (covered
    # by the test below), so this round trip restores the param groups and
    # allocates this DP rank's shard of every optimizer state tensor.
    _, fresh_optimizer = _build_stepped_optimizer(model_parallel, step=False)
    fresh_distributed = fresh_optimizer.chained_optimizers[0]
    fresh_optimizer.load_state_dict(saved_state)

    restored_groups = fresh_optimizer.state_dict()['optimizer']['param_groups']
    assert len(restored_groups) == len(saved_groups)
    for restored_group, saved_group in zip(restored_groups, saved_groups):
        assert set(restored_group.keys()) == set(saved_group.keys())
        for key, saved_value in saved_group.items():
            if isinstance(saved_value, torch.Tensor):
                torch.testing.assert_close(restored_group[key].cpu(), saved_value.cpu())
            else:
                assert restored_group[key] == saved_value

    for fresh_group, stepped_group in zip(
        fresh_distributed.optimizer.param_groups, distributed_optimizer.optimizer.param_groups
    ):
        for fresh_param, stepped_param in zip(fresh_group['params'], stepped_group['params']):
            fresh_state = fresh_distributed.optimizer.state[fresh_param]
            stepped_state = distributed_optimizer.optimizer.state[stepped_param]
            assert set(fresh_state) == set(stepped_state)
            for key, stepped_value in stepped_state.items():
                assert isinstance(stepped_value, torch.Tensor)
                fresh_value = fresh_state[key]
                assert fresh_value.shape == stepped_value.shape
                assert fresh_value.dtype == stepped_value.dtype
                if key == 'step':
                    torch.testing.assert_close(fresh_value, stepped_value)


def test_distributed_optimizer_save_and_load_parameter_state(model_parallel, tmp_path):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    distributed_optimizer = optimizer.chained_optimizers[0]
    checkpoint = str(tmp_path / 'dist_optim.pt')

    optimizer.save_parameter_state(checkpoint)
    captured = _capture_inner_states(distributed_optimizer)
    _perturb_inner_states(distributed_optimizer)

    optimizer.load_parameter_state(checkpoint)

    _assert_states_restored(distributed_optimizer, captured)
    # One unclipped Adam step ran on the deterministic grad, so the restored
    # first moments equal (1 - beta1) * grad over this rank's owned slice.
    owned_grad = _owned_grad_slice(distributed_optimizer)
    for group in distributed_optimizer.optimizer.param_groups:
        for param in group['params']:
            torch.testing.assert_close(
                distributed_optimizer.optimizer.state[param]['exp_avg'].cpu(),
                0.1 * owned_grad,
                rtol=1e-5,
                atol=1e-6,
            )


def test_distributed_optimizer_load_parameter_state_legacy_format(model_parallel, tmp_path):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    distributed_optimizer = optimizer.chained_optimizers[0]
    modern_path = str(tmp_path / 'modern.pt')
    legacy_path = str(tmp_path / 'legacy.pt')

    optimizer.save_parameter_state(modern_path)
    captured = _capture_inner_states(distributed_optimizer)

    dp_rank = distributed_optimizer.data_parallel_group.rank()
    if dp_rank == 0:
        modern_state = torch.load(modern_path)
        legacy_state = {}
        for gbuf_idx, dtype_map in modern_state.items():
            if gbuf_idx == 'buckets_coalesced':
                continue
            world_tensors = next(iter(dtype_map.values()))
            legacy_state[gbuf_idx] = {
                torch.float32: {
                    key: [world_tensors[key]] for key in ('param', 'exp_avg', 'exp_avg_sq')
                }
            }
        torch.save(legacy_state, legacy_path)
    torch.distributed.barrier()

    _perturb_inner_states(distributed_optimizer)

    optimizer.load_parameter_state(legacy_path, update_legacy_format=True)

    _assert_states_restored(distributed_optimizer, captured)
    owned_grad = _owned_grad_slice(distributed_optimizer)
    for group in distributed_optimizer.optimizer.param_groups:
        for param in group['params']:
            torch.testing.assert_close(
                distributed_optimizer.optimizer.state[param]['exp_avg_sq'].cpu(),
                0.001 * owned_grad * owned_grad,
                rtol=1e-4,
                atol=1e-6,
            )


def test_distributed_optimizer_sharded_state_dict_fully_reshardable(model_parallel):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    model_sharded_state_dict = ddp_model.sharded_state_dict()

    sharded_state_dict = optimizer.sharded_state_dict(
        model_sharded_state_dict,
        metadata={'distrib_optim_sharding_type': 'fully_reshardable'},
        is_loading=True,
    )

    assert sharded_state_dict['param_state_sharding_type'] == 'fully_reshardable'
    param_state = sharded_state_dict['param_state']
    assert len(param_state) == 1
    entry = param_state[0]
    assert set(entry.keys()) == {'param', 'exp_avg', 'exp_avg_sq'}
    for key, sharded_tensor in entry.items():
        assert isinstance(sharded_tensor, ShardedTensor)
        assert sharded_tensor.key.startswith(f'optimizer.state.{key}.')
        assert sharded_tensor.key.endswith('.weight')
        assert tuple(sharded_tensor.data.shape) == (16, 16)


def test_distributed_optimizer_sharded_state_dict_dp_zero_gather_scatter(model_parallel):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    distributed_optimizer = optimizer.chained_optimizers[0]
    model_sharded_state_dict = ddp_model.sharded_state_dict()

    sharded_state_dict = optimizer.sharded_state_dict(
        model_sharded_state_dict, metadata={'distrib_optim_sharding_type': 'dp_zero_gather_scatter'}
    )

    assert sharded_state_dict['param_state_sharding_type'] == 'dp_zero_gather_scatter'
    param_state = sharded_state_dict['param_state']
    if distributed_optimizer.data_parallel_group.rank() == 0:
        assert isinstance(param_state, ShardedObject)
        assert 'param_state' in param_state.key
    else:
        assert isinstance(param_state, LocalNonpersistentObject)


def test_distributed_optimizer_sharded_state_dict_fs_model_space(model_parallel):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    model_sharded_state_dict = ddp_model.sharded_state_dict()

    sharded_state_dict = optimizer.sharded_state_dict(model_sharded_state_dict)

    assert sharded_state_dict['param_state_sharding_type'] == 'fully_sharded_model_space'
    param_state = sharded_state_dict['param_state']
    assert len(param_state) == 1
    entry = param_state[0]
    assert set(entry.keys()) == {'fp32_param', 'exp_avg', 'exp_avg_sq'}
    fp32_sharded_tensor = entry['fp32_param']
    assert isinstance(fp32_sharded_tensor, ShardedTensor)
    assert fp32_sharded_tensor.key.startswith('optimizer.state.fp32_param.')
    assert fp32_sharded_tensor.key.endswith('.weight')
    # The fp32 state is sharded over DP; with tp=1/pp=1 the DP group spans the
    # whole world and each rank owns one contiguous slice of the 256 elements.
    shard_numel = 256 // torch.distributed.get_world_size()
    expected_start = torch.distributed.get_rank() * shard_numel
    assert fp32_sharded_tensor.flattened_range == slice(
        expected_start, expected_start + shard_numel
    )
    assert fp32_sharded_tensor.replica_id[-1] == 0


def test_distributed_optimizer_sharded_state_dict_dp_reshardable(model_parallel):
    ddp_model, optimizer = _build_stepped_optimizer(model_parallel)
    model_sharded_state_dict = ddp_model.sharded_state_dict()

    sharded_state_dict = optimizer.sharded_state_dict(
        model_sharded_state_dict, metadata={'distrib_optim_sharding_type': 'dp_reshardable'}
    )

    assert sharded_state_dict['param_state_sharding_type'] == 'dp_reshardable'
    param_state = sharded_state_dict['param_state']
    assert set(param_state) == {0, 'per_bucket_numel', 'per_bucket_numel_unpadded'}
    per_bucket_numel = param_state['per_bucket_numel']
    assert isinstance(per_bucket_numel, ShardedObject)
    assert per_bucket_numel.key == 'optimizer.distributed.dp_group_idx_0.per_bucket_numel'
    # The single bf16 model param fills one unpadded 256-element bucket.
    assert next(iter(per_bucket_numel.data[0].values())) == [256]

    buckets = param_state[0][(torch.bfloat16, torch.bfloat16)]
    assert len(buckets) == 1
    assert len(buckets[0]) == 1
    entry = buckets[0][0]
    assert set(entry) == {'param', 'exp_avg', 'exp_avg_sq', 'padding'}
    assert isinstance(entry['padding'], LocalNonpersistentObject)
    world_size = torch.distributed.get_world_size()
    dp_rank = torch.distributed.get_rank()
    for key in ('param', 'exp_avg', 'exp_avg_sq'):
        sharded_tensor = entry[key]
        assert isinstance(sharded_tensor, ShardedTensor)
        assert 'bucket_idx_0' in sharded_tensor.key
        assert sharded_tensor.key.endswith(f'.{key}')
        # Each DP rank owns a contiguous 256 / world_size slice of the bucket.
        assert tuple(sharded_tensor.data.shape) == (256 // world_size,)
        assert sharded_tensor.global_shape == (256,)
        assert sharded_tensor.global_offset == (dp_rank * (256 // world_size),)
        assert sharded_tensor.replica_id == (0, 0, 0)
