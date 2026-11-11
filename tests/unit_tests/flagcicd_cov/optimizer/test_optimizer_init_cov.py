# Copyright (c) 2026, BAAI. All rights reserved.
"""Optimizer factory: validation, param group overrides, expert params, CPU offload, MuP."""

import pathlib

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import (
    OptimizerConfig,
    ParamKey,
    _get_megatron_optimizer_based_on_param_groups,
    check_config_overrides_consistency,
    get_megatron_optimizer,
    get_mup_config_overrides,
)
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
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


def _param_group(params, lr=0.01):
    return {
        'params': list(params),
        'is_expert_parallel': False,
        'is_engram_parallel': False,
        'is_vision_model_param': False,
        'default_config': True,
        'wd_mult': 1.0,
        'lr_mult': 1.0,
        'is_decoupled_lr': False,
        'max_lr': lr,
        'min_lr': None,
    }


def _build_chunk(dtype=torch.bfloat16, with_bias=True, with_frozen=False, with_expert=False):
    linear = torch.nn.Linear(8, 8, bias=with_bias, dtype=dtype, device=_device())
    linear.weight.data.fill_(1.0)
    if with_bias:
        linear.bias.data.fill_(0.5)
    extra_params = []
    if with_frozen:
        extra_params.append(
            torch.nn.Parameter(torch.full((4,), 7.0, dtype=dtype, device=_device()))
        )
        extra_params[-1].requires_grad_(False)
    if with_expert:
        extra_params.append(
            torch.nn.Parameter(torch.full((8,), 3.0, dtype=dtype, device=_device()))
        )
        extra_params[-1].allreduce = False

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.extra_params = torch.nn.ParameterList(extra_params)

    return _Chunk()


def _wrap_ddp(chunk):
    return DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        chunk,
    )


def _config(**overrides):
    defaults = dict(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        weight_decay=0.0,
        use_distributed_optimizer=False,
        clip_grad=0.0,
    )
    return OptimizerConfig(**{**defaults, **overrides})


def _set_main_grads(model, value):
    for index, param in enumerate(model.parameters()):
        if param.requires_grad:
            with torch.no_grad():
                param.main_grad.copy_(torch.full_like(param.main_grad, value + 0.001 * index))


@pytest.mark.parametrize(
    "extra_config,match",
    [
        pytest.param(
            dict(
                use_precision_aware_optimizer=True,
                use_distributed_optimizer=True,
                optimizer_cpu_offload=True,
            ),
            'skip_megatron_wrapping=True is incompatible with use_precision_aware_optimizer',
            id='precision-aware',
        ),
        pytest.param(
            dict(optimizer_cpu_offload=True),
            'skip_megatron_wrapping=True is incompatible with optimizer_cpu_offload',
            id='cpu-offload',
        ),
    ],
)
def test_get_megatron_optimizer_rejects_skip_wrapping_with_incompatible_config(extra_config, match):
    config = _config(**extra_config)

    with pytest.raises(ValueError, match=match):
        _get_megatron_optimizer_based_on_param_groups(
            config=config,
            model_chunks=[],
            param_groups=[_param_group([torch.nn.Parameter(torch.ones(2))])],
            skip_megatron_wrapping=True,
        )


def test_get_megatron_optimizer_based_on_param_groups_rejects_unknown_optimizer():
    config = _config(optimizer='rmsprop')

    with pytest.raises(Exception, match='rmsprop optimizer is not supported'):
        _get_megatron_optimizer_based_on_param_groups(
            config=config,
            model_chunks=[],
            param_groups=[_param_group([torch.nn.Parameter(torch.ones(2))])],
        )


def test_check_config_overrides_consistency_rejects_optimizer_override():
    config = _config()
    inconsistent_overrides = {ParamKey(name='*.bias'): {'optimizer': 'sgd'}}

    with pytest.raises(ValueError, match='Field optimizer should not be overriden'):
        check_config_overrides_consistency(config, inconsistent_overrides)

    assert (
        check_config_overrides_consistency(config, {ParamKey(name='*.bias'): {'wd_mult': 0.0}})
        is True
    )


def test_get_megatron_optimizer_applies_config_overrides_and_skips_frozen_params(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(with_bias=True, with_frozen=True))
    frozen_param = [param for param in ddp_model.parameters() if not param.requires_grad].pop()
    config = _config(bf16=True)
    overrides = {ParamKey(name='*.bias'): {'max_lr': 0.02}}

    optimizer = get_megatron_optimizer(config, [ddp_model], config_overrides=overrides)

    assert isinstance(optimizer, ChainedOptimizer)
    group_params = [
        (group['max_lr'], tuple(p.shape))
        for group in optimizer.param_groups
        for p in group['params']
    ]
    assert set(group_params) == {(0.01, (8, 8)), (0.02, (8,))}
    assert all(
        param is not frozen_param for group in optimizer.param_groups for param in group['params']
    )


def test_get_megatron_optimizer_adam_init_state_fn_initializes_inner_states(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(with_bias=True))
    optimizer = get_megatron_optimizer(_config(bf16=True), [ddp_model])

    child = optimizer.chained_optimizers[0]
    assert isinstance(child, Float16OptimizerWithFloat16Params)
    assert child.optimizer.state == {}

    state_dict = child.state_dict(is_loading=True)

    inner_state = state_dict['optimizer']['state']
    assert set(inner_state) == {0, 1}
    for state in inner_state.values():
        assert set(state) == {'exp_avg', 'exp_avg_sq'}
        for moment in state.values():
            assert moment.dtype == torch.float32
            assert torch.equal(moment, torch.zeros_like(moment))


def test_get_megatron_optimizer_sgd_step_matches_reference(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(dtype=torch.float32, with_bias=True))
    config = _config(optimizer='sgd', sgd_momentum=0.0)
    optimizer = get_megatron_optimizer(config, [ddp_model])

    assert isinstance(optimizer.chained_optimizers[0], FP32Optimizer)
    original_values = [param.detach().clone() for param in ddp_model.parameters()]
    _set_main_grads(ddp_model, 0.02)

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm is None
    assert num_zeros_in_grad is None
    grads = [0.02 + 0.001 * index for index, _ in enumerate(ddp_model.parameters())]
    for param, original, grad_value in zip(ddp_model.parameters(), original_values, grads):
        expected = original - config.lr * grad_value
        torch.testing.assert_close(param.detach(), expected, rtol=1e-6, atol=1e-7)


def test_get_megatron_optimizer_lion_branch_wraps_lion_with_bf16(model_parallel):
    from emerging_optimizers.scalar_optimizers import Lion

    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(with_bias=False))
    config = _config(optimizer='lion', bf16=True, lion_beta1=0.9, lion_beta2=0.99)

    optimizer = _get_megatron_optimizer_based_on_param_groups(
        config=config,
        model_chunks=[ddp_model],
        param_groups=[_param_group(ddp_model.parameters(), lr=config.lr)],
    )

    child = optimizer
    assert isinstance(child, Float16OptimizerWithFloat16Params)
    assert isinstance(child.optimizer, Lion)
    assert child.optimizer.param_groups[0]['betas'] == (0.9, 0.99)
    assert child.optimizer.param_groups[0]['lr'] == config.lr

    state_dict = child.state_dict(is_loading=True)
    for state in state_dict['optimizer']['state'].values():
        assert set(state) == {'exp_avg'}
        assert state['exp_avg'].dtype == torch.float32


def test_get_megatron_optimizer_cpu_offload_steps_against_reference(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(dtype=torch.float32, with_bias=True))
    config = _config(
        optimizer='adam',
        decoupled_weight_decay=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=0.5,
    )
    optimizer = get_megatron_optimizer(config, [ddp_model])

    child = optimizer.chained_optimizers[0]
    assert isinstance(child, FP32Optimizer)
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    hybrid = child.optimizer
    assert isinstance(hybrid, HybridDeviceOptimizer)
    assert hybrid.param_update_in_fp32 is True
    assert hybrid.defaults['offload_fraction'] == 0.5
    assert hybrid.defaults['lr'] == config.lr
    weight = ddp_model.module.linear.weight
    bias = ddp_model.module.linear.bias
    assert set(hybrid.gpu_params_map_cpu_copy) == {weight}
    assert hybrid.gpu_params_map_cpu_copy[weight].device.type == 'cpu'
    assert len(hybrid.cpu_optimizers) == 1
    assert hybrid.sub_optimizers == [*hybrid.cpu_optimizers, hybrid.gpu_optimizer]

    original_weight = weight.detach().clone()
    original_bias = bias.detach().clone()
    _set_main_grads(ddp_model, 0.02)

    success, _, _ = optimizer.step()

    assert success is True
    for param, original in ((weight, original_weight), (bias, original_bias)):
        grad = 0.02 + 0.001 * (0 if param is weight else 1)
        reference = torch.nn.Parameter(original.clone())
        reference.grad = torch.full_like(reference, grad)
        torch.optim.AdamW(
            [reference], lr=config.lr, betas=(0.9, 0.999), eps=config.adam_eps, weight_decay=0.0
        ).step()
        torch.testing.assert_close(param.detach(), reference.detach(), rtol=1e-5, atol=1e-6)


def test_get_megatron_optimizer_with_expert_params_and_dump_map(model_parallel, tmp_path):
    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(with_bias=False, with_expert=True))
    config = _config(bf16=True)
    dump_path = str(tmp_path / 'param_group_map')

    optimizer = get_megatron_optimizer(config, [ddp_model], dump_param_to_param_group_map=dump_path)

    assert isinstance(optimizer, ChainedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    dense_child, expert_child = optimizer.chained_optimizers
    dense_shapes = [tuple(p.shape) for p in dense_child.get_parameters()]
    expert_shapes = [tuple(p.shape) for p in expert_child.get_parameters()]
    assert dense_shapes == [(8, 8)]
    assert expert_shapes == [(8,)]

    # Every rank wrote its own shard file into the dump directory; only the
    # coordinating rank also wrote the metadata, which lists the dumped
    # parameter names.
    rank = torch.distributed.get_rank()
    assert (pathlib.Path(dump_path) / f'__{rank}_0.distcp').is_file()
    if rank == 0:
        metadata = FileSystemReader(dump_path).read_metadata()
        assert sorted(metadata.state_dict_metadata) == [
            'module.extra_params.0',
            'module.linear.weight',
        ]

    original_values = [param.detach().clone() for param in ddp_model.parameters()]
    _set_main_grads(ddp_model, 0.02)
    success, _, _ = optimizer.step()
    assert success is True
    for param, original in zip(ddp_model.parameters(), original_values):
        assert not torch.equal(param.detach(), original)


def test_get_megatron_optimizer_splits_first_chunk_for_overlapped_param_gather(model_parallel):
    model_parallel()
    first_chunk = _wrap_ddp(_build_chunk(with_bias=False))
    second_chunk = _wrap_ddp(_build_chunk(with_bias=False))
    config = _config(bf16=True, overlap_param_gather_with_optimizer_step=True)

    optimizer = get_megatron_optimizer(config, [first_chunk, second_chunk])

    assert isinstance(optimizer, ChainedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    assert first_chunk.overlap_param_gather_with_optimizer_step is True
    assert second_chunk.overlap_param_gather_with_optimizer_step is False


def test_get_mup_config_overrides_exclude_muon_matrices_from_lr_scaling(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_chunk(with_bias=True))
    config = _config(
        optimizer='muon',
        bf16=True,
        decoupled_lr=0.004,
        min_lr=0.0004,
        muon_num_ns_steps=5,
        muon_tp_mode='duplicated',
    )
    mup_overrides = get_mup_config_overrides(config, mup_width_mult=2.0)

    assert mup_overrides

    optimizer = get_megatron_optimizer(
        config, [ddp_model], config_overrides=mup_overrides, use_gloo_process_groups=True
    )

    assert isinstance(optimizer, ChainedOptimizer)
    group_lrs = {
        (group['max_lr'], tuple(p.shape))
        for group in optimizer.param_groups
        for p in group['params']
    }
    # Muon-managed matrices and vector-like params are excluded from MuP LR
    # scaling, so both groups keep the base LR instead of lr / mup_width_mult.
    assert group_lrs == {(0.01, (8, 8)), (0.01, (8,))}
