# Copyright (c) 2026, BAAI. All rights reserved.
"""Emerging optimizers: muon stepping, coefficient validation, Lion and muon factory chains."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
    validate_coefficient_type,
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


def _matrix_param(fill=1.0):
    param = torch.nn.Parameter(torch.full((8, 8), fill, dtype=torch.float32, device=_device()))
    return param


def _wrap_ddp(module):
    return DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        module,
    )


def _build_bf16_chunk():
    matrices = torch.nn.ModuleList(
        [
            torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16, device=_device())
            for _ in range(2)
        ]
    )
    vectors = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.full((64,), 0.5, dtype=torch.bfloat16, device=_device()))
            for _ in range(2)
        ]
    )

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrices = matrices
            self.vectors = vectors

    return _Chunk()


def _set_main_grads(model, value):
    for index, param in enumerate(model.parameters()):
        with torch.no_grad():
            param.main_grad.copy_(torch.full_like(param.main_grad, value + 0.001 * index))


def _momentum_buffer_equals(optimizer, param, expected):
    # buf = grad + momentum * buf from a zero state: after the first step the
    # momentum buffer equals the raw gradient, and a further step with the same
    # gradient makes it equal momentum * grad + grad.
    return any(
        torch.equal(value, expected)
        for value in optimizer.state[param].values()
        if isinstance(value, torch.Tensor)
    )


def _config(**overrides):
    defaults = dict(
        lr=0.01, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=0.0, adam_eps=1e-8
    )
    return OptimizerConfig(**{**defaults, **overrides})


def test_validate_coefficient_type_accepts_quintic():
    assert validate_coefficient_type('quintic') is None


def test_validate_coefficient_type_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported muon coefficient type 'bogus_type'"):
        validate_coefficient_type('bogus_type')


def test_tensor_parallel_muon_rejects_num_ns_steps_below_one():
    param = _matrix_param()
    with pytest.raises(ValueError, match='num_ns_steps must be at least 1, got 0'):
        TensorParallelMuon(params=[param], num_ns_steps=0)


def test_tensor_parallel_muon_step_is_noop_with_zero_lr():
    param = _matrix_param()
    optimizer = TensorParallelMuon(
        params=[param], lr=0.0, momentum=0.95, weight_decay=0.0, pg_collection=None
    )
    before = param.detach().clone()
    param.grad = torch.full_like(param, 0.5)

    optimizer.step()

    assert torch.equal(param.detach(), before)


def test_tensor_parallel_muon_steps_with_none_tp_group_and_negative_partition_dim():
    param = _matrix_param()
    param.partition_dim = -1
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode='duplicated',
    )
    before = param.detach().clone()
    param.grad = torch.linspace(-1.0, 1.0, param.numel()).view_as(param)

    optimizer.step()

    assert not torch.equal(param.detach(), before)
    assert _momentum_buffer_equals(optimizer, param, param.grad)
    assert torch.isfinite(param.detach()).all()

    optimizer.step()
    assert _momentum_buffer_equals(optimizer, param, param.grad * 0.95 + param.grad)


def test_tensor_parallel_muon_uses_process_group_for_partitioned_params(model_parallel):
    model_parallel()
    from megatron.core.process_groups_config import ProcessGroupCollection

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    param = _matrix_param()
    param.partition_dim = 0
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=5,
        pg_collection=pg_collection,
        tp_mode='distributed',
    )
    before = param.detach().clone()
    param.grad = torch.linspace(-1.0, 1.0, param.numel()).view_as(param)

    optimizer.step()

    assert not torch.equal(param.detach(), before)
    assert _momentum_buffer_equals(optimizer, param, param.grad)
    assert torch.isfinite(param.detach()).all()


def test_tensor_parallel_adaptive_muon_sets_group_defaults_and_initializes_state():
    param = _matrix_param()
    optimizer = TensorParallelAdaptiveMuon(
        params=[param],
        lr=0.0,
        momentum=0.95,
        weight_decay=0.0,
        pg_collection=None,
        moment2_method='adamuon',
        beta2=0.9,
        eps=1e-6,
    )

    assert optimizer.param_groups[0]['beta2'] == 0.9
    assert optimizer.param_groups[0]['eps'] == 1e-6
    assert optimizer.moment2_method == 'adamuon'

    before = param.detach().clone()
    param.grad = torch.linspace(-1.0, 1.0, param.numel()).view_as(param)
    optimizer.step()

    assert torch.equal(param.detach(), before)
    state_tensors = [
        value for value in optimizer.state[param].values() if isinstance(value, torch.Tensor)
    ]
    assert state_tensors
    for state_tensor in state_tensors:
        assert state_tensor.shape == param.shape
        assert torch.isfinite(state_tensor).all()


def test_tensor_parallel_adaptive_muon_step_updates_weights():
    param = _matrix_param()
    optimizer = TensorParallelAdaptiveMuon(
        params=[param], lr=0.01, momentum=0.95, weight_decay=0.0, pg_collection=None
    )
    before = param.detach().clone()
    param.grad = torch.linspace(-1.0, 1.0, param.numel()).view_as(param)

    optimizer.step()

    assert not torch.equal(param.detach(), before)
    assert _momentum_buffer_equals(optimizer, param, param.grad)
    assert torch.isfinite(param.detach()).all()
    # AdaptiveMuon adds a second moment on top of the Muon momentum buffer;
    # as a squared-quantity accumulator its first-step value is non-negative.
    second_moment = [
        value
        for value in optimizer.state[param].values()
        if isinstance(value, torch.Tensor)
        and value.shape == param.shape
        and not torch.equal(value, param.grad)
    ]
    assert len(second_moment) == 1
    assert torch.isfinite(second_moment[0]).all()
    assert (second_moment[0] >= 0).all()


def test_get_megatron_optimizer_builds_adaptive_muon_chain(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_bf16_chunk())
    config = _config(
        optimizer='adaptive_muon',
        bf16=True,
        adaptive_muon_moment2_method='adamuon',
        adaptive_muon_beta2=0.95,
        muon_num_ns_steps=5,
        muon_tp_mode='duplicated',
    )
    optimizer = get_megatron_optimizer(config, [ddp_model], use_gloo_process_groups=True)

    assert isinstance(optimizer, ChainedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    muon_child, adam_child = optimizer.chained_optimizers
    assert isinstance(muon_child, Float16OptimizerWithFloat16Params)
    assert isinstance(muon_child.optimizer, TensorParallelAdaptiveMuon)
    assert isinstance(adam_child, Float16OptimizerWithFloat16Params)
    assert not isinstance(adam_child.optimizer, TensorParallelAdaptiveMuon)
    muon_shapes = [p.shape for p in muon_child.get_parameters()]
    adam_shapes = [p.shape for p in adam_child.get_parameters()]
    assert muon_shapes == [torch.Size((8, 8)), torch.Size((8, 8))]
    assert adam_shapes == [torch.Size((64,)), torch.Size((64,))]
    assert all(p.dtype == torch.float32 for p in optimizer.get_parameters())

    original_values = [param.detach().clone() for param in ddp_model.parameters()]
    _set_main_grads(ddp_model, 0.02)
    success, grad_norm, num_zeros_in_grad = optimizer.step()
    assert success is True
    assert grad_norm > 0.0
    assert num_zeros_in_grad is None
    for param, original in zip(ddp_model.parameters(), original_values):
        assert not torch.equal(param.detach(), original)


def test_get_megatron_optimizer_builds_lion_with_default_kwargs(model_parallel):
    from emerging_optimizers.scalar_optimizers import Lion

    model_parallel()
    ddp_model = _wrap_ddp(_build_bf16_chunk())
    config = _config(optimizer='lion', bf16=True)
    optimizer = get_megatron_optimizer(config, [ddp_model], use_gloo_process_groups=True)

    assert isinstance(optimizer, ChainedOptimizer)
    lion_child = optimizer.chained_optimizers[0]
    assert isinstance(lion_child.optimizer, Lion)
    assert lion_child.optimizer.param_groups[0]['lr'] == config.lr
    assert lion_child.optimizer.param_groups[0]['betas'] == (config.adam_beta1, config.adam_beta2)
    assert [p.shape for p in lion_child.get_parameters()] == [
        torch.Size((8, 8)),
        torch.Size((8, 8)),
    ]

    _set_main_grads(ddp_model, 0.02)
    success, _, _ = optimizer.step()
    assert success is True


def test_get_megatron_optimizer_builds_fp32_muon_chain(model_parallel):
    model_parallel()
    chunk = _build_bf16_chunk().float()
    ddp_model = _wrap_ddp(chunk)
    config = _config(optimizer='muon', muon_num_ns_steps=5, muon_tp_mode='duplicated')
    optimizer = get_megatron_optimizer(config, [ddp_model], use_gloo_process_groups=True)

    assert isinstance(optimizer, ChainedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    muon_child, adam_child = optimizer.chained_optimizers
    assert isinstance(muon_child, FP32Optimizer)
    assert isinstance(muon_child.optimizer, TensorParallelMuon)
    assert isinstance(adam_child, FP32Optimizer)
    assert [p.shape for p in muon_child.get_parameters()] == [
        torch.Size((8, 8)),
        torch.Size((8, 8)),
    ]

    original_values = [param.detach().clone() for param in ddp_model.parameters()]
    _set_main_grads(ddp_model, 0.02)
    success, grad_norm, _ = optimizer.step()
    assert success is True
    assert grad_norm > 0.0
    for param, original in zip(ddp_model.parameters(), original_values):
        assert not torch.equal(param.detach(), original)


def test_get_megatron_optimizer_rejects_unknown_emerging_optimizer(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_bf16_chunk())
    config = _config(optimizer='no_such_emerging_optimizer', bf16=True)

    with pytest.raises(
        ValueError, match='Unsupported emerging optimizer: no_such_emerging_optimizer'
    ):
        get_megatron_optimizer(config, [ddp_model])


def test_get_megatron_optimizer_rejects_fp16_for_emerging_optimizer(model_parallel):
    model_parallel()
    ddp_model = _wrap_ddp(_build_bf16_chunk())
    config = _config(optimizer='muon', fp16=True, muon_num_ns_steps=5, muon_tp_mode='duplicated')

    with pytest.raises(ValueError, match='emerging optimizer with fp16 is not supported'):
        get_megatron_optimizer(config, [ddp_model])
