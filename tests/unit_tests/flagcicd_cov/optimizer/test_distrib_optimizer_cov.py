# Copyright (c) 2026, BAAI. All rights reserved.
"""DistributedOptimizer stepping: Adam parity, clipping, fp16 unscaling, zero_grad, param reload."""

import pytest
import torch

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


def _build_optimizer(model_parallel, **config_overrides):
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
    return ddp_model, optimizer


def _deterministic_grad():
    grad = torch.arange(256, dtype=torch.float32, device=_device()).reshape(16, 16) / 8.0
    grad.view(-1)[::64] = 0.0
    return grad


def _set_main_grads(ddp_model, grad):
    with torch.no_grad():
        ddp_model.module.weight.main_grad.copy_(grad)


def _reference_adam_step(weight_before, grad, lr=0.01):
    reference = torch.nn.Parameter(weight_before.detach().clone().float())
    reference.grad = grad.clone()
    torch.optim.Adam([reference], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
    return reference.detach().to(weight_before.dtype)


def test_distributed_optimizer_step_matches_reference_adam_on_full_params(model_parallel):
    ddp_model, optimizer = _build_optimizer(model_parallel)
    weight_before = ddp_model.module.weight.detach().clone()
    grad = _deterministic_grad()
    _set_main_grads(ddp_model, grad)

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    # ChainedOptimizer.step always computes the grad norm; each DP rank only
    # contributes its own shard, so the result is the norm of the full gradient.
    assert grad_norm == pytest.approx(grad.double().pow(2).sum().sqrt().item(), rel=1e-4)
    assert num_zeros_in_grad is None
    torch.testing.assert_close(
        ddp_model.module.weight.detach(),
        _reference_adam_step(weight_before, grad),
        rtol=1.6e-2,
        atol=1e-2,
    )


def test_distributed_optimizer_step_clips_to_full_grad_norm_and_counts_zeros(model_parallel):
    ddp_model, optimizer = _build_optimizer(
        model_parallel, clip_grad=1.0, log_num_zeros_in_grad=True
    )
    weight_before = ddp_model.module.weight.detach().clone()
    grad = _deterministic_grad()
    _set_main_grads(ddp_model, grad)

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    # Each DP rank contributes its own shard, so the reduced norm equals the
    # norm of the full gradient.
    assert grad_norm == pytest.approx(grad.double().pow(2).sum().sqrt().item(), rel=1e-4)
    assert num_zeros_in_grad == 4
    clip_coeff = 1.0 / (grad_norm + 1.0e-6)
    torch.testing.assert_close(
        ddp_model.module.weight.detach(),
        _reference_adam_step(weight_before, grad * clip_coeff),
        rtol=1.6e-2,
        atol=1e-2,
    )


def test_distributed_optimizer_fp16_step_unscales_grads_before_update(model_parallel):
    model_parallel()
    torch.manual_seed(1234)
    linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16, device=_device())
    linear.weight.data.fill_(1.0)
    ddp_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        linear,
    )
    config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        weight_decay=0.0,
        fp16=True,
        use_distributed_optimizer=True,
        clip_grad=0.0,
        initial_loss_scale=4.0,
        hysteresis=2,
        loss_scale_window=100,
    )
    optimizer = get_megatron_optimizer(config, [ddp_model])
    weight_before = ddp_model.module.weight.detach().clone()
    grad = _deterministic_grad()

    # Grads are stored scaled by the loss scale; the optimizer must unscale
    # them before the Adam update.
    with torch.no_grad():
        ddp_model.module.weight.main_grad.copy_(4.0 * grad)

    success, _, _ = optimizer.step()

    assert success is True
    torch.testing.assert_close(
        ddp_model.module.weight.detach(),
        _reference_adam_step(weight_before, grad),
        rtol=1e-3,
        atol=1e-3,
    )


def test_distributed_optimizer_zero_grad_clears_shard_grads(model_parallel):
    ddp_model, optimizer = _build_optimizer(model_parallel)
    _set_main_grads(ddp_model, _deterministic_grad())
    optimizer.step()
    distributed_optimizer = optimizer.chained_optimizers[0]
    shard_params = [
        p for group in distributed_optimizer.shard_fp32_from_float16_groups for p in group
    ]
    assert all(p.grad is not None for p in shard_params)

    optimizer.zero_grad(set_to_none=False)
    assert all(torch.equal(p.grad, torch.zeros_like(p.grad)) for p in shard_params)

    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in shard_params)


def test_distributed_optimizer_step_defers_param_sync_with_grad_buf_reuse(model_parallel):
    ddp_model, optimizer = _build_optimizer(model_parallel, reuse_grad_buf_for_mxfp8_param_ag=True)
    weight_before = ddp_model.module.weight.detach().clone()
    grad = _deterministic_grad()
    _set_main_grads(ddp_model, grad)

    success, _, _ = optimizer.step()

    assert success is True
    torch.testing.assert_close(
        ddp_model.module.weight.detach(),
        _reference_adam_step(weight_before, grad),
        rtol=1.6e-2,
        atol=1e-2,
    )


def test_distributed_optimizer_with_frozen_params_steps_as_stub(model_parallel):
    model_parallel()
    torch.manual_seed(1234)
    linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=_device())
    linear.weight.data.fill_(1.0)
    linear.weight.requires_grad_(False)
    ddp_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        linear,
    )
    config = OptimizerConfig(optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=True)
    optimizer = get_megatron_optimizer(config, [ddp_model])
    distributed_optimizer = optimizer.chained_optimizers[0]

    assert distributed_optimizer.is_stub_optimizer is True

    success, grad_norm, num_zeros_in_grad = optimizer.step()

    assert success is True
    assert grad_norm == 0.0
    assert num_zeros_in_grad is None

    optimizer.zero_grad()

    # A stub optimizer has no parameter state to load and must not touch the file.
    optimizer.load_parameter_state('/nonexistent/stub/path.pt')


def test_distributed_optimizer_param_name_uses_model_chunk_names(model_parallel):
    model_parallel()
    torch.manual_seed(1234)
    chunk_specs = [('l0', 1.0), ('l1', 2.0)]
    model_chunks = []
    for name, value in chunk_specs:
        chunk = torch.nn.ModuleDict(
            {name: torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=_device())}
        )
        chunk[name].weight.data.fill_(value)
        model_chunks.append(
            DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                chunk,
            )
        )
    config = OptimizerConfig(optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=True)
    optimizer = get_megatron_optimizer(config, model_chunks)
    distributed_optimizer = optimizer.chained_optimizers[0]

    assert (
        distributed_optimizer._param_name(model_chunks[0].module['l0'].weight) == 'module.l0.weight'
    )
    assert (
        distributed_optimizer._param_name(model_chunks[1].module['l1'].weight) == 'module.l1.weight'
    )


def test_distributed_optimizer_param_name_rejects_conflicting_chunk_names(model_parallel):
    model_parallel()
    torch.manual_seed(1234)
    model_chunks = []
    for _ in range(2):
        chunk = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=_device())
        chunk.weight.data.fill_(1.0)
        model_chunks.append(
            DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                chunk,
            )
        )
    config = OptimizerConfig(optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=True)
    optimizer = get_megatron_optimizer(config, model_chunks)
    distributed_optimizer = optimizer.chained_optimizers[0]

    with pytest.raises(ValueError, match='Parameter names conflict between model chunks'):
        distributed_optimizer._param_name(model_chunks[0].module.weight)


def test_distributed_optimizer_reload_model_params_from_model_and_state_dict(model_parallel):
    model_parallel()
    torch.manual_seed(1234)
    model_chunks = []
    for name in ('l0', 'l1'):
        chunk = torch.nn.ModuleDict(
            {name: torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=_device())}
        )
        chunk[name].weight.data.fill_(1.0)
        model_chunks.append(
            DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                chunk,
            )
        )
    config = OptimizerConfig(optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=True)
    optimizer = get_megatron_optimizer(config, model_chunks)
    distributed_optimizer = optimizer.chained_optimizers[0]

    for chunk in model_chunks:
        for param in chunk.parameters():
            param.data.fill_(2.0)

    optimizer.reload_model_params()
    for group in distributed_optimizer.param_groups:
        for main_param in group['params']:
            assert main_param.dtype == torch.float32
            torch.testing.assert_close(
                main_param.detach(), torch.full_like(main_param, 2.0), atol=0, rtol=0
            )

    state_dict = {
        'model0': {'l0.weight': torch.full((16, 16), 3.0, dtype=torch.bfloat16, device=_device())},
        'model1': {'l1.weight': torch.full((16, 16), 3.0, dtype=torch.bfloat16, device=_device())},
    }
    optimizer.reload_model_params(state_dict)

    for chunk in model_chunks:
        for param in chunk.parameters():
            torch.testing.assert_close(param.detach(), torch.full_like(param, 2.0), atol=0, rtol=0)
    for group in distributed_optimizer.param_groups:
        for main_param in group['params']:
            torch.testing.assert_close(
                main_param.detach(), torch.full_like(main_param, 3.0), atol=0, rtol=0
            )
