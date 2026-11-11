# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronFSDP one-step training matches a single-process PyTorch reference across sharding strategies."""

import pytest
import torch
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    MixedPrecisionPolicy,
    fully_shard_model,
    fully_shard_optimizer,
)
from tests.unit_tests.test_utilities import Utils

DIM = 8
LR = 0.01


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TinyFSDPModel(torch.nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TiedWeightsModel(torch.nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim, bias=False)
        self.head = torch.nn.Linear(dim, dim, bias=False)
        self.head.weight = self.fc.weight

    def forward(self, x):
        return self.head(torch.relu(self.fc(x)))


def _rank_inputs():
    """Deterministic per-rank (input, target) pairs known identically on every rank."""
    pairs = []
    for rank in range(torch.distributed.get_world_size()):
        generator = torch.Generator().manual_seed(4100 + rank)
        x = torch.randn(4, DIM, generator=generator).cuda()
        target = torch.randn(4, DIM, generator=generator).cuda()
        pairs.append((x, target))
    return pairs


def _eval_input():
    generator = torch.Generator().manual_seed(9999)
    return torch.randn(4, DIM, generator=generator).cuda()


def _reference_step(module_class, pairs, *, frozen_second_layer=False):
    """Run the same warmup + one gradient step as fully_shard_optimizer + one FSDP step."""
    torch.manual_seed(123)
    reference = module_class().cuda()
    if frozen_second_layer:
        for param in _second_layer_params(reference):
            param.requires_grad_(False)
    optimizer = torch.optim.Adam(reference.parameters(), lr=LR)
    world_size = torch.distributed.get_world_size()

    # fully_shard_optimizer() pre-initializes state with a zero-gradient step.
    for param in reference.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param)
    optimizer.step()
    optimizer.zero_grad()

    # The FSDP gradient equals the mean of the per-rank gradients.
    for x, target in pairs:
        loss = torch.nn.functional.mse_loss(reference(x), target)
        (loss / world_size).backward()
    optimizer.step()
    optimizer.zero_grad()
    return reference


def _second_layer_params(module):
    return [module.fc2.weight, module.fc2.bias] if hasattr(module, "fc2") else []


def _train_one_step(model, optimizer, x, target):
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _observed_and_expected(model, reference):
    with torch.no_grad():
        return model(_eval_input()), reference(_eval_input())


def _build(module, strategy, **kwargs):
    return fully_shard_model(
        module, fsdp_unit_modules=[torch.nn.Linear], zero_dp_strategy=strategy, **kwargs
    )


@pytest.mark.parametrize("strategy", ["optim_grads_params", "optim_grads", "optim"])
def test_fully_shard_step_matches_reference_for_sharding_strategies(model_parallel, strategy):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), strategy)
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    reference = _reference_step(TinyFSDPModel, _rank_inputs())
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_no_shard_step_matches_reference(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "no_shard")
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    # With no sharding, parameters are replicated: perturb them per rank and
    # check that broadcast_params restores rank-0 values before training.
    saved = {name: value.detach().clone() for name, value in model.module.state_dict().items()}
    with torch.no_grad():
        for param in model.module.parameters():
            param.fill_(float(torch.distributed.get_rank()))
    model.broadcast_params()
    for name, value in model.module.state_dict().items():
        assert torch.count_nonzero(value) == 0
    with torch.no_grad():
        model.module.load_state_dict(saved)

    model.start_param_sync()

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    reference = _reference_step(TinyFSDPModel, _rank_inputs())
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_step_with_fine_grained_hooks_and_nan_reporting(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(
        TinyFSDPModel().cuda(),
        "optim_grads_params",
        enable_fine_grained_param_gather=True,
        report_nan_in_param_grad=True,
    )
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    reference = _reference_step(TinyFSDPModel, _rank_inputs())
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_step_with_sync_param_gather(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "optim", overlap_param_gather=False)
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    model.start_param_sync(force_sync=True)

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    reference = _reference_step(TinyFSDPModel, _rank_inputs())
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_calculate_per_token_loss_sums_gradients(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "optim_grads_params", calculate_per_token_loss=True)
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    # Without per-token loss scaling the gradients are summed, not averaged.
    torch.manual_seed(123)
    reference = TinyFSDPModel().cuda()
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=LR)
    for param in reference.parameters():
        param.grad = torch.zeros_like(param)
    reference_optimizer.step()
    reference_optimizer.zero_grad()
    for x_r, target_r in _rank_inputs():
        torch.nn.functional.mse_loss(reference(x_r), target_r).backward()
    reference_optimizer.step()
    reference_optimizer.zero_grad()
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_frozen_params_are_not_updated(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    module = TinyFSDPModel().cuda()
    frozen_before = module.fc2.weight.detach().clone()
    module.fc2.weight.requires_grad_(False)
    module.fc2.bias.requires_grad_(False)
    model = _build(module, "optim_grads_params")
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    frozen_group_ids = {
        model.param_and_grad_buffer.param_to_param_group[param]
        for param in (module.fc2.weight, module.fc2.bias)
    }
    assert all(
        not model.param_and_grad_buffer.parameter_groups[group_id].requires_grad
        for group_id in frozen_group_ids
    )
    assert all(
        model.param_and_grad_buffer.parameter_groups[group_id].main_grad_buffer is None
        for group_id in frozen_group_ids
    )

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    reference = _reference_step(TinyFSDPModel, _rank_inputs(), frozen_second_layer=True)
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(module.fc2.weight.detach(), frozen_before, rtol=0, atol=0)


def test_fully_shard_tied_weights_stay_tied_and_match_reference(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TiedWeightsModel().cuda(), "optim_grads_params")
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    # After the raw-parameter swap of the forward pass the weights are still tied.
    model(_eval_input())
    assert model.module.head.weight is model.module.fc.weight

    reference = _reference_step(TiedWeightsModel, _rank_inputs())
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_accepts_fsdp_unit_modules_as_import_paths(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = fully_shard_model(
        TinyFSDPModel().cuda(),
        fsdp_unit_modules=["torch.nn.Linear"],
        zero_dp_strategy="optim_grads_params",
    )
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    reference = _reference_step(TinyFSDPModel, _rank_inputs())
    observed, expected = _observed_and_expected(model, reference)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_fully_shard_rejects_expert_params_without_expert_group(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    module = TinyFSDPModel().cuda()
    module.fc1.weight.allreduce = False

    with pytest.raises(ValueError, match="Megatron Expert Parallelism is enabled"):
        fully_shard_model(module, fsdp_unit_modules=[torch.nn.Linear])


def test_megatron_fsdp_context_managers_toggle_sync_flags(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "optim_grads_params")

    assert model.get_distributed_index() is model.dist_index
    assert model.dist_index.use_hybrid_fsdp is False

    with model.no_sync():
        assert model.is_last_microbatch is False
    assert model.is_last_microbatch is True

    with model.sync():
        assert model.model_auto_sync is True
    assert model.model_auto_sync is False


def test_megatron_fsdp_mixed_precision_context_restores_policy(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(
        TinyFSDPModel().cuda(),
        "optim_grads_params",
        mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
    )

    with model.mixed_precision_context(MixedPrecisionPolicy(grad_comm_dtype=torch.bfloat16)):
        assert model.mp_policy.main_params_dtype == torch.float32
        assert model.mp_policy.grad_comm_dtype == torch.bfloat16
        assert model.param_and_grad_buffer.mp_policy.grad_comm_dtype == torch.bfloat16

    assert model.mp_policy.main_params_dtype == torch.float32
    assert model.mp_policy.grad_comm_dtype is None
    assert model.param_and_grad_buffer.mp_policy.grad_comm_dtype is None


def test_megatron_fsdp_scale_gradients_scales_grad_buffers(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "optim_grads_params")
    for group in model.param_and_grad_buffer.parameter_groups:
        group.main_grad_buffer.data.fill_(2.0)

    model.scale_gradients(0.5)

    for group in model.param_and_grad_buffer.parameter_groups:
        torch.testing.assert_close(
            group.main_grad_buffer.data,
            torch.full_like(group.main_grad_buffer.data, 1.0),
            rtol=0,
            atol=0,
        )


def test_megatron_fsdp_state_dict_round_trip_preserves_outputs(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "optim_grads_params")
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    _train_one_step(model, optimizer, x, target)

    state_dict = model.state_dict()
    assert set(state_dict) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert all(isinstance(value, DTensor) for value in state_dict.values())
    optimizer_state = optimizer.state_dict()
    assert set(optimizer_state) == {"state", "param_groups"}
    assert len(optimizer_state["param_groups"]) == 1

    # A differently-initialized model produces different outputs before the load.
    torch.manual_seed(321)
    reloaded = _build(TinyFSDPModel().cuda(), "optim_grads_params")
    with torch.no_grad():
        before = reloaded(_eval_input())
        expected = model(_eval_input())
    assert not torch.allclose(before, expected)

    reloaded.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        after = reloaded(_eval_input())
    torch.testing.assert_close(after, expected, rtol=1e-6, atol=1e-6)


def test_megatron_fsdp_zero_grad_buffer_clears_gradients(model_parallel):
    model_parallel()
    torch.manual_seed(123)
    model = _build(TinyFSDPModel().cuda(), "optim_grads_params")
    optimizer = fully_shard_optimizer(torch.optim.Adam(model.parameters(), lr=LR))

    x, target = _rank_inputs()[torch.distributed.get_rank()]
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    model.finish_grad_sync()

    model.zero_grad_buffer()

    for group in model.param_and_grad_buffer.parameter_groups:
        gbuf = group.main_grad_buffer
        if gbuf is not None and gbuf.data.numel() > 0:
            assert torch.count_nonzero(gbuf.data) == 0
    for param in model.parameters():
        assert param.grad is None
    optimizer.zero_grad()
