# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

import megatron.core.optimizer.clip_grads as clip_grads_module
import megatron.core.optimizer.grad_scaler as grad_scaler_module
import megatron.core.optimizer.layer_wise_optimizer as layer_wise_optimizer_module
import megatron.core.optimizer.optimizer as optimizer_module
# FP8 recipe will be used to test precision-aware-optimizer.
from transformer_engine.pytorch.fp8 import fp8_autocast

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import (
    ChainedOptimizer,
    OptimizerConfig,
    ParamKey,
    ParamPredicate,
    _get_param_groups,
    check_config_overrides_consistency,
    get_megatron_optimizer,
    get_standard_config_overrides,
)
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version, is_torch_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _init_distributed

try:
    # Check if FP8 block scaling is available.
    from transformer_engine.pytorch.fp8 import check_fp8_block_scaling_support

    fp8_block_scaling_available, reason_for_no_fp8_block_scaling = check_fp8_block_scaling_support()
    from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, Format
except:
    fp8_block_scaling_available = False
    reason_for_no_fp8_block_scaling = "FP8 block scaled GEMM requires Hopper and CUDA >= 12.9."
    try:
        from transformer_engine.common.recipe import DelayedScaling
    except:
        delayed_scaling_available = False


class Net(nn.Module):
    def __init__(self, add_layernorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if add_layernorm:
            self.q_layernorm = nn.LayerNorm(10, bias=False)
            self.k_layernorm = nn.LayerNorm(10, bias=False)
            self.layernorm = nn.LayerNorm(10, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_no_overrides(mock_get_world_size):
    net = Net()
    # NOTE: to get no overrides, supply an empty dictionary rather than None.
    param_groups = _get_param_groups([net], OptimizerConfig(optimizer='adam', lr=0.01), {})
    assert len(param_groups) == 1
    pg0 = param_groups[0]
    assert pg0.keys() == {
        'params',
        'is_expert_parallel',
        'is_engram_parallel',
        'is_vision_model_param',
        'default_config',
        'wd_mult',
        'lr_mult',
        'is_decoupled_lr',
        'max_lr',
        'min_lr',
    }
    assert pg0['params'] == list(net.parameters())
    assert pg0['is_expert_parallel'] == False
    assert pg0['is_engram_parallel'] == False
    assert pg0['is_vision_model_param'] == False
    assert pg0['default_config'] == True
    assert pg0['wd_mult'] == 1.0
    assert pg0['lr_mult'] == 1.0
    assert pg0['is_decoupled_lr'] == False
    assert pg0['max_lr'] == 0.01  # from the optimizer config default for lr
    assert pg0['min_lr'] is None  # from the optimizer config default.


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_default_overrides(mock_get_world_size):
    """Test that the default overrides are applied to the parameter groups."""
    net = Net()
    # NOTE: to get legacy default overrides, supply None.
    opt_config = OptimizerConfig(optimizer='adam', lr=0.01)
    check_config_overrides_consistency(opt_config, None)
    param_groups = _get_param_groups([net], opt_config, None)
    assert len(param_groups) == 2
    pg0, pg1 = param_groups
    wd_mults = {pg0['wd_mult'], pg1['wd_mult']}
    assert wd_mults == {1.0, 0.0}


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_with_overrides(mock_get_world_size):
    net = Net()
    config_overrides = {
        ParamKey(
            name="*.bias",
            predicate=ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1),
        ): ParamGroupOverride(wd_mult=0.0)
    }
    opt_config = OptimizerConfig(optimizer='adam', lr=0.01)
    check_config_overrides_consistency(opt_config, config_overrides)
    param_groups = _get_param_groups([net], opt_config, config_overrides)
    assert len(param_groups) == 2
    p_set = set(net.parameters())

    assert p_set == set(param_groups[0]['params']) | set(param_groups[1]['params'])
    assert len(p_set) == len(param_groups[0]['params']) + len(param_groups[1]['params'])
    assert param_groups[0]['wd_mult'] == 0.0 or param_groups[1]['wd_mult'] == 0.0
    assert param_groups[0]['wd_mult'] == 1.0 or param_groups[1]['wd_mult'] == 1.0
    assert len(param_groups[0]['params']) > 0 and len(param_groups[1]['params']) > 0


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_multiple_matches(mock_get_world_size):
    net = Net()

    param_groups = _get_param_groups(
        [net],
        OptimizerConfig(optimizer='adam', lr=0.01),
        {
            ParamKey(name="*.bias"): ParamGroupOverride(min_lr=1e-4, wd_mult=0.0),
            ParamKey(
                predicate=ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1)
            ): ParamGroupOverride(wd_mult=0.0, min_lr=1e-4),
        },
    )
    config_overrides = {
        ParamKey(
            name="*.bias",
            predicate=ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1),
        ): ParamGroupOverride(min_lr=1e-4, wd_mult=0.0)
    }
    opt_config = OptimizerConfig(optimizer='adam', lr=0.01)
    check_config_overrides_consistency(opt_config, config_overrides)
    param_groups2 = _get_param_groups([net], opt_config, config_overrides)
    assert len(param_groups) == 2
    assert param_groups == param_groups2


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_overlapping_matches(mock_get_world_size):
    """In this test, we see if we can have two matches that create three param groups."""
    net = Net()
    # We expect that all convolution parameters will have wd_mult=0.0
    #  However the conv1 related parameters will additionally have a different LR schedule.
    #  this should create three param groups (no match, conv1 (both wd_mult=0.0 and LR schedule), conv2 (only wd_mult=0.0))
    config_overrides = {
        ParamKey(name="*conv*"): ParamGroupOverride(wd_mult=0.0),
        ParamKey(name="*conv1*"): ParamGroupOverride(min_lr=10, max_lr=20),
    }
    opt_config = OptimizerConfig(optimizer='adam', lr=0.01)
    check_config_overrides_consistency(opt_config, config_overrides)
    param_groups = _get_param_groups([net], opt_config, config_overrides)
    assert len(param_groups) == 3
    p_set = set(net.parameters())
    assert p_set == set(param_groups[0]['params']) | set(param_groups[1]['params']) | set(
        param_groups[2]['params']
    )
    assert len(p_set) == len(param_groups[0]['params']) + len(param_groups[1]['params']) + len(
        param_groups[2]['params']
    )
    assert (
        param_groups[0]['wd_mult'] == 1.0
    ), "We expect the first param group to be the None one, which should have wd_mult=1.0"
    assert (
        param_groups[1]['wd_mult'] == 0.0
    ), "We expect the second param group to be the conv1 one, which should have wd_mult=0.0"
    assert (
        param_groups[2]['wd_mult'] == 0.0
    ), "We expect the third param group to be the conv2 one, which should have wd_mult=0.0"
    assert param_groups[1]['min_lr'] == 10
    assert param_groups[1]['max_lr'] == 20
    assert param_groups[2]['min_lr'] is None
    assert param_groups[2]['max_lr'] == 0.01


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_with_standard_config_overrides(apply_wd_to_qk_layernorm: bool):
    """In this test, we see if the standard config overrides are applied correctly."""

    # Initialize the model with layernorm
    net = Net()

    config = OptimizerConfig(optimizer='adam', lr=0.01)
    config_overrides = get_standard_config_overrides(config=config)
    param_groups = _get_param_groups([net], config, config_overrides)

    assert len(param_groups) == 2
    p_set = set(net.parameters())

    assert p_set == set(param_groups[0]['params']) | set(param_groups[1]['params'])
    assert len(p_set) == len(param_groups[0]['params']) + len(param_groups[1]['params'])
    assert param_groups[0]['wd_mult'] == 0.0 or param_groups[1]['wd_mult'] == 0.0
    assert param_groups[0]['wd_mult'] == 1.0 or param_groups[1]['wd_mult'] == 1.0
    assert len(param_groups[0]['params']) > 0 and len(param_groups[1]['params']) > 0

    # Both param groups should have 5 parameters.
    # Param group A (wd_mult=1.0): conv1.weight, conv2.weight, fc1.weight, fc2.weight, fc3.weight
    # Param group B (wd_mult=0.0): conv1.bias, conv2.bias, fc1.bias, fc2.bias, fc3.bias
    assert len(param_groups[0]['params']) == 5, (
        f"Expected 5 parameters in the first param group, "
        f"but got {len(param_groups[0]['params'])}"
    )
    assert len(param_groups[1]['params']) == 5, (
        f"Expected 5 parameters in the second param group, "
        f"but got {len(param_groups[1]['params'])}"
    )


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
def test_get_param_groups_appling_wd_to_qk_layernorm(apply_wd_to_qk_layernorm: bool):
    """In this test, we see if the `apply_wd_to_qk_layernorm` config is applied correctly."""

    # Initialize the model with layernorm
    net = Net(add_layernorm=True)

    config = OptimizerConfig(
        optimizer='adam', lr=0.01, apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm
    )
    config_overrides = get_standard_config_overrides(config=config)
    param_groups = _get_param_groups([net], config, config_overrides)

    assert len(param_groups) == 2
    p_set = set(net.parameters())

    assert p_set == set(param_groups[0]['params']) | set(param_groups[1]['params'])
    assert len(p_set) == len(param_groups[0]['params']) + len(param_groups[1]['params'])
    assert param_groups[0]['wd_mult'] == 1.0
    assert param_groups[1]['wd_mult'] == 0.0

    # There are two param groups, having 7, and 6 parameters respectively.
    # Param group A (wd_mult=1.0): conv1.weight, conv2.weight, fc1.weight, fc2.weight, fc3.weight,
    #    q_layernorm.weight, k_layernorm.weight
    # Param group B (wd_mult=0.0): conv1.bias, conv2.bias, fc1.bias, fc2.bias, fc3.bias,
    #    layernorm.weight
    assert len(param_groups[0]['params']) == 7, (
        f"Expected 5 parameters in the first param group, "
        f"but got {len(param_groups[0]['params'])}"
    )
    assert len(param_groups[1]['params']) == 6, (
        f"Expected 6 parameters in the second param group, "
        f"but got {len(param_groups[1]['params'])}"
    )


def test_chained_optimizer():
    net = Net()
    optimizer_1 = Adam(list(net.parameters())[:2], lr=0.01)
    optimizer_2 = SGD(list(net.parameters())[2:], lr=0.1, momentum=0.9)
    chained_optimizer = ChainedOptimizer([optimizer_1, optimizer_2])

    # Test the chained optimizer's param groups is a reference of the underlying optimizers' param groups
    assert optimizer_1.param_groups[0]["lr"] == 0.01
    chained_optimizer.param_groups[0]["lr"] = 0.02
    assert optimizer_1.param_groups[0]["lr"] == 0.02

    # Test the chained optimizer's state is a reference of the underlying optimizers' state
    # 1. run step on optimizers, make sure there is state
    assert len(chained_optimizer.state) == 0
    input = torch.randn(1, 3, 32, 32)
    output = net(input)
    output.sum().backward()
    optimizer_1.step()
    optimizer_2.step()
    assert len(chained_optimizer.state) != 0

    # 2. check the state is a reference
    assert not list(optimizer_1.state.values())[0]["exp_avg"].is_cuda
    assert not list(optimizer_2.state.values())[0]["momentum_buffer"].is_cuda

    def to_cuda(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to("cuda")
            elif isinstance(v, dict):
                to_cuda(v)
        return d

    for k, v in chained_optimizer.state.items():
        chained_optimizer.state[k] = to_cuda(v)

    assert list(optimizer_1.state.values())[0]["exp_avg"].is_cuda
    assert list(optimizer_2.state.values())[0]["momentum_buffer"].is_cuda


def test_chained_optimizer_get_parameters():
    """Test ChainedOptimizer.get_parameters() aggregates params from all sub-optimizers.

    Regression test: without the get_parameters() override, ChainedOptimizer would
    access self.optimizer which asserts only one optimizer exists, failing with VPP/MoE.
    """

    class MockOptimizer:
        """Mock that mimics MegatronOptimizer's get_parameters() interface."""

        def __init__(self, params):
            self.params = list(params)
            self.param_groups = [{"params": self.params}]

        def get_parameters(self):
            return self.params

    net = Net()
    all_params = list(net.parameters())

    # Test empty
    assert ChainedOptimizer([]).get_parameters() == []

    # Test single optimizer
    opt1 = MockOptimizer(all_params[:3])
    assert ChainedOptimizer([opt1]).get_parameters() == opt1.params

    # Test multiple optimizers (the case that previously failed)
    opt2 = MockOptimizer(all_params[3:6])
    opt3 = MockOptimizer(all_params[6:])
    chained = ChainedOptimizer([opt1, opt2, opt3])
    result = chained.get_parameters()

    assert len(result) == len(all_params)
    assert result == opt1.params + opt2.params + opt3.params


class _ToyMegatronOptimizer(optimizer_module.MegatronOptimizer):
    def __init__(self, optimizer, config):
        super().__init__(optimizer, config)
        self.is_stub_optimizer = False
        self.zero_grad_calls = []
        self.reloaded_state_dicts = []
        self.loaded_state_dicts = []
        self.stepped = False

    def prepare_grads(self):
        return False

    def step_with_ready_grads(self):
        return True

    def zero_grad(self, set_to_none=True):
        self.zero_grad_calls.append(set_to_none)

    def get_loss_scale(self):
        return torch.tensor([2.0])

    def reload_model_params(self, state_dict=None):
        self.reloaded_state_dicts.append(state_dict)

    def state_dict(self):
        return {"state": self.optimizer.state, "param_groups": self.optimizer.param_groups}

    def load_state_dict(self, state_dict):
        self.loaded_state_dicts.append(state_dict)

    def step(self):
        self.stepped = True
        return True

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        return {
            "model": model_sharded_state_dict,
            "is_loading": is_loading,
            "metadata": metadata,
        }


def _make_base_optimizer_with_params(params):
    return SimpleNamespace(param_groups=[{"params": list(params)}], state={})


def test_megatron_optimizer_base_grad_filters_stats_and_properties(monkeypatch):
    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2 = torch.nn.Parameter(torch.tensor([5.0]))
    p2.grad = torch.tensor([6.0])
    p2.shared = True
    p3 = torch.nn.Parameter(torch.tensor([7.0]))
    p3.grad = torch.tensor([8.0])
    p3.tp_duplicate = True
    fsdp_grad = torch.tensor([9.0])
    fsdp_param = SimpleNamespace(grad=SimpleNamespace(_local_tensor=fsdp_grad), __fsdp_param__=True)

    monkeypatch.setattr(optimizer_module, "param_is_not_shared", lambda p: not getattr(p, "shared", False))
    monkeypatch.setattr(
        optimizer_module.tensor_parallel,
        "param_is_not_tensor_parallel_duplicate",
        lambda p, tp_group=None: not getattr(p, "tp_duplicate", False),
    )

    base = _make_base_optimizer_with_params([p1, p2, p3, fsdp_param])
    opt = _ToyMegatronOptimizer(base, OptimizerConfig())

    assert all(
        actual is expected
        for actual, expected in zip(opt.get_parameters(), [p1, p2, p3, fsdp_param])
    )
    grads_for_norm = opt.get_main_grads_for_grad_norm()
    assert grads_for_norm[0] is p1.grad
    assert grads_for_norm[1] is fsdp_grad

    opt.model_parallel_group = "legacy-group"
    with pytest.warns(UserWarning, match="model_parallel_group"):
        assert opt.get_grad_stats_parallel_group() == "legacy-group"
    assert not hasattr(opt, "model_parallel_group")
    assert opt.get_grad_stats_parallel_group() == "legacy-group"
    delattr(opt, "grad_stats_parallel_group")
    monkeypatch.setattr(optimizer_module.parallel_state, "get_model_parallel_group", lambda: "mp")
    assert opt.get_grad_stats_parallel_group() == "mp"

    assert opt.scale_loss(torch.tensor([3.0])).item() == 6.0
    assert opt.param_groups is base.param_groups
    opt.param_groups = [{"params": [p2]}]
    assert len(base.param_groups) == 1
    assert base.param_groups[0]["params"][0] is p2
    opt.is_stub_optimizer = True
    assert opt.param_groups == []


def test_megatron_optimizer_base_decoupled_clip_count_offload_and_steps(monkeypatch):
    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p1.decoupled_grad = torch.tensor([0.0, 1.0])
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p2.decoupled_grad = torch.tensor([2.0])
    base = _make_base_optimizer_with_params([p1, p2])
    base.state[p1] = {"momentum": torch.tensor([4.0])}

    config = OptimizerConfig()
    config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
    opt = _ToyMegatronOptimizer(base, config)

    monkeypatch.setattr(optimizer_module, "param_is_not_shared", lambda p: True)
    monkeypatch.setattr(
        optimizer_module.tensor_parallel,
        "param_is_not_tensor_parallel_duplicate",
        lambda p, tp_group=None: True,
    )
    monkeypatch.setattr(optimizer_module.parallel_state, "get_model_parallel_group", lambda: "mp")

    norm_calls = []
    clip_calls = []
    zero_calls = []
    monkeypatch.setattr(
        optimizer_module,
        "get_grad_norm_fp32",
        lambda grads, grad_stats_parallel_group=None: norm_calls.append(
            (list(grads), grad_stats_parallel_group)
        )
        or 7.0,
    )
    monkeypatch.setattr(
        optimizer_module,
        "clip_grad_by_total_norm_fp32",
        lambda params, clip_grad, grad_norm, use_decoupled_grad: clip_calls.append(
            (list(params), clip_grad, grad_norm, use_decoupled_grad)
        ),
    )
    monkeypatch.setattr(
        optimizer_module,
        "count_zeros_fp32",
        lambda params, grad_stats_parallel_group=None, use_decoupled_grad=False, tp_group=None: (
            zero_calls.append((list(params), grad_stats_parallel_group, use_decoupled_grad, tp_group))
            or 3
        ),
    )

    decoupled_grads = opt.get_main_grads_for_grad_norm()
    assert decoupled_grads[0] is p1.decoupled_grad
    assert decoupled_grads[1] is p2.decoupled_grad
    assert opt.get_grad_norm() == 7.0
    assert opt.clip_grad_norm(0.5) == 7.0
    assert clip_calls[0][0][0] is p1
    assert clip_calls[0][0][1] is p2
    assert clip_calls[0][1:] == (0.5, 7.0, True)
    assert opt.count_zeros() == 3
    assert zero_calls[0][0][0] is p1
    assert zero_calls[0][0][1] is p2
    assert zero_calls[0][1:] == ("mp", True, None)
    assert norm_calls[-1][0][0] is p1.decoupled_grad
    assert norm_calls[-1][0][1] is p2.decoupled_grad
    assert norm_calls[-1][1] == "mp"

    empty_cache_calls = []
    monkeypatch.setattr(
        optimizer_module,
        "cur_platform",
        SimpleNamespace(
            empty_cache=lambda: empty_cache_calls.append("empty"),
            device=lambda: torch.device("cpu"),
        ),
    )
    opt.offload_to_cpu()
    assert empty_cache_calls == ["empty"]
    opt.restore_from_cpu()
    assert base.state[p1]["momentum"].device.type == "cpu"

    assert optimizer_module.MegatronOptimizer._extract_common_per_param_step(
        {"state": {0: {"step": 11}, 1: {"step": 11}, 2: {}}}
    ) == 11
    with pytest.raises(ValueError, match="differs per parameter"):
        optimizer_module.MegatronOptimizer._extract_common_per_param_step(
            {"state": {0: {"step": 1}, 1: {"step": 2}}}
        )
    state = {"state": {0: {}, 1: {"step": 0}}}
    optimizer_module.MegatronOptimizer._restore_common_per_param_step(state, 5)
    assert state == {"state": {0: {"step": 5}, 1: {"step": 5}}}


def test_clip_grads_decoupled_cpu_paths_and_dynamic_grad_scaler(monkeypatch):
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    def fake_multi_tensor_applier(_impl, _overflow_buf, tensor_lists, scale):
        scale_value = scale.item() if isinstance(scale, torch.Tensor) else scale
        for grad in tensor_lists[0]:
            grad.mul_(scale_value)
        return torch.tensor([0.0]), None

    monkeypatch.setattr(clip_grads_module.torch, "zeros", cpu_zeros)
    monkeypatch.setattr(clip_grads_module.torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(clip_grads_module, "param_is_not_shared", lambda p: True)
    monkeypatch.setattr(
        clip_grads_module,
        "param_is_not_tensor_parallel_duplicate",
        lambda p, tp_group=None: True,
    )
    monkeypatch.setattr(clip_grads_module, "multi_tensor_applier", fake_multi_tensor_applier)
    monkeypatch.setattr(clip_grads_module, "multi_tensor_scale_tensor_impl", object())

    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p1.decoupled_grad = torch.tensor([0.0, 4.0])
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p2.decoupled_grad = torch.tensor([0.0])

    assert clip_grads_module.count_zeros_fp32(
        [p1, p2], grad_stats_parallel_group="mp", use_decoupled_grad=True
    ) == 2

    clip_grads_module.clip_grad_by_total_norm_fp32(
        [p1, p2], max_norm=2.0, total_norm=8.0, use_decoupled_grad=True
    )
    assert torch.allclose(p1.decoupled_grad, torch.tensor([0.0, 1.0]), atol=1e-5)

    clip_grads_module.clip_grad_by_total_norm_fp32(
        [p1], max_norm=torch.tensor(1.0), total_norm=torch.tensor(4.0), use_decoupled_grad=True
    )
    assert p1.decoupled_grad[1].item() < 1.0

    monkeypatch.setattr(
        grad_scaler_module,
        "cur_platform",
        SimpleNamespace(device_name=lambda: "cpu", current_device=lambda: torch.device("cpu")),
    )
    constant = grad_scaler_module.ConstantGradScaler(2.0)
    assert constant.inv_scale.item() == 0.5
    constant.update(True)
    assert constant.state_dict() == {}

    scaler = grad_scaler_module.DynamicGradScaler(
        initial_scale=8.0,
        min_scale=2.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
        hysteresis=2,
    )
    scaler.update(False)
    assert scaler.scale.item() == 8.0
    scaler.update(False)
    assert scaler.scale.item() == 16.0
    scaler.update(True)
    assert scaler.scale.item() == 16.0
    scaler.update(True)
    assert scaler.scale.item() == 8.0
    state_dict = scaler.state_dict()
    restored = grad_scaler_module.DynamicGradScaler(4.0, 1.0, 2.0, 0.5, 3, 1)
    restored.load_state_dict(state_dict)
    assert restored.scale.item() == scaler.scale.item()
    assert restored._growth_tracker == scaler._growth_tracker
    assert restored._hysteresis_tracker == scaler._hysteresis_tracker


def test_chained_optimizer_state_split_load_step_and_parameter_state(monkeypatch, tmp_path):
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p1.grad = torch.tensor([1.0])
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    p2.grad = torch.tensor([2.0])
    base1 = _make_base_optimizer_with_params([p1])
    base1.param_groups[0]["step"] = 3
    base1.state["state-a"] = 1
    base2 = _make_base_optimizer_with_params([p2])
    base2.param_groups[0]["step"] = 3
    base2.state["state-b"] = 2

    config = OptimizerConfig(clip_grad=0.7, log_num_zeros_in_grad=True)
    config.overlap_param_gather_with_optimizer_step = True
    opt1 = _ToyMegatronOptimizer(base1, config)
    opt2 = _ToyMegatronOptimizer(base2, config)
    sync_calls = []
    opt1.model_chunks = [SimpleNamespace(start_param_sync=lambda **kwargs: sync_calls.append(kwargs))]
    opt2.model_chunks = [object(), object()]
    opt1.grad_stats_parallel_group = "shared"
    opt2.grad_stats_parallel_group = "shared"

    chained = ChainedOptimizer([opt1, opt2])
    assert chained.model_chunks == opt1.model_chunks + opt2.model_chunks
    assert ChainedOptimizer([opt1]).optimizer is opt1.optimizer
    with pytest.raises(AssertionError, match="more than one optimizer"):
        _ = chained.optimizer
    assert chained.get_loss_scale().item() == 2.0
    chained.zero_grad(set_to_none=False)
    assert opt1.zero_grad_calls == [False]
    assert opt2.zero_grad_calls == [False]

    proxy_state = chained.state
    assert len(proxy_state) == 2
    assert proxy_state[(0, "state-a")] == 1
    proxy_state[(1, "extra")] = 9
    assert base2.state["extra"] == 9
    assert sorted(list(proxy_state)) == [(0, "state-a"), (1, "extra"), (1, "state-b")]
    assert ((1, "extra"), 9) in list(proxy_state.items())

    split = chained._split_state_dict({"model0": "a", "model1": "b", "model2": "c"})
    assert split == [{"model0": "a"}, {"model0": "b", "model1": "c"}]
    chained.reload_model_params({"model0": "a", "model1": "b", "model2": "c"})
    assert opt1.reloaded_state_dicts[-1] == {"model0": "a"}
    assert opt2.reloaded_state_dicts[-1] == {"model0": "b", "model1": "c"}
    with pytest.raises(AssertionError, match="Wrong state_dict format"):
        chained._split_state_dict({"model0": "only-one"})

    assert isinstance(chained.state_dict(), list)
    chained.load_state_dict({1: {"opt": "two"}, 0: {"opt": "one"}})
    assert opt1.loaded_state_dicts[-1] == {"opt": "one"}
    assert opt2.loaded_state_dicts[-1] == {"opt": "two"}
    with pytest.raises(RuntimeError, match="Expected 2 entries"):
        chained.load_state_dict([{"only": "one"}])

    norm_calls = []
    clip_calls = []
    monkeypatch.setattr(
        optimizer_module,
        "get_grad_norm_fp32",
        lambda grads, grad_stats_parallel_group=None: norm_calls.append(
            (len(grads), grad_stats_parallel_group)
        )
        or 13.0,
    )
    monkeypatch.setattr(
        optimizer_module,
        "clip_grad_by_total_norm_fp32",
        lambda parameters, max_norm, total_norm, use_decoupled_grad: clip_calls.append(
            (list(parameters), max_norm, total_norm, use_decoupled_grad)
        ),
    )
    monkeypatch.setattr(
        optimizer_module,
        "count_zeros_fp32",
        lambda parameters, grad_stats_parallel_group=None, use_decoupled_grad=False: 6,
    )
    success, grad_norm, zeros = chained.step()
    assert success is True
    assert grad_norm == 13.0
    assert zeros == 6
    assert norm_calls == [(2, "shared")]
    assert sync_calls == [{"force_dispatch": True}]
    assert clip_calls[0][0][0] is p1
    assert clip_calls[1][0][0] is p2

    opt2.grad_stats_parallel_group = "different"
    opt1.get_grad_norm = lambda: 3.0
    opt2.get_grad_norm = lambda: 4.0
    opt1.count_zeros = lambda: 2
    opt2.count_zeros = lambda: 5
    assert chained.grads_states_parallel_group_is_shared() is False
    with pytest.raises(AssertionError, match="not shared"):
        chained.get_grad_stats_parallel_group()
    assert chained.get_grad_norm() == 5.0
    assert chained.count_zeros() == 7

    opt1.data_parallel_group = SimpleNamespace(rank=lambda: 0)
    opt2.data_parallel_group = SimpleNamespace(rank=lambda: 1)
    opt1.get_parameter_state_dp_zero = lambda: {"rank0": True}
    opt2.get_parameter_state_dp_zero = lambda: None
    parameter_state_path = tmp_path / "parameter_state.pt"
    chained.save_parameter_state(str(parameter_state_path))
    assert torch.load(parameter_state_path) == [{"rank0": True}, None]

    loaded_parameter_states = []
    opt1.load_parameter_state_from_dp_zero = lambda state, update_legacy_format=False: (
        loaded_parameter_states.append((state, update_legacy_format))
    )
    opt2.load_parameter_state_from_dp_zero = lambda state, update_legacy_format=False: (
        loaded_parameter_states.append((state, update_legacy_format))
    )
    chained.load_parameter_state(str(parameter_state_path), update_legacy_format=True)
    assert loaded_parameter_states == [({"rank0": True}, True), (None, True)]


def test_chained_optimizer_sharded_state_dict_convert_and_prefix_paths(monkeypatch):
    config = OptimizerConfig()
    opt1 = _ToyMegatronOptimizer(_make_base_optimizer_with_params([]), config)
    opt2 = _ToyMegatronOptimizer(_make_base_optimizer_with_params([]), config)
    opt1.optimizer.param_groups = [{"params": [1], "step": 4}]
    opt2.optimizer.param_groups = [{"params": [2], "step": 4}]

    opt1.sharded_state_dict = lambda *args, **kwargs: {
        "optimizer": {"main": "one"},
        "param_state": {0: {"a": 1}},
        "param_state_sharding_type": {0: "type-a"},
    }
    opt2.sharded_state_dict = lambda *args, **kwargs: {
        "optimizer": {"main": "two"},
        "param_state": {3: {"b": 2}},
        "param_state_sharding_type": {3: "type-b"},
    }
    chained = ChainedOptimizer([opt1, opt2])

    converted = chained.sharded_state_dict({}, is_loading=True, convert_to_ep=True)
    assert converted["param_state"] == {0: {"a": 1}, 1: {"b": 2}}
    assert chained.mapping_idx == {0: {0: 0}, 1: {3: 1}}
    assert chained.original_sharded_state_dict[0]["len_param_state"] == 1

    prefixes = []
    monkeypatch.setattr(
        optimizer_module,
        "add_prefix_for_sharding",
        lambda state_dict, prefix: prefixes.append(prefix) or state_dict.setdefault(
            "prefix", prefix
        ),
    )
    sharded = chained.sharded_state_dict(
        {},
        is_loading=False,
        metadata={"distrib_optim_sharding_type": "legacy"},
    )
    assert prefixes == ["chained_0.", "chained_1."]
    assert sharded[0]["prefix"] == "chained_0."
    assert sharded[1]["prefix"] == "chained_1."

    no_prefix = chained.sharded_state_dict(
        {},
        is_loading=False,
        metadata={"chained_optim_avoid_prefix": True},
    )
    assert "prefix" not in no_prefix[0]


def test_optimizer_package_mup_overrides_and_param_group_buffer_helpers(monkeypatch):
    import megatron.core.optimizer as optimizer_pkg

    warnings_seen = []
    monkeypatch.setattr(
        optimizer_pkg,
        "log_single_rank",
        lambda _logger, _level, message: warnings_seen.append(message),
    )

    adam_config = OptimizerConfig(
        optimizer="adam",
        lr=0.1,
        min_lr=0.01,
        decoupled_lr=0.2,
        decoupled_min_lr=0.02,
        adam_eps=1.0e-6,
    )
    adam_overrides = optimizer_pkg.get_mup_config_overrides(adam_config, 2.0, "adam")
    assert any("decoupled_lr" in message for message in warnings_seen)
    assert {"max_lr": 0.05, "min_lr": 0.005} in adam_overrides.values()
    assert {"eps": 5.0e-7} in adam_overrides.values()

    hidden = torch.nn.Parameter(torch.ones(2, 2))
    vector = torch.nn.Parameter(torch.ones(2))
    output = torch.nn.Parameter(torch.ones(2, 2))
    output.is_embedding_or_output_parameter = True
    hidden_matches = [
        override
        for key, override in adam_overrides.items()
        if key.matches(hidden, "decoder.layers.0.mlp.weight")
    ]
    output_matches = [
        override
        for key, override in adam_overrides.items()
        if key.matches(output, "output_layer.weight")
    ]
    assert {"max_lr": 0.05, "min_lr": 0.005} in hidden_matches
    assert {"eps": 5.0e-7} in hidden_matches
    assert {"max_lr": 0.05, "min_lr": 0.005} not in output_matches

    sgd_overrides = optimizer_pkg.get_mup_config_overrides(
        OptimizerConfig(optimizer="sgd", lr=0.1, min_lr=0.01), 3.0, "sgd"
    )
    assert list(sgd_overrides.values()) == [{"max_lr": 0.30000000000000004, "min_lr": 0.03}]
    assert next(iter(sgd_overrides)).matches(vector, "norm.weight")
    assert optimizer_pkg.get_mup_config_overrides(OptimizerConfig(lr=0.1), 1.0, "adam") == {}

    monkeypatch.setattr(optimizer_pkg.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        optimizer_pkg.torch.distributed,
        "all_gather_object",
        lambda gathered, value: gathered.__setitem__(0, value),
    )

    class Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Parameter(torch.ones(2, 2))
            self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.expert = torch.nn.Parameter(torch.ones(2))
            self.expert.allreduce = False
            self.engram = torch.nn.Parameter(torch.ones(3))
            self.engram.is_engram_embedding = True
            self.buffers = ["dense-buffer"]
            self.expert_parallel_buffers = ["expert-buffer"]
            self.engram_embedding_buffers = ["engram-buffer"]

    chunk = Chunk()
    dense_groups, dense_buffers = optimizer_pkg._get_param_groups_and_buffers(
        [chunk],
        model_chunk_offset=4,
        config=OptimizerConfig(lr=0.01),
        config_overrides={},
        filter_fn=lambda group: not group["is_expert_parallel"] and not group["is_engram_parallel"],
        buffer_name="buffers",
    )
    assert dense_buffers == {4: ["dense-buffer"]}
    assert any(param is chunk.dense for param in dense_groups[0]["params"])
    assert not any(param is chunk.expert for param in dense_groups[0]["params"])
    assert not any(param is chunk.frozen for param in dense_groups[0]["params"])

    expert_groups, expert_buffers = optimizer_pkg._get_param_groups_and_buffers(
        [chunk],
        model_chunk_offset=0,
        config=OptimizerConfig(lr=0.01),
        config_overrides={},
        filter_fn=lambda group: group["is_expert_parallel"],
        buffer_name="expert_parallel_buffers",
    )
    assert expert_buffers == {0: ["expert-buffer"]}
    assert expert_groups[0]["is_expert_parallel"] is True
    assert any(param is chunk.expert for param in expert_groups[0]["params"])

    engram_groups, engram_buffers = optimizer_pkg._get_param_groups_and_buffers(
        [chunk],
        model_chunk_offset=0,
        config=OptimizerConfig(lr=0.01),
        config_overrides={},
        filter_fn=lambda group: group["is_engram_parallel"],
        buffer_name="engram_embedding_buffers",
    )
    assert engram_buffers == {0: ["engram-buffer"]}
    assert engram_groups[0]["is_engram_parallel"] is True
    assert any(param is chunk.engram for param in engram_groups[0]["params"])


def test_optimizer_package_optimizer_factory_selection_paths(monkeypatch):
    import megatron.core.optimizer as optimizer_pkg

    class FakeTorchOptimizer:
        def __init__(self, params, **kwargs):
            self.param_groups = params
            self.kwargs = kwargs
            self.state = {}
            for group in params:
                for param in group["params"]:
                    self.state[param] = {}

    wrappers = []

    def fake_fp32_optimizer(base_optimizer, config, init_state_fn):
        wrapper = SimpleNamespace(
            kind="fp32",
            optimizer=base_optimizer,
            config=config,
            init_state_fn=init_state_fn,
        )
        wrappers.append(wrapper)
        return wrapper

    def fake_float16_optimizer(base_optimizer, config, grad_scaler, init_state_fn):
        wrapper = SimpleNamespace(
            kind="float16",
            optimizer=base_optimizer,
            config=config,
            grad_scaler=grad_scaler,
            init_state_fn=init_state_fn,
        )
        wrappers.append(wrapper)
        return wrapper

    def fake_distributed_optimizer(*args, **kwargs):
        wrapper = SimpleNamespace(kind="distributed", args=args, kwargs=kwargs)
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(optimizer_pkg, "SGD", FakeTorchOptimizer)
    monkeypatch.setattr(optimizer_pkg, "Adam", FakeTorchOptimizer)
    monkeypatch.setattr(optimizer_pkg, "FP32Optimizer", fake_fp32_optimizer)
    monkeypatch.setattr(optimizer_pkg, "Float16OptimizerWithFloat16Params", fake_float16_optimizer)
    monkeypatch.setattr(optimizer_pkg, "DistributedOptimizer", fake_distributed_optimizer)
    monkeypatch.setattr(optimizer_pkg.parallel_state, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(optimizer_pkg, "HAVE_EO_V02", False)

    param = torch.nn.Parameter(torch.ones(1))
    param_groups = [{"params": [param]}]
    fp32 = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="sgd", lr=0.1),
        model_chunks=["model"],
        param_groups=param_groups,
        model_parallel_group="mp",
    )
    assert fp32.kind == "fp32"
    assert fp32.optimizer.kwargs["momentum"] == OptimizerConfig().sgd_momentum
    assert fp32.grad_stats_parallel_group == "mp"
    assert fp32.tp_group == "tp"

    empty = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="sgd", lr=0.1),
        model_chunks=[],
        param_groups=[],
        pg_collection=SimpleNamespace(tp="custom-tp"),
    )
    assert empty.kind == "fp32"
    assert empty.optimizer is None
    assert empty.tp_group == "custom-tp"

    float16 = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="sgd", lr=0.1, bf16=True),
        model_chunks=["model"],
        param_groups=param_groups,
        model_parallel_group="mp16",
    )
    assert float16.kind == "float16"
    assert float16.grad_scaler is None
    assert float16.grad_stats_parallel_group == "mp16"

    distributed = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="sgd", lr=0.1, use_distributed_optimizer=True),
        model_chunks=["model"],
        param_groups=param_groups,
        per_model_buffers={0: ["buffer"]},
        data_parallel_group="dp",
        data_parallel_group_gloo="gloo",
        data_parallel_group_idx=7,
        intra_dist_opt_group="intra",
        distributed_optimizer_instance_id=3,
    )
    assert distributed.kind == "distributed"
    assert distributed.kwargs["per_model_buffers"] == {0: ["buffer"]}
    assert distributed.kwargs["data_parallel_group_idx"] == 7
    assert distributed.grad_stats_parallel_group == "intra"

    with pytest.raises(ImportError, match="Lion optimizer requires"):
        optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
            OptimizerConfig(optimizer="lion", lr=0.1),
            model_chunks=["model"],
            param_groups=param_groups,
        )
    with pytest.raises(Exception, match="not supported"):
        optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
            OptimizerConfig(optimizer="unknown", lr=0.1),
            model_chunks=["model"],
            param_groups=param_groups,
        )


def test_layer_wise_optimizer_shards_buckets_gathers_broadcasts_and_steps(monkeypatch):
    monkeypatch.setattr(layer_wise_optimizer_module, "get_pg_size", lambda group: {"dp": 2, "expt": 2}[group])
    monkeypatch.setattr(layer_wise_optimizer_module, "get_pg_rank", lambda group: 0)

    config = OptimizerConfig(clip_grad=0.5, log_num_zeros_in_grad=True)
    dense_a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    dense_b = torch.nn.Parameter(torch.tensor([3.0]))
    expert_a = torch.nn.Parameter(torch.tensor([4.0, 5.0, 6.0]))
    expert_b = torch.nn.Parameter(torch.tensor([7.0]))

    base1 = SimpleNamespace(
        param_groups=[
            {"params": [dense_a, dense_b], "is_expert_parallel": False},
            {"params": [expert_a, expert_b], "is_expert_parallel": True},
        ],
        state={},
    )
    opt1 = _ToyMegatronOptimizer(base1, config)
    pg_collection = SimpleNamespace(dp_cp="dp", expt_dp="expt")

    layerwise = layer_wise_optimizer_module.LayerWiseDistributedOptimizer(
        [opt1], config, pg_collection=pg_collection
    )

    assert layerwise.dp_cp_params_list is not None
    assert layerwise.expt_dp_params_list is not None
    assert any(param is dense_a for param in layerwise.dp_cp_params_list[0])
    assert any(param is dense_b for param in layerwise.dp_cp_params_list[1])
    assert any(param is expert_a for param in layerwise.expt_dp_params_list[0])
    assert any(param is expert_b for param in layerwise.expt_dp_params_list[1])
    assert len(opt1.optimizer.param_groups[0]["params"]) == 1
    assert opt1.optimizer.param_groups[0]["params"][0] is dense_a
    assert len(opt1.optimizer.param_groups[1]["params"]) == 1
    assert opt1.optimizer.param_groups[1]["params"][0] is expert_a

    bucket_calls = []

    class _Bucket:
        def __init__(self, params):
            self.params = params

        def set_layerwise_params_list(self, params_list):
            bucket_calls.append(params_list)

    model_chunk = SimpleNamespace(
        bucket_groups=[SimpleNamespace(buckets=[_Bucket([dense_a])])],
        expert_parallel_bucket_groups=[SimpleNamespace(buckets=[_Bucket([expert_a])])],
    )
    layerwise.set_bucket_layerwise_params_list([model_chunk])
    assert len(bucket_calls) == 2
    assert any(param is dense_a for param in bucket_calls[0][0])
    assert any(param is expert_a for param in bucket_calls[1][0])

    monkeypatch.setattr(
        layer_wise_optimizer_module.torch.distributed,
        "all_gather",
        lambda gather_list, src, group=None: [
            tensor.copy_(torch.full_like(tensor, 10.0 + idx))
            for idx, tensor in enumerate(gather_list)
            if tensor is not src and tensor.numel() > 0
        ],
    )
    layerwise.allgather_params()
    assert torch.equal(dense_b.data, torch.full_like(dense_b.data, 11.0))
    assert torch.equal(expert_b.data, torch.full_like(expert_b.data, 11.0))

    broadcast_calls = []
    monkeypatch.setattr(
        layer_wise_optimizer_module.torch.distributed,
        "get_global_rank",
        lambda group, rank: {"dp": 100, "expt": 200}[group] + rank,
    )
    monkeypatch.setattr(
        layer_wise_optimizer_module.torch.distributed,
        "broadcast",
        lambda param, src, group: broadcast_calls.append((param, src, group)),
    )
    layerwise.broadcast_params()
    assert any(src == 100 and group == "dp" for _, src, group in broadcast_calls)
    assert any(src == 200 and group == "expt" for _, src, group in broadcast_calls)

    norm_calls = []
    clip_calls = []
    monkeypatch.setattr(
        layer_wise_optimizer_module,
        "get_grad_norm_fp32",
        lambda grads, grad_stats_parallel_group=None: norm_calls.append(len(grads)) or 5.0,
    )
    monkeypatch.setattr(
        layer_wise_optimizer_module,
        "count_zeros_fp32",
        lambda params, grad_stats_parallel_group=None, use_decoupled_grad=False: 2,
    )
    monkeypatch.setattr(
        optimizer_module,
        "clip_grad_by_total_norm_fp32",
        lambda params, max_norm, total_norm, use_decoupled_grad: clip_calls.append(
            (list(params), max_norm, total_norm, use_decoupled_grad)
        ),
    )
    for param in [dense_a, expert_a]:
        param.grad = torch.ones_like(param.data)
    success, grad_norm, zeros = layerwise.step()
    assert success is True
    assert grad_norm == 5.0
    assert zeros == 2
    assert norm_calls[-1] == 2
    assert clip_calls

    single_pg = SimpleNamespace(dp_cp="single", expt_dp="single")
    monkeypatch.setattr(layer_wise_optimizer_module, "get_pg_size", lambda group: 1)
    no_shard = layer_wise_optimizer_module.LayerWiseDistributedOptimizer(
        [_ToyMegatronOptimizer(SimpleNamespace(param_groups=[{"params": [torch.nn.Parameter(torch.ones(1))]}], state={}), config)],
        config,
        pg_collection=single_pg,
    )
    assert no_shard.dp_cp_params_list is None
    no_shard.allgather_params()
    no_shard.broadcast_params()


def test_layer_wise_optimizer_state_dict_files_and_sharded_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(layer_wise_optimizer_module, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(layer_wise_optimizer_module, "get_pg_rank", lambda group: 0)
    config = OptimizerConfig()
    base = SimpleNamespace(param_groups=[{"params": [torch.nn.Parameter(torch.ones(1))]}], state={})
    toy = _ToyMegatronOptimizer(base, config)
    layerwise = layer_wise_optimizer_module.LayerWiseDistributedOptimizer(
        [toy], config, pg_collection=SimpleNamespace(dp_cp="dp", expt_dp="expt")
    )

    loaded = []
    monkeypatch.setattr(
        layer_wise_optimizer_module.ChainedOptimizer,
        "load_state_dict",
        lambda self, state_dict: loaded.append(state_dict),
    )
    state = {"fp32_from_fp16_params": {1: ["b"], 0: ["a"]}}
    layerwise.load_state_dict(state)
    assert state["fp32_from_fp16_params"] == [["a"], ["b"]]
    assert loaded[-1] is state

    class _Shard:
        def __init__(self, replica_id):
            self.replica_id = replica_id

    local_params = SimpleNamespace(unwrap=lambda: [torch.tensor([1.0])])
    empty_params = SimpleNamespace(unwrap=lambda: [])
    sharded_state = {
        "tensor": _Shard((3, 4, 5)),
        "fp32_from_fp16_params": [[], [torch.tensor([2.0])]],
        "optimizer": {
            "state": {},
            "param_groups": [{"params": local_params}, {"params": empty_params}],
        },
    }
    monkeypatch.setattr(
        layer_wise_optimizer_module.ChainedOptimizer,
        "sharded_state_dict",
        lambda self, model_sharded_state_dict, is_loading=False, **kwargs: sharded_state,
    )
    monkeypatch.setattr(
        layer_wise_optimizer_module.torch.distributed,
        "get_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        layer_wise_optimizer_module.torch.distributed,
        "all_gather_object",
        lambda out, obj: (out.__setitem__(0, dict(obj)), out.__setitem__(1, {**dict(obj), "params": True})),
    )

    result = layerwise.sharded_state_dict({}, is_loading=False)
    assert result["tensor"].replica_id == (3, 4, 0)
    assert isinstance(
        result["fp32_from_fp16_params"][0],
        layer_wise_optimizer_module.LocalNonpersistentObject,
    )
    assert len(result["fp32_from_fp16_params"][1]) == 1
    assert torch.equal(result["fp32_from_fp16_params"][1][0], torch.tensor([2.0]))
    assert isinstance(result["fp32_from_fp16_params"], dict)
    assert isinstance(
        result["optimizer"]["state"],
        layer_wise_optimizer_module.LocalNonpersistentObject,
    )
    assert result["optimizer"]["param_groups"][0]["params"] is local_params
    assert result["optimizer"]["param_groups"][1]["params"] is empty_params

    save_path = tmp_path / "layerwise.pt"
    monkeypatch.setattr(
        layer_wise_optimizer_module.ChainedOptimizer,
        "state_dict",
        lambda self: {"saved": True},
    )
    layerwise.save_state_dict_to_file(str(save_path))
    assert torch.load(save_path) == {"saved": True}
    layerwise.load_state_dict_from_file(str(save_path))
    assert loaded[-1] == {"saved": True}


def test_precision_aware_fused_adam():
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        # Older versions of TE don't have FusedAdam.
        return

    import inspect

    adam_args = inspect.signature(FusedAdam).parameters
    arg_names = ["master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype", "use_decoupled_grad"]
    for name in arg_names:
        if name not in adam_args:
            # Skip the test if TE doesn't support precision aware FusedAdam.
            return

    tensor = torch.rand(278011, dtype=torch.bfloat16).cuda()
    params_1 = [torch.nn.Parameter(tensor.float())]  # FP32 reference
    params_2 = [torch.nn.Parameter(tensor.clone())]  # BF16

    options = {"lr": 1, "betas": (0.1, 0.25), "eps": 1e-08, "weight_decay": 0, "amsgrad": False}

    optimizer_1 = FusedAdam(params_1, **options)
    optimizer_2 = FusedAdam(params_2, master_weights=True, use_decoupled_grad=True, **options)

    for _ in range(1000):
        for p_1, p_2 in zip(params_1, params_2):
            p_1.grad = torch.rand_like(p_1)
            p_2.decoupled_grad = p_1.grad.clone()

        optimizer_1.step()
        optimizer_2.step()

        master_params = [optimizer_2.get_unscaled_state(p, "master_param") for p in params_2]
        for p_1, p_2 in zip(params_1, master_params):
            bytes_1 = p_1.data.view(torch.uint8)
            bytes_2 = p_2.data.view(torch.uint8)
            # Make sure bit-wise matched
            assert torch.all(bytes_1 == bytes_2)

        for p_1, p_2 in zip(params_1, params_2):
            bytes_1 = p_1.data.bfloat16().view(torch.uint8)
            bytes_2 = p_2.data.view(torch.uint8)
            # Make sure bit-wise matched
            assert torch.all(bytes_1 == bytes_2)


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TE 1.13.0 is required for precision aware optimizer"
)
@pytest.mark.parametrize("precision", ['bf16', 'fp8'])
@pytest.mark.parametrize("main_params_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("main_grads_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    # use the same dtype for exp_avg and exp_avg_sq to reduce the number of tests
    "moment_dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.uint8],
)
@pytest.mark.skip(reason="inconsistent ci test runs resulting in NCCL errors")
def test_precision_aware_optimizer(
    precision: str,
    main_params_dtype: torch.dtype,
    main_grads_dtype: torch.dtype,
    moment_dtype: torch.dtype,
):
    # Skip because bf16 optimizer states are not supported before TE 2.3.0
    if (moment_dtype == torch.bfloat16) and not is_te_min_version("2.3.0"):
        pytest.skip("bfloat16 for moment_dtype requires TE >= 2.3.0")

    if precision == 'fp8':
        if not fp8_block_scaling_available:
            fp8_recipe = "delayed"
            fp8_recipe_settings = DelayedScaling()
        else:
            fp8_recipe = "blockwise"
            fp8_recipe_settings = Float8BlockScaling(fp8_format=Format.E4M3)
    else:
        fp8_recipe = None
        fp8_recipe_settings = None

    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model, mock_args.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    # First create baseline model with float32 optimizer states
    baseline_model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    baseline_model.requires_grad_(True)
    baseline_model.weight.data.fill_(1.0)
    baseline_ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    baseline_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), baseline_ddp_config, baseline_model
    )
    baseline_optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    baseline_optim = get_megatron_optimizer(baseline_optimizer_config, [baseline_model])

    # Create test model with specified dtypes for optimizer states
    test_model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    test_model.requires_grad_(True)
    test_model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    test_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, test_model
    )
    test_optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        bf16=True,
        fp8_recipe=fp8_recipe,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_params_dtype=main_params_dtype,
        main_grads_dtype=main_grads_dtype,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
    )
    test_optim = get_megatron_optimizer(test_optimizer_config, [test_model])

    # Use same input for both models
    input = torch.randn(8, 100, dtype=torch.bfloat16, device='cuda')

    # Run model
    def run_model(model, input, optim, fp8_recipe, fp8_recipe_settings):
        if not fp8_recipe:
            output = model(input)
        else:
            with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe_settings):
                output = model(input)
        loss = output.sum()
        loss.backward()
        optim.step()
        return loss.item(), optim.get_grad_norm()

    # Run baseline model and test model
    baseline_loss, baseline_grad_norm = run_model(
        baseline_model, input, baseline_optim, fp8_recipe, fp8_recipe_settings
    )
    test_loss, test_grad_norm = run_model(
        test_model, input, test_optim, fp8_recipe, fp8_recipe_settings
    )

    rtol, atol = 1.6e-2, 1e-5

    # Compare grad norms - allow small difference due to precision
    torch.testing.assert_close(test_grad_norm, baseline_grad_norm, atol=atol, rtol=rtol)

    # Compare losses - allow small difference due to precision
    torch.testing.assert_close(test_loss, baseline_loss, atol=atol, rtol=rtol)

    # Save and reload state dict for the test model
    state_dict = test_optim.state_dict()
    test_optim.load_state_dict(state_dict)


@pytest.mark.parametrize("use_precision_aware", [True, False])
def test_distrib_optimizer_save_load_with_non_tensor_state(use_precision_aware):
    """Test that save/load of distributed optimizer handles non-tensor state entries.

    Optimizers may store non-tensor entries (e.g. `found_inf: bool`) in the per-parameter
    state dict. The distrib_optimizer's _get_main_param_and_optimizer_states and
    _set_main_param_and_optimizer_states must skip these to avoid crashes when calling
    tensor operations (.copy_(), get_unscaled_state, set_scaled_state) on non-tensors.

    Tests both the precision-aware path (TE FusedAdam with scaled states) and the
    non-precision-aware path (standard optimizer with .copy_()).
    """
    if use_precision_aware:
        try:
            from transformer_engine.pytorch.optimizers import FusedAdam
        except ImportError:
            pytest.skip("TE FusedAdam not available")

        import inspect

        adam_args = inspect.signature(FusedAdam).parameters
        arg_names = [
            "master_weight_dtype",
            "exp_avg_dtype",
            "exp_avg_sq_dtype",
            "use_decoupled_grad",
        ]
        for name in arg_names:
            if name not in adam_args:
                pytest.skip("TE FusedAdam does not support precision-aware args")

    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )

    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=use_precision_aware,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    optim = get_megatron_optimizer(optimizer_config, [model])

    # Run a training step to populate optimizer state
    input_data = torch.randn(8, 100, dtype=torch.bfloat16, device='cuda')
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optim.step()

    # Access the underlying distrib_optimizer
    distrib_optim = optim.chained_optimizers[0]
    if use_precision_aware:
        assert distrib_optim.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8

    # Inject non-tensor entries into optimizer state (simulates found_inf, etc.)
    inner_optimizer = distrib_optim.optimizer
    for param in inner_optimizer.state:
        inner_optimizer.state[param]['found_inf'] = False
        inner_optimizer.state[param]['non_tensor_int'] = 42

    # Test 1: _get_main_param_and_optimizer_states should skip non-tensor entries
    for gbuf_range_maps in distrib_optim.gbuf_ranges:
        for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
            for gbuf_range_map in gbuf_range_map_for_all_buckets:
                for model_param in gbuf_range_map["param_map"]:
                    tensors = distrib_optim._get_main_param_and_optimizer_states(model_param)
                    for k, v in tensors.items():
                        assert isinstance(
                            v, torch.Tensor
                        ), f"Non-tensor value for key '{k}': {type(v)}"

    # Test 2: Full save/load roundtrip via dp_reshardable path
    saved_state = distrib_optim.get_parameter_state_dp_reshardable()

    # Verify saved state doesn't contain non-tensor entries (except metadata keys)
    metadata_keys = {"per_bucket_numel", "per_bucket_numel_unpadded"}
    for key, value in saved_state.items():
        if key in metadata_keys:
            continue
        for dtype, buckets_state in value.items():
            for bucket_state in buckets_state:
                for param_dict in bucket_state:
                    for k, v in param_dict.items():
                        if k in ('gbuf_local_start', 'gbuf_local_end', 'padding'):
                            continue
                        assert isinstance(
                            v, torch.Tensor
                        ), f"Non-tensor in saved state key '{k}': {type(v)}"

    # Test 3: load_parameter_state_from_dp_reshardable should not crash
    # Add 'padding' key required by the load path (normally added by fully_reshardable save)
    for key, value in saved_state.items():
        if key in metadata_keys:
            continue
        for dtype, buckets_state in value.items():
            for bucket_state in buckets_state:
                for param_dict in bucket_state:
                    param_dict['padding'] = False
    distrib_optim.load_parameter_state_from_dp_reshardable(saved_state)

    # Test 4: Inject non-tensor entries directly into the saved state and verify load handles them
    for key, value in saved_state.items():
        if key in metadata_keys:
            continue
        for dtype, buckets_state in value.items():
            for bucket_state in buckets_state:
                for param_dict in bucket_state:
                    param_dict['found_inf'] = False
                    param_dict['step_count'] = 42

    # This should not crash - non-tensor entries should be skipped
    distrib_optim.load_parameter_state_from_dp_reshardable(saved_state)


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("precision", ['bf16', 'fp32'])
def test_optim_sharded_state_dict(use_distributed_optimizer: bool, precision: str):
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model, mock_args.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=use_distributed_optimizer)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    for param in model.parameters():
        assert param.requires_grad

    if precision == 'bf16':
        optimizer_config = OptimizerConfig(
            optimizer='adam', bf16=True, use_distributed_optimizer=use_distributed_optimizer
        )
    elif precision == 'fp32':
        optimizer_config = OptimizerConfig(
            optimizer='adam',
            bf16=False,
            fp16=False,
            use_distributed_optimizer=use_distributed_optimizer,
        )
    optim = get_megatron_optimizer(optimizer_config, [model])

    model_sharded_state_dict = model.sharded_state_dict()
    metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}
    if precision == 'bf16' or use_distributed_optimizer:
        sharded_state_dict = optim.sharded_state_dict(
            model_sharded_state_dict, metadata=metadata, is_loading=True
        )
    else:
        sharded_state_dict = optim.sharded_state_dict(model_sharded_state_dict)

    if 'optimizer' in sharded_state_dict and 'state' in sharded_state_dict['optimizer']:
        assert (
            'common_step' not in sharded_state_dict['optimizer']['state']
            or sharded_state_dict['optimizer']['state']['common_step'] is not None
        ), "Found 'optimizer.state.common_step=None' in sharded state dict."


def test_optimizer_reload_model_params():
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    model = Net().bfloat16().cuda()
    # Initial values of model params are 1.
    for param in model.parameters():
        param.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    optimizer_config = OptimizerConfig(optimizer='adam', bf16=True, use_distributed_optimizer=True)
    optim = get_megatron_optimizer(optimizer_config, [model])

    # Set all model params to 2.
    for param in model.parameters():
        param.data.fill_(2.0)

    # Although model params are 2 now, but we haven't called reload_model_params() yet, so
    # main_params should be 1.
    for group in optim.param_groups:
        for main_param in group['params']:
            assert main_param.dtype == torch.float32
            torch.testing.assert_close(
                main_param, torch.empty_like(main_param).fill_(1.0), atol=0, rtol=0
            )

    # Copy model params to main_params, so main_params should be 2 now.
    optim.reload_model_params()
    for group in optim.param_groups:
        for main_param in group['params']:
            assert main_param.dtype == torch.float32
            torch.testing.assert_close(
                main_param, torch.empty_like(main_param).fill_(2.0), atol=0, rtol=0
            )

    # Create a new state_dict with all params set to 3.
    state_dict = model.state_dict()
    new_state_dict = {}
    for name, param in state_dict.items():
        new_state_dict[name] = torch.empty_like(param).fill_(3.0)

    # Initialize main_params with the new state_dict, so main_params should be 3 now, but model
    # params should still be 2.
    optim.reload_model_params(new_state_dict)
    for param in model.parameters():
        torch.testing.assert_close(param, torch.empty_like(param).fill_(2.0), atol=0, rtol=0)
    for group in optim.param_groups:
        for main_param in group['params']:
            assert main_param.dtype == torch.float32
            torch.testing.assert_close(
                main_param, torch.empty_like(main_param).fill_(3.0), atol=0, rtol=0
            )


@pytest.mark.skipif(
    not is_torch_min_version("2.4.0"),
    reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
)
@pytest.mark.parametrize(
    "world_size, tp_size, cp_size, dp_size",
    [
        (1, 1, 1, 1),  # Single GPU, no parallelism
        (2, 1, 2, 1),  # 2 GPUs, 1 TP, 2 CP
        (2, 2, 1, 1),  # 2 GPUs, 2 TP, 1 CP
        (8, 8, 1, 1),  # 8 GPUs, 8 TP, 1 CP
        (8, 2, 4, 1),  # 8 GPUs, 2 TP, 4 CP
        (8, 4, 2, 1),  # 8 GPUs, 4 TP, 2 CP
        (8, 1, 1, 8),  # 8 GPUs, 1 TP, 1 CP, 8 DP
        (8, 2, 1, 4),  # 8 GPUs, 2 TP, 1 CP, 4 DP
        (8, 2, 2, 2),  # 8 GPUs, 2 TP, 2 CP, 2 DP
    ],
)
def test_get_megatron_optimizer_with_custom_process_groups(world_size, tp_size, cp_size, dp_size):
    """
    Test that get_megatron_optimizer works correctly with custom process groups
    provided via pg_collection parameters.
    """
    # Skip if world size doesn't match available GPUs
    actual_world_size = torch.cuda.device_count()
    if actual_world_size != world_size:
        pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")

    # Initialize model parallel with default settings first
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )

    # Create device mesh for custom process groups
    device_mesh = torch.distributed.init_device_mesh(
        "cuda", (1, dp_size, 1, cp_size, tp_size), mesh_dim_names=("pp", "dp", "ep", "cp", "tp")
    )

    # Create custom process groups from device mesh
    dp_group = device_mesh.get_group(mesh_dim="dp")
    cp_group = device_mesh.get_group(mesh_dim="cp")
    tp_group = device_mesh.get_group(mesh_dim="tp")
    pp_group = device_mesh.get_group(mesh_dim="pp")

    # Create dp_cp group
    dp_cp_mesh = device_mesh["dp", "cp"]
    dp_cp_group = dp_cp_mesh._flatten().get_group()

    # Create model parallel group (tp + pp)
    mp_mesh = device_mesh["pp", "tp"]
    mp_group = mp_mesh._flatten().get_group()

    # Create intra_dist_opt group
    # It has the same ranks as dp_cp group when num_distributed_optimizer_instances is not > 1
    intra_dist_opt_mesh = device_mesh["dp", "cp"]
    intra_dist_opt_group = intra_dist_opt_mesh._flatten().get_group()

    # Create process group configurations
    pg_collection = ProcessGroupCollection()
    pg_collection.dp = dp_group
    pg_collection.dp_cp = dp_cp_group
    pg_collection.intra_dist_opt = intra_dist_opt_group
    pg_collection.expt_dp = None  # Not using expert parallelism in this test

    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    pg_collection.pp = pp_group
    pg_collection.mp = mp_group
    pg_collection.tp_ep_pp = None  # Not using expert parallelism in this test

    # Create a simple model for testing
    model = torch.nn.Linear(100, 100, bias=False, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    for param in model.parameters():
        assert param.requires_grad
    model_chunks = [model]

    # Create optimizer config
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.001,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    )

    # Test 1: Create optimizer with custom process groups
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=model_chunks,
        use_gloo_process_groups=False,  # Required when using custom process groups
        pg_collection=pg_collection,
    )

    # Verify optimizer was created successfully
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test 2: Verify optimizer can perform forward and backward pass
    input_tensor = torch.randn(32, 100, device='cuda', requires_grad=True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Test 3: Optimizer step should work
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original parameters
    original_weight = model.module.weight.data.clone()
    original_bias = model.module.bias.data.clone() if model.module.bias is not None else None

    # Perform optimizer step
    optimizer.step()

    # Verify parameters were updated
    assert not torch.equal(
        model.module.weight.data, original_weight
    ), "Weight should be updated after optimizer step"
    if model.module.bias is not None:
        assert not torch.equal(
            model.module.bias.data, original_bias
        ), "Bias should be updated after optimizer step"

    # Test 4: Compare with default process groups optimizer (if world_size allows)
    if world_size == 1:  # Only test on single GPU to avoid complex setup
        # Create optimizer with default process groups
        default_optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=model_chunks
        )

        # Both optimizers should have the same structure
        assert len(optimizer.param_groups) == len(
            default_optimizer.param_groups
        ), "Custom and default optimizers should have same number of parameter groups"


def test_get_megatron_optimizer_custom_process_groups_validation():
    """
    Test validation logic for custom process groups in get_megatron_optimizer.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    # Create a simple model
    model = torch.nn.Linear(100, 100, bias=False, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    for param in model.parameters():
        assert param.requires_grad
    model_chunks = [model]
    optimizer_config = OptimizerConfig(optimizer='adam', lr=0.001)

    # Test 2: Missing dp process group in pg_collection
    pg_collection_no_dp = ProcessGroupCollection()

    with pytest.raises(ValueError, match="dp process group is required"):
        get_megatron_optimizer(
            config=optimizer_config, model_chunks=model_chunks, pg_collection=pg_collection_no_dp
        )

    # Test 3: Missing expt_dp attribute in pg_collection
    pg_collection_no_expt_dp = ProcessGroupCollection()
    pg_collection_no_expt_dp.dp = torch.distributed.new_group()
    # Missing required 'expt_dp' attribute

    with pytest.raises(ValueError, match="expt_dp process group is required"):
        get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model_chunks,
            pg_collection=pg_collection_no_expt_dp,
        )

    # Test 4: Missing intra_dist_opt and mp attribute in pg_collection
    pg_collection_complete = ProcessGroupCollection()
    pg_collection_complete.dp = torch.distributed.new_group()
    pg_collection_complete.expt_dp = None  # Explicitly set to None as allowed

    # Missing required 'intra_dist_opt' attribute
    with pytest.raises(ValueError, match="intra_dist_opt process group is required"):
        get_megatron_optimizer(
            config=optimizer_config, model_chunks=model_chunks, pg_collection=pg_collection_complete
        )

    pg_collection_complete.intra_dist_opt = None  # Explicitly set to None as allowed
    # Missing required 'mp' attribute
    with pytest.raises(ValueError, match="mp process group is required"):
        get_megatron_optimizer(
            config=optimizer_config, model_chunks=model_chunks, pg_collection=pg_collection_complete
        )

    # Test 5: Missing tp_ep_pp attribute in pg_collection
    pg_collection_complete.mp = None  # Explicitly set to None as allowed

    with pytest.raises(ValueError, match="tp_ep_pp process group is required"):
        get_megatron_optimizer(
            config=optimizer_config, model_chunks=model_chunks, pg_collection=pg_collection_complete
        )

    # Test 6: Gloo process groups should not be used with custom process groups
    pg_collection_complete.mp = None  # Explicitly set to None as allowed
    pg_collection_complete.tp_ep_pp = None  # Explicitly set to None as allowed

    with pytest.raises(ValueError, match="Gloo process groups are not supported"):
        get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model_chunks,
            use_gloo_process_groups=True,  # Should be False when using custom groups
            pg_collection=pg_collection_complete,
        )
