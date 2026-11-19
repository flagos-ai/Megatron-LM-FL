# Copyright (c) 2026, BAAI. All rights reserved.
"""OptimizerConfig: precision-aware optimizer flags with CPU offload and mxfp8 grad-buffer warnings; ParamKey: predicate tuple matching rules."""

import pytest
import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey


def test_param_key_matches_when_any_predicate_in_tuple_matches():
    param = torch.nn.Parameter(torch.zeros(2, 3))
    key = ParamKey(predicate=(lambda p: p.ndim == 1, lambda p: p.shape == torch.Size([2, 3])))
    assert key.matches(param, 'module.weight') is True


def test_param_key_rejects_when_no_predicate_in_tuple_matches():
    param = torch.nn.Parameter(torch.zeros(2, 3))
    key = ParamKey(predicate=(lambda p: p.ndim == 1, lambda p: p.dtype == torch.float64))
    assert key.matches(param, 'module.weight') is False


def test_param_key_matches_when_any_with_name_predicate_in_tuple_matches():
    param = torch.nn.Parameter(torch.zeros(1))
    key = ParamKey(
        with_name_predicate=(lambda p, n: n.endswith('weight'), lambda p, n: n.startswith('mlp'))
    )
    assert key.matches(param, 'mlp.fc1.bias') is True


def test_param_key_rejects_when_no_with_name_predicate_in_tuple_matches():
    param = torch.nn.Parameter(torch.zeros(1))
    key = ParamKey(
        with_name_predicate=(lambda p, n: n.endswith('bias'), lambda p, n: n.startswith('mlp'))
    )
    assert key.matches(param, 'module.weight') is False


def test_optimizer_config_warns_when_mxfp8_reuses_grad_buffer():
    with pytest.warns(UserWarning, match='mxfp8 without using reuse_grad_buf_for_mxfp8_param_ag'):
        OptimizerConfig(fp8_recipe='mxfp8')


def test_optimizer_config_sets_precision_aware_flag_with_cpu_offload():
    config = OptimizerConfig(
        use_precision_aware_optimizer=True,
        optimizer='adam',
        use_distributed_optimizer=True,
        optimizer_cpu_offload=True,
    )
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 is True
