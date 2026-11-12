# Copyright (c) 2026, BAAI. All rights reserved.
"""TEFusedMLP op-fuser MLP: forward equivalence with unfused submodules and hook emulation."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEFusedMLP,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

RTOL, ATOL = 1.6e-2, 1e-2


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _fused_config(**overrides):
    overrides.setdefault('params_dtype', torch.bfloat16)
    overrides.setdefault('bf16', True)
    overrides.setdefault('gradient_accumulation_fusion', False)
    overrides.setdefault('tensor_model_parallel_size', 2)
    overrides.setdefault('gated_linear_unit', True)
    overrides.setdefault('activation_func', torch.nn.functional.silu)
    overrides.setdefault('add_bias_linear', False)
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)
    config.init_method = lambda w: None
    config.output_layer_init_method = lambda w: None
    return config


def _det(rows, cols):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((values % 7) - 3).to(torch.bfloat16)


def _det_input(seq, batch, hidden):
    values = torch.arange(seq * batch * hidden, dtype=torch.float32)
    return (((values % 11) - 5) / 4.0).to(torch.bfloat16).reshape(seq, batch, hidden)


def _make_fused_mlp(model_parallel, **config_overrides):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    mlp = TEFusedMLP(
        _fused_config(**config_overrides),
        MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear),
        is_expert=False,
    )
    mlp.linear_fc1.weight.data.copy_(_det(128, 32).cuda())
    mlp.linear_fc1.layer_norm_weight.data.fill_(1.0)
    mlp.linear_fc1.layer_norm_bias.data.zero_()
    mlp.linear_fc2.weight.data.copy_(_det(32, 64).cuda())
    return mlp


def test_te_fused_mlp_forward_matches_unfused_submodules(model_parallel):
    mlp = _make_fused_mlp(model_parallel)
    hidden = _det_input(6, 2, 32).cuda()

    fused_out, fused_bias = mlp(hidden)
    assert fused_bias is None

    intermediate, bias_parallel = mlp.linear_fc1(hidden)
    assert bias_parallel is None
    x_glu, x_linear = torch.chunk(intermediate, 2, dim=-1)
    activated = mlp.config.activation_func(x_glu) * x_linear
    unfused_out, unfused_bias = mlp.linear_fc2(activated)
    assert unfused_bias is None

    assert fused_out.shape == (6, 2, 32)
    torch.testing.assert_close(fused_out, unfused_out, rtol=RTOL, atol=ATOL)


def test_te_fused_mlp_reuses_fused_impl_across_forwards(model_parallel):
    mlp = _make_fused_mlp(model_parallel)
    hidden = _det_input(6, 2, 32).cuda()

    first_out, _ = mlp(hidden)
    fused_impl = mlp._fused_impl[0]
    second_out, _ = mlp(hidden)
    assert mlp._fused_impl[0] is fused_impl
    torch.testing.assert_close(first_out, second_out, rtol=RTOL, atol=ATOL)


def test_te_fused_mlp_runs_registered_submodule_hooks(model_parallel):
    mlp = _make_fused_mlp(model_parallel)
    hook_calls = []
    mlp.linear_fc1.register_forward_hook(
        lambda module, args, output: hook_calls.append((module, args, output))
    )
    hidden = _det_input(6, 2, 32).cuda()
    with pytest.warns(
        UserWarning, match='TEFusedMLP module has a submodule with a post-forward hook'
    ):
        mlp(hidden)
    assert hook_calls == [(mlp.linear_fc1, (), None)]


def test_te_fused_mlp_rejects_modifying_submodule_pre_hooks(model_parallel):
    mlp = _make_fused_mlp(model_parallel)
    mlp.linear_fc1.register_forward_pre_hook(lambda module, args: torch.zeros(1))
    hidden = _det_input(6, 2, 32).cuda()
    with pytest.raises(RuntimeError, match='pre-forward hook that modifies input tensor'):
        mlp(hidden)


def test_te_fused_mlp_rejects_modifying_submodule_post_hooks(model_parallel):
    mlp = _make_fused_mlp(model_parallel)
    mlp.linear_fc1.register_forward_hook(lambda module, args, output: torch.zeros(1))
    hidden = _det_input(6, 2, 32).cuda()
    with pytest.raises(RuntimeError, match='post-forward hook that modifies output tensor'):
        mlp(hidden)


def test_te_fused_mlp_rejects_submodule_backward_hooks(model_parallel):
    mlp = _make_fused_mlp(model_parallel)
    mlp.linear_fc1.register_full_backward_pre_hook(lambda module, grad_output: None)
    hidden = _det_input(6, 2, 32).cuda()
    with pytest.raises(RuntimeError, match='does not support submodules with pre-backward hooks'):
        mlp(hidden)
