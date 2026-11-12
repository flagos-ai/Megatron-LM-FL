# Copyright (c) 2026, BAAI. All rights reserved.
"""SharedExpertMLP overlapped fc1 activation: fused bias-activation paths, TE activation
modules, GLU clamping and ungated activations."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _tanh_gelu(x):
    """GELU with the tanh approximation the fused geglu kernel computes."""
    return F.gelu(x, approximate='tanh')


def _shared_expert_config(**overrides):
    """Config with one small shared expert, following the developer test conventions."""
    defaults = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        moe_shared_expert_intermediate_size=16,
        use_cpu_initialization=True,
        add_bias_linear=False,
        moe_token_dispatcher_type='alltoall',
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.0,
        activation_func=F.silu,
        gated_linear_unit=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _local_submodules():
    """MLP submodules with local (non-TE) linear modules."""
    return MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)


def _shared_expert(config, gate):
    """SharedExpertMLP with deterministic silu-glu weights and gate."""
    expert = SharedExpertMLP(
        config=config,
        submodules=_local_submodules(),
        gate=gate,
        pg_collection=get_default_pg_collection(),
    )
    with torch.no_grad():
        fc1_weight = expert.linear_fc1.weight
        fc1_weight.copy_(
            torch.arange(fc1_weight.numel(), dtype=torch.float32).reshape(fc1_weight.shape) / 64.0
        )
        fc2_weight = expert.linear_fc2.weight
        fc2_weight.copy_(
            torch.arange(fc2_weight.numel(), dtype=torch.float32).reshape(fc2_weight.shape) / 128.0
        )
        if gate:
            expert.gate_weight.copy_(
                torch.arange(expert.gate_weight.numel(), dtype=torch.float32).reshape(
                    expert.gate_weight.shape
                )
                / 8.0
            )
    return expert


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestOverlappedActivationPaths:
    """Activation handling inside the overlapped fc1 step on the A100 CI node."""

    def _run_fc1_activation(self, config, hidden):
        expert = _shared_expert(
            _shared_expert_config(moe_shared_expert_overlap=True, **config), gate=False
        ).cuda()
        expert.pre_forward_comm(hidden)
        expert.linear_fc1_forward_and_act()
        return expert.cached_fc2_input

    def test_silu_glu_bias_fusion_matches_unfused_activation(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        fused = self._run_fc1_activation(
            dict(bias_activation_fusion=True),
            torch.arange(32, device=device).reshape(2, 2, 8) / 8.0,
        )
        plain = self._run_fc1_activation(
            dict(), torch.arange(32, device=device).reshape(2, 2, 8) / 8.0
        )

        assert fused.shape == (2, 2, 16)
        torch.testing.assert_close(fused, plain, rtol=1e-5, atol=1e-6)

    def test_gelu_glu_bias_fusion_matches_unfused_activation(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        hidden = torch.arange(32, device=device).reshape(2, 2, 8) / 8.0
        fused = self._run_fc1_activation(
            dict(activation_func=F.gelu, bias_activation_fusion=True), hidden
        )
        plain = self._run_fc1_activation(dict(activation_func=_tanh_gelu), hidden)

        torch.testing.assert_close(fused, plain, rtol=1e-5, atol=1e-6)

    def test_bias_fusion_rejects_unsupported_activation(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        # relu is rejected up front by the config validation.
        with pytest.raises(
            ValueError,
            match='When bias_activation_fusion is True, activation function should be '
            'either gelu, swiglu, or quick_geglu',
        ):
            self._run_fc1_activation(
                dict(activation_func=F.relu, bias_activation_fusion=True), torch.zeros(2, 2, 8)
            )

        # quick_gelu passes the config check but the shared-expert fused path does not
        # implement it, so the layer itself rejects it during linear_fc1_forward_and_act.
        with pytest.raises(ValueError, match='Only support fusion of gelu and swiglu'):
            self._run_fc1_activation(
                dict(activation_func=quick_gelu, bias_activation_fusion=True),
                torch.zeros(2, 2, 8, device='cuda'),
            )

    def test_te_activation_module_matches_plain_glu(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        from megatron.core.extensions.transformer_engine import (
            TEActivationOp,
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        te_config = _shared_expert_config(
            moe_shared_expert_overlap=True, use_te_activation_func=True
        )
        te_expert = SharedExpertMLP(
            config=te_config,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
                activation_func=TEActivationOp,
            ),
            gate=False,
            pg_collection=get_default_pg_collection(),
        ).cuda()
        plain = _shared_expert(
            _shared_expert_config(moe_shared_expert_overlap=True), gate=False
        ).cuda()
        with torch.no_grad():
            te_expert.linear_fc1.weight.copy_(plain.linear_fc1.weight)
            te_expert.linear_fc2.weight.copy_(plain.linear_fc2.weight)
        hidden = torch.arange(32, device=device).reshape(2, 2, 8) / 8.0

        te_expert.pre_forward_comm(hidden)
        te_expert.linear_fc1_forward_and_act()
        plain.pre_forward_comm(hidden)
        plain.linear_fc1_forward_and_act()

        # The TE activation module returns the gated half-width output like the plain
        # GLU path. Both fc1 GEMMs run in fp32 TF32 on the CI image, so compare the
        # fp32 result with the fp32 reference at the matmul tolerance.
        torch.testing.assert_close(
            te_expert.cached_fc2_input,
            plain.cached_fc2_input,
            rtol=1e-3,
            atol=1e-3 * plain.cached_fc2_input.abs().max().item(),
        )

    def test_glu_clamp_limits_gate_and_linear_halves(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        clamp_value = 0.5
        expert = _shared_expert(
            _shared_expert_config(
                moe_shared_expert_overlap=True,
                activation_func_clamp_value=clamp_value,
                # SwiGLU clamping is only validated for MoE configs
                # (transformer_config.py:2304).
                num_moe_experts=2,
            ),
            gate=False,
        ).cuda()
        hidden = torch.arange(32, device=device).reshape(2, 2, 8) / 8.0

        expert.pre_forward_comm(hidden)
        expert.linear_fc1_forward_and_act()

        intermediate = hidden @ expert.linear_fc1.weight.t()
        gate, linear = torch.chunk(intermediate, 2, dim=-1)
        expected = F.silu(gate.clamp(max=clamp_value)) * linear.clamp(
            min=-clamp_value, max=clamp_value
        )
        assert expert.cached_fc2_input.shape == (2, 2, 16)
        torch.testing.assert_close(expert.cached_fc2_input, expected, rtol=1e-5, atol=1e-6)

    def test_ungated_relu_applies_activation_without_glu(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        expert = _shared_expert(
            _shared_expert_config(
                moe_shared_expert_overlap=True, activation_func=F.relu, gated_linear_unit=False
            ),
            gate=False,
        ).cuda()
        hidden = torch.arange(32, device=device).reshape(2, 2, 8) / 8.0 - 2.0

        expert.pre_forward_comm(hidden)
        expert.linear_fc1_forward_and_act()

        intermediate = hidden @ expert.linear_fc1.weight.t()
        assert expert.cached_fc2_input.shape == (2, 2, 16)
        torch.testing.assert_close(
            expert.cached_fc2_input, F.relu(intermediate), rtol=1e-5, atol=1e-6
        )
