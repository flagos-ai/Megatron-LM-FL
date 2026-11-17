# Copyright (c) 2026, BAAI. All rights reserved.
"""SharedExpertMLP forward: gated silu-glu output scaled by the gate score and gate-weight
entries in the sharded state dict."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


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


def _reference_forward(expert, hidden):
    """Plain-torch gated silu-glu forward, used as an independent reference."""
    intermediate = hidden @ expert.linear_fc1.weight.t()
    gate, linear = torch.chunk(intermediate, 2, dim=-1)
    activated = F.silu(gate) * linear
    output = activated @ expert.linear_fc2.weight.t()
    if expert.use_shared_expert_gate:
        gate_score = torch.sigmoid(hidden @ expert.gate_weight.t())
        output = output * gate_score
    return output


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestSharedExpertForward:

    @pytest.mark.parametrize(
        'gate', [pytest.param(True, id='gated'), pytest.param(False, id='ungated')]
    )
    def test_forward_scales_mlp_output_by_the_gate_score(self, model_parallel, gate):
        model_parallel()
        expert = _shared_expert(_shared_expert_config(), gate=gate)
        hidden = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8) / 8.0

        output = expert(hidden)

        expected = _reference_forward(expert, hidden)
        assert output.shape == (2, 2, 8)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_forward_passes_input_through_unchanged_when_no_expert_gate(self, model_parallel):
        model_parallel()
        expert = _shared_expert(_shared_expert_config(), gate=False)
        hidden = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8) / 8.0

        output = expert(hidden)

        torch.testing.assert_close(output, _reference_forward(expert, hidden), rtol=1e-5, atol=1e-6)

    def test_sharded_state_dict_adds_gate_weight_entry(self, model_parallel):
        model_parallel()
        expert = _shared_expert(_shared_expert_config(), gate=True)
        gate_weight = expert.gate_weight.detach().clone()

        sharded = expert.sharded_state_dict(metadata={'dp_cp_group': expert.tp_group})

        assert 'gate_weight' in sharded
        assert sharded['gate_weight'].key == 'gate_weight'
        torch.testing.assert_close(sharded['gate_weight'].data, gate_weight, rtol=1e-6, atol=1e-6)
        assert 'linear_fc1.weight' in sharded
        assert 'linear_fc2.weight' in sharded
