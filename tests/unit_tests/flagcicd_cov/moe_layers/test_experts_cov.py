# Copyright (c) 2026, BAAI. All rights reserved.
"""SequentialMLP experts: per-expert token splitting with on-input probability scaling and
sharded checkpoint keys with global indices for local expert shards."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _experts_config(**overrides):
    """Expert config with two small local experts, following the developer test conventions."""
    defaults = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=16,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        use_cpu_initialization=True,
        add_bias_linear=False,
        moe_token_dispatcher_type='allgather',
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.0,
        activation_func=F.silu,
        gated_linear_unit=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _sequential_experts(config):
    """SequentialMLP with two experts and deterministic silu-glu weights."""
    experts = SequentialMLP(
        num_local_experts=config.num_moe_experts,
        config=config,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
        pg_collection=get_default_pg_collection(),
    )
    for index, expert in enumerate(experts.local_experts):
        with torch.no_grad():
            fc1_weight = expert.linear_fc1.weight
            fc1_weight.copy_(
                torch.arange(fc1_weight.numel(), dtype=torch.float32).reshape(fc1_weight.shape)
                + 8.0 * index
            )
            fc2_weight = expert.linear_fc2.weight
            fc2_weight.copy_(
                torch.arange(fc2_weight.numel(), dtype=torch.float32).reshape(fc2_weight.shape)
                + 16.0 * index
            )
    return experts


def _reference_expert(expert, tokens, probs):
    """Plain-torch silu-glu MLP for one expert, used as an independent reference."""
    intermediate = tokens @ expert.linear_fc1.weight.t()
    gate, linear = torch.chunk(intermediate, 2, dim=-1)
    activated = F.silu(gate) * linear
    return (activated * probs.unsqueeze(-1)) @ expert.linear_fc2.weight.t()


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestSequentialMLP:

    def test_forward_applies_probs_on_input_and_matches_manual_expert_computation(
        self, model_parallel
    ):
        model_parallel()
        config = _experts_config(moe_apply_probs_on_input=True)
        experts = _sequential_experts(config)
        hidden = torch.arange(40, dtype=torch.float32).reshape(5, 8) / 8.0
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5])
        tokens_per_expert = torch.tensor([3, 2])

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        scaled = hidden * probs.unsqueeze(-1)
        expected = torch.cat(
            [
                _reference_expert(experts.local_experts[0], scaled[:3], torch.ones(3)),
                _reference_expert(experts.local_experts[1], scaled[3:], torch.ones(2)),
            ]
        )
        assert output_bias is None
        assert output.shape == (5, 8)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_forward_splits_tokens_across_experts_and_matches_manual_computation(
        self, model_parallel
    ):
        model_parallel()
        experts = _sequential_experts(_experts_config())
        hidden = torch.arange(40, dtype=torch.float32).reshape(5, 8) / 8.0
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5])
        tokens_per_expert = torch.tensor([3, 2])

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        expected = torch.cat(
            [
                _reference_expert(experts.local_experts[0], hidden[:3], probs[:3]),
                _reference_expert(experts.local_experts[1], hidden[3:], probs[3:]),
            ]
        )
        assert output_bias is None
        assert output.shape == (5, 8)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_sharded_state_dict_maps_local_experts_to_global_indices_for_singleton_shards(
        self, model_parallel
    ):
        model_parallel()
        config = _experts_config(activation_func=F.relu, gated_linear_unit=False)
        experts = _sequential_experts(config)
        dp_group = experts.dp_group
        metadata = {'dp_cp_group': dp_group}

        singleton = experts.sharded_state_dict(metadata=dict(metadata, singleton_local_shards=True))
        default = experts.sharded_state_dict(metadata=dict(metadata, singleton_local_shards=False))

        assert sorted(singleton) == [
            'local_experts.0.linear_fc1._extra_state',
            'local_experts.0.linear_fc1.weight',
            'local_experts.0.linear_fc2._extra_state',
            'local_experts.0.linear_fc2.weight',
            'local_experts.1.linear_fc1._extra_state',
            'local_experts.1.linear_fc1.weight',
            'local_experts.1.linear_fc2._extra_state',
            'local_experts.1.linear_fc2.weight',
        ]
        singleton_fc1 = singleton['local_experts.1.linear_fc1.weight']
        default_fc2 = default['local_experts.1.linear_fc2.weight']
        assert singleton_fc1.key == 'experts.1.linear_fc1.weight'
        assert singleton_fc1.global_shape == (16, 8)
        assert singleton_fc1.global_offset == (0, 0)
        assert singleton_fc1.replica_id == (0, 0, dp_group.rank())
        assert default_fc2.key == 'experts.linear_fc2.weight'
        assert default_fc2.global_shape == (2, 8, 16)
        assert default_fc2.global_offset == (1, 0, 0)
        assert default_fc2.replica_id == (0, 0, dp_group.rank())
        torch.testing.assert_close(
            singleton['local_experts.0.linear_fc1.weight'].data,
            experts.local_experts[0].linear_fc1.weight,
            rtol=1e-6,
            atol=1e-6,
        )
