# Copyright (c) 2026, BAAI. All rights reserved.
"""SharedExpertMLP FP8 constructor: saving the original fc1 input for FP8 unless
shared-expert recompute is enabled."""

import pytest
import torch
import torch.nn.functional as F

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


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestSharedExpertFP8Constructor:
    """FP8 constructor flag on TE shared-expert linears, A100 CI node only."""

    def test_constructor_saves_original_fc1_input_for_fp8(self, model_parallel):
        model_parallel()
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        config = _shared_expert_config(
            fp8=True, bf16=True, params_dtype=torch.bfloat16, fp8_recipe='tensorwise'
        )
        expert = SharedExpertMLP(
            config=config,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            gate=False,
            pg_collection=get_default_pg_collection(),
        )

        assert expert.linear_fc1.save_original_input is True

    def test_constructor_skips_saving_original_input_when_recompute_enabled(self, model_parallel):
        model_parallel()
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        config = _shared_expert_config(
            fp8=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8_recipe='tensorwise',
            recompute_granularity='selective',
            recompute_modules=['shared_experts'],
        )
        expert = SharedExpertMLP(
            config=config,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            gate=False,
            pg_collection=get_default_pg_collection(),
        )

        assert expert.linear_fc1.save_original_input is False
