# Copyright (c) 2026, BAAI. All rights reserved.
"""TEGroupedMLP construction: fused-impl eligibility guards for unsupported configurations and
FP8 quantization padding setup."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    Fp8Padding,
    Fp8Unpadding,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
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


def _grouped_submodules():
    """Grouped-linear submodules with real Transformer Engine modules (A100 only)."""
    return GroupedMLPSubmodules(
        linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
    )


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class _PlainLinear(torch.nn.Module):
    """Spec-provided grouped-linear stand-in that is not a TE GroupedLinear."""

    def __init__(self, num_local_experts, input_size, output_size, **kwargs):
        super().__init__()
        self.num_local_experts = num_local_experts

    def forward(self, hidden_states, tokens_per_expert):
        return hidden_states, None

    def backward_dw(self):
        pass


class TestTEGroupedMLPConstructorGuards:
    """Construction guards run on the A100 CI image with real Transformer Engine."""

    @pytest.mark.parametrize(
        'config_overrides',
        [
            pytest.param(
                dict(fine_grained_activation_offloading=True, offload_modules=['moe_act']),
                id='offload-moe-act',
            ),
            pytest.param(dict(moe_apply_probs_on_input=True), id='probs-on-input'),
            pytest.param(dict(), id='fc1-is-not-a-grouped-linear'),
        ],
    )
    def test_constructor_rejects_fused_impl_for_unsupported_configurations(
        self, model_parallel, config_overrides
    ):
        model_parallel()
        config = _experts_config(use_transformer_engine_op_fuser=True, **config_overrides)
        submodules = GroupedMLPSubmodules(linear_fc1=_PlainLinear, linear_fc2=_PlainLinear)

        with pytest.raises(
            AssertionError, match='Fused GroupedMLP is not supported for this configuration.'
        ):
            TEGroupedMLP(
                num_local_experts=config.num_moe_experts,
                config=config,
                submodules=submodules,
                pg_collection=get_default_pg_collection(),
            )

    def test_constructor_rejects_fused_impl_for_tensor_parallel_experts(self, model_parallel):
        model_parallel(tensor_model_parallel_size=2)
        config = _experts_config(use_transformer_engine_op_fuser=True, tensor_model_parallel_size=2)
        submodules = GroupedMLPSubmodules(linear_fc1=_PlainLinear, linear_fc2=_PlainLinear)

        with pytest.raises(
            AssertionError, match='Fused GroupedMLP is not supported for this configuration.'
        ):
            TEGroupedMLP(
                num_local_experts=config.num_moe_experts,
                config=config,
                submodules=submodules,
                pg_collection=get_default_pg_collection(),
            )

    def test_constructor_rejects_fused_impl_when_fc2_is_not_a_grouped_linear(self, model_parallel):
        model_parallel()
        config = _experts_config(
            use_transformer_engine_op_fuser=True, bf16=True, params_dtype=torch.bfloat16
        )
        submodules = GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=_PlainLinear
        )

        with pytest.raises(
            AssertionError, match='Fused GroupedMLP is not supported for this configuration.'
        ):
            TEGroupedMLP(
                num_local_experts=config.num_moe_experts,
                config=config,
                submodules=submodules,
                pg_collection=get_default_pg_collection(),
            )


class TestTEGroupedMLPConstructor:
    """Constructor behaviour that needs real Transformer Engine on the A100 CI node."""

    def test_constructor_creates_quantization_padding_for_fp8(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(
            fp8=True, bf16=True, params_dtype=torch.bfloat16, moe_grouped_gemm=True
        )
        experts = TEGroupedMLP(
            num_local_experts=config.num_moe_experts,
            config=config,
            submodules=_grouped_submodules(),
            pg_collection=get_default_pg_collection(),
        )

        assert experts._with_fused_impl is False
        assert isinstance(experts.quantization_padding, Fp8Padding)
        assert isinstance(experts.quantization_unpadding, Fp8Unpadding)
