# Copyright (c) 2026, BAAI. All rights reserved.
"""MoELayer behaviour: dispatcher selection guards, staged forward execution, shared-expert
recompute and delayed weight-gradient overlap."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.router as router_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoELayer
from megatron.core.transformer.moe.moe_utils import (
    get_default_pg_collection as moe_layer_default_pg,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP, SharedExpertState
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _cuda_or_cpu_device():
    """cuda on the A100 CI node, cpu on a laptop: what the real platform reports."""
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _moe_config(**overrides):
    """MoE layer config following the developer test conventions."""
    defaults = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        num_moe_experts=4,
        use_cpu_initialization=True,
        moe_token_dispatcher_type='allgather',
        moe_router_load_balancing_type='aux_loss',
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_ffn_hidden_size=16,
        add_bias_linear=False,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _local_submodules():
    """SequentialMLP experts with local (non-TE) linear modules."""
    return get_submodules(get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False).mlp)


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def pinned_device(monkeypatch):
    """Pin the router's platform object to one device for both run environments."""
    device = _cuda_or_cpu_device()
    monkeypatch.setattr(
        router_module, 'cur_platform', SimpleNamespace(current_device=lambda: device)
    )
    return device


class _DelegatingMoELayer(BaseMoELayer):
    """Concrete MoE layer that calls the abstract base forward."""

    def forward(self, hidden_states):
        return super().forward(hidden_states)


class TestBaseMoELayerContract:

    def test_forward_is_a_noop_placeholder_on_base_moe_layer(self, model_parallel):
        model_parallel()
        layer = _DelegatingMoELayer(config=_moe_config(), pg_collection=moe_layer_default_pg())

        assert layer.forward(torch.zeros(2, 2, 8)) is None


class TestConstructor:

    def test_constructor_rejects_unknown_dispatcher_type(self, model_parallel, pinned_device):
        model_parallel()

        with pytest.raises(ValueError, match='Unsupported token dispatcher type: fastpath'):
            MoELayer(_moe_config(moe_token_dispatcher_type='fastpath'), _local_submodules())

    def test_constructor_wires_shared_experts_into_dispatcher_when_overlap_enabled(
        self, model_parallel
    ):
        model_parallel()
        layer = MoELayer(
            _moe_config(
                moe_token_dispatcher_type='alltoall',
                moe_shared_expert_intermediate_size=16,
                moe_shared_expert_overlap=True,
            ),
            _local_submodules(),
        )
        shared_experts = layer.shared_experts

        assert layer.shared_expert_overlap is True
        assert layer.token_dispatcher.shared_experts is shared_experts
        assert isinstance(shared_experts, SharedExpertMLP)
        assert shared_experts.config.ffn_hidden_size == 16
        assert shared_experts.linear_fc1.explicit_expert_comm is True
        assert shared_experts.linear_fc2.explicit_expert_comm is True
        assert shared_experts._overlap_state is SharedExpertState.IDLE
        assert shared_experts.cached_fc1_input is None
        assert shared_experts.cached_output is None
        assert shared_experts.gate_weight is None


class TestForwardStaging:

    def test_forward_returns_route_stage_tensors_when_intermediate_tensors_given(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        layer = MoELayer(_moe_config(), _local_submodules()).to(pinned_device)
        hidden = torch.randn(4, 2, 8, device=pinned_device)

        hidden_states, probs, shared_expert_output = layer.forward(hidden, intermediate_tensors=())

        assert hidden_states.shape == (8, 8)
        assert probs.shape == (8, 4)
        assert shared_expert_output is None
        assert (probs > 0).sum(dim=1).tolist() == [2] * 8
        torch.testing.assert_close(
            probs.sum(dim=1), torch.ones(8, device=pinned_device), rtol=1e-5, atol=1e-6
        )

    def test_forward_requires_sequence_parallel_when_training_with_tensor_parallel(
        self, model_parallel
    ):
        model_parallel(tensor_model_parallel_size=2)
        layer = MoELayer(
            _moe_config(tensor_model_parallel_size=2, sequence_parallel=False), _local_submodules()
        ).cuda()

        with pytest.raises(
            ValueError, match='performance may degrade if MoE and tensor parallelism'
        ):
            layer(torch.randn(4, 2, 8, device='cuda'))


class TestSharedExpertsCompute:
    """Full forwards need the fused permute kernels: A100 only."""

    def test_shared_experts_compute_uses_checkpoint_when_recompute_enabled(self, model_parallel):
        model_parallel()
        base_config = dict(
            moe_token_dispatcher_type='alltoall', moe_shared_expert_intermediate_size=16
        )
        recompute_layer = MoELayer(
            _moe_config(
                **base_config,
                recompute_granularity='selective',
                recompute_modules=['shared_experts'],
            ),
            _local_submodules(),
        ).cuda()
        plain_layer = MoELayer(_moe_config(**base_config), _local_submodules()).cuda()
        recompute_layer.load_state_dict(plain_layer.state_dict())
        torch.manual_seed(123)
        hidden = torch.randn(4, 2, 8, device='cuda', dtype=torch.float32, requires_grad=True)

        recompute_output, _ = recompute_layer(hidden)

        plain_hidden = hidden.detach().clone().requires_grad_(True)
        plain_output, _ = plain_layer(plain_hidden)

        torch.testing.assert_close(recompute_output, plain_output, rtol=1e-5, atol=1e-6)
        recompute_output.sum().backward()
        plain_output.sum().backward()
        torch.testing.assert_close(hidden.grad, plain_hidden.grad, rtol=1e-5, atol=1e-6)
        assert recompute_layer.shared_experts_recompute is True


class TestSetForRecomputePreMlpLayernorm:

    def test_marks_shared_expert_fc1_to_save_original_input(self, model_parallel):
        model_parallel()
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=4, moe_grouped_gemm=False
            ).mlp
        )
        layer = MoELayer(
            _moe_config(
                moe_token_dispatcher_type='alltoall', moe_shared_expert_intermediate_size=16
            ),
            submodules,
        ).cuda()

        layer.set_for_recompute_pre_mlp_layernorm()

        assert layer.shared_experts.linear_fc1.save_original_input is True


class TestDelayedWgradOverlap:

    def test_forward_and_backward_match_plain_layer_when_delayed_wgrad_enabled(
        self, model_parallel
    ):
        model_parallel()
        plain_layer = MoELayer(
            _moe_config(moe_token_dispatcher_type='alltoall'), _local_submodules()
        ).cuda()
        delayed_layer = MoELayer(
            _moe_config(
                moe_token_dispatcher_type='alltoall',
                overlap_dispatch_backward_with_experts_wgrad=True,
            ),
            _local_submodules(),
        ).cuda()
        delayed_layer.load_state_dict(plain_layer.state_dict())
        torch.manual_seed(321)
        hidden = torch.randn(4, 2, 8, device='cuda', requires_grad=True)

        delayed_output, _ = delayed_layer(hidden)

        plain_hidden = hidden.detach().clone().requires_grad_(True)
        plain_output, _ = plain_layer(plain_hidden)

        torch.testing.assert_close(delayed_output, plain_output, rtol=1e-5, atol=1e-6)
        delayed_output.sum().backward()
        plain_output.sum().backward()
        torch.testing.assert_close(hidden.grad, plain_hidden.grad, rtol=1e-5, atol=1e-6)
        for name, delayed_param in delayed_layer.named_parameters():
            if 'local_experts' in name:
                plain_param = plain_layer.get_parameter(name)
                torch.testing.assert_close(
                    delayed_param.grad, plain_param.grad, rtol=1e-5, atol=1e-6
                )
