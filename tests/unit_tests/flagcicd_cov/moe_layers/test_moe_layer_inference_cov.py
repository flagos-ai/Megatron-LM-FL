# Copyright (c) 2026, BAAI. All rights reserved.
"""Inference-optimized MoE layer behaviour: dispatcher swapping under InferenceMode, latent
projections and their shared-expert overlap guard rails."""

import pytest
import torch

from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.token_dispatcher_inference import (
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


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


class _RoutingAwareExperts(torch.nn.Module):
    """Spec-provided experts that return the dispatched tokens unchanged.

    Records the routing_map the MoE layer passes so the test can check the
    inference call signature; a probability-weighted combine of identity
    experts reproduces the input because top-k softmax weights sum to one.
    """

    def __init__(self, num_local_experts, config, **kwargs):
        super().__init__()
        self.received_routing_map = None

    def forward(self, dispatched_input, tokens_per_expert, permuted_probs, routing_map=None):
        self.received_routing_map = routing_map
        return dispatched_input, None

    def backward_dw(self):
        pass


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestInferenceSetup:

    def test_constructor_swaps_to_nccl_inference_dispatcher_when_inference_optimized(
        self, model_parallel
    ):
        model_parallel()
        layer = MoELayer(
            _moe_config(
                transformer_impl='inference_optimized',
                moe_router_dtype='fp32',
                normalization='RMSNorm',
                inference_moe_token_dispatcher_type='nccl',
            ),
            _local_submodules(),
        )

        assert layer.token_dispatcher is layer._training_token_dispatcher
        assert isinstance(layer._inference_token_dispatcher, NCCLAllGatherDispatcher)
        assert layer._latent_shared_expert_output is None

    def test_constructor_uses_inference_linear_for_latent_projections(self, model_parallel):
        model_parallel()
        layer = MoELayer(
            _moe_config(
                transformer_impl='inference_optimized',
                moe_router_dtype='fp32',
                normalization='RMSNorm',
                inference_moe_token_dispatcher_type='nccl',
                moe_latent_size=4,
            ),
            _local_submodules(),
        ).cuda()
        from megatron.core.tensor_parallel.inference_layers import InferenceLinear

        assert isinstance(layer.fc1_latent_proj, InferenceLinear)
        assert isinstance(layer.fc2_latent_proj, InferenceLinear)


class TestInferenceBackendGuards:

    def test_constructor_accepts_torch_grouped_gemm_backend(self, model_parallel):
        model_parallel()
        layer = MoELayer(
            _moe_config(
                transformer_impl='inference_optimized',
                moe_router_dtype='fp32',
                normalization='RMSNorm',
                inference_moe_token_dispatcher_type='nccl',
                inference_grouped_gemm_backend='torch',
            ),
            _local_submodules(),
        )

        assert isinstance(layer._inference_token_dispatcher, NCCLAllGatherDispatcher)
        assert (
            layer.experts.config.inference_grouped_gemm_backend is InferenceGroupedGemmBackend.TORCH
        )

    def test_constructor_wires_shared_experts_into_the_nvls_inference_dispatcher(
        self, model_parallel
    ):
        model_parallel()
        layer = MoELayer(
            _moe_config(
                transformer_impl='inference_optimized',
                moe_router_dtype='fp32',
                normalization='RMSNorm',
                inference_moe_token_dispatcher_type='nvls',
                moe_shared_expert_intermediate_size=16,
                moe_shared_expert_overlap=True,
                moe_latent_size=4,
            ),
            _local_submodules(),
        ).cuda()

        assert isinstance(layer._inference_token_dispatcher, NVLSAllGatherVDispatcher)
        assert layer._inference_token_dispatcher.shared_experts is layer.shared_experts
        assert layer._inference_token_dispatcher._external_shared_expert_launch is True
        assert layer._latent_shared_expert_output is None


class TestLatentOverlapGuards:

    def _latent_overlap_layer(self):
        return MoELayer(
            _moe_config(
                moe_token_dispatcher_type='alltoall',
                moe_latent_size=4,
                moe_shared_expert_intermediate_size=16,
                moe_shared_expert_overlap=True,
            ),
            _local_submodules(),
        ).cuda()

    def test_preprocess_rejects_shared_expert_overlap_with_latent_in_training(self, model_parallel):
        model_parallel()
        layer = self._latent_overlap_layer()
        layer.train()

        with pytest.raises(
            AssertionError,
            match='Shared expert overlap with MoE latent projections is not supported',
        ):
            layer.preprocess(
                torch.zeros(2, 1, 8, device='cuda'),
                torch.zeros(2, 4, device='cuda'),
                torch.zeros(2, 4, dtype=torch.bool, device='cuda'),
            )

    def test_preprocess_rejects_shared_expert_overlap_without_nvls_dispatcher(self, model_parallel):
        model_parallel()
        layer = self._latent_overlap_layer()
        layer.eval()

        with pytest.raises(AssertionError, match='requires the NVLS inference dispatcher'):
            layer.preprocess(
                torch.zeros(2, 1, 8, device='cuda'),
                torch.zeros(2, 4, device='cuda'),
                torch.zeros(2, 4, dtype=torch.bool, device='cuda'),
            )


class TestLatentBackwardDw:

    def test_backward_dw_runs_experts_shared_experts_and_latent_projections(self, model_parallel):
        model_parallel()
        base_config = dict(
            moe_token_dispatcher_type='alltoall',
            moe_latent_size=4,
            moe_shared_expert_intermediate_size=16,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        delayed = MoELayer(
            _moe_config(overlap_moe_expert_parallel_comm=True, **base_config), _local_submodules()
        ).cuda()
        plain = MoELayer(
            _moe_config(overlap_moe_expert_parallel_comm=False, **base_config), _local_submodules()
        ).cuda()
        plain.load_state_dict(delayed.state_dict())
        torch.manual_seed(321)
        hidden = torch.randn(4, 2, 8, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        delayed_output, _ = delayed(hidden)
        delayed_output.sum().backward()

        plain_hidden = hidden.detach().clone().requires_grad_(True)
        plain_output, _ = plain(plain_hidden)
        plain_output.sum().backward()

        assert delayed.backward_dw(routed_experts=True, shared_experts=True) is None
        assert plain.backward_dw(routed_experts=True, shared_experts=True) is None
        torch.testing.assert_close(delayed_output, plain_output, rtol=1.6e-2, atol=1e-2)
        torch.testing.assert_close(hidden.grad, plain_hidden.grad, rtol=1.6e-2, atol=1e-2)
        for name, param in delayed.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(
                    param.grad, plain.get_parameter(name).grad, rtol=1.6e-2, atol=1e-2
                )


class TestInferenceForward:

    def test_forward_swaps_dispatchers_and_passes_routing_map_to_experts(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _moe_config(
            transformer_impl='inference_optimized',
            moe_router_dtype='fp32',
            normalization='RMSNorm',
            inference_moe_token_dispatcher_type='nccl',
        )
        layer = MoELayer(config, MoESubmodules(experts=_RoutingAwareExperts)).cuda()
        torch.manual_seed(123)
        hidden = torch.randn(4, 2, 8, device='cuda')

        with InferenceMode.active():
            inference_output, bias = layer(hidden)

        assert bias is None
        assert layer.token_dispatcher is layer._inference_token_dispatcher
        routing_map = layer.token_dispatcher.routing_map
        assert routing_map.shape == (8, 4)
        assert (routing_map.sum(dim=1) == 2).all()
        assert layer.experts.received_routing_map is routing_map
        torch.testing.assert_close(inference_output, hidden, rtol=1e-4, atol=1e-5)

        training_output, bias = layer(hidden)

        assert bias is None
        assert layer.token_dispatcher is layer._training_token_dispatcher
        torch.testing.assert_close(training_output, hidden, rtol=1e-4, atol=1e-5)
