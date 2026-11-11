# Copyright (c) 2026, BAAI. All rights reserved.
"""InferenceGroupedMLP: mcore activation-type resolution, torch-backend grouped-GEMM forwards
against a manual reference and concatenated weights aliasing expert parameters."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import InferenceGroupedMLP
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


def _grouped_reference(experts, index, tokens, probs, activation=None, deinterleave=None):
    """Plain-torch reference for one grouped expert on the given token slice."""
    fc1_weight = getattr(experts.linear_fc1, f'weight{index}').float()
    fc2_weight = getattr(experts.linear_fc2, f'weight{index}').float()
    intermediate = tokens.float() @ fc1_weight.t()
    if deinterleave is not None:
        width = intermediate.shape[-1]
        intermediate = (
            intermediate.reshape(-1, width // (2 * deinterleave), 2, deinterleave)
            .transpose(1, 2)
            .contiguous()
            .view(-1, width)
        )
    if experts.config.gated_linear_unit:
        gate, linear = torch.chunk(intermediate, 2, dim=-1)
        if experts.config.activation_func_clamp_value is not None:
            value = experts.config.activation_func_clamp_value
            gate = gate.clamp(max=value)
            linear = linear.clamp(min=-value, max=value)
        activated = experts.config.activation_func(gate) * (
            linear + experts.config.glu_linear_offset
        )
    else:
        activated = activation(intermediate)
    return (activated * probs.unsqueeze(-1).float()) @ fc2_weight.t()


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestInferenceGroupedMLP:
    """Inference-optimized grouped experts on the A100 CI node."""

    def _inference_experts(self, activation_func):
        from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

        config = _experts_config(
            activation_func=activation_func,
            gated_linear_unit=False,
            moe_grouped_gemm=True,
            transformer_impl='inference_optimized',
            inference_grouped_gemm_backend='torch',
            inference_moe_token_dispatcher_type='nccl',
            moe_router_dtype='fp32',
            normalization='RMSNorm',
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        layer = get_inference_optimized_moe_spec()(config=config)
        layer = layer.cuda()
        return layer, layer.experts

    def test_constructor_resolves_mcore_activation_type_and_backend(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        from megatron.core.inference.moe import InferenceGroupedGemmBackend, McoreActivationType

        _, experts = self._inference_experts(squared_relu)

        assert isinstance(experts, InferenceGroupedMLP)
        assert experts._mcore_activation_type is McoreActivationType.SQUARED_RELU
        assert experts.inference_grouped_gemm_backend is InferenceGroupedGemmBackend.TORCH
        assert experts._nvls_dispatcher is False
        assert experts._concatenated_weights_built is False

    def test_constructor_rejects_activation_without_mcore_mapping(self, model_parallel):
        model_parallel()

        with pytest.raises(
            ValueError, match='No mcore_fused_moe ActivationType mapping for activation_func=silu'
        ):
            self._inference_experts(F.silu)

    def test_training_forward_delegates_to_grouped_mlp_and_matches_manual_computation(
        self, model_parallel
    ):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        layer, experts = self._inference_experts(squared_relu)
        with torch.no_grad():
            for index in range(2):
                fc1_weight = getattr(experts.linear_fc1, f'weight{index}')
                fc1_weight.copy_(
                    (
                        torch.arange(fc1_weight.numel(), dtype=torch.float32).reshape(
                            fc1_weight.shape
                        )
                        / 64.0
                    ).to(fc1_weight.dtype)
                )
                fc2_weight = getattr(experts.linear_fc2, f'weight{index}')
                fc2_weight.copy_(
                    (
                        torch.arange(fc2_weight.numel(), dtype=torch.float32).reshape(
                            fc2_weight.shape
                        )
                        / 128.0
                    ).to(fc2_weight.dtype)
                )
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        expected = torch.cat(
            [
                _grouped_reference(
                    experts, 0, hidden[:3], probs[:3], activation=lambda x: torch.relu(x).square()
                ),
                _grouped_reference(
                    experts, 1, hidden[3:], probs[3:], activation=lambda x: torch.relu(x).square()
                ),
            ]
        ).to(output.dtype)
        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)

    def test_builds_concatenated_weights_aliasing_expert_parameters(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        _, experts = self._inference_experts(squared_relu)
        with torch.no_grad():
            for index in range(2):
                fc1_weight = getattr(experts.linear_fc1, f'weight{index}')
                fc1_weight.copy_(
                    (
                        torch.arange(fc1_weight.numel(), dtype=torch.float32).reshape(
                            fc1_weight.shape
                        )
                        / 64.0
                        + index
                    ).to(fc1_weight.dtype)
                )
                fc2_weight = getattr(experts.linear_fc2, f'weight{index}')
                fc2_weight.copy_(
                    (
                        torch.arange(fc2_weight.numel(), dtype=torch.float32).reshape(
                            fc2_weight.shape
                        )
                        / 128.0
                        + index
                    ).to(fc2_weight.dtype)
                )

        experts._build_concatenated_weights()

        torch.testing.assert_close(
            experts._fc1_weight[0].float(),
            getattr(experts.linear_fc1, 'weight0').float(),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            experts._fc2_weight[1].float(),
            getattr(experts.linear_fc2, 'weight1').float(),
            rtol=1e-6,
            atol=1e-6,
        )
        assert getattr(experts.linear_fc1, 'weight0').data_ptr() == (
            experts._fc1_weight[0].data_ptr()
        )
        assert getattr(experts.linear_fc2, 'weight1').data_ptr() == (
            experts._fc2_weight[1].data_ptr()
        )
        assert experts._fc1_weight.shape == (2, *getattr(experts.linear_fc1, 'weight0').shape)
