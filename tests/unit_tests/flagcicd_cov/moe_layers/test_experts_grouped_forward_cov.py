# Copyright (c) 2026, BAAI. All rights reserved.
"""TEGroupedMLP forwards: per-expert token slices with weighted silu-glu, interleaved GLU halves
with clamping, fused bias-activation paths, TE activation modules and the fused op-fuser impl."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.extensions.transformer_engine import (
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


def _grouped_experts(config):
    """TEGroupedMLP on the current CUDA device with deterministic silu-glu weights."""
    experts = TEGroupedMLP(
        num_local_experts=config.num_moe_experts,
        config=config,
        submodules=_grouped_submodules(),
        pg_collection=get_default_pg_collection(),
    ).cuda()
    with torch.no_grad():
        for index in range(config.num_moe_experts):
            fc1_weight = getattr(experts.linear_fc1, f'weight{index}')
            fc1_weight.copy_(
                (
                    torch.arange(fc1_weight.numel(), dtype=torch.float32).reshape(fc1_weight.shape)
                    / 64.0
                    + 0.25 * index
                ).to(fc1_weight.dtype)
            )
            fc2_weight = getattr(experts.linear_fc2, f'weight{index}')
            fc2_weight.copy_(
                (
                    torch.arange(fc2_weight.numel(), dtype=torch.float32).reshape(fc2_weight.shape)
                    / 128.0
                    + 0.5 * index
                ).to(fc2_weight.dtype)
            )
    return experts


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


class TestTEGroupedMLPForward:
    """Grouped-GEMM expert forwards on the A100 CI node with real Transformer Engine."""

    def test_forward_runs_each_expert_on_its_token_slice_with_weighted_silu_glu(
        self, model_parallel
    ):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        experts = _grouped_experts(_experts_config(bf16=True, params_dtype=torch.bfloat16))
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        expected = torch.cat(
            [
                _grouped_reference(experts, 0, hidden[:3], probs[:3]),
                _grouped_reference(experts, 1, hidden[3:], probs[3:]),
            ]
        ).to(output.dtype)
        assert output_bias is None
        assert output.shape == (5, 8)
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)

    def test_grouped_forward_applies_probs_on_input_and_matches_manual_computation(
        self, model_parallel
    ):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(moe_apply_probs_on_input=True, bf16=True)
        config.params_dtype = torch.bfloat16
        experts = _grouped_experts(config)
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        scaled = (hidden.float() * probs.unsqueeze(-1).float()).to(hidden.dtype)
        expected = torch.cat(
            [
                _grouped_reference(experts, 0, scaled[:3], torch.ones(3, device='cuda')),
                _grouped_reference(experts, 1, scaled[3:], torch.ones(2, device='cuda')),
            ]
        ).to(output.dtype)
        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)

    def test_forward_clamps_interleaved_glu_halves_when_clamp_value_is_set(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(
            moe_mlp_glu_interleave_size=2,
            activation_func_clamp_value=0.5,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        experts = _grouped_experts(config)
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        expected = torch.cat(
            [
                _grouped_reference(
                    experts,
                    0,
                    hidden[:3],
                    probs[:3],
                    deinterleave=config.moe_mlp_glu_interleave_size,
                ),
                _grouped_reference(
                    experts,
                    1,
                    hidden[3:],
                    probs[3:],
                    deinterleave=config.moe_mlp_glu_interleave_size,
                ),
            ]
        ).to(output.dtype)
        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)

    def test_forward_matches_unfused_layer_when_bias_act_fusion_is_enabled(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        from megatron.core.fusions.fused_bias_geglu import quick_gelu

        fused_config = _experts_config(
            activation_func=quick_gelu,
            bias_activation_fusion=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        plain_config = _experts_config(
            activation_func=quick_gelu, bf16=True, params_dtype=torch.bfloat16
        )
        fused = _grouped_experts(fused_config)
        plain = _grouped_experts(plain_config)
        with torch.no_grad():
            for index in range(2):
                getattr(fused.linear_fc1, f'weight{index}').copy_(
                    getattr(plain.linear_fc1, f'weight{index}')
                )
                getattr(fused.linear_fc2, f'weight{index}').copy_(
                    getattr(plain.linear_fc2, f'weight{index}')
                )
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        fused_output, _ = fused(hidden, tokens_per_expert, probs)
        plain_output, _ = plain(hidden, tokens_per_expert, probs)

        torch.testing.assert_close(fused_output, plain_output, rtol=1.6e-2, atol=1e-2)

    def test_forward_matches_unfused_layer_for_fused_weighted_squared_relu(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        fused_config = _experts_config(
            activation_func=squared_relu,
            gated_linear_unit=False,
            use_fused_weighted_squared_relu=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        plain_config = _experts_config(
            activation_func=squared_relu,
            gated_linear_unit=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        fused = _grouped_experts(fused_config)
        plain = _grouped_experts(plain_config)
        with torch.no_grad():
            for index in range(2):
                getattr(fused.linear_fc1, f'weight{index}').copy_(
                    getattr(plain.linear_fc1, f'weight{index}')
                )
                getattr(fused.linear_fc2, f'weight{index}').copy_(
                    getattr(plain.linear_fc2, f'weight{index}')
                )
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        fused_output, _ = fused(hidden, tokens_per_expert, probs)
        plain_output, _ = plain(hidden, tokens_per_expert, probs)

        torch.testing.assert_close(fused_output, plain_output, rtol=1.6e-2, atol=1e-2)
        expected = torch.cat(
            [
                _grouped_reference(
                    plain, 0, hidden[:3], probs[:3], activation=lambda x: torch.relu(x).square()
                ),
                _grouped_reference(
                    plain, 1, hidden[3:], probs[3:], activation=lambda x: torch.relu(x).square()
                ),
            ]
        ).to(fused_output.dtype)
        torch.testing.assert_close(plain_output, expected, rtol=1.6e-2, atol=1e-2)

    def test_forward_uses_te_activation_module_and_prob_multiply(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(
            use_te_activation_func=True, bf16=True, params_dtype=torch.bfloat16
        )
        experts = _grouped_experts(config)
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        expected = torch.cat(
            [
                _grouped_reference(experts, 0, hidden[:3], probs[:3]),
                _grouped_reference(experts, 1, hidden[3:], probs[3:]),
            ]
        ).to(output.dtype)
        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)

    def test_forward_deinterleaves_before_the_te_activation_module(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(
            use_te_activation_func=True,
            moe_mlp_glu_interleave_size=2,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        experts = _grouped_experts(config)
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = experts(hidden, tokens_per_expert, probs)

        expected = torch.cat(
            [
                _grouped_reference(
                    experts,
                    0,
                    hidden[:3],
                    probs[:3],
                    deinterleave=config.moe_mlp_glu_interleave_size,
                ),
                _grouped_reference(
                    experts,
                    1,
                    hidden[3:],
                    probs[3:],
                    deinterleave=config.moe_mlp_glu_interleave_size,
                ),
            ]
        ).to(output.dtype)
        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)

    def test_forward_rejects_unsupported_bias_activation_fusion(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(
            activation_func=F.gelu,
            bias_activation_fusion=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        experts = _grouped_experts(config)
        hidden = (torch.arange(8, device='cuda').reshape(1, 8) / 8.0).to(torch.bfloat16)
        probs = torch.ones(1, device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([1, 0], device='cuda', dtype=torch.long)

        with pytest.raises(
            ValueError, match='Only support fusion of swiglu and quick_gelu in TEGroupedMLP.'
        ):
            experts(hidden, tokens_per_expert, probs)


class TestTEGroupedMLPFusedImpl:
    """Fused TE op-fuser grouped MLP forwards on the A100 CI node."""

    def _run_fused_forward(self, experts, hidden, probs, tokens_per_expert):
        return experts(hidden, tokens_per_expert, probs)

    def test_fused_forward_matches_manual_expert_computation(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        config = _experts_config(
            use_transformer_engine_op_fuser=True, bf16=True, params_dtype=torch.bfloat16
        )
        experts = _grouped_experts(config)
        hidden = (torch.arange(40, device='cuda').reshape(5, 8) / 8.0).to(torch.bfloat16)
        probs = torch.tensor([0.5, 0.25, 0.75, 1.0, 0.5], device='cuda', dtype=torch.bfloat16)
        tokens_per_expert = torch.tensor([3, 2], device='cuda', dtype=torch.long)

        output, output_bias = self._run_fused_forward(experts, hidden, probs, tokens_per_expert)

        expected = torch.cat(
            [
                _grouped_reference(experts, 0, hidden[:3], probs[:3]),
                _grouped_reference(experts, 1, hidden[3:], probs[3:]),
            ]
        ).to(output.dtype)
        assert output_bias is None
        assert experts._with_fused_impl is True
        assert len(experts._fused_ops[0]) == 3
        torch.testing.assert_close(output, expected, rtol=1.6e-2, atol=1e-2)
