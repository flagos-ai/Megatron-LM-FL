# Copyright (c) 2026, BAAI. All rights reserved.
"""TransformerConfig runtime feature checks: CUDA graphs, MoE overlap, inference, MLA."""

import pytest
import torch

from megatron.core.transformer import transformer_config
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

REJECTED = [
    pytest.param(
        dict(cpu_offloading=True, cpu_offloading_num_layers=1, cuda_graph_impl='local'),
        ValueError,
        'CUDA graphs not supported with CPU offloading.',
        id='graph_with_cpu_offloading',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=['moe_preprocess'],
        ),
        AssertionError,
        'moe_preprocess cuda graph is only supported with moe_router cuda graph.',
        id='moe_preprocess_without_moe_router',
    ),
    pytest.param(
        dict(num_moe_experts=4, cuda_graph_impl='local', cuda_graph_modules=['mlp']),
        AssertionError,
        'mlp cuda graph is only supported for dense layers, but not found in the model.',
        id='mlp_graph_with_all_moe',
    ),
    pytest.param(
        dict(num_moe_experts=4, cuda_graph_impl='local', cuda_graph_modules=['moe']),
        AssertionError,
        'moe cuda graph is only supported with drop-padding MoE.',
        id='moe_graph_without_drop_padding',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            cuda_graph_impl='local',
            cuda_graph_modules=['moe_router'],
            moe_token_dispatcher_type='alltoall',
            moe_expert_capacity_factor=1.0,
        ),
        AssertionError,
        'moe_preprocess cuda graph is not supported when there are DtoH copies',
        id='moe_preprocess_graph_with_alltoall_copies',
    ),
    pytest.param(
        dict(
            cuda_graph_impl='local',
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
        ),
        AssertionError,
        'full recompute is only supported with full iteration CUDA graph.',
        id='full_recompute_with_local_graph',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            cuda_graph_impl='transformer_engine',
            recompute_granularity='selective',
            recompute_modules=['moe'],
            cuda_graph_modules=['moe_router'],
        ),
        AssertionError,
        'moe recompute is not supported with moe_router CUDA graph',
        id='moe_recompute_with_router_graph',
    ),
    pytest.param(
        dict(cuda_graph_impl='local', recompute_granularity='selective'),
        AssertionError,
        'attention dropout is not supported with graphed attention recomputation.',
        id='attention_dropout_with_graphed_core_attn',
    ),
    pytest.param(
        dict(
            cuda_graph_impl='local',
            recompute_granularity='selective',
            recompute_modules=['mlp'],
            cuda_graph_modules=['mlp'],
            attention_dropout=0.0,
            hidden_dropout=0.1,
        ),
        AssertionError,
        'hidden dropout is not supported with graphed MLP/MoE recomputation.',
        id='hidden_dropout_with_graphed_mlp',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            cuda_graph_impl='local',
            recompute_granularity='selective',
            recompute_modules=['moe'],
            moe_input_jitter_eps=0.1,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        ),
        AssertionError,
        'moe_input_jitter_eps is not supported with graphed moe recomputation.',
        id='moe_input_jitter_with_graphed_moe',
    ),
    pytest.param(
        dict(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn'],
            cuda_graph_impl='local',
        ),
        AssertionError,
        'fine-grained activation offloading is only supported with transformer_engine',
        id='offloading_with_local_graph',
    ),
    pytest.param(
        dict(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn'],
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=['moe'],
            num_moe_experts=4,
            moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True,
        ),
        AssertionError,
        'Token-drop MoE is temporarily not supported with activation offloading.',
        id='offloading_with_token_drop_moe',
    ),
    pytest.param(
        dict(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn'],
            cuda_graph_impl='transformer_engine',
            cuda_graph_warmup_steps=0,
        ),
        AssertionError,
        'cuda_graph_warmup_steps must be greater than 0 when enabling',
        id='offloading_with_zero_warmup',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_token_dispatcher_type='allgather', variable_seq_lengths=True),
        ValueError,
        'Token dispatcher type: allgather does not support variable sequence length',
        id='allgather_with_variable_sequences',
    ),
    pytest.param(
        dict(
            overlap_moe_expert_parallel_comm=True,
            num_layers=1,
            mtp_num_layers=1,
            pipeline_model_parallel_layout='Et|m|L',
            pipeline_model_parallel_size=3,
            pipeline_dtype=torch.float32,
        ),
        ValueError,
        'overlap_moe_expert_parallel_comm requires MTP to share a pipeline stage',
        id='overlap_with_standalone_mtp_stage',
    ),
    pytest.param(
        dict(
            overlap_moe_expert_parallel_comm=True,
            enable_hyper_connections=True,
            cuda_graph_impl='local',
        ),
        NotImplementedError,
        "mHC EP overlap currently requires cuda_graph_impl='none'",
        id='mhc_overlap_with_cuda_graph',
    ),
    pytest.param(
        dict(
            overlap_moe_expert_parallel_comm=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            bf16=True,
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_token_dispatcher_type='alltoall',
        ),
        AssertionError,
        'If enabling EP A2A overlap, virtual_pipeline_model_parallel_size',
        id='overlap_with_pp_without_vpp',
    ),
    pytest.param(
        dict(
            overlap_moe_expert_parallel_comm=True,
            bf16=True,
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_token_dispatcher_type='alltoall',
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=['moe'],
            moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True,
        ),
        AssertionError,
        'CUDA graph scope on moe and mlp is not supported with overlap_moe_expert_parallel_comm',
        id='overlap_te_graph_with_moe_modules',
    ),
    pytest.param(
        dict(context_parallel_size=2, cp_comm_type=['p2p']),
        AssertionError,
        'Length of cp_comm_type (1) should equal to the total number of transformer layers (2)!',
        id='cp_comm_type_list_length',
    ),
    pytest.param(
        dict(inference_fuse_tp_communication=True),
        AssertionError,
        'inference_fuse_tp_communication is only supported for inference_optimized',
        id='fuse_tp_wrong_impl',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            normalization='RMSNorm',
            add_bias_linear=False,
            num_moe_experts=2,
            moe_router_dtype='fp32',
            inference_fuse_tp_communication=True,
        ),
        AssertionError,
        '--inference-fuse-tp-communication is not supported for MoE models.',
        id='fuse_tp_with_moe',
    ),
    pytest.param(
        dict(inference_disable_triton_nvls_kernels=True),
        AssertionError,
        'inference_disable_triton_nvls_kernels is only supported for inference_optimized',
        id='nvls_wrong_impl',
    ),
]

DERIVED = [
    pytest.param(
        dict(num_moe_experts=4, cuda_graph_impl='local', cuda_graph_modules=['moe_router']),
        dict(cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess]),
        id='moe_router_module_adds_moe_preprocess',
    ),
    pytest.param(
        dict(num_moe_experts=4, cuda_graph_impl='local', cuda_graph_modules=['moe_preprocess']),
        dict(cuda_graph_modules=[CudaGraphModule.moe_preprocess, CudaGraphModule.moe_router]),
        id='moe_preprocess_module_adds_moe_router',
    ),
    pytest.param(
        dict(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn'],
            cuda_graph_impl='full_iteration',
            cuda_graph_warmup_steps=1,
            fine_grained_offloading_max_inflight_offloads=2,
        ),
        dict(cuda_graph_impl='full_iteration'),
        id='offloading_with_full_iteration_graph',
    ),
    pytest.param(
        dict(
            overlap_moe_expert_parallel_comm=True,
            bf16=True,
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_token_dispatcher_type='alltoall',
            cuda_graph_impl='transformer_engine',
        ),
        dict(cuda_graph_impl='transformer_engine'),
        id='overlap_with_te_graph',
    ),
    pytest.param(
        dict(symmetric_ar_type='two_shot'),
        dict(symmetric_ar_type='two_shot'),
        id='symmetric_ar_accepted',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            normalization='RMSNorm',
            add_bias_linear=False,
            inference_fuse_tp_communication=True,
            inference_disable_triton_nvls_kernels=True,
        ),
        dict(transformer_impl='inference_optimized', normalization='RMSNorm'),
        id='inference_optimized_accepted',
    ),
]

OVERLAP = dict(
    overlap_moe_expert_parallel_comm=True,
    bf16=True,
    expert_model_parallel_size=2,
    num_moe_experts=4,
    moe_token_dispatcher_type='alltoall',
    cuda_graph_impl='transformer_engine',
)


def build(**overrides):
    arguments = dict(num_layers=2, hidden_size=12, num_attention_heads=4)
    arguments.update(overrides)
    return TransformerConfig(**arguments)


@pytest.fixture(autouse=True)
def transformer_engine(monkeypatch):
    """Pin Transformer Engine version checks; a test may mark specific versions as missing."""

    def set_available(available):
        monkeypatch.setattr(transformer_config, 'is_te_min_version', lambda version: available)

    monkeypatch.setattr(transformer_config, 'is_torch_min_version', lambda version: True)
    monkeypatch.setattr(transformer_config, 'get_te_version', lambda: 'test-version')
    set_available(True)
    return set_available


@pytest.mark.parametrize(('overrides', 'error', 'message'), REJECTED)
def test_incompatible_combination_is_rejected(overrides, error, message):
    with pytest.raises(error) as caught:
        build(**overrides)
    assert message in str(caught.value)


@pytest.mark.parametrize(('overrides', 'expected'), DERIVED)
def test_compatible_settings_are_normalised(overrides, expected):
    config = build(**overrides)
    assert {name: getattr(config, name) for name in expected} == expected


def test_permute_fusion_requires_the_fused_kernels(monkeypatch):
    for name in (
        'fused_permute',
        'fused_permute_with_probs',
        'fused_sort_chunks_by_index',
        'fused_sort_chunks_by_index_with_probs',
        'fused_unpermute',
    ):
        monkeypatch.setattr(moe_utils, name, None, raising=False)
    with pytest.raises(ValueError) as caught:
        build(moe_permute_fusion=True)
    assert 'fused permutation is not available. Please install TE >= 2.1.0.' in str(caught.value)


def test_delay_wgrad_compute_needs_a_recent_transformer_engine(transformer_engine):
    transformer_engine(False)
    with pytest.raises(AssertionError) as caught:
        build(delay_wgrad_compute=True, **OVERLAP)
    assert 'TE version >= 2.10.0 is required for delay_wgrad_compute' in str(caught.value)


def test_symmetric_ar_type_needs_packaging_installed(monkeypatch):
    monkeypatch.setattr(transformer_config, 'HAVE_PACKAGING', False)
    with pytest.raises(ImportError) as caught:
        build(symmetric_ar_type='two_shot')
    assert 'packaging is not installed. Please install it with `pip install packaging`.' in str(
        caught.value
    )


@pytest.mark.parametrize(
    ('overrides', 'error', 'message'),
    [
        (
            dict(apply_rope_fusion=True, rope_type='rope'),
            ValueError,
            'apply_rope_fusion for MLA only works with YARN RoPE.',
        ),
        (
            dict(attention_output_gate=True),
            NotImplementedError,
            'Output gate is not supported for MLA yet.',
        ),
        (
            dict(apply_rope_fusion=True, cache_mla_latents=True),
            AssertionError,
            'Rope Fusion is not compatible with caching latents',
        ),
    ],
    ids=['rope_fusion_without_yarn', 'output_gate', 'cache_latents_with_rope_fusion'],
)
def test_mla_incompatible_settings_are_rejected(overrides, error, message):
    with pytest.raises(error) as caught:
        MLATransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, **overrides)
    assert message in str(caught.value)


def test_mla_config_is_accepted_with_yarn_rope():
    config = MLATransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4)
    assert config.multi_latent_attention
    assert config.rope_type == 'yarn'
