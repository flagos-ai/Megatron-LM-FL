# Copyright (c) 2026, BAAI. All rights reserved.
"""TransformerConfig argument validation, derived defaults and the indexer-type rule parser."""

import pytest
import torch

from megatron.core.transformer import transformer_config
from megatron.core.transformer.transformer_config import TransformerConfig

REJECTED = [
    pytest.param(
        dict(fp16=True, bf16=True),
        ValueError,
        'Only one of self.fp16: True and self.bf16 True should be True.',
        id='fp16_and_bf16',
    ),
    pytest.param(
        dict(tensor_model_parallel_size=3),
        ValueError,
        'num_attention_heads (4) must be a multiple of tensor_model_parallel_size (3).',
        id='heads_not_multiple_of_tp',
    ),
    pytest.param(
        dict(num_attention_heads=6, num_query_groups=4, tensor_model_parallel_size=6),
        ValueError,
        'num_query_groups (4) must be a multiple or divisor of tensor_model_parallel_size (6).',
        id='query_groups_vs_tp',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=None,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        ),
        AssertionError,
        'linear_attention_freq must be set for linear gated_delta_net.',
        id='gdn_missing_freq',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=None,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        ),
        AssertionError,
        'linear_conv_kernel_dim must be set for gated delta net.',
        id='gdn_missing_conv_kernel',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=None,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        ),
        AssertionError,
        'linear_key_head_dim must be set for gated delta net.',
        id='gdn_missing_key_head_dim',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=2,
            linear_value_head_dim=None,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        ),
        AssertionError,
        'linear_value_head_dim must be set for gated delta net.',
        id='gdn_missing_value_head_dim',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=None,
            linear_num_value_heads=4,
        ),
        AssertionError,
        'linear_num_key_heads must be set for gated delta net.',
        id='gdn_missing_key_heads',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=None,
        ),
        AssertionError,
        'linear_num_value_heads must be set for gated delta net.',
        id='gdn_missing_value_heads',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=3,
        ),
        AssertionError,
        'linear_num_value_heads (3) must be a multiple of linear_num_key_heads (2).',
        id='gdn_value_heads_not_multiple_of_key_heads',
    ),
    pytest.param(
        dict(
            experimental_attention_variant='gated_delta_net',
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            tensor_model_parallel_size=4,
        ),
        AssertionError,
        'self.linear_num_key_heads=2 must be a multiple of (self.tensor_model_parallel_size=4 * self.context_parallel_size=1).',
        id='gdn_key_heads_not_multiple_of_tp',
    ),
    pytest.param(
        dict(experimental_attention_variant='dsa', indexer_types=['full']),
        AssertionError,
        'indexer_types length (1) must equal num_layers (2)',
        id='dsa_indexer_types_length',
    ),
    pytest.param(
        dict(experimental_attention_variant='dsa', indexer_types=['full', 'bogus']),
        AssertionError,
        'indexer_types must only contain ',
        id='dsa_indexer_types_invalid',
    ),
    pytest.param(
        dict(fp8='e4m3', fp8_recipe='delayed', first_last_layers_bf16=True),
        ValueError,
        'Delayed scaling does not support first / last layer in BF16.',
        id='fp8_delayed_with_first_last_bf16',
    ),
    pytest.param(
        dict(
            fp8='e4m3',
            fp8_recipe='tensorwise',
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=3,
        ),
        ValueError,
        'num_layers_at_start_in_bf16 (3) must be between 0 and number of layers per pipeline stage (2).',
        id='fp8_start_bf16_out_of_range',
    ),
    pytest.param(
        dict(
            fp8='e4m3',
            fp8_recipe='tensorwise',
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=-1,
        ),
        ValueError,
        'num_layers_at_end_in_bf16 (-1) must be between 0 and number of layers per pipeline stage (2).',
        id='fp8_end_bf16_out_of_range',
    ),
    pytest.param(
        dict(fp8='e4m3', fp8_recipe='custom'),
        ValueError,
        "fp8_quantizer_factory must be provided when fp8_recipe is 'custom'. Specify a Python import path (e.g., package.module.quantizer_factory) via --fp8-quantizer-factory.",
        id='fp8_custom_without_factory',
    ),
    pytest.param(
        dict(fp8_param=True),
        ValueError,
        'fp8_param must be used together with fp8 mode.',
        id='fp8_param_without_fp8',
    ),
    pytest.param(
        dict(fp8_output_proj=True),
        ValueError,
        'fp8_output_proj must be used together with fp8 mode.',
        id='fp8_output_proj_without_fp8',
    ),
    pytest.param(
        dict(fp8='e4m3', fp8_recipe='tensorwise', fp8_output_proj=True),
        ValueError,
        "fp8_output_proj requires fp8_recipe='mxfp8', got 'tensorwise'.",
        id='fp8_output_proj_wrong_recipe',
    ),
    pytest.param(
        dict(fp4_param=True),
        ValueError,
        'fp4_param must be used together with fp4 mode.',
        id='fp4_param_without_fp4',
    ),
    pytest.param(
        dict(fp4='nvfp4', fp8='e4m3', fp8_recipe='tensorwise'),
        ValueError,
        'fp4 and fp8 cannot be used simultaneously. Please choose one.',
        id='fp4_with_fp8',
    ),
    pytest.param(
        dict(fp4='nvfp4', fp4_recipe='custom'),
        ValueError,
        "fp4_quantizer_factory must be provided when fp4_recipe is 'custom'. Specify a Python import path (e.g., package.module.quantizer_factory) via --fp4-quantizer-factory.",
        id='fp4_custom_without_factory',
    ),
    pytest.param(
        dict(expert_model_parallel_size=2),
        ValueError,
        'num_moe_experts must be non None to use expert-parallel.',
        id='expert_parallel_without_experts',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            expert_tensor_parallel_size=2,
        ),
        ValueError,
        'Inference-optimized MoE layers does not support expert tensor parallelism.',
        id='inference_expert_tensor_parallel',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            moe_expert_capacity_factor=1.0,
        ),
        ValueError,
        'Inference-optimized MoE layers only support dropless MoE ',
        id='inference_capacity_factor',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            moe_router_padding_for_quantization=True,
        ),
        ValueError,
        'Inference-optimized MoE layers do not support padded routing map for quantization.',
        id='inference_padding_for_quantization',
    ),
    pytest.param(
        dict(transformer_impl='inference_optimized', num_moe_experts=2, moe_router_dtype='fp64'),
        ValueError,
        "--transformer-impl='inference_optimized' requires --moe-router-dtype=fp32 to avoid costly dtype conversions during decode.",
        id='inference_router_dtype',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            gated_linear_unit=True,
        ),
        ValueError,
        "--transformer-impl='inference_optimized' does not yet support gated linear units (SwiGLU/GeGLU).",
        id='inference_gated_linear_unit',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            fp8='mxfp8',
        ),
        ValueError,
        "fp8_param must be enabled when using --transformer-impl='inference_optimized' with --fp8-recipe='mxfp8'. Please set --fp8-param-gather.",
        id='inference_mxfp8_without_fp8_param',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            inference_grouped_gemm_backend='bogus',
        ),
        ValueError,
        "inference_grouped_gemm_backend must be 'flashinfer', 'torch', or 'vllm', got 'bogus'",
        id='inference_unknown_gemm_backend',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            fp8='mxfp8',
            fp8_param=True,
            inference_grouped_gemm_backend='flashinfer',
        ),
        ValueError,
        "FlashInfer is not compatible with MXFP8 quantization. Set inference_grouped_gemm_backend to 'torch'.",
        id='inference_flashinfer_with_mxfp8',
    ),
    pytest.param(
        dict(
            transformer_impl='inference_optimized',
            num_moe_experts=2,
            moe_router_dtype='fp32',
            fp8='mxfp8',
            fp8_param=True,
            inference_grouped_gemm_backend='vllm',
        ),
        ValueError,
        "vLLM Triton fused MoE only supports BF16. Set inference_grouped_gemm_backend to 'torch' for MXFP8.",
        id='inference_vllm_with_mxfp8',
    ),
    pytest.param(
        dict(num_moe_experts=0),
        ValueError,
        'num_moe_experts must be non-negative.',
        id='experts_not_positive',
    ),
    pytest.param(
        dict(moe_ffn_hidden_size=8),
        ValueError,
        'moe_ffn_hidden_size is set but num_moe_experts is None. Please set num_moe_experts or remove moe_ffn_hidden_size.',
        id='moe_ffn_size_without_experts',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_single_grouped_bias=True),
        ValueError,
        'moe_single_grouped_weight and moe_single_grouped_bias require moe_grouped_gemm=True.',
        id='single_grouped_without_grouped_gemm',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_grouped_gemm=True, moe_single_grouped_weight=True),
        ValueError,
        "moe_single_grouped_weight is currently supported only with fp8 mode and fp8_recipe='mxfp8'.",
        id='single_grouped_weight_without_mxfp8',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_grouped_gemm=True,
            moe_single_grouped_bias=True,
            add_bias_linear=False,
        ),
        ValueError,
        'moe_single_grouped_bias requires add_bias_linear=True.',
        id='single_grouped_bias_without_linear_bias',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_enable_deepep=True),
        ValueError,
        'DeepEP backend is only supported with flex token dispatcher.',
        id='deepep_without_flex_dispatcher',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_enable_deepep=True,
            moe_token_dispatcher_type='flex',
            moe_flex_dispatcher_backend='hybridep',
        ),
        ValueError,
        'Only one backend is supported for flex token dispatcher.',
        id='deepep_with_hybridep_backend',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_token_dispatcher_type='flex',
            moe_flex_dispatcher_backend='deepep',
            moe_pad_expert_input_to_capacity=True,
        ),
        ValueError,
        'Flex token dispatcher with deepep backend does not support moe_pad_expert_input_to_capacity',
        id='flex_deepep_pad_to_capacity',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_shared_expert_intermediate_size=0),
        ValueError,
        'moe_shared_expert_intermediate_size must be num_shared_experts * ffn_size_of_each_shared_expert, but got 0',
        id='shared_expert_size_not_positive',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_shared_expert_intermediate_size=8,
            moe_shared_expert_overlap=True,
            moe_token_dispatcher_type='allgather',
        ),
        ValueError,
        'moe_shared_expert_overlap only works with alltoall or flex token dispatcher.',
        id='shared_expert_overlap_with_allgather',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_router_load_balancing_type=['aux_loss', 'none'],
            moe_aux_loss_coeff=[0.1],
        ),
        AssertionError,
        'moe_aux_loss_coeff must be a list of the same length as moe_router_load_balancing_type',
        id='aux_loss_coeff_length_mismatch',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_expert_capacity_factor=1.0,
            moe_router_load_balancing_type=['sinkhorn'],
            moe_aux_loss_coeff=[0.0],
        ),
        ValueError,
        'moe_expert_capacity_factor only works with aux_loss, seq_aux_loss, global_aux_loss or none load balancing',
        id='capacity_factor_with_listed_sinkhorn',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_expert_capacity_factor=1.0,
            moe_router_load_balancing_type='sinkhorn',
        ),
        ValueError,
        'moe_expert_capacity_factor only works with aux_loss, seq_aux_loss, global_aux_loss or none load balancing',
        id='capacity_factor_with_sinkhorn',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_pad_expert_input_to_capacity=True),
        ValueError,
        'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity',
        id='pad_to_capacity_without_capacity_factor',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_expert_rank_capacity_factor=1.0),
        ValueError,
        'moe_expert_rank_capacity_factor requires use_transformer_engine_op_fuser to be enabled.',
        id='rank_capacity_without_op_fuser',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_expert_rank_capacity_factor=1.0,
            use_transformer_engine_op_fuser=True,
        ),
        ValueError,
        "moe_expert_rank_capacity_factor requires moe_flex_dispatcher_backend to be 'hybridep'.",
        id='rank_capacity_without_hybridep',
    ),
    pytest.param(
        dict(cpu_offloading=True, cpu_offloading_num_layers=2),
        ValueError,
        'CPU offloading can be done only for layers less than 2',
        id='cpu_offloading_too_many_layers',
    ),
    pytest.param(
        dict(
            cpu_offloading=True,
            cpu_offloading_num_layers=1,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        ),
        ValueError,
        'Currently there is no support for Pipeline parallelism with CPU offloading',
        id='cpu_offloading_with_pipeline_parallel',
    ),
    pytest.param(
        dict(cpu_offloading=True, cpu_offloading_num_layers=1, recompute_granularity='selective'),
        ValueError,
        'CPU offloading does not work when activation recomputation is enabled',
        id='cpu_offloading_with_recompute',
    ),
    pytest.param(
        dict(recompute_granularity='partial'),
        ValueError,
        'When using recompute_granuarlity: partial must be "full"or "selective".',
        id='recompute_granularity_unknown',
    ),
    pytest.param(
        dict(recompute_granularity='full', recompute_method='diagonal'),
        ValueError,
        'recompute_method: diagonal must be "block" or "uniform".',
        id='recompute_method_unknown',
    ),
    pytest.param(
        dict(recompute_granularity='full'),
        ValueError,
        'Using recompute_granularity: full so recompute_method must be "block" or "uniform"',
        id='full_recompute_without_method',
    ),
    pytest.param(
        dict(recompute_granularity='full', recompute_method='uniform'),
        ValueError,
        'When using recompute_granularity: full recompute_num_layers must be between 1 and num_layers_per_pipeline_rank: 2',
        id='full_recompute_without_num_layers',
    ),
    pytest.param(
        dict(recompute_granularity='selective', recompute_num_layers=1),
        ValueError,
        'When using recompute_granularity: selective recompute_num_layers must be None.',
        id='selective_recompute_with_num_layers',
    ),
    pytest.param(
        dict(
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            distribute_saved_activations=True,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        ),
        ValueError,
        'distribute_saved_activations: True must be false when sequence parallel is enabled: True',
        id='distribute_saved_activations_with_sequence_parallel',
    ),
    pytest.param(
        dict(recompute_granularity='selective', recompute_modules=['bogus']),
        AssertionError,
        'Invalid choices for recompute_modules: ',
        id='selective_unknown_module',
    ),
    pytest.param(
        dict(recompute_granularity='selective', recompute_modules=['moe_act']),
        ValueError,
        'moe_act in recompute_modules is only supported with moe_grouped_gemm.',
        id='selective_moe_act_without_grouped_gemm',
    ),
    pytest.param(
        dict(recompute_granularity='selective', recompute_modules=['mla_up_proj']),
        ValueError,
        'mla_up_proj in recompute_modules is only supported with multi_latent_attention.',
        id='selective_mla_up_proj_without_mla',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            num_moe_experts=2,
            recompute_modules=['shared_experts'],
            moe_shared_expert_intermediate_size=8,
            moe_shared_expert_overlap=True,
            moe_token_dispatcher_type='alltoall',
        ),
        ValueError,
        'shared_experts recompute cannot work with --moe-shared-expert-overlap.',
        id='selective_shared_experts_with_overlap',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            fp8='e4m3',
            fp8_recipe='delayed',
            recompute_modules=['layernorm'],
        ),
        ValueError,
        'Delayed scaling does not support moe_act and layernorm recompute for fp8.',
        id='selective_fp8_delayed_layernorm',
    ),
    pytest.param(
        dict(
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            moe_layer_recompute=True,
        ),
        ValueError,
        'Do not set --moe-layer-recompute with full recompute granularity. ',
        id='moe_layer_recompute_with_full_granularity',
    ),
]

DERIVED = [
    pytest.param(
        dict(),
        dict(
            ffn_hidden_size=48, kv_channels=3, num_query_groups=4, recompute_modules=['core_attn']
        ),
        id='defaults',
    ),
    pytest.param(
        dict(fp32_residual_connection=True, pipeline_dtype=torch.bfloat16),
        dict(pipeline_dtype=torch.float32),
        id='fp32_residual_overrides_pipeline_dtype',
    ),
    pytest.param(
        dict(fp32_residual_connection=True, pipeline_dtype=torch.float32),
        dict(pipeline_dtype=torch.float32),
        id='fp32_residual_keeps_float_pipeline_dtype',
    ),
    pytest.param(
        dict(apply_query_key_layer_scaling=True, attention_softmax_in_fp32=False),
        dict(attention_softmax_in_fp32=True),
        id='query_key_scaling_forces_fp32_softmax',
    ),
    pytest.param(
        dict(
            num_layers=4,
            experimental_attention_variant='dsa',
            indexer_type_rule='full+repeat(shared)',
        ),
        dict(indexer_types=['full', 'shared', 'shared', 'shared']),
        id='dsa_rule_expands_to_indexer_types',
    ),
    pytest.param(
        dict(num_moe_experts=2), dict(moe_ffn_hidden_size=48), id='experts_default_moe_ffn_size'
    ),
    pytest.param(
        dict(moe_ffn_hidden_size=8, moe_layer_freq=2),
        dict(moe_ffn_hidden_size=None),
        id='mixed_model_by_frequency_drops_moe_ffn_size',
    ),
    pytest.param(
        dict(moe_ffn_hidden_size=8, moe_layer_freq=[1, 0]),
        dict(moe_ffn_hidden_size=None),
        id='mixed_model_by_layer_list_drops_moe_ffn_size',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_enable_deepep=True, moe_token_dispatcher_type='flex'),
        dict(moe_flex_dispatcher_backend='deepep'),
        id='deepep_flag_selects_deepep_backend',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_expert_capacity_factor=-1.0),
        dict(moe_expert_capacity_factor=None),
        id='negative_capacity_factor_disables_capacity',
    ),
    pytest.param(
        dict(
            num_moe_experts=2,
            moe_expert_capacity_factor=1.0,
            moe_router_load_balancing_type=['aux_loss', 'none'],
            moe_aux_loss_coeff=[0.1, 0.0],
        ),
        dict(moe_expert_capacity_factor=1.0),
        id='capacity_factor_with_listed_aux_losses',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_layer_recompute=True),
        dict(recompute_granularity='selective', recompute_modules=['core_attn', 'moe']),
        id='moe_layer_recompute_adds_moe_module',
    ),
    pytest.param(
        dict(num_moe_experts=2, moe_layer_recompute=True, recompute_modules=['moe']),
        dict(recompute_granularity='selective', recompute_modules=['moe']),
        id='moe_layer_recompute_keeps_listed_moe_module',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            fp8='e4m3',
            fp8_recipe='tensorwise',
            recompute_modules=['layernorm'],
        ),
        dict(recompute_modules=['layernorm']),
        id='selective_fp8_layernorm_recompute',
    ),
    pytest.param(
        dict(recompute_granularity='selective', recompute_modules=[]),
        dict(recompute_modules=[]),
        id='selective_without_modules',
    ),
]

INDEXER_RULES = [
    ('full+repeat(shared)', 4, ['full', 'shared', 'shared', 'shared']),
    (
        '2*full+repeat(shared,full)+shared',
        6,
        ['full', 'full', 'shared', 'full', 'shared', 'shared'],
    ),
    ('repeat(full)+2*shared', 3, ['full', 'shared', 'shared']),
    ('full+shared', 2, ['full', 'shared']),
    (' full ', 1, ['full']),
]

REJECTED_INDEXER_RULES = [
    ('repeat(a)+repeat(b)', 4, 'indexer_type_rule: only one repeat() is allowed'),
    ('full+!bad', 2, "indexer_type_rule: cannot parse segment '!bad'"),
    (
        '3*full+repeat(shared)',
        2,
        'indexer_type_rule: prefix (3) + suffix (0) = 3 exceeds num_layers (2)',
    ),
    (
        'full+shared',
        3,
        'indexer_type_rule: expanded length (2) != num_layers (3). Use repeat() to fill remaining layers.',
    ),
]


def build(**overrides):
    arguments = dict(num_layers=2, hidden_size=12, num_attention_heads=4)
    arguments.update(overrides)
    return TransformerConfig(**arguments)


@pytest.fixture(autouse=True)
def transformer_engine(monkeypatch):
    """Pin the Transformer Engine version checks so laptops and CI take the same branches."""

    def set_available(available):
        monkeypatch.setattr(transformer_config, 'is_te_min_version', lambda version: available)
        monkeypatch.setattr(transformer_config, 'get_te_version', lambda: 'test-version')

    set_available(True)
    return set_available


@pytest.fixture
def bf16_reduction():
    """Start with reduced-precision BF16 reduction allowed and restore the original setting."""
    matmul = torch.backends.cuda.matmul
    original = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = True
    yield matmul
    matmul.allow_bf16_reduced_precision_reduction = original


@pytest.mark.parametrize(('overrides', 'error', 'message'), REJECTED)
def test_invalid_combination_is_rejected(overrides, error, message):
    with pytest.raises(error) as caught:
        build(**overrides)
    assert message in str(caught.value)


def test_single_grouped_parameters_need_a_recent_transformer_engine(transformer_engine):
    transformer_engine(False)
    with pytest.raises(ValueError) as caught:
        build(num_moe_experts=2, moe_grouped_gemm=True, moe_single_grouped_bias=True)
    assert 'require transformer-engine>=2.14.0, but your version is test-version.' in str(
        caught.value
    )


def test_fp8_layernorm_recompute_needs_a_recent_transformer_engine(transformer_engine):
    transformer_engine(False)
    with pytest.raises(ValueError) as caught:
        build(
            recompute_granularity='selective',
            fp8='e4m3',
            fp8_recipe='tensorwise',
            recompute_modules=['layernorm'],
        )
    assert 'needs transformer-engine>=2.6.0dev0, but your version is test-version.' in str(
        caught.value
    )


@pytest.mark.parametrize(('overrides', 'expected'), DERIVED)
def test_settings_are_derived_from_the_arguments(overrides, expected):
    config = build(**overrides)
    assert {name: getattr(config, name) for name in expected} == expected


def test_bf16_can_disable_reduced_precision_matmul(bf16_reduction):
    build(bf16=True, disable_bf16_reduced_precision_matmul=True)
    assert bf16_reduction.allow_bf16_reduced_precision_reduction is False


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        (dict(num_moe_experts=2), 'moe_ffn_hidden_size is not set, using ffn_hidden_size instead.'),
        (
            dict(moe_ffn_hidden_size=8, moe_layer_freq=2),
            'moe_ffn_hidden_size is set but num_moe_experts is None',
        ),
        (
            dict(num_moe_experts=2, moe_enable_deepep=True, moe_token_dispatcher_type='flex'),
            'moe_enable_deepep is deprecated.',
        ),
        (dict(num_moe_experts=2, moe_layer_recompute=True), '--moe-layer-recompute is deprecated.'),
        (dict(recompute_granularity='selective'), 'the core_attn is from transformer_engine'),
    ],
)
def test_deprecated_or_ignored_arguments_warn(overrides, message):
    with pytest.warns(UserWarning) as caught:
        build(**overrides)
    assert any(message in str(warning.message) for warning in caught)


@pytest.mark.parametrize(('rule', 'num_layers', 'expected'), INDEXER_RULES)
def test_indexer_rule_expands_to_one_type_per_layer(rule, num_layers, expected):
    assert TransformerConfig._parse_indexer_type_rule(rule, num_layers) == expected


@pytest.mark.parametrize(('rule', 'num_layers', 'message'), REJECTED_INDEXER_RULES)
def test_malformed_indexer_rule_is_rejected(rule, num_layers, message):
    with pytest.raises(ValueError) as caught:
        TransformerConfig._parse_indexer_type_rule(rule, num_layers)
    assert str(caught.value) == message
