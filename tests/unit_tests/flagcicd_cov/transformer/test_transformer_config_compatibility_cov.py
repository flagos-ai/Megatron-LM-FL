# Copyright (c) 2026, BAAI. All rights reserved.
"""TransformerConfig feature-compatibility checks: mHC, offloading, pipeline splits, MoE routing."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings import rope_utils
from megatron.core.transformer import transformer_config
from megatron.core.transformer.transformer_config import TransformerConfig, quick_gelu

REJECTED = [
    pytest.param(
        dict(mhc_fused_backend='bogus'),
        ValueError,
        'mhc_fused_backend must be auto, native, triton, or cutile',
        id='mhc_backend_unknown',
    ),
    pytest.param(
        dict(recompute_granularity='selective', recompute_modules=['mhc']),
        ValueError,
        "'mhc' in recompute_modules requires enable_hyper_connections=True.",
        id='mhc_recompute_without_hyper_connections',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            recompute_modules=['mhc', 'mlp'],
            enable_hyper_connections=True,
        ),
        ValueError,
        "'mhc' and 'mlp' in recompute_modules cannot be used together. They use different checkpoint mechanisms that may conflict.",
        id='mhc_recompute_with_mlp',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            recompute_modules=['mhc'],
            enable_hyper_connections=True,
            mhc_recompute_layer_num=0,
        ),
        ValueError,
        "mhc_recompute_layer_num must be a positive integer when 'mhc' is in recompute_modules.",
        id='mhc_recompute_layer_num_zero',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            recompute_modules=['mhc'],
            enable_hyper_connections=True,
            mhc_recompute_layer_num=True,
        ),
        ValueError,
        "mhc_recompute_layer_num must be a positive integer when 'mhc' is in recompute_modules.",
        id='mhc_recompute_layer_num_bool',
    ),
    pytest.param(
        dict(
            recompute_granularity='selective',
            recompute_modules=['mhc'],
            enable_hyper_connections=True,
            fine_grained_activation_offloading=True,
            offload_modules=['attn_norm'],
        ),
        ValueError,
        "'mhc' in recompute_modules is incompatible with offload_modules ",
        id='mhc_recompute_with_norm_offload',
    ),
    pytest.param(
        dict(fine_grained_activation_offloading=True, offload_modules=['attn_proj']),
        ValueError,
        'attn_proj cannot be set to offload_modules alone without core_attn because the input of attn_proj is the output of core_attn, which is needed in core_attn.backward().',
        id='offload_attn_proj_alone',
    ),
    pytest.param(
        dict(
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn'],
            cpu_offloading=True,
            cpu_offloading_num_layers=1,
        ),
        AssertionError,
        'fine_grained_activation_offloading cannot be enabled with cpu_offloading.',
        id='offload_with_cpu_offloading',
    ),
    pytest.param(
        dict(fine_grained_activation_offloading=True, offload_modules=['bogus']),
        AssertionError,
        'Invalid choices for offload_modules: ',
        id='offload_unknown_module',
    ),
    pytest.param(
        dict(moe_paged_stash=True, cpu_offloading=True, cpu_offloading_num_layers=1),
        ValueError,
        'moe_paged_stash cannot be enabled with cpu_offloading.',
        id='paged_stash_with_cpu_offloading',
    ),
    pytest.param(
        dict(moe_paged_stash=True),
        ValueError,
        'moe_paged_stash requires moe_expert_rank_capacity_factor to be set; there is no need to use paged stashing without it.',
        id='paged_stash_without_rank_capacity',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_expert_rank_capacity_factor=1.0,
            use_transformer_engine_op_fuser=True,
            moe_flex_dispatcher_backend='hybridep',
            moe_token_dispatcher_type='flex',
            moe_paged_stash=True,
            offload_modules=['moe_act'],
        ),
        ValueError,
        'When moe_paged_stash is enabled, offload_modules must not include expert_fc1 or moe_act (paged stash covers those activations). Remove: ',
        id='paged_stash_with_moe_offload',
    ),
    pytest.param(
        dict(num_layers_in_first_pipeline_stage=1, account_for_embedding_in_pipeline_split=True),
        ValueError,
        'num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage cannot beset at the same time with account_for_embedding_in_pipeline_splitand account_for_loss_in_pipeline_split',
        id='uneven_pp_with_embedding_accounting',
    ),
    pytest.param(
        dict(pipeline_model_parallel_layout='E|t*2|L', num_layers_in_first_pipeline_stage=1),
        ValueError,
        'pipeline_model_parallel_layout cannot be set with other pipeline layout arguments. self.num_layers_in_first_pipeline_stage=1, self.num_layers_in_last_pipeline_stage=None, self.account_for_embedding_in_pipeline_split=False, self.account_for_loss_in_pipeline_split=False.',
        id='layout_with_uneven_pp',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            num_layers_in_first_pipeline_stage=0,
        ),
        ValueError,
        'num_layers_in_first_pipeline_stage must be larger than 0',
        id='first_stage_not_positive',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            num_layers=4,
            num_layers_in_first_pipeline_stage=1,
            virtual_pipeline_model_parallel_size=2,
        ),
        ValueError,
        'number of layers at first stage: 1must be divisible by virtual pipelineparallel degree 2',
        id='first_stage_not_divisible_by_vpp',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            num_layers_in_last_pipeline_stage=0,
        ),
        ValueError,
        'num_layers_in_last_pipeline_stage must be larger than 0',
        id='last_stage_not_positive',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            num_layers=4,
            num_layers_in_last_pipeline_stage=1,
            virtual_pipeline_model_parallel_size=2,
        ),
        ValueError,
        'number of layers at last stage: 1must be divisible by virtual pipelineparallel degree 2',
        id='last_stage_not_divisible_by_vpp',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            num_layers=4,
            num_layers_in_first_pipeline_stage=1,
            num_layers_in_last_pipeline_stage=1,
        ),
        ValueError,
        'Mismatch: 2 middle layers remaining but 0 middle PP stages available.',
        id='middle_layers_without_middle_stages',
    ),
    pytest.param(
        dict(
            num_layers=6,
            pipeline_model_parallel_size=4,
            pipeline_dtype=torch.float32,
            num_layers_in_first_pipeline_stage=1,
            num_layers_in_last_pipeline_stage=2,
        ),
        ValueError,
        'number of layers at middle stage: 3 must be divisible bythe middle pipeline model parallel size 2',
        id='middle_layers_not_divisible',
    ),
    pytest.param(
        dict(
            num_layers=6,
            pipeline_model_parallel_size=4,
            pipeline_dtype=torch.float32,
            num_layers_in_first_pipeline_stage=2,
            num_layers_in_last_pipeline_stage=2,
            virtual_pipeline_model_parallel_size=2,
        ),
        ValueError,
        'number of layers on each middle pipeline rank:1 must be divisible by virtualpipeline parallel degree 2',
        id='middle_rank_not_divisible_by_vpp',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            account_for_embedding_in_pipeline_split=True,
        ),
        ValueError,
        'number of middle layers: 3 must be divisible by middle pipeline_model_parallel_size 2',
        id='accounted_layers_not_divisible',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            account_for_embedding_in_pipeline_split=True,
            virtual_pipeline_model_parallel_size=2,
        ),
        ValueError,
        'num_layers: 3 after enableaccount_for_embedding_in_pipeline_split or account_for_loss_in_pipeline_split must be divisibleby pipeline_model_parallel_size 2',
        id='accounted_layers_not_divisible_vpp',
    ),
    pytest.param(
        dict(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            account_for_embedding_in_pipeline_split=True,
            account_for_loss_in_pipeline_split=True,
            virtual_pipeline_model_parallel_size=3,
        ),
        ValueError,
        'number of layers on each pipeline rank: 2(after enable account_for_embedding_in_pipeline_split or account_for_loss_in_pipeline_split) must be divisible byvirtual_pipeline_model_parallel_size3',
        id='accounted_rank_not_divisible_by_vpp',
    ),
    pytest.param(
        dict(bias_activation_fusion=True, activation_func=F.relu),
        ValueError,
        'When bias_activation_fusion is True, activation function should be either gelu, swiglu, or quick_geglu',
        id='bias_fusion_unsupported_activation',
    ),
    pytest.param(
        dict(bias_activation_fusion=True, add_bias_linear=False),
        ValueError,
        'When bias_activation_fusion is True, gated_linear_unit is False and activation function is gelu, add_bias_linear must also be True.',
        id='bias_fusion_gelu_without_bias',
    ),
    pytest.param(
        dict(bias_activation_fusion=True, activation_func=quick_gelu),
        ValueError,
        'When bias_activation_fusion is True and activation function is quick_gelu, gated_linear_unit must be True.',
        id='bias_fusion_quick_gelu_without_glu',
    ),
    pytest.param(
        dict(bias_activation_fusion=True, glu_linear_offset=1.0),
        ValueError,
        'When bias_activation_fusion is True and glu_linear_offset is non-zero, activation function must be quick_gelu.',
        id='bias_fusion_offset_without_quick_gelu',
    ),
    pytest.param(
        dict(bias_activation_fusion=True, use_te_activation_func=True),
        ValueError,
        'bias_activation_fusion and use_te_activation_func cannot be both true. If you use bias in MLP FC1, we recommend setting bias_activation_fusion to True and use_te_activation_func to False.',
        id='bias_fusion_with_te_activation',
    ),
    pytest.param(
        dict(fused_residual_rmsnorm=True),
        ValueError,
        'fused_residual_rmsnorm is only supported when normalization is RMSNorm.',
        id='fused_residual_rmsnorm_with_layernorm',
    ),
    pytest.param(
        dict(use_te_activation_func=True, activation_func=torch.tanh),
        ValueError,
        "TransformerEngine only support gelu, geglu, silu, swiglu, relu, reglu. If you don't want to use TransformerEngine activation function, set use_te_activation_func to False",
        id='te_activation_unsupported',
    ),
    pytest.param(
        dict(activation_func_fp8_input_store=True),
        ValueError,
        'Storing activation input in FP8 is supported only for SwiGLU.',
        id='fp8_input_store_without_swiglu',
    ),
    pytest.param(
        dict(activation_func_clamp_value=1.0, activation_func=F.silu, gated_linear_unit=True),
        ValueError,
        'activation_func_clamp_value for SwiGLU is only supported with MoE.',
        id='clamp_swiglu_without_moe',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            activation_func_clamp_value=1.0,
            activation_func=F.silu,
            gated_linear_unit=True,
            use_te_activation_func=True,
        ),
        ValueError,
        'use_te_activation_func must be False when activation_func_clamp_value is not None for SwiGLU',
        id='clamp_swiglu_with_te_activation',
    ),
    pytest.param(
        dict(fused_single_qkv_rope=True, attention_output_gate=True),
        ValueError,
        'fused_single_qkv_rope does not support gated attention for now.',
        id='fused_qkv_rope_with_output_gate',
    ),
    pytest.param(
        dict(multi_latent_attention=True, rotary_interleaved=True),
        ValueError,
        'rotary_interleaved does not work with multi_latent_attention.',
        id='mla_with_rotary_interleaved',
    ),
    pytest.param(
        dict(num_moe_experts=4, expert_tensor_parallel_size=2, add_bias_linear=True),
        AssertionError,
        'Bias in Moe is only supported when ETP==1',
        id='moe_bias_with_etp',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_enable_expert_bias=True),
        ValueError,
        "Expert bias for aux-loss-free routing only supports 'sigmoid' and 'sqrtsoftplus' score functions. Please set --moe-router-score-function to 'sigmoid' or 'sqrtsoftplus', or unset --moe-router-enable-expert-bias.",
        id='expert_bias_score_function',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_n_hash_layers=1),
        AssertionError,
        'actual_vocab_size must be set when moe_n_hash_layers > 0.',
        id='hash_layers_without_vocab',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_n_hash_layers=1,
            actual_vocab_size=8,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        ),
        AssertionError,
        'pipeline_model_parallel_layout must be set when using hash MoE layers with pipeline parallelism (PP > 1).',
        id='hash_layers_pp_without_layout',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_n_hash_layers=2,
            actual_vocab_size=8,
            pipeline_model_parallel_layout='Et|tL',
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        ),
        AssertionError,
        'Currently, All hash MoE layers must be in the same virtual pipeline stage as the embedding. The embedding stage has 1 decoder layers, but moe_n_hash_layers=2.',
        id='hash_layers_outside_embedding_stage',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_n_hash_layers=1,
            actual_vocab_size=8,
            overlap_moe_expert_parallel_comm=True,
        ),
        AssertionError,
        'overlap_moe_expert_parallel_comm does not support moe_n_hash_layers > 0 for now.',
        id='hash_layers_with_overlap',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_padding_for_quantization=True),
        ValueError,
        'fp8/fp4 must be specified when moe_router_padding_for_quantization is True.',
        id='padding_without_quantization',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_padding_for_fp8=True),
        ValueError,
        'fp8/fp4 must be specified when moe_router_padding_for_quantization is True.',
        id='padding_for_fp8_without_quantization',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_router_padding_for_quantization=True,
            fp8='e4m3',
            fp8_recipe='tensorwise',
        ),
        ValueError,
        'allgather and alltoall_seq dispatcher does not support moe_router_padding_for_quantization.',
        id='padding_with_allgather',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_topk=1),
        ValueError,
        'Please use --moe-router-pre-softmax when topk is 1.',
        id='topk_one_without_pre_softmax',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_group_topk=1, moe_router_topk_limited_devices=1),
        ValueError,
        'moe_router_topk_limited_devices is deprecated and replaced by moe_router_group_topk and moe_router_num_groups.',
        id='group_topk_with_limited_devices',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_group_topk=1),
        ValueError,
        'When using group limited routing, moe_router_num_groups must be specified.',
        id='group_topk_without_groups',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_group_topk=1, moe_router_num_groups=3),
        AssertionError,
        'num_moe_experts (4) should be divisible by moe_router_num_groups (3).',
        id='experts_not_divisible_by_groups',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_group_topk=3, moe_router_num_groups=2),
        AssertionError,
        'moe_router_group_topk (3) should be smaller than moe_router_num_groups (2).',
        id='group_topk_above_groups',
    ),
    pytest.param(
        dict(enable_cuda_graph=True, cuda_graph_impl='local'),
        AssertionError,
        'Do not use enable_cuda_graph or external_cuda_graph with cuda_graph_impl.',
        id='deprecated_graph_flag_with_impl',
    ),
    pytest.param(
        dict(enable_cuda_graph=True, external_cuda_graph=True),
        AssertionError,
        'enable_cuda_graph and external_cuda_graph cannot be enabled at the same time.',
        id='both_deprecated_graph_flags',
    ),
    pytest.param(
        dict(cuda_graph_scope='attn', cuda_graph_modules=['attn']),
        AssertionError,
        'cuda_graph_scope and cuda_graph_modules cannot be set together.',
        id='scope_with_modules',
    ),
]

DERIVED = [
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_n_hash_layers=2,
            actual_vocab_size=8,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            pipeline_model_parallel_layout='Ett|L',
        ),
        dict(virtual_pipeline_model_parallel_size=None, moe_n_hash_layers=2),
        id='hash_layers_inside_embedding_stage',
    ),
    pytest.param(
        dict(
            num_moe_experts=4,
            moe_router_padding_for_fp8=True,
            fp8='e4m3',
            fp8_recipe='tensorwise',
            moe_token_dispatcher_type='alltoall',
        ),
        dict(moe_router_padding_for_quantization=True),
        id='padding_for_fp8_enables_quantization_padding',
    ),
    pytest.param(
        dict(num_moe_experts=4, moe_router_topk_limited_devices=1),
        dict(moe_router_group_topk=1, moe_router_num_groups=1),
        id='limited_devices_maps_to_group_routing',
    ),
    pytest.param(
        dict(enable_cuda_graph=True),
        dict(cuda_graph_impl='local'),
        id='enable_cuda_graph_selects_local_impl',
    ),
    pytest.param(
        dict(use_mup=True), dict(mup_base_hidden_size=12), id='mup_defaults_base_hidden_size'
    ),
    pytest.param(
        dict(fine_grained_activation_offloading=True, offload_modules=['core_attn', 'attn_proj']),
        dict(offload_modules=['core_attn', 'attn_proj']),
        id='valid_offload_modules_are_kept',
    ),
]


def build(**overrides):
    arguments = dict(num_layers=2, hidden_size=12, num_attention_heads=4)
    arguments.update(overrides)
    return TransformerConfig(**arguments)


@pytest.fixture(autouse=True)
def transformer_engine(monkeypatch):
    """Pin Transformer Engine version checks; a test may mark specific versions as missing."""

    def set_missing(*versions, everything=False):
        monkeypatch.setattr(
            transformer_config,
            'is_te_min_version',
            lambda version: not everything and version not in versions,
        )
        monkeypatch.setattr(transformer_config, 'get_te_version', lambda: 'test-version')

    set_missing()
    return set_missing


@pytest.fixture
def platform(monkeypatch):
    """Report a chosen platform name to the mHC backend check."""

    def select(name):
        monkeypatch.setattr(
            transformer_config, 'get_platform', lambda: SimpleNamespace(platform_name=lambda: name)
        )

    return select


@pytest.mark.parametrize(('overrides', 'error', 'message'), REJECTED)
def test_incompatible_combination_is_rejected(overrides, error, message):
    with pytest.raises(error) as caught:
        build(**overrides)
    assert message in str(caught.value)


@pytest.mark.parametrize(('overrides', 'expected'), DERIVED)
def test_compatible_settings_are_normalised(overrides, expected):
    config = build(**overrides)
    assert {name: getattr(config, name) for name in expected} == expected


@pytest.mark.parametrize(
    ('missing', 'overrides', 'message'),
    [
        (
            {'everything': True},
            dict(apply_rope_fusion=True, rotary_interleaved=True),
            'rotary_interleaved does not work with apply_rope_fusion for TE < 2.3.0',
        ),
        (
            {'everything': True},
            dict(num_moe_experts=4, fp8='e4m3', fp8_recipe='tensorwise'),
            'Only transformer-engine>=1.7.0 supports MoE FP8 training, but your version is test-version.',
        ),
        (
            {'versions': ('1.11.0',)},
            dict(num_moe_experts=4, fp8='e4m3', fp8_recipe='tensorwise', moe_grouped_gemm=True),
            'Only transformer-engine>=1.11.0 supports FP8 grouped gemm, but your version is test-version.',
        ),
    ],
)
def test_old_transformer_engine_is_rejected(transformer_engine, missing, overrides, message):
    transformer_engine(*missing.get('versions', ()), everything=missing.get('everything', False))
    with pytest.raises(ValueError) as caught:
        build(**overrides)
    assert message in str(caught.value)


def test_rope_fusion_requires_fused_kernels(monkeypatch):
    monkeypatch.setattr(rope_utils, 'fused_apply_rotary_pos_emb', None)
    monkeypatch.setattr(rope_utils, 'fused_apply_rotary_pos_emb_thd', None)
    with pytest.raises(
        ValueError, match='apply_rope_fusion is not available. Please install TE >= 1.4.'
    ):
        build(apply_rope_fusion=True)


@pytest.mark.parametrize(
    ('name', 'backend', 'maca'), [('kunlunxin', 'auto', None), ('cuda', 'triton', '1.0')]
)
def test_triton_mhc_is_rejected_where_it_cannot_compile(platform, monkeypatch, name, backend, maca):
    platform(name)
    if maca is not None:
        monkeypatch.setattr(torch.version, 'maca', maca, raising=False)
    with pytest.raises(ValueError) as caught:
        build(use_fused_mhc=True, mhc_fused_backend=backend)
    expected = 'kunlunxin' if maca is None else 'metax'
    assert f'Triton fused mHC kernels are incompatible with {expected}' in str(caught.value)


def test_triton_mhc_warns_about_slow_compilation(platform):
    platform('npu')
    with pytest.warns(UserWarning, match='known compilation timeouts on npu'):
        build(use_fused_mhc=True)


def test_native_mhc_backend_is_accepted_everywhere(platform, monkeypatch):
    monkeypatch.delattr(torch.version, 'maca', raising=False)
    platform('kunlunxin')
    config = build(use_fused_mhc=True, mhc_fused_backend='native')
    assert config.mhc_fused_backend == 'native'
    platform('cuda')
    config = build(use_fused_mhc=True)
    assert config.mhc_fused_backend == 'auto'


@pytest.mark.parametrize(
    ('overrides', 'category', 'message'),
    [
        (
            dict(enable_hyper_connections=True),
            UserWarning,
            "HyperConnections are enabled but 'mhc' is not in recompute_modules",
        ),
        (dict(external_cuda_graph=True), UserWarning, 'external_cuda_graph is deprecated'),
        (
            dict(apply_rope_fusion=True, multi_latent_attention=True),
            UserWarning,
            'apply_rope_fusion for multi-latent attention only supports training.',
        ),
        (
            dict(cuda_graph_scope='attn'),
            DeprecationWarning,
            'cuda_graph_scope is deprecated, use cuda_graph_modules instead.',
        ),
    ],
)
def test_deprecated_or_advisory_settings_warn(overrides, category, message):
    with pytest.warns(category) as caught:
        build(**overrides)
    assert any(message in str(warning.message) for warning in caught)


def test_external_cuda_graph_selects_the_transformer_engine_impl():
    with pytest.warns(UserWarning):
        config = build(external_cuda_graph=True)
    assert config.cuda_graph_impl == 'transformer_engine'


@pytest.mark.parametrize(
    ('scope', 'expected'), [('attn', ['attn']), (['attn', 'mlp'], ['attn', 'mlp'])]
)
def test_cuda_graph_scope_is_translated_to_modules(scope, expected):
    with pytest.warns(DeprecationWarning):
        config = build(cuda_graph_scope=scope)
    assert config.cuda_graph_scope is None
    assert [module.name for module in config.cuda_graph_modules] == expected
