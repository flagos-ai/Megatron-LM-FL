# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig, quick_gelu


def _cfg(**overrides):
    kwargs = {
        "num_layers": 8,
        "hidden_size": 32,
        "num_attention_heads": 4,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_transformer_config_defaults_mutations_and_warning_paths(monkeypatch):
    cfg = _cfg()
    assert cfg.ffn_hidden_size == 128
    assert cfg.kv_channels == 8
    assert cfg.num_query_groups == 4
    assert cfg.recompute_modules == ["core_attn"]

    cfg = _cfg(fp32_residual_connection=True, pipeline_dtype=torch.bfloat16)
    assert cfg.pipeline_dtype is torch.float

    cfg = _cfg(apply_query_key_layer_scaling=True)
    assert cfg.attention_softmax_in_fp32 is True

    with pytest.warns(UserWarning, match="moe_ffn_hidden_size is not set"):
        moe_cfg = _cfg(num_moe_experts=4)
    assert moe_cfg.moe_ffn_hidden_size == moe_cfg.ffn_hidden_size

    with pytest.warns(UserWarning, match="moe_enable_deepep is deprecated"):
        deepep_cfg = _cfg(
            num_moe_experts=4,
            moe_enable_deepep=True,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
        )
    assert deepep_cfg.moe_flex_dispatcher_backend == "deepep"

    with pytest.warns(UserWarning, match="moe-layer-recompute is deprecated"):
        recompute_cfg = _cfg(moe_layer_recompute=True, recompute_modules=[])
    assert recompute_cfg.recompute_granularity == "selective"
    assert "moe" in recompute_cfg.recompute_modules

    with pytest.warns(UserWarning, match="core_attn recompute"):
        selective_cfg = _cfg(recompute_granularity="selective", recompute_modules=["core_attn"])
    assert selective_cfg.recompute_modules == ["core_attn"]

    offload_cfg = _cfg(
        fine_grained_activation_offloading=True,
        offload_modules=["core_attn", "attn_proj"],
    )
    assert offload_cfg.fine_grained_activation_offloading is True

    layout_cfg = _cfg(
        num_layers=3,
        pipeline_model_parallel_size=2,
        pipeline_model_parallel_layout=[["embedding", "decoder"], ["decoder", "decoder", "loss"]],
    )
    assert layout_cfg.pipeline_model_parallel_layout is not None


@pytest.mark.parametrize(
    ("overrides", "exception", "match"),
    [
        ({"fp16": True, "bf16": True}, ValueError, "Only one"),
        ({"num_attention_heads": 3, "tensor_model_parallel_size": 2}, ValueError, "multiple"),
        (
            {"num_query_groups": 3, "tensor_model_parallel_size": 2},
            ValueError,
            "num_query_groups",
        ),
        (
            {"fp8": "e4m3", "first_last_layers_bf16": True, "fp8_recipe": Fp8Recipe.delayed},
            ValueError,
            "Delayed scaling",
        ),
        (
            {
                "fp8": "e4m3",
                "first_last_layers_bf16": True,
                "fp8_recipe": Fp8Recipe.tensorwise,
                "num_layers_at_start_in_bf16": -1,
            },
            ValueError,
            "num_layers_at_start_in_bf16",
        ),
        (
            {
                "fp8": "e4m3",
                "first_last_layers_bf16": True,
                "fp8_recipe": Fp8Recipe.tensorwise,
                "num_layers_at_end_in_bf16": 99,
            },
            ValueError,
            "num_layers_at_end_in_bf16",
        ),
        ({"fp8": "e4m3", "fp8_recipe": Fp8Recipe.custom}, ValueError, "fp8_quantizer_factory"),
        ({"fp8_param": True}, ValueError, "fp8_param"),
        ({"fp4_param": True}, ValueError, "fp4_param"),
        ({"fp4": "e2m1", "fp8": "e4m3"}, ValueError, "fp4 and fp8"),
        ({"fp4": "e2m1", "fp4_recipe": Fp4Recipe.custom}, ValueError, "fp4_quantizer_factory"),
        ({"expert_model_parallel_size": 2}, ValueError, "num_moe_experts"),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "expert_tensor_parallel_size": 2,
            },
            ValueError,
            "expert tensor",
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "moe_expert_capacity_factor": 1.0,
            },
            ValueError,
            "dropless",
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "moe_router_padding_for_quantization": True,
            },
            ValueError,
            "routing map",
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "moe_router_dtype": "fp64",
            },
            ValueError,
            "moe-router-dtype=fp32",
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "moe_router_dtype": "fp32",
                "gated_linear_unit": True,
                "cuda_graph_impl": "local",
            },
            ValueError,
            "gated linear",
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "moe_router_dtype": "fp32",
                "inference_grouped_gemm_backend": "bad",
            },
            AssertionError,
            "inference_grouped_gemm_backend",
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 4,
                "moe_router_dtype": "fp32",
                "cuda_graph_impl": "local",
                "inference_grouped_gemm_backend": "te",
            },
            ValueError,
            "TE GroupedGEMM",
        ),
        ({"num_moe_experts": 0}, ValueError, "non-negative"),
        ({"moe_ffn_hidden_size": 64}, AssertionError, "moe_ffn_hidden_size"),
        (
            {"moe_enable_deepep": True, "moe_token_dispatcher_type": "allgather"},
            ValueError,
            "DeepEP backend",
        ),
        (
            {
                "moe_enable_deepep": True,
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
            },
            ValueError,
            "Only one backend",
        ),
        (
            {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "deepep",
                "moe_pad_expert_input_to_capacity": True,
                "moe_expert_capacity_factor": 1.0,
            },
            ValueError,
            "moe_pad_expert_input_to_capacity",
        ),
        ({"moe_shared_expert_intermediate_size": 0}, ValueError, "moe_shared_expert"),
        (
            {
                "moe_shared_expert_intermediate_size": 16,
                "moe_shared_expert_overlap": True,
                "moe_token_dispatcher_type": "allgather",
            },
            ValueError,
            "alltoall",
        ),
        (
            {"moe_router_load_balancing_type": ["aux_loss", "none"], "moe_aux_loss_coeff": [0.1]},
            AssertionError,
            "same length",
        ),
        (
            {"moe_expert_capacity_factor": 1.0, "moe_router_load_balancing_type": "sinkhorn"},
            ValueError,
            "capacity_factor",
        ),
        (
            {"moe_pad_expert_input_to_capacity": True},
            ValueError,
            "moe_expert_capacity_factor must be set",
        ),
        ({"cpu_offloading": True, "cpu_offloading_num_layers": -1}, ValueError, "CPU offloading"),
        (
            {
                "cpu_offloading": True,
                "cpu_offloading_num_layers": 1,
                "pipeline_model_parallel_size": 2,
            },
            ValueError,
            "Pipeline parallelism",
        ),
        (
            {
                "cpu_offloading": True,
                "cpu_offloading_num_layers": 1,
                "recompute_granularity": "full",
                "recompute_method": "block",
                "recompute_num_layers": 1,
            },
            ValueError,
            "activation recomputation",
        ),
        ({"recompute_granularity": "bad"}, ValueError, "recompute_granuarlity"),
        ({"recompute_granularity": "full", "recompute_method": "bad"}, ValueError, "recompute_method"),
        ({"recompute_granularity": "full"}, ValueError, "recompute_method"),
        (
            {"recompute_granularity": "full", "recompute_method": "block"},
            ValueError,
            "recompute_num_layers",
        ),
        (
            {"recompute_granularity": "selective", "recompute_num_layers": 1},
            ValueError,
            "recompute_num_layers must be None",
        ),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["bad"]},
            AssertionError,
            "Invalid choices",
        ),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["moe_act"]},
            ValueError,
            "moe_grouped_gemm",
        ),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["mla_up_proj"]},
            ValueError,
            "multi_latent_attention",
        ),
        (
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["shared_experts"],
                "moe_shared_expert_intermediate_size": 16,
                "moe_shared_expert_overlap": True,
                "moe_token_dispatcher_type": "alltoall",
            },
            ValueError,
            "shared_experts",
        ),
        (
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["layernorm"],
                "fp8": "e4m3",
                "fp8_recipe": "delayed",
            },
            ValueError,
            "Delayed scaling",
        ),
        (
            {
                "moe_layer_recompute": True,
                "recompute_granularity": "full",
                "recompute_method": "block",
                "recompute_num_layers": 1,
            },
            ValueError,
            "moe-layer-recompute",
        ),
        (
            {
                "fine_grained_activation_offloading": True,
                "cpu_offloading": True,
                "cpu_offloading_num_layers": 1,
                "offload_modules": ["core_attn"],
            },
            AssertionError,
            "fine_grained",
        ),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": []},
            AssertionError,
            "",
        ),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": ["bad"]},
            AssertionError,
            "Invalid choices",
        ),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": ["attn_proj"]},
            ValueError,
            "attn_proj cannot",
        ),
        (
            {
                "num_layers_in_first_pipeline_stage": 1,
                "account_for_embedding_in_pipeline_split": True,
            },
            ValueError,
            "cannot be",
        ),
        (
            {"pipeline_model_parallel_layout": "E|tL", "num_layers_in_first_pipeline_stage": 1},
            ValueError,
            "pipeline_model_parallel_layout cannot be set",
        ),
        (
            {"num_layers_in_first_pipeline_stage": 0, "pipeline_model_parallel_size": 2},
            ValueError,
            "first_pipeline_stage",
        ),
        (
            {"num_layers_in_last_pipeline_stage": 0, "pipeline_model_parallel_size": 2},
            ValueError,
            "last_pipeline_stage",
        ),
        (
            {
                "num_layers": 5,
                "num_layers_in_first_pipeline_stage": 2,
                "pipeline_model_parallel_size": 2,
            },
            ValueError,
            "middle stage",
        ),
        (
            {
                "num_layers": 5,
                "pipeline_model_parallel_size": 2,
                "account_for_embedding_in_pipeline_split": True,
            },
            ValueError,
            "middle layers",
        ),
        (
            {"bias_activation_fusion": True, "activation_func": torch.tanh},
            ValueError,
            "bias_activation_fusion",
        ),
        (
            {"bias_activation_fusion": True, "activation_func": F.gelu, "gated_linear_unit": True},
            ValueError,
            "gated_linear_unit is False",
        ),
        (
            {"bias_activation_fusion": True, "activation_func": quick_gelu},
            ValueError,
            "quick_gelu",
        ),
        (
            {"bias_activation_fusion": True, "activation_func": F.silu, "glu_linear_offset": 1.0},
            ValueError,
            "glu_linear_offset",
        ),
        (
            {"bias_activation_fusion": True, "use_te_activation_func": True},
            ValueError,
            "cannot be both true",
        ),
        (
            {"use_te_activation_func": True, "activation_func": torch.tanh},
            ValueError,
            "use_te_activation_func",
        ),
        (
            {"activation_func_fp8_input_store": True, "activation_func": F.gelu},
            ValueError,
            "activation_func_fp8_input_store",
        ),
    ],
)
def test_transformer_config_validation_guard_matrix(overrides, exception, match):
    with pytest.raises(exception, match=match):
        _cfg(**overrides)


def test_transformer_config_cuda_graph_scope_mutation_paths():
    cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=None)
    assert cfg.cuda_graph_scope == []

    cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=CudaGraphScope.full_iteration)
    assert cfg.cuda_graph_scope == [CudaGraphScope.full_iteration]

    cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope="attn,mlp")
    assert cfg.cuda_graph_scope == [CudaGraphScope.attn, CudaGraphScope.mlp]

    with pytest.raises(AssertionError, match="full"):
        _cfg(cuda_graph_impl="local", cuda_graph_scope="full,attn")
    with pytest.raises(AssertionError, match="Invalid cuda graph implementation"):
        _cfg(cuda_graph_impl="bad", cuda_graph_scope=[])
    with pytest.raises(AssertionError, match="moe_preprocess"):
        _cfg(
            cuda_graph_impl="local",
            cuda_graph_scope=[CudaGraphScope.moe_preprocess],
            num_moe_experts=4,
            moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True,
        )
