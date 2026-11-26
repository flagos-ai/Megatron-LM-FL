# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small MCore API adapters; no model, router, or parallel-group replacements.

FL exposes the two helpers below. Upstream currently keeps their semantics
inside finalize_model_grads. Keep those differences here, outside the engine.
The actual TP reduction still calls MCore's native finalizer.
"""
from importlib import import_module

import torch.distributed as dist
from torch import nn


def te_fused_weight_parameters(module):
    """Resolve TE Linear and GroupedLinear weights before enabling fused wgrad.

    GroupedLinear exposes weight0..weightN, whereas Linear uses weight or its
    explicit weight_names. Every fused target must acquire an engine-owned
    main_grad view before TE backward; silently enabling an unknown layout
    would defer the failure until backward (or drop an expert gradient).
    """
    names = getattr(module, "weight_names", None)
    if names is None:
        gemms = getattr(module, "num_gemms", None)
        names = tuple(f"weight{i}" for i in range(gemms)) if gemms is not None else ("weight",)
    weights = tuple(getattr(module, name, None) for name in names)
    if not weights or any(not isinstance(weight, nn.Parameter) for weight in weights):
        raise ValueError(f"Unsupported TE fused weight layout: {type(module).__qualname__}")
    return weights


def get_tensor_parallel_gradient_reduction(name, param, config):
    module = import_module("megatron.core.distributed.finalize_model_grads")

    native = getattr(module, "get_tensor_parallel_gradient_reduction", None)
    if native is not None:
        return native(name, param, config)
    if getattr(param, "average_gradients_across_tp_domain", False):
        return "avg"
    if (
        (getattr(config, "sequence_parallel", False) and getattr(param, "sequence_parallel", False))
        or (getattr(config, "qk_layernorm", False) and ("q_layernorm" in name or "k_layernorm" in name))
        or getattr(param, "sum_gradients_across_tp_domain", False)
    ):
        return "sum"
    return None


def get_global_num_tokens(num_tokens, pp_group, dp_cp_group):
    module = import_module("megatron.core.distributed.finalize_model_grads")

    native = getattr(module, "get_global_num_tokens", None)
    if native is not None:
        return native(num_tokens, pp_group, dp_cp_group)
    if isinstance(pp_group, list):
        raise NotImplementedError("Multiple pipeline groups are not supported")
    from megatron.core.pipeline_parallel.utils import get_pp_last_rank

    tokens = num_tokens.detach().clone()
    if pp_group is not None:
        dist.broadcast(tokens, src=get_pp_last_rank(pp_group), group=pp_group)
    if dp_cp_group is not None:
        dist.all_reduce(tokens, group=dp_cp_group)
    return tokens
