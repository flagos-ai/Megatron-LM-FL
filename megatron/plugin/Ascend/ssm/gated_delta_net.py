# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Ascend GDN dispatch; native MCore owns model and sequence handling."""

import torch


def normalize_qk(self, x):
    return x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-6)


def gated_delta_rule(self, query, key, value, g, beta):
    kwargs = {"g": g, "beta": beta}
    if self.config.deterministic_mode:
        return _torch_fallback(query, key, value, **kwargs)

    try:
        from transformer_engine.plugin.core.manager import get_default_manager
    except ImportError:
        return _torch_fallback(query, key, value, **kwargs)

    manager = get_default_manager()
    manager.ensure_initialized()
    if not manager.registry.get_implementations("gated_delta_net_forward"):
        return _torch_fallback(query, key, value, **kwargs)

    result = manager.call(
        "gated_delta_net_forward",
        query=query,
        key=key,
        value=value,
        g=kwargs["g"],
        beta=kwargs["beta"],
        initial_state=kwargs.get("initial_state"),
        output_final_state=kwargs.get("output_final_state", False),
        use_qk_l2norm=kwargs.get("use_qk_l2norm_in_kernel", False),
        chunk_size=64,
    )

    if result is NotImplemented:
        return _torch_fallback(query, key, value, **kwargs)
    return result


def _torch_fallback(query, key, value, **kwargs):
    """Reuse MCore's native math with NPU-safe normalization for dense inputs."""
    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule

    if kwargs.get("use_qk_l2norm_in_kernel", False):
        query = normalize_qk(None, query)
        key = normalize_qk(None, key)
        kwargs["use_qk_l2norm_in_kernel"] = False
    return torch_chunk_gated_delta_rule(query, key, value, **kwargs)
