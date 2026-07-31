# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace metadata helpers for routed MoE phases."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from megatron.core import utils

ROUTER_WORKLOAD_SLOTS = (
    "aux_loss",
    "z_loss",
    "num_tokens",
    "routed_tokens",
    "dropped_tokens",
    "drop_rate",
    "expert_cv",
    "top1_expert_share",
    "routing_entropy",
)

EXPERT_WORKLOAD_SLOTS = (
    "routed_tokens",
    "expert_cv",
    "top1_expert_share",
    "expert_max_over_mean",
    "tokens_per_expert",
)


def _ep_size(owner: Any) -> int:
    return int(utils.get_pg_size(owner.ep_group))


def router_trace_context(router: Any) -> dict[str, Any]:
    """Build source-compatible topology metadata for one router invocation."""
    ep_size = _ep_size(router)
    num_experts = int(router.config.num_moe_experts)
    return {
        "layer": router.layer_number,
        "num_experts": num_experts,
        "num_local_experts": num_experts // ep_size,
        "ep_size": ep_size,
        "router_topk": int(router.topk),
    }


def _moe_layer_trace_context(layer: Any) -> dict[str, Any]:
    return {
        "layer": layer.layer_number,
        "ep_size": _ep_size(layer),
        "num_experts": int(layer.config.num_moe_experts),
        "num_local_experts": int(layer.num_local_experts),
    }


def dispatch_trace_context(layer: Any, hidden_states: torch.Tensor) -> dict[str, Any]:
    """Build topology and input-workload metadata for token dispatch."""
    context = _moe_layer_trace_context(layer)
    context.update(
        {
            "router_topk": int(layer.config.moe_router_topk),
            "dispatcher": layer.config.moe_token_dispatcher_type,
            "num_tokens": int(hidden_states.shape[0]),
            "capacity_factor": layer.config.moe_expert_capacity_factor,
        }
    )
    return context


def experts_trace_context(layer: Any) -> dict[str, Any]:
    """Build topology metadata for routed local-expert compute."""
    return _moe_layer_trace_context(layer)


def combine_trace_context(layer: Any, output: torch.Tensor) -> dict[str, Any]:
    """Build topology and input-workload metadata for token combine."""
    context = _moe_layer_trace_context(layer)
    context.update(
        {"dispatcher": layer.config.moe_token_dispatcher_type, "num_tokens": int(output.shape[0])}
    )
    return context


@torch.no_grad()
def router_workload(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    capacity_factor: float | None,
    pad_to_capacity: bool,
) -> dict[str, int | float | None]:
    """Summarize post-capacity token-to-expert assignments for one router call."""
    num_tokens = int(routing_map.shape[0])
    padded_execution_map = capacity_factor is not None and pad_to_capacity

    if padded_execution_map:
        routed_tokens = None
        dropped_tokens = None
        drop_rate = None
        expert_cv = None
        top1_expert_share = None
    else:
        raw_tokens_per_expert = routing_map.detach().sum(dim=0)
        tokens_per_expert = raw_tokens_per_expert.float()
        routed_tokens = int(raw_tokens_per_expert.sum().item())
        if routed_tokens > 0 and tokens_per_expert.numel() > 0:
            mean_tokens = routed_tokens / tokens_per_expert.numel()
            std_tokens = float(tokens_per_expert.std(unbiased=False).item())
            expert_cv = std_tokens / mean_tokens if mean_tokens > 0 else 0.0
            top1_expert_share = int(raw_tokens_per_expert.max().item()) / routed_tokens
        else:
            expert_cv = 0.0
            top1_expert_share = 0.0

        dropped_tokens = 0 if capacity_factor is None else None
        drop_rate = 0.0 if capacity_factor is None else None

    routing_entropy = 0.0
    detached_probs = probs.detach().float()
    if detached_probs.numel() > 0:
        routing_entropy = float(
            (-(detached_probs.clamp_min(1e-12).log() * detached_probs).sum(dim=1).mean()).item()
        )

    return {
        "num_tokens": num_tokens,
        "routed_tokens": routed_tokens,
        "dropped_tokens": dropped_tokens,
        "drop_rate": drop_rate,
        "expert_cv": expert_cv,
        "top1_expert_share": top1_expert_share,
        "routing_entropy": routing_entropy,
    }


@torch.no_grad()
def expert_workload(tokens_per_expert: torch.Tensor | None) -> dict[str, Any]:
    """Summarize the local expert assignment distribution after dispatch."""
    if tokens_per_expert is None:
        return {}

    raw_counts = tokens_per_expert.detach()
    counts = raw_counts.float()
    routed_tokens = int(raw_counts.sum().item())

    if routed_tokens > 0 and counts.numel() > 0:
        mean_tokens = routed_tokens / counts.numel()
        std_tokens = float(counts.std(unbiased=False).item())
        max_tokens = int(raw_counts.max().item())
        expert_cv = std_tokens / mean_tokens if mean_tokens > 0 else 0.0
        top1_expert_share = max_tokens / routed_tokens
        expert_max_over_mean = max_tokens / mean_tokens if mean_tokens > 0 else 0.0
    else:
        expert_cv = 0.0
        top1_expert_share = 0.0
        expert_max_over_mean = 0.0

    return {
        "routed_tokens": routed_tokens,
        "expert_cv": float(expert_cv),
        "top1_expert_share": float(top1_expert_share),
        "expert_max_over_mean": float(expert_max_over_mean),
        "tokens_per_expert": raw_counts.long().tolist(),
    }


def set_trace_fields(scope: Any, fields: Mapping[str, Any]) -> None:
    """Fill predeclared output slots on an active trace scope."""
    for name, value in fields.items():
        scope.set(name, value)


__all__ = [
    "EXPERT_WORKLOAD_SLOTS",
    "ROUTER_WORKLOAD_SLOTS",
    "combine_trace_context",
    "dispatch_trace_context",
    "expert_workload",
    "experts_trace_context",
    "router_trace_context",
    "router_workload",
    "set_trace_fields",
]
