# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace metadata helpers for routed MoE phases."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

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

_AUX_LOSS_NAMES = frozenset(
    {"load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss"}
)


class _RouterLossCollector:
    """Invocation-local normalized loss observations for one router scope."""

    __slots__ = ("_aux_loss_values", "_z_loss_values")

    def __init__(self) -> None:
        self._aux_loss_values: list[torch.Tensor | None] = []
        self._z_loss_values: list[torch.Tensor | None] = []

    @staticmethod
    def _detach_normalized(value: torch.Tensor, coefficient: float) -> torch.Tensor | None:
        try:
            return value.detach() / coefficient
        except Exception:
            return None

    @staticmethod
    def _to_scalar(value: torch.Tensor | None) -> float | None:
        if value is None:
            return None
        try:
            scalar = float(value.item())
        except Exception:
            return None
        return scalar if math.isfinite(scalar) else None

    def observe(self, name: str, value: torch.Tensor, coefficient: float) -> None:
        if name in _AUX_LOSS_NAMES:
            self._aux_loss_values.append(self._detach_normalized(value, coefficient))
        elif name == "z_loss":
            self._z_loss_values.append(self._detach_normalized(value, coefficient))

    def fields(self) -> dict[str, float | None]:
        # The source schema has one aux_loss slot. Multiple enabled target
        # subtypes cannot be represented without silently choosing one.
        aux_loss = (
            self._to_scalar(self._aux_loss_values[0]) if len(self._aux_loss_values) == 1 else None
        )
        z_loss = self._to_scalar(self._z_loss_values[0]) if len(self._z_loss_values) == 1 else None
        return {"aux_loss": aux_loss, "z_loss": z_loss}


_ACTIVE_ROUTER_LOSS_COLLECTOR: ContextVar[_RouterLossCollector | None] = ContextVar(
    "megatron_moe_router_loss_collector", default=None
)


@contextmanager
def collect_router_loss_fields() -> Iterator[_RouterLossCollector]:
    """Bind loss observations to the current accepted router invocation."""
    collector = _RouterLossCollector()
    token = _ACTIVE_ROUTER_LOSS_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_ROUTER_LOSS_COLLECTOR.reset(token)


def observe_router_loss(name: str, value: torch.Tensor, coefficient: float = 1.0) -> None:
    """Record a source-compatible base loss only for an active eager trace scope."""
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        return

    collector = _ACTIVE_ROUTER_LOSS_COLLECTOR.get()
    if collector is not None:
        collector.observe(name, value, coefficient)


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


def shared_experts_trace_context(layer: Any) -> dict[str, Any]:
    """Build the source-compatible shared-expert topology metadata."""
    return {"layer": layer.layer_number, "ep_size": _ep_size(layer)}


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
    "collect_router_loss_fields",
    "combine_trace_context",
    "dispatch_trace_context",
    "expert_workload",
    "experts_trace_context",
    "observe_router_loss",
    "router_trace_context",
    "router_workload",
    "set_trace_fields",
    "shared_experts_trace_context",
]
