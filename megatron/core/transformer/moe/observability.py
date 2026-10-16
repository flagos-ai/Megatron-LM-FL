# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility exports for MegaLens MOE probes."""

# BEGIN MEGALENS OBSERVABILITY
from megatron.megalens.probes.moe import (
    DISPATCH_ROUTER_FIELDS,
    EXPERT_WORKLOAD_SLOTS,
    ROUTER_WORKLOAD_SLOTS,
    bind_dispatch_fields,
    capture_dispatch_fields,
    collect_router_assignment_fields,
    collect_router_loss_fields,
    combine_trace_context,
    current_dispatch_fields,
    dispatch_fields_requested,
    dispatch_trace_context,
    ep_collective_trace_context,
    expert_workload,
    experts_trace_context,
    observe_router_assignments_before_drop,
    observe_router_loss,
    publish_dispatch_fields,
    router_trace_context,
    router_workload,
    set_trace_fields,
    shared_experts_trace_context,
)

__all__ = [
    'DISPATCH_ROUTER_FIELDS',
    'EXPERT_WORKLOAD_SLOTS',
    'ROUTER_WORKLOAD_SLOTS',
    'bind_dispatch_fields',
    'capture_dispatch_fields',
    'collect_router_assignment_fields',
    'collect_router_loss_fields',
    'combine_trace_context',
    'current_dispatch_fields',
    'dispatch_fields_requested',
    'dispatch_trace_context',
    'ep_collective_trace_context',
    'expert_workload',
    'experts_trace_context',
    'observe_router_assignments_before_drop',
    'observe_router_loss',
    'publish_dispatch_fields',
    'router_trace_context',
    'router_workload',
    'set_trace_fields',
    'shared_experts_trace_context',
]
# END MEGALENS OBSERVABILITY
