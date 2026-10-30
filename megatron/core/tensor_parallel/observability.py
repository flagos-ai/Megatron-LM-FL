# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility exports for MegaLens TP probes."""

# BEGIN MEGALENS OBSERVABILITY
from megatron.megalens.probes.tp import (
    async_linear_collective_launch_scope,
    sync_linear_all_gather_scope,
    wait_async_linear_collective,
)

__all__ = [
    'async_linear_collective_launch_scope',
    'sync_linear_all_gather_scope',
    'wait_async_linear_collective',
]
# END MEGALENS OBSERVABILITY
