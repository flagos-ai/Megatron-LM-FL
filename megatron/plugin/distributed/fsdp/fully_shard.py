# Copyright (c) 2026, BAAI. All rights reserved.
"""Shared compatibility for FSDP optimizers on older PyTorch versions."""

import logging

import torch
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import is_torch_min_version

logger = logging.getLogger(__name__)


def _configure_optimizer_for_dtensor_meshes(optimizer: torch.optim.Optimizer) -> None:
    """Preserve mixed-mesh parameter groups while avoiding old foreach dispatch."""
    if is_torch_min_version("2.7.0"):
        return

    def configure_foreach(optimizer):
        for group in optimizer.param_groups:
            if "foreach" not in group or group["foreach"] is False or group.get("fused"):
                continue
            meshes = {param.device_mesh for param in group["params"] if isinstance(param, DTensor)}
            if len(meshes) > 1:
                # PyTorch < 2.7 assumes a foreach tensor list shares one mesh.
                # Empty local shards can change which mesh appears first on each
                # rank. Per-tensor updates preserve groups and checkpoint indices.
                group["foreach"] = False
                logger.warning(
                    "Disabling foreach for an optimizer group with DTensors on different "
                    "DeviceMeshes: PyTorch < 2.7 requires per-tensor updates."
                )

    configure_foreach(optimizer)
    # Older checkpoints can restore foreach=True/None in the parameter groups.
    optimizer.register_load_state_dict_post_hook(configure_foreach)
