# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit, shared configuration for the CLI and portable dense-DP fast paths."""
from dataclasses import asdict, dataclass, replace
import json
import os
import socket

import torch
import torch.distributed as dist


def _enabled(name, default="0"):
    return os.getenv("MEGATRON_SLIDEFORMER_" + name, default).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DataParallelTransport:
    """Same-host Dense DP, FP32 CPU Adam, one batch and known token count.

    Default within this explicitly selected configuration is native GPU
    reduce-scatter with sharded H2D and native parameter gather. Legacy callers without a configuration
    keep their existing path. No process-wide environment mutation is needed.
    """
    gradient_reduce: str = "gpu_reduce"
    parameter_transfer: str = "shard"

    def __post_init__(self):
        if self.gradient_reduce not in {"hsr", "gpu_reduce"}:
            raise ValueError("gradient_reduce must be hsr or gpu_reduce")
        if self.parameter_transfer not in {"full", "shard"}:
            raise ValueError("parameter_transfer must be full or shard")

    @classmethod
    def from_env(cls, *, preset_only=False):
        preset = os.getenv("MEGATRON_SLIDEFORMER_DP_PRESET", "")
        if preset:
            if preset not in {"dense-dp", "hsr-dp"}:
                raise ValueError("DP_PRESET must be dense-dp or hsr-dp")
            gpu = preset == "dense-dp"
            for name, expected in (("SHARDED_CPU_OPT", True), ("EAGER_SHARDED_UPDATE", True),
                                   ("GPU_REDUCE", gpu)):
                if "MEGATRON_SLIDEFORMER_" + name in os.environ and _enabled(name) != expected:
                    raise ValueError(f"DP_PRESET conflicts with {name}")
            return cls("gpu_reduce" if gpu else "hsr",
                       "shard" if _enabled("SHARD_H2D", "1" if gpu else "0") else "full")
        if preset_only:
            return None
        if _enabled("SHARDED_CPU_OPT"):
            if not _enabled("EAGER_SHARDED_UPDATE"):
                raise ValueError("Portable sharded transport requires EAGER_SHARDED_UPDATE=1")
            return cls("gpu_reduce" if _enabled("GPU_REDUCE") else "hsr",
                       "shard" if _enabled("SHARD_H2D") else "full")
        if _enabled("GPU_REDUCE") or _enabled("SHARD_H2D"):
            raise ValueError("GPU_REDUCE/SHARD_H2D requires a sharded transport configuration")
        return None

    def prepare(self, model, groups, config):
        """Validate before allocating layer/optimizer storage; preserve native operators."""
        native = model.config
        if any(getattr(native, key, 1) != 1 for key in (
                "tensor_model_parallel_size", "context_parallel_size",
                "expert_model_parallel_size", "pipeline_model_parallel_size")):
            raise NotImplementedError("Explicit DP transport is qualified only for pure DP")
        if getattr(native, "num_moe_experts", None) is not None:
            raise NotImplementedError("Explicit DP transport is qualified for dense models")
        if not getattr(native, "calculate_per_token_loss", False):
            raise ValueError("Explicit DP transport requires native per-token loss")
        if getattr(native, "params_dtype", None) != torch.bfloat16:
            raise ValueError("Explicit DP transport requires BF16 execution weights")
        if (not config.defer_optimizer_step or not config.fp32_master_params or
                not config.fp32_optimizer_state or config.nvme_offload_fraction or
                config.skip_grad_d2h or config.skip_cpu_adam_step):
            raise ValueError("Explicit DP transport requires FP32 CPU master/Adam without optimizer NVMe or skipped work")
        if any(_enabled(name) for name in ("LAYERWISE_UPDATE", "ASYNC_LAYER_CAPTURE", "GPU_GRADIENT_PIPELINE")):
            raise ValueError("Explicit DP transport cannot attach another gradient/update owner")
        size = dist.get_world_size(groups.dp_cp)
        if size < 2:
            raise ValueError("Explicit DP transport requires at least two replicas")
        hosts = [None] * size
        dist.all_gather_object(hosts, socket.gethostname(), group=groups.dp_cp)
        if len(set(hosts)) != 1:
            raise ValueError("Explicit DP transport requires replicas on one host")
        return replace(config, flat_grad_d2h=self.gradient_reduce == "gpu_reduce")

    def attach(self, engine, groups):
        enabled = engine.enable_sharded_cpu_optimizer(
            groups.dp_cp, groups.tp, groups.pp,
            gpu_reduce=self.gradient_reduce == "gpu_reduce",
            shard_h2d=self.parameter_transfer == "shard")
        owners = [o for o in engine.managed_layers if o.params]
        if not enabled or len(engine._sharded_opt.owners) != len(owners):
            raise RuntimeError("Requested DP transport did not cover every trainable owner")
        if self.gradient_reduce == "gpu_reduce" and (
                engine._gpu_reduce is None or len(engine._gpu_reduce.layers) != len(owners)):
            raise RuntimeError("Requested GPU reduction did not cover every trainable owner")
        if self.parameter_transfer == "shard" and any(o.shard_h2d is None for o in owners):
            raise RuntimeError("Requested shard H2D did not cover every trainable owner")
        engine._dp_transport = self
        if dist.get_rank() == 0:
            print("SLIDEFORMER_DP_TRANSPORT " + json.dumps(asdict(self), sort_keys=True), flush=True)
