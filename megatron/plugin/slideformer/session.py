# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit full-finetuning lifecycle for MCore trainers (initially PP=1).

The caller constructs the native model and ProcessGroupCollection. This adapter
does not install hooks into megatron.training or monkey-patch parallel_state.
The FL training adapter remains available for its existing PP/checkpoint path.
"""
from dataclasses import replace
from importlib import import_module
import inspect
import math

import torch.distributed as dist

from .engine import MegatronSlideFormerEngineConfig, apply_true_megatron_slideformer
from .transport import DataParallelTransport


def validate_native_config(model, groups, config):
    """Reject unsupported compositions before allocating offload buffers."""
    native = model.config
    if not config.defer_optimizer_step:
        raise ValueError("Native parallel integration requires deferred optimizer updates")
    if config.qkv_layout_backend != "megatron":
        raise ValueError("FullFineTuneSession preserves native attention/layout; use qkv_layout_backend='megatron'")
    if getattr(model, "module", None) is not None or hasattr(model, "ddp_config"):
        raise ValueError("Pass the raw native model before DDP/optimizer construction")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("FullFineTuneSession supports full fine-tuning only")
    if getattr(model, "_megatron_slideformer_engine", None) is not None:
        raise ValueError("Model already has a SlideFormer engine")
    for attr in ("pipeline_model_parallel_size", "gtp_weight_remat_size", "expert_gtp_weight_remat_size"):
        if getattr(native, attr, 1) != 1:
            raise NotImplementedError(f"Portable adapter does not yet support {attr} != 1")
    if getattr(native, "virtual_pipeline_model_parallel_size", None) is not None:
        raise NotImplementedError("Portable adapter does not support virtual pipeline parallelism")
    if getattr(native, "recompute_granularity", None) is not None:
        raise ValueError("Use SlideFormer activation checkpointing without native recompute")
    if getattr(native, "fp16", False) or getattr(native, "fp8", None):
        raise NotImplementedError("Portable adapter has no FP16 loss scaler or FP8 state handling")
    if "quantile_balancing" in getattr(native, "moe_router_load_balancing_type", ""):
        raise NotImplementedError("Quantile-balancing router state is not integrated")
    for attr in ("dp_cp", "tp", "pp", "cp", "ep", "expt_dp", "expt_tp"):
        group = getattr(groups, attr, None)
        if group is None or isinstance(group, list):
            raise ValueError(f"An explicit native ProcessGroupCollection.{attr} is required")
        if dist.get_rank(group) < 0:
            raise ValueError(f"Current rank is not a member of {attr}")
    for attr, field in (("tp", "tensor_model_parallel_size"), ("cp", "context_parallel_size"),
                        ("ep", "expert_model_parallel_size"), ("pp", "pipeline_model_parallel_size")):
        if dist.get_world_size(getattr(groups, attr)) != getattr(native, field, 1):
            raise ValueError(f"{attr} group does not match native model configuration")
    if dist.get_world_size(groups.expt_tp) != dist.get_world_size(groups.tp):
        raise NotImplementedError("Different dense/expert TP degrees need a separate gradient adapter")


class FullFineTuneSession:
    """Own offload/update state; retain native model, operators and groups.

    begin_step -> native forward/backward(s) -> finalize_gradients -> step.
    Call engine.save_checkpoint/load_checkpoint only at completed step boundaries.
    Do not attach another optimizer or native DDP to this model.
    """

    def __init__(self, model, *, groups, device, config=None, transport=None):
        config = config or replace(MegatronSlideFormerEngineConfig(), defer_optimizer_step=True)
        validate_native_config(model, groups, config)
        self.transport = transport if transport is not None else DataParallelTransport.from_env()
        if self.transport is not None:
            if not isinstance(self.transport, DataParallelTransport):
                raise TypeError("transport must be a DataParallelTransport")
            config = self.transport.prepare(model, groups, config)
        self.model, self.groups = model, groups
        self.engine = apply_true_megatron_slideformer(model, device=device, config=config)
        try:
            if self.transport is not None:
                self.transport.attach(self.engine, groups)
            else:
                self.engine.enable_layer_update_pipeline(
                    group=groups.dp_cp, tp_group=groups.tp, expert_group=groups.expt_dp,
                    pp_group=groups.pp)
        except BaseException:
            self.engine.close()
            raise
        self._phase = "ready"

    def begin_step(self, num_microbatches=1, *, num_tokens=None):
        if self._phase != "ready":
            raise RuntimeError(f"begin_step requires ready state, got {self._phase}")
        if not isinstance(num_microbatches, int) or num_microbatches < 1:
            raise ValueError("num_microbatches must be a positive integer")
        if self.transport is not None and (num_microbatches != 1 or num_tokens is None):
            raise ValueError("Explicit DP transport requires one batch and its local token count before backward")
        total = None
        if num_tokens is not None and (self.transport is not None or self.engine.layer_update_pipeline is not None):
            from .compat import get_global_num_tokens
            total = float(get_global_num_tokens(num_tokens, self.groups.pp, self.groups.dp_cp).item())
            if self.transport is not None and (not math.isfinite(total) or total <= 0):
                raise ValueError("Explicit DP transport requires a positive finite global token count")
        self.model.zero_grad(set_to_none=True)
        self.engine.zero_unmanaged_grads(num_microbatches=num_microbatches)
        if total is not None:
            self.engine.set_expected_scale(1.0 / total if total > 0 else 1.0)
        self._phase = "backward"

    def finalize_gradients(self, num_tokens=None):
        if self._phase != "backward":
            raise RuntimeError(f"finalize_gradients requires backward state, got {self._phase}")
        if self.model.config.calculate_per_token_loss and num_tokens is None:
            raise ValueError("Per-token loss requires this rank's local non-padding token count")
        groups = self.groups
        self.engine.finalize_data_parallel_gradients(
            group=groups.dp_cp, num_tokens=num_tokens, tp_group=groups.tp,
            expert_group=groups.expt_dp, expert_tp_group=groups.expt_tp,
            pp_group=groups.pp, defer_commit=True,
        )
        native = import_module("megatron.core.distributed.finalize_model_grads")
        if self.model.config.moe_router_enable_expert_bias:
            update = native._update_router_expert_bias
            kwargs = {}
            if "tp_dp_cp_group" in inspect.signature(update).parameters:
                group = getattr(groups, "tp_dp_cp", None)
                if group is None:
                    raise ValueError("Native router bias update requires groups.tp_dp_cp")
                kwargs["tp_dp_cp_group"] = group
            update([self.model], self.model.config, **kwargs)
        native.reset_model_temporary_tensors(self.model.config, [self.model])
        self.engine.commit_pipeline_updates()
        self._phase = "finalized"

    def step(self):
        if self._phase != "finalized":
            raise RuntimeError(f"step requires finalized state, got {self._phase}")
        self.engine.step_unmanaged_params()
        self._phase = "ready"

    def close(self):
        self.engine.close()
        self._phase = "closed"
