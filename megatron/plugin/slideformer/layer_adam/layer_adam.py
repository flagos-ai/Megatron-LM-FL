# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import glob
import itertools
import logging
import math
import os
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from typing import Any

import torch

from .builder import CPUAdamLoader

logger = logging.getLogger(__name__)


class LayerAdam:
    """Layer-wise CPU Adam optimizer for SlideFormer-managed parameters."""

    _optimizer_id_counter = itertools.count()

    def __init__(
        self,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adamw_mode=True,
        fp32_optimizer_state=True,
        num_layer: int = 0,
        nvme_offload_fraction: float = 0.0,
        offload_dir: str = "/NVME1",
        prefetch: bool = True,
        distributed_cfg=None,
        per_layer_cpu_adam: bool = False,
    ):
        """Create an initially empty optimizer shared by all managed layers.

        Args:
            lr: Learning rate.
            bias_correction: Apply Adam bias correction.
            betas: Adam beta coefficients.
            eps: Adam numerical-stability epsilon.
            weight_decay: Weight-decay coefficient.
            adamw_mode: Use decoupled AdamW weight decay.
            fp32_optimizer_state: Keep optimizer moments in FP32.
            num_layer: Total number of managed layer units.
            nvme_offload_fraction: Fraction of each layer state offloaded to NVMe.
            offload_dir: NVMe offload directory.
            prefetch: Enable NVMe state prefetch.
        """
        self.defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adamw_mode=adamw_mode,
        )

        self.param_groups = []

        self.cpu_adam = CPUAdamLoader().load()
        self.per_layer_cpu_adam = per_layer_cpu_adam
        self._layer_optimizer_ids: dict[int, int] = {}

        self.optimizer_id = next(self._optimizer_id_counter)
        self._create_cpu_adam(self.optimizer_id, should_log=True)

        self.fp32_optimizer_state = fp32_optimizer_state
        self.num_layer = num_layer
        self.distributed_cfg = distributed_cfg

        self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)

        assert 0.0 <= nvme_offload_fraction <= 1.0
        self.nvme_offload_fraction = nvme_offload_fraction
        self.prefetch = prefetch
        self.pin_memory = nvme_offload_fraction > 0.0

        self.exp_avg_flat = OrderedDict()
        self.exp_avg_sq_flat = OrderedDict()

        if self.nvme_offload_fraction > 0.0:
            try:
                from tensornvme import DiskOffloader
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Please install tensornvme to use NVMeOptimizer")

            assert offload_dir is not None, "offload_dir cannot be None"
            self.offload_dir = offload_dir
            self.offloader = DiskOffloader(self.offload_dir, 16, "uring")
            self.offload_numel = OrderedDict()
            self._offloaded_layers: set[int] = set()

        else:
            self.offload_dir = None
            self.offloader = None
            self._offloaded_layers: set[int] = set()

    def _create_cpu_adam(self, optimizer_id: int, *, should_log: bool = False) -> None:
        self.cpu_adam.create_adam(
            optimizer_id,
            self.defaults["lr"],
            self.defaults["betas"][0],
            self.defaults["betas"][1],
            self.defaults["eps"],
            self.defaults["weight_decay"],
            self.defaults["adamw_mode"],
            should_log,
        )

    def __del__(self):
        """Release native optimizer instances and temporary offload files."""
        if hasattr(self, "cpu_adam") and hasattr(self, "optimizer_id"):
            try:
                for optimizer_id in getattr(self, "_layer_optimizer_ids", {}).values():
                    self.cpu_adam.destroy_adam(optimizer_id)
                self.cpu_adam.destroy_adam(self.optimizer_id)
            except Exception:
                pass

        if hasattr(self, "offloader"):
            del self.offloader
            if self.offload_dir and os.path.exists(self.offload_dir):
                try:
                    for file in glob.glob(os.path.join(self.offload_dir, "offload-*")):
                        os.remove(file)
                except OSError:
                    pass

    def _get_layer_numel(self, layer_idx: int) -> int:
        """Return the number of parameters managed for one layer unit."""
        numel = 0
        for p in self.param_groups[layer_idx]["params"]:
            numel += p.numel()
        return numel

    def add_layer_params(self, layer_idx: int, params: Iterable[torch.nn.Parameter]) -> int:
        """Register a layer unit's parameters with the optimizer.

        Args:
            layer_idx: Layer-unit index.
            params: Parameters owned by the layer unit.
        """
        while len(self.param_groups) < layer_idx:
            self.param_groups.append(
                {"params": [], "layer_idx": len(self.param_groups), "step": 0, **self.defaults}
            )

        params = list(params)
        if not params:
            if len(self.param_groups) == layer_idx:
                self.param_groups.append(
                    {"params": [], "layer_idx": layer_idx, "step": 0, **self.defaults}
                )
            return -1

        param_group = {"params": params, "layer_idx": layer_idx, "step": 0, **self.defaults}

        if len(self.param_groups) == layer_idx:
            self.param_groups.append(param_group)
        else:
            self.param_groups[layer_idx] = param_group

        layer_numel = sum(p.numel() for p in params)

        state_type = torch.float32 if self.fp32_optimizer_state else params[0].dtype

        self.exp_avg_flat[layer_idx] = torch.zeros(
            layer_numel, dtype=state_type, device=torch.device("cpu")
        )

        self.exp_avg_sq_flat[layer_idx] = torch.zeros(
            layer_numel, dtype=state_type, device=torch.device("cpu")
        )

        offset = 0
        for p in params:
            size = p.numel()
            shape = p.shape

            if p not in self.state:
                self.state[p] = {}

            self.state[p]["exp_avg"] = self.exp_avg_flat[layer_idx][offset : offset + size].view(
                shape
            )
            self.state[p]["exp_avg_sq"] = self.exp_avg_sq_flat[layer_idx][
                offset : offset + size
            ].view(shape)

            offset += size

        if self.offloader is not None:
            if layer_idx != self.num_layer - 1:
                self._offload_layer(layer_idx)
            else:
                self.offloader.sync_write_events()
        if self.per_layer_cpu_adam and layer_idx not in self._layer_optimizer_ids:
            optimizer_id = next(self._optimizer_id_counter)
            self._create_cpu_adam(optimizer_id, should_log=False)
            self._layer_optimizer_ids[layer_idx] = optimizer_id

        return layer_idx

    def _optimizer_id_for_layer(self, layer_idx: int | None) -> int:
        if layer_idx is None or not self.per_layer_cpu_adam:
            return self.optimizer_id
        return self._layer_optimizer_ids.get(layer_idx, self.optimizer_id)

    def _load_layer(self, layer_idx: int):
        """Issue asynchronous reads for an offloaded layer state."""
        if self.offloader is None:
            return

        if layer_idx < 0:
            layer_idx = self.num_layer + layer_idx

        if layer_idx not in self.exp_avg_flat:
            return
        if layer_idx not in self._offloaded_layers:
            return

        self.offloader.async_read(self.exp_avg_flat[layer_idx])
        if self.nvme_offload_fraction > 0.5:
            self.offloader.async_read(self.exp_avg_sq_flat[layer_idx])

    def _offload_layer(self, layer_idx: int):
        """Issue asynchronous writes for a resident layer state."""
        if self.offloader is None:
            return

        if layer_idx not in self.exp_avg_flat:
            return

        self.offloader.async_write(self.exp_avg_flat[layer_idx])
        if self.nvme_offload_fraction > 0.5:
            self.offloader.async_write(self.exp_avg_sq_flat[layer_idx])
        self._offloaded_layers.add(layer_idx)

    def _pre_step(self, layer_idx: int):
        """Ensure the current layer state is resident before CPU Adam updates."""
        if self.offloader is None:
            return

        # Correctness first: the layer being updated must be fully loaded
        # before the native CPU Adam kernel receives state tensor views.
        self.offloader.sync_read_events()
        self._load_layer(layer_idx)
        self.offloader.sync_read_events()

    def _post_step(self, layer_idx: int):
        """Offload updated state and maintain the NVMe pipeline."""
        if self.offloader is None:
            return

        if self.prefetch:
            self.offloader.sync_write_events()

        self._offload_layer(layer_idx)

    def state_dict(self) -> dict[str, Any]:
        if self.offloader is not None:
            self.offloader.sync_read_events()
            self.offloader.sync_write_events()
        return {
            "defaults": dict(self.defaults),
            "param_group_steps": {group["layer_idx"]: group["step"] for group in self.param_groups},
            "exp_avg_flat": {
                layer_idx: tensor.detach().cpu().clone()
                for layer_idx, tensor in self.exp_avg_flat.items()
            },
            "exp_avg_sq_flat": {
                layer_idx: tensor.detach().cpu().clone()
                for layer_idx, tensor in self.exp_avg_sq_flat.items()
            },
            "num_layer": self.num_layer,
            "nvme_offload_fraction": self.nvme_offload_fraction,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        steps = state_dict.get("param_group_steps", {})
        for group in self.param_groups:
            layer_idx = group["layer_idx"]
            if layer_idx in steps:
                group["step"] = int(steps[layer_idx])

        for source_name, target in (
            ("exp_avg_flat", self.exp_avg_flat),
            ("exp_avg_sq_flat", self.exp_avg_sq_flat),
        ):
            for layer_idx, source in state_dict.get(source_name, {}).items():
                if layer_idx not in target:
                    raise KeyError(f"LayerAdam state has unknown layer index {layer_idx}")
                if target[layer_idx].shape != source.shape:
                    raise ValueError(
                        f"LayerAdam {source_name}[{layer_idx}] shape mismatch: "
                        f"{tuple(target[layer_idx].shape)} != {tuple(source.shape)}"
                    )
                target[layer_idx].copy_(
                    source.to(dtype=target[layer_idx].dtype), non_blocking=False
                )

        if self.offloader is not None:
            for layer_idx in self.exp_avg_flat:
                self._offload_layer(layer_idx)
            self.offloader.sync_write_events()

    def torch_adam_update(
        self,
        data,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
    ):
        """Apply an Adam update using PyTorch tensor operations."""
        grad = grad.to(data.dtype)

        if weight_decay != 0:
            if self.defaults["adamw_mode"]:
                data.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(data, alpha=weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1

        data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, layer_idx: int | None = None, closure=None):
        """Update one layer unit, or every unit when ``layer_idx`` is omitted.

        Args:
            layer_idx: Optional layer-unit index.
            closure: Optional closure that evaluates the model.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if layer_idx is not None:
            param_groups_to_update = [self.param_groups[layer_idx]]
        else:
            assert self.offloader is None, "NVMe offload requires ordered per-layer updates"
            param_groups_to_update = self.param_groups

        self._pre_step(layer_idx)

        for param_group in param_groups_to_update:
            beta1, beta2 = param_group["betas"]
            param_group["step"] += 1
            cpu_optimizer_id = self._optimizer_id_for_layer(param_group["layer_idx"])

            for p in param_group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                target_device = p.device

                if target_device.type == "cpu":
                    assert (
                        p.data.numel() == p.grad.data.numel()
                    ), "parameter and gradient sizes must match"

                    self.cpu_adam.adam_update(
                        cpu_optimizer_id,
                        param_group["step"],
                        param_group["lr"],
                        beta1,
                        beta2,
                        param_group["eps"],
                        param_group["weight_decay"],
                        param_group["bias_correction"],
                        p.data,
                        p.grad.data,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                    )

                else:
                    raise RuntimeError(f"unsupported parameter device: {target_device.type}")

        self._post_step(layer_idx)

        return loss

    @torch.no_grad()
    def _synchronize_cpu_gradient(self, grad: torch.Tensor) -> None:
        del grad
        if self.distributed_cfg is not None and getattr(self.distributed_cfg, "world_size", 1) > 1:
            raise NotImplementedError(
                "Megatron-LM-FL SlideFormer LayerAdam currently supports DP=1 only"
            )

    @torch.no_grad()
    def step_with_grad_views(
        self,
        layer_idx: int | None = None,
        grad_views: dict[torch.nn.Parameter, torch.Tensor] | None = None,
        param_views: dict[torch.nn.Parameter, torch.Tensor] | None = None,
        closure=None,
    ):
        """Update from supplied gradient and CPU-master parameter views.

        Args:
            layer_idx: Optional layer-unit index.
            grad_views: Mapping from parameters to gradient views.
            param_views: Mapping from parameters to CPU-master parameter views.
            closure: Optional closure that evaluates the model.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if layer_idx is not None:
            param_groups_to_update = [self.param_groups[layer_idx]]
        else:
            assert self.offloader is None, "NVMe offload requires ordered per-layer updates"
            param_groups_to_update = self.param_groups

        self._pre_step(layer_idx)

        for param_group in param_groups_to_update:
            beta1, beta2 = param_group["betas"]
            param_group["step"] += 1
            cpu_optimizer_id = self._optimizer_id_for_layer(param_group["layer_idx"])

            for p in param_group["params"]:
                grad = grad_views.get(p) if grad_views else None
                if grad is None:
                    logger.warning(f"Parameter {p} has no gradient view; skipping update.")
                    continue
                self._synchronize_cpu_gradient(grad)

                state = self.state[p]
                target = param_views.get(p, p.data) if param_views else p.data
                target_device = target.device

                if target_device.type == "cpu":
                    assert target.numel() == grad.numel(), "parameter and gradient sizes must match"
                    self.cpu_adam.adam_update(
                        cpu_optimizer_id,
                        param_group["step"],
                        param_group["lr"],
                        beta1,
                        beta2,
                        param_group["eps"],
                        param_group["weight_decay"],
                        param_group["bias_correction"],
                        target,
                        grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                    )
                else:
                    raise RuntimeError(f"unsupported parameter device: {target_device.type}")

        self._post_step(layer_idx)

        return loss

    def update_learning_rate(self, new_lr):
        """Update the learning rate for every parameter group.

        Args:
            new_lr: New learning rate.
        """
        self.defaults["lr"] = new_lr
        for param_group in self.param_groups:
            param_group["lr"] = new_lr
