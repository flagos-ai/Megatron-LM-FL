from __future__ import annotations

import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from megatron.plugin.slideformer.checkpoint import (
    MegatronSlidingCheckpoint,
    slideformer_slot_checkpoint,
)
from megatron.plugin.slideformer.layer_adam import LayerAdam
from megatron.plugin.slideformer.layout import (
    MegatronDecoderLayout,
    assert_trainable_parameter_coverage,
    resolve_megatron_decoder_layout,
)


@dataclass(frozen=True)
class _ManagedModuleSpec:
    module: nn.Module
    params: list[nn.Parameter]
    keep_loaded_after_forward: bool = False
    is_transformer_layer: bool = False


class _CheckpointBackwardBoundary(torch.autograd.Function):
    """Run a callback after a checkpointed layer has fully propagated dgrad."""

    @staticmethod
    def forward(ctx: Any, hidden_states: torch.Tensor, callback: Any) -> torch.Tensor:
        ctx.callback = callback
        return hidden_states

    @staticmethod
    def backward(ctx: Any, grad_hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        ctx.callback()
        return grad_hidden_states, None


@dataclass(frozen=True)
class MegatronSlideFormerEngineConfig:
    lr: float = 1e-5
    bias_correction: bool = True
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    adamw_mode: bool = True
    fp32_optimizer_state: bool = True
    fp32_master_params: bool = True
    nvme_offload_fraction: float = 0.0
    offload_dir: str = "/NVME1"
    pin_memory: bool = True
    prefetch: bool = True
    chunk_size_mb: int = 32
    double_buffer: bool = False
    gpu_buffer_count: int = 2
    window_size: int = 1
    offload_after_forward: bool = False
    activation_offload: bool = False
    activation_backend: str = "checkpoint"
    activation_slot_prefetch: bool = False
    activation_slot_gpu_window: int = 0
    unified_h2d_scheduler: bool = False
    max_outstanding_h2d: int = 2
    activation_offload_min_numel: int = 1
    activation_offload_max_dim: int = 99
    activation_offload_leaf_tensors: bool = True
    activation_offload_max_tensors_per_layer: int = -1
    overlap_grad_d2h_cpu_adam: bool = True
    optimizer_pipeline_depth: int = 2
    optimizer_worker_count: int = 1
    cpu_adam_per_layer_optimizer: bool = False
    optimizer_backpressure: bool = True
    skip_grad_d2h: bool = False
    skip_cpu_adam_step: bool = False
    flat_grad_d2h: bool = False
    te_fused_main_grad: bool = False
    shared_cpu_buffers: bool = True
    cpu_grad_buffer_count: int = 0
    cpu_param_staging_buffer_count: int = 1
    use_native_slideformer_overlap: bool = False
    native_slideformer_backward_lifecycle: bool = False
    profile_slideformer_lifecycle: bool = False
    profile_slideformer_lifecycle_summary: bool = False
    profile_slideformer_lifecycle_memory: bool = True
    profile_activation_saved_tensors: bool = False
    profile_activation_lifecycle: bool = False
    enable_timing: bool = False

    def __post_init__(self) -> None:
        if self.activation_backend not in {"checkpoint", "slideformer-slot"}:
            raise ValueError("activation_backend must be 'checkpoint' or 'slideformer-slot'")
        if self.activation_backend == "slideformer-slot" and not self.activation_offload:
            raise ValueError(
                "activation_backend='slideformer-slot' requires activation_offload=True"
            )
        if self.activation_slot_prefetch and self.activation_backend != "slideformer-slot":
            raise ValueError(
                "activation_slot_prefetch requires activation_backend='slideformer-slot'"
            )
        if self.activation_slot_gpu_window < 0:
            raise ValueError("activation_slot_gpu_window must be non-negative")
        if self.activation_slot_gpu_window and self.activation_backend != "slideformer-slot":
            raise ValueError(
                "activation_slot_gpu_window requires activation_backend='slideformer-slot'"
            )
        if self.max_outstanding_h2d < 1:
            raise ValueError("max_outstanding_h2d must be at least 1")
        if self.native_slideformer_backward_lifecycle and self.flat_grad_d2h:
            raise ValueError(
                "native_slideformer_backward_lifecycle is incompatible with flat_grad_d2h; "
                "the experimental lifecycle keeps per-parameter early D2H."
            )
        if self.native_slideformer_backward_lifecycle and not self.optimizer_backpressure:
            raise ValueError(
                "native_slideformer_backward_lifecycle requires optimizer backpressure; "
                "use the existing path for no-backpressure ablations."
            )
        if self.optimizer_worker_count < 1:
            raise ValueError("optimizer_worker_count must be at least 1")
        if self.cpu_grad_buffer_count < 0 or self.cpu_param_staging_buffer_count < 1:
            raise ValueError(
                "CPU grad buffer count must be non-negative and parameter staging "
                "buffer count must be positive"
            )
        if self.optimizer_worker_count > 1 and not self.cpu_adam_per_layer_optimizer:
            object.__setattr__(self, "cpu_adam_per_layer_optimizer", True)
        if not self.use_native_slideformer_overlap:
            return
        if not self.offload_after_forward:
            object.__setattr__(self, "offload_after_forward", True)
        if self.optimizer_pipeline_depth == 2:
            object.__setattr__(self, "optimizer_pipeline_depth", 1)

    @staticmethod
    def native_overlap_defaults() -> dict[str, Any]:
        return {
            "pin_memory": True,
            "prefetch": True,
            "window_size": 1,
            "double_buffer": False,
            "gpu_buffer_count": 2,
            "offload_after_forward": True,
            "overlap_grad_d2h_cpu_adam": True,
            "optimizer_pipeline_depth": 1,
            "optimizer_backpressure": True,
            "flat_grad_d2h": False,
        }

    def overlap_config_summary(self) -> dict[str, Any]:
        return {
            "use_native_slideformer_overlap": self.use_native_slideformer_overlap,
            "pin_memory": self.pin_memory,
            "prefetch": self.prefetch,
            "window_size": self.window_size,
            "double_buffer": self.double_buffer,
            "gpu_buffer_count": self.gpu_buffer_count,
            "offload_after_forward": self.offload_after_forward,
            "activation_offload": self.activation_offload,
            "activation_backend": self.activation_backend,
            "activation_slot_prefetch": self.activation_slot_prefetch,
            "activation_slot_gpu_window": self.activation_slot_gpu_window,
            "unified_h2d_scheduler": self.unified_h2d_scheduler,
            "max_outstanding_h2d": self.max_outstanding_h2d,
            "activation_offload_min_numel": self.activation_offload_min_numel,
            "activation_offload_max_dim": self.activation_offload_max_dim,
            "activation_offload_leaf_tensors": self.activation_offload_leaf_tensors,
            "activation_offload_max_tensors_per_layer": self.activation_offload_max_tensors_per_layer,
            "overlap_grad_d2h_cpu_adam": self.overlap_grad_d2h_cpu_adam,
            "optimizer_pipeline_depth": self.optimizer_pipeline_depth,
            "optimizer_worker_count": self.optimizer_worker_count,
            "cpu_adam_per_layer_optimizer": self.cpu_adam_per_layer_optimizer,
            "optimizer_backpressure": self.optimizer_backpressure,
            "skip_grad_d2h": self.skip_grad_d2h,
            "skip_cpu_adam_step": self.skip_cpu_adam_step,
            "flat_grad_d2h": self.flat_grad_d2h,
            "te_fused_main_grad": self.te_fused_main_grad,
            "shared_cpu_buffers": self.shared_cpu_buffers,
            "cpu_grad_buffer_count": self.cpu_grad_buffer_count,
            "cpu_param_staging_buffer_count": self.cpu_param_staging_buffer_count,
            "native_slideformer_backward_lifecycle": self.native_slideformer_backward_lifecycle,
            "profile_slideformer_lifecycle": self.profile_slideformer_lifecycle,
            "profile_slideformer_lifecycle_summary": self.profile_slideformer_lifecycle_summary,
            "profile_slideformer_lifecycle_memory": self.profile_slideformer_lifecycle_memory,
            "profile_activation_saved_tensors": self.profile_activation_saved_tensors,
            "profile_activation_lifecycle": self.profile_activation_lifecycle,
            "chunk_size_mb": self.chunk_size_mb,
            "nvme_offload_fraction": self.nvme_offload_fraction,
            "offload_dir": self.offload_dir,
        }

    def native_overlap_semantic_gaps(self) -> list[dict[str, str]]:
        if not self.use_native_slideformer_overlap:
            return []
        return [
            {
                "mechanism": "shared_gpu_cache_queue",
                "reason": "Megatron parameters stay owned by their modules and hooks cannot swap a single native wrapper cache unit across heterogeneous module boundaries safely.",
                "approximation": "Per-owner GPU parameter buffers with native-sized window_size=1 and forward offload.",
                "semantic_difference": "Memory reuse and backpressure are buffer-pool based instead of queue-slot based.",
            },
            {
                "mechanism": "native_layer_flat_grad_d2h",
                "reason": "Megatron post-accumulate hooks expose gradients per parameter; staging a layer-flat GPU buffer increases peak memory and delayed clearing in current measurements.",
                "approximation": "Keep default per-parameter D2H unless --flat-grad-d2h is explicitly used in benchmark experiments.",
                "semantic_difference": "Copy submission granularity differs from native SlideFormer's single flat layer copy.",
            },
            {
                "mechanism": "forward_internal_backward",
                "reason": "Megatron training calls backward outside model forward and integrates optimizer stepping in the training loop.",
                "approximation": "Use module backward hooks plus engine.wait_for_completion at the optimizer phase.",
                "semantic_difference": "Autograd trigger point is outside forward, so some queue timing differs.",
            },
            {
                "mechanism": "fixed_hf_activation_checkpoint_buffers",
                "reason": "Megatron layers save a dynamic set of tensors rather than native HF hidden-state/mask pairs.",
                "approximation": "MegatronSlidingCheckpoint dynamically records offloaded tensors and prefetches by layer/tensor index.",
                "semantic_difference": "Activation storage layout and prefetch granularity are dynamic, not native fixed slots.",
            },
        ]

    def activation_backend_semantic_gaps(self) -> list[dict[str, str]]:
        if self.activation_backend != "slideformer-slot":
            return []
        return [
            {
                "mechanism": "native_save_on_cpu_all_saved_tensors",
                "reason": "Megatron's non-reentrant checkpoint path does not expose the real internal saved tensors to the outer saved_tensors_hooks context.",
                "approximation": "Explicitly offload the TransformerLayer boundary hidden state and recompute layer internals during backward; compatible bias-free split SwiGLU elides the unused final FC2 value.",
                "semantic_difference": "QKV/attention/MLP intermediates are recomputed rather than saved to CPU and restored, with an early-stop-equivalent FC2 Jacobian path when hidden dropout is zero.",
            },
            {
                "mechanism": "native_attention_mask_rotary_cpu_slots",
                "reason": "Megatron passes masks/RoPE metadata as non-gradient constants that are already available from the parent GPTModel call.",
                "approximation": "Keep these references as metadata for recompute instead of copying them into per-layer CPU slots.",
                "semantic_difference": "Only hidden-state boundary activations use fixed CPU slots.",
            },
        ]

    def backward_lifecycle_semantic_gaps(self) -> list[dict[str, str]]:
        if not self.native_slideformer_backward_lifecycle:
            return []
        return [
            {
                "mechanism": "embedding_final_norm_output_lifecycle",
                "reason": "The prototype intentionally targets homogeneous TransformerLayer units first.",
                "approximation": "Non-transformer managed modules keep the existing Megatron hook lifecycle.",
                "semantic_difference": "Only decoder TransformerLayer units use last-param-grad-ready completion.",
            },
            {
                "mechanism": "native_forward_internal_backward",
                "reason": "Megatron training still owns loss.backward() and optimizer phase boundaries.",
                "approximation": "TransformerLayer owners can become unit-ready at last param grad ready without waiting for the module backward hook.",
                "semantic_difference": "Backward is still launched by Megatron, but layer completion ownership moves earlier toward SlideFormer.",
            },
        ]


class _TimingRecorder:
    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}

    def add(self, name: str, duration: float) -> None:
        self.records.setdefault(name, []).append(duration)

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(self.records.items()):
            if not values:
                continue
            total = sum(values)
            result[name] = {
                "count": len(values),
                "total_s": total,
                "avg_ms": total * 1000.0 / len(values),
                "max_ms": max(values) * 1000.0,
            }
        return result

    def reset(self) -> None:
        self.records.clear()


class _TimedBlock:
    def __init__(self, recorder: _TimingRecorder | None, name: str) -> None:
        self.recorder = recorder
        self.name = name
        self.start = 0.0

    def __enter__(self) -> None:
        if self.recorder is not None:
            self.start = time.perf_counter()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.recorder is not None:
            self.recorder.add(self.name, time.perf_counter() - self.start)


class _SharedCPUBufferPool:
    """Bounded pinned-CPU tensor pool with optional CUDA-copy completion guards."""

    def __init__(self, *, numel: int, dtype: torch.dtype, count: int, pin_memory: bool) -> None:
        self._condition = threading.Condition()
        self._available = list(range(count))
        self._pending_events: list[torch.cuda.Event | None] = [None] * count
        try:
            self.tensors = [
                torch.empty(numel, dtype=dtype, device="cpu", pin_memory=pin_memory)
                for _ in range(count)
            ]
        except RuntimeError:
            self.tensors = [torch.empty(numel, dtype=dtype, device="cpu") for _ in range(count)]

    def acquire(self) -> tuple[int, torch.Tensor]:
        with self._condition:
            while not self._available:
                self._condition.wait()
            slot = self._available.pop()
            event = self._pending_events[slot]
            self._pending_events[slot] = None
        if event is not None:
            event.synchronize()
        return slot, self.tensors[slot]

    def release(self, slot: int, *, event: torch.cuda.Event | None = None) -> None:
        with self._condition:
            self._pending_events[slot] = event
            self._available.append(slot)
            self._condition.notify()


class _SharedGPUBufferPool:
    """Bounded layer-flat CUDA buffer pool guarded by D2H completion events."""

    def __init__(self, *, numel: int, dtype: torch.dtype, count: int, device: torch.device) -> None:
        self._condition = threading.Condition()
        self._available = list(range(count))
        self._pending_events: list[torch.cuda.Event | None] = [None] * count
        self.tensors = [torch.empty(numel, dtype=dtype, device=device) for _ in range(count)]

    def acquire(self) -> tuple[int, torch.Tensor]:
        with self._condition:
            while not self._available:
                self._condition.wait()
            slot = self._available.pop()
            event = self._pending_events[slot]
            self._pending_events[slot] = None
        if event is not None:
            event.synchronize()
        return slot, self.tensors[slot]

    def release(self, slot: int, *, event: torch.cuda.Event | None = None) -> None:
        with self._condition:
            self._pending_events[slot] = event
            self._available.append(slot)
            self._condition.notify()


class _SharedGPUParameterPool:
    """Layer-flat parameter slots whose reuse is ordered on the H2D stream."""

    def __init__(self, *, numel: int, dtype: torch.dtype, count: int, device: torch.device) -> None:
        self._condition = threading.Condition()
        self._available = list(range(count))
        self._pending_events: list[torch.cuda.Event | None] = [None] * count
        self.tensors = [torch.empty(numel, dtype=dtype, device=device) for _ in range(count)]

    def acquire(self, stream: torch.cuda.Stream) -> tuple[int, torch.Tensor]:
        with self._condition:
            while not self._available:
                self._condition.wait()
            slot = self._available.pop()
            event = self._pending_events[slot]
            self._pending_events[slot] = None
        if event is not None:
            stream.wait_event(event)
        return slot, self.tensors[slot]

    def release(self, slot: int, *, event: torch.cuda.Event) -> None:
        with self._condition:
            self._pending_events[slot] = event
            self._available.append(slot)
            self._condition.notify()


class _H2DScheduler:
    """Shared H2D stream plus low-overhead exposed-wait accounting."""

    def __init__(
        self, *, device: torch.device, max_outstanding: int, counters: dict[str, int]
    ) -> None:
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.max_outstanding = max(max_outstanding, 1)
        self.counters = counters
        self._outstanding = 0
        self._outstanding_samples = 0
        self._outstanding_total = 0
        self._wait_records: list[dict[str, Any]] = []
        self._next_request_id = 0

    def _inc(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def _sample_outstanding(self) -> None:
        self._outstanding_samples += 1
        self._outstanding_total += self._outstanding
        self.counters["h2d_outstanding_samples"] = self._outstanding_samples
        self.counters["h2d_outstanding_total"] = self._outstanding_total
        self.counters["h2d_outstanding_peak"] = max(
            self.counters.get("h2d_outstanding_peak", 0), self._outstanding
        )

    def can_submit_prefetch(self, *, pending_depth: int = 1) -> bool:
        if self._outstanding < self.max_outstanding:
            return True
        self.counters["h2d_pending_queue_peak"] = max(
            self.counters.get("h2d_pending_queue_peak", 0), pending_depth
        )
        self._inc("h2d_prefetch_throttle_count")
        return False

    def begin_submit(
        self, *, kind: str, layer_idx: int, bytes_: int, allow_over_limit: bool = False
    ) -> int | None:
        if not allow_over_limit and self._outstanding >= self.max_outstanding:
            self.counters["h2d_pending_queue_peak"] = max(
                self.counters.get("h2d_pending_queue_peak", 0), 1
            )
            self._inc("h2d_submit_throttle_count")
            return None
        self._next_request_id += 1
        request_id = self._next_request_id
        self._outstanding += 1
        self._sample_outstanding()
        self._inc("h2d_submit_count")
        self._inc(f"{kind}_submit_count")
        self._inc(f"{kind}_bytes", bytes_)
        self._inc(f"{kind}_layer_{layer_idx}_submit_count")
        return request_id

    def record_use_wait(
        self,
        *,
        kind: str,
        layer_idx: int,
        bytes_: int,
        event: torch.cuda.Event,
        request_id: int | None = None,
    ) -> None:
        current = torch.cuda.current_stream(self.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(current)
        current.wait_event(event)
        end_event.record(current)
        self._wait_records.append(
            {
                "kind": kind,
                "layer_idx": layer_idx,
                "bytes": bytes_,
                "request_id": request_id,
                "start_event": start_event,
                "end_event": end_event,
            }
        )
        if self._outstanding > 0:
            self._outstanding -= 1
        self._sample_outstanding()

    def wait_stream(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, dict[str, float | int]] = {}
        total_wait_ms = 0.0
        for record in self._wait_records:
            kind = str(record["kind"])
            stats = by_kind.setdefault(
                kind,
                {
                    "count": 0,
                    "bytes": 0,
                    "exposed_wait_ms": 0.0,
                    "max_exposed_wait_ms": 0.0,
                    "deadline_miss_count": 0,
                },
            )
            try:
                elapsed_ms = float(record["start_event"].elapsed_time(record["end_event"]))
            except RuntimeError:
                elapsed_ms = 0.0
            stats["count"] = int(stats["count"]) + 1
            stats["bytes"] = int(stats["bytes"]) + int(record["bytes"])
            stats["exposed_wait_ms"] = float(stats["exposed_wait_ms"]) + elapsed_ms
            stats["max_exposed_wait_ms"] = max(float(stats["max_exposed_wait_ms"]), elapsed_ms)
            # A sub-microsecond event interval is effectively ready at use.
            if elapsed_ms > 0.001:
                stats["deadline_miss_count"] = int(stats["deadline_miss_count"]) + 1
            total_wait_ms += elapsed_ms
        average_outstanding = (
            self._outstanding_total / self._outstanding_samples
            if self._outstanding_samples
            else 0.0
        )
        return {
            "enabled": True,
            "max_outstanding_h2d": self.max_outstanding,
            "outstanding_peak": self.counters.get("h2d_outstanding_peak", 0),
            "outstanding_average": average_outstanding,
            "pending_queue_peak": self.counters.get("h2d_pending_queue_peak", 0),
            "total_exposed_wait_ms": total_wait_ms,
            "by_kind": by_kind,
        }

    def reset(self) -> None:
        self._outstanding = 0
        self._outstanding_samples = 0
        self._outstanding_total = 0
        self._wait_records.clear()


class _ManagedLayer:
    """Single-GPU Megatron layer owner with four hook lifecycle.

    This is the first true-SlideFormer Megatron path: decoder layer parameters
    are owned by CPU master tensors and loaded for forward/backward. It keeps
    the implementation intentionally conservative so correctness tests can
    harden the Megatron structure contract before adding more overlap.
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        layer_idx: int,
        device: torch.device,
        config: MegatronSlideFormerEngineConfig,
        layer_optimizer: LayerAdam,
        params: list[nn.Parameter] | None = None,
        keep_loaded_after_forward: bool = False,
        is_transformer_layer: bool = False,
        on_backward_ready: Any | None = None,
        timing: _TimingRecorder | None = None,
        traffic_counters: dict[str, int] | None = None,
        h2d_scheduler: _H2DScheduler | None = None,
        cpu_grad_pool: _SharedCPUBufferPool | None = None,
        cpu_param_staging_pool: _SharedCPUBufferPool | None = None,
        gpu_grad_pool: _SharedGPUBufferPool | None = None,
        gpu_param_pool: _SharedGPUParameterPool | None = None,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Megatron true SlideFormer requires a CUDA device")
        self.layer = layer
        self.layer_idx = layer_idx
        self.device = device
        self.config = config
        self.layer_optimizer = layer_optimizer
        self.keep_loaded_after_forward = keep_loaded_after_forward
        self.is_transformer_layer = is_transformer_layer
        self.on_backward_ready = on_backward_ready
        self.timing = timing
        self.traffic_counters = traffic_counters
        self.h2d_scheduler = h2d_scheduler
        self.cpu_grad_pool = cpu_grad_pool
        self.cpu_param_staging_pool = cpu_param_staging_pool
        self.gpu_grad_pool = gpu_grad_pool
        self.gpu_param_pool = gpu_param_pool
        self._cpu_grad_pool_slot: int | None = None
        self._cpu_param_staging_pool_slot: int | None = None
        self._gpu_grad_pool_slot: int | None = None
        self._gpu_param_pool_slot: int | None = None
        self._gpu_param_flat: torch.Tensor | None = None
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.compute_event = torch.cuda.Event()
        self.backward_event = torch.cuda.Event()
        self.grad_copy_event = torch.cuda.Event()
        self.loaded = False
        self.prefetched = False
        self.prefetch_event: torch.cuda.Event | None = None
        self.prefetch_request_id: int | None = None
        self._pending_backward_step = False
        self._module_backward_done = False
        self._backward_ready_reported = False
        self._seen_backward_params = 0
        self._expected_backward_params = 0
        self._grad_seen: set[nn.Parameter] = set()
        self._param_accumulation_counts: dict[nn.Parameter, int] = {}
        self._grad_copy_submitted = False
        self._forward_compute_start = 0.0
        self._backward_compute_start = 0.0
        self.step = 0
        self.params = (
            params
            if params is not None
            else [param for param in layer.parameters(recurse=True) if param.requires_grad]
        )
        self._te_fused_params: set[nn.Parameter] = set()
        if self.config.te_fused_main_grad and self.is_transformer_layer:
            param_ids = {id(param): param for param in self.params}
            for module in self.layer.modules():
                if not hasattr(module, "fuse_wgrad_accumulation"):
                    continue
                weight_names = getattr(module, "weight_names", ("weight",))
                for name in weight_names:
                    weight = getattr(module, name, None)
                    if isinstance(weight, nn.Parameter) and id(weight) in param_ids:
                        fused_param = param_ids[id(weight)]
                        self._te_fused_params.add(fused_param)
                        # TE snapshots this flag during forward, before the
                        # engine leases a gradient slot for backward.
                        fused_param.overwrite_main_grad = True
                module.fuse_wgrad_accumulation = True
        self._param_expected_accumulations = {
            param: int(getattr(param, "_slideformer_expected_grad_accumulations", 1))
            for param in self.params
        }
        self.execution_dtypes = {param: param.dtype for param in self.params}
        self.cpu_params: dict[nn.Parameter, torch.Tensor] = {}
        self.cpu_master: dict[nn.Parameter, torch.Tensor] = {}
        self.cpu_grads: dict[nn.Parameter, torch.Tensor] = {}
        self.cpu_execution_params: dict[nn.Parameter, torch.Tensor] = {}
        self.cpu_param_flat: torch.Tensor | None = None
        self.cpu_grad_flat: torch.Tensor | None = None
        self.cpu_param_staging_flat: torch.Tensor | None = None
        self.gpu_grad_flat: torch.Tensor | None = None
        self._param_flat_offsets: dict[nn.Parameter, tuple[int, int]] = {}
        self.gpu_params: dict[nn.Parameter, torch.Tensor] = {}
        self.gpu_buffer_pool: dict[nn.Parameter, list[torch.Tensor]] = {}
        self.gpu_buffer_cursor: dict[nn.Parameter, int] = {}
        self._flat_grad_copy_supported = False
        self._param_hook_handles = []
        for param in self.params:
            if hasattr(param, "register_post_accumulate_grad_hook"):
                self._param_hook_handles.append(
                    param.register_post_accumulate_grad_hook(
                        self._make_post_accumulate_param_hook(param)
                    )
                )
            else:
                if self._param_expected_accumulations[param] != 1:
                    raise RuntimeError(
                        "Tiled MLP requires register_post_accumulate_grad_hook support"
                    )
                self._param_hook_handles.append(param.register_hook(self._make_param_hook(param)))
        self._init_state()
        self.lifecycle_events: list[dict[str, Any]] = []
        self.lifecycle_summary: dict[str, Any] = {
            "layer_idx": self.layer_idx,
            "is_transformer_layer": self.is_transformer_layer,
            "event_counts": {},
            "durations": {},
            "bytes": {},
        }
        self._lifecycle_last_times: dict[str, float] = {}
        self.offload_params()
        self.optimizer_layer_idx = self.layer_optimizer.add_layer_params(
            self.layer_idx, self.params
        )

    def _record_lifecycle(self, event: str, **extra: Any) -> None:
        if not (
            self.config.profile_slideformer_lifecycle
            or self.config.profile_slideformer_lifecycle_summary
        ):
            return
        now = time.perf_counter()
        if self.config.profile_slideformer_lifecycle_summary:
            counts = self.lifecycle_summary["event_counts"]
            counts[event] = counts.get(event, 0) + 1
            for key in ("grad_d2h_bytes", "param_h2d_bytes"):
                if key in extra:
                    current = int(extra[key])
                    self.lifecycle_summary["bytes"][key] = max(
                        self.lifecycle_summary["bytes"].get(key, 0), current
                    )
            self._record_lifecycle_pairs(event, now)
            self._lifecycle_last_times[event] = now
            return
        if not self.config.profile_slideformer_lifecycle:
            return
        record: dict[str, Any] = {
            "layer_idx": self.layer_idx,
            "is_transformer_layer": self.is_transformer_layer,
            "event": event,
            "timestamp_kind": "cpu_perf_counter_s",
            "t": now,
        }
        record.update(extra)
        if self.config.profile_slideformer_lifecycle_memory and torch.cuda.is_available():
            record["cuda_memory_allocated"] = torch.cuda.memory_allocated(self.device)
            record["cuda_memory_reserved"] = torch.cuda.memory_reserved(self.device)
        self.lifecycle_events.append(record)

    def _record_lifecycle_pairs(self, event: str, now: float) -> None:
        pairs = {
            "grad_d2h_first_submit": ("grad_first_ready", "grad_first_ready_to_d2h_first_submit_s"),
            "grad_last_ready": ("grad_first_ready", "grad_first_ready_to_last_ready_s"),
            "prepare_grads_for_async_step": ("grad_last_ready", "grad_last_ready_to_prepare_s"),
            "optimizer_enqueue": ("grad_last_ready", "grad_last_ready_to_optimizer_enqueue_s"),
            "grad_d2h_complete": ("grad_first_ready", "grad_first_ready_to_d2h_complete_s"),
            "optimizer_start": ("optimizer_enqueue", "optimizer_enqueue_to_start_s"),
            "optimizer_end": ("optimizer_start", "optimizer_start_to_end_s"),
            "param_prefetch_complete": (
                "param_prefetch_submit",
                "param_prefetch_submit_to_complete_s",
            ),
            "param_h2d_sync_load_complete": ("param_h2d_sync_load_start", "param_h2d_sync_load_s"),
            "forward_end": ("forward_start", "forward_duration_s"),
            "backward_end": ("backward_start", "backward_duration_s"),
        }
        if event not in pairs:
            return
        start_event, name = pairs[event]
        start = self._lifecycle_last_times.get(start_event)
        if start is None:
            return
        duration = now - start
        durations = self.lifecycle_summary["durations"].setdefault(
            name, {"count": 0, "total_s": 0.0, "max_s": 0.0}
        )
        durations["count"] += 1
        durations["total_s"] += duration
        durations["max_s"] = max(durations["max_s"], duration)

    def _param_h2d_bytes(self) -> int:
        return sum(
            param.numel() * torch.empty((), dtype=self.execution_dtypes[param]).element_size()
            for param in self.params
        )

    def _grad_d2h_bytes(self) -> int:
        if self.config.skip_grad_d2h:
            return 0
        return sum(
            self.cpu_grads[param].numel() * self.cpu_grads[param].element_size()
            for param in self._grad_seen
        )

    def _cpu_empty_like(
        self, param: nn.Parameter, *, dtype: torch.dtype, pin_memory: bool = True
    ) -> torch.Tensor:
        try:
            return torch.empty(
                param.shape,
                dtype=dtype,
                device="cpu",
                pin_memory=self.config.pin_memory and pin_memory,
            )
        except RuntimeError:
            return torch.empty(param.shape, dtype=dtype, device="cpu")

    def _init_state(self) -> None:
        with torch.no_grad():
            if self.params and len({param.dtype for param in self.params}) == 1:
                total_numel = sum(param.numel() for param in self.params)
                execution_dtype = self.params[0].dtype
                master_dtype = torch.float32 if self.config.fp32_master_params else execution_dtype
                self.cpu_param_flat = torch.empty(total_numel, dtype=master_dtype, device="cpu")
                try:
                    if self.cpu_grad_pool is None:
                        self.cpu_grad_flat = torch.empty(
                            total_numel,
                            dtype=execution_dtype,
                            device="cpu",
                            pin_memory=self.config.pin_memory,
                        )
                    if self.cpu_param_staging_pool is None:
                        self.cpu_param_staging_flat = torch.empty(
                            total_numel,
                            dtype=execution_dtype,
                            device="cpu",
                            pin_memory=self.config.pin_memory,
                        )
                except RuntimeError:
                    if self.cpu_grad_pool is None:
                        self.cpu_grad_flat = torch.empty(
                            total_numel, dtype=execution_dtype, device="cpu"
                        )
                    if self.cpu_param_staging_pool is None:
                        self.cpu_param_staging_flat = torch.empty(
                            total_numel, dtype=execution_dtype, device="cpu"
                        )
                if self.cpu_grad_pool is not None:
                    self.cpu_grad_flat = self.cpu_grad_pool.tensors[0][:total_numel]
                if self.cpu_param_staging_pool is not None:
                    self.cpu_param_staging_flat = self.cpu_param_staging_pool.tensors[0][
                        :total_numel
                    ]
                assert self.cpu_grad_flat is not None
                assert self.cpu_param_staging_flat is not None
                offset = 0
                for param in self.params:
                    end = offset + param.numel()
                    cpu_param = self.cpu_param_flat[offset:end].view_as(param)
                    cpu_grad = self.cpu_grad_flat[offset:end].view_as(param)
                    execution_param = self.cpu_param_staging_flat[offset:end].view_as(param)
                    if self.cpu_param_staging_pool is not None:
                        execution_param.copy_(param.detach(), non_blocking=False)
                        cpu_param.copy_(execution_param, non_blocking=False)
                    else:
                        cpu_param.copy_(param.detach(), non_blocking=False)
                        execution_param.copy_(cpu_param, non_blocking=False)
                    self.cpu_params[param] = cpu_param
                    self.cpu_master[param] = cpu_param
                    self.cpu_grads[param] = cpu_grad
                    self.cpu_execution_params[param] = execution_param
                    self._param_flat_offsets[param] = (offset, end)
                    offset = end
                self._flat_grad_copy_supported = (
                    self.config.flat_grad_d2h or bool(self._te_fused_params)
                ) and all(
                    hasattr(param, "register_post_accumulate_grad_hook") for param in self.params
                )
                return
            for param in self.params:
                master_dtype = torch.float32 if self.config.fp32_master_params else param.dtype
                cpu_param = self._cpu_empty_like(param, dtype=master_dtype, pin_memory=False)
                cpu_param.copy_(param.detach(), non_blocking=False)
                self.cpu_params[param] = cpu_param
                # Backwards-compatible handle for tests/status checks; LayerAdam
                # owns optimizer state, while these CPU tensors are the params.
                self.cpu_master[param] = cpu_param
                self.cpu_grads[param] = self._cpu_empty_like(param, dtype=param.dtype)
                execution_param = self._cpu_empty_like(param, dtype=param.dtype)
                execution_param.copy_(cpu_param, non_blocking=False)
                self.cpu_execution_params[param] = execution_param

    def _bind_grad_buffer(self, tensor: torch.Tensor) -> None:
        total_numel = sum(param.numel() for param in self.params)
        self.cpu_grad_flat = tensor[:total_numel]
        for param in self.params:
            start, end = self._param_flat_offsets[param]
            self.cpu_grads[param] = self.cpu_grad_flat[start:end].view_as(param)

    def _acquire_grad_buffer(self) -> None:
        if self.cpu_grad_pool is None or self._cpu_grad_pool_slot is not None:
            return
        slot, tensor = self.cpu_grad_pool.acquire()
        self._cpu_grad_pool_slot = slot
        self._bind_grad_buffer(tensor)

    def _release_grad_buffer(self) -> None:
        if self.cpu_grad_pool is None or self._cpu_grad_pool_slot is None:
            pass
        else:
            slot = self._cpu_grad_pool_slot
            self._cpu_grad_pool_slot = None
            self.cpu_grad_pool.release(slot)
        if self.gpu_grad_pool is not None and self._gpu_grad_pool_slot is not None:
            slot = self._gpu_grad_pool_slot
            self._gpu_grad_pool_slot = None
            self.gpu_grad_pool.release(
                slot, event=self.grad_copy_event if self._grad_copy_submitted else None
            )
            self.gpu_grad_flat = None

    def _acquire_gpu_grad_buffer(self) -> None:
        if (
            not self._te_fused_params
            or self.gpu_grad_pool is None
            or self._gpu_grad_pool_slot is not None
        ):
            return
        slot, tensor = self.gpu_grad_pool.acquire()
        self._gpu_grad_pool_slot = slot
        assert self.cpu_grad_flat is not None
        self.gpu_grad_flat = tensor[: self.cpu_grad_flat.numel()]
        for param in self._te_fused_params:
            start, end = self._param_flat_offsets[param]
            param.main_grad = self.gpu_grad_flat[start:end].view_as(param)
            param.overwrite_main_grad = True

    def _bind_execution_staging_buffer(self, tensor: torch.Tensor) -> None:
        total_numel = sum(param.numel() for param in self.params)
        self.cpu_param_staging_flat = tensor[:total_numel]
        for param in self.params:
            start, end = self._param_flat_offsets[param]
            self.cpu_execution_params[param] = self.cpu_param_staging_flat[start:end].view_as(param)

    def _acquire_execution_staging(self) -> None:
        if self.cpu_param_staging_pool is None:
            return
        if self._cpu_param_staging_pool_slot is not None:
            raise RuntimeError("SlideFormer parameter staging buffer is already in use")
        slot, tensor = self.cpu_param_staging_pool.acquire()
        self._cpu_param_staging_pool_slot = slot
        self._bind_execution_staging_buffer(tensor)
        assert self.cpu_param_flat is not None
        assert self.cpu_param_staging_flat is not None
        self.cpu_param_staging_flat.copy_(self.cpu_param_flat, non_blocking=False)

    def _release_execution_staging(self, event: torch.cuda.Event | None = None) -> None:
        if self.cpu_param_staging_pool is None or self._cpu_param_staging_pool_slot is None:
            return
        slot = self._cpu_param_staging_pool_slot
        self._cpu_param_staging_pool_slot = None
        self.cpu_param_staging_pool.release(slot, event=event)

    def _refresh_execution_params(self) -> None:
        if self.cpu_param_staging_pool is not None:
            return
        if self.cpu_param_flat is not None and self.cpu_param_staging_flat is not None:
            self.cpu_param_staging_flat.copy_(self.cpu_param_flat, non_blocking=False)
            return
        for param in self.params:
            self.cpu_execution_params[param].copy_(self.cpu_params[param], non_blocking=False)

    def _step_layer_optimizer(self) -> None:
        if self.optimizer_layer_idx < 0:
            return
        # The native kernel parallelizes individual tensors better than one
        # heterogeneous layer-sized BF16 buffer on the current CPU. Keep the
        # storage flat for transfer locality, but update through tensor views.
        self.layer_optimizer.step_with_grad_views(
            self.optimizer_layer_idx, self.cpu_grads, self.cpu_params
        )
        with _TimedBlock(self.timing, "cpu_param_cast"):
            self._refresh_execution_params()
        self.step = self.layer_optimizer.param_groups[self.optimizer_layer_idx]["step"]

    def _get_gpu_param_buffer(self, param: nn.Parameter) -> torch.Tensor:
        if not self.config.double_buffer:
            return torch.empty(param.shape, dtype=self.execution_dtypes[param], device=self.device)
        pool = self.gpu_buffer_pool.setdefault(param, [])
        cursor = self.gpu_buffer_cursor.get(param, 0)
        if len(pool) < self.config.gpu_buffer_count:
            gpu_param = torch.empty(
                param.shape, dtype=self.execution_dtypes[param], device=self.device
            )
            pool.append(gpu_param)
        else:
            gpu_param = pool[cursor % len(pool)]
        self.gpu_buffer_cursor[param] = (cursor + 1) % max(self.config.gpu_buffer_count, 1)
        return gpu_param

    def _acquire_gpu_param_flat(self, stream: torch.cuda.Stream) -> torch.Tensor | None:
        if self.gpu_param_pool is None:
            return None
        if self._gpu_param_pool_slot is not None:
            raise RuntimeError("SlideFormer GPU parameter slot is already in use")
        slot, tensor = self.gpu_param_pool.acquire(stream)
        self._gpu_param_pool_slot = slot
        total_numel = sum(param.numel() for param in self.params)
        self._gpu_param_flat = tensor[:total_numel]
        return self._gpu_param_flat

    def _gpu_param_view(self, param: nn.Parameter) -> torch.Tensor:
        if self._gpu_param_flat is None:
            return self._get_gpu_param_buffer(param)
        start, end = self._param_flat_offsets[param]
        return self._gpu_param_flat[start:end].view_as(param)

    def _release_gpu_param_flat(self, stream: torch.cuda.Stream) -> None:
        if self.gpu_param_pool is None or self._gpu_param_pool_slot is None:
            return
        event = torch.cuda.Event()
        event.record(stream)
        slot = self._gpu_param_pool_slot
        self._gpu_param_pool_slot = None
        self._gpu_param_flat = None
        self.gpu_param_pool.release(slot, event=event)

    def _copy_tensor(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if self.config.chunk_size_mb <= 0:
            dst.copy_(src, non_blocking=True)
            return
        bytes_per_elem = max(src.element_size(), 1)
        chunk_numel = max((self.config.chunk_size_mb * 1024 * 1024) // bytes_per_elem, 1)
        if src.numel() <= chunk_numel:
            dst.copy_(src, non_blocking=True)
            return
        dst_flat = dst.view(-1)
        src_flat = src.view(-1)
        for start in range(0, src.numel(), chunk_numel):
            end = min(start + chunk_numel, src.numel())
            dst_flat[start:end].copy_(src_flat[start:end], non_blocking=True)

    def load_params(self) -> None:
        with _TimedBlock(self.timing, "load_params_total"):
            self._load_params()

    def _load_params(self) -> None:
        if not self.params:
            self.loaded = False
            self.prefetched = False
            return
        if self.loaded:
            return
        if self.prefetched:
            with _TimedBlock(self.timing, "prefetch_wait"):
                if self.h2d_scheduler is not None and self.prefetch_event is not None:
                    self.h2d_scheduler.record_use_wait(
                        kind="parameter_prefetch",
                        layer_idx=self.layer_idx,
                        bytes_=self._param_h2d_bytes(),
                        event=self.prefetch_event,
                        request_id=self.prefetch_request_id,
                    )
                else:
                    torch.cuda.current_stream(self.device).wait_stream(self.h2d_stream)
            self._record_lifecycle(
                "param_prefetch_complete", param_h2d_bytes=self._param_h2d_bytes()
            )
            self.loaded = True
            self.prefetched = False
            self.prefetch_event = None
            self.prefetch_request_id = None
            return
        with _TimedBlock(self.timing, "h2d_sync_load"):
            param_h2d_bytes = self._param_h2d_bytes()
            self._record_lifecycle("param_h2d_sync_load_start", param_h2d_bytes=param_h2d_bytes)
            if self.traffic_counters is not None:
                self.traffic_counters["parameter_sync_h2d_bytes"] = (
                    self.traffic_counters.get("parameter_sync_h2d_bytes", 0) + param_h2d_bytes
                )
                self.traffic_counters["parameter_sync_h2d_count"] = (
                    self.traffic_counters.get("parameter_sync_h2d_count", 0) + 1
                )
            h2d_stream = (
                self.h2d_scheduler.stream if self.h2d_scheduler is not None else self.h2d_stream
            )
            request_id = (
                self.h2d_scheduler.begin_submit(
                    kind="parameter_sync",
                    layer_idx=self.layer_idx,
                    bytes_=param_h2d_bytes,
                    allow_over_limit=True,
                )
                if self.h2d_scheduler is not None
                else None
            )
            self._acquire_execution_staging()
            self._acquire_gpu_param_flat(h2d_stream)
            with torch.cuda.stream(h2d_stream), torch.no_grad():
                for param in self.params:
                    gpu_param = self._gpu_param_view(param)
                    self.gpu_params[param] = gpu_param
                    self._copy_tensor(gpu_param, self.cpu_execution_params[param])
                    param.data = gpu_param
                    param.grad = None
                sync_event = torch.cuda.Event()
                sync_event.record(h2d_stream)
            self._release_execution_staging(sync_event)
            if self.h2d_scheduler is not None:
                self.h2d_scheduler.record_use_wait(
                    kind="parameter_sync",
                    layer_idx=self.layer_idx,
                    bytes_=param_h2d_bytes,
                    event=sync_event,
                    request_id=request_id,
                )
            else:
                torch.cuda.current_stream(self.device).wait_stream(self.h2d_stream)
            self._record_lifecycle(
                "param_h2d_sync_load_complete", param_h2d_bytes=self._param_h2d_bytes()
            )
        self.loaded = True
        self.prefetched = False

    def prefetch_params(self) -> None:
        with _TimedBlock(self.timing, "prefetch_submit"):
            self._prefetch_params()

    def _prefetch_params(self) -> None:
        if not self.params:
            return
        if self.loaded or self.prefetched:
            return
        param_h2d_bytes = self._param_h2d_bytes()
        request_id = None
        if self.h2d_scheduler is not None:
            request_id = self.h2d_scheduler.begin_submit(
                kind="parameter_prefetch", layer_idx=self.layer_idx, bytes_=param_h2d_bytes
            )
            if request_id is None:
                return
        self._record_lifecycle("param_prefetch_submit", param_h2d_bytes=self._param_h2d_bytes())
        if self.traffic_counters is not None:
            self.traffic_counters["parameter_prefetch_h2d_bytes"] = (
                self.traffic_counters.get("parameter_prefetch_h2d_bytes", 0) + param_h2d_bytes
            )
            self.traffic_counters["parameter_prefetch_count"] = (
                self.traffic_counters.get("parameter_prefetch_count", 0) + 1
            )
        h2d_stream = (
            self.h2d_scheduler.stream if self.h2d_scheduler is not None else self.h2d_stream
        )
        self._acquire_execution_staging()
        self._acquire_gpu_param_flat(h2d_stream)
        with torch.cuda.stream(h2d_stream), torch.no_grad():
            for param in self.params:
                gpu_param = self._gpu_param_view(param)
                self.gpu_params[param] = gpu_param
                self._copy_tensor(gpu_param, self.cpu_execution_params[param])
                param.data = gpu_param
                param.grad = None
            if self.h2d_scheduler is not None or self.cpu_param_staging_pool is not None:
                self.prefetch_event = torch.cuda.Event()
                self.prefetch_event.record(h2d_stream)
            if self.h2d_scheduler is not None:
                self.prefetch_request_id = request_id
        self._release_execution_staging(self.prefetch_event)
        self.prefetched = True

    def offload_params(self, *, force: bool = False) -> None:
        with _TimedBlock(self.timing, "offload_params"):
            self._offload_params(force=force)

    def _offload_params(self, *, force: bool = False) -> None:
        if not self.params:
            self.loaded = False
            self.prefetched = False
            return
        if self.keep_loaded_after_forward and self.loaded and not force:
            return
        if self.prefetched:
            if self.h2d_scheduler is not None and self.prefetch_event is not None:
                self.h2d_scheduler.record_use_wait(
                    kind="parameter_prefetch",
                    layer_idx=self.layer_idx,
                    bytes_=self._param_h2d_bytes(),
                    event=self.prefetch_event,
                    request_id=self.prefetch_request_id,
                )
            else:
                torch.cuda.current_stream(self.device).wait_stream(self.h2d_stream)
        current_stream = torch.cuda.current_stream(self.device)
        preserve_te_parameter_storage = bool(self._te_fused_params and self._pending_backward_step)
        with torch.no_grad():
            for param in self.params:
                gpu_param = self.gpu_params.get(param)
                if gpu_param is not None and gpu_param.is_cuda:
                    # Swapping param.data drops the owner's last reference when
                    # double buffering is disabled. Keep the allocation alive
                    # until all kernels already queued on this stream finish.
                    gpu_param.record_stream(current_stream)
                # TE custom autograd retains a weak reference to the Python
                # Parameter beyond its CUDA GEMM. Its execution storage is
                # stale after this point but remains harmless; the next H2D
                # bind replaces it with the updated slot. Mutating param.data
                # here races that saved state at large batch sizes.
                if not preserve_te_parameter_storage:
                    param.data = self.cpu_params[param]
                param.grad = None
        self.gpu_params.clear()
        self._release_gpu_param_flat(current_stream)
        self.loaded = False
        self.prefetched = False
        self.prefetch_event = None
        self.prefetch_request_id = None

    def offload_grads_and_step(self) -> None:
        try:
            with _TimedBlock(self.timing, "offload_grads_and_step_total"):
                self._offload_grads_and_step()
        finally:
            self._release_grad_buffer()

    def _offload_grads_and_step(self) -> None:
        if not self.params:
            self.offload_params()
            return

        with _TimedBlock(self.timing, "zero_missing_grads"):
            with torch.no_grad():
                for param in self.params:
                    if self.config.skip_grad_d2h or param not in self._grad_seen:
                        self.cpu_grads[param].zero_()
        self._wait_for_grad_copies()
        with _TimedBlock(self.timing, "sync_cpu_grads"):
            self._sync_cpu_grads_if_needed()
        self.offload_params(force=True)
        if self.optimizer_layer_idx >= 0 and not self.config.skip_cpu_adam_step:
            with _TimedBlock(self.timing, "cpu_adam_step"):
                self._record_lifecycle("optimizer_start")
                self._step_layer_optimizer()
                self._record_lifecycle("optimizer_end")
        self._pending_backward_step = False
        self._module_backward_done = False
        self._backward_ready_reported = False
        self._grad_seen.clear()
        self._param_accumulation_counts.clear()
        self._grad_copy_submitted = False

    def prepare_grads_for_async_step(self) -> None:
        """Detach GPU state after backward so a CPU worker can finish the step."""
        if not self.params:
            self.offload_params()
            return

        with _TimedBlock(self.timing, "zero_missing_grads"):
            with torch.no_grad():
                for param in self.params:
                    if self.config.skip_grad_d2h or param not in self._grad_seen:
                        self.cpu_grads[param].zero_()
        if self._grad_copy_submitted:
            self.grad_copy_event.record(self.d2h_stream)
            # Transformer Engine can keep consulting a Parameter's CUDA
            # storage while the enclosing backward hook unwinds. Replacing
            # param.data before its gradient copy completes makes the async
            # path race with that work and corrupts upstream gradients at
            # large batch sizes. Keep CPUAdam asynchronous, but make storage
            # ownership transfer obey the same D2H-before-offload order as
            # the synchronous path.
            if not self._te_fused_params:
                with _TimedBlock(self.timing, "async_grad_copy_before_param_release_wait"):
                    self.grad_copy_event.synchronize()
        self.offload_params(force=True)
        self._record_lifecycle(
            "prepare_grads_for_async_step", grad_d2h_bytes=self._grad_d2h_bytes()
        )

    def finish_async_step(self) -> None:
        """Wait for D2H and run LayerAdam outside the autograd thread."""
        if not self.params:
            return
        if self._grad_copy_submitted:
            with _TimedBlock(self.timing, "async_grad_copy_to_cpu_wait"):
                self.grad_copy_event.synchronize()
            self._record_lifecycle("grad_d2h_complete", grad_d2h_bytes=self._grad_d2h_bytes())
        with _TimedBlock(self.timing, "async_sync_cpu_grads"):
            self._sync_cpu_grads_if_needed()
        if self.optimizer_layer_idx >= 0 and not self.config.skip_cpu_adam_step:
            with _TimedBlock(self.timing, "async_cpu_adam_step"):
                self._record_lifecycle("optimizer_start")
                self._step_layer_optimizer()
                self._record_lifecycle("optimizer_end")
        self._pending_backward_step = False
        self._module_backward_done = False
        self._backward_ready_reported = False
        self._grad_seen.clear()
        self._param_accumulation_counts.clear()
        self._grad_copy_submitted = False

    def _submit_grad_copy(self, param: nn.Parameter, grad: torch.Tensor) -> None:
        if self.config.skip_grad_d2h:
            return
        source = grad.detach()
        if self.traffic_counters is not None:
            bytes_ = int(source.numel() * source.element_size())
            self.traffic_counters["grad_d2h_bytes"] = (
                self.traffic_counters.get("grad_d2h_bytes", 0) + bytes_
            )
            self.traffic_counters["grad_d2h_count"] = (
                self.traffic_counters.get("grad_d2h_count", 0) + 1
            )
        current_stream = torch.cuda.current_stream(self.device)
        with _TimedBlock(self.timing, "grad_copy_to_cpu_submit"):
            if not self._grad_copy_submitted:
                self._record_lifecycle("grad_d2h_first_submit")
            with torch.cuda.stream(self.d2h_stream):
                self.d2h_stream.wait_stream(current_stream)
                self.cpu_grads[param].copy_(source, non_blocking=True)
            if source.is_cuda:
                source.record_stream(self.d2h_stream)
        self._grad_copy_submitted = True

    def _ensure_gpu_grad_flat(self) -> torch.Tensor:
        if not self._flat_grad_copy_supported or self.cpu_grad_flat is None:
            raise RuntimeError("Layer-flat grad copy is not available for this layer")
        if self.gpu_grad_flat is None or self.gpu_grad_flat.numel() != self.cpu_grad_flat.numel():
            self.gpu_grad_flat = torch.empty(
                self.cpu_grad_flat.shape, dtype=self.cpu_grad_flat.dtype, device=self.device
            )
        return self.gpu_grad_flat

    def _stage_param_grad_to_flat(self, param: nn.Parameter, grad: torch.Tensor) -> None:
        gpu_grad_flat = self._ensure_gpu_grad_flat()
        start, end = self._param_flat_offsets[param]
        current_stream = torch.cuda.current_stream(self.device)
        source = grad.detach()
        with _TimedBlock(self.timing, "grad_pack_to_gpu_flat"):
            with torch.cuda.stream(self.d2h_stream):
                self.d2h_stream.wait_stream(current_stream)
                gpu_grad_flat[start:end].view_as(param).copy_(source, non_blocking=True)
            if source.is_cuda:
                source.record_stream(self.d2h_stream)

    def _submit_layer_grad_copy(self) -> None:
        if not self._flat_grad_copy_supported or self.cpu_grad_flat is None:
            return
        gpu_grad_flat = self._ensure_gpu_grad_flat()
        if self.traffic_counters is not None:
            bytes_ = int(gpu_grad_flat.numel() * gpu_grad_flat.element_size())
            self.traffic_counters["grad_d2h_bytes"] = (
                self.traffic_counters.get("grad_d2h_bytes", 0) + bytes_
            )
            self.traffic_counters["grad_d2h_count"] = (
                self.traffic_counters.get("grad_d2h_count", 0) + 1
            )
        current_stream = torch.cuda.current_stream(self.device)
        with _TimedBlock(self.timing, "grad_copy_to_cpu_submit"):
            with torch.cuda.stream(self.d2h_stream):
                self.d2h_stream.wait_stream(current_stream)
                self.cpu_grad_flat.copy_(gpu_grad_flat, non_blocking=True)
            gpu_grad_flat.record_stream(self.d2h_stream)
        self.gpu_grad_flat = None
        self._grad_copy_submitted = True

    def _wait_for_grad_copies(self) -> None:
        if not self._grad_copy_submitted:
            return
        with _TimedBlock(self.timing, "grad_copy_to_cpu_wait"):
            self.d2h_stream.synchronize()
        self._record_lifecycle("grad_d2h_complete", grad_d2h_bytes=self._grad_d2h_bytes())

    def _sync_cpu_grads_if_needed(self) -> None:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        world_size = dist.get_world_size()
        for param in self.params:
            grad = self.cpu_grads[param]
            work = grad.float().contiguous()
            dist.all_reduce(work, op=dist.ReduceOp.SUM)
            work.div_(world_size)
            grad.copy_(work.to(dtype=grad.dtype))

    def mark_backward_pending(self) -> None:
        self._acquire_grad_buffer()
        self._acquire_gpu_grad_buffer()
        self._pending_backward_step = bool(self.params)
        self._module_backward_done = False
        self._backward_ready_reported = False
        self._seen_backward_params = 0
        self._expected_backward_params = len(self.params)
        self._grad_seen.clear()
        self._param_accumulation_counts.clear()
        self._grad_copy_submitted = False
        self._record_lifecycle("backward_start")

    def mark_module_backward_done(self) -> None:
        if not self._pending_backward_step:
            return
        for param in self._te_fused_params:
            if param not in self._grad_seen:
                self._grad_seen.add(param)
                self._seen_backward_params += 1
        if self._flat_grad_copy_supported and self.gpu_grad_flat is not None:
            with torch.no_grad():
                for param in self.params:
                    if param in self._grad_seen:
                        continue
                    start, end = self._param_flat_offsets[param]
                    self.gpu_grad_flat[start:end].zero_()
        self._module_backward_done = True
        self._record_lifecycle("backward_end")
        self._maybe_finish_backward()

    def _maybe_finish_backward(self) -> None:
        module_completion_required = not (
            self.config.native_slideformer_backward_lifecycle and self.is_transformer_layer
        )
        if (
            self._pending_backward_step
            and (self._module_backward_done or not module_completion_required)
            and self._seen_backward_params >= self._expected_backward_params
            and not self._backward_ready_reported
        ):
            if os.environ.get("FLAGSCALE_SLIDEFORMER_DEBUG_HOOKS") == "1":
                tiled = {
                    name: (
                        self._param_accumulation_counts.get(param, 0),
                        self._param_expected_accumulations[param],
                    )
                    for name, param in self.layer.named_parameters()
                    if self._param_expected_accumulations.get(param, 1) > 1
                }
                print(
                    f"[SlideFormer hooks] layer={self.layer_idx} ready "
                    f"seen={self._seen_backward_params}/{self._expected_backward_params} "
                    f"tiled={tiled}",
                    flush=True,
                )
            self._backward_ready_reported = True
            self._submit_layer_grad_copy()
            self._record_lifecycle("grad_last_ready")
            self._record_lifecycle("grad_d2h_last_submit", grad_d2h_bytes=self._grad_d2h_bytes())
            if self.on_backward_ready is not None:
                self.on_backward_ready(self)
            else:
                self.offload_grads_and_step()

    def _make_param_hook(self, param: nn.Parameter):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if self._pending_backward_step:
                if not self._grad_seen:
                    self._record_lifecycle("grad_first_ready")
                self._submit_grad_copy(param, grad)
                self._grad_seen.add(param)
                self._seen_backward_params += 1
                self._maybe_finish_backward()
            return grad

        return hook

    def _make_post_accumulate_param_hook(self, param: nn.Parameter):
        def hook(_param: nn.Parameter) -> None:
            if not self._pending_backward_step or _param.grad is None:
                return
            count = self._param_accumulation_counts.get(param, 0) + 1
            self._param_accumulation_counts[param] = count
            if count < self._param_expected_accumulations[param]:
                return
            if not self._grad_seen:
                self._record_lifecycle("grad_first_ready")
            if not self._flat_grad_copy_supported:
                self._submit_grad_copy(param, _param.grad)
            else:
                self._stage_param_grad_to_flat(param, _param.grad)
                _param.grad = None
            if not self._flat_grad_copy_supported:
                _param.grad = None
            self._record_lifecycle("grad_gpu_clear")
            self._grad_seen.add(param)
            self._seen_backward_params += 1
            self._maybe_finish_backward()

        return hook


class MegatronSlideFormerEngine:
    """Single-GPU true-SlideFormer adapter for Megatron decoder layers."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        config: MegatronSlideFormerEngineConfig | None = None,
        layout: MegatronDecoderLayout | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config or MegatronSlideFormerEngineConfig()
        self._closed = False
        self.layout = layout or resolve_megatron_decoder_layout(model)
        assert_trainable_parameter_coverage(model, self.layout)
        self.managed_module_specs = self._collect_managed_module_specs()
        self.managed_modules = [spec.module for spec in self.managed_module_specs]
        self.layer_optimizer = self._build_layer_optimizer()
        self.timing = _TimingRecorder() if self.config.enable_timing else None
        # NVMe state movement has a separate ordered async queue. Keep that
        # path synchronous until its read/write dependencies are pipelined too.
        self._async_optimizer_enabled = bool(
            self.config.overlap_grad_d2h_cpu_adam
            and self.config.nvme_offload_fraction == 0.0
            and (not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1)
        )
        self._optimizer_queue: queue.Queue[_ManagedLayer | None] | None = None
        self._optimizer_workers: list[threading.Thread] = []
        self._optimizer_error: BaseException | None = None
        self._optimizer_error_lock = threading.Lock()
        self._optimizer_slots: threading.Semaphore | None = None
        if self._async_optimizer_enabled:
            if self.config.optimizer_pipeline_depth < 1:
                raise ValueError("optimizer_pipeline_depth must be at least 1")
            self._optimizer_queue = queue.Queue()
            if self.config.optimizer_backpressure:
                self._optimizer_slots = threading.Semaphore(self.config.optimizer_pipeline_depth)
            for worker_idx in range(self.config.optimizer_worker_count):
                worker = threading.Thread(
                    target=self._optimizer_worker_main,
                    name=f"slideformer-layeradam-{worker_idx}",
                    daemon=True,
                )
                worker.start()
                self._optimizer_workers.append(worker)
        self._managed_param_ids = {
            id(param)
            for module in self.managed_modules
            for param in module.parameters(recurse=True)
            if param.requires_grad
        }
        self.unmanaged_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in self._managed_param_ids
        ]
        self.unmanaged_optimizer = (
            torch.optim.Adam(
                self.unmanaged_params,
                lr=self.config.lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
            if self.unmanaged_params
            else None
        )
        self.activation_cpu_tensors: dict[int, list[torch.Tensor]] = {}
        self.activation_copy_events: dict[int, list[torch.cuda.Event | None]] = {}
        self.activation_prefetch_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self.activation_prefetch_events: dict[tuple[int, int], torch.cuda.Event] = {}
        self.activation_tensor_events: list[dict[str, Any]] = []
        self.activation_slot_cpu_tensors: dict[int, torch.Tensor] = {}
        self.activation_slot_copy_events: dict[int, torch.cuda.Event] = {}
        self.activation_slot_gpu_resident_tensors: dict[int, torch.Tensor] = {}
        self.activation_slot_prefetch_tensors: dict[int, torch.Tensor] = {}
        self.activation_slot_prefetch_events: dict[int, torch.cuda.Event] = {}
        self.activation_lifecycle_events: list[dict[str, Any]] = []
        self.traffic_counters: dict[str, int] = {}
        self.activation_stream = torch.cuda.Stream(device=device)
        self.h2d_scheduler = (
            _H2DScheduler(
                device=device,
                max_outstanding=self.config.max_outstanding_h2d,
                counters=self.traffic_counters,
            )
            if self.config.unified_h2d_scheduler
            else None
        )
        self.cpu_grad_pools, self.cpu_param_staging_pools = self._build_shared_cpu_pools()
        self.gpu_grad_pools = self._build_shared_gpu_grad_pools()
        self.gpu_param_pools, self.gpu_tied_param_pools = self._build_shared_gpu_param_pools()
        self.managed_layers = [
            _ManagedLayer(
                spec.module,
                layer_idx=idx,
                device=device,
                config=self.config,
                layer_optimizer=self.layer_optimizer,
                params=spec.params,
                keep_loaded_after_forward=spec.keep_loaded_after_forward,
                is_transformer_layer=spec.is_transformer_layer,
                on_backward_ready=self._mark_layer_backward_ready,
                timing=self.timing,
                traffic_counters=self.traffic_counters,
                h2d_scheduler=self.h2d_scheduler,
                cpu_grad_pool=self.cpu_grad_pools.get(self._spec_execution_dtype(spec)),
                cpu_param_staging_pool=self.cpu_param_staging_pools.get(
                    self._spec_execution_dtype(spec)
                ),
                gpu_grad_pool=self.gpu_grad_pools.get(self._spec_execution_dtype(spec)),
                gpu_param_pool=(
                    self.gpu_tied_param_pools
                    if self._tied_embedding_output and spec.module is self.layout.embedding
                    else self.gpu_param_pools
                ).get(self._spec_execution_dtype(spec)),
            )
            for idx, spec in enumerate(self.managed_module_specs)
        ]
        self._transformer_owner_by_module_id = {
            id(owner.layer): owner for owner in self.managed_layers if owner.is_transformer_layer
        }
        if self.config.activation_offload:
            self._wrap_activation_offload_forwards()
        self._ready_backward_layers: set[int] = set()
        self._hook_handles: list[Any] = []
        self._register_hooks()

    @staticmethod
    def _spec_execution_dtype(spec: _ManagedModuleSpec) -> torch.dtype | None:
        dtypes = {param.dtype for param in spec.params}
        return next(iter(dtypes)) if len(dtypes) == 1 else None

    def _build_shared_cpu_pools(
        self,
    ) -> tuple[dict[torch.dtype, _SharedCPUBufferPool], dict[torch.dtype, _SharedCPUBufferPool]]:
        if not self.config.shared_cpu_buffers:
            return {}, {}
        max_numel_by_dtype: dict[torch.dtype, int] = {}
        for spec in self.managed_module_specs:
            dtype = self._spec_execution_dtype(spec)
            if dtype is None:
                continue
            max_numel_by_dtype[dtype] = max(
                max_numel_by_dtype.get(dtype, 0), sum(param.numel() for param in spec.params)
            )
        grad_count = self.config.cpu_grad_buffer_count or (
            self.config.optimizer_pipeline_depth if self._async_optimizer_enabled else 1
        )
        grad_count = max(grad_count, 1)
        grad_pools = {
            dtype: _SharedCPUBufferPool(
                numel=numel, dtype=dtype, count=grad_count, pin_memory=self.config.pin_memory
            )
            for dtype, numel in max_numel_by_dtype.items()
            if numel
        }
        staging_pools = {
            dtype: _SharedCPUBufferPool(
                numel=numel,
                dtype=dtype,
                count=self.config.cpu_param_staging_buffer_count,
                pin_memory=self.config.pin_memory,
            )
            for dtype, numel in max_numel_by_dtype.items()
            if numel
        }
        return grad_pools, staging_pools

    def _build_shared_gpu_grad_pools(self) -> dict[torch.dtype, _SharedGPUBufferPool]:
        if not self.config.te_fused_main_grad:
            return {}
        max_numel_by_dtype: dict[torch.dtype, int] = {}
        for spec in self.managed_module_specs:
            if not spec.is_transformer_layer:
                continue
            dtype = self._spec_execution_dtype(spec)
            if dtype is None:
                continue
            max_numel_by_dtype[dtype] = max(
                max_numel_by_dtype.get(dtype, 0), sum(param.numel() for param in spec.params)
            )
        return {
            dtype: _SharedGPUBufferPool(
                numel=numel,
                dtype=dtype,
                count=max(self.config.optimizer_pipeline_depth, 1),
                device=self.device,
            )
            for dtype, numel in max_numel_by_dtype.items()
            if numel
        }

    def _build_shared_gpu_param_pools(
        self,
    ) -> tuple[
        dict[torch.dtype, _SharedGPUParameterPool], dict[torch.dtype, _SharedGPUParameterPool]
    ]:
        max_numel_by_dtype: dict[torch.dtype, int] = {}
        tied_numel_by_dtype: dict[torch.dtype, int] = {}
        for spec in self.managed_module_specs:
            dtype = self._spec_execution_dtype(spec)
            if dtype is None:
                continue
            numel = sum(param.numel() for param in spec.params)
            if self._tied_embedding_output and spec.module is self.layout.embedding:
                tied_numel_by_dtype[dtype] = max(tied_numel_by_dtype.get(dtype, 0), numel)
                continue
            max_numel_by_dtype[dtype] = max(max_numel_by_dtype.get(dtype, 0), numel)
        # Match SlideFormer's cache-unit ownership model: every managed
        # module leases one of two max-layer parameter units. This lets the
        # output projection remain resident after forward while the second
        # unit prefetches the preceding layer for backward. The optional
        # double-buffer mode uses the original SlideFormer three-unit depth.
        # A tied embedding/output weight remains resident in a dedicated slot
        # across the loss and output backward. Keeping that slot separate lets
        # Transformer/final-norm owners retain a true two-unit sliding window
        # without sizing all three units to the vocabulary projection.
        unit_count = max(self.config.gpu_buffer_count, 3 if self.config.double_buffer else 2)
        sliding_pools = {
            dtype: _SharedGPUParameterPool(
                numel=numel, dtype=dtype, count=unit_count, device=self.device
            )
            for dtype, numel in max_numel_by_dtype.items()
            if numel
        }
        tied_pools = {
            dtype: _SharedGPUParameterPool(numel=numel, dtype=dtype, count=1, device=self.device)
            for dtype, numel in tied_numel_by_dtype.items()
            if numel
        }
        return sliding_pools, tied_pools

    def _build_layer_optimizer(self) -> LayerAdam:
        distributed_cfg = SimpleNamespace(
            parallel_strategy="none", mode="single", world_size=1, gradient_average=True
        )
        return LayerAdam(
            lr=self.config.lr,
            bias_correction=self.config.bias_correction,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            adamw_mode=self.config.adamw_mode,
            fp32_optimizer_state=self.config.fp32_optimizer_state,
            num_layer=len(self.managed_modules),
            nvme_offload_fraction=self.config.nvme_offload_fraction,
            offload_dir=self.config.offload_dir,
            prefetch=self.config.prefetch,
            distributed_cfg=distributed_cfg,
            per_layer_cpu_adam=self.config.cpu_adam_per_layer_optimizer,
        )

    def _collect_managed_module_specs(self) -> list[_ManagedModuleSpec]:
        specs: list[_ManagedModuleSpec] = []
        seen_modules: set[int] = set()
        seen_params: set[int] = set()
        output_param_ids = {
            id(param)
            for param in (
                self.layout.output_layer.parameters(recurse=True)
                if self.layout.output_layer is not None
                else []
            )
        }

        def add(
            module: nn.Module | None,
            *,
            keep_loaded_after_forward: bool = False,
            is_transformer_layer: bool = False,
        ) -> None:
            if module is None or id(module) in seen_modules:
                return
            seen_modules.add(id(module))
            params: list[nn.Parameter] = []
            for param in module.parameters(recurse=True):
                if not param.requires_grad or id(param) in seen_params:
                    continue
                seen_params.add(id(param))
                params.append(param)
            specs.append(
                _ManagedModuleSpec(
                    module=module,
                    params=params,
                    keep_loaded_after_forward=keep_loaded_after_forward,
                    is_transformer_layer=is_transformer_layer,
                )
            )

        embedding_param_ids = {
            id(param)
            for param in (
                self.layout.embedding.parameters(recurse=True)
                if self.layout.embedding is not None
                else []
            )
        }
        # MCore normally implements tied embeddings by constructing the output
        # projection without its own weight and passing the embedding weight to
        # it at call time.  In that representation there is no literal
        # Parameter alias for the identity test below to find.
        tied_embedding_output = bool(
            getattr(self.layout.model, "share_embeddings_and_output_weights", False)
            or embedding_param_ids & output_param_ids
        )
        self._tied_embedding_output = tied_embedding_output
        add(self.layout.embedding, keep_loaded_after_forward=tied_embedding_output)
        for layer in self.layout.layers:
            add(layer, is_transformer_layer=True)
        add(self.layout.final_norm)
        # SlideFormer preloads the output projection as a backward-ready
        # cache unit and deliberately keeps it resident across the loss
        # forward. Releasing it here would force a second full-vocabulary
        # H2D load at the start of backward.
        add(self.layout.output_layer, keep_loaded_after_forward=True)
        return specs

    def _wrap_activation_offload_forwards(self) -> None:
        decoder_layers = list(self.layout.layers)
        total_layers = len(decoder_layers)
        for layer_idx, module in enumerate(decoder_layers):
            if hasattr(module, "_megatron_slideformer_original_forward"):
                continue
            original_forward = module.forward

            def wrapped_forward(
                *args: Any, _original_forward=original_forward, _layer_idx=layer_idx, **kwargs: Any
            ) -> Any:
                def add_backward_boundary(checkpoint_args: tuple[Any, ...]) -> tuple[Any, ...]:
                    module = self.layout.layers[_layer_idx]
                    owner = self._transformer_owner_by_module_id[id(module)]
                    if not owner._te_fused_params:
                        return checkpoint_args
                    hidden_states = checkpoint_args[0]
                    if not isinstance(hidden_states, torch.Tensor):
                        raise RuntimeError(
                            "TE fused SlideFormer expected the checkpoint's first argument "
                            "to be hidden_states"
                        )
                    callback = lambda: self._mark_checkpoint_backward_done(_layer_idx)
                    return (
                        _CheckpointBackwardBoundary.apply(hidden_states, callback),
                        *checkpoint_args[1:],
                    )

                if self.config.activation_backend == "slideformer-slot":
                    if args:
                        if kwargs:

                            def run_function(*flat_args: Any) -> Any:
                                return _original_forward(*flat_args, **kwargs)

                        else:
                            run_function = _original_forward
                        checkpoint_args = args
                    elif "hidden_states" in kwargs:
                        hidden_states = kwargs["hidden_states"]
                        recompute_kwargs = dict(kwargs)
                        del recompute_kwargs["hidden_states"]

                        def run_function(hidden: torch.Tensor) -> Any:
                            return _original_forward(hidden_states=hidden, **recompute_kwargs)

                        checkpoint_args = (hidden_states,)
                    else:
                        raise RuntimeError(
                            "slideformer-slot activation backend could not find hidden_states "
                            "as a positional input or keyword argument"
                        )
                    checkpoint_args = add_backward_boundary(checkpoint_args)
                    return slideformer_slot_checkpoint(
                        run_function,
                        layer_idx=_layer_idx,
                        device=self.device,
                        pin_memory=self.config.pin_memory,
                        stream=self.activation_stream,
                        cpu_slots=self.activation_slot_cpu_tensors,
                        copy_events=self.activation_slot_copy_events,
                        gpu_resident_slots=self.activation_slot_gpu_resident_tensors,
                        prefetch_slots=self.activation_slot_prefetch_tensors,
                        prefetch_events=self.activation_slot_prefetch_events,
                        traffic_counters=self.traffic_counters,
                        lifecycle_records=(
                            self.activation_lifecycle_events
                            if self.config.profile_activation_lifecycle
                            else None
                        ),
                        prefetch_previous=self.config.activation_slot_prefetch,
                        total_layers=total_layers,
                        gpu_window=self.config.activation_slot_gpu_window,
                        h2d_scheduler=self.h2d_scheduler,
                        args=checkpoint_args,
                    )
                checkpoint_args = args
                checkpoint_kwargs = kwargs
                if args:
                    checkpoint_args = add_backward_boundary(args)
                elif "hidden_states" in kwargs:
                    module = self.layout.layers[_layer_idx]
                    owner = self._transformer_owner_by_module_id[id(module)]
                    if not owner._te_fused_params:
                        checkpoint_kwargs = kwargs
                    else:
                        checkpoint_kwargs = dict(kwargs)
                        hidden_states = checkpoint_kwargs["hidden_states"]
                        if not isinstance(hidden_states, torch.Tensor):
                            raise RuntimeError("hidden_states must be a Tensor")
                        callback = lambda: self._mark_checkpoint_backward_done(_layer_idx)
                        checkpoint_kwargs["hidden_states"] = _CheckpointBackwardBoundary.apply(
                            hidden_states, callback
                        )
                with MegatronSlidingCheckpoint(
                    layer_idx=_layer_idx,
                    total_layers=total_layers,
                    layer_cpu_tensors=self.activation_cpu_tensors,
                    layer_copy_events=self.activation_copy_events,
                    prefetch_cache=self.activation_prefetch_cache,
                    prefetch_events=self.activation_prefetch_events,
                    device=self.device,
                    pin_memory=self.config.pin_memory,
                    stream=self.activation_stream,
                    min_numel=self.config.activation_offload_min_numel,
                    max_dim=self.config.activation_offload_max_dim,
                    offload_leaf_tensors=self.config.activation_offload_leaf_tensors,
                    max_offloaded_tensors=self.config.activation_offload_max_tensors_per_layer,
                    profile_records=(
                        self.activation_tensor_events
                        if self.config.profile_activation_saved_tensors
                        else None
                    ),
                ):
                    return checkpoint(
                        _original_forward,
                        *checkpoint_args,
                        use_reentrant=False,
                        **checkpoint_kwargs,
                    )

            module._megatron_slideformer_original_forward = original_forward
            module._megatron_slideformer_wrapped_forward = wrapped_forward
            module.forward = wrapped_forward

    def _register_hooks(self) -> None:
        for idx, layer_owner in enumerate(self.managed_layers):
            layer = layer_owner.layer
            self._hook_handles.append(
                layer.register_forward_pre_hook(self._make_forward_pre_hook(idx), with_kwargs=True)
            )
            self._hook_handles.append(
                layer.register_forward_hook(self._make_forward_hook(idx), with_kwargs=True)
            )
            self._hook_handles.append(
                layer.register_full_backward_pre_hook(self._make_backward_pre_hook(idx))
            )
            self._hook_handles.append(
                layer.register_full_backward_hook(self._make_backward_hook(idx))
            )

    def _make_forward_pre_hook(self, idx: int):
        def hook(
            module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            if idx == 0 and self.config.activation_offload:
                self.activation_prefetch_cache.clear()
                self.activation_prefetch_events.clear()
            layer_owner = self.managed_layers[idx]
            layer_owner._record_lifecycle("forward_start")
            in_backward_recompute = layer_owner._pending_backward_step
            if self.config.prefetch and not in_backward_recompute:
                with _TimedBlock(self.timing, "forward_prefetch_window"):
                    for next_idx in range(
                        idx + 1, min(idx + 1 + self.config.window_size, len(self.managed_layers))
                    ):
                        self.managed_layers[next_idx].prefetch_params()
            layer_owner.load_params()
            if self.timing is not None:
                layer_owner._forward_compute_start = time.perf_counter()
            # Megatron deliberately keeps tensors such as RoPE frequencies in
            # fp32 even when execution parameters are bf16. Casting every
            # floating input here accumulates visible error in deep models.
            return inputs, kwargs

        return hook

    def _make_forward_hook(self, idx: int):
        def hook(
            module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any], output: Any
        ) -> Any:
            if self.timing is not None:
                start = self.managed_layers[idx]._forward_compute_start
                if start:
                    self.timing.add("forward_compute", time.perf_counter() - start)
            self.managed_layers[idx]._record_lifecycle("forward_end")
            self.managed_layers[idx].compute_event.record(torch.cuda.current_stream(self.device))
            # Checkpoint recomputation runs a nested forward during backward.
            # Keep those parameters resident until their backward is complete.
            if (
                self.config.offload_after_forward
                and not self.managed_layers[idx]._pending_backward_step
            ):
                offload_idx = idx - max(self.config.window_size, 1) + 1
                if offload_idx >= 0:
                    self.managed_layers[offload_idx].offload_params()
            return output

        return hook

    def _make_backward_pre_hook(self, idx: int):
        def hook(module: nn.Module, grad_output: tuple[Any, ...]) -> None:
            if self.config.prefetch:
                with _TimedBlock(self.timing, "backward_prefetch_window"):
                    for prev_idx in range(max(idx - self.config.window_size, 0), idx):
                        self.managed_layers[prev_idx].prefetch_params()
            self.managed_layers[idx].load_params()
            self.managed_layers[idx].mark_backward_pending()
            if self.timing is not None:
                self.managed_layers[idx]._backward_compute_start = time.perf_counter()

        return hook

    def _make_backward_hook(self, idx: int):
        def hook(
            module: nn.Module, grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]
        ) -> None:
            if self.timing is not None:
                start = self.managed_layers[idx]._backward_compute_start
                if start:
                    self.timing.add("backward_compute", time.perf_counter() - start)
            if (
                self.config.activation_offload
                and self.managed_layers[idx].is_transformer_layer
                and self.managed_layers[idx]._te_fused_params
            ):
                return
            self.managed_layers[idx].mark_module_backward_done()

        return hook

    def _mark_checkpoint_backward_done(self, decoder_layer_idx: int) -> None:
        module = self.layout.layers[decoder_layer_idx]
        self._transformer_owner_by_module_id[id(module)].mark_module_backward_done()

    def _mark_layer_backward_ready(self, layer_owner: _ManagedLayer) -> None:
        if not self._async_optimizer_enabled:
            layer_owner.offload_grads_and_step()
            return
        layer_owner.prepare_grads_for_async_step()
        if self._optimizer_slots is not None:
            with _TimedBlock(self.timing, "async_optimizer_backpressure"):
                self._optimizer_slots.acquire()
        assert self._optimizer_queue is not None
        with _TimedBlock(self.timing, "async_optimizer_enqueue"):
            layer_owner._record_lifecycle("optimizer_enqueue")
            self._optimizer_queue.put(layer_owner)

    def _optimizer_worker_main(self) -> None:
        assert self._optimizer_queue is not None
        with torch.cuda.device(self.device):
            while True:
                layer_owner = self._optimizer_queue.get()
                try:
                    if layer_owner is None:
                        return
                    with self._optimizer_error_lock:
                        failed = self._optimizer_error is not None
                    if not failed:
                        layer_owner.finish_async_step()
                except BaseException as exc:
                    with self._optimizer_error_lock:
                        if self._optimizer_error is None:
                            self._optimizer_error = exc
                finally:
                    if layer_owner is not None:
                        layer_owner._release_grad_buffer()
                        if self._optimizer_slots is not None:
                            self._optimizer_slots.release()
                    self._optimizer_queue.task_done()

    def _raise_optimizer_error(self) -> None:
        with self._optimizer_error_lock:
            error = self._optimizer_error
            self._optimizer_error = None
        if error is not None:
            raise RuntimeError("SlideFormer asynchronous LayerAdam worker failed") from error

    def wait_for_completion(self) -> None:
        if self._optimizer_queue is not None:
            with _TimedBlock(self.timing, "async_optimizer_flush"):
                self._optimizer_queue.join()
            self._raise_optimizer_error()
        while self._ready_backward_layers:
            layer_idx = max(self._ready_backward_layers)
            self._ready_backward_layers.remove(layer_idx)
            self.managed_layers[layer_idx].offload_grads_and_step()

    def state_dict(self) -> dict[str, Any]:
        self.wait_for_completion()
        return {
            "layer_optimizer": self.layer_optimizer.state_dict(),
            "layers": [
                {
                    "step": owner.step,
                    "cpu_params": [
                        owner.cpu_params[param].detach().cpu().clone() for param in owner.params
                    ],
                }
                for owner in self.managed_layers
            ],
        }

    def prepare_for_checkpoint(self) -> None:
        """Flush work and expose CPU master params to Megatron's model state dict."""
        self.wait_for_completion()
        for owner in self.managed_layers:
            owner.offload_params(force=True)

    def optimizer_checkpoint_state(self) -> dict[str, Any]:
        """Return only state not already covered by Megatron's model checkpoint."""
        self.prepare_for_checkpoint()
        return {
            "version": 1,
            "layer_optimizer": self.layer_optimizer.state_dict(),
            "owner_steps": [owner.step for owner in self.managed_layers],
        }

    def finish_checkpoint_load(
        self, optimizer_state: dict[str, Any] | None = None, *, load_optimizer: bool = True
    ) -> None:
        """Synchronize engine-owned buffers after Megatron restores model weights."""
        self.wait_for_completion()
        if load_optimizer:
            if optimizer_state is None or "layer_optimizer" not in optimizer_state:
                raise KeyError("SlideFormer checkpoint does not contain LayerAdam state")
            version = int(optimizer_state.get("version", 0))
            if version != 1:
                raise ValueError(f"Unsupported SlideFormer checkpoint version: {version}")
            self.layer_optimizer.load_state_dict(optimizer_state["layer_optimizer"])
            owner_steps = optimizer_state.get("owner_steps", [])
            if owner_steps and len(owner_steps) != len(self.managed_layers):
                raise ValueError(
                    "SlideFormer checkpoint owner count mismatch: "
                    f"{len(owner_steps)} != {len(self.managed_layers)}"
                )
        else:
            owner_steps = []
            for group in self.layer_optimizer.param_groups:
                group["step"] = 0
            for state_buffer in (
                self.layer_optimizer.exp_avg_flat,
                self.layer_optimizer.exp_avg_sq_flat,
            ):
                for tensor in state_buffer.values():
                    tensor.zero_()

        for index, owner in enumerate(self.managed_layers):
            owner.offload_params(force=True)
            if owner.cpu_grad_flat is not None:
                owner.cpu_grad_flat.zero_()
            else:
                for grad in owner.cpu_grads.values():
                    grad.zero_()
            if load_optimizer:
                if owner_steps:
                    owner.step = int(owner_steps[index])
                elif owner.optimizer_layer_idx >= 0:
                    owner.step = int(
                        self.layer_optimizer.param_groups[owner.optimizer_layer_idx]["step"]
                    )
            else:
                owner.step = 0
            owner._refresh_execution_params()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        layers = state_dict.get("layers", [])
        if len(layers) != len(self.managed_layers):
            raise ValueError(
                f"SlideFormer checkpoint layer count mismatch: "
                f"{len(layers)} != {len(self.managed_layers)}"
            )
        for owner, layer_state in zip(self.managed_layers, layers, strict=True):
            owner.step = int(layer_state.get("step", 0))
            cpu_params = layer_state.get("cpu_params", [])
            if len(cpu_params) != len(owner.params):
                raise ValueError("SlideFormer checkpoint parameter count mismatch")
            for param, source_param in zip(owner.params, cpu_params, strict=True):
                owner.cpu_params[param].copy_(
                    source_param.to(dtype=owner.cpu_params[param].dtype), non_blocking=False
                )
            owner.offload_params(force=True)
            owner._refresh_execution_params()
        self.layer_optimizer.load_state_dict(state_dict["layer_optimizer"])

    def save_checkpoint(
        self, checkpoint_dir: str | Path, iteration: int, opt_param_scheduler=None
    ) -> Path:
        checkpoint_root = Path(checkpoint_dir) / "slideformer"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 2,
            "iteration": iteration,
            "engine": self.state_dict(),
            "opt_param_scheduler": (
                opt_param_scheduler.state_dict() if opt_param_scheduler is not None else None
            ),
        }
        path = checkpoint_root / f"iter_{iteration:07d}.pt"
        latest_path = checkpoint_root / "latest.pt"
        torch.save(state, path)
        torch.save(state, latest_path)
        return path

    def load_checkpoint(self, checkpoint_dir: str | Path, opt_param_scheduler=None) -> int | None:
        latest_path = Path(checkpoint_dir) / "slideformer" / "latest.pt"
        if not latest_path.exists():
            return None
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["engine"])
        scheduler_state = state.get("opt_param_scheduler")
        if opt_param_scheduler is not None and scheduler_state is not None:
            opt_param_scheduler.load_state_dict(scheduler_state)
        return int(state.get("iteration", 0))

    def step_unmanaged_params(self) -> None:
        """Update Megatron-owned trainable params not handled by layer offload.

        The expected true-SlideFormer path covers embedding, decoder layers,
        final norm, and output layer with LayerAdam. This fallback only exists
        for future Megatron modules that expose extra trainable parameters.
        """
        self.wait_for_completion()
        if self.unmanaged_optimizer is None:
            return
        self.unmanaged_optimizer.step()

    def zero_unmanaged_grads(self) -> None:
        if self.unmanaged_optimizer is not None:
            self.unmanaged_optimizer.zero_grad(set_to_none=True)
        # Prime the first cache unit before the embedding forward, mirroring
        # SlideFormerOffloader.forward(). Subsequent hooks keep the two-unit
        # queue one layer ahead in both directions.
        if self.config.prefetch and self.managed_layers:
            self.managed_layers[0].prefetch_params()

    def timing_summary(self) -> dict[str, Any]:
        if self.timing is None:
            return {}
        return self.timing.summary()

    def lifecycle_timeline(self) -> list[dict[str, Any]]:
        if not self.config.profile_slideformer_lifecycle:
            return []
        events: list[dict[str, Any]] = []
        for owner in self.managed_layers:
            events.extend(owner.lifecycle_events)
        events.sort(key=lambda event: event["t"])
        if not events:
            return events
        origin = events[0]["t"]
        for event in events:
            event["relative_t_s"] = event["t"] - origin
            del event["t"]
        return events

    def lifecycle_summary(self) -> dict[str, Any]:
        if not self.config.profile_slideformer_lifecycle_summary:
            return {}
        layers = [owner.lifecycle_summary for owner in self.managed_layers]
        aggregate: dict[str, dict[str, float | int]] = {}
        event_counts: dict[str, int] = {}
        for layer in layers:
            for event, count in layer.get("event_counts", {}).items():
                event_counts[event] = event_counts.get(event, 0) + int(count)
            for name, stats in layer.get("durations", {}).items():
                target = aggregate.setdefault(name, {"count": 0, "total_s": 0.0, "max_s": 0.0})
                target["count"] += int(stats["count"])
                target["total_s"] += float(stats["total_s"])
                target["max_s"] = max(float(target["max_s"]), float(stats["max_s"]))
        for stats in aggregate.values():
            count = int(stats["count"])
            stats["avg_ms"] = (float(stats["total_s"]) * 1000.0 / count) if count else 0.0
            stats["max_ms"] = float(stats["max_s"]) * 1000.0
        return {
            "timestamp_kind": "cpu_perf_counter_s",
            "event_counts": event_counts,
            "durations": aggregate,
            "layers": layers,
        }

    def overlap_config_summary(self) -> dict[str, Any]:
        return self.config.overlap_config_summary()

    def native_overlap_semantic_gaps(self) -> list[dict[str, str]]:
        return self.config.native_overlap_semantic_gaps()

    def backward_lifecycle_semantic_gaps(self) -> list[dict[str, str]]:
        return self.config.backward_lifecycle_semantic_gaps()

    def activation_backend_semantic_gaps(self) -> list[dict[str, str]]:
        return self.config.activation_backend_semantic_gaps()

    def traffic_summary(self) -> dict[str, Any]:
        counters = dict(self.traffic_counters)
        gib = {
            key.replace("_bytes", "_gib"): value / 1024**3
            for key, value in counters.items()
            if key.endswith("_bytes")
        }
        total_h2d_bytes = (
            counters.get("activation_restore_h2d_bytes", 0)
            + counters.get("activation_prefetch_h2d_bytes", 0)
            + counters.get("parameter_prefetch_h2d_bytes", 0)
            + counters.get("parameter_sync_h2d_bytes", 0)
        )
        total_d2h_bytes = counters.get("activation_offload_d2h_bytes", 0) + counters.get(
            "grad_d2h_bytes", 0
        )
        h2d_scheduler = (
            self.h2d_scheduler.summary()
            if self.h2d_scheduler is not None
            else {
                "enabled": False,
                "max_outstanding_h2d": None,
                "outstanding_peak": None,
                "outstanding_average": None,
                "pending_queue_peak": None,
                "total_exposed_wait_ms": None,
                "by_kind": {},
            }
        )
        return {
            "counters": counters,
            "gib": gib,
            "total_h2d_gib": total_h2d_bytes / 1024**3,
            "total_d2h_gib": total_d2h_bytes / 1024**3,
            "h2d_scheduler": h2d_scheduler,
        }

    @staticmethod
    def _unique_cpu_storage_bytes(tensors: list[torch.Tensor]) -> int:
        storages: dict[tuple[int, int], int] = {}
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                continue
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            storages[key] = storage.nbytes()
        return sum(storages.values())

    def cpu_memory_summary(self) -> dict[str, Any]:
        """Account for engine-owned CPU tensor storage without double-counting views."""

        categories = {
            "master_parameters": [
                tensor for owner in self.managed_layers for tensor in owner.cpu_params.values()
            ],
            "gradients": [
                *(tensor for pool in self.cpu_grad_pools.values() for tensor in pool.tensors),
                *(
                    tensor
                    for owner in self.managed_layers
                    if owner.cpu_grad_pool is None
                    for tensor in owner.cpu_grads.values()
                ),
            ],
            "execution_parameter_staging": [
                *(
                    tensor
                    for pool in self.cpu_param_staging_pools.values()
                    for tensor in pool.tensors
                ),
                *(
                    tensor
                    for owner in self.managed_layers
                    if owner.cpu_param_staging_pool is None
                    for tensor in owner.cpu_execution_params.values()
                ),
            ],
            "optimizer_exp_avg": list(self.layer_optimizer.exp_avg_flat.values()),
            "optimizer_exp_avg_sq": list(self.layer_optimizer.exp_avg_sq_flat.values()),
            "activation_slots": [
                *self.activation_slot_cpu_tensors.values(),
                *(tensor for tensors in self.activation_cpu_tensors.values() for tensor in tensors),
            ],
        }
        category_bytes = {
            name: self._unique_cpu_storage_bytes(tensors) for name, tensors in categories.items()
        }
        all_tensors = [tensor for tensors in categories.values() for tensor in tensors]
        total_bytes = self._unique_cpu_storage_bytes(all_tensors)
        return {
            "bytes": {**category_bytes, "total_unique": total_bytes},
            "gib": {
                **{name: value / 1024**3 for name, value in category_bytes.items()},
                "total_unique": total_bytes / 1024**3,
            },
        }

    def activation_saved_tensor_summary(self, *, limit: int = 20) -> dict[str, Any] | None:
        if not self.config.profile_activation_saved_tensors:
            return None
        pack_events = [
            event
            for event in self.activation_tensor_events
            if event["event"] in {"pack_keep", "pack_offload"}
        ]
        offloaded = [event for event in pack_events if event["offloaded"]]
        kept = [event for event in pack_events if not event["offloaded"]]
        largest = sorted(pack_events, key=lambda event: event["bytes"], reverse=True)[:limit]
        return {
            "event_count": len(self.activation_tensor_events),
            "packed_count": len(pack_events),
            "offloaded_count": len(offloaded),
            "kept_count": len(kept),
            "offloaded_bytes": sum(event["bytes"] for event in offloaded),
            "kept_bytes": sum(event["bytes"] for event in kept),
            "largest_packed_tensors": largest,
        }

    def activation_lifecycle_summary(self, *, limit: int = 20) -> dict[str, Any] | None:
        if not self.config.profile_activation_lifecycle:
            return None
        events = self.activation_lifecycle_events
        by_layer: dict[int, dict[str, Any]] = {}
        offloaded_bytes = 0
        for event in events:
            layer_idx = int(event["layer_idx"])
            layer = by_layer.setdefault(
                layer_idx,
                {
                    "layer_idx": layer_idx,
                    "event_counts": {},
                    "offloaded_bytes": 0,
                    "max_allocated_gb": 0.0,
                },
            )
            counts = layer["event_counts"]
            counts[event["event"]] = counts.get(event["event"], 0) + 1
            if event["event"] == "activation_d2h_submit":
                bytes_ = int(event.get("bytes", 0))
                layer["offloaded_bytes"] += bytes_
                offloaded_bytes += bytes_
            if "cuda_memory_allocated" in event:
                layer["max_allocated_gb"] = max(
                    float(layer["max_allocated_gb"]),
                    float(event["cuda_memory_allocated"]) / 1024**3,
                )
        largest = sorted(
            (event for event in events if "bytes" in event and "shape" in event),
            key=lambda event: int(event.get("bytes", 0)),
            reverse=True,
        )[:limit]
        timeline = list(events)
        if timeline:
            origin = timeline[0]["t"]
            timeline = [dict(event) for event in timeline]
            for event in timeline:
                event["relative_t_s"] = event["t"] - origin
                del event["t"]
        return {
            "event_count": len(events),
            "total_offloaded_bytes": offloaded_bytes,
            "layers": [by_layer[key] for key in sorted(by_layer)],
            "largest_boundary_activations": largest,
            "timeline": timeline[: 10 * limit],
        }

    def reset_timing(self) -> None:
        if self.timing is not None:
            self.timing.reset()
        if self.config.profile_slideformer_lifecycle:
            for owner in self.managed_layers:
                owner.lifecycle_events.clear()
        if self.config.profile_slideformer_lifecycle_summary:
            for owner in self.managed_layers:
                owner.lifecycle_summary = {
                    "layer_idx": owner.layer_idx,
                    "is_transformer_layer": owner.is_transformer_layer,
                    "event_counts": {},
                    "durations": {},
                    "bytes": {},
                }
                owner._lifecycle_last_times.clear()
        if self.config.profile_activation_saved_tensors:
            self.activation_tensor_events.clear()
        if self.config.profile_activation_lifecycle:
            self.activation_lifecycle_events.clear()
        self.traffic_counters.clear()
        if self.h2d_scheduler is not None:
            self.h2d_scheduler.reset()
        self.activation_slot_gpu_resident_tensors.clear()
        self.activation_slot_prefetch_tensors.clear()
        self.activation_slot_prefetch_events.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.current_stream(self.device).wait_stream(self.activation_stream)
        if self.h2d_scheduler is not None:
            self.h2d_scheduler.wait_stream()
        completion_error: BaseException | None = None
        try:
            self.wait_for_completion()
        except BaseException as exc:
            completion_error = exc
        if self._optimizer_queue is not None and self._optimizer_workers:
            for _ in self._optimizer_workers:
                self._optimizer_queue.put(None)
            self._optimizer_queue.join()
            for worker in self._optimizer_workers:
                worker.join()
            self._optimizer_workers.clear()
            self._optimizer_queue = None
        for owner in self.managed_layers:
            torch.cuda.current_stream(self.device).wait_stream(owner.h2d_stream)
            torch.cuda.current_stream(self.device).wait_stream(owner.d2h_stream)
        torch.cuda.synchronize(self.device)

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for owner in self.managed_layers:
            for handle in owner._param_hook_handles:
                handle.remove()
            owner._param_hook_handles.clear()
            owner.offload_params()

        for module in self.managed_modules:
            original_forward = getattr(module, "_megatron_slideformer_original_forward", None)
            wrapped_forward = getattr(module, "_megatron_slideformer_wrapped_forward", None)
            if original_forward is not None and module.forward is wrapped_forward:
                module.forward = original_forward
                delattr(module, "_megatron_slideformer_original_forward")
                delattr(module, "_megatron_slideformer_wrapped_forward")

        self.activation_prefetch_cache.clear()
        self.activation_prefetch_events.clear()
        self.activation_cpu_tensors.clear()
        self.activation_copy_events.clear()
        self.activation_slot_cpu_tensors.clear()
        self.activation_slot_copy_events.clear()
        self.activation_slot_gpu_resident_tensors.clear()
        self.activation_slot_prefetch_tensors.clear()
        self.activation_slot_prefetch_events.clear()
        for module in getattr(self, "_registration_modules", (self.model,)):
            if getattr(module, "_megatron_slideformer_engine", None) is self:
                delattr(module, "_megatron_slideformer_engine")
        if completion_error is not None:
            raise completion_error

    def __del__(self) -> None:
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            pass


def apply_true_megatron_slideformer(
    model: nn.Module,
    *,
    device: torch.device | None = None,
    config: MegatronSlideFormerEngineConfig | None = None,
) -> MegatronSlideFormerEngine:
    device = device or torch.device("cuda", torch.cuda.current_device())
    engine = MegatronSlideFormerEngine(model, device=device, config=config)
    registration_modules = [model]
    if engine.layout.model is not model:
        registration_modules.append(engine.layout.model)
    engine._registration_modules = tuple(registration_modules)
    for module in registration_modules:
        module._megatron_slideformer_engine = engine
    return engine
