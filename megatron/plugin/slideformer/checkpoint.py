from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import torch
from torch.autograd.graph import saved_tensors_hooks


class MegatronSlidingCheckpoint(saved_tensors_hooks):
    """SlideFormer-style activation offload hooks for Megatron modules.

    This follows the SlideFormer saved-tensor hook design, but uses dynamic
    per-layer CPU buffers because Megatron/Qwen-style layers may save more than
    the fixed hidden-state/mask pair used by the original HF wrapper.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        total_layers: int,
        layer_cpu_tensors: dict[int, list[torch.Tensor]],
        layer_copy_events: dict[int, list[torch.cuda.Event | None]],
        prefetch_cache: OrderedDict[tuple[int, int], torch.Tensor],
        prefetch_events: dict[tuple[int, int], torch.cuda.Event],
        device: torch.device,
        pin_memory: bool = True,
        stream: torch.cuda.Stream | None = None,
        min_numel: int = 1,
        max_dim: int = 99,
        offload_leaf_tensors: bool = False,
        max_offloaded_tensors: int = -1,
        profile_records: list[dict[str, Any]] | None = None,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MegatronSlidingCheckpoint requires CUDA")

        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.layer_cpu_tensors = layer_cpu_tensors
        self.layer_copy_events = layer_copy_events
        self.prefetch_cache = prefetch_cache
        self.prefetch_events = prefetch_events
        self.device = device
        self.pin_memory = pin_memory
        self.stream = stream or torch.cuda.Stream(device=device)
        self.min_numel = min_numel
        self.max_dim = max_dim
        self.offload_leaf_tensors = offload_leaf_tensors
        self.max_offloaded_tensors = max_offloaded_tensors
        self.profile_records = profile_records
        self.pack_counter = 0
        self.unpack_counter = 0
        self.offload_counter = 0
        self.pre_pack_event = torch.cuda.Event()
        self.pre_unpack_event = torch.cuda.Event()

        def pack_hook(
            tensor: torch.Tensor,
        ) -> tuple[int, torch.device, torch.Tensor, torch.cuda.Event | None, bool]:
            tensor_idx = self.pack_counter
            self.pack_counter += 1
            if (
                not tensor.is_cuda
                or tensor.numel() == 0
                or not tensor.is_floating_point()
                or tensor.numel() < self.min_numel
                or tensor.dim() > self.max_dim
                or (tensor.is_leaf and tensor.requires_grad and not self.offload_leaf_tensors)
                or (
                    self.max_offloaded_tensors >= 0
                    and self.offload_counter >= self.max_offloaded_tensors
                )
            ):
                self._record_tensor_event("pack_keep", tensor_idx, tensor, offloaded=False)
                return tensor_idx, tensor.device, tensor, None, False
            self.offload_counter += 1
            self._record_tensor_event("pack_offload", tensor_idx, tensor, offloaded=True)

            tensors = self.layer_cpu_tensors.setdefault(self.layer_idx, [])
            slot_idx = tensor_idx
            if slot_idx < len(tensors) and tensors[slot_idx].shape == tensor.shape:
                cpu_tensor = tensors[slot_idx]
                if cpu_tensor.dtype != tensor.dtype:
                    cpu_tensor = self._new_cpu_tensor(tensor)
                    tensors[slot_idx] = cpu_tensor
            else:
                cpu_tensor = self._new_cpu_tensor(tensor)
                if slot_idx < len(tensors):
                    tensors[slot_idx] = cpu_tensor
                else:
                    while len(tensors) < slot_idx:
                        tensors.append(torch.empty(0, dtype=tensor.dtype))
                    tensors.append(cpu_tensor)

            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready_event)
                cpu_tensor.copy_(tensor.detach(), non_blocking=self.pin_memory)
                tensor.record_stream(self.stream)
                copy_event = torch.cuda.Event()
                copy_event.record(self.stream)
            events = self.layer_copy_events.setdefault(self.layer_idx, [])
            if slot_idx < len(events):
                events[slot_idx] = copy_event
            else:
                while len(events) < slot_idx:
                    events.append(None)
                events.append(copy_event)

            return tensor_idx, tensor.device, cpu_tensor, copy_event, True

        def unpack_hook(
            packed: tuple[int, torch.device, torch.Tensor, torch.cuda.Event | None, bool]
        ) -> torch.Tensor:
            tensor_idx, original_device, cpu_tensor, copy_event, offloaded = packed
            if not offloaded:
                self._record_tensor_event("unpack_keep", tensor_idx, cpu_tensor, offloaded=False)
                return cpu_tensor
            if copy_event is not None:
                torch.cuda.current_stream(self.device).wait_event(copy_event)
            if self.unpack_counter == 0:
                self.pre_unpack_event.record(torch.cuda.current_stream(self.device))

            key = (self.layer_idx, tensor_idx)
            prefetched = self.prefetch_cache.pop(key, None)
            event = self.prefetch_events.pop(key, None)
            if prefetched is not None:
                if event is not None:
                    torch.cuda.current_stream(self.device).wait_event(event)
                result = prefetched
            else:
                result = cpu_tensor.to(original_device, non_blocking=self.pin_memory)

            self.unpack_counter += 1
            self._record_tensor_event("unpack_restore", tensor_idx, result, offloaded=True)
            return result

        super().__init__(pack_hook, unpack_hook)

    def _record_tensor_event(
        self, event: str, tensor_idx: int, tensor: torch.Tensor, *, offloaded: bool
    ) -> None:
        if self.profile_records is None:
            return
        self.profile_records.append(
            {
                "layer_idx": self.layer_idx,
                "tensor_idx": tensor_idx,
                "event": event,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
                "bytes": int(tensor.numel() * tensor.element_size()),
                "offloaded": offloaded,
            }
        )

    def _new_cpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        try:
            return torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                layout=tensor.layout,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError:
            return torch.empty(tensor.shape, dtype=tensor.dtype, layout=tensor.layout, device="cpu")

    def _prefetch_previous_layer(self) -> None:
        previous_idx = self.layer_idx - 1
        if previous_idx < 0:
            return
        previous_tensors = self.layer_cpu_tensors.get(previous_idx)
        if not previous_tensors:
            return
        previous_events = self.layer_copy_events.get(previous_idx, [])
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.pre_unpack_event)
            for tensor_idx, cpu_tensor in enumerate(previous_tensors):
                if cpu_tensor.numel() == 0:
                    continue
                key = (previous_idx, tensor_idx)
                if key in self.prefetch_cache:
                    continue
                if tensor_idx < len(previous_events) and previous_events[tensor_idx] is not None:
                    self.stream.wait_event(previous_events[tensor_idx])
                gpu_tensor = torch.empty_like(cpu_tensor, device=self.device)
                gpu_tensor.copy_(cpu_tensor, non_blocking=self.pin_memory)
                event = torch.cuda.Event()
                event.record(self.stream)
                self.prefetch_cache[key] = gpu_tensor
                self.prefetch_events[key] = event


class _SlideFormerSlotCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        run_function: Any,
        layer_idx: int,
        device: torch.device,
        pin_memory: bool,
        stream: torch.cuda.Stream,
        cpu_slots: dict[int, torch.Tensor],
        copy_events: dict[int, torch.cuda.Event],
        gpu_resident_slots: dict[int, torch.Tensor],
        prefetch_slots: dict[int, torch.Tensor],
        prefetch_events: dict[int, torch.cuda.Event],
        traffic_counters: dict[str, int],
        lifecycle_records: list[dict[str, Any]] | None,
        prefetch_previous: bool,
        total_layers: int,
        gpu_window: int,
        h2d_scheduler: Any | None,
        *args: Any,
    ) -> torch.Tensor:
        if not args or not torch.is_tensor(args[0]) or not args[0].is_cuda:
            raise RuntimeError(
                "slideformer-slot activation backend expects CUDA hidden_states as the first tensor input"
            )
        hidden_states = args[0]
        if not hidden_states.is_floating_point():
            raise RuntimeError("slideformer-slot activation backend expects floating hidden_states")

        def record(event: str, **extra: Any) -> None:
            if lifecycle_records is None:
                return
            payload = {
                "layer_idx": layer_idx,
                "event": event,
                "timestamp_kind": "cpu_perf_counter_s",
                "t": time.perf_counter(),
            }
            payload.update(extra)
            if torch.cuda.is_available():
                payload["cuda_memory_allocated"] = torch.cuda.memory_allocated(device)
                payload["cuda_memory_reserved"] = torch.cuda.memory_reserved(device)
            lifecycle_records.append(payload)

        bytes_ = int(hidden_states.numel() * hidden_states.element_size())
        record(
            "forward_entry",
            shape=tuple(hidden_states.shape),
            dtype=str(hidden_states.dtype),
            bytes=bytes_,
        )
        with torch.no_grad():
            output = run_function(*args)
        if torch.is_tensor(output):
            output_tensors = (output,)
            ctx.output_is_tuple = False
        elif isinstance(output, tuple) and all(
            item is None or torch.is_tensor(item) for item in output
        ):
            output_tensors = tuple(item for item in output if torch.is_tensor(item))
            ctx.output_is_tuple = True
        else:
            raise RuntimeError(
                "slideformer-slot activation backend currently supports tensor or tensor/None tuple layer outputs only"
            )
        record(
            "forward_exit",
            output_shapes=[tuple(item.shape) for item in output_tensors],
            output_dtypes=[str(item.dtype) for item in output_tensors],
        )

        keep_on_gpu = gpu_window > 0 and layer_idx >= max(total_layers - gpu_window, 0)
        if keep_on_gpu:
            gpu_resident_slots[layer_idx] = hidden_states.detach()
            traffic_counters["activation_gpu_resident_bytes"] = (
                traffic_counters.get("activation_gpu_resident_bytes", 0) + bytes_
            )
            traffic_counters["activation_gpu_resident_count"] = (
                traffic_counters.get("activation_gpu_resident_count", 0) + 1
            )
            record("activation_gpu_resident_keep", bytes=bytes_)
        else:
            slot = cpu_slots.get(layer_idx)
            if (
                slot is None
                or slot.shape != hidden_states.shape
                or slot.dtype != hidden_states.dtype
            ):
                try:
                    slot = torch.empty(
                        hidden_states.shape,
                        dtype=hidden_states.dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                except RuntimeError:
                    slot = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device="cpu")
                cpu_slots[layer_idx] = slot

            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                stream.wait_event(ready_event)
                record("activation_d2h_submit", bytes=bytes_)
                slot.copy_(hidden_states.detach(), non_blocking=pin_memory)
                hidden_states.record_stream(stream)
                copy_event = torch.cuda.Event()
                copy_event.record(stream)
            copy_events[layer_idx] = copy_event
            traffic_counters["activation_offload_d2h_bytes"] = (
                traffic_counters.get("activation_offload_d2h_bytes", 0) + bytes_
            )
            traffic_counters["activation_offload_count"] = (
                traffic_counters.get("activation_offload_count", 0) + 1
            )
            record("activation_gpu_release", bytes=bytes_)

        ctx.run_function = run_function
        ctx.layer_idx = layer_idx
        ctx.device = device
        ctx.pin_memory = pin_memory
        ctx.stream = stream
        ctx.cpu_slots = cpu_slots
        ctx.copy_events = copy_events
        ctx.gpu_resident_slots = gpu_resident_slots
        ctx.prefetch_slots = prefetch_slots
        ctx.prefetch_events = prefetch_events
        ctx.traffic_counters = traffic_counters
        ctx.lifecycle_records = lifecycle_records
        ctx.prefetch_previous = prefetch_previous
        ctx.total_layers = total_layers
        ctx.gpu_window = gpu_window
        ctx.h2d_scheduler = h2d_scheduler
        ctx.args_tail = args[1:]
        ctx.hidden_requires_grad = bool(hidden_states.requires_grad)
        ctx.hidden_shape = tuple(hidden_states.shape)
        ctx.hidden_dtype = str(hidden_states.dtype)
        ctx.hidden_bytes = bytes_
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        def record(event: str, **extra: Any) -> None:
            if ctx.lifecycle_records is None:
                return
            payload = {
                "layer_idx": ctx.layer_idx,
                "event": event,
                "timestamp_kind": "cpu_perf_counter_s",
                "t": time.perf_counter(),
            }
            payload.update(extra)
            if torch.cuda.is_available():
                payload["cuda_memory_allocated"] = torch.cuda.memory_allocated(ctx.device)
                payload["cuda_memory_reserved"] = torch.cuda.memory_reserved(ctx.device)
            ctx.lifecycle_records.append(payload)

        resident_hidden = ctx.gpu_resident_slots.pop(ctx.layer_idx, None)
        if resident_hidden is not None:
            restored_hidden = resident_hidden
            ctx.traffic_counters["activation_gpu_resident_hit_count"] = (
                ctx.traffic_counters.get("activation_gpu_resident_hit_count", 0) + 1
            )
            record("activation_gpu_resident_hit", bytes=ctx.hidden_bytes)
        else:
            slot = ctx.cpu_slots[ctx.layer_idx]
            copy_event = ctx.copy_events.get(ctx.layer_idx)
            if copy_event is not None:
                torch.cuda.current_stream(ctx.device).wait_event(copy_event)
            record("activation_d2h_complete", bytes=ctx.hidden_bytes)

            prefetched = ctx.prefetch_slots.pop(ctx.layer_idx, None)
            prefetched_event = ctx.prefetch_events.pop(ctx.layer_idx, None)
            if prefetched is not None:
                if prefetched_event is not None:
                    if ctx.h2d_scheduler is not None:
                        ctx.h2d_scheduler.record_use_wait(
                            kind="activation_prefetch",
                            layer_idx=ctx.layer_idx,
                            bytes_=ctx.hidden_bytes,
                            event=prefetched_event,
                        )
                    else:
                        torch.cuda.current_stream(ctx.device).wait_event(prefetched_event)
                restored_hidden = prefetched
                record("activation_h2d_prefetch_hit", bytes=ctx.hidden_bytes)
            else:
                h2d_stream = (
                    ctx.h2d_scheduler.stream if ctx.h2d_scheduler is not None else ctx.stream
                )
                request_id = (
                    ctx.h2d_scheduler.begin_submit(
                        kind="activation_restore",
                        layer_idx=ctx.layer_idx,
                        bytes_=ctx.hidden_bytes,
                        allow_over_limit=True,
                    )
                    if ctx.h2d_scheduler is not None
                    else None
                )
                with torch.cuda.stream(h2d_stream):
                    record("activation_h2d_submit", bytes=ctx.hidden_bytes)
                    restored_hidden = slot.to(ctx.device, non_blocking=ctx.pin_memory)
                    restore_event = torch.cuda.Event()
                    restore_event.record(h2d_stream)
                if ctx.h2d_scheduler is not None:
                    ctx.h2d_scheduler.record_use_wait(
                        kind="activation_restore",
                        layer_idx=ctx.layer_idx,
                        bytes_=ctx.hidden_bytes,
                        event=restore_event,
                        request_id=request_id,
                    )
                else:
                    torch.cuda.current_stream(ctx.device).wait_event(restore_event)
            ctx.traffic_counters["activation_restore_h2d_bytes"] = (
                ctx.traffic_counters.get("activation_restore_h2d_bytes", 0) + ctx.hidden_bytes
            )
            ctx.traffic_counters["activation_restore_count"] = (
                ctx.traffic_counters.get("activation_restore_count", 0) + 1
            )
        restored_hidden.requires_grad_(ctx.hidden_requires_grad)
        record("activation_h2d_complete", bytes=ctx.hidden_bytes)

        recompute_args = (restored_hidden, *ctx.args_tail)
        with torch.enable_grad():
            record("activation_recompute_start")
            _submit_previous_slot_prefetch(ctx, record)
            from megatron.plugin.slideformer.kernels import split_te_recompute_early_stop

            with split_te_recompute_early_stop():
                recomputed = ctx.run_function(*recompute_args)
            record("activation_recompute_end")
        if torch.is_tensor(recomputed):
            recomputed_items = (recomputed,)
        elif isinstance(recomputed, tuple):
            recomputed_items = recomputed
        else:
            raise RuntimeError(
                "slideformer-slot activation backend currently supports tensor or tensor/None tuple layer outputs only"
            )
        outputs_with_grad: list[torch.Tensor] = []
        grads_with_grad: list[torch.Tensor] = []
        for output, grad in zip(recomputed_items, grad_outputs, strict=True):
            if torch.is_tensor(output) and grad is not None:
                outputs_with_grad.append(output)
                grads_with_grad.append(grad)
        if outputs_with_grad:
            torch.autograd.backward(outputs_with_grad, grads_with_grad)
        arg_grads: list[Any] = []
        for arg in recompute_args:
            if torch.is_tensor(arg) and arg.requires_grad:
                arg_grads.append(arg.grad)
            else:
                arg_grads.append(None)
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *arg_grads,
        )


def _submit_previous_slot_prefetch(ctx: Any, record: Any) -> None:
    if not ctx.prefetch_previous:
        return
    previous_idx = ctx.layer_idx - 1
    if previous_idx < 0:
        return
    if previous_idx in ctx.prefetch_slots:
        return
    previous_slot = ctx.cpu_slots.get(previous_idx)
    if previous_slot is None:
        return
    previous_copy_event = ctx.copy_events.get(previous_idx)
    h2d_scheduler = getattr(ctx, "h2d_scheduler", None)
    if h2d_scheduler is not None:
        request_id = h2d_scheduler.begin_submit(
            kind="activation_prefetch",
            layer_idx=previous_idx,
            bytes_=int(previous_slot.numel() * previous_slot.element_size()),
        )
        if request_id is None:
            return
        h2d_stream = h2d_scheduler.stream
    else:
        h2d_stream = ctx.stream
    with torch.cuda.stream(h2d_stream):
        if previous_copy_event is not None:
            h2d_stream.wait_event(previous_copy_event)
        record(
            "activation_h2d_prefetch_submit",
            target_layer_idx=previous_idx,
            bytes=int(previous_slot.numel() * previous_slot.element_size()),
        )
        gpu_tensor = torch.empty_like(previous_slot, device=ctx.device)
        gpu_tensor.copy_(previous_slot, non_blocking=ctx.pin_memory)
        event = torch.cuda.Event()
        event.record(h2d_stream)
    ctx.traffic_counters["activation_prefetch_h2d_bytes"] = ctx.traffic_counters.get(
        "activation_prefetch_h2d_bytes", 0
    ) + int(previous_slot.numel() * previous_slot.element_size())
    ctx.traffic_counters["activation_prefetch_count"] = (
        ctx.traffic_counters.get("activation_prefetch_count", 0) + 1
    )
    ctx.prefetch_slots[previous_idx] = gpu_tensor
    ctx.prefetch_events[previous_idx] = event


def slideformer_slot_checkpoint(
    run_function: Any,
    *,
    layer_idx: int,
    device: torch.device,
    pin_memory: bool,
    stream: torch.cuda.Stream,
    cpu_slots: dict[int, torch.Tensor],
    copy_events: dict[int, torch.cuda.Event],
    gpu_resident_slots: dict[int, torch.Tensor],
    prefetch_slots: dict[int, torch.Tensor],
    prefetch_events: dict[int, torch.cuda.Event],
    traffic_counters: dict[str, int],
    lifecycle_records: list[dict[str, Any]] | None,
    prefetch_previous: bool,
    total_layers: int,
    gpu_window: int,
    h2d_scheduler: Any | None,
    args: tuple[Any, ...],
) -> torch.Tensor:
    return _SlideFormerSlotCheckpointFunction.apply(
        run_function,
        layer_idx,
        device,
        pin_memory,
        stream,
        cpu_slots,
        copy_events,
        gpu_resident_slots,
        prefetch_slots,
        prefetch_events,
        traffic_counters,
        lifecycle_records,
        prefetch_previous,
        total_layers,
        gpu_window,
        h2d_scheduler,
        *args,
    )
