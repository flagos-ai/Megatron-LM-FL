# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU reduce-scatter of layer gradients for the CPU-sharded optimizer (NVLink machines).

The CPU path sums the gradients of all replicas on the host: every rank D2H-copies
the whole bf16 layer gradient, accumulates it into a shared fp32 buffer, and then each
rank reads all N buffers to sum its own slice. That is ~100 GB of host-memory traffic per
rank and step, which is the bottleneck on a machine with fast GPU interconnect.

Here the replica sum uses native NCCL collectives on a dedicated communicator.
Default BF16 reduction and D2H match the original SlideDP precision. CPU master
weights, Adam moments, arithmetic and cross-microbatch accumulation remain FP32.
Set gpu_reduce_dtype="fp32" for the preceding FP32 reduction/D2H behavior.
The independent MCore FP32-accumulation option sends BF16, sums in FP32 and casts
to gpu_reduce_dtype on GPU before scaling and D2H. BF16 reduction need not match
FP32 reduction bitwise, particularly for larger groups or cancellation.
Single-microbatch steps lease a bounded host pool through CPU Adam completion;
multiple microbatches convert the copied BF16 slice into FP32 host accumulators.

Layers without flat BF16 gradient hooks and layers with TP-special gradients
keep using the host path. Single-parameter embedding/output owners can reuse
their contiguous autograd gradient directly.
"""
from __future__ import annotations

import os
import json

import torch
import torch.distributed as dist

from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import (
    reduce_scatter_with_fp32_accumulation,
)
from megatron.core.distributed.param_and_grad_buffer import shard_buffer

ENV_FLAG = "MEGATRON_SLIDEFORMER_GPU_REDUCE"


def requested() -> bool:
    return os.getenv(ENV_FLAG, "0").lower() in {"1", "true", "yes", "on"}


def _nccl_twin(group, *, gradient=False):
    """Collectively create a dedicated NCCL communicator per distinct group membership."""
    from megatron.core.parallel_state import create_group, get_nccl_options

    options = None
    if gradient:
        min_ctas = int(os.getenv("MEGATRON_SLIDEFORMER_GRAD_MIN_CTAS", "2"))
        max_ctas = int(os.getenv("MEGATRON_SLIDEFORMER_GRAD_MAX_CTAS", "4"))
        if min_ctas < 0 or max_ctas < min_ctas:
            raise ValueError("Expected 0 <= gradient min CTAs <= max CTAs")
        if max_ctas:
            options = get_nccl_options(
                "slide_grad", {"slide_grad": {"min_ctas": min_ctas, "max_ctas": max_ctas}}
            )

    members = tuple(dist.get_process_group_ranks(group))
    everyone = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, members)
    mine = None
    for ranks in sorted(set(everyone)):
        twin = create_group(
            ranks=list(ranks), backend="nccl", pg_options=options, group_desc="slide-layer-transfer"
        )
        if dist.get_rank() in ranks:
            mine = twin
    return mine


class GPUReduceScatter:
    def __init__(self, engine, sharded, slots: int = 2):
        self.engine = engine
        self.sharded = sharded
        self.size = sharded.size
        self.rank = sharded.rank
        self.group = _nccl_twin(sharded.dp_group, gradient=True)
        self.device = engine.device
        self.fp32_accumulation = engine.config.reduce_scatter_with_fp32_accumulation
        precision = engine.config.gpu_reduce_dtype
        if precision not in {"bf16", "fp32"}:
            raise ValueError("gpu_reduce_dtype must be bf16 or fp32")
        self.transport_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.reduce_dtype = torch.float32 if self.fp32_accumulation else self.transport_dtype
        self.layout = os.getenv("MEGATRON_SLIDEFORMER_GPU_REDUCE_LAYOUT", "auto")
        if self.layout not in {"auto", "bucketed"}:
            raise ValueError("GPU reduce layout must be auto or bucketed")
        reuse = os.getenv("MEGATRON_SLIDEFORMER_GPU_GRAD_REUSE", "stream")
        if reuse not in {"stream", "host"}:
            raise ValueError("GPU gradient reuse must be stream or host")
        self.stream_ordered_reuse = reuse == "stream"
        print("GPU_REDUCE_PIPELINE " + json.dumps(dict(
            rank=self.rank, layout=self.layout, gradient_slot_reuse=reuse,
            d2h_chunk_mb=engine.config.chunk_size_mb,
            host_slot_release="cpu_future_complete", parameter_prefetch="after_update_drain")), flush=True)
        print("GPU_REDUCE_PRECISION " + json.dumps(dict(
            rank=self.rank, wire_dtype=str(torch.bfloat16 if self.fp32_accumulation else self.transport_dtype),
            reduced_dtype=str(self.reduce_dtype), d2h_dtype=str(self.transport_dtype),
            microbatch_accumulation_dtype="torch.float32", cpu_master_m_v_dtype="torch.float32")), flush=True)
        self.layers = {}  # layer_idx -> (lo, hi)
        self.sacc = {}  # active gradient views consumed by the CPU worker
        self._persistent_sacc = {}  # only multi-microbatch steps need full-model accumulation
        self.stmp = {}  # bounded-precision DMA input; microbatch accumulation stays FP32
        self.last_job = {}
        longest = 0
        for owner in sharded.owners:
            L = owner.layer_idx
            if not (owner._flat_grad_copy_supported and owner.cpu_grad_flat is not None):
                continue
            if sharded.special[L]:
                continue
            if owner.cpu_grad_flat.dtype != torch.bfloat16:
                continue
            bounds = sharded.bounds[L]
            lo, hi = bounds[self.rank]
            self.layers[L] = (lo, hi)
            longest = max(longest, max(b - a for a, b in bounds))
        self.longest = longest
        # Scratch is only needed for uneven shards or a precision conversion.
        # Eligible equal-dtype/equal-shard layers use the native in-place layout.
        # Explicit bucketed mode retains the preceding path for comparisons.
        bucket_mb = float(os.getenv("MEGATRON_SLIDEFORMER_GPU_REDUCE_BUCKET_MB", "128"))
        if bucket_mb < 0:
            raise ValueError("GPU reduce bucket size must be nonnegative")
        self.bucket = (
            min(longest, max(1, int(bucket_mb * 1024**2 // (4 * self.size))))
            if bucket_mb
            else longest
        )
        self._slots = slots
        self._padded = [None] * slots
        self._out = [None] * slots
        self._slot_event = [None] * slots
        self._host = [None] * slots
        self._host_jobs = [None] * slots
        self._host_slot_for_layer = {}
        self._next = 0

    def eligible(self, owner) -> bool:
        return owner.layer_idx in self.layers

    def active(self, owner) -> bool:
        """Use the GPU path now? Needs the per-token scale known before backward."""
        return (
            owner.layer_idx in self.layers
            and self.engine._eager_scale is not None
            and self.sharded.owns(owner)
        )

    def _acquire_slot(self, stream, *, needs_scratch=True):
        slot = self._next
        self._next = (self._next + 1) % self._slots
        if needs_scratch and self._padded[slot] is None:
            dtype = torch.bfloat16 if self.fp32_accumulation else self.transport_dtype
            self._padded[slot] = torch.empty(
                self.size * self.bucket, dtype=dtype, device=self.device
            )
            self._out[slot] = torch.empty(self.bucket, dtype=self.reduce_dtype, device=self.device)
        if self._slot_event[slot] is not None:
            stream.wait_event(self._slot_event[slot])  # previous use finished its D2H
        return slot

    def _can_reduce_in_place(self, gpu_flat, bounds):
        width = bounds[0][1] - bounds[0][0]
        return (
            self.layout == "auto"
            and not self.fp32_accumulation
            and gpu_flat.dtype == self.reduce_dtype
            and gpu_flat.ndim == 1
            and gpu_flat.is_contiguous()
            and width > 0
            and gpu_flat.numel() == self.size * width
            and all((a, b) == (r * width, (r + 1) * width)
                    for r, (a, b) in enumerate(bounds))
        )

    def _reduce_in_place(self, gpu_flat, host, gpu_scale):
        # Match MCore _ParamAndGradBucketGroup.start_grad_sync(): the local
        # output aliases its input shard. No packed input or output scratch.
        # Use the explicit native BF16/FP32 collective; MCore's module-global
        # alias can be rebound by an unrelated FP32-accumulation DDP instance.
        mine = shard_buffer(gpu_flat, self.size)[self.rank]
        dist.reduce_scatter_tensor(mine, gpu_flat, op=dist.ReduceOp.SUM, group=self.group)
        if gpu_scale is not None:
            mine.mul_(gpu_scale)
        # The input gradient slot remains leased through the last DMA. Chunking
        # lets simultaneous parameter H2D progress without a second layer copy.
        chunk_mb = self.engine.config.chunk_size_mb
        chunk = max(1, int(chunk_mb * 1024**2 // mine.element_size())) if chunk_mb > 0 else mine.numel()
        for offset in range(0, mine.numel(), chunk):
            host[offset : offset + chunk].copy_(mine[offset : offset + chunk], non_blocking=True)
        counters = getattr(self.engine, "traffic_counters", None)
        if counters is not None:
            counters["gpu_reduce_inplace_layers"] = counters.get("gpu_reduce_inplace_layers", 0) + 1
            counters["gpu_reduce_d2h_bytes"] = counters.get("gpu_reduce_d2h_bytes", 0) + mine.numel() * mine.element_size()

    def register_update(self, layer_idx, future):
        """Keep a pooled D2H slot leased until CPU Adam has consumed its contents."""
        slot = self._host_slot_for_layer.pop(layer_idx, None)
        if slot is not None:
            self._host_jobs[slot] = future

    def submit(self, owner, gpu_flat: torch.Tensor) -> None:
        """Reduce-scatter one finished layer gradient (called from the autograd thread)."""
        L = owner.layer_idx
        engine = self.engine
        lo, hi = self.layers[L]
        bounds = self.sharded.bounds[L]
        first = engine._layer_bwd_count.get(L, 0) == 0
        if not first:
            # The temp buffer is reused by every later micro-batch of this layer: make sure
            # the worker consumed the previous one.
            previous = self.last_job.get(L)
            if previous is not None:
                previous.result()
        if engine._expected_microbatches > 1:
            if L not in self._persistent_sacc:
                self._persistent_sacc[L] = torch.empty(
                    hi - lo, dtype=torch.float32, pin_memory=True
                )
            self.sacc[L] = self._persistent_sacc[L]
            if L not in self.stmp:
                self.stmp[L] = torch.empty(hi - lo, dtype=self.transport_dtype, pin_memory=True)
            host = self.stmp[L]
        gpu_scale = engine._eager_scale if engine._expected_microbatches == 1 else None
        stream = owner.d2h_stream
        current = torch.cuda.current_stream(self.device)
        in_place = self._can_reduce_in_place(gpu_flat, bounds)
        with torch.cuda.stream(stream):
            stream.wait_stream(current)
            slot = self._acquire_slot(stream, needs_scratch=not in_place)
            if engine._expected_microbatches == 1:
                previous = self._host_jobs[slot]
                if previous is not None:
                    previous.result()
                if self._host[slot] is None:
                    self._host[slot] = torch.empty(
                        self.longest, dtype=self.transport_dtype, pin_memory=True
                    )
                host = self._host[slot][: hi - lo]
                self.sacc[L] = host
                self._host_slot_for_layer[L] = slot
            if in_place:
                self._reduce_in_place(gpu_flat, host, gpu_scale)
            else:
                self._reduce_bucketed(gpu_flat, host, bounds, lo, hi, slot, gpu_scale)
            event = torch.cuda.Event()
            event.record(stream)
            self._slot_event[slot] = event
        gpu_flat.record_stream(stream)
        owner.gpu_grad_flat = None
        owner._grad_copy_submitted = True
        owner._gpu_reduce_pending = True
        owner._gpu_reduce_event = event
        owner._gpu_reduce_first = first

    def _reduce_bucketed(self, gpu_flat, host, bounds, lo, hi, slot, gpu_scale):
        """Compatibility fallback; called in the owner's D2H stream context."""
        engine = self.engine
        counters = getattr(engine, "traffic_counters", None)
        if counters is not None:
            counters["gpu_reduce_bucketed_layers"] = counters.get("gpu_reduce_bucketed_layers", 0) + 1
        padded, out = self._padded[slot], self._out[slot]
        longest = max(b - a for a, b in bounds)
        equal_shards = all(b - a == longest for a, b in bounds)
        for offset in range(0, longest, self.bucket):
            width = min(self.bucket, longest - offset)
            packed = padded[: self.size * width].view(self.size, width)
            if equal_shards:
                # Pack all replicas in one copy. Bounded slices use a
                # strided kernel instead of per-shard DMA submissions that
                # compete with concurrent parameter H2D / gradient D2H.
                packed.copy_(gpu_flat.view(self.size, longest)[:, offset : offset + width])
            else:
                for r, (a, b) in enumerate(bounds):
                    count = max(0, min(width, b - a - offset))
                    if count:
                        packed[r, :count].copy_(gpu_flat[a + offset : a + offset + count])
                    # Only the unfilled tail needs initialization. Full
                    # buckets overwrite every element before communication.
                    if count < width:
                        packed[r, count:].zero_()
            if self.fp32_accumulation:
                # Reuse MCore's lower-precision transport with FP32 local
                # summation. Keep the destination FP32 for CPU Adam.
                reduce_scatter_with_fp32_accumulation(
                    out[:width],
                    packed.flatten(),
                    op=dist.ReduceOp.SUM,
                    group=self.group,
                    async_op=False,
                )
            else:
                dist.reduce_scatter_tensor(
                    out[:width], packed.flatten(), op=dist.ReduceOp.SUM, group=self.group
                )
            count = max(0, min(width, hi - lo - offset))
            if count:
                # Cast on GPU before DMA: casting into a CPU BF16 target
                # alone may still transfer an FP32 intermediate.
                mine = out[:count].to(self.transport_dtype)
                if gpu_scale is not None:
                    mine.mul_(gpu_scale)
                host[offset : offset + count].copy_(mine, non_blocking=True)
                counters = getattr(engine, "traffic_counters", None)
                if counters is not None:
                    counters["gpu_reduce_d2h_bytes"] = counters.get("gpu_reduce_d2h_bytes", 0) + count * mine.element_size()
