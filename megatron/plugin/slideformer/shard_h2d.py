# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sharded parameter H2D + native Megatron parameter gather (NVLink machines).

Every rank used to copy the whole bf16 layer from the shared pinned buffer to its GPU, so
N ranks pull N full copies through host memory and PCIe at once (~7 GB/s per rank on the
A800 box). Here each rank copies only its own 1/N slice of the layer into its place in the
GPU parameter buffer, and the missing slices are filled in over the GPU interconnect with
one broadcast per slice (root = the rank that owns it). Slices are the same ones the CPU
sharded optimizer uses, so the slice a rank loads is exactly the slice it updates.

Equal slices use Megatron's in-place all-gather. Uneven slices (boundaries
aligned to parameter starts) use N broadcasts without a padded model buffer.

Ordering rule: NCCL collectives must be issued in the same order on every rank. Parameter
loads happen in hooks whose order is the program order, identical on all ranks; the H2D
scheduler's throttling is bypassed for these layers so that no rank can skip a prefetch that
another rank issues. A dedicated communicator keeps this traffic apart from everything else.

The source is either the shared BF16 execution cache or bounded pinned staging
cast from only this rank's FP32 master slice. Both deliver the same BF16 values.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from megatron.core.distributed.param_and_grad_buffer import dist_all_gather_func, shard_buffer

from .gpu_reduce import _nccl_twin

ENV_FLAG = "MEGATRON_SLIDEFORMER_SHARD_H2D"


def requested() -> bool:
    return os.getenv(ENV_FLAG, "0").lower() in {"1", "true", "yes", "on"}


class ShardedH2D:
    def __init__(self, engine, sharded):
        self.engine = engine
        self.sharded = sharded
        self.rank = sharded.rank
        self.group = _nccl_twin(sharded.dp_group)
        self.roots = list(
            dist.get_process_group_ranks(sharded.dp_group)
        )  # global rank per group rank
        self.layers = set()
        for owner in sharded.owners:
            if owner.gpu_param_pool is not None and all(
                dtype == torch.bfloat16 for dtype in owner.execution_dtypes.values()
            ):
                self.layers.add(owner.layer_idx)
        self._host = [None, None]
        self._host_events = [None, None]
        self._next_host = 0
        chunk_mb = float(os.getenv("MEGATRON_SLIDEFORMER_H2D_CHUNK_MB", "64"))
        if chunk_mb <= 0:
            raise ValueError("H2D chunk size must be positive")
        self._chunk_bytes = max(1, int(chunk_mb * 1024**2))
        self._host_numel = max(
            (
                sharded.bounds[L][self.rank][1] - sharded.bounds[L][self.rank][0]
                for L in self.layers
            ),
            default=0,
        )

    def eligible(self, owner) -> bool:
        return owner.layer_idx in self.layers

    def load(self, owner) -> None:
        """Fill the owner's GPU flat parameter buffer; call inside the H2D stream context."""
        L = owner.layer_idx
        flat = owner._gpu_param_flat
        bounds = self.sharded.bounds[L]
        lo, hi = bounds[self.rank]
        if self.sharded.exec_cache:
            source = self.sharded.exec[L][lo:hi]
        else:
            # Cast only the rank-owned shard into bounded pinned staging, as
            # opposed to retaining a full-model BF16 execution copy on the host.
            slot = self._next_host
            self._next_host = (slot + 1) % len(self._host)
            if self._host_events[slot] is not None:
                self._host_events[slot].synchronize()
            if self._host[slot] is None:
                self._host[slot] = torch.empty(self._host_numel, dtype=flat.dtype, pin_memory=True)
            source = self._host[slot][: hi - lo]
        # Bound each DMA submission so gradient D2H can make progress while a
        # large vocabulary/decoder owner is being prefetched on another stream.
        chunk = max(1, self._chunk_bytes // flat.element_size())
        for offset in range(0, hi - lo, chunk):
            count = min(chunk, hi - lo - offset)
            if not self.sharded.exec_cache:
                source[offset : offset + count].copy_(
                    self.sharded.master[L][lo + offset : lo + offset + count]
                )
            flat[lo + offset : lo + offset + count].copy_(
                source[offset : offset + count], non_blocking=True
            )
        if not self.sharded.exec_cache:
            self._host_events[slot] = torch.cuda.Event()
            self._host_events[slot].record(torch.cuda.current_stream(self.engine.device))
        if all(b - a == hi - lo for a, b in bounds):
            # Same in-place shard view and all-gather used by Megatron's native
            # distributed optimizer. Uneven SIMD-aligned slices retain the
            # broadcast fallback rather than allocating a padded model buffer.
            local_view = shard_buffer(flat, len(bounds))[self.rank]
            dist_all_gather_func(flat, local_view, group=self.group)
        else:
            for r, (a, b) in enumerate(bounds):
                if b > a:
                    dist.broadcast(flat[a:b], src=self.roots[r], group=self.group)
