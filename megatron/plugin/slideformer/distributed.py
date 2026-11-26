# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded gradient synchronization for CPU-owned SlideFormer parameters.

Process groups are supplied by Megatron; never infer replica ownership from
world size. Dense parameters use DP×CP groups; experts use expert DP groups.
"""
import os
import socket
import uuid

import numpy as np
import torch
import torch.distributed as dist


def synchronize_gradient(gradient, *, group, device, chunk_size_mb, average, async_op=False):
    """Reduce FP32 chunks on the communication device and copy back to CPU."""
    size = dist.get_world_size(group) if group is not None else 1
    if size == 1:
        return
    flat = gradient.view(-1)
    chunk_numel = max(1, int(chunk_size_mb * 1024**2) // 4)
    for start in range(0, flat.numel(), chunk_numel):
        destination = flat[start : start + chunk_numel]
        work = destination.to(device=device, dtype=torch.float32)
        handle = dist.all_reduce(work, group=group, async_op=async_op)
        if handle is not None:
            # CUDA Work.wait orders the current communication stream; CPU
            # callers wait on this worker, never the training Python thread.
            handle.wait()
        if average:
            work.div_(size)
        destination.copy_(work.to(device=destination.device))


class SharedCPUGradientReducer:
    """Sum FP32 CPU gradients across same-host ranks through /dev/shm.

    Each rank writes a segment of its gradients into its own mapped slot,
    owns one slice of the segment and sums that slice over ranks 0..N-1 in
    fixed order, then every rank reads the full reduced segment back. This
    replaces the GPU NCCL all-reduce (plus pageable D2H/H2D copies) used by
    ``synchronize_gradient``. It only reduces; scaling stays with the caller.
    """

    def __init__(self, group, segment_numel=128 * 2**20, num_threads=None):
        self.group = group
        self.size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.segment = int(segment_numel)
        self.num_threads = num_threads or int(
            os.getenv("MEGATRON_SLIDEFORMER_SHARED_CPU_REDUCE_THREADS", "16")
        )
        self._maps = []
        self._parity = 0
        self.usable = self._same_host()
        if self.usable:
            self._setup()

    def _same_host(self):
        names = [None] * self.size
        dist.all_gather_object(names, socket.gethostname(), group=self.group)
        return len(set(names)) == 1

    def _map(self, path, create):
        array = np.memmap(path, dtype=np.float32, mode="w+" if create else "r+",
                          shape=(self.segment,))
        self._maps.append(array)
        return torch.from_numpy(array)

    def _setup(self):
        token = [uuid.uuid4().hex if self.rank == 0 else None]
        dist.broadcast_object_list(token, src=dist.get_global_rank(self.group, 0), group=self.group)
        base = f"/dev/shm/slideformer_grad_{token[0]}"
        paths_in = [f"{base}_in{r}" for r in range(self.size)]
        paths_out = [f"{base}_out{k}" for k in range(2)]
        mine = self._map(paths_in[self.rank], create=True)
        outs = [self._map(p, create=True) for p in paths_out] if self.rank == 0 else None
        dist.barrier(group=self.group)
        self.bufs = [mine if r == self.rank else self._map(paths_in[r], create=False)
                     for r in range(self.size)]
        if outs is None:
            outs = [self._map(p, create=False) for p in paths_out]
        self.outs = outs
        dist.barrier(group=self.group)
        # Mappings stay valid after unlink, so nothing leaks if a job dies.
        try:
            os.unlink(paths_in[self.rank])
            if self.rank == 0:
                for p in paths_out:
                    os.unlink(p)
        except FileNotFoundError:
            pass

    def reduce(self, gradients):
        """In-place SUM of contiguous FP32 CPU tensors over the group."""
        views = [g.view(-1) for g in gradients]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.num_threads)
        try:
            index, offset = 0, 0
            while index < len(views):
                pieces, filled = [], 0
                while index < len(views) and filled < self.segment:
                    view = views[index]
                    count = min(view.numel() - offset, self.segment - filled)
                    if count > 0:
                        pieces.append((view, offset, count, filled))
                        filled += count
                        offset += count
                    if offset >= view.numel():
                        index, offset = index + 1, 0
                if filled:
                    self._reduce_segment(pieces, filled)
        finally:
            torch.set_num_threads(previous_threads)

    def _reduce_segment(self, pieces, filled):
        mine = self.bufs[self.rank]
        for view, start, count, at in pieces:
            mine[at : at + count].copy_(view[start : start + count])
        dist.barrier(group=self.group)
        per_rank = (filled + self.size - 1) // self.size
        low = min(self.rank * per_rank, filled)
        high = min(low + per_rank, filled)
        out = self.outs[self._parity]
        self._parity ^= 1
        if high > low:
            torch.add(self.bufs[0][low:high], self.bufs[1][low:high], out=out[low:high])
            for r in range(2, self.size):
                out[low:high].add_(self.bufs[r][low:high])
        dist.barrier(group=self.group)
        for view, start, count, at in pieces:
            view[start : start + count].copy_(out[at : at + count])
