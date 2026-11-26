# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-sharded data-parallel optimizer for SlideFormer (same-host replica groups).

A dense parameter is replicated across its dp_cp group (the ranks that differ
only in data/context-parallel coordinate). Previously every replica summed the
gradients of the whole model and ran CPU Adam on all of it. Here, inside each
replica group of N ranks:

* the fp32 master weights and fp32 gradient accumulators of every layer live in
  ``/dev/shm`` (one namespace per replica group), so the N ranks share ONE
  master copy;
* rank r owns the r-th of N contiguous slices of every layer: it sums the N
  accumulators on its slice in rank order, applies Megatron's TP-special
  gradient reduction to that slice (TP>1), scales, and runs Adam on the slice;
* after one CPU barrier the next forward reads the complete updated layers.

Adam, the replica sum and the TP all-reduce are element-wise, so slice results
equal the matching part of the unsliced computation. Slice boundaries are
aligned to 64 elements from a parameter start so SIMD body/tail splits match
the unsliced call. N is arbitrary (>= 2); nothing assumes 2.

Scope: tensor-parallel, context-parallel (inside dp_cp) and data-parallel
layouts, one host per replica group, fp32 flat masters. Layers with expert
parameters (and anything ineligible) stay on the regular path, per layer.
Pipeline parallelism and NVMe optimizer offload are not supported. Optimizer
checkpoints contain the active DP-local moments and require the same topology.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist

ENV_FLAG = "MEGATRON_SLIDEFORMER_SHARDED_CPU_OPT"
_ALIGN = 64


def requested() -> bool:
    return os.getenv(ENV_FLAG, "0").lower() in {"1", "true", "yes", "on"}


def _group_size(group) -> int:
    return dist.get_world_size(group) if group is not None else 1


def _gloo_like(group):
    """Collectively create a gloo group per distinct membership of ``group``.

    Every rank must call this; each rank gets back the gloo twin of its own group.
    """
    from megatron.core.parallel_state import create_group

    members = tuple(dist.get_process_group_ranks(group))
    everyone = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, members)
    mine = None
    for ranks in sorted(set(everyone)):
        twin = create_group(ranks=list(ranks), backend="gloo", group_desc="slide-cpu-replica")
        if dist.get_rank() in ranks:
            mine = twin
    return mine


class ShardedCPUOptimizer:
    def __init__(self, engine, dp_group, tp_group, pp_group, special_kind=None):
        """``special_kind(param)`` -> "sum" | "avg" | None for TP-special gradients."""
        self.engine = engine
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.special_kind = special_kind or (lambda param: None)
        self.size = _group_size(dp_group)
        self.rank = dist.get_rank(dp_group) if dp_group is not None else 0
        self.tp_size = _group_size(tp_group)
        self.owners = []  # eligible layers only; the rest stay on the regular path
        self.done = set()  # layer indices already reduced + updated this step
        self.exec_cache = False  # shared bf16 copy of the master used as the H2D source
        # Opt-in Python fused reduction (bit-identical, but measured SLOWER on the A800: thousands of tiny
        # torch calls contend for the GIL). Kept for reference; a C++ kernel would be the real fix.
        # Fused reduction: blocks small enough that torch runs each op inline (<= its
        # 32768-element grain) and that stay cache-resident across the chained ops.
        self.reduce_chunk = int(os.getenv("MEGATRON_SLIDEFORMER_SHARDED_REDUCE_CHUNK", "32768"))
        self.reduce_threads = int(os.getenv("MEGATRON_SLIDEFORMER_SHARDED_REDUCE_THREADS", "0"))  # 0 = off (measured slower: GIL contention)
        self._reduce_pool = None
        self._reduce_local = threading.local()
        self.phase_times = {"barrier": 0.0, "reduce": 0.0, "adam": 0.0, "cast": 0.0, "layers": 0}
        self.exec = {}
        self._futures = []
        self._executor = None
        reason = self._group_ineligible_reason(pp_group)
        # Every rank must reach the same decision before any further collective.
        reasons = [None] * dist.get_world_size()
        dist.all_gather_object(reasons, reason)
        self.reason = next((r for r in reasons if r is not None), None)
        self.usable = False
        if self.reason is None:
            self.gloo_dp = _gloo_like(dp_group)
            self.gloo_tp = _gloo_like(tp_group) if self.tp_size > 1 else None
            self._select_layers()
            if self.owners:
                self._setup()
                self.usable = True
            else:
                self.reason = "no eligible layers"

    # ------------------------------------------------------------ eligibility
    def _group_ineligible_reason(self, pp_group):
        engine = self.engine
        if self.size < 2:
            return "replica group size < 2"
        if _group_size(pp_group) != 1:
            return "PP > 1 not supported"
        if engine.gpu_gradient_pipeline is not None:
            return "GPU gradient pipeline active"
        if engine.config.skip_cpu_adam_step:
            return "skip_cpu_adam_step"
        if float(getattr(engine.config, "clip_grad", 0.0) or 0.0) > 0.0:
            return "gradient clipping needs the full-model norm"
        if getattr(engine.layer_optimizer, "offloader", None) is not None:
            return "NVMe optimizer offload"
        if engine.unmanaged_params:
            return "unmanaged parameters"
        names = [None] * self.size
        dist.all_gather_object(names, socket.gethostname(), group=self.dp_group)
        if len(set(names)) != 1:
            return "ranks of a replica group span several hosts"
        return None

    def _select_layers(self):
        local = [
            o.layer_idx
            for o in self.engine.managed_layers
            if o.params
            and o.cpu_param_flat is not None
            and o.cpu_param_flat.dtype == torch.float32
            and all(getattr(p, "allreduce", True) for p in o.params)
        ]
        gathered = [None] * self.size
        dist.all_gather_object(gathered, local, group=self.gloo_dp)
        common = set(gathered[0]).intersection(*map(set, gathered[1:]))
        self.owners = [o for o in self.engine.managed_layers if o.params and o.layer_idx in common]

    def owns(self, owner) -> bool:
        return owner.layer_idx in self._layer_ids

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _map(path, numel, create):
        array = np.memmap(path, dtype=np.float32, mode="w+" if create else "r+", shape=(numel,))
        return array, torch.from_numpy(array)

    def _split_points(self, owner):
        total = owner.cpu_param_flat.numel()
        ranges = sorted(owner._param_flat_offsets[p] for p in owner.params)
        points = [0]
        for r in range(1, self.size):
            target = r * total // self.size
            point = target
            for start, end in ranges:
                if start <= target < end:
                    point = start + ((target - start) // _ALIGN) * _ALIGN
                    break
            points.append(max(points[-1], point))
        points.append(total)
        return points

    def _setup(self):
        self._layer_ids = {o.layer_idx for o in self.owners}
        token = [uuid.uuid4().hex if self.rank == 0 else None]
        dist.broadcast_object_list(
            token, src=dist.get_process_group_ranks(self.gloo_dp)[0], group=self.gloo_dp
        )
        base = f"/dev/shm/slideformer_shardopt_{token[0]}"
        self._base = base
        keep, created = [], []  # np.memmap objects must outlive the torch views
        self.master, self.acc, self.bounds = {}, {}, {}
        self.exp_avg, self.exp_avg_sq, self.special = {}, {}, {}

        # Phase 1: every rank creates the files it owns.
        pending = {}
        for owner in self.owners:
            L, n = owner.layer_idx, owner.cpu_param_flat.numel()
            points = self._split_points(owner)
            self.bounds[L] = [(points[r], points[r + 1]) for r in range(self.size)]
            master_path = f"{base}_m{L}"
            acc_paths = [f"{base}_a{r}_{L}" for r in range(self.size)]
            own = {}
            if self.rank == 0:
                array, own["master"] = self._map(master_path, n, create=True)
                keep.append(array)
                created.append(master_path)
            array, own["acc"] = self._map(acc_paths[self.rank], n, create=True)
            keep.append(array)
            created.append(acc_paths[self.rank])
            pending[L] = (master_path, acc_paths, n, own)
            self.special[L] = (
                [
                    (*owner._param_flat_offsets[p], self.special_kind(p))
                    for p in owner.params
                    if self.tp_size > 1 and self.special_kind(p)
                ]
            )
        dist.barrier(group=self.gloo_dp)

        # Phase 2: map the files owned by the other ranks of the replica group.
        for owner in self.owners:
            L = owner.layer_idx
            master_path, acc_paths, n, own = pending[L]
            if self.rank == 0:
                master = own["master"]
            else:
                array, master = self._map(master_path, n, create=False)
                keep.append(array)
            accs = []
            for r in range(self.size):
                if r == self.rank:
                    accs.append(own["acc"])
                else:
                    array, tensor = self._map(acc_paths[r], n, create=False)
                    keep.append(array)
                    accs.append(tensor)
            self.master[L], self.acc[L] = master, accs
        dist.barrier(group=self.gloo_dp)
        for path in created:  # mappings survive unlink: nothing leaks if the job dies
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._keep = keep

        # Phase 3: seed the shared master (each rank writes its slice; masters
        # are identical across the replica group), then rebind the layers.
        for owner in self.owners:
            L = owner.layer_idx
            lo, hi = self.bounds[L][self.rank]
            self.master[L][lo:hi].copy_(owner.cpu_param_flat[lo:hi])
            optimizer = self.engine.layer_optimizer
            optimizer.shard_layer_state(owner.optimizer_layer_idx, lo, hi)
            self.exp_avg[L] = optimizer.exp_avg_flat[owner.optimizer_layer_idx]
            self.exp_avg_sq[L] = optimizer.exp_avg_sq_flat[owner.optimizer_layer_idx]
        dist.barrier(group=self.gloo_dp)
        for owner in self.owners:
            master = self.master[owner.layer_idx]
            owner.cpu_param_flat = master
            for param in owner.params:
                start, end = owner._param_flat_offsets[param]
                view = master[start:end].view_as(param)
                owner.cpu_params[param] = view
                owner.cpu_master[param] = view
        dist.barrier(group=self.gloo_dp)
        self._setup_exec_cache()

    def _setup_exec_cache(self):
        """Keep one shared, pinned bf16 copy of every master layer up to date.

        Loading a layer used to cast the whole fp32 master to bf16 on the CPU
        (twice per step, on the thread that drives the GPU). Now each rank casts
        only its own slice right after Adam, and H2D copies read the shared
        buffer directly. The cast is the same fp32->bf16 rounding, so the values
        are bit-identical to what the per-load cast produced.
        """
        if os.getenv("MEGATRON_SLIDEFORMER_SHARDED_EXEC_CACHE", "1").lower() not in {
            "1", "true", "yes", "on"}:
            return
        keep, created = [], []
        paths = {o.layer_idx: f"{self._base}_x{o.layer_idx}" for o in self.owners}
        arrays = {}
        for owner in self.owners:
            n = owner.cpu_param_flat.numel()
            if self.rank == 0:
                arrays[owner.layer_idx] = np.memmap(paths[owner.layer_idx], dtype=np.int16,
                                                    mode="w+", shape=(n,))
                created.append(paths[owner.layer_idx])
        dist.barrier(group=self.gloo_dp)
        for owner in self.owners:
            L, n = owner.layer_idx, owner.cpu_param_flat.numel()
            if L not in arrays:
                arrays[L] = np.memmap(paths[L], dtype=np.int16, mode="r+", shape=(n,))
            keep.append(arrays[L])
            # int16 and bfloat16 have the same width: reinterpret the shared bytes.
            self.exec[L] = torch.from_numpy(arrays[L]).view(torch.bfloat16)
        dist.barrier(group=self.gloo_dp)
        for path in created:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._keep.extend(keep)
        for owner in self.owners:
            L = owner.layer_idx
            lo, hi = self.bounds[L][self.rank]
            self.exec[L][lo:hi].copy_(self.master[L][lo:hi])
        dist.barrier(group=self.gloo_dp)
        self._pin_exec_buffers()
        for owner in self.owners:
            L = owner.layer_idx
            flat = self.exec[L]
            owner.cpu_param_staging_pool = None  # no per-load cast into a pool slot
            owner.cpu_param_staging_flat = flat
            for param in owner.params:
                start, end = owner._param_flat_offsets[param]
                owner.cpu_execution_params[param] = flat[start:end].view_as(param)
        self.exec_cache = True
        dist.barrier(group=self.gloo_dp)

    @staticmethod
    def _clear_cuda_error():
        """Clear libcudart's sticky last-error so a failed best-effort call does not
        surface later as an unrelated CUDA error."""
        import ctypes

        try:
            with open("/proc/self/maps") as maps:
                libs = sorted({line.split()[-1] for line in maps if "libcudart" in line})
            for lib in libs:
                ctypes.CDLL(lib).cudaGetLastError()
        except Exception:
            pass

    def _pin_exec_buffers(self):
        """Page-lock the shared buffers so H2D copies run at full PCIe speed.

        Best effort: a buffer that cannot be pinned still works (slower H2D).
        Every failure is reported with its rank, layer and size.
        """
        if not torch.cuda.is_available():
            return
        runtime = torch.cuda.cudart()
        failures = []
        for layer, tensor in self.exec.items():
            nbytes = tensor.numel() * tensor.element_size()
            try:
                status = runtime.cudaHostRegister(tensor.data_ptr(), nbytes, 0)
                code = int(getattr(status, "value", status)) if status is not None else 0
            except Exception as exc:  # noqa: BLE001
                status, code = repr(exc)[:80], -1
            if code != 0:
                failures.append((layer, nbytes, str(status)))
                self._clear_cuda_error()
        if failures:
            print(f"SlideFormer sharded optimizer [group rank {self.rank}, global rank "
                  f"{dist.get_rank()}]: {len(failures)} shared bf16 buffers could not be pinned "
                  f"(H2D from them is slower): {failures[:4]}", flush=True)

    # --------------------------------------------------------------- per step
    def barrier(self):
        dist.barrier(group=self.gloo_dp)

    def accumulator(self, owner, param):
        start, end = owner._param_flat_offsets[param]
        return self.acc[owner.layer_idx][self.rank][start:end].view_as(param)

    def accumulate(self, owner, param, source, first):
        view = self.accumulator(owner, param)
        if first:
            view.copy_(source)
        else:
            view.add_(source)
        return view

    def _tp_special_reduce(self, owner, out, lo, hi):
        """Megatron's TP-special gradient reduction, applied to this rank's slice."""
        if self.gloo_tp is None:
            return
        entries = self.special[owner.layer_idx]
        for kind in ("sum", "avg"):
            pieces = []
            for start, end, k in entries:
                a, b = max(start, lo), min(end, hi)
                if k == kind and b > a:
                    pieces.append(out[a - lo : b - lo])
            if not pieces:
                continue  # tp peers share slice boundaries, so they skip together
            flat = torch.cat(pieces)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.gloo_tp)
            if kind == "avg":
                flat.div_(self.tp_size)
            offset = 0
            for piece in pieces:
                piece.copy_(flat[offset : offset + piece.numel()])
                offset += piece.numel()

    def reduce_owner(self, owner, divide_by_size, scale):
        """Sum the replicas on this rank's slice (fixed rank order), TP-reduce, scale."""
        L = owner.layer_idx
        lo, hi = self.bounds[L][self.rank]
        accs = self.acc[L]
        if not self.special[L] and self.reduce_threads > 0:
            # No TP-special step has to run between the sum and the scale: do it in one pass.
            self._fused_reduce(accs, lo, hi, divide_by_size, scale)
            return
        out = accs[self.rank][lo:hi]
        if self.rank < 2:
            # `out` is an input for rank 0/1 and is only consumed elementwise
            # before being overwritten, so the sum can land in place.
            torch.add(accs[0][lo:hi], accs[1][lo:hi], out=out)
            for r in range(2, self.size):
                out.add_(accs[r][lo:hi])
        else:
            # For rank >= 2 `out` still holds this rank's own contribution, which the
            # ordered sum must read after ranks 0 and 1: accumulate in scratch.
            scratch = self._scratch_buffer(hi - lo)
            torch.add(accs[0][lo:hi], accs[1][lo:hi], out=scratch)
            for r in range(2, self.size):
                scratch.add_(accs[r][lo:hi])
            out.copy_(scratch)
        if divide_by_size:
            out.div_(self.size)
        self._tp_special_reduce(owner, out, lo, hi)
        if scale is not None:
            out.mul_(scale)

    def _fused_reduce(self, accs, lo, hi, divide_by_size, scale):
        """out = (((a0 + a1) + a2) ...) [/ N] [* scale] with a single pass over memory.

        The operations and their order are exactly those of the unfused path, so the
        result is bit-identical; only the memory traffic changes.
        """
        n = hi - lo
        chunk = self.reduce_chunk
        threads = self.reduce_threads
        if threads <= 1 or n < chunk * threads * 2:
            self._reduce_range(accs, lo, hi, divide_by_size, scale)
            return
        if self._reduce_pool is None:
            self._reduce_pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="slideformer-reduce")
        per = ((n + threads - 1) // threads + chunk - 1) // chunk * chunk
        futures = []
        for a in range(lo, hi, per):
            futures.append(self._reduce_pool.submit(
                self._reduce_range, accs, a, min(a + per, hi), divide_by_size, scale))
        for future in futures:
            future.result()

    def _reduce_range(self, accs, a, b, divide_by_size, scale):
        chunk = self.reduce_chunk
        tmp_all = getattr(self._reduce_local, "tmp", None)
        if tmp_all is None or tmp_all.numel() < chunk:
            tmp_all = self._reduce_local.tmp = torch.empty(chunk, dtype=torch.float32)
        out = accs[self.rank]
        for start in range(a, b, chunk):
            end = min(start + chunk, b)
            tmp = tmp_all[: end - start]
            torch.add(accs[0][start:end], accs[1][start:end], out=tmp)
            for r in range(2, self.size):
                tmp.add_(accs[r][start:end])
            if divide_by_size:
                tmp.div_(self.size)
            if scale is not None:
                tmp.mul_(scale)
            out[start:end].copy_(tmp)

    def _scratch_buffer(self, numel):
        buffer = getattr(self, "_scratch", None)
        if buffer is None or buffer.numel() < numel:
            self._scratch = buffer = torch.empty(numel, dtype=torch.float32)
        return buffer[:numel]

    def reduce_slices(self, divide_by_size, scale, owners=None):
        for owner in (self.owners if owners is None else owners):
            self.reduce_owner(owner, divide_by_size, scale)

    def adam_owner(self, owner, grad_slice=None):
        """Run CPU Adam on this rank's slice of one layer.

        ``grad_slice`` (length hi-lo, already reduced and scaled) replaces the shared
        accumulator as the gradient source when the replica sum was done on the GPU.
        """
        t_adam = time.perf_counter()
        optimizer = self.engine.layer_optimizer
        L = owner.layer_idx
        lo, hi = self.bounds[L][self.rank]
        group = optimizer.param_groups[owner.optimizer_layer_idx]
        beta1, beta2 = group["betas"]
        group["step"] += 1
        optimizer_id = optimizer._optimizer_id_for_layer(group["layer_idx"])
        master, grad = self.master[L], self.acc[L][self.rank]
        ranges = []
        for param in owner.params:
            start, end = owner._param_flat_offsets[param]
            a, b = max(start, lo), min(end, hi)
            if b <= a:
                continue
            decay = group["weight_decay"] * getattr(param, "_slideformer_wd_mult", 1.0)
            # Fuse adjacent parameter segments only when their Adam settings
            # and SIMD boundaries match. Odd-sized parameters retain the old
            # calls so scalar tails keep identical arithmetic.
            if (ranges and ranges[-1][1] == a and ranges[-1][2] == decay
                    and (a - ranges[-1][0]) % _ALIGN == 0):
                ranges[-1] = (ranges[-1][0], b, decay)
            else:
                ranges.append((a, b, decay))
        for a, b, decay in ranges:
            optimizer.cpu_adam.adam_update(
                optimizer_id,
                group["step"],
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                decay,
                group["bias_correction"],
                master[a:b],
                grad_slice[a - lo : b - lo] if grad_slice is not None else grad[a:b],
                self.exp_avg[L][a - lo : b - lo],
                self.exp_avg_sq[L][a - lo : b - lo],
            )
        owner.step = group["step"]
        self.phase_times["adam"] += time.perf_counter() - t_adam
        t_cast = time.perf_counter()
        if self.exec_cache:  # refresh this rank's slice of the shared bf16 copy
            self.exec[L][lo:hi].copy_(master[lo:hi])
        self.phase_times["cast"] += time.perf_counter() - t_cast

    def adam_slices(self, owners=None):
        for owner in (self.owners if owners is None else owners):
            self.adam_owner(owner)

    # ------------------------------------------------- eager (per-layer) mode
    def submit(self, owner, scale):
        """Queue reduce + scale + Adam for a layer whose gradients are final.

        A single FIFO worker keeps the layer order identical on every rank of
        the replica group, so the per-layer gloo collectives always pair up.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="slideformer-shardopt"
            )
        self._futures.append(self._executor.submit(self._eager_job, owner, scale))

    def _eager_job(self, owner, scale):
        t = time.perf_counter()
        self.barrier()  # every replica finished accumulating this layer
        t1 = time.perf_counter()
        self.reduce_owner(owner, divide_by_size=False, scale=scale)
        t2 = time.perf_counter()
        self.adam_owner(owner)
        self.done.add(owner.layer_idx)
        pt = self.phase_times
        pt["barrier"] += t1 - t
        pt["reduce"] += t2 - t1
        pt["layers"] += 1

    def submit_gpu_job(self, owner, event, first, last, cpu_scale, ctx):
        """Finish a layer whose replica sum was done by the GPU reduce-scatter.

        No cross-rank barrier: the reduce-scatter only completes once every rank has
        finished this layer's backward (and therefore its parameter loads), so updating
        the shared master / bf16 copy here cannot race with a peer's H2D copy.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="slideformer-shardopt"
            )
        future = self._executor.submit(self._gpu_job, owner, event, first, last, cpu_scale, ctx)
        ctx.last_job[owner.layer_idx] = future
        ctx.register_update(owner.layer_idx, future)
        self._futures.append(future)
        return future

    def _gpu_job(self, owner, event, first, last, cpu_scale, ctx):
        L = owner.layer_idx
        t = time.perf_counter()
        event.synchronize()  # reduced slice is in pinned host memory
        t1 = time.perf_counter()
        if self.engine._expected_microbatches > 1:
            # The DMA buffer may be BF16, but never round the running sum to BF16.
            if first:
                ctx.sacc[L].copy_(ctx.stmp[L])
            else:
                ctx.sacc[L].add_(ctx.stmp[L])
        if last:
            if cpu_scale is not None:
                ctx.sacc[L].mul_(cpu_scale)
            self.adam_owner(owner, grad_slice=ctx.sacc[L])
            self.done.add(L)
            self.phase_times["layers"] += 1
        self.phase_times["barrier"] += t1 - t  # here: waiting for GPU reduce + D2H

    def drain(self):
        futures, self._futures = self._futures, []
        for future in futures:
            future.result()

    def reset_step(self):
        self.done.clear()
        self._futures = []
        self.phase_times = {"barrier": 0.0, "reduce": 0.0, "adam": 0.0, "cast": 0.0, "layers": 0}
