"""Group-aware SlideFormer gradient pipeline for Megatron parallel layouts.

Megatron owns model-parallel computation. This module accumulates local
parameter shards, runs replica reductions on Megatron's groups, and streams
the reduced shards to LayerAdam.
"""
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.distributed as dist


class GPUGradientPipeline:
    @staticmethod
    def estimate_accumulator_bytes(owners):
        size = torch.empty((), dtype=torch.float32).element_size()
        return sum(owner.cpu_grad_flat.numel() * size for owner in owners if owner.params)

    @staticmethod
    def validate_memory_budget(required_bytes, max_memory_mb):
        if max_memory_mb < 1:
            raise ValueError("GPU gradient memory budget must be at least 1 MiB")
        if required_bytes > max_memory_mb * 1024**2:
            raise MemoryError(
                f"GPU gradient accumulators require {required_bytes / 1024**2:.1f} MiB, "
                f"exceeding the configured {max_memory_mb} MiB budget"
            )

    def __init__(
        self,
        engine,
        dense_replica_group,
        tp_group,
        *,
        expert_replica_group=None,
        expert_tp_group=None,
        pp_group=None,
        model_chunk_id=0,
    ):
        self.engine = engine
        self.model_chunk_id = int(model_chunk_id)
        self.tp_group = tp_group
        self.expert_tp_group = expert_tp_group
        self.pp_group = pp_group
        self.dense_replica_group = dense_replica_group
        self.expert_replica_group = expert_replica_group or dense_replica_group
        self.dense_replica_size = self._group_size(dense_replica_group)
        self.tp_size = self._group_size(tp_group)
        if engine.unmanaged_params:
            raise ValueError("GPU gradient pipeline requires fully managed parameters")
        if any(not owner._flat_grad_copy_supported for owner in engine.managed_layers if owner.params):
            raise ValueError("GPU gradient pipeline requires layer-flat gradient hooks")

        self.owners = [owner for owner in reversed(engine.managed_layers) if owner.params]
        self.names = {id(param): name for name, param in engine.model.named_parameters()}
        self.accumulator_bytes = self.estimate_accumulator_bytes(self.owners)
        self._validate_budget_across_ranks()

        # Separate communicators keep layer-gradient traffic independent from
        # CP attention and Megatron's native DP collectives.
        self.groups = []
        self.dense_group = self._duplicate_membership(dense_replica_group)
        self.expert_group = (
            self.dense_group
            if self._same_membership(self.expert_replica_group, dense_replica_group)
            else self._duplicate_membership(self.expert_replica_group)
        )
        self.stream = torch.cuda.Stream(device=engine.device)
        self.buffers = {
            owner.layer_idx: torch.zeros(
                owner.cpu_grad_flat.numel(), dtype=torch.float32, device=engine.device
            )
            for owner in self.owners
        }
        self.host = {
            owner.layer_idx: torch.empty(
                owner.cpu_grad_flat.numel(), dtype=torch.float32, pin_memory=True
            )
            for owner in self.owners
        }
        self.host_bytes = sum(t.numel() * t.element_size() for t in self.host.values())
        self.update_executor = ThreadPoolExecutor(
            max_workers=max(1, int(engine.config.optimizer_worker_count)),
            thread_name_prefix=f"slideformer-chunk{self.model_chunk_id}-adam",
        )
        self.active = False
        if dist.get_rank() == 0:
            print(
                "SlideFormer group-aware gradient pipeline: "
                f"chunk={self.model_chunk_id}, device={self.accumulator_bytes / 1024**2:.1f} MiB, "
                f"host={self.host_bytes / 1024**2:.1f} MiB",
                flush=True,
            )

    @staticmethod
    def _group_size(group):
        return dist.get_world_size(group) if group is not None else 1

    @staticmethod
    def _members(group):
        if group is None:
            return tuple(range(dist.get_world_size()))
        return tuple(dist.get_process_group_ranks(group))

    def _same_membership(self, lhs, rhs):
        return self._members(lhs) == self._members(rhs)

    def _duplicate_membership(self, source_group):
        from megatron.core.parallel_state import create_group

        local_members = self._members(source_group)
        memberships = [None] * dist.get_world_size()
        dist.all_gather_object(memberships, local_members)
        selected = None
        for members in sorted(set(memberships)):
            group = create_group(ranks=list(members), backend="nccl", group_desc="slide-grad-pipeline")
            self.groups.append(group)
            if dist.get_rank() in members and members == local_members:
                selected = group
        if selected is None:
            raise RuntimeError(f"rank is absent from its gradient group {local_members}")
        return selected

    def _validate_budget_across_ranks(self):
        error = None
        max_memory_mb = 0
        free_bytes = 0
        try:
            max_memory_mb = int(os.getenv("MEGATRON_SLIDEFORMER_GPU_GRADIENT_MAX_MB", "8192"))
            self.validate_memory_budget(self.accumulator_bytes, max_memory_mb)
            free_bytes, _ = torch.cuda.mem_get_info(self.engine.device)
            if self.accumulator_bytes > free_bytes:
                raise MemoryError(
                    f"requires {self.accumulator_bytes / 1024**2:.1f} MiB, "
                    f"but only {free_bytes / 1024**2:.1f} MiB is free"
                )
        except (ValueError, MemoryError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, (error, self.accumulator_bytes, max_memory_mb, free_bytes))
        failures = {rank: report[0] for rank, report in enumerate(reports) if report[0]}
        if failures:
            raise MemoryError(f"GPU gradient pipeline memory preflight failed: {failures}")

    def begin(self, microbatches, *, per_token_loss=False):
        if self.active:
            raise RuntimeError("Previous gradient pipeline step was not consumed")
        if microbatches < 1:
            raise ValueError("microbatches must be positive")
        self.microbatches = int(microbatches)
        self.per_token_loss = bool(per_token_loss)
        self.counts = {owner.layer_idx: 0 for owner in self.owners}
        self.ready = {}
        self.cursor = 0
        self.works = []
        self.update_futures: list[Future] = []
        self.committed = set()
        self.scale = None if self.per_token_loss else 1.0 / self.dense_replica_size
        self.early_update = self.scale is not None and self._group_size(self.pp_group) <= 1
        self.active = True
        self.reductions_finalized = False
        self.finalized = False
        with torch.cuda.stream(self.stream):
            for buffer in self.buffers.values():
                buffer.zero_()
            self.zero_event = torch.cuda.Event()
            self.zero_event.record(self.stream)

    def capture(self, owner):
        if not self.active:
            raise RuntimeError("GPU gradient pipeline must begin before backward")
        idx = owner.layer_idx
        if self.counts[idx] >= self.microbatches:
            raise RuntimeError("Unexpected extra layer backward in gradient pipeline")
        source = owner._ensure_gpu_grad_flat()
        compute = torch.cuda.current_stream(self.engine.device)
        with torch.cuda.stream(owner.d2h_stream):
            owner.d2h_stream.wait_stream(compute)
            owner.d2h_stream.wait_event(self.zero_event)
            with torch.profiler.record_function("slideformer/gpu_gradient_accumulate"):
                self.buffers[idx].add_(source)
            event = torch.cuda.Event()
            event.record(owner.d2h_stream)
            source.record_stream(owner.d2h_stream)
        owner.grad_copy_event = event
        owner._grad_copy_submitted = True
        self.counts[idx] += 1
        if self.counts[idx] == self.microbatches:
            self.ready[idx] = event
            self._drain_ready()

    @staticmethod
    def _is_expert(param):
        return not getattr(param, "allreduce", True)

    def _owner_segments(self, owner):
        result = []
        current_kind = None
        current_start = None
        current_end = None
        for param in owner.params:
            start, end = owner._param_flat_offsets[param]
            kind = "expert" if self._is_expert(param) else "dense"
            if current_kind is not None and (kind != current_kind or start != current_end):
                result.append((current_kind, current_start, current_end))
                current_start = None
            if current_start is None:
                current_kind, current_start = kind, start
            current_end = end
        if current_start is not None:
            result.append((current_kind, current_start, current_end))
        return result

    def _reduce_tp_special_grads(self, owner, buffer):
        from .compat import get_tensor_parallel_gradient_reduction
        config = getattr(self.engine.layout.model, "config", None)
        selected = []
        saved = {}
        for param in owner.params:
            name = self.names[id(param)]
            if get_tensor_parallel_gradient_reduction(name, param, config):
                start, end = owner._param_flat_offsets[param]
                saved[param] = getattr(param, "main_grad", None)
                param.main_grad = buffer[start:end].view_as(owner.cpu_params[param])
                selected.append((name, param))
        try:
            if selected and self.tp_size > 1:
                from megatron.core.distributed.finalize_model_grads import (
                    _allreduce_non_tensor_model_parallel_grads,
                )
                adapter = SimpleNamespace(
                    ddp_config=SimpleNamespace(use_megatron_fsdp=False),
                    named_parameters=lambda: iter(selected),
                )
                _allreduce_non_tensor_model_parallel_grads([adapter], config, self.tp_group)
        finally:
            for param, old in saved.items():
                param.main_grad = old

    def _drain_ready(self):
        with torch.cuda.stream(self.stream), torch.no_grad():
            while self.cursor < len(self.owners):
                owner = self.owners[self.cursor]
                event = self.ready.get(owner.layer_idx)
                if event is None:
                    break
                self.stream.wait_event(event)
                buffer = self.buffers[owner.layer_idx]
                chunk = max(1, self.engine.config.chunk_size_mb * 1024**2 // 4)
                with torch.profiler.record_function("slideformer/async_gradient_reduce"):
                    for kind, begin, end in self._owner_segments(owner):
                        group = self.expert_group if kind == "expert" else self.dense_group
                        if self._group_size(group) <= 1:
                            continue
                        for start in range(begin, end, chunk):
                            stop = min(start + chunk, end)
                            work = dist.all_reduce(buffer[start:stop], group=group, async_op=True)
                            work.wait()
                            self.works.append(work)
                # Preserve Megatron's native ordering: replica reductions
                # (DP/CP or expert-DP) finish before SP/QK/TP-special grads
                # are reduced across the TP domain.
                self._reduce_tp_special_grads(owner, buffer)
                reduced = torch.cuda.Event()
                reduced.record(self.stream)
                self.ready[owner.layer_idx] = reduced
                self.cursor += 1
                if self.early_update:
                    self._commit_owner(owner)

    def set_global_token_count(self, num_tokens, *, pp_group=None):
        if num_tokens is None:
            return
        from .compat import get_global_num_tokens

        tokens = get_global_num_tokens(num_tokens, pp_group, self.dense_replica_group)
        self.scale = 1.0 / float(tokens.item()) if tokens.item() > 0 else 1.0

    def finalize_reductions(self, num_tokens=None, *, pp_group=None):
        if not self.active or self.reductions_finalized:
            raise RuntimeError("Invalid GPU gradient finalization lifecycle")
        if any(count != self.microbatches for count in self.counts.values()):
            raise RuntimeError(f"Incomplete microbatch accumulation: {self.counts}")
        self._drain_ready()
        if self.cursor != len(self.owners):
            raise RuntimeError("Not all gradient buckets were submitted")
        if self.per_token_loss:
            self.set_global_token_count(num_tokens, pp_group=pp_group)
        if self.scale is None:
            raise RuntimeError("Per-token SlideFormer update is missing its global token count")
        self.reductions_finalized = True

    @contextmanager
    def bind_main_grads(self):
        saved = {}
        current = torch.cuda.current_stream(self.engine.device)
        current.wait_stream(self.stream)
        try:
            for owner in self.owners:
                buffer = self.buffers[owner.layer_idx]
                for param in owner.params:
                    start, end = owner._param_flat_offsets[param]
                    saved[param] = getattr(param, "main_grad", None)
                    param.main_grad = buffer[start:end].view_as(owner.cpu_params[param])
            yield
        finally:
            # Native PP collectives run on the caller's stream. D2H/Adam must
            # consume their result, rather than the pre-collective buffer.
            self.stream.wait_stream(current)
            for param, old in saved.items():
                param.main_grad = old

    def _commit_owner(self, owner):
        idx = owner.layer_idx
        if idx in self.committed or self.scale is None:
            return
        ready = self.ready.get(idx)
        if ready is None:
            return
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.stream.wait_event(ready)
            self.host[idx].copy_(self.buffers[idx].mul_(self.scale), non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(self.stream)
        gradients = {
            param: self.host[idx][slice(*owner._param_flat_offsets[param])].view_as(owner.cpu_params[param])
            for param in owner.params
        }

        def update():
            copied.synchronize()
            original = owner.cpu_grads
            owner.cpu_grads = gradients
            try:
                if not owner.config.skip_cpu_adam_step:
                    owner._record_lifecycle("optimizer_start")
                    owner._step_layer_optimizer()
                    owner._record_lifecycle("optimizer_end")
            finally:
                owner.cpu_grads = original

        self.update_futures.append(self.update_executor.submit(update))
        self.committed.add(idx)

    def commit_updates(self):
        if not self.reductions_finalized:
            raise RuntimeError("Gradient reductions must finish before updates are committed")
        for owner in self.owners:
            self._commit_owner(owner)
        for future in self.update_futures:
            future.result()
        self.finalized = True

    def consumed(self):
        if not self.finalized:
            raise RuntimeError("Gradient pipeline update precedes finalization")
        self.works.clear()
        self.active = False

    def close(self):
        self.update_executor.shutdown(wait=True)
