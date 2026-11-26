# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer-ready reductions and CPU Adam without full-model GPU accumulators.

MCore owns the model, native dense/expert/TP groups and reduction semantics.
The adapter retains CPU microbatch accumulators and leases bounded communication
scratch. Only a layer's last backward may submit its update. Unknown token
scales permit early reduction, but postpone Adam until native finalization.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
import os
import threading
import time

import torch
import torch.distributed as dist

from .compat import get_global_num_tokens, get_tensor_parallel_gradient_reduction
from .distributed import synchronize_gradient


def group_size(group):
    return dist.get_world_size(group) if group is not None else 1


class GradientLease:
    """Keep a D2H destination alive until a worker has consumed its contents."""
    def __init__(self, gradients, event, pool, slot):
        self.gradients, self.event, self.pool, self.slot = gradients, event, pool, slot
        self.released = False

    def wait(self):
        if self.event is not None:
            self.event.synchronize()

    def release(self):
        if not self.released:
            # Also used on cancellation: a queued D2H may still be writing.
            self.wait()
            if self.slot is not None:
                self.pool.release(self.slot)
            self.released = True


def communication_device(device, config):
    """Keep background collectives off native model-parallel CUDA streams.

    Separate NCCL groups do not establish a globally consistent launch order
    with native TP/CP/EP work. In particular, CP checkpoint recomputation may
    wait for a peer whose D2H stream already waits for our replica reduction.
    These configurations reduce CPU-owned gradients through private Gloo
    groups instead. Pure DP retains its bounded GPU communication path.
    """
    parallel = any(getattr(config, name, 1) > 1 for name in (
        "tensor_model_parallel_size", "context_parallel_size", "expert_model_parallel_size"))
    return torch.device("cpu") if parallel else device


def copy_native_membership(group, device):
    """Register a communicator with MCore; keep its existing rank ownership.

    Background replica/TP reductions must not interleave with native forward,
    backward or CP collectives on the same communicator.
    """
    from megatron.core.parallel_state import create_group

    members = tuple(dist.get_process_group_ranks(group))
    memberships = [None] * dist.get_world_size()
    dist.all_gather_object(memberships, members)
    local = None
    for ranks in sorted(set(memberships)):
        new_group = create_group(
            ranks=list(ranks), backend="nccl" if device.type == "cuda" else "gloo",
            group_desc="slide-layer-update",
        )
        if dist.get_rank() in ranks:
            local = new_group
    if local is None:
        raise RuntimeError("Missing local native gradient group")
    return local


class OrderedLayerWork:
    """Bounded submission in identical layer order on every replica."""
    def __init__(self, owners, operation, depth):
        if depth < 1:
            raise ValueError("Layer update queue depth must be positive")
        self.owners = tuple(owners)
        self.operation = operation
        self.slots = threading.BoundedSemaphore(depth)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slide-native-update")
        self.reset()

    def reset(self):
        if getattr(self, "futures", None):
            self.drain()
        self.ready, self.futures, self.cursor = set(), [], 0
        self.error = None

    def submit_ready(self, owner):
        key = owner.layer_idx
        if key in self.ready or key not in {o.layer_idx for o in self.owners}:
            raise RuntimeError(f"Duplicate or unknown ready layer {key}")
        self.ready.add(key)
        while self.cursor < len(self.owners):
            current = self.owners[self.cursor]
            if current.layer_idx not in self.ready:
                break
            # Observe errors before enqueueing more work. The worker always
            # releases its slot, so an exception cannot strand backpressure.
            for future in self.futures:
                if future.done():
                    future.result()
            self.slots.acquire()
            try:
                if self.error is not None:
                    raise RuntimeError("Earlier layer update failed") from self.error
                self.futures.append(self.pool.submit(self._run, current))
            except BaseException:
                self.slots.release()
                raise
            self.cursor += 1

    def _run(self, owner):
        try:
            if self.error is not None:
                raise RuntimeError("Earlier layer update failed") from self.error
            self.operation(owner)
        except BaseException as exc:
            if self.error is None:
                self.error = exc
            raise
        finally:
            self.slots.release()

    def drain(self):
        for future in self.futures:
            future.result()

    def finish(self):
        if self.cursor != len(self.owners):
            raise RuntimeError("Missing final backward for layer update pipeline")
        self.drain()

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True)


class LayerUpdatePipeline:
    def __init__(self, engine, *, group, tp_group, expert_group, pp_group):
        if group_size(pp_group) != 1:
            raise NotImplementedError("Layer updates require PP=1")
        if engine._sharded_opt is not None or engine.gpu_gradient_pipeline is not None:
            raise ValueError("Select one gradient/update owner, not two pipelines")
        if not engine.config.defer_optimizer_step or engine.unmanaged_params:
            raise ValueError("Layer pipeline needs fully managed deferred updates")
        if engine.config.skip_grad_d2h or engine.layer_optimizer.offloader is not None:
            raise NotImplementedError("Layer pipeline requires gradient copies and CPU-resident Adam states")
        self.engine, self.device = engine, torch.device(engine.device)
        self.trace = (self.device.type == "cuda" and
                      os.getenv("MEGATRON_SLIDEFORMER_TRACE_PIPELINE", "0") == "1")
        self.original_group, self.pp_group = group, pp_group
        self.replica_size = group_size(group)
        self.native_config = engine.layout.model.config
        self.communication_device = communication_device(self.device, self.native_config)
        self.group = copy_native_membership(group, self.communication_device)
        self.expert_group = copy_native_membership(expert_group, self.communication_device)
        self.tp_group = copy_native_membership(tp_group, self.communication_device)
        self.names = {id(p): n for n, p in engine.model.named_parameters()}
        self.owners = tuple(o for o in reversed(engine.managed_layers) if o.params)
        self.stream = (torch.cuda.Stream(device=self.device)
                       if self.communication_device.type == "cuda" else None)
        self.work = OrderedLayerWork(self.owners, self._reduce_and_maybe_update,
                                     engine.config.optimizer_pipeline_depth)
        self.counts, self.updated = {}, set()
        self.scale, self.active = None, False
        self.stats = {}
        self.leases = {}

    def begin(self, microbatches):
        if self.active or microbatches < 1:
            raise RuntimeError("Invalid layer update step boundary")
        self.work.reset()
        if self.leases:
            raise RuntimeError("Unconsumed layer gradient leases at step boundary")
        self.microbatches = microbatches
        self.counts = {o.layer_idx: 0 for o in self.owners}
        self.updated = set()
        self.scale = None if self.native_config.calculate_per_token_loss else 1.0 / self.replica_size
        self.stats = {"layers_reduced": 0, "layers_updated_before_finalize": 0,
                      "layers_updated_with_precomputed_scale": 0,
                      "first_reduce_started": None, "first_update_started": None,
                      "finalize_entered": None, "gpu_full_model_accumulator_bytes": 0,
                      "communication_device": str(self.communication_device),
                      "communication_backend": "nccl" if self.communication_device.type == "cuda" else "gloo",
                      "communication_chunk_bytes": self.engine.config.chunk_size_mb * 1024**2}
        self.stats["async_gradient_captures"] = 0
        self.active = True

    def capture_single_batch(self, owner):
        """Move the single-batch D2H wait and CPU copy off the autograd thread.

        Model-parallel and accumulation paths retain their qualified behavior.
        The worker owns the detached CPU pool slot; GPU slot reuse waits on the
        recorded copy event. Finalization drains every update before next-step
        parameter prefetch. Queue and pool bounds supply backpressure.
        """
        enabled = os.getenv("MEGATRON_SLIDEFORMER_ASYNC_LAYER_CAPTURE", "0") == "1"
        parallel = any(getattr(self.native_config, name, 1) > 1 for name in (
            "tensor_model_parallel_size", "context_parallel_size", "expert_model_parallel_size"))
        if not enabled or self.microbatches != 1 or parallel:
            return False
        if not self.active or owner.layer_idx not in self.counts or self.counts[owner.layer_idx]:
            raise RuntimeError("Unexpected single-batch layer backward")
        event = None
        if owner._grad_copy_submitted:
            event = torch.cuda.Event()
            event.record(owner.d2h_stream)
            owner.grad_copy_event = event
            # Match the original prepare_grads_for_async_step storage fence:
            # non-fused TE backwards can still consult Parameter.data. Only
            # fused owners preserve their CUDA storage during offload.
            if not owner._te_fused_params:
                event.synchronize()
        gradients = {param: owner.cpu_grads[param] for param in owner.params if param in owner._grad_seen}
        lease = GradientLease(gradients, event, owner.cpu_grad_pool, owner._cpu_grad_pool_slot)
        self.leases[owner.layer_idx] = lease
        owner._cpu_grad_pool_slot = None
        try:
            owner.offload_params(force=True)
            owner._release_grad_buffer()  # GPU only; its reuse is fenced by event.
            owner._pending_backward_step = False
            owner._module_backward_done = False
            owner._backward_ready_reported = False
            owner._grad_seen.clear()
            owner._param_accumulation_counts.clear()
            owner._grad_copy_submitted = False
            self.counts[owner.layer_idx] = 1
            self.stats["async_gradient_captures"] += 1
            self.work.submit_ready(owner)
        except BaseException:
            # The close path also drains workers before reclaiming any leases.
            # Never release an enqueued lease here while its worker can read it.
            raise
        return True

    def _consume_lease(self, owner):
        lease = self.leases.get(owner.layer_idx)
        if lease is None:
            return
        try:
            with self._trace_range("gradient_wait_copy", owner):
                lease.wait()
                gradients = self.engine._accumulated_grads.setdefault(owner.layer_idx, {})
                for param in owner.params:
                    source = lease.gradients.get(param)
                    gradients[param] = self.engine._pooled_grad_accumulator(
                        owner.layer_idx, param, source if source is not None else owner.cpu_params[param],
                        zero=source is None)
        finally:
            lease.release()
            del self.leases[owner.layer_idx]

    def set_expected_scale(self, scale):
        if not self.active or self.work.ready:
            raise RuntimeError("Token scale must be set before any layer finishes backward")
        self.scale = scale

    def capture(self, owner):
        if not self.active or owner.layer_idx not in self.counts:
            raise RuntimeError("Unexpected layer backward")
        self.counts[owner.layer_idx] += 1
        count = self.counts[owner.layer_idx]
        if count > self.microbatches:
            raise RuntimeError("Too many microbatch backwards")
        if count == self.microbatches:
            gradients = self.engine._accumulated_grads.setdefault(owner.layer_idx, {})
            for param in owner.params:
                if param not in gradients:
                    gradients[param] = self.engine._pooled_grad_accumulator(
                        owner.layer_idx, param, owner.cpu_params[param], zero=True)
            self.work.submit_ready(owner)

    def _context(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()

    def _trace_range(self, phase, owner):
        return (torch.cuda.nvtx.range(f"slideformer.{phase}.layer.{owner.layer_idx}")
                if self.trace else nullcontext())

    def _reduce_and_maybe_update(self, owner):
        with self._context(), torch.no_grad(), self._trace_range("reduce_update", owner):
            self._consume_lease(owner)
            self.stats["first_reduce_started"] = self.stats["first_reduce_started"] or time.perf_counter()
            gradients = self.engine._accumulated_grads[owner.layer_idx]
            for param in owner.params:
                group = self.group if getattr(param, "allreduce", True) else self.expert_group
                synchronize_gradient(gradients[param], group=group, device=self.communication_device,
                                     chunk_size_mb=self.engine.config.chunk_size_mb,
                                     average=False, async_op=True)
            # Reuse MCore's TP selection, ordering and SUM/AVG implementation.
            if group_size(self.tp_group) > 1 and self.communication_device.type == "cpu":
                # Keep MCore's parameter selection and SUM/AVG semantics.
                # Gloo lacks ReduceOp.AVG, so use SUM followed by division.
                for param in owner.params:
                    reduction = get_tensor_parallel_gradient_reduction(
                        self.names[id(param)], param, self.native_config)
                    if reduction:
                        synchronize_gradient(
                            gradients[param], group=self.tp_group, device=self.communication_device,
                            chunk_size_mb=self.engine.config.chunk_size_mb,
                            average=reduction == "avg", async_op=True)
            elif group_size(self.tp_group) > 1:
                from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
                selected, previous = [], {}
                for param in owner.params:
                    name = self.names[id(param)]
                    if get_tensor_parallel_gradient_reduction(name, param, self.native_config):
                        previous[param] = getattr(param, "main_grad", None)
                        param.main_grad = gradients[param].to(self.device)
                        selected.append((name, param))
                try:
                    adapter = SimpleNamespace(
                        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
                        named_parameters=lambda: iter(selected))
                    _allreduce_non_tensor_model_parallel_grads([adapter], self.native_config, self.tp_group)
                    for _, param in selected:
                        gradients[param].copy_(param.main_grad.to("cpu"))
                finally:
                    for param, old in previous.items():
                        param.main_grad = old
            self.stats["layers_reduced"] += 1
            if self.scale is not None:
                self._update(owner, self.scale, early=True)

    def _update(self, owner, scale, *, early):
        gradients = self.engine._accumulated_grads[owner.layer_idx]
        for grad in gradients.values():
            grad.mul_(scale)
        original = owner.cpu_grads
        owner.cpu_grads = gradients
        try:
            if not self.engine.config.skip_cpu_adam_step:
                self.stats["first_update_started"] = self.stats["first_update_started"] or time.perf_counter()
                owner._record_lifecycle("optimizer_start")
                with self._trace_range("adam", owner):
                    owner._step_layer_optimizer()
                owner._record_lifecycle("optimizer_end")
        finally:
            owner.cpu_grads = original
        self.updated.add(owner.layer_idx)
        self.stats["layers_updated_with_precomputed_scale"] += int(early)
        self.stats["layers_updated_before_finalize"] += int(self.stats["finalize_entered"] is None)

    def finalize(self, num_tokens):
        self.stats["finalize_entered"] = time.perf_counter()
        if not self.active or any(n != self.microbatches for n in self.counts.values()):
            raise RuntimeError("Incomplete layer backward accumulation")
        self.work.finish()
        actual_scale = 1.0 / self.replica_size
        if num_tokens is not None:
            total = get_global_num_tokens(num_tokens, self.pp_group, self.original_group)
            actual_scale = 1.0 / float(total.item()) if total.item() > 0 else 1.0
        elif self.native_config.calculate_per_token_loss:
            raise ValueError("Per-token loss requires local num_tokens")
        if self.scale is not None and self.scale != actual_scale:
            raise RuntimeError(f"Early token scale {self.scale} != final native scale {actual_scale}")
        with torch.no_grad():
            for owner in self.owners:
                if owner.layer_idx not in self.updated:
                    self._update(owner, actual_scale, early=False)
        self.active = False

    def close(self):
        try:
            self.work.close()
        finally:
            # A prior failure skips queued operations. Their slots must remain
            # leased until the worker has stopped and their D2H writes finish.
            for lease in self.leases.values():
                lease.release()
            self.leases.clear()
