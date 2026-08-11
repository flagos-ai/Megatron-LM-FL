import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed

from megatron.core import parallel_state
from megatron.megalens.hardware_monitor import HardwareMonitor

# -- Tracing Granularity Sets --
# These sets define which events are captured at different granularity levels.

# All events
# megatron/training/training.py
#     1379:optimizer

# megatron/core/tensor_parallel/mappings.py
#     35:_reduce
#     113:_gather_along_last_dim
#     163:_reduce_scatter_along_last_dim
#     213:_gather_along_first_dim
#     287:_reduce_scatter_along_first_dim

# megatron/core/transformer/transformer_layer.py
#     393:transformer_layer_{self.layer_number}
#     396:_forward_attention_{self.layer_number}
#     398:_forward_mlp_{self.layer_number}
#     543:recompute_mlp_{self.layer_number}
#     556:mlp_{self.layer_number}

# megatron/core/transformer/mlp.py
#     110:MLP.forward

# megatron/core/transformer/attention.py
#     513:attention

# megatron/core/pipeline_parallel/schedules.py
#     294:loss
#     1897:recv-warmup
#     1904:forward-warmup
#     1919:send-warmup
#     1956:recv-extra
#     1983:forward
#     2028:exchange-next
#     2055:backward
#     2068:send-extra
#     2075:exchange-prev
#     2102:recv-cooldown
#     2105:backward-cooldown
#     2110:send-cooldown
#     2127:grad-sync
#     2143:allreduce

# megatron/core/models/gpt/gpt_model.py
#     337:decoder
BASE_TRACING_EVENTS = {
    '_reduce',
    '_gather_along_last_dim',
    '_gather_along_first_dim',
    '_reduce_scatter_along_last_dim',
    '_reduce_scatter_along_first_dim',
    'forward',
    'forward-warmup',
    'forward-step',
    'combined-forward-backward-step',
    'backward',
    'backward-cooldown',
    'backward-step',
    'optimizer',
    'loss',
    'allreduce',
    'grad-sync',
    'all-grads-sync',
    'p2p-launch',
    'p2p-batch-complete',
    'p2p-batch-device-sync',
    'bridge-p2p-launch',
    'bridge-grid-broadcast',
    'bridge-send-forward',
    'bridge-recv-forward',
    'bridge-send-backward',
    'bridge-recv-backward',
    'send-warmup',
    'send-extra',
    'send-forward',
    'send-backward',
    'send-cooldown',
    'exchange-next',
    'exchange-prev',
    'recv-warmup',
    'recv-extra',
    'recv-forward',
    'recv-backward',
    'recv-cooldown',
    # DP communication (param_and_grad_buffer.py)
    'dp-reduce-scatter',
    'dp-allreduce',
    'dp-param-all-gather',
    'dp-grad-sync-complete',
    'dp-param-sync-complete',
    # TP communication (mappings.py, renamed for specificity)
    'tp-allreduce',
    'tp-all-gather-first',
    'tp-all-gather-last',
    'tp-reduce-scatter',
    'tp-reduce-scatter-last',
    'tp-linear-async-launch',
    'tp-linear-async-complete',
    # Gradient finalization sub-scopes (finalize_model_grads.py)
    'sp-layernorm-allreduce',
    'embedding-grads-allreduce',
    # EP / MoE (moe_layer.py, router.py, token_dispatcher.py)
    'moe-router',
    'moe-dispatch',
    'moe-experts',
    'moe-combine',
    'moe-shared-expert',
    'ep-alltoall-dispatch',
    'ep-alltoall-combine',
    'ep-alltoall-async-launch',
    'ep-alltoall-async-complete',
    'ep-allgather-dispatch',
    'ep-allgather-combine',
}
FULL_TRACING_EVENTS = {
    'optimizer',
    '_reduce',
    "_gather_along_last_dim",
    "_gather_along_first_dim",
    "_reduce_scatter_along_last_dim",
    "_reduce_scatter_along_first_dim",
    "transformer_layer",
    "_forward_attention",
    "_forward_mlp",
    "recompute_mlp",
    "mlp",
    "MLP.forward",
    "attention",
    "loss",
    "recv-warmup",
    "forward-warmup",
    "send-warmup",
    "recv-extra",
    "forward",
    "combined-forward-backward-step",
    "exchange-next",
    "backward",
    "send-extra",
    "exchange-prev",
    "recv-cooldown",
    "backward-cooldown",
    "send-cooldown",
    "grad-sync",
    "allreduce",
    "decoder",
    # DP communication
    "dp-reduce-scatter",
    "dp-allreduce",
    "dp-param-all-gather",
    "dp-grad-sync-complete",
    "dp-param-sync-complete",
    # TP communication (renamed)
    "tp-allreduce",
    "tp-all-gather-first",
    "tp-all-gather-last",
    "tp-reduce-scatter",
    "tp-reduce-scatter-last",
    "tp-linear-async-launch",
    "tp-linear-async-complete",
    # Gradient finalization sub-scopes
    "sp-layernorm-allreduce",
    "embedding-grads-allreduce",
    # EP / MoE (moe_layer.py, router.py, token_dispatcher.py)
    "moe-router",
    "moe-dispatch",
    "moe-experts",
    "moe-combine",
    "moe-shared-expert",
    "ep-alltoall-dispatch",
    "ep-alltoall-combine",
    "ep-alltoall-async-launch",
    "ep-alltoall-async-complete",
    "ep-allgather-dispatch",
    "ep-allgather-combine",
}


class _TracerScope:
    def __init__(
        self,
        tracer: "Tracer",
        name: Optional[str],
        in_attrs: Dict[str, Any],
        out_attrs: Dict[str, Any],
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.in_attrs = in_attrs
        self.out_attrs = out_attrs

    def __enter__(self) -> "_TracerScope":
        self.tracer._push_scope(self)
        if self.name is not None:
            # Only copy in_attrs if we need to remove slot placeholders.
            # Most scopes have no slots, so the copy is unnecessary.
            need_filter = any(
                k in self.out_attrs
                and self.out_attrs.get(k) is None
                and self.in_attrs.get(k) is True
                for k in self.in_attrs
            )
            if need_filter:
                begin_attrs = {
                    k: v
                    for k, v in self.in_attrs.items()
                    if not (k in self.out_attrs and self.out_attrs.get(k) is None and v is True)
                }
            else:
                begin_attrs = self.in_attrs  # no copy needed
            self.tracer._tick(self.name, "B", begin_attrs)
        return self

    def __exit__(self, type, value, traceback) -> None:
        if self.name is not None:
            self.tracer._tick(self.name, "E", self.out_attrs)
        self.tracer._pop_scope()

    def get(self, q: str) -> Optional[Any]:
        """Get from in_attrs."""
        return self.in_attrs.get(q)

    def set(self, q: str, v: Any) -> bool:
        """Set to out_attrs, if this is required."""
        if q in self.out_attrs and self.out_attrs[q] is not None:
            return True
        if q in self.out_attrs and self.out_attrs[q] is None:
            self.out_attrs[q] = v
            return True
        else:
            return False


class _NoopTracerScope:
    """Cheap context manager used by sentinel mode to skip framework scopes."""

    def __enter__(self) -> "_NoopTracerScope":
        return self

    def __exit__(self, type, value, traceback) -> None:
        return None

    def get(self, q: str) -> Optional[Any]:
        return None

    def set(self, q: str, v: Any) -> bool:
        return True


@dataclass
class _Pending:
    name: str
    phase: str
    event: Any
    attrs: Dict[str, Any]


def _trace_filename(
    *, global_rank: int, dp_rank: int, pp_rank: int, tp_rank: int, mode0: bool
) -> str:
    prefix = "mode0-sentinel" if mode0 else "benchmark"
    extension = "jsonl" if mode0 else "json"
    return (
        f"{prefix}-global-{global_rank}-data-{dp_rank}-"
        f"pipeline-{pp_rank}-tensor-{tp_rank}.{extension}"
    )


def _save_traces_to_disk_thread(work_queue: queue.Queue, trace_dir: str, error_queue: queue.Queue):
    """
    Worker thread that pulls trace data from a queue and saves it to disk.

    This runs in a separate thread to avoid blocking the main training loop.
    It supports two payload formats:
      * ``(filename, records)`` or ``(filename, records, "json")``: append
        records into a JSON array file.
      * ``(filename, records, "jsonl")``: append one JSON object per line.

    Args:
        work_queue: The queue to get trace data from.
        trace_dir: The directory where trace files will be saved.
    """
    while True:
        payloads = work_queue.get()
        try:
            if payloads is None:  # Sentinel for termination
                break

            for payload in payloads:
                if len(payload) == 2:
                    filename, new_records = payload
                    write_mode = "json"
                else:
                    filename, new_records, write_mode = payload

                if not new_records:
                    continue

                filepath = os.path.join(trace_dir, filename)

                if write_mode == "jsonl":
                    with open(filepath, 'a') as f:
                        for record in new_records:
                            json.dump(record, f)
                            f.write('\n')
                        f.flush()
                        os.fsync(f.fileno())
                    continue

                existing_records = []
                try:
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if content:  # Avoid error on empty file
                                existing_records = json.loads(content)
                        if not isinstance(existing_records, list):
                            print(
                                f"Warning: Trace file {filepath} appears to be corrupted. Overwriting.",
                                flush=True,
                            )
                            existing_records = []
                except (json.JSONDecodeError, IOError) as e:
                    print(
                        f"Warning: Could not read or parse existing trace file {filepath}. Overwriting. Error: {e}",
                        flush=True,
                    )
                    existing_records = []

                existing_records.extend(new_records)

                with open(filepath, 'w') as f:
                    json.dump(existing_records, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as error:
            error_queue.put(error)
            print(f"Error in trace saving thread: {error}", flush=True)
        finally:
            work_queue.task_done()


class Tracer:
    """Global tracer to record and print timestamp during training process"""

    def __init__(self) -> None:
        self._records: List[Any] = []
        self._cur: Optional[int] = None
        self._pending_pad_before: Optional[int] = None
        self._pendings: Optional[List[_Pending]] = None
        self._scopes: List[_TracerScope] = []
        self._iteration_open = False
        self._iteration_record_start = 0
        self._closed = False
        self.iter = 0
        self.global_args = None
        self.interval: int = 1000
        self.continuous_trace_iters: int = 1

        self._work_queue: Optional[queue.Queue] = None
        self._save_thread: Optional[threading.Thread] = None
        self._save_error_queue: Optional[queue.Queue] = None
        self._hw_monitor = HardwareMonitor(interval=0.01)

        self._mode0_started: bool = False
        self._mode0_trace_start_ns: Optional[int] = None
        self._mode0_step_start_ns: Optional[int] = None
        self._noop_scope = _NoopTracerScope()
        self._mode0_rank_cache_valid: bool = False

        # Lazy-resolved on first use (after global_args is set):
        # - True  → fallback path: rank 0 collects all payloads via gather_object
        # - False → default path: each rank writes its own trace file directly
        self._gather_to_rank0: Optional[bool] = None

        # CUDA kernel-level tracing via torch.profiler (tier2 only by default).
        # _kernel_capture_enabled is lazy-resolved from --trace-cupti-kernels +
        # trace_granularity. _kernel_profiler holds the active profile() context
        # while a trace window is open. _kernel_profiler_anchor_ns records the
        # wall-clock at profiler enter so each kernel's profiler-relative
        # start_us / end_us can be aligned to the framework rel_ts timeline.
        # _iter_wall_us_starts maps iteration → wall-clock-us at iteration_begin
        # so _extract_kernel_records can attribute each kernel to its true iter
        # (torch.profiler returns kernels for all iters in the window at once).
        self._kernel_capture_enabled: Optional[bool] = None
        self._kernel_profiler: Any = None
        self._kernel_profiler_anchor_ns: int = 0
        self._iter_wall_us_starts: Dict[int, int] = {}

    def configure(self, args: Any) -> None:
        """Bind validated runtime arguments to this fresh tracer instance."""
        if self._closed:
            raise RuntimeError("Cannot configure a closed Tracer")
        interval = int(getattr(args, "trace_interval", 1000))
        continuous = int(getattr(args, "continuous_trace_iterations", 1))
        sample_ms = float(getattr(args, "sentinel_hw_sample_ms", 100.0))
        flush_interval = int(getattr(args, "sentinel_flush_interval", 100))
        cupti_mode = getattr(args, "trace_cupti_kernels", "off")
        if interval <= 0:
            raise ValueError("trace_interval must be greater than zero")
        if continuous <= 0 or continuous > interval:
            raise ValueError("continuous_trace_iterations must be in [1, trace_interval]")
        if sample_ms <= 0:
            raise ValueError("sentinel_hw_sample_ms must be greater than zero")
        if flush_interval <= 0:
            raise ValueError("sentinel_flush_interval must be greater than zero")
        if getattr(args, "hardware_monitor", False) and not getattr(args, "trace", False):
            raise ValueError("hardware_monitor requires trace")
        kernel_capture_requested = (
            getattr(args, "trace_mode", 1) != 0 and cupti_mode != "off"
        )
        if (
            kernel_capture_requested
            and getattr(args, "profile", False)
            and getattr(args, "use_pytorch_profiler", False)
        ):
            raise ValueError("MegaLens kernel capture cannot nest the Megatron PyTorch profiler")

        self.global_args = args
        self.interval = interval
        self.continuous_trace_iters = continuous
        self._hw_monitor.interval = sample_ms / 1000.0

    def _resolve_gather_mode(self) -> bool:
        """Resolve --trace-gather-to-rank0 once global_args is available.

        Returns False (rank-local write) by default. The legacy
        gather-to-rank-0 behavior is enabled only when the user explicitly
        passes --trace-gather-to-rank0.
        """
        if self._gather_to_rank0 is None:
            self._gather_to_rank0 = bool(
                self.global_args and getattr(self.global_args, "trace_gather_to_rank0", False)
            )
        return self._gather_to_rank0

    def _resolve_kernel_capture_mode(self) -> bool:
        """Decide whether to capture CUDA kernel events via torch.profiler.

        Returns True only in framework trace mode (mode 1) when:
          - --trace-cupti-kernels=on, OR
          - --trace-cupti-kernels=auto AND --trace-granularity=full (tier2)

        Mode 0 (sentinel) always returns False; capturing kernels there would
        defeat the <1% overhead goal. Result is cached on first call.
        """
        if self._kernel_capture_enabled is not None:
            return self._kernel_capture_enabled
        if self.global_args is None:
            return False
        if self.is_mode0():
            self._kernel_capture_enabled = False
            return False
        setting = getattr(self.global_args, "trace_cupti_kernels", "auto")
        granularity = getattr(self.global_args, "trace_granularity", "full")
        if setting == "off":
            enabled = False
        elif setting == "on":
            enabled = True
        else:  # auto
            enabled = granularity == "full"
        self._kernel_capture_enabled = enabled
        return enabled

    def _start_kernel_profiler(self) -> None:
        """Enter a torch.profiler scope at trace-window start.

        Called from iteration_begin() when a new trace window opens.
        Errors are caught and logged; capture is best-effort.
        """
        if not self._resolve_kernel_capture_mode():
            return
        if self._kernel_profiler is not None:
            return  # already inside a window
        try:
            from torch.profiler import ProfilerActivity, profile

            self._kernel_profiler = profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
            )
            # Record wall-clock at enter so kernel start_us (profiler-relative)
            # can be mapped to the same wall-clock timeline used by framework
            # events. Anchor goes into each cuda_kernel record.
            self._kernel_profiler.__enter__()
            self._kernel_profiler_anchor_ns = time.time_ns()
        except Exception as e:
            print(f"Warning: failed to start kernel profiler: {e}", flush=True)
            self._kernel_profiler = None
            self._kernel_profiler_anchor_ns = 0

    def _stop_kernel_profiler_and_extract(self) -> None:
        """Exit profiler at trace-window end and merge kernel records.

        Called from iteration_end() right before log() so the kernel
        records flush together with framework records in the same payload.
        """
        if self._kernel_profiler is None:
            return
        try:
            self._kernel_profiler.__exit__(None, None, None)
            self._extract_kernel_records()
        except Exception as e:
            print(f"Warning: failed to stop kernel profiler: {e}", flush=True)
        finally:
            self._kernel_profiler = None

    def _extract_kernel_records(self) -> None:
        """Walk profiler events and append CUDA kernel records to self._records.

        Each kernel becomes a Chrome-trace 'X' (complete) event with:
            record_type=cuda_kernel, name, duration_us, device, iteration,
            dp_rk, pp_rk, tp_rk, g_rk

        Supports two PyTorch profiler APIs:
        - New (kineto, PyTorch 2.x): kernel events appear directly in events()
          with device_type == DeviceType.CUDA; .kernels list is empty.
        - Legacy (autograd profiler): each FunctionEvent has a .kernels list
          of Kernel sub-objects with name/duration/device.

        Rank fields use the cache populated by iteration_end's _cache_ranks().
        Empty profiler / event-list is silently OK.
        """
        if self._kernel_profiler is None:
            return
        try:
            events = self._kernel_profiler.events()
        except Exception:
            events = []
        if not events:
            return

        try:
            from torch.autograd import DeviceType

            cuda_type = DeviceType.CUDA
        except Exception:
            cuda_type = None

        dp_rank = getattr(self, "_cached_dp_rank", 0)
        pp_rank = getattr(self, "_cached_pp_rank", 0)
        tp_rank = getattr(self, "_cached_tp_rank", 0)
        device = getattr(self, "_cached_device", 0)
        global_rank = getattr(self, "_cached_global_rank", 0)

        # Pre-compute sorted iter boundaries for kernel→iter attribution.
        # bisect_right(starts, wall_us) - 1 gives the iter whose [start, next_start)
        # window contains wall_us. Falls back to self.iter if map empty.
        import bisect as _bisect

        iter_items = sorted(self._iter_wall_us_starts.items())
        iter_starts_us = [w for _, w in iter_items]
        iter_nums = [n for n, _ in iter_items]

        def _attribute_iter(wall_us: int) -> int:
            if not iter_starts_us:
                return self.iter
            idx = _bisect.bisect_right(iter_starts_us, wall_us) - 1
            if idx < 0:
                return iter_nums[0]
            return iter_nums[idx]

        kernel_count = 0
        for evt in events:
            # Detect new-API CUDA kernel events first (PyTorch 2.x kineto path)
            is_kernel = False
            try:
                is_kernel = cuda_type is not None and evt.device_type == cuda_type
            except Exception:
                pass

            if is_kernel:
                # New API: this event is the kernel itself
                # Pull start/end from time_range (profiler-relative microseconds);
                # these are needed by round-4 kernel-aware analyzers (P1/P2/etc.)
                # to reconstruct the GPU timeline within a trace window.
                start_us = 0
                end_us = 0
                try:
                    tr = evt.time_range
                    start_us = int(getattr(tr, "start", 0) or 0)
                    end_us = int(getattr(tr, "end", 0) or 0)
                except Exception:
                    pass

                duration_us = 0
                try:
                    d = getattr(evt, "dur", None)
                    if d:
                        duration_us = int(d)
                except Exception:
                    pass
                if duration_us == 0 and end_us > start_us:
                    duration_us = end_us - start_us

                # Wall-clock-aligned timestamps: anchor + profiler-relative.
                # Lets downstream analysis correlate kernel events with
                # framework rel_ts on the same rank.
                anchor_us = self._kernel_profiler_anchor_ns // 1000
                wall_start_us = anchor_us + start_us
                wall_end_us = anchor_us + end_us
                # Attribute kernel to its true iter using wall-clock map
                kernel_iter = _attribute_iter(wall_start_us)
                # Round 5: iteration-anchored timestamps for cross-rank soft
                # alignment. Subtracting iter_begin_wall_us puts kernels from
                # different ranks at the same iter into a comparable timeline
                # (error = NCCL barrier skew, typically <100us). Falls back to
                # anchor_us when iter_begin missing (degenerates to start_us /
                # end_us, same as profiler-relative).
                iter_begin_wall_us = self._iter_wall_us_starts.get(kernel_iter, anchor_us)
                iter_rel_start_us = wall_start_us - iter_begin_wall_us
                iter_rel_end_us = wall_end_us - iter_begin_wall_us

                self._add_record(
                    {
                        "record_type": "cuda_kernel",
                        "name": getattr(evt, "name", "unknown"),
                        "ph": "X",
                        "start_us": start_us,
                        "end_us": end_us,
                        "wall_start_us": wall_start_us,
                        "wall_end_us": wall_end_us,
                        "iter_rel_start_us": iter_rel_start_us,
                        "iter_rel_end_us": iter_rel_end_us,
                        "duration_us": duration_us,
                        "device": getattr(evt, "device_index", device),
                        "iteration": kernel_iter,
                        "dp_rk": dp_rank,
                        "pp_rk": pp_rank,
                        "tp_rk": tp_rank,
                        "g_rk": global_rank,
                    }
                )
                kernel_count += 1
            else:
                # Legacy autograd profiler path: walk .kernels children
                kernels = getattr(evt, "kernels", None) or []
                for k in kernels:
                    self._add_record(
                        {
                            "record_type": "cuda_kernel",
                            "name": getattr(k, "name", "unknown"),
                            "ph": "X",
                            "duration_us": int(getattr(k, "duration", 0)),
                            "device": getattr(k, "device", device),
                            "iteration": self.iter,
                            "dp_rk": dp_rank,
                            "pp_rk": pp_rank,
                            "tp_rk": tp_rank,
                            "g_rk": global_rank,
                        }
                    )
                    kernel_count += 1

        if kernel_count > 0 and global_rank == 0:
            # Lightweight rank-0 trace so users can see capture is active.
            print(
                f"[trace] extracted {kernel_count} cuda kernel events at iter {self.iter}",
                flush=True,
            )

    def _cleanup_trace_dir(self, trace_dir: str) -> None:
        """Remove existing MegaLens trace files in trace_dir.

        Called by rank 0 in legacy gather mode before its writer spins up. Safe
        to call when the directory does not yet exist.
        """
        if os.path.exists(trace_dir):
            try:
                for filename in os.listdir(trace_dir):
                    owned_prefix = filename.startswith(
                        (
                            'benchmark-global-',
                            'benchmark-data-',
                            'mode0-sentinel-global-',
                            'mode0-sentinel-data-',
                        )
                    )
                    if owned_prefix and (filename.endswith('.json') or filename.endswith('.jsonl')):
                        os.remove(os.path.join(trace_dir, filename))
            except OSError as e:
                print(f"Warning: Could not clean up trace directory {trace_dir}: {e}", flush=True)

    def _cleanup_rank_local_trace_file(self, trace_dir: str, filename: str) -> None:
        """Remove only the shard owned by the current rank-local writer."""
        filepath = os.path.join(trace_dir, filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError as e:
            print(f"Warning: Could not clean up trace file {filepath}: {e}", flush=True)

    def _initialize_save_thread(self, *, rank_local_filename: Optional[str] = None):
        """Initialize the background saver thread.

        Default behavior (rank-local write): every rank starts its own saver
        thread and removes only its own stale shard before writing. This path
        performs no distributed collective.

        Fallback behavior (--trace-gather-to-rank0): only rank 0 starts a saver
        thread; non-zero ranks short-circuit. Rank 0 still cleans the directory
        before spinning up its writer.
        """
        if self._save_thread is not None:
            return

        assert self.global_args is not None, "Tracer's global_args has not been set"
        trace_dir = self.global_args.trace_dir
        gather_mode = self._resolve_gather_mode()

        if gather_mode:
            # Legacy: only rank 0 cleans + writes; non-zero ranks have no thread
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
            self._cleanup_trace_dir(trace_dir)
        else:
            assert rank_local_filename is not None, "Rank-local writer requires its shard filename"
            self._cleanup_rank_local_trace_file(trace_dir, rank_local_filename)

        os.makedirs(trace_dir, exist_ok=True)
        self._work_queue = queue.Queue()
        self._save_error_queue = queue.Queue()
        self._save_thread = threading.Thread(
            target=_save_traces_to_disk_thread,
            args=(self._work_queue, trace_dir, self._save_error_queue),
            daemon=True,
        )
        self._save_thread.start()

    def _raise_save_error_if_any(self) -> None:
        if self._save_error_queue is None:
            return
        try:
            error = self._save_error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("MegaLens trace writer failed") from error

    def iteration_begin(self, iteration: int, enable_hw_monitor: bool = False) -> None:
        """Open one training iteration for framework probes."""
        if self._closed:
            raise RuntimeError("Cannot begin an iteration on a closed Tracer")
        if self._iteration_open:
            raise RuntimeError(f"Iteration {self.iter} is already open")

        self._iteration_record_start = len(self._records)
        self._iteration_open = True
        try:
            self._iteration_begin_impl(iteration, enable_hw_monitor)
        except BaseException:
            self.abort_iteration()
            raise

    def close_trace_window(self) -> None:
        """Flush a retained mode-1 window before skipped training work."""
        if self._iteration_open:
            raise RuntimeError("Cannot close a trace window during an open iteration")
        if self.is_mode0() or self._pendings is None:
            return
        self._stop_kernel_profiler_and_extract()
        self.log()
        self._pendings = None

    def _iteration_begin_impl(self, iteration: int, enable_hw_monitor: bool = False) -> None:
        """Start collecting one iteration.

        Time alignment strategy:
            ``_calibrate()`` records ``self._cur = time.time_ns()`` (wall-clock).
            The first CUDA event ("iteration B") is recorded at approximately the
            same instant.  The HardwareMonitor receives ``self._cur`` as its
            reference origin so that ``hw_rel_ts = time.time_ns() - _cur`` is
            on the same timeline as CUDA ``rel_ts`` produced by
            ``_process_pending_scope``.  The residual skew between CPU and GPU
            clocks is negligible relative to the 10 ms HW sampling interval.
        """
        self.iter = iteration

        if self.is_mode0():
            self._mode0_iteration_begin(iteration, enable_hw_monitor)
            return

        if self.is_tracing_active():
            if self._pendings is None:  # Start of a tracing window
                self._pendings = []
                # Reset iter→wall map at window start
                self._iter_wall_us_starts = {}
                # Open the kernel profiler for this window (tier2 + auto/on)
                self._start_kernel_profiler()
            self._pending_pad_before = self._calibrate()
            # Record this iter's wall-clock start (before any in-iter work)
            # so kernel records can be attributed to their true iter.
            if self._resolve_kernel_capture_mode():
                self._iter_wall_us_starts[iteration] = time.time_ns() // 1000
            # Mark the beginning of the iteration
            self._add_cuda_event("iteration", "B", {"iteration": self.iter})
            if enable_hw_monitor:
                self._hw_monitor.start(self._cur)
        else:
            self._pendings = None

    def _mode0_iteration_begin(self, iteration: int, enable_hw_monitor: bool = False) -> None:
        if not self._mode0_started:
            self._mode0_started = True
            self._mode0_trace_start_ns = time.time_ns()
            self._mode0_step_start_ns = self._mode0_trace_start_ns
            self._records = []
            if enable_hw_monitor:
                self._hw_monitor.start(self._mode0_trace_start_ns)
            return

        self._mode0_step_start_ns = time.time_ns()

    def _calibrate(self) -> int:
        """Reset the clock and get delta."""
        cur = time.time_ns()
        if self._cur is None:
            delta = 0
        else:
            delta = cur - self._cur
        self._cur = cur
        return delta

    def _add_record(self, attrs: Dict[str, Any]) -> None:
        self._records.append(attrs)

    def _last_record(self) -> Dict[str, Any]:
        return self._records[-1]

    def _add_pending(self, pending: _Pending) -> None:
        if self._pendings is not None:
            self._pendings.append(pending)

    def _add_cuda_event(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()  # type: ignore
        pending = _Pending(name, phase, event, attrs)  # type: ignore
        self._add_pending(pending)

    def is_tracing(self) -> bool:
        return self._pendings is not None

    def _cache_ranks(self) -> None:
        """Cache parallelism ranks once per iteration (avoid repeated queries)."""
        try:
            self._cached_dp_rank = parallel_state.get_data_parallel_rank()
        except Exception:
            self._cached_dp_rank = 0
        try:
            self._cached_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except Exception:
            self._cached_pp_rank = 0
        try:
            self._cached_tp_rank = parallel_state.get_tensor_model_parallel_rank()
        except Exception:
            self._cached_tp_rank = 0
        try:
            self._cached_device = torch.cuda.current_device()
        except Exception:
            self._cached_device = 0
        try:
            self._cached_global_rank = torch.distributed.get_rank()
        except Exception:
            self._cached_global_rank = 0
        self._mode0_rank_cache_valid = True

    def _process_pending_scope(self, ref_ts: int, ref_event: torch.cuda.Event, i: int) -> int:
        """Process the pending scopes.
        ref must be a "B".
        Args:
            ref_ts: reference timestamp.
            ref_event: reference event.
            i: index of the pending scope to be processed.
        Returns:
            The next index to process.
        """
        assert self._pendings is not None
        # Use cached ranks (set once per iteration in iteration_end)
        dp_rank = self._cached_dp_rank
        pp_rank = self._cached_pp_rank
        tp_rank = self._cached_tp_rank
        device = self._cached_device
        global_rank = self._cached_global_rank

        while i < len(self._pendings):
            pending = self._pendings[i]
            elapsed = int(ref_event.elapsed_time(pending.event) * 1e6)
            rel_ts = ref_ts + elapsed
            chrome_event = {
                **pending.attrs,
                "name": pending.name,
                "ph": pending.phase,
                "rel_ts": rel_ts,
                "dp_rk": dp_rank,
                "pp_rk": pp_rank,
                "tp_rk": tp_rank,
                "dev": device,
                "g_rk": global_rank,
            }

            self._add_record(chrome_event)
            i += 1
            if pending.phase == "B":
                # Nested scope
                i = self._process_pending_scope(rel_ts, pending.event, i)
            elif pending.phase == "E":
                # End of this scope
                if "data" in pending.attrs:
                    last = self._last_record()
                    if pending.attrs["data"] is None:
                        last["bandwidth"] = None
                    else:
                        # 1 Gb = 2 ** 30 b = 2 ** 27 B
                        gb = pending.attrs["data"] / (2**27)
                        secs = elapsed / 1e9
                        bandwidth = gb / secs  # Gbps
                        last["bandwidth"] = bandwidth
                return i
        assert i == len(self._pendings), "Mismatched scopes"
        return i

    def iteration_end(self, enable_hw_monitor: bool = False) -> None:
        """Close the current training iteration or discard its partial state."""
        if not self._iteration_open:
            raise RuntimeError("No MegaLens iteration is open")
        try:
            self._iteration_end_impl(enable_hw_monitor)
        except BaseException:
            self.abort_iteration()
            raise
        else:
            self._iteration_open = False
            self._iteration_record_start = len(self._records)

    def _iteration_end_impl(self, enable_hw_monitor: bool = False) -> None:
        if self.is_mode0():
            self._mode0_iteration_end(enable_hw_monitor)
            return

        if self.is_tracing_active() and self._pendings is not None:
            # Mark the end of the iteration
            self._add_cuda_event("iteration", "E", {"iteration": self.iter})
            # Wait for the last recorded CUDA event only (instead of a full
            # device synchronize which blocks the entire GPU pipeline).
            if self._pendings:
                self._pendings[-1].event.synchronize()
            else:
                torch.cuda.synchronize()
            # Cache rank info once (avoids per-event queries in _process_pending_scope)
            self._cache_ranks()
            # Get wall clock duration for this iteration
            wall_duration = self._calibrate()

            self._add_record(
                {
                    "name": "iteration",
                    "ph": "B",
                    "pad_before": self._pending_pad_before,
                    "iteration": self.iter,
                }
            )

            if enable_hw_monitor:
                hw_records = self._hw_monitor.collect_and_clear()
                self._hw_monitor.stop()
                for rec in hw_records:
                    rec.setdefault("iteration", self.iter)
                self._records.extend(hw_records)

            if not self._pendings:
                return

            iteration_begin_event = self._pendings[0].event
            # We cannot know the absolute timestamp of the first event, so we set it to 0.
            self._process_pending_scope(0, iteration_begin_event, 1)
            end = self._last_record()
            end["duration_wall"] = wall_duration
            end["duration_cuda"] = end["rel_ts"]
            self._pendings = []

        if self.should_log_this_iter():
            # Close the kernel profiler (if open) and merge its records into
            # self._records before flushing to disk so kernel events ride
            # in the same payload as framework events.
            self._stop_kernel_profiler_and_extract()
            self.log()
            self._pendings = None

    def abort_iteration(self) -> None:
        """Discard incomplete iteration state without distributed coordination."""
        if not self._iteration_open:
            return
        try:
            self._hw_monitor.stop()
            self._stop_kernel_profiler_and_extract()
        finally:
            completed_kernel_records = [
                record
                for record in self._records[self._iteration_record_start :]
                if record.get("record_type") == "cuda_kernel"
                and record.get("iteration") != self.iter
            ]
            del self._records[self._iteration_record_start :]
            self._records.extend(completed_kernel_records)
            self._pendings = None
            self._scopes = []
            self._pending_pad_before = None
            self._iteration_open = False
            self._iteration_record_start = len(self._records)

    def _mode0_iteration_end(self, enable_hw_monitor: bool = False) -> None:
        if self._mode0_step_start_ns is None:
            self._mode0_step_start_ns = time.time_ns()

        now_ns = time.time_ns()
        trace_start_ns = self._mode0_trace_start_ns or self._mode0_step_start_ns
        step_start_rel_ts = self._mode0_step_start_ns - trace_start_ns
        step_end_rel_ts = now_ns - trace_start_ns
        step_duration_ns = now_ns - self._mode0_step_start_ns

        if not self._mode0_rank_cache_valid:
            self._cache_ranks()
        dp_rank = self._cached_dp_rank
        pp_rank = self._cached_pp_rank
        tp_rank = self._cached_tp_rank
        device = self._cached_device
        global_rank = self._cached_global_rank

        hw_samples: List[Dict[str, Any]] = []
        if enable_hw_monitor:
            hw_records = self._hw_monitor.collect_and_clear()
            hw_samples = self._mode0_build_hw_samples(
                hw_records=hw_records,
                step_start_rel_ts=step_start_rel_ts,
                step_end_rel_ts=step_end_rel_ts,
            )

        self._add_record(
            {
                "record_type": "mode0_step",
                "iteration": self.iter,
                "mode": 0,
                "rank": {
                    "g_rk": global_rank,
                    "dp_rk": dp_rank,
                    "pp_rk": pp_rank,
                    "tp_rk": tp_rank,
                    "dev": device,
                },
                "step_time": {
                    "start_rel_ts": step_start_rel_ts,
                    "end_rel_ts": step_end_rel_ts,
                    "duration_wall": step_duration_ns,
                    "duration_ms": step_duration_ns / 1e6,
                },
                "hardware_metrics": hw_samples,
            }
        )

        if self.should_log_this_iter():
            self.log()
            self._records = []

    def _mode0_build_hw_samples(
        self, hw_records: List[Dict[str, Any]], step_start_rel_ts: int, step_end_rel_ts: int
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        pending_sample: Dict[int, Dict[str, Any]] = {}

        for rec in hw_records:
            rel_ts = rec.get("rel_ts")
            if not isinstance(rel_ts, int):
                continue
            if rel_ts < step_start_rel_ts or rel_ts > step_end_rel_ts:
                continue

            sample = pending_sample.setdefault(rel_ts, {"rel_ts": rel_ts, "cpu": None, "gpu": None})
            name = rec.get("name")
            if name == "CPU_Metrics":
                sample["cpu"] = dict(rec.get("args", {}))
            elif name == "GPU_Metrics":
                sample["gpu"] = dict(rec.get("args", {}))

        for rel_ts in sorted(pending_sample.keys()):
            sample = pending_sample[rel_ts]
            if sample["cpu"] is None and sample["gpu"] is None:
                continue
            samples.append(sample)

        return samples

    def _tick(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        if self.is_tracing():
            self._add_cuda_event(name, phase, attrs)

    def tick(self, name: str, **attrs: Any) -> None:
        """Record an event."""
        self._tick(name, "i", attrs)

    def scope(
        self,
        name: Optional[str],
        *args,
        ctx: Optional[Mapping[str, Any]] = None,
        slots: Optional[Sequence[str]] = None,
        attrs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> _TracerScope:
        """
        Create a scope of code, selectively tracing based on granularity.

        Args:
            name: Name of the scope. If None, the scope is not timed.
            ctx: Parameters to be passed to the scope.
            kwargs: Items to be recorded. If an item is None, it should be filled by some inner scope.
            slots: Parameters that are passed to the scope and must be filled. (They go to both ctx and kwargs.)
        """
        assert len(args) == 0, "Positional arguments are not supported"
        ctx_values = dict(ctx or {})
        slot_names = list(slots or ())
        out_attrs = dict(attrs or {})
        duplicate_attrs = out_attrs.keys() & kwargs.keys()
        if duplicate_attrs:
            duplicates = ", ".join(sorted(duplicate_attrs))
            raise TypeError(f"Duplicate trace attributes: {duplicates}")
        out_attrs.update(kwargs)
        if self.is_mode0():
            return self._noop_scope
        for slot in slot_names:
            ctx_values[slot] = True
            out_attrs[slot] = None

        trace_name = name
        if self.global_args and self.global_args.trace and name is not None:
            granularity = self.global_args.trace_granularity
            if granularity == 'base' and name not in BASE_TRACING_EVENTS:
                trace_name = None  # Mute non-core events in tier1/base.
        if trace_name is None and not slot_names:
            return self._noop_scope
        return _TracerScope(self, name=trace_name, in_attrs=ctx_values, out_attrs=out_attrs)

    # def scoped(self, func):
    #     """Decorator to time a function."""
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         with self.scope(func.__name__):
    #             return func(*args, **kwargs)
    #     return wrapper

    def scoped(
        self,
        name: Optional[str] = None,
        ctx: Optional[Dict[str, Any]] = None,
        slots: Optional[List[str]] = None,
        **kwargs0: Any,
    ):
        if ctx is None:
            ctx = {}
        if slots is None:
            slots = []

        def decorator(func):
            from megatron.training import get_args

            args = get_args()
            # if we are not tracing, just return the function
            if args is None or not args.trace:
                return func
            if getattr(args, "trace_mode", 1) == 0:
                return func

            @wraps(func)
            def wrapper(*args, **kwargs):
                if name is None:
                    with self.scope(func.__name__, ctx=ctx, slots=slots, **kwargs0):
                        return func(*args, **kwargs)
                else:
                    with self.scope(name, ctx=ctx, slots=slots, **kwargs0):
                        return func(*args, **kwargs)

            return wrapper

        return decorator

    def _push_scope(self, scope) -> None:
        self._scopes.append(scope)

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def get(self, q: str) -> Optional[Any]:
        """Query parameter from scopes."""
        for scope in reversed(self._scopes):
            v = scope.get(q)
            if v is not None:
                return v
        return None

    def set(self, q: str, v: Any) -> None:
        """Set parameter to the nearest requiring scope."""
        if self.is_mode0():
            return
        for scope in reversed(self._scopes):
            if scope.set(q, v):
                return
        # for scope in reversed(self._scopes):
        #     print(f"scope: '{scope.name}', q: '{q}', v: '{v}', in_attrs: {scope.in_attrs}, out_attrs: {scope.out_attrs}")
        assert False, f"Cannot find a requiring scope for '{q}'"

    def set_group(self, group: torch.distributed.ProcessGroup | List[int]) -> None:
        if self.is_mode0():
            return
        # get ranks in the group
        if isinstance(group, torch.distributed.ProcessGroup):
            ranks = torch.distributed.get_process_group_ranks(group)
        else:
            ranks = group
        cur_rk = torch.distributed.get_rank()
        # print(f"cur_rk: {cur_rk}, ranks: {ranks}")
        assert cur_rk is not None and cur_rk in ranks
        ranks.remove(cur_rk)
        self.set("group", ranks)

    def is_mode0(self) -> bool:
        args = self.global_args
        return bool(args and getattr(args, "trace", False) and getattr(args, "trace_mode", 1) == 0)

    def log(self):
        """
        Persist trace records.

        Default (rank-local write): each rank writes its own file directly via
        a per-rank saver thread. No collective communication.

        Fallback (--trace-gather-to-rank0): all ranks gather their payloads to
        rank 0, which serializes everything onto a single saver-thread queue.
        """
        if not self.is_tracing_active():
            return
        self._raise_save_error_if_any()

        # Each rank constructs its own filename and payload (file naming is
        # rank-aware in both modes, so the on-disk layout is identical).
        if not self._mode0_rank_cache_valid:
            self._cache_ranks()
        dp_rank = self._cached_dp_rank
        pp_rank = self._cached_pp_rank
        tp_rank = self._cached_tp_rank
        global_rank = self._cached_global_rank
        if self.is_mode0():
            filename = _trace_filename(
                global_rank=global_rank,
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                mode0=True,
            )
            payload = (filename, self._records, "jsonl")
        else:
            filename = _trace_filename(
                global_rank=global_rank,
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                mode0=False,
            )
            payload = (filename, self._records)

        gather_mode = self._resolve_gather_mode()

        if gather_mode:
            # ===== Fallback: gather to rank 0 (legacy behavior) =====
            if global_rank == 0 and self._save_thread is None:
                self._initialize_save_thread()

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                gathered_payloads = [None] * world_size
                torch.distributed.gather_object(
                    payload, gathered_payloads if global_rank == 0 else None, dst=0
                )
            else:
                gathered_payloads = [payload]

            if global_rank == 0:
                assert self._work_queue is not None, "Work queue is not initialized on rank 0"
                payloads_to_save = [p for p in gathered_payloads if p and p[1]]
                if payloads_to_save:
                    self._work_queue.put(payloads_to_save)
        else:
            # ===== Default: rank-local write (no gather) =====
            # Each rank lazily initializes its own saver thread and cleans only
            # its own stale shard on first flush.
            if self._save_thread is None:
                self._initialize_save_thread(rank_local_filename=filename)
            if payload[1]:  # skip empty records
                assert self._work_queue is not None, "Work queue is not initialized"
                self._work_queue.put([payload])

        # Clear records after handing them off
        self._records = []

    def shutdown(self, *, graceful: bool = True) -> None:
        """Idempotently release tracer resources."""
        if self._closed:
            return
        try:
            self._shutdown_impl(graceful=graceful)
        finally:
            self._hw_monitor.shutdown()
            self.global_args = None
            self._closed = True

    def _shutdown_impl(self, *, graceful: bool) -> None:
        """
        Flush remaining records and shut down per-rank saver thread(s).

        In rank-local mode every rank waits its own saver thread.
        In fallback mode only rank 0 has a thread to wait.
        """
        if self._iteration_open:
            self.abort_iteration()
        self._hw_monitor.stop()
        # Close any still-open kernel profiler and merge its records before
        # the final flush. Safe no-op if not enabled / not currently in window.
        self._stop_kernel_profiler_and_extract()
        if graceful and self.global_args and self.global_args.trace and self._records:
            self.log()
        elif not graceful:
            self._records = []
        gather_mode = self._resolve_gather_mode()

        # Decide whether the current rank owns a saver thread to wait on.
        # - Fallback mode: only rank 0 owns one.
        # - Rank-local mode: every rank owns its own.
        # - Single-process: always wait.
        if gather_mode:
            should_wait = (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            )
        else:
            should_wait = True

        if should_wait and self._save_thread is not None:
            save_thread = self._save_thread
            if save_thread.is_alive() and self._work_queue is not None:
                # The sentinel is ordered after all pending payloads. A bounded
                # thread join therefore waits for durability without an
                # unbounded Queue.join() when the writer stalls.
                self._work_queue.put(None)
            save_thread.join(timeout=10)
            if save_thread.is_alive():
                raise TimeoutError("MegaLens trace writer did not terminate within 10 seconds")
            try:
                self._raise_save_error_if_any()
            finally:
                self._save_thread = None
                self._work_queue = None
                self._save_error_queue = None

    def should_log_this_iter(self) -> bool:
        """Checks if we should log at the end of this iteration."""
        args = self.global_args
        if not args or not args.trace or not self.is_tracing_active():
            return False
        if self.is_mode0():
            flush_interval = max(1, int(getattr(args, "sentinel_flush_interval", 100)))
            return self.iter % flush_interval == 0

        # self.iter is 1-based.
        idx = (self.iter - 1) % self.interval
        return idx == self.continuous_trace_iters - 1

    def is_tracing_active(self) -> bool:
        """Checks if we are in a tracing interval."""
        args = self.global_args
        if args is None:
            return False
        if not args.trace:
            return False
        if self.is_mode0():
            return True
        if self.interval is None or self.continuous_trace_iters is None:
            return False

        # Training step IDs are 1-based.
        idx = (self.iter - 1) % self.interval
        return 0 <= idx < self.continuous_trace_iters

    def is_event_enabled(self, name: str) -> bool:
        """Return whether a Core probe should open a concrete trace scope."""
        args = self.global_args
        if (
            args is None
            or not getattr(args, "trace", False)
            or self._closed
            or not self._iteration_open
            or self.is_mode0()
            or not self.is_tracing_active()
            or not self.is_tracing()
        ):
            return False
        if getattr(args, "trace_granularity", "full") == "base":
            return name in BASE_TRACING_EVENTS
        return True


def get_tensor_bytes(obj):
    """calculate the number of bytes of a tensor or a list/tuple of tensors"""
    if obj is None:
        return 0
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(get_tensor_bytes(x) for x in obj)
    return 0
