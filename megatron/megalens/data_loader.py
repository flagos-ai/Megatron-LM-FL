"""Trace Data Engine for MegaLens.

Loads aggregated Chrome Trace JSON files and provides efficient windowed
queries over CUDA span events (``ph: "X"``) and hardware counter time-series
(``ph: "C"``).

Usage::

    loader = TraceDataLoader.from_file("benchmark.json")
    avg_temp = loader.get_hardware_metrics_in_window(
        start_ts=100_000, end_ts=200_000, rank=0, metric_name="Temp_C",
    )
"""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ============================================================================
# Data classes
# ============================================================================


@dataclass
class SpanEvent:
    """A complete CUDA event (``ph: "X"``)."""

    name: str
    ts: int  # start timestamp in microseconds
    dur: int  # duration in microseconds
    rank: int  # global rank (pid)
    args: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_ts(self) -> int:
        return self.ts + self.dur

    # Topology helpers
    @property
    def dp_rank(self) -> int:
        return self.args.get("dp_rk", -1)

    @property
    def pp_rank(self) -> int:
        return self.args.get("pp_rk", -1)

    @property
    def tp_rank(self) -> int:
        return self.args.get("tp_rk", -1)

    @property
    def iteration(self) -> int:
        return self.args.get("iteration", -1)


@dataclass
class KernelEvent:
    """A CUDA kernel event captured by torch.profiler in tier 2 cupti mode.

    Records appear in the trace JSON as objects with
    ``record_type == "cuda_kernel"`` (see Tracer._extract_kernel_records).

    Three time-field tiers, each with different cross-rank semantics:

    1. ``start_us`` / ``end_us``  --  profiler-relative microseconds
       (anchor = ``torch.profiler.__enter__()`` moment, per-window per-rank).
       Comparable: same rank + same trace window only.

    2. ``wall_start_us`` / ``wall_end_us``  --  rank-local wall-clock-aligned
       microseconds (``anchor_ns // 1000 + start_us``). Comparable: same rank
       only (each rank has its own ``time.time_ns()``; cross-rank skew is
       unbounded without explicit reconciliation).

    3. ``iter_rel_start_us`` / ``iter_rel_end_us``  --  iteration-anchored
       microseconds (``wall_*_us - iter_begin_wall_us[iteration]``). Comparable:
       cross-rank within barrier-skew tolerance. Megatron's ``iteration_begin``
       carries an implicit NCCL collective barrier (grad clear / data load),
       so per-rank ``iter_begin_wall_us[same_iter]`` differ by at most ~10us
       (single node) to ~100us (multi-node IB). Use this field for cross-rank
       kernel timeline analysis (P1 link contention, PP P2P stall,
       cross-rank kernel dispersion).

    The ``iteration`` and ``duration_us`` fields are always cross-rank-safe:
    same iter on different ranks is the same training step; duration is a
    GPU-side number with no clock dependence.
    """

    name: str
    start_us: int
    end_us: int
    duration_us: int
    rank: int  # global rank (g_rk)
    device: int
    iteration: int
    dp_rank: int
    pp_rank: int
    tp_rank: int
    wall_start_us: int = 0
    wall_end_us: int = 0
    iter_rel_start_us: int = 0
    iter_rel_end_us: int = 0

    @property
    def name_short(self) -> str:
        """Strip C++ template/anonymous-ns mangle for display."""
        if "<" in self.name:
            return self.name.split("<", 1)[0]
        return self.name

    def overlaps(self, other: "KernelEvent") -> bool:
        """True if [start_us, end_us] intersects other's interval (same window)."""
        return not (self.end_us <= other.start_us or other.end_us <= self.start_us)

    def overlap_us_with(self, other: "KernelEvent") -> int:
        """Length of overlap in microseconds (0 if disjoint)."""
        s = max(self.start_us, other.start_us)
        e = min(self.end_us, other.end_us)
        return max(0, e - s)


@dataclass
class CounterSample:
    """A single hardware counter measurement (``ph: "C"``)."""

    name: str
    ts: int  # timestamp in microseconds
    rank: int  # global rank (pid)
    metrics: Dict[str, float] = field(default_factory=dict)
    iteration: int = -1


@dataclass
class MetricWindowResult:
    """Aggregated metric statistics over a time window."""

    metric_name: str
    rank: int
    window_start_ts: int
    window_end_ts: int
    num_samples: int
    mean: float
    peak: float
    minimum: float
    values: np.ndarray  # raw sample values within the window


# ============================================================================
# Main loader
# ============================================================================


class TraceDataLoader:
    """Load and index Chrome Trace JSON data for efficient querying.

    Supports two input modes:
        1. A single aggregated JSON file (all ranks mixed).
        2. A directory of per-rank ``benchmark-*.json`` files.
    """

    def __init__(self) -> None:
        self.span_events: List[SpanEvent] = []
        self.counter_samples: List[CounterSample] = []
        self.kernel_events: List[KernelEvent] = []

        # Indices built by ``_build_indices``
        self._spans_by_rank: Dict[int, List[SpanEvent]] = {}
        self._counters_by_rank: Dict[int, List[CounterSample]] = {}
        self._counter_ts_by_rank: Dict[int, np.ndarray] = {}
        self._kernels_by_rank: Dict[int, List[KernelEvent]] = {}

        self._topology: Dict[int, Dict[str, int]] = {}  # rank -> {dp, pp, tp}
        self._ranks: List[int] = []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "TraceDataLoader":
        """Load from a single aggregated Chrome Trace JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)
        loader = cls()
        loader._ingest(raw)
        loader._build_indices()
        return loader

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "TraceDataLoader":
        """Load and aggregate raw per-rank ``benchmark-*.json`` files."""
        from megatron.megalens.trace_aggregate import (
            aggregate_benchmark_data,
            benchmark_to_chrome_trace,
            collect_benchmark_files,
            read_benchmark_file,
        )

        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Benchmark directory does not exist: {directory}")

        files = sorted(
            collect_benchmark_files(directory),
            key=lambda item: (
                item[0].global_rank is None,
                item[0].global_rank if item[0].global_rank is not None else -1,
                item[0].data,
                item[0].pipeline,
                item[0].tensor,
            ),
        )
        if not files:
            raise FileNotFoundError(f"No benchmark-*.json files found in directory: {directory}")

        decoded_files = []
        raw_flags = []
        for rank, content in files:
            rows = json.loads(content)
            if not isinstance(rows, list):
                raise ValueError(f"Benchmark shard for rank {rank} must contain a JSON list")
            decoded_files.append(rows)
            raw_flags.append(
                any(
                    row.get("name") == "iteration" and row.get("ph") == "B" and "pad_before" in row
                    for row in rows
                )
            )

        if any(raw_flags) and not all(raw_flags):
            raise ValueError("Benchmark directory mixes raw and post-transform trace shards")

        if all(raw_flags):
            contents = [read_benchmark_file(rank, content) for rank, content in files]
            iterations, _, _, _ = aggregate_benchmark_data(contents)
            all_events = benchmark_to_chrome_trace(iterations)
        else:
            invalid = [
                row
                for rows in decoded_files
                for row in rows
                if row.get("record_type") != "cuda_kernel" and row.get("ph") not in {"X", "C", "M"}
            ]
            if invalid:
                raise ValueError("Post-transform benchmark shards may contain only X/C/M events")
            all_events = [row for rows in decoded_files for row in rows]
        loader = cls()
        loader._ingest(all_events)
        loader._build_indices()
        return loader

    @classmethod
    def from_traces(cls, traces: List[Dict[str, Any]]) -> "TraceDataLoader":
        """Load from an in-memory list of trace dicts (post-``transform``)."""
        loader = cls()
        loader._ingest(traces)
        loader._build_indices()
        return loader

    # ------------------------------------------------------------------
    # Core query API
    # ------------------------------------------------------------------

    def get_hardware_metrics_in_window(
        self, start_ts: int, end_ts: int, rank: int, metric_name: str
    ) -> Optional[MetricWindowResult]:
        """Return aggregated hardware metric within ``[start_ts, end_ts]``.

        Uses binary search on the pre-sorted timestamp index for O(log N)
        lookups.  Returns ``None`` if no samples exist in the window.

        Args:
            start_ts: Window start (microseconds, inclusive).
            end_ts:   Window end (microseconds, inclusive).
            rank:     Global rank (pid).
            metric_name: Key inside the counter ``args``, e.g. ``"Temp_C"``.
        """
        samples = self._counters_by_rank.get(rank)
        ts_arr = self._counter_ts_by_rank.get(rank)
        if samples is None or ts_arr is None or len(samples) == 0:
            return None

        lo = bisect.bisect_left(ts_arr, start_ts)
        hi = bisect.bisect_right(ts_arr, end_ts)
        if lo >= hi:
            return None

        values: List[float] = []
        for i in range(lo, hi):
            v = samples[i].metrics.get(metric_name)
            if v is not None:
                values.append(float(v))

        if not values:
            return None

        arr = np.array(values, dtype=np.float64)
        return MetricWindowResult(
            metric_name=metric_name,
            rank=rank,
            window_start_ts=start_ts,
            window_end_ts=end_ts,
            num_samples=len(arr),
            mean=float(np.mean(arr)),
            peak=float(np.max(arr)),
            minimum=float(np.min(arr)),
            values=arr,
        )

    def get_events_by_name(
        self, name: str, rank: Optional[int] = None, iteration: Optional[int] = None
    ) -> List[SpanEvent]:
        """Filter span events by name and optionally by rank / iteration."""
        if rank is not None:
            source = self._spans_by_rank.get(rank, [])
        else:
            source = self.span_events
        results: List[SpanEvent] = []
        for ev in source:
            if ev.name != name:
                continue
            if iteration is not None and ev.iteration != iteration:
                continue
            results.append(ev)
        return results

    def get_events_matching(
        self, name_contains: str, rank: Optional[int] = None, iteration: Optional[int] = None
    ) -> List[SpanEvent]:
        """Filter span events whose name contains *name_contains*."""
        if rank is not None:
            source = self._spans_by_rank.get(rank, [])
        else:
            source = self.span_events
        results: List[SpanEvent] = []
        for ev in source:
            if name_contains not in ev.name:
                continue
            if iteration is not None and ev.iteration != iteration:
                continue
            results.append(ev)
        return results

    def get_iteration_events(self, iteration: Optional[int] = None) -> List[SpanEvent]:
        """Return all ``iteration`` boundary events."""
        return self.get_events_by_name("iteration", iteration=iteration)

    def get_ranks(self) -> List[int]:
        return list(self._ranks)

    def get_dp_ranks(self) -> List[int]:
        """Return global-rank IDs that share a DP group (unique PP=0, TP=0)."""
        return [
            r for r, t in self._topology.items() if t.get("pp", -1) == 0 and t.get("tp", -1) == 0
        ]

    def get_tp_group(self, dp_rank: int, pp_rank: int) -> List[int]:
        """Global ranks that belong to the given TP group."""
        return [
            r
            for r, t in self._topology.items()
            if t.get("dp", -1) == dp_rank and t.get("pp", -1) == pp_rank
        ]

    @property
    def topology(self) -> Dict[int, Dict[str, int]]:
        return self._topology

    # ------------------------------------------------------------------
    # Kernel-aware query API (round 4: kernel-aware analyzer)
    # ------------------------------------------------------------------

    def has_kernel_events(self, rank: Optional[int] = None) -> bool:
        """True if at least one cuda_kernel record was loaded for `rank` (or any rank)."""
        if rank is None:
            return bool(self.kernel_events)
        return bool(self._kernels_by_rank.get(rank))

    def get_kernels(
        self,
        rank: Optional[int] = None,
        iteration: Optional[int] = None,
        name_contains: Optional[str] = None,
    ) -> List[KernelEvent]:
        """Filter kernel events by rank / iteration / name substring.

        Linear scan; the kernel index is small enough (~few×10k per window)
        that this is fast for analysis use cases.
        """
        if rank is not None:
            source = self._kernels_by_rank.get(rank, [])
        else:
            source = self.kernel_events
        out: List[KernelEvent] = []
        for k in source:
            if iteration is not None and k.iteration != iteration:
                continue
            if name_contains is not None and name_contains not in k.name:
                continue
            out.append(k)
        return out

    def get_nccl_kernels(
        self, rank: Optional[int] = None, iteration: Optional[int] = None
    ) -> List[KernelEvent]:
        """Return NCCL communication kernels.

        NCCL kernel names typically start with ``ncclDevKernel_`` or
        ``ncclKernel_`` (depends on backend version). This helper centralizes
        the substring filter so analyzers don't have to rediscover the
        naming convention.
        """
        if rank is not None:
            source = self._kernels_by_rank.get(rank, [])
        else:
            source = self.kernel_events
        out: List[KernelEvent] = []
        for k in source:
            n = k.name
            if not (
                n.startswith("ncclDevKernel")
                or n.startswith("ncclKernel")
                or "nccl" in n.lower()[:32]
            ):
                continue
            if iteration is not None and k.iteration != iteration:
                continue
            out.append(k)
        return out

    def get_compute_kernels(
        self, rank: Optional[int] = None, iteration: Optional[int] = None
    ) -> List[KernelEvent]:
        """Return non-NCCL CUDA kernels (i.e. compute kernels).

        Coarse approximation: anything not starting with ``ncclDevKernel`` /
        ``ncclKernel``. Includes cuBLAS, Triton, aten elementwise, memcpy, etc.
        """
        if rank is not None:
            source = self._kernels_by_rank.get(rank, [])
        else:
            source = self.kernel_events
        out: List[KernelEvent] = []
        for k in source:
            n = k.name
            is_nccl = (
                n.startswith("ncclDevKernel")
                or n.startswith("ncclKernel")
                or "nccl" in n.lower()[:32]
            )
            if is_nccl:
                continue
            if iteration is not None and k.iteration != iteration:
                continue
            out.append(k)
        return out

    @staticmethod
    def _assert_same_rank(events: Sequence[KernelEvent], label: str = "events") -> None:
        """Round 5 defensive check: all events must come from the same rank.

        ``start_us`` / ``end_us`` and ``wall_*_us`` are NOT cross-rank
        comparable (each rank has its own profiler anchor and wall clock).
        Helpers that read these fields must scope inputs to a single rank;
        violating this would silently produce nonsense overlap / merge results.

        For cross-rank analysis use the iter-anchored helpers
        (``cross_rank_kernel_overlap_at_iter`` etc.) which read
        ``iter_rel_*_us`` instead.
        """
        if not events:
            return
        first = next(iter(events))
        rank0 = first.rank
        for k in events:
            if k.rank != rank0:
                raise AssertionError(
                    f"{label}: all kernel events must be from the same rank; "
                    f"got rank {rank0} and rank {k.rank}. "
                    f"For cross-rank analysis use cross_rank_*_at_iter helpers."
                )

    @staticmethod
    def merge_intervals(events: Sequence[KernelEvent]) -> List[Tuple[int, int]]:
        """Merge overlapping (start_us, end_us) intervals from a kernel list.

        Returns disjoint sorted intervals. Useful for computing GPU-active
        time without double-counting concurrent kernels on the same stream.

        WARNING: Single-rank only. ``start_us`` / ``end_us`` are
        profiler-relative (per-rank anchor); mixing ranks produces meaningless
        merged intervals. Enforced via :meth:`_assert_same_rank`.
        """
        if not events:
            return []
        TraceDataLoader._assert_same_rank(events, "merge_intervals input")
        intervals = sorted([(e.start_us, e.end_us) for e in events])
        merged: List[Tuple[int, int]] = [intervals[0]]
        for s, e in intervals[1:]:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def kernel_overlap_total_us(
        group_a: Sequence[KernelEvent], group_b: Sequence[KernelEvent]
    ) -> int:
        """Total GPU-timeline overlap length between two kernel groups (single rank).

        Both inputs are first interval-merged, then the two merged intervals
        are intersected. Use for P2 R_actual (comm vs compute overlap) and
        cross-dim CONTENTION detection.

        WARNING: Single-rank only. group_a and group_b must share the same
        rank; mixing ranks invalidates the overlap calculation. Enforced via
        :meth:`_assert_same_rank`. For cross-rank overlap use
        :meth:`cross_rank_kernel_overlap_at_iter`.
        """
        if not group_a or not group_b:
            return 0
        TraceDataLoader._assert_same_rank(group_a, "kernel_overlap_total_us group_a")
        TraceDataLoader._assert_same_rank(group_b, "kernel_overlap_total_us group_b")
        # Cross-group rank check: a and b must also share a rank
        if next(iter(group_a)).rank != next(iter(group_b)).rank:
            raise AssertionError(
                f"kernel_overlap_total_us: group_a and group_b must share the "
                f"same rank; got {next(iter(group_a)).rank} vs "
                f"{next(iter(group_b)).rank}. For cross-rank overlap use "
                f"cross_rank_kernel_overlap_at_iter."
            )
        A = TraceDataLoader.merge_intervals(group_a)
        B = TraceDataLoader.merge_intervals(group_b)
        i = j = 0
        total = 0
        while i < len(A) and j < len(B):
            s = max(A[i][0], B[j][0])
            e = min(A[i][1], B[j][1])
            if e > s:
                total += e - s
            # advance the interval that ends earlier
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return total

    @staticmethod
    def split_solo_vs_overlap(
        targets: Sequence[KernelEvent], competitors: Sequence[KernelEvent]
    ) -> Tuple[List[KernelEvent], List[KernelEvent]]:
        """Split `targets` into (solo, overlapping) groups vs competitor intervals.

        A target kernel is "solo" iff it has zero overlap with any competitor.
        Used to construct BW_solo (P1) baseline samples.

        WARNING: Single-rank only. targets and competitors must share the same
        rank; mixing ranks would put all targets in the wrong bucket.
        """
        if not targets:
            return [], []
        if not competitors:
            return list(targets), []
        TraceDataLoader._assert_same_rank(targets, "split_solo_vs_overlap targets")
        TraceDataLoader._assert_same_rank(competitors, "split_solo_vs_overlap competitors")
        if next(iter(targets)).rank != next(iter(competitors)).rank:
            raise AssertionError(
                f"split_solo_vs_overlap: targets and competitors must share "
                f"the same rank; got {next(iter(targets)).rank} vs "
                f"{next(iter(competitors)).rank}."
            )
        comp = sorted([(c.start_us, c.end_us) for c in competitors])
        comp_starts = [c[0] for c in comp]
        solo: List[KernelEvent] = []
        overlap: List[KernelEvent] = []
        for t in targets:
            ts, te = t.start_us, t.end_us
            # Find competitor intervals potentially overlapping [ts, te]
            i = bisect.bisect_left(comp_starts, te)  # first comp with start >= te
            collides = False
            # check competitors with start < te, walking back while their end > ts
            j = i - 1
            while j >= 0 and comp[j][1] > ts:
                if comp[j][0] < te and comp[j][1] > ts:
                    collides = True
                    break
                j -= 1
            if collides:
                overlap.append(t)
            else:
                solo.append(t)
        return solo, overlap

    # ------------------------------------------------------------------
    # Round 5: cross-rank helpers (iter-anchored soft alignment)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_intervals_iter_rel(events: Sequence[KernelEvent]) -> List[Tuple[int, int]]:
        """Same as merge_intervals but uses iter_rel_*_us fields.

        Internal helper for cross-rank analysis. Single rank input is allowed
        but not required; cross-rank input is valid because iter_rel_*_us are
        comparable across ranks within barrier-skew tolerance.
        """
        if not events:
            return []
        intervals = sorted([(e.iter_rel_start_us, e.iter_rel_end_us) for e in events])
        merged: List[Tuple[int, int]] = [intervals[0]]
        for s, e in intervals[1:]:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def cross_rank_kernel_overlap_at_iter(
        loader: "TraceDataLoader",
        ranks: Sequence[int],
        iteration: int,
        name_pattern: Optional[str] = None,
    ) -> int:
        """Total GPU-timeline overlap among multiple ranks' kernels at the same iter.

        Uses ``iter_rel_*_us`` (iteration-anchored, see KernelEvent docstring).
        Valid within NCCL barrier-skew tolerance (~10-100us depending on node
        topology). Returns 0 when fewer than 2 ranks have kernels at this iter.

        Use case: P1 contention - whether multiple ranks' communication kernels
        share GPU/link time at the same training step.

        Args:
            loader: TraceDataLoader instance
            ranks: list of global ranks to consider
            iteration: training iteration ID (cross-rank synchronization point)
            name_pattern: optional substring filter (e.g. "ncclDevKernel")

        Returns:
            Sum of pairwise overlap durations (us), iter-relative timeline.
        """
        if len(ranks) < 2:
            return 0
        per_rank: Dict[int, List[KernelEvent]] = {}
        for r in ranks:
            evts = loader.get_kernels(rank=r, iteration=iteration)
            if name_pattern is not None:
                evts = [e for e in evts if name_pattern in e.name]
            if evts:
                per_rank[r] = evts
        if len(per_rank) < 2:
            return 0
        # Merge intervals per rank in iter-relative timeline
        merged_per_rank = {
            r: TraceDataLoader._merge_intervals_iter_rel(evts) for r, evts in per_rank.items()
        }
        # Pairwise overlap sum
        rank_list = sorted(merged_per_rank.keys())
        total = 0
        for i in range(len(rank_list)):
            for j in range(i + 1, len(rank_list)):
                A = merged_per_rank[rank_list[i]]
                B = merged_per_rank[rank_list[j]]
                ai = bi = 0
                while ai < len(A) and bi < len(B):
                    s = max(A[ai][0], B[bi][0])
                    e = min(A[ai][1], B[bi][1])
                    if e > s:
                        total += e - s
                    if A[ai][1] < B[bi][1]:
                        ai += 1
                    else:
                        bi += 1
        return total

    @staticmethod
    def same_iter_kernel_dispersion(
        loader: "TraceDataLoader", iteration: int, name_contains: str
    ) -> Dict[int, Dict[str, float]]:
        """Per-rank statistics for a kernel name at a given iteration.

        Returns ``{rank: {sum_dur_us, count, mean_dur_us}}`` so callers can
        compute CV / max-min ratios for straggler detection (e.g. which rank
        spent more time on the same NCCL AllReduce kernel).

        Uses ``duration_us`` (cross-rank-safe) and ``iteration``
        (cross-rank-safe). Does NOT use timing fields - dispersion is on
        durations alone, which is the right choice for kernel-level straggler
        analysis.
        """
        result: Dict[int, Dict[str, float]] = {}
        for rank in loader.get_ranks():
            if rank < 0:
                continue
            kernels = loader.get_kernels(
                rank=rank, iteration=iteration, name_contains=name_contains
            )
            if not kernels:
                continue
            durs = [k.duration_us for k in kernels]
            sum_dur = float(sum(durs))
            count = len(durs)
            result[rank] = {
                "sum_dur_us": sum_dur,
                "count": float(count),
                "mean_dur_us": sum_dur / count if count > 0 else 0.0,
            }
        return result

    @staticmethod
    def cross_rank_iter_relative_intervals(
        loader: "TraceDataLoader", ranks: Sequence[int], iteration: int
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Per-rank merged kernel intervals in iter-relative microseconds.

        Used by P1 cross-rank link contention analysis to inspect GPU-timeline
        kernel placement at the same training step across multiple ranks.

        Returns ``{rank: [(iter_rel_start, iter_rel_end), ...]}`` with
        intervals merged within each rank.
        """
        result: Dict[int, List[Tuple[int, int]]] = {}
        for r in ranks:
            kernels = loader.get_kernels(rank=r, iteration=iteration)
            if kernels:
                result[r] = TraceDataLoader._merge_intervals_iter_rel(kernels)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest(self, raw_events: Sequence[Dict[str, Any]]) -> None:
        for ev in raw_events:
            # Tier-2 cuda_kernel records carry record_type and ph=X but use
            # top-level start_us/end_us (profiler-relative) instead of ts/dur.
            # Route them to kernel_events to avoid polluting the span index
            # with ts=0 placeholder entries.
            if ev.get("record_type") == "cuda_kernel":
                self.kernel_events.append(
                    KernelEvent(
                        name=ev.get("name", ""),
                        start_us=int(ev.get("start_us", 0) or 0),
                        end_us=int(ev.get("end_us", 0) or 0),
                        duration_us=int(ev.get("duration_us", 0) or 0),
                        rank=int(ev.get("g_rk", ev.get("pid", -1))),
                        device=int(ev.get("device", 0) or 0),
                        iteration=int(ev.get("iteration", -1)),
                        dp_rank=int(ev.get("dp_rk", -1)),
                        pp_rank=int(ev.get("pp_rk", -1)),
                        tp_rank=int(ev.get("tp_rk", -1)),
                        wall_start_us=int(ev.get("wall_start_us", 0) or 0),
                        wall_end_us=int(ev.get("wall_end_us", 0) or 0),
                        # Round 5: iteration-anchored time fields.
                        # Older trace files predating round 5 will not have these
                        # keys; fall back to 0 (callers must check field presence
                        # before doing cross-rank analysis).
                        iter_rel_start_us=int(ev.get("iter_rel_start_us", 0) or 0),
                        iter_rel_end_us=int(ev.get("iter_rel_end_us", 0) or 0),
                    )
                )
                continue
            ph = ev.get("ph")
            if ph == "X":
                self.span_events.append(
                    SpanEvent(
                        name=ev.get("name", ""),
                        ts=ev.get("ts", 0),
                        dur=ev.get("dur", 0),
                        rank=ev.get("pid", -1),
                        args=ev.get("args", {}),
                    )
                )
            elif ph == "C":
                args = ev.get("args", {})
                self.counter_samples.append(
                    CounterSample(
                        name=ev.get("name", ""),
                        ts=ev.get("ts", 0),
                        rank=ev.get("pid", -1),
                        iteration=int(args.get("iteration", -1)),
                        metrics={
                            k: v
                            for k, v in args.items()
                            if k not in {"iteration", "pid", "tid"} and isinstance(v, (int, float))
                        },
                    )
                )
            elif ph == "M" and ev.get("name") == "process_name":
                # Extract topology from metadata events
                pname = ev.get("args", {}).get("name", "")
                rank = ev.get("pid", -1)
                self._parse_topology_from_pname(rank, pname)

    def _parse_topology_from_pname(self, rank: int, pname: str) -> None:
        """Parse ``DP0-PP1-TP2`` style process names into topology dict."""
        parts: Dict[str, int] = {}
        for token in pname.split("-"):
            token = token.strip()
            if token.startswith("DP"):
                parts["dp"] = int(token[2:])
            elif token.startswith("PP"):
                parts["pp"] = int(token[2:])
            elif token.startswith("TP"):
                parts["tp"] = int(token[2:])
        if parts:
            self._topology[rank] = parts

    def _build_indices(self) -> None:
        """Sort events and build per-rank lookup structures."""
        # Also extract topology from span event args if metadata is absent
        for ev in self.span_events:
            r = ev.rank
            if r not in self._topology and ev.dp_rank >= 0:
                self._topology[r] = {"dp": ev.dp_rank, "pp": ev.pp_rank, "tp": ev.tp_rank}

        # Spans by rank
        spans_by_rank: Dict[int, List[SpanEvent]] = {}
        for ev in self.span_events:
            spans_by_rank.setdefault(ev.rank, []).append(ev)
        for lst in spans_by_rank.values():
            lst.sort(key=lambda e: e.ts)
        self._spans_by_rank = spans_by_rank

        # Counters by rank (sorted by ts for binary search)
        counters_by_rank: Dict[int, List[CounterSample]] = {}
        for cs in self.counter_samples:
            counters_by_rank.setdefault(cs.rank, []).append(cs)
        for lst in counters_by_rank.values():
            lst.sort(key=lambda c: c.ts)
        self._counters_by_rank = counters_by_rank

        # Pre-compute numpy timestamp arrays for bisect
        self._counter_ts_by_rank = {
            rank: np.array([c.ts for c in samples], dtype=np.int64)
            for rank, samples in counters_by_rank.items()
        }

        # Kernel events by rank, sorted by (iteration, start_us)
        kernels_by_rank: Dict[int, List[KernelEvent]] = {}
        for k in self.kernel_events:
            kernels_by_rank.setdefault(k.rank, []).append(k)
        for lst in kernels_by_rank.values():
            lst.sort(key=lambda k: (k.iteration, k.start_us))
        self._kernels_by_rank = kernels_by_rank

        self._ranks = sorted(
            set(
                list(spans_by_rank.keys())
                + list(counters_by_rank.keys())
                + list(kernels_by_rank.keys())
            )
        )
