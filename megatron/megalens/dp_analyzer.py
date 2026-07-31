# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Data Parallelism Inefficiency Analyzer.

Diagnoses two primary DP performance issues:

1. **Straggler detection** — identifies the DP rank that arrives latest at
   synchronisation barriers (e.g. ``allreduce``, ``grad-sync``) and
   correlates the delay with hardware metrics (thermal throttling,
   clock frequency drops).

2. **Communication / computation overlap** — quantifies how much of the
   ``allreduce`` or ``reduce-scatter`` time is hidden behind backward
   computation, and reports the *exposed* (non-overlapped) communication
   cost.

All public methods return JSON-serialisable ``list[dict]`` results suitable
for rendering in a diagnostic report.
"""

from __future__ import annotations

import collections
import os
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from megatron.megalens.data_loader import TraceDataLoader, SpanEvent
from megatron.megalens.dp_lifecycle import aggregate_dp_lifecycle_partition
from megatron.megalens.paper_style import (
    FIG_W_SINGLE, FIG_W_DOUBLE, FIG_H, FIG_H_TALL, FIG_H_SHORT,
)


# ============================================================================
# Helpers
# ============================================================================

def _merge_intervals(
    intervals: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Classic greedy interval merging (sorted by start)."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _total_length(intervals: List[Tuple[int, int]]) -> int:
    return sum(e - s for s, e in intervals)


def _overlap_length(
    a_intervals: List[Tuple[int, int]],
    b_intervals: List[Tuple[int, int]],
) -> int:
    """Compute the total overlap between two sets of merged intervals."""
    overlap = 0
    j = 0
    for a_s, a_e in a_intervals:
        while j < len(b_intervals) and b_intervals[j][1] <= a_s:
            j += 1
        k = j
        while k < len(b_intervals) and b_intervals[k][0] < a_e:
            o_s = max(a_s, b_intervals[k][0])
            o_e = min(a_e, b_intervals[k][1])
            if o_s < o_e:
                overlap += o_e - o_s
            k += 1
    return overlap


# ============================================================================
# Thresholds
# ============================================================================

_THERMAL_THROTTLE_TEMP_C: float = 80.0
_CLOCK_DROP_RATIO: float = 0.92   # current / base < 0.92 → throttling

_DP_LIFECYCLE_EVENT_NAMES: Tuple[str, ...] = (
    "dp-param-all-gather",
    "dp-reduce-scatter",
    "dp-allreduce",
    "dp-param-sync-complete",
    "dp-grad-sync-complete",
)


def _load_reporting_dependencies() -> Tuple[Any, Any]:
    """Load optional plotting/reporting dependencies only when required."""
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


def _json_value(value: Any) -> Any:
    """Convert frozen reducer output into plain JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


# ============================================================================
# Analyzer
# ============================================================================

class DPAnalyzer:
    """Analyses Data-Parallelism inefficiencies using a loaded trace.

    Args:
        loader: A :class:`TraceDataLoader` populated with trace data.
    """

    def __init__(self, loader: TraceDataLoader) -> None:
        self.loader = loader

    # ------------------------------------------------------------------
    # 1. Straggler Diagnosis
    # ------------------------------------------------------------------

    def diagnose_stragglers(
        self,
        sync_event_names: Optional[List[str]] = None,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect DP stragglers at synchronisation barriers.

        Algorithm:
            1. Collect all sync events across DP ranks.
            2. Group by ``(iteration, event_name)`` and compute the start-time
               gap ``max(start) - min(start)`` across DP ranks.
            3. The rank that starts last is the *straggler*.
            4. Query the hardware monitor for the straggler's temperature and
               clock during the *preceding computation window* (the span
               event immediately before the sync on that rank).

        Returns:
            A list of dicts, one per (iteration, sync_event) group::

                {
                    "iteration": int,
                    "sync_event": str,
                    "gap_us": float,
                    "straggler_rank": int,
                    "fastest_rank": int,
                    "per_rank_start_ts": {rank: ts, ...},
                    "hardware_diagnosis": str | None,
                    "hw_detail": {...} | None,
                }
        """
        if sync_event_names is None:
            sync_event_names = [
                "allreduce", "grad-sync", "backward-cooldown",
                "dp-reduce-scatter", "dp-allreduce", "all-grads-sync",
            ]

        # Identify DP ranks: unique global ranks with PP=0 (or all ranks if
        # topology is unavailable).
        dp_ranks = self.loader.get_dp_ranks()
        if not dp_ranks:
            dp_ranks = self.loader.get_ranks()

        # Gather sync events per (iteration, name, rank)
        GroupKey = Tuple[int, str]
        grouped: Dict[GroupKey, Dict[int, SpanEvent]] = collections.defaultdict(dict)

        for rank in dp_ranks:
            for sname in sync_event_names:
                for ev in self.loader.get_events_by_name(sname, rank=rank, iteration=iteration):
                    key: GroupKey = (ev.iteration, ev.name)
                    if rank not in grouped[key] or ev.ts < grouped[key][rank].ts:
                        grouped[key][rank] = ev

        results: List[Dict[str, Any]] = []
        for (iter_id, ev_name), rank_events in sorted(grouped.items()):
            if len(rank_events) < 2:
                continue

            starts = {r: ev.ts for r, ev in rank_events.items()}
            min_ts = min(starts.values())
            max_ts = max(starts.values())
            gap_us = float(max_ts - min_ts)

            straggler_rank = max(starts, key=starts.get)  # type: ignore[arg-type]
            fastest_rank = min(starts, key=starts.get)     # type: ignore[arg-type]

            hw_diagnosis, hw_detail = self._correlate_hardware(
                straggler_rank, rank_events[straggler_rank]
            )

            results.append({
                "iteration": iter_id,
                "sync_event": ev_name,
                "gap_us": gap_us,
                "straggler_rank": straggler_rank,
                "fastest_rank": fastest_rank,
                "per_rank_start_ts": starts,
                "hardware_diagnosis": hw_diagnosis,
                "hw_detail": hw_detail,
            })

        return results

    def _correlate_hardware(
        self, rank: int, sync_event: SpanEvent
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Check HW metrics in the computation window preceding *sync_event*.

        Uses a dynamic window from the *iteration start* on this rank up to
        the sync event start, covering the full backward computation phase.
        Falls back to a 50 ms look-back if no explicit iteration boundary is
        found (e.g. trace without iteration scopes).  A minimum of 1 ms is
        enforced so NVML 10 ms sampling has at least one sample.
        """
        window_end = sync_event.ts
        iter_id = sync_event.iteration

        # Try to anchor window to the iteration start on this rank
        iter_start: Optional[int] = None
        if iter_id >= 0:
            for ev in self.loader.get_events_by_name(
                "iteration", rank=rank, iteration=iter_id
            ):
                if ev.dur > 0:
                    iter_start = ev.ts
                    break

        if iter_start is not None and iter_start < window_end:
            window_start = iter_start
        else:
            # Fallback: 50 ms look-back (covers typical backward pass)
            window_start = max(0, window_end - 50_000)

        # Enforce minimum 1 ms window so NVML sampling can contribute
        if window_end - window_start < 1_000:
            window_start = max(0, window_end - 1_000)

        temp_result = self.loader.get_hardware_metrics_in_window(
            window_start, window_end, rank, "Temp_C"
        )
        clock_result = self.loader.get_hardware_metrics_in_window(
            window_start, window_end, rank, "SM_Clock_MHz"
        )
        base_clock_result = self.loader.get_hardware_metrics_in_window(
            window_start, window_end, rank, "SM_Base_Clock_MHz"
        )

        if temp_result is None and clock_result is None:
            return None, None

        detail: Dict[str, Any] = {}
        diagnosis_parts: List[str] = []

        if temp_result is not None:
            detail["Temp_C_peak"] = temp_result.peak
            detail["Temp_C_mean"] = round(temp_result.mean, 1)

        if clock_result is not None:
            detail["SM_Clock_MHz_mean"] = round(clock_result.mean, 1)

        base_clock: float = 0.0
        if base_clock_result is not None and base_clock_result.mean > 0:
            base_clock = base_clock_result.mean
            detail["SM_Base_Clock_MHz"] = round(base_clock, 1)

        is_hot = (temp_result is not None
                  and temp_result.peak >= _THERMAL_THROTTLE_TEMP_C)
        is_throttled = (clock_result is not None
                        and base_clock > 0
                        and clock_result.mean < base_clock * _CLOCK_DROP_RATIO)

        if is_hot and is_throttled:
            diagnosis_parts.append(
                "Hardware-Induced Straggler (Thermal Throttling): "
                f"Temp {temp_result.peak:.0f}°C, "
                f"Clock {clock_result.mean:.0f}/{base_clock:.0f} MHz"
            )
        elif is_throttled:
            diagnosis_parts.append(
                f"Clock Throttling Detected: {clock_result.mean:.0f}"
                f"/{base_clock:.0f} MHz"
            )
        elif is_hot:
            diagnosis_parts.append(
                f"High Temperature Warning: {temp_result.peak:.0f}°C"
            )

        hw_diag = "; ".join(diagnosis_parts) if diagnosis_parts else None
        return hw_diag, detail if detail else None

    # ------------------------------------------------------------------
    # 2. DP Step Time Balance (Load Imbalance)
    # ------------------------------------------------------------------

    def analyze_step_time_balance(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compare per-iteration step time across DP ranks.

        For each iteration, collects forward+backward total duration per
        DP rank and computes cross-rank CV (coefficient of variation).
        A high CV indicates DP load imbalance.  Also flags the slowest
        rank per iteration as the DP straggler.

        Returns:
            List of dicts, one per iteration::

                {
                    "iteration": int,
                    "per_rank_step_ms": {rank: float, ...},
                    "mean_step_ms": float,
                    "cv": float,
                    "slowest_rank": int,
                    "fastest_rank": int,
                    "imbalance_pct": float,
                }
        """
        dp_ranks = self.loader.get_dp_ranks()
        if not dp_ranks:
            dp_ranks = self.loader.get_ranks()

        rank_iter_dur: Dict[int, Dict[int, float]] = collections.defaultdict(dict)

        for rank in dp_ranks:
            for phase_name in ["forward-step", "backward-step"]:
                for ev in self.loader.get_events_by_name(
                    phase_name, rank=rank, iteration=iteration
                ):
                    if ev.dur <= 0:
                        continue
                    it = ev.iteration
                    rank_iter_dur[rank][it] = (
                        rank_iter_dur[rank].get(it, 0.0) + ev.dur / 1e3
                    )

        all_iters = sorted(
            set(it for rd in rank_iter_dur.values() for it in rd.keys())
        )

        results: List[Dict[str, Any]] = []
        for it in all_iters:
            per_rank: Dict[int, float] = {}
            for r in dp_ranks:
                v = rank_iter_dur.get(r, {}).get(it)
                if v is not None:
                    per_rank[r] = v
            if len(per_rank) < 2:
                continue
            vals = list(per_rank.values())
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            cv = std_v / mean_v if mean_v > 0 else 0.0
            slowest = max(per_rank, key=per_rank.get)  # type: ignore[arg-type]
            fastest = min(per_rank, key=per_rank.get)   # type: ignore[arg-type]
            imbalance = (
                (per_rank[slowest] - per_rank[fastest]) / per_rank[fastest] * 100
                if per_rank[fastest] > 0 else 0.0
            )
            results.append({
                "iteration": it,
                "per_rank_step_ms": per_rank,
                "mean_step_ms": round(mean_v, 3),
                "cv": round(cv, 4),
                "slowest_rank": slowest,
                "fastest_rank": fastest,
                "imbalance_pct": round(imbalance, 2),
            })

        return results

    # ------------------------------------------------------------------
    # 3. Gradient Sync Overhead
    # ------------------------------------------------------------------

    def analyze_grad_sync_overhead(
        self,
        sync_names: Optional[List[str]] = None,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Quantify gradient synchronisation cost relative to iteration time.

        Computes ``sync_ratio = total_grad_sync_time / iteration_time``
        for each DP rank and iteration.

        Returns:
            List of dicts, one per (rank, iteration)::

                {
                    "rank": int,
                    "iteration": int,
                    "grad_sync_us": float,
                    "iter_time_us": float,
                    "sync_ratio": float,
                    "n_sync_events": int,
                }
        """
        if sync_names is None:
            sync_names = [
                "grad-sync", "all-grads-sync", "allreduce",
                "_reduce_scatter_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "dp-reduce-scatter", "dp-allreduce",
                "sp-layernorm-allreduce", "embedding-grads-allreduce",
            ]

        dp_ranks = self.loader.get_dp_ranks()
        if not dp_ranks:
            dp_ranks = self.loader.get_ranks()

        # Iteration boundaries
        iter_dur: Dict[Tuple[int, int], float] = {}
        for ev in self.loader.get_iteration_events(iteration=iteration):
            if ev.dur > 0 and ev.rank in dp_ranks:
                iter_dur[(ev.rank, ev.iteration)] = float(ev.dur)

        sync_by_ri: Dict[Tuple[int, int], List[float]] = collections.defaultdict(list)
        for rank in dp_ranks:
            for sname in sync_names:
                for ev in self.loader.get_events_by_name(
                    sname, rank=rank, iteration=iteration
                ):
                    if ev.dur > 0:
                        sync_by_ri[(rank, ev.iteration)].append(float(ev.dur))

        results: List[Dict[str, Any]] = []
        for (rank, it), durations in sorted(sync_by_ri.items()):
            total_sync = sum(durations)
            it_time = iter_dur.get((rank, it), 0.0)
            ratio = total_sync / it_time if it_time > 0 else 0.0
            results.append({
                "rank": rank,
                "iteration": it,
                "grad_sync_us": round(total_sync, 1),
                "iter_time_us": round(it_time, 1),
                "sync_ratio": round(ratio, 4),
                "n_sync_events": len(durations),
            })

        return results

    # ------------------------------------------------------------------
    # 4. Memory Efficiency Analysis
    # ------------------------------------------------------------------

    def analyze_memory_efficiency(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Profile GPU memory utilisation across DP ranks using HW counters.

        Queries ``Mem_Used_MB`` and ``Mem_Util_pct`` from the hardware
        monitor during each iteration window.  High variance across DP
        ranks indicates memory fragmentation or uneven ZeRO sharding.

        Returns:
            List of dicts, one per (rank, iteration)::

                {
                    "rank": int,
                    "iteration": int,
                    "mem_peak_mb": float,
                    "mem_mean_mb": float,
                    "mem_util_peak_pct": float,
                    "mem_util_mean_pct": float,
                }
        """
        dp_ranks = self.loader.get_dp_ranks()
        if not dp_ranks:
            dp_ranks = self.loader.get_ranks()

        results: List[Dict[str, Any]] = []
        for ev in self.loader.get_iteration_events(iteration=iteration):
            if ev.dur <= 0 or ev.rank not in dp_ranks:
                continue

            mem_mb = self.loader.get_hardware_metrics_in_window(
                ev.ts, ev.end_ts, ev.rank, "Mem_Used_MB"
            )
            mem_pct = self.loader.get_hardware_metrics_in_window(
                ev.ts, ev.end_ts, ev.rank, "Mem_Util_pct"
            )

            results.append({
                "rank": ev.rank,
                "iteration": ev.iteration,
                "mem_peak_mb": round(mem_mb.peak, 1) if mem_mb else None,
                "mem_mean_mb": round(mem_mb.mean, 1) if mem_mb else None,
                "mem_util_peak_pct": round(mem_pct.peak, 1) if mem_pct else None,
                "mem_util_mean_pct": round(mem_pct.mean, 1) if mem_pct else None,
            })

        return results

    # ------------------------------------------------------------------
    # 5. Communication / Computation Overlap
    # ------------------------------------------------------------------

    def analyze_comm_overlap(
        self,
        compute_event_names: Optional[List[str]] = None,
        comm_event_names: Optional[List[str]] = None,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Quantify overlap between backward computation and DP communication.

        Algorithm:
            For each DP rank and iteration:
            1. Collect all *compute* intervals (e.g. ``backward-step``).
            2. Collect all *comm* intervals (e.g. ``_reduce_scatter*``,
               ``allreduce``).
            3. Merge each set independently.
            4. Compute the pairwise overlap duration.
            5. **Exposed comm** = total_comm - overlap.

        Returns:
            A list of dicts, one per (rank, iteration)::

                {
                    "rank": int,
                    "iteration": int,
                    "total_compute_us": float,
                    "total_comm_us": float,
                    "overlap_us": float,
                    "exposed_comm_us": float,
                    "overlap_ratio": float,   # overlap / total_comm
                }
        """
        if compute_event_names is None:
            compute_event_names = ["backward-step", "backward", "backward-cooldown"]
        if comm_event_names is None:
            comm_event_names = [
                "_reduce_scatter_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "allreduce", "_reduce", "grad-sync",
                "dp-reduce-scatter", "dp-allreduce",
                "tp-reduce-scatter", "tp-allreduce",
                "tp-all-gather-first", "tp-all-gather-last",
                "tp-reduce-scatter-last",
            ]

        dp_ranks = self.loader.get_dp_ranks()
        if not dp_ranks:
            dp_ranks = self.loader.get_ranks()

        results: List[Dict[str, Any]] = []

        for rank in dp_ranks:
            # Group by iteration
            comp_by_iter: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            comm_by_iter: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            for cname in compute_event_names:
                for ev in self.loader.get_events_by_name(cname, rank=rank, iteration=iteration):
                    if ev.dur > 0:
                        comp_by_iter[ev.iteration].append((ev.ts, ev.end_ts))

            for cname in comm_event_names:
                for ev in self.loader.get_events_by_name(cname, rank=rank, iteration=iteration):
                    if ev.dur > 0:
                        comm_by_iter[ev.iteration].append((ev.ts, ev.end_ts))

            all_iters = sorted(set(comp_by_iter.keys()) | set(comm_by_iter.keys()))
            for it in all_iters:
                comp_merged = _merge_intervals(comp_by_iter.get(it, []))
                comm_merged = _merge_intervals(comm_by_iter.get(it, []))

                total_comp = _total_length(comp_merged)
                total_comm = _total_length(comm_merged)
                overlap = _overlap_length(comm_merged, comp_merged)

                exposed = max(0, total_comm - overlap)
                ratio = (overlap / total_comm) if total_comm > 0 else 0.0

                results.append({
                    "rank": rank,
                    "iteration": it,
                    "total_compute_us": float(total_comp),
                    "total_comm_us": float(total_comm),
                    "overlap_us": float(overlap),
                    "exposed_comm_us": float(exposed),
                    "overlap_ratio": round(ratio, 4),
                })

        return results

    # ------------------------------------------------------------------
    # 6. Typed Collective Lifecycle Evidence
    # ------------------------------------------------------------------

    def analyze_collective_lifecycle(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Correlate DP dispatch and current-stream completion evidence.

        This target-specific semantic view is additive: the five source
        analyses above retain their original event lists and calculations.
        Dispatch span durations are excluded from lifecycle timing.  A
        successful completion proves only its declared current-stream
        dependency, and a multi-operation stream-join boundary contributes
        its interval once.
        """
        dp_ranks = self.loader.get_dp_ranks()
        if not dp_ranks:
            dp_ranks = self.loader.get_ranks()

        partitions: Dict[Tuple[int, int, int], List[SpanEvent]] = collections.defaultdict(list)
        invalid_partition_sequence = 0
        for rank in dp_ranks:
            for event_name in _DP_LIFECYCLE_EVENT_NAMES:
                for event in self.loader.get_events_by_name(
                    event_name, rank=rank, iteration=iteration
                ):
                    event_iteration = event.iteration
                    if (
                        type(event.rank) is int
                        and event.rank >= 0
                        and type(event_iteration) is int
                        and event_iteration >= 0
                    ):
                        key = (0, event.rank, event_iteration)
                    else:
                        key = (1, invalid_partition_sequence, 0)
                        invalid_partition_sequence += 1
                    partitions[key].append(event)

        results: List[Dict[str, Any]] = []
        for key in sorted(partitions):
            lifecycle = aggregate_dp_lifecycle_partition(partitions[key])
            results.append(_json_value(asdict(lifecycle)))
        return results


# ============================================================================
# Report Logger (mirrors pp_analyzer.ReportLogger)
# ============================================================================

class _ReportLogger:
    def __init__(self, filepath: str, title: str) -> None:
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write(f" {title}\n")
            f.write("=" * 90 + "\n\n")

    def log(self, message: str, end: str = "\n") -> None:
        print(message, end=end)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(message + end)


def _pick_style(plt: Any) -> None:
    for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if s in plt.style.available:
            plt.style.use(s)
            return
    plt.style.use("default")


# ============================================================================
# Visualization — Straggler Diagnosis
# ============================================================================

def generate_straggler_plots(
    straggler_data: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate 4-panel root-cause visualisation for DP stragglers."""
    if not straggler_data:
        print("[DP Straggler] No straggler data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style(plt)
    df = pd.DataFrame(straggler_data)
    ranks = sorted({r for d in straggler_data for r in d["per_rank_start_ts"]})
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_DOUBLE, FIG_H_TALL))
    fig.suptitle("Data Parallelism: Straggler Diagnosis", fontsize=22, fontweight="bold")

    # ---- Panel 1: Gap (us) per sync event over iterations ----
    ax = axes[0, 0]
    for sname in df["sync_event"].unique():
        sub = df[df["sync_event"] == sname].sort_values("iteration")
        ax.plot(sub["iteration"], sub["gap_us"], marker="o", markersize=5,
                linewidth=1.5, alpha=0.8, label=sname)
    ax.set_title("Sync Barrier Gap Over Iterations\n(Higher = Worse Imbalance)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Gap (μs)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Panel 2: Straggler frequency per rank ----
    ax = axes[0, 1]
    straggler_counts = df["straggler_rank"].value_counts().reindex(ranks, fill_value=0)
    bar_colors = [rank_color[r] for r in straggler_counts.index]
    bars = ax.bar([str(r) for r in straggler_counts.index],
                  straggler_counts.values, color=bar_colors, edgecolor="black", alpha=0.85)
    if len(straggler_counts) > 0 and straggler_counts.max() > 0:
        worst_idx = int(np.argmax(straggler_counts.values))
        bars[worst_idx].set_edgecolor("red")
        bars[worst_idx].set_linewidth(3)
        bars[worst_idx].set_hatch("//")
    ax.set_title("Straggler Frequency per Rank\n(Tallest bar = most frequent straggler)", fontsize=14, fontweight="bold")
    ax.set_ylabel("# Times Straggler", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)

    # ---- Panel 3: Gap timeline coloured by HW diagnosis ----
    ax = axes[1, 0]
    hw_mask = df["hardware_diagnosis"].notna()
    normal = df[~hw_mask]
    flagged = df[hw_mask]
    ax.scatter(normal["iteration"], normal["gap_us"], c="steelblue", alpha=0.6,
               s=40, edgecolors="none", label="Normal")
    if not flagged.empty:
        ax.scatter(flagged["iteration"], flagged["gap_us"], c="red", marker="X",
                   s=120, edgecolors="black", linewidth=0.8, zorder=10,
                   label="HW Throttle Flagged")
    ax.set_title("Barrier Gap Timeline\n(Red X = Hardware-Correlated Straggler)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Gap (μs)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Panel 4: Straggler rank per-iteration heatmap ----
    ax = axes[1, 1]
    iterations = sorted(df["iteration"].unique())
    heatmap = np.zeros((len(ranks), len(iterations)))
    rank_idx_map = {r: i for i, r in enumerate(ranks)}
    iter_idx_map = {it: j for j, it in enumerate(iterations)}
    for _, row in df.iterrows():
        ri = rank_idx_map.get(row["straggler_rank"])
        ii = iter_idx_map.get(row["iteration"])
        if ri is not None and ii is not None:
            heatmap[ri, ii] += 1
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(r) for r in ranks])
    step = max(1, len(iterations) // 15)
    ax.set_xticks(range(0, len(iterations), step))
    ax.set_xticklabels([str(iterations[i]) for i in range(0, len(iterations), step)], fontsize=8)
    ax.set_ylabel("Global Rank", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_title("Straggler Heatmap (Rank × Iteration)\n(Hot cells = repeated straggler)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# Times Straggler")

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "dp_straggler_diagnosis.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[DP Straggler] Plot saved to {path}")


# ============================================================================
# Visualization — Comm / Compute Overlap
# ============================================================================

def generate_overlap_plots(
    overlap_data: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate 3-panel visualisation for communication overlap analysis."""
    if not overlap_data:
        print("[DP Overlap] No overlap data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style(plt)
    df = pd.DataFrame(overlap_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Data Parallelism: Communication / Computation Overlap Analysis",
                 fontsize=22, fontweight="bold")

    # ---- Panel 1: Stacked bar — time breakdown per rank (averaged) ----
    ax = axes[0]
    avg = df.groupby("rank").agg({
        "overlap_us": "mean",
        "exposed_comm_us": "mean",
        "total_compute_us": "mean",
    }).reindex(ranks)
    x = np.arange(len(ranks))
    w = 0.55
    ax.bar(x, avg["total_compute_us"] / 1e3, w, label="Backward Compute", color="mediumseagreen", edgecolor="black")
    ax.bar(x, avg["overlap_us"] / 1e3, w, bottom=avg["total_compute_us"] / 1e3,
           label="Overlapped Comm (hidden)", color="steelblue", edgecolor="black")
    ax.bar(x, avg["exposed_comm_us"] / 1e3,  w,
           bottom=(avg["total_compute_us"] + avg["overlap_us"]) / 1e3,
           label="Exposed Comm (blocking)", color="salmon", hatch="//", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.set_title("Avg Iteration Time Breakdown\n(Red = Wasted GPU Time)", fontsize=14, fontweight="bold")
    ax.legend(fontsize="small", frameon=True)

    # ---- Panel 2: Overlap ratio per rank (bar) ----
    ax = axes[1]
    avg_ratio = df.groupby("rank")["overlap_ratio"].mean().reindex(ranks)
    bar_colors = [rank_color[r] for r in ranks]
    ax.bar([str(r) for r in ranks], avg_ratio.values * 100, color=bar_colors,
           edgecolor="black", alpha=0.85)
    ax.axhline(y=100, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Ideal (100%)")
    ax.set_ylabel("Overlap Ratio (%)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.set_title("Avg Comm Overlap Ratio\n(100% = Fully Hidden)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize="small")
    for i, v in enumerate(avg_ratio.values):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # ---- Panel 3: Exposed comm over iterations (per rank) ----
    ax = axes[2]
    for rank in ranks:
        sub = df[df["rank"] == rank].sort_values("iteration")
        ax.plot(sub["iteration"], sub["exposed_comm_us"] / 1e3, marker="o",
                markersize=4, linewidth=1.2, alpha=0.8,
                label=f"Rank {rank}", color=rank_color[rank])
    ax.set_title("Exposed Comm per Iteration\n(Non-overlapped blocking time)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Exposed Comm (ms)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "dp_comm_overlap_analysis.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[DP Overlap] Plot saved to {path}")


# ============================================================================
# Visualization — Step Time Balance
# ============================================================================

def generate_step_balance_plots(
    balance_data: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate 3-panel plot for DP load imbalance analysis."""
    if not balance_data:
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style(plt)
    df = pd.DataFrame(balance_data)
    ranks = sorted({r for d in balance_data for r in d["per_rank_step_ms"]})
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Data Parallelism: Step Time Balance Across DP Ranks",
                 fontsize=22, fontweight="bold")

    # Panel 1: Per-rank step time over iterations
    ax = axes[0]
    for rank in ranks:
        iters, times = [], []
        for _, row in df.iterrows():
            v = row["per_rank_step_ms"].get(rank)
            if v is not None:
                iters.append(row["iteration"])
                times.append(v)
        if iters:
            ax.plot(iters, times, marker="o", markersize=3, linewidth=1, alpha=0.7,
                    label=f"Rank {rank}", color=rank_color[rank])
    ax.set_title("Step Time per DP Rank\n(Divergence = load imbalance)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Step Time (ms)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: CV over iterations
    ax = axes[1]
    ax.plot(df["iteration"], df["cv"] * 100, marker="s", markersize=4,
            linewidth=1.5, color="steelblue")
    ax.axhline(y=5, color="orange", linestyle="--", linewidth=2, label="Warning (5%)")
    ax.axhline(y=10, color="red", linestyle="--", linewidth=2, label="Critical (10%)")
    ax.set_title("Step Time CV Across DP Ranks\n(>5% = imbalanced)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Slowest rank frequency
    ax = axes[2]
    slowest_counts = df["slowest_rank"].value_counts().reindex(ranks, fill_value=0)
    bar_colors = [rank_color[r] for r in slowest_counts.index]
    bars = ax.bar([str(r) for r in slowest_counts.index],
                  slowest_counts.values, color=bar_colors,
                  edgecolor="black", alpha=0.85)
    if len(slowest_counts) > 0 and slowest_counts.max() > 0:
        worst_idx = int(np.argmax(slowest_counts.values))
        bars[worst_idx].set_edgecolor("red")
        bars[worst_idx].set_linewidth(3)
        bars[worst_idx].set_hatch("//")
    ax.set_title("Slowest DP Rank Frequency\n(Tallest = chronic straggler)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("# Iterations Slowest", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "dp_step_balance.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[DP Balance] Plot saved to {path}")


# ============================================================================
# Visualization — Grad Sync Overhead
# ============================================================================

def generate_grad_sync_plots(
    sync_data: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate 2-panel plot for gradient sync overhead."""
    if not sync_data:
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style(plt)
    df = pd.DataFrame(sync_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Data Parallelism: Gradient Synchronisation Overhead",
                 fontsize=22, fontweight="bold")

    # Panel 1: Sync ratio over iterations per rank
    ax = axes[0]
    for rank in ranks:
        sub = df[df["rank"] == rank].sort_values("iteration")
        ax.plot(sub["iteration"], sub["sync_ratio"] * 100, marker="o",
                markersize=3, linewidth=1, alpha=0.7,
                label=f"Rank {rank}", color=rank_color[rank])
    ax.axhline(y=20, color="orange", linestyle="--", linewidth=2, label="Warning (20%)")
    ax.set_title("Grad Sync Ratio per Iteration\n(% of iteration time in sync)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Sync / Iteration (%)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Average sync time per rank (stacked: sync vs compute)
    ax = axes[1]
    avg = df.groupby("rank").agg(
        sync_ms=("grad_sync_us", lambda s: s.mean() / 1e3),
        iter_ms=("iter_time_us", lambda s: s.mean() / 1e3),
    ).reindex(ranks)
    avg["compute_ms"] = (avg["iter_ms"] - avg["sync_ms"]).clip(lower=0)
    x = np.arange(len(ranks))
    ax.bar(x, avg["compute_ms"], 0.55, label="Compute", color="mediumseagreen",
           edgecolor="black")
    ax.bar(x, avg["sync_ms"], 0.55, bottom=avg["compute_ms"],
           label="Grad Sync", color="salmon", hatch="//", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_title("Avg Iteration Time Breakdown\n(Red = sync overhead)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.legend(fontsize="small", frameon=True)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "dp_grad_sync_overhead.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[DP Grad Sync] Plot saved to {path}")


# ============================================================================
# Report Generation
# ============================================================================

def _write_step_balance_report(
    balance_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[DP Step Time Balance Report]")
    logger.log("=" * 90)

    if not balance_data:
        logger.log("  Insufficient data for DP step time balance analysis.")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(balance_data)
    mean_cv = df["cv"].mean()
    max_cv = df["cv"].max()
    logger.log(f"  Iterations analysed: {len(df)}")
    logger.log(f"  Mean CV across iterations: {mean_cv:.2%}")
    logger.log(f"  Max CV: {max_cv:.2%}")

    if mean_cv > 0.05:
        logger.log("  [WARNING] DP ranks have > 5% step time variance on average.")
        logger.log("            Possible causes: data skew, heterogeneous hardware, "
                    "or uneven model sharding (ZeRO).")
    else:
        logger.log("  DP ranks are well-balanced.")

    freq = df["slowest_rank"].value_counts()
    logger.log("\n  Slowest-rank frequency:")
    for rank, count in freq.head(5).items():
        pct = count / len(df) * 100
        marker = " *** CHRONIC STRAGGLER ***" if pct > 50 else ""
        logger.log(f"    Rank {rank}: {count}/{len(df)} iterations ({pct:.1f}%){marker}")

    logger.log("=" * 90 + "\n")


def _write_grad_sync_report(
    sync_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[DP Gradient Sync Overhead Report]")
    logger.log("=" * 90)

    if not sync_data:
        logger.log("  No gradient sync data available.")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(sync_data)
    avg = df.groupby("rank").agg(
        mean_sync_ms=("grad_sync_us", lambda s: round(s.mean() / 1e3, 2)),
        mean_ratio=("sync_ratio", "mean"),
        n=("sync_ratio", "count"),
    )
    avg["mean_ratio"] = avg["mean_ratio"].map(lambda x: f"{x:.1%}")
    logger.log(avg.to_string())
    logger.log("")

    for rank in sorted(df["rank"].unique()):
        r_data = df[df["rank"] == rank]
        mean_ratio = r_data["sync_ratio"].mean()
        if mean_ratio > 0.20:
            logger.log(f"  [WARNING] Rank {rank}: Grad sync occupies {mean_ratio:.1%} "
                        f"of iteration time.  Consider async overlap or reducing "
                        f"DP communication frequency.")

    logger.log("=" * 90 + "\n")


def _write_memory_report(
    mem_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[DP Memory Efficiency Report]")
    logger.log("=" * 90)

    if not mem_data:
        logger.log("  No memory HW counter data available.")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(mem_data)
    has_mem = df["mem_peak_mb"].notna().any()
    if not has_mem:
        logger.log("  HW monitor did not record Mem_Used_MB. Skipping.")
        logger.log("=" * 90 + "\n")
        return

    avg = df.groupby("rank").agg(
        mean_peak_mb=("mem_peak_mb", "mean"),
        max_peak_mb=("mem_peak_mb", "max"),
        mean_util_pct=("mem_util_mean_pct", "mean"),
    )
    logger.log(avg.to_string(float_format="%.1f", na_rep="N/A"))

    peaks = avg["mean_peak_mb"].dropna()
    if len(peaks) >= 2:
        spread = (peaks.max() - peaks.min()) / peaks.min() * 100 if peaks.min() > 0 else 0
        if spread > 10:
            logger.log(f"\n  [WARNING] GPU memory usage spread across DP ranks: "
                        f"{spread:.1f}%. Possible ZeRO sharding imbalance or "
                        f"memory fragmentation.")
        else:
            logger.log(f"\n  Memory usage is balanced across DP ranks (spread {spread:.1f}%).")

    logger.log("=" * 90 + "\n")


def _write_straggler_report(
    straggler_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[DP Straggler Diagnosis Report]")
    logger.log("=" * 90)

    if not straggler_data:
        logger.log("  No synchronisation barriers found across DP ranks.")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(straggler_data)
    ranks = sorted({r for d in straggler_data for r in d["per_rank_start_ts"]})

    logger.log(f"  Analysed {len(df)} sync barrier events across {len(ranks)} DP ranks.\n")

    # Per-event summary table
    summary_rows = []
    for _, row in df.iterrows():
        summary_rows.append({
            "Iter": row["iteration"],
            "Sync Event": row["sync_event"],
            "Gap (μs)": f"{row['gap_us']:.1f}",
            "Straggler": f"Rank {row['straggler_rank']}",
            "HW Diag": row["hardware_diagnosis"] or "—",
        })
    summary_df = pd.DataFrame(summary_rows)
    logger.log(summary_df.to_string(index=False))

    # Straggler frequency
    logger.log("\n" + "-" * 90)
    logger.log("[Straggler Frequency]")
    freq = df["straggler_rank"].value_counts()
    for rank, count in freq.items():
        pct = count / len(df) * 100
        marker = " *** FREQUENT STRAGGLER ***" if pct > 50 else ""
        logger.log(f"  Rank {rank}: {count}/{len(df)} ({pct:.1f}%){marker}")

    # HW-correlated stragglers
    hw_flagged = df[df["hardware_diagnosis"].notna()]
    if not hw_flagged.empty:
        logger.log("\n" + "-" * 90)
        logger.log("[Hardware-Correlated Stragglers]")
        for _, row in hw_flagged.iterrows():
            logger.log(f"  Iter {row['iteration']} | {row['sync_event']} | "
                        f"Rank {row['straggler_rank']} | {row['hardware_diagnosis']}")
    else:
        logger.log("\n  No hardware-correlated stragglers detected.")

    logger.log("=" * 90 + "\n")


def _write_overlap_report(
    overlap_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[DP Communication / Computation Overlap Report]")
    logger.log("=" * 90)

    if not overlap_data:
        logger.log("  No overlap data available.")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(overlap_data)
    ranks = sorted(df["rank"].unique())

    logger.log(f"  Analysed {len(df)} (rank, iteration) data points.\n")

    # Per-rank average table
    avg = df.groupby("rank").agg({
        "total_compute_us": "mean",
        "total_comm_us": "mean",
        "overlap_us": "mean",
        "exposed_comm_us": "mean",
        "overlap_ratio": "mean",
    })
    avg.columns = ["Avg Compute (μs)", "Avg Comm (μs)", "Avg Overlap (μs)",
                    "Avg Exposed (μs)", "Avg Overlap Ratio"]
    avg["Avg Overlap Ratio"] = avg["Avg Overlap Ratio"].map(lambda x: f"{x:.1%}")
    logger.log(avg.to_string(float_format="%.1f"))
    logger.log("")

    # Flag poorly-overlapped ranks
    for rank in ranks:
        r_data = df[df["rank"] == rank]
        mean_ratio = r_data["overlap_ratio"].mean()
        mean_exposed_ms = r_data["exposed_comm_us"].mean() / 1e3
        if mean_ratio < 0.5:
            logger.log(f"  [WARNING] Rank {rank}: Only {mean_ratio:.1%} overlap, "
                        f"avg exposed comm = {mean_exposed_ms:.2f} ms.")
            logger.log(f"            Action: Increase gradient accumulation or "
                        f"enable async grad-reduce overlap.")

    logger.log("=" * 90 + "\n")


def _lifecycle_metric_value(row: Dict[str, Any], name: str) -> Any:
    return row[name]["value"]


def _write_lifecycle_report(
    lifecycle_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[DP Collective Lifecycle Evidence Report]")
    logger.log("=" * 90)

    if not lifecycle_data:
        logger.log("  No typed DP lifecycle data available.")
        return

    status_counts = collections.Counter(row["status"] for row in lifecycle_data)
    logger.log(
        "  Partitions: "
        + ", ".join(
            f"{status}={status_counts[status]}" for status in sorted(status_counts)
        )
    )
    logger.log("")
    for row in lifecycle_data:
        logger.log(
            f"  Rank {row['rank']} | Iter {row['iteration']} | {row['status']} | "
            f"dispatch={_lifecycle_metric_value(row, 'dispatch_attempt_count')} | "
            f"completion={_lifecycle_metric_value(row, 'completion_attempt_count')} | "
            "current-stream-ready="
            f"{_lifecycle_metric_value(row, 'current_stream_guaranteed_operation_count')} | "
            f"pending={_lifecycle_metric_value(row, 'pending_operation_count')} | "
            "failed-attempts="
            f"{_lifecycle_metric_value(row, 'failed_completion_attempt_count')} | "
            f"orphan-boundaries={_lifecycle_metric_value(row, 'orphan_completion_count')} | "
            f"dependency-union-us={_lifecycle_metric_value(row, 'exposed_dependency_union_us')}"
        )
        if row["status"] in ("partial", "unknown"):
            logger.log(f"    Evidence reason: {row['reason']}")

    logger.log("")
    logger.log(
        "  Completion guarantees are current-stream dependencies only; "
        "dispatch spans are excluded from physical communication duration."
    )
    logger.log("=" * 90 + "\n")


# ============================================================================
# Master Orchestrator
# ============================================================================

def analyze_dp_traces(
    traces: List[Dict[str, Any]],
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Run all DP analyses, generate report + plots.

    Args:
        traces: Aggregated Chrome Trace event list (post-``transform``).
        output_dir: Directory for PDF plots and TXT report.

    Returns:
        Dict containing the five source analysis result lists plus the
        target-specific ``"lifecycle_data"`` evidence list.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, pd = _load_reporting_dependencies()
    from megatron.megalens.paper_style import apply_global_rcparams
    apply_global_rcparams()
    report_file = os.path.join(output_dir, "dp_diagnostic_report.txt")
    logger = _ReportLogger(report_file, "MegaLens Data Parallelism Diagnostic Report")

    logger.log("[DP Analyzer] Loading trace data...")
    loader = TraceDataLoader.from_traces(traces)
    analyzer = DPAnalyzer(loader)

    logger.log(f"[DP Analyzer] Detected {len(loader.get_ranks())} ranks, "
               f"DP ranks: {loader.get_dp_ranks() or 'all'}")

    # 1. Straggler diagnosis
    logger.log("\n[Step 1/6] Running Straggler Diagnosis...")
    straggler_data = analyzer.diagnose_stragglers()
    _write_straggler_report(straggler_data, logger)

    # 2. Step time balance (DP load imbalance)
    logger.log("[Step 2/6] Running Step Time Balance Analysis...")
    balance_data = analyzer.analyze_step_time_balance()
    _write_step_balance_report(balance_data, logger)

    # 3. Gradient sync overhead
    logger.log("[Step 3/6] Running Gradient Sync Overhead Analysis...")
    sync_data = analyzer.analyze_grad_sync_overhead()
    _write_grad_sync_report(sync_data, logger)

    # 4. Memory efficiency
    logger.log("[Step 4/6] Running Memory Efficiency Analysis...")
    mem_data = analyzer.analyze_memory_efficiency()
    _write_memory_report(mem_data, logger)

    # 5. Comm overlap analysis
    logger.log("[Step 5/6] Running Comm/Compute Overlap Analysis...")
    overlap_data = analyzer.analyze_comm_overlap()
    _write_overlap_report(overlap_data, logger)

    # 6. Typed collective lifecycle evidence
    logger.log("[Step 6/6] Reducing Typed Collective Lifecycle Evidence...")
    lifecycle_data = analyzer.analyze_collective_lifecycle()
    _write_lifecycle_report(lifecycle_data, logger)

    # 7. Visualizations
    logger.log("\n[DP Analyzer] Generating visualizations...")
    generate_straggler_plots(straggler_data, output_dir)
    generate_step_balance_plots(balance_data, output_dir)
    generate_grad_sync_plots(sync_data, output_dir)
    generate_overlap_plots(overlap_data, output_dir)

    # 8. CSV export
    if straggler_data:
        pd.DataFrame(straggler_data).drop(
            columns=["per_rank_start_ts", "hw_detail"], errors="ignore"
        ).to_csv(os.path.join(output_dir, "dp_straggler_stats.csv"), index=False)
    if balance_data:
        pd.DataFrame(balance_data).drop(
            columns=["per_rank_step_ms"], errors="ignore"
        ).to_csv(os.path.join(output_dir, "dp_step_balance.csv"), index=False)
    if sync_data:
        pd.DataFrame(sync_data).to_csv(
            os.path.join(output_dir, "dp_grad_sync_stats.csv"), index=False)
    if overlap_data:
        pd.DataFrame(overlap_data).to_csv(
            os.path.join(output_dir, "dp_overlap_stats.csv"), index=False)

    logger.log(f"\n[DP Analyzer] All analysis complete. Report -> {report_file}")
    return {
        "straggler_data": straggler_data,
        "balance_data": balance_data,
        "sync_data": sync_data,
        "mem_data": mem_data,
        "overlap_data": overlap_data,
        "lifecycle_data": lifecycle_data,
    }
