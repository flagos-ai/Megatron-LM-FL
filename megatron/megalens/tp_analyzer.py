# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tensor / Sequence Parallelism Inefficiency Analyzer.

Diagnoses two primary TP/SP performance issues:

1. **Kernel launch overhead** — detects when small TP GEMM kernels are
   CPU-bound because the CUDA launch latency dominates the actual GPU
   execution time.

2. **NVLink saturation** — checks whether TP/SP collective communications
   (``all-gather``, ``reduce-scatter``) achieve a reasonable fraction of
   the hardware's theoretical NVLink peak bandwidth, using the hardware
   counter data collected by :class:`HardwareMonitor`.

All public methods return JSON-serialisable ``list[dict]`` results.
"""

from __future__ import annotations

import collections
import json
import math
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from megatron.megalens.data_loader import SpanEvent, TraceDataLoader
from megatron.megalens.event_catalog import (
    CapabilityStatus,
    EventRole,
    MetricKind,
    get_event_spec,
    is_event_eligible_for_metric,
)
from megatron.megalens.nested_aggregation import (
    MetricResult,
    TPReduceScatterMetrics,
    aggregate_tp_reduce_scatter_partition,
)
from megatron.megalens.paper_style import FIG_H, FIG_H_SHORT, FIG_H_TALL, FIG_W_DOUBLE, FIG_W_SINGLE
from megatron.megalens.utils import get_gpu_p2p_theory_bw_gbps

# ============================================================================
# Helpers
# ============================================================================


_AnalysisPartitionKey = Tuple[int, int, str]
_Timestamp = int | float


def _merge_intervals(
    intervals: List[Tuple[_Timestamp, _Timestamp]]
) -> List[Tuple[_Timestamp, _Timestamp]]:
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


def _total_length(intervals: List[Tuple[_Timestamp, _Timestamp]]) -> _Timestamp:
    return sum(e - s for s, e in intervals)


def _is_finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive_span_interval(event: SpanEvent) -> Optional[Tuple[_Timestamp, _Timestamp]]:
    if not _is_finite_number(event.ts) or not _is_finite_number(event.dur):
        return None
    if event.ts < 0 or event.dur <= 0:
        return None
    end = event.ts + event.dur
    if not _is_finite_number(end) or end <= event.ts:
        return None
    return event.ts, end


def _has_invalid_span_interval(event: SpanEvent) -> bool:
    invalid_value = (
        not _is_finite_number(event.ts)
        or not _is_finite_number(event.dur)
        or event.ts < 0
        or event.dur < 0
    )
    if invalid_value:
        return True
    return event.dur > 0 and _positive_span_interval(event) is None


def _safe_event_iteration(event: SpanEvent) -> Optional[int]:
    if not isinstance(event.args, Mapping):
        return None
    iteration = event.args.get("iteration", -1)
    return iteration if type(iteration) is int else None


def _has_valid_partition_identity(event: SpanEvent) -> bool:
    iteration = _safe_event_iteration(event)
    return type(event.rank) is int and event.rank >= 0 and iteration is not None and iteration >= 0


def _analysis_partition_key(event: SpanEvent) -> _AnalysisPartitionKey:
    rank = event.rank if type(event.rank) is int else -1
    iteration = _safe_event_iteration(event)
    if _has_valid_partition_identity(event):
        assert iteration is not None
        return rank, iteration, ""

    raw_iteration = (
        event.args.get("iteration", -1) if isinstance(event.args, Mapping) else event.args
    )
    marker = (
        f"rank={type(event.rank).__name__}:{event.rank!r};"
        f"iteration={type(raw_iteration).__name__}:{raw_iteration!r}"
    )
    return rank, -1, marker


def _reduce_cataloged_partitions(
    events: List[SpanEvent],
) -> Dict[_AnalysisPartitionKey, TPReduceScatterMetrics]:
    cataloged_by_partition: Dict[_AnalysisPartitionKey, List[SpanEvent]] = collections.defaultdict(
        list
    )
    for event in events:
        if get_event_spec(event.name) is not None and is_event_eligible_for_metric(
            event.name, MetricKind.INTERVAL_UNION
        ):
            cataloged_by_partition[_analysis_partition_key(event)].append(event)

    return {
        key: aggregate_tp_reduce_scatter_partition(partition_events)
        for key, partition_events in cataloged_by_partition.items()
    }


def _select_physical_metric_events(
    events: List[SpanEvent],
    nested_by_partition: Optional[Dict[_AnalysisPartitionKey, TPReduceScatterMetrics]] = None,
) -> List[SpanEvent]:
    """Select validated physical leaves and pass through uncataloged events."""
    if nested_by_partition is None:
        nested_by_partition = _reduce_cataloged_partitions(events)

    approved_cataloged_ids: set[int] = set()
    for event in events:
        spec = get_event_spec(event.name)
        if spec is None or not is_event_eligible_for_metric(event.name, MetricKind.LEAF_SUM):
            continue
        metrics = nested_by_partition[_analysis_partition_key(event)]
        if metrics.leaf_sum_us.status is not CapabilityStatus.AVAILABLE:
            continue
        approved_cataloged_ids.add(id(event))

    return [
        event
        for event in events
        if is_event_eligible_for_metric(event.name, MetricKind.LEAF_SUM, allow_uncataloged=True)
        and (get_event_spec(event.name) is None or id(event) in approved_cataloged_ids)
    ]


def _metric_result_to_dict(result: MetricResult) -> Dict[str, Any]:
    return {"value": result.value, "status": result.status.value, "reason": result.reason}


def _nested_metrics_to_dict(metrics: TPReduceScatterMetrics) -> Dict[str, Dict[str, Any]]:
    return {
        "phase_wall_us": _metric_result_to_dict(metrics.phase_wall_us),
        "leaf_sum_us": _metric_result_to_dict(metrics.leaf_sum_us),
        "interval_union_us": _metric_result_to_dict(metrics.interval_union_us),
        "leaf_data_bytes": _metric_result_to_dict(metrics.leaf_data_bytes),
    }


def _has_complete_comm_ratio(row: Dict[str, Any]) -> bool:
    status = row.get("comm_ratio_status")
    return (
        status in (None, CapabilityStatus.AVAILABLE.value)
        and _is_finite_number(row.get("comm_ratio"))
        and _is_finite_number(row.get("total_compute_us"))
        and _is_finite_number(row.get("total_tp_comm_us"))
    )


def _comm_ratio_status_label(row: Dict[str, Any]) -> str:
    status = row.get("comm_ratio_status")
    if isinstance(status, str) and status:
        return status
    return "legacy" if _has_complete_comm_ratio(row) else CapabilityStatus.UNKNOWN.value


# ============================================================================
# Thresholds
# ============================================================================

_LAUNCH_OVERHEAD_WALL_RATIO: float = 2.0
_LAUNCH_OVERHEAD_MIN_DUR_US: float = 20.0

_NVLINK_LOW_UTIL_RATIO: float = 0.30  # flag if < 30 % of peak


def _load_reporting_dependencies() -> Tuple[Any, Any]:
    """Load optional plotting/reporting dependencies only when required."""
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


# ============================================================================
# Analyzer
# ============================================================================


class TPAnalyzer:
    """Analyses Tensor/Sequence-Parallelism inefficiencies.

    Args:
        loader: A populated :class:`TraceDataLoader`.
        nvlink_theory_peak_gbps: Bi-directional theoretical NVLink peak in
            **GB/s** (e.g. 600 for A100, 900 for H100).  Used as the
            denominator for bandwidth utilisation calculations.
    """

    def __init__(self, loader: TraceDataLoader, nvlink_theory_peak_gbps: float = 300.0) -> None:
        self.loader = loader
        self.nvlink_peak_gbps: float = nvlink_theory_peak_gbps
        self.nvlink_peak_mbs: float = nvlink_theory_peak_gbps * 1024.0

    # ------------------------------------------------------------------
    # 1. Kernel Launch Overhead Detection
    # ------------------------------------------------------------------

    def analyze_kernel_launch_overhead(
        self, gemm_event_names: Optional[List[str]] = None, iteration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Detect CPU-bound TP GEMM kernels whose launch overhead dominates.

        Algorithm:
            For each GEMM event the trace records two durations:
            * **Wall duration** (``duration_wall`` in args, or the outer scope
              duration) — time from CPU launch to GPU completion including
              launch overhead.
            * **CUDA duration** (``duration_cuda`` in args, or ``dur`` on the
              event itself) — actual GPU execution time measured via CUDA
              events.

            If ``wall_dur > 2 × cuda_dur`` **and** ``cuda_dur < 20 μs``, the
            kernel is flagged as *CPU-Bound: Kernel Launch Overhead*.

        Returns:
            List of dicts per flagged event::

                {
                    "rank": int,
                    "iteration": int,
                    "event_name": str,
                    "ts": int,
                    "cuda_dur_us": float,
                    "wall_dur_us": float,
                    "overhead_ratio": float,
                    "diagnosis": str,
                }
        """
        if gemm_event_names is None:
            gemm_event_names = [
                "_reduce",
                "_gather_along_last_dim",
                "_gather_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "_reduce_scatter_along_first_dim",
                "tp-allreduce",
                "tp-all-gather-first",
                "tp-all-gather-last",
                "tp-reduce-scatter",
                "tp-reduce-scatter-last",
            ]

        results: List[Dict[str, Any]] = []

        selected_events: List[SpanEvent] = []
        for gname in gemm_event_names:
            selected_events.extend(self.loader.get_events_by_name(gname, iteration=iteration))

        for ev in _select_physical_metric_events(selected_events):
            cuda_dur = float(ev.dur)
            wall_dur = float(ev.args.get("duration_wall", ev.dur))

            if wall_dur <= 0 or cuda_dur <= 0:
                continue

            ratio = wall_dur / cuda_dur

            if ratio > _LAUNCH_OVERHEAD_WALL_RATIO and cuda_dur < _LAUNCH_OVERHEAD_MIN_DUR_US:
                results.append(
                    {
                        "rank": ev.rank,
                        "iteration": ev.iteration,
                        "event_name": ev.name,
                        "ts": ev.ts,
                        "cuda_dur_us": cuda_dur,
                        "wall_dur_us": wall_dur,
                        "overhead_ratio": round(ratio, 2),
                        "diagnosis": (
                            f"CPU-Bound: Kernel Launch Overhead "
                            f"(wall {wall_dur:.1f} μs vs cuda {cuda_dur:.1f} μs, "
                            f"{ratio:.1f}× ratio)"
                        ),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # 2. NVLink Saturation Analysis
    # ------------------------------------------------------------------

    def analyze_nvlink_saturation(
        self, sp_comm_event_names: Optional[List[str]] = None, iteration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Check NVLink bandwidth utilisation during TP/SP collective comms.

        Algorithm:
            For each TP/SP communication event:
            1. Query the ``NVLink_Tx_MBs`` hardware counter during the
               event's ``[ts, ts + dur]`` window.
            2. Compare the observed peak/mean bandwidth against the
               theoretical peak (constructor parameter).
            3. Flag events where utilisation is below 30 %.

        Also computes a per-rank summary with average utilisation across
        all selected TP/SP comm events.

        Returns:
            List of dicts::

                {
                    "rank": int,
                    "iteration": int,
                    "event_name": str,
                    "ts": int,
                    "dur_us": float,
                    "data_bytes": int | None,
                    "achieved_bw_mbs": float | None,
                    "peak_bw_mbs": float,
                    "utilisation_pct": float | None,
                    "diagnosis": str | None,
                }
        """
        if sp_comm_event_names is None:
            sp_comm_event_names = [
                "_gather_along_first_dim",
                "_gather_along_last_dim",
                "_reduce_scatter_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "tp-allreduce",
                "tp-all-gather-first",
                "tp-all-gather-last",
                "tp-reduce-scatter",
                "tp-reduce-scatter-last",
                "sp-layernorm-allreduce",
                "embedding-grads-allreduce",
            ]

        results: List[Dict[str, Any]] = []
        min_payload_bytes = 1 * 1024 * 1024  # small payloads are latency-dominated

        selected_events: List[SpanEvent] = []
        for cname in sp_comm_event_names:
            selected_events.extend(self.loader.get_events_by_name(cname, iteration=iteration))

        nested_by_partition = _reduce_cataloged_partitions(selected_events)
        for ev in _select_physical_metric_events(selected_events, nested_by_partition):
            if ev.dur <= 0:
                continue

            spec = get_event_spec(ev.name)
            payload_available = spec is None or (
                nested_by_partition[_analysis_partition_key(ev)].leaf_data_bytes.status
                is CapabilityStatus.AVAILABLE
            )
            data_bytes = int(ev.args.get("data_bytes", 0) or 0) if payload_available else None
            group_size = int(ev.args.get("group_size", 1) or 1)

            # Primary method: query NVLink HW counter for this window.
            # NVML sampling is asynchronous (10 ms) while comm events are often
            # sub-ms, so windows may have 0 samples. In that case fall back to
            # tensor-size / duration with topology-aware transferred bytes.
            nvlink_result = self.loader.get_hardware_metrics_in_window(
                ev.ts, ev.end_ts, ev.rank, "NVLink_Tx_MBs"
            )

            achieved_mbs: Optional[float] = None
            utilisation: Optional[float] = None
            diagnosis: Optional[str] = None

            if nvlink_result is not None and nvlink_result.num_samples > 0:
                achieved_mbs = max(nvlink_result.mean, nvlink_result.peak)
            elif payload_available and data_bytes is not None and data_bytes > 0 and ev.dur > 0:
                transferred_bytes = data_bytes
                name = ev.name.lower()
                if "allreduce" in name and group_size > 1:
                    # Ring all-reduce transfers 2*(N-1)/N payload bytes per rank.
                    transferred_bytes = int(data_bytes * (2.0 * (group_size - 1) / group_size))
                elif "all-gather" in name and group_size > 1:
                    transferred_bytes = data_bytes * (group_size - 1)
                elif "reduce-scatter" in name and group_size > 1:
                    transferred_bytes = int(data_bytes * (group_size - 1) / group_size)
                achieved_mbs = (transferred_bytes / (1024.0 * 1024.0)) / (ev.dur / 1e6)

            if achieved_mbs is not None and self.nvlink_peak_mbs > 0:
                utilisation = achieved_mbs / self.nvlink_peak_mbs
                util_pct = utilisation * 100.0

                if not payload_available:
                    diagnosis = "Payload unavailable: skip low-util judgement"
                elif data_bytes is not None and data_bytes < min_payload_bytes:
                    diagnosis = (
                        f"Small payload ({data_bytes / 1024.0:.1f} KiB): "
                        f"latency-dominated; skip low-util judgement"
                    )
                elif payload_available and utilisation < _NVLINK_LOW_UTIL_RATIO:
                    diagnosis = (
                        f"Low NVLink Utilisation: {util_pct:.1f}% of "
                        f"{self.nvlink_peak_mbs:.0f} MB/s peak "
                        f"(achieved {achieved_mbs:.0f} MB/s)"
                    )

            results.append(
                {
                    "rank": ev.rank,
                    "iteration": ev.iteration,
                    "event_name": ev.name,
                    "ts": ev.ts,
                    "dur_us": float(ev.dur),
                    "data_bytes": data_bytes,
                    "group_size": group_size,
                    "achieved_bw_mbs": round(achieved_mbs, 1) if achieved_mbs else None,
                    "peak_bw_mbs": self.nvlink_peak_mbs,
                    "utilisation_pct": (
                        round(utilisation * 100.0, 2) if utilisation is not None else None
                    ),
                    "diagnosis": diagnosis,
                }
            )

        return results

    # ------------------------------------------------------------------
    # 3. TP Communication Overhead Ratio
    # ------------------------------------------------------------------

    def analyze_tp_comm_overhead(
        self,
        tp_comm_names: Optional[List[str]] = None,
        compute_names: Optional[List[str]] = None,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Quantify TP/SP communication with role-aware interval aggregation.

        For each rank and iteration, communication spans are merged into an
        interval union so nested logical composites and physical leaves do not
        double count wall time. Cataloged reduce-scatter spans are validated by
        :func:`aggregate_tp_reduce_scatter_partition`. The ratio is emitted only
        when both communication aggregation and compute evidence are complete::

            comm_ratio = total_tp_comm / (total_compute + total_tp_comm)

        Returns:
            List of dicts, one per (rank, iteration)::

                {
                    "rank": int,
                    "iteration": int,
                    "total_compute_us": float | None,
                    "total_tp_comm_us": float | None,
                    "comm_ratio": float | None,
                    "n_comm_events": int,
                    "aggregation_mode": "interval_union_us",
                    "aggregation_status": str,
                    "aggregation_reason": str,
                    "comm_ratio_status": str,
                    "comm_ratio_reason": str,
                    "tp_reduce_scatter_metrics": dict,
                }
        """
        if tp_comm_names is None:
            tp_comm_names = [
                "allreduce",
                "all-gather",
                "reduce-scatter",
                "_gather_along_first_dim",
                "_gather_along_last_dim",
                "_reduce_scatter_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "_reduce",
                "tp-allreduce",
                "tp-all-gather-first",
                "tp-all-gather-last",
                "tp-reduce-scatter",
                "tp-reduce-scatter-last",
            ]
        if compute_names is None:
            compute_names = [
                "forward-step",
                "backward-step",
                "_forward_attention",
                "_forward_mlp",
                "MLP.forward",
                "attention",
            ]

        ranks = self.loader.get_ranks()

        comm_by_ri: Dict[_AnalysisPartitionKey, List[SpanEvent]] = collections.defaultdict(list)
        comp_by_ri: Dict[_AnalysisPartitionKey, List[Tuple[_Timestamp, _Timestamp]]] = (
            collections.defaultdict(list)
        )
        invalid_compute_identity_partitions: set[_AnalysisPartitionKey] = set()
        invalid_compute_interval_partitions: set[_AnalysisPartitionKey] = set()

        for rank in ranks:
            for cname in tp_comm_names:
                for ev in self.loader.get_events_by_name(cname, rank=rank, iteration=iteration):
                    spec = get_event_spec(ev.name)
                    if not is_event_eligible_for_metric(
                        ev.name, MetricKind.INTERVAL_UNION, allow_uncataloged=True
                    ):
                        continue
                    if (
                        spec is not None
                        or _positive_span_interval(ev) is not None
                        or _has_invalid_span_interval(ev)
                    ):
                        comm_by_ri[_analysis_partition_key(ev)].append(ev)

            for cname in compute_names:
                for ev in self.loader.get_events_by_name(cname, rank=rank, iteration=iteration):
                    key = _analysis_partition_key(ev)
                    interval_value = _positive_span_interval(ev)
                    if interval_value is not None:
                        comp_by_ri[key].append(interval_value)
                    if not _has_valid_partition_identity(ev):
                        invalid_compute_identity_partitions.add(key)
                    if _has_invalid_span_interval(ev):
                        invalid_compute_interval_partitions.add(key)

        results: List[Dict[str, Any]] = []
        all_keys = sorted(
            set(comm_by_ri.keys())
            | set(comp_by_ri.keys())
            | invalid_compute_identity_partitions
            | invalid_compute_interval_partitions
        )
        for key in all_keys:
            rank, it, _ = key
            comm_events = comm_by_ri.get(key, [])
            nested_metrics = aggregate_tp_reduce_scatter_partition(comm_events)
            nested_union = nested_metrics.interval_union_us
            has_cataloged_events = any(
                get_event_spec(event.name) is not None for event in comm_events
            )
            invalid_uncataloged_event = any(
                get_event_spec(event.name) is None and _has_invalid_span_interval(event)
                for event in comm_events
            )
            invalid_partition_identity = any(
                not _has_valid_partition_identity(event) for event in comm_events
            )
            comm_intervals = [
                interval_value
                for event in comm_events
                if (interval_value := _positive_span_interval(event)) is not None
            ]

            if invalid_partition_identity:
                total_comm: Optional[float] = None
                aggregation_status = CapabilityStatus.UNKNOWN
                aggregation_reason = (
                    "one or more TP communication spans lack a valid non-negative "
                    "rank and iteration"
                )
            elif invalid_uncataloged_event:
                total_comm = None
                aggregation_status = CapabilityStatus.UNKNOWN
                aggregation_reason = (
                    "one or more uncataloged TP communication spans have an invalid "
                    "timestamp or duration"
                )
            elif has_cataloged_events and nested_union.status is CapabilityStatus.UNKNOWN:
                total_comm = None
                aggregation_status = CapabilityStatus.UNKNOWN
                aggregation_reason = nested_union.reason
            elif has_cataloged_events and nested_union.status is CapabilityStatus.AVAILABLE:
                total_comm = float(_total_length(_merge_intervals(comm_intervals)))
                aggregation_status = CapabilityStatus.AVAILABLE
                aggregation_reason = (
                    f"{nested_union.reason}; merged all selected TP communication intervals"
                )
            elif has_cataloged_events and nested_union.status is CapabilityStatus.PARTIAL:
                total_comm = None
                aggregation_status = CapabilityStatus.PARTIAL
                aggregation_reason = nested_union.reason
            elif comm_intervals:
                total_comm = float(_total_length(_merge_intervals(comm_intervals)))
                aggregation_status = CapabilityStatus.AVAILABLE
                aggregation_reason = (
                    "merged selected TP communication intervals; "
                    "no cataloged nested spans were observed"
                )
            else:
                total_comm = None
                aggregation_status = CapabilityStatus.UNAVAILABLE
                aggregation_reason = "no positive-duration TP communication spans were observed"

            comp_intervals = comp_by_ri.get(key, [])
            total_comp = (
                None
                if key in invalid_compute_identity_partitions
                or key in invalid_compute_interval_partitions
                else (
                    float(_total_length(_merge_intervals(comp_intervals)))
                    if comp_intervals
                    else None
                )
            )
            if key in invalid_compute_identity_partitions:
                ratio: Optional[float] = None
                ratio_status = CapabilityStatus.UNKNOWN
                ratio_reason = (
                    "one or more compute spans lack a valid non-negative rank and iteration"
                )
            elif key in invalid_compute_interval_partitions:
                ratio = None
                ratio_status = CapabilityStatus.UNKNOWN
                ratio_reason = "one or more compute spans have an invalid timestamp or duration"
            elif aggregation_status is CapabilityStatus.AVAILABLE and total_comp is not None:
                assert total_comm is not None
                denom = total_comp + total_comm
                ratio = total_comm / denom if denom > 0 else None
                ratio_status = CapabilityStatus.AVAILABLE
                ratio_reason = "communication and compute timing evidence are available"
            elif aggregation_status is CapabilityStatus.PARTIAL:
                ratio = None
                ratio_status = CapabilityStatus.PARTIAL
                ratio_reason = (
                    "communication timing is a lower bound; "
                    "a complete overhead ratio is unavailable"
                )
            elif aggregation_status is CapabilityStatus.UNKNOWN:
                ratio = None
                ratio_status = CapabilityStatus.UNKNOWN
                ratio_reason = aggregation_reason
            elif aggregation_status is CapabilityStatus.UNAVAILABLE:
                ratio = None
                ratio_status = CapabilityStatus.UNAVAILABLE
                ratio_reason = aggregation_reason
            else:
                ratio = None
                ratio_status = CapabilityStatus.UNAVAILABLE
                ratio_reason = "no positive-duration compute spans were observed"

            physical_event_count = sum(
                _positive_span_interval(event) is not None
                and (
                    get_event_spec(event.name) is None
                    or (
                        (spec := get_event_spec(event.name)) is not None
                        and spec.role is EventRole.PHYSICAL_LEAF
                        and nested_metrics.leaf_sum_us.status is CapabilityStatus.AVAILABLE
                    )
                )
                for event in comm_events
            )
            results.append(
                {
                    "rank": rank,
                    "iteration": it,
                    "total_compute_us": round(total_comp, 1) if total_comp is not None else None,
                    "total_tp_comm_us": (round(total_comm, 1) if total_comm is not None else None),
                    "comm_ratio": round(ratio, 4) if ratio is not None else None,
                    "n_comm_events": physical_event_count,
                    "aggregation_mode": "interval_union_us",
                    "aggregation_status": aggregation_status.value,
                    "aggregation_reason": aggregation_reason,
                    "comm_ratio_status": ratio_status.value,
                    "comm_ratio_reason": ratio_reason,
                    "tp_reduce_scatter_metrics": _nested_metrics_to_dict(nested_metrics),
                }
            )

        return results

    # ------------------------------------------------------------------
    # 4. GPU SM Efficiency
    # ------------------------------------------------------------------

    def analyze_gpu_sm_efficiency(
        self, compute_names: Optional[List[str]] = None, iteration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Profile GPU SM utilisation from HW counters during compute phases.

        Queries ``SM_Util_pct`` during each compute event to detect
        under-utilisation caused by over-aggressive tensor splitting.

        Returns:
            List of dicts, one per rank::

                {
                    "rank": int,
                    "mean_sm_util_pct": float | None,
                    "min_sm_util_pct": float | None,
                    "max_sm_util_pct": float | None,
                    "n_samples": int,
                    "diagnosis": str | None,
                }
        """
        if compute_names is None:
            compute_names = ["forward-step", "backward-step"]

        ranks = self.loader.get_ranks()
        rank_utils: Dict[int, List[float]] = collections.defaultdict(list)

        for rank in ranks:
            for cname in compute_names:
                for ev in self.loader.get_events_by_name(cname, rank=rank, iteration=iteration):
                    if ev.dur <= 0:
                        continue
                    result = self.loader.get_hardware_metrics_in_window(
                        ev.ts, ev.end_ts, ev.rank, "SM_Util_pct"
                    )
                    if result is not None and result.num_samples > 0:
                        rank_utils[rank].extend(result.values.tolist())

        results: List[Dict[str, Any]] = []
        for rank in sorted(ranks):
            vals = rank_utils.get(rank, [])
            if not vals:
                results.append(
                    {
                        "rank": rank,
                        "mean_sm_util_pct": None,
                        "min_sm_util_pct": None,
                        "max_sm_util_pct": None,
                        "n_samples": 0,
                        "diagnosis": None,
                    }
                )
                continue
            arr = np.array(vals)
            mean_u = float(np.mean(arr))
            min_u = float(np.min(arr))
            diag = None
            if mean_u < 50:
                diag = (
                    f"Low SM Utilisation ({mean_u:.1f}%): TP splitting may be "
                    f"too aggressive — consider reducing TP degree or increasing "
                    f"hidden size."
                )
            results.append(
                {
                    "rank": rank,
                    "mean_sm_util_pct": round(mean_u, 1),
                    "min_sm_util_pct": round(min_u, 1),
                    "max_sm_util_pct": round(float(np.max(arr)), 1),
                    "n_samples": len(vals),
                    "diagnosis": diag,
                }
            )

        return results

    # ------------------------------------------------------------------
    # 5. Compute Fragmentation (many short kernels)
    # ------------------------------------------------------------------

    def analyze_compute_fragmentation(
        self,
        kernel_names: Optional[List[str]] = None,
        short_threshold_us: float = 10.0,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect kernel fragmentation — too many tiny compute kernels.

        When TP/SP splits sequences or tensors too finely, each GEMM or
        collective becomes very short, and CUDA kernel launch overhead
        dominates.  This method counts the fraction of events shorter
        than *short_threshold_us* per rank.

        Returns:
            List of dicts, one per rank::

                {
                    "rank": int,
                    "total_events": int,
                    "short_events": int,
                    "short_ratio": float,
                    "mean_dur_us": float,
                    "median_dur_us": float,
                    "diagnosis": str | None,
                }
        """
        if kernel_names is None:
            kernel_names = [
                "allreduce",
                "all-gather",
                "reduce-scatter",
                "_gather_along_first_dim",
                "_gather_along_last_dim",
                "_reduce_scatter_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "_reduce",
                "_forward_attention",
                "_forward_mlp",
                "attention",
                "MLP.forward",
                "tp-allreduce",
                "tp-all-gather-first",
                "tp-all-gather-last",
                "tp-reduce-scatter",
                "tp-reduce-scatter-last",
            ]

        ranks = self.loader.get_ranks()
        rank_durs: Dict[int, List[float]] = collections.defaultdict(list)

        for rank in ranks:
            selected_events: List[SpanEvent] = []
            for kname in kernel_names:
                selected_events.extend(
                    self.loader.get_events_by_name(kname, rank=rank, iteration=iteration)
                )
            for ev in _select_physical_metric_events(selected_events):
                if ev.dur > 0:
                    rank_durs[rank].append(float(ev.dur))

        results: List[Dict[str, Any]] = []
        for rank in sorted(ranks):
            durs = rank_durs.get(rank, [])
            if not durs:
                continue
            arr = np.array(durs)
            short_count = int(np.sum(arr < short_threshold_us))
            short_ratio = short_count / len(arr)
            diag = None
            if short_ratio > 0.30 and len(arr) > 20:
                diag = (
                    f"High kernel fragmentation: {short_ratio:.0%} of events "
                    f"< {short_threshold_us} μs.  Consider kernel fusion, "
                    f"reducing SP degree, or using CUDA Graphs."
                )
            results.append(
                {
                    "rank": rank,
                    "total_events": len(arr),
                    "short_events": short_count,
                    "short_ratio": round(short_ratio, 4),
                    "mean_dur_us": round(float(np.mean(arr)), 1),
                    "median_dur_us": round(float(np.median(arr)), 1),
                    "diagnosis": diag,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Convenience: per-rank summary
    # ------------------------------------------------------------------

    def summarise_nvlink_utilisation(
        self, sp_comm_event_names: Optional[List[str]] = None, iteration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate NVLink utilisation per rank across all selected TP/SP comm events.

        Returns one dict per rank::

            {
                "rank": int,
                "num_events": int,
                "mean_utilisation_pct": float | None,
                "min_utilisation_pct": float | None,
                "num_flagged_low": int,
            }
        """
        raw = self.analyze_nvlink_saturation(sp_comm_event_names, iteration)

        by_rank: Dict[int, List[Optional[float]]] = collections.defaultdict(list)
        judged_by_rank: Dict[int, List[float]] = collections.defaultdict(list)
        flagged: Dict[int, int] = collections.defaultdict(int)

        for entry in raw:
            rank = entry["rank"]
            by_rank[rank].append(entry["utilisation_pct"])
            diag = entry.get("diagnosis")
            util = entry.get("utilisation_pct")
            skip_judgement = isinstance(diag, str) and (
                diag.startswith("Small payload") or diag.startswith("Payload unavailable")
            )
            if util is not None and not skip_judgement:
                judged_by_rank[rank].append(util)
            if isinstance(diag, str) and diag.startswith("Low NVLink Utilisation"):
                flagged[rank] += 1

        summary: List[Dict[str, Any]] = []
        for rank in sorted(by_rank.keys()):
            vals = judged_by_rank.get(rank, [])
            summary.append(
                {
                    "rank": rank,
                    "num_events": len(by_rank[rank]),
                    "num_events_judged": len(vals),
                    "mean_utilisation_pct": (round(sum(vals) / len(vals), 2) if vals else None),
                    "min_utilisation_pct": round(min(vals), 2) if vals else None,
                    "num_flagged_low": flagged.get(rank, 0),
                }
            )
        return summary

    # ------------------------------------------------------------------
    # 6. TP Group Straggler Diagnosis
    # ------------------------------------------------------------------

    def diagnose_tp_stragglers(
        self, sync_event_names: Optional[List[str]] = None, iteration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Detect the rank that arrives *latest* at a TP AllReduce barrier.

        In TP, every rank in the same TP group must participate in each
        AllReduce / ReduceScatter / AllGather before any can proceed.  The
        rank that starts the collective last is the *straggler* — it was
        the slowest to finish the preceding GEMM or other compute kernel,
        and it blocked all peers until it joined.

        Algorithm:
            1. Collect all TP collective events per (iteration, event_name).
            2. Group events by TP group key ``(dp_rk, pp_rk)``.
            3. Within each group, ``start_ts`` spread = ``max - min``.
               The rank with the *latest* start is the straggler.
            4. Correlate hardware metrics (Temp, Clock) in the window
               ``[iteration_start, sync_event.ts]`` on the straggler rank
               to distinguish hardware vs software root causes.

        Returns:
            One record per ``(iteration, sync_event, tp_group)``::

                {
                    "iteration": int,
                    "sync_event": str,
                    "tp_group": str,          # "DP{d}-PP{p}"
                    "gap_us": float,          # max_start - min_start (μs)
                    "straggler_rank": int,
                    "fastest_rank": int,
                    "per_rank_start_ts": {rank: ts, ...},
                    "likely_cause": str,      # "hardware" | "compute_skew" | "unknown"
                    "hardware_diagnosis": str | None,
                    "hw_detail": dict | None,
                }
        """
        if sync_event_names is None:
            sync_event_names = [
                "allreduce",
                "all-gather",
                "reduce-scatter",
                "tp-allreduce",
                "tp-all-gather-first",
                "tp-all-gather-last",
                "tp-reduce-scatter",
                "tp-reduce-scatter-last",
                "_gather_along_first_dim",
                "_gather_along_last_dim",
                "_reduce_scatter_along_first_dim",
                "_reduce_scatter_along_last_dim",
                "_reduce",
            ]

        # Build TP group mapping: (dp_rk, pp_rk) -> list of global ranks
        ranks = self.loader.get_ranks()
        tp_groups: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        for rank in ranks:
            # Infer dp/pp from any span event on this rank
            for ev in self.loader.get_events_by_name("iteration", rank=rank):
                dp = ev.dp_rank
                pp = ev.pp_rank
                if dp >= 0 and pp >= 0:
                    tp_groups[(dp, pp)].append(rank)
                    break
        # Fallback: single group with all ranks
        if not tp_groups:
            tp_groups[(0, 0)] = list(ranks)

        # Also build iteration start ts per rank for dynamic HW window
        iter_start_by_rank: Dict[int, Dict[int, int]] = collections.defaultdict(dict)
        for rank in ranks:
            for ev in self.loader.get_events_by_name("iteration", rank=rank, iteration=iteration):
                if ev.dur > 0:
                    iter_start_by_rank[rank][ev.iteration] = ev.ts

        # Gather sync events: (iteration, event_name, dp, pp) -> {rank: SpanEvent}
        GroupKey = Tuple[int, str, int, int]
        grouped: Dict[GroupKey, Dict[int, SpanEvent]] = collections.defaultdict(dict)

        for (dp, pp), group_ranks in tp_groups.items():
            if len(group_ranks) < 2:
                continue
            for rank in group_ranks:
                selected_events: List[SpanEvent] = []
                for sname in sync_event_names:
                    selected_events.extend(
                        self.loader.get_events_by_name(sname, rank=rank, iteration=iteration)
                    )
                for ev in _select_physical_metric_events(selected_events):
                    spec = get_event_spec(ev.name)
                    event_name = spec.name if spec is not None else ev.name
                    key: GroupKey = (ev.iteration, event_name, dp, pp)
                    # Keep the earliest start per rank per group
                    if rank not in grouped[key] or ev.ts < grouped[key][rank].ts:
                        grouped[key][rank] = ev

        results: List[Dict[str, Any]] = []
        for (iter_id, ev_name, dp, pp), rank_events in sorted(grouped.items()):
            if len(rank_events) < 2:
                continue

            starts = {r: ev.ts for r, ev in rank_events.items()}
            min_ts = min(starts.values())
            max_ts = max(starts.values())
            gap_us = float(max_ts - min_ts)

            straggler_rank = max(starts, key=starts.get)  # type: ignore[arg-type]
            fastest_rank = min(starts, key=starts.get)  # type: ignore[arg-type]

            # Dynamic HW window: from iteration start to sync event start
            # This covers the full preceding computation phase, not just 2ms
            straggler_ev = rank_events[straggler_rank]
            iter_start = iter_start_by_rank.get(straggler_rank, {}).get(
                iter_id, max(0, straggler_ev.ts - 50_000)  # fallback: 50ms lookback
            )
            hw_diag, hw_detail = self._correlate_hardware_window(
                straggler_rank, iter_start, straggler_ev.ts
            )

            # Determine likely cause
            likely_cause = "unknown"
            if hw_diag:
                likely_cause = "hardware"
            elif gap_us > 200:  # >200μs gap with no HW cause = compute skew
                likely_cause = "compute_skew"

            results.append(
                {
                    "iteration": iter_id,
                    "sync_event": ev_name,
                    "tp_group": f"DP{dp}-PP{pp}",
                    "gap_us": round(gap_us, 1),
                    "straggler_rank": straggler_rank,
                    "fastest_rank": fastest_rank,
                    "per_rank_start_ts": starts,
                    "likely_cause": likely_cause,
                    "hardware_diagnosis": hw_diag,
                    "hw_detail": hw_detail,
                }
            )

        return results

    def _correlate_hardware_window(
        self, rank: int, window_start: int, window_end: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Query HW metrics over an arbitrary ``[window_start, window_end]`` window.

        Used by both :meth:`diagnose_tp_stragglers` (full compute window) and
        other callers that need a precise time range instead of the fixed 2 ms
        look-back.
        """
        _THERMAL_THROTTLE_TEMP_C: float = 80.0
        _CLOCK_DROP_RATIO: float = 0.92

        # Clamp minimum window to 1 ms so NVML 10 ms sampling has a chance
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

        base_mhz: float = 0.0
        if base_clock_result is not None and base_clock_result.mean > 0:
            base_mhz = base_clock_result.mean
            detail["SM_Base_Clock_MHz"] = round(base_mhz, 1)

        is_hot = temp_result is not None and temp_result.peak >= _THERMAL_THROTTLE_TEMP_C
        is_throttled = (
            clock_result is not None
            and base_mhz > 0
            and clock_result.mean < base_mhz * _CLOCK_DROP_RATIO
        )

        if is_hot and is_throttled:
            diagnosis_parts.append(
                f"Thermal Throttling: {temp_result.peak:.0f}\u00b0C, "
                f"Clock {clock_result.mean:.0f}/{base_mhz:.0f} MHz"
            )
        elif is_throttled:
            diagnosis_parts.append(f"Clock Throttling: {clock_result.mean:.0f}/{base_mhz:.0f} MHz")
        elif is_hot:
            diagnosis_parts.append(f"High Temp: {temp_result.peak:.0f}\u00b0C")

        hw_diag = "; ".join(diagnosis_parts) if diagnosis_parts else None
        return hw_diag, detail if detail else None


# ============================================================================
# Report Logger
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


def _pick_style() -> None:
    plt, _ = _load_reporting_dependencies()
    for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if s in plt.style.available:
            plt.style.use(s)
            return
    plt.style.use("default")


# ============================================================================
# Visualization — Kernel Launch Overhead
# ============================================================================


def generate_launch_overhead_plots(overhead_data: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate 4-panel visualisation for CPU-bound kernel launch overhead."""
    if not overhead_data:
        print("[TP Launch Overhead] No overhead events to plot.")
        return

    _pick_style()
    plt, pd = _load_reporting_dependencies()
    df = pd.DataFrame(overhead_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_DOUBLE, FIG_H_TALL))
    fig.suptitle(
        "Tensor Parallelism: Kernel Launch Overhead Analysis", fontsize=22, fontweight="bold"
    )

    # ---- Panel 1: Wall vs CUDA duration scatter (diagonal = ideal) ----
    ax = axes[0, 0]
    for rank in ranks:
        sub = df[df["rank"] == rank]
        ax.scatter(
            sub["cuda_dur_us"],
            sub["wall_dur_us"],
            s=30,
            alpha=0.6,
            color=rank_color[rank],
            edgecolors="none",
            label=f"Rank {rank}",
        )
    lim_max = max(df["wall_dur_us"].max(), df["cuda_dur_us"].max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, alpha=0.5, label="Ideal (1:1)")
    ax.plot(
        [0, lim_max / _LAUNCH_OVERHEAD_WALL_RATIO],
        [0, lim_max],
        "r--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Threshold ({_LAUNCH_OVERHEAD_WALL_RATIO}× ratio)",
    )
    ax.set_xlabel("CUDA Duration (μs)", fontsize=12)
    ax.set_ylabel("Wall Duration (μs)", fontsize=12)
    ax.set_title(
        "Wall vs CUDA Duration\n(Above red line = CPU-bound)", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Panel 2: Overhead ratio histogram ----
    ax = axes[0, 1]
    ax.hist(df["overhead_ratio"], bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(
        x=_LAUNCH_OVERHEAD_WALL_RATIO,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({_LAUNCH_OVERHEAD_WALL_RATIO}×)",
    )
    ax.set_xlabel("Overhead Ratio (Wall / CUDA)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of Overhead Ratios\n(Right-skewed = many CPU-bound kernels)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize="small")

    # ---- Panel 3: Flagged events per rank (bar) ----
    ax = axes[1, 0]
    counts = df.groupby("rank").size().reindex(ranks, fill_value=0)
    bar_colors = [rank_color[r] for r in counts.index]
    bars = ax.bar(
        [str(r) for r in counts.index],
        counts.values,
        color=bar_colors,
        edgecolor="black",
        alpha=0.85,
    )
    if len(counts) > 0 and counts.max() > 0:
        worst_idx = int(np.argmax(counts.values))
        bars[worst_idx].set_edgecolor("red")
        bars[worst_idx].set_linewidth(3)
        bars[worst_idx].set_hatch("//")
    ax.set_ylabel("# Flagged Events", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.set_title("Flagged CPU-Bound Kernels per Rank", fontsize=14, fontweight="bold")

    # ---- Panel 4: Flagged events per kernel name (horizontal bar) ----
    ax = axes[1, 1]
    name_counts = df["event_name"].value_counts()
    y_pos = np.arange(len(name_counts))
    ax.barh(y_pos, name_counts.values, color="coral", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(name_counts.index, fontsize=10)
    ax.set_xlabel("# Flagged Events", fontsize=12)
    ax.set_title(
        "Flagged Events by Kernel Name\n(Most frequent = optimization target)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "tp_kernel_launch_overhead.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[TP Launch Overhead] Plot saved to {path}")


# ============================================================================
# Visualization — NVLink Saturation
# ============================================================================


def generate_nvlink_plots(
    nvlink_data: List[Dict[str, Any]],
    nvlink_summary: List[Dict[str, Any]],
    nvlink_peak_mbs: float,
    output_dir: str,
) -> None:
    """Generate 4-panel visualisation for NVLink bandwidth utilisation."""
    if not nvlink_data:
        print("[TP NVLink] No NVLink data to plot.")
        return

    _pick_style()
    plt, pd = _load_reporting_dependencies()
    df = pd.DataFrame(nvlink_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_DOUBLE, FIG_H_TALL))
    fig.suptitle(
        "Tensor / Sequence Parallelism: NVLink Saturation Analysis", fontsize=22, fontweight="bold"
    )

    # ---- Panel 1: Achieved BW over time (per rank) ----
    ax = axes[0, 0]
    valid = df[df["achieved_bw_mbs"].notna()].copy()
    if not valid.empty:
        valid["ts_rel"] = valid["ts"] - valid["ts"].min()
        for rank in ranks:
            sub = valid[valid["rank"] == rank].sort_values("ts_rel")
            if sub.empty:
                continue
            ax.plot(
                sub["ts_rel"],
                sub["achieved_bw_mbs"] / 1024.0,
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.7,
                label=f"Rank {rank}",
                color=rank_color[rank],
            )
        ax.axhline(
            y=nvlink_peak_mbs / 1024.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Theoretical Peak ({nvlink_peak_mbs / 1024.0:.0f} GB/s)",
        )
    ax.set_title("Achieved NVLink BW Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_xlabel("Timestamp (μs offset)", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Panel 2: BW distribution boxplot per rank ----
    ax = axes[0, 1]
    box_data = []
    box_labels = []
    for rank in ranks:
        vals = valid[valid["rank"] == rank]["achieved_bw_mbs"].dropna().values / 1024.0
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(str(rank))
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True)
        ax.set_xticks(range(1, len(box_labels) + 1))
        ax.set_xticklabels(box_labels)
        for patch, color in zip(bp["boxes"], [rank_color[int(l)] for l in box_labels]):
            patch.set_facecolor(color)
        ax.axhline(y=nvlink_peak_mbs / 1024.0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_title(
        "BW Distribution per Rank\n(Low median = sustained bottleneck)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)

    # ---- Panel 3: Per-rank avg utilisation bar chart ----
    ax = axes[1, 0]
    if nvlink_summary:
        sdf = pd.DataFrame(nvlink_summary)
        x = np.arange(len(sdf))
        vals = sdf["mean_utilisation_pct"].fillna(0).values
        bar_cols = [
            "salmon" if v < _NVLINK_LOW_UTIL_RATIO * 100 else "mediumseagreen" for v in vals
        ]
        ax.bar(x, vals, color=bar_cols, edgecolor="black", alpha=0.85)
        ax.axhline(
            y=_NVLINK_LOW_UTIL_RATIO * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Low Threshold ({_NVLINK_LOW_UTIL_RATIO * 100:.0f}%)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in sdf["rank"]])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Utilisation (%)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.set_title(
        "Avg NVLink Utilisation per Rank\n(Red bar = below threshold)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize="small")

    # ---- Panel 4: Utilisation per event name (grouped bar) ----
    ax = axes[1, 1]
    has_util = df[df["utilisation_pct"].notna()]
    if not has_util.empty:
        event_avg = has_util.groupby("event_name")["utilisation_pct"].agg(["mean", "min", "count"])
        event_avg = event_avg.sort_values("mean")
        y_pos = np.arange(len(event_avg))
        ax.barh(
            y_pos,
            event_avg["mean"].values,
            color="steelblue",
            edgecolor="black",
            alpha=0.85,
            label="Mean Util %",
        )
        ax.barh(
            y_pos,
            event_avg["min"].values,
            color="salmon",
            edgecolor="black",
            alpha=0.5,
            height=0.4,
            label="Min Util %",
        )
        ax.axvline(
            x=_NVLINK_LOW_UTIL_RATIO * 100, color="red", linestyle="--", linewidth=2, alpha=0.7
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(event_avg.index, fontsize=10)
        ax.legend(fontsize="small")
    ax.set_xlabel("Utilisation (%)", fontsize=12)
    ax.set_title(
        "NVLink Utilisation by Collective Type\n(Left of red = bottleneck collective)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "tp_nvlink_saturation.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[TP NVLink] Plot saved to {path}")


# ============================================================================
# Visualization — TP Comm Overhead
# ============================================================================


def generate_tp_comm_overhead_plots(overhead_data: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate 2-panel plot for TP communication overhead ratio."""
    if not overhead_data:
        return

    valid_data = [row for row in overhead_data if _has_complete_comm_ratio(row)]
    if not valid_data:
        print("[TP Comm Overhead] No complete communication-ratio evidence to plot.")
        return

    _pick_style()
    plt, pd = _load_reporting_dependencies()
    df = pd.DataFrame(valid_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Tensor Parallelism: Communication Overhead Ratio", fontsize=22, fontweight="bold")

    # Panel 1: Comm ratio over iterations per rank
    ax = axes[0]
    for rank in ranks:
        sub = df[df["rank"] == rank].sort_values("iteration")
        ax.plot(
            sub["iteration"],
            sub["comm_ratio"] * 100,
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.7,
            label=f"Rank {rank}",
            color=rank_color[rank],
        )
    ax.axhline(y=30, color="orange", linestyle="--", linewidth=2, label="Warning (30%)")
    ax.axhline(y=50, color="red", linestyle="--", linewidth=2, label="Critical (50%)")
    ax.set_title(
        "TP Comm Ratio per Iteration\n(High = communication-bound)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Comm / (Compute + Comm) (%)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Average breakdown per rank (stacked bar)
    ax = axes[1]
    avg = (
        df.groupby("rank")
        .agg(
            comp_ms=("total_compute_us", lambda s: s.mean() / 1e3),
            comm_ms=("total_tp_comm_us", lambda s: s.mean() / 1e3),
        )
        .reindex(ranks)
    )
    x = np.arange(len(ranks))
    ax.bar(x, avg["comp_ms"], 0.55, label="Compute", color="mediumseagreen", edgecolor="black")
    ax.bar(
        x,
        avg["comm_ms"],
        0.55,
        bottom=avg["comp_ms"],
        label="TP Comm",
        color="steelblue",
        hatch="//",
        edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_title(
        "Avg Step Time: Compute vs TP Comm\n(Blue = TP overhead)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.legend(fontsize="small", frameon=True)
    for i, (c, m) in enumerate(zip(avg["comp_ms"], avg["comm_ms"])):
        total = c + m
        if total > 0:
            ax.text(
                i,
                total + total * 0.01,
                f"{m / total * 100:.0f}%",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="steelblue",
            )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "tp_comm_overhead.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[TP Comm Overhead] Plot saved to {path}")


# ============================================================================
# Visualization — SM Efficiency + Fragmentation
# ============================================================================


def generate_sm_and_frag_plots(
    sm_data: List[Dict[str, Any]], frag_data: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate 2-panel plot for SM utilisation and kernel fragmentation."""
    if not sm_data and not frag_data:
        return

    _pick_style()
    plt, pd = _load_reporting_dependencies()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Tensor Parallelism: GPU Efficiency Analysis", fontsize=22, fontweight="bold")

    # Panel 1: SM Utilisation per rank
    ax = axes[0]
    if sm_data:
        sdf = pd.DataFrame(sm_data)
        ranks = sorted(sdf["rank"].unique())
        vals = sdf.set_index("rank").reindex(ranks)["mean_sm_util_pct"].fillna(0).values
        bar_colors = ["salmon" if v < 50 else "mediumseagreen" for v in vals]
        ax.bar([str(r) for r in ranks], vals, color=bar_colors, edgecolor="black", alpha=0.85)
        ax.axhline(
            y=50, color="orange", linestyle="--", linewidth=2, label="Low Util Threshold (50%)"
        )
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.legend(fontsize="small")
    ax.set_title(
        "Mean SM Utilisation During Compute\n(Low = TP splitting too fine)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("SM Utilisation (%)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.set_ylim(0, 110)

    # Panel 2: Kernel fragmentation per rank
    ax = axes[1]
    if frag_data:
        fdf = pd.DataFrame(frag_data)
        ranks = sorted(fdf["rank"].unique())
        short_ratios = fdf.set_index("rank").reindex(ranks)["short_ratio"].fillna(0).values * 100
        bar_colors = ["salmon" if v > 30 else "steelblue" for v in short_ratios]
        ax.bar(
            [str(r) for r in ranks], short_ratios, color=bar_colors, edgecolor="black", alpha=0.85
        )
        ax.axhline(
            y=30, color="red", linestyle="--", linewidth=2, label="Fragmentation Threshold (30%)"
        )
        for i, v in enumerate(short_ratios):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.legend(fontsize="small")
    ax.set_title(
        "Short Kernel Ratio (< 10 μs)\n(High = excessive fragmentation)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Short Kernels (%)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    ax.set_ylim(0, max(110, ax.get_ylim()[1]))

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "tp_gpu_efficiency.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[TP Efficiency] Plot saved to {path}")


# ============================================================================
# Report Generation
# ============================================================================


def _write_tp_comm_overhead_report(
    overhead_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[TP Communication Overhead Report]")
    logger.log("=" * 90)

    if not overhead_data:
        logger.log("  No TP communication data found.")
        logger.log("=" * 90 + "\n")
        return

    status_counts = collections.Counter(_comm_ratio_status_label(row) for row in overhead_data)
    logger.log(
        "  Evidence status: "
        + ", ".join(f"{key}={status_counts[key]}" for key in sorted(status_counts))
    )
    valid_rows = [row for row in overhead_data if _has_complete_comm_ratio(row)]
    incomplete_rows = [row for row in overhead_data if not _has_complete_comm_ratio(row)]
    if incomplete_rows:
        logger.log("  Incomplete partitions:")
        for row in incomplete_rows:
            logger.log(
                f"    rank={row['rank']} iteration={row['iteration']} "
                f"status={_comm_ratio_status_label(row)}: "
                f"{row.get('comm_ratio_reason', 'incomplete legacy evidence')}"
            )

    if not valid_rows:
        logger.log("  No complete TP communication-ratio evidence; KPI judgement skipped.")
        logger.log("=" * 90 + "\n")
        return

    _, pd = _load_reporting_dependencies()
    valid = pd.DataFrame(valid_rows)
    avg = valid.groupby("rank").agg(
        mean_comp_ms=("total_compute_us", lambda s: round(s.mean() / 1e3, 2)),
        mean_comm_ms=("total_tp_comm_us", lambda s: round(s.mean() / 1e3, 2)),
        mean_ratio=("comm_ratio", "mean"),
        n_comm=("n_comm_events", "mean"),
    )
    avg["mean_ratio_pct"] = avg["mean_ratio"].map(lambda x: f"{x:.1%}")
    logger.log(avg.to_string(float_format="%.2f"))
    logger.log("")

    overall_ratio = valid["comm_ratio"].mean()
    logger.log(f"  Overall TP comm overhead: {overall_ratio:.1%} of step time")
    if overall_ratio > 0.30:
        logger.log("  [WARNING] TP communication exceeds 30% of step time.")
        logger.log(
            "            Consider reducing TP degree, using SP overlap "
            "strategies, or ensuring TP stays within NVLink domain."
        )
    elif overall_ratio > 0.15:
        logger.log(
            "  [INFO] TP communication is moderate. Consider overlapping "
            "communication with computation if not already done."
        )
    else:
        logger.log("  TP communication overhead is healthy.")

    logger.log("=" * 90 + "\n")


def _write_sm_efficiency_report(sm_data: List[Dict[str, Any]], logger: _ReportLogger) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[TP GPU SM Efficiency Report]")
    logger.log("=" * 90)

    if not sm_data:
        logger.log("  No SM utilisation data from HW monitor.")
        logger.log("=" * 90 + "\n")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(sm_data)
    has_data = df["mean_sm_util_pct"].notna().any()
    if not has_data:
        logger.log("  HW monitor did not record SM_Util_pct. Skipping.")
        logger.log("=" * 90 + "\n")
        return

    logger.log(
        df[
            ["rank", "mean_sm_util_pct", "min_sm_util_pct", "max_sm_util_pct", "n_samples"]
        ].to_string(index=False, float_format="%.1f", na_rep="N/A")
    )

    low_ranks = df[(df["mean_sm_util_pct"].notna()) & (df["mean_sm_util_pct"] < 50)]
    if not low_ranks.empty:
        logger.log(
            f"\n  [WARNING] {len(low_ranks)} ranks have mean SM utilisation "
            f"below 50%.  The GPU is under-utilised during compute phases."
        )
        logger.log(
            "            Possible causes: TP degree too high for the hidden "
            "size, or too many small GEMM operations."
        )
    else:
        logger.log("\n  SM utilisation is healthy across all ranks.")

    logger.log("=" * 90 + "\n")


def _write_fragmentation_report(frag_data: List[Dict[str, Any]], logger: _ReportLogger) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[TP/SP Kernel Fragmentation Report]")
    logger.log("=" * 90)

    if not frag_data:
        logger.log("  No fragmentation data available.")
        logger.log("=" * 90 + "\n")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(frag_data)
    logger.log(
        df.drop(columns=["diagnosis"], errors="ignore").to_string(index=False, float_format="%.2f")
    )

    flagged = df[df["diagnosis"].notna()]
    if not flagged.empty:
        logger.log(f"\n  [WARNING] {len(flagged)} ranks have high kernel fragmentation:")
        for _, row in flagged.iterrows():
            logger.log(
                f"    Rank {row['rank']}: {row['short_events']}/{row['total_events']} "
                f"events < 10 μs ({row['short_ratio']:.0%}), "
                f"median {row['median_dur_us']:.1f} μs"
            )
        logger.log("  Action: Consider kernel fusion, CUDA Graphs, or reducing " "SP/CP degree.")
    else:
        logger.log("\n  Kernel durations are healthy (no excessive fragmentation).")

    logger.log("=" * 90 + "\n")


def _write_launch_overhead_report(
    overhead_data: List[Dict[str, Any]], logger: _ReportLogger
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[TP Kernel Launch Overhead Report]")
    logger.log("=" * 90)

    if not overhead_data:
        logger.log("  No CPU-bound kernels detected (all kernels are GPU-bound).")
        logger.log("=" * 90 + "\n")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(overhead_data)

    # Summary statistics
    logger.log(f"  Total flagged events: {len(df)}")
    logger.log(f"  Affected ranks: {sorted(df['rank'].unique().tolist())}")
    logger.log(f"  Mean overhead ratio: {df['overhead_ratio'].mean():.2f}×")
    logger.log(f"  Max overhead ratio:  {df['overhead_ratio'].max():.2f}×\n")

    # Per-kernel-name breakdown
    logger.log("-" * 90)
    logger.log("[Breakdown by Kernel Name]")
    name_stats = (
        df.groupby("event_name")
        .agg(
            count=("overhead_ratio", "size"),
            mean_ratio=("overhead_ratio", "mean"),
            mean_cuda_us=("cuda_dur_us", "mean"),
            mean_wall_us=("wall_dur_us", "mean"),
        )
        .sort_values("count", ascending=False)
    )
    logger.log(name_stats.to_string(float_format="%.2f"))

    # Per-rank breakdown
    logger.log("\n" + "-" * 90)
    logger.log("[Breakdown by Rank]")
    rank_stats = (
        df.groupby("rank")
        .agg(
            count=("overhead_ratio", "size"),
            mean_ratio=("overhead_ratio", "mean"),
            max_ratio=("overhead_ratio", "max"),
        )
        .sort_values("count", ascending=False)
    )
    logger.log(rank_stats.to_string(float_format="%.2f"))

    # Worst offenders
    logger.log("\n" + "-" * 90)
    logger.log("[Top 10 Worst Offenders]")
    worst = df.nlargest(10, "overhead_ratio")[
        ["rank", "iteration", "event_name", "cuda_dur_us", "wall_dur_us", "overhead_ratio"]
    ]
    logger.log(worst.to_string(index=False, float_format="%.2f"))

    # Actionable advice
    logger.log("\n" + "-" * 90)
    total_wasted_us = (df["wall_dur_us"] - df["cuda_dur_us"]).sum()
    logger.log(
        f"[Impact] Total CPU launch overhead: {total_wasted_us:.0f} μs "
        f"({total_wasted_us / 1e3:.2f} ms) across all flagged events."
    )
    logger.log(
        "  Action: Consider kernel fusion, CUDA Graphs, or increasing TP "
        "hidden-size to amortise launch cost."
    )
    logger.log("=" * 90 + "\n")


def _write_nvlink_report(
    nvlink_data: List[Dict[str, Any]],
    nvlink_summary: List[Dict[str, Any]],
    nvlink_peak_mbs: float,
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log("[TP/SP NVLink Saturation Report]")
    logger.log("=" * 90)

    if not nvlink_data:
        logger.log("  No TP/SP communication events found.")
        logger.log("=" * 90 + "\n")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(nvlink_data)
    peak_gbps = nvlink_peak_mbs / 1024.0

    logger.log(f"  Theoretical NVLink peak: {peak_gbps:.0f} GB/s ({nvlink_peak_mbs:.0f} MB/s)")
    logger.log(f"  Total TP/SP comm events analysed: {len(df)}")

    flagged = df[df["diagnosis"].fillna("").str.startswith("Low NVLink Utilisation")]
    logger.log(
        f"  Events flagged as low utilisation (<{_NVLINK_LOW_UTIL_RATIO * 100:.0f}%): "
        f"{len(flagged)} ({len(flagged) / len(df) * 100:.1f}%)"
    )
    skipped_small = df[df["diagnosis"].fillna("").str.startswith("Small payload")]
    if not skipped_small.empty:
        logger.log(f"  Small-payload events skipped from low-util judgement: {len(skipped_small)}")
    logger.log("")

    # Per-rank summary
    if nvlink_summary:
        logger.log("-" * 90)
        logger.log("[Per-Rank NVLink Summary]")
        sdf = pd.DataFrame(nvlink_summary)
        logger.log(sdf.to_string(index=False, float_format="%.2f"))

    # Per-event-name summary
    has_util = df[df["utilisation_pct"].notna()]
    if not has_util.empty:
        logger.log("\n" + "-" * 90)
        logger.log("[Per-Collective Summary]")
        event_stats = (
            has_util.groupby("event_name")
            .agg(
                count=("utilisation_pct", "size"),
                mean_util_pct=("utilisation_pct", "mean"),
                min_util_pct=("utilisation_pct", "min"),
                mean_bw_mbs=("achieved_bw_mbs", "mean"),
            )
            .sort_values("mean_util_pct")
        )
        logger.log(event_stats.to_string(float_format="%.2f"))

    # Worst events
    if not flagged.empty:
        logger.log("\n" + "-" * 90)
        logger.log("[Top 10 Lowest-Utilisation Events]")
        worst = flagged.nsmallest(10, "utilisation_pct")[
            [
                "rank",
                "iteration",
                "event_name",
                "dur_us",
                "achieved_bw_mbs",
                "utilisation_pct",
                "diagnosis",
            ]
        ]
        logger.log(worst.to_string(index=False, float_format="%.2f"))

    # Actionable advice
    logger.log("\n" + "-" * 90)
    for _, row in pd.DataFrame(nvlink_summary).iterrows():
        mean_u = row.get("mean_utilisation_pct")
        if mean_u is not None and mean_u < _NVLINK_LOW_UTIL_RATIO * 100:
            logger.log(
                f"  [WARNING] Rank {row['rank']}: Mean NVLink util = {mean_u:.1f}% "
                f"(below {_NVLINK_LOW_UTIL_RATIO * 100:.0f}% threshold)."
            )
            logger.log(
                f"            Possible causes: small message size, NVLink congestion, "
                f"or misconfigured TP/SP topology."
            )
    logger.log("=" * 90 + "\n")


def _write_tp_straggler_report(
    straggler_data: List[Dict[str, Any]], logger: "_ReportLogger"
) -> None:
    """Write TP group straggler diagnosis report."""
    logger.log("\n" + "=" * 90)
    logger.log("[TP Group Straggler Diagnosis Report]")
    logger.log("=" * 90)

    if not straggler_data:
        logger.log("  No TP sync barriers found with multiple ranks — cannot detect stragglers.")
        logger.log("=" * 90 + "\n")
        return

    _, pd = _load_reporting_dependencies()
    df = pd.DataFrame(straggler_data)
    tp_groups = sorted(df["tp_group"].unique())
    logger.log(f"  Analysed {len(df)} sync barrier events across {len(tp_groups)} TP group(s).\n")

    # Per-event summary
    summary_rows = []
    for _, row in df.iterrows():
        summary_rows.append(
            {
                "Iter": row["iteration"],
                "Sync Event": row["sync_event"],
                "TP Group": row["tp_group"],
                "Gap (μs)": f"{row['gap_us']:.1f}",
                "Straggler": f"Rank {row['straggler_rank']}",
                "Cause": row["likely_cause"],
                "HW Diag": row["hardware_diagnosis"] or "—",
            }
        )
    if summary_rows:
        logger.log(pd.DataFrame(summary_rows).to_string(index=False))

    # Straggler frequency per rank
    logger.log("\n" + "-" * 90)
    logger.log("[Straggler Frequency per Rank]")
    freq = df["straggler_rank"].value_counts()
    for rank, count in freq.items():
        pct = count / len(df) * 100
        marker = " *** FREQUENT TP STRAGGLER ***" if pct > 50 else ""
        logger.log(f"  Rank {rank}: {count}/{len(df)} ({pct:.1f}%){marker}")

    # Hardware-correlated stragglers
    hw_flagged = df[df["hardware_diagnosis"].notna()]
    if not hw_flagged.empty:
        logger.log("\n" + "-" * 90)
        logger.log("[Hardware-Correlated TP Stragglers]")
        for _, row in hw_flagged.iterrows():
            logger.log(
                f"  Iter {row['iteration']} | {row['sync_event']} | "
                f"TP group {row['tp_group']} | "
                f"Rank {row['straggler_rank']} | {row['hardware_diagnosis']}"
            )
        logger.log(
            "  [RECOMMENDATION] Hardware-induced TP stragglers block the entire "
            "TP group at every AllReduce. Check thermal / clock health on the "
            "flagged ranks."
        )
    else:
        logger.log("\n  No hardware-correlated TP stragglers detected.")

    # Compute-skew stragglers
    skew_flagged = df[df["likely_cause"] == "compute_skew"]
    if not skew_flagged.empty:
        logger.log("\n" + "-" * 90)
        logger.log("[Compute-Skew TP Stragglers (no HW cause)]")
        skew_freq = skew_flagged["straggler_rank"].value_counts()
        for rank, count in skew_freq.head(5).items():
            pct = count / len(skew_flagged) * 100
            logger.log(f"  Rank {rank}: {count} events ({pct:.1f}% of skew events)")
        logger.log(
            "  [RECOMMENDATION] Non-hardware TP straggler likely caused by "
            "uneven kernel execution (e.g. data skew, variable sequence length, "
            "or non-uniform layer assignment across TP ranks)."
        )

    logger.log("=" * 90 + "\n")


# ============================================================================
# Master Orchestrator
# ============================================================================


def analyze_tp_traces(
    traces: List[Dict[str, Any]],
    nvlink_theory_peak_gbps: Optional[float] = None,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Run all TP/SP analyses, generate report + plots.

    Args:
        traces: Aggregated Chrome Trace event list (post-``transform``).
        nvlink_theory_peak_gbps: NVLink theoretical peak in GB/s.
        output_dir: Directory for PDF plots and TXT report.

    Returns:
        Dict with ``"launch_overhead_data"``, ``"nvlink_data"``,
        ``"nvlink_summary"`` keys.
    """
    _, pd = _load_reporting_dependencies()
    os.makedirs(output_dir, exist_ok=True)
    from megatron.megalens.paper_style import apply_global_rcparams

    apply_global_rcparams()
    report_file = os.path.join(output_dir, "tp_diagnostic_report.txt")
    logger = _ReportLogger(report_file, "MegaLens Tensor / Sequence Parallelism Diagnostic Report")

    logger.log("[TP Analyzer] Loading trace data...")
    loader = TraceDataLoader.from_traces(traces)

    if nvlink_theory_peak_gbps is None:
        nvlink_theory_peak_gbps = get_gpu_p2p_theory_bw_gbps()
        logger.log(
            f"[TP Analyzer] Auto-detected NVLink/P2P peak (unidirectional): "
            f"{nvlink_theory_peak_gbps} GB/s"
        )
    else:
        logger.log(
            f"[TP Analyzer] Using user-provided NVLink/P2P peak "
            f"(unidirectional): {nvlink_theory_peak_gbps} GB/s"
        )

    analyzer = TPAnalyzer(loader, nvlink_theory_peak_gbps)

    logger.log(f"[TP Analyzer] Detected {len(loader.get_ranks())} ranks")

    # 1. TP Communication overhead ratio
    logger.log("\n[Step 1/5] Running TP Communication Overhead Analysis...")
    tp_comm_data = analyzer.analyze_tp_comm_overhead()
    _write_tp_comm_overhead_report(tp_comm_data, logger)

    # 2. Kernel launch overhead
    logger.log("[Step 2/5] Running Kernel Launch Overhead Detection...")
    overhead_data = analyzer.analyze_kernel_launch_overhead()
    _write_launch_overhead_report(overhead_data, logger)

    # 3. NVLink saturation
    logger.log("[Step 3/5] Running NVLink Saturation Analysis...")
    nvlink_data = analyzer.analyze_nvlink_saturation()
    nvlink_summary = analyzer.summarise_nvlink_utilisation()
    _write_nvlink_report(nvlink_data, nvlink_summary, analyzer.nvlink_peak_mbs, logger)

    # 4. SM efficiency
    logger.log("[Step 4/5] Running GPU SM Efficiency Analysis...")
    sm_data = analyzer.analyze_gpu_sm_efficiency()
    _write_sm_efficiency_report(sm_data, logger)

    # 5. Compute fragmentation
    logger.log("[Step 5/5] Running Kernel Fragmentation Analysis...")
    frag_data = analyzer.analyze_compute_fragmentation()
    _write_fragmentation_report(frag_data, logger)

    # 6. TP Group Straggler Diagnosis
    logger.log("[Step 6/6] Running TP Group Straggler Diagnosis...")
    straggler_data = analyzer.diagnose_tp_stragglers()
    _write_tp_straggler_report(straggler_data, logger)

    # 6. Visualizations
    logger.log("\n[TP Analyzer] Generating visualizations...")
    generate_tp_comm_overhead_plots(tp_comm_data, output_dir)
    generate_launch_overhead_plots(overhead_data, output_dir)
    generate_nvlink_plots(nvlink_data, nvlink_summary, analyzer.nvlink_peak_mbs, output_dir)
    generate_sm_and_frag_plots(sm_data, frag_data, output_dir)

    # 7. CSV export
    if tp_comm_data:
        tp_comm_frame = pd.DataFrame(tp_comm_data)
        tp_comm_frame["tp_reduce_scatter_metrics"] = tp_comm_frame["tp_reduce_scatter_metrics"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
        tp_comm_frame.to_csv(os.path.join(output_dir, "tp_comm_overhead_stats.csv"), index=False)
    if overhead_data:
        pd.DataFrame(overhead_data).to_csv(
            os.path.join(output_dir, "tp_launch_overhead_stats.csv"), index=False
        )
    if nvlink_data:
        pd.DataFrame(nvlink_data).to_csv(
            os.path.join(output_dir, "tp_nvlink_stats.csv"), index=False
        )
    if nvlink_summary:
        pd.DataFrame(nvlink_summary).to_csv(
            os.path.join(output_dir, "tp_nvlink_summary.csv"), index=False
        )
    if frag_data:
        pd.DataFrame(frag_data).to_csv(
            os.path.join(output_dir, "tp_fragmentation_stats.csv"), index=False
        )

    if straggler_data:
        pd.DataFrame(straggler_data).drop(
            columns=["per_rank_start_ts", "hw_detail"], errors="ignore"
        ).to_csv(os.path.join(output_dir, "tp_straggler_stats.csv"), index=False)

    logger.log(f"\n[TP Analyzer] All analysis complete. Report -> {report_file}")
    return {
        "tp_comm_data": tp_comm_data,
        "launch_overhead_data": overhead_data,
        "nvlink_data": nvlink_data,
        "nvlink_summary": nvlink_summary,
        "sm_data": sm_data,
        "frag_data": frag_data,
        "straggler_data": straggler_data,
    }
