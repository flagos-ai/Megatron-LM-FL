"""Hybrid Parallelism Collaborative Analyzer.

Diagnoses cross-dimensional performance inefficiencies in mixed-parallel
(PP + DP + TP + EP) large model training.
"""

from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from megatron.megalens.data_loader import TraceDataLoader
from megatron.megalens.utils import infer_parallel_sizes_from_loader
from megatron.megalens.paper_style import (
    FIG_W_SINGLE, FIG_W_DOUBLE, FIG_H, FIG_H_TALL, FIG_H_SHORT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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


def _overlap_length(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    overlap = 0
    j = 0
    for a_s, a_e in a:
        while j < len(b) and b[j][1] <= a_s:
            j += 1
        k = j
        while k < len(b) and b[k][0] < a_e:
            o_s = max(a_s, b[k][0])
            o_e = min(a_e, b[k][1])
            if o_s < o_e:
                overlap += o_e - o_s
            k += 1
    return overlap


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return 0.0
    xa, ya = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    if xa.std() < 1e-12 or ya.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _pick_style() -> None:
    for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if s in plt.style.available:
            plt.style.use(s)
            return
    plt.style.use("default")


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_EP_PP_CORR_WARN: float = 0.50
_EP_PP_CORR_CRIT: float = 0.70
_EP_TP_CONTENTION_WARN: float = 0.20
_EP_TP_CONTENTION_CRIT: float = 0.40
_EP_DP_CORR_WARN: float = 0.40
_EP_DP_CORR_CRIT: float = 0.65
_PP_DP_UTIL_WARN: float = 0.30
_PP_DP_UTIL_CRIT: float = 0.10
_TP_EP_FRAG_CORR_WARN: float = 0.40
_TP_EP_FRAG_CORR_CRIT: float = 0.60
_GLOBAL_STRAGGLER_WARN: float = 0.30
_GLOBAL_STRAGGLER_CRIT: float = 0.50


# ===========================================================================
# HybridAnalyzer
# ===========================================================================

class HybridAnalyzer:
    """Cross-dimension collaborative analyzer.

    Args:
        loader: A populated :class:`TraceDataLoader`.
        pp_result: Output of ``analyze_pp_traces`` (list of trace events or bubble stats dict).
        dp_result: Output of ``analyze_dp_traces``.
        tp_result: Output of ``analyze_tp_traces``.
        ep_result: Output of ``analyze_ep_traces``.
    """

    def __init__(
        self,
        loader: TraceDataLoader,
        pp_result: Optional[Any] = None,
        dp_result: Optional[Any] = None,
        tp_result: Optional[Any] = None,
        ep_result: Optional[Any] = None,
    ) -> None:
        self.loader = loader

        def _as_result_dict(obj: Optional[Any]) -> Dict[str, Any]:
            # Some analyzers may return raw trace list instead of a structured dict.
            # Hybrid modules consume key-based dict payloads, so we normalize here.
            return obj if isinstance(obj, dict) else {}

        self.pp = _as_result_dict(pp_result)
        self.dp = _as_result_dict(dp_result)
        self.tp = _as_result_dict(tp_result)
        self.ep = _as_result_dict(ep_result)

        self.parallel_sizes = infer_parallel_sizes_from_loader(self.loader)

    def _enabled(self, dim: str) -> bool:
        return self.parallel_sizes.get(dim, 1) > 1

    @staticmethod
    def _empty_pair_result() -> Dict[str, Any]:
        return {
            "correlation_ep_cv_pp_bubble": 0.0,
            "avg_ep_expert_cv": 0.0,
            "avg_pp_bubble_rate": 0.0,
            "num_common_iterations": 0,
            "worst_iteration": -1,
            "root_ep_rank": None,
            "estimated_bubble_overhead_us": 0.0,
            "severity": "SKIPPED",
            "_ep_cv_series": [],
            "_pp_bubble_series": [],
            "_common_iters": [],
        }

    # ------------------------------------------------------------------
    # 1. EP → PP Bubble Amplification
    # ------------------------------------------------------------------

    def analyze_ep_pp_bubble_amplification(self) -> Dict[str, Any]:
        """Correlate EP expert_cv with PP bubble rate across iterations."""
        if not (self._enabled("ep") and self._enabled("pp")):
            return self._empty_pair_result()
        balance_data: List[Dict] = self.ep.get("balance_data", [])
        ep_by_iter: Dict[int, List[float]] = collections.defaultdict(list)
        for d in balance_data:
            expert_cv = d["expert_cv"]
            if expert_cv is not None:
                ep_by_iter[d["iteration"]].append(expert_cv)
        ep_cv_by_iter = {it: float(np.mean(v)) for it, v in ep_by_iter.items()}

        bubble_stats: List[Dict] = self.pp.get("bubble_stats", [])
        pp_bubble_by_iter: Dict[int, float] = {}
        for d in bubble_stats:
            it = d.get("iteration", d.get("iter", d.get("Iteration", -1)))
            rate = d.get("bubble_rate", d.get("Bubble_Rate", None))
            if it >= 0 and rate is not None:
                pp_bubble_by_iter[it] = float(rate)

        common = sorted(set(ep_cv_by_iter) & set(pp_bubble_by_iter))
        ep_s = [ep_cv_by_iter[i] for i in common]
        pp_s = [pp_bubble_by_iter[i] for i in common]
        corr = _pearson(ep_s, pp_s)

        avg_ep_cv = float(np.mean(ep_s)) if ep_s else 0.0
        avg_pp_bub = float(np.mean(pp_s)) if pp_s else 0.0

        if corr > _EP_PP_CORR_CRIT and avg_ep_cv > 0.30:
            severity = "CRITICAL"
        elif corr > _EP_PP_CORR_WARN and avg_ep_cv > 0.20:
            severity = "WARNING"
        else:
            severity = "OK"

        worst_iter = -1
        if common:
            scores = [ep_s[i] * pp_s[i] for i in range(len(common))]
            worst_iter = common[int(np.argmax(scores))]

        straggler_data: List[Dict] = self.ep.get("straggler_data", [])
        root_ep_rank: Optional[int] = None
        if straggler_data:
            freq = collections.Counter(d["straggler_rank"] for d in straggler_data)
            root_ep_rank = freq.most_common(1)[0][0]

        ep_worst = [d for d in straggler_data if d["iteration"] == worst_iter]
        estimated_overhead_us = float(np.mean([d["gap_us"] for d in ep_worst])) if ep_worst else 0.0

        return {
            "correlation_ep_cv_pp_bubble": round(corr, 4),
            "avg_ep_expert_cv": round(avg_ep_cv, 4),
            "avg_pp_bubble_rate": round(avg_pp_bub, 4),
            "num_common_iterations": len(common),
            "worst_iteration": worst_iter,
            "root_ep_rank": root_ep_rank,
            "estimated_bubble_overhead_us": round(estimated_overhead_us, 1),
            "severity": severity,
            "_ep_cv_series": ep_s,
            "_pp_bubble_series": pp_s,
            "_common_iters": common,
        }

    # ------------------------------------------------------------------
    # 2. EP / TP Communication Contention
    # ------------------------------------------------------------------

    def analyze_ep_tp_comm_contention(self) -> List[Dict[str, Any]]:
        """Measure temporal overlap of EP and TP communications per (rank, iteration)."""
        if not (self._enabled("ep") and self._enabled("tp")):
            return []
        ep_comm_names = ["ep-alltoall-dispatch", "ep-alltoall-combine",
                         "ep-allgather-dispatch", "ep-allgather-combine"]
        tp_comm_names = [
            "tp-allreduce",
            "tp-all-gather-first",
            "tp-all-gather-last",
            "tp-reduce-scatter",
            "tp-reduce-scatter-last",
        ]
        results: List[Dict[str, Any]] = []

        for rank in self.loader.get_ranks():
            ep_by_iter: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            tp_by_iter: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            for name in ep_comm_names:
                for ev in self.loader.get_events_by_name(name, rank=rank):
                    if ev.dur > 0:
                        ep_by_iter[ev.iteration].append((ev.ts, ev.end_ts))
            for name in tp_comm_names:
                for ev in self.loader.get_events_by_name(name, rank=rank):
                    if ev.dur > 0:
                        tp_by_iter[ev.iteration].append((ev.ts, ev.end_ts))

            for it in sorted(set(ep_by_iter) | set(tp_by_iter)):
                ep_m = _merge_intervals(ep_by_iter.get(it, []))
                tp_m = _merge_intervals(tp_by_iter.get(it, []))
                ep_t = _total_length(ep_m)
                tp_t = _total_length(tp_m)
                contention = _overlap_length(ep_m, tp_m)
                total = ep_t + tp_t
                ratio = contention / total if total > 0 else 0.0

                if ratio > _EP_TP_CONTENTION_CRIT:
                    severity = "CRITICAL"
                elif ratio > _EP_TP_CONTENTION_WARN:
                    severity = "WARNING"
                else:
                    severity = "OK"

                results.append({
                    "rank": rank, "iteration": it,
                    "ep_comm_us": float(ep_t), "tp_comm_us": float(tp_t),
                    "contention_us": float(contention),
                    "contention_ratio": round(ratio, 4),
                    "severity": severity,
                })
        return results

    # ------------------------------------------------------------------
    # 3. EP Load → DP Step Jitter Coupling
    # ------------------------------------------------------------------

    def analyze_ep_dp_load_coupling(self) -> Dict[str, Any]:
        """Correlate EP expert skew with DP step-time CV."""
        if not (self._enabled("ep") and self._enabled("dp")):
            return {
                "correlation_ep_skew_dp_cv": 0.0,
                "ep_contribution_pct": 0.0,
                "num_common_iterations": 0,
                "worst_iteration": -1,
                "avg_ep_cv": 0.0,
                "avg_dp_step_cv": 0.0,
                "severity": "SKIPPED",
                "_ep_cv_series": [],
                "_dp_cv_series": [],
                "_common_iters": [],
            }
        balance_data: List[Dict] = self.ep.get("balance_data", [])
        ep_by_iter: Dict[int, List[float]] = collections.defaultdict(list)
        for d in balance_data:
            expert_cv = d["expert_cv"]
            if expert_cv is not None:
                ep_by_iter[d["iteration"]].append(expert_cv)
        ep_cv_by_iter = {it: float(np.mean(v)) for it, v in ep_by_iter.items()}

        dp_cv_by_iter: Dict[int, float] = {}
        for d in self.dp.get("balance_data", []):
            it = d.get("iteration", -1)
            cv = d.get("cv", None)
            if it >= 0 and cv is not None:
                dp_cv_by_iter[it] = float(cv)

        common = sorted(set(ep_cv_by_iter) & set(dp_cv_by_iter))
        ep_s = [ep_cv_by_iter[i] for i in common]
        dp_s = [dp_cv_by_iter[i] for i in common]
        corr = _pearson(ep_s, dp_s)

        worst_iter = -1
        if common:
            scores = [ep_s[i] * dp_s[i] for i in range(len(common))]
            worst_iter = common[int(np.argmax(scores))]

        if corr > _EP_DP_CORR_CRIT:
            severity = "CRITICAL"
        elif corr > _EP_DP_CORR_WARN:
            severity = "WARNING"
        else:
            severity = "OK"

        ep_contribution_pct = round(corr ** 2 * 100, 1) if corr > 0 and len(dp_s) >= 3 else 0.0

        return {
            "correlation_ep_skew_dp_cv": round(corr, 4),
            "ep_contribution_pct": ep_contribution_pct,
            "num_common_iterations": len(common),
            "worst_iteration": worst_iter,
            "avg_ep_cv": round(float(np.mean(ep_s)) if ep_s else 0.0, 4),
            "avg_dp_step_cv": round(float(np.mean(dp_s)) if dp_s else 0.0, 4),
            "severity": severity,
            "_ep_cv_series": ep_s,
            "_dp_cv_series": dp_s,
            "_common_iters": common,
        }

    # ------------------------------------------------------------------
    # 4. PP Bubble ↔ DP Grad-Sync Serialization
    # ------------------------------------------------------------------

    def analyze_pp_dp_sync_serialization(self) -> List[Dict[str, Any]]:
        """Measure how much DP communication overlaps with PP bubble windows."""
        if not (self._enabled("pp") and self._enabled("dp")):
            return []
        dp_comm_names = ["allreduce", "grad-sync", "dp-reduce-scatter",
                         "dp-allreduce", "all-grads-sync"]
        active_names = ["forward-step", "backward-step", "optimizer"]
        results: List[Dict[str, Any]] = []

        for rank in self.loader.get_ranks():
            for iter_ev in self.loader.get_events_by_name("iteration", rank=rank):
                if iter_ev.dur <= 0:
                    continue
                it = iter_ev.iteration
                iter_start, iter_end = iter_ev.ts, iter_ev.end_ts

                active_ivs: List[Tuple[int, int]] = []
                for name in active_names:
                    for ev in self.loader.get_events_by_name(name, rank=rank, iteration=it):
                        if ev.dur > 0:
                            active_ivs.append((ev.ts, ev.end_ts))

                active_merged = _merge_intervals(active_ivs)
                bubble_us = max(0, iter_ev.dur - _total_length(active_merged))
                if bubble_us <= 0:
                    continue

                bubble_ivs: List[Tuple[int, int]] = []
                cursor = iter_start
                for a_s, a_e in active_merged:
                    if cursor < a_s:
                        bubble_ivs.append((cursor, a_s))
                    cursor = max(cursor, a_e)
                if cursor < iter_end:
                    bubble_ivs.append((cursor, iter_end))

                dp_ivs: List[Tuple[int, int]] = []
                for name in dp_comm_names:
                    for ev in self.loader.get_events_by_name(name, rank=rank, iteration=it):
                        if ev.dur > 0:
                            dp_ivs.append((ev.ts, ev.end_ts))

                dp_merged = _merge_intervals(dp_ivs)
                dp_total = _total_length(dp_merged)
                dp_in_bubble = _overlap_length(bubble_ivs, dp_merged)
                util_ratio = dp_in_bubble / bubble_us if bubble_us > 0 else 1.0

                if util_ratio < _PP_DP_UTIL_CRIT:
                    severity = "CRITICAL"
                elif util_ratio < _PP_DP_UTIL_WARN:
                    severity = "WARNING"
                else:
                    severity = "OK"

                results.append({
                    "rank": rank, "iteration": it,
                    "bubble_us": float(bubble_us),
                    "dp_comm_us": float(dp_total),
                    "dp_comm_in_bubble_us": float(dp_in_bubble),
                    "bubble_utilization_ratio": round(util_ratio, 4),
                    "severity": severity,
                })
        return results

    # ------------------------------------------------------------------
    # 5. TP Fragmentation → EP Expert Compute Slowdown
    # ------------------------------------------------------------------

    def analyze_tp_ep_compute_fragmentation(self) -> Dict[str, Any]:
        """Correlate TP kernel fragmentation with EP expert compute duration per rank."""
        if not (self._enabled("tp") and self._enabled("ep")):
            return {
                "correlation_tp_frag_ep_compute": 0.0,
                "num_common_ranks": 0,
                "severity": "SKIPPED",
                "per_rank": [],
            }
        frag_data: List[Dict] = self.tp.get("frag_data", self.tp.get("fragmentation_data", []))
        frag_by_rank: Dict[int, float] = {}
        for d in frag_data:
            r = d.get("rank", -1)
            if r >= 0:
                frag_by_rank[r] = float(d.get("short_ratio", 0.0))

        experts_by_rank: Dict[int, List[float]] = collections.defaultdict(list)
        for d in self.ep.get("comm_data", []):
            r = d.get("rank", -1)
            v = d.get("experts_dur_us", None)
            if r >= 0 and v is not None:
                experts_by_rank[r].append(float(v))
        avg_experts_by_rank = {r: float(np.mean(v)) for r, v in experts_by_rank.items()}

        common_ranks = sorted(set(frag_by_rank) & set(avg_experts_by_rank))
        frag_s = [frag_by_rank[r] for r in common_ranks]
        exp_s = [avg_experts_by_rank[r] for r in common_ranks]
        corr = _pearson(frag_s, exp_s)

        per_rank = []
        for r in common_ranks:
            fv, ev = frag_by_rank[r], avg_experts_by_rank[r]
            sev = ("CRITICAL" if corr > _TP_EP_FRAG_CORR_CRIT and fv > 0.30
                   else "WARNING" if corr > _TP_EP_FRAG_CORR_WARN and fv > 0.20
                   else "OK")
            per_rank.append({"rank": r, "short_kernel_ratio": round(fv, 4),
                             "avg_experts_dur_us": round(ev, 1), "severity": sev})

        return {
            "correlation_tp_frag_ep_compute": round(corr, 4),
            "num_common_ranks": len(common_ranks),
            "severity": ("CRITICAL" if corr > _TP_EP_FRAG_CORR_CRIT else
                         "WARNING" if corr > _TP_EP_FRAG_CORR_WARN else "OK"),
            "per_rank": per_rank,
        }

    # ------------------------------------------------------------------
    # 6. Global Straggler (cross-dimension)
    # ------------------------------------------------------------------

    def diagnose_global_straggler(self) -> List[Dict[str, Any]]:
        """Identify ranks repeatedly slowest across PP/DP/TP/EP and diagnose root cause.

        Improvements over the original implementation:

        * **PP**: Uses ``straggler_data`` (standardised jitter ROOT_CAUSE + P2P
          dep-stall sender ranks) instead of receiver-side p2p_stats, which
          incorrectly blamed the downstream victim rather than the slow upstream.
        * **TP**: Uses ``straggler_data`` from the new
          :meth:`TPAnalyzer.diagnose_tp_stragglers` method instead of low-SM-util
          heuristics, which were a proxy metric not a direct straggler signal.
        * **HW root cause**: For every rank with WARNING/CRITICAL severity,
          consolidates hardware evidence (temperature, clock throttling) from
          *all* sub-analyzer ``hw_detail`` fields to produce a unified
          ``hardware_root_cause`` string and structured ``hw_evidence`` dict.
        """
        score: Dict[int, Dict[str, int]] = collections.defaultdict(
            lambda: {"pp": 0, "dp": 0, "tp": 0, "ep": 0, "total": 0, "n": 0}
        )
        # Accumulate hw_detail evidence per rank from all sub-analyzers
        hw_evidence: Dict[int, List[Dict]] = collections.defaultdict(list)

        # ------------------------------------------------------------------
        # PP: use standardised straggler_data (jitter ROOT_CAUSE + p2p sender)
        # ------------------------------------------------------------------
        for d in self.pp.get("straggler_data", []):
            r = d.get("rank", -1)
            if r >= 0:
                score[r]["pp"] += 1
                score[r]["total"] += 1
                score[r]["n"] += 1

        # ------------------------------------------------------------------
        # DP: use straggler_rank from diagnose_stragglers() + hw_detail
        # ------------------------------------------------------------------
        for d in self.dp.get("straggler_data", []):
            r = d.get("straggler_rank", -1)
            if r >= 0:
                score[r]["dp"] += 1
                score[r]["total"] += 1
                score[r]["n"] += 1
                hw = d.get("hw_detail")
                if hw:
                    hw_evidence[r].append({"source": "dp", **hw})

        # ------------------------------------------------------------------
        # TP: use straggler_rank from diagnose_tp_stragglers() + hw_detail
        # (falls back to low-SM-util heuristic if new field absent)
        # ------------------------------------------------------------------
        tp_straggler_data = self.tp.get("straggler_data", [])
        if tp_straggler_data:
            for d in tp_straggler_data:
                r = d.get("straggler_rank", -1)
                if r >= 0:
                    score[r]["tp"] += 1
                    score[r]["total"] += 1
                    score[r]["n"] += 1
                    hw = d.get("hw_detail")
                    if hw:
                        hw_evidence[r].append({"source": "tp", **hw})
        else:
            # Legacy fallback: low SM utilisation as TP straggler proxy
            for d in self.tp.get("sm_data", []):
                r = d.get("rank", -1)
                util_raw = d.get("mean_sm_util_pct", 100.0)
                try:
                    util = float(util_raw) if util_raw is not None else 100.0
                except (TypeError, ValueError):
                    util = 100.0
                if r >= 0 and util < 50.0:
                    score[r]["tp"] += 1
                    score[r]["total"] += 1
                    score[r]["n"] += 1

        # ------------------------------------------------------------------
        # EP: use straggler_rank from diagnose_ep_stragglers() + hw_detail
        # ------------------------------------------------------------------
        for d in self.ep.get("straggler_data", []):
            r = d.get("straggler_rank", -1)
            if r >= 0:
                score[r]["ep"] += 1
                score[r]["total"] += 1
                score[r]["n"] += 1
                hw = d.get("hw_detail")
                if hw:
                    hw_evidence[r].append({"source": "ep", **hw})

        total_events = sum(v["n"] for v in score.values()) or 1
        results = []
        for rank, s in sorted(score.items()):
            straggler_score = s["total"] / total_events
            if straggler_score > _GLOBAL_STRAGGLER_CRIT:
                sev = "CRITICAL"
            elif straggler_score > _GLOBAL_STRAGGLER_WARN:
                sev = "WARNING"
            else:
                sev = "OK"

            # ---- Consolidate hardware root cause from all sub-analyzers ----
            hw_root_cause, hw_summary = self._consolidate_hw_evidence(
                rank, hw_evidence.get(rank, [])
            )

            results.append({
                "rank": rank,
                "straggler_score": round(straggler_score, 4),
                "pp_straggler_count": s["pp"],
                "dp_straggler_count": s["dp"],
                "tp_straggler_count": s["tp"],
                "ep_straggler_count": s["ep"],
                "severity": sev,
                "hardware_root_cause": hw_root_cause,
                "hw_evidence": hw_summary,
            })
        return sorted(results, key=lambda x: -x["straggler_score"])

    def _consolidate_hw_evidence(
        self,
        rank: int,
        evidence_list: List[Dict],
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Merge hw_detail dicts from multiple sub-analyzers into a unified diagnosis.

        Aggregates peak temperature, mean SM clock, and base clock across all
        evidence entries, then applies the standard thermal/throttle thresholds
        to produce a single ``hardware_root_cause`` string.

        If HW counter data is unavailable (no hw_detail entries), also checks
        the ``loader`` for a HW metrics window around the most recent iteration
        for this rank as a fallback.
        """
        _THERMAL_THROTTLE_TEMP_C = 80.0
        _CLOCK_DROP_RATIO = 0.92

        # Gather raw values from accumulated evidence
        temp_peaks: List[float] = []
        clock_means: List[float] = []
        base_clocks: List[float] = []
        sources: List[str] = []

        for ev in evidence_list:
            sources.append(ev.get("source", "unknown"))
            v = ev.get("Temp_C_peak")
            if v is not None:
                try:
                    temp_peaks.append(float(v))
                except (TypeError, ValueError):
                    pass
            v = ev.get("SM_Clock_MHz_mean")
            if v is not None:
                try:
                    clock_means.append(float(v))
                except (TypeError, ValueError):
                    pass
            v = ev.get("SM_Base_Clock_MHz")
            if v is not None:
                try:
                    base_clocks.append(float(v))
                except (TypeError, ValueError):
                    pass

        # If no evidence from sub-analyzers, try a direct HW query over the
        # last observed iteration window on this rank
        if not temp_peaks and not clock_means:
            try:
                iter_events = self.loader.get_events_by_name(
                    "iteration", rank=rank
                )
                if iter_events:
                    # Use the most recent iteration
                    latest = max(iter_events, key=lambda e: e.ts)
                    for metric, bucket in [
                        ("Temp_C", temp_peaks),
                        ("SM_Clock_MHz", clock_means),
                    ]:
                        res = self.loader.get_hardware_metrics_in_window(
                            latest.ts, latest.end_ts, rank, metric
                        )
                        if res is not None and res.num_samples > 0:
                            bucket.append(res.peak if metric == "Temp_C" else res.mean)
                    base_res = self.loader.get_hardware_metrics_in_window(
                        latest.ts, latest.end_ts, rank, "SM_Base_Clock_MHz"
                    )
                    if base_res is not None and base_res.num_samples > 0:
                        base_clocks.append(base_res.mean)
            except Exception:
                pass

        if not temp_peaks and not clock_means:
            return None, None

        import numpy as np  # already imported at module level, safe
        peak_temp = float(max(temp_peaks)) if temp_peaks else None
        mean_clock = float(np.mean(clock_means)) if clock_means else None
        base_clock = float(np.mean(base_clocks)) if base_clocks else None

        is_hot = peak_temp is not None and peak_temp >= _THERMAL_THROTTLE_TEMP_C
        is_throttled = (
            mean_clock is not None
            and base_clock is not None
            and base_clock > 0
            and mean_clock < base_clock * _CLOCK_DROP_RATIO
        )

        diag_parts: List[str] = []
        if is_hot and is_throttled:
            diag_parts.append(
                f"Thermal Throttling: {peak_temp:.0f}\u00b0C, "
                f"Clock {mean_clock:.0f}/{base_clock:.0f} MHz "
                f"({mean_clock / base_clock * 100:.0f}% of base)"
            )
        elif is_throttled:
            diag_parts.append(
                f"Clock Throttling: {mean_clock:.0f}/{base_clock:.0f} MHz "
                f"({mean_clock / base_clock * 100:.0f}% of base)"
            )
        elif is_hot:
            diag_parts.append(f"High Temperature: {peak_temp:.0f}\u00b0C")

        hw_root_cause = "; ".join(diag_parts) if diag_parts else None
        hw_summary: Dict = {
            "sources": list(set(sources)),
        }
        if peak_temp is not None:
            hw_summary["Temp_C_peak"] = round(peak_temp, 1)
        if mean_clock is not None:
            hw_summary["SM_Clock_MHz_mean"] = round(mean_clock, 1)
        if base_clock is not None:
            hw_summary["SM_Base_Clock_MHz"] = round(base_clock, 1)
        if mean_clock is not None and base_clock is not None and base_clock > 0:
            hw_summary["clock_ratio"] = round(mean_clock / base_clock, 3)

        return hw_root_cause, hw_summary if hw_summary else None

    # ------------------------------------------------------------------
    # 7. Hybrid Health Score
    # ------------------------------------------------------------------

    def compute_hybrid_health_score(
        self,
        ep_pp_result: Dict[str, Any],
        contention_data: List[Dict[str, Any]],
        ep_dp_result: Dict[str, Any],
        pp_dp_data: List[Dict[str, Any]],
        tp_ep_result: Dict[str, Any],
        straggler_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute a 0-100 composite health score."""
        score = 100
        deductions: List[Tuple[str, int]] = []

        def deduct(reason: str, pts: int) -> None:
            nonlocal score
            score -= pts
            deductions.append((reason, pts))

        # PP bubble
        for d in self.pp.get("bubble_stats", []):
            bub = d.get("bubble_rate", d.get("Bubble_Rate", 0.0))
            theory = d.get("theory_bubble_rate", d.get("Theory_Bubble_Rate", 0.0))
            excess = bub - theory
            if excess > 0.50:
                deduct("PP bubble > theory+50%", 20)
                break
            elif excess > 0.20:
                deduct("PP bubble > theory+20%", 10)
                break

        # DP grad sync
        for d in self.dp.get("sync_data", self.dp.get("grad_sync_data", [])):
            if d.get("sync_ratio", 0.0) > 0.40:
                deduct("DP grad sync ratio > 40%", 10)
                break
            elif d.get("sync_ratio", 0.0) > 0.20:
                deduct("DP grad sync ratio > 20%", 5)
                break

        # TP comm ratio
        for d in self.tp.get("tp_comm_data", self.tp.get("overhead_data", [])):
            comm_ratio = d.get("comm_ratio")
            if comm_ratio is None:
                continue
            if comm_ratio > 0.50:
                deduct("TP comm ratio > 50%", 10)
                break
            elif comm_ratio > 0.30:
                deduct("TP comm ratio > 30%", 5)
                break

        # TP NVLink
        for d in self.tp.get("nvlink_summary", self.tp.get("nvlink_data", [])):
            utilisation = d.get("mean_utilisation_pct")
            if utilisation is None:
                continue
            if utilisation < 50.0:
                deduct("TP NVLink util < 50%", 10)
                break

        # EP expert balance
        for d in self.ep.get("balance_data", []):
            expert_cv = d.get("expert_cv")
            if expert_cv is None:
                continue
            if expert_cv > 0.50:
                deduct("EP expert_cv > 0.50", 10)
                break
            elif expert_cv > 0.30:
                deduct("EP expert_cv > 0.30", 5)
                break

        # EP token drop
        for d in self.ep.get("drop_data", []):
            if d.get("avg_drop_rate", 0.0) > 0.05:
                deduct("EP token drop_rate > 5%", 10)
                break
            elif d.get("avg_drop_rate", 0.0) > 0.01:
                deduct("EP token drop_rate > 1%", 5)
                break

        # EP router collapse
        for d in self.ep.get("health_data", []):
            if d.get("status") == "Router Collapse Risk":
                deduct("EP Router Collapse Risk", 20)
                break

        # Cross-dimension
        if ep_pp_result.get("severity") == "CRITICAL":
            deduct("EP-PP bubble amplification CRITICAL", 10)
        elif ep_pp_result.get("severity") == "WARNING":
            deduct("EP-PP bubble amplification WARNING", 5)

        if contention_data:
            max_cont = max(d["contention_ratio"] for d in contention_data)
            if max_cont > _EP_TP_CONTENTION_CRIT:
                deduct("EP-TP comm contention CRITICAL", 10)
            elif max_cont > _EP_TP_CONTENTION_WARN:
                deduct("EP-TP comm contention WARNING", 5)

        if any(d["severity"] == "CRITICAL" for d in straggler_data):
            deduct("Global straggler CRITICAL", 10)
        elif any(d["severity"] == "WARNING" for d in straggler_data):
            deduct("Global straggler WARNING", 5)

        score = max(0, score)
        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 45:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": score,
            "grade": grade,
            "deductions": deductions,
            "total_deducted": 100 - score,
        }


# ===========================================================================
# Report Logger
# ===========================================================================

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


# ===========================================================================
# Visualization helpers
# ===========================================================================

def _generate_ep_pp_plot(result: Dict[str, Any], output_dir: str) -> None:
    iters = result.get("_common_iters", [])
    ep_s  = result.get("_ep_cv_series", [])
    pp_s  = result.get("_pp_bubble_series", [])
    if not iters:
        return
    _pick_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Hybrid: EP Expert Skew vs PP Bubble Amplification",
                 fontsize=18, fontweight="bold")
    ax = axes[0]
    ax.plot(iters, [v * 100 for v in ep_s], marker="o", linewidth=1.5,
            color="steelblue", label="EP Expert CV")
    ax2 = ax.twinx()
    ax2.plot(iters, [v * 100 for v in pp_s], marker="s", linewidth=1.5,
             color="salmon", label="PP Bubble Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("EP Expert CV (%)", color="steelblue")
    ax2.set_ylabel("PP Bubble Rate (%)", color="salmon")
    ax.set_title(f"Time Series  corr={result['correlation_ep_cv_pp_bubble']:.3f}",
                 fontsize=12, fontweight="bold")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize="small")
    ax = axes[1]
    ax.scatter(ep_s, pp_s, c="steelblue", edgecolors="black", s=50, alpha=0.75)
    if len(ep_s) >= 3:
        m, b = np.polyfit(ep_s, pp_s, 1)
        xr = np.linspace(min(ep_s), max(ep_s), 100)
        ax.plot(xr, m * xr + b, "r--", linewidth=2, label="Trend")
        ax.legend(fontsize="small")
    ax.set_xlabel("EP Expert CV")
    ax.set_ylabel("PP Bubble Rate")
    ax.set_title("Scatter: EP CV vs PP Bubble", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "hybrid_ep_pp_bubble.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[Hybrid EP-PP] Plot saved to {path}")


def _generate_ep_tp_contention_plot(
    data: List[Dict[str, Any]], output_dir: str
) -> None:
    if not data:
        return
    _pick_style()
    df = pd.DataFrame(data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rc = {r: colors[i % 10] for i, r in enumerate(ranks)}
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Hybrid: EP / TP Communication Contention",
                 fontsize=18, fontweight="bold")
    ax = axes[0]
    for r in ranks:
        sub = df[df["rank"] == r].sort_values("iteration")
        ax.plot(sub["iteration"], sub["contention_ratio"] * 100,
                marker="o", markersize=3, linewidth=1.2, alpha=0.8,
                label=f"Rank {r}", color=rc[r])
    ax.axhline(y=_EP_TP_CONTENTION_WARN * 100, color="orange",
               linestyle="--", linewidth=2, label="Warning")
    ax.axhline(y=_EP_TP_CONTENTION_CRIT * 100, color="red",
               linestyle="--", linewidth=2, label="Critical")
    ax.set_title("EP-TP Contention Ratio Over Iterations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contention Ratio (%)")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax = axes[1]
    avg = df.groupby("rank")[["ep_comm_us", "tp_comm_us", "contention_us"]].mean().reindex(ranks)
    x = np.arange(len(ranks))
    ax.bar(x, avg["ep_comm_us"] / 1e3, 0.5, label="EP Comm",
           color="steelblue", edgecolor="black")
    ax.bar(x, avg["tp_comm_us"] / 1e3, 0.5, bottom=avg["ep_comm_us"] / 1e3,
           label="TP Comm", color="mediumseagreen", edgecolor="black")
    ax.bar(x, avg["contention_us"] / 1e3, 0.5,
           bottom=(avg["ep_comm_us"] + avg["tp_comm_us"]) / 1e3,
           label="Contention", color="salmon", hatch="//", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_title("Avg Comm Breakdown per Rank", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Global Rank")
    ax.legend(fontsize="small")
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "hybrid_ep_tp_contention.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[Hybrid EP-TP] Plot saved to {path}")


def _generate_ep_dp_coupling_plot(result: Dict[str, Any], output_dir: str) -> None:
    iters = result.get("_common_iters", [])
    ep_s  = result.get("_ep_cv_series", [])
    dp_s  = result.get("_dp_cv_series", [])
    if not iters:
        return
    _pick_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Hybrid: EP Load Imbalance → DP Step-Time Jitter",
                 fontsize=18, fontweight="bold")
    ax = axes[0]
    ax.plot(iters, ep_s, marker="o", linewidth=1.5, color="steelblue", label="EP Expert CV")
    ax2 = ax.twinx()
    ax2.plot(iters, dp_s, marker="s", linewidth=1.5, color="salmon", label="DP Step CV")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("EP Expert CV", color="steelblue")
    ax2.set_ylabel("DP Step CV", color="salmon")
    ax.set_title(f"Time Series  corr={result['correlation_ep_skew_dp_cv']:.3f}",
                 fontsize=12, fontweight="bold")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize="small")
    ax = axes[1]
    ax.scatter(ep_s, dp_s, c="steelblue", edgecolors="black", s=50, alpha=0.75)
    if len(ep_s) >= 3:
        m, b = np.polyfit(ep_s, dp_s, 1)
        xr = np.linspace(min(ep_s), max(ep_s), 100)
        ax.plot(xr, m * xr + b, "r--", linewidth=2)
    ax.set_xlabel("EP Expert CV")
    ax.set_ylabel("DP Step CV")
    ax.set_title("Scatter: EP Skew vs DP Step Jitter", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "hybrid_ep_dp_coupling.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[Hybrid EP-DP] Plot saved to {path}")


def _generate_pp_dp_serialization_plot(
    data: List[Dict[str, Any]], output_dir: str
) -> None:
    if not data:
        return
    _pick_style()
    df = pd.DataFrame(data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rc = {r: colors[i % 10] for i, r in enumerate(ranks)}
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Hybrid: PP Bubble Utilization by DP Grad-Sync",
                 fontsize=18, fontweight="bold")
    ax = axes[0]
    avg = df.groupby("rank")["bubble_utilization_ratio"].mean().reindex(ranks)
    bar_colors = ["salmon" if v < _PP_DP_UTIL_WARN else "mediumseagreen"
                  for v in avg.values]
    ax.bar([str(r) for r in ranks], avg.values * 100,
           color=bar_colors, edgecolor="black", alpha=0.85)
    ax.axhline(y=_PP_DP_UTIL_WARN * 100, color="orange", linestyle="--",
               linewidth=2, label=f"Warning ({_PP_DP_UTIL_WARN:.0%})")
    ax.axhline(y=_PP_DP_UTIL_CRIT * 100, color="red", linestyle="--",
               linewidth=2, label=f"Critical ({_PP_DP_UTIL_CRIT:.0%})")
    ax.set_title("Avg Bubble Utilization per Rank", fontsize=12, fontweight="bold")
    ax.set_ylabel("DP Comm in Bubble (%)")
    ax.set_xlabel("Global Rank")
    ax.set_ylim(0, 110)
    ax.legend(fontsize="small")
    ax = axes[1]
    for r in ranks:
        sub = df[df["rank"] == r].sort_values("iteration")
        ax.plot(sub["iteration"], sub["bubble_utilization_ratio"] * 100,
                marker="o", markersize=3, linewidth=1.2, alpha=0.8,
                label=f"Rank {r}", color=rc[r])
    ax.axhline(y=_PP_DP_UTIL_WARN * 100, color="orange", linestyle="--", linewidth=2)
    ax.set_title("Bubble Utilization Over Iterations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Utilization (%)")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "hybrid_pp_dp_serialization.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[Hybrid PP-DP] Plot saved to {path}")


def _generate_global_straggler_plot(
    data: List[Dict[str, Any]], output_dir: str
) -> None:
    if not data:
        return
    _pick_style()
    df = pd.DataFrame(data)
    ranks = df["rank"].tolist()
    dims = ["pp_straggler_count", "dp_straggler_count",
            "tp_straggler_count", "ep_straggler_count"]
    dim_labels = ["PP", "DP", "TP", "EP"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle("Hybrid: Global Straggler Cross-Dimension Heatmap",
                 fontsize=18, fontweight="bold")
    ax = axes[0]
    matrix = df[dims].values.T.astype(float)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(dim_labels)))
    ax.set_yticklabels(dim_labels)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_title("Straggler Heatmap (Rank × Dimension)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Global Rank")
    ax.set_ylabel("Dimension")
    plt.colorbar(im, ax=ax, shrink=0.8)
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(dim_labels)):
        for j in range(len(ranks)):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if matrix[i, j] > vmax * 0.6 else "black")
    ax = axes[1]
    scores = df["straggler_score"].values
    bar_colors = ["red" if s > _GLOBAL_STRAGGLER_CRIT
                  else "orange" if s > _GLOBAL_STRAGGLER_WARN
                  else "mediumseagreen" for s in scores]
    ax.bar([str(r) for r in ranks], scores * 100,
           color=bar_colors, edgecolor="black", alpha=0.85)
    ax.axhline(y=_GLOBAL_STRAGGLER_WARN * 100, color="orange", linestyle="--",
               linewidth=2, label="Warning")
    ax.axhline(y=_GLOBAL_STRAGGLER_CRIT * 100, color="red", linestyle="--",
               linewidth=2, label="Critical")
    ax.set_title("Global Straggler Score per Rank", fontsize=12, fontweight="bold")
    ax.set_ylabel("Straggler Score (%)")
    ax.set_xlabel("Global Rank")
    ax.legend(fontsize="small")
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "hybrid_global_straggler.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[Hybrid Straggler] Plot saved to {path}")


def _generate_health_score_plot(health: Dict[str, Any], output_dir: str) -> None:
    _pick_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        f"Hybrid Health Score: {health['score']}/100  (Grade {health['grade']})",
        fontsize=20, fontweight="bold",
    )
    ax = axes[0]
    theta = np.linspace(0, np.pi, 200)
    score_pct = health["score"] / 100.0
    for color, start, end in [
        ("red", 0.0, np.pi * 0.45),
        ("orange", np.pi * 0.45, np.pi * 0.75),
        ("mediumseagreen", np.pi * 0.75, np.pi),
    ]:
        t = np.linspace(start, end, 100)
        ax.plot(np.cos(t), np.sin(t), color=color, linewidth=10, solid_capstyle="butt")
    needle_angle = np.pi * (1.0 - score_pct)
    ax.annotate(
        "",
        xy=(np.cos(needle_angle) * 0.72, np.sin(needle_angle) * 0.72),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=3),
    )
    ax.text(0, -0.25, str(health["score"]), ha="center", va="center",
            fontsize=44, fontweight="bold")
    ax.text(0, -0.48, f"Grade {health['grade']}", ha="center", va="center",
            fontsize=20, color="dimgray")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.65, 1.15)
    ax.axis("off")
    ax.set_title("Composite Health Score", fontsize=14, fontweight="bold")
    ax = axes[1]
    deductions = health.get("deductions", [])
    if deductions:
        reasons = [d[0] for d in deductions]
        pts = [d[1] for d in deductions]
        y_pos = np.arange(len(reasons))
        ax.barh(y_pos, pts, color="salmon", edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(reasons, fontsize=9)
        ax.set_xlabel("Points Deducted")
        ax.set_title("Score Deduction Breakdown", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No deductions — Perfect score!",
                ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.axis("off")
        ax.set_title("Score Deduction Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90], h_pad=1.6, w_pad=1.6)
    path = os.path.join(output_dir, "hybrid_health_score.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[Hybrid Health] Plot saved to {path}")


# ===========================================================================
# Report writing helpers
# ===========================================================================

def _write_hybrid_report(
    logger: "_ReportLogger",
    ep_pp: Dict[str, Any],
    contention: List[Dict[str, Any]],
    ep_dp: Dict[str, Any],
    pp_dp: List[Dict[str, Any]],
    tp_ep: Dict[str, Any],
    stragglers: List[Dict[str, Any]],
    health: Dict[str, Any],
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 1: EP-Induced PP Bubble Amplification")
    logger.log("=" * 90)
    logger.log(f"  Correlation EP expert_cv vs PP bubble_rate: {ep_pp['correlation_ep_cv_pp_bubble']:.4f}")
    logger.log(f"  Avg EP expert_cv: {ep_pp['avg_ep_expert_cv']:.4f}")
    logger.log(f"  Avg PP bubble_rate: {ep_pp['avg_pp_bubble_rate']:.4f}")
    logger.log(f"  Common iterations analysed: {ep_pp['num_common_iterations']}")
    logger.log(f"  Worst iteration: {ep_pp['worst_iteration']}")
    logger.log(f"  Root EP rank: {ep_pp['root_ep_rank']}")
    logger.log(f"  Estimated extra bubble overhead: {ep_pp['estimated_bubble_overhead_us']:.1f} us")
    logger.log(f"  Severity: {ep_pp['severity']}")
    if ep_pp["severity"] in ("WARNING", "CRITICAL"):
        logger.log("  [RECOMMENDATION] Reduce EP expert skew (increase aux_loss_coeff or"
                   " use Sinkhorn routing) to relieve PP pipeline stall.")

    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 2: EP / TP Communication Contention")
    logger.log("=" * 90)
    if contention:
        df = pd.DataFrame(contention)
        max_ratio = df["contention_ratio"].max()
        mean_ratio = df["contention_ratio"].mean()
        crit_cnt = int((df["severity"] == "CRITICAL").sum())
        warn_cnt = int((df["severity"] == "WARNING").sum())
        logger.log(f"  Max contention ratio: {max_ratio:.2%}")
        logger.log(f"  Mean contention ratio: {mean_ratio:.2%}")
        logger.log(f"  CRITICAL entries: {crit_cnt}  WARNING entries: {warn_cnt}")
        if max_ratio > _EP_TP_CONTENTION_WARN:
            logger.log("  [RECOMMENDATION] Schedule EP all-to-all and TP collectives in"
                       " separate CUDA streams or use async EP dispatch to avoid NVLink"
                       " bandwidth contention.")
    else:
        logger.log("  No EP-TP contention data available.")

    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 3: EP Load Imbalance → DP Step-Time Jitter")
    logger.log("=" * 90)
    logger.log(f"  Correlation EP skew vs DP step CV: {ep_dp['correlation_ep_skew_dp_cv']:.4f}")
    logger.log(f"  EP contribution to DP step variance: {ep_dp['ep_contribution_pct']:.1f}%")
    logger.log(f"  Severity: {ep_dp['severity']}")
    if ep_dp["severity"] in ("WARNING", "CRITICAL"):
        logger.log("  [RECOMMENDATION] Balance EP routing (reduce expert_cv) to stabilise"
                   " DP step time and avoid AllReduce wait at gradient sync barrier.")

    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 4: PP Bubble ↔ DP Grad-Sync Serialization")
    logger.log("=" * 90)
    if pp_dp:
        df = pd.DataFrame(pp_dp)
        mean_util = df["bubble_utilization_ratio"].mean()
        crit_cnt = int((df["severity"] == "CRITICAL").sum())
        logger.log(f"  Mean bubble utilization by DP comm: {mean_util:.2%}")
        logger.log(f"  CRITICAL entries (util < {_PP_DP_UTIL_CRIT:.0%}): {crit_cnt}")
        if mean_util < _PP_DP_UTIL_WARN:
            logger.log("  [RECOMMENDATION] Enable async DP AllReduce / ReduceScatter so"
                       " gradient sync starts during the PP warmup bubble period.")
    else:
        logger.log("  No PP-DP serialization data available.")

    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 5: TP Fragmentation → EP Expert Compute Slowdown")
    logger.log("=" * 90)
    logger.log(f"  Correlation TP frag vs EP experts_dur: {tp_ep['correlation_tp_frag_ep_compute']:.4f}")
    logger.log(f"  Ranks analysed: {tp_ep['num_common_ranks']}")
    logger.log(f"  Severity: {tp_ep['severity']}")
    for pr in tp_ep.get("per_rank", []):
        if pr["severity"] != "OK":
            logger.log(f"    Rank {pr['rank']}: short_ratio={pr['short_kernel_ratio']:.3f}"
                       f"  avg_experts_dur={pr['avg_experts_dur_us']:.0f} us  [{pr['severity']}]")

    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 6: Global Straggler Diagnosis")
    logger.log("=" * 90)
    if stragglers:
        for d in stragglers[:10]:
            hw_note = f"  HW: {d['hardware_root_cause']}" if d.get("hardware_root_cause") else ""
            logger.log(f"  Rank {d['rank']:3d}  score={d['straggler_score']:.3f}"
                       f"  PP={d['pp_straggler_count']} DP={d['dp_straggler_count']}"
                       f" TP={d['tp_straggler_count']} EP={d['ep_straggler_count']}"
                       f"  [{d['severity']}]{hw_note}")
        worst = stragglers[0]
        if worst["severity"] != "OK":
            hw_rc = worst.get("hardware_root_cause")
            if hw_rc:
                # Hardware root cause identified — give specific actionable advice
                logger.log(f"  [CRITICAL] Rank {worst['rank']} hardware root cause: {hw_rc}")
                hw_ev = worst.get("hw_evidence") or {}
                clock_ratio = hw_ev.get("clock_ratio")
                temp_peak = hw_ev.get("Temp_C_peak")
                if clock_ratio is not None and clock_ratio < 0.92:
                    freq_drop_pct = round((1.0 - clock_ratio) * 100, 1)
                    logger.log(
                        f"  [RECOMMENDATION] Rank {worst['rank']} SM clock is running "
                        f"{freq_drop_pct}% below base frequency — GPU is frequency-throttled. "
                        f"Action: check power capping (nvidia-smi -pl), thermal paste, "
                        f"and cooling system.  Consider draining this node."
                    )
                if temp_peak is not None and temp_peak >= 80.0:
                    logger.log(
                        f"  [RECOMMENDATION] Rank {worst['rank']} GPU temperature peaked at "
                        f"{temp_peak:.0f}\u00b0C — above thermal throttle threshold. "
                        f"Action: verify data-centre cooling, check GPU fan health, "
                        f"and reduce ambient temperature."
                    )
                sources = hw_ev.get("sources", [])
                logger.log(
                    f"  [INFO] Hardware evidence corroborated by sub-analyzers: "
                    f"{', '.join(sources) if sources else 'direct HW query'}."
                )
            else:
                # No HW evidence — likely software root cause
                logger.log(
                    f"  [RECOMMENDATION] Rank {worst['rank']} is a persistent straggler "
                    "across multiple dimensions with no hardware throttle detected. "
                    "Likely software root cause: check data skew (uneven sequence lengths), "
                    "expert routing imbalance (MoE), or PP layer partition asymmetry."
                )
    else:
        logger.log("  No global straggler data available.")

    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 7: Hybrid Health Score")
    logger.log("=" * 90)
    logger.log(f"  Overall Score : {health['score']} / 100")
    logger.log(f"  Grade         : {health['grade']}")
    logger.log(f"  Total deducted: {health['total_deducted']} pts")
    logger.log("  Deductions:")
    for reason, pts in health.get("deductions", []):
        logger.log(f"    -{pts:2d}  {reason}")
    logger.log("=" * 90 + "\n")


# ===========================================================================
# Main Orchestrator
# ===========================================================================

def analyze_hybrid_traces(
    traces: List[Dict[str, Any]],
    output_dir: str = ".",
    pp_result: Optional[Dict[str, Any]] = None,
    dp_result: Optional[Dict[str, Any]] = None,
    tp_result: Optional[Dict[str, Any]] = None,
    ep_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all hybrid cross-dimension analyses.

    Sub-analyzer results can be passed in to avoid recomputation.  If any
    are omitted the relevant cross-dimension modules will gracefully return
    empty / zero results rather than raising errors.

    Args:
        traces: Aggregated Chrome Trace event list.
        output_dir: Directory for reports, PDFs, and CSVs.
        pp_result: Optional cached output of ``analyze_pp_traces``.
        dp_result: Optional cached output of ``analyze_dp_traces``.
        tp_result: Optional cached output of ``analyze_tp_traces``.
        ep_result: Optional cached output of ``analyze_ep_traces``.

    Returns:
        Dict with all intermediate and final results.
    """
    os.makedirs(output_dir, exist_ok=True)
    from megatron.megalens.paper_style import apply_global_rcparams
    apply_global_rcparams()
    report_file = os.path.join(output_dir, "hybrid_diagnostic_report.txt")
    logger = _ReportLogger(
        report_file, "MegaLens Hybrid Parallelism Collaborative Diagnostic Report"
    )

    logger.log("[Hybrid Analyzer] Loading trace data...")
    loader = TraceDataLoader.from_traces(traces)

    analyzer = HybridAnalyzer(
        loader,
        pp_result=pp_result,
        dp_result=dp_result,
        tp_result=tp_result,
        ep_result=ep_result,
    )

    ps = analyzer.parallel_sizes
    logger.log(
        f"[Hybrid Analyzer] Parallel sizes: "
        f"DP={ps['dp']} PP={ps['pp']} TP={ps['tp']} EP={ps['ep']}"
    )

    logger.log("\n[Step 1/7] EP → PP Bubble Amplification...")
    ep_pp = analyzer.analyze_ep_pp_bubble_amplification()

    logger.log("[Step 2/7] EP / TP Communication Contention...")
    contention = analyzer.analyze_ep_tp_comm_contention()

    logger.log("[Step 3/7] EP Load → DP Step Jitter Coupling...")
    ep_dp = analyzer.analyze_ep_dp_load_coupling()

    logger.log("[Step 4/7] PP Bubble ↔ DP Grad-Sync Serialization...")
    pp_dp = analyzer.analyze_pp_dp_sync_serialization()

    logger.log("[Step 5/7] TP Fragmentation → EP Expert Compute Slowdown...")
    tp_ep = analyzer.analyze_tp_ep_compute_fragmentation()

    logger.log("[Step 6/7] Global Straggler Diagnosis...")
    stragglers = analyzer.diagnose_global_straggler()

    logger.log("[Step 7/7] Hybrid Health Score...")
    health = analyzer.compute_hybrid_health_score(
        ep_pp, contention, ep_dp, pp_dp, tp_ep, stragglers
    )

    _write_hybrid_report(logger, ep_pp, contention, ep_dp, pp_dp, tp_ep, stragglers, health)

    # ---- Visualizations ----
    logger.log("\n[Hybrid Analyzer] Generating visualizations...")
    _generate_ep_pp_plot(ep_pp, output_dir)
    _generate_ep_tp_contention_plot(contention, output_dir)
    _generate_ep_dp_coupling_plot(ep_dp, output_dir)
    _generate_pp_dp_serialization_plot(pp_dp, output_dir)
    _generate_global_straggler_plot(stragglers, output_dir)
    _generate_health_score_plot(health, output_dir)

    # ---- CSV export ----
    logger.log("[Hybrid Analyzer] Exporting CSVs...")
    if contention:
        pd.DataFrame(contention).to_csv(
            os.path.join(output_dir, "hybrid_ep_tp_contention.csv"), index=False
        )
    if pp_dp:
        pd.DataFrame(pp_dp).to_csv(
            os.path.join(output_dir, "hybrid_pp_dp_serialization.csv"), index=False
        )
    if stragglers:
        pd.DataFrame(stragglers).to_csv(
            os.path.join(output_dir, "hybrid_global_straggler.csv"), index=False
        )

    # ---- per-iter summary CSV ----
    summary_rows: List[Dict[str, Any]] = []
    all_iters = sorted(
        set(ep_pp.get("_common_iters", []))
        | set(ep_dp.get("_common_iters", []))
    )
    ep_cv_map = dict(zip(ep_pp.get("_common_iters", []),
                         ep_pp.get("_ep_cv_series", [])))
    pp_bub_map = dict(zip(ep_pp.get("_common_iters", []),
                          ep_pp.get("_pp_bubble_series", [])))
    ep_cv_dp_map = dict(zip(ep_dp.get("_common_iters", []),
                            ep_dp.get("_ep_cv_series", [])))
    dp_cv_map = dict(zip(ep_dp.get("_common_iters", []),
                         ep_dp.get("_dp_cv_series", [])))
    for it in all_iters:
        summary_rows.append({
            "iteration": it,
            "ep_expert_cv": ep_cv_map.get(it, None),
            "pp_bubble_rate": pp_bub_map.get(it, None),
            "dp_step_cv": dp_cv_map.get(it, None),
        })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(output_dir, "hybrid_summary.csv"), index=False
        )

    logger.log(f"\n[Hybrid Analyzer] Complete. Report -> {report_file}")
    logger.log(f"[Hybrid Analyzer] Health Score = {health['score']}/100 (Grade {health['grade']})")

    return {
        "ep_pp": ep_pp,
        "contention": contention,
        "ep_dp": ep_dp,
        "pp_dp": pp_dp,
        "tp_ep": tp_ep,
        "stragglers": stragglers,
        "health": health,
    }
