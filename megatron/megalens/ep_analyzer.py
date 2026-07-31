"""Expert Parallelism (EP) Inefficiency Analyzer.

Diagnoses EP-specific performance issues in MoE (Mixture of Experts) layers:

1. **Expert Load Balance** — detects hot experts and skewed token routing.
2. **EP Communication Overhead** — quantifies All-to-All dispatch/combine cost.
3. **Comm-Comp Overlap** — measures how well EP communication is hidden.
4. **Token Dropping** — flags excessive token loss from capacity overflow.
5. **Router Health** — assesses routing collapse risk and stability.
6. **Aux/Z Loss Drift** — detects load-balancing loss anomalies over time.
7. **EP Straggler** — identifies slow ranks within EP groups.

All public methods return JSON-serialisable ``list[dict]`` results suitable
for rendering in a diagnostic report.
"""

from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


from megatron.megalens.data_loader import TraceDataLoader, SpanEvent
from megatron.megalens.paper_style import (
    FIG_W_SINGLE, FIG_W_DOUBLE, FIG_H, FIG_H_TALL, FIG_H_SHORT,
    apply_global_rcparams,
)


def _load_pandas():
    """Load the optional tabular reporting dependency when first required."""
    import pandas as pd

    return pd


def _load_reporting_dependencies():
    """Load optional plotting dependencies when first required."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


# ============================================================================
# Interval Helpers
# ============================================================================

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Greedy interval merging (sorted by start)."""
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
    """Total overlap between two sets of *merged* intervals."""
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


def _safe_cv(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.mean(arr))
    if m <= 0:
        return 0.0
    return float(np.std(arr) / m)


def _safe_top1_share(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0:
        return 0.0
    return float(np.max(arr) / total)


def _safe_entropy(values: List[float], eps: float = 1e-12) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0:
        return 0.0
    p = arr / total
    return float(-(p * np.log(p + eps)).sum())


# ============================================================================
# Thresholds (design doc §9)
# ============================================================================

_EXPERT_CV_INFO: float = 0.20
_EXPERT_CV_WARN: float = 0.30
_EXPERT_CV_CRIT: float = 0.50

_TOP1_SHARE_WARN: float = 0.35
_TOP1_SHARE_CRIT: float = 0.50

_EP_COMM_RATIO_WARN: float = 0.25
_EP_COMM_RATIO_CRIT: float = 0.40

_EP_OVERLAP_WARN: float = 0.50
_EP_OVERLAP_CRIT: float = 0.30

_DROP_RATE_WARN: float = 0.01
_DROP_RATE_CRIT: float = 0.05

_THERMAL_THROTTLE_TEMP_C: float = 80.0
_CLOCK_DROP_RATIO: float = 0.92


# ============================================================================
# EPAnalyzer
# ============================================================================

class EPAnalyzer:
    """Analyses Expert-Parallelism inefficiencies from a loaded trace.

    Args:
        loader: A populated :class:`TraceDataLoader`.
    """

    def __init__(self, loader: TraceDataLoader) -> None:
        self.loader = loader

    def _get_ep_ranks(self) -> List[int]:
        """Return global ranks that have at least one MoE span event."""
        ranks = self.loader.get_ranks()
        ep_ranks: List[int] = []
        for r in ranks:
            if self.loader.get_events_matching("moe-", rank=r):
                ep_ranks.append(r)
        return ep_ranks if ep_ranks else ranks

    # ------------------------------------------------------------------
    # 1. Expert Load Balance
    # ------------------------------------------------------------------

    def analyze_expert_load_balance(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect expert load imbalance from router / dispatch metrics.

        Priority of signal extraction:
        1) ``tokens_per_expert`` array (preferred)
        2) Pre-computed scalars: ``expert_cv``, ``top1_expert_share``

        Returns one record per ``(iteration, layer)``.
        """
        source_events: List[SpanEvent] = []
        # moe-experts is the primary source: it carries tokens_per_expert array
        # (post-dispatch, actual local token counts) plus pre-computed scalars.
        # moe-router / moe-dispatch are secondary: they carry router-side metrics
        # (pre-dispatch estimate from router._last_router_metrics).
        # All three are merged; moe-experts entries take precedence because their
        # tokens_per_expert reflects true post-All-to-All local distribution.
        for name in ("moe-experts", "moe-router", "moe-dispatch"):
            source_events.extend(
                self.loader.get_events_by_name(name, iteration=iteration)
            )

        GroupKey = Tuple[int, int]  # (iteration, layer)
        grouped: Dict[GroupKey, List[SpanEvent]] = collections.defaultdict(list)
        for ev in source_events:
            layer = ev.args.get("layer", -1)
            grouped[(ev.iteration, layer)].append(ev)

        results: List[Dict[str, Any]] = []
        for (iter_id, layer), evs in sorted(grouped.items()):
            cv_vals: List[float] = []
            top1_vals: List[float] = []
            maxmean_vals: List[float] = []
            entropy_vals: List[float] = []
            # Per-rank token load: sum of tokens_per_expert for each rank.
            # This is the global imbalance signal — ranks receiving more tokens
            # take longer in All-to-All and are more likely to be stragglers.
            rank_token_loads: Dict[int, float] = {}
            # Store raw per-rank arrays for distribution visualisation.
            # Key: rank, Value: list of token counts per local expert.
            tpe_by_rank: Dict[int, List[float]] = {}

            for ev in evs:
                tpe = ev.args.get("tokens_per_expert")
                if isinstance(tpe, (list, tuple)) and len(tpe) > 0:
                    # Primary path: full tokens_per_expert array available
                    # (written by moe-experts scope after dispatch_postprocess).
                    # Each rank provides LOCAL experts only (len = num_local_experts).
                    tpe_f = [float(x) for x in tpe]
                    cv_vals.append(_safe_cv(tpe_f))
                    top1_vals.append(_safe_top1_share(tpe_f))
                    entropy_vals.append(_safe_entropy(tpe_f))
                    mean_v = float(np.mean(tpe_f)) if tpe_f else 0.0
                    maxmean_vals.append(
                        float(max(tpe_f) / mean_v) if mean_v > 0 else 0.0
                    )
                    # Rank-level total token load (for cross-rank imbalance)
                    rank_token_loads[ev.rank] = float(sum(tpe_f))
                    # Store the array keyed by rank, last microbatch wins
                    tpe_by_rank[ev.rank] = tpe_f
                else:
                    # Fallback path: only pre-computed scalars available.
                    # NOTE: When only scalars are provided (no tokens_per_expert array),
                    # routing_entropy and expert_max_over_mean cannot be computed and
                    # will be reported as None (not 0.0) to avoid misleading zeros.
                    # This typically occurs when moe-router spans carry expert_cv /
                    # top1_expert_share as args but not the full distribution.
                    if ev.args.get("expert_cv") is not None:
                        cv_vals.append(float(ev.args["expert_cv"]))
                    if ev.args.get("top1_expert_share") is not None:
                        top1_vals.append(float(ev.args["top1_expert_share"]))
                    if ev.args.get("expert_max_over_mean") is not None:
                        maxmean_vals.append(float(ev.args["expert_max_over_mean"]))

            if not cv_vals and not top1_vals:
                continue

            avg_cv: Optional[float] = (
                float(np.mean(cv_vals)) if cv_vals else None
            )
            avg_top1: Optional[float] = (
                float(np.mean(top1_vals)) if top1_vals else None
            )
            # Use None when data is unavailable to avoid misleading zero values.
            # This happens when tokens_per_expert array is absent from the trace
            # and the fallback scalar args don't include max_over_mean / entropy.
            avg_maxmean: Optional[float] = (
                float(np.mean(maxmean_vals)) if maxmean_vals else None
            )
            avg_entropy: Optional[float] = (
                float(np.mean(entropy_vals)) if entropy_vals else None
            )
            # Count unique ranks and events per rank for context
            unique_ranks = len(set(e.rank for e in evs))
            num_events_per_rank = (
                len(evs) // unique_ranks if unique_ranks > 0 else len(evs)
            )

            # Cross-rank token load imbalance: CV of per-rank total token sums.
            # High rank_load_cv means some EP ranks receive significantly more
            # tokens than others — this is the primary cause of EP stragglers.
            rank_load_vals = list(rank_token_loads.values())
            rank_load_cv: Optional[float] = (
                round(_safe_cv(rank_load_vals), 4) if len(rank_load_vals) >= 2 else None
            )
            rank_load_max_over_mean: Optional[float] = None
            if len(rank_load_vals) >= 2:
                rl_arr = np.asarray(rank_load_vals, dtype=np.float64)
                rl_mean = float(rl_arr.mean())
                if rl_mean > 0:
                    rank_load_max_over_mean = round(
                        float(rl_arr.max() / rl_mean), 4
                    )

            if (
                (avg_cv is not None and avg_cv > _EXPERT_CV_CRIT)
                or (avg_top1 is not None and avg_top1 > _TOP1_SHARE_CRIT)
            ):
                severity = "CRITICAL"
            elif (
                (avg_cv is not None and avg_cv > _EXPERT_CV_WARN)
                or (avg_top1 is not None and avg_top1 > _TOP1_SHARE_WARN)
            ):
                severity = "WARNING"
            elif avg_cv is not None and avg_cv > _EXPERT_CV_INFO:
                severity = "INFO"
            else:
                severity = "OK"

            ep_size = next(
                (ev.args["ep_size"] for ev in evs if "ep_size" in ev.args), None
            )
            num_experts = next(
                (ev.args["num_experts"] for ev in evs if "num_experts" in ev.args),
                None,
            )

            results.append({
                "iteration": iter_id,
                "layer": layer,
                "expert_cv": round(avg_cv, 4) if avg_cv is not None else None,
                "top1_expert_share": (
                    round(avg_top1, 4) if avg_top1 is not None else None
                ),
                # None when tokens_per_expert array absent; do NOT interpret 0 as zero-skew.
                "expert_max_over_mean": (
                    round(avg_maxmean, 4) if avg_maxmean is not None else None
                ),
                "routing_entropy": (
                    round(avg_entropy, 6) if avg_entropy is not None else None
                ),
                # Cross-rank load imbalance metrics (global EP view).
                # rank_load_cv: CV of per-rank total token sums; high value means
                # some ranks receive many more tokens than others (EP straggler risk).
                "rank_load_cv": rank_load_cv,
                "rank_load_max_over_mean": rank_load_max_over_mean,
                # Raw per-rank distribution for visualisation (may be None if
                # tokens_per_expert array was not written to the trace).
                "tokens_per_expert_by_rank": tpe_by_rank if tpe_by_rank else None,
                "num_ranks_sampled": len(evs),
                # num_events_per_rank > 1 means CV is averaged over multiple microbatches.
                # Small micro-batch sizes produce naturally high CV; compare against
                # iteration-level aggregates rather than per-microbatch thresholds.
                "num_events_per_rank": num_events_per_rank,
                "ep_size": ep_size,
                "num_experts": num_experts,
                "severity": severity,
            })

        return results

    # ------------------------------------------------------------------
    # 2. EP Communication Overhead
    # ------------------------------------------------------------------

    def analyze_ep_comm_overhead(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Quantify EP All-to-All communication cost per step.

        .. math::

            \\text{ep\\_comm\\_ratio} =
            \\frac{T_{dispatch\\_comm} + T_{combine\\_comm}}
                  {T_{dispatch} + T_{experts} + T_{combine}}

        Returns:
            One record per ``(rank, iteration)``.
        """
        ep_ranks = self._get_ep_ranks()
        results: List[Dict[str, Any]] = []

        for rank in ep_ranks:
            dispatch_by_iter: Dict[int, float] = collections.defaultdict(float)
            experts_by_iter: Dict[int, float] = collections.defaultdict(float)
            combine_by_iter: Dict[int, float] = collections.defaultdict(float)
            dispatch_comm_by_iter: Dict[int, float] = collections.defaultdict(float)
            combine_comm_by_iter: Dict[int, float] = collections.defaultdict(float)

            for ev in self.loader.get_events_by_name(
                "moe-dispatch", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    dispatch_by_iter[ev.iteration] += ev.dur
            for ev in self.loader.get_events_by_name(
                "moe-experts", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    experts_by_iter[ev.iteration] += ev.dur
            for ev in self.loader.get_events_by_name(
                "moe-combine", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    combine_by_iter[ev.iteration] += ev.dur
            for ev in self.loader.get_events_by_name(
                "ep-alltoall-dispatch", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    dispatch_comm_by_iter[ev.iteration] += ev.dur
            for ev in self.loader.get_events_by_name(
                "ep-allgather-dispatch", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    dispatch_comm_by_iter[ev.iteration] += ev.dur
            for ev in self.loader.get_events_by_name(
                "ep-alltoall-combine", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    combine_comm_by_iter[ev.iteration] += ev.dur
            for ev in self.loader.get_events_by_name(
                "ep-allgather-combine", rank=rank, iteration=iteration
            ):
                if ev.dur > 0:
                    combine_comm_by_iter[ev.iteration] += ev.dur

            all_iters = sorted(set(
                list(dispatch_by_iter)
                + list(experts_by_iter)
                + list(combine_by_iter)
            ))

            for it in all_iters:
                disp = dispatch_by_iter.get(it, 0.0)
                exp = experts_by_iter.get(it, 0.0)
                comb = combine_by_iter.get(it, 0.0)
                disp_comm = dispatch_comm_by_iter.get(it, 0.0)
                comb_comm = combine_comm_by_iter.get(it, 0.0)

                moe_window = disp + exp + comb
                ep_comm = disp_comm + comb_comm
                if moe_window <= 0:
                    continue

                ep_comm_ratio = ep_comm / moe_window
                comm_comp_ratio = ep_comm / exp if exp > 0 else 0.0
                dispatch_share = disp_comm / ep_comm if ep_comm > 0 else 0.0

                if ep_comm_ratio > _EP_COMM_RATIO_CRIT:
                    severity = "CRITICAL"
                elif ep_comm_ratio > _EP_COMM_RATIO_WARN:
                    severity = "WARNING"
                else:
                    severity = "OK"

                results.append({
                    "rank": rank,
                    "iteration": it,
                    "dispatch_dur_us": round(disp, 1),
                    "experts_dur_us": round(exp, 1),
                    "combine_dur_us": round(comb, 1),
                    "dispatch_comm_us": round(disp_comm, 1),
                    "combine_comm_us": round(comb_comm, 1),
                    "moe_window_us": round(moe_window, 1),
                    "ep_comm_us": round(ep_comm, 1),
                    "ep_comm_ratio": round(ep_comm_ratio, 4),
                    "comm_comp_ratio": round(comm_comp_ratio, 4),
                    "dispatch_share": round(dispatch_share, 4),
                    "severity": severity,
                })

        return results

    # ------------------------------------------------------------------
    # 3. EP Comm-Comp Overlap
    # ------------------------------------------------------------------

    def analyze_ep_overlap(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Measure overlap between EP communication and expert computation.

        Communication intervals come from EP dispatch/combine events
        (``ep-alltoall-*`` and ``ep-allgather-*``).  Computation intervals
        come from ``moe-experts`` and ``moe-shared-expert``.

        Returns:
            One record per ``(rank, iteration)`` with ``overlap_ratio``.
        """
        ep_ranks = self._get_ep_ranks()
        comm_names = [
            "ep-alltoall-dispatch",
            "ep-alltoall-combine",
            "ep-allgather-dispatch",
            "ep-allgather-combine",
        ]
        comp_names = ["moe-experts", "moe-shared-expert"]

        results: List[Dict[str, Any]] = []
        for rank in ep_ranks:
            comm_by_iter: Dict[int, List[Tuple[int, int]]] = (
                collections.defaultdict(list)
            )
            comp_by_iter: Dict[int, List[Tuple[int, int]]] = (
                collections.defaultdict(list)
            )

            for cname in comm_names:
                for ev in self.loader.get_events_by_name(
                    cname, rank=rank, iteration=iteration
                ):
                    if ev.dur > 0:
                        comm_by_iter[ev.iteration].append((ev.ts, ev.end_ts))

            for cname in comp_names:
                for ev in self.loader.get_events_by_name(
                    cname, rank=rank, iteration=iteration
                ):
                    if ev.dur > 0:
                        comp_by_iter[ev.iteration].append((ev.ts, ev.end_ts))

            all_iters = sorted(set(comm_by_iter) | set(comp_by_iter))
            for it in all_iters:
                comm_merged = _merge_intervals(comm_by_iter.get(it, []))
                comp_merged = _merge_intervals(comp_by_iter.get(it, []))

                total_comm = _total_length(comm_merged)
                total_comp = _total_length(comp_merged)
                overlap = _overlap_length(comm_merged, comp_merged)
                exposed = max(0, total_comm - overlap)
                if total_comm > 0:
                    ratio: Optional[float] = overlap / total_comm
                    if ratio < _EP_OVERLAP_CRIT:
                        severity = "CRITICAL"
                    elif ratio < _EP_OVERLAP_WARN:
                        severity = "WARNING"
                    else:
                        severity = "OK"
                else:
                    ratio = None
                    severity = "N/A"

                results.append({
                    "rank": rank,
                    "iteration": it,
                    "total_comm_us": float(total_comm),
                    "total_comp_us": float(total_comp),
                    "overlap_us": float(overlap),
                    "exposed_comm_us": float(exposed),
                    "overlap_ratio": (
                        round(ratio, 4) if ratio is not None else None
                    ),
                    # NOTE: overlap_ratio=0.0 does NOT always mean "no overlap attempted".
                    # When --overlap-moe-expert-parallel-comm is False (Megatron default),
                    # comm and comp run sequentially on the same CUDA stream even if
                    # --moe-shared-expert-overlap is True; the latter only schedules
                    # shared-expert work before waiting on the All-to-All but does NOT
                    # use a separate CUDA stream. True GPU-level overlap requires
                    # --overlap-moe-expert-parallel-comm to be enabled.
                    "severity": severity,
                })

        return results

    # ------------------------------------------------------------------
    # 4. Token Dropping
    # ------------------------------------------------------------------

    def analyze_token_dropping(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect excessive token dropping in MoE dispatch.

        Reads ``dropped_tokens`` / ``drop_rate`` from ``moe-dispatch`` or
        ``moe-router`` event args and groups by ``(iteration, layer)``.

        Returns:
            One record per ``(iteration, layer)`` with severity flags.
        """
        source_events: List[SpanEvent] = []
        for name in ("moe-dispatch", "moe-router"):
            source_events.extend(
                self.loader.get_events_by_name(name, iteration=iteration)
            )

        GroupKey = Tuple[int, int]
        grouped: Dict[GroupKey, List[SpanEvent]] = collections.defaultdict(list)
        for ev in source_events:
            if "drop_rate" in ev.args or "dropped_tokens" in ev.args:
                layer = ev.args.get("layer", -1)
                grouped[(ev.iteration, layer)].append(ev)

        results: List[Dict[str, Any]] = []
        for (iter_id, layer), evs in sorted(grouped.items()):
            drop_rates = [
                ev.args["drop_rate"]
                for ev in evs
                if ev.args.get("drop_rate") is not None
            ]
            dropped = [
                ev.args["dropped_tokens"]
                for ev in evs
                if ev.args.get("dropped_tokens") is not None
            ]
            routed = [
                ev.args["num_tokens"]
                for ev in evs
                if ev.args.get("num_tokens") is not None
            ]

            if not drop_rates:
                continue

            avg_rate = float(np.mean(drop_rates))
            total_dropped = int(sum(dropped)) if dropped else None
            total_routed = int(sum(routed)) if routed else None

            if avg_rate > _DROP_RATE_CRIT:
                severity = "CRITICAL"
            elif avg_rate > _DROP_RATE_WARN:
                severity = "WARNING"
            else:
                severity = "OK"

            cap_factor = next(
                (ev.args["capacity_factor"] for ev in evs
                 if "capacity_factor" in ev.args),
                None,
            )
            topk = next(
                (ev.args["router_topk"] for ev in evs
                 if "router_topk" in ev.args),
                None,
            )

            results.append({
                "iteration": iter_id,
                "layer": layer,
                "avg_drop_rate": round(avg_rate, 6),
                "total_dropped_tokens": total_dropped,
                "total_routed_tokens": total_routed,
                "capacity_factor": cap_factor,
                "router_topk": topk,
                "num_ranks_sampled": len(evs),
                "severity": severity,
            })

        return results

    # ------------------------------------------------------------------
    # 5. Router Health (composite)
    # ------------------------------------------------------------------

    def analyze_router_health(
        self,
        balance_data: List[Dict[str, Any]],
        drop_data: List[Dict[str, Any]],
        loss_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Composite router health assessment per iteration.

        Fuses three signal families:
        1) distribution sharpness (CV, Top1 share, entropy)
        2) trend deterioration (CV slope)
        3) optimisation signal health (Aux loss level / slope)

        State machine output:
            ``Healthy`` → ``Mild Skew`` → ``Router Collapse Risk``
        """
        bal_by_iter: Dict[int, List[Dict]] = collections.defaultdict(list)
        for d in balance_data:
            bal_by_iter[d["iteration"]].append(d)

        drop_by_iter: Dict[int, float] = {}
        for d in drop_data:
            it = d["iteration"]
            drop_rate = d.get("avg_drop_rate")
            if drop_rate is not None:
                drop_by_iter[it] = max(drop_by_iter.get(it, 0.0), drop_rate)

        loss_by_iter: Dict[int, Dict[str, float]] = {}
        for d in loss_data:
            loss_by_iter[d["iteration"]] = {
                "aux": float(d.get("aux_loss_mean", 0.0)),
                "aux_slope": float(d.get("aux_slope", 0.0)),
                "z": float(d.get("z_loss_mean", 0.0)),
                "z_slope": float(d.get("z_slope", 0.0)),
            }

        all_iters = sorted(set(bal_by_iter) | set(drop_by_iter) | set(loss_by_iter))

        cv_series: List[float] = []
        results: List[Dict[str, Any]] = []
        for it in all_iters:
            bal = bal_by_iter.get(it, [])
            cv_vals = [
                b["expert_cv"] for b in bal
                if b.get("expert_cv") is not None
            ]
            top1_vals = [
                b["top1_expert_share"] for b in bal
                if b.get("top1_expert_share") is not None
            ]
            entropy_vals = [
                b["routing_entropy"] for b in bal
                if b.get("routing_entropy") is not None
            ]
            avg_cv: Optional[float] = (
                float(np.mean(cv_vals)) if cv_vals else None
            )
            avg_top1: Optional[float] = (
                float(np.mean(top1_vals)) if top1_vals else None
            )
            avg_entropy: Optional[float] = (
                float(np.mean(entropy_vals)) if entropy_vals else None
            )
            dr: Optional[float] = drop_by_iter.get(it)
            loss = loss_by_iter.get(it, {"aux": 0.0, "aux_slope": 0.0, "z": 0.0, "z_slope": 0.0})
            aux, aux_slope = loss["aux"], loss["aux_slope"]
            z_mean, z_slope = loss["z"], loss["z_slope"]

            if avg_cv is not None:
                cv_series.append(avg_cv)
                cv_slope: Optional[float] = self._linear_slope(cv_series)
            else:
                cv_slope = None

            score = 0
            reasons: List[str] = []

            # Distribution sharpness
            if avg_cv is not None and avg_cv > _EXPERT_CV_CRIT:
                score += 3
                reasons.append(f"expert_cv={avg_cv:.3f} (>{_EXPERT_CV_CRIT})")
            elif avg_cv is not None and avg_cv > _EXPERT_CV_WARN:
                score += 2
                reasons.append(f"expert_cv={avg_cv:.3f} (>{_EXPERT_CV_WARN})")

            if avg_top1 is not None and avg_top1 > _TOP1_SHARE_CRIT:
                score += 3
                reasons.append(f"top1_share={avg_top1:.3f} (>{_TOP1_SHARE_CRIT})")
            elif avg_top1 is not None and avg_top1 > _TOP1_SHARE_WARN:
                score += 2
                reasons.append(f"top1_share={avg_top1:.3f} (>{_TOP1_SHARE_WARN})")

            # Trend deterioration
            if cv_slope is not None and cv_slope > 0.003:
                score += 1
                reasons.append(f"cv_slope={cv_slope:.5f} (rising)")

            # Capacity stress
            if dr is not None and dr > _DROP_RATE_CRIT:
                score += 2
                reasons.append(f"drop_rate={dr:.4f} (>{_DROP_RATE_CRIT})")
            elif dr is not None and dr > _DROP_RATE_WARN:
                score += 1
                reasons.append(f"drop_rate={dr:.4f} (>{_DROP_RATE_WARN})")

            # Optimisation signal health
            if aux > 0.0 and aux_slope > 0.001:
                score += 1
                reasons.append(f"aux_loss rising (slope={aux_slope:.5f})")
            if z_mean > 0 and z_slope > 0.01:
                score += 1
                reasons.append(f"z_loss spike risk (slope={z_slope:.5f})")

            if score >= 7:
                status = "Router Collapse Risk"
            elif score >= 3:
                status = "Mild Skew"
            else:
                status = "Healthy"

            results.append({
                "iteration": it,
                "status": status,
                "risk_score": score,
                "expert_cv": round(avg_cv, 4) if avg_cv is not None else None,
                "top1_expert_share": (
                    round(avg_top1, 4) if avg_top1 is not None else None
                ),
                "routing_entropy": (
                    round(avg_entropy, 6) if avg_entropy is not None else None
                ),
                "cv_slope": (
                    round(cv_slope, 8) if cv_slope is not None else None
                ),
                "drop_rate": round(dr, 6) if dr is not None else None,
                "aux_loss": round(aux, 6),
                "aux_slope": round(aux_slope, 8),
                "z_loss": round(z_mean, 6),
                "z_slope": round(z_slope, 8),
                "reasons": reasons,
            })

        return results

    # ------------------------------------------------------------------
    # 6. Aux / Z Loss Drift
    # ------------------------------------------------------------------

    def analyze_router_loss_drift(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect aux-loss and z-loss drift over iterations.

        Builds a per-iteration time series from ``moe-router`` event args
        (``aux_loss``, ``z_loss``) and ``EP_Metrics`` counter events.
        Reports rolling mean, standard deviation, and linear trend slope.
        """
        source_events = self.loader.get_events_by_name(
            "moe-router", iteration=iteration
        )

        aux_by_iter: Dict[int, List[float]] = collections.defaultdict(list)
        z_by_iter: Dict[int, List[float]] = collections.defaultdict(list)

        stats_by_iter: Dict[int, Dict[str, int]] = collections.defaultdict(
            lambda: {
                "router_events": 0,
                "aux_valid": 0,
                "aux_missing_or_invalid": 0,
                "z_valid": 0,
                "z_missing_or_invalid": 0,
                "aux_counter_windows": 0,
                "z_counter_windows": 0,
            }
        )

        for ev in source_events:
            it = ev.iteration
            stats_by_iter[it]["router_events"] += 1

            aux_v = ev.args.get("aux_loss")
            if aux_v is not None:
                try:
                    aux_by_iter[it].append(float(aux_v))
                    stats_by_iter[it]["aux_valid"] += 1
                except (TypeError, ValueError):
                    stats_by_iter[it]["aux_missing_or_invalid"] += 1
            else:
                stats_by_iter[it]["aux_missing_or_invalid"] += 1

            z_v = ev.args.get("z_loss")
            if z_v is not None:
                try:
                    z_by_iter[it].append(float(z_v))
                    stats_by_iter[it]["z_valid"] += 1
                except (TypeError, ValueError):
                    stats_by_iter[it]["z_missing_or_invalid"] += 1
            else:
                stats_by_iter[it]["z_missing_or_invalid"] += 1

        # Supplement from counter events inside each iteration window.
        for rank in self.loader.get_ranks():
            for ie in self.loader.get_events_by_name(
                "iteration", rank=rank, iteration=iteration
            ):
                if ie.dur <= 0:
                    continue
                aux_res = self.loader.get_hardware_metrics_in_window(
                    ie.ts, ie.end_ts, rank, "EP_Aux_Loss",
                )
                z_res = self.loader.get_hardware_metrics_in_window(
                    ie.ts, ie.end_ts, rank, "EP_Z_Loss",
                )
                if aux_res is not None:
                    aux_by_iter[ie.iteration].append(aux_res.mean)
                    stats_by_iter[ie.iteration]["aux_counter_windows"] += 1
                if z_res is not None:
                    z_by_iter[ie.iteration].append(z_res.mean)
                    stats_by_iter[ie.iteration]["z_counter_windows"] += 1

        all_iters = sorted(set(aux_by_iter) | set(z_by_iter))
        if not all_iters:
            return []

        results: List[Dict[str, Any]] = []
        aux_series: List[float] = []
        z_series: List[float] = []

        for it in all_iters:
            aux_vals = aux_by_iter.get(it, [])
            z_vals = z_by_iter.get(it, [])
            aux_mean = float(np.mean(aux_vals)) if aux_vals else 0.0
            z_mean = float(np.mean(z_vals)) if z_vals else 0.0
            aux_series.append(aux_mean)
            z_series.append(z_mean)

            window = min(5, len(aux_series))
            rolling_aux_mean = float(np.mean(aux_series[-window:]))
            rolling_aux_std = (
                float(np.std(aux_series[-window:])) if window > 1 else 0.0
            )

            aux_slope = self._linear_slope(aux_series)
            z_slope = self._linear_slope(z_series)

            risk_flags: List[str] = []
            if aux_slope > 0 and rolling_aux_std > rolling_aux_mean * 0.3:
                risk_flags.append("aux_loss_unstable")
            if aux_slope > 0.001:
                risk_flags.append("aux_loss_rising")
            if z_mean > 0 and z_slope > 0.01:
                risk_flags.append("z_loss_spike_risk")

            s = stats_by_iter.get(it, {})
            results.append({
                "iteration": it,
                "aux_loss_mean": round(aux_mean, 6),
                "z_loss_mean": round(z_mean, 6),
                "aux_rolling_mean": round(rolling_aux_mean, 6),
                "aux_rolling_std": round(rolling_aux_std, 6),
                "aux_slope": round(aux_slope, 8),
                "z_slope": round(z_slope, 8),
                "risk_flags": risk_flags,
                "router_events": int(s.get("router_events", 0)),
                "aux_valid_count": int(s.get("aux_valid", 0)),
                "aux_missing_or_invalid_count": int(s.get("aux_missing_or_invalid", 0)),
                "z_valid_count": int(s.get("z_valid", 0)),
                "z_missing_or_invalid_count": int(s.get("z_missing_or_invalid", 0)),
                "aux_counter_windows": int(s.get("aux_counter_windows", 0)),
                "z_counter_windows": int(s.get("z_counter_windows", 0)),
            })

        return results

    @staticmethod
    def _linear_slope(series: List[float], tail: int = 10) -> float:
        """Least-squares slope over the last *tail* samples."""
        if len(series) < 3:
            return 0.0
        recent = series[-min(tail, len(series)):]
        x = np.arange(len(recent), dtype=np.float64)
        if np.std(x) == 0:
            return 0.0
        return float(np.polyfit(x, recent, 1)[0])

    # ------------------------------------------------------------------
    # 7. EP Group Straggler Diagnosis
    # ------------------------------------------------------------------

    def diagnose_ep_stragglers(
        self,
        iteration: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect stragglers within EP groups at dispatch/combine sync points.

        Ranks sharing the same ``(dp_rank, pp_rank)`` form an EP group.
        The method compares completion timestamps and correlates the
        slowest rank with hardware metrics.

        Returns:
            One record per ``(iteration, sync_event, ep_group)``.
        """
        ep_ranks = self._get_ep_ranks()
        if len(ep_ranks) < 2:
            return []

        sync_names = [
            "ep-alltoall-dispatch",
            "ep-alltoall-combine",
            "ep-allgather-dispatch",
            "ep-allgather-combine",
            "moe-dispatch",
            "moe-combine",
        ]

        # Group key: (iteration, event_name, dp_rank, pp_rank).
        # This correctly identifies EP groups when expert_tensor_parallel_size=1
        # (ETP=1): all EP ranks within a (dp, pp) group share the same expert
        # shards and synchronise at All-to-All barriers.
        # NOTE: When ETP > 1, multiple TP ranks co-own each expert shard and the
        # EP group boundary changes. In that case an explicit ep_group_id field
        # from the trace args should be used instead of (dp, pp).
        GroupKey = Tuple[int, str, int, int]
        grouped: Dict[GroupKey, Dict[int, SpanEvent]] = (
            collections.defaultdict(dict)
        )

        for rank in ep_ranks:
            for sname in sync_names:
                for ev in self.loader.get_events_by_name(
                    sname, rank=rank, iteration=iteration
                ):
                    dp = ev.dp_rank
                    pp = ev.pp_rank
                    key: GroupKey = (ev.iteration, ev.name, dp, pp)
                    if (rank not in grouped[key]
                            or ev.end_ts > grouped[key][rank].end_ts):
                        grouped[key][rank] = ev

        results: List[Dict[str, Any]] = []
        for (iter_id, ev_name, dp, pp), rank_events in sorted(grouped.items()):
            if len(rank_events) < 2:
                continue

            end_times = {r: ev.end_ts for r, ev in rank_events.items()}
            min_end = min(end_times.values())
            max_end = max(end_times.values())
            gap_us = float(max_end - min_end)

            straggler_rank = max(end_times, key=end_times.get)  # type: ignore[arg-type]
            fastest_rank = min(end_times, key=end_times.get)    # type: ignore[arg-type]

            hw_diag, hw_detail = self._correlate_hardware(
                straggler_rank, rank_events[straggler_rank],
            )

            cause = "unknown"
            straggler_ev = rank_events[straggler_rank]
            if hw_diag:
                cause = "hardware"
            elif ev_name in (
                "ep-alltoall-dispatch",
                "ep-alltoall-combine",
                "ep-allgather-dispatch",
                "ep-allgather-combine",
            ):
                # For EP communication events, expert_cv is not in args.
                # Diagnose based on gap magnitude: large gap implies token count
                # asymmetry between EP ranks (caused by routing skew) or network
                # load imbalance; small gap is likely minor token count variation.
                if gap_us > 500:
                    cause = "network_or_load"
                elif gap_us > 0:
                    cause = "token_imbalance"  # token counts differ between EP ranks
            elif ev_name in ("moe-dispatch", "moe-combine"):
                # For dispatch/combine span events, expert_cv in args reflects
                # the routing distribution of the straggler rank's microbatch.
                if straggler_ev.args.get("expert_cv", 0) > _EXPERT_CV_WARN:
                    cause = "routing_skew"
                elif gap_us > 500:
                    cause = "network_or_load"

            results.append({
                "iteration": iter_id,
                "sync_event": ev_name,
                "ep_group": f"DP{dp}-PP{pp}",
                "gap_us": gap_us,
                "straggler_rank": straggler_rank,
                "fastest_rank": fastest_rank,
                "per_rank_end_ts": end_times,
                "likely_cause": cause,
                "hardware_diagnosis": hw_diag,
                "hw_detail": hw_detail,
            })

        return results

    def _correlate_hardware(
        self,
        rank: int,
        event: SpanEvent,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Check HW metrics in a 2 ms window before *event* ends."""
        window_end = event.end_ts
        window_start = max(0, window_end - 2000)

        temp = self.loader.get_hardware_metrics_in_window(
            window_start, window_end, rank, "Temp_C",
        )
        clock = self.loader.get_hardware_metrics_in_window(
            window_start, window_end, rank, "SM_Clock_MHz",
        )
        base_clock = self.loader.get_hardware_metrics_in_window(
            window_start, window_end, rank, "SM_Base_Clock_MHz",
        )

        if temp is None and clock is None:
            return None, None

        detail: Dict[str, Any] = {}
        parts: List[str] = []

        if temp is not None:
            detail["Temp_C_peak"] = temp.peak
            detail["Temp_C_mean"] = round(temp.mean, 1)
        if clock is not None:
            detail["SM_Clock_MHz_mean"] = round(clock.mean, 1)

        base_mhz = (
            base_clock.mean if base_clock and base_clock.mean > 0 else 0.0
        )
        if base_mhz > 0:
            detail["SM_Base_Clock_MHz"] = round(base_mhz, 1)

        is_hot = temp is not None and temp.peak >= _THERMAL_THROTTLE_TEMP_C
        is_throttled = (
            clock is not None
            and base_mhz > 0
            and clock.mean < base_mhz * _CLOCK_DROP_RATIO
        )

        if is_hot and is_throttled:
            parts.append(
                f"Thermal Throttling: {temp.peak:.0f}\u00b0C, "
                f"Clock {clock.mean:.0f}/{base_mhz:.0f} MHz"
            )
        elif is_throttled:
            parts.append(
                f"Clock Throttling: {clock.mean:.0f}/{base_mhz:.0f} MHz"
            )
        elif is_hot:
            parts.append(f"High Temp: {temp.peak:.0f}\u00b0C")

        diag = "; ".join(parts) if parts else None
        return diag, detail if detail else None


# ============================================================================
# Report Logger (shared pattern with dp_analyzer / pp_analyzer)
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
# Visualization — Expert Load Balance
# ============================================================================

def generate_expert_balance_plots(
    balance_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """3-panel chart: CV over iterations, Top-1 share, severity distribution."""
    if not balance_data:
        print("[EP Balance] No balance data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style()
    df = pd.DataFrame(balance_data)
    layers = sorted(df["layer"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(layers))))
    layer_color = {ly: colors[i % 10] for i, ly in enumerate(layers)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: Expert Load Balance Analysis",
        fontsize=22, fontweight="bold",
    )

    # Panel 1: Expert CV over iterations (per layer)
    ax = axes[0]
    for ly in layers:
        sub = df[df["layer"] == ly].sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["expert_cv"],
            marker="o", markersize=4, linewidth=1.2, alpha=0.8,
            label=f"Layer {ly}", color=layer_color[ly],
        )
    ax.axhline(y=_EXPERT_CV_WARN, color="orange", linestyle="--",
               linewidth=2, label=f"Warning ({_EXPERT_CV_WARN})")
    ax.axhline(y=_EXPERT_CV_CRIT, color="red", linestyle="--",
               linewidth=2, label=f"Critical ({_EXPERT_CV_CRIT})")
    ax.set_title("Expert CV Over Iterations\n(Higher = More Imbalanced)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("Expert CV", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Top-1 Expert Share over iterations
    ax = axes[1]
    for ly in layers:
        sub = df[df["layer"] == ly].sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["top1_expert_share"],
            marker="s", markersize=4, linewidth=1.2, alpha=0.8,
            label=f"Layer {ly}", color=layer_color[ly],
        )
    ax.axhline(y=_TOP1_SHARE_WARN, color="orange", linestyle="--",
               linewidth=2, label=f"Warning ({_TOP1_SHARE_WARN})")
    ax.set_title("Top-1 Expert Share Over Iterations\n(Hot expert ratio)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("Top-1 Share", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: CV heatmap (layer x iteration)
    ax = axes[2]
    pivot = df.pivot_table(index="layer", columns="iteration", values="expert_cv", aggfunc="mean")
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_title("Expert CV Heatmap\n(Hot = More Skewed)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Layer", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    # Show sparse x ticks for readability
    cols = list(pivot.columns)
    if cols:
        tick_idx = np.linspace(0, len(cols) - 1, num=min(8, len(cols)), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(cols[i]) for i in tick_idx], rotation=30)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_expert_balance.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Balance] Plot saved to {path}")


# ============================================================================
# Visualization — Token Distribution per Expert
# ============================================================================

def generate_token_distribution_plots(
    balance_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """3-panel token distribution chart.

    Requires ``tokens_per_expert_by_rank`` in balance_data (written when
    the ``moe-experts`` span includes the ``tokens_per_expert`` list arg).

    Panels:
        A) Boxplot: per-expert token count distribution across iterations
           (one box per local expert, coloured by EP rank).
        B) Heatmap: EP rank × expert_id, average token count (global view).
        C) Line chart: per-rank total token load over iterations (rank_load_cv).
    """
    # Collect records that have the raw per-rank arrays
    records_with_tpe = [
        r for r in balance_data
        if r.get("tokens_per_expert_by_rank")
    ]
    if not records_with_tpe:
        print(
            "[EP Token Distribution] No tokens_per_expert_by_rank data available. "
            "Ensure moe-experts spans include tokens_per_expert list arg."
        )
        return

    # ---- Collect data structures ----------------------------------------
    # rank -> expert_local_idx -> list of token counts across (iter, layer)
    from collections import defaultdict
    rank_expert_tokens: Dict[int, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # rank -> iteration -> total token load
    rank_iter_load: Dict[int, Dict[int, float]] = defaultdict(dict)

    for rec in records_with_tpe:
        it = rec["iteration"]
        tpe_by_rank: Dict[str, List[float]] = rec["tokens_per_expert_by_rank"]
        for rank_key, tpe_list in tpe_by_rank.items():
            rank = int(rank_key)
            for local_idx, tok_count in enumerate(tpe_list):
                rank_expert_tokens[rank][local_idx].append(tok_count)
            rank_iter_load[rank][it] = float(sum(tpe_list))

    ranks = sorted(rank_expert_tokens.keys())
    if not ranks:
        return

    plt, _ = _load_reporting_dependencies()
    _pick_style()
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: Token Distribution per Expert",
        fontsize=22, fontweight="bold",
    )

    # ---- Panel A: Boxplot per local expert (coloured by rank) -----------
    ax = axes[0]
    n_experts_per_rank = max(
        len(rank_expert_tokens[r]) for r in ranks
    )
    # Offset boxes slightly for each rank so they don't overlap
    width = 0.8 / max(len(ranks), 1)
    for ri, rank in enumerate(ranks):
        expert_data = [
            rank_expert_tokens[rank].get(ei, [0.0])
            for ei in range(n_experts_per_rank)
        ]
        positions = [
            ei + (ri - len(ranks) / 2.0 + 0.5) * width
            for ei in range(n_experts_per_rank)
        ]
        bp = ax.boxplot(
            expert_data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=rank_color[rank], alpha=0.6),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            flierprops=dict(marker="x", markersize=3, alpha=0.4),
            showfliers=True,
        )
        # Legend proxy
        ax.plot([], [], color=rank_color[rank], linewidth=8,
                alpha=0.6, label=f"Rank {rank}")
    ax.set_title(
        "Token Count per Local Expert\n(distribution across iterations)",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("Local Expert Index", fontsize=16)
    ax.set_ylabel("Tokens Received", fontsize=16)
    ax.set_xticks(range(n_experts_per_rank))
    ax.set_xticklabels(
        [str(i) for i in range(n_experts_per_rank)], fontsize=16
    )
    ax.legend(fontsize="small", frameon=True, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    # Draw mean-load reference line
    all_vals = [
        v
        for r in ranks
        for ei_vals in rank_expert_tokens[r].values()
        for v in ei_vals
    ]
    if all_vals:
        grand_mean = float(np.mean(all_vals))
        ax.axhline(
            y=grand_mean, color="grey", linestyle=":",
            linewidth=1.5, label=f"Grand mean ({grand_mean:.0f})",
        )

    # ---- Panel B: Heatmap EP rank × local expert (mean tokens) ---------
    ax = axes[1]
    matrix = np.zeros((len(ranks), n_experts_per_rank))
    for ri, rank in enumerate(ranks):
        for ei in range(n_experts_per_rank):
            vals = rank_expert_tokens[rank].get(ei, [])
            matrix[ri, ei] = float(np.mean(vals)) if vals else 0.0
    im = ax.imshow(
        matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest"
    )
    ax.set_title(
        "Mean Token Load\nEP Rank × Local Expert",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("Local Expert Index", fontsize=16)
    ax.set_ylabel("EP Rank", fontsize=16)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"Rank {r}" for r in ranks], fontsize=9)
    ax.set_xticks(range(n_experts_per_rank))
    ax.set_xticklabels(
        [str(i) for i in range(n_experts_per_rank)], fontsize=8
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean tokens")
    # Annotate cells with values if matrix is small enough
    if n_experts_per_rank * len(ranks) <= 64:
        for ri in range(len(ranks)):
            for ei in range(n_experts_per_rank):
                ax.text(
                    ei, ri, f"{matrix[ri, ei]:.0f}",
                    ha="center", va="center",
                    fontsize=7, color="black",
                )

    # ---- Panel C: Per-rank total token load over iterations ------------
    ax = axes[2]
    all_iters = sorted(
        {it for r in ranks for it in rank_iter_load[r]}
    )
    for rank in ranks:
        loads = [rank_iter_load[rank].get(it, None) for it in all_iters]
        valid_iters = [
            it for it, ld in zip(all_iters, loads) if ld is not None
        ]
        valid_loads = [ld for ld in loads if ld is not None]
        ax.plot(
            valid_iters, valid_loads,
            marker="o", markersize=5, linewidth=1.5,
            color=rank_color[rank], label=f"Rank {rank}",
        )
    # Ideal uniform load reference
    if all_iters and all_vals:
        # Total tokens across all ranks for each iteration
        iter_totals = {
            it: sum(
                rank_iter_load[r].get(it, 0.0) for r in ranks
            )
            for it in all_iters
        }
        uniform_loads = [
            iter_totals.get(it, 0.0) / max(len(ranks), 1)
            for it in all_iters
        ]
        ax.plot(
            all_iters, uniform_loads,
            color="grey", linestyle="--", linewidth=1.5,
            label="Ideal uniform",
        )
    ax.set_title(
        "Per-Rank Total Token Load over Iterations\n"
        "(deviation from uniform = EP straggler risk)",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Total Tokens Received", fontsize=16)
    ax.legend(fontsize=14, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_token_distribution.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Token Distribution] Plot saved to {path}")


# ============================================================================
# Visualization — EP Communication Overhead
# ============================================================================

def generate_ep_comm_plots(
    comm_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """3-panel chart: time breakdown, comm ratio, dispatch vs combine."""
    if not comm_data:
        print("[EP Comm] No communication data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style()
    df = pd.DataFrame(comm_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: EP Communication Overhead",
        fontsize=22, fontweight="bold",
    )

    # Panel 1: Stacked bar — average time breakdown per rank
    ax = axes[0]
    avg = df.groupby("rank").agg({
        "dispatch_comm_us": "mean",
        "combine_comm_us": "mean",
        "experts_dur_us": "mean",
    }).reindex(ranks)
    x = np.arange(len(ranks))
    w = 0.55
    ax.bar(x, avg["experts_dur_us"] / 1e3, w,
           label="Expert Compute", color="mediumseagreen", edgecolor="black")
    ax.bar(x, avg["dispatch_comm_us"] / 1e3, w,
           bottom=avg["experts_dur_us"] / 1e3,
           label="Dispatch All-to-All", color="steelblue", edgecolor="black")
    ax.bar(x, avg["combine_comm_us"] / 1e3, w,
           bottom=(avg["experts_dur_us"] + avg["dispatch_comm_us"]) / 1e3,
           label="Combine All-to-All", color="salmon", hatch="//",
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylabel("Time (ms)", fontsize=16)
    ax.set_xlabel("Global Rank", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Avg MoE Time Breakdown\n(Red = combine comm)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, frameon=True)

    # Panel 2: EP comm ratio over iterations (per rank)
    ax = axes[1]
    for rank in ranks:
        sub = df[df["rank"] == rank].sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["ep_comm_ratio"] * 100,
            marker="o", markersize=3, linewidth=1.2, alpha=0.8,
            label=f"Rank {rank}", color=rank_color[rank],
        )
    ax.axhline(y=_EP_COMM_RATIO_WARN * 100, color="orange", linestyle="--",
               linewidth=2, label=f"Warning ({_EP_COMM_RATIO_WARN:.0%})")
    ax.axhline(y=_EP_COMM_RATIO_CRIT * 100, color="red", linestyle="--",
               linewidth=2, label=f"Critical ({_EP_COMM_RATIO_CRIT:.0%})")
    ax.set_title("EP Comm Ratio Over Iterations\n(% of MoE window in comm)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("EP Comm Ratio (%)", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Dispatch vs Combine share (bar per rank)
    ax = axes[2]
    avg_disp = df.groupby("rank")["dispatch_share"].mean().reindex(ranks)
    avg_comb = 1.0 - avg_disp
    ax.bar(x, avg_disp.values * 100, w,
           label="Dispatch Share", color="steelblue", edgecolor="black")
    ax.bar(x, avg_comb.values * 100, w, bottom=avg_disp.values * 100,
           label="Combine Share", color="salmon", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylabel("Share of EP Comm (%)", fontsize=16)
    ax.set_xlabel("Global Rank", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Dispatch vs Combine Breakdown\n(Which phase dominates?)",
                 fontsize=16, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=12, frameon=True)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_comm_overhead.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Comm] Plot saved to {path}")


# ============================================================================
# Visualization — EP Overlap
# ============================================================================

def generate_ep_overlap_plots(
    overlap_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """3-panel chart: time decomposition, overlap ratio, exposed comm trend."""
    if not overlap_data:
        print("[EP Overlap] No overlap data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style()
    df = pd.DataFrame(overlap_data)
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: Communication-Computation Overlap",
        fontsize=22, fontweight="bold",
    )

    # Panel 1: Stacked bar — compute / overlapped / exposed per rank
    ax = axes[0]
    avg = df.groupby("rank").agg({
        "total_comp_us": "mean",
        "overlap_us": "mean",
        "exposed_comm_us": "mean",
    }).reindex(ranks)
    x = np.arange(len(ranks))
    w = 0.55
    ax.bar(x, avg["total_comp_us"] / 1e3, w,
           label="Expert Compute", color="mediumseagreen", edgecolor="black")
    ax.bar(x, avg["overlap_us"] / 1e3, w,
           bottom=avg["total_comp_us"] / 1e3,
           label="Overlapped Comm (hidden)", color="steelblue",
           edgecolor="black")
    ax.bar(x, avg["exposed_comm_us"] / 1e3, w,
           bottom=(avg["total_comp_us"] + avg["overlap_us"]) / 1e3,
           label="Exposed Comm (blocking)", color="salmon", hatch="//",
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylabel("Time (ms)", fontsize=16)
    ax.set_xlabel("Global Rank", fontsize=16)
    ax.set_title("Avg Time Breakdown per Rank\n(Red = blocking comm)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize="small", frameon=True)

    # Panel 2: Overlap ratio per rank (bar)
    ax = axes[1]
    avg_ratio = df.groupby("rank")["overlap_ratio"].mean().reindex(ranks)
    bar_colors = [rank_color[r] for r in ranks]
    ax.bar([str(r) for r in ranks], avg_ratio.values * 100,
           color=bar_colors, edgecolor="black", alpha=0.85)
    ax.axhline(y=100, color="green", linestyle="--", linewidth=2,
               alpha=0.7, label="Ideal (100%)")
    ax.axhline(y=_EP_OVERLAP_WARN * 100, color="orange", linestyle="--",
               linewidth=2, label=f"Warning ({_EP_OVERLAP_WARN:.0%})")
    ax.set_ylabel("Overlap Ratio (%)", fontsize=16)
    ax.set_xlabel("Global Rank", fontsize=16)
    ax.set_title("Avg Comm Overlap Ratio\n(100% = Fully Hidden)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize="small")
    for i, v in enumerate(avg_ratio.values):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.1f}%",
                ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Exposed comm over iterations (per rank)
    ax = axes[2]
    for rank in ranks:
        sub = df[df["rank"] == rank].sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["exposed_comm_us"] / 1e3,
            marker="o", markersize=4, linewidth=1.2, alpha=0.8,
            label=f"Rank {rank}", color=rank_color[rank],
        )
    ax.set_title("Exposed Comm per Iteration\n(Non-overlapped blocking time)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Exposed Comm (ms)", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_overlap_analysis.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Overlap] Plot saved to {path}")


# ============================================================================
# Visualization — Token Dropping
# ============================================================================

def generate_token_drop_plots(
    drop_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """2-panel chart: drop rate trend and per-layer distribution."""
    if not drop_data:
        print("[EP Token Drop] No token drop data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style()
    df = pd.DataFrame(drop_data)
    layers = sorted(df["layer"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(layers))))
    layer_color = {ly: colors[i % 10] for i, ly in enumerate(layers)}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: Token Dropping Analysis",
        fontsize=22, fontweight="bold",
    )

    # Panel 1: Drop rate over iterations (per layer)
    ax = axes[0]
    for ly in layers:
        sub = df[df["layer"] == ly].sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["avg_drop_rate"] * 100,
            marker="o", markersize=4, linewidth=1.2, alpha=0.8,
            label=f"Layer {ly}", color=layer_color[ly],
        )
    ax.axhline(y=_DROP_RATE_WARN * 100, color="orange", linestyle="--",
               linewidth=2, label=f"Warning ({_DROP_RATE_WARN:.0%})")
    ax.axhline(y=_DROP_RATE_CRIT * 100, color="red", linestyle="--",
               linewidth=2, label=f"Critical ({_DROP_RATE_CRIT:.0%})")
    ax.set_title("Token Drop Rate Over Iterations",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Drop Rate (%)", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Total dropped tokens per layer (bar)
    ax = axes[1]
    layer_drops = (
        df.groupby("layer")["total_dropped_tokens"]
        .sum(min_count=1)
        .reindex(layers)
    )
    cmap = plt.get_cmap("Blues")
    bar_colors = [cmap(0.35 + 0.4*i/len(layers)) for i in range(len(layers))]
    ax.bar(
        [str(ly) for ly in layers], layer_drops.values,
        color=bar_colors, edgecolor="black", alpha=0.85,
    )
    ax.set_title("Total Dropped Tokens per Layer",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Dropped Tokens", fontsize=16)
    ax.set_xlabel("Layer", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_token_drop_analysis.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Token Drop] Plot saved to {path}")


# ============================================================================
# Visualization — Router Health & Loss Drift
# ============================================================================

def generate_router_health_plots(
    health_data: List[Dict[str, Any]],
    loss_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """3-panel chart: health status, risk score, aux/z loss trends."""
    if not health_data and not loss_data:
        print("[EP Router] No router health data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: Router Health & Aux/Z Loss",
        fontsize=22, fontweight="bold",
    )

    # Panel 1: Health status timeline
    ax = axes[0]
    if health_data:
        hdf = pd.DataFrame(health_data)
        status_map = {"Healthy": 0, "Mild Skew": 1, "Router Collapse Risk": 2}
        status_color = {0: "mediumseagreen", 1: "orange", 2: "red"}
        hdf["status_code"] = hdf["status"].map(status_map)
        for code, color in status_color.items():
            mask = hdf["status_code"] == code
            label = [k for k, v in status_map.items() if v == code][0]
            ax.scatter(
                hdf.loc[mask, "iteration"], hdf.loc[mask, "status_code"],
                c=color, s=60, edgecolors="black", linewidth=0.5,
                label=label, zorder=5,
            )
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Healthy", "Mild Skew", "Collapse Risk"])
        ax.legend(fontsize="small", frameon=True)
    ax.set_title("Router Health Status Over Iterations",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Risk score over iterations
    ax = axes[1]
    if health_data:
        hdf = pd.DataFrame(health_data)
        ax.plot(
            hdf["iteration"], hdf["risk_score"],
            marker="s", markersize=4, linewidth=1.5, color="steelblue",
        )
        # Thresholds must match the scoring logic in analyze_router_health:
        # score >= 3 -> Mild Skew, score >= 7 -> Router Collapse Risk
        ax.axhline(y=3, color="orange", linestyle="--", linewidth=2,
                    label="Mild Skew threshold (\u22653)")
        ax.axhline(y=7, color="red", linestyle="--", linewidth=2,
                    label="Collapse Risk threshold (\u22657)")
        ax.legend(fontsize="small")
    ax.set_title("Router Risk Score Over Iterations",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Risk Score", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Aux/Z loss time series
    ax = axes[2]
    if loss_data:
        ldf = pd.DataFrame(loss_data)
        ax.plot(
            ldf["iteration"], ldf["aux_loss_mean"],
            marker="o", markersize=3, linewidth=1.2, color="steelblue",
            label="Aux Loss",
        )
        if ldf["z_loss_mean"].sum() > 0:
            ax2 = ax.twinx()
            ax2.plot(
                ldf["iteration"], ldf["z_loss_mean"],
                marker="^", markersize=3, linewidth=1.2, color="salmon",
                label="Z Loss",
            )
            ax2.set_ylabel("Z Loss", fontsize=16, color="salmon")
            ax2.legend(loc="upper left", fontsize="small")
    ax.set_title("Aux / Z Loss Over Iterations",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Aux Loss", fontsize=16, color="steelblue")
    ax.set_xlabel("Iteration", fontsize=16)
    ax.legend(loc="upper right", fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_router_health.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Router] Plot saved to {path}")


# ============================================================================
# Visualization — EP Straggler
# ============================================================================

def generate_ep_straggler_plots(
    straggler_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """3-panel chart: gap trend, straggler frequency, cause distribution."""
    if not straggler_data:
        print("[EP Straggler] No straggler data to plot.")
        return

    plt, pd = _load_reporting_dependencies()
    _pick_style()
    df = pd.DataFrame(straggler_data)
    ranks = sorted({r for d in straggler_data for r in d["per_rank_end_ts"]})
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color = {r: colors[i % 10] for i, r in enumerate(ranks)}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle(
        "Expert Parallelism: EP Group Straggler Diagnosis",
        fontsize=22, fontweight="bold",
    )

    # Panel 1: Gap over iterations per sync event
    ax = axes[0]
    for sname in df["sync_event"].unique():
        sub = df[df["sync_event"] == sname].sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["gap_us"],
            marker="o", markersize=5, linewidth=1.5, alpha=0.8, label=sname,
        )
    ax.set_title("EP Sync Barrier Gap Over Iterations\n(Higher = Worse)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Gap (\u03bcs)", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.legend(fontsize="small", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Straggler frequency per rank
    ax = axes[1]
    straggler_counts = (
        df["straggler_rank"].value_counts().reindex(ranks, fill_value=0)
    )
    bar_colors = [rank_color[r] for r in straggler_counts.index]
    bars = ax.bar(
        [str(r) for r in straggler_counts.index],
        straggler_counts.values, color=bar_colors,
        edgecolor="black", alpha=0.85,
    )
    if len(straggler_counts) > 0 and straggler_counts.max() > 0:
        worst_idx = int(np.argmax(straggler_counts.values))
        bars[worst_idx].set_edgecolor("red")
        bars[worst_idx].set_linewidth(3)
        bars[worst_idx].set_hatch("//")
    ax.set_title("Straggler Frequency per Rank\n(Tallest = most frequent)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("# Times Straggler", fontsize=16)
    ax.set_xlabel("Global Rank", fontsize=16)

    # Panel 3: Likely cause distribution (pie)
    ax = axes[2]
    cause_counts = df["likely_cause"].value_counts()
    cause_colors = {
        "hardware": "salmon",
        "routing_skew": "orange",
        "network_or_load": "steelblue",
        "unknown": "lightgray",
    }
    pie_colors = [cause_colors.get(c, "lightgray") for c in cause_counts.index]
    wedges, texts, autotexts = ax.pie(
        cause_counts.values,
        labels=cause_counts.index,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Straggler Root-Cause Distribution",
                 fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.92], h_pad=1.5, w_pad=1.5)
    path = os.path.join(output_dir, "ep_straggler_diagnosis.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[EP Straggler] Plot saved to {path}")


# ============================================================================
# Report Writing
# ============================================================================

def _write_expert_balance_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 1: Expert Load Balance")
    logger.log("=" * 90)

    if not data:
        logger.log("  No expert balance data available.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    cv_values = df["expert_cv"].dropna()
    mean_cv = float(cv_values.mean()) if not cv_values.empty else None
    max_cv = float(cv_values.max()) if not cv_values.empty else None
    logger.log(f"  Data points (layer x iteration): {len(df)}")
    if mean_cv is None:
        logger.log("  Mean Expert CV: N/A")
        logger.log("  Max Expert CV:  N/A")
    else:
        logger.log(f"  Mean Expert CV: {mean_cv:.4f}")
        logger.log(f"  Max Expert CV:  {max_cv:.4f}")
    # Warn when CV is averaged over multiple microbatches per rank, which
    # naturally inflates CV values due to small per-microbatch batch sizes.
    if "num_events_per_rank" in df.columns:
        max_eprank = int(df["num_events_per_rank"].max())
        if max_eprank > 1:
            logger.log(
                f"  [NOTE] num_events_per_rank={max_eprank}: CV is averaged over "
                f"{max_eprank} microbatch(es) per rank per (iter, layer). "
                "Small micro-batch sizes produce naturally high per-microbatch CV. "
                "Compare CV trends across iterations rather than absolute thresholds."
            )

    for sev in ("CRITICAL", "WARNING"):
        flagged = df[df["severity"] == sev]
        if not flagged.empty:
            logger.log(f"\n  [{sev}] {len(flagged)} entries:")
            for _, row in flagged.head(10).iterrows():
                expert_cv = row["expert_cv"]
                top1_share = row["top1_expert_share"]
                expert_cv_text = (
                    f"{expert_cv:.4f}" if pd.notna(expert_cv) else "N/A"
                )
                top1_share_text = (
                    f"{top1_share:.4f}" if pd.notna(top1_share) else "N/A"
                )
                logger.log(
                    f"    Iter {row['iteration']}  Layer {row['layer']}  "
                    f"CV={expert_cv_text}  Top1={top1_share_text}"
                )

    if mean_cv is None:
        logger.log(
            "\n  Expert CV is unavailable; no CV-based balance conclusion was made."
        )
    elif mean_cv > _EXPERT_CV_WARN:
        logger.log(
            "\n  [RECOMMENDATION] Expert skew is high. Consider:\n"
            "    - Increasing aux_loss_coeff to penalise imbalance.\n"
            "    - Switching to a more balanced router (e.g. Sinkhorn).\n"
            "    - Increasing capacity_factor if tokens are also being dropped."
        )
    else:
        logger.log("\n  Expert load balance is within healthy range.")

    logger.log("=" * 90 + "\n")


def _write_ep_comm_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 2: EP Communication Overhead")
    logger.log("=" * 90)

    if not data:
        logger.log("  No EP communication data available.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    avg = df.groupby("rank").agg({
        "ep_comm_ratio": "mean",
        "comm_comp_ratio": "mean",
        "dispatch_share": "mean",
        "ep_comm_us": "mean",
        "moe_window_us": "mean",
    })
    avg.columns = [
        "Avg Comm Ratio", "Avg Comm/Comp", "Avg Dispatch Share",
        "Avg EP Comm (us)", "Avg MoE Window (us)",
    ]
    avg["Avg Comm Ratio"] = avg["Avg Comm Ratio"].map(lambda x: f"{x:.1%}")
    logger.log(avg.to_string(float_format="%.1f"))
    logger.log("")

    for rank in sorted(df["rank"].unique()):
        r_data = df[df["rank"] == rank]
        mean_ratio = r_data["ep_comm_ratio"].mean()
        mean_disp_share = r_data["dispatch_share"].mean()
        if mean_ratio > _EP_COMM_RATIO_WARN:
            phase = "dispatch" if mean_disp_share > 0.6 else (
                "combine" if mean_disp_share < 0.4 else "both phases"
            )
            logger.log(
                f"  [WARNING] Rank {rank}: EP comm ratio = {mean_ratio:.1%}. "
                f"Bottleneck appears in {phase}."
            )

    logger.log("=" * 90 + "\n")


def _write_ep_overlap_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 3: EP Communication-Computation Overlap")
    logger.log("=" * 90)

    if not data:
        logger.log("  No EP overlap data available.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    avg = df.groupby("rank").agg({
        "total_comm_us": "mean",
        "total_comp_us": "mean",
        "overlap_us": "mean",
        "exposed_comm_us": "mean",
        "overlap_ratio": "mean",
    })
    avg.columns = [
        "Avg Comm (us)", "Avg Comp (us)", "Avg Overlap (us)",
        "Avg Exposed (us)", "Avg Overlap Ratio",
    ]
    avg["Avg Overlap Ratio"] = avg["Avg Overlap Ratio"].map(
        lambda x: f"{x:.1%}"
    )
    logger.log(avg.to_string(float_format="%.1f"))
    logger.log("")

    for rank in sorted(df["rank"].unique()):
        r_data = df[df["rank"] == rank]
        mean_ratio = r_data["overlap_ratio"].mean()
        mean_exposed_ms = r_data["exposed_comm_us"].mean() / 1e3
        if mean_ratio < _EP_OVERLAP_WARN:
            logger.log(
                f"  [WARNING] Rank {rank}: Only {mean_ratio:.1%} overlap, "
                f"avg exposed comm = {mean_exposed_ms:.2f} ms."
            )
            if mean_ratio == 0.0:
                logger.log(
                    "            [ROOT CAUSE] overlap_ratio=0.0 indicates comm and comp "
                    "run sequentially on the same CUDA stream.\n"
                    "            If --moe-shared-expert-overlap is enabled but "
                    "--overlap-moe-expert-parallel-comm is NOT, shared-expert work\n"
                    "            is scheduled before the All-to-All barrier but on the "
                    "same stream, giving zero true GPU parallelism.\n"
                    "            [ACTION] Add --overlap-moe-expert-parallel-comm to "
                    "enable a dedicated communication CUDA stream for All-to-All."
                )
            else:
                logger.log(
                    "            Consider enabling EP comm-compute pipelining "
                    "or using shared experts to fill comm gaps."
                )

    logger.log("=" * 90 + "\n")


def _write_token_drop_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
    balance_data: Optional[List[Dict[str, Any]]] = None,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 4: Token Dropping")
    logger.log("=" * 90)

    if not data:
        logger.log("  No token dropping data available.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    mean_rate = df["avg_drop_rate"].mean()
    max_rate = df["avg_drop_rate"].max()
    dropped_values = [
        row["total_dropped_tokens"]
        for row in data
        if row.get("total_dropped_tokens") is not None
    ]
    total_dropped = sum(dropped_values) if dropped_values else None

    logger.log(f"  Mean drop rate: {mean_rate:.4%}")
    logger.log(f"  Max drop rate:  {max_rate:.4%}")
    if total_dropped is None:
        logger.log("  Total tokens dropped: N/A")
    else:
        logger.log(f"  Total tokens dropped: {total_dropped:,}")

    for sev in ("CRITICAL", "WARNING"):
        flagged = df[df["severity"] == sev]
        if not flagged.empty:
            logger.log(f"\n  [{sev}] {len(flagged)} entries:")
            for _, row in flagged.head(10).iterrows():
                dropped = row["total_dropped_tokens"]
                dropped_text = str(dropped) if pd.notna(dropped) else "N/A"
                logger.log(
                    f"    Iter {row['iteration']}  Layer {row['layer']}  "
                    f"Rate={row['avg_drop_rate']:.4%}  "
                    f"Dropped={dropped_text}"
                )

    if max_rate > _DROP_RATE_CRIT:
        logger.log(
            "\n  [RECOMMENDATION] High token drop rate detected. Consider:\n"
            "    - Increasing capacity_factor (e.g. 1.25 -> 1.5).\n"
            "    - Reducing router_topk if over-dispatching.\n"
            "    - Strengthening aux loss to rebalance routing."
        )

    # Root-cause association with load-balance signals
    if balance_data:
        bdf = pd.DataFrame(balance_data)
        cv_values = bdf["expert_cv"].dropna()
        mean_cv = float(cv_values.mean()) if not cv_values.empty else None
        if mean_rate > _DROP_RATE_WARN:
            if mean_cv is None:
                logger.log(
                    "\n  Load-balance evidence is unavailable; no drop root-cause "
                    "association was made."
                )
            elif mean_cv > _EXPERT_CV_WARN:
                logger.log(
                    "\n  [ROOT CAUSE] High drop + high expert skew -> likely router "
                    "imbalance causing token pile-up on hot experts."
                )
            else:
                logger.log(
                    "\n  [ROOT CAUSE] High drop + balanced load -> capacity_factor may "
                    "be too tight (configuration bottleneck)."
                )

    logger.log("=" * 90 + "\n")


def _write_router_health_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 5: Router Health")
    logger.log("=" * 90)

    if not data:
        logger.log("  No router health data available.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    status_counts = df["status"].value_counts()
    logger.log("  Status distribution:")
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        logger.log(f"    {status}: {count}/{len(df)} ({pct:.1f}%)")

    collapse_risk = df[df["status"] == "Router Collapse Risk"]
    if not collapse_risk.empty:
        logger.log(
            f"\n  [CRITICAL] Router collapse risk detected in "
            f"{len(collapse_risk)} iterations."
        )
        for _, row in collapse_risk.head(5).iterrows():
            reasons = "; ".join(row["reasons"]) if row["reasons"] else "N/A"
            logger.log(f"    Iter {row['iteration']}: {reasons}")

    logger.log("=" * 90 + "\n")


def _write_router_loss_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 5b: Aux / Z Loss Drift")
    logger.log("=" * 90)

    if not data:
        logger.log("  No aux/z loss data available.")
        logger.log(
            "  [CONFIG NOTE] aux/z loss data is absent. Common causes:\n"
            "    1. moe_aux_loss_coeff=0.0 — the load-balancing loss weight is zero;\n"
            "       seq_aux_loss routing computes the value but it contributes nothing\n"
            "       to the gradient. Set moe_aux_loss_coeff > 0 (e.g. 0.001) to\n"
            "       activate load-balancing regularization.\n"
            "    2. aux_loss is not written into moe-router span args — instrument\n"
            "       the router forward to emit aux_loss / z_loss in event args, or\n"
            "       add EP_Aux_Loss / EP_Z_Loss Counter (ph=C) events so the\n"
            "       hardware-metrics fallback path can pick them up."
        )
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    logger.log(f"  Iterations with loss data: {len(df)}")
    logger.log(f"  Aux loss range: [{df['aux_loss_mean'].min():.6f}, "
               f"{df['aux_loss_mean'].max():.6f}]")

    if df["z_loss_mean"].sum() > 0:
        logger.log(f"  Z loss range:   [{df['z_loss_mean'].min():.6f}, "
                   f"{df['z_loss_mean'].max():.6f}]")

    # Data-quality / semantics diagnostics for aux_loss & z_loss.
    if "router_events" in df.columns:
        total_router_events = int(df["router_events"].sum())
        total_aux_valid = int(df["aux_valid_count"].sum())
        total_aux_missing = int(df["aux_missing_or_invalid_count"].sum())
        total_z_valid = int(df["z_valid_count"].sum())
        total_z_missing = int(df["z_missing_or_invalid_count"].sum())
        total_aux_counter = int(df["aux_counter_windows"].sum())
        total_z_counter = int(df["z_counter_windows"].sum())

        logger.log("\n  [Data Semantics Check]")
        logger.log(
            f"    Router events: {total_router_events}  "
            f"aux valid/missing: {total_aux_valid}/{total_aux_missing}  "
            f"z valid/missing: {total_z_valid}/{total_z_missing}"
        )
        logger.log(
            f"    Counter windows used: EP_Aux_Loss={total_aux_counter}, "
            f"EP_Z_Loss={total_z_counter}"
        )

        if total_aux_valid == 0 and total_aux_counter == 0:
            logger.log(
                "    [ROOT CAUSE] aux_loss has no numeric samples in trace. "
                "For seq_aux_loss routing, this is usually a logging-semantic gap "
                "instead of runtime failure."
            )
        if total_z_valid == 0 and total_z_counter == 0:
            logger.log(
                "    [ROOT CAUSE] z_loss has no numeric samples in trace. "
                "This is expected when moe_z_loss_coeff is not enabled."
            )

    flagged_iters = [
        row for _, row in df.iterrows() if row["risk_flags"]
    ]
    if flagged_iters:
        logger.log(f"\n  Risk flags detected in {len(flagged_iters)} iterations:")
        for row in flagged_iters[:10]:
            flags = ", ".join(row["risk_flags"])
            logger.log(
                f"    Iter {row['iteration']}: {flags}  "
                f"(aux={row['aux_loss_mean']:.6f}, "
                f"slope={row['aux_slope']:.6f})"
            )

    last_row = df.iloc[-1]
    if last_row["aux_slope"] > 0.001:
        logger.log(
            "\n  [WARNING] Aux loss shows an upward trend "
            f"(slope={last_row['aux_slope']:.6f}). "
            "Load balancing may be degrading."
        )

    logger.log("=" * 90 + "\n")


def _write_ep_straggler_report(
    data: List[Dict[str, Any]],
    logger: _ReportLogger,
) -> None:
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 6: EP Group Straggler Diagnosis")
    logger.log("=" * 90)

    if not data:
        logger.log("  No EP straggler data found.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    df = pd.DataFrame(data)
    ranks = sorted({r for d in data for r in d["per_rank_end_ts"]})

    logger.log(
        f"  Analysed {len(df)} sync barrier events across "
        f"{len(ranks)} EP ranks.\n"
    )

    summary_rows = []
    for _, row in df.iterrows():
        summary_rows.append({
            "Iter": row["iteration"],
            "Sync Event": row["sync_event"],
            "EP Group": row["ep_group"],
            "Gap (us)": f"{row['gap_us']:.1f}",
            "Straggler": f"Rank {row['straggler_rank']}",
            "Cause": row["likely_cause"],
            "HW Diag": row["hardware_diagnosis"] or "\u2014",
        })
    logger.log(pd.DataFrame(summary_rows).to_string(index=False))

    logger.log("\n" + "-" * 90)
    logger.log("  [Straggler Frequency]")
    freq = df["straggler_rank"].value_counts()
    for rank, count in freq.items():
        pct = count / len(df) * 100
        marker = " *** FREQUENT ***" if pct > 50 else ""
        logger.log(f"    Rank {rank}: {count}/{len(df)} ({pct:.1f}%){marker}")

    logger.log("\n  [Cause Summary]")
    for cause, cnt in df["likely_cause"].value_counts().items():
        logger.log(f"    {cause}: {cnt}")

    # Actionable root-cause guidance
    if (df["likely_cause"] == "hardware").any():
        logger.log(
            "\n  [RECOMMENDATION] Hardware-linked stragglers detected. "
            "Check thermal throttling, SM clock drops, and GPU health logs."
        )
    if (df["likely_cause"] == "routing_skew").any():
        logger.log(
            "  [RECOMMENDATION] Routing-skew-linked stragglers detected. "
            "Tune aux loss / router regularization and verify per-expert token balance."
        )
    if (df["likely_cause"] == "network_or_load").any():
        logger.log(
            "  [RECOMMENDATION] Network/load linked stragglers detected. "
            "Check EP group link health and investigate dispatch/combine asymmetry."
        )
    if (df["likely_cause"] == "token_imbalance").any():
        logger.log(
            "  [RECOMMENDATION] Token-imbalance-linked stragglers detected on "
            "ep-alltoall events. Ranks with more assigned tokens take longer in "
            "All-to-All. Root cause is typically routing skew (uneven expert load). "
            "Increase moe_aux_loss_coeff or switch to a more balanced routing strategy."
        )

    logger.log("=" * 90 + "\n")


def _write_layer_root_cause_topk(
    logger: _ReportLogger,
    balance_data: List[Dict[str, Any]],
    drop_data: List[Dict[str, Any]],
    health_data: List[Dict[str, Any]],
    overlap_data: List[Dict[str, Any]],
    comm_data: List[Dict[str, Any]],
    topk: int = 5,
) -> None:
    """Rank Top-K risky layers with actionable root-cause hints."""
    logger.log("\n" + "=" * 90)
    logger.log(" Chapter 7: Layer-wise Root-Cause Top-K")
    logger.log("=" * 90)

    if not balance_data and not drop_data:
        logger.log("  No layer-level balance/drop data available.")
        logger.log("=" * 90 + "\n")
        return

    pd = _load_pandas()
    bal_df = pd.DataFrame(balance_data) if balance_data else pd.DataFrame()
    drop_df = pd.DataFrame(drop_data) if drop_data else pd.DataFrame()

    layer_scores: Dict[int, Dict[str, Any]] = {}

    # Balance-driven signals
    if not bal_df.empty:
        grouped = bal_df.groupby("layer").agg({
            "expert_cv": "mean",
            "top1_expert_share": "mean",
            "routing_entropy": "mean",
        })
        for layer, row in grouped.iterrows():
            score = 0.0
            causes: List[str] = []
            cv_value = row.get("expert_cv")
            top1_value = row.get("top1_expert_share")
            entropy_value = row.get("routing_entropy")
            cv = float(cv_value) if pd.notna(cv_value) else None
            top1 = float(top1_value) if pd.notna(top1_value) else None
            entropy = float(entropy_value) if pd.notna(entropy_value) else None

            if cv is not None and cv > _EXPERT_CV_CRIT:
                score += 4.0
                causes.append(f"routing_skew: CV={cv:.3f} (> {_EXPERT_CV_CRIT})")
            elif cv is not None and cv > _EXPERT_CV_WARN:
                score += 2.5
                causes.append(f"routing_skew: CV={cv:.3f} (> {_EXPERT_CV_WARN})")

            if top1 is not None and top1 > _TOP1_SHARE_CRIT:
                score += 3.0
                causes.append(f"hot_expert: Top1={top1:.3f} (> {_TOP1_SHARE_CRIT})")
            elif top1 is not None and top1 > _TOP1_SHARE_WARN:
                score += 1.5
                causes.append(f"hot_expert: Top1={top1:.3f} (> {_TOP1_SHARE_WARN})")

            layer_scores[int(layer)] = {
                "layer": int(layer),
                "risk_score": score,
                "expert_cv": cv,
                "top1_expert_share": top1,
                "routing_entropy": entropy,
                "drop_rate": None,
                "root_causes": causes,
            }

    # Drop-driven signals + association with balance
    if not drop_df.empty:
        grouped = drop_df.groupby("layer").agg({"avg_drop_rate": "mean"})
        for layer, row in grouped.iterrows():
            layer = int(layer)
            entry = layer_scores.setdefault(layer, {
                "layer": layer,
                "risk_score": 0.0,
                "expert_cv": None,
                "top1_expert_share": None,
                "routing_entropy": None,
                "drop_rate": None,
                "root_causes": [],
            })
            dr = float(row.get("avg_drop_rate", 0.0))
            entry["drop_rate"] = dr

            if dr > _DROP_RATE_CRIT:
                entry["risk_score"] += 3.0
            elif dr > _DROP_RATE_WARN:
                entry["risk_score"] += 1.5

            cv = entry.get("expert_cv")
            if dr > _DROP_RATE_WARN and cv is not None and cv > _EXPERT_CV_WARN:
                entry["root_causes"].append(
                    f"drop+skew: drop_rate={dr:.3%}, CV={cv:.3f}"
                )
            elif dr > _DROP_RATE_WARN and cv is not None:
                entry["root_causes"].append(
                    f"capacity_tight: drop_rate={dr:.3%} with moderate CV"
                )
            elif dr > _DROP_RATE_WARN:
                entry["root_causes"].append(
                    f"drop_rate={dr:.3%}; load-balance evidence unavailable"
                )

    ranked = sorted(
        layer_scores.values(),
        key=lambda x: x.get("risk_score", 0.0),
        reverse=True,
    )
    ranked = [r for r in ranked if r.get("risk_score", 0.0) > 0][:max(1, topk)]

    if not ranked:
        logger.log("  No risky layers identified (all scores are low).")
        logger.log("=" * 90 + "\n")
        return

    rows = []
    for r in ranked:
        causes = "; ".join(r["root_causes"][:3]) if r["root_causes"] else "low risk"
        rows.append({
            "Layer": r["layer"],
            "Risk Score": f"{r['risk_score']:.1f}",
            "CV": f"{r['expert_cv']:.3f}" if r["expert_cv"] is not None else "N/A",
            "Top1": (
                f"{r['top1_expert_share']:.3f}"
                if r["top1_expert_share"] is not None else "N/A"
            ),
            "Drop Rate": (
                f"{r['drop_rate']:.3%}" if r["drop_rate"] is not None else "N/A"
            ),
            "Primary Root-Cause": causes,
        })

    logger.log(pd.DataFrame(rows).to_string(index=False))

    # Global context for communication/exposed bottlenecks.
    if overlap_data:
        ov = pd.DataFrame(overlap_data)
        mean_overlap = float(ov["overlap_ratio"].mean()) if not ov.empty else 1.0
        if mean_overlap < _EP_OVERLAP_WARN:
            logger.log(
                f"\n  [Global Context] EP overlap is low ({mean_overlap:.1%}). "
                "Layer-local skew may be amplified by exposed communication."
            )
    if comm_data:
        cm = pd.DataFrame(comm_data)
        mean_comm_ratio = float(cm["ep_comm_ratio"].mean()) if not cm.empty else 0.0
        if mean_comm_ratio > _EP_COMM_RATIO_WARN:
            logger.log(
                f"  [Global Context] EP comm ratio is high ({mean_comm_ratio:.1%}). "
                "Check dispatch/combine bottlenecks together with Top-K layers above."
            )

    if health_data:
        hd = pd.DataFrame(health_data)
        risk_ratio = float((hd["status"] == "Router Collapse Risk").mean()) if not hd.empty else 0.0
        if risk_ratio > 0.2:
            logger.log(
                f"  [Global Context] Router collapse risk appears in {risk_ratio:.1%} iterations. "
                "Prioritize regularization and load-balancing stability."
            )

    logger.log("=" * 90 + "\n")


# ============================================================================
# Master Orchestrator
# ============================================================================

def analyze_ep_traces(
    traces: List[Dict[str, Any]],
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Run all EP analyses, generate report + plots + CSV exports.

    This is the public entry point called by
    :func:`megatron.megalens.analyzer.run_parallelism_analyses`.

    Args:
        traces: Aggregated Chrome Trace event list (post-``transform``).
        output_dir: Directory for PDF plots, TXT report, and CSV files.

    Returns:
        Dict with keys for each analysis result set.
    """
    pd = _load_pandas()
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "ep_diagnostic_report.txt")
    logger = _ReportLogger(
        report_file, "MegaLens Expert Parallelism (EP) Diagnostic Report",
    )

    logger.log("[EP Analyzer] Loading trace data...")
    loader = TraceDataLoader.from_traces(traces)
    analyzer = EPAnalyzer(loader)

    ep_ranks = analyzer._get_ep_ranks()
    logger.log(
        f"[EP Analyzer] Detected {len(loader.get_ranks())} total ranks, "
        f"{len(ep_ranks)} with MoE events."
    )

    # 1. Expert Load Balance
    logger.log("\n[Step 1/8] Running Expert Load Balance Analysis...")
    balance_data = analyzer.analyze_expert_load_balance()
    _write_expert_balance_report(balance_data, logger)

    # 2. EP Communication Overhead
    logger.log("[Step 2/8] Running EP Communication Overhead Analysis...")
    comm_data = analyzer.analyze_ep_comm_overhead()
    _write_ep_comm_report(comm_data, logger)

    # 3. EP Comm-Comp Overlap
    logger.log("[Step 3/8] Running EP Comm-Comp Overlap Analysis...")
    overlap_data = analyzer.analyze_ep_overlap()
    _write_ep_overlap_report(overlap_data, logger)

    # 4. Token Dropping
    logger.log("[Step 4/8] Running Token Dropping Analysis...")
    drop_data = analyzer.analyze_token_dropping()
    _write_token_drop_report(drop_data, logger, balance_data=balance_data)

    # 5. Aux / Z Loss Drift
    logger.log("[Step 5/8] Running Aux/Z Loss Drift Analysis...")
    loss_data = analyzer.analyze_router_loss_drift()
    _write_router_loss_report(loss_data, logger)

    # 6. Router Health (composite — depends on 1, 4, 5)
    logger.log("[Step 6/8] Running Router Health Assessment...")
    health_data = analyzer.analyze_router_health(
        balance_data, drop_data, loss_data,
    )
    _write_router_health_report(health_data, logger)

    # 7. EP Straggler Diagnosis
    logger.log("[Step 7/8] Running EP Straggler Diagnosis...")
    straggler_data = analyzer.diagnose_ep_stragglers()
    _write_ep_straggler_report(straggler_data, logger)

    # 8. Layer-wise Root-Cause Top-K
    logger.log("[Step 8/8] Ranking Layer-wise Root-Causes (Top-K)...")
    _write_layer_root_cause_topk(
        logger,
        balance_data=balance_data,
        drop_data=drop_data,
        health_data=health_data,
        overlap_data=overlap_data,
        comm_data=comm_data,
        topk=5,
    )

    # ---- Visualizations ----
    logger.log("\n[EP Analyzer] Generating visualizations...")
    generate_expert_balance_plots(balance_data, output_dir)
    generate_token_distribution_plots(balance_data, output_dir)
    generate_ep_comm_plots(comm_data, output_dir)
    generate_ep_overlap_plots(overlap_data, output_dir)
    generate_token_drop_plots(drop_data, output_dir)
    generate_router_health_plots(health_data, loss_data, output_dir)
    generate_ep_straggler_plots(straggler_data, output_dir)

    # ---- CSV Export ----
    logger.log("[EP Analyzer] Exporting CSV stats...")

    # Token distribution (for ep_token_distribution.pdf)
    # Export two flat tables so each sub-figure has reusable source data:
    #   1) rank-expert statistics (boxplot + heatmap)
    #   2) rank-iteration total load (line chart)
    records_with_tpe = [
        r for r in balance_data
        if r.get("tokens_per_expert_by_rank")
    ]
    if records_with_tpe:
        rank_expert_rows: List[Dict[str, Any]] = []
        rank_iter_rows: List[Dict[str, Any]] = []

        for rec in records_with_tpe:
            it = int(rec.get("iteration", -1))
            layer = int(rec.get("layer", -1))
            tpe_by_rank = rec.get("tokens_per_expert_by_rank") or {}

            for rank_key, tpe_list in tpe_by_rank.items():
                rank = int(rank_key)
                tpe_vals = [float(x) for x in tpe_list]

                # B3 data: per-rank total token load per iteration/layer
                rank_iter_rows.append({
                    "iteration": it,
                    "layer": layer,
                    "rank": rank,
                    "total_tokens": float(sum(tpe_vals)),
                    "num_local_experts": len(tpe_vals),
                })

                # B1/B2 data: per-rank per-local-expert distribution
                for local_expert_idx, tok_count in enumerate(tpe_vals):
                    rank_expert_rows.append({
                        "iteration": it,
                        "layer": layer,
                        "rank": rank,
                        "local_expert_idx": int(local_expert_idx),
                        "tokens": float(tok_count),
                    })

        if rank_expert_rows:
            exp_df = pd.DataFrame(rank_expert_rows)
            exp_stats = (
                exp_df.groupby(["rank", "local_expert_idx"], as_index=False)
                .agg(
                    samples=("tokens", "count"),
                    mean_tokens=("tokens", "mean"),
                    std_tokens=("tokens", "std"),
                    min_tokens=("tokens", "min"),
                    p50_tokens=("tokens", "median"),
                    p95_tokens=("tokens", lambda x: np.percentile(x, 95)),
                    max_tokens=("tokens", "max"),
                )
            )
            exp_stats["std_tokens"] = exp_stats["std_tokens"].fillna(0.0)
            exp_stats.to_csv(
                os.path.join(output_dir, "ep_token_distribution_rank_expert_stats.csv"),
                index=False,
            )

        if rank_iter_rows:
            iter_df = pd.DataFrame(rank_iter_rows)
            iter_load = (
                iter_df.groupby(["iteration", "rank"], as_index=False)
                .agg(
                    total_tokens=("total_tokens", "sum"),
                    num_layers=("layer", "nunique"),
                )
                .sort_values(["iteration", "rank"])
            )
            iter_totals = iter_load.groupby("iteration")["total_tokens"].transform("sum")
            ranks_per_iter = iter_load.groupby("iteration")["rank"].transform("nunique")
            iter_load["ideal_uniform_tokens"] = iter_totals / ranks_per_iter
            iter_load["load_vs_uniform_ratio"] = (
                iter_load["total_tokens"] / iter_load["ideal_uniform_tokens"]
            )
            iter_load.to_csv(
                os.path.join(output_dir, "ep_token_distribution_rank_iteration_load.csv"),
                index=False,
            )

    if balance_data:
        # tokens_per_expert_by_rank is a nested dict, not CSV-serialisable.
        # Export a flat copy without that field.
        _CSV_EXCLUDE = {"tokens_per_expert_by_rank"}
        balance_csv = [
            {k: v for k, v in row.items() if k not in _CSV_EXCLUDE}
            for row in balance_data
        ]
        pd.DataFrame(balance_csv).to_csv(
            os.path.join(output_dir, "ep_expert_balance_stats.csv"),
            index=False,
        )
    if comm_data:
        pd.DataFrame(comm_data).to_csv(
            os.path.join(output_dir, "ep_comm_overhead_stats.csv"),
            index=False,
        )
    if overlap_data:
        pd.DataFrame(overlap_data).to_csv(
            os.path.join(output_dir, "ep_overlap_stats.csv"),
            index=False,
        )
    if drop_data:
        pd.DataFrame(drop_data).to_csv(
            os.path.join(output_dir, "ep_token_drop_stats.csv"),
            index=False,
        )
    if health_data:
        pd.DataFrame(health_data).drop(columns=["reasons"], errors="ignore").to_csv(
            os.path.join(output_dir, "ep_router_metrics.csv"),
            index=False,
        )
    if loss_data:
        pd.DataFrame(loss_data).drop(columns=["risk_flags"], errors="ignore").to_csv(
            os.path.join(output_dir, "ep_router_loss_stats.csv"),
            index=False,
        )
    if straggler_data:
        pd.DataFrame(straggler_data).drop(
            columns=["per_rank_end_ts", "hw_detail"], errors="ignore",
        ).to_csv(
            os.path.join(output_dir, "ep_straggler_stats.csv"),
            index=False,
        )

    logger.log(f"\n[EP Analyzer] All analysis complete. Report -> {report_file}")
    return {
        "balance_data": balance_data,
        "comm_data": comm_data,
        "overlap_data": overlap_data,
        "drop_data": drop_data,
        "loss_data": loss_data,
        "health_data": health_data,
        "straggler_data": straggler_data,
    }
