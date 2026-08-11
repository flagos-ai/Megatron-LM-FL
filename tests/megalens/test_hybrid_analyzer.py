# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Any

import pytest

from megatron.megalens import hybrid_analyzer as hybrid_module
from megatron.megalens.data_loader import TraceDataLoader
from megatron.megalens.hybrid_analyzer import HybridAnalyzer, analyze_hybrid_traces

def _span(
    name: str,
    ts: int,
    dur: int,
    *,
    rank: int = 0,
    iteration: int = 7,
    dp_rank: int = 0,
    pp_rank: int = 0,
    tp_rank: int = 0,
    **args: Any,
) -> dict[str, Any]:
    event_args: dict[str, Any] = {
        "iteration": iteration,
        "dp_rk": dp_rank,
        "pp_rk": pp_rank,
        "tp_rk": tp_rank,
    }
    event_args.update(args)
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "pid": rank, "args": event_args}


def _analyzer(
    *events: dict[str, Any],
    pp: Any = None,
    dp: Any = None,
    tp: Any = None,
    ep: Any = None,
    sizes: dict[str, int] | None = None,
) -> HybridAnalyzer:
    analyzer = HybridAnalyzer(
        TraceDataLoader.from_traces(list(events)),
        pp_result=pp,
        dp_result=dp,
        tp_result=tp,
        ep_result=ep,
    )
    if sizes is not None:
        analyzer.parallel_sizes = sizes
    return analyzer


def test_public_api_signatures_defaults_and_thresholds_match_source() -> None:
    constructor = inspect.signature(HybridAnalyzer)
    assert tuple(constructor.parameters) == (
        "loader",
        "pp_result",
        "dp_result",
        "tp_result",
        "ep_result",
    )
    assert all(
        constructor.parameters[name].default is None
        for name in ("pp_result", "dp_result", "tp_result", "ep_result")
    )

    methods = (
        "analyze_ep_pp_bubble_amplification",
        "analyze_ep_tp_comm_contention",
        "analyze_ep_dp_load_coupling",
        "analyze_pp_dp_sync_serialization",
        "analyze_tp_ep_compute_fragmentation",
        "diagnose_global_straggler",
    )
    for method in methods:
        assert tuple(inspect.signature(getattr(HybridAnalyzer, method)).parameters) == ("self",)

    health = inspect.signature(HybridAnalyzer.compute_hybrid_health_score)
    assert tuple(health.parameters) == (
        "self",
        "ep_pp_result",
        "contention_data",
        "ep_dp_result",
        "pp_dp_data",
        "tp_ep_result",
        "straggler_data",
    )
    master = inspect.signature(analyze_hybrid_traces)
    assert tuple(master.parameters) == (
        "traces",
        "output_dir",
        "pp_result",
        "dp_result",
        "tp_result",
        "ep_result",
    )
    assert master.parameters["output_dir"].default == "."

    assert hybrid_module._EP_PP_CORR_WARN == 0.50
    assert hybrid_module._EP_PP_CORR_CRIT == 0.70
    assert hybrid_module._EP_TP_CONTENTION_WARN == 0.20
    assert hybrid_module._EP_TP_CONTENTION_CRIT == 0.40
    assert hybrid_module._EP_DP_CORR_WARN == 0.40
    assert hybrid_module._EP_DP_CORR_CRIT == 0.65
    assert hybrid_module._PP_DP_UTIL_WARN == 0.30
    assert hybrid_module._PP_DP_UTIL_CRIT == 0.10
    assert hybrid_module._TP_EP_FRAG_CORR_WARN == 0.40
    assert hybrid_module._TP_EP_FRAG_CORR_CRIT == 0.60
    assert hybrid_module._GLOBAL_STRAGGLER_WARN == 0.30
    assert hybrid_module._GLOBAL_STRAGGLER_CRIT == 0.50


def test_constructor_normalizes_non_dict_results_and_disabled_dimensions() -> None:
    analyzer = _analyzer(pp=[], dp="raw", tp=(), ep=object())

    assert analyzer.pp == {}
    assert analyzer.dp == {}
    assert analyzer.tp == {}
    assert analyzer.ep == {}
    assert analyzer.analyze_ep_pp_bubble_amplification()["severity"] == "SKIPPED"
    assert analyzer.analyze_ep_tp_comm_contention() == []
    assert analyzer.analyze_ep_dp_load_coupling()["severity"] == "SKIPPED"
    assert analyzer.analyze_pp_dp_sync_serialization() == []
    assert analyzer.analyze_tp_ep_compute_fragmentation()["severity"] == "SKIPPED"
    assert analyzer.diagnose_global_straggler() == []


def test_ep_pp_correlation_keeps_source_iteration_key_contract_and_formula() -> None:
    ep = {
        "balance_data": [
            {"iteration": 1, "expert_cv": 0.2},
            {"iteration": 2, "expert_cv": 0.4},
            {"iteration": 3, "expert_cv": 0.6},
        ],
        "straggler_data": [
            {"iteration": 3, "straggler_rank": 5, "gap_us": 300.0},
            {"iteration": 3, "straggler_rank": 5, "gap_us": 500.0},
        ],
    }
    sizes = {"dp": 1, "pp": 2, "tp": 1, "ep": 2}
    actual_pp_producer_shape = {"bubble_stats": [{"Iteration": 1, "Bubble_Rate": 0.2}]}

    producer_shape_result = _analyzer(pp=actual_pp_producer_shape, ep=ep, sizes=sizes)
    assert producer_shape_result.analyze_ep_pp_bubble_amplification()["num_common_iterations"] == 0

    hybrid_accepted_shape = {
        "bubble_stats": [
            {"iteration": 1, "Bubble_Rate": 0.1},
            {"iteration": 2, "Bubble_Rate": 0.2},
            {"iteration": 3, "Bubble_Rate": 0.3},
        ]
    }
    result = _analyzer(
        pp=hybrid_accepted_shape, ep=ep, sizes=sizes
    ).analyze_ep_pp_bubble_amplification()

    assert result == {
        "correlation_ep_cv_pp_bubble": 1.0,
        "avg_ep_expert_cv": 0.4,
        "avg_pp_bubble_rate": 0.2,
        "num_common_iterations": 3,
        "worst_iteration": 3,
        "root_ep_rank": 5,
        "estimated_bubble_overhead_us": 400.0,
        "severity": "CRITICAL",
        "_ep_cv_series": [0.2, 0.4, 0.6],
        "_pp_bubble_series": [0.1, 0.2, 0.3],
        "_common_iters": [1, 2, 3],
    }


def test_ep_correlations_ignore_nullable_expert_cv() -> None:
    analyzer = _analyzer(
        pp={
            "bubble_stats": [
                {"iteration": 1, "Bubble_Rate": 0.2},
                {"iteration": 2, "Bubble_Rate": 0.4},
                {"iteration": 3, "Bubble_Rate": 0.0},
            ]
        },
        dp={
            "balance_data": [
                {"iteration": 1, "cv": 0.2},
                {"iteration": 2, "cv": 0.4},
                {"iteration": 3, "cv": 0.0},
            ]
        },
        ep={
            "balance_data": [
                {"iteration": 1, "expert_cv": None},
                {"iteration": 1, "expert_cv": 0.2},
                {"iteration": 2, "expert_cv": None},
                {"iteration": 3, "expert_cv": 0.0},
            ]
        },
        sizes={"dp": 2, "pp": 2, "tp": 1, "ep": 2},
    )

    ep_pp = analyzer.analyze_ep_pp_bubble_amplification()
    ep_dp = analyzer.analyze_ep_dp_load_coupling()

    assert ep_pp["_common_iters"] == [1, 3]
    assert ep_pp["_ep_cv_series"] == [0.2, 0.0]
    assert ep_dp["_common_iters"] == [1, 3]
    assert ep_dp["_ep_cv_series"] == [0.2, 0.0]


def test_ep_tp_contention_uses_interval_union_and_legacy_tp_names() -> None:
    sizes = {"dp": 1, "pp": 1, "tp": 2, "ep": 2}
    rows = _analyzer(
        _span("ep-alltoall-dispatch", 0, 100),
        _span("tp-allreduce", 0, 100),
        _span("allreduce", 50, 100),
        sizes=sizes,
    ).analyze_ep_tp_comm_contention()

    assert rows == [
        {
            "rank": 0,
            "iteration": 7,
            "ep_comm_us": 100.0,
            "tp_comm_us": 100.0,
            "contention_us": 50.0,
            "contention_ratio": 0.25,
            "severity": "WARNING",
        }
    ]


def test_ep_dp_load_coupling_keeps_source_correlation_and_contribution() -> None:
    ep = {
        "balance_data": [
            {"iteration": 1, "expert_cv": 0.1},
            {"iteration": 2, "expert_cv": 0.2},
            {"iteration": 3, "expert_cv": 0.3},
        ]
    }
    dp = {
        "balance_data": [
            {"iteration": 1, "cv": 0.2},
            {"iteration": 2, "cv": 0.4},
            {"iteration": 3, "cv": 0.6},
        ]
    }
    sizes = {"dp": 2, "pp": 1, "tp": 1, "ep": 2}

    result = _analyzer(dp=dp, ep=ep, sizes=sizes).analyze_ep_dp_load_coupling()

    assert result["correlation_ep_skew_dp_cv"] == 1.0
    assert result["ep_contribution_pct"] == 100.0
    assert result["num_common_iterations"] == 3
    assert result["worst_iteration"] == 3
    assert result["avg_ep_cv"] == 0.2
    assert result["avg_dp_step_cv"] == 0.4
    assert result["severity"] == "CRITICAL"


def test_pp_dp_serialization_uses_bubble_and_comm_interval_unions() -> None:
    sizes = {"dp": 2, "pp": 2, "tp": 1, "ep": 1}
    rows = _analyzer(
        _span("iteration", 0, 1000),
        _span("forward-step", 100, 200),
        _span("backward-step", 500, 200),
        _span("grad-sync", 50, 100),
        _span("all-grads-sync", 350, 100),
        sizes=sizes,
    ).analyze_pp_dp_sync_serialization()

    assert rows == [
        {
            "rank": 0,
            "iteration": 7,
            "bubble_us": 600.0,
            "dp_comm_us": 200.0,
            "dp_comm_in_bubble_us": 150.0,
            "bubble_utilization_ratio": 0.25,
            "severity": "WARNING",
        }
    ]


def test_tp_ep_fragmentation_keeps_rank_correlation_and_thresholds() -> None:
    tp = {
        "frag_data": [
            {"rank": 0, "short_ratio": 0.1},
            {"rank": 1, "short_ratio": 0.3},
            {"rank": 2, "short_ratio": 0.5},
        ]
    }
    ep = {
        "comm_data": [
            {"rank": 0, "experts_dur_us": 100.0},
            {"rank": 1, "experts_dur_us": 200.0},
            {"rank": 2, "experts_dur_us": 300.0},
        ]
    }
    sizes = {"dp": 1, "pp": 1, "tp": 2, "ep": 2}

    result = _analyzer(tp=tp, ep=ep, sizes=sizes).analyze_tp_ep_compute_fragmentation()

    assert result["correlation_tp_frag_ep_compute"] == 1.0
    assert result["num_common_ranks"] == 3
    assert result["severity"] == "CRITICAL"
    assert result["per_rank"] == [
        {"rank": 0, "short_kernel_ratio": 0.1, "avg_experts_dur_us": 100.0, "severity": "OK"},
        {"rank": 1, "short_kernel_ratio": 0.3, "avg_experts_dur_us": 200.0, "severity": "WARNING"},
        {"rank": 2, "short_kernel_ratio": 0.5, "avg_experts_dur_us": 300.0, "severity": "CRITICAL"},
    ]


def test_global_straggler_keeps_cross_dimension_counts_and_hw_merge() -> None:
    pp = {"straggler_data": [{"rank": 1}, {"rank": 2}]}
    dp = {
        "straggler_data": [
            {
                "straggler_rank": 1,
                "hw_detail": {
                    "Temp_C_peak": 85.0,
                    "SM_Clock_MHz_mean": 800.0,
                    "SM_Base_Clock_MHz": 1000.0,
                },
            }
        ]
    }
    tp = {"straggler_data": [{"straggler_rank": 1}]}
    ep = {"straggler_data": [{"straggler_rank": 1}]}

    rows = _analyzer(pp=pp, dp=dp, tp=tp, ep=ep).diagnose_global_straggler()

    assert rows[0]["rank"] == 1
    assert rows[0]["straggler_score"] == 0.8
    assert rows[0]["pp_straggler_count"] == 1
    assert rows[0]["dp_straggler_count"] == 1
    assert rows[0]["tp_straggler_count"] == 1
    assert rows[0]["ep_straggler_count"] == 1
    assert rows[0]["severity"] == "CRITICAL"
    assert rows[0]["hardware_root_cause"].startswith("Thermal Throttling")
    assert rows[0]["hw_evidence"]["clock_ratio"] == 0.8
    assert rows[1]["rank"] == 2
    assert rows[1]["straggler_score"] == 0.2
    assert rows[1]["severity"] == "OK"


def test_health_score_keeps_source_deductions_and_zero_floor() -> None:
    analyzer = _analyzer(
        pp={"bubble_stats": [{"bubble_rate": 0.8, "theory_bubble_rate": 0.0}]},
        dp={"sync_data": [{"sync_ratio": 0.5}]},
        tp={
            "tp_comm_data": [{"comm_ratio": 0.6}],
            "nvlink_summary": [{"mean_utilisation_pct": 40.0}],
        },
        ep={
            "balance_data": [{"expert_cv": 0.6}],
            "drop_data": [{"avg_drop_rate": 0.06}],
            "health_data": [{"status": "Router Collapse Risk"}],
        },
    )

    result = analyzer.compute_hybrid_health_score(
        {"severity": "CRITICAL"},
        [{"contention_ratio": 0.5}],
        {},
        [],
        {},
        [{"severity": "CRITICAL"}],
    )

    assert result["score"] == 0
    assert result["grade"] == "F"
    assert result["total_deducted"] == 100
    assert sum(points for _, points in result["deductions"]) == 120


def test_health_score_ignores_three_cross_results_and_breaks_on_first_pp_threshold() -> None:
    analyzer = _analyzer(
        pp={
            "bubble_stats": [
                {"bubble_rate": 0.3, "theory_bubble_rate": 0.0},
                {"bubble_rate": 0.9, "theory_bubble_rate": 0.0},
            ]
        }
    )

    result = analyzer.compute_hybrid_health_score(
        {"severity": "OK"},
        [],
        {"severity": "CRITICAL"},
        [{"severity": "CRITICAL"}],
        {"severity": "CRITICAL"},
        [],
    )

    assert result == {
        "score": 90,
        "grade": "A",
        "deductions": [("PP bubble > theory+20%", 10)],
        "total_deducted": 10,
    }


@pytest.mark.parametrize(
    ("tp", "ep", "expected_deduction"),
    [
        (
            {"tp_comm_data": [{"comm_ratio": None}, {"comm_ratio": 0.6}]},
            {},
            ("TP comm ratio > 50%", 10),
        ),
        (
            {
                "nvlink_summary": [
                    {"mean_utilisation_pct": None},
                    {"mean_utilisation_pct": 40.0},
                ]
            },
            {},
            ("TP NVLink util < 50%", 10),
        ),
        (
            {},
            {"balance_data": [{"expert_cv": None}, {"expert_cv": 0.6}]},
            ("EP expert_cv > 0.50", 10),
        ),
    ],
)
def test_health_score_ignores_nullable_target_results(
    tp: dict[str, Any], ep: dict[str, Any], expected_deduction: tuple[str, int]
) -> None:
    analyzer = _analyzer(tp=tp, ep=ep)

    result = analyzer.compute_hybrid_health_score({}, [], {}, [], {}, [])

    assert result["score"] == 90
    assert result["deductions"] == [expected_deduction]


def test_master_orchestration_order_result_keys_and_csv_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    ep_pp = {
        "severity": "OK",
        "_common_iters": [1],
        "_ep_cv_series": [0.2],
        "_pp_bubble_series": [0.3],
    }
    contention = [{"rank": 0, "iteration": 1, "contention_ratio": 0.1}]
    ep_dp = {
        "severity": "OK",
        "_common_iters": [1, 2],
        "_ep_cv_series": [0.2, 0.9],
        "_dp_cv_series": [0.4, 0.8],
    }
    pp_dp = [{"rank": 0, "iteration": 1}]
    tp_ep = {"severity": "OK", "per_rank": []}
    stragglers = [{"rank": 0, "severity": "OK"}]
    health = {"score": 100, "grade": "A", "deductions": [], "total_deducted": 0}

    values: list[tuple[str, Any]] = [
        ("ep_pp", ep_pp),
        ("contention", contention),
        ("ep_dp", ep_dp),
        ("pp_dp", pp_dp),
        ("tp_ep", tp_ep),
        ("stragglers", stragglers),
    ]
    method_names = (
        "analyze_ep_pp_bubble_amplification",
        "analyze_ep_tp_comm_contention",
        "analyze_ep_dp_load_coupling",
        "analyze_pp_dp_sync_serialization",
        "analyze_tp_ep_compute_fragmentation",
        "diagnose_global_straggler",
    )
    for method_name, (call_name, value) in zip(method_names, values):
        monkeypatch.setattr(
            HybridAnalyzer,
            method_name,
            lambda self, _name=call_name, _value=value: (calls.append(_name), _value)[1],
        )

    def fake_health(self: HybridAnalyzer, *args: Any) -> dict[str, Any]:
        calls.append("health")
        return health

    monkeypatch.setattr(HybridAnalyzer, "compute_hybrid_health_score", fake_health)
    monkeypatch.setattr(
        hybrid_module, "_write_hybrid_report", lambda *args, **kwargs: calls.append("report")
    )
    plot_names = (
        "_generate_ep_pp_plot",
        "_generate_ep_tp_contention_plot",
        "_generate_ep_dp_coupling_plot",
        "_generate_pp_dp_serialization_plot",
        "_generate_global_straggler_plot",
        "_generate_health_score_plot",
    )
    for name in plot_names:
        monkeypatch.setattr(
            hybrid_module, name, lambda *args, _name=name, **kwargs: calls.append(_name)
        )

    result = analyze_hybrid_traces([], output_dir=str(tmp_path))

    assert tuple(result) == (
        "ep_pp",
        "contention",
        "ep_dp",
        "pp_dp",
        "tp_ep",
        "stragglers",
        "health",
    )
    assert result == {
        "ep_pp": ep_pp,
        "contention": contention,
        "ep_dp": ep_dp,
        "pp_dp": pp_dp,
        "tp_ep": tp_ep,
        "stragglers": stragglers,
        "health": health,
    }
    assert calls == [
        "ep_pp",
        "contention",
        "ep_dp",
        "pp_dp",
        "tp_ep",
        "stragglers",
        "health",
        "report",
        *plot_names,
    ]
    assert {path.name for path in tmp_path.iterdir()} == {
        "hybrid_diagnostic_report.txt",
        "hybrid_ep_tp_contention.csv",
        "hybrid_pp_dp_serialization.csv",
        "hybrid_global_straggler.csv",
        "hybrid_summary.csv",
    }
    with (tmp_path / "hybrid_summary.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert summary_rows == [
        {"iteration": "1", "ep_expert_cv": "0.2", "pp_bubble_rate": "0.3", "dp_step_cv": "0.4"},
        {"iteration": "2", "ep_expert_cv": "", "pp_bubble_rate": "", "dp_step_cv": "0.8"},
    ]


def test_hybrid_analyzer_remains_offline_without_core_or_training_wiring() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (repository_root / "megatron/core", repository_root / "megatron/training")
        for path in root.rglob("*.py")
    )

    assert "megatron.megalens.hybrid_analyzer" not in source_text
    assert "analyze_hybrid_traces" not in source_text
