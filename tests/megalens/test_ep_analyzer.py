# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import inspect
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from megatron.megalens import ep_analyzer as ep_module
from megatron.megalens.data_loader import TraceDataLoader
from megatron.megalens.ep_analyzer import EPAnalyzer, analyze_ep_traces

_SOURCE_BASELINE = "12fb7169ce30fdb62b50f86b41afa09336a523ea"
_SOURCE_SHA256 = "981a664d36dd26cedfe1efb8c695d743690219016df0d539ff65c4fb082855fd"
_SOURCE_LINE_COUNT = 2613
_SOURCE_BYTE_COUNT = 103_314
_ADAPTED_SHA256 = "60cd39d857e8094ec19b34c85de2a04a3457c49f6207644d7edf4dabf40460cf"
_ADAPTED_LINE_COUNT = 2733
_ADAPTED_BYTE_COUNT = 107_499


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


def _loader(*events: dict[str, Any]) -> TraceDataLoader:
    return TraceDataLoader.from_traces(list(events))


def test_locked_mixedpara_source_identity_and_adapter_snapshot_are_recorded() -> None:
    payload = Path(ep_module.__file__).resolve().read_bytes()

    assert _SOURCE_BASELINE == "12fb7169ce30fdb62b50f86b41afa09336a523ea"
    assert _SOURCE_SHA256 == "981a664d36dd26cedfe1efb8c695d743690219016df0d539ff65c4fb082855fd"
    assert _SOURCE_LINE_COUNT == 2613
    assert _SOURCE_BYTE_COUNT == 103_314
    assert hashlib.sha256(payload).hexdigest() == _ADAPTED_SHA256
    assert len(payload) == _ADAPTED_BYTE_COUNT
    assert len(payload.splitlines()) == _ADAPTED_LINE_COUNT
    assert _ADAPTED_SHA256 != _SOURCE_SHA256


def test_public_api_signatures_defaults_and_thresholds_match_source() -> None:
    assert tuple(inspect.signature(EPAnalyzer).parameters) == ("loader",)
    assert tuple(inspect.signature(EPAnalyzer.analyze_expert_load_balance).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(EPAnalyzer.analyze_ep_comm_overhead).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(EPAnalyzer.analyze_ep_overlap).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(EPAnalyzer.analyze_token_dropping).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(EPAnalyzer.analyze_router_loss_drift).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(EPAnalyzer.analyze_router_health).parameters) == (
        "self",
        "balance_data",
        "drop_data",
        "loss_data",
    )
    assert tuple(inspect.signature(EPAnalyzer.diagnose_ep_stragglers).parameters) == (
        "self",
        "iteration",
    )
    assert inspect.signature(EPAnalyzer._linear_slope).parameters["tail"].default == 10
    assert inspect.signature(ep_module._safe_entropy).parameters["eps"].default == 1e-12

    master = inspect.signature(analyze_ep_traces)
    assert tuple(master.parameters) == ("traces", "output_dir")
    assert master.parameters["output_dir"].default == "."

    assert ep_module._EXPERT_CV_INFO == 0.20
    assert ep_module._EXPERT_CV_WARN == 0.30
    assert ep_module._EXPERT_CV_CRIT == 0.50
    assert ep_module._TOP1_SHARE_WARN == 0.35
    assert ep_module._TOP1_SHARE_CRIT == 0.50
    assert ep_module._EP_COMM_RATIO_WARN == 0.25
    assert ep_module._EP_COMM_RATIO_CRIT == 0.40
    assert ep_module._EP_OVERLAP_WARN == 0.50
    assert ep_module._EP_OVERLAP_CRIT == 0.30
    assert ep_module._DROP_RATE_WARN == 0.01
    assert ep_module._DROP_RATE_CRIT == 0.05
    assert "matplotlib" not in ep_module.__dict__
    assert "plt" not in ep_module.__dict__
    assert "pd" not in ep_module.__dict__
    assert "cm" not in ep_module.__dict__


def test_base_install_import_and_empty_plots_do_not_require_reporting_dependencies() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = f"""
import importlib.abc
import sys
import tempfile

sys.path.insert(0, {str(repository_root)!r})

class BlockReportingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pandas" or fullname.startswith("matplotlib"):
            raise AssertionError(f"unexpected reporting import: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockReportingImports())
from megatron.megalens import ep_analyzer

with tempfile.TemporaryDirectory() as output_dir:
    ep_analyzer.generate_expert_balance_plots([], output_dir)
    ep_analyzer.generate_token_distribution_plots([], output_dir)
    ep_analyzer.generate_token_distribution_plots(
        [{{"iteration": 1, "tokens_per_expert_by_rank": {{0: []}}}}],
        output_dir,
    )
    ep_analyzer.generate_ep_comm_plots([], output_dir)
    ep_analyzer.generate_ep_overlap_plots([], output_dir)
    ep_analyzer.generate_token_drop_plots([], output_dir)
    ep_analyzer.generate_router_health_plots([], [], output_dir)
    ep_analyzer.generate_ep_straggler_plots([], output_dir)
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", source], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_reporting_loader_uses_agg_and_token_drop_plot_supports_matplotlib_311(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = f"""
import sys

sys.path.insert(0, {str(repository_root)!r})
from megatron.megalens import ep_analyzer

assert "matplotlib" not in sys.modules
plt, _ = ep_analyzer._load_reporting_dependencies()
assert plt.get_backend().lower() == "agg"
ep_analyzer.generate_token_drop_plots(
    [{{
        "iteration": 1,
        "layer": 0,
        "avg_drop_rate": 0.02,
        "total_dropped_tokens": 3,
    }}],
    {str(tmp_path)!r},
)
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", source], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "ep_token_drop_analysis.pdf"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_expert_balance_report_preserves_unknown_nullable_fields(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial-balance.txt"
    partial_logger = ep_module._ReportLogger(str(partial_path), "test")
    ep_module._write_expert_balance_report(
        [
            {
                "iteration": 7,
                "layer": 1,
                "expert_cv": None,
                "top1_expert_share": 0.4,
                "severity": "WARNING",
            },
            {
                "iteration": 8,
                "layer": 2,
                "expert_cv": 0.4,
                "top1_expert_share": None,
                "severity": "WARNING",
            },
        ],
        partial_logger,
    )
    partial = partial_path.read_text(encoding="utf-8")

    assert "Mean Expert CV: 0.4000" in partial
    assert "Max Expert CV:  0.4000" in partial
    assert "CV=N/A  Top1=0.4000" in partial
    assert "CV=0.4000  Top1=N/A" in partial
    assert "CV=0.0000" not in partial

    missing_path = tmp_path / "missing-balance.txt"
    missing_logger = ep_module._ReportLogger(str(missing_path), "test")
    ep_module._write_expert_balance_report(
        [
            {
                "iteration": 9,
                "layer": 3,
                "expert_cv": None,
                "top1_expert_share": 0.2,
                "severity": "OK",
            }
        ],
        missing_logger,
    )
    missing = missing_path.read_text(encoding="utf-8")

    assert "Mean Expert CV: N/A" in missing
    assert "Max Expert CV:  N/A" in missing
    assert "Expert CV is unavailable" in missing
    assert "within healthy range" not in missing

    numeric_path = tmp_path / "numeric-balance.txt"
    numeric_logger = ep_module._ReportLogger(str(numeric_path), "test")
    ep_module._write_expert_balance_report(
        [
            {
                "iteration": 10,
                "layer": 4,
                "expert_cv": 0.4,
                "top1_expert_share": 0.4,
                "severity": "WARNING",
            }
        ],
        numeric_logger,
    )
    numeric = numeric_path.read_text(encoding="utf-8")

    assert "CV=0.4000  Top1=0.4000" in numeric
    assert "[RECOMMENDATION] Expert skew is high." in numeric

    zero_path = tmp_path / "zero-balance.txt"
    zero_logger = ep_module._ReportLogger(str(zero_path), "test")
    ep_module._write_expert_balance_report(
        [
            {
                "iteration": 11,
                "layer": 5,
                "expert_cv": 0.0,
                "top1_expert_share": 0.0,
                "severity": "OK",
            }
        ],
        zero_logger,
    )
    zero = zero_path.read_text(encoding="utf-8")

    assert "Mean Expert CV: 0.0000" in zero
    assert "within healthy range" in zero


def test_token_drop_report_does_not_infer_from_unknown_counts_or_balance(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-drop.txt"
    missing_logger = ep_module._ReportLogger(str(missing_path), "test")
    ep_module._write_token_drop_report(
        [
            {
                "iteration": 7,
                "layer": 1,
                "avg_drop_rate": 0.02,
                "total_dropped_tokens": None,
                "severity": "WARNING",
            }
        ],
        missing_logger,
        balance_data=[{"expert_cv": None}],
    )
    missing = missing_path.read_text(encoding="utf-8")

    assert "Mean drop rate: 2.0000%" in missing
    assert "Total tokens dropped: N/A" in missing
    assert "Dropped=N/A" in missing
    assert "Load-balance evidence is unavailable" in missing
    assert "Total tokens dropped: 0" not in missing
    assert "[ROOT CAUSE]" not in missing

    numeric_path = tmp_path / "numeric-drop.txt"
    numeric_logger = ep_module._ReportLogger(str(numeric_path), "test")
    ep_module._write_token_drop_report(
        [
            {
                "iteration": 8,
                "layer": 2,
                "avg_drop_rate": 0.1,
                "total_dropped_tokens": 12,
                "severity": "CRITICAL",
            }
        ],
        numeric_logger,
        balance_data=[{"expert_cv": 0.4}],
    )
    numeric = numeric_path.read_text(encoding="utf-8")

    assert "Mean drop rate: 10.0000%" in numeric
    assert "Total tokens dropped: 12" in numeric
    assert "Dropped=12" in numeric
    assert "[RECOMMENDATION] High token drop rate detected." in numeric
    assert "[ROOT CAUSE] High drop + high expert skew" in numeric


def test_token_drop_plot_keeps_unknown_dropped_token_bars_as_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.axes import Axes

    bar_calls: list[tuple[list[str], list[float]]] = []
    original_bar = Axes.bar

    def record_bar(self: Axes, x: Any, height: Any, *args: Any, **kwargs: Any) -> Any:
        bar_calls.append((list(x), list(height)))
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", record_bar)
    ep_module.generate_token_drop_plots(
        [
            {"iteration": 1, "layer": 0, "avg_drop_rate": 0.02, "total_dropped_tokens": None},
            {"iteration": 1, "layer": 1, "avg_drop_rate": 0.03, "total_dropped_tokens": 7},
            {"iteration": 1, "layer": 2, "avg_drop_rate": 0.0, "total_dropped_tokens": 0},
        ],
        str(tmp_path),
    )

    assert len(bar_calls) == 1
    bar_layers, bar_heights = bar_calls[0]
    assert bar_layers == ["0", "1", "2"]
    assert math.isnan(bar_heights[0])
    assert bar_heights[1:] == [7.0, 0.0]
    output = tmp_path / "ep_token_drop_analysis.pdf"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_layer_root_cause_topk_keeps_missing_evidence_unknown(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial-topk.txt"
    partial_logger = ep_module._ReportLogger(str(partial_path), "test")
    ep_module._write_layer_root_cause_topk(
        partial_logger,
        balance_data=[
            {"layer": 1, "expert_cv": None, "top1_expert_share": 0.6, "routing_entropy": None}
        ],
        drop_data=[{"layer": 2, "avg_drop_rate": 0.02}],
        health_data=[],
        overlap_data=[],
        comm_data=[],
    )
    partial = partial_path.read_text(encoding="utf-8")

    assert "hot_expert: Top1=0.600" in partial
    assert "drop_rate=2.000%; load-balance evidence unavailable" in partial
    assert partial.count("N/A") >= 4
    assert "nan" not in partial
    assert "capacity_tight" not in partial
    assert "drop+skew" not in partial

    numeric_path = tmp_path / "numeric-topk.txt"
    numeric_logger = ep_module._ReportLogger(str(numeric_path), "test")
    ep_module._write_layer_root_cause_topk(
        numeric_logger,
        balance_data=[
            {"layer": 3, "expert_cv": 0.4, "top1_expert_share": 0.4, "routing_entropy": 0.5}
        ],
        drop_data=[{"layer": 3, "avg_drop_rate": 0.02}],
        health_data=[],
        overlap_data=[],
        comm_data=[],
    )
    numeric = numeric_path.read_text(encoding="utf-8")

    assert "5.5" in numeric
    assert "routing_skew: CV=0.400" in numeric
    assert "hot_expert: Top1=0.400" in numeric
    assert "drop+skew: drop_rate=2.000%, CV=0.400" in numeric

    zero_path = tmp_path / "zero-topk.txt"
    zero_logger = ep_module._ReportLogger(str(zero_path), "test")
    ep_module._write_layer_root_cause_topk(
        zero_logger,
        balance_data=[
            {"layer": 4, "expert_cv": 0.0, "top1_expert_share": 0.0, "routing_entropy": 0.0}
        ],
        drop_data=[{"layer": 4, "avg_drop_rate": 0.02}],
        health_data=[],
        overlap_data=[],
        comm_data=[],
    )
    zero = zero_path.read_text(encoding="utf-8")

    assert "0.000 0.000" in zero
    assert "capacity_tight: drop_rate=2.000% with moderate CV" in zero


def test_ep_rank_detection_prefers_moe_ranks_and_falls_back_to_all_ranks() -> None:
    analyzer = EPAnalyzer(
        _loader(
            _span("iteration", 0, 100, rank=0),
            _span("iteration", 0, 100, rank=1),
            _span("moe-router", 10, 10, rank=1),
        )
    )
    assert analyzer._get_ep_ranks() == [1]

    fallback = EPAnalyzer(
        _loader(_span("iteration", 0, 100, rank=0), _span("iteration", 0, 100, rank=1))
    )
    assert fallback._get_ep_ranks() == [0, 1]


def test_expert_balance_preserves_source_averaging_and_rank_load_contract() -> None:
    row = EPAnalyzer(
        _loader(
            _span(
                "moe-experts",
                0,
                10,
                rank=0,
                layer=2,
                ep_size=2,
                num_experts=4,
                tokens_per_expert=[10, 30],
            ),
            _span(
                "moe-experts",
                20,
                10,
                rank=1,
                layer=2,
                ep_size=2,
                num_experts=4,
                tokens_per_expert=[20, 20],
            ),
        )
    ).analyze_expert_load_balance()[0]

    assert row == {
        "iteration": 7,
        "layer": 2,
        "expert_cv": 0.25,
        "top1_expert_share": 0.625,
        "expert_max_over_mean": 1.25,
        "routing_entropy": 0.627741,
        "rank_load_cv": 0.0,
        "rank_load_max_over_mean": 1.0,
        "tokens_per_expert_by_rank": {0: [10.0, 30.0], 1: [20.0, 20.0]},
        "num_ranks_sampled": 2,
        "num_events_per_rank": 1,
        "ep_size": 2,
        "num_experts": 4,
        "severity": "CRITICAL",
    }


def test_expert_balance_scalar_fallback_keeps_unavailable_fields_as_none() -> None:
    row = EPAnalyzer(
        _loader(_span("moe-router", 0, 10, layer=3, expert_cv=0.2, top1_expert_share=0.3))
    ).analyze_expert_load_balance()[0]

    assert row["expert_cv"] == 0.2
    assert row["top1_expert_share"] == 0.3
    assert row["expert_max_over_mean"] is None
    assert row["routing_entropy"] is None
    assert row["rank_load_cv"] is None
    assert row["tokens_per_expert_by_rank"] is None
    assert row["severity"] == "OK"


def test_nullable_balance_fields_are_skipped_without_zero_substitution() -> None:
    analyzer = EPAnalyzer(
        _loader(
            _span(
                "moe-router",
                0,
                10,
                rank=0,
                layer=3,
                expert_cv=None,
                top1_expert_share=0.2,
                expert_max_over_mean=None,
            ),
            _span(
                "moe-router",
                20,
                10,
                rank=1,
                layer=4,
                expert_cv=0.4,
                top1_expert_share=None,
                expert_max_over_mean=1.5,
            ),
        )
    )

    rows = analyzer.analyze_expert_load_balance()

    assert rows[0]["expert_cv"] is None
    assert rows[0]["top1_expert_share"] == 0.2
    assert rows[0]["expert_max_over_mean"] is None
    assert rows[0]["severity"] == "OK"
    assert rows[1]["expert_cv"] == 0.4
    assert rows[1]["top1_expert_share"] is None
    assert rows[1]["expert_max_over_mean"] == 1.5
    assert rows[1]["severity"] == "WARNING"

    aggregated = EPAnalyzer(
        _loader(
            _span(
                "moe-router",
                0,
                10,
                rank=0,
                layer=5,
                expert_cv=None,
                top1_expert_share=0.2,
                expert_max_over_mean=None,
            ),
            _span(
                "moe-router",
                20,
                10,
                rank=1,
                layer=5,
                expert_cv=0.4,
                top1_expert_share=None,
                expert_max_over_mean=1.5,
            ),
        )
    ).analyze_expert_load_balance()[0]
    assert aggregated["expert_cv"] == 0.4
    assert aggregated["top1_expert_share"] == 0.2
    assert aggregated["expert_max_over_mean"] == 1.5

    explicit_zero = EPAnalyzer(
        _loader(_span("moe-router", 0, 10, layer=6, expert_cv=0.0, top1_expert_share=0.0))
    ).analyze_expert_load_balance()[0]
    assert explicit_zero["expert_cv"] == 0.0
    assert explicit_zero["top1_expert_share"] == 0.0
    assert explicit_zero["severity"] == "OK"


@pytest.mark.parametrize(
    "event",
    [
        _span(
            "moe-router",
            0,
            10,
            routed_tokens=None,
            dropped_tokens=None,
            drop_rate=None,
            expert_cv=None,
            top1_expert_share=None,
        ),
        _span(
            "moe-experts",
            0,
            10,
            routed_tokens=None,
            expert_cv=None,
            top1_expert_share=None,
            expert_max_over_mean=None,
            tokens_per_expert=None,
        ),
    ],
)
def test_nullable_balance_family_without_valid_driver_is_omitted(event: dict[str, Any]) -> None:
    assert EPAnalyzer(_loader(event)).analyze_expert_load_balance() == []


def test_nullable_drop_fields_are_skipped_without_zero_substitution() -> None:
    unknown = EPAnalyzer(
        _loader(
            _span("moe-router", 0, 10, layer=1, drop_rate=None, dropped_tokens=None, num_tokens=8)
        )
    )
    assert unknown.analyze_token_dropping() == []

    partial = EPAnalyzer(
        _loader(
            _span("moe-router", 0, 10, layer=1, drop_rate=0.1, dropped_tokens=None, num_tokens=None)
        )
    ).analyze_token_dropping()[0]
    assert partial["avg_drop_rate"] == 0.1
    assert partial["total_dropped_tokens"] is None
    assert partial["total_routed_tokens"] is None

    mixed = EPAnalyzer(
        _loader(
            _span(
                "moe-router",
                0,
                10,
                rank=0,
                layer=1,
                drop_rate=None,
                dropped_tokens=None,
                num_tokens=None,
            ),
            _span(
                "moe-router",
                20,
                10,
                rank=1,
                layer=1,
                drop_rate=0.02,
                dropped_tokens=2,
                num_tokens=100,
            ),
        )
    ).analyze_token_dropping()[0]
    assert mixed["avg_drop_rate"] == 0.02
    assert mixed["total_dropped_tokens"] == 2
    assert mixed["total_routed_tokens"] == 100
    assert mixed["num_ranks_sampled"] == 2

    explicit_zero = EPAnalyzer(
        _loader(_span("moe-router", 0, 10, layer=2, drop_rate=0.0, dropped_tokens=0, num_tokens=0))
    ).analyze_token_dropping()[0]
    assert explicit_zero["avg_drop_rate"] == 0.0
    assert explicit_zero["total_dropped_tokens"] == 0
    assert explicit_zero["total_routed_tokens"] == 0
    assert explicit_zero["severity"] == "OK"


def test_router_health_preserves_missing_metrics_as_none() -> None:
    analyzer = EPAnalyzer(_loader())
    balance_data = [
        {"iteration": 7, "expert_cv": None, "top1_expert_share": 0.3, "routing_entropy": None}
    ]

    row = analyzer.analyze_router_health(balance_data, [], [])[0]

    assert row == {
        "iteration": 7,
        "status": "Healthy",
        "risk_score": 0,
        "expert_cv": None,
        "top1_expert_share": 0.3,
        "routing_entropy": None,
        "cv_slope": None,
        "drop_rate": None,
        "aux_loss": 0.0,
        "aux_slope": 0.0,
        "z_loss": 0.0,
        "z_slope": 0.0,
        "reasons": [],
    }


def test_ep_comm_overhead_keeps_raw_sums_and_parent_iteration_selection() -> None:
    rows = EPAnalyzer(
        _loader(
            _span("moe-dispatch", 0, 100),
            _span("moe-experts", 100, 200),
            _span("moe-combine", 300, 100),
            _span("ep-alltoall-dispatch", 10, 50),
            _span("ep-allgather-dispatch", 20, 25),
            _span("ep-alltoall-combine", 310, 25),
            _span("ep-alltoall-dispatch", 500, 999, iteration=8),
        )
    ).analyze_ep_comm_overhead()

    assert rows == [
        {
            "rank": 0,
            "iteration": 7,
            "dispatch_dur_us": 100.0,
            "experts_dur_us": 200.0,
            "combine_dur_us": 100.0,
            "dispatch_comm_us": 75.0,
            "combine_comm_us": 25.0,
            "moe_window_us": 400.0,
            "ep_comm_us": 100.0,
            "ep_comm_ratio": 0.25,
            "comm_comp_ratio": 0.5,
            "dispatch_share": 0.75,
            "severity": "OK",
        }
    ]


def test_ep_overlap_uses_interval_union_and_exact_source_severity() -> None:
    rows = EPAnalyzer(
        _loader(
            _span("ep-alltoall-dispatch", 0, 100),
            _span("ep-allgather-dispatch", 50, 100),
            _span("moe-experts", 25, 50),
            _span("moe-shared-expert", 120, 50),
        )
    ).analyze_ep_overlap()

    assert rows == [
        {
            "rank": 0,
            "iteration": 7,
            "total_comm_us": 150.0,
            "total_comp_us": 100.0,
            "overlap_us": 80.0,
            "exposed_comm_us": 70.0,
            "overlap_ratio": 0.5333,
            "severity": "OK",
        }
    ]


def test_ep_overlap_zero_report_keeps_stream_and_kernel_causes_unknown(
    tmp_path: Path,
) -> None:
    rows = EPAnalyzer(
        _loader(
            _span("ep-alltoall-dispatch", 0, 50),
            _span("moe-shared-expert", 100, 50),
        )
    ).analyze_ep_overlap()
    report_path = tmp_path / "ep-overlap-zero.txt"
    logger = ep_module._ReportLogger(str(report_path), "test")

    ep_module._write_ep_overlap_report(rows, logger)
    report = report_path.read_text(encoding="utf-8")

    assert "captured EP communication and expert-computation intervals did not intersect" in report
    assert "cannot identify CUDA stream placement or physical kernel concurrency" in report
    assert "Check event coverage, stream dependencies, and a device timeline" in report
    assert "run sequentially on the same CUDA stream" not in report
    assert "Add --overlap-moe-expert-parallel-comm" not in report


def test_token_drop_and_router_health_keep_source_thresholds() -> None:
    analyzer = EPAnalyzer(
        _loader(
            _span(
                "moe-dispatch",
                0,
                10,
                rank=0,
                layer=1,
                drop_rate=0.02,
                dropped_tokens=2,
                num_tokens=100,
                capacity_factor=1.25,
                router_topk=2,
            ),
            _span(
                "moe-router",
                20,
                10,
                rank=1,
                layer=1,
                drop_rate=0.06,
                dropped_tokens=6,
                num_tokens=100,
            ),
        )
    )
    drop_row = analyzer.analyze_token_dropping()[0]

    assert drop_row == {
        "iteration": 7,
        "layer": 1,
        "avg_drop_rate": 0.04,
        "total_dropped_tokens": 8,
        "total_routed_tokens": 200,
        "capacity_factor": 1.25,
        "router_topk": 2,
        "num_ranks_sampled": 2,
        "severity": "WARNING",
    }

    health_row = analyzer.analyze_router_health(
        [{"iteration": 7, "expert_cv": 0.6, "top1_expert_share": 0.6, "routing_entropy": 0.1}],
        [drop_row | {"avg_drop_rate": 0.06}],
        [
            {
                "iteration": 7,
                "aux_loss_mean": 1.0,
                "aux_slope": 0.002,
                "z_loss_mean": 1.0,
                "z_slope": 0.02,
            }
        ],
    )[0]
    assert health_row["status"] == "Router Collapse Risk"
    assert health_row["risk_score"] == 10
    assert health_row["cv_slope"] == 0.0
    assert len(health_row["reasons"]) == 5


def test_router_loss_drift_keeps_missing_counts_and_linear_trends() -> None:
    rows = EPAnalyzer(
        _loader(
            _span("moe-router", 0, 10, iteration=1, aux_loss=0.1),
            _span("moe-router", 20, 10, iteration=2, aux_loss="bad", z_loss=0.02),
            _span("moe-router", 40, 10, iteration=3, aux_loss=0.4, z_loss=0.05),
        )
    ).analyze_router_loss_drift()

    assert [row["iteration"] for row in rows] == [1, 2, 3]
    assert rows[0]["z_missing_or_invalid_count"] == 1
    assert rows[1]["aux_missing_or_invalid_count"] == 1
    assert rows[2]["aux_loss_mean"] == 0.4
    assert rows[2]["z_loss_mean"] == 0.05
    assert rows[2]["aux_rolling_mean"] == 0.166667
    assert rows[2]["aux_rolling_std"] == 0.169967
    assert rows[2]["aux_slope"] == 0.15
    assert rows[2]["z_slope"] == 0.025
    assert rows[2]["risk_flags"] == ["aux_loss_unstable", "aux_loss_rising", "z_loss_spike_risk"]


def test_ep_straggler_keeps_latest_rank_event_and_raw_cross_rank_end_times() -> None:
    rows = EPAnalyzer(
        _loader(
            _span("moe-router", 0, 1, rank=0),
            _span("moe-router", 0, 1, rank=1),
            _span("ep-alltoall-dispatch", 0, 100, rank=0),
            _span("ep-alltoall-dispatch", 0, 600, rank=1),
            _span("ep-alltoall-dispatch", 0, 700, rank=1),
        )
    ).diagnose_ep_stragglers()

    assert rows == [
        {
            "iteration": 7,
            "sync_event": "ep-alltoall-dispatch",
            "ep_group": "DP0-PP0",
            "gap_us": 600.0,
            "straggler_rank": 1,
            "fastest_rank": 0,
            "per_rank_end_ts": {0: 100, 1: 700},
            "likely_cause": "network_or_load",
            "hardware_diagnosis": None,
            "hw_detail": None,
        }
    ]


def test_master_orchestration_order_result_keys_and_csv_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    balance = [{"iteration": 1, "layer": 2, "tokens_per_expert_by_rank": {0: [1.0, 2.0]}}]
    comm = [{"rank": 0, "iteration": 1}]
    overlap = [{"rank": 0, "iteration": 1}]
    drop = [{"iteration": 1, "layer": 2}]
    loss = [{"iteration": 1, "risk_flags": []}]
    health = [{"iteration": 1, "reasons": []}]
    straggler = [{"iteration": 1, "per_rank_end_ts": {}, "hw_detail": None}]

    def record_method(name: str, value: list[dict[str, Any]]):
        def method(self: EPAnalyzer, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(name)
            return value

        return method

    monkeypatch.setattr(
        EPAnalyzer, "analyze_expert_load_balance", record_method("balance", balance)
    )
    monkeypatch.setattr(EPAnalyzer, "analyze_ep_comm_overhead", record_method("comm", comm))
    monkeypatch.setattr(EPAnalyzer, "analyze_ep_overlap", record_method("overlap", overlap))
    monkeypatch.setattr(EPAnalyzer, "analyze_token_dropping", record_method("drop", drop))
    monkeypatch.setattr(EPAnalyzer, "analyze_router_loss_drift", record_method("loss", loss))
    monkeypatch.setattr(EPAnalyzer, "analyze_router_health", record_method("health", health))
    monkeypatch.setattr(EPAnalyzer, "diagnose_ep_stragglers", record_method("straggler", straggler))

    report_names = (
        "_write_expert_balance_report",
        "_write_ep_comm_report",
        "_write_ep_overlap_report",
        "_write_token_drop_report",
        "_write_router_loss_report",
        "_write_router_health_report",
        "_write_ep_straggler_report",
        "_write_layer_root_cause_topk",
    )
    plot_names = (
        "generate_expert_balance_plots",
        "generate_token_distribution_plots",
        "generate_ep_comm_plots",
        "generate_ep_overlap_plots",
        "generate_token_drop_plots",
        "generate_router_health_plots",
        "generate_ep_straggler_plots",
    )
    for name in report_names + plot_names:
        monkeypatch.setattr(
            ep_module, name, lambda *args, _name=name, **kwargs: calls.append(_name)
        )

    result = analyze_ep_traces([], output_dir=str(tmp_path))

    assert tuple(result) == (
        "balance_data",
        "comm_data",
        "overlap_data",
        "drop_data",
        "loss_data",
        "health_data",
        "straggler_data",
    )
    assert result == {
        "balance_data": balance,
        "comm_data": comm,
        "overlap_data": overlap,
        "drop_data": drop,
        "loss_data": loss,
        "health_data": health,
        "straggler_data": straggler,
    }
    assert calls == [
        "balance",
        "_write_expert_balance_report",
        "comm",
        "_write_ep_comm_report",
        "overlap",
        "_write_ep_overlap_report",
        "drop",
        "_write_token_drop_report",
        "loss",
        "_write_router_loss_report",
        "health",
        "_write_router_health_report",
        "straggler",
        "_write_ep_straggler_report",
        "_write_layer_root_cause_topk",
        *plot_names,
    ]

    expected_files = {
        "ep_diagnostic_report.txt",
        "ep_token_distribution_rank_expert_stats.csv",
        "ep_token_distribution_rank_iteration_load.csv",
        "ep_expert_balance_stats.csv",
        "ep_comm_overhead_stats.csv",
        "ep_overlap_stats.csv",
        "ep_token_drop_stats.csv",
        "ep_router_metrics.csv",
        "ep_router_loss_stats.csv",
        "ep_straggler_stats.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files


def test_master_accepts_target_nullable_router_and_expert_payloads(tmp_path: Path) -> None:
    traces = [
        _span(
            "moe-router",
            0,
            10,
            rank=0,
            layer=3,
            num_tokens=8,
            routed_tokens=8,
            dropped_tokens=None,
            drop_rate=None,
            expert_cv=0.2,
            top1_expert_share=0.3,
            routing_entropy=0.5,
        ),
        _span(
            "moe-router",
            20,
            10,
            rank=1,
            layer=4,
            num_tokens=8,
            routed_tokens=None,
            dropped_tokens=None,
            drop_rate=None,
            expert_cv=None,
            top1_expert_share=None,
            routing_entropy=0.5,
        ),
        _span(
            "moe-experts",
            40,
            10,
            rank=1,
            layer=5,
            routed_tokens=None,
            expert_cv=None,
            top1_expert_share=None,
            expert_max_over_mean=None,
            tokens_per_expert=None,
        ),
    ]

    result = analyze_ep_traces(traces, output_dir=str(tmp_path))

    assert tuple(result) == (
        "balance_data",
        "comm_data",
        "overlap_data",
        "drop_data",
        "loss_data",
        "health_data",
        "straggler_data",
    )
    assert len(result["balance_data"]) == 1
    assert result["balance_data"][0]["expert_cv"] == 0.2
    assert result["balance_data"][0]["top1_expert_share"] == 0.3
    assert result["balance_data"][0]["routing_entropy"] is None
    assert result["drop_data"] == []
    assert result["health_data"][0]["routing_entropy"] is None
    assert result["health_data"][0]["drop_rate"] is None
    assert (tmp_path / "ep_diagnostic_report.txt").is_file()
    assert (tmp_path / "ep_expert_balance_stats.csv").is_file()
    assert (tmp_path / "ep_router_metrics.csv").is_file()


def test_ep_analyzer_remains_offline_without_core_or_training_wiring() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (repository_root / "megatron/core", repository_root / "megatron/training")
        for path in root.rglob("*.py")
    )

    assert "megatron.megalens.ep_analyzer" not in source_text
    assert "analyze_ep_traces" not in source_text
