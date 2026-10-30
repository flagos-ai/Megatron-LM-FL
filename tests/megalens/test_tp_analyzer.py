# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from megatron.megalens import tp_analyzer as tp_module
from megatron.megalens.data_loader import SpanEvent, TraceDataLoader
from megatron.megalens.tp_analyzer import TPAnalyzer, analyze_tp_traces

_LAUNCH_NAMES = (
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
)

_NVLINK_NAMES = (
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
)

_COMM_NAMES = (
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
)

_COMPUTE_NAMES = (
    "forward-step",
    "backward-step",
    "_forward_attention",
    "_forward_mlp",
    "MLP.forward",
    "attention",
)

_FRAGMENTATION_NAMES = (
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
)

_STRAGGLER_NAMES = (
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
)


def _span(
    name: str, ts: int, dur: int, *, rank: int = 0, iteration: int = 7, **args: Any
) -> dict[str, Any]:
    event_args: dict[str, Any] = {"iteration": iteration}
    event_args.update(args)
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "pid": rank, "args": event_args}


def _iteration(rank: int, tp_rank: int, *, iteration: int = 7) -> dict[str, Any]:
    return _span(
        "iteration", 0, 10_000, rank=rank, iteration=iteration, dp_rk=0, pp_rk=0, tp_rk=tp_rank
    )


def _counter(ts: int, rank: int = 0, *, iteration: int = 7, **metrics: float) -> dict[str, Any]:
    args: dict[str, Any] = {"iteration": iteration}
    args.update(metrics)
    return {"name": "GPU_Metrics", "ph": "C", "ts": ts, "pid": rank, "args": args}


def _loader(*events: dict[str, Any]) -> TraceDataLoader:
    return TraceDataLoader.from_traces(list(events))


class _RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, int | None]] = []

    def get_ranks(self) -> list[int]:
        return [0, 1]

    def get_events_by_name(
        self, name: str, rank: int | None = None, iteration: int | None = None
    ) -> list[SpanEvent]:
        self.calls.append((name, rank, iteration))
        if name == "iteration" and rank is not None:
            return [
                SpanEvent(
                    name="iteration",
                    ts=0,
                    dur=10_000,
                    rank=rank,
                    args={"iteration": 7, "dp_rk": 0, "pp_rk": 0, "tp_rk": rank},
                )
            ]
        return []

    def get_hardware_metrics_in_window(self, *args: Any, **kwargs: Any) -> None:
        return None


def _queried_names(loader: _RecordingLoader, rank: int | None) -> tuple[str, ...]:
    return tuple(
        name for name, query_rank, _ in loader.calls if name != "iteration" and query_rank == rank
    )


def _assert_query_iteration(loader: _RecordingLoader, expected: int) -> None:
    assert loader.calls
    assert {iteration for name, _, iteration in loader.calls if name != "iteration"} == {expected}
    assert {iteration for name, _, iteration in loader.calls if name == "iteration"} <= {
        None,
        expected,
    }


def test_public_signatures_and_lazy_reporting_import_match_the_port_contract() -> None:
    constructor = inspect.signature(TPAnalyzer)
    assert tuple(constructor.parameters) == ("loader", "nvlink_theory_peak_gbps")
    assert constructor.parameters["nvlink_theory_peak_gbps"].default == 300.0

    master = inspect.signature(analyze_tp_traces)
    assert tuple(master.parameters) == ("traces", "nvlink_theory_peak_gbps", "output_dir")
    assert master.parameters["nvlink_theory_peak_gbps"].default is None
    assert master.parameters["output_dir"].default == "."

    assert "plt" not in tp_module.__dict__
    assert "pd" not in tp_module.__dict__


def test_base_install_import_does_not_require_reporting_dependencies() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(repository_root)!r})

class BlockReportingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pandas" or fullname.startswith("matplotlib"):
            raise AssertionError(f"unexpected reporting import: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockReportingImports())
from megatron.megalens.tp_analyzer import TPAnalyzer
assert TPAnalyzer
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", source], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_source_default_event_lists_and_exact_name_queries_are_preserved() -> None:
    loader = _RecordingLoader()
    analyzer = TPAnalyzer(loader)  # type: ignore[arg-type]

    analyzer.analyze_kernel_launch_overhead(iteration=7)
    assert _queried_names(loader, None) == _LAUNCH_NAMES
    _assert_query_iteration(loader, 7)

    loader.calls.clear()
    analyzer.analyze_nvlink_saturation(iteration=7)
    assert _queried_names(loader, None) == _NVLINK_NAMES
    _assert_query_iteration(loader, 7)

    loader.calls.clear()
    analyzer.analyze_tp_comm_overhead(iteration=7)
    assert _queried_names(loader, 0) == _COMM_NAMES + _COMPUTE_NAMES
    _assert_query_iteration(loader, 7)

    loader.calls.clear()
    analyzer.analyze_gpu_sm_efficiency(iteration=7)
    assert _queried_names(loader, 0) == ("forward-step", "backward-step")
    _assert_query_iteration(loader, 7)

    loader.calls.clear()
    analyzer.analyze_compute_fragmentation(iteration=7)
    assert _queried_names(loader, 0) == _FRAGMENTATION_NAMES
    _assert_query_iteration(loader, 7)

    loader.calls.clear()
    analyzer.diagnose_tp_stragglers(iteration=7)
    assert _queried_names(loader, 0) == _STRAGGLER_NAMES
    _assert_query_iteration(loader, 7)


def test_explicit_empty_event_lists_disable_source_defaults() -> None:
    loader = _RecordingLoader()
    analyzer = TPAnalyzer(loader)  # type: ignore[arg-type]

    assert analyzer.analyze_kernel_launch_overhead(gemm_event_names=[]) == []
    assert analyzer.analyze_nvlink_saturation(sp_comm_event_names=[]) == []
    assert analyzer.analyze_tp_comm_overhead(tp_comm_names=[], compute_names=[]) == []
    analyzer.analyze_gpu_sm_efficiency(compute_names=[])
    assert analyzer.analyze_compute_fragmentation(kernel_names=[]) == []
    assert analyzer.diagnose_tp_stragglers(sync_event_names=[]) == []
    assert all(name == "iteration" for name, _, _ in loader.calls)


def test_role_aware_comm_overhead_uses_interval_union_and_exposes_nested_evidence() -> None:
    analyzer = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 200),
            _span("backward-step", 150, 100),
            _span("tp-reduce-scatter-last", 300, 100),
            _span("tp-reduce-scatter", 320, 60),
            _span("reduce-scatter", 500, 40),
            _span("tp-reduce-scatter", 700, 999, iteration=8),
        )
    )

    rows = analyzer.analyze_tp_comm_overhead(iteration=7)

    assert len(rows) == 1
    row = rows[0]
    assert row["rank"] == 0
    assert row["iteration"] == 7
    assert row["total_compute_us"] == 250.0
    assert row["total_tp_comm_us"] == 140.0
    assert row["comm_ratio"] == 0.359
    assert row["n_comm_events"] == 2
    assert row["aggregation_mode"] == "interval_union_us"
    assert row["aggregation_status"] == "available"
    assert row["comm_ratio_status"] == "available"

    evidence = row["tp_reduce_scatter_metrics"]
    assert evidence["phase_wall_us"]["value"] == 100
    assert evidence["phase_wall_us"]["status"] == "available"
    assert evidence["leaf_sum_us"]["value"] == 100
    assert evidence["leaf_sum_us"]["status"] == "available"
    assert evidence["interval_union_us"]["value"] == 140
    assert evidence["interval_union_us"]["status"] == "available"
    assert evidence["leaf_data_bytes"]["value"] is None
    assert evidence["leaf_data_bytes"]["status"] == "partial"
    json.dumps(rows)


def test_incomplete_nested_evidence_does_not_emit_comm_kpis() -> None:
    parent_only = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 100),
            _span("tp-reduce-scatter-last", 200, 80, data_bytes=4_096),
        )
    ).analyze_tp_comm_overhead()

    assert len(parent_only) == 1
    row = parent_only[0]
    assert row["total_compute_us"] == 100.0
    assert row["total_tp_comm_us"] is None
    assert row["comm_ratio"] is None
    assert row["n_comm_events"] == 0
    assert row["aggregation_status"] == "partial"
    assert row["comm_ratio_status"] == "partial"
    assert row["tp_reduce_scatter_metrics"]["phase_wall_us"]["value"] == 80
    assert row["tp_reduce_scatter_metrics"]["interval_union_us"] == {
        "value": 80,
        "status": "partial",
        "reason": (
            "merged role-valid observed intervals: "
            "one or more composite spans have no observed physical leaf"
        ),
    }

    conflicting = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 100),
            _span("tp-reduce-scatter-last", 200, 80, op="reduce-scatter", dim="last"),
            _span("tp-reduce-scatter", 210, 50, op="reduce-scatter", dim="last", data_bytes=4_096),
        )
    ).analyze_tp_comm_overhead()

    assert len(conflicting) == 1
    row = conflicting[0]
    assert row["total_tp_comm_us"] is None
    assert row["comm_ratio"] is None
    assert row["n_comm_events"] == 0
    assert row["aggregation_status"] == "unknown"
    assert row["comm_ratio_status"] == "unknown"
    assert "expected dim=first" in row["aggregation_reason"]


def test_comm_overhead_reduces_each_rank_iteration_partition_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tp_module.aggregate_tp_reduce_scatter_partition
    calls: list[tuple[tuple[int, int], ...]] = []

    def record_partition(events: list[SpanEvent]) -> Any:
        calls.append(tuple((event.rank, event.iteration) for event in events))
        return original(events)

    monkeypatch.setattr(tp_module, "aggregate_tp_reduce_scatter_partition", record_partition)
    analyzer = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 100, rank=0, iteration=7),
            _span("tp-reduce-scatter-last", 200, 80, rank=0, iteration=7),
            _span("tp-reduce-scatter", 210, 50, rank=0, iteration=7, data_bytes=100),
            _span("forward-step", 0, 100, rank=0, iteration=8),
            _span("reduce-scatter", 200, 40, rank=0, iteration=8, data_bytes=100),
            _span("forward-step", 0, 100, rank=1, iteration=7),
            _span("tp-reduce-scatter", 200, 40, rank=1, iteration=7, data_bytes=100),
        )
    )

    rows = analyzer.analyze_tp_comm_overhead()

    assert [(row["rank"], row["iteration"]) for row in rows] == [(0, 7), (0, 8), (1, 7)]
    assert calls == [((0, 7), (0, 7)), ((0, 8),), ((1, 7),)]


def test_missing_compute_or_communication_evidence_does_not_create_zero_ratios() -> None:
    compute_only = TPAnalyzer(_loader(_span("forward-step", 0, 100))).analyze_tp_comm_overhead()
    assert compute_only[0]["total_compute_us"] == 100.0
    assert compute_only[0]["total_tp_comm_us"] is None
    assert compute_only[0]["comm_ratio"] is None
    assert compute_only[0]["aggregation_status"] == "unavailable"
    assert compute_only[0]["comm_ratio_status"] == "unavailable"

    comm_only = TPAnalyzer(_loader(_span("tp-allreduce", 0, 100))).analyze_tp_comm_overhead()
    assert comm_only[0]["total_compute_us"] is None
    assert comm_only[0]["total_tp_comm_us"] == 100.0
    assert comm_only[0]["comm_ratio"] is None
    assert comm_only[0]["aggregation_status"] == "available"
    assert comm_only[0]["comm_ratio_status"] == "unavailable"
    assert "compute spans" in comm_only[0]["comm_ratio_reason"]


def test_noncatalog_communication_spans_also_use_interval_union() -> None:
    row = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 100),
            _span("tp-allreduce", 200, 100),
            _span("allreduce", 250, 100),
        )
    ).analyze_tp_comm_overhead()[0]

    assert row["total_tp_comm_us"] == 150.0
    assert row["comm_ratio"] == 0.6
    assert row["n_comm_events"] == 2
    assert row["aggregation_status"] == "available"


def test_finite_float_intervals_are_counted_and_invalid_intervals_fail_closed() -> None:
    float_row = TPAnalyzer(
        _loader(
            _span("forward-step", 0.25, 100.0),
            _span("tp-reduce-scatter-last", 200, 80),
            _span("tp-reduce-scatter", 210, 50, data_bytes=1_024),
            _span("tp-allreduce", 400.25, 25.5),
        )
    ).analyze_tp_comm_overhead()[0]

    assert float_row["total_compute_us"] == 100.0
    assert float_row["total_tp_comm_us"] == 105.5
    assert float_row["comm_ratio"] == 0.5134
    assert float_row["n_comm_events"] == 2
    assert float_row["aggregation_status"] == "available"

    invalid_row = TPAnalyzer(
        _loader(_span("tp-allreduce", "invalid", 25))
    ).analyze_tp_comm_overhead()[0]

    assert invalid_row["total_tp_comm_us"] is None
    assert invalid_row["comm_ratio"] is None
    assert invalid_row["n_comm_events"] == 0
    assert invalid_row["aggregation_status"] == "unknown"
    assert invalid_row["comm_ratio_status"] == "unknown"
    assert "invalid timestamp or duration" in invalid_row["aggregation_reason"]

    precise_integer_row = TPAnalyzer(
        _loader(_span("tp-allreduce", 2**53, 1))
    ).analyze_tp_comm_overhead()[0]
    assert precise_integer_row["total_tp_comm_us"] == 1.0
    assert precise_integer_row["aggregation_status"] == "available"

    collapsed_float_row = TPAnalyzer(
        _loader(_span("tp-allreduce", 1.7e15, 0.1))
    ).analyze_tp_comm_overhead()[0]
    assert collapsed_float_row["total_tp_comm_us"] is None
    assert collapsed_float_row["aggregation_status"] == "unknown"


def test_malformed_cataloged_timing_reaches_reducer_and_fails_closed() -> None:
    row = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 100), _span("tp-reduce-scatter", 200, None, data_bytes=1_024)
        )
    ).analyze_tp_comm_overhead()[0]

    assert row["total_tp_comm_us"] is None
    assert row["comm_ratio"] is None
    assert row["n_comm_events"] == 0
    assert row["aggregation_status"] == "unknown"
    assert "timestamps and durations" in row["aggregation_reason"]


def test_invalid_partition_identity_is_grouped_and_fails_closed() -> None:
    rows = TPAnalyzer(
        _loader(
            _span("forward-step", 0, 100, iteration="bad"),
            _span("tp-allreduce", 200, 25, iteration="bad"),
        )
    ).analyze_tp_comm_overhead()

    assert len(rows) == 1
    row = rows[0]
    assert row["rank"] == 0
    assert row["iteration"] == -1
    assert row["total_compute_us"] is None
    assert row["total_tp_comm_us"] is None
    assert row["comm_ratio"] is None
    assert row["aggregation_status"] == "unknown"
    assert row["comm_ratio_status"] == "unknown"
    assert "rank and iteration" in row["aggregation_reason"]
    assert "rank and iteration" in row["comm_ratio_reason"]


def test_legacy_numeric_comm_rows_remain_complete_for_reporting() -> None:
    legacy_row = {
        "rank": 0,
        "iteration": 7,
        "total_compute_us": 100.0,
        "total_tp_comm_us": 25.0,
        "comm_ratio": 0.2,
        "n_comm_events": 1,
    }

    assert tp_module._has_complete_comm_ratio(legacy_row)
    assert tp_module._comm_ratio_status_label(legacy_row) == "legacy"


def test_physical_analyses_exclude_nested_composite_samples() -> None:
    two_mib = 2 * 1024 * 1024
    analyzer = TPAnalyzer(
        _loader(
            _span("tp-reduce-scatter-last", 100, 10, duration_wall=30, data_bytes=two_mib),
            _span("tp-reduce-scatter", 102, 5, duration_wall=15, data_bytes=two_mib, group_size=2),
        ),
        nvlink_theory_peak_gbps=10.0,
    )

    launch = analyzer.analyze_kernel_launch_overhead()
    assert [row["event_name"] for row in launch] == ["tp-reduce-scatter"]

    nvlink = analyzer.analyze_nvlink_saturation()
    assert [row["event_name"] for row in nvlink] == ["tp-reduce-scatter"]
    assert nvlink[0]["data_bytes"] == two_mib

    fragmentation = analyzer.analyze_compute_fragmentation()
    assert fragmentation[0]["total_events"] == 1
    assert fragmentation[0]["mean_dur_us"] == 5.0


def test_async_lifecycle_spans_are_excluded_from_physical_and_comm_metrics() -> None:
    two_mib = 2 * 1024 * 1024
    names = ["tp-linear-async-launch", "tp-linear-async-complete", "tp-all-gather-first"]
    analyzer = TPAnalyzer(
        _loader(
            _span("tp-linear-async-launch", 80, 10, duration_wall=40, data_bytes=two_mib),
            _span(
                "tp-all-gather-first", 100, 5, duration_wall=20, data_bytes=two_mib, group_size=2
            ),
            _counter(102, NVLink_Tx_MBs=100.0),
            _span("tp-linear-async-complete", 120, 20, duration_wall=80, data_bytes=two_mib),
            _span("forward-step", 0, 200),
        ),
        nvlink_theory_peak_gbps=10.0,
    )

    launch = analyzer.analyze_kernel_launch_overhead(gemm_event_names=names)
    assert [row["event_name"] for row in launch] == ["tp-all-gather-first"]

    nvlink = analyzer.analyze_nvlink_saturation(sp_comm_event_names=names)
    assert [row["event_name"] for row in nvlink] == ["tp-all-gather-first"]

    fragmentation = analyzer.analyze_compute_fragmentation(kernel_names=names)
    assert fragmentation[0]["total_events"] == 1
    assert fragmentation[0]["mean_dur_us"] == 5.0

    overhead = analyzer.analyze_tp_comm_overhead(
        tp_comm_names=names, compute_names=["forward-step"]
    )
    assert len(overhead) == 1
    assert overhead[0]["total_tp_comm_us"] == 5.0
    assert overhead[0]["n_comm_events"] == 1
    assert overhead[0]["aggregation_status"] == "available"


def test_async_lifecycle_only_input_has_no_communication_evidence() -> None:
    analyzer = TPAnalyzer(
        _loader(
            _span("tp-linear-async-launch", 80, 10, data_bytes=2 * 1024 * 1024),
            _span("tp-linear-async-complete", 120, 20, data_bytes=2 * 1024 * 1024),
            _span("forward-step", 0, 200),
        )
    )

    rows = analyzer.analyze_tp_comm_overhead(
        tp_comm_names=["tp-linear-async-launch", "tp-linear-async-complete"],
        compute_names=["forward-step"],
    )

    assert len(rows) == 1
    assert rows[0]["total_tp_comm_us"] is None
    assert rows[0]["n_comm_events"] == 0
    assert rows[0]["aggregation_status"] == "unavailable"


def test_async_lifecycle_only_input_cannot_form_a_straggler_group() -> None:
    analyzer = TPAnalyzer(
        _loader(
            _iteration(0, 0),
            _iteration(1, 1),
            _span("tp-linear-async-launch", 100, 5, rank=0),
            _span("tp-linear-async-launch", 300, 5, rank=1),
            _span("tp-linear-async-complete", 200, 10, rank=0),
            _span("tp-linear-async-complete", 500, 10, rank=1),
        )
    )

    assert (
        analyzer.diagnose_tp_stragglers(
            sync_event_names=["tp-linear-async-launch", "tp-linear-async-complete"]
        )
        == []
    )


def test_nvlink_payload_evidence_skips_unjustified_utilisation_judgement() -> None:
    missing_payload = TPAnalyzer(
        _loader(_span("tp-reduce-scatter", 100, 20), _counter(110, NVLink_Tx_MBs=100.0)),
        nvlink_theory_peak_gbps=10.0,
    )

    row = missing_payload.analyze_nvlink_saturation()[0]
    assert row["data_bytes"] is None
    assert row["achieved_bw_mbs"] == 100.0
    assert row["utilisation_pct"] == 0.98
    assert row["diagnosis"] == "Payload unavailable: skip low-util judgement"
    assert missing_payload.summarise_nvlink_utilisation() == [
        {
            "rank": 0,
            "num_events": 1,
            "num_events_judged": 0,
            "mean_utilisation_pct": None,
            "min_utilisation_pct": None,
            "num_flagged_low": 0,
        }
    ]

    malformed_payload = TPAnalyzer(
        _loader(
            _span("tp-reduce-scatter", 100, 20, data_bytes="bad"),
            _counter(110, NVLink_Tx_MBs=100.0),
        ),
        nvlink_theory_peak_gbps=10.0,
    ).analyze_nvlink_saturation()[0]
    assert malformed_payload["data_bytes"] is None
    assert malformed_payload["achieved_bw_mbs"] == 100.0
    assert malformed_payload["diagnosis"] == "Payload unavailable: skip low-util judgement"


def test_nvlink_plot_omits_missing_utilisation_and_preserves_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plt, _ = tp_module._load_reporting_dependencies()
    from matplotlib.axes import Axes

    original_bar = Axes.bar
    bar_calls: list[tuple[list[float], list[str]]] = []

    def record_bar(self: Axes, x: Any, height: Any, *args: Any, **kwargs: Any) -> Any:
        if x is not None:
            bar_calls.append((list(height), list(kwargs["color"])))
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", record_bar)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    tp_module.generate_nvlink_plots(
        [
            {
                "rank": 0,
                "ts": 0,
                "achieved_bw_mbs": 1_000.0,
                "utilisation_pct": 10.0,
                "event_name": "tp-allreduce",
            }
        ],
        [
            {"rank": 0, "mean_utilisation_pct": None},
            {"rank": 1, "mean_utilisation_pct": 0.0},
            {"rank": 2, "mean_utilisation_pct": 30.0},
        ],
        10_000.0,
        str(tmp_path),
    )

    assert bar_calls == [([0.0, 30.0], ["salmon", "mediumseagreen"])]


def test_straggler_uses_physical_leaf_and_normalises_legacy_alias() -> None:
    analyzer = TPAnalyzer(
        _loader(
            _iteration(0, 0),
            _iteration(1, 1),
            _span("tp-reduce-scatter-last", 100, 40, rank=0),
            _span("tp-reduce-scatter", 110, 20, rank=0, data_bytes=1_024),
            _span("tp-reduce-scatter-last", 300, 40, rank=1),
            _span("reduce-scatter", 310, 20, rank=1, data_bytes=1_024),
        )
    )

    rows = analyzer.diagnose_tp_stragglers()

    assert len(rows) == 1
    assert rows[0]["sync_event"] == "tp-reduce-scatter"
    assert rows[0]["per_rank_start_ts"] == {0: 110, 1: 310}
    assert rows[0]["gap_us"] == 200.0
    assert rows[0]["straggler_rank"] == 1


def test_partial_comm_report_skips_kpi_judgement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = TPAnalyzer(
        _loader(_span("forward-step", 0, 100), _span("tp-reduce-scatter-last", 200, 80))
    ).analyze_tp_comm_overhead()
    report_path = tmp_path / "partial-report.txt"
    logger = tp_module._ReportLogger(str(report_path), "Partial Evidence")

    tp_module._write_tp_comm_overhead_report(rows, logger)

    output = capsys.readouterr().out
    report = report_path.read_text(encoding="utf-8")
    assert "partial=1" in output
    assert "KPI judgement skipped" in report
    assert "healthy" not in report
    assert "nan" not in report.lower()


def test_launch_overhead_fields_thresholds_and_natural_errors_are_preserved() -> None:
    analyzer = TPAnalyzer(
        _loader(
            _span("tp-allreduce", 100, 10, duration_wall=25),
            _span("_reduce", 200, 10, duration_wall=20),
            _span("tp-allreduce", 300, 20, duration_wall=100),
        )
    )

    assert analyzer.analyze_kernel_launch_overhead() == [
        {
            "rank": 0,
            "iteration": 7,
            "event_name": "tp-allreduce",
            "ts": 100,
            "cuda_dur_us": 10.0,
            "wall_dur_us": 25.0,
            "overhead_ratio": 2.5,
            "diagnosis": (
                "CPU-Bound: Kernel Launch Overhead " "(wall 25.0 μs vs cuda 10.0 μs, 2.5× ratio)"
            ),
        }
    ]

    malformed = TPAnalyzer(_loader(_span("tp-allreduce", 0, 5, duration_wall="bad")))
    with pytest.raises(ValueError):
        malformed.analyze_kernel_launch_overhead()


def test_nvlink_counter_fallback_summary_and_result_fields_match_source() -> None:
    two_mib = 2 * 1024 * 1024
    analyzer = TPAnalyzer(
        _loader(
            _span("tp-allreduce", 0, 1_000, data_bytes=two_mib, group_size=2),
            _span("tp-all-gather-first", 2_000, 1_000, data_bytes=1_024, group_size=2),
            _span("tp-reduce-scatter", 4_000, 100, data_bytes=two_mib, group_size=2),
            _counter(4_050, NVLink_Tx_MBs=5_000.0),
        ),
        nvlink_theory_peak_gbps=10.0,
    )

    rows = analyzer.analyze_nvlink_saturation()
    assert [set(row) for row in rows] == [
        {
            "rank",
            "iteration",
            "event_name",
            "ts",
            "dur_us",
            "data_bytes",
            "group_size",
            "achieved_bw_mbs",
            "peak_bw_mbs",
            "utilisation_pct",
            "diagnosis",
        }
    ] * 3
    by_name = {row["event_name"]: row for row in rows}
    assert by_name["tp-allreduce"]["achieved_bw_mbs"] == 2_000.0
    assert by_name["tp-allreduce"]["utilisation_pct"] == 19.53
    assert str(by_name["tp-allreduce"]["diagnosis"]).startswith("Low NVLink Utilisation")
    assert by_name["tp-all-gather-first"]["achieved_bw_mbs"] == 1.0
    assert str(by_name["tp-all-gather-first"]["diagnosis"]).startswith("Small payload")
    assert by_name["tp-reduce-scatter"]["achieved_bw_mbs"] == 5_000.0
    assert by_name["tp-reduce-scatter"]["utilisation_pct"] == 48.83

    assert analyzer.summarise_nvlink_utilisation() == [
        {
            "rank": 0,
            "num_events": 3,
            "num_events_judged": 2,
            "mean_utilisation_pct": 34.18,
            "min_utilisation_pct": 19.53,
            "num_flagged_low": 1,
        }
    ]

    malformed = TPAnalyzer(_loader(_span("tp-allreduce", 0, 5, data_bytes="bad")))
    with pytest.raises(ValueError):
        malformed.analyze_nvlink_saturation()


def test_sm_efficiency_and_fragmentation_fields_match_source() -> None:
    sm_analyzer = TPAnalyzer(
        _loader(
            _iteration(0, 0),
            _iteration(1, 1),
            _span("forward-step", 100, 500),
            _counter(200, SM_Util_pct=40.0),
            _counter(300, SM_Util_pct=60.0),
        )
    )
    assert sm_analyzer.analyze_gpu_sm_efficiency() == [
        {
            "rank": 0,
            "mean_sm_util_pct": 50.0,
            "min_sm_util_pct": 40.0,
            "max_sm_util_pct": 60.0,
            "n_samples": 2,
            "diagnosis": None,
        },
        {
            "rank": 1,
            "mean_sm_util_pct": None,
            "min_sm_util_pct": None,
            "max_sm_util_pct": None,
            "n_samples": 0,
            "diagnosis": None,
        },
    ]

    frag_analyzer = TPAnalyzer(
        _loader(*[_span("tp-allreduce", index * 10, 5) for index in range(21)])
    )
    frag = frag_analyzer.analyze_compute_fragmentation()
    assert len(frag) == 1
    assert set(frag[0]) == {
        "rank",
        "total_events",
        "short_events",
        "short_ratio",
        "mean_dur_us",
        "median_dur_us",
        "diagnosis",
    }
    assert frag[0] == {
        "rank": 0,
        "total_events": 21,
        "short_events": 21,
        "short_ratio": 1.0,
        "mean_dur_us": 5.0,
        "median_dur_us": 5.0,
        "diagnosis": (
            "High kernel fragmentation: 100% of events < 10.0 μs.  "
            "Consider kernel fusion, reducing SP degree, or using CUDA Graphs."
        ),
    }


def test_tp_straggler_fields_and_compute_skew_diagnosis_match_source() -> None:
    analyzer = TPAnalyzer(
        _loader(
            _iteration(0, 0),
            _iteration(1, 1),
            _span("tp-allreduce", 1_000, 50, rank=0),
            _span("tp-allreduce", 1_300, 50, rank=1),
        )
    )

    assert analyzer.diagnose_tp_stragglers() == [
        {
            "iteration": 7,
            "sync_event": "tp-allreduce",
            "tp_group": "DP0-PP0",
            "gap_us": 300.0,
            "straggler_rank": 1,
            "fastest_rank": 0,
            "per_rank_start_ts": {0: 1_000, 1: 1_300},
            "likely_cause": "compute_skew",
            "hardware_diagnosis": None,
            "hw_detail": None,
        }
    ]


def test_empty_inputs_and_master_result_keys_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from megatron.megalens import paper_style

    analyzer = TPAnalyzer(_loader())
    assert analyzer.analyze_kernel_launch_overhead() == []
    assert analyzer.analyze_nvlink_saturation() == []
    assert analyzer.analyze_tp_comm_overhead() == []
    assert analyzer.analyze_gpu_sm_efficiency() == []
    assert analyzer.analyze_compute_fragmentation() == []
    assert analyzer.summarise_nvlink_utilisation() == []
    assert analyzer.diagnose_tp_stragglers() == []

    monkeypatch.setattr(tp_module, "_load_reporting_dependencies", lambda: (None, None))
    monkeypatch.setattr(paper_style, "apply_global_rcparams", lambda: None)
    result = analyze_tp_traces([], nvlink_theory_peak_gbps=10.0, output_dir=str(tmp_path))

    assert tuple(result) == (
        "tp_comm_data",
        "launch_overhead_data",
        "nvlink_data",
        "nvlink_summary",
        "sm_data",
        "frag_data",
        "straggler_data",
    )
    assert all(value == [] for value in result.values())
    assert (tmp_path / "tp_diagnostic_report.txt").is_file()
