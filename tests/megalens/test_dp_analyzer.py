# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from megatron.megalens import dp_analyzer as dp_module
from megatron.megalens.data_loader import SpanEvent, TraceDataLoader
from megatron.megalens.dp_analyzer import DPAnalyzer, analyze_dp_traces


_STRAGGLER_NAMES = (
    "allreduce",
    "grad-sync",
    "backward-cooldown",
    "dp-reduce-scatter",
    "dp-allreduce",
    "all-grads-sync",
)

_GRAD_SYNC_NAMES = (
    "grad-sync",
    "all-grads-sync",
    "allreduce",
    "_reduce_scatter_along_first_dim",
    "_reduce_scatter_along_last_dim",
    "dp-reduce-scatter",
    "dp-allreduce",
    "sp-layernorm-allreduce",
    "embedding-grads-allreduce",
)

_COMPUTE_NAMES = ("backward-step", "backward", "backward-cooldown")

_COMM_NAMES = (
    "_reduce_scatter_along_first_dim",
    "_reduce_scatter_along_last_dim",
    "allreduce",
    "_reduce",
    "grad-sync",
    "dp-reduce-scatter",
    "dp-allreduce",
    "tp-reduce-scatter",
    "tp-allreduce",
    "tp-all-gather-first",
    "tp-all-gather-last",
    "tp-reduce-scatter-last",
)

def _span(
    name: str,
    ts: int,
    dur: int,
    *,
    rank: int = 0,
    iteration: int = 7,
    dp_rank: int | None = None,
    **args: Any,
) -> dict[str, Any]:
    event_args: dict[str, Any] = {
        "iteration": iteration,
        "dp_rk": rank if dp_rank is None else dp_rank,
        "pp_rk": 0,
        "tp_rk": 0,
    }
    event_args.update(args)
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "pid": rank, "args": event_args}


def _loader(*events: dict[str, Any]) -> TraceDataLoader:
    return TraceDataLoader.from_traces(list(events))


class _RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, int | None]] = []

    def get_dp_ranks(self) -> list[int]:
        return [0, 1]

    def get_ranks(self) -> list[int]:
        return [0, 1]

    def get_events_by_name(
        self, name: str, rank: int | None = None, iteration: int | None = None
    ) -> list[SpanEvent]:
        self.calls.append((name, rank, iteration))
        return []

    def get_iteration_events(self, iteration: int | None = None) -> list[SpanEvent]:
        return []


def _queried_names(loader: _RecordingLoader, rank: int | None) -> tuple[str, ...]:
    return tuple(name for name, query_rank, _ in loader.calls if query_rank == rank)


def test_public_signatures_and_lazy_reporting_import_match_port_contract() -> None:
    assert tuple(inspect.signature(DPAnalyzer).parameters) == ("loader",)
    assert tuple(inspect.signature(DPAnalyzer.diagnose_stragglers).parameters) == (
        "self",
        "sync_event_names",
        "iteration",
    )
    assert tuple(inspect.signature(DPAnalyzer.analyze_step_time_balance).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(DPAnalyzer.analyze_grad_sync_overhead).parameters) == (
        "self",
        "sync_names",
        "iteration",
    )
    assert tuple(inspect.signature(DPAnalyzer.analyze_memory_efficiency).parameters) == (
        "self",
        "iteration",
    )
    assert tuple(inspect.signature(DPAnalyzer.analyze_comm_overlap).parameters) == (
        "self",
        "compute_event_names",
        "comm_event_names",
        "iteration",
    )
    assert not hasattr(DPAnalyzer, "analyze_collective_lifecycle")

    master = inspect.signature(analyze_dp_traces)
    assert tuple(master.parameters) == ("traces", "output_dir")
    assert master.parameters["output_dir"].default == "."
    assert "plt" not in dp_module.__dict__
    assert "pd" not in dp_module.__dict__


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
from megatron.megalens.dp_analyzer import DPAnalyzer
assert DPAnalyzer
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", source], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_source_default_event_lists_and_exact_name_queries_are_preserved() -> None:
    loader = _RecordingLoader()
    analyzer = DPAnalyzer(loader)  # type: ignore[arg-type]

    analyzer.diagnose_stragglers(iteration=7)
    assert _queried_names(loader, 0) == _STRAGGLER_NAMES
    assert _queried_names(loader, 1) == _STRAGGLER_NAMES

    loader.calls.clear()
    analyzer.analyze_step_time_balance(iteration=7)
    assert _queried_names(loader, 0) == ("forward-step", "backward-step")
    assert _queried_names(loader, 1) == ("forward-step", "backward-step")

    loader.calls.clear()
    analyzer.analyze_grad_sync_overhead(iteration=7)
    assert _queried_names(loader, 0) == _GRAD_SYNC_NAMES
    assert _queried_names(loader, 1) == _GRAD_SYNC_NAMES

    loader.calls.clear()
    analyzer.analyze_comm_overlap(iteration=7)
    assert _queried_names(loader, 0) == _COMPUTE_NAMES + _COMM_NAMES
    assert _queried_names(loader, 1) == _COMPUTE_NAMES + _COMM_NAMES
    assert {query_iteration for _, _, query_iteration in loader.calls} == {7}

def test_grad_sync_preserves_source_raw_sum_for_nested_events() -> None:
    row = DPAnalyzer(
        _loader(
            _span("iteration", 0, 1_000),
            _span("grad-sync", 100, 100),
            _span("all-grads-sync", 110, 80),
            _span("dp-allreduce", 120, 40),
        )
    ).analyze_grad_sync_overhead()[0]

    assert row == {
        "rank": 0,
        "iteration": 7,
        "grad_sync_us": 220.0,
        "iter_time_us": 1_000.0,
        "sync_ratio": 0.22,
        "n_sync_events": 3,
    }


def test_straggler_preserves_earliest_same_name_event_per_rank() -> None:
    rows = DPAnalyzer(
        _loader(
            _span("iteration", 0, 1_000, rank=0),
            _span("iteration", 0, 1_000, rank=1),
            _span("grad-sync", 100, 20, rank=0),
            _span("grad-sync", 300, 20, rank=0),
            _span("grad-sync", 200, 20, rank=1),
        )
    ).diagnose_stragglers(sync_event_names=["grad-sync"])

    assert rows == [
        {
            "iteration": 7,
            "sync_event": "grad-sync",
            "gap_us": 100.0,
            "straggler_rank": 1,
            "fastest_rank": 0,
            "per_rank_start_ts": {0: 100, 1: 200},
            "hardware_diagnosis": None,
            "hw_detail": None,
        }
    ]


def test_comm_overlap_preserves_interval_union_and_tp_default_names() -> None:
    rows = DPAnalyzer(
        _loader(
            _span("backward-step", 0, 200),
            _span("dp-allreduce", 50, 100),
            _span("tp-allreduce", 180, 80),
        )
    ).analyze_comm_overlap()

    assert rows == [
        {
            "rank": 0,
            "iteration": 7,
            "total_compute_us": 200.0,
            "total_comm_us": 180.0,
            "overlap_us": 120.0,
            "exposed_comm_us": 60.0,
            "overlap_ratio": 0.6667,
        }
    ]


def test_empty_inputs_and_master_result_keys_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from megatron.megalens import paper_style

    analyzer = DPAnalyzer(_loader())
    assert analyzer.diagnose_stragglers() == []
    assert analyzer.analyze_step_time_balance() == []
    assert analyzer.analyze_grad_sync_overhead() == []
    assert analyzer.analyze_memory_efficiency() == []
    assert analyzer.analyze_comm_overlap() == []

    monkeypatch.setattr(dp_module, "_load_reporting_dependencies", lambda: (None, None))
    monkeypatch.setattr(paper_style, "apply_global_rcparams", lambda: None)
    result = analyze_dp_traces([], output_dir=str(tmp_path))

    assert tuple(result) == (
        "straggler_data",
        "balance_data",
        "sync_data",
        "mem_data",
        "overlap_data",
    )
    assert all(value == [] for value in result.values())
    report = (tmp_path / "dp_diagnostic_report.txt").read_text(encoding="utf-8")
    assert "DP Collective Lifecycle Evidence Report" not in report
