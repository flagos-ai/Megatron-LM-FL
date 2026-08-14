# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Pytest conftest for DSA kernel tests — metrics collection and report generation.

Provides:
- ``dsa_metrics`` fixture: records accuracy, performance, and memory metrics
- ``--dsa-report`` CLI option: controls output path (default: test_results.md)
- Session-finish hook: writes collected metrics as Markdown tables

Usage:
    pytest tests/unit_tests/plugin/dsa_kernel/ -v --dsa-report=report.md
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


@dataclass
class AccuracyRecord:
    test_class: str
    test_name: str
    params: Dict[str, Any]
    cos_sim: float
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    target: str = "output"
    status: str = "PASS"


@dataclass
class PerformanceRecord:
    test_class: str
    test_name: str
    params: Dict[str, Any]
    fused_ms: float
    unfused_ms: float
    speedup: float
    label: str = "fwd"


@dataclass
class MemoryRecord:
    test_class: str
    test_name: str
    params: Dict[str, Any]
    fused_mb: float
    unfused_mb: float
    ratio: float


@dataclass
class MetricCollector:
    accuracy: List[AccuracyRecord] = field(default_factory=list)
    performance: List[PerformanceRecord] = field(default_factory=list)
    memory: List[MemoryRecord] = field(default_factory=list)

    def has_data(self) -> bool:
        return bool(self.accuracy or self.performance or self.memory)


class MetricRecorder:
    """Per-test facade for recording metrics."""

    def __init__(self, test_class: str, test_name: str, collector: MetricCollector):
        self._class = test_class
        self._name = test_name
        self._collector = collector

    def record_accuracy(self, params, cos_sim, max_diff=None, mean_diff=None,
                        target="output", status="PASS"):
        self._collector.accuracy.append(AccuracyRecord(
            self._class, self._name, params, cos_sim, max_diff, mean_diff, target, status))

    def record_performance(self, params, fused_ms, unfused_ms, speedup, label="fwd"):
        self._collector.performance.append(PerformanceRecord(
            self._class, self._name, params, fused_ms, unfused_ms, speedup, label))

    def record_memory(self, params, fused_mb, unfused_mb, ratio):
        self._collector.memory.append(MemoryRecord(
            self._class, self._name, params, fused_mb, unfused_mb, ratio))


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _param_cols(records):
    seen = {}
    for r in records:
        for k in r.params:
            seen.setdefault(k, None)
    return list(seen.keys())


def _md_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _render_accuracy(records):
    if not records:
        return "_No accuracy data._\n"
    pcols = _param_cols(records)
    hdrs = ["test_class", "test_name", "target"] + pcols + ["cos_sim", "max_diff", "mean_diff", "status"]
    lines = [_md_row(hdrs), _md_row(["---"] * len(hdrs))]
    for r in records:
        row = [r.test_class, r.test_name, r.target]
        row += [r.params.get(k, "") for k in pcols]
        row += [f"{r.cos_sim:.6f}",
                f"{r.max_diff:.4e}" if r.max_diff is not None else "-",
                f"{r.mean_diff:.4e}" if r.mean_diff is not None else "-",
                r.status]
        lines.append(_md_row(row))
    return "\n".join(lines) + "\n"


def _render_performance(records):
    if not records:
        return "_No performance data._\n"
    pcols = _param_cols(records)
    hdrs = ["test_class", "test_name", "label"] + pcols + ["fused_ms", "unfused_ms", "speedup"]
    lines = [_md_row(hdrs), _md_row(["---"] * len(hdrs))]
    for r in records:
        row = [r.test_class, r.test_name, r.label]
        row += [r.params.get(k, "") for k in pcols]
        row += [f"{r.fused_ms:.3f}", f"{r.unfused_ms:.3f}", f"{r.speedup:.2f}x"]
        lines.append(_md_row(row))
    return "\n".join(lines) + "\n"


def _render_memory(records):
    if not records:
        return "_No memory data._\n"
    pcols = _param_cols(records)
    hdrs = ["test_class", "test_name"] + pcols + ["fused_MB", "unfused_MB", "ratio"]
    lines = [_md_row(hdrs), _md_row(["---"] * len(hdrs))]
    for r in records:
        row = [r.test_class, r.test_name]
        row += [r.params.get(k, "") for k in pcols]
        row += [f"{r.fused_mb:.1f}", f"{r.unfused_mb:.1f}", f"{r.ratio:.2f}x"]
        lines.append(_md_row(row))
    return "\n".join(lines) + "\n"


def _generate_report(collector: MetricCollector) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"# DSA Fused Kernel Test Results\n",
        f"Generated: {now}\n",
        "## Accuracy\n",
        _render_accuracy(collector.accuracy),
        "\n## Performance\n",
        _render_performance(collector.performance),
        "\n## Memory\n",
        _render_memory(collector.memory),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pytest hooks and fixtures
# ---------------------------------------------------------------------------

_collector = MetricCollector()


def pytest_addoption(parser):
    parser.addoption(
        "--dsa-report",
        action="store",
        default=None,
        help="Enable DSA test report generation. Pass a path (relative to test dir) "
             "to write the markdown report. Omit to skip report generation (CI default).",
    )
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="Run performance benchmark tests (skipped by default).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: mark test as performance benchmark (skipped unless --run-perf)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-perf"):
        return
    skip_perf = pytest.mark.skip(reason="Performance test skipped. Use --run-perf to enable.")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)


@pytest.fixture
def dsa_metrics(request):
    """Fixture that provides a MetricRecorder for the current test."""
    cls_name = request.node.cls.__name__ if request.node.cls else ""
    test_name = request.node.name
    return MetricRecorder(cls_name, test_name, _collector)


def pytest_sessionfinish(session, exitstatus):
    """Write collected metrics to markdown — only when --dsa-report is specified."""
    report_path_str = session.config.getoption("--dsa-report", default=None)
    if report_path_str is None:
        return
    if not _collector.has_data():
        return
    test_dir = Path(__file__).parent
    report_path = test_dir / report_path_str
    report_path.write_text(_generate_report(_collector), encoding="utf-8")
    print(f"\n[dsa_metrics] Report written to: {report_path}")
