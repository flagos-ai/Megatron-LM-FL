# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from megatron.megalens import pp_analyzer as pp_module
from megatron.megalens.pp_analyzer import (
    ReportLogger,
    TracedWaitHandle,
    TraceGraph,
    analyze_pp_traces,
    bubble_analysis,
    compute_load_analysis,
    get_pure_compute_dur,
    hardware_jitter_analysis,
    p2p_comm_analysis,
    wrap_p2p_communicate_reqs,
)
_SOURCE_BASELINE = "12fb7169ce30fdb62b50f86b41afa09336a523ea"
_SOURCE_SHA256 = "38c1bb91c0677505169bbe742776de4dc07db2462b5f80479d47da1cde5c6bc7"
_SOURCE_LINE_COUNT = 1711
_ADAPTED_SHA256 = "9af4503bab41589117e4aa8556b3c011df65fd80da7c5b093435a7586d00f3f3"
_ADAPTED_LINE_COUNT = 1774


def _span(
    name: str,
    ts: int,
    dur: int,
    *,
    rank: int = 0,
    iteration: int = 7,
    phase: str = "X",
    **args: Any,
) -> dict[str, Any]:
    event_args: dict[str, Any] = {"iteration": iteration, "dp_rk": 0, "tp_rk": 0, "pp_rk": rank}
    event_args.update(args)
    return {"name": name, "ph": phase, "ts": ts, "dur": dur, "pid": rank, "args": event_args}


class _MemoryLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str, end: str = "\n") -> None:
        self.messages.append(message + end)


def test_locked_mixedpara_source_identity_and_adapter_snapshot_are_recorded() -> None:
    module_path = Path(pp_module.__file__).resolve()
    payload = module_path.read_bytes()

    assert _SOURCE_BASELINE == "12fb7169ce30fdb62b50f86b41afa09336a523ea"
    assert _SOURCE_SHA256 == "38c1bb91c0677505169bbe742776de4dc07db2462b5f80479d47da1cde5c6bc7"
    assert _SOURCE_LINE_COUNT == 1711
    assert hashlib.sha256(payload).hexdigest() == _ADAPTED_SHA256
    assert len(payload.splitlines()) == _ADAPTED_LINE_COUNT
    assert _ADAPTED_SHA256 != _SOURCE_SHA256
    assert b"megatron.megalens.pig" not in payload


def test_base_import_defers_optional_reporting_and_scipy_imports() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(repository_root)!r})
attempts = []

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in {{"matplotlib", "pandas", "scipy"}}:
            attempts.append(root)
            raise ModuleNotFoundError(f"blocked optional dependency {{fullname}}")
        return None

sys.meta_path.insert(0, BlockOptionalImports())
from megatron.megalens import pp_analyzer

assert attempts == []

try:
    pp_analyzer._load_pandas()
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("pandas loader did not attempt the optional import")
assert attempts == ["pandas"]

attempts.clear()
try:
    pp_analyzer._load_reporting_dependencies()
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("plot loader did not attempt the optional import")
assert attempts == ["matplotlib"]

attempts.clear()
rho, pvalue = pp_analyzer._spearmanr_compat([1, 2, 3], [1, 2, 3])
assert attempts == ["scipy"]
assert rho == 1.0
assert pvalue == 0.0
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr


def test_p2p_bandwidth_plot_uses_supported_matplotlib_tick_labels(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    pp_module.generate_p2p_bw_plots(
        [
            {"ts_s": 1.0, "rank": 0, "eff_bw_gbps": 100.0},
            {"ts_s": 2.0, "rank": 0, "eff_bw_gbps": 120.0},
        ],
        theory_bw_gbps=450.0,
        output_dir=str(tmp_path),
    )

    output = tmp_path / "p2p_actual_bandwidth_analysis.pdf"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_public_api_signatures_defaults_and_thresholds_match_source() -> None:
    wait_handle = inspect.signature(TracedWaitHandle)
    assert tuple(wait_handle.parameters) == ("req", "name", "peer_rank", "tracer", "tensor_bytes")
    assert tuple(inspect.signature(TracedWaitHandle.wait).parameters) == ("self",)
    assert tuple(inspect.signature(wrap_p2p_communicate_reqs).parameters) == (
        "reqs",
        "communicator",
        "tensors_dict",
    )
    assert tuple(inspect.signature(TraceGraph).parameters) == ("traces",)

    master = inspect.signature(analyze_pp_traces)
    assert tuple(master.parameters) == ("traces", "theory_bw_gbps", "output_dir")
    assert master.parameters["theory_bw_gbps"].default is None
    assert master.parameters["output_dir"].default == "."

    assert pp_module.P2P_THEORY_BW_GBPS == 450.0
    assert pp_module.COMPUTE_JITTER_CV_THRESHOLD == 0.1
    assert pp_module.COMPUTE_SKEW_MAX_MIN_RATIO == 1.2
    assert pp_module.COMPUTE_SKEW_MAX_MEAN_RATIO == 1.2
    assert pp_module.ACTIVE_EVENTS_WHITELIST == {
        "forward-step",
        "backward-step",
        "grad-sync",
        "optimizer",
    }


def test_traced_wait_handle_preserves_source_wait_scope_and_return_behavior() -> None:
    class Request:
        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self) -> str:
            self.wait_count += 1
            return "backend-result"

    class Tracer:
        def __init__(self) -> None:
            self.scopes: list[tuple[str, dict[str, Any]]] = []

        def is_tracing(self) -> bool:
            return True

        @contextmanager
        def scope(self, name: str, ctx: dict[str, Any]):
            self.scopes.append((name, ctx))
            yield

    request = Request()
    tracer = Tracer()
    handle = TracedWaitHandle(request, "recv-forward", 3, tracer, 4096)

    assert handle.wait() is None
    assert request.wait_count == 1
    assert tracer.scopes == [
        (
            "recv-forward",
            {"peer_rank": 3, "comm_type": "p2p", "is_blocking": True, "data_bytes": 4096},
        )
    ]


def test_legacy_wrapper_symbols_remain_source_compatible_without_core_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.training import global_vars

    tracer = SimpleNamespace(is_tracing=lambda: True)
    monkeypatch.setattr(global_vars, "get_args", lambda: SimpleNamespace(trace=True))
    monkeypatch.setattr(global_vars, "get_tracer", lambda: tracer)
    monkeypatch.setattr(pp_module, "get_tensor_bytes", lambda tensor: len(tensor or ""))

    requests = {
        "send_next": object(),
        "recv_prev": object(),
        "send_prev": object(),
        "recv_next": object(),
        "custom": object(),
    }
    communicator = SimpleNamespace(next_rank=7, prev_rank=5)
    wrapped = wrap_p2p_communicate_reqs(
        requests,
        communicator,
        {"send_next": "abcd", "recv_prev": "xy", "send_prev": "123", "recv_next": "z"},
    )

    assert wrapped is not requests
    assert isinstance(wrapped["send_next"], TracedWaitHandle)
    assert wrapped["send_next"].name == "send-forward"
    assert wrapped["send_next"].peer_rank == 7
    assert wrapped["send_next"].tensor_bytes == 4
    assert isinstance(wrapped["recv_prev"], TracedWaitHandle)
    assert wrapped["recv_prev"].name == "recv-forward"
    assert wrapped["recv_prev"].peer_rank == 5
    assert wrapped["recv_prev"].tensor_bytes == 2
    assert isinstance(wrapped["send_prev"], TracedWaitHandle)
    assert wrapped["send_prev"].name == "send-backward"
    assert wrapped["send_prev"].peer_rank == 5
    assert wrapped["send_prev"].tensor_bytes == 3
    assert isinstance(wrapped["recv_next"], TracedWaitHandle)
    assert wrapped["recv_next"].name == "recv-backward"
    assert wrapped["recv_next"].peer_rank == 7
    assert wrapped["recv_next"].tensor_bytes == 1
    assert wrapped["custom"] is requests["custom"]

    monkeypatch.setattr(global_vars, "get_args", lambda: SimpleNamespace(trace=False))
    assert wrap_p2p_communicate_reqs(requests, communicator, {}) is requests

    repository_root = Path(__file__).resolve().parents[2]
    core_and_training = [repository_root / "megatron/core", repository_root / "megatron/training"]
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in core_and_training
        for path in root.rglob("*.py")
    )
    assert "wrap_p2p_communicate_reqs" not in source_text
    assert "TracedWaitHandle" not in source_text
    assert "megatron.megalens.pp_analyzer" not in source_text


def test_trace_graph_keeps_source_x_filter_fifo_pairing_and_no_backtracking() -> None:
    early_recv = _span("recv-forward", 10, 2, rank=1, peer_rank=0, data_bytes=16)
    first_send = _span("send-forward", 20, 2, rank=0, peer_rank=1, data_bytes=16)
    second_send = _span("send-forward", 25, 2, rank=0, peer_rank=1, data_bytes=16)
    first_recv = _span("recv-forward", 30, 2, rank=1, peer_rank=0, data_bytes=16)
    second_recv = _span("recv-forward", 35, 2, rank=1, peer_rank=0, data_bytes=16)
    graph = TraceGraph(
        [
            _span("iteration", 0, 100, rank=0, dp_rk=2, tp_rk=3, pp_rk=4),
            _span("p2p-launch", 15, 3, rank=0),
            second_recv,
            second_send,
            early_recv,
            first_recv,
            first_send,
            _span("send-forward", 40, 2, rank=0, phase="C", peer_rank=1),
        ]
    )

    def event_for(raw: dict[str, Any]):
        return next(event for event in graph.comm_events if event.raw is raw)

    assert len(graph.comm_events) == 5
    assert graph.rank_topology == {0: {"dp": 2, "tp": 3, "pp": 4}}
    assert event_for(early_recv).paired_event is None
    assert event_for(first_send).paired_event is event_for(first_recv)
    assert event_for(second_send).paired_event is event_for(second_recv)
    assert event_for(first_recv).paired_event is event_for(first_send)
    assert event_for(second_recv).paired_event is event_for(second_send)
    assert event_for(first_send).launch_event is graph.p2p_launch_events[0]


def test_pure_compute_duration_subtracts_union_of_tp_intervals() -> None:
    forward = _span("forward-step", 0, 200)
    graph = TraceGraph([forward, _span("tp-allreduce", 20, 80), _span("reduce-scatter", 80, 70)])

    assert get_pure_compute_dur(graph.compute_events[0], graph) == 70.0


def test_bubble_analysis_merges_active_intervals_truncates_at_optimizer_and_mutates_trace() -> None:
    iteration = _span("iteration", 0, 1_000)
    graph = TraceGraph(
        [
            iteration,
            _span("forward-step", 0, 300),
            _span("backward-step", 200, 200),
            _span("optimizer", 800, 100),
        ]
    )

    rows = bubble_analysis(graph, _MemoryLogger())  # type: ignore[arg-type]

    assert rows == [
        {
            "Rank": 0,
            "DP_Rank": 0,
            "TP_Rank": 0,
            "PP_Rank": 0,
            "Group": "DP=0 | TP=0",
            "Iteration": 7,
            "T_actual_ms": 0.8,
            "T_active_ms": 0.4,
            "Bubble_Rate": 0.5,
        }
    ]
    assert iteration["args"]["Analysis_Bubble_Rate"] == "50.00%"
    assert iteration["args"]["Analysis_Active_Time_ms"] == "0.40"
    assert iteration["cname"] == "bad"


def test_p2p_analysis_uses_fifo_pair_and_preserves_source_trace_mutation() -> None:
    sender = _span("send-forward", 100, 20, rank=0, peer_rank=1, data_bytes=1_000_000)
    receiver = _span("recv-forward", 200, 400, rank=1, peer_rank=0, data_bytes=1_000_000)
    graph = TraceGraph(
        [
            _span("iteration", 0, 1_000, rank=0),
            _span("iteration", 0, 1_000, rank=1),
            _span("p2p-launch", 80, 10, rank=0),
            sender,
            receiver,
        ]
    )

    rows = p2p_comm_analysis(graph, 100.0, _MemoryLogger())  # type: ignore[arg-type]

    assert len(rows) == 1
    assert rows[0]["rank"] == 1
    assert rows[0]["sender_rank"] == 0
    assert rows[0]["direction"] == "forward"
    assert rows[0]["stall_us"] == 0.0
    assert rows[0]["transfer_ub_us"] == 400.0
    assert rows[0]["eff_bw_gbps"] == pytest.approx(2.5)
    assert receiver["args"]["Has_Cross_Rank_Pair"] is True
    assert receiver["args"]["Wait_Total_us"] == 400
    assert receiver["args"]["Transfer_Upper_Bound_us"] == 400
    assert receiver["args"]["Analysis_Diagnosis"] == "Normal Transfer"


def test_source_fifo_pairing_keeps_dependency_stall_and_straggler_path_unreachable() -> None:
    graph = TraceGraph(
        [
            _span("iteration", 0, 2_000, rank=0),
            _span("iteration", 0, 2_000, rank=1),
            _span("p2p-launch", 90, 60, rank=0),
            _span("send-forward", 100, 10, rank=0, peer_rank=1, data_bytes=2_000_000),
            _span("recv-forward", 100, 1_000, rank=1, peer_rank=0, data_bytes=2_000_000),
        ]
    )

    rows = p2p_comm_analysis(graph, 100.0, _MemoryLogger())  # type: ignore[arg-type]

    assert rows[0]["stall_us"] == 50.0
    assert pp_module.PP_P2P_STALL_THRESHOLD_US == 100.0
    assert pp_module._extract_pp_straggler_data([], rows, graph) == []

    recv_before_send = TraceGraph(
        [
            _span("recv-forward", 99, 1_000, rank=1, peer_rank=0, data_bytes=2_000_000),
            _span("send-forward", 100, 10, rank=0, peer_rank=1, data_bytes=2_000_000),
        ]
    )
    unpaired = p2p_comm_analysis(recv_before_send, 100.0, _MemoryLogger())  # type: ignore[arg-type]
    assert unpaired[0]["sender_rank"] == -1
    assert unpaired[0]["stall_us"] == 0.0


def test_compute_and_jitter_analysis_keep_source_workload_normalization_and_mutation() -> None:
    events = [
        _span("iteration", 0, 10_000),
        *[
            _span(
                "forward-step",
                index * 2_500,
                duration,
                current_microbatch=index,
                num_tokens=128,
                sum_sq_seq_len=16_384,
            )
            for index, duration in enumerate((1_000, 1_000, 1_000, 2_000))
        ],
    ]
    graph = TraceGraph(events)

    compute_stats, raw_dfs = compute_load_analysis(graph, _MemoryLogger())  # type: ignore[arg-type]
    jitter_data = hardware_jitter_analysis(graph, _MemoryLogger())  # type: ignore[arg-type]

    assert len(compute_stats["forward"]) == 1
    assert compute_stats["forward"][0]["Mean_Dur_ms"] == 1.25
    assert compute_stats["forward"][0]["Diagnosis"] == "Hardware Jitter (Static)"
    assert list(raw_dfs["forward"]) == [0]
    assert events[-1]["cname"] == "terrible"
    assert events[-1]["args"]["Analysis_Rank_Mean_ms"] == "1.25"

    assert len(jitter_data) == 4
    assert sum(row["is_spike"] for row in jitter_data) == 1
    assert jitter_data[-1]["slowdown_ratio"] == 2.0
    assert "event_ref" not in jitter_data[-1]
    assert events[-1]["args"]["Analysis_Slowdown_Ratio"] == "2.00x"
    assert events[-1]["args"]["Analysis_Diagnosis"] == "True Hardware Jitter Spike"


def test_master_orchestrator_preserves_source_order_result_shape_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def record(name: str, result: Any = None):
        def call(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return result

        return call

    p2p_rows = [{"rank": 0, "iteration": 7}]
    bubble_rows = [{"Rank": 0, "Iteration": 7}]
    compute_rows = {"forward": [{"Rank": 0}], "backward": [], "optimizer": []}
    jitter_rows = [{"rank": 0, "iteration": 7}]
    straggler_rows = [{"rank": 0, "iteration": 7}]
    empty_dfs = {"forward": {}, "backward": {}, "optimizer": {}}

    from megatron.megalens import paper_style

    monkeypatch.setattr(paper_style, "apply_global_rcparams", record("style"))
    monkeypatch.setattr(pp_module, "p2p_comm_analysis", record("p2p", p2p_rows))
    monkeypatch.setattr(pp_module, "bubble_analysis", record("bubble", bubble_rows))
    monkeypatch.setattr(
        pp_module, "compute_load_analysis", record("compute", (compute_rows, empty_dfs))
    )
    monkeypatch.setattr(pp_module, "hardware_jitter_analysis", record("jitter", jitter_rows))
    monkeypatch.setattr(pp_module, "generate_p2p_bw_plots", record("plot-p2p-bw"))
    monkeypatch.setattr(pp_module, "generate_p2p_plots", record("plot-p2p"))
    monkeypatch.setattr(pp_module, "generate_compute_plots2", record("plot-compute"))
    monkeypatch.setattr(pp_module, "generate_bubble_plots2", record("plot-bubble"))
    monkeypatch.setattr(pp_module, "generate_jitter_plots", record("plot-jitter"))
    monkeypatch.setattr(
        pp_module, "_extract_pp_straggler_data", record("straggler", straggler_rows)
    )

    traces = [
        _span("iteration", 0, 1_000, rank=0, pp_rk=0),
        _span("iteration", 0, 1_000, rank=1, pp_rk=1),
        _span("forward-step", 100, 200, rank=0, pp_rk=0),
        _span("forward-step", 100, 200, rank=1, pp_rk=1),
    ]
    result = analyze_pp_traces(traces, theory_bw_gbps=123.0, output_dir=str(tmp_path))

    assert calls == [
        "style",
        "p2p",
        "bubble",
        "compute",
        "jitter",
        "plot-p2p-bw",
        "plot-p2p",
        "plot-compute",
        "plot-bubble",
        "plot-jitter",
        "straggler",
    ]
    assert result == {
        "p2p_stats": p2p_rows,
        "bubble_stats": bubble_rows,
        "compute_stats": compute_rows,
        "jitter_data": jitter_rows,
        "pp_config": {"n_dp": 1, "n_tp": 1, "n_pp": 2, "n_microbatches": 1},
        "straggler_data": straggler_rows,
    }
    report = (tmp_path / "pp_diagnostic_report.txt").read_text(encoding="utf-8")
    assert "MegaLens Pipeline Parallelism Diagnostic Report" in report
    assert "Using user-provided P2P BW" in report
    assert "'n_pp': 2" in report
    assert "Exporting CSV stats" in report
    assert {path.name for path in tmp_path.iterdir()} == {
        "pp_diagnostic_report.txt",
        "pp_p2p_stats.csv",
        "pp_bubble_stats.csv",
        "pp_compute_stats.csv",
        "pp_jitter_stats.csv",
        "pp_straggler_stats.csv",
    }
