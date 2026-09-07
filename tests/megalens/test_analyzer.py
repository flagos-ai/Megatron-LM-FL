# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from megatron.megalens import analyzer as analyzer_module

_SOURCE_BASELINE = "12fb7169ce30fdb62b50f86b41afa09336a523ea"
_SOURCE_SHA256 = "73ff14a902ba5cff507187dd3693f4c2479dac1b11194d5a4bbf9ca885777ef7"
_SOURCE_LINE_COUNT = 350
_SOURCE_BYTE_COUNT = 11_905
_TARGET_SHA256 = "1e9e36cd4a07f28e5c287a05e838d15ed560670e7ec2f13eb0579aff8f067668"
_TARGET_LINE_COUNT = 364
_TARGET_BYTE_COUNT = 12_561


def _install_fake_analyzers(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> tuple[dict[str, tuple[list[dict[str, Any]], dict[str, Any]]], dict[str, object]]:
    captured: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    sentinels = {name: object() for name in ("pp", "dp", "tp", "ep")}
    analyzer_specs = (
        ("pp", "megatron.megalens.pp_analyzer", "analyze_pp_traces", sentinels["pp"]),
        ("dp", "megatron.megalens.dp_analyzer", "analyze_dp_traces", sentinels["dp"]),
        ("tp", "megatron.megalens.tp_analyzer", "analyze_tp_traces", sentinels["tp"]),
        ("ep", "megatron.megalens.ep_analyzer", "analyze_ep_traces", sentinels["ep"]),
        ("hybrid", "megatron.megalens.hybrid_analyzer", "analyze_hybrid_traces", None),
    )
    for label, module_name, symbol, result in analyzer_specs:
        module = ModuleType(module_name)

        def fake_analyzer(
            traces: list[dict[str, Any]],
            *,
            _label: str = label,
            _result: object | None = result,
            **kwargs: Any,
        ) -> object | None:
            calls.append(_label)
            captured[_label] = (traces, kwargs)
            return _result

        setattr(module, symbol, fake_analyzer)
        monkeypatch.setitem(sys.modules, module_name, module)

    return captured, sentinels


def _write_raw_benchmark(directory: Path) -> None:
    directory.mkdir()
    rank = {"g_rk": 0, "dp_rk": 0, "pp_rk": 0, "tp_rk": 0}
    rows = [
        {"name": "iteration", "ph": "B", "pad_before": 0, "iteration": 7},
        {"name": "forward", "ph": "B", "rel_ts": 1_000, **rank},
        {"name": "forward", "ph": "E", "rel_ts": 5_000, **rank},
        {"name": "iteration", "ph": "E", "duration_wall": 10_000, "iteration": 7},
    ]
    path = directory / "benchmark-data-0-pipeline-0-tensor-0.json"
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_locked_mixedpara_source_and_no_pig_target_adaptation() -> None:
    payload = Path(analyzer_module.__file__).resolve().read_bytes()

    assert _SOURCE_BASELINE == "12fb7169ce30fdb62b50f86b41afa09336a523ea"
    assert _SOURCE_SHA256 == "73ff14a902ba5cff507187dd3693f4c2479dac1b11194d5a4bbf9ca885777ef7"
    assert _SOURCE_BYTE_COUNT == 11_905
    assert _SOURCE_LINE_COUNT == 350
    assert hashlib.sha256(payload).hexdigest() == _TARGET_SHA256
    assert len(payload) == _TARGET_BYTE_COUNT
    assert len(payload.splitlines()) == _TARGET_LINE_COUNT
    assert b"progressive_decoupler" not in payload

    signatures = {
        "_expand_run_modes": ("modes",),
        "_infer_parallel_sizes": ("traces",),
        "aggregate_traces_from_benchmark_dir": ("bench_dir",),
        "load_traces_from_json": ("path",),
        "write_trace_json": ("traces", "path"),
        "run_parallelism_analyses": (
            "traces",
            "modes",
            "output_dir",
            "pp_theory_bw_gbps",
            "tp_nvlink_peak_gbps",
        ),
        "build_arg_parser": (),
        "main": ("argv",),
    }
    for name, parameters in signatures.items():
        assert tuple(inspect.signature(getattr(analyzer_module, name)).parameters) == parameters

    run_signature = inspect.signature(analyzer_module.run_parallelism_analyses)
    assert run_signature.parameters["pp_theory_bw_gbps"].default is None
    assert run_signature.parameters["tp_nvlink_peak_gbps"].default is None
    assert inspect.signature(analyzer_module.main).parameters["argv"].default is None


@pytest.mark.parametrize(
    ("modes", "expected"),
    [
        ([], ["pp", "dp", "tp", "ep", "hybrid"]),
        (["tp", "all", "pp"], ["pp", "dp", "tp", "ep", "hybrid"]),
        (["hybrid", "pp", "hybrid"], ["hybrid", "pp"]),
    ],
)
def test_expand_run_modes_keeps_source_contract(modes: list[str], expected: list[str]) -> None:
    assert analyzer_module._expand_run_modes(modes) == expected


def test_run_parallelism_analyses_uses_fixed_source_order_and_cached_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    captured, sentinels = _install_fake_analyzers(monkeypatch, calls)
    sizes = {"dp": 2, "pp": 2, "tp": 2, "ep": 2}
    monkeypatch.setattr(analyzer_module, "_infer_parallel_sizes", lambda traces: sizes)
    traces = [{"name": "iteration"}]

    result = analyzer_module.run_parallelism_analyses(
        traces,
        ["hybrid", "ep", "tp", "dp", "pp"],
        tmp_path,
        pp_theory_bw_gbps=12.5,
        tp_nvlink_peak_gbps=250.0,
    )

    assert result is None
    assert calls == ["pp", "dp", "tp", "ep", "hybrid"]
    assert captured["pp"] == (traces, {"theory_bw_gbps": 12.5, "output_dir": str(tmp_path / "pp")})
    assert captured["dp"] == (traces, {"output_dir": str(tmp_path / "dp")})
    assert captured["tp"] == (
        traces,
        {"nvlink_theory_peak_gbps": 250.0, "output_dir": str(tmp_path / "tp")},
    )
    assert captured["ep"] == (traces, {"output_dir": str(tmp_path / "ep")})
    assert captured["hybrid"] == (
        traces,
        {
            "output_dir": str(tmp_path / "hybrid"),
            "pp_result": sentinels["pp"],
            "dp_result": sentinels["dp"],
            "tp_result": sentinels["tp"],
            "ep_result": sentinels["ep"],
        },
    )


@pytest.mark.parametrize(
    ("sizes", "expected_calls"),
    [
        ({"dp": 1, "pp": 1, "tp": 1, "ep": 1}, []),
        ({"dp": 1, "pp": 2, "tp": 1, "ep": 1}, ["pp", "hybrid"]),
        ({"dp": 2, "pp": 1, "tp": 1, "ep": 1}, ["dp", "hybrid"]),
        ({"dp": 1, "pp": 1, "tp": 2, "ep": 1}, ["tp", "hybrid"]),
        ({"dp": 1, "pp": 1, "tp": 1, "ep": 2}, ["ep", "hybrid"]),
    ],
)
def test_size_gates_each_dimension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sizes: dict[str, int],
    expected_calls: list[str],
) -> None:
    calls: list[str] = []
    _install_fake_analyzers(monkeypatch, calls)
    monkeypatch.setattr(analyzer_module, "_infer_parallel_sizes", lambda traces: sizes)

    analyzer_module.run_parallelism_analyses([], ["all"], tmp_path)

    assert calls == expected_calls


def test_direct_hybrid_does_not_precompute_dimension_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    captured, _ = _install_fake_analyzers(monkeypatch, calls)
    sizes = {"dp": 1, "pp": 2, "tp": 1, "ep": 1}
    monkeypatch.setattr(analyzer_module, "_infer_parallel_sizes", lambda traces: sizes)

    analyzer_module.run_parallelism_analyses([], ["hybrid"], tmp_path)

    assert calls == ["hybrid"]
    _, kwargs = captured["hybrid"]
    assert kwargs == {
        "output_dir": str(tmp_path / "hybrid"),
        "pp_result": None,
        "dp_result": None,
        "tp_result": None,
        "ep_result": None,
    }


def test_parser_requires_exactly_one_trace_source(tmp_path: Path) -> None:
    parser = analyzer_module.build_arg_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            ["--bench-dir", str(tmp_path / "bench"), "--trace", str(tmp_path / "trace.json")]
        )

    assert parser.parse_args(["--bench-dir", str(tmp_path / "bench")]).bench_dir == (
        tmp_path / "bench"
    )
    assert parser.parse_args(["--trace", str(tmp_path / "trace.json")]).trace == (
        tmp_path / "trace.json"
    )
    assert parser.parse_args(
        ["--trace", str(tmp_path / "trace.json"), "--align-framework-timeline"]
    ).align_framework_timeline
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["--trace", str(tmp_path / "trace.json"), "--run", "pig"])


def test_aggregate_only_rejects_trace_source_before_reading(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        analyzer_module.main(["--trace", str(tmp_path / "missing.json"), "--aggregate-only"])


class _FixedDatetime:
    @classmethod
    def now(cls) -> _FixedDatetime:
        return cls()

    def strftime(self, pattern: str) -> str:
        assert pattern == "%Y%m%d_%H%M%S"
        return "20260728_010203"


@pytest.mark.parametrize("output_mode", ["explicit", "output-dir", "cwd"])
def test_bench_input_trace_output_precedence_and_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_mode: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analyzer_module, "datetime", _FixedDatetime)
    bench_dir = tmp_path / "bench"
    _write_raw_benchmark(bench_dir)
    args = ["--bench-dir", str(bench_dir), "--aggregate-only"]

    if output_mode == "explicit":
        expected = tmp_path / "explicit.json"
        args.extend(
            ["--trace-output", str(expected), "--output-dir", str(tmp_path / "unused-output")]
        )
    elif output_mode == "output-dir":
        expected = tmp_path / "analysis" / "aggregated_trace.json"
        args.extend(["--output-dir", str(tmp_path / "analysis")])
    else:
        expected = tmp_path / "megalens_trace_20260728_010203.json"

    assert analyzer_module.main(args) == 0
    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert any(event["name"] == "forward" and event["ph"] == "X" for event in payload)
    if output_mode == "explicit":
        assert not (tmp_path / "unused-output" / "aggregated_trace.json").exists()


@pytest.mark.parametrize(
    ("run_args", "expected_modes"),
    [(["--run", "pp", "tp"], ["pp", "tp"]), ([], ["pp", "dp", "tp", "ep", "hybrid"])],
)
def test_trace_input_ignores_trace_output_and_runs_selected_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_args: list[str], expected_modes: list[str]
) -> None:
    traces = [{"name": "iteration", "ph": "X", "ts": 0, "dur": 1, "args": {}}]
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(traces), encoding="utf-8")
    ignored = tmp_path / "ignored.json"
    captured: dict[str, Any] = {}

    def fake_run(
        actual_traces: list[dict[str, Any]], modes: list[str], output_dir: Path, **kwargs: Any
    ) -> None:
        captured.update(traces=actual_traces, modes=modes, output_dir=output_dir, kwargs=kwargs)

    monkeypatch.setattr(analyzer_module, "run_parallelism_analyses", fake_run)
    args = [
        "--trace",
        str(trace_path),
        "--trace-output",
        str(ignored),
        "--output-dir",
        str(tmp_path / "reports"),
        "--pp-theory-bw",
        "12.5",
        "--tp-nvlink-peak",
        "250",
        *run_args,
    ]

    assert analyzer_module.main(args) == 0
    assert captured == {
        "traces": traces,
        "modes": expected_modes,
        "output_dir": tmp_path / "reports",
        "kwargs": {"pp_theory_bw_gbps": 12.5, "tp_nvlink_peak_gbps": 250.0},
    }
    assert not ignored.exists()


def test_trace_input_can_explicitly_align_and_write_framework_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    traces: list[dict[str, Any]] = []
    for rank, offset in ((0, 0), (1, 50)):
        common_args = {"iteration": 2, "g_rk": rank, "dp_rk": 0, "pp_rk": 0, "tp_rk": rank}
        traces.append(
            {
                "name": "iteration",
                "ph": "X",
                "ts": 0,
                "dur": 1_000,
                "pid": rank,
                "args": dict(common_args),
            }
        )
        for completion in (200, 400, 600):
            traces.append(
                {
                    "name": "tp-allreduce",
                    "ph": "X",
                    "ts": completion + offset - 20,
                    "dur": 20,
                    "pid": rank,
                    "args": {
                        **common_args,
                        "group": [1 - rank],
                        "group_size": 2,
                        "op": "all_reduce",
                        "timing_phase": "collective_call",
                        "data_bytes": 4096,
                        "payload_role": "inplace_input_output",
                    },
                }
            )

    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(traces), encoding="utf-8")
    aligned_path = tmp_path / "aligned.json"
    captured: dict[str, Any] = {}

    def fake_run(
        actual_traces: list[dict[str, Any]], modes: list[str], output_dir: Path, **kwargs: Any
    ) -> None:
        captured["traces"] = actual_traces

    monkeypatch.setattr(analyzer_module, "run_parallelism_analyses", fake_run)

    assert (
        analyzer_module.main(
            [
                "--trace",
                str(trace_path),
                "--trace-output",
                str(aligned_path),
                "--align-framework-timeline",
                "--run",
                "tp",
                "--output-dir",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )

    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    assert captured["traces"] == aligned
    anchors = [event for event in aligned if event["name"] == "tp-allreduce"]
    assert {event["ts"] + event["dur"] for event in anchors[:3]} == {200, 400, 600}
    assert {event["ts"] + event["dur"] for event in anchors[3:]} == {200, 400, 600}
    assert [event["dur"] for event in anchors] == [20] * 6
