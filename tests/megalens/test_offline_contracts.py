from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from megatron.megalens.data_loader import CounterSample, TraceDataLoader
from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    Rank,
    aggregate_benchmark_data,
    align_framework_trace_timeline,
    benchmark_to_chrome_trace,
    read_benchmark_file,
    transform,
)
from megatron.megalens.utils import (
    get_tensor_bytes,
    infer_parallel_sizes_from_loader,
    infer_parallel_sizes_from_traces,
)


def test_loader_and_raw_events_infer_the_same_parallel_sizes() -> None:
    traces = [
        {"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "DP0-PP0-TP0"}},
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "DP1-PP1-TP1"}},
        {
            "ph": "X",
            "name": "forward-step",
            "pid": 0,
            "ts": 1,
            "dur": 3,
            "args": {"dp_rk": 0, "pp_rk": 0, "tp_rk": 0},
        },
        {
            "ph": "X",
            "name": "moe-router",
            "pid": 1,
            "ts": 2,
            "dur": 1,
            "args": {"dp_rk": 1, "pp_rk": 1, "tp_rk": 1, "ep_size": 4},
        },
    ]
    loader = TraceDataLoader.from_traces(traces)

    assert loader.get_ranks() == [0, 1]
    assert len(loader.get_events_by_name("forward-step")) == 1
    assert len(loader.span_events) == 2
    assert infer_parallel_sizes_from_loader(loader) == {"dp": 2, "pp": 2, "tp": 2, "ep": 4}
    assert infer_parallel_sizes_from_traces(traces) == {"dp": 2, "pp": 2, "tp": 2, "ep": 4}


def test_tensor_bytes_accepts_missing_and_nested_non_tensor_payloads() -> None:
    assert get_tensor_bytes(None) == 0
    assert get_tensor_bytes([None, (None, [None])]) == 0


def test_report_style_applies_without_leaking_global_settings() -> None:
    mpl = pytest.importorskip("matplotlib")
    from megatron.megalens import paper_style

    with mpl.rc_context():
        paper_style.apply_global_rcparams()


def _event(rank: Rank, iteration_id: int, suffix: str) -> Event:
    return Event(
        rel_ts=1_000,
        rank=rank,
        name=f"forward-rank-{rank.data}-iter-{iteration_id}-{suffix}",
        ph="B",
        attrs={
            "g_rk": rank.global_rank if rank.global_rank is not None else rank.data,
            "dp_rk": rank.data,
            "pp_rk": rank.pipeline,
            "tp_rk": rank.tensor,
        },
    )


def _iteration(
    rank: Rank, iteration_id: int | None, *, pad_before: int = 0, duration: int = 10_000
) -> Iteration:
    event_id = iteration_id if iteration_id is not None else -1
    return Iteration(
        pad_before=pad_before,
        events=[_event(rank, event_id, "begin")],
        duration=duration,
        iteration_id=iteration_id,
        ranks=(rank,),
    )


def _raw_iteration(rank: Rank, iteration_id: int) -> list[dict[str, object]]:
    attrs = {
        "g_rk": rank.global_rank if rank.global_rank is not None else rank.data,
        "dp_rk": rank.data,
        "pp_rk": rank.pipeline,
        "tp_rk": rank.tensor,
    }
    return [
        {"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration_id},
        {"name": "forward", "ph": "B", "rel_ts": 1_000, **attrs},
        {"name": "forward", "ph": "E", "rel_ts": 5_000, **attrs},
        {"name": "iteration", "ph": "E", "duration_wall": 10_000, "iteration": iteration_id},
    ]


def _write_rank_trace(directory: Path, rank: Rank, iteration_ids: list[int]) -> None:
    rows = [row for iteration_id in iteration_ids for row in _raw_iteration(rank, iteration_id)]
    global_prefix = f"global-{rank.global_rank}-" if rank.global_rank is not None else ""
    path = directory / (
        f"benchmark-{global_prefix}data-{rank.data}-pipeline-{rank.pipeline}-"
        f"tensor-{rank.tensor}.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_aggregate_joins_ranks_by_iteration_id_and_preserves_id() -> None:
    rank0 = Rank(data=0, pipeline=0, tensor=0)
    rank1 = Rank(data=1, pipeline=0, tensor=0)
    contents = [
        [_iteration(rank0, 138), _iteration(rank0, 137, pad_before=100, duration=1_000)],
        [_iteration(rank1, 138), _iteration(rank1, 137, pad_before=300, duration=900)],
    ]

    iterations, dp, pp, tp = aggregate_benchmark_data(contents)

    assert [iteration.iteration_id for iteration in iterations] == [137, 138]
    assert (dp, pp, tp) == (2, 1, 1)
    assert {event.name for event in iterations[0].events} == {
        "forward-rank-0-iter-137-begin",
        "forward-rank-1-iter-137-begin",
    }
    rank1_event = next(event for event in iterations[0].events if event.rank == rank1)
    assert rank1_event.rel_ts == 1_200
    assert iterations[0].duration == 1_100


def test_aggregate_preserves_per_rank_scope_order_when_cuda_timestamps_cross() -> None:
    rank0 = Rank(data=0, pipeline=0, tensor=0, global_rank=0)
    rank1 = Rank(data=1, pipeline=0, tensor=0, global_rank=1)
    attrs0 = {"g_rk": 0, "dp_rk": 0, "pp_rk": 0, "tp_rk": 0}
    attrs1 = {"g_rk": 1, "dp_rk": 1, "pp_rk": 0, "tp_rk": 0}
    rank0_iteration = Iteration(
        pad_before=0,
        events=[
            Event(100_000, rank0, "optimizer", "B", attrs0),
            Event(301_888, rank0, "optimizer", "E", attrs0),
            Event(301_156, rank0, "optimizer-postprocess", "B", attrs0),
            Event(401_156, rank0, "optimizer-postprocess", "E", attrs0),
        ],
        duration=500_000,
        iteration_id=1,
        ranks=(rank0,),
    )
    rank1_iteration = Iteration(
        pad_before=0,
        events=[
            Event(200_000, rank1, "forward", "B", attrs1),
            Event(250_000, rank1, "forward", "E", attrs1),
        ],
        duration=500_000,
        iteration_id=1,
        ranks=(rank1,),
    )

    iterations, _, _, _ = aggregate_benchmark_data(
        [[rank0_iteration], [rank1_iteration]]
    )
    assert [
        (event.rank.global_rank, event.name, event.ph)
        for event in iterations[0].events
    ] == [
        (0, "optimizer", "B"),
        (1, "forward", "B"),
        (1, "forward", "E"),
        (0, "optimizer", "E"),
        (0, "optimizer-postprocess", "B"),
        (0, "optimizer-postprocess", "E"),
    ]
    traces = benchmark_to_chrome_trace(iterations)

    spans = {trace["name"]: trace for trace in traces if trace.get("ph") == "X"}
    assert spans["optimizer"]["dur"] == 201
    assert spans["optimizer-postprocess"]["dur"] == 100
    assert spans["forward"]["dur"] == 50


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            [[_iteration(Rank(0, 0, 0), 137)], [_iteration(Rank(1, 0, 0), 138)]],
            "Mismatched iteration IDs",
        ),
        (
            [[_iteration(Rank(0, 0, 0), 137), _iteration(Rank(0, 0, 0), 137)]],
            "Duplicate iteration ID 137",
        ),
        (
            [[_iteration(Rank(0, 0, 0), 137)], [_iteration(Rank(1, 0, 0), None)]],
            "present for only part",
        ),
    ],
)
def test_aggregate_rejects_ambiguous_iteration_alignment(
    contents: list[list[Iteration]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_benchmark_data(contents)


def test_aggregate_keeps_legacy_position_alignment_when_all_ids_are_absent() -> None:
    with pytest.warns(RuntimeWarning, match="file position"):
        iterations, _, _, _ = aggregate_benchmark_data(
            [[_iteration(Rank(0, 0, 0), None)], [_iteration(Rank(1, 0, 0), None)]]
        )

    assert len(iterations) == 1
    assert iterations[0].iteration_id is None


def _framework_alignment_trace() -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    offsets = {0: 0, 1: 300, 2: 500, 3: 400}
    for rank, offset in offsets.items():
        common_args = {"iteration": 2, "g_rk": rank, "dp_rk": 0, "pp_rk": 0, "tp_rk": rank}
        traces.append(
            {
                "name": "iteration",
                "ph": "X",
                "ts": 0,
                "dur": 10_000,
                "pid": rank,
                "tid": 0,
                "args": dict(common_args),
            }
        )
        traces.append(
            {
                "name": "forward",
                "ph": "X",
                "ts": 500 + offset,
                "dur": 1_000,
                "pid": rank,
                "tid": 0,
                "args": dict(common_args),
            }
        )
        traces.append(
            {
                "name": "gpu-utilization",
                "ph": "C",
                "ts": 700 + offset,
                "pid": rank,
                "tid": "Hardware Monitor",
                "args": {**common_args, "utilization": 90},
            }
        )
        peers = [peer for peer in offsets if peer != rank]
        for index, completion in enumerate((2_000, 4_000, 6_000)):
            duration = 100 + rank * 10 + index
            completion_outlier = 30 if rank == 3 and index == 1 else 0
            traces.append(
                {
                    "name": "tp-allreduce",
                    "ph": "X",
                    "ts": completion + offset + completion_outlier - duration,
                    "dur": duration,
                    "pid": rank,
                    "tid": 0,
                    "args": {
                        **common_args,
                        "group": peers,
                        "group_size": 4,
                        "op": "all_reduce",
                        "timing_phase": "collective_call",
                        "data_bytes": 4096,
                        "reduce_op": "SUM",
                        "payload_role": "inplace_input_output",
                    },
                }
            )
    traces.append({"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "rank 0"}})
    traces.append(
        {
            "record_type": "cuda_kernel",
            "name": "ncclKernel_AllReduce",
            "ph": "X",
            "ts": 900,
            "dur": 50,
            "pid": 0,
            "args": {"iteration": 2, "g_rk": 0},
        }
    )
    traces.append(
        {
            "record_type": "cuda_kernel",
            "name": "rank1Kernel",
            "ph": "X",
            "ts": 900 + offsets[1],
            "dur": 50,
            "pid": 1,
            "iteration": 2,
            "g_rk": 1,
            "iter_rel_start_us": 900,
        }
    )
    return traces


def test_framework_timeline_alignment_removes_rank_offset_only() -> None:
    traces = _framework_alignment_trace()
    original = deepcopy(traces)

    aligned, reports = align_framework_trace_timeline(traces)

    assert traces == original
    assert len(reports) == 1
    report = reports[0]
    assert report.iteration_id == 2
    assert report.group == (0, 1, 2, 3)
    assert report.anchor_count == 3
    assert dict(report.rank_offsets_us) == {0: 0, 1: 300, 2: 500, 3: 400}
    assert dict(report.applied_shifts_us) == {0: 0, 1: -300, 2: -500, 3: -400}

    anchors = [event for event in aligned if event.get("name") == "tp-allreduce"]
    for index in range(3):
        operation = sorted(
            (event for event in anchors if event["pid"] in range(4)),
            key=lambda event: (event["pid"], event["ts"]),
        )[index::3]
        expected_completions = {(2_000, 4_000, 6_000)[index]}
        if index == 1:
            expected_completions.add(4_030)
        assert {event["ts"] + event["dur"] for event in operation} == expected_completions

    expected_shifts = {0: 0, 1: -300, 2: -500, 3: -400}
    for before, after in zip(original, aligned):
        is_framework_record = (
            before.get("ph") in ("X", "C")
            and before.get("name") != "iteration"
            and before.get("record_type") != "cuda_kernel"
        )
        if is_framework_record:
            assert after["ts"] == before["ts"] + expected_shifts[before["pid"]]

    forwards = [event for event in aligned if event.get("name") == "forward"]
    assert {event["ts"] for event in forwards} == {500}
    counters = [event for event in aligned if event.get("ph") == "C"]
    assert {event["ts"] for event in counters} == {700}
    iterations = [event for event in aligned if event.get("name") == "iteration"]
    assert {(event["ts"], event["dur"]) for event in iterations} == {(0, 10_000)}
    kernels = [event for event in aligned if event.get("record_type") == "cuda_kernel"]
    assert {(kernel["pid"], kernel["ts"], kernel["dur"]) for kernel in kernels} == {
        (0, 900, 50),
        (1, 900, 50),
    }
    assert next(kernel for kernel in kernels if kernel["pid"] == 1)["iter_rel_start_us"] == 900
    assert [event.get("name") for event in aligned] == [event.get("name") for event in traces]
    assert [event.get("args") for event in aligned] == [event.get("args") for event in traces]
    assert [event.get("dur") for event in aligned] == [event.get("dur") for event in traces]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-rank", "missing ranks"),
        ("mismatched-count", "mismatched counts"),
        ("mismatched-payload", "mismatched payload/order"),
        ("missing-iteration", "no integer iteration"),
    ],
)
def test_framework_timeline_alignment_rejects_incomplete_or_ambiguous_anchors(
    case: str, message: str
) -> None:
    traces = _framework_alignment_trace()
    if case == "missing-rank":
        traces[:] = [
            event
            for event in traces
            if not (event.get("name") == "tp-allreduce" and event.get("pid") == 3)
        ]
    elif case == "mismatched-count":
        traces.remove(
            next(
                event
                for event in traces
                if event.get("name") == "tp-allreduce"
                and event.get("pid") == 3
                and event.get("ts") > 5_000
            )
        )
    elif case == "mismatched-payload":
        anchor = next(
            event
            for event in traces
            if event.get("name") == "tp-allreduce" and event.get("pid") == 2
        )
        anchor["args"]["data_bytes"] = 8192
    else:
        anchor = next(event for event in traces if event.get("name") == "tp-allreduce")
        del anchor["args"]["iteration"]

    with pytest.raises(ValueError, match=message):
        align_framework_trace_timeline(traces)


def test_directory_loader_converts_raw_rank_traces_end_to_end(tmp_path: Path) -> None:
    _write_rank_trace(tmp_path, Rank(0, 0, 0), [137, 138])
    _write_rank_trace(tmp_path, Rank(1, 0, 0), [138, 137])

    loader = TraceDataLoader.from_directory(tmp_path)

    rank0_forward = loader.get_events_by_name("forward", rank=0, iteration=137)
    rank1_forward = loader.get_events_by_name("forward", rank=1, iteration=137)
    assert len(rank0_forward) == 1
    assert len(rank1_forward) == 1
    assert rank0_forward[0].dur == 4
    assert rank1_forward[0].dur == 4
    assert loader.get_ranks() == [0, 1]
    assert loader.topology == {0: {"dp": 0, "pp": 0, "tp": 0}, 1: {"dp": 1, "pp": 0, "tp": 0}}


def test_global_rank_shards_keep_duplicate_parallel_coordinates_distinct(tmp_path: Path) -> None:
    rank0 = Rank(0, 0, 0, global_rank=0)
    rank1 = Rank(0, 0, 0, global_rank=1)
    _write_rank_trace(tmp_path, rank0, [137])
    _write_rank_trace(tmp_path, rank1, [137])

    loader = TraceDataLoader.from_directory(tmp_path)

    assert loader.get_ranks() == [0, 1]
    assert loader.topology == {0: {"dp": 0, "pp": 0, "tp": 0}, 1: {"dp": 0, "pp": 0, "tp": 0}}


def test_empty_inner_iteration_preserves_rank_identity(tmp_path: Path) -> None:
    for global_rank in (0, 1):
        rank = Rank(0, 0, 0, global_rank=global_rank)
        rows = [_raw_iteration(rank, 137)[0], _raw_iteration(rank, 137)[-1]]
        global_prefix = f"global-{global_rank}-"
        path = tmp_path / (f"benchmark-{global_prefix}data-0-pipeline-0-tensor-0.json")
        path.write_text(json.dumps(rows), encoding="utf-8")

    loader = TraceDataLoader.from_directory(tmp_path)

    assert loader.get_ranks() == [0, 1]
    assert len(loader.get_events_by_name("iteration", iteration=137)) == 2


def test_directory_loader_keeps_post_transform_shard_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "benchmark-data-0-pipeline-0-tensor-0.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "forward",
                    "ph": "X",
                    "ts": 10,
                    "dur": 5,
                    "pid": 0,
                    "args": {"iteration": 137, "dp_rk": 0, "pp_rk": 0, "tp_rk": 0},
                }
            ]
        ),
        encoding="utf-8",
    )

    loader = TraceDataLoader.from_directory(tmp_path)

    assert len(loader.get_events_by_name("forward", rank=0, iteration=137)) == 1


def test_directory_loader_fails_when_no_rank_trace_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No benchmark-\*\.json files"):
        TraceDataLoader.from_directory(tmp_path)


def test_reader_rejects_orphan_kernel_and_unclosed_iteration() -> None:
    rank = Rank(0, 0, 0)
    orphan_kernel = _raw_iteration(rank, 137) + [
        {"record_type": "cuda_kernel", "name": "kernel", "iteration": 999}
    ]
    with pytest.raises(ValueError, match="unknown iteration ID: 999"):
        read_benchmark_file(rank, json.dumps(orphan_kernel))

    unclosed = _raw_iteration(rank, 137)[:-1]
    with pytest.raises(ValueError, match="no matching end"):
        read_benchmark_file(rank, json.dumps(unclosed))


def test_reader_rejects_conflicting_iteration_and_rank_identity() -> None:
    rank = Rank(0, 0, 0, global_rank=3)
    mismatched_end = _raw_iteration(rank, 137)
    mismatched_end[-1]["iteration"] = 138
    with pytest.raises(ValueError, match="boundary IDs do not match"):
        read_benchmark_file(rank, json.dumps(mismatched_end))

    mismatched_inner = _raw_iteration(rank, 137)
    mismatched_inner[1]["iteration"] = 999
    with pytest.raises(ValueError, match="enclosing iteration is 137"):
        read_benchmark_file(rank, json.dumps(mismatched_inner))

    mismatched_rank = _raw_iteration(rank, 137)
    mismatched_rank[1]["g_rk"] = 4
    with pytest.raises(ValueError, match="shard identity requires 3"):
        read_benchmark_file(rank, json.dumps(mismatched_rank))


def test_canonical_iteration_overrides_event_attrs() -> None:
    rank = Rank(0, 0, 0, global_rank=0)
    attrs = {"g_rk": 0, "dp_rk": 0, "pp_rk": 0, "tp_rk": 0, "iteration": 999}
    iteration = Iteration(
        pad_before=0,
        events=[
            Event(1_000, rank, "forward", "B", dict(attrs)),
            Event(5_000, rank, "forward", "E", dict(attrs)),
        ],
        duration=10_000,
        iteration_id=137,
        ranks=(rank,),
    )

    traces = benchmark_to_chrome_trace([iteration])
    forward = next(trace for trace in traces if trace.get("name") == "forward")

    assert forward["args"]["iteration"] == 137


def test_counter_and_kernel_keep_iteration_without_metric_pollution(tmp_path: Path) -> None:
    source_sample = CounterSample("GPU_Metrics", 10, 2, {"SM_Util_pct": 42.0})
    assert source_sample.metrics == {"SM_Util_pct": 42.0}
    assert source_sample.iteration == -1

    rank = Rank(0, 0, 0, global_rank=0)
    rows = _raw_iteration(rank, 137)
    rows.insert(
        -1,
        {
            "name": "GPU_Metrics",
            "ph": "C",
            "rel_ts": 2_000,
            "g_rk": 0,
            "args": {"SM_Util_pct": 42.0},
        },
    )
    rows.append(
        {
            "record_type": "cuda_kernel",
            "name": "kernel",
            "ph": "X",
            "iteration": 137,
            "g_rk": 0,
            "dp_rk": 0,
            "pp_rk": 0,
            "tp_rk": 0,
            "start_us": 2,
            "end_us": 7,
            "duration_us": 5,
        }
    )
    path = tmp_path / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    loader = TraceDataLoader.from_directory(tmp_path)

    assert loader.counter_samples[0].iteration == 137
    assert loader.counter_samples[0].metrics == {"SM_Util_pct": 42.0}
    assert loader.kernel_events[0].iteration == 137
    assert loader.kernel_events[0].duration_us == 5


def test_raw_cuda_kernel_exports_viewer_complete_event() -> None:
    rank0 = Rank(0, 0, 0)
    rank1 = Rank(1, 0, 0)
    rank0_rows = _raw_iteration(rank0, 137) + _raw_iteration(rank0, 138)
    rank1_rows = _raw_iteration(rank1, 137) + _raw_iteration(rank1, 138)
    rank1_rows.extend(
        [
            {
                "record_type": "cuda_kernel",
                "name": f"kernel-{iteration}",
                "ph": "X",
                "iteration": iteration,
                "dp_rk": rank1.data,
                "pp_rk": 0,
                "tp_rk": 0,
                "start_us": 12,
                "end_us": 17,
                "iter_rel_start_us": 2,
                "iter_rel_end_us": 7,
                "duration_us": 5,
                "device": 0,
            }
            for iteration in (137, 138)
        ]
    )

    rank0_contents = read_benchmark_file(rank0, json.dumps(rank0_rows))
    rank1_contents = read_benchmark_file(rank1, json.dumps(rank1_rows))
    rank0_contents[0].pad_before = 1_000
    rank0_contents[1].pad_before = 20_000
    rank1_contents[0].pad_before = 4_000
    rank1_contents[1].pad_before = 25_000
    iterations, _, _, _ = aggregate_benchmark_data([rank0_contents, rank1_contents])
    traces = benchmark_to_chrome_trace(iterations)
    kernels = [trace for trace in traces if trace.get("record_type") == "cuda_kernel"]

    assert [(kernel["pid"], kernel["ts"], kernel["dur"]) for kernel in kernels] == [
        (1, 6, 5),
        (1, 41, 5),
    ]
    assert all(kernel["cat"] == "cuda_kernel" for kernel in kernels)
    assert all(kernel["start_us"] == 12 for kernel in kernels)
    assert all(kernel["iter_rel_start_us"] == 2 for kernel in kernels)


def test_transform_rejects_unbalanced_or_mismatched_spans() -> None:
    args = {"dp_rk": 0, "pp_rk": 0, "tp_rk": 0}
    begin = {"name": "forward", "ph": "B", "pid": 0, "ts": 1, "args": args}

    with pytest.raises(ValueError, match="no matching end"):
        transform([begin.copy()])

    wrong_end = {"name": "backward", "ph": "E", "pid": 0, "ts": 2, "args": args}
    with pytest.raises(ValueError, match="closes begin 'forward'"):
        transform([begin.copy(), wrong_end])

    early_end = {"name": "forward", "ph": "E", "pid": 0, "ts": 0, "args": args}
    with pytest.raises(ValueError, match="precedes its begin"):
        transform([begin.copy(), early_end])

    wrong_thread = {"name": "forward", "ph": "E", "pid": 0, "tid": 1, "ts": 2, "args": args}
    with pytest.raises(ValueError, match="pid/tid"):
        transform([begin.copy(), wrong_thread])


def test_trace_aggregate_import_has_no_dependency_or_logging_side_effect() -> None:
    script = """
import builtins
import logging

blocked = {"numpy", "pytz"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise RuntimeError(f"unexpected dependency import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
root_logger = logging.getLogger()
handlers_before = tuple(root_logger.handlers)
import megatron.megalens.trace_aggregate
handlers_after = tuple(root_logger.handlers)
if handlers_after != handlers_before:
    raise RuntimeError("trace_aggregate configured the root logger during import")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
