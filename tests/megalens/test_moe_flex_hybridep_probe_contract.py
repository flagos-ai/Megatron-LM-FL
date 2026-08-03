# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from tests.test_utils.runners import moe_flex_hybridep_probe_contract as contract


def _write_rank_trace(trace_root: Path, rank: int) -> None:
    timestamp = 0
    rows: list[dict[str, object]] = []

    def event(name: str, phase: str, iteration: int, **attrs: object) -> None:
        nonlocal timestamp
        timestamp += 1
        rows.append(
            {
                "name": name,
                "ph": phase,
                "rel_ts": timestamp,
                "iteration": iteration,
                **attrs,
            }
        )

    metadata = {
        "comm_type": "ep-hybridep",
        "dispatcher": "flex",
        "group_size": 4,
        "ep_size": 4,
        "tp_size": 1,
        "data_bytes": 16384,
    }
    for iteration in (1, 2):
        rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration})
        for _layer in (1, 2):
            event("ep-alltoall-dispatch", "B", iteration)
            event("ep-alltoall-dispatch", "E", iteration, **metadata)
            event("ep-alltoall-combine", "B", iteration)
            event("ep-alltoall-combine", "E", iteration, **metadata)
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    trace_root.mkdir(parents=True, exist_ok=True)
    path = trace_root / (
        f"benchmark-global-{rank}-data-0-pipeline-0-tensor-{rank % 2}.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_profile(trace_root: Path) -> None:
    for rank in range(8):
        _write_rank_trace(trace_root, rank)


def _mutate_rank_zero(
    trace_root: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    path = trace_root / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    mutate(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _first_dispatch_end(rows: list[dict[str, object]]) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("name") == "ep-alltoall-dispatch"
        and row.get("ph") == "E"
        and row.get("iteration") == 1
    )


def test_tp2_ep4_flex_hybridep_contract_accepts_eight_rank_training_trace(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    assert contract.validate_tp2_ep4_flex_hybridep(trace_root) == ()


def test_flex_hybridep_contract_requires_strict_begin_end_phases(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def remove_first_begin(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "ep-alltoall-dispatch"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        rows.pop(index)

    _mutate_rank_zero(trace_root, remove_first_begin)

    failures = contract.validate_tp2_ep4_flex_hybridep(trace_root)

    assert "trace.moe_flex_hybridep.phases" in {failure.code for failure in failures}


def test_flex_hybridep_contract_requires_backend_metadata_and_actual_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def change_metadata(rows: list[dict[str, object]]) -> None:
        primitive = _first_dispatch_end(rows)
        primitive["comm_type"] = "ep-deepep"
        primitive["group_size"] = 8

    _mutate_rank_zero(trace_root, change_metadata)

    failures = contract.validate_tp2_ep4_flex_hybridep(trace_root)

    metadata_failures = [
        failure for failure in failures if failure.code == "trace.moe_flex_hybridep.metadata"
    ]
    assert len(metadata_failures) == 2


def test_flex_hybridep_contract_rejects_async_lifecycle_events(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def add_async_launch(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "ep-alltoall-dispatch"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        rows.insert(
            index,
            {
                "name": "ep-alltoall-async-launch",
                "ph": "B",
                "rel_ts": 0,
                "iteration": 1,
            },
        )

    _mutate_rank_zero(trace_root, add_async_launch)

    failures = contract.validate_tp2_ep4_flex_hybridep(trace_root)

    assert "trace.moe_flex_hybridep.lifecycle" in {
        failure.code for failure in failures
    }


def test_flex_hybridep_contract_rejects_async_fields_on_fused_primitive(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_dispatch_end(rows).__setitem__("operation_id", "a2a:0"),
    )

    failures = contract.validate_tp2_ep4_flex_hybridep(trace_root)

    assert "trace.moe_flex_hybridep.lifecycle" in {
        failure.code for failure in failures
    }
