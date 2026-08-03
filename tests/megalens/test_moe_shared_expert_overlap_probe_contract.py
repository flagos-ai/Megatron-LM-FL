# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from tests.test_utils.runners import moe_shared_expert_overlap_probe_contract as contract

_STAGES = (
    "pre_forward_comm",
    "linear_fc1_forward_and_act",
    "linear_fc2_forward",
    "post_forward_comm",
    "get_output",
)


def _write_rank_trace(trace_root: Path, rank: int) -> None:
    rows: list[dict[str, object]] = []

    def event(
        name: str,
        phase: str,
        iteration: int,
        rel_ts: int,
        **attrs: object,
    ) -> None:
        rows.append(
            {
                "name": name,
                "ph": phase,
                "rel_ts": rel_ts,
                "iteration": iteration,
                **attrs,
            }
        )

    for iteration in (1, 2):
        rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration})
        for layer in (1, 2):
            base = (iteration - 1) * 10_000 + (layer - 1) * 1_000

            def shared_scope(stage: str, start: int, end: int) -> None:
                event("moe-shared-expert", "B", iteration, base + start)
                event(
                    "moe-shared-expert",
                    "E",
                    iteration,
                    base + end,
                    layer=layer,
                    ep_size=2,
                    stage=stage,
                )

            def ep_scope(name: str, start: int, end: int) -> None:
                event(name, "B", iteration, base + start)
                event(name, "E", iteration, base + end)

            shared_scope("pre_forward_comm", 10, 20)
            ep_scope("ep-alltoall-dispatch", 30, 90)
            shared_scope("linear_fc1_forward_and_act", 50, 70)
            shared_scope("linear_fc2_forward", 110, 130)
            ep_scope("ep-alltoall-combine", 120, 180)
            shared_scope("post_forward_comm", 190, 200)
            shared_scope("get_output", 210, 220)
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": iteration * 10_000,
            }
        )

    trace_root.mkdir(parents=True, exist_ok=True)
    path = trace_root / f"benchmark-global-{rank}-data-0-pipeline-0-tensor-0.json"
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_profile(trace_root: Path) -> None:
    for rank in (0, 1):
        _write_rank_trace(trace_root, rank)


def _mutate_rank_zero(
    trace_root: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    path = trace_root / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    mutate(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _first_end(rows: list[dict[str, object]]) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("name") == "moe-shared-expert"
        and row.get("ph") == "E"
        and row.get("iteration") == 1
    )


def test_ep2_shared_expert_overlap_contract_accepts_forty_scopes(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    assert contract.validate_ep2_shared_expert_overlap(trace_root) == ()


def test_shared_expert_overlap_contract_accepts_cross_stream_timestamp_regression(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    path = trace_root / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    timestamps = [
        row["rel_ts"]
        for row in rows
        if row.get("iteration") == 1 and "rel_ts" in row
    ]

    assert timestamps != sorted(timestamps)
    assert contract.validate_ep2_shared_expert_overlap(trace_root) == ()


def _move_shared_intervals_outside_ep(rows: list[dict[str, object]]) -> None:
    for iteration in (1, 2):
        events = [
            row
            for row in rows
            if row.get("name") == "moe-shared-expert"
            and row.get("iteration") == iteration
        ]
        for index, row in enumerate(events):
            scope = index // 2
            layer_index = scope // len(_STAGES)
            stage_index = scope % len(_STAGES)
            base = (iteration - 1) * 10_000 + layer_index * 1_000
            row["rel_ts"] = base + 300 + stage_index * 20 + index % 2


def _remove_profile_overlap(trace_root: Path) -> None:
    _mutate_rank_zero(trace_root, _move_shared_intervals_outside_ep)
    rank_one = trace_root / "benchmark-global-1-data-0-pipeline-0-tensor-0.json"
    rank_one_rows = json.loads(rank_one.read_text(encoding="utf-8"))
    _move_shared_intervals_outside_ep(rank_one_rows)
    rank_one.write_text(json.dumps(rank_one_rows), encoding="utf-8")


def test_shared_expert_overlap_contract_accepts_disjoint_device_intervals(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _remove_profile_overlap(trace_root)

    assert contract.validate_ep2_shared_expert_overlap(trace_root) == ()


def test_shared_expert_overlap_contract_requires_positive_ep_intervals(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def collapse_first_dispatch(rows: list[dict[str, object]]) -> None:
        events = [
            row
            for row in rows
            if row.get("name") == "ep-alltoall-dispatch"
            and row.get("iteration") == 1
        ]
        events[1]["rel_ts"] = events[0]["rel_ts"]

    _mutate_rank_zero(trace_root, collapse_first_dispatch)

    failures = contract.validate_ep2_shared_expert_overlap(trace_root)

    assert "trace.moe_shared_expert_overlap.interval" in {
        failure.code for failure in failures
    }


def test_shared_expert_overlap_contract_requires_strict_begin_end_phases(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def remove_first_begin(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-shared-expert"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        rows.pop(index)

    _mutate_rank_zero(trace_root, remove_first_begin)

    failures = contract.validate_ep2_shared_expert_overlap(trace_root)

    assert "trace.moe_shared_expert_overlap.phases" in {
        failure.code for failure in failures
    }


def test_shared_expert_overlap_contract_requires_layers_one_and_two(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_end(rows).__setitem__("layer", 2),
    )

    failures = contract.validate_ep2_shared_expert_overlap(trace_root)

    assert "trace.moe_shared_expert_overlap.layers" in {
        failure.code for failure in failures
    }


def test_shared_expert_overlap_contract_requires_five_stage_order(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def swap_first_two_stages(rows: list[dict[str, object]]) -> None:
        events = [
            row
            for row in rows
            if row.get("name") == "moe-shared-expert"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
            and row.get("layer") == 1
        ]
        events[0]["stage"], events[1]["stage"] = events[1]["stage"], events[0]["stage"]

    _mutate_rank_zero(trace_root, swap_first_two_stages)

    failures = contract.validate_ep2_shared_expert_overlap(trace_root)

    assert "trace.moe_shared_expert_overlap.stages" in {
        failure.code for failure in failures
    }


def test_shared_expert_overlap_contract_requires_ep2_scope(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_end(rows).__setitem__("ep_size", 1),
    )

    failures = contract.validate_ep2_shared_expert_overlap(trace_root)

    assert "trace.moe_shared_expert_overlap.ep_size" in {
        failure.code for failure in failures
    }
