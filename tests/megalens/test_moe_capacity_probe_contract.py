# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from tests.test_utils.runners import moe_capacity_probe_contract as contract


def _end_attrs(occurrence: int) -> dict[str, object]:
    return {
        "layer": occurrence + 1,
        "ep_size": 2,
        "num_experts": 4,
        "num_local_experts": 2,
        "router_topk": 2,
        "dropped_tokens": 128,
        "drop_rate": 0.5,
        "expert_cv": 0.25,
        "top1_expert_share": 0.5,
        "aux_loss": 0.01,
        "z_loss": 0.02,
    }


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

    for iteration in (1, 2):
        rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration})
        for occurrence in range(2):
            handoff = _end_attrs(occurrence)
            event("moe-router", "B", iteration)
            event(
                "moe-router",
                "E",
                iteration,
                num_tokens=128,
                routed_tokens=128,
                routing_entropy=1.0,
                **handoff,
            )
            event("moe-dispatch", "B", iteration)
            event(
                "moe-dispatch",
                "E",
                iteration,
                dispatcher="alltoall",
                num_tokens=128,
                capacity_factor=0.5,
                **handoff,
            )
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
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


def _first_end(
    rows: list[dict[str, object]],
    name: str,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("name") == name
        and row.get("ph") == "E"
        and row.get("iteration") == 1
    )


def test_ep2_capacity_drop_contract_accepts_two_rank_training_trace(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    assert contract.validate_ep2_capacity_drop(trace_root) == ()


def test_ep2_capacity_drop_contract_requires_nonzero_drop(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def remove_drop(rows: list[dict[str, object]]) -> None:
        router = _first_end(rows, "moe-router")
        dispatch = _first_end(rows, "moe-dispatch")
        router.update(routed_tokens=256, dropped_tokens=0, drop_rate=0.0)
        dispatch.update(dropped_tokens=0, drop_rate=0.0)

    _mutate_rank_zero(trace_root, remove_drop)

    failures = contract.validate_ep2_capacity_drop(trace_root)

    assert "trace.moe_capacity.dropped_tokens" in {failure.code for failure in failures}


def test_ep2_capacity_drop_contract_enforces_the_configured_capacity_bound(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def exceed_capacity(rows: list[dict[str, object]]) -> None:
        router = _first_end(rows, "moe-router")
        dispatch = _first_end(rows, "moe-dispatch")
        router.update(routed_tokens=129, dropped_tokens=127, drop_rate=127 / 256)
        dispatch.update(num_tokens=129, dropped_tokens=127, drop_rate=127 / 256)

    _mutate_rank_zero(trace_root, exceed_capacity)

    failures = contract.validate_ep2_capacity_drop(trace_root)

    assert "trace.moe_capacity.capacity_bound" in {
        failure.code for failure in failures
    }


def test_ep2_capacity_drop_contract_requires_matching_handoff(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_end(rows, "moe-dispatch").__setitem__("expert_cv", 0.75),
    )

    failures = contract.validate_ep2_capacity_drop(trace_root)

    assert "trace.moe_capacity.handoff" in {failure.code for failure in failures}


def test_ep2_capacity_drop_contract_requires_router_then_dispatch_order(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def reorder_first_dispatch_end(rows: list[dict[str, object]]) -> None:
        dispatch_index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-dispatch"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        dispatch = rows.pop(dispatch_index)
        router_ends = [
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-router"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        ]
        rows.insert(router_ends[1] + 1, dispatch)

    _mutate_rank_zero(trace_root, reorder_first_dispatch_end)

    failures = contract.validate_ep2_capacity_drop(trace_root)

    assert "trace.moe_capacity.order" in {failure.code for failure in failures}


def test_ep2_capacity_drop_contract_requires_profile_capacity_factor(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_end(rows, "moe-dispatch").__setitem__(
            "capacity_factor", 1.0
        ),
    )

    failures = contract.validate_ep2_capacity_drop(trace_root)

    assert "trace.moe_capacity.capacity_factor" in {
        failure.code for failure in failures
    }
