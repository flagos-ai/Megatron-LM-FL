# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from tests.test_utils.runners import moe_shared_expert_probe_contract as contract


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
        for layer in (1, 2):
            event("moe-shared-expert", "B", iteration)
            event("moe-shared-expert", "E", iteration, layer=layer, ep_size=2)
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


def _first_end(rows: list[dict[str, object]]) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("name") == "moe-shared-expert"
        and row.get("ph") == "E"
        and row.get("iteration") == 1
    )


def test_ep2_shared_expert_contract_accepts_two_rank_training_trace(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    assert contract.validate_ep2_shared_expert(trace_root) == ()


def test_ep2_shared_expert_contract_requires_strict_begin_end_phases(
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

    failures = contract.validate_ep2_shared_expert(trace_root)

    assert "trace.moe_shared_expert.phases" in {failure.code for failure in failures}


def test_ep2_shared_expert_contract_requires_layers_one_and_two(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_end(rows).__setitem__("layer", 2),
    )

    failures = contract.validate_ep2_shared_expert(trace_root)

    assert "trace.moe_shared_expert.layers" in {failure.code for failure in failures}


def test_ep2_shared_expert_contract_requires_ep2_scope(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_end(rows).__setitem__("ep_size", 1),
    )

    failures = contract.validate_ep2_shared_expert(trace_root)

    assert "trace.moe_shared_expert.ep_size" in {failure.code for failure in failures}
