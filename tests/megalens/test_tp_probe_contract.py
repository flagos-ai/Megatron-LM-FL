# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from tests.test_utils.runners import tp_probe_contract


def _write_collective_trace(
    trace_root: Path,
    *,
    rank: int,
    omit_nested_reduce_scatter: bool = False,
    cross_all_gather_scopes: bool = False,
) -> None:
    rows: list[dict[str, object]] = []
    timestamp = 0

    def event(name: str, phase: str, **attrs: object) -> None:
        nonlocal timestamp
        timestamp += 1
        rows.append(
            {
                "name": name,
                "ph": phase,
                "rel_ts": timestamp,
                "g_rk": rank,
                "dp_rk": 0,
                "pp_rk": 0,
                "tp_rk": rank,
                **attrs,
            }
        )

    def collective(name: str, *, op: str, dim: str) -> None:
        event(name, "B", op=op, dim=dim, data_bytes=32768, group_size=2)
        event(name, "E", group=[1 - rank])

    for iteration in (1, 2):
        rows.append(
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        if cross_all_gather_scopes:
            event(
                "tp-all-gather-first",
                "B",
                op="all-gather",
                dim="first",
                data_bytes=32768,
                group_size=2,
            )
            event(
                "tp-all-gather-last",
                "B",
                op="all-gather",
                dim="last",
                data_bytes=32768,
                group_size=2,
            )
            event("tp-all-gather-first", "E", group=[1 - rank])
            event("tp-all-gather-last", "E", group=[1 - rank])
        else:
            collective("tp-all-gather-first", op="all-gather", dim="first")
            collective("tp-all-gather-first", op="all-gather", dim="first")
            collective("tp-all-gather-last", op="all-gather", dim="last")
        event(
            "tp-reduce-scatter-last",
            "B",
            op="reduce-scatter",
            dim="last",
            data_bytes=32768,
            group_size=2,
        )
        if not omit_nested_reduce_scatter:
            collective("tp-reduce-scatter", op="reduce-scatter", dim="first")
        event("tp-reduce-scatter-last", "E", group=[1 - rank])
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    trace_root.mkdir(parents=True, exist_ok=True)
    path = trace_root / f"benchmark-global-{rank}-data-0-pipeline-0-tensor-{rank}.json"
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_tp2_collective_contract_accepts_first_last_hierarchy(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(tmp_path, rank=rank)

    assert tp_probe_contract.validate_tp2_gqa_collective_hierarchy(tmp_path) == ()


def test_tp2_collective_contract_requires_nested_physical_reduce_scatter(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            omit_nested_reduce_scatter=rank == 0,
        )

    failures = tp_probe_contract.validate_tp2_gqa_collective_hierarchy(tmp_path)

    assert "trace.tp.collective_hierarchy" in {
        failure.code for failure in failures
    }


def test_tp2_collective_contract_rejects_crossed_scopes(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            cross_all_gather_scopes=rank == 0,
        )

    failures = tp_probe_contract.validate_tp2_gqa_collective_hierarchy(tmp_path)

    assert "trace.tp.nesting" in {failure.code for failure in failures}
