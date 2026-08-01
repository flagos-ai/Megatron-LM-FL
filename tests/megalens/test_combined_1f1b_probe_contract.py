# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from tests.test_utils.runners import combined_1f1b_probe_contract as contract
from tests.test_utils.runners import run_flagscale_megalens as gate

_EVENT = "combined-forward-backward-step"
_KEYS = tuple((microbatch, vp_stage) for microbatch in range(4) for vp_stage in range(2))


def _operation_id(key: tuple[int, int] | None) -> str | None:
    if key is None:
        return None
    return f"pp:microbatch={key[0]}:vp={key[1]}"


def _context(
    mode: str,
    forward: tuple[int, int] | None,
    backward: tuple[int, int] | None,
) -> dict[str, object]:
    forward_id = _operation_id(forward)
    backward_id = _operation_id(backward)
    return {
        "operation_id": f"pp-combined:forward={forward_id}:backward={backward_id}",
        "forward_operation_id": forward_id,
        "backward_operation_id": backward_id,
        "forward_microbatch": None if forward is None else forward[0],
        "backward_microbatch": None if backward is None else backward[0],
        "forward_vp_stage": None if forward is None else forward[1],
        "backward_vp_stage": None if backward is None else backward[1],
        "execution_mode": mode,
        "overlap_active": mode == "combined",
        "schedule": "combined-1f1b",
        "timing_phase": "framework_phase",
    }


def _steps(
    pipeline_rank: int,
) -> tuple[tuple[str, tuple[int, int] | None, tuple[int, int] | None], ...]:
    warmup = 5 if pipeline_rank == 0 else 3
    return (
        *(("forward", key, None) for key in _KEYS[:warmup]),
        *(("combined", forward, backward) for forward, backward in zip(_KEYS[warmup:], _KEYS)),
        *(("backward", None, key) for key in _KEYS[len(_KEYS) - warmup :]),
    )


def _write_trace(
    trace_root: Path,
    rank: int,
    iterations: tuple[int, int] = (1, 2),
) -> None:
    pipeline_rank = rank // 2
    data_rank = rank % 2
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
                "g_rk": rank,
                "dp_rk": data_rank,
                "pp_rk": pipeline_rank,
                "tp_rk": 0,
                **attrs,
            }
        )

    for iteration in iterations:
        rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration})
        for mode, forward, backward in _steps(pipeline_rank):
            event(_EVENT, "B", iteration, **_context(mode, forward, backward))
            event(_EVENT, "E", iteration)
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
        f"benchmark-global-{rank}-data-{data_rank}-pipeline-{pipeline_rank}-tensor-0.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_profile(
    trace_root: Path,
    iterations: tuple[int, int] = (1, 2),
) -> None:
    for rank in range(4):
        _write_trace(trace_root, rank, iterations)


def _mutate_rank_zero(
    trace_root: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    path = trace_root / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    mutate(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _first_begin(rows: list[dict[str, object]], mode: str) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("name") == _EVENT
        and row.get("ph") == "B"
        and row.get("iteration") == 1
        and row.get("execution_mode") == mode
    )


def test_ep2_profile_registers_and_accepts_the_combined_1f1b_contract(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    profile = gate.PROFILES["ep2-fine-grained"]

    assert profile.contract is contract.validate_ep2_fine_grained_combined
    assert profile.contract(trace_root) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("overlap_active", True),
        ("schedule", "other-schedule"),
        ("timing_phase", "other-phase"),
    ),
)
def test_contract_rejects_mode_field_inconsistency(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_begin(rows, "forward").__setitem__(field, value),
    )

    failures = contract.validate_ep2_fine_grained_combined(trace_root)

    assert "trace.combined_1f1b.field" in {failure.code for failure in failures}


def test_contract_rejects_unpaired_scope(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def remove_first_end(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == _EVENT and row.get("ph") == "E"
        )
        rows.pop(index)

    _mutate_rank_zero(trace_root, remove_first_end)

    failures = contract.validate_ep2_fine_grained_combined(trace_root)

    assert "trace.combined_1f1b.pairing" in {failure.code for failure in failures}


def test_contract_rejects_identity_and_coverage_drift(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def replace_first_identity(rows: list[dict[str, object]]) -> None:
        begin = _first_begin(rows, "forward")
        begin["forward_microbatch"] = 99
        begin["forward_operation_id"] = "pp:microbatch=99:vp=0"
        begin["operation_id"] = "pp-combined:forward=pp:microbatch=99:vp=0:backward=None"

    _mutate_rank_zero(trace_root, replace_first_identity)

    failures = contract.validate_ep2_fine_grained_combined(trace_root)
    codes = {failure.code for failure in failures}

    assert "trace.combined_1f1b.coverage" in codes
    assert "trace.combined_1f1b.operation_id_coverage" in codes


def test_contract_requires_all_three_execution_modes(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def replace_forward_modes(rows: list[dict[str, object]]) -> None:
        for row in rows:
            if (
                row.get("name") == _EVENT
                and row.get("ph") == "B"
                and row.get("execution_mode") == "forward"
            ):
                row["execution_mode"] = "combined"
                row["overlap_active"] = True

    _mutate_rank_zero(trace_root, replace_forward_modes)

    failures = contract.validate_ep2_fine_grained_combined(trace_root)

    assert "trace.combined_1f1b.mode_coverage" in {failure.code for failure in failures}


def test_contract_requires_the_profile_iteration_ids(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, iterations=(1, 3))

    failures = contract.validate_ep2_fine_grained_combined(trace_root)

    assert "trace.combined_1f1b.iterations" in {failure.code for failure in failures}
