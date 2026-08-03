# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import pytest

from tests.test_utils.runners import moe_recompute_fp8_probe_contract as contract
from tests.test_utils.runners.megalens_run_manifest import Failure


def _topology(layer: int) -> dict[str, object]:
    return {
        "layer": layer,
        "ep_size": 2,
        "num_experts": 4,
        "num_local_experts": 2,
    }


def _handoff(
    layer: int,
    occurrence: int,
    *,
    recompute: bool,
) -> dict[str, object]:
    return {
        "dropped_tokens": 0,
        "drop_rate": 0.0,
        "expert_cv": 0.1 + layer,
        "top1_expert_share": 0.5,
        "aux_loss": None if recompute and occurrence < 2 else 0.01 + occurrence,
        "z_loss": None if recompute and occurrence < 2 else 0.02 + occurrence,
    }


def _write_rank_trace(trace_root: Path, rank: int, layers: Sequence[int]) -> None:
    timestamp = 0
    rows: list[dict[str, object]] = []
    recompute = len(layers) == 4

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
                "dp_rk": 0,
                "pp_rk": 0,
                "tp_rk": 0,
                **attrs,
            }
        )

    for iteration in (1, 2):
        rows.append(
            {"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration}
        )
        for occurrence, layer in enumerate(layers):
            topology = _topology(layer)
            handoff = _handoff(layer, occurrence, recompute=recompute)
            event("moe-router", "B", iteration)
            event(
                "moe-router",
                "E",
                iteration,
                router_topk=2,
                num_tokens=128,
                routed_tokens=256,
                routing_entropy=1.0,
                **topology,
                **handoff,
            )
            event("moe-dispatch", "B", iteration)
            event(
                "ep-alltoall-dispatch",
                "B",
                iteration,
                comm_type="ep-alltoall",
                dispatcher="alltoall",
                group_size=2,
            )
            event("ep-alltoall-dispatch", "E", iteration)
            event(
                "moe-dispatch",
                "E",
                iteration,
                router_topk=2,
                dispatcher="alltoall",
                num_tokens=128,
                capacity_factor=None,
                **topology,
                **handoff,
            )
            event("moe-experts", "B", iteration)
            event(
                "moe-experts",
                "E",
                iteration,
                routed_tokens=128,
                expert_cv=0.25,
                top1_expert_share=0.5,
                expert_max_over_mean=1.5,
                tokens_per_expert=[64, 64],
                **topology,
            )
            event("moe-combine", "B", iteration)
            event(
                "ep-alltoall-combine",
                "B",
                iteration,
                comm_type="ep-alltoall",
                dispatcher="alltoall",
                group_size=2,
            )
            event("ep-alltoall-combine", "E", iteration)
            event(
                "moe-combine",
                "E",
                iteration,
                dispatcher="alltoall",
                num_tokens=128,
                **topology,
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


def _write_profile(trace_root: Path, layers: Sequence[int]) -> None:
    for rank in (0, 1):
        _write_rank_trace(trace_root, rank, layers)


def _mutate_rank_zero(
    trace_root: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    path = trace_root / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    mutate(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


@pytest.mark.parametrize(
    ("validate", "layers"),
    (
        (contract.validate_ep2_recompute, (1, 2, 2, 1)),
        (contract.validate_ep2_fp8, (1, 2)),
        (contract.validate_ep2_fp8_recompute, (1, 2, 2, 1)),
    ),
)
def test_recompute_fp8_contract_accepts_the_exact_layer_calls(
    tmp_path: Path,
    validate: Callable[[Path], tuple[Failure, ...]],
    layers: Sequence[int],
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, layers)

    assert validate(trace_root) == ()


def test_recompute_contract_requires_reverse_order_backward_calls(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2, 1, 2))

    failures = contract.validate_ep2_recompute(trace_root)

    assert "trace.moe_recompute_fp8.layers" in {failure.code for failure in failures}


def test_recompute_contract_requires_complete_adjacent_phase_scopes(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2, 2, 1))

    def remove_experts_end(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-experts"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        rows.pop(index)

    _mutate_rank_zero(trace_root, remove_experts_end)
    failures = contract.validate_ep2_recompute(trace_root)

    assert "trace.moe_recompute_fp8.sequence" in {failure.code for failure in failures}


def test_recompute_contract_requires_invocation_local_router_handoff(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2, 2, 1))

    def change_dispatch_field(rows: list[dict[str, object]]) -> None:
        dispatch = next(
            row
            for row in rows
            if row.get("name") == "moe-dispatch"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        dispatch["aux_loss"] = 99.0

    _mutate_rank_zero(trace_root, change_dispatch_field)
    failures = contract.validate_ep2_recompute(trace_root)

    assert "trace.moe_recompute_fp8.handoff" in {failure.code for failure in failures}


def test_recompute_contract_requires_grad_enabled_aux_loss_on_backward_calls(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2, 2, 1))

    def clear_aux_loss(rows: list[dict[str, object]]) -> None:
        for row in rows:
            if row.get("name") in {"moe-router", "moe-dispatch"} and row.get("ph") == "E":
                row["aux_loss"] = None

    _mutate_rank_zero(trace_root, clear_aux_loss)
    failures = contract.validate_ep2_recompute(trace_root)

    assert "trace.moe_recompute_fp8.aux_loss_mode" in {
        failure.code for failure in failures
    }


@pytest.mark.parametrize(
    ("layer", "reentry_occurrence"),
    ((1, 3), (2, 2)),
)
def test_recompute_contract_pairs_router_workload_with_same_layer_reentry(
    tmp_path: Path,
    layer: int,
    reentry_occurrence: int,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2, 2, 1))

    def change_reentry_workload(rows: list[dict[str, object]]) -> None:
        for event_name in ("moe-router", "moe-dispatch"):
            end_events = [
                row
                for row in rows
                if row.get("name") == event_name
                and row.get("ph") == "E"
                and row.get("iteration") == 1
            ]
            end_events[reentry_occurrence]["expert_cv"] = 99.0

    _mutate_rank_zero(trace_root, change_reentry_workload)
    failures = contract.validate_ep2_recompute(trace_root)

    failure_codes = {failure.code for failure in failures}
    assert "trace.moe_recompute_fp8.reentry" in failure_codes
    assert "trace.moe_recompute_fp8.handoff" not in failure_codes
    assert any(
        f"layer={layer}" in failure.message
        and f"occurrence={3 - reentry_occurrence}<->{reentry_occurrence}"
        in failure.message
        for failure in failures
    )


@pytest.mark.parametrize("phases", (("B", "E"), ("B",), ("E",)))
def test_contract_rejects_unconsumed_ep_collective_outside_moe_scopes(
    tmp_path: Path,
    phases: Sequence[str],
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2))

    def add_extra_collective(rows: list[dict[str, object]]) -> None:
        iteration_end = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "iteration"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        rows[iteration_end:iteration_end] = [
            {
                "name": "ep-alltoall-dispatch",
                "ph": phase,
                "iteration": 1,
                "rel_ts": 999_000 + offset,
            }
            for offset, phase in enumerate(phases)
        ]

    _mutate_rank_zero(trace_root, add_extra_collective)
    failures = contract.validate_ep2_fp8(trace_root)

    assert "trace.moe_recompute_fp8.collective_sequence" in {
        failure.code for failure in failures
    }


def test_fp8_contract_requires_alltoall_inside_dispatch_and_combine(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2))

    def remove_dispatch_collective_end(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "ep-alltoall-dispatch"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        rows.pop(index)

    _mutate_rank_zero(trace_root, remove_dispatch_collective_end)
    failures = contract.validate_ep2_fp8(trace_root)

    assert "trace.moe_recompute_fp8.collective_phases" in {
        failure.code for failure in failures
    }


def test_fp8_contract_rejects_unconfigured_recompute_calls(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root, (1, 2, 2, 1))

    failures = contract.validate_ep2_fp8(trace_root)

    assert "trace.moe_recompute_fp8.sequence" in {failure.code for failure in failures}
