# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from tests.test_utils.runners import moe_flex_deepep_probe_contract as contract


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
        "comm_type": "ep-deepep",
        "dispatcher": "flex",
        "group_size": 4,
        "ep_size": 4,
        "tp_size": 1,
        "data_bytes": 16384,
    }

    def tp_collective(
        name: str,
        iteration: int,
        *,
        op: str,
        dim: str | None = None,
    ) -> None:
        attrs: dict[str, object] = {
            "op": op,
            "data_bytes": 16384,
            "group_size": 2,
        }
        if dim is not None:
            attrs["dim"] = dim
        if name == "tp-allreduce":
            attrs.update(
                {
                    "timing_phase": "collective_call",
                    "payload_role": "inplace_input_output",
                }
            )
        event(name, "B", iteration, **attrs)
        event(name, "E", iteration, group=[rank ^ 1])

    def linear_route(
        iteration: int,
        *,
        operation_id: str,
        collective_op: str,
        launch_site: str,
        payload_role: str,
        completion_site: str,
        wait_role: str,
    ) -> None:
        route = {
            "operation_id": operation_id,
            "operation_id_scope": "rank_local",
            "execution_route": "local_linear_direct_async",
            "collective_op": collective_op,
            "data_bytes": 16384,
            "group_size": 2,
            "launch_site": launch_site,
            "pass_direction": "backward",
            "payload_role": payload_role,
            "dim": "first",
        }
        event(
            "tp-linear-async-launch",
            "B",
            iteration,
            **route,
            async_op=True,
            completion_included=False,
            timing_phase="launch_attempt",
        )
        event(
            "tp-linear-async-launch",
            "E",
            iteration,
            api_returned=True,
            error_type=None,
        )
        event(
            "tp-linear-async-complete",
            "B",
            iteration,
            **route,
            completion_guarantee="current_stream_after_wait",
            completion_included=True,
            completion_kind="work_wait",
            completion_site=completion_site,
            duration_attribution="per_request",
            global_device_completion_guaranteed=False,
            host_blocking_guaranteed=False,
            launch_observed=True,
            op="wait",
            terminal=True,
            timing_phase="stream_dependency",
            wait_role=wait_role,
        )
        event(
            "tp-linear-async-complete",
            "E",
            iteration,
            completed=True,
            error_type=None,
        )

    for iteration in (1, 2):
        rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": iteration})
        tp_collective(
            "tp-reduce-scatter",
            iteration,
            op="reduce-scatter",
            dim="first",
        )
        for _layer in (1, 2):
            event("moe-router", "B", iteration)
            tp_collective("tp-allreduce", iteration, op="all_reduce")
            event("moe-router", "E", iteration)
            event("moe-dispatch", "B", iteration)
            event("ep-alltoall-dispatch", "B", iteration)
            event("ep-alltoall-dispatch", "E", iteration, **metadata)
            event("moe-dispatch", "E", iteration)
            event("moe-combine", "B", iteration)
            event("ep-alltoall-combine", "B", iteration)
            event("ep-alltoall-combine", "E", iteration, **metadata)
            event("moe-combine", "E", iteration)
        tp_collective(
            "tp-all-gather-first",
            iteration,
            op="all-gather",
            dim="first",
        )
        operation_base = (iteration - 1) * 2
        linear_route(
            iteration,
            operation_id=f"tp-linear:{operation_base + 1}",
            collective_op="all-gather",
            launch_site="linear_backward_wgrad_input_all_gather",
            payload_role="weight_gradient_input",
            completion_site="linear_backward_wgrad_input_ready",
            wait_role="dependency",
        )
        linear_route(
            iteration,
            operation_id=f"tp-linear:{operation_base + 2}",
            collective_op="reduce-scatter",
            launch_site="linear_backward_dgrad_reduce_scatter",
            payload_role="input_gradient",
            completion_site="linear_backward_dgrad_reduce_scatter_return",
            wait_role="return",
        )
        tp_collective(
            "tp-all-gather-first",
            iteration,
            op="all-gather",
            dim="first",
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
    path = trace_root / (
        f"benchmark-global-{rank}-data-{rank // 2}-pipeline-0-"
        f"tensor-{rank % 2}.json"
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


def test_tp2_ep4_flex_deepep_contract_accepts_eight_rank_training_trace(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    assert contract.validate_tp2_ep4_flex_deepep(trace_root) == ()


def test_flex_deepep_contract_requires_strict_begin_end_phases(
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

    failures = contract.validate_tp2_ep4_flex_deepep(trace_root)

    assert "trace.moe_flex_deepep.phases" in {failure.code for failure in failures}


def test_flex_deepep_contract_requires_source_metadata_and_actual_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def change_metadata(rows: list[dict[str, object]]) -> None:
        primitive = _first_dispatch_end(rows)
        primitive["comm_type"] = "ep-alltoall"
        primitive["group_size"] = 8

    _mutate_rank_zero(trace_root, change_metadata)

    failures = contract.validate_tp2_ep4_flex_deepep(trace_root)

    metadata_failures = [
        failure for failure in failures if failure.code == "trace.moe_flex_deepep.metadata"
    ]
    assert len(metadata_failures) == 2


def test_flex_deepep_contract_rejects_async_lifecycle_events(tmp_path: Path) -> None:
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

    failures = contract.validate_tp2_ep4_flex_deepep(trace_root)

    assert "trace.moe_flex_deepep.lifecycle" in {
        failure.code for failure in failures
    }


def test_flex_deepep_contract_rejects_async_fields_on_fused_primitive(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)
    _mutate_rank_zero(
        trace_root,
        lambda rows: _first_dispatch_end(rows).__setitem__("operation_id", "a2a:0"),
    )

    failures = contract.validate_tp2_ep4_flex_deepep(trace_root)

    assert "trace.moe_flex_deepep.lifecycle" in {
        failure.code for failure in failures
    }


def test_flex_deepep_contract_requires_the_model_tp_peer_group(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def change_tp_peer(rows: list[dict[str, object]]) -> None:
        event = next(
            row
            for row in rows
            if row.get("name") == "tp-reduce-scatter"
            and row.get("ph") == "E"
            and row.get("iteration") == 1
        )
        event["group"] = [2]

    _mutate_rank_zero(trace_root, change_tp_peer)

    failures = contract.validate_tp2_ep4_flex_deepep(trace_root)

    assert "trace.tp_ep.collective_group" in {
        failure.code for failure in failures
    }


def test_flex_deepep_contract_requires_fused_ep_inside_the_moe_scope(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_profile(trace_root)

    def remove_moe_dispatch_begin(rows: list[dict[str, object]]) -> None:
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("name") == "moe-dispatch"
            and row.get("ph") == "B"
            and row.get("iteration") == 1
        )
        rows.pop(index)

    _mutate_rank_zero(trace_root, remove_moe_dispatch_begin)

    failures = contract.validate_tp2_ep4_flex_deepep(trace_root)

    assert "trace.moe_flex_deepep.scope_order" in {
        failure.code for failure in failures
    }
