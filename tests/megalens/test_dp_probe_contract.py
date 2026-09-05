# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_utils.runners import dp_probe_contract


def _write_trace(
    trace_root: Path,
    *,
    rank: int,
    distopt: bool,
    layerwise: bool = False,
    multi_instance: bool = False,
    duplicate_id: bool = False,
    omit_param_completion: bool = False,
    wrong_grad_completion_kind: bool = False,
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
                "dp_rk": rank,
                "pp_rk": 0,
                "tp_rk": 0,
                **attrs,
            }
        )

    def dispatch(name: str, operation_id: str, *, async_op: bool = True, **route: object) -> None:
        event(
            name,
            "B",
            api_async_op=async_op,
            async_op=async_op,
            completion_included=False,
            data_bytes=32768,
            group=[1 - rank],
            group_size=2,
            n_buckets=1,
            operation_id=operation_id,
            operation_id_scope="rank_local",
            overlap_enabled=True,
            timing_phase="async_dispatch" if async_op else "collective_call",
            **route,
        )
        event(name, "E")

    grad_id = f"dp:grad:{rank}"
    param_id = grad_id if duplicate_id else f"dp:param:{rank}"
    rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": 1})
    grad_operations: list[dict[str, str]] = []
    if distopt:
        dispatch(
            "dp-reduce-scatter",
            grad_id,
            async_op=not multi_instance,
            op="reduce_scatter",
            group_role="intra_optimizer_instance",
            payload_role="gradient_bucket",
            stage="intra_instance_reduce_scatter",
        )
        grad_operations.append(
            {
                "event_name": "dp-reduce-scatter",
                "operation_id": grad_id,
                "stage": "intra_instance_reduce_scatter",
            }
        )
        if multi_instance:
            inter_id = f"dp:inter:{rank}"
            dispatch(
                "dp-allreduce",
                inter_id,
                async_op=False,
                op="all_reduce",
                group_role="inter_optimizer_instance",
                payload_role="gradient_shard",
                stage="inter_instance_shard_allreduce",
            )
            grad_operations.append(
                {
                    "event_name": "dp-allreduce",
                    "operation_id": inter_id,
                    "stage": "inter_instance_shard_allreduce",
                }
            )
        dispatch(
            "dp-param-all-gather",
            param_id,
            op="all_gather",
            group_role="intra_optimizer_instance",
            optimizer_kind="distributed",
            payload_role="parameter_bucket",
            stage="distributed_optimizer_param_allgather",
        )
    else:
        dispatch(
            "dp-allreduce",
            grad_id,
            op="all_reduce",
            group_role="data_parallel",
            payload_role="gradient_bucket",
            stage="main_bucket_allreduce",
        )
        grad_operations.append(
            {
                "event_name": "dp-allreduce",
                "operation_id": grad_id,
                "stage": "main_bucket_allreduce",
            }
        )
        if layerwise:
            dispatch(
                "dp-param-all-gather",
                param_id,
                op="all_gather",
                group_role="intra_optimizer_instance",
                optimizer_kind="layerwise",
                payload_role="parameter_bucket",
                stage="layerwise_optimizer_param_allgather",
            )
    rows.append({"name": "iteration", "ph": "E", "iteration": 1, "duration_wall": timestamp})

    rows.append({"name": "iteration", "ph": "B", "pad_before": 0, "iteration": 2})
    stream_join = multi_instance and not wrong_grad_completion_kind
    event(
        "dp-grad-sync-complete",
        "B",
        completion_guarantee=(
            "current_stream_after_join" if stream_join else "current_stream_after_wait"
        ),
        completion_included=True,
        completion_kind="stream_join" if stream_join else "work_wait",
        completion_site="finish_grad_sync",
        force_all_reduce=False,
        host_blocking_guaranteed=False,
        launch_observed=True,
        num_distributed_optimizer_instances=2 if multi_instance else 1,
        op="wait_stream" if stream_join else "wait",
        operation_count=len(grad_operations),
        operation_ids=[operation["operation_id"] for operation in grad_operations],
        operation_id_scope="rank_local",
        operations=grad_operations,
        stage="gradient_collective_completion",
        timing_phase="stream_dependency",
        use_distributed_optimizer=distopt,
    )
    event("dp-grad-sync-complete", "E", completed=True, error_type=None)
    if (distopt or layerwise) and not omit_param_completion:
        event(
            "dp-param-sync-complete",
            "B",
            completion_guarantee="current_stream_after_wait",
            completion_included=True,
            completion_kind="work_wait",
            completion_site="finish_param_sync",
            host_blocking_guaranteed=False,
            launch_observed=True,
            op="wait",
            operation_count=1,
            operation_id=param_id,
            operation_ids=[param_id],
            operation_id_scope="rank_local",
            stage="parameter_allgather_completion",
            timing_phase="stream_dependency",
        )
        event("dp-param-sync-complete", "E", completed=True, error_type=None)
    rows.append({"name": "iteration", "ph": "E", "iteration": 2, "duration_wall": timestamp})

    trace_root.mkdir(parents=True, exist_ok=True)
    path = trace_root / f"benchmark-global-{rank}-data-{rank}-pipeline-0-tensor-0.json"
    path.write_text(json.dumps(rows), encoding="utf-8")


@pytest.mark.parametrize(
    ("distopt", "layerwise", "multi_instance", "rank_count", "validator"),
    (
        (False, False, False, 2, dp_probe_contract.validate_dp_standard_overlap),
        (True, False, False, 2, dp_probe_contract.validate_dp_distopt_overlap),
        (False, True, False, 2, dp_probe_contract.validate_dp_layerwise_overlap),
        (True, False, True, 4, dp_probe_contract.validate_dp_multi_instance_distopt_overlap),
    ),
)
def test_dp_overlap_contract_accepts_cross_iteration_lifecycle(
    tmp_path: Path, distopt: bool, layerwise: bool, multi_instance: bool, rank_count: int, validator
) -> None:
    for rank in range(rank_count):
        _write_trace(
            tmp_path, rank=rank, distopt=distopt, layerwise=layerwise, multi_instance=multi_instance
        )

    assert validator(tmp_path) == ()


def test_dp_distopt_contract_rejects_duplicate_operation_identity(tmp_path: Path) -> None:
    _write_trace(tmp_path, rank=0, distopt=True, duplicate_id=True)

    failures = dp_probe_contract.validate_dp_distopt_overlap(tmp_path)

    assert "trace.dp.operation_id" in {failure.code for failure in failures}


def test_dp_distopt_contract_rejects_missing_parameter_completion(tmp_path: Path) -> None:
    _write_trace(tmp_path, rank=0, distopt=True, omit_param_completion=True)

    failures = dp_probe_contract.validate_dp_distopt_overlap(tmp_path)

    assert {failure.code for failure in failures} == {
        "trace.dp.event_count",
        "trace.dp.operation_id",
    }


def test_dp_multi_instance_contract_requires_stream_join(tmp_path: Path) -> None:
    _write_trace(
        tmp_path, rank=0, distopt=True, multi_instance=True, wrong_grad_completion_kind=True
    )

    failures = dp_probe_contract.validate_dp_multi_instance_distopt_overlap(tmp_path)

    assert "trace.dp.field" in {failure.code for failure in failures}


def test_dp_layerwise_contract_rejects_distopt_route(tmp_path: Path) -> None:
    _write_trace(tmp_path, rank=0, distopt=True)

    failures = dp_probe_contract.validate_dp_layerwise_overlap(tmp_path)

    assert {"trace.dp.route", "trace.dp.field"} <= {failure.code for failure in failures}
