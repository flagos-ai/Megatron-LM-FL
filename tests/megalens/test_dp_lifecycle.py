# Copyright (c) 2026, MegaLens Authors. All rights reserved.

from __future__ import annotations

import importlib.abc
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from megatron.megalens.data_loader import SpanEvent
from megatron.megalens.dp_lifecycle import (
    DPOperationState,
    aggregate_dp_lifecycle_partition,
)
from megatron.megalens.event_catalog import CapabilityStatus


_OPS = {
    "dp-param-all-gather": "all_gather",
    "dp-reduce-scatter": "reduce_scatter",
    "dp-allreduce": "all_reduce",
}


def _span(
    name: str,
    ts: int,
    dur: int,
    *,
    rank: int = 0,
    iteration: int = 7,
    **args: Any,
) -> SpanEvent:
    return SpanEvent(
        name=name,
        ts=ts,
        dur=dur,
        rank=rank,
        args={"iteration": iteration, **args},
    )


def _dispatch(
    name: str,
    operation_id: str,
    *,
    ts: int = 100,
    dur: int = 10,
    async_op: bool,
    overlap_enabled: bool,
    stage: str,
    **overrides: Any,
) -> SpanEvent:
    args: dict[str, Any] = {
        "api_async_op": async_op,
        "async_op": async_op,
        "completion_included": False,
        "op": _OPS[name],
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "overlap_enabled": overlap_enabled,
        "stage": stage,
        "timing_phase": "async_dispatch" if async_op else "collective_call",
    }
    args.update(overrides)
    return _span(name, ts, dur, **args)


def _param_completion(
    operation_id: str | None,
    *,
    ts: int = 200,
    dur: int = 20,
    completed: bool = True,
    **overrides: Any,
) -> SpanEvent:
    operation_ids = [] if operation_id is None else [operation_id]
    args: dict[str, Any] = {
        "completed": completed,
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "work_wait",
        "completion_site": "finish_param_sync",
        "host_blocking_guaranteed": False,
        "launch_observed": bool(operation_ids),
        "op": "wait",
        "operation_count": len(operation_ids),
        "operation_id": operation_id,
        "operation_ids": operation_ids,
        "operation_id_scope": "rank_local",
        "stage": "parameter_allgather_completion",
        "timing_phase": "stream_dependency",
    }
    if not completed:
        args["error_type"] = "RuntimeError"
    args.update(overrides)
    return _span("dp-param-sync-complete", ts, dur, **args)


def _grad_completion(
    operations: list[tuple[str, str, str]],
    *,
    ts: int = 200,
    dur: int = 20,
    kind: str = "work_wait",
    completed: bool = True,
    **overrides: Any,
) -> SpanEvent:
    operation_ids = [operation_id for _, operation_id, _ in operations]
    guarantee = (
        "current_stream_after_join" if kind == "stream_join" else "current_stream_after_wait"
    )
    args: dict[str, Any] = {
        "completed": completed,
        "completion_guarantee": guarantee,
        "completion_included": True,
        "completion_kind": kind,
        "completion_site": "finish_grad_sync",
        "host_blocking_guaranteed": False,
        "launch_observed": bool(operation_ids),
        "op": "wait_stream" if kind == "stream_join" else "wait",
        "operation_count": len(operation_ids),
        "operation_ids": operation_ids,
        "operation_id_scope": "rank_local",
        "operations": [
            {"event_name": name, "operation_id": operation_id, "stage": stage}
            for name, operation_id, stage in operations
        ],
        "stage": "gradient_collective_completion",
        "timing_phase": "stream_dependency",
    }
    if not completed:
        args["error_type"] = "RuntimeError"
    args.update(overrides)
    return _span("dp-grad-sync-complete", ts, dur, **args)


def test_empty_and_unrelated_inputs_are_unavailable() -> None:
    for events in ([], [_span("tp-linear-async-launch", 0, 1)]):
        result = aggregate_dp_lifecycle_partition(events)
        assert result.status is CapabilityStatus.UNAVAILABLE
        assert result.dispatch_attempt_count.value is None
        assert result.operations == ()


def test_param_dispatch_and_successful_wait_establish_current_stream_guarantee() -> None:
    operation_id = "dp:param-all-gather:1"
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-param-all-gather",
                operation_id,
                async_op=True,
                overlap_enabled=True,
                stage="distributed_optimizer_param_allgather",
            ),
            _param_completion(operation_id),
        ]
    )

    assert result.status is CapabilityStatus.AVAILABLE
    assert result.dispatch_attempt_count.value == 1
    assert result.completion_attempt_count.value == 1
    assert result.current_stream_guaranteed_operation_count.value == 1
    assert result.pending_operation_count.value == 0
    assert result.failed_completion_attempt_count.value == 0
    assert result.orphan_completion_count.value == 0
    assert result.exposed_dependency_union_us.value == 20
    assert result.operations[0].state is DPOperationState.CURRENT_STREAM_GUARANTEED
    assert result.operations[0].completion_guarantee == "current_stream_after_wait"
    assert result.boundaries[0].guarantee_established is True
    assert result.boundaries[0].duration_attribution == "single_operation_boundary"
    json.dumps(asdict(result))


def test_synchronous_param_dispatch_does_not_expect_a_completion() -> None:
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-param-all-gather",
                "dp:param-all-gather:sync",
                async_op=False,
                overlap_enabled=True,
                stage="distributed_optimizer_param_allgather",
            )
        ]
    )

    assert result.status is CapabilityStatus.AVAILABLE
    assert result.pending_operation_count.value == 0
    assert result.exposed_dependency_union_us.value == 0
    assert result.operations[0].completion_expected is False
    assert result.operations[0].state is DPOperationState.COMPLETION_NOT_EXPECTED


def test_multi_instance_stream_join_covers_two_sync_api_dispatches_once() -> None:
    rs_id = "dp:reduce-scatter:3"
    ar_id = "dp:inter-instance-allreduce:4"
    operations = [
        ("dp-reduce-scatter", rs_id, "intra_instance_reduce_scatter"),
        ("dp-allreduce", ar_id, "inter_instance_shard_allreduce"),
    ]
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-reduce-scatter",
                rs_id,
                async_op=False,
                overlap_enabled=True,
                stage="intra_instance_reduce_scatter",
            ),
            _dispatch(
                "dp-allreduce",
                ar_id,
                ts=120,
                async_op=False,
                overlap_enabled=True,
                stage="inter_instance_shard_allreduce",
            ),
            _grad_completion(operations, ts=200, dur=50, kind="stream_join"),
        ]
    )

    assert result.status is CapabilityStatus.AVAILABLE
    assert result.dispatch_attempt_count.value == 2
    assert result.current_stream_guaranteed_operation_count.value == 2
    assert result.exposed_dependency_union_us.value == 50
    assert {row.state for row in result.operations} == {
        DPOperationState.CURRENT_STREAM_GUARANTEED
    }
    assert result.boundaries[0].duration_attribution == "shared_nonexclusive_boundary"


def test_zero_id_and_unobserved_id_completions_remain_partial_orphans() -> None:
    zero_id = aggregate_dp_lifecycle_partition([_param_completion(None)])
    assert zero_id.status is CapabilityStatus.PARTIAL
    assert zero_id.completion_attempt_count.value == 1
    assert zero_id.orphan_completion_count.value == 1
    assert zero_id.exposed_dependency_union_us.value == 20
    assert zero_id.operations == ()
    assert zero_id.boundaries[0].status is CapabilityStatus.PARTIAL

    unobserved_id = aggregate_dp_lifecycle_partition(
        [_param_completion("dp:param-all-gather:outside-window")]
    )
    assert unobserved_id.status is CapabilityStatus.PARTIAL
    assert unobserved_id.orphan_completion_count.value == 1
    assert "unobserved dispatch" in unobserved_id.reason


def test_failed_wait_can_retry_with_the_same_identity_then_succeed() -> None:
    operation_id = "dp:allreduce:retry"
    operation = ("dp-allreduce", operation_id, "main_bucket_allreduce")
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-allreduce",
                operation_id,
                async_op=True,
                overlap_enabled=True,
                stage="main_bucket_allreduce",
            ),
            _grad_completion([operation], ts=200, dur=30, completed=False),
            _grad_completion([operation], ts=300, dur=7, completed=True),
        ]
    )

    assert result.status is CapabilityStatus.AVAILABLE
    assert result.completion_attempt_count.value == 2
    assert result.failed_completion_attempt_count.value == 1
    assert result.current_stream_guaranteed_operation_count.value == 1
    assert result.pending_operation_count.value == 0
    assert result.exposed_dependency_union_us.value == 7
    assert result.operations[0].completion_attempt_count == 2
    assert result.operations[0].state is DPOperationState.CURRENT_STREAM_GUARANTEED
    assert [boundary.guarantee_established for boundary in result.boundaries] == [False, True]


def test_failed_wait_without_retry_is_a_known_retryable_outcome() -> None:
    operation_id = "dp:allreduce:failed"
    operation = ("dp-allreduce", operation_id, "main_bucket_allreduce")
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-allreduce",
                operation_id,
                async_op=True,
                overlap_enabled=True,
                stage="main_bucket_allreduce",
            ),
            _grad_completion([operation], completed=False),
        ]
    )

    assert result.status is CapabilityStatus.AVAILABLE
    assert result.failed_completion_attempt_count.value == 1
    assert result.pending_operation_count.value == 1
    assert result.exposed_dependency_union_us.value == 0
    assert result.operations[0].state is DPOperationState.RETRYABLE_FAILURE
    assert result.operations[0].completion_guarantee is None


def test_expected_launch_without_completion_is_partial_and_pending() -> None:
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-reduce-scatter",
                "dp:reduce-scatter:pending",
                async_op=True,
                overlap_enabled=True,
                stage="intra_instance_reduce_scatter",
            )
        ]
    )

    assert result.status is CapabilityStatus.PARTIAL
    assert result.pending_operation_count.value == 1
    assert result.operations[0].state is DPOperationState.PENDING_COMPLETION


def test_success_followed_by_another_completion_fails_closed() -> None:
    operation_id = "dp:allreduce:duplicate-success"
    operation = ("dp-allreduce", operation_id, "main_bucket_allreduce")
    result = aggregate_dp_lifecycle_partition(
        [
            _dispatch(
                "dp-allreduce",
                operation_id,
                async_op=True,
                overlap_enabled=True,
                stage="main_bucket_allreduce",
            ),
            _grad_completion([operation], ts=200),
            _grad_completion([operation], ts=300),
        ]
    )

    assert result.status is CapabilityStatus.UNKNOWN
    assert result.current_stream_guaranteed_operation_count.value is None
    assert result.operations == ()


@pytest.mark.parametrize(
    "completion",
    [
        _param_completion("dp:param-all-gather:bad-count", operation_count=2),
        _param_completion(
            "dp:param-all-gather:bad-guarantee",
            completion_guarantee="current_stream_after_join",
        ),
        _param_completion(
            "dp:param-all-gather:bad-scope", operation_id_scope="global"
        ),
        _param_completion(
            "dp:param-all-gather:bad-bool", completion_included=1
        ),
    ],
)
def test_present_completion_metadata_contradictions_fail_closed(
    completion: SpanEvent,
) -> None:
    result = aggregate_dp_lifecycle_partition([completion])

    assert result.status is CapabilityStatus.UNKNOWN
    assert result.completion_attempt_count.value is None


def test_duplicate_dispatch_identity_fails_closed() -> None:
    operation_id = "dp:allreduce:duplicate"
    launch = _dispatch(
        "dp-allreduce",
        operation_id,
        async_op=True,
        overlap_enabled=True,
        stage="main_bucket_allreduce",
    )
    result = aggregate_dp_lifecycle_partition([launch, launch])

    assert result.status is CapabilityStatus.UNKNOWN
    assert "multiple dispatch" in result.reason


def test_missing_legacy_typed_fields_degrade_to_partial_evidence() -> None:
    result = aggregate_dp_lifecycle_partition([_span("dp-allreduce", 100, 10)])

    assert result.status is CapabilityStatus.PARTIAL
    assert result.dispatch_attempt_count.value == 1
    assert result.operations == ()
    assert "lacks typed field" in result.reason


def test_cross_partition_input_fails_closed() -> None:
    result = aggregate_dp_lifecycle_partition(
        [
            _span("dp-allreduce", 100, 10, rank=0),
            _span("dp-grad-sync-complete", 200, 10, rank=1),
        ]
    )

    assert result.status is CapabilityStatus.UNKNOWN
    assert "one rank and iteration" in result.reason


def test_reducer_import_does_not_load_runtime_or_reporting_dependencies() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(repository_root)!r})

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname == "pandas" or fullname.startswith("matplotlib"):
            raise AssertionError(f"unexpected heavy import: {{fullname}}")
        if fullname in ("megatron.megalens.runtime", "megatron.megalens.trace"):
            raise AssertionError(f"unexpected runtime import: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockHeavyImports())
from megatron.megalens.dp_lifecycle import aggregate_dp_lifecycle_partition
assert aggregate_dp_lifecycle_partition
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", source], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
