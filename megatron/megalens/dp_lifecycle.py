# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Fail-closed offline reduction for DP collective lifecycle spans.

The reducer consumes one rank and iteration at a time.  Dispatch spans count
observed collective call attempts; their CUDA-event duration is deliberately
not treated as physical communication time.  Completion spans describe an
existing caller-stream dependency boundary.  A boundary can correlate several
operations, but its interval is merged exactly once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from megatron.megalens.event_catalog import CapabilityStatus, EventRole, get_event_spec
from megatron.megalens.nested_aggregation import MetricResult


_DISPATCH_NAMES = frozenset(("dp-param-all-gather", "dp-reduce-scatter", "dp-allreduce"))
_COMPLETION_NAMES = frozenset(("dp-param-sync-complete", "dp-grad-sync-complete"))

_DISPATCH_OPS = {
    "dp-param-all-gather": "all_gather",
    "dp-reduce-scatter": "reduce_scatter",
    "dp-allreduce": "all_reduce",
}
_DISPATCH_STAGES = {
    "dp-param-all-gather": frozenset(
        ("distributed_optimizer_param_allgather", "layerwise_optimizer_param_allgather")
    ),
    "dp-reduce-scatter": frozenset(("intra_instance_reduce_scatter",)),
    "dp-allreduce": frozenset(
        ("main_bucket_allreduce", "inter_instance_shard_allreduce")
    ),
}
_GRADIENT_STAGES = frozenset(
    (
        "main_bucket_allreduce",
        "intra_instance_reduce_scatter",
        "inter_instance_shard_allreduce",
    )
)


class SpanRecord(Protocol):
    """Minimal offline span shape accepted by the reducer."""

    @property
    def name(self) -> str: ...

    @property
    def ts(self) -> int: ...

    @property
    def dur(self) -> int: ...

    @property
    def rank(self) -> int: ...

    @property
    def args(self) -> Mapping[str, Any]: ...


class DPOperationState(str, Enum):
    """Observed state of one rank-local DP operation identity."""

    COMPLETION_NOT_EXPECTED = "completion-not-expected"
    PENDING_COMPLETION = "pending-completion"
    RETRYABLE_FAILURE = "retryable-failure"
    CURRENT_STREAM_GUARANTEED = "current-stream-guaranteed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DPOperationOutcome:
    """Correlation result for one observed dispatch identity."""

    operation_id: str
    event_name: str
    stage: str | None
    timing_phase: str | None
    async_op: bool | None
    completion_expected: bool | None
    state: DPOperationState
    completion_attempt_count: int
    completion_kind: str | None
    completion_guarantee: str | None
    status: CapabilityStatus
    reason: str


@dataclass(frozen=True, slots=True)
class DPCompletionBoundary:
    """One observed Work.wait or stream-join attempt."""

    event_name: str
    ts: int
    dur: int
    completion_kind: str | None
    operation_ids: tuple[str, ...]
    completed: bool | None
    completion_guarantee: str | None
    guarantee_established: bool
    duration_attribution: str
    status: CapabilityStatus
    reason: str


@dataclass(frozen=True, slots=True)
class DPLifecycleResult:
    """Typed lifecycle metrics for one rank/iteration partition."""

    rank: int | None
    iteration: int | None
    status: CapabilityStatus
    reason: str
    dispatch_attempt_count: MetricResult
    completion_attempt_count: MetricResult
    current_stream_guaranteed_operation_count: MetricResult
    pending_operation_count: MetricResult
    failed_completion_attempt_count: MetricResult
    orphan_completion_count: MetricResult
    exposed_dependency_union_us: MetricResult
    operations: tuple[DPOperationOutcome, ...]
    boundaries: tuple[DPCompletionBoundary, ...]


@dataclass(frozen=True, slots=True)
class _Dispatch:
    operation_id: str
    event_name: str
    stage: str | None
    timing_phase: str | None
    async_op: bool | None
    overlap_enabled: bool | None
    completion_expected: bool | None


@dataclass(frozen=True, slots=True)
class _Boundary:
    index: int
    event_name: str
    ts: int
    dur: int
    completion_kind: str | None
    operation_ids: tuple[str, ...]
    completed: bool | None
    completion_guarantee: str | None
    guarantee_established: bool
    descriptors: Mapping[str, tuple[str, str]]
    issues: tuple[str, ...]


def _reason(prefix: str, messages: Sequence[str]) -> str:
    unique = sorted(set(messages))
    return prefix if not unique else f"{prefix}: {'; '.join(unique)}"


def _uniform_result(status: CapabilityStatus, reason: str) -> DPLifecycleResult:
    metric = MetricResult(value=None, status=status, reason=reason)
    return DPLifecycleResult(
        rank=None,
        iteration=None,
        status=status,
        reason=reason,
        dispatch_attempt_count=metric,
        completion_attempt_count=metric,
        current_stream_guaranteed_operation_count=metric,
        pending_operation_count=metric,
        failed_completion_attempt_count=metric,
        orphan_completion_count=metric,
        exposed_dependency_union_us=metric,
        operations=(),
        boundaries=(),
    )


def _cataloged_dp_events(events: Sequence[SpanRecord]) -> list[SpanRecord]:
    selected = []
    for event in events:
        spec = get_event_spec(event.name)
        if spec is None:
            continue
        if spec.name in _DISPATCH_NAMES and spec.role is EventRole.COLLECTIVE_DISPATCH:
            selected.append(event)
        elif spec.name in _COMPLETION_NAMES and spec.role is EventRole.STREAM_DEPENDENCY:
            selected.append(event)
    return selected


def _validate_partition(events: Sequence[SpanRecord]) -> tuple[int, int] | str:
    if any(
        type(event.ts) is not int or event.ts < 0 or type(event.dur) is not int or event.dur < 0
        for event in events
    ):
        return "cataloged DP lifecycle spans require non-negative integer timestamps and durations"
    if any(type(event.rank) is not int or event.rank < 0 for event in events):
        return "cataloged DP lifecycle spans require a non-negative integer rank"
    if any(not isinstance(event.args, Mapping) for event in events):
        return "cataloged DP lifecycle spans require mapping-like args"

    iterations = [event.args.get("iteration") for event in events]
    if any(type(iteration) is not int or iteration < 0 for iteration in iterations):
        return "cataloged DP lifecycle spans require a non-negative integer iteration"
    partitions = {(event.rank, iteration) for event, iteration in zip(events, iterations)}
    if len(partitions) != 1:
        return "cataloged DP lifecycle spans must belong to one rank and iteration"
    return next(iter(partitions))


def _required_field(
    args: Mapping[str, Any],
    key: str,
    *,
    event_name: str,
    issues: list[str],
) -> Any:
    if key not in args:
        issues.append(f"{event_name} lacks typed field {key}")
        return None
    return args[key]


def _expect_exact(
    args: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    event_name: str,
    issues: list[str],
    conflicts: list[str],
) -> Any:
    value = _required_field(args, key, event_name=event_name, issues=issues)
    type_mismatch = type(expected) is bool and type(value) is not bool
    if key in args and (type_mismatch or value != expected):
        conflicts.append(f"{event_name} has {key}={value!r}, expected {expected!r}")
    return value


def _expect_bool(
    args: Mapping[str, Any],
    key: str,
    *,
    event_name: str,
    issues: list[str],
    conflicts: list[str],
) -> bool | None:
    value = _required_field(args, key, event_name=event_name, issues=issues)
    if key not in args:
        return None
    if type(value) is not bool:
        conflicts.append(f"{event_name} field {key} must be bool")
        return None
    return value


def _expect_trimmed_string(
    args: Mapping[str, Any],
    key: str,
    *,
    event_name: str,
    issues: list[str],
    conflicts: list[str],
) -> str | None:
    value = _required_field(args, key, event_name=event_name, issues=issues)
    if key not in args:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        conflicts.append(f"{event_name} field {key} must be a non-empty trimmed string")
        return None
    return value


def _completion_expected(
    event_name: str,
    *,
    async_op: bool | None,
    overlap_enabled: bool | None,
    stage: str | None,
) -> bool | None:
    if event_name == "dp-param-all-gather":
        return async_op
    if async_op is True:
        return True
    if async_op is False and overlap_enabled is not None and stage is not None:
        return overlap_enabled and stage in _GRADIENT_STAGES
    return None


def _parse_dispatch(
    event: SpanRecord, issues: list[str], conflicts: list[str]
) -> _Dispatch | None:
    args = event.args
    local_issues: list[str] = []
    operation_id = _expect_trimmed_string(
        args,
        "operation_id",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    _expect_exact(
        args,
        "operation_id_scope",
        "rank_local",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    _expect_exact(
        args,
        "completion_included",
        False,
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    async_op = _expect_bool(
        args,
        "async_op",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    api_async_op = _expect_bool(
        args,
        "api_async_op",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    if async_op is not None and api_async_op is not None and async_op != api_async_op:
        conflicts.append(f"{event.name} async_op and api_async_op disagree")

    timing_phase = _expect_trimmed_string(
        args,
        "timing_phase",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    if async_op is not None and timing_phase is not None:
        expected_phase = "async_dispatch" if async_op else "collective_call"
        if timing_phase != expected_phase:
            conflicts.append(
                f"{event.name} has timing_phase={timing_phase!r}, expected {expected_phase!r}"
            )

    stage = _expect_trimmed_string(
        args, "stage", event_name=event.name, issues=local_issues, conflicts=conflicts
    )
    if stage is not None and stage not in _DISPATCH_STAGES[event.name]:
        conflicts.append(f"{event.name} has unsupported stage={stage!r}")
    _expect_exact(
        args,
        "op",
        _DISPATCH_OPS[event.name],
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    overlap_enabled = _expect_bool(
        args,
        "overlap_enabled",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    issues.extend(local_issues)
    if operation_id is None:
        return None
    return _Dispatch(
        operation_id=operation_id,
        event_name=event.name,
        stage=stage,
        timing_phase=timing_phase,
        async_op=async_op,
        overlap_enabled=overlap_enabled,
        completion_expected=_completion_expected(
            event.name, async_op=async_op, overlap_enabled=overlap_enabled, stage=stage
        ),
    )


def _parse_operation_ids(
    args: Mapping[str, Any],
    *,
    event_name: str,
    issues: list[str],
    conflicts: list[str],
) -> tuple[str, ...]:
    raw_ids = _required_field(args, "operation_ids", event_name=event_name, issues=issues)
    if "operation_ids" not in args:
        return ()
    if not isinstance(raw_ids, list):
        conflicts.append(f"{event_name} operation_ids must be a list")
        return ()
    if any(
        not isinstance(operation_id, str)
        or not operation_id.strip()
        or operation_id != operation_id.strip()
        for operation_id in raw_ids
    ):
        conflicts.append(f"{event_name} operation_ids must contain trimmed non-empty strings")
        return ()
    if len(set(raw_ids)) != len(raw_ids):
        conflicts.append(f"{event_name} operation_ids must be unique")
    return tuple(raw_ids)


def _parse_grad_descriptors(
    args: Mapping[str, Any],
    operation_ids: tuple[str, ...],
    *,
    event_name: str,
    issues: list[str],
    conflicts: list[str],
) -> Mapping[str, tuple[str, str]]:
    raw_operations = _required_field(args, "operations", event_name=event_name, issues=issues)
    if "operations" not in args:
        return {}
    if not isinstance(raw_operations, list):
        conflicts.append(f"{event_name} operations must be a list")
        return {}

    descriptor_ids: list[str] = []
    descriptors: dict[str, tuple[str, str]] = {}
    for index, operation in enumerate(raw_operations):
        if not isinstance(operation, Mapping):
            conflicts.append(f"{event_name} operations[{index}] must be a mapping")
            continue
        descriptor_issues: list[str] = []
        operation_id = _expect_trimmed_string(
            operation,
            "operation_id",
            event_name=f"{event_name} operations[{index}]",
            issues=descriptor_issues,
            conflicts=conflicts,
        )
        dispatch_name = _expect_trimmed_string(
            operation,
            "event_name",
            event_name=f"{event_name} operations[{index}]",
            issues=descriptor_issues,
            conflicts=conflicts,
        )
        stage = _expect_trimmed_string(
            operation,
            "stage",
            event_name=f"{event_name} operations[{index}]",
            issues=descriptor_issues,
            conflicts=conflicts,
        )
        issues.extend(descriptor_issues)
        if dispatch_name is not None and dispatch_name not in (
            "dp-reduce-scatter",
            "dp-allreduce",
        ):
            conflicts.append(f"{event_name} references unsupported event_name={dispatch_name!r}")
        if (
            dispatch_name is not None
            and stage is not None
            and stage not in _DISPATCH_STAGES[dispatch_name]
        ):
            conflicts.append(
                f"{event_name} descriptor stage={stage!r} conflicts with {dispatch_name}"
            )
        if operation_id is not None:
            descriptor_ids.append(operation_id)
            if dispatch_name is not None and stage is not None:
                descriptors[operation_id] = (dispatch_name, stage)

    if tuple(descriptor_ids) != operation_ids:
        conflicts.append(f"{event_name} operations identities disagree with operation_ids")
    return descriptors


def _parse_boundary(
    event: SpanRecord, index: int, issues: list[str], conflicts: list[str]
) -> _Boundary:
    args = event.args
    local_issues: list[str] = []
    _expect_exact(
        args,
        "operation_id_scope",
        "rank_local",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    _expect_exact(
        args,
        "completion_included",
        True,
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    _expect_exact(
        args,
        "timing_phase",
        "stream_dependency",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    _expect_exact(
        args,
        "host_blocking_guaranteed",
        False,
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )

    operation_ids = _parse_operation_ids(
        args, event_name=event.name, issues=local_issues, conflicts=conflicts
    )
    operation_count = _required_field(
        args, "operation_count", event_name=event.name, issues=local_issues
    )
    if "operation_count" in args and (
        type(operation_count) is not int or operation_count < 0
    ):
        conflicts.append(f"{event.name} operation_count must be a non-negative integer")
    elif "operation_count" in args and operation_count != len(operation_ids):
        conflicts.append(f"{event.name} operation_count disagrees with operation_ids")

    launch_observed = _expect_bool(
        args,
        "launch_observed",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    if launch_observed is not None and launch_observed != bool(operation_ids):
        conflicts.append(f"{event.name} launch_observed disagrees with operation_ids")

    completion_kind = _expect_trimmed_string(
        args,
        "completion_kind",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    allowed_kinds = (
        frozenset(("work_wait",))
        if event.name == "dp-param-sync-complete"
        else frozenset(("work_wait", "stream_join"))
    )
    if completion_kind is not None and completion_kind not in allowed_kinds:
        conflicts.append(f"{event.name} has unsupported completion_kind={completion_kind!r}")

    completion_guarantee = _expect_trimmed_string(
        args,
        "completion_guarantee",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    expected_guarantee = {
        "work_wait": "current_stream_after_wait",
        "stream_join": "current_stream_after_join",
    }.get(completion_kind)
    if (
        completion_guarantee is not None
        and expected_guarantee is not None
        and completion_guarantee != expected_guarantee
    ):
        conflicts.append(
            f"{event.name} completion guarantee conflicts with {completion_kind}"
        )
    expected_op = {"work_wait": "wait", "stream_join": "wait_stream"}.get(completion_kind)
    if expected_op is not None:
        _expect_exact(
            args,
            "op",
            expected_op,
            event_name=event.name,
            issues=local_issues,
            conflicts=conflicts,
        )
    elif "op" not in args:
        local_issues.append(f"{event.name} lacks typed field op")

    expected_stage = (
        "parameter_allgather_completion"
        if event.name == "dp-param-sync-complete"
        else "gradient_collective_completion"
    )
    _expect_exact(
        args,
        "stage",
        expected_stage,
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )

    completed = _expect_bool(
        args,
        "completed",
        event_name=event.name,
        issues=local_issues,
        conflicts=conflicts,
    )
    error_type = args.get("error_type")
    if "error_type" in args and error_type is not None:
        if (
            not isinstance(error_type, str)
            or not error_type.strip()
            or error_type != error_type.strip()
        ):
            conflicts.append(f"{event.name} error_type must be null or a trimmed string")
        elif completed is True:
            conflicts.append(f"{event.name} completed=True conflicts with error_type")
    if completed is False and ("error_type" not in args or error_type is None):
        local_issues.append(f"{event.name} failed completion lacks error_type")

    descriptors: Mapping[str, tuple[str, str]] = {}
    if event.name == "dp-param-sync-complete":
        scalar_operation_id = _required_field(
            args, "operation_id", event_name=event.name, issues=local_issues
        )
        if "operation_id" in args and scalar_operation_id is not None and (
            not isinstance(scalar_operation_id, str)
            or not scalar_operation_id.strip()
            or scalar_operation_id != scalar_operation_id.strip()
        ):
            conflicts.append(f"{event.name} operation_id must be null or a trimmed string")
        expected_scalar = operation_ids[0] if len(operation_ids) == 1 else None
        if len(operation_ids) > 1:
            conflicts.append(f"{event.name} can reference at most one operation")
        if "operation_id" in args and scalar_operation_id != expected_scalar:
            conflicts.append(f"{event.name} operation_id disagrees with operation_ids")
    else:
        descriptors = _parse_grad_descriptors(
            args,
            operation_ids,
            event_name=event.name,
            issues=local_issues,
            conflicts=conflicts,
        )

    core_keys = (
        "operation_id_scope",
        "completion_included",
        "timing_phase",
        "host_blocking_guaranteed",
        "completion_kind",
        "completion_guarantee",
        "op",
        "stage",
        "completed",
    )
    core_present = all(key in args for key in core_keys)
    guarantee_established = (
        completed is True
        and core_present
        and completion_kind in allowed_kinds
        and completion_guarantee == expected_guarantee
    )
    issues.extend(local_issues)
    return _Boundary(
        index=index,
        event_name=event.name,
        ts=event.ts,
        dur=event.dur,
        completion_kind=completion_kind,
        operation_ids=operation_ids,
        completed=completed,
        completion_guarantee=completion_guarantee,
        guarantee_established=guarantee_established,
        descriptors=descriptors,
        issues=tuple(local_issues),
    )


def _interval_union_us(boundaries: Sequence[_Boundary]) -> int:
    intervals = sorted((boundary.ts, boundary.ts + boundary.dur) for boundary in boundaries)
    if not intervals:
        return 0
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _unknown_result(conflicts: Sequence[str]) -> DPLifecycleResult:
    return _uniform_result(
        CapabilityStatus.UNKNOWN,
        _reason("DP lifecycle metadata is contradictory", conflicts),
    )


def aggregate_dp_lifecycle_partition(events: Sequence[SpanRecord]) -> DPLifecycleResult:
    """Reduce cataloged DP dispatch and completion spans from one partition.

    Missing typed fields are treated as partial legacy evidence.  Present but
    contradictory fields fail the partition closed.  A successful completion
    proves only the declared current-stream dependency; it does not prove host
    blocking, global device completion, or physical collective duration.
    """

    cataloged = _cataloged_dp_events(events)
    if not cataloged:
        return _uniform_result(
            CapabilityStatus.UNAVAILABLE,
            "no cataloged DP lifecycle spans were observed",
        )

    partition = _validate_partition(cataloged)
    if isinstance(partition, str):
        return _uniform_result(CapabilityStatus.UNKNOWN, partition)
    rank, iteration = partition

    issues: list[str] = []
    conflicts: list[str] = []
    dispatches: dict[str, _Dispatch] = {}
    boundaries: list[_Boundary] = []
    dispatch_attempt_count = 0
    completion_attempt_count = 0

    for index, event in enumerate(cataloged):
        if event.name in _DISPATCH_NAMES:
            dispatch_attempt_count += 1
            dispatch = _parse_dispatch(event, issues, conflicts)
            if dispatch is None:
                continue
            previous = dispatches.get(dispatch.operation_id)
            if previous is not None:
                conflicts.append(
                    f"operation_id {dispatch.operation_id!r} is used by multiple dispatch spans"
                )
            else:
                dispatches[dispatch.operation_id] = dispatch
        else:
            completion_attempt_count += 1
            boundaries.append(_parse_boundary(event, index, issues, conflicts))

    if conflicts:
        return _unknown_result(conflicts)

    attempts_by_operation: dict[str, list[_Boundary]] = defaultdict(list)
    boundary_orphans: dict[int, tuple[str, ...]] = {}
    for boundary in boundaries:
        orphan_ids: list[str] = []
        for operation_id in boundary.operation_ids:
            dispatch = dispatches.get(operation_id)
            if dispatch is None:
                orphan_ids.append(operation_id)
                continue
            if boundary.event_name == "dp-param-sync-complete":
                if dispatch.event_name != "dp-param-all-gather":
                    conflicts.append(
                        "parameter completion references "
                        f"{dispatch.event_name} operation {operation_id!r}"
                    )
                    continue
            elif dispatch.event_name not in ("dp-reduce-scatter", "dp-allreduce"):
                conflicts.append(
                    "gradient completion references "
                    f"{dispatch.event_name} operation {operation_id!r}"
                )
                continue

            descriptor = boundary.descriptors.get(operation_id)
            if boundary.event_name == "dp-grad-sync-complete" and descriptor is not None:
                if descriptor != (dispatch.event_name, dispatch.stage):
                    conflicts.append(
                        f"gradient completion descriptor disagrees with dispatch {operation_id!r}"
                    )
                    continue
            attempts_by_operation[operation_id].append(boundary)

        if not boundary.operation_ids:
            issues.append(f"{boundary.event_name} has no observed launch identity")
            boundary_orphans[boundary.index] = ()
        elif orphan_ids:
            issues.append(
                f"{boundary.event_name} references unobserved dispatch IDs "
                f"{', '.join(sorted(orphan_ids))}"
            )
            boundary_orphans[boundary.index] = tuple(sorted(orphan_ids))

    if conflicts:
        return _unknown_result(conflicts)

    outcomes: list[DPOperationOutcome] = []
    current_stream_guaranteed = 0
    pending_operations = 0
    for operation_id, dispatch in sorted(dispatches.items()):
        attempts = sorted(
            attempts_by_operation.get(operation_id, ()), key=lambda boundary: boundary.ts
        )
        successful = [boundary for boundary in attempts if boundary.completed is True]
        if len(successful) > 1:
            conflicts.append(f"operation {operation_id!r} has multiple successful completions")
            continue
        if successful:
            success = successful[0]
            if any(boundary.ts >= success.ts and boundary is not success for boundary in attempts):
                conflicts.append(f"operation {operation_id!r} has a completion at or after success")
                continue

        if dispatch.completion_expected is False:
            if attempts:
                conflicts.append(
                    f"operation {operation_id!r} has a completion although none is expected"
                )
                continue
            outcomes.append(
                DPOperationOutcome(
                    operation_id=operation_id,
                    event_name=dispatch.event_name,
                    stage=dispatch.stage,
                    timing_phase=dispatch.timing_phase,
                    async_op=dispatch.async_op,
                    completion_expected=False,
                    state=DPOperationState.COMPLETION_NOT_EXPECTED,
                    completion_attempt_count=0,
                    completion_kind=None,
                    completion_guarantee=None,
                    status=CapabilityStatus.AVAILABLE,
                    reason="synchronous dispatch does not require a separate completion boundary",
                )
            )
            continue

        if dispatch.completion_expected is None:
            pending_operations += 1
            issues.append(f"operation {operation_id!r} has incomplete dispatch metadata")
            outcomes.append(
                DPOperationOutcome(
                    operation_id=operation_id,
                    event_name=dispatch.event_name,
                    stage=dispatch.stage,
                    timing_phase=dispatch.timing_phase,
                    async_op=dispatch.async_op,
                    completion_expected=None,
                    state=DPOperationState.UNKNOWN,
                    completion_attempt_count=len(attempts),
                    completion_kind=attempts[-1].completion_kind if attempts else None,
                    completion_guarantee=None,
                    status=CapabilityStatus.PARTIAL,
                    reason=(
                        "typed dispatch metadata cannot determine whether completion is expected"
                    ),
                )
            )
            continue

        established = [boundary for boundary in successful if boundary.guarantee_established]
        if established:
            boundary = established[0]
            current_stream_guaranteed += 1
            outcomes.append(
                DPOperationOutcome(
                    operation_id=operation_id,
                    event_name=dispatch.event_name,
                    stage=dispatch.stage,
                    timing_phase=dispatch.timing_phase,
                    async_op=dispatch.async_op,
                    completion_expected=True,
                    state=DPOperationState.CURRENT_STREAM_GUARANTEED,
                    completion_attempt_count=len(attempts),
                    completion_kind=boundary.completion_kind,
                    completion_guarantee=boundary.completion_guarantee,
                    status=CapabilityStatus.AVAILABLE,
                    reason=(
                        "a successful completion established the declared current-stream dependency"
                    ),
                )
            )
        elif any(boundary.completed is False for boundary in attempts):
            pending_operations += 1
            boundary = attempts[-1]
            outcomes.append(
                DPOperationOutcome(
                    operation_id=operation_id,
                    event_name=dispatch.event_name,
                    stage=dispatch.stage,
                    timing_phase=dispatch.timing_phase,
                    async_op=dispatch.async_op,
                    completion_expected=True,
                    state=DPOperationState.RETRYABLE_FAILURE,
                    completion_attempt_count=len(attempts),
                    completion_kind=boundary.completion_kind,
                    completion_guarantee=None,
                    status=CapabilityStatus.AVAILABLE,
                    reason=(
                        "all observed completion attempts failed; "
                        "no current-stream guarantee was established"
                    ),
                )
            )
        else:
            pending_operations += 1
            issues.append(f"operation {operation_id!r} has no successful completion boundary")
            boundary = attempts[-1] if attempts else None
            outcomes.append(
                DPOperationOutcome(
                    operation_id=operation_id,
                    event_name=dispatch.event_name,
                    stage=dispatch.stage,
                    timing_phase=dispatch.timing_phase,
                    async_op=dispatch.async_op,
                    completion_expected=True,
                    state=DPOperationState.PENDING_COMPLETION,
                    completion_attempt_count=len(attempts),
                    completion_kind=boundary.completion_kind if boundary else None,
                    completion_guarantee=None,
                    status=CapabilityStatus.PARTIAL,
                    reason=(
                        "no successful completion with a valid current-stream "
                        "guarantee was observed"
                    ),
                )
            )

    if conflicts:
        return _unknown_result(conflicts)

    successful_boundaries = [boundary for boundary in boundaries if boundary.guarantee_established]
    if any(boundary.dur == 0 for boundary in successful_boundaries):
        issues.append(
            "one or more successful completion durations were quantized to zero microseconds"
        )

    public_boundaries: list[DPCompletionBoundary] = []
    for boundary in sorted(boundaries, key=lambda item: (item.ts, item.index)):
        boundary_issues = list(boundary.issues)
        if boundary.index in boundary_orphans:
            orphan_ids = boundary_orphans[boundary.index]
            boundary_issues.append(
                "completion has no correlated dispatch"
                if not orphan_ids
                else f"completion has unobserved dispatch IDs {', '.join(orphan_ids)}"
            )
        boundary_status = (
            CapabilityStatus.PARTIAL if boundary_issues else CapabilityStatus.AVAILABLE
        )
        if boundary.completion_kind == "stream_join":
            attribution = "shared_nonexclusive_boundary"
        elif len(boundary.operation_ids) == 1 and boundary.index not in boundary_orphans:
            attribution = "single_operation_boundary"
        else:
            attribution = "uncorrelated_boundary"
        public_boundaries.append(
            DPCompletionBoundary(
                event_name=boundary.event_name,
                ts=boundary.ts,
                dur=boundary.dur,
                completion_kind=boundary.completion_kind,
                operation_ids=boundary.operation_ids,
                completed=boundary.completed,
                completion_guarantee=boundary.completion_guarantee,
                guarantee_established=boundary.guarantee_established,
                duration_attribution=attribution,
                status=boundary_status,
                reason=_reason(
                    "completion boundary evidence is partial"
                    if boundary_issues
                    else "completion boundary evidence is available",
                    boundary_issues,
                ),
            )
        )

    status = CapabilityStatus.PARTIAL if issues else CapabilityStatus.AVAILABLE
    result_reason = _reason(
        "DP lifecycle evidence is partial"
        if status is CapabilityStatus.PARTIAL
        else "DP lifecycle evidence is internally consistent",
        issues,
    )

    def metric(value: int) -> MetricResult:
        return MetricResult(value=value, status=status, reason=result_reason)

    return DPLifecycleResult(
        rank=rank,
        iteration=iteration,
        status=status,
        reason=result_reason,
        dispatch_attempt_count=metric(dispatch_attempt_count),
        completion_attempt_count=metric(completion_attempt_count),
        current_stream_guaranteed_operation_count=metric(current_stream_guaranteed),
        pending_operation_count=metric(pending_operations),
        failed_completion_attempt_count=metric(
            sum(boundary.completed is False for boundary in boundaries)
        ),
        orphan_completion_count=metric(len(boundary_orphans)),
        exposed_dependency_union_us=metric(_interval_union_us(successful_boundaries)),
        operations=tuple(outcomes),
        boundaries=tuple(public_boundaries),
    )


__all__ = [
    "DPCompletionBoundary",
    "DPLifecycleResult",
    "DPOperationOutcome",
    "DPOperationState",
    "aggregate_dp_lifecycle_partition",
]
