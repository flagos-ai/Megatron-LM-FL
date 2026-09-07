# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Fail-closed metrics for cataloged nested trace spans.

The reducer operates on one rank and iteration at a time.  It uses the event
catalog for role selection, keeps physical-leaf payload separate from logical
composite spans, and never imports the runtime tracer or Megatron Core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from megatron.megalens.event_catalog import (
    CapabilityStatus,
    EventRole,
    EventSpec,
    MetricKind,
    get_event_spec,
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


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One metric value together with its evidence status and reason."""

    value: int | None
    status: CapabilityStatus
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, CapabilityStatus):
            raise TypeError("metric status must be a CapabilityStatus")
        if (
            not isinstance(self.reason, str)
            or not self.reason.strip()
            or self.reason != self.reason.strip()
        ):
            raise ValueError("metric reason must be a non-empty, trimmed string")
        if self.value is not None and (type(self.value) is not int or self.value < 0):
            raise ValueError("metric value must be a non-negative integer or None")
        if self.status is CapabilityStatus.AVAILABLE and self.value is None:
            raise ValueError("available metrics require a value")
        if self.status in (CapabilityStatus.UNAVAILABLE, CapabilityStatus.UNKNOWN):
            if self.value is not None:
                raise ValueError(f"{self.status.value} metrics cannot expose a value")


@dataclass(frozen=True, slots=True)
class TPReduceScatterMetrics:
    """De-duplicated TP reduce-scatter results for one span partition."""

    phase_wall_us: MetricResult
    leaf_sum_us: MetricResult
    interval_union_us: MetricResult
    leaf_data_bytes: MetricResult

    def __post_init__(self) -> None:
        for result in (
            self.phase_wall_us,
            self.leaf_sum_us,
            self.interval_union_us,
            self.leaf_data_bytes,
        ):
            if not isinstance(result, MetricResult):
                raise TypeError("nested metric fields must be MetricResult values")


def _uniform_result(status: CapabilityStatus, reason: str) -> TPReduceScatterMetrics:
    result = MetricResult(value=None, status=status, reason=reason)
    return TPReduceScatterMetrics(
        phase_wall_us=result, leaf_sum_us=result, interval_union_us=result, leaf_data_bytes=result
    )


def _validate_partition(events: Sequence[SpanRecord]) -> str | None:
    if any(
        type(event.ts) is not int or event.ts < 0 or type(event.dur) is not int or event.dur < 0
        for event in events
    ):
        return "cataloged spans require non-negative integer timestamps and durations"
    if any(type(event.rank) is not int or event.rank < 0 for event in events):
        return "cataloged spans require a non-negative integer rank"
    if any(not isinstance(event.args, Mapping) for event in events):
        return "cataloged spans require mapping-like args"

    iterations = [event.args.get("iteration") for event in events]
    if any(type(iteration) is not int or iteration < 0 for iteration in iterations):
        return "cataloged spans require a non-negative integer iteration"

    partitions = {(event.rank, iteration) for event, iteration in zip(events, iterations)}
    if len(partitions) > 1:
        return "cataloged spans must belong to one rank and iteration"
    return None


def _contains(parent: SpanRecord, child: SpanRecord) -> bool:
    return parent.ts <= child.ts and child.ts + child.dur <= parent.ts + parent.dur


def _overlaps(left: SpanRecord, right: SpanRecord) -> bool:
    return max(left.ts, right.ts) < min(left.ts + left.dur, right.ts + right.dur)


def _interval_union_us(events: Sequence[SpanRecord]) -> int:
    intervals = sorted((event.ts, event.ts + event.dur) for event in events)
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _overlapping_composite_reason(composites: Sequence[SpanRecord]) -> str | None:
    for index, parent in enumerate(composites):
        if any(_overlaps(parent, other) for other in composites[index + 1 :]):
            return "logical composite spans overlap without thread identity"
    return None


def _containment_assessment(
    composites: Sequence[SpanRecord], leaves: Sequence[SpanRecord]
) -> tuple[CapabilityStatus, str]:
    composite_overlap = _overlapping_composite_reason(composites)
    if composite_overlap is not None:
        return CapabilityStatus.UNKNOWN, composite_overlap

    child_indices: list[list[int]] = [[] for _ in composites]
    parent_counts = [0] * len(leaves)
    for parent_index, parent in enumerate(composites):
        for leaf_index, leaf in enumerate(leaves):
            if _contains(parent, leaf):
                child_indices[parent_index].append(leaf_index)
                parent_counts[leaf_index] += 1
            elif _overlaps(parent, leaf):
                return (
                    CapabilityStatus.UNKNOWN,
                    "a composite and physical leaf overlap without containment",
                )

    multiple_children = sum(len(indices) > 1 for indices in child_indices)
    shared_children = sum(count > 1 for count in parent_counts)
    if multiple_children or shared_children:
        return (
            CapabilityStatus.UNKNOWN,
            "nested containment is ambiguous: expected one distinct leaf per composite",
        )

    missing_children = sum(not indices for indices in child_indices)
    if missing_children:
        return (
            CapabilityStatus.PARTIAL,
            "one or more composite spans have no observed physical leaf",
        )

    pairs = [(parent, leaves[indices[0]]) for parent, indices in zip(composites, child_indices)]
    for key in ("group_size", "group"):
        for parent, leaf in pairs:
            parent_value = parent.args.get(key)
            leaf_value = leaf.args.get(key)
            if parent_value is not None and leaf_value is not None and parent_value != leaf_value:
                return (CapabilityStatus.UNKNOWN, f"nested parent and leaf disagree on known {key}")

    return CapabilityStatus.AVAILABLE, "every composite has one distinct physical leaf"


def _metadata_conflict_reason(events: Sequence[SpanRecord], *, expected_dim: str) -> str | None:
    for key, expected in (("op", "reduce-scatter"), ("dim", expected_dim)):
        conflicting_names = sorted(
            {event.name for event in events if key in event.args and event.args[key] != expected}
        )
        if conflicting_names:
            names = ", ".join(conflicting_names)
            return f"{names} metadata conflicts with expected {key}={expected}"
    return None


def _payload_result(
    leaves: Sequence[SpanRecord],
    structure_status: CapabilityStatus,
    structure_reason: str,
    leaf_metadata_conflict: str | None,
) -> MetricResult:
    if leaf_metadata_conflict is not None:
        return MetricResult(
            value=None, status=CapabilityStatus.UNKNOWN, reason=leaf_metadata_conflict
        )
    if structure_status is not CapabilityStatus.AVAILABLE:
        return MetricResult(value=None, status=structure_status, reason=structure_reason)

    values: list[int] = []
    for leaf in leaves:
        value: Any = leaf.args.get("data_bytes")
        if type(value) is not int or value < 0:
            return MetricResult(
                value=None,
                status=CapabilityStatus.PARTIAL,
                reason="one or more physical leaves lack a non-negative data_bytes value",
            )
        values.append(value)
    return MetricResult(
        value=sum(values),
        status=CapabilityStatus.AVAILABLE,
        reason="summed data_bytes from each physical leaf once",
    )


def aggregate_tp_reduce_scatter_partition(events: Sequence[SpanRecord]) -> TPReduceScatterMetrics:
    """Aggregate TP reduce-scatter spans from one rank and iteration.

    Only the two exact catalog entries for the physical leaf and logical
    last-dimension composite are consumed; other catalog entries and unknown
    names are ignored. Invalid or cross-partition consumed spans return
    ``unknown`` metrics without values. A composite is expected to contain
    exactly one distinct physical leaf. Partial interval values preserve the
    union of role-valid observed intervals while marking incomplete evidence.
    """

    cataloged: list[tuple[SpanRecord, EventSpec]] = []
    for event in events:
        spec = get_event_spec(event.name)
        if spec is not None and spec.name in ("tp-reduce-scatter", "tp-reduce-scatter-last"):
            cataloged.append((event, spec))

    if not cataloged:
        return _uniform_result(
            CapabilityStatus.UNAVAILABLE, "no cataloged nested spans were observed"
        )

    cataloged_events = [event for event, _ in cataloged]
    invalid_reason = _validate_partition(cataloged_events)
    if invalid_reason is not None:
        return _uniform_result(CapabilityStatus.UNKNOWN, invalid_reason)

    composites = [
        event
        for event, spec in cataloged
        if spec.role is EventRole.LOGICAL_COMPOSITE and MetricKind.PHASE_WALL in spec.metrics
    ]
    leaves = [
        event
        for event, spec in cataloged
        if spec.role is EventRole.PHYSICAL_LEAF and MetricKind.LEAF_SUM in spec.metrics
    ]
    union_events = [event for event, spec in cataloged if MetricKind.INTERVAL_UNION in spec.metrics]

    composite_overlap = _overlapping_composite_reason(composites)
    composite_metadata_conflict = _metadata_conflict_reason(composites, expected_dim="last")
    leaf_metadata_conflict = _metadata_conflict_reason(leaves, expected_dim="first")

    if composites:
        if composite_overlap is not None:
            phase = MetricResult(
                value=None, status=CapabilityStatus.UNKNOWN, reason=composite_overlap
            )
        elif composite_metadata_conflict is not None:
            phase = MetricResult(
                value=None, status=CapabilityStatus.UNKNOWN, reason=composite_metadata_conflict
            )
        elif any(event.dur == 0 for event in composites):
            phase = MetricResult(
                value=None,
                status=CapabilityStatus.PARTIAL,
                reason="one or more composite durations were quantized to zero microseconds",
            )
        else:
            phase = MetricResult(
                value=sum(event.dur for event in composites),
                status=CapabilityStatus.AVAILABLE,
                reason="summed logical composite wall spans",
            )
    else:
        phase = MetricResult(
            value=None,
            status=CapabilityStatus.UNAVAILABLE,
            reason="no logical composite spans were observed",
        )

    structure_status, structure_reason = _containment_assessment(composites, leaves)
    if leaf_metadata_conflict is not None:
        leaf_sum = MetricResult(
            value=None, status=CapabilityStatus.UNKNOWN, reason=leaf_metadata_conflict
        )
    elif structure_status is CapabilityStatus.AVAILABLE and any(event.dur == 0 for event in leaves):
        leaf_sum = MetricResult(
            value=None,
            status=CapabilityStatus.PARTIAL,
            reason="one or more leaf durations were quantized to zero microseconds",
        )
    elif structure_status is CapabilityStatus.AVAILABLE:
        leaf_sum = MetricResult(
            value=sum(event.dur for event in leaves),
            status=CapabilityStatus.AVAILABLE,
            reason="summed each physical leaf duration once",
        )
    else:
        leaf_sum = MetricResult(value=None, status=structure_status, reason=structure_reason)

    if union_events:
        has_zero_duration = any(event.dur == 0 for event in union_events)
        metadata_conflicts = [
            reason
            for reason in (composite_metadata_conflict, leaf_metadata_conflict)
            if reason is not None
        ]
        if metadata_conflicts:
            interval_union = MetricResult(
                value=None, status=CapabilityStatus.UNKNOWN, reason="; ".join(metadata_conflicts)
            )
        else:
            union_issues: list[str] = []
            if structure_status is not CapabilityStatus.AVAILABLE:
                union_issues.append(structure_reason)
            if has_zero_duration:
                union_issues.append("one or more durations were quantized to zero microseconds")
            union_status = CapabilityStatus.PARTIAL if union_issues else CapabilityStatus.AVAILABLE
            union_reason = "merged all TP reduce-scatter intervals"
            if union_issues:
                union_reason = "merged role-valid observed intervals: " + "; ".join(union_issues)
            interval_union = MetricResult(
                value=_interval_union_us(union_events), status=union_status, reason=union_reason
            )
    else:
        interval_union = MetricResult(
            value=None,
            status=CapabilityStatus.UNAVAILABLE,
            reason="no interval-union eligible spans were observed",
        )

    return TPReduceScatterMetrics(
        phase_wall_us=phase,
        leaf_sum_us=leaf_sum,
        interval_union_us=interval_union,
        leaf_data_bytes=_payload_result(
            leaves, structure_status, structure_reason, leaf_metadata_conflict
        ),
    )


__all__ = [
    "MetricResult",
    "SpanRecord",
    "TPReduceScatterMetrics",
    "aggregate_tp_reduce_scatter_partition",
]
