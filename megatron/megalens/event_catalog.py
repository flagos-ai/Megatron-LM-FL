# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Typed metadata for trace-event aggregation and asynchronous lifecycles.

This module is intentionally offline-only and standard-library-only.  It
describes how consumers may interpret known trace event names; it does not
change producers, load the external audit contract, or compute metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class EventRole(str, Enum):
    """A catalog event's aggregation or asynchronous-lifecycle role."""

    LOGICAL_COMPOSITE = "logical-composite"
    PHYSICAL_LEAF = "physical-leaf"
    ASYNC_LAUNCH = "async-launch"
    ASYNC_WAIT = "async-wait"
    COLLECTIVE_DISPATCH = "collective-dispatch"
    STREAM_DEPENDENCY = "stream-dependency"


class MetricKind(str, Enum):
    """Consumer metric families for which an event is eligible.

    ``LAUNCH_ATTEMPT_COUNT`` and ``DISPATCH_ATTEMPT_COUNT`` deliberately
    ignore the span duration.  MegaLens scopes are timed by CUDA events on the
    selected stream, so that duration is not Python/API wall time or proof of
    physical collective completion.  ``COMPLETION_ATTEMPT_COUNT`` counts
    observed completion boundaries independently of their outcome.
    ``EXPOSED_WAIT`` is the caller-stream interval around an existing
    ``Work.wait()`` attempt.
    ``STREAM_DEPENDENCY`` is the shared caller-stream boundary established
    by an existing ``Work.wait()`` or ``wait_stream()`` call; one boundary
    may correlate several operations and its duration must be counted once.
    """

    PHASE_WALL = "phase_wall_us"
    LEAF_SUM = "leaf_sum_us"
    INTERVAL_UNION = "interval_union_us"
    LAUNCH_ATTEMPT_COUNT = "launch_attempt_count"
    DISPATCH_ATTEMPT_COUNT = "dispatch_attempt_count"
    COMPLETION_ATTEMPT_COUNT = "completion_attempt_count"
    EXPOSED_WAIT = "exposed_wait_us"
    STREAM_DEPENDENCY = "stream_dependency_us"


class CapabilityStatus(str, Enum):
    """Evidence status for an analyzer metric."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


_ALLOWED_METRICS_BY_ROLE: Mapping[EventRole, frozenset[MetricKind]] = MappingProxyType(
    {
        EventRole.LOGICAL_COMPOSITE: frozenset((MetricKind.PHASE_WALL, MetricKind.INTERVAL_UNION)),
        EventRole.PHYSICAL_LEAF: frozenset((MetricKind.LEAF_SUM, MetricKind.INTERVAL_UNION)),
        EventRole.ASYNC_LAUNCH: frozenset((MetricKind.LAUNCH_ATTEMPT_COUNT,)),
        EventRole.ASYNC_WAIT: frozenset((MetricKind.EXPOSED_WAIT,)),
        EventRole.COLLECTIVE_DISPATCH: frozenset((MetricKind.DISPATCH_ATTEMPT_COUNT,)),
        EventRole.STREAM_DEPENDENCY: frozenset(
            (MetricKind.COMPLETION_ATTEMPT_COUNT, MetricKind.STREAM_DEPENDENCY)
        ),
    }
)


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Immutable interpretation metadata for one canonical event name."""

    name: str
    role: EventRole
    metrics: frozenset[MetricKind]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name != self.name.strip()
        ):
            raise ValueError("event name must be a non-empty, trimmed string")
        if not isinstance(self.role, EventRole):
            raise TypeError("event role must be an EventRole")
        if not isinstance(self.metrics, frozenset) or not self.metrics:
            raise ValueError("event metrics must be a non-empty frozenset")
        if any(not isinstance(metric, MetricKind) for metric in self.metrics):
            raise TypeError("event metrics must contain only MetricKind values")
        unsupported_metrics = self.metrics - _ALLOWED_METRICS_BY_ROLE[self.role]
        if unsupported_metrics:
            names = ", ".join(sorted(metric.value for metric in unsupported_metrics))
            raise ValueError(f"{self.role.value} events cannot select {names}")
        if not isinstance(self.aliases, tuple):
            raise TypeError("event aliases must be a tuple")
        if any(
            not isinstance(alias, str) or not alias.strip() or alias != alias.strip()
            for alias in self.aliases
        ):
            raise ValueError("event aliases must contain non-empty, trimmed strings")
        if self.name in self.aliases or len(set(self.aliases)) != len(self.aliases):
            raise ValueError(f"event aliases must be unique for {self.name!r}")


EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec(
        name="tp-reduce-scatter",
        aliases=("reduce-scatter",),
        role=EventRole.PHYSICAL_LEAF,
        metrics=frozenset((MetricKind.LEAF_SUM, MetricKind.INTERVAL_UNION)),
    ),
    EventSpec(
        name="tp-reduce-scatter-last",
        role=EventRole.LOGICAL_COMPOSITE,
        metrics=frozenset((MetricKind.PHASE_WALL, MetricKind.INTERVAL_UNION)),
    ),
    EventSpec(
        name="tp-linear-async-launch",
        role=EventRole.ASYNC_LAUNCH,
        metrics=frozenset((MetricKind.LAUNCH_ATTEMPT_COUNT,)),
    ),
    EventSpec(
        name="tp-linear-async-complete",
        role=EventRole.ASYNC_WAIT,
        metrics=frozenset((MetricKind.EXPOSED_WAIT,)),
    ),
    EventSpec(
        name="dp-param-all-gather",
        role=EventRole.COLLECTIVE_DISPATCH,
        metrics=frozenset((MetricKind.DISPATCH_ATTEMPT_COUNT,)),
    ),
    EventSpec(
        name="dp-reduce-scatter",
        role=EventRole.COLLECTIVE_DISPATCH,
        metrics=frozenset((MetricKind.DISPATCH_ATTEMPT_COUNT,)),
    ),
    EventSpec(
        name="dp-allreduce",
        role=EventRole.COLLECTIVE_DISPATCH,
        metrics=frozenset((MetricKind.DISPATCH_ATTEMPT_COUNT,)),
    ),
    EventSpec(
        name="dp-param-sync-complete",
        role=EventRole.STREAM_DEPENDENCY,
        metrics=frozenset((MetricKind.COMPLETION_ATTEMPT_COUNT, MetricKind.STREAM_DEPENDENCY)),
    ),
    EventSpec(
        name="dp-grad-sync-complete",
        role=EventRole.STREAM_DEPENDENCY,
        metrics=frozenset((MetricKind.COMPLETION_ATTEMPT_COUNT, MetricKind.STREAM_DEPENDENCY)),
    ),
)


def _index_event_specs(specs: tuple[EventSpec, ...]) -> Mapping[str, EventSpec]:
    indexed: dict[str, EventSpec] = {}
    for spec in specs:
        for name in (spec.name, *spec.aliases):
            if name in indexed:
                raise ValueError(f"duplicate event catalog name: {name!r}")
            indexed[name] = spec
    return MappingProxyType(indexed)


_EVENT_SPECS_BY_NAME = _index_event_specs(EVENT_SPECS)


def get_event_spec(name: str) -> EventSpec | None:
    """Return the exact canonical/alias match, or ``None`` for unknown input."""

    if not isinstance(name, str):
        return None
    return _EVENT_SPECS_BY_NAME.get(name)


def supports_metric(name: str, metric: MetricKind) -> bool:
    """Return whether a known event may participate in ``metric``."""

    if not isinstance(metric, MetricKind):
        return False
    spec = get_event_spec(name)
    return spec is not None and metric in spec.metrics


def is_event_eligible_for_metric(
    name: str, metric: MetricKind, *, allow_uncataloged: bool = False
) -> bool:
    """Select an event for one metric with explicit legacy compatibility.

    Known events always follow their typed catalog entry.  Uncataloged names
    pass through only when a caller explicitly owns a legacy compatibility
    path; invalid inputs fail closed.
    """

    if not isinstance(name, str) or not isinstance(metric, MetricKind):
        return False
    if type(allow_uncataloged) is not bool:
        return False
    spec = get_event_spec(name)
    if spec is None:
        return allow_uncataloged
    return metric in spec.metrics


def event_names_for_metric(metric: MetricKind, *, include_aliases: bool = False) -> tuple[str, ...]:
    """Return catalog names eligible for a metric, preserving declaration order."""

    if not isinstance(metric, MetricKind):
        return ()
    names: list[str] = []
    for spec in EVENT_SPECS:
        if metric not in spec.metrics:
            continue
        names.append(spec.name)
        if include_aliases:
            names.extend(spec.aliases)
    return tuple(names)


__all__ = [
    "CapabilityStatus",
    "EVENT_SPECS",
    "EventRole",
    "EventSpec",
    "MetricKind",
    "event_names_for_metric",
    "get_event_spec",
    "is_event_eligible_for_metric",
    "supports_metric",
]
