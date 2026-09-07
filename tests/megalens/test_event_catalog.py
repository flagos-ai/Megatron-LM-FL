# Copyright (c) 2026, MegaLens Authors. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from megatron.megalens.event_catalog import (
    EVENT_SPECS,
    CapabilityStatus,
    EventRole,
    EventSpec,
    MetricKind,
    event_names_for_metric,
    get_event_spec,
    is_event_eligible_for_metric,
    supports_metric,
)


def test_catalog_vocabulary_matches_nested_and_async_lifecycle_contracts() -> None:
    assert {role.value for role in EventRole} == {
        "logical-composite",
        "physical-leaf",
        "async-launch",
        "async-wait",
        "collective-dispatch",
        "stream-dependency",
    }
    assert {metric.value for metric in MetricKind} == {
        "phase_wall_us",
        "leaf_sum_us",
        "interval_union_us",
        "launch_attempt_count",
        "dispatch_attempt_count",
        "completion_attempt_count",
        "exposed_wait_us",
        "stream_dependency_us",
    }
    assert {status.value for status in CapabilityStatus} == {
        "available",
        "partial",
        "unavailable",
        "unknown",
    }


def test_catalog_registers_nested_reduce_scatter_and_linear_async_roles() -> None:
    leaf = get_event_spec("tp-reduce-scatter")
    composite = get_event_spec("tp-reduce-scatter-last")
    launch = get_event_spec("tp-linear-async-launch")
    completion = get_event_spec("tp-linear-async-complete")

    assert leaf is not None
    assert leaf.role is EventRole.PHYSICAL_LEAF
    assert leaf.metrics == frozenset((MetricKind.LEAF_SUM, MetricKind.INTERVAL_UNION))
    assert leaf.aliases == ("reduce-scatter",)
    assert composite is not None
    assert composite.role is EventRole.LOGICAL_COMPOSITE
    assert composite.metrics == frozenset((MetricKind.PHASE_WALL, MetricKind.INTERVAL_UNION))
    assert composite.aliases == ()
    assert launch is not None
    assert launch.role is EventRole.ASYNC_LAUNCH
    assert launch.metrics == frozenset((MetricKind.LAUNCH_ATTEMPT_COUNT,))
    assert completion is not None
    assert completion.role is EventRole.ASYNC_WAIT
    assert completion.metrics == frozenset((MetricKind.EXPOSED_WAIT,))
    assert tuple(spec.name for spec in EVENT_SPECS) == (
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "dp-param-all-gather",
        "dp-reduce-scatter",
        "dp-allreduce",
        "dp-param-sync-complete",
        "dp-grad-sync-complete",
    )


def test_catalog_registers_dp_dispatch_and_stream_dependency_roles() -> None:
    for name in ("dp-param-all-gather", "dp-reduce-scatter", "dp-allreduce"):
        dispatch = get_event_spec(name)
        assert dispatch is not None
        assert dispatch.role is EventRole.COLLECTIVE_DISPATCH
        assert dispatch.metrics == frozenset((MetricKind.DISPATCH_ATTEMPT_COUNT,))

    for name in ("dp-param-sync-complete", "dp-grad-sync-complete"):
        completion = get_event_spec(name)
        assert completion is not None
        assert completion.role is EventRole.STREAM_DEPENDENCY
        assert completion.metrics == frozenset(
            (MetricKind.COMPLETION_ATTEMPT_COUNT, MetricKind.STREAM_DEPENDENCY)
        )


def test_catalog_resolves_the_emitted_legacy_reduce_scatter_name() -> None:
    legacy = get_event_spec("reduce-scatter")

    assert legacy is get_event_spec("tp-reduce-scatter")
    assert legacy is not None
    assert legacy.name == "tp-reduce-scatter"


def test_unknown_events_and_metrics_fail_closed() -> None:
    assert get_event_spec("tp-all-gather-first") is None
    assert get_event_spec("_reduce_scatter_along_first_dim") is None
    assert get_event_spec("_reduce_scatter_along_last_dim") is None
    assert get_event_spec("TP-REDUCE-SCATTER") is None
    assert get_event_spec(None) is None  # type: ignore[arg-type]
    assert not supports_metric("unknown", MetricKind.LEAF_SUM)
    assert not supports_metric("tp-reduce-scatter", "leaf_sum_us")  # type: ignore[arg-type]
    assert event_names_for_metric("leaf_sum_us") == ()  # type: ignore[arg-type]


def test_metric_eligibility_requires_explicit_legacy_passthrough() -> None:
    assert is_event_eligible_for_metric("tp-reduce-scatter", MetricKind.LEAF_SUM)
    assert not is_event_eligible_for_metric("legacy-tp-event", MetricKind.LEAF_SUM)
    assert is_event_eligible_for_metric(
        "legacy-tp-event", MetricKind.LEAF_SUM, allow_uncataloged=True
    )
    assert not is_event_eligible_for_metric(
        "tp-linear-async-launch", MetricKind.INTERVAL_UNION, allow_uncataloged=True
    )
    assert not is_event_eligible_for_metric(
        "tp-linear-async-complete", MetricKind.INTERVAL_UNION, allow_uncataloged=True
    )
    assert not is_event_eligible_for_metric(
        None, MetricKind.LEAF_SUM, allow_uncataloged=True  # type: ignore[arg-type]
    )
    assert not is_event_eligible_for_metric(
        "legacy-tp-event", "leaf_sum_us", allow_uncataloged=True  # type: ignore[arg-type]
    )
    assert not is_event_eligible_for_metric(
        "legacy-tp-event", MetricKind.LEAF_SUM, allow_uncataloged="yes"  # type: ignore[arg-type]
    )


def test_tp_prefix_collection_still_requires_metric_eligibility() -> None:
    names = (
        "tp-all-gather-first",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "forward-step",
    )

    selected = tuple(
        name
        for name in names
        if name.startswith("tp-")
        and is_event_eligible_for_metric(name, MetricKind.INTERVAL_UNION, allow_uncataloged=True)
    )

    assert selected == ("tp-all-gather-first",)


def test_metric_selection_keeps_composite_and_leaf_roles_separate() -> None:
    assert event_names_for_metric(MetricKind.PHASE_WALL) == ("tp-reduce-scatter-last",)
    assert event_names_for_metric(MetricKind.LEAF_SUM) == ("tp-reduce-scatter",)
    assert event_names_for_metric(MetricKind.INTERVAL_UNION) == (
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
    )
    assert event_names_for_metric(MetricKind.LEAF_SUM, include_aliases=True) == (
        "tp-reduce-scatter",
        "reduce-scatter",
    )
    assert event_names_for_metric(MetricKind.EXPOSED_WAIT) == ("tp-linear-async-complete",)
    assert event_names_for_metric(MetricKind.STREAM_DEPENDENCY) == (
        "dp-param-sync-complete",
        "dp-grad-sync-complete",
    )
    assert event_names_for_metric(MetricKind.LAUNCH_ATTEMPT_COUNT) == (
        "tp-linear-async-launch",
    )
    assert event_names_for_metric(MetricKind.DISPATCH_ATTEMPT_COUNT) == (
        "dp-param-all-gather",
        "dp-reduce-scatter",
        "dp-allreduce",
    )
    assert event_names_for_metric(MetricKind.COMPLETION_ATTEMPT_COUNT) == (
        "dp-param-sync-complete",
        "dp-grad-sync-complete",
    )
    assert not supports_metric("tp-reduce-scatter-last", MetricKind.LEAF_SUM)
    assert not supports_metric("tp-reduce-scatter", MetricKind.PHASE_WALL)
    assert not supports_metric("tp-linear-async-launch", MetricKind.INTERVAL_UNION)
    assert not supports_metric("tp-linear-async-complete", MetricKind.LEAF_SUM)
    assert not supports_metric("dp-allreduce", MetricKind.INTERVAL_UNION)
    assert not supports_metric("dp-grad-sync-complete", MetricKind.EXPOSED_WAIT)


@pytest.mark.parametrize(
    ("role", "metric"),
    [
        (EventRole.LOGICAL_COMPOSITE, MetricKind.LEAF_SUM),
        (EventRole.PHYSICAL_LEAF, MetricKind.PHASE_WALL),
        (EventRole.ASYNC_LAUNCH, MetricKind.EXPOSED_WAIT),
        (EventRole.ASYNC_LAUNCH, MetricKind.INTERVAL_UNION),
        (EventRole.ASYNC_WAIT, MetricKind.LAUNCH_ATTEMPT_COUNT),
        (EventRole.ASYNC_WAIT, MetricKind.LEAF_SUM),
        (EventRole.COLLECTIVE_DISPATCH, MetricKind.COMPLETION_ATTEMPT_COUNT),
        (EventRole.COLLECTIVE_DISPATCH, MetricKind.STREAM_DEPENDENCY),
        (EventRole.COLLECTIVE_DISPATCH, MetricKind.INTERVAL_UNION),
        (EventRole.STREAM_DEPENDENCY, MetricKind.LAUNCH_ATTEMPT_COUNT),
        (EventRole.STREAM_DEPENDENCY, MetricKind.DISPATCH_ATTEMPT_COUNT),
        (EventRole.STREAM_DEPENDENCY, MetricKind.EXPOSED_WAIT),
    ],
)
def test_event_spec_rejects_metrics_that_conflict_with_its_role(
    role: EventRole, metric: MetricKind
) -> None:
    with pytest.raises(ValueError, match=metric.value):
        EventSpec(name="invalid-role-metric", role=role, metrics=frozenset((metric,)))


@pytest.mark.parametrize(
    ("name", "aliases", "message"),
    [
        (" tp-leaf", (), "name"),
        ("tp-leaf ", (), "name"),
        ("tp-leaf", (" legacy",), "aliases"),
        ("tp-leaf", ("legacy ",), "aliases"),
    ],
)
def test_event_spec_rejects_untrimmed_names_and_aliases(
    name: str, aliases: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EventSpec(
            name=name,
            aliases=aliases,
            role=EventRole.PHYSICAL_LEAF,
            metrics=frozenset((MetricKind.LEAF_SUM,)),
        )


def test_catalog_entries_are_immutable() -> None:
    leaf = get_event_spec("tp-reduce-scatter")
    assert leaf is not None

    with pytest.raises(FrozenInstanceError):
        leaf.role = EventRole.LOGICAL_COMPOSITE  # type: ignore[misc]
    with pytest.raises(AttributeError):
        leaf.metrics.add(MetricKind.PHASE_WALL)  # type: ignore[attr-defined]
