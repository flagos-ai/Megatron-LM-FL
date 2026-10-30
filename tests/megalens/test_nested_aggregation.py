# Copyright (c) 2026, MegaLens Authors. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest

from megatron.megalens.data_loader import SpanEvent
from megatron.megalens.event_catalog import CapabilityStatus
from megatron.megalens.nested_aggregation import MetricResult, aggregate_tp_reduce_scatter_partition

_MISSING = object()


def _span(
    name: str,
    ts: int,
    dur: int,
    *,
    rank: int = 0,
    iteration: object = 7,
    data_bytes: object = _MISSING,
    metadata: dict[str, object] | None = None,
) -> SpanEvent:
    args: dict[str, object] = {}
    if iteration is not _MISSING:
        args["iteration"] = iteration
    if data_bytes is not _MISSING:
        args["data_bytes"] = data_bytes
    if metadata is not None:
        args.update(metadata)
    return SpanEvent(name=name, ts=ts, dur=dur, rank=rank, args=args)


def test_valid_nested_pair_separates_phase_leaf_union_and_payload() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span(
                "tp-reduce-scatter-last",
                100,
                100,
                data_bytes=4_096,
                metadata={"op": "reduce-scatter", "dim": "last", "group_size": 2, "group": [1]},
            ),
            _span(
                "tp-reduce-scatter",
                120,
                60,
                data_bytes=4_096,
                metadata={"op": "reduce-scatter", "dim": "first", "group_size": 2, "group": [1]},
            ),
        ]
    )

    assert result.phase_wall_us.value == 100
    assert result.leaf_sum_us.value == 60
    assert result.interval_union_us.value == 100
    assert result.leaf_data_bytes.value == 4_096
    assert {
        result.phase_wall_us.status,
        result.leaf_sum_us.status,
        result.interval_union_us.status,
        result.leaf_data_bytes.status,
    } == {CapabilityStatus.AVAILABLE}


@pytest.mark.parametrize("leaf_name", ["tp-reduce-scatter", "reduce-scatter"])
def test_standalone_canonical_and_legacy_leaves_have_the_same_metrics(leaf_name: str) -> None:
    result = aggregate_tp_reduce_scatter_partition([_span(leaf_name, 10, 40, data_bytes=512)])

    assert result.phase_wall_us.value is None
    assert result.phase_wall_us.status is CapabilityStatus.UNAVAILABLE
    assert result.leaf_sum_us.value == 40
    assert result.leaf_sum_us.status is CapabilityStatus.AVAILABLE
    assert result.interval_union_us.value == 40
    assert result.interval_union_us.status is CapabilityStatus.AVAILABLE
    assert result.leaf_data_bytes.value == 512
    assert result.leaf_data_bytes.status is CapabilityStatus.AVAILABLE


def test_historical_legacy_leaf_without_payload_keeps_timing_evidence() -> None:
    result = aggregate_tp_reduce_scatter_partition([_span("reduce-scatter", 10, 40)])

    assert result.leaf_sum_us.value == 40
    assert result.leaf_sum_us.status is CapabilityStatus.AVAILABLE
    assert result.interval_union_us.value == 40
    assert result.interval_union_us.status is CapabilityStatus.AVAILABLE
    assert result.leaf_data_bytes.value is None
    assert result.leaf_data_bytes.status is CapabilityStatus.PARTIAL


def test_parent_only_keeps_observed_phase_and_partial_union_without_fake_leaf_values() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [_span("tp-reduce-scatter-last", 0, 100, data_bytes=4_096)]
    )

    assert result.phase_wall_us.value == 100
    assert result.phase_wall_us.status is CapabilityStatus.AVAILABLE
    assert result.phase_wall_us.reason
    assert result.leaf_sum_us.value is None
    assert result.leaf_sum_us.status is CapabilityStatus.PARTIAL
    assert result.interval_union_us.value == 100
    assert result.interval_union_us.status is CapabilityStatus.PARTIAL
    assert result.leaf_data_bytes.value is None
    assert result.leaf_data_bytes.status is CapabilityStatus.PARTIAL


def test_multiple_valid_pairs_sum_each_role_and_payload_once() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter", 170, 30, data_bytes=2_000),
            _span("tp-reduce-scatter-last", 150, 80, data_bytes=2_000),
            _span("tp-reduce-scatter", 20, 60, data_bytes=1_000),
            _span("tp-reduce-scatter-last", 0, 100, data_bytes=1_000),
        ]
    )

    assert result.phase_wall_us.value == 180
    assert result.leaf_sum_us.value == 90
    assert result.interval_union_us.value == 180
    assert result.leaf_data_bytes.value == 3_000


def test_overlapping_and_adjacent_standalone_leaves_use_interval_union() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter", 0, 10, data_bytes=10),
            _span("tp-reduce-scatter", 5, 10, data_bytes=20),
            _span("tp-reduce-scatter", 15, 5, data_bytes=30),
        ]
    )

    assert result.leaf_sum_us.value == 25
    assert result.interval_union_us.value == 20
    assert result.leaf_data_bytes.value == 60


def test_equal_start_and_end_boundaries_are_valid_containment() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 25, 50, data_bytes=100),
            _span("tp-reduce-scatter", 25, 50, data_bytes=100),
        ]
    )

    assert result.leaf_sum_us.status is CapabilityStatus.AVAILABLE
    assert result.interval_union_us.value == 50


def test_multiple_children_fail_closed_for_leaf_and_payload() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 100, data_bytes=3_000),
            _span("tp-reduce-scatter", 10, 20, data_bytes=1_000),
            _span("tp-reduce-scatter", 40, 20, data_bytes=2_000),
        ]
    )

    assert result.phase_wall_us.value == 100
    assert result.phase_wall_us.status is CapabilityStatus.AVAILABLE
    assert result.leaf_sum_us.value is None
    assert result.leaf_sum_us.status is CapabilityStatus.UNKNOWN
    assert result.leaf_data_bytes.value is None
    assert result.leaf_data_bytes.status is CapabilityStatus.UNKNOWN
    assert result.interval_union_us.value == 100
    assert result.interval_union_us.status is CapabilityStatus.PARTIAL


def test_leaf_contained_by_multiple_parents_is_ambiguous() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 100),
            _span("tp-reduce-scatter-last", 10, 80),
            _span("tp-reduce-scatter", 20, 20, data_bytes=100),
        ]
    )

    assert result.leaf_sum_us.status is CapabilityStatus.UNKNOWN
    assert result.leaf_sum_us.value is None
    assert "overlap" in result.leaf_sum_us.reason
    assert result.phase_wall_us.status is CapabilityStatus.UNKNOWN
    assert result.phase_wall_us.value is None


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        (
            [
                _span("tp-reduce-scatter-last", 0, 100),
                _span("tp-reduce-scatter", 90, 20, data_bytes=100),
            ],
            "without containment",
        ),
        (
            [
                _span("tp-reduce-scatter-last", 0, 100),
                _span("tp-reduce-scatter-last", 50, 100),
                _span("tp-reduce-scatter", 10, 20, data_bytes=100),
                _span("tp-reduce-scatter", 120, 20, data_bytes=100),
            ],
            "without thread identity",
        ),
    ],
)
def test_ambiguous_overlap_fails_closed_for_leaf_metrics(
    events: list[SpanEvent], reason: str
) -> None:
    result = aggregate_tp_reduce_scatter_partition(events)

    assert result.leaf_sum_us.status is CapabilityStatus.UNKNOWN
    assert result.leaf_sum_us.value is None
    assert reason in result.leaf_sum_us.reason
    assert result.interval_union_us.status is CapabilityStatus.PARTIAL


def test_outer_payload_mismatch_does_not_pollute_leaf_metrics() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 100, data_bytes=200),
            _span("tp-reduce-scatter", 10, 20, data_bytes=100),
        ]
    )

    assert result.phase_wall_us.value == 100
    assert result.leaf_sum_us.value == 20
    assert result.interval_union_us.value == 100
    assert result.leaf_data_bytes.value == 100
    assert {
        result.phase_wall_us.status,
        result.leaf_sum_us.status,
        result.interval_union_us.status,
        result.leaf_data_bytes.status,
    } == {CapabilityStatus.AVAILABLE}


@pytest.mark.parametrize(
    ("event", "affected_metrics"),
    [
        (
            _span("tp-reduce-scatter", 0, 10, data_bytes=100, metadata={"op": "all-gather"}),
            ("leaf_sum_us", "leaf_data_bytes"),
        ),
        (_span("tp-reduce-scatter-last", 0, 10, metadata={"dim": "first"}), ("phase_wall_us",)),
    ],
)
def test_explicit_role_metadata_conflicts_fail_closed(
    event: SpanEvent, affected_metrics: tuple[str, ...]
) -> None:
    result = aggregate_tp_reduce_scatter_partition([event])

    for metric_name in affected_metrics:
        metric = getattr(result, metric_name)
        assert metric.status is CapabilityStatus.UNKNOWN
        assert metric.value is None
        assert "metadata conflicts" in metric.reason
    assert result.interval_union_us.status is CapabilityStatus.UNKNOWN
    assert result.interval_union_us.value is None
    assert "metadata conflicts" in result.interval_union_us.reason


@pytest.mark.parametrize(
    ("parent_metadata", "leaf_metadata", "conflict_key"),
    [
        ({"group_size": 2}, {"group_size": 4}, "group_size"),
        ({"group_size": 2, "group": [1]}, {"group_size": 2, "group": [2]}, "group"),
    ],
)
def test_known_group_identity_conflict_downgrades_nested_pairing_only(
    parent_metadata: dict[str, object], leaf_metadata: dict[str, object], conflict_key: str
) -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 100, data_bytes=200, metadata=parent_metadata),
            _span("tp-reduce-scatter", 10, 20, data_bytes=100, metadata=leaf_metadata),
        ]
    )

    assert result.phase_wall_us.value == 100
    assert result.phase_wall_us.status is CapabilityStatus.AVAILABLE
    assert result.leaf_sum_us.value is None
    assert result.leaf_sum_us.status is CapabilityStatus.UNKNOWN
    assert result.leaf_data_bytes.value is None
    assert result.leaf_data_bytes.status is CapabilityStatus.UNKNOWN
    assert result.interval_union_us.value == 100
    assert result.interval_union_us.status is CapabilityStatus.PARTIAL
    assert conflict_key in result.interval_union_us.reason


def test_unknown_group_identity_does_not_reject_an_otherwise_valid_pair() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span(
                "tp-reduce-scatter-last",
                0,
                100,
                data_bytes=100,
                metadata={"group_size": 2, "group": None},
            ),
            _span(
                "tp-reduce-scatter",
                10,
                20,
                data_bytes=100,
                metadata={"group_size": 2, "group": [1]},
            ),
        ]
    )

    assert result.leaf_sum_us.status is CapabilityStatus.AVAILABLE
    assert result.leaf_data_bytes.status is CapabilityStatus.AVAILABLE
    assert result.interval_union_us.status is CapabilityStatus.AVAILABLE


def test_zero_duration_is_partial_timing_evidence_but_keeps_valid_payload() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 10, data_bytes=100),
            _span("tp-reduce-scatter", 5, 0, data_bytes=100),
        ]
    )

    assert result.phase_wall_us.value == 10
    assert result.phase_wall_us.status is CapabilityStatus.AVAILABLE
    assert result.leaf_sum_us.value is None
    assert result.leaf_sum_us.status is CapabilityStatus.PARTIAL
    assert result.interval_union_us.value == 10
    assert result.interval_union_us.status is CapabilityStatus.PARTIAL
    assert result.leaf_data_bytes.value == 100
    assert result.leaf_data_bytes.status is CapabilityStatus.AVAILABLE


@pytest.mark.parametrize("leaf_overrides", [{"rank": 1}, {"iteration": 8}])
def test_cross_partition_spans_fail_closed(leaf_overrides: dict[str, int]) -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 100),
            _span("tp-reduce-scatter", 20, 20, data_bytes=100, **leaf_overrides),
        ]
    )

    for metric in (
        result.phase_wall_us,
        result.leaf_sum_us,
        result.interval_union_us,
        result.leaf_data_bytes,
    ):
        assert metric.status is CapabilityStatus.UNKNOWN
        assert metric.value is None
        assert "one rank and iteration" in metric.reason


@pytest.mark.parametrize(
    "events",
    [
        [],
        [_span("tp-all-gather-first", 0, 10)],
        [_span("_reduce_scatter_along_first_dim", 0, 10)],
        [_span("_reduce_scatter_along_last_dim", 0, 10)],
    ],
)
def test_empty_unknown_and_stale_names_are_unavailable(events: list[SpanEvent]) -> None:
    result = aggregate_tp_reduce_scatter_partition(events)

    for metric in (
        result.phase_wall_us,
        result.leaf_sum_us,
        result.interval_union_us,
        result.leaf_data_bytes,
    ):
        assert metric.status is CapabilityStatus.UNAVAILABLE
        assert metric.value is None


def test_unknown_events_do_not_change_valid_cataloged_metrics() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("unrelated", 0, 1_000, rank=9, iteration=99),
            _span("tp-reduce-scatter", 20, 10, data_bytes=64),
        ]
    )

    assert result.leaf_sum_us.value == 10
    assert result.interval_union_us.value == 10
    assert result.leaf_data_bytes.value == 64


def test_input_order_does_not_change_results() -> None:
    events = [
        _span("tp-reduce-scatter-last", 0, 100, data_bytes=100),
        _span("tp-reduce-scatter", 20, 20, data_bytes=100),
        _span("unrelated", 0, 1_000, rank=9, iteration=99),
    ]

    assert aggregate_tp_reduce_scatter_partition(events) == aggregate_tp_reduce_scatter_partition(
        list(reversed(events))
    )


def test_conflict_reasons_do_not_depend_on_input_order() -> None:
    events = [
        _span("tp-reduce-scatter", 0, 10, data_bytes=100, metadata={"dim": "last"}),
        _span("reduce-scatter", 20, 10, data_bytes=100, metadata={"op": "all-gather"}),
    ]

    assert aggregate_tp_reduce_scatter_partition(events) == aggregate_tp_reduce_scatter_partition(
        list(reversed(events))
    )


def test_pair_identity_conflict_priority_does_not_depend_on_input_order() -> None:
    events = [
        _span("tp-reduce-scatter-last", 0, 100, metadata={"group_size": 2}),
        _span("tp-reduce-scatter", 10, 20, data_bytes=100, metadata={"group_size": 4}),
        _span("tp-reduce-scatter-last", 200, 100, metadata={"group": [1]}),
        _span("tp-reduce-scatter", 210, 20, data_bytes=100, metadata={"group": [2]}),
    ]

    assert aggregate_tp_reduce_scatter_partition(events) == aggregate_tp_reduce_scatter_partition(
        list(reversed(events))
    )


@pytest.mark.parametrize(
    "span",
    [
        _span("tp-reduce-scatter", True, 10, data_bytes=1),
        _span("tp-reduce-scatter", -1, 10, data_bytes=1),
        _span("tp-reduce-scatter", 0, -1, data_bytes=1),
        _span("tp-reduce-scatter", 0, True, data_bytes=1),
        _span("tp-reduce-scatter", 0, 10, rank=True, data_bytes=1),
        _span("tp-reduce-scatter", 0, 10, rank=-1, data_bytes=1),
        _span("tp-reduce-scatter", 0, 10, iteration="7", data_bytes=1),
        _span("tp-reduce-scatter", 0, 10, iteration=-1, data_bytes=1),
        _span("tp-reduce-scatter", 0, 10, iteration=_MISSING, data_bytes=1),
    ],
)
def test_malformed_cataloged_spans_fail_closed(span: SpanEvent) -> None:
    result = aggregate_tp_reduce_scatter_partition([span])

    for metric in (
        result.phase_wall_us,
        result.leaf_sum_us,
        result.interval_union_us,
        result.leaf_data_bytes,
    ):
        assert metric.status is CapabilityStatus.UNKNOWN
        assert metric.value is None


@pytest.mark.parametrize("data_bytes", [_MISSING, None, -1, True, "1024"])
def test_invalid_leaf_payload_does_not_hide_valid_time_metrics(data_bytes: object) -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [_span("tp-reduce-scatter", 0, 10, data_bytes=data_bytes)]
    )

    assert result.leaf_sum_us.value == 10
    assert result.leaf_sum_us.status is CapabilityStatus.AVAILABLE
    assert result.interval_union_us.value == 10
    assert result.leaf_data_bytes.value is None
    assert result.leaf_data_bytes.status is CapabilityStatus.PARTIAL


def test_invalid_nested_leaf_payload_does_not_hide_valid_time_metrics() -> None:
    result = aggregate_tp_reduce_scatter_partition(
        [
            _span("tp-reduce-scatter-last", 0, 100, data_bytes=100),
            _span("tp-reduce-scatter", 20, 20),
        ]
    )

    assert result.phase_wall_us.value == 100
    assert result.leaf_sum_us.value == 20
    assert result.interval_union_us.value == 100
    assert result.leaf_data_bytes.value is None
    assert result.leaf_data_bytes.status is CapabilityStatus.PARTIAL


def test_public_annotations_resolve_at_runtime() -> None:
    hints = get_type_hints(aggregate_tp_reduce_scatter_partition)

    assert hints["events"]
    assert hints["return"].__name__ == "TPReduceScatterMetrics"


def test_metric_results_are_immutable_and_reject_inconsistent_statuses() -> None:
    result = MetricResult(
        value=10, status=CapabilityStatus.PARTIAL, reason="observed lower-bound interval"
    )

    with pytest.raises(FrozenInstanceError):
        result.value = 20  # type: ignore[misc]
    with pytest.raises(ValueError, match="require a value"):
        MetricResult(value=None, status=CapabilityStatus.AVAILABLE, reason="missing")
    with pytest.raises(ValueError, match="cannot expose"):
        MetricResult(value=1, status=CapabilityStatus.UNKNOWN, reason="ambiguous")
