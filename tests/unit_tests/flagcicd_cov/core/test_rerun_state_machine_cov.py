# Copyright (c) 2026, BAAI. All rights reserved.
"""Rerun state machine helpers that need no training loop: stats, replay, injection, tracker."""

import math
from types import SimpleNamespace

import pytest

from megatron.core import rerun_state_machine
from megatron.core.rerun_state_machine import (
    COMPARISON_MATCH,
    COMPARISON_MISMATCH,
    QuickStats,
    RerunDataIterator,
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    RerunState,
    RerunStateMachine,
)


def test_quick_stats_counts_zeros_apart_from_samples():
    stats = QuickStats()
    for value in (0.0, 2.0, 0.0, 5.0, 1.0):
        stats.record(value)
    assert stats.zero_cnt == 2
    assert stats.samples == [2.0, 5.0, 1.0]
    assert stats.max == 5.0


def test_quick_stats_overwrites_oldest_slot_once_full():
    stats = QuickStats(max_size=2)
    for value in (1.0, 2.0, 3.0, 4.0):
        stats.record(value)
    assert stats.samples == [3.0, 4.0]
    assert stats.pos == 4


def test_quick_stats_combine_merges_samples_zero_counts_and_max():
    first, second, third = QuickStats(), QuickStats(), QuickStats()
    first.record(1.0)
    second.record(0.0)
    second.record(7.0)
    third.record(3.0)

    first.combine([])
    assert first.samples == [1.0]

    first.combine([second, third])
    assert first.samples == [1.0, 7.0, 3.0]
    assert first.pos == 3
    assert first.zero_cnt == 1
    assert first.max == 7.0


def test_quick_stats_combine_keeps_own_samples_when_the_union_overflows():
    first, second = QuickStats(max_size=2), QuickStats()
    first.record(1.0)
    second.record(2.0)
    second.record(9.0)

    first.combine([second])

    assert first.samples == [1.0]
    assert first.max == 9.0


def test_quick_stats_report_and_reset():
    stats = QuickStats()
    assert stats.print_stats() == "0 samples, all identical"

    for value in (0.0, 4.0, 2.0, 0.0):
        stats.record(value)
    report = stats.print_stats()
    assert report.startswith("4/2 total/identical samples, rel. variability: avg= 1.500E+00")
    # Percentiles index into the non-zero samples after discounting the zeros.
    assert "p50= 2.000E+00" in report
    assert "p99.9= 4.000E+00" in report
    assert report.endswith("max: 4.000E+00")

    stats.reset()
    assert (stats.samples, stats.pos, stats.zero_cnt, stats.max) == ([], 0, 0, 0.0)


@pytest.fixture
def rerun_mode(monkeypatch):
    """Let RerunDataIterator see a rerun mode without building the global state machine."""
    holder = SimpleNamespace(mode=RerunMode.VALIDATE_RESULTS)
    monkeypatch.setattr(
        rerun_state_machine,
        "get_rerun_state_machine",
        lambda: SimpleNamespace(get_mode=lambda: holder.mode),
    )
    return holder


def test_data_iterator_replays_the_buffered_microbatches(rerun_mode):
    iterator = RerunDataIterator(iter(range(10)))
    assert [next(iterator), next(iterator)] == [0, 1]

    iterator.rewind()
    assert [next(iterator), next(iterator)] == [0, 1]
    with pytest.raises(AssertionError, match="No more batches to replay"):
        next(iterator)

    iterator.advance()
    assert next(iterator) == 2
    assert iterator.saved_microbatches == [2]


def test_data_iterator_does_not_buffer_when_reruns_are_disabled(rerun_mode):
    rerun_mode.mode = RerunMode.DISABLED
    iterator = RerunDataIterator(iter(range(3)))
    assert next(iterator) == 0
    assert iterator.saved_microbatches == []


def test_data_iterator_state_round_trip(rerun_mode):
    iterator = RerunDataIterator(iter(range(10)))
    next(iterator)
    next(iterator)
    iterator.rewind()
    next(iterator)

    restored = RerunDataIterator(iter(()))
    restored.load_state_dict(iterator.state_dict())

    assert restored.replaying
    assert next(restored) == 1


@pytest.fixture
def deterministic_injection(monkeypatch):
    monkeypatch.setattr(rerun_state_machine.random, "randint", lambda low, high: 0)
    monkeypatch.setattr(rerun_state_machine, "safe_get_rank", lambda: 0)


def test_injector_is_inert_without_a_rate():
    injector = RerunErrorInjector()
    assert not injector.maybe_inject()
    assert injector.injected_error_type is None


def test_injector_rejects_a_non_diagnostic_type():
    with pytest.raises(AssertionError, match="valid RerunDiagnostic"):
        RerunErrorInjector(error_injection_rate=1, error_injection_type="transient")


def test_injector_fires_once_per_iteration(deterministic_injection):
    injector = RerunErrorInjector(error_injection_rate=4)
    assert injector.maybe_inject()
    assert injector.injected_error_type == RerunDiagnostic.TRANSIENT_ERROR
    assert not injector.maybe_inject()


def test_injector_skips_when_the_draw_misses(monkeypatch):
    monkeypatch.setattr(rerun_state_machine.random, "randint", lambda low, high: 1)
    monkeypatch.setattr(rerun_state_machine, "safe_get_rank", lambda: 0)
    injector = RerunErrorInjector(error_injection_rate=4)
    assert not injector.maybe_inject()


def test_miscompare_defers_to_the_comparison_when_nothing_was_injected():
    injector = RerunErrorInjector(error_injection_rate=4)
    result = injector.maybe_miscompare(lambda a, b: abs(a - b), 1.0, 3.5, RerunState.NOT_RUNNING_YET)
    assert result == 2.5


def test_transient_error_mismatches_on_the_in_place_rerun(deterministic_injection):
    injector = RerunErrorInjector(error_injection_rate=1)
    injector.maybe_inject()
    assert (
        injector.maybe_miscompare(None, None, None, RerunState.RERUNNING_IN_PLACE)
        == COMPARISON_MISMATCH
    )
    assert injector.injected_error_type is None


@pytest.mark.parametrize(
    ("error_type", "from_checkpoint_result"),
    [
        (RerunDiagnostic.PERSISTENT_ERROR, COMPARISON_MISMATCH),
        (RerunDiagnostic.CORRECT_RESULT, COMPARISON_MATCH),
    ],
)
def test_reproducible_injection_resolves_on_the_checkpoint_rerun(
    deterministic_injection, error_type, from_checkpoint_result
):
    injector = RerunErrorInjector(error_injection_rate=1, error_injection_type=error_type)
    injector.maybe_inject()

    assert (
        injector.maybe_miscompare(None, None, None, RerunState.RERUNNING_IN_PLACE)
        == COMPARISON_MATCH
    )
    assert injector.injected_error_type == error_type
    assert (
        injector.maybe_miscompare(None, None, None, RerunState.RERUNNING_FROM_CHECKPOINT)
        == from_checkpoint_result
    )
    assert injector.injected_error_type is None


@pytest.mark.parametrize(
    "state", [RerunState.RERUNNING_FROM_CHECKPOINT, RerunState.NOT_RUNNING_YET]
)
def test_transient_injection_outside_the_in_place_rerun_is_a_bug(deterministic_injection, state):
    injector = RerunErrorInjector(error_injection_rate=1)
    injector.maybe_inject()
    with pytest.raises(RuntimeError, match="Should not be here"):
        injector.maybe_miscompare(None, None, None, state)


def test_injector_state_round_trip(deterministic_injection):
    injector = RerunErrorInjector(
        error_injection_rate=3, error_injection_type=RerunDiagnostic.PERSISTENT_ERROR
    )
    injector.maybe_inject()

    restored = RerunErrorInjector()
    restored.load_state_dict(injector.state_dict())

    assert restored.should_inject_errors
    assert restored.error_injection_rate == 3
    assert restored.injected_error_type == RerunDiagnostic.PERSISTENT_ERROR


def tracker_line(job, rank, iteration, status):
    return (
        f"ts=2026-01-01 node=n0 device=0 jobID={job} rank={rank} iteration={iteration} "
        f"status=RerunValidationStatus.{status} result=nan message=None\n"
    )


def test_tracker_skips_iterations_that_fail_twice_on_one_rank(tmp_path):
    tracker = tmp_path / "tracker.txt"
    tracker.write_text(
        "unrelated line\n"
        # Iteration 5: the same rank fails in two jobs -> skip.
        + tracker_line("a", 0, 5, "RERUN_DISABLED")
        + tracker_line("b", 0, 5, "SECOND_RERUN_REPRODUCIBLE")
        # Iteration 7: a single failure is not enough.
        + tracker_line("a", 1, 7, "FIRST_RERUN_REPRODUCIBLE")
        # Iteration 9: repeated, but several ranks failed in one job -> not a data problem.
        + tracker_line("a", 0, 9, "RERUN_DISABLED")
        + tracker_line("a", 1, 9, "RERUN_DISABLED")
        + tracker_line("b", 0, 9, "RERUN_DISABLED")
        # Iteration 11: statuses that do not indicate a reproducible failure are ignored.
        + tracker_line("a", 0, 11, "FIRST_RERUN_NOT_REPRODUCIBLE")
        + tracker_line("b", 0, 11, "FIRST_RERUN_NOT_REPRODUCIBLE")
    )

    assert RerunStateMachine.get_skipped_iterations_from_tracker_file(str(tracker)) == [5]


def test_tracker_read_failure_skips_nothing(tmp_path):
    missing = tmp_path / "missing.txt"
    assert RerunStateMachine.get_skipped_iterations_from_tracker_file(str(missing)) == []


def test_comparison_constants_are_ordered():
    assert COMPARISON_MATCH == 0.0
    assert math.isinf(COMPARISON_MISMATCH)
