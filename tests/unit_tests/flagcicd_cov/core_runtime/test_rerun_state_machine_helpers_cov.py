# Copyright (c) 2026, BAAI. All rights reserved.
"""Rerun state machine helpers: mode, rank reductions, magnitude tracking, tracker and stats."""

import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.core import rerun_state_machine
from megatron.core.rerun_state_machine import (
    COMPARISON_MATCH,
    COMPARISON_MISMATCH,
    Caller,
    QuickStats,
    RerunDataIterator,
    RerunMode,
    RerunStateMachine,
    _compare_floats,
    destroy_rerun_state_machine,
)


@pytest.fixture
def make_machine(monkeypatch):
    """Build a machine with the platform and single-process distributed boundaries patched."""
    monkeypatch.setattr(
        rerun_state_machine,
        'cur_platform',
        SimpleNamespace(
            device=lambda value: torch.device('cpu'),
            current_device=lambda: 'gpu0',
            get_rng_state=torch.get_rng_state,
            set_rng_state=lambda state: None,
        ),
    )
    distributed = rerun_state_machine.torch.distributed
    monkeypatch.setattr(distributed, 'all_reduce', lambda tensor: None)
    monkeypatch.setattr(distributed, 'is_initialized', lambda: False)
    for name in ('RANK', 'SLURM_PROCID', 'SLURM_NTASKS'):
        monkeypatch.delenv(name, raising=False)

    def build(**kwargs):
        return RerunStateMachine(**kwargs)

    return build


@pytest.fixture
def clean_singleton():
    """Reset the rerun machine singleton to its initial value around each test."""
    rerun_state_machine.destroy_rerun_state_machine()
    yield
    rerun_state_machine.destroy_rerun_state_machine()


def test_tracker_file_is_created_and_a_missing_directory_is_rejected(tmp_path):
    tracker = tmp_path / 'tracker.log'
    RerunStateMachine(result_rejected_tracker_filename=str(tracker))
    assert tracker.exists()
    with pytest.raises(RuntimeError, match='cannot be appended to'):
        RerunStateMachine(
            result_rejected_tracker_filename=str(tmp_path / 'missing' / 'tracker.log')
        )


def test_set_mode_updates_the_operating_mode(make_machine):
    machine = make_machine(mode=RerunMode.DISABLED)
    machine.set_mode(RerunMode.REPORT_DETERMINISM_STATS)
    assert machine.get_mode() == RerunMode.REPORT_DETERMINISM_STATS


def test_reduce_any_or_combines_scalar_flags_across_ranks(make_machine, monkeypatch):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    remote = {'flag': 0}

    def all_reduce_or(tensor):
        tensor.add_(remote['flag'])
        tensor.clamp_(max=1)

    monkeypatch.setattr(rerun_state_machine.torch.distributed, 'all_reduce', all_reduce_or)
    remote['flag'] = 0
    assert machine._reduce_any(True) is True
    assert machine._reduce_any(False) is False
    remote['flag'] = 1
    assert machine._reduce_any(False) is True


def test_reduce_any_or_combines_list_flags_element_wise(make_machine, monkeypatch):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    remote = {'flags': torch.tensor([0, 0, 0], dtype=torch.int32)}

    def all_reduce_or(tensor):
        tensor.add_(remote['flags'])
        tensor.clamp_(max=1)

    monkeypatch.setattr(rerun_state_machine.torch.distributed, 'all_reduce', all_reduce_or)
    remote['flags'] = torch.tensor([0, 0, 0], dtype=torch.int32)
    assert machine._reduce_any([False, True, False]) == (False, True, False)
    remote['flags'] = torch.tensor([1, 0, 1], dtype=torch.int32)
    assert machine._reduce_any([False, True, False]) == (True, True, True)


def test_is_unexpectedly_large_collects_samples_before_flagging(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    assert machine.is_unexpectedly_large(torch.tensor(1.0), 2.0, 'loss', num_samples=2) is False
    assert machine.is_unexpectedly_large(torch.tensor(4.0), 2.0, 'loss', num_samples=2) is False
    assert machine.max_values['loss'] == 4.0
    assert machine.large_value_counts['loss'] == 2


def test_is_unexpectedly_large_flags_values_at_or_above_the_threshold(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    for value in (1.0, 2.0):
        machine.is_unexpectedly_large(torch.tensor(value), 2.0, 'loss', num_samples=2)
    assert machine.max_values['loss'] == 2.0
    assert machine.is_unexpectedly_large(torch.tensor(4.0), 2.0, 'loss', num_samples=2) is True
    assert machine.is_unexpectedly_large(torch.tensor(3.999), 2.0, 'loss', num_samples=2) is False


def test_is_unexpectedly_large_ignores_nan_and_inf(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    assert machine.is_unexpectedly_large(torch.tensor(float('nan')), 2.0, 'loss') is False
    assert machine.is_unexpectedly_large(torch.tensor(float('inf')), 2.0, 'loss') is False
    assert machine.large_value_counts == {}


def test_is_unexpectedly_large_resamples_the_context(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    machine.is_unexpectedly_large(torch.tensor(4.0), 2.0, 'loss', num_samples=1)
    assert (
        machine.is_unexpectedly_large(torch.tensor(4.0), 2.0, 'loss', num_samples=1, resample=True)
        is False
    )
    assert machine.large_value_counts['loss'] == 1


def test_sanitize_data_iterators_rejects_unwrapped_iterators(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    iterator = RerunDataIterator(iter(range(4)))
    assert machine._sanitize_data_iterators([iterator, None]) == [iterator]
    assert machine._sanitize_data_iterators(iterator) == [iterator]
    with pytest.raises(AssertionError, match='not wrapped with RerunDataIterator'):
        machine._sanitize_data_iterators([object()])


def test_validation_calls_are_counted_per_caller(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    iterator = RerunDataIterator(iter(range(4)))
    assert machine.should_run_forward_backward(iterator) is True
    machine.should_checkpoint_and_exit()
    machine.validate_result(torch.tensor(1.0), lambda value: False, 'loss')
    machine.validate_result(torch.tensor(1.0), lambda value: False, 'loss')
    assert machine.validation_counts[Caller('loss', 0)] == 2


def test_stats_are_reported_without_distributed_when_the_iteration_is_even(make_machine, caplog):
    machine = make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
    iterator = RerunDataIterator(iter(range(4)))
    assert machine.should_run_forward_backward(iterator) is True
    assert machine.should_run_forward_backward(iterator) is False
    assert machine.current_iteration == 0
    with caplog.at_level(logging.INFO, logger='megatron.core.rerun_state_machine'):
        machine._maybe_report_stats()
    assert 'Stats on computation determinism in validation calls' in caplog.text


def test_stats_are_reset_on_non_zero_ranks_when_distributed(make_machine, monkeypatch):
    machine = make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
    iterator = RerunDataIterator(iter(range(8)))
    gathered = []

    def fake_gather(obj, dst=None):
        gathered.append((obj, dst))

    distributed = rerun_state_machine.torch.distributed
    monkeypatch.setattr(distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(distributed, 'get_rank', lambda: 1)
    monkeypatch.setattr(distributed, 'gather_object', fake_gather)
    assert machine.should_run_forward_backward(iterator) is True
    assert machine.should_checkpoint_and_exit() == (False, False, 0)
    machine.validate_result(torch.tensor(3.0), lambda value: False, 'loss')
    assert machine.should_run_forward_backward(iterator) is True
    machine.validate_result(torch.tensor(6.0), lambda value: False, 'loss')
    assert machine.should_run_forward_backward(iterator) is False
    assert machine.should_checkpoint_and_exit() == (False, False, 0)
    assert len(gathered) == 1
    assert gathered[0][1] is None
    assert machine.stats[Caller('loss', 0)].samples == []


def test_stats_are_combined_on_rank_zero_when_distributed(make_machine, monkeypatch, caplog):
    machine = make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
    iterator = RerunDataIterator(iter(range(4)))
    assert machine.should_run_forward_backward(iterator) is True
    assert machine.should_run_forward_backward(iterator) is False

    def fake_gather(obj, dst=None):
        if dst is not None:
            for index in range(len(dst)):
                dst[index] = dict(obj)

    distributed = rerun_state_machine.torch.distributed
    monkeypatch.setattr(distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(distributed, 'gather_object', fake_gather)
    with caplog.at_level(logging.INFO, logger='megatron.core.rerun_state_machine'):
        machine._maybe_report_stats()
    assert 'Stats on computation determinism in validation calls' in caplog.text


def test_tracker_write_failures_are_logged_not_raised(make_machine, tmp_path, caplog):
    tracker = tmp_path / 'tracker.log'
    machine = make_machine(
        mode=RerunMode.VALIDATE_RESULTS, result_rejected_tracker_filename=str(tracker)
    )
    tracker.unlink()
    tracker.mkdir()
    with caplog.at_level(logging.ERROR, logger='megatron.core.rerun_state_machine'):
        machine._log_validation_error_to_file(
            rerun_state_machine.RerunValidationStatus.INITIAL_RUN, 1, 'loss'
        )
    assert 'Could not log validation error!' in caplog.text


def test_quick_stats_state_is_restored_through_the_pickle_hooks(make_machine):
    stats = QuickStats()
    stats.record(0.0)
    stats.record(2.5)
    restored = QuickStats()
    getattr(restored, '_QuickStats__setstate')(getattr(stats, '_QuickStats__getstate_')())
    assert restored.samples == [2.5]
    assert restored.zero_cnt == 1
    assert restored.pos == 1
    assert restored.max == 2.5


def test_destroy_rerun_state_machine_clears_the_global_instance(clean_singleton):
    rerun_state_machine.initialize_rerun_state_machine(mode=RerunMode.DISABLED)
    assert rerun_state_machine.get_rerun_state_machine().get_mode() == RerunMode.DISABLED
    destroy_rerun_state_machine()
    rerun_state_machine.initialize_rerun_state_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
    assert (
        rerun_state_machine.get_rerun_state_machine().get_mode()
        == RerunMode.REPORT_DETERMINISM_STATS
    )


def test_compare_floats_reports_matches_and_mismatches(make_machine):
    assert _compare_floats(torch.tensor(1.0), torch.tensor(1.0)) == COMPARISON_MATCH
    assert _compare_floats(torch.tensor(float('nan')), torch.tensor(float('nan'))) == (
        COMPARISON_MATCH
    )
    assert _compare_floats(torch.tensor(float('nan')), torch.tensor(1.0)) == COMPARISON_MISMATCH
    assert _compare_floats(torch.tensor(1.0), torch.tensor(float('nan'))) == COMPARISON_MISMATCH
    assert _compare_floats(torch.tensor(float('inf')), torch.tensor(1.0)) == COMPARISON_MISMATCH
    assert _compare_floats(torch.tensor(1.0), torch.tensor(float('inf'))) == COMPARISON_MISMATCH


def test_compare_floats_returns_the_relative_difference(make_machine):
    assert _compare_floats(torch.tensor(2.0), torch.tensor(4.0)) == pytest.approx(2 / 3)
    assert _compare_floats(torch.tensor(-2.0), torch.tensor(4.0)) == pytest.approx(6.0)
