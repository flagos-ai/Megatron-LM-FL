# Copyright (c) 2026, BAAI. All rights reserved.
"""Step batch-size schedule parsing and the deprecated ramp-up entry points."""

import logging

import pytest

from megatron.core import num_microbatches_calculator as calculator
from megatron.core.num_microbatches_calculator import (
    ConstantNumMicroBatchesCalculator,
    NumMicroBatchesCalculator,
    StepBatchsizeNumMicroBatchesCalculator,
)


class BareCalculator(NumMicroBatchesCalculator):
    def update(self, consumed_samples, consistency_check, verbose=False):
        return super().update(consumed_samples, consistency_check, verbose)


@pytest.fixture
def unset_global(monkeypatch):
    """Start from an unset global calculator and restore whatever was there afterwards."""
    monkeypatch.setattr(calculator, '_GLOBAL_NUM_MICROBATCHES_CALCULATOR', None)


@pytest.fixture
def messages(caplog):
    caplog.set_level(logging.INFO, logger=calculator.logger.name)
    return lambda: [record.getMessage() for record in caplog.records]


@pytest.mark.parametrize(
    ('text', 'value'),
    [
        ('12', 12),
        (' 3k ', 3000),
        ('1.5M', 1_500_000),
        ('2B', 2_000_000_000),
        ('1T', 1_000_000_000_000),
    ],
)
def test_numeric_values_accept_magnitude_suffixes(text, value):
    assert StepBatchsizeNumMicroBatchesCalculator._parse_numeric_value(text) == value


def test_schedule_is_sorted_and_accepts_commas():
    assert StepBatchsizeNumMicroBatchesCalculator._parse_schedule('1K:64, 0:32', None) == [
        (0, 32),
        (1000, 64),
    ]


def test_schedule_thresholds_convert_tokens_to_samples():
    assert StepBatchsizeNumMicroBatchesCalculator._parse_schedule('0:32 1K:64', 10) == [
        (0, 32),
        (100, 64),
    ]


def test_schedule_entry_without_separator_is_rejected():
    with pytest.raises(
        ValueError, match='Invalid schedule entry "64". Expected format: "THRESHOLD:BATCH_SIZE"'
    ):
        StepBatchsizeNumMicroBatchesCalculator._parse_schedule('0:32 64', None)


def test_token_schedule_is_logged_on_rank_zero_and_drives_updates(messages):
    schedule = StepBatchsizeNumMicroBatchesCalculator(
        micro_batch_size=2,
        data_parallel_size=4,
        decrease_batch_size_if_needed=False,
        rank=0,
        schedule='0:32 1K:64',
        seq_length=10,
    )
    assert schedule.schedule == [(0, 32), (100, 64)]
    assert schedule.global_batch_size == 64
    assert schedule.get() == 4
    assert schedule.get_current_global_batch_size() == 32
    assert '  >= 1,000 tokens (100 samples) -> batch_size=64, num_microbatches=8' in messages()
    assert '  seq_length: 10 (thresholds interpreted as tokens)' in messages()
    schedule.update(100, True)
    assert (
        schedule.get(),
        schedule.get_current_global_batch_size(),
        schedule.get_current_running_global_batch_size(),
    ) == (8, 64, 64)
    schedule.update(99, True)
    assert schedule.get() == 4


@pytest.mark.parametrize(
    ('schedule', 'message'),
    [
        ('5:32', 'first schedule entry must have threshold 0, got 5'),
        ('0:32 0:64', 'schedule thresholds must be strictly increasing, got 0 before 0'),
        ('0:0', 'batch size must be positive, got 0'),
        ('', 'schedule must have at least one entry'),
    ],
)
def test_schedule_validation_rejects_malformed_schedules(schedule, message):
    with pytest.raises(AssertionError, match=message):
        StepBatchsizeNumMicroBatchesCalculator(2, 4, False, 1, schedule)


def test_consistency_check_requires_divisible_batch_size():
    schedule = StepBatchsizeNumMicroBatchesCalculator(2, 3, False, 1, '0:32')
    with pytest.raises(AssertionError, match=r'current global batch size \(32\) is not divisible'):
        schedule.update(0, True)


def test_step_schedule_cannot_decrease_batch_size():
    with pytest.raises(
        ValueError, match='Step batch size schedules do not support decrease_batch_size_if_needed'
    ):
        StepBatchsizeNumMicroBatchesCalculator(2, 4, True, 1, '0:32')


def test_builder_selects_step_schedule_and_validates_arguments():
    built = calculator._build_num_microbatches_calculator(1, None, 2, 4, False, '0:32', None)
    assert isinstance(built, StepBatchsizeNumMicroBatchesCalculator)
    assert built.get() == 4
    with pytest.raises(
        ValueError,
        match='Cannot specify both --step-batch-size-schedule and '
        '--decrease-batch-size-if-needed',
    ):
        calculator._build_num_microbatches_calculator(1, None, 2, 4, True, '0:32', None)
    with pytest.raises(AssertionError, match='--global-batch-size is required'):
        calculator._build_num_microbatches_calculator(1, None, 2, 4, False)


def test_init_warns_about_rampup_on_rank_zero_and_refuses_reinitialisation(unset_global, caplog):
    caplog.set_level(logging.WARNING, logger=calculator.logger.name)
    calculator.init_num_microbatches_calculator(0, [1, 2, 3], 32, 2, 4)
    assert [record.levelname for record in caplog.records] == ['WARNING']
    assert 'rampup_batch_size is deprecated' in caplog.records[0].getMessage()
    assert (
        calculator.get_num_microbatches(),
        calculator.get_micro_batch_size(),
        calculator.get_current_global_batch_size(),
        calculator.get_current_running_global_batch_size(),
    ) == (4, 2, 32, 32)
    with pytest.raises(AssertionError, match='num microbatches calculator is already initialized'):
        calculator.init_num_microbatches_calculator(1, None, 32, 2, 4)


def test_reconfigure_warns_only_on_rank_zero(unset_global, caplog):
    caplog.set_level(logging.WARNING, logger=calculator.logger.name)
    calculator.reconfigure_num_microbatches_calculator(1, [1], 64, 2, 4)
    assert caplog.records == []
    assert calculator.get_num_microbatches() == 8
    calculator.reconfigure_num_microbatches_calculator(0, [1], 32, 2, 4)
    assert len(caplog.records) == 1
    assert calculator.get_num_microbatches() == 4
    calculator.update_num_microbatches(0)
    calculator.unset_num_microbatches_calculator()
    assert calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None


def test_base_calculator_has_no_values_until_a_subclass_sets_them():
    bare = BareCalculator()
    assert bare.update(0, True) is None
    for getter in (
        bare.get,
        bare.get_current_global_batch_size,
        bare.get_micro_batch_size,
        bare.get_current_running_global_batch_size,
    ):
        with pytest.raises(AssertionError):
            getter()


def test_constant_calculator_rounds_down_when_allowed():
    constant = ConstantNumMicroBatchesCalculator(30, 2, 4, True, 0)
    assert (
        constant.get(),
        constant.get_current_global_batch_size(),
        constant.get_current_running_global_batch_size(),
        constant.get_micro_batch_size(),
    ) == (3, 30, 24, 2)
