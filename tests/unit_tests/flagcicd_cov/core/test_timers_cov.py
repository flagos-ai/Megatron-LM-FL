# Copyright (c) 2026, BAAI. All rights reserved.
"""Timer bookkeeping and cross-rank report formatting, with collectives replaced."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import timers
from megatron.core.timers import DummyTimer, Timer, TimerBase, Timers


class PassThroughTimer(TimerBase):
    def start(self, barrier=False):
        return super().start(barrier)

    def stop(self, barrier=False):
        return super().stop(barrier)

    def reset(self):
        return super().reset()

    def elapsed(self, reset=True, barrier=False):
        return super().elapsed(reset, barrier)


class RecordingSummaryWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, *args):
        self.scalars.append(args)


class RecordingWandb:
    def __init__(self):
        self.logged = []

    def log(self, *args):
        self.logged.append(args)


@pytest.fixture
def world(monkeypatch):
    """A two-rank world seen from rank 1; collectives only record that they were called."""
    state = SimpleNamespace(backend='nccl', barriers=[], gathers=[])
    monkeypatch.setattr(
        timers,
        'cur_platform',
        SimpleNamespace(current_device=lambda: 'cpu', synchronize=lambda: None),
    )
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda *args, **kwargs: 2)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda *args, **kwargs: 1)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda *args, **kwargs: state.backend)
    monkeypatch.setattr(
        torch.distributed, 'barrier', lambda *args, **kwargs: state.barriers.append(kwargs)
    )
    monkeypatch.setattr(
        timers,
        'dist_all_gather_func',
        lambda output, tensor: state.gathers.append((tuple(output.shape), tuple(tensor.shape))),
    )
    return state


def test_timer_base_defaults_do_nothing():
    timer = PassThroughTimer('x')
    assert timer.name == 'x'
    assert (timer.start(), timer.stop(), timer.reset(), timer.elapsed()) == (None, None, None, None)


def test_dummy_timer_ignores_control_and_refuses_readout():
    timer = DummyTimer()
    assert (timer.start(), timer.stop(), timer.reset()) == (None, None, None)
    with pytest.raises(Exception, match='dummy timer should not be used'):
        timer.elapsed()
    with pytest.raises(Exception, match='active timer should not be used'):
        timer.active_time()


def test_elapsed_restarts_a_running_timer_with_its_barrier_group(world):
    timer = Timer('z')
    timer.set_barrier_group('group')
    timer.start(barrier=True)
    value = timer.elapsed(barrier=True)
    assert value >= 0.0
    assert timer._started is True
    assert timer._elapsed == 0.0
    assert world.barriers == [{'group': 'group'}] * 3


def test_active_time_survives_reset(world):
    timer = Timer('z')
    timer.start()
    timer.stop()
    active = timer.active_time()
    timer.reset()
    assert active >= 0.0
    assert timer.active_time() == active
    assert timer.elapsed() == 0.0


def test_set_elapsed_injects_a_precomputed_value(world):
    timer = Timer('z')
    timer.set_elapsed(0.5)
    assert timer.elapsed(reset=False) == 0.5
    assert timer.elapsed() == 0.5
    assert timer.elapsed() == 0.0


def test_registry_returns_existing_timer_and_checks_its_level(world):
    registry = Timers(log_level=1, log_option='max')
    first = registry('a', log_level=0)
    assert registry('a') is first
    assert registry('a', log_level=0) is first
    with pytest.raises(
        AssertionError,
        match='input log level 1 does not match already existing log level 0 for a timer',
    ):
        registry('a', log_level=1)


def test_registry_defaults_to_max_level_and_hands_out_dummies_above_its_own(world):
    registry = Timers(log_level=1, log_option='max')
    assert isinstance(registry('implicit'), DummyTimer)
    assert isinstance(registry('explicit', log_level=2), DummyTimer)
    assert registry._log_levels == {}
    verbose = Timers(log_level=2, log_option='max')
    assert isinstance(verbose('implicit'), Timer)
    assert verbose._log_levels == {'implicit': 2}
    with pytest.raises(
        AssertionError, match='log level 3 is larger than max supported log level 2'
    ):
        registry('too-high', log_level=3)


def test_registry_rejects_unknown_log_option():
    with pytest.raises(AssertionError, match='input log option bad is invalid'):
        Timers(0, 'bad')


def test_max_report_formats_milliseconds_and_resets(world):
    registry = Timers(1, 'max')
    registry('a', 0).set_elapsed(0.5)
    assert registry.get_all_timers_string(['a']) == (
        'max time across ranks (ms):\n    ' + ('a' + ' ').ljust(48, '.') + ': 500.00'
    )
    assert registry('a').elapsed() == 0.0


def test_report_defaults_to_all_registered_timers_and_applies_normalizer(world):
    registry = Timers(1, 'max')
    registry('a', 0).set_elapsed(0.5)
    assert registry.get_all_timers_string(None, normalizer=2.0, reset=False) == (
        'max time across ranks (ms):\n    ' + ('a' + ' ').ljust(48, '.') + ': 250.00'
    )
    assert registry('a').elapsed(reset=False) == 0.5


def test_minmax_report_skips_timers_without_samples(world):
    registry = Timers(1, 'minmax')
    registry('a', 0).set_elapsed(0.25)
    assert registry.get_all_timers_string(['a', 'missing']) == (
        '(min, max) time across ranks (ms):\n    '
        + ('a' + ' ').ljust(48, '.')
        + ': (250.00, 250.00)'
    )
    assert registry.get_all_timers_string(['missing']) is None


def test_all_ranks_report_lists_only_ranks_with_samples(world):
    registry = Timers(1, 'all')
    registry('a', 0).set_elapsed(0.5)
    registry('b', 1)
    assert registry.get_all_timers_string(['a', 'b']) == (
        'times across ranks (ms):\n  a:\n     rank  1: 500.00'
    )
    assert registry.get_all_timers_string(['b']) is None


def test_unknown_log_option_at_report_time_raises(world):
    registry = Timers(1, 'max')
    registry._log_option = 'bogus'
    with pytest.raises(Exception, match='unknown timing log option bogus'):
        registry.get_all_timers_string(['a'])


def test_gather_returns_none_without_names(world):
    registry = Timers(1, 'max')
    assert registry._get_elapsed_time_all_ranks([], True, False) is None
    assert world.gathers == []


def test_gather_synchronises_and_moves_to_cpu_for_gloo(world):
    world.backend = 'cpu:gloo'
    registry = Timers(1, 'max')
    registry('a', 0).set_elapsed(0.5)
    times = registry._get_elapsed_time_all_ranks(['a', 'missing'], True, True)
    assert world.barriers == [{}]
    assert world.gathers == [((4,), (2,))]
    assert times.device.type == 'cpu'
    assert times.tolist() == [[0.0, 0.0], [0.5, 0.0]]


def test_write_sends_max_time_to_tensorboard(monkeypatch, world):
    monkeypatch.setattr(timers, 'SummaryWriter', RecordingSummaryWriter, raising=False)
    monkeypatch.setattr(timers, 'wandb', None, raising=False)
    registry = Timers(1, 'max')
    registry('a', 0).set_elapsed(0.5)
    writer = RecordingSummaryWriter()
    assert registry.write(['a'], writer, 7, reset=False) is None
    assert writer.scalars == [('a-time', 0.5, 7)]


def test_write_sends_max_time_to_wandb(monkeypatch, world):
    tracker = RecordingWandb()
    monkeypatch.setattr(timers, 'SummaryWriter', RecordingSummaryWriter, raising=False)
    monkeypatch.setattr(timers, 'wandb', tracker, raising=False)
    registry = Timers(1, 'max')
    registry('a', 0).set_elapsed(0.5)
    registry.write(['a'], tracker, 8, normalizer=0.5, reset=False)
    assert tracker.logged == [({'a-time': 1.0}, 8)]


def test_write_without_a_writer_still_resets_and_rejects_bad_normalizer(monkeypatch, world):
    monkeypatch.setattr(timers, 'SummaryWriter', RecordingSummaryWriter, raising=False)
    monkeypatch.setattr(timers, 'wandb', None, raising=False)
    registry = Timers(1, 'max')
    registry('a', 0).set_elapsed(0.5)
    assert registry.write(['a'], None, 9) is None
    assert registry('a').elapsed() == 0.0
    with pytest.raises(AssertionError):
        registry.write(['a'], None, 9, normalizer=0.0)
