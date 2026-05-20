# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

from megatron.core import timers as timers_module


def test_timer_start_stop_elapsed_reset_and_active_time(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(timers_module.cur_platform, "synchronize", synchronize)
    monkeypatch.setattr(timers_module.time, "time", Mock(side_effect=[0.0, 10.0, 12.5]))

    timer = timers_module.Timer("unit")
    timer.start()
    timer.stop()

    assert timer.elapsed(reset=False) == pytest.approx(2.5)
    assert timer.active_time() == pytest.approx(2.5)
    assert synchronize.call_count == 2

    timer.reset()
    assert timer.elapsed(reset=False) == 0.0


def test_timer_barrier_uses_configured_group(monkeypatch):
    barrier = Mock()
    group = object()
    monkeypatch.setattr(timers_module.cur_platform, "synchronize", Mock())
    monkeypatch.setattr(timers_module.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(timers_module.time, "time", Mock(side_effect=[0.0, 1.0, 2.0]))

    timer = timers_module.Timer("barrier")
    timer.set_barrier_group(group)
    timer.start(barrier=True)
    timer.stop(barrier=True)

    assert barrier.call_count == 2
    barrier.assert_called_with(group=group)


def test_dummy_timer_rejects_elapsed_and_active_time():
    timer = timers_module.DummyTimer()

    timer.start()
    timer.stop()
    timer.reset()

    with pytest.raises(Exception, match="dummy timer should not be used"):
        timer.elapsed()

    with pytest.raises(Exception, match="active timer should not be used"):
        timer.active_time()


def test_timers_returns_real_timer_or_dummy_by_log_level():
    timers = timers_module.Timers(log_level=1, log_option="minmax")

    real_timer = timers("real", log_level=1)
    same_timer = timers("real", log_level=1)
    dummy_timer = timers("ignored", log_level=2)

    assert isinstance(real_timer, timers_module.Timer)
    assert same_timer is real_timer
    assert isinstance(dummy_timer, timers_module.DummyTimer)

    with pytest.raises(AssertionError, match="does not match already existing"):
        timers("real", log_level=0)

    with pytest.raises(AssertionError, match="larger than max supported"):
        timers("too-high", log_level=3)

    with pytest.raises(AssertionError, match="input log option"):
        timers_module.Timers(log_level=1, log_option="invalid")


def test_timers_minmax_and_max_strings(monkeypatch):
    timers = timers_module.Timers(log_level=2, log_option="minmax")
    monkeypatch.setattr(
        timers,
        "_get_global_min_max_time",
        Mock(return_value={"forward": (1.25, 2.5)}),
    )

    minmax_output = timers.get_all_timers_string(["forward"], normalizer=1.0)
    assert "(min, max) time across ranks" in minmax_output
    assert "forward" in minmax_output
    assert "(1.25, 2.50)" in minmax_output

    timers._log_option = "max"
    max_output = timers.get_all_timers_string(["forward"], normalizer=1.0)
    assert "max time across ranks" in max_output
    assert "2.50" in max_output


def test_timers_all_ranks_string(monkeypatch):
    timers = timers_module.Timers(log_level=2, log_option="all")
    monkeypatch.setattr(timers_module.torch.distributed, "get_world_size", Mock(return_value=2))
    monkeypatch.setattr(
        timers,
        "_get_elapsed_time_all_ranks",
        Mock(return_value=torch.tensor([[0.0, 0.1], [0.2, 0.0]])),
    )

    output = timers.get_all_timers_string(["forward", "backward"], normalizer=1.0)

    assert "times across ranks" in output
    assert "forward" in output
    assert "backward" in output
    assert "rank  1" in output


def test_timers_write_uses_summary_writer_like_object(monkeypatch):
    class FakeSummaryWriter:
        def __init__(self):
            self.add_scalar = Mock()

    timers = timers_module.Timers(log_level=2, log_option="minmax")
    monkeypatch.setattr(
        timers,
        "_get_global_min_max_time",
        Mock(return_value={"forward": (1.0, 2.0)}),
    )
    monkeypatch.setattr(timers_module, "SummaryWriter", FakeSummaryWriter)

    writer = FakeSummaryWriter()
    timers.write(["forward"], writer, iteration=7)

    writer.add_scalar.assert_called_once_with("forward-time", 2.0, 7)
