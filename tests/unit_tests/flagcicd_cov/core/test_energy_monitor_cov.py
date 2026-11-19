# Copyright (c) 2026, BAAI. All rights reserved.
"""Energy accounting against a scripted NVML, with the all-reduce replaced."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import energy_monitor
from megatron.core.energy_monitor import EnergyMonitor


class ScriptedNvmlError(Exception):
    pass


@pytest.fixture
def nvml(monkeypatch):
    """NVML that returns scripted millijoule readings; all-reduce only records its input."""
    state = SimpleNamespace(readings=[], fail=False, init=0, shutdown=0, reduced=[])

    def total_energy(handle):
        if state.fail:
            raise ScriptedNvmlError()
        return state.readings.pop(0)

    def count(name):
        setattr(state, name, getattr(state, name) + 1)

    monkeypatch.setattr(energy_monitor, 'has_nvml', True)
    monkeypatch.setattr(energy_monitor, 'NVMLError', ScriptedNvmlError, raising=False)
    monkeypatch.setattr(energy_monitor, 'nvmlInit', lambda: count('init'), raising=False)
    monkeypatch.setattr(energy_monitor, 'nvmlShutdown', lambda: count('shutdown'), raising=False)
    monkeypatch.setattr(
        energy_monitor, 'nvmlDeviceGetHandleByIndex', lambda index: ('handle', index), raising=False
    )
    monkeypatch.setattr(
        energy_monitor, 'nvmlDeviceGetTotalEnergyConsumption', total_energy, raising=False
    )
    monkeypatch.setattr(
        energy_monitor,
        'cur_platform',
        SimpleNamespace(current_device=lambda: 3, device_name=lambda *args: 'cpu'),
    )
    monkeypatch.setattr(
        energy_monitor,
        'dist',
        SimpleNamespace(
            all_reduce=lambda tensor, op=None: state.reduced.append((tensor.tolist(), op)),
            ReduceOp=SimpleNamespace(SUM='SUM'),
        ),
    )
    return state


def test_setup_binds_the_current_device_and_shutdown_releases_nvml(nvml):
    monitor = EnergyMonitor()
    monitor.setup()
    assert monitor._handle == ('handle', 3)
    assert nvml.init == 1
    monitor.shutdown()
    assert nvml.shutdown == 1


def test_reading_without_a_handle_returns_the_last_reading(nvml):
    monitor = EnergyMonitor()
    monitor._last_energy = 42
    assert monitor._get_energy() == 42
    assert nvml.readings == []


def test_lap_adds_paused_and_running_energy_in_joules(nvml):
    nvml.readings = [1000, 4000, 9000]
    monitor = EnergyMonitor()
    monitor.setup()
    monitor.resume()
    assert monitor._last_energy == 1000
    monitor.pause()
    assert monitor._lap_energy == 3000
    assert monitor._last_energy == 1000
    assert monitor.lap() == 11.0
    assert monitor._total_energy == 11000
    assert monitor._lap_energy == 0
    assert monitor._last_energy == 9000
    assert nvml.reduced == [([11000], 'SUM')]


def test_total_is_reduced_across_ranks_in_joules(nvml):
    monitor = EnergyMonitor()
    monitor._total_energy = 2500
    assert monitor.get_total() == 2.5
    assert nvml.reduced == [([2500], 'SUM')]


def test_nvml_error_falls_back_to_the_last_reading(nvml):
    monitor = EnergyMonitor()
    monitor.setup()
    monitor._last_energy = 9000
    nvml.fail = True
    assert monitor._get_energy() == 9000
    assert monitor.lap() == 0.0
    assert nvml.reduced == [([0], 'SUM')]


def test_everything_is_a_no_op_without_nvml(monkeypatch, nvml):
    monkeypatch.setattr(energy_monitor, 'has_nvml', False)
    monitor = EnergyMonitor()
    monitor.setup()
    monitor.resume()
    monitor.pause()
    monitor.shutdown()
    assert monitor.lap() == 0.0
    assert monitor.get_total() == 0.0
    assert monitor._handle is None
    assert (nvml.init, nvml.shutdown, nvml.reduced) == (0, 0, [])
