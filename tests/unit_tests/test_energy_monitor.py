# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core import energy_monitor


def test_pause_and_resume_skip_nvml_before_setup(monkeypatch):
    monitor = energy_monitor.EnergyMonitor()
    monitor._last_energy = 17

    monkeypatch.setattr(energy_monitor, "has_nvml", True)

    def unexpected_nvml_call(_handle):
        raise AssertionError("NVML must not receive an uninitialized device handle")

    monkeypatch.setattr(
        energy_monitor,
        "nvmlDeviceGetTotalEnergyConsumption",
        unexpected_nvml_call,
        raising=False,
    )

    monitor.pause()
    monitor.resume()

    assert monitor._lap_energy == 0
    assert monitor._last_energy == 17
