# Copyright (c) 2026, BAAI. All rights reserved.
"""register_platforms: vendor registration when available, npu registry_patch hook."""

import importlib

import pytest

from megatron.plugin.platform import platform_cuda, platform_npu, platform_register


@pytest.mark.parametrize(
    "key, class_name",
    [
        pytest.param("musa", "PlatformMUSA", id="musa"),
        pytest.param("txda", "PlatformTXDA", id="txda"),
        pytest.param("enflame", "PlatformENFLAME", id="enflame"),
        pytest.param("kunlunxin", "PlatformKunLunXin", id="kunlunxin"),
        pytest.param("supa", "PlatformSUPA", id="supa"),
    ],
)
def test_register_platforms_registers_vendor_when_available(monkeypatch, key, class_name):
    module = importlib.import_module(f"megatron.plugin.platform.platform_{key}")
    platform_class = getattr(module, class_name)
    monkeypatch.setattr(platform_class, "is_available", lambda self: True)
    monkeypatch.setattr(platform_cuda.PlatformCUDA, "is_available", lambda self: False)
    monkeypatch.setattr(platform_register, "PLATFORMS", {})
    platform_register.register_platforms()
    registered = platform_register.PLATFORMS
    assert set(registered) == {"cpu", key}
    assert isinstance(registered[key], platform_class)
    assert registered[key].platform_name() == key


def test_register_platforms_calls_registry_patch_when_npu_available(monkeypatch):
    monkeypatch.setattr(platform_npu.PlatformNPU, "is_available", lambda self: True)
    monkeypatch.setattr(platform_cuda.PlatformCUDA, "is_available", lambda self: False)
    calls = []
    monkeypatch.setattr(platform_npu, "registry_patch", lambda: calls.append("patched"))
    monkeypatch.setattr(platform_register, "PLATFORMS", {})
    platform_register.register_platforms()
    registered = platform_register.PLATFORMS
    assert set(registered) == {"cpu", "npu"}
    assert isinstance(registered["npu"], platform_npu.PlatformNPU)
    assert calls == ["patched"]
