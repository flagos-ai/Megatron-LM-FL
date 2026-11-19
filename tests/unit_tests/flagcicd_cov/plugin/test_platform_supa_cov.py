# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformSUPA adapter: availability probe, supa forwarding, bf16 fallback."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.platform import platform_supa
from megatron.plugin.platform.platform_supa import PlatformSUPA


def _fake_torch(supa):
    return SimpleNamespace(supa=supa)


def test_module_import_tolerates_missing_torch():
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "torch", None)
        importlib.reload(platform_supa)
        assert platform_supa.PlatformSUPA().device_name() == "supa"
    importlib.reload(platform_supa)


def test_is_available_false_when_supa_reports_no_devices(monkeypatch):
    monkeypatch.setattr(
        torch,
        "supa",
        SimpleNamespace(device_count=lambda: 0, is_available=lambda: True),
        raising=False,
    )
    assert PlatformSUPA().is_available() is False


def test_set_device_forwards_to_supa(monkeypatch):
    calls = []
    supa = SimpleNamespace(set_device=lambda index: calls.append(("set_device", index)))
    monkeypatch.setattr(platform_supa, "torch", _fake_torch(supa))
    platform = PlatformSUPA()
    platform.set_device(2)
    assert calls == [("set_device", 2)]


def test_is_bf16_supported_false_when_probe_raises(monkeypatch):
    def probe():
        raise RuntimeError("supa bf16 probe unavailable")

    supa = SimpleNamespace(is_bf16_supported=probe)
    monkeypatch.setattr(platform_supa, "torch", _fake_torch(supa))
    assert PlatformSUPA().is_bf16_supported() is False


def test_amp_returns_none_when_supa_has_no_amp(monkeypatch):
    supa = SimpleNamespace()
    monkeypatch.setattr(platform_supa, "torch", _fake_torch(supa))
    assert PlatformSUPA().amp() is None


def test_build_extension_returns_torch_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    assert PlatformSUPA().build_extension() is BuildExtension
