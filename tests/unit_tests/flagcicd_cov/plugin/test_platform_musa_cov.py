# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformMUSA adapter: availability probe, musa forwarding, graph pool handle."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.platform import platform_musa
from megatron.plugin.platform.platform_musa import PlatformMUSA


def _fake_torch(musa):
    return SimpleNamespace(musa=musa)


def test_module_import_tolerates_missing_torch():
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "torch", None)
        importlib.reload(platform_musa)
        assert platform_musa.PlatformMUSA().device_name() == "musa"
    importlib.reload(platform_musa)


def test_is_available_false_when_musa_reports_no_devices(monkeypatch):
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(device_count=lambda: 0, is_available=lambda: True),
        raising=False,
    )
    assert PlatformMUSA().is_available() is False


def test_set_device_forwards_to_musa(monkeypatch):
    calls = []
    musa = SimpleNamespace(set_device=lambda index: calls.append(("set_device", index)))
    monkeypatch.setattr(platform_musa, "torch", _fake_torch(musa))
    platform = PlatformMUSA()
    platform.set_device(2)
    assert calls == [("set_device", 2)]


def test_is_fp16_supported_false_without_explicit_musa_check(monkeypatch):
    musa = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(platform_musa, "torch", _fake_torch(musa))
    assert PlatformMUSA().is_fp16_supported() is False


def test_amp_returns_none_when_musa_has_no_amp(monkeypatch):
    musa = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(platform_musa, "torch", _fake_torch(musa))
    assert PlatformMUSA().amp() is None


def test_graph_pool_handle_uses_musa_pool_handle(monkeypatch):
    musa = SimpleNamespace(graph_pool_handle=lambda: "musa-pool")
    monkeypatch.setattr(platform_musa, "torch", _fake_torch(musa))
    assert PlatformMUSA().graph_pool_handle() == "musa-pool"

    bare = SimpleNamespace()
    monkeypatch.setattr(platform_musa, "torch", _fake_torch(bare))
    assert PlatformMUSA().graph_pool_handle() is None


def test_build_extension_returns_torch_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    assert PlatformMUSA().build_extension() is BuildExtension
