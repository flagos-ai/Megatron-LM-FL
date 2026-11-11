# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformENFLAME adapter: availability probe, gcu forwarding, graph pool handle."""

from types import SimpleNamespace

import torch

from megatron.plugin.platform import platform_enflame
from megatron.plugin.platform.platform_enflame import PlatformENFLAME


def _fake_torch(gcu):
    return SimpleNamespace(gcu=gcu)


def test_is_available_false_when_gcu_reports_no_devices(monkeypatch):
    monkeypatch.setattr(
        torch,
        "gcu",
        SimpleNamespace(device_count=lambda: 0, is_available=lambda: True),
        raising=False,
    )
    assert PlatformENFLAME().is_available() is False


def test_set_device_forwards_to_gcu(monkeypatch):
    calls = []
    gcu = SimpleNamespace(set_device=lambda index: calls.append(("set_device", index)))
    monkeypatch.setattr(platform_enflame, "torch", _fake_torch(gcu))
    platform = PlatformENFLAME()
    platform.set_device(2)
    assert calls == [("set_device", 2)]


def test_is_fp16_supported_false_without_explicit_gcu_check(monkeypatch):
    gcu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(platform_enflame, "torch", _fake_torch(gcu))
    assert PlatformENFLAME().is_fp16_supported() is False


def test_amp_returns_none_when_gcu_has_no_amp(monkeypatch):
    gcu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(platform_enflame, "torch", _fake_torch(gcu))
    assert PlatformENFLAME().amp() is None


def test_graph_pool_handle_uses_gcu_pool_handle(monkeypatch):
    gcu = SimpleNamespace(graph_pool_handle=lambda: "gcu-pool")
    monkeypatch.setattr(platform_enflame, "torch", _fake_torch(gcu))
    assert PlatformENFLAME().graph_pool_handle() == "gcu-pool"

    bare = SimpleNamespace()
    monkeypatch.setattr(platform_enflame, "torch", _fake_torch(bare))
    assert PlatformENFLAME().graph_pool_handle() is None


def test_build_extension_returns_torch_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    assert PlatformENFLAME().build_extension() is BuildExtension
