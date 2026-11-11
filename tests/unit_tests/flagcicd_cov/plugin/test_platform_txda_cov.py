# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformTXDA adapter: availability probe, cuda forwarding, placeholders."""

import sys
from types import SimpleNamespace

import torch

from megatron.plugin.platform import platform_txda
from megatron.plugin.platform.platform_txda import PlatformTXDA


def _fake_torch(cuda):
    return SimpleNamespace(cuda=cuda, txda=cuda)


def test_is_available_false_when_txda_reports_no_devices(monkeypatch):
    monkeypatch.setattr(torch, "txda", SimpleNamespace(device_count=lambda: 0), raising=False)
    monkeypatch.setitem(sys.modules, "torch_txda", SimpleNamespace())
    assert PlatformTXDA().is_available() is False


def test_set_device_forwards_to_cuda(monkeypatch):
    calls = []
    cuda = SimpleNamespace(set_device=lambda index: calls.append(("set_device", index)))
    monkeypatch.setattr(platform_txda, "torch", _fake_torch(cuda))
    platform = PlatformTXDA()
    platform.set_device(2)
    assert calls == [("set_device", 2)]


def test_is_fp16_supported_false_without_explicit_cuda_check(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(platform_txda, "torch", _fake_torch(cuda))
    assert PlatformTXDA().is_fp16_supported() is False


def test_amp_returns_none_when_cuda_has_no_amp(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(platform_txda, "torch", _fake_torch(cuda))
    assert PlatformTXDA().amp() is None


def test_graph_pool_handle_returns_none():
    assert PlatformTXDA().graph_pool_handle() is None


def test_build_extension_returns_torch_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    assert PlatformTXDA().build_extension() is BuildExtension
