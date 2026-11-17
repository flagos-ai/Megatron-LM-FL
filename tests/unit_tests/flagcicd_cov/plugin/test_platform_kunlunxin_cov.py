# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformKunLunXin: XPU/xpu-smi availability, modelopt mocks, idempotent XME init."""

import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

import pytest

from megatron.plugin.platform import platform_kunlunxin
from megatron.plugin.platform.platform_kunlunxin import PlatformKunLunXin


@pytest.mark.parametrize(
    "xpu_flag", [pytest.param("1", id="xpu-one"), pytest.param("True", id="xpu-true")]
)
def test_is_available_true_when_xpu_flag_set(monkeypatch, xpu_flag):
    monkeypatch.setenv("XPU", xpu_flag)
    assert PlatformKunLunXin().is_available() is True


def test_is_available_true_when_xpu_smi_present_without_nvidia_smi(monkeypatch):
    monkeypatch.delenv("XPU", raising=False)
    monkeypatch.setattr(
        platform_kunlunxin,
        "shutil",
        types.SimpleNamespace(which=lambda name: "/usr/bin/xpu-smi" if name == "xpu-smi" else None),
    )
    assert PlatformKunLunXin().is_available() is True


def test_is_available_false_without_xpu_indicators(monkeypatch):
    monkeypatch.delenv("XPU", raising=False)
    monkeypatch.setattr(
        platform_kunlunxin, "shutil", types.SimpleNamespace(which=lambda name: None)
    )
    assert PlatformKunLunXin().is_available() is False


def test_ensure_modelopt_mocks_replaces_bridge_modules(monkeypatch):
    for key in (
        "modelopt.torch.distill",
        "modelopt.torch.distill.plugins.megatron",
        "modelopt.torch.opt.plugins",
    ):
        monkeypatch.setitem(sys.modules, key, object())
    PlatformKunLunXin.ensure_modelopt_mocks()
    assert isinstance(sys.modules["modelopt.torch.distill"], MagicMock)
    assert isinstance(sys.modules["modelopt.torch.distill.plugins.megatron"], MagicMock)
    assert isinstance(sys.modules["modelopt.torch.opt.plugins"], MagicMock)


def test_ensure_xme_init_installs_mocks_and_is_idempotent(monkeypatch):
    for key in (
        "modelopt.torch.distill",
        "modelopt.torch.distill.plugins.megatron",
        "modelopt.torch.opt.plugins",
    ):
        monkeypatch.setitem(sys.modules, key, object())
    bridge = types.ModuleType("megatron.bridge")
    bridge.__spec__ = importlib.machinery.ModuleSpec("megatron.bridge", None)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    monkeypatch.setattr(platform_kunlunxin, "_XME_INITIALIZED", False)

    PlatformKunLunXin.ensure_xme_init()
    assert platform_kunlunxin._XME_INITIALIZED is True
    installed = sys.modules["modelopt.torch.distill"]
    assert isinstance(installed, MagicMock)

    PlatformKunLunXin.ensure_xme_init()
    assert sys.modules["modelopt.torch.distill"] is installed


def test_ensure_xme_init_forwards_plugin_init_arguments(monkeypatch):
    monkeypatch.setattr(platform_kunlunxin, "_XME_INITIALIZED", False)
    calls = []
    ext = types.ModuleType("xmegatron_ext")
    ext.megatron_plugin_init = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "xmegatron_ext", ext)
    PlatformKunLunXin.ensure_xme_init()
    assert calls == [{"use_version": "0.17.1", "check_version": True}]
    assert platform_kunlunxin._XME_INITIALIZED is True
