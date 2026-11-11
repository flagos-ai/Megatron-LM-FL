# Copyright (c) 2026, BAAI. All rights reserved.
"""platform_manager selection: kunlunxin chosen with deferred XME init."""

import sys
import types

from megatron.plugin.platform import platform_kunlunxin, platform_manager
from megatron.plugin.platform.platform_kunlunxin import PlatformKunLunXin


def test_get_platform_runs_xme_init_when_kunlunxin_selected(monkeypatch):
    calls = []
    ext = types.ModuleType("xmegatron_ext")
    ext.megatron_plugin_init = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "xmegatron_ext", ext)
    monkeypatch.setattr(platform_kunlunxin, "_XME_INITIALIZED", False)
    kunlunxin = PlatformKunLunXin()
    monkeypatch.setattr(PlatformKunLunXin, "is_available", lambda self: True)
    monkeypatch.setattr(platform_manager, "PLATFORMS", {"kunlunxin": kunlunxin})
    monkeypatch.setattr(platform_manager, "cur_platform", None)
    selected = platform_manager.get_platform()
    assert selected is kunlunxin
    assert calls == [{"use_version": "0.17.1", "check_version": True}]
    assert platform_manager.cur_platform is kunlunxin
