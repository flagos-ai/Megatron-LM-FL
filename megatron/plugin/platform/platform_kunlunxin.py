# Copyright (c) 2026, Baidu Inc. All rights reserved.
"""KunLunXin XPU Platform for Megatron-LM-FL.

Inherits from PlatformCUDA since XMLIR makes XPU appear as CUDA.
XME init is deferred to first use (via _ensure_xme_init) to avoid circular import
during platform registration (parallel_state calls get_platform() at module level).
"""

import os
import shutil

from .platform_cuda import PlatformCUDA

_XME_INITIALIZED = False


class PlatformKunLunXin(PlatformCUDA):

    def __init__(self):
        super().__init__()
        self._name = "kunlunxin"

    def is_available(self):
        """Detect KunLunXin XPU. Does NOT trigger XME init to avoid circular import."""
        xpu_flag = os.getenv("XPU", "")
        if xpu_flag in ("1", "True", "true", "TRUE"):
            return True
        if shutil.which("xpu-smi") is not None and shutil.which("nvidia-smi") is None:
            return True
        return False

    @staticmethod
    def ensure_xme_init():
        """Call once after get_platform() to init XME patches. Idempotent."""
        global _XME_INITIALIZED
        if _XME_INITIALIZED:
            return
        _XME_INITIALIZED = True
        try:
            from xmegatron_ext import megatron_plugin_init
            megatron_plugin_init(use_version="0.17.1", check_version=True)
        except Exception:
            pass

    def device_name(self, device_index=None):
        if device_index is None:
            return "kunlunxin"
        return f"kunlunxin:{device_index}"

    def current_device_name(self):
        return f"kunlunxin:{super().current_device()}"