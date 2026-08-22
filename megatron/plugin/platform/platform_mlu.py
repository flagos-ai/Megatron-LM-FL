# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.
"""Cambricon MLU Platform for Megatron-LM-FL.

Inherits from PlatformCUDA since torch_mlu's ``gpu_migration`` module aliases
the ``torch.cuda.*`` surface (is_available, device_count, default_generators,
Stream, Event, RNG, memory, ...) onto MLU devices — the same bridge a
standalone cambricon shim used to provide. Importing the bridge at module load
time makes every inherited PlatformCUDA method work unchanged on MLU, and keeps
``megatron/training/initialize.py``'s ``assert torch.cuda.is_available()``
satisfied.

``is_available()`` also forces device initialization so that
``torch.mlu.default_generators`` is non-empty before any tensor-parallel RNG
path indexes into it (``megatron/core/tensor_parallel/random.py`` uses
``cur_platform.default_generators[idx]``).

Because the bridge makes ``torch.cuda.is_available()`` report True on MLU hosts,
``platform_manager`` must prefer this platform over the generic CUDA one
(which would otherwise be picked first).
"""

import torch

try:
    import torch_mlu.utils.gpu_migration  # noqa: F401  (aliases torch.cuda.* onto MLU)
except ImportError:
    pass

from .platform_cuda import PlatformCUDA


class PlatformMLU(PlatformCUDA):

    def __init__(self):
        super().__init__()
        self._name = "mlu"

    def is_available(self):
        """Detect Cambricon MLU.

        Forces device init (idempotent) so ``torch.mlu.default_generators`` is
        populated before tensor-parallel RNG paths use it.
        """
        try:
            if torch.mlu.device_count() > 0:
                if not torch.mlu.is_initialized():
                    _ = torch.ones(1, device="mlu:0")
                    torch.mlu.synchronize()
                return True
            return False
        except Exception:
            return False

    def device_name(self, device_index=None):
        if device_index is None:
            return "mlu"
        return f"mlu:{device_index}"

    def current_device_name(self):
        return f"mlu:{torch.mlu.current_device()}"

    def on_accelerator(self, tensor):
        return str(tensor.device).startswith("mlu:")

    def visible_devices_envs(self):
        return ["MLU_VISIBLE_DEVICES"]
