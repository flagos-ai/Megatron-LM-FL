# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.
"""Iluvatar CoreX Platform for Megatron-LM-FL.

The CoreX torch build keeps the CUDA surface: ``torch.cuda`` is the device API,
devices are named ``cuda:N``, and the vendor part shows up only in
``get_device_name()``. So every PlatformCUDA method is the right one here and
this class inherits them unchanged.

What differs is the kernel set. flash-attn builds its kernels with nvcc for
NVIDIA architectures, so there is no build of it for this device: the varlen
kernels the dynamic-batching path normally calls are missing, and the flag_gems
paged kernel is the only route. ``supports_paged_attention`` answers True for
that reason, the same way the MLU and MUSA platforms do.

Because the CUDA surface is the real device API here, ``PlatformCUDA`` itself
would also be selected on a CoreX host, so ``platform_manager`` must prefer this
platform over the generic CUDA one (see the note there).
"""

import torch

from .platform_cuda import PlatformCUDA


class PlatformIluvatar(PlatformCUDA):

    def is_available(self):
        """Detect an Iluvatar CoreX device.

        CoreX exposes no module or torch attribute of its own, so the device
        name is what identifies it: the vendor build reports the part
        ("Iluvatar BI-V150", ...) where an NVIDIA build reports the board.
        """
        try:
            if torch.cuda.device_count() <= 0 or not torch.cuda.is_available():
                return False
            return "iluvatar" in torch.cuda.get_device_name(0).lower()
        except Exception:
            return False

    # Attention backend capabilities
    def supports_paged_attention(self) -> bool:
        # No flash-attn build for this device, so dynamic batching runs the
        # flag_gems paged kernel through the PlatformBase implementations. This
        # asserts the wiring, not that flag_gems is importable: the base
        # implementation names the missing package when it is not.
        return True
