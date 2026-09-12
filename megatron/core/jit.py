# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_torch_min_version

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2


def noop_decorator(func):
    '''No-op decorator'''
    return func


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser
    try:
        if is_torch_min_version("2.2.0a0"):
            from megatron.plugin.platform.platform_manager import get_platform

            # JIT fusion (bias+gelu / bias+dropout+add / fused cross-entropy)
            # is a CUDA-only optimization. On non-CUDA platforms torch.compile
            # routes through the inductor, whose autotuner needs a platform
            # triton backend (triton.backends.mtgpu for MUSA) that isn't
            # guaranteed importable — keep the plain eager functions.
            if get_platform().device_name() == "cuda":
                jit_fuser = torch.compile
            else:
                jit_fuser = noop_decorator
    except ImportError:

        jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
