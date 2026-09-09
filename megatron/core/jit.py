# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_torch_min_version
from megatron.plugin.platform import get_platform

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
            jit_fuser = torch.compile
    except ImportError:

        jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


# torch.compile jit fusion is only available on CUDA. On non-CUDA backends
# (e.g. MUSA) inductor autotuning requires a Triton backend that is not
# shipped, so fall back to the no-op decorator.
enable_jit_fuser()
if get_platform().device_name() != "cuda":
    disable_jit_fuser()
