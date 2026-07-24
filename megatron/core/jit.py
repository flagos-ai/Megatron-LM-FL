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
            # Verify that the inductor backend can resolve triton's cache key.
            # Triton 3.7+ removed `triton_key` which torch._inductor expects,
            # causing torch.compile to fail at graph-cache time.
            try:
                from triton.compiler.compiler import triton_key  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                jit_fuser = noop_decorator
                return
            jit_fuser = torch.compile
    except ImportError:

        jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
