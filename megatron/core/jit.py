# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools

import torch

from megatron.core.utils import is_torch_min_version

# ``jit_fuser`` is a lazy decorator: it wraps the decorated function and
# resolves the active fuser (torch.compile / torch.jit.script / no-op) at the
# first call, not at module import time. The indirection is required because
# ``--disable-jit-fuser`` is parsed at runtime, after the many ``@jit_fuser``
# consumers have already been imported -- binding ``torch.compile`` eagerly at
# import time would make that flag a no-op for the already-decorated functions.
_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2, hence the upgrade to
# torch.compile below.


def noop_decorator(func):
    '''No-op decorator'''
    return func


def jit_fuser(func):
    '''Decorate a function to be JIT-fused.

    The active fuser is resolved at the first call and cached, so a function
    decorated here still respects a later ``disable_jit_fuser()`` call even
    though it was imported before training arguments were parsed.
    '''
    fused = None

    @functools.wraps(func)
    def inner(*args, **kwargs):
        nonlocal fused
        if fused is None:
            target = _fuser
            fused = target(func) if target is not noop_decorator else func
        return fused(*args, **kwargs)

    return inner


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global _fuser
    try:
        if is_torch_min_version("2.2.0a0"):
            _fuser = torch.compile
    except ImportError:

        _fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global _fuser
    _fuser = noop_decorator


enable_jit_fuser()
