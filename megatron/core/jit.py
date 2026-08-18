# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools

import torch

from megatron.core.utils import is_torch_min_version

# Whether functions decorated with @jit_fuser are compiled via torch.compile.
# Enabled by default to match upstream behavior; `--disable-jit-fuser` flips
# it off after CLI argument parsing. Decoration is deferred to first call so
# that the flag is honored even though the decorated modules are imported
# before the CLI args are parsed.
jit_fuser_enabled = True


def jit_fuser(func):
    '''Lazily compile `func` with torch.compile (or torch.jit.script).

    The decision is made at first call, not at decoration time. Decorated
    modules are imported before the CLI args are parsed, so eagerly binding
    torch.compile at import made `--disable-jit-fuser` ineffective: warmup
    still triggered torch.compile -> inductor, crashing on backends whose
    flagtree cannot compile those kernels (e.g. Hygon DTK).
    '''
    resolved = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal resolved
        if resolved is None:
            if jit_fuser_enabled:
                try:
                    if is_torch_min_version("2.2.0a0"):
                        resolved = torch.compile(func)
                    else:
                        resolved = torch.jit.script(func)
                except ImportError:
                    resolved = func
            else:
                resolved = func
        return resolved(*args, **kwargs)

    return wrapper


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser_enabled
    jit_fuser_enabled = True


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser_enabled
    jit_fuser_enabled = False
