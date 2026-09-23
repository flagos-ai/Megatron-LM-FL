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
            if get_platform().device_name() == "cuda" and not _is_metax_torch():
                jit_fuser = torch.compile
            else:
                jit_fuser = noop_decorator
    except ImportError:

        jit_fuser = noop_decorator


def _is_metax_torch():
    """Detect MetaX (MACA) torch builds.

    torch.compile on this stack recompiles jit_fuser'd ops every step
    (requires_grad flips between the frozen ViT path and the LM trunk,
    and between no_grad/grad under full recompute). Each recompile adds a
    dynamo/AOTAutograd cache entry whose example_value pins real activation
    tensors, producing a strictly linear GPU-memory leak (~0.8 GiB/iter on
    Qwen3.5-4B) until OOM. TORCHDYNAMO_DISABLE=1 confirms the diagnosis.
    Keep jit_fuser as a no-op on MetaX until that is fixed upstream.
    """
    try:
        version = getattr(torch, "__version__", "")
        if "metax" in version:
            return True
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) or ""
            if "MetaX" in name or "MACA" in name:
                return True
    except Exception:
        pass
    return False


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
