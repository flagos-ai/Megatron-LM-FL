# Copyright (c) BAAI Corporation.

import os
from .platform_register import PLATFORMS

cur_platform = None


def is_current_platform_supported():
    return get_platform().device_name() in PLATFORMS.keys()


def get_platform():
    global cur_platform
    if cur_platform is not None:
        return cur_platform

    # Both are checked before CUDA. The mlu case: the gpu_migration bridge makes
    # the CUDA surface (and thus PLATFORMS["cuda"].is_available()) report True on
    # MLU hosts. The npu case: torch_npu's transfer_to_npu shim
    # (TORCH_TRANSFER_TO_NPU=1) makes torch.cuda.is_available()/device_count()
    # report the NPU as "cuda", so the real device must win the selection. On a
    # host without the vendor's torch each check falls through (is_available False).
    if "mlu" in PLATFORMS.keys() and PLATFORMS["mlu"].is_available():
        cur_platform = PLATFORMS["mlu"]
        print(f"Megatron-LM-FL Platform: mlu Selected")
    elif "npu" in PLATFORMS.keys() and PLATFORMS["npu"].is_available():
        cur_platform = PLATFORMS["npu"]
        print(f"Megatron-LM-FL Platform: npu Selected")
    elif "cuda" in PLATFORMS.keys() and PLATFORMS["cuda"].is_available():
        cur_platform = PLATFORMS["cuda"]
        print(f"Megatron-LM-FL Platform: cuda Selected")
    elif "musa" in PLATFORMS.keys() and PLATFORMS["musa"].is_available():
        cur_platform = PLATFORMS["musa"]
        print(f"Megatron-LM-FL Platform: musa Selected")
    elif "txda" in PLATFORMS.keys() and PLATFORMS["txda"].is_available():
        cur_platform = PLATFORMS["txda"]
        print(f"Megatron-LM-FL Platform: txda Selected")
    elif "enflame" in PLATFORMS.keys() and PLATFORMS["enflame"].is_available():
        cur_platform = PLATFORMS["enflame"]
        print(f"Megatron-LM-FL Platform: enflame Selected")
    elif "kunlunxin" in PLATFORMS.keys() and PLATFORMS["kunlunxin"].is_available():
        cur_platform = PLATFORMS["kunlunxin"]
        print(f"Megatron-LM-FL Platform: kunlunxin Selected")
        # Deferred XME init (after platform selection to avoid circular import)
        cur_platform.ensure_xme_init()
    elif "ptpu" in PLATFORMS.keys() and PLATFORMS["ptpu"].is_available():
        # PT-PU does not alias torch.cuda, but like every vendor check it must
        # come before the CPU platform: platform_cpu.is_available() is
        # unconditionally True, so reaching it means no accelerator was found.
        cur_platform = PLATFORMS["ptpu"]
        print(f"Megatron-LM-FL Platform: ptpu Selected")
    elif "cpu" in PLATFORMS.keys() and PLATFORMS["cpu"].is_available():
        cur_platform = PLATFORMS["cpu"]
        print(f"Megatron-LM-FL Platform: cpu Selected")
    else:
        raise ValueError("No platform is available")
    
    return cur_platform


def set_platform(platform_obj):
    global cur_platform
    cur_platform = platform_obj
