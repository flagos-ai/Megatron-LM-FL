# Copyright (c) BAAI Corporation.

import os

SUPPORTED_PLATFORM_LIST = ['cuda', 'cpu']
cur_platform = None


def is_current_platform_supported():
    return get_platform().device_name() in SUPPORTED_PLATFORM_LIST


def get_platform():
    global cur_platform
    if cur_platform is not None:
        return cur_platform

    platform_name = None
    # 1. Detect whether there is override of Megatron-LM-FL platforms from environment variable.
    if "MG_PLATFORM" in os.environ.keys():
        platform_name = os.environ["MG_ACCELERATOR"]
        if platform_name == "cpu":
            pass
        elif platform_name not in SUPPORTED_PLATFORM_LIST:
            raise ValueError(f'MG_PLATFORM must be one of {SUPPORTED_PLATFORM_LIST}. '
                             f'Value "{platform_name}" is not supported')

    # 2. If no override, detect which platform to use automatically
    if platform_name is None:
        # We need a way to choose among different platform types.
        # Currently we detect which platform extension is installed
        # in the environment and use it if the installing answer is True.

        if platform_name is None:
            try:
                import torch

                # Determine if we are on a GPU or x86 CPU with torch.
                if torch.cuda.device_count() > 0 and torch.cuda.is_available():  #ignore-cuda
                    platform_name = "cuda"
            except (RuntimeError, ImportError) as e:
                # TODO need a more decent way to detect which accelerator to use, consider using nvidia-smi command for detection
                pass
        if platform_name is None:
            # cpu added as catch-all when accelerator detection fails
            platform_name = "cpu"

    # 3. Set cur_platform accordingly
    if platform_name == "cuda":
        from .platform_cuda import PlatformCUDA

        cur_platform = PlatformCUDA()
    elif platform_name == "cpu":
        from .platform_cpu import PlatformCPU

        cur_platform = PlatformCPU()
    
    return cur_platform


def set_platform(platform_obj):
    global cur_platform
    cur_platform = platform_obj