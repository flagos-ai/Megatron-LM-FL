# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

PLATFORMS = {}


def register_platforms() -> None:
    """
    Register all platforms

    """
    # Register CPU Platform
    from .platform_cpu import PlatformCPU
    platform_cpu = PlatformCPU()
    if platform_cpu.is_available():
        PLATFORMS["cpu"] = platform_cpu # use lower keys: cpu
        print(f"Megatron-LM-FL Platform: cpu Registered")

    # Register CUDA Platform
    from .platform_cuda import PlatformCUDA
    platform_cuda = PlatformCUDA()
    if platform_cuda.is_available():
        PLATFORMS["cuda"] = platform_cuda # use lower keys: cuda
        print(f"Megatron-LM-FL Platform: cuda Registered")
