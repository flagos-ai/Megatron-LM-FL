# Copyright (c) FlagOS Team, BAAI Corporation.
"""
Build script for the SM90 MegaMoE C++ extension (_mega_moe_C).

Usage:
    cd megatron/plugin/mega_moe/
    python setup_ext.py build_ext --inplace

Requirements:
    - deep_gemm source tree (set DEEP_GEMM_ROOT env var, or it will try to auto-detect)
    - CUDA >= 12.3 (CUDA_HOME must be set)
    - PyTorch with CUDA support
"""
import os
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

# ============================================================================
# Resolve paths
# ============================================================================

THIS_DIR = Path(__file__).resolve().parent

# deep_gemm source root (contains csrc/, deep_gemm/include/, third-party/)
DEEP_GEMM_ROOT = os.environ.get("DEEP_GEMM_ROOT", "")
if not DEEP_GEMM_ROOT:
    # Try common relative locations
    candidates = [
        THIS_DIR.parents[3] / "DeepGEMM",           # sibling project
        Path.home() / "DeepGEMM",
        Path("/opt/DeepGEMM"),
    ]
    for c in candidates:
        if (c / "csrc" / "jit" / "compiler.hpp").exists():
            DEEP_GEMM_ROOT = str(c)
            break

if not DEEP_GEMM_ROOT or not Path(DEEP_GEMM_ROOT).exists():
    print("ERROR: Cannot find DeepGEMM source tree.", file=sys.stderr)
    print("  Set DEEP_GEMM_ROOT environment variable to the DeepGEMM repo root.", file=sys.stderr)
    print("  e.g.: export DEEP_GEMM_ROOT=/path/to/DeepGEMM", file=sys.stderr)
    sys.exit(1)

DEEP_GEMM_ROOT = Path(DEEP_GEMM_ROOT)
print(f"Using DeepGEMM source at: {DEEP_GEMM_ROOT}")

assert CUDA_HOME is not None, "CUDA_HOME is not set. Please set it to your CUDA toolkit path."

# ============================================================================
# Include directories
# ============================================================================

include_dirs = [
    # Our own csrc/ (for the apis/ and jit_kernels/ relative includes)
    str(THIS_DIR / "csrc"),
    # Our own include/ (for the .cuh kernel sources used by JIT — also needed for
    # deep_gemm/layout/mega_moe.cuh which is #included by host headers)
    str(THIS_DIR / "include"),
    # DeepGEMM's csrc/ — resolves shared infrastructure headers:
    # jit/compiler.hpp, jit/kernel_runtime.hpp, utils/exception.hpp, etc.
    str(DEEP_GEMM_ROOT / "csrc"),
    # DeepGEMM's include/ — resolves <deep_gemm/common/types.cuh> etc.
    str(DEEP_GEMM_ROOT / "deep_gemm" / "include"),
    # Third-party: cutlass, fmt
    str(DEEP_GEMM_ROOT / "third-party" / "cutlass" / "include"),
    str(DEEP_GEMM_ROOT / "third-party" / "fmt" / "include"),
    # CUDA
    os.path.join(CUDA_HOME, "include"),
]

# Filter out non-existent paths (tolerate missing third-party)
include_dirs = [d for d in include_dirs if os.path.isdir(d)]

# ============================================================================
# Compiler flags
# ============================================================================

cxx_flags = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
]

# Define DG_TENSORMAP_COMPATIBLE so the sm90 APIs are compiled
cxx_flags.append("-DDG_TENSORMAP_COMPATIBLE=1")

# ============================================================================
# Extension module
# ============================================================================

ext_modules = [
    CUDAExtension(
        name="_mega_moe_C",
        sources=[str(THIS_DIR / "csrc" / "python_api.cpp")],
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": [],  # No CUDA sources compiled at build time
        },
        libraries=["cudart", "nvrtc", "cublasLt", "cuda"],
        library_dirs=[os.path.join(CUDA_HOME, "lib64")],
    )
]

# ============================================================================
# Setup
# ============================================================================

setup(
    name="mega_moe_ext",
    version="0.1.0",
    description="SM90 MegaMoE C++ extension for Megatron-LM",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
