# Copyright (c) FlagOS Team, BAAI Corporation.
# Dependency checks for MegaMoE plugin.

import importlib


def check_deep_gemm_installed() -> bool:
    """Check if deep_gemm package is installed (needed for JIT infrastructure)."""
    return importlib.util.find_spec("deep_gemm") is not None


def check_mega_moe_ext_built() -> bool:
    """Check if _mega_moe_C extension is compiled and loadable."""
    try:
        import _mega_moe_C  # noqa: F401
        return True
    except ImportError:
        return False


def require_mega_moe_ext():
    """Raise ImportError with instructions if the extension is not available."""
    if not check_deep_gemm_installed():
        raise ImportError(
            "MegaMoE requires the `deep_gemm` package for JIT kernel compilation.\n"
            "Install it with: pip install deep_gemm>=2.5.0\n"
            "Or build from source: https://github.com/DeepSeek-AI/DeepGEMM"
        )

    if not check_mega_moe_ext_built():
        raise ImportError(
            "MegaMoE C++ extension (_mega_moe_C) is not built.\n"
            "Please compile it first:\n"
            "  cd megatron/plugin/mega_moe/\n"
            "  export DEEP_GEMM_ROOT=/path/to/DeepGEMM\n"
            "  python setup_ext.py build_ext --inplace\n"
            "\n"
            "Requirements:\n"
            "  - DeepGEMM source tree (for shared C++ headers)\n"
            "  - CUDA >= 12.3 with nvcc\n"
            "  - SM90 (Hopper) GPU"
        )
