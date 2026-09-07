# Copyright (c) FlagOS Team, BAAI Corporation.
#
# Installs MegaMoE kernel source files (.cuh) into deep_gemm's include
# directory so that the JIT compiler can resolve them at runtime.
#
# The JIT compiler resolves #include <deep_gemm/...> from deep_gemm's
# library_include_path (= <deep_gemm_package_root>/include/). Since our
# kernel implementations live in this plugin, we copy/symlink them into
# that directory at first import.

import os
import shutil
from pathlib import Path


def _get_deep_gemm_include_path() -> Path:
    """Return the include/ directory of the installed deep_gemm package."""
    import deep_gemm
    dg_root = Path(deep_gemm.__file__).parent
    include_path = dg_root / "include"
    if not include_path.exists():
        raise RuntimeError(
            f"deep_gemm include directory not found at {include_path}.\n"
            "Please reinstall deep_gemm or check your installation."
        )
    return include_path


def _get_our_include_path() -> Path:
    """Return this plugin's include/ directory."""
    return Path(__file__).parent / "include"


def install_kernel_sources():
    """Copy our .cuh files into deep_gemm's include tree if not already present.

    Files are placed at:
      <deep_gemm>/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh
      <deep_gemm>/include/deep_gemm/impls/sm90_bf16_mega_moe_backward.cuh
      <deep_gemm>/include/deep_gemm/layout/mega_moe.cuh
      <deep_gemm>/include/deep_gemm/layout/sym_buffer.cuh
      <deep_gemm>/include/deep_gemm/scheduler/mega_moe.cuh

    This is idempotent — files are only copied when the destination is missing
    or has a different size (indicating an update).
    """
    dg_include = _get_deep_gemm_include_path()
    our_include = _get_our_include_path()

    # Files to install (relative to include/)
    kernel_files = [
        "deep_gemm/impls/sm90_fp8_mega_moe.cuh",
        "deep_gemm/impls/sm90_bf16_mega_moe_backward.cuh",
        "deep_gemm/layout/mega_moe.cuh",
        "deep_gemm/layout/sym_buffer.cuh",
        "deep_gemm/scheduler/mega_moe.cuh",
    ]

    for rel_path in kernel_files:
        src = our_include / rel_path
        dst = dg_include / rel_path

        if not src.exists():
            raise RuntimeError(
                f"MegaMoE kernel source not found: {src}\n"
                "Plugin installation may be corrupted."
            )

        # Skip if already installed and same size
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Try symlink first (faster, doesn't duplicate data), fall back to copy
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            # Symlinks may not be supported (e.g., Windows without privileges)
            shutil.copy2(src, dst)
