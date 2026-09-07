# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import importlib
import platform
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


class CPUAdamLoader:
    """Load the x86-64 CPU Adam extension used by SlideFormer LayerAdam."""

    def __init__(self) -> None:
        architecture = platform.machine().lower()
        if architecture not in {"x86_64", "amd64"}:
            raise RuntimeError(
                "SlideFormer CPU Adam supports x86-64 only; "
                f"detected architecture {architecture}"
            )

    @staticmethod
    def _torch_major_minor() -> str:
        return ".".join(torch.__version__.split(".")[:2])

    @staticmethod
    def _cpu_flags() -> set[str]:
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                name, separator, value = line.partition(":")
                if separator and name.strip().lower() in {"flags", "features"}:
                    return set(value.split())
        except OSError:
            pass
        return set()

    def get_extension_path(self) -> str:
        project_root = Path(__file__).resolve().parent.parent
        source_hash = hashlib.sha256(str(project_root).encode()).hexdigest()[:8]
        build_dir = (
            Path.home()
            / ".cache"
            / "oom_optimizer"
            / "torch_extensions"
            / f"torch{self._torch_major_minor()}_cpu-{source_hash}"
        )
        build_dir.mkdir(parents=True, exist_ok=True)
        return str(build_dir)

    @staticmethod
    def get_sources() -> list[str]:
        source_dir = Path(__file__).parent / "csrc"
        return [str(source_dir / "cpu_adam.cpp"), str(source_dir / "cpu_adam_impl.cpp")]

    @staticmethod
    def get_include_dirs() -> list[str]:
        return [str(Path(__file__).parent / "csrc")]

    def get_compile_args(self) -> list[str]:
        flags = self._cpu_flags()
        if "avx512f" in flags:
            vector_macro = "-D__AVX512__"
        elif "avx2" in flags:
            vector_macro = "-D__AVX256__"
        else:
            vector_macro = "-D__SCALAR__"
        return [
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            vector_macro,
            "-O3",
            "-std=c++17",
            "-Wno-reorder",
            "-fopenmp",
            "-march=native",
        ]

    def load(self):
        try:
            return importlib.import_module("oom_optimizer._C.cpu_adam")
        except ImportError:
            pass

        build_dir = self.get_extension_path()
        compiled_before = (Path(build_dir) / "cpu_adam.so").exists()
        action = "Loading" if compiled_before else "Compiling"
        print(f"{action} CPU Adam optimizer...")

        started = time.perf_counter()
        module = load(
            name="cpu_adam",
            sources=self.get_sources(),
            extra_include_paths=self.get_include_dirs(),
            extra_cflags=self.get_compile_args(),
            build_directory=build_dir,
        )
        elapsed = time.perf_counter() - started
        action = "load" if compiled_before else "compile"
        print(f"CPU Adam {action} finished in {elapsed:.2f} seconds")
        return module
