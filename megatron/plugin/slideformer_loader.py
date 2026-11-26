# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Select one SlideFormer implementation for every FL training hook."""
from importlib import import_module
import json
import os
from pathlib import Path


def _implementation():
    mode = os.getenv("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "in_tree")
    if mode not in {"in_tree", "package"}:
        raise ValueError("MEGATRON_SLIDEFORMER_IMPLEMENTATION must be in_tree or package")
    # Explicit disable remains usable even when the optional wheel is absent.
    enabled = os.getenv("MEGATRON_SLIDEFORMER_ENABLE", os.getenv("FLAGSCALE_SLIDEFORMER_MEGATRON", "0"))
    if enabled.lower() not in {"1", "true", "yes", "on"}:
        return "megatron.plugin.slideformer"
    return "slideformer_megatron" if mode == "package" else "megatron.plugin.slideformer"


def load_slideformer(submodule=None):
    name = _implementation()
    try:
        api = import_module(name)
    except ImportError as error:
        raise ImportError(f"Cannot load requested SlideFormer implementation {name}: {error}") from error
    if name == "slideformer_megatron" and getattr(api, "API_VERSION", None) != 1:
        raise RuntimeError("Installed slideformer-megatron does not provide FL adapter API_VERSION=1")
    return import_module(f"{name}.{submodule}") if submodule else api


def implementation_report():
    api = load_slideformer()
    report = {"module": api.__name__, "file": api.__file__, "api_version": getattr(api, "API_VERSION", None)}
    manifest = Path(api.__file__).with_name("source_manifest.json")
    if manifest.is_file():
        report["source_commit"] = json.loads(manifest.read_text())["source_commit"]
    return report
