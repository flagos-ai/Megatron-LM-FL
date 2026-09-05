# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run the minimum import and file checks against an installed MegaLens wheel."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path


CORE_MODULES = (
    "megatron.core",
    "megatron.core.observability",
    "megatron.core.tensor_parallel.observability",
    "megatron.core.transformer.attention",
    "megatron.core.transformer.mlp",
    "megatron.core.transformer.moe.moe_layer",
    "megatron.core.transformer.moe.observability",
    "megatron.core.transformer.moe.router",
    "megatron.plugin.dualpipev.observability",
)

MEGALENS_MODULES = (
    # Trace aggregation and the Probe runtime bridge.
    "megatron.megalens.trace",
    "megatron.megalens.runtime",
    "megatron.megalens.core_adapter",
    "megatron.megalens.trace_aggregate",
    "megatron.megalens.data_loader",
    # Shared analysis support and retained parallelism analyzers.
    "megatron.megalens.event_catalog",
    "megatron.megalens.nested_aggregation",
    "megatron.megalens.pp_analyzer",
    "megatron.megalens.dp_analyzer",
    "megatron.megalens.tp_analyzer",
    "megatron.megalens.ep_analyzer",
    # Hybrid Analysis is included in the file manifest and loaded lazily by
    # the CLI because it requires the optional ``megalens`` dependencies.
    "megatron.megalens.analyzer",
)

REQUIRED_WHEEL_FILES = frozenset(
    {
        "megatron/core/observability.py",
        "megatron/core/tensor_parallel/observability.py",
        "megatron/core/transformer/moe/observability.py",
        "megatron/plugin/dualpipev/dualpipev_schedules.py",
        "megatron/plugin/dualpipev/observability.py",
        "megatron/megalens/__init__.py",
        "megatron/megalens/analyzer.py",
        "megatron/megalens/core_adapter.py",
        "megatron/megalens/data_loader.py",
        "megatron/megalens/dp_analyzer.py",
        "megatron/megalens/ep_analyzer.py",
        "megatron/megalens/event_catalog.py",
        "megatron/megalens/hardware_monitor.py",
        "megatron/megalens/hybrid_analyzer.py",
        "megatron/megalens/nested_aggregation.py",
        "megatron/megalens/paper_style.py",
        "megatron/megalens/pp_analyzer.py",
        "megatron/megalens/runtime.py",
        "megatron/megalens/trace.py",
        "megatron/megalens/trace_aggregate.py",
        "megatron/megalens/tp_analyzer.py",
        "megatron/megalens/utils.py",
    }
)

FORBIDDEN_WHEEL_PREFIXES = (
    "megatron/megalens/dp_lifecycle.py",
    "megatron/megalens/docs/",
    "megatron/megalens/migration_checks/",
    "megatron/megalens/pig",
    "megatron/megalens/progressive_decoupler.py",
    "megatron/megalens/mitigation_engine.py",
    "megatron/megalens/route_capability.py",
    "megatron/megalens/all_grads_capability.py",
    "megatron/megalens/moe_route_capability.py",
)


def _loaded_megalens_modules() -> list[str]:
    return sorted(name for name in sys.modules if name.startswith("megatron.megalens"))


def _import_core_without_megalens():
    initially_loaded = _loaded_megalens_modules()
    if initially_loaded:
        raise AssertionError(
            f"MegaLens was loaded before explicit Core imports: {initially_loaded}"
        )

    imported_core = [importlib.import_module(name) for name in CORE_MODULES]
    loaded_by_core = _loaded_megalens_modules()
    if loaded_by_core:
        raise AssertionError(f"Core imports loaded MegaLens modules: {loaded_by_core}")
    return imported_core


def _distribution_files(distribution: importlib.metadata.Distribution) -> set[str]:
    if distribution.files is None:
        raise AssertionError("wheel metadata does not expose an installed-file list")
    return {str(path).replace("\\", "/") for path in distribution.files}


def _assert_wheel_files(distribution: importlib.metadata.Distribution) -> None:
    files = _distribution_files(distribution)
    missing = sorted(REQUIRED_WHEEL_FILES - files)
    if missing:
        raise AssertionError(f"wheel is missing required files: {missing}")

    forbidden = sorted(
        path
        for path in files
        if any(path.startswith(prefix) for prefix in FORBIDDEN_WHEEL_PREFIXES)
    )
    if forbidden:
        raise AssertionError(f"wheel contains files outside the Probe-only package: {forbidden}")


def _assert_expected_version(
    distribution: importlib.metadata.Distribution,
    imported_core,
    expected_version: str | None,
) -> None:
    if expected_version is None:
        return
    imported_version = getattr(imported_core, "__version__", None)
    if distribution.version != expected_version or imported_version != expected_version:
        raise AssertionError(
            "installed version mismatch: "
            f"expected={expected_version!r}, metadata={distribution.version!r}, "
            f"imported={imported_version!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--require-source-root-absent", action="store_true")
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if args.require_source_root_absent and source_root.exists():
        raise AssertionError(f"source root is present during wheel check: {source_root}")

    imported_core = _import_core_without_megalens()
    imported_megalens = [importlib.import_module(name) for name in MEGALENS_MODULES]

    distribution = importlib.metadata.distribution("megatron-core")
    _assert_wheel_files(distribution)
    _assert_expected_version(distribution, imported_core[0], args.expected_version)

    print("Core imports did not load MegaLens")
    print(f"installed MegaLens origin: {imported_megalens[0].__file__}")
    print("wheel Trace, Probe, analysis imports and file manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
