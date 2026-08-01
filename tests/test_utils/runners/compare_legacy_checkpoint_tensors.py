# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare tensor leaves in trace-off/on legacy PP2 checkpoints."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_ITERATION = "iter_0000002"
_CHECKPOINT_FILES = tuple(
    Path(_ITERATION) / f"mp_rank_00_{pipeline_rank:03d}" / filename
    for pipeline_rank in range(2)
    for filename in ("model_optim_rng.pt", "distrib_optim.pt")
)

_PathSegment = tuple[str, str]
_TensorPath = tuple[_PathSegment, ...]


def _mapping_segment(key: object) -> _PathSegment:
    return ("key", repr(key))


def _index_segment(index: int) -> _PathSegment:
    return ("index", str(index))


def _display_path(path: _TensorPath) -> str:
    parts = ["$"]
    for _kind, value in path:
        parts.append(f"[{value}]")
    return "".join(parts)


def tensor_leaves(
    payload: object,
    path: _TensorPath = (),
) -> dict[_TensorPath, torch.Tensor]:
    """Return tensors reachable through mappings, lists, and tuples."""

    if isinstance(payload, torch.Tensor):
        return {path: payload}

    leaves: dict[_TensorPath, torch.Tensor] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            leaves.update(tensor_leaves(value, path + (_mapping_segment(key),)))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            leaves.update(tensor_leaves(value, path + (_index_segment(index),)))
    return leaves


def _load_tensor_leaves(path: Path) -> dict[_TensorPath, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tensor_leaves(payload)


def _tensor_mismatch(
    path: _TensorPath,
    trace_off: torch.Tensor,
    trace_on: torch.Tensor,
) -> dict[str, Any] | None:
    differences: dict[str, Any] = {}
    if trace_off.dtype != trace_on.dtype:
        differences["dtype"] = {
            "trace_off": str(trace_off.dtype),
            "trace_on": str(trace_on.dtype),
        }
    if trace_off.shape != trace_on.shape:
        differences["shape"] = {
            "trace_off": list(trace_off.shape),
            "trace_on": list(trace_on.shape),
        }
    if trace_off.layout != trace_on.layout:
        differences["layout"] = {
            "trace_off": str(trace_off.layout),
            "trace_on": str(trace_on.layout),
        }
    if not differences and not torch.equal(trace_off, trace_on):
        differences["value"] = "torch.equal returned false"
    if not differences:
        return None
    return {"path": _display_path(path), "differences": differences}


def _compare_file(
    trace_off_run: Path,
    trace_on_run: Path,
    relative_path: Path,
) -> dict[str, Any]:
    trace_off_path = trace_off_run / "checkpoints" / relative_path
    trace_on_path = trace_on_run / "checkpoints" / relative_path
    load_errors: dict[str, str] = {}
    leaves: dict[str, dict[_TensorPath, torch.Tensor]] = {}
    for mode, path in (("trace_off", trace_off_path), ("trace_on", trace_on_path)):
        try:
            leaves[mode] = _load_tensor_leaves(path)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            load_errors[mode] = f"{type(error).__name__}: {error}"

    result: dict[str, Any] = {
        "checkpoint": relative_path.as_posix(),
        "passed": False,
    }
    if load_errors:
        result["load_errors"] = load_errors
        return result

    trace_off_leaves = leaves["trace_off"]
    trace_on_leaves = leaves["trace_on"]
    trace_off_paths = set(trace_off_leaves)
    trace_on_paths = set(trace_on_leaves)
    missing_in_trace_on = sorted(
        (_display_path(path) for path in trace_off_paths - trace_on_paths)
    )
    missing_in_trace_off = sorted(
        (_display_path(path) for path in trace_on_paths - trace_off_paths)
    )
    mismatches = [
        mismatch
        for path in sorted(trace_off_paths & trace_on_paths, key=_display_path)
        if (
            mismatch := _tensor_mismatch(
                path,
                trace_off_leaves[path],
                trace_on_leaves[path],
            )
        )
        is not None
    ]
    result.update(
        {
            "tensor_count": {
                "trace_off": len(trace_off_leaves),
                "trace_on": len(trace_on_leaves),
            },
            "missing_in_trace_on": missing_in_trace_on,
            "missing_in_trace_off": missing_in_trace_off,
            "mismatches": mismatches,
        }
    )
    result["passed"] = not (
        missing_in_trace_on or missing_in_trace_off or mismatches
    )
    return result


def compare_runs(trace_off_run: Path, trace_on_run: Path) -> dict[str, Any]:
    """Compare the fixed iteration-2 legacy checkpoint tensor set."""

    files = [
        _compare_file(trace_off_run, trace_on_run, relative_path)
        for relative_path in _CHECKPOINT_FILES
    ]
    return {
        "passed": all(file_result["passed"] for file_result in files),
        "trace_off_run": str(trace_off_run),
        "trace_on_run": str(trace_on_run),
        "iteration": _ITERATION,
        "files": files,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare trace-off/on legacy checkpoint tensor leaves."
    )
    parser.add_argument("--trace-off-run", required=True, type=Path)
    parser.add_argument("--trace-on-run", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_runs(args.trace_off_run, args.trace_on_run)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
