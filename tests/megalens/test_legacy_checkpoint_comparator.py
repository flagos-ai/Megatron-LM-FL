# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tests.test_utils.runners import compare_legacy_checkpoint_tensors as comparator


def _write_checkpoints(run_root: Path, payload: object) -> None:
    for relative_path in comparator._CHECKPOINT_FILES:
        path = run_root / "checkpoints" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


def test_cli_accepts_equal_tensor_leaves_and_ignores_other_state(
    tmp_path: Path,
    capsys,
) -> None:
    trace_off_run = tmp_path / "trace-off"
    trace_on_run = tmp_path / "trace-on"
    common = {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": [torch.tensor([3], dtype=torch.int64)],
    }
    _write_checkpoints(
        trace_off_run,
        {
            **common,
            "args": argparse.Namespace(hidden=torch.tensor([1.0])),
            "numpy_rng": np.array([1]),
            "iteration": 2,
        },
    )
    _write_checkpoints(
        trace_on_run,
        {
            **common,
            "args": argparse.Namespace(hidden=torch.tensor([9.0])),
            "numpy_rng": np.array([9]),
            "iteration": 99,
        },
    )

    returncode = comparator.main(
        [
            "--trace-off-run",
            str(trace_off_run),
            "--trace-on-run",
            str(trace_on_run),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert returncode == 0
    assert report["passed"] is True
    assert len(report["files"]) == 4
    assert all(
        file_result["tensor_count"] == {"trace_off": 2, "trace_on": 2}
        for file_result in report["files"]
    )


def test_cli_reports_path_dtype_shape_layout_and_value_differences(
    tmp_path: Path,
    capsys,
) -> None:
    trace_off_run = tmp_path / "trace-off"
    trace_on_run = tmp_path / "trace-on"
    equal_payload = {"weight": torch.tensor([1.0])}
    _write_checkpoints(trace_off_run, equal_payload)
    _write_checkpoints(trace_on_run, equal_payload)

    relative_path = comparator._CHECKPOINT_FILES[0]
    trace_off_payload = {
        "only_off": torch.tensor([1.0]),
        "dtype": torch.tensor([1], dtype=torch.int64),
        "shape": torch.ones(2),
        "layout": torch.sparse_coo_tensor([[0]], [1.0], (1,)),
        "value": torch.tensor([1.0]),
    }
    trace_on_payload = {
        "only_on": torch.tensor([1.0]),
        "dtype": torch.tensor([1.0], dtype=torch.float32),
        "shape": torch.ones(1),
        "layout": torch.ones(1),
        "value": torch.tensor([2.0]),
    }
    torch.save(
        trace_off_payload,
        trace_off_run / "checkpoints" / relative_path,
    )
    torch.save(
        trace_on_payload,
        trace_on_run / "checkpoints" / relative_path,
    )

    returncode = comparator.main(
        [
            "--trace-off-run",
            str(trace_off_run),
            "--trace-on-run",
            str(trace_on_run),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    failed_file = report["files"][0]
    mismatches = {
        item["path"]: set(item["differences"])
        for item in failed_file["mismatches"]
    }

    assert returncode == 1
    assert report["passed"] is False
    assert failed_file["missing_in_trace_on"] == ["$['only_off']"]
    assert failed_file["missing_in_trace_off"] == ["$['only_on']"]
    assert mismatches == {
        "$['dtype']": {"dtype"},
        "$['layout']": {"layout"},
        "$['shape']": {"shape"},
        "$['value']": {"value"},
    }
