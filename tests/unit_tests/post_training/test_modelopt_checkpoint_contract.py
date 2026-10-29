# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU regressions for the ModelOpt checkpoint/helper boundary."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest


@pytest.fixture
def checkpoint_api():
    # Execute the production functions without importing optional ModelOpt/CUDA packages.
    root = Path(__file__).resolve().parents[3]
    source = root / "megatron/post_training/checkpointing.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"has_modelopt_state", "load_modelopt_state"}
    ]
    native = root / "megatron/training/checkpointing.py"
    helper = next(
        node
        for node in ast.parse(native.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_base_checkpoint"
    )
    helper.body = [ast.Pass()]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            helper,
            *functions,
        ],
        type_ignores=[],
    )
    args = SimpleNamespace(ckpt_format="torch", load="default-checkpoint")
    namespace = {
        "get_args": lambda: args,
        "print_rank_0": Mock(),
        "mto": SimpleNamespace(restore_from_modelopt_state=Mock()),
        "get_sharded_load_dir": Mock(),
        "restore_sharded_modelopt_state": Mock(),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    namespace["_load_base_checkpoint"] = create_autospec(
        namespace["_load_base_checkpoint"]
    )
    return SimpleNamespace(args=args, **namespace)


@pytest.mark.parametrize(
    "state,expected", [(None, False), ({}, False), ({"modelopt_state": {}}, True)]
)
def test_inspect_uses_current_checkpoint_contract(checkpoint_api, state, expected):
    api = checkpoint_api
    api._load_base_checkpoint.return_value = (state, "checkpoint.pt", False, object())

    assert api.has_modelopt_state("inspect-checkpoint") is expected
    api._load_base_checkpoint.assert_called_once_with(
        "inspect-checkpoint", api.args, rank0=False
    )


@pytest.mark.parametrize("load_dir", [None, "override-checkpoint"])
@pytest.mark.parametrize("state", [None, {}, {"modelopt_state": {"version": 1}}])
def test_restore_uses_resolved_directory_and_four_element_result(
    checkpoint_api, load_dir, state
):
    api = checkpoint_api
    model = object()
    api._load_base_checkpoint.return_value = (state, "checkpoint.pt", False, object())

    assert api.load_modelopt_state(model, load_dir) is None
    api._load_base_checkpoint.assert_called_once_with(
        load_dir or api.args.load, api.args, rank0=False
    )
    if state and "modelopt_state" in state:
        api.mto.restore_from_modelopt_state.assert_called_once_with(
            model, state["modelopt_state"]
        )
    else:
        api.mto.restore_from_modelopt_state.assert_not_called()


@pytest.mark.parametrize("function", ["has_modelopt_state", "load_modelopt_state"])
def test_checkpoint_read_failure_retains_existing_fallback(checkpoint_api, function):
    api = checkpoint_api
    api._load_base_checkpoint.side_effect = OSError("checkpoint read failed")

    result = getattr(api, function)("checkpoint-or-model")

    assert result is (False if function == "has_modelopt_state" else None)
    api.mto.restore_from_modelopt_state.assert_not_called()
    api.print_rank_0.assert_called()


def test_sharded_inspection_keeps_directory_protocol(checkpoint_api, tmp_path):
    api = checkpoint_api
    api.args.ckpt_format = "torch_dist"
    api.get_sharded_load_dir.return_value = (tmp_path, "")
    (tmp_path / "modelopt_state").mkdir()

    assert api.has_modelopt_state("sharded-checkpoint") is True
    api.get_sharded_load_dir.assert_called_once_with("sharded-checkpoint")
    api._load_base_checkpoint.assert_not_called()


def test_sharded_restore_keeps_existing_loader(checkpoint_api, tmp_path):
    api = checkpoint_api
    api.args.ckpt_format = "torch_dist"
    api.get_sharded_load_dir.return_value = (tmp_path, "")
    model = object()

    api.load_modelopt_state(model, "sharded-checkpoint")

    api.restore_sharded_modelopt_state.assert_called_once_with([model], tmp_path)
    api._load_base_checkpoint.assert_not_called()
