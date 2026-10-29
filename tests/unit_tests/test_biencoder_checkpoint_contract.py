# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Load real CPU checkpoint files through the production biencoder functions."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("custom_path", [False, True])
@pytest.mark.parametrize("selection", ["full", "query", "context"])
def test_biencoder_loads_requested_checkpoint(
    tmp_path, use_distributed_optimizer, custom_path, selection
):
    source = Path(__file__).resolve().parents[2] / "megatron/training/checkpointing.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "get_checkpoint_name",
            "get_checkpoint_tracker_filename",
            "load_biencoder_checkpoint",
        }
    ]
    args = SimpleNamespace(
        load=str(tmp_path / "default"),
        use_distributed_optimizer=use_distributed_optimizer,
    )
    barrier = Mock()
    distributed = SimpleNamespace(get_rank=lambda: 0, barrier=barrier)
    namespace = {
        "os": os,
        "get_args": lambda: args,
        "unwrap_model": lambda model: model,
        "open_file": open,
        "torch": SimpleNamespace(load=torch.load, distributed=distributed),
        "mpu": SimpleNamespace(
            get_data_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0,
            get_expert_model_parallel_world_size=lambda: 1,
            get_expert_model_parallel_rank=lambda: 0,
        ),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    load_dir = str(tmp_path / "override") if custom_path else args.load
    checkpoint = Path(namespace["get_checkpoint_name"](load_dir, 7, release=False))
    checkpoint.parent.mkdir(parents=True)
    Path(namespace["get_checkpoint_tracker_filename"](load_dir)).write_text("7")
    state = {
        "query_model": {"weight": torch.tensor([1.0])},
        "context_model": {"weight": torch.tensor([2.0])},
    }
    torch.save({"model": state}, checkpoint)
    model = [Mock()]

    result = namespace["load_biencoder_checkpoint"](
        model,
        only_query_model=selection == "query",
        only_context_model=selection == "context",
        custom_load_path=load_dir if custom_path else None,
    )

    assert result is model
    loaded = model[0].load_state_dict.call_args.args[0]
    expected_keys = (
        {"query_model", "context_model"}
        if selection == "full"
        else {selection + "_model"}
    )
    assert loaded.keys() == expected_keys
    model[0].load_state_dict.assert_called_once()
    for key in expected_keys:
        torch.testing.assert_close(loaded[key]["weight"], state[key]["weight"])
    barrier.assert_called_once_with()
