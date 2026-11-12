# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
import inspect
import textwrap

import pytest

from megatron.plugin.dualpipev.fb_overlap.overlap_funcs import fwdbwd


def _shared_backward_guard_actions(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    initializers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "has_backward_shared_experts"
            for target in node.targets
        )
    ]
    assert len(initializers) == 1
    assert ast.unparse(initializers[0].value) == (
        "bwd_layer_graph.shared_experts_graph[0] is not None"
    )
    guards = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "has_backward_shared_experts"
        ),
        key=lambda node: node.lineno,
    )
    assert len(guards) == 3
    assignments = [
        node
        for node in ast.walk(guards[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "backward_shared"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    calls = [
        tuple(
            node.func.id
            for node in sorted(
                (
                    node
                    for node in ast.walk(guard)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                ),
                key=lambda node: node.lineno,
            )
        )
        for guard in guards[1:]
    ]
    return ast.unparse(assignments[0].value), calls


@pytest.mark.parametrize(
    "function",
    (
        fwdbwd.transformer_layer_forward_dense_backward_moe_overlapping,
        fwdbwd.transformer_layer_forward_moe_backward_moe_overlapping,
    ),
)
def test_dualpipev_combined_backward_guards_optional_shared_experts(function) -> None:
    assert _shared_backward_guard_actions(function) == (
        "bwd_layer_graph.shared_experts_graph[1].grad",
        [
            ("turn_shared_experts_delay_wgrad_compute", "run_graph_backward"),
            ("call_shared_experts_backward_dw", "turn_shared_experts_delay_wgrad_compute"),
        ],
    )
