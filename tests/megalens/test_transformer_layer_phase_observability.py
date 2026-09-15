# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.transformer import transformer_layer

ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER_LAYER = ROOT / "megatron/core/transformer/transformer_layer.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _scope_name(node: ast.With) -> str | None:
    if len(node.items) != 1 or not isinstance(node.items[0].context_expr, ast.Call):
        return None
    call = node.items[0].context_expr
    name_position = 1 if _call_name(call) == "open_trace_scope" else 0
    if len(call.args) <= name_position or not isinstance(call.args[name_position], ast.Constant):
        return None
    return call.args[name_position].value


def test_transformer_layer_scopes_match_the_source_eager_forward_boundary() -> None:
    module = ast.parse(TRANSFORMER_LAYER.read_text(encoding="utf-8"))
    layer_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TransformerLayer"
    )
    forward = next(
        node
        for node in layer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )

    gate_assignment = next(
        node
        for node in forward.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "prepare_trace_scope"
    )
    assert isinstance(gate_assignment.targets[0], ast.Name)
    assert gate_assignment.targets[0].id == "transformer_layer_gate"
    assert len(gate_assignment.value.args) == 1
    assert isinstance(gate_assignment.value.args[0], ast.Constant)
    assert gate_assignment.value.args[0].value == "transformer_layer"
    assert gate_assignment.value.keywords == []

    branch = next(node for node in forward.body if isinstance(node, ast.If))
    assert len(branch.body) == 1
    outer_scope = branch.body[0]
    assert isinstance(outer_scope, ast.With)
    assert _scope_name(outer_scope) == "transformer_layer"
    assert outer_scope.items[0].context_expr.keywords == []
    assert [_scope_name(node) for node in outer_scope.body] == [
        "_forward_attention",
        "_forward_mlp",
    ]

    attention_scope, mlp_scope = outer_scope.body
    assert len(attention_scope.body) == 1
    attention_assignment = attention_scope.body[0]
    assert isinstance(attention_assignment, ast.Assign)
    assert isinstance(attention_assignment.value, ast.Call)
    assert isinstance(attention_assignment.value.func, ast.Attribute)
    assert attention_assignment.value.func.attr == "_forward_attention"

    assert len(mlp_scope.body) == 1
    mlp_assignment = mlp_scope.body[0]
    assert isinstance(mlp_assignment, ast.Assign)
    assert isinstance(mlp_assignment.value, ast.Call)
    assert isinstance(mlp_assignment.value.func, ast.Attribute)
    assert mlp_assignment.value.func.attr == "_forward_mlp"
    assert [keyword.arg for keyword in mlp_assignment.value.keywords] == [
        "padding_mask",
        "input_ids",
    ]

    assert len(branch.orelse) == 2
    assert [
        node.value.func.attr
        for node in branch.orelse
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
    ] == ["_forward_attention", "_forward_mlp"]
    final_return = forward.body[-1]
    assert isinstance(final_return, ast.Return)
    assert branch.end_lineno < final_return.lineno


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", name: str) -> None:
        self.sink = sink
        self.name = name

    def __enter__(self):
        self.sink.order.append(("B", self.name))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.sink.order.append(("E", self.name, exc_type))
        return False

    def get(self, key: str):
        return None

    def set(self, key: str, value: Any) -> bool:
        return True


class _RecordingSink:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.gate_calls: list[str] = []
        self.scope_calls: list[str] = []
        self.order: list[tuple[Any, ...]] = []

    def is_enabled(self, name: str) -> bool:
        self.gate_calls.append(name)
        return self.enabled

    def scope(self, name: str, **_kwargs):
        self.scope_calls.append(name)
        return _RecordingScope(self, name)


def _forward(model, hidden_states, *, inference_context, padding_mask, input_ids):
    return transformer_layer.TransformerLayer.forward(
        model,
        hidden_states,
        inference_context=inference_context,
        padding_mask=padding_mask,
        input_ids=input_ids,
    )


def test_transformer_layer_scopes_preserve_forward_backward_and_arguments() -> None:
    hidden_states = torch.tensor([1.0, 2.0], requires_grad=True)
    context = object()
    inference_context = object()
    padding_mask = object()
    input_ids = object()
    sink = _RecordingSink()

    def forward_attention(actual_hidden_states, **kwargs):
        sink.order.append(("call", "_forward_attention", actual_hidden_states, kwargs))
        return actual_hidden_states * 2, context

    def forward_mlp(actual_hidden_states, actual_inference_context, **kwargs):
        sink.order.append(
            ("call", "_forward_mlp", actual_hidden_states, actual_inference_context, kwargs)
        )
        return actual_hidden_states + 1

    model = SimpleNamespace(
        _forward_attention=forward_attention,
        _forward_mlp=forward_mlp,
    )
    install_trace_sink(sink)
    try:
        output, actual_context = _forward(
            model,
            hidden_states,
            inference_context=inference_context,
            padding_mask=padding_mask,
            input_ids=input_ids,
        )
        output.sum().backward()
    finally:
        reset_trace_sink()

    assert actual_context is context
    assert torch.equal(output, hidden_states.detach() * 2 + 1)
    assert torch.equal(hidden_states.grad, torch.full_like(hidden_states, 2))
    assert sink.gate_calls == ["transformer_layer", "_forward_attention", "_forward_mlp"]
    assert sink.scope_calls == ["transformer_layer", "_forward_attention", "_forward_mlp"]
    assert sink.order[0:2] == [
        ("B", "transformer_layer"),
        ("B", "_forward_attention"),
    ]
    assert sink.order[2][0:2] == ("call", "_forward_attention")
    assert sink.order[2][2] is hidden_states
    assert sink.order[2][3]["inference_context"] is inference_context
    assert sink.order[3:5] == [
        ("E", "_forward_attention", None),
        ("B", "_forward_mlp"),
    ]
    assert sink.order[5][0:2] == ("call", "_forward_mlp")
    assert sink.order[5][3] is inference_context
    assert sink.order[5][4] == {"padding_mask": padding_mask, "input_ids": input_ids}
    assert sink.order[6:] == [
        ("E", "_forward_mlp", None),
        ("E", "transformer_layer", None),
    ]


@pytest.mark.parametrize("failing_phase", ["_forward_attention", "_forward_mlp"])
def test_transformer_layer_scopes_preserve_exception_identity(failing_phase: str) -> None:
    sink = _RecordingSink()
    error = RuntimeError(f"{failing_phase} failed")

    def forward_attention(hidden_states, **_kwargs):
        if failing_phase == "_forward_attention":
            raise error
        return hidden_states, None

    def forward_mlp(hidden_states, _inference_context, **_kwargs):
        if failing_phase == "_forward_mlp":
            raise error
        return hidden_states

    model = SimpleNamespace(
        _forward_attention=forward_attention,
        _forward_mlp=forward_mlp,
    )
    install_trace_sink(sink)
    try:
        with pytest.raises(RuntimeError, match="failed") as raised:
            _forward(
                model,
                torch.ones(1),
                inference_context=None,
                padding_mask=None,
                input_ids=None,
            )
    finally:
        reset_trace_sink()

    assert raised.value is error
    assert sink.order[-1] == ("E", "transformer_layer", RuntimeError)
    assert ("E", failing_phase, RuntimeError) in sink.order
    if failing_phase == "_forward_attention":
        assert "_forward_mlp" not in sink.scope_calls


def test_transformer_layer_trace_off_uses_the_source_fast_path() -> None:
    hidden_states = torch.tensor([1.0, 2.0], requires_grad=True)
    context = object()
    calls: list[str] = []
    sink = _RecordingSink(enabled=False)

    def forward_attention(actual_hidden_states, **_kwargs):
        calls.append("_forward_attention")
        return actual_hidden_states * 3, context

    def forward_mlp(actual_hidden_states, _inference_context, **_kwargs):
        calls.append("_forward_mlp")
        return actual_hidden_states + 1

    model = SimpleNamespace(
        _forward_attention=forward_attention,
        _forward_mlp=forward_mlp,
    )
    install_trace_sink(sink)
    try:
        output, actual_context = _forward(
            model,
            hidden_states,
            inference_context=None,
            padding_mask=None,
            input_ids=None,
        )
        output.sum().backward()
    finally:
        reset_trace_sink()

    assert actual_context is context
    assert calls == ["_forward_attention", "_forward_mlp"]
    assert torch.equal(hidden_states.grad, torch.full_like(hidden_states, 3))
    assert sink.gate_calls == ["transformer_layer"]
    assert sink.scope_calls == []
