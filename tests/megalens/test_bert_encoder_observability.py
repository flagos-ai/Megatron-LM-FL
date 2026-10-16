# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from megatron.core.models.bert import bert_model
from megatron.core.observability import install_trace_sink, reset_trace_sink

ROOT = Path(__file__).resolve().parents[2]
BERT_MODEL = ROOT / "megatron/core/models/bert/bert_model.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


class _RecordingScope:
    def __init__(self, events: list[tuple[Any, ...]], name: str) -> None:
        self.events = events
        self.name = name

    def __enter__(self):
        self.events.append(("B", self.name, None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.events.append(("E", self.name, exc_type))
        return False


def _fake_model(hidden_states: torch.Tensor, encoder):
    return SimpleNamespace(
        pre_process=True,
        post_process=False,
        position_embedding_type="learned_absolute",
        embedding=lambda **_kwargs: hidden_states,
        encoder=encoder,
        bert_extended_attention_mask=lambda attention_mask: attention_mask[:, None, None, :],
        bert_position_ids=lambda input_ids: torch.arange(input_ids.shape[1]).unsqueeze(0),
    )


def _forward(model):
    return bert_model.BertModel.forward(
        model,
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
    )


def test_bert_encoder_scope_matches_the_source_call_boundary() -> None:
    module = ast.parse(BERT_MODEL.read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("megatron.core.observability", "trace_scope") in imports

    bert_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "BertModel"
    )
    forward = next(
        node
        for node in bert_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    scopes = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Call)
        and _call_name(node.items[0].context_expr) == "trace_scope"
        and len(node.items[0].context_expr.args) == 1
        and isinstance(node.items[0].context_expr.args[0], ast.Constant)
        and node.items[0].context_expr.args[0].value == "encoder"
    ]

    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.items[0].context_expr.keywords == []
    assert len(scope.body) == 1
    assignment = scope.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Attribute)
    assert isinstance(assignment.value.func.value, ast.Name)
    assert assignment.value.func.value.id == "self"
    assert assignment.value.func.attr == "encoder"
    assert [keyword.arg for keyword in assignment.value.keywords] == [
        "hidden_states",
        "attention_mask",
        "inference_context",
        "rotary_pos_emb",
    ]


def test_bert_encoder_scope_preserves_forward_backward_and_arguments(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    calls: list[dict[str, Any]] = []
    hidden_states = torch.tensor([[[1.0], [2.0]]], requires_grad=True)

    def encoder(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"] * 2

    monkeypatch.setattr(bert_model.parallel_state, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(bert_model, "trace_scope", lambda name: _RecordingScope(events, name))

    output = _forward(_fake_model(hidden_states, encoder))
    output.sum().backward()

    assert torch.equal(output, hidden_states.detach() * 2)
    assert torch.equal(hidden_states.grad, torch.full_like(hidden_states, 2))
    assert len(calls) == 1
    assert calls[0]["hidden_states"] is hidden_states
    assert calls[0]["inference_context"] is None
    assert calls[0]["rotary_pos_emb"] is None
    assert events == [("B", "encoder", None), ("E", "encoder", None)]


def test_bert_encoder_scope_preserves_exception_identity(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    error = RuntimeError("encoder failed")

    def encoder(**_kwargs):
        raise error

    monkeypatch.setattr(bert_model.parallel_state, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(bert_model, "trace_scope", lambda name: _RecordingScope(events, name))

    with pytest.raises(RuntimeError, match="encoder failed") as raised:
        _forward(_fake_model(torch.ones(1, 2, 1), encoder))

    assert raised.value is error
    assert events == [("B", "encoder", None), ("E", "encoder", RuntimeError)]


def test_bert_encoder_trace_off_does_not_open_a_scope(monkeypatch) -> None:
    class DisabledSink:
        def __init__(self) -> None:
            self.gate_calls: list[str] = []
            self.scope_calls = 0

        def is_enabled(self, name: str) -> bool:
            self.gate_calls.append(name)
            return False

        def scope(self, name: str, **_kwargs):
            self.scope_calls += 1
            raise AssertionError(f"disabled sink opened {name}")

    monkeypatch.setattr(bert_model.parallel_state, "is_pipeline_first_stage", lambda: True)
    hidden_states = torch.tensor([[[1.0], [2.0]]], requires_grad=True)
    sink = DisabledSink()
    install_trace_sink(sink)
    try:
        output = _forward(_fake_model(hidden_states, lambda **kwargs: kwargs["hidden_states"] * 3))
        output.sum().backward()
    finally:
        reset_trace_sink()

    assert torch.equal(hidden_states.grad, torch.full_like(hidden_states, 3))
    assert sink.gate_calls == ["encoder"]
    assert sink.scope_calls == 0
