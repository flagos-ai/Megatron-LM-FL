# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from megatron.core.models.gpt import gpt_model
from megatron.core.observability import install_trace_sink, reset_trace_sink

ROOT = Path(__file__).resolve().parents[2]
GPT_MODEL = ROOT / "megatron/core/models/gpt/gpt_model.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _fake_model(compute_loss):
    return SimpleNamespace(
        training=True,
        share_embeddings_and_output_weights=False,
        post_process=True,
        config=SimpleNamespace(mtp_num_layers=0),
        output_layer=lambda hidden_states, **_kwargs: (hidden_states + 1, None),
        _scale_logits=lambda logits: logits,
        compute_language_model_loss=compute_loss,
    )


def _postprocess(model, *, labels):
    return gpt_model.GPTModel._postprocess(
        model,
        hidden_states=torch.tensor([[[1.0]]]),
        input_ids=None,
        position_ids=None,
        labels=labels,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=False,
        runtime_gather_output=False,
    )


def test_gpt_loss_scope_matches_the_source_call_boundary() -> None:
    module = ast.parse(GPT_MODEL.read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("megatron.core.observability", "trace_scope") in imports

    gpt_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GPTModel"
    )
    postprocess = next(
        node
        for node in gpt_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_postprocess"
    )
    scopes = [
        node
        for node in ast.walk(postprocess)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Call)
        and _call_name(node.items[0].context_expr) == "trace_scope"
        and len(node.items[0].context_expr.args) == 1
        and isinstance(node.items[0].context_expr.args[0], ast.Constant)
        and node.items[0].context_expr.args[0].value == "loss"
    ]

    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.items[0].context_expr.keywords == []
    assert len(scope.body) == 1
    assignment = scope.body[0]
    assert isinstance(assignment, ast.Assign)
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "loss"
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Attribute)
    assert isinstance(assignment.value.func.value, ast.Name)
    assert assignment.value.func.value.id == "self"
    assert assignment.value.func.attr == "compute_language_model_loss"
    assert [
        argument.id for argument in assignment.value.args if isinstance(argument, ast.Name)
    ] == ["labels", "logits"]
    assert assignment.value.keywords == []


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


def test_gpt_loss_scope_wraps_success_and_preserves_the_result(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    labels = torch.tensor([[1]])
    expected = torch.tensor([[3.0]])
    calls = []

    def compute_loss(actual_labels, logits):
        calls.append((actual_labels, logits))
        return expected

    monkeypatch.setattr(gpt_model, "has_config_logger_enabled", lambda _config: False)
    monkeypatch.setattr(gpt_model, "trace_scope", lambda name: _RecordingScope(events, name))

    result = _postprocess(_fake_model(compute_loss), labels=labels)

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] is labels
    assert torch.equal(calls[0][1], torch.tensor([[[2.0]]]))
    assert events == [
        ("B", "output_layer", None),
        ("E", "output_layer", None),
        ("B", "loss", None),
        ("E", "loss", None),
    ]


def test_gpt_loss_scope_preserves_exception_identity(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    error = RuntimeError("loss failed")

    def compute_loss(_labels, _logits):
        raise error

    monkeypatch.setattr(gpt_model, "has_config_logger_enabled", lambda _config: False)
    monkeypatch.setattr(gpt_model, "trace_scope", lambda name: _RecordingScope(events, name))

    with pytest.raises(RuntimeError, match="loss failed") as raised:
        _postprocess(_fake_model(compute_loss), labels=torch.tensor([[1]]))

    assert raised.value is error
    assert events == [
        ("B", "output_layer", None),
        ("E", "output_layer", None),
        ("B", "loss", None),
        ("E", "loss", RuntimeError),
    ]


def test_gpt_logits_path_does_not_open_a_loss_scope(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    monkeypatch.setattr(gpt_model, "has_config_logger_enabled", lambda _config: False)
    monkeypatch.setattr(gpt_model, "trace_scope", lambda name: _RecordingScope(events, name))
    logits = _postprocess(
        _fake_model(lambda _labels, _logits: pytest.fail("loss was computed")), labels=None
    )
    assert torch.equal(logits, torch.tensor([[[2.0]]]))
    assert events == [("B", "output_layer", None), ("E", "output_layer", None)]


def test_gpt_loss_trace_off_path_does_not_allocate_a_scope(monkeypatch) -> None:
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

    monkeypatch.setattr(gpt_model, "has_config_logger_enabled", lambda _config: False)
    sink = DisabledSink()
    install_trace_sink(sink)
    try:
        expected = torch.tensor([[3.0]])
        result = _postprocess(
            _fake_model(lambda _labels, _logits: expected), labels=torch.tensor([[1]])
        )
    finally:
        reset_trace_sink()

    assert result is expected
    assert sink.gate_calls == ["output_layer", "loss"]
    assert sink.scope_calls == 0
