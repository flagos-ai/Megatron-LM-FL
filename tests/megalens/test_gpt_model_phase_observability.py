# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch

from megatron.core.models.gpt import gpt_model
from megatron.core.observability import install_trace_sink, reset_trace_sink

ROOT = Path(__file__).resolve().parents[2]
GPT_MODEL = ROOT / "megatron/core/models/gpt/gpt_model.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _literal_scope(function: ast.FunctionDef, name: str) -> ast.With:
    scopes = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Call)
        and _call_name(node.items[0].context_expr) == "trace_scope"
        and len(node.items[0].context_expr.args) == 1
        and isinstance(node.items[0].context_expr.args[0], ast.Constant)
        and node.items[0].context_expr.args[0].value == name
    ]
    assert len(scopes) == 1
    return scopes[0]


def test_gpt_model_phase_scopes_match_the_source_call_boundaries() -> None:
    module = ast.parse(GPT_MODEL.read_text(encoding="utf-8"))
    gpt_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GPTModel"
    )
    forward = next(
        node
        for node in gpt_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    postprocess = next(
        node
        for node in gpt_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_postprocess"
    )
    mtp_single_step = next(
        node
        for node in gpt_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_mtp_single_step"
    )

    decoder_scope = _literal_scope(forward, "decoder")
    assert decoder_scope.items[0].context_expr.keywords == []
    assert len(decoder_scope.body) == 1
    decoder_assignment = decoder_scope.body[0]
    assert isinstance(decoder_assignment, ast.Assign)
    assert isinstance(decoder_assignment.targets[0], ast.Name)
    assert decoder_assignment.targets[0].id == "decoder_output"
    assert isinstance(decoder_assignment.value, ast.Call)
    assert isinstance(decoder_assignment.value.func, ast.Attribute)
    assert decoder_assignment.value.func.attr == "decoder"

    tuple_normalization = next(
        node
        for node in forward.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and _call_name(node.test) == "isinstance"
    )
    assert decoder_scope.end_lineno < tuple_normalization.lineno

    postprocess_scope = _literal_scope(forward, "decoder-postprocess")
    assert tuple_normalization.end_lineno < postprocess_scope.lineno
    assert postprocess_scope.items[0].context_expr.keywords == []
    assert len(postprocess_scope.body) == 1
    postprocess_return = postprocess_scope.body[0]
    assert isinstance(postprocess_return, ast.Return)
    assert isinstance(postprocess_return.value, ast.Call)
    assert isinstance(postprocess_return.value.func, ast.Attribute)
    assert postprocess_return.value.func.attr == "_postprocess"
    assert {"is_spec_decode", "mhc_multistream"} <= {
        keyword.arg for keyword in postprocess_return.value.keywords
    }

    output_scope = _literal_scope(postprocess, "output_layer")
    assert output_scope.items[0].context_expr.keywords == []
    assert len(output_scope.body) == 1
    output_assignment = output_scope.body[0]
    assert isinstance(output_assignment, ast.Assign)
    assert isinstance(output_assignment.targets[0], ast.Tuple)
    assert isinstance(output_assignment.value, ast.Call)
    assert isinstance(output_assignment.value.func, ast.Attribute)
    assert output_assignment.value.func.attr == "output_layer"
    scale_assignment = next(
        node
        for node in postprocess.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_scale_logits"
    )
    assert output_scope.end_lineno < scale_assignment.lineno
    assert not any(
        isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _call_name(item.context_expr) == "trace_scope"
            and item.context_expr.args
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == "output_layer"
            for item in node.items
        )
        for node in ast.walk(mtp_single_step)
    )


class _RecordingScope:
    def __init__(self, order: list[tuple[Any, ...]], name: str) -> None:
        self.order = order
        self.name = name

    def __enter__(self):
        self.order.append(("B", self.name))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.order.append(("E", self.name, exc_type))
        return False


def _fake_forward_model(decoder, postprocess):
    return SimpleNamespace(
        config=SimpleNamespace(
            fine_grained_activation_offloading=False,
            moe_n_hash_layers=0,
        ),
        mtp_process=False,
        _preprocess=lambda **kwargs: (
            kwargs["decoder_input"],
            None,
            None,
            None,
            None,
            kwargs["padding_mask"],
        ),
        decoder=decoder,
        _postprocess=postprocess,
    )


def _forward(model, *, decoder_input):
    return gpt_model.GPTModel.forward(
        model,
        input_ids=torch.tensor([[1]]),
        position_ids=torch.tensor([[0]]),
        attention_mask=None,
        decoder_input=decoder_input,
    )


def test_gpt_model_scopes_preserve_mhc_normalization_and_call_order(monkeypatch) -> None:
    order: list[tuple[Any, ...]] = []
    decoder_input = torch.tensor([[[1.0]]])
    contracted = torch.tensor([[[2.0]]])
    multistream = torch.tensor([[[3.0]]])
    expected = object()

    def decoder(**kwargs):
        order.append(("call", "decoder", kwargs["hidden_states"]))
        return contracted, multistream

    def postprocess(**kwargs):
        order.append(
            ("call", "postprocess", kwargs["hidden_states"], kwargs["mhc_multistream"])
        )
        return expected

    monkeypatch.setattr(gpt_model, "trace_scope", lambda name: _RecordingScope(order, name))

    result = _forward(_fake_forward_model(decoder, postprocess), decoder_input=decoder_input)

    assert result is expected
    assert order == [
        ("B", "decoder"),
        ("call", "decoder", decoder_input),
        ("E", "decoder", None),
        ("B", "decoder-postprocess"),
        ("call", "postprocess", contracted, multistream),
        ("E", "decoder-postprocess", None),
    ]


def test_gpt_postprocess_early_return_does_not_open_output_layer(monkeypatch) -> None:
    order: list[tuple[Any, ...]] = []
    hidden_states = torch.tensor([[[1.0]]])

    model = _fake_forward_model(lambda **_kwargs: hidden_states, None)
    model.training = True
    model.share_embeddings_and_output_weights = False
    model.post_process = False
    model.config.mtp_num_layers = 0
    model._postprocess = MethodType(gpt_model.GPTModel._postprocess, model)

    monkeypatch.setattr(gpt_model, "trace_scope", lambda name: _RecordingScope(order, name))

    result = _forward(model, decoder_input=hidden_states)

    assert result is hidden_states
    assert order == [
        ("B", "decoder"),
        ("E", "decoder", None),
        ("B", "decoder-postprocess"),
        ("E", "decoder-postprocess", None),
    ]


def test_gpt_model_trace_off_does_not_open_phase_scopes() -> None:
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

    hidden_states = torch.tensor([[[1.0]]])
    expected = object()
    sink = DisabledSink()
    install_trace_sink(sink)
    try:
        result = _forward(
            _fake_forward_model(lambda **_kwargs: hidden_states, lambda **_kwargs: expected),
            decoder_input=hidden_states,
        )
    finally:
        reset_trace_sink()

    assert result is expected
    assert sink.gate_calls == ["decoder", "decoder-postprocess"]
    assert sink.scope_calls == 0
