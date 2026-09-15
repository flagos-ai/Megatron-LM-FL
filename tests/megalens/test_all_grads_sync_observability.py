# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink, trace_scope
from megatron.core.pipeline_parallel import schedules
from megatron.core.process_groups_config import ProcessGroupCollection

finalize_grads = importlib.import_module("megatron.core.distributed.finalize_model_grads")

ROOT = Path(__file__).resolve().parents[2]


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.active.append(self.record["name"])
        self.sink.transitions.append(("B", self.record["name"], None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self.sink.active.pop() == self.record["name"]
        self.sink.transitions.append(("E", self.record["name"], exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []
        self.active: list[str] = []

    def is_enabled(self, name: str) -> bool:
        return self.enabled

    def scope(
        self,
        name: str,
        *,
        ctx: Mapping[str, Any] | None = None,
        slots: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> _RecordingScope:
        record = {
            "name": name,
            "ctx": dict(ctx or {}),
            "slots": tuple(slots or ()),
            "attrs": dict(attrs or {}),
            "values": {},
        }
        self.records.append(record)
        return _RecordingScope(self, record)


class _Timers:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events

    def __call__(self, name: str, log_level: int | None = None) -> "_Timers":
        self.name = name
        return self

    def start(self, *, barrier: bool) -> None:
        self.events.append(("timer-start", self.name, barrier))

    def stop(self) -> None:
        self.events.append(("timer-stop", self.name))


class _Chunk:
    def __init__(
        self,
        name: str,
        events: list[tuple[Any, ...]],
        sink: _RecordingSink,
        *,
        child_scope: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.sink = sink
        self.child_scope = child_scope
        self.error = error

    def finish_grad_sync(self, *, force_all_reduce: bool = False) -> None:
        self.events.append(("finish", self.name, force_all_reduce, tuple(self.sink.active)))
        if self.child_scope:
            with trace_scope("dp-grad-sync-complete"):
                self.events.append(("child-body", self.name, tuple(self.sink.active)))
        if self.error is not None:
            raise self.error


class _FakeGroup:
    def size(self) -> int:
        return 1


class _ScheduleModel(torch.nn.Module):
    def __init__(
        self, config: SimpleNamespace, events: list[tuple[Any, ...]], sink: _RecordingSink
    ) -> None:
        super().__init__()
        self.config = config
        self.events = events
        self.sink = sink
        self.model_type = "unit-test"
        self.input_tensor = None

    def set_input_tensor(self, input_tensor) -> None:
        self.input_tensor = input_tensor

    def finish_grad_sync(self, *, force_all_reduce: bool = False) -> None:
        self.events.append(("schedule-finish", force_all_reduce, tuple(self.sink.active)))


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def _configure_finalize(
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[Any, ...]],
    sink: _RecordingSink,
    *,
    timers: _Timers | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    config = SimpleNamespace(
        timers=timers, barrier_with_L1_time=False, moe_router_enable_expert_bias=False
    )
    groups = SimpleNamespace(
        tp=object(), pp=object(), embd=object(), pos_embd=object(), dp_cp=object()
    )
    monkeypatch.setattr(finalize_grads, "get_model_config", lambda model: config)

    helpers = (
        "_allreduce_conditional_embedding_grads",
        "_allreduce_non_tensor_model_parallel_grads",
        "_allreduce_word_embedding_grads",
        "_allreduce_position_embedding_grads",
        "reset_model_temporary_tensors",
    )
    for helper_name in helpers:

        def record_helper(*args, _name=helper_name, **kwargs):
            events.append(("helper", _name, tuple(sink.active)))

        monkeypatch.setattr(finalize_grads, helper_name, record_helper)

    return config, groups


def _configure_no_pipeline_schedule(
    config: SimpleNamespace, finalize_func
) -> ProcessGroupCollection:
    config.enable_autocast = False
    config.autocast_dtype = torch.float32
    config.calculate_per_token_loss = False
    config.grad_scale_func = None
    config.deallocate_pipeline_outputs = False
    config.num_moe_experts = None
    config.mtp_num_layers = None
    config.overlap_moe_expert_parallel_comm = False
    config.hybrid_context_parallel = False
    config.finalize_model_grads_func = finalize_func
    config.no_sync_func = None
    config.fine_grained_activation_offloading = False

    group = _FakeGroup()
    groups = ProcessGroupCollection()
    groups.tp = group
    groups.pp = group
    groups.embd = group
    groups.pos_embd = group
    groups.dp_cp = group
    groups.cp = group
    return groups


def _one_microbatch_forward(data_iterator, active_model):
    leaf = torch.tensor(1.0, requires_grad=True)
    output = leaf * 1.0
    return output, lambda value: (value, {"loss": value.detach()})


def test_all_grads_parent_wraps_each_chunk_and_preserves_target_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    sink = _RecordingSink()
    install_trace_sink(sink)
    timers = _Timers(events)
    _, groups = _configure_finalize(monkeypatch, events, sink, timers=timers)
    chunks = [_Chunk("first", events, sink, child_scope=True), _Chunk("second", events, sink)]

    result = finalize_grads.finalize_model_grads(
        chunks, pg_collection=groups, force_all_reduce=True
    )

    assert result is None
    assert [event for event in events if event[0] == "finish"] == [
        ("finish", "first", True, ("all-grads-sync",)),
        ("finish", "second", True, ("all-grads-sync",)),
    ]
    assert [event for event in events if event[0] == "helper"] == [
        ("helper", "_allreduce_conditional_embedding_grads", ()),
        ("helper", "_allreduce_non_tensor_model_parallel_grads", ()),
        ("helper", "_allreduce_word_embedding_grads", ()),
        ("helper", "_allreduce_position_embedding_grads", ()),
        ("helper", "reset_model_temporary_tensors", ()),
    ]
    assert max(index for index, event in enumerate(events) if event[0] == "finish") < min(
        index for index, event in enumerate(events) if event[0] == "helper"
    )
    assert sink.transitions == [
        ("B", "all-grads-sync", None),
        ("B", "dp-grad-sync-complete", None),
        ("E", "dp-grad-sync-complete", None),
        ("E", "all-grads-sync", None),
    ]
    parent = next(record for record in sink.records if record["name"] == "all-grads-sync")
    assert parent["ctx"] == {}
    assert parent["slots"] == ()
    assert parent["attrs"] == {}
    assert events.index(("timer-start", "all-grads-sync", False)) < events.index(
        ("finish", "first", True, ("all-grads-sync",))
    )
    assert events.index(("finish", "second", True, ("all-grads-sync",))) < events.index(
        ("timer-stop", "all-grads-sync")
    )


def test_all_grads_parent_closes_and_preserves_chunk_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    sink = _RecordingSink()
    install_trace_sink(sink)
    timers = _Timers(events)
    _, groups = _configure_finalize(monkeypatch, events, sink, timers=timers)
    error = RuntimeError("gradient sync failed")
    chunks = [_Chunk("broken", events, sink, error=error), _Chunk("unreached", events, sink)]

    with pytest.raises(RuntimeError) as raised:
        finalize_grads.finalize_model_grads(chunks, pg_collection=groups)

    assert raised.value is error
    assert [event for event in events if event[0] == "finish"] == [
        ("finish", "broken", False, ("all-grads-sync",))
    ]
    assert not any(event[0] == "helper" for event in events)
    assert ("timer-stop", "all-grads-sync") not in events
    assert sink.transitions == [
        ("B", "all-grads-sync", None),
        ("E", "all-grads-sync", RuntimeError),
    ]


@pytest.mark.parametrize("enabled,suppressed", [(False, False), (True, True)])
def test_closed_all_grads_gate_preserves_finalize_behavior(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, suppressed: bool
) -> None:
    events: list[tuple[Any, ...]] = []
    sink = _RecordingSink(enabled=enabled)
    install_trace_sink(sink, suppress_scope=(lambda: True) if suppressed else None)
    _, groups = _configure_finalize(monkeypatch, events, sink)
    chunks = [_Chunk("only", events, sink)]

    finalize_grads.finalize_model_grads(chunks, pg_collection=groups, force_all_reduce=True)

    assert [event for event in events if event[0] in {"finish", "helper"}] == [
        ("finish", "only", True, ()),
        ("helper", "_allreduce_conditional_embedding_grads", ()),
        ("helper", "_allreduce_non_tensor_model_parallel_grads", ()),
        ("helper", "_allreduce_word_embedding_grads", ()),
        ("helper", "_allreduce_position_embedding_grads", ()),
        ("helper", "reset_model_temporary_tensors", ()),
    ]
    assert sink.records == []
    assert sink.transitions == []


def test_no_pipeline_schedule_nests_all_grads_under_grad_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    sink = _RecordingSink()
    install_trace_sink(sink)
    config, _ = _configure_finalize(monkeypatch, events, sink)
    groups = _configure_no_pipeline_schedule(config, finalize_grads.finalize_model_grads)
    model = _ScheduleModel(config, events, sink)

    schedules.forward_backward_no_pipelining(
        forward_step_func=_one_microbatch_forward,
        data_iterator=None,
        model=model,
        num_microbatches=1,
        seq_length=1,
        micro_batch_size=1,
        forward_only=False,
        pg_collection=groups,
    )

    assert ("schedule-finish", False, ("grad-sync", "all-grads-sync")) in events
    grad_begin = sink.transitions.index(("B", "grad-sync", None))
    parent_begin = sink.transitions.index(("B", "all-grads-sync", None))
    parent_end = sink.transitions.index(("E", "all-grads-sync", None))
    grad_end = sink.transitions.index(("E", "grad-sync", None))
    assert grad_begin < parent_begin < parent_end < grad_end


def test_forward_only_schedule_suppresses_nonempty_finalize_callback() -> None:
    events: list[tuple[Any, ...]] = []
    sink = _RecordingSink()
    install_trace_sink(sink)
    called = False

    def fail_finalize(*args, **kwargs):
        nonlocal called
        called = True
        pytest.fail("forward-only schedule called finalize_model_grads")

    config = SimpleNamespace(timers=None, barrier_with_L1_time=False)
    groups = _configure_no_pipeline_schedule(config, fail_finalize)
    model = _ScheduleModel(config, events, sink)

    schedules.forward_backward_no_pipelining(
        forward_step_func=_one_microbatch_forward,
        data_iterator=None,
        model=model,
        num_microbatches=1,
        seq_length=1,
        micro_batch_size=1,
        forward_only=True,
        pg_collection=groups,
    )

    assert called is False
    assert all(record["name"] not in {"grad-sync", "all-grads-sync"} for record in sink.records)


def test_invalid_process_group_collection_fails_before_parent_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    sink = _RecordingSink()
    install_trace_sink(sink)
    _configure_finalize(monkeypatch, events, sink)
    chunks = [_Chunk("unreached", events, sink)]
    incomplete_groups = SimpleNamespace(tp=object(), pp=object())

    with pytest.raises(AssertionError, match="pg_collection must have a embd"):
        finalize_grads.finalize_model_grads(chunks, pg_collection=incomplete_groups)

    assert sink.records == []
    assert not any(event[0] == "finish" for event in events)


def test_all_grads_source_scope_is_literal_dependency_light_and_narrow() -> None:
    path = ROOT / "megatron/core/distributed/finalize_model_grads.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "finalize_model_grads"
    )
    scopes = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and isinstance(node.items[0].context_expr.func, ast.Name)
        and node.items[0].context_expr.func.id == "trace_scope"
    ]

    assert "megatron.training" not in source
    assert "megatron.megalens" not in source
    assert len(scopes) == 1
    scope_call = scopes[0].items[0].context_expr
    assert len(scope_call.args) == 1
    assert isinstance(scope_call.args[0], ast.Constant)
    assert scope_call.args[0].value == "all-grads-sync"
    assert scope_call.keywords == []
    assert len(scopes[0].body) == 1
    assert isinstance(scopes[0].body[0], ast.For)
    scoped_calls = [node for node in ast.walk(scopes[0]) if isinstance(node, ast.Call)]
    assert (
        sum(
            isinstance(call.func, ast.Attribute) and call.func.attr == "finish_grad_sync"
            for call in scoped_calls
        )
        == 1
    )
    assert not any(
        isinstance(call.func, ast.Name)
        and call.func.id
        in {
            "_allreduce_conditional_embedding_grads",
            "_allreduce_non_tensor_model_parallel_grads",
            "_allreduce_word_embedding_grads",
            "_allreduce_position_embedding_grads",
            "reset_model_temporary_tensors",
        }
        for call in scoped_calls
    )


def _schedule_finalize_guards(relative_path: str) -> list[bool]:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    guards: list[bool] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.guard_stack: list[bool] = []

        def visit_If(self, node: ast.If) -> None:
            guarded = any(
                isinstance(candidate, ast.UnaryOp)
                and isinstance(candidate.op, ast.Not)
                and isinstance(candidate.operand, ast.Name)
                and candidate.operand.id == "forward_only"
                for candidate in ast.walk(node.test)
            )
            self.guard_stack.append(guarded or any(self.guard_stack))
            for child in node.body:
                self.visit(child)
            self.guard_stack.pop()
            for child in node.orelse:
                self.visit(child)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "finalize_model_grads_func"
            ):
                guards.append(any(self.guard_stack))
            self.generic_visit(node)

    Visitor().visit(tree)
    return guards


def test_every_schedule_finalize_call_remains_forward_only_guarded() -> None:
    standard_guards = _schedule_finalize_guards("megatron/core/pipeline_parallel/schedules.py")
    dualpipe_guards = _schedule_finalize_guards("megatron/plugin/dualpipev/dualpipev_schedules.py")

    assert standard_guards == [True, True, True]
    assert dualpipe_guards == [True]
