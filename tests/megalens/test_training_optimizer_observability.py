from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "megatron/training/training.py"
CORE_OPTIMIZER = ROOT / "megatron/core/optimizer/optimizer.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def test_optimizer_scope_preserves_source_and_target_boundaries():
    module = ast.parse(TRAINING.read_text(encoding="utf-8"))
    train_step = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )

    optimizer_scopes = []
    for node in ast.walk(train_step):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context_expr = node.items[0].context_expr
        if (
            isinstance(context_expr, ast.Call)
            and _call_name(context_expr) == "trace_scope"
            and len(context_expr.args) == 1
            and isinstance(context_expr.args[0], ast.Constant)
            and context_expr.args[0].value == "optimizer"
        ):
            optimizer_scopes.append(node)

    assert len(optimizer_scopes) == 1
    scope = optimizer_scopes[0]
    assert len(scope.body) == 1
    assignment = scope.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Attribute)
    assert isinstance(assignment.value.func.value, ast.Name)
    assert assignment.value.func.value.id == "optimizer"
    assert assignment.value.func.attr == "step"
    assert assignment.value.args == []
    assert assignment.value.keywords == []

    calls = [node for node in ast.walk(train_step) if isinstance(node, ast.Call)]
    timer_start = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "start"
        and isinstance(call.func.value, ast.Call)
        and _call_name(call.func.value) == "timers"
        and isinstance(call.func.value.args[0], ast.Constant)
        and call.func.value.args[0].value == "optimizer"
    )
    clip_qk = next(call for call in calls if _call_name(call) == "clip_qk")
    timer_stop = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "stop"
        and isinstance(call.func.value, ast.Call)
        and _call_name(call.func.value) == "timers"
        and isinstance(call.func.value.args[0], ast.Constant)
        and call.func.value.args[0].value == "optimizer"
    )
    model_parallel_postprocess = next(
        call for call in calls if _call_name(call) == "logical_and_across_model_parallel_group"
    )

    assert timer_start.lineno < scope.lineno < clip_qk.lineno < timer_stop.lineno
    assert timer_stop.lineno < model_parallel_postprocess.lineno


def test_optimizer_postprocess_scope_matches_the_source_tail_boundary():
    module = ast.parse(TRAINING.read_text(encoding="utf-8"))
    train_step = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )
    scopes = [
        node
        for node in ast.walk(train_step)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Call)
        and _call_name(node.items[0].context_expr) == "trace_scope"
        and len(node.items[0].context_expr.args) == 1
        and isinstance(node.items[0].context_expr.args[0], ast.Constant)
        and node.items[0].context_expr.args[0].value == "optimizer-postprocess"
    ]

    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.items[0].context_expr.keywords == []
    first_statement = scope.body[0]
    assert isinstance(first_statement, ast.Assign)
    assert isinstance(first_statement.value, ast.Call)
    assert _call_name(first_statement.value) == "logical_and_across_model_parallel_group"

    calls = [node for node in ast.walk(train_step) if isinstance(node, ast.Call)]
    timer_stop = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "stop"
        and isinstance(call.func.value, ast.Call)
        and _call_name(call.func.value) == "timers"
        and isinstance(call.func.value.args[0], ast.Constant)
        and call.func.value.args[0].value == "optimizer"
    )
    assert timer_stop.lineno < scope.lineno

    scoped_calls = [node for node in ast.walk(scope) if isinstance(node, ast.Call)]
    assert not any(_call_name(call) == "clip_qk" for call in scoped_calls)
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "step"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "optimizer"
        for call in scoped_calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr in {"start", "stop"}
        and isinstance(call.func.value, ast.Call)
        and _call_name(call.func.value) == "timers"
        for call in scoped_calls
    )
    assert len([node for node in ast.walk(scope) if isinstance(node, ast.Return)]) == 2


def test_training_imports_dependency_light_trace_facade():
    module = ast.parse(TRAINING.read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("megatron.core.observability", "trace_scope") in imports


def test_core_optimizer_step_scopes_match_source_call_boundaries():
    module = ast.parse(CORE_OPTIMIZER.read_text(encoding="utf-8"))
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}

    mixed_method = next(
        node
        for node in classes["MixedPrecisionOptimizer"].body
        if isinstance(node, ast.FunctionDef) and node.name == "step_with_ready_grads"
    )
    fp32_method = next(
        node
        for node in classes["FP32Optimizer"].body
        if isinstance(node, ast.FunctionDef) and node.name == "step_with_ready_grads"
    )

    for method in (mixed_method, fp32_method):
        scopes = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.With)
            and len(node.items) == 1
            and isinstance(node.items[0].context_expr, ast.Call)
            and _call_name(node.items[0].context_expr) == "trace_scope"
            and len(node.items[0].context_expr.args) == 1
            and isinstance(node.items[0].context_expr.args[0], ast.Constant)
            and node.items[0].context_expr.args[0].value == "optimizer-step"
        ]
        assert len(scopes) == 1

        scope = scopes[0]
        step_calls = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
            and node.func.value.attr == "optimizer"
        ]
        assert len(step_calls) == 1

        timer_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"start", "stop"}
            and isinstance(node.func.value, ast.Call)
            and _call_name(node.func.value) == "timers"
            and isinstance(node.func.value.args[0], ast.Constant)
            and node.func.value.args[0].value == "optimizer-inner-step"
        ]
        assert len(timer_calls) == 2
        assert timer_calls[0].lineno < scope.lineno < timer_calls[1].lineno


def test_fp32_optimizer_step_scope_preserves_timer_and_backend_order(monkeypatch):
    from megatron.core.optimizer import optimizer as optimizer_module

    events = []

    @contextmanager
    def recording_scope(name):
        events.append(("scope-enter", name))
        yield
        events.append(("scope-exit", name))

    class Timer:
        def start(self, *, barrier):
            events.append(("timer-start", barrier))

        def stop(self):
            events.append(("timer-stop",))

    class Timers:
        def __call__(self, name, log_level=None):
            assert name == "optimizer-inner-step"
            assert log_level in (None, 1)
            return Timer()

    class BackendOptimizer:
        def step(self):
            events.append(("backend-step",))

    monkeypatch.setattr(optimizer_module, "trace_scope", recording_scope)
    optimizer = SimpleNamespace(
        config=SimpleNamespace(timers=Timers(), barrier_with_L1_time=False),
        is_stub_optimizer=False,
        optimizer=BackendOptimizer(),
    )

    result = optimizer_module.FP32Optimizer.step_with_ready_grads(optimizer)

    assert result is True
    assert events == [
        ("timer-start", False),
        ("scope-enter", "optimizer-step"),
        ("backend-step",),
        ("scope-exit", "optimizer-step"),
        ("timer-stop",),
    ]
