# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU and structural contracts for DualPipeV framework-phase probes."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.pipeline_parallel import schedules
from megatron.plugin.dualpipev import dualpipev_schedules
from megatron.plugin.dualpipev import observability as dualpipev_observability


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.transitions.append(("B", self.record["name"], None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.sink.transitions.append(("E", self.record["name"], exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []

    def is_enabled(self, name: str) -> bool:
        return True

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
            "values": {slot: None for slot in slots or ()},
        }
        self.records.append(record)
        return _RecordingScope(self, record)


class _PhaseModel(torch.nn.Module):
    def __init__(self, config, *, dualpipev_stage: int) -> None:
        super().__init__()
        self.config = config
        self.dualpipev_stage = dualpipev_stage
        self.input_tensor = None

    def set_input_tensor(self, input_tensor) -> None:
        self.input_tensor = input_tensor


class _ObservedMicrobatch(int):
    format_calls = 0

    def __format__(self, format_spec: str) -> str:
        type(self).format_calls += 1
        return super().__format__(format_spec)


def _phase_config(**overrides):
    values = {
        "timers": None,
        "enable_autocast": False,
        "autocast_dtype": torch.float32,
        "calculate_per_token_loss": True,
        "grad_scale_func": None,
        "deallocate_pipeline_outputs": False,
        "num_moe_experts": None,
        "mtp_num_layers": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _event_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {**record["ctx"], **record["values"]}


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def test_dualpipev_phase_and_combined_contexts_use_separate_stage_identity() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)

    with dualpipev_observability.phase_scope(
        "forward-step",
        current_microbatch=5,
        dualpipev_stage=1,
        schedule_phase=dualpipev_observability.WARMUP,
        is_first_microbatch=False,
        is_last_stage=True,
        uses_model_graph=False,
    ):
        pass
    with dualpipev_observability.combined_scope(
        forward_microbatch=6,
        backward_microbatch=2,
        forward_dualpipev_stage=0,
        backward_dualpipev_stage=1,
        schedule_phase=dualpipev_observability.STEADY,
    ):
        pass
    with dualpipev_observability.grad_sync_scope():
        pass

    assert _event_fields(sink.records[0]) == {
        "current_microbatch": 5,
        "vp_stage": None,
        "dualpipev_stage": 1,
        "is_first_microbatch": False,
        "is_last_stage": True,
        "operation_id": "pp:microbatch=5:dualpipev_stage=1",
        "schedule": "dualpipev",
        "schedule_phase": "warmup",
        "uses_model_graph": False,
        "timing_phase": "framework_phase",
    }
    assert _event_fields(sink.records[1]) == {
        "operation_id": (
            "pp-combined:"
            "forward=pp:microbatch=6:dualpipev_stage=0:"
            "backward=pp:microbatch=2:dualpipev_stage=1"
        ),
        "forward_operation_id": "pp:microbatch=6:dualpipev_stage=0",
        "backward_operation_id": "pp:microbatch=2:dualpipev_stage=1",
        "forward_microbatch": 6,
        "backward_microbatch": 2,
        "forward_vp_stage": None,
        "backward_vp_stage": None,
        "forward_dualpipev_stage": 0,
        "backward_dualpipev_stage": 1,
        "execution_mode": "combined",
        "overlap_active": True,
        "schedule": "dualpipev",
        "schedule_phase": "steady",
        "uses_model_graph": True,
        "timing_phase": "framework_phase",
    }
    assert _event_fields(sink.records[2]) == {
        "schedule": "dualpipev",
        "timing_phase": "framework_phase",
    }


def test_dualpipev_trace_off_skips_context_and_operation_identity(monkeypatch) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        pytest.fail("trace-off path built DualPipeV metadata")

    monkeypatch.setattr(dualpipev_observability, "_phase_context", fail)
    monkeypatch.setattr(dualpipev_observability, "_combined_context", fail)
    _ObservedMicrobatch.format_calls = 0
    current_microbatch = _ObservedMicrobatch(1)

    with dualpipev_observability.phase_scope(
        "forward-step",
        current_microbatch=current_microbatch,
        dualpipev_stage=0,
        schedule_phase=dualpipev_observability.WARMUP,
        is_first_microbatch=True,
        is_last_stage=False,
        uses_model_graph=False,
    ):
        pass
    with dualpipev_observability.combined_scope(
        forward_microbatch=_ObservedMicrobatch(2),
        backward_microbatch=current_microbatch,
        forward_dualpipev_stage=1,
        backward_dualpipev_stage=0,
        schedule_phase=dualpipev_observability.STEADY,
    ):
        pass
    with dualpipev_observability.grad_sync_scope():
        pass

    assert _ObservedMicrobatch.format_calls == 0


def test_no_model_graph_forward_preserves_result_and_emits_nested_loss(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    model = _PhaseModel(config, dualpipev_stage=1)
    forward_data_store: list[Any] = []
    monkeypatch.setattr(
        dualpipev_schedules, "cur_platform", SimpleNamespace(device_name=lambda: "cpu")
    )

    def forward_step_func(data_iterator, active_model):
        del data_iterator
        assert active_model is model
        output = torch.tensor(3.0, requires_grad=True)

        def loss_func(value):
            return value, 11, {"loss": value.detach()}

        return output, loss_func

    output, num_tokens = dualpipev_schedules.forward_step_no_model_graph(
        forward_step_func,
        data_iterator=None,
        model=model,
        num_microbatches=4,
        input_tensor=None,
        forward_data_store=forward_data_store,
        config=config,
        cp_group_size=1,
        current_microbatch=1,
        dualpipev_stage=1,
        schedule_phase=dualpipev_observability.WARMUP,
        is_last_stage=True,
    )

    assert output.requires_grad
    assert num_tokens == 11
    assert [record["name"] for record in sink.records] == ["forward-step", "forward-step-calc-loss"]
    for record in sink.records:
        assert record["ctx"]["operation_id"] == "pp:microbatch=1:dualpipev_stage=1"
        assert record["ctx"]["vp_stage"] is None
        assert record["ctx"]["schedule"] == "dualpipev"
        assert record["ctx"]["schedule_phase"] == "warmup"
        assert record["ctx"]["uses_model_graph"] is False
    assert sink.records[0]["values"]["num_tokens"] == 11


def test_no_model_graph_forward_keeps_existing_positional_parameter_order() -> None:
    parameters = tuple(
        inspect.signature(dualpipev_schedules.forward_step_no_model_graph).parameters
    )
    assert parameters[-3:] == ("dualpipev_stage", "is_last_stage", "schedule_phase")


def test_no_model_graph_loss_error_closes_nested_phases_and_preserves_identity(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    model = _PhaseModel(config, dualpipev_stage=1)
    error = RuntimeError("loss failed")
    monkeypatch.setattr(
        dualpipev_schedules, "cur_platform", SimpleNamespace(device_name=lambda: "cpu")
    )

    def forward_step_func(data_iterator, active_model):
        del data_iterator, active_model

        def loss_func(output_tensor):
            del output_tensor
            raise error

        return torch.tensor(3.0, requires_grad=True), loss_func

    with pytest.raises(RuntimeError) as captured:
        dualpipev_schedules.forward_step_no_model_graph(
            forward_step_func,
            data_iterator=None,
            model=model,
            num_microbatches=4,
            input_tensor=None,
            forward_data_store=[],
            config=config,
            cp_group_size=1,
            current_microbatch=1,
            dualpipev_stage=1,
            schedule_phase=dualpipev_observability.WARMUP,
            is_last_stage=True,
        )

    assert captured.value is error
    assert sink.transitions == [
        ("B", "forward-step", None),
        ("B", "forward-step-calc-loss", None),
        ("E", "forward-step-calc-loss", RuntimeError),
        ("E", "forward-step", RuntimeError),
    ]


def test_model_graph_forward_emits_loss_child_and_fused_mode_suppresses_it(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    model = _PhaseModel(config, dualpipev_stage=1)
    model_graph = object()

    monkeypatch.setattr(dualpipev_schedules, "is_dualpipev_last_stgae", lambda stage: True)

    def fake_forward(data_iterator, active_model, extra_block_kwargs=None):
        del data_iterator, extra_block_kwargs
        assert active_model is model
        output = torch.tensor(8.0, requires_grad=True)
        return (output, model_graph), lambda value: (value, 13, {"loss": value.detach()})

    monkeypatch.setattr(dualpipev_schedules, "pretrain_gpt_forward_step_dualpipe", fake_forward)

    output, num_tokens = dualpipev_schedules.forward_step_with_model_graph(
        lambda *args: None,
        1,
        None,
        model,
        4,
        None,
        [],
        config,
        current_microbatch=7,
        schedule_phase=dualpipev_observability.STEADY,
    )

    assert output[1] is model_graph
    assert num_tokens == 13
    assert [record["name"] for record in sink.records] == ["forward-step", "forward-step-calc-loss"]
    assert all(record["ctx"]["uses_model_graph"] for record in sink.records)

    sink.records.clear()
    output, num_tokens = dualpipev_schedules.forward_step_with_model_graph(
        lambda *args: None,
        1,
        None,
        model,
        4,
        None,
        [],
        config,
        current_microbatch=8,
        schedule_phase=dualpipev_observability.STEADY,
        trace_phase=False,
    )
    assert output[1] is model_graph
    assert num_tokens == 13
    assert sink.records == []


def test_model_graph_backward_groups_loss_head_and_graph_as_one_phase(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    calls = []
    model_graph = object()

    def fake_backward(input_tensor, output_tensor, output_tensor_grad, config, model_graph=None):
        del config
        calls.append((input_tensor, output_tensor, output_tensor_grad, model_graph))
        return "loss-head-grad" if model_graph is None else "model-input-grad"

    monkeypatch.setattr(dualpipev_schedules, "backward_step_with_model_graph", fake_backward)

    result = dualpipev_schedules._backward_step_with_phase(
        "model-input",
        "model-output",
        "output-grad",
        object(),
        model_graph,
        logits_input_tensor="logits-input",
        current_microbatch=4,
        num_microbatches=4,
        dualpipev_stage=1,
        schedule_phase=dualpipev_observability.COOLDOWN,
        is_last_stage=True,
    )

    assert result == "model-input-grad"
    assert calls == [
        ("logits-input", "model-output", "output-grad", None),
        ("model-input", "model-output", "loss-head-grad", model_graph),
    ]
    assert [record["name"] for record in sink.records] == ["backward-step"]
    assert sink.records[0]["ctx"] == {
        "current_microbatch": 4,
        "vp_stage": None,
        "dualpipev_stage": 1,
        "is_first_microbatch": True,
        "is_last_stage": True,
        "operation_id": "pp:microbatch=4:dualpipev_stage=1",
        "schedule": "dualpipev",
        "schedule_phase": "cooldown",
        "uses_model_graph": True,
        "timing_phase": "framework_phase",
    }


def test_model_graph_queue_helper_preserves_pop_order_on_logits_error(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    error = RuntimeError("loss-head backward failed")
    input_queue = [(4, "model-input")]
    output_queue = ["model-output"]
    model_graph = object()
    model_graph_queue = [model_graph]
    logits_queue = ["logits-input"]

    def fail_logits_backward(
        input_tensor, output_tensor, output_tensor_grad, config, model_graph=None
    ):
        del input_tensor, output_tensor, output_tensor_grad, config
        assert model_graph is None
        raise error

    monkeypatch.setattr(dualpipev_schedules, "backward_step_with_model_graph", fail_logits_backward)

    with pytest.raises(RuntimeError) as captured:
        dualpipev_schedules._backward_step_from_queues(
            input_queue,
            output_queue,
            model_graph_queue,
            logits_queue,
            "output-grad",
            object(),
            include_logits_backward=True,
            num_microbatches=4,
            dualpipev_stage=1,
            schedule_phase=dualpipev_observability.STEADY,
            is_last_stage=True,
        )

    assert captured.value is error
    assert logits_queue == []
    assert input_queue == [(4, "model-input")]
    assert output_queue == ["model-output"]
    assert model_graph_queue == [model_graph]
    assert sink.transitions == [("B", "backward-step", None), ("E", "backward-step", RuntimeError)]


def test_core_backward_accepts_dualpipev_metadata_without_changing_gradient() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    input_tensor = torch.tensor(3.0, requires_grad=True)
    output_tensor = input_tensor * 5

    input_grad = schedules.backward_step(
        input_tensor,
        output_tensor,
        None,
        _phase_config(),
        current_microbatch=6,
        vp_stage=None,
        is_first_microbatch=False,
        is_last_stage=False,
        schedule="dualpipev",
        schedule_phase="steady",
        dualpipev_stage=0,
        uses_model_graph=False,
    )

    assert input_grad is input_tensor.grad
    assert input_grad.item() == 5.0
    assert _event_fields(sink.records[0]) == {
        "current_microbatch": 6,
        "vp_stage": None,
        "is_first_microbatch": False,
        "is_last_stage": False,
        "schedule": "dualpipev",
        "schedule_phase": "steady",
        "dualpipev_stage": 0,
        "uses_model_graph": False,
        "timing_phase": "framework_phase",
        "operation_id": "pp:microbatch=6:dualpipev_stage=0",
        "num_tokens": None,
        "sum_sq_seq_len": None,
    }


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_dualpipev_schedule_owns_combined_backward_and_final_sync_boundaries() -> None:
    source_path = Path(dualpipev_schedules.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    schedule = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "forward_backward_pipelining_with_dualpipev"
    )

    combined_with = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.With)
        and any(_call_name(item.context_expr) == "combined_scope" for item in node.items)
    ]
    assert len(combined_with) == 1
    combined_calls = [node for node in ast.walk(combined_with[0]) if isinstance(node, ast.Call)]
    combined_backward_calls = [
        node for node in combined_calls if _call_name(node) == "backward_step_with_model_graph"
    ]
    assert len(combined_backward_calls) == 1
    fused_forward_calls = [
        node for node in combined_calls if _call_name(node) == "forward_step_helper"
    ]
    assert len(fused_forward_calls) == 1
    fused_forward = fused_forward_calls[0]
    fused_keywords = {keyword.arg: keyword.value for keyword in fused_forward.keywords}
    assert isinstance(fused_keywords["trace_phase"], ast.Constant)
    assert fused_keywords["trace_phase"].value is False
    assert isinstance(fused_keywords["schedule_phase"], ast.Name)
    assert fused_keywords["schedule_phase"].id == "STEADY"

    grad_sync_with = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.With)
        and any(_call_name(item.context_expr) == "grad_sync_scope" for item in node.items)
    ]
    assert len(grad_sync_with) == 1
    scoped_finalizers = [
        node
        for node in ast.walk(grad_sync_with[0])
        if _call_name(node) == "finalize_model_grads_func"
    ]
    all_finalizers = [
        node for node in ast.walk(schedule) if _call_name(node) == "finalize_model_grads_func"
    ]
    assert len(scoped_finalizers) == 1
    assert all_finalizers == scoped_finalizers
    assert not [
        node
        for node in ast.walk(grad_sync_with[0])
        if _call_name(node) == "finish_embedding_wgrad_compute"
    ]

    core_backward_calls = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.Call) and _call_name(node) == "backward_step"
    ]
    assert len(core_backward_calls) == 1
    core_keywords = {keyword.arg: keyword.value for keyword in core_backward_calls[0].keywords}
    assert isinstance(core_keywords["current_microbatch"], ast.Name)
    assert core_keywords["current_microbatch"].id == "current_microbatch"
    assert isinstance(core_keywords["vp_stage"], ast.Constant)
    assert core_keywords["vp_stage"].value is None
    assert isinstance(core_keywords["schedule"], ast.Constant)
    assert core_keywords["schedule"].value == "dualpipev"
    assert isinstance(core_keywords["schedule_phase"], ast.Name)
    assert core_keywords["schedule_phase"].id == "schedule_phase"
    assert isinstance(core_keywords["dualpipev_stage"], ast.Name)
    assert core_keywords["dualpipev_stage"].id == "model_chunk_id"
    assert isinstance(core_keywords["uses_model_graph"], ast.Constant)
    assert core_keywords["uses_model_graph"].value is False

    input_appends = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Subscript)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "input_tensors"
    ]
    assert input_appends
    assert all(
        len(node.args) == 1 and isinstance(node.args[0], ast.Tuple) and len(node.args[0].elts) == 2
        for node in input_appends
    )

    backward_helper_calls = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.Call) and _call_name(node) == "backward_step_helper"
    ]
    assert len(backward_helper_calls) == 4
    for call in backward_helper_calls:
        assert isinstance(call.args[2], ast.Name)
        assert call.args[2].id == "STEADY"
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords["include_logits_backward"], ast.Constant)
        assert keywords["include_logits_backward"].value is True

    cooldown_calls = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.Call)
        and _call_name(node) == "run_backward_step"
        and len(node.args) > 5
        and isinstance(node.args[5], ast.Name)
        and node.args[5].id == "COOLDOWN"
    ]
    assert len(cooldown_calls) == 1
    cooldown_keywords = {keyword.arg: keyword.value for keyword in cooldown_calls[0].keywords}
    assert isinstance(cooldown_keywords["include_logits_backward"], ast.Constant)
    assert cooldown_keywords["include_logits_backward"].value is False


def test_dualpipev_observability_facade_remains_dependency_light() -> None:
    source_path = Path(dualpipev_observability.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(
        name == "torch"
        or name.startswith("megatron.training")
        or name.startswith("megatron.megalens")
        for name in imports
    )
