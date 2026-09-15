# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib
import inspect
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink

shared_experts = importlib.import_module("megatron.core.transformer.moe.shared_experts")
SharedExpertMLP = shared_experts.SharedExpertMLP


_STAGES = (
    "pre_forward_comm",
    "linear_fc1_forward_and_act",
    "linear_fc2_forward",
    "post_forward_comm",
    "get_output",
)


class _FakeStream:
    def __init__(self, name: str, timeline: list[tuple[Any, ...]]) -> None:
        self.name = name
        self.timeline = timeline

    def wait_stream(self, other: "_FakeStream") -> None:
        self.timeline.append(("wait_stream", self.name, other.name))


class _UseStream(AbstractContextManager[None]):
    def __init__(self, platform: "_FakePlatform", stream: _FakeStream) -> None:
        self.platform = platform
        self.stream = stream
        self.previous: _FakeStream | None = None

    def __enter__(self) -> None:
        self.previous = self.platform.current
        self.platform.current = self.stream
        self.platform.timeline.append(("stream-B", self.stream.name))
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.platform.timeline.append(("stream-E", self.stream.name, exc_type))
        assert self.previous is not None
        self.platform.current = self.previous
        return False


class _FakePlatform:
    def __init__(self, timeline: list[tuple[Any, ...]]) -> None:
        self.timeline = timeline
        self.main = _FakeStream("main", timeline)
        self.shared = _FakeStream("shared", timeline)
        self.current = self.main

    def current_stream(self) -> _FakeStream:
        return self.current

    def stream(self, stream: _FakeStream) -> _UseStream:
        return _UseStream(self, stream)


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        stage = self.record["attrs"]["stage"]
        self.sink.active.append(stage)
        self.sink.timeline.append(("B", stage, self.sink.current_stream().name))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        stage = self.record["attrs"]["stage"]
        assert self.sink.active.pop() == stage
        self.record["exit_exception"] = exc_type
        self.record["exit_value"] = exc_value
        self.sink.timeline.append(("E", stage, self.sink.current_stream().name, exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> bool:
        return True


class _RecordingSink:
    def __init__(
        self, timeline: list[tuple[Any, ...]], current_stream, *, enabled: bool = True
    ) -> None:
        self.timeline = timeline
        self.current_stream = current_stream
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
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
        }
        self.records.append(record)
        return _RecordingScope(self, record)


class _Linear:
    def __init__(self, scale: float, timeline: list[tuple[Any, ...]], name: str) -> None:
        self.scale = scale
        self.timeline = timeline
        self.name = name

    def __call__(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.timeline.append(("compute", self.name))
        return value * self.scale, None


class _FailingLinear:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __call__(self, value: torch.Tensor):
        raise self.error


class _UnpackFails:
    def __iter__(self):
        pytest.fail("closed trace gate constructed shared-expert overlap attrs")


class _SharedExpertHarness:
    _overlap_trace_scope = SharedExpertMLP._overlap_trace_scope
    pre_forward_comm = SharedExpertMLP.pre_forward_comm
    linear_fc1_forward_and_act = SharedExpertMLP.linear_fc1_forward_and_act
    linear_fc2_forward = SharedExpertMLP.linear_fc2_forward
    post_forward_comm = SharedExpertMLP.post_forward_comm
    get_output = SharedExpertMLP.get_output

    def __init__(
        self,
        platform: _FakePlatform,
        timeline: list[tuple[Any, ...]],
        *,
        identity: object = (7, 4),
    ) -> None:
        self.config = SimpleNamespace(
            moe_shared_expert_overlap=True,
            sequence_parallel=False,
            use_te_activation_func=False,
            bias_activation_fusion=False,
            gated_linear_unit=False,
        )
        self.stream = platform.shared
        self._shared_expert_trace_identity = identity
        self.use_shared_expert_gate = False
        self.gate_score = None
        self.cached_fc1_input = None
        self.cached_fc2_input = None
        self.cached_fc2_output = None
        self.cached_output = None
        self.linear_fc1 = _Linear(2.0, timeline, "fc1")
        self.linear_fc2 = _Linear(3.0, timeline, "fc2")
        self.activation_func = lambda value: value + 1.0


@pytest.fixture(autouse=True)
def _reset_trace_sink() -> None:
    reset_trace_sink()
    yield
    reset_trace_sink()


@pytest.fixture
def overlap_runtime(monkeypatch: pytest.MonkeyPatch):
    timeline: list[tuple[Any, ...]] = []
    platform = _FakePlatform(timeline)
    monkeypatch.setattr(shared_experts, "cur_platform", platform)
    monkeypatch.setattr(shared_experts, "apply_module", lambda module: module)
    monkeypatch.setattr(shared_experts, "copy_to_tensor_model_parallel_region", lambda value: value)
    monkeypatch.setattr(
        shared_experts, "reduce_from_tensor_model_parallel_region", lambda value: value - 4.0
    )
    monkeypatch.setattr(shared_experts, "set_tensor_grad_fn_sequence_sr", lambda *args: None)
    return timeline, platform


def _run_overlap_path(
    identity: object, timeline, platform
) -> tuple[_SharedExpertHarness, torch.Tensor]:
    module = _SharedExpertHarness(platform, timeline, identity=identity)
    module.pre_forward_comm(torch.tensor([2.0]))
    module.linear_fc1_forward_and_act()
    module.linear_fc2_forward()
    module.post_forward_comm()
    output = module.get_output()
    return module, output


def test_overlap_stages_emit_exact_order_fields_and_shared_stream_boundaries(
    overlap_runtime,
) -> None:
    timeline, platform = overlap_runtime
    sink = _RecordingSink(timeline, platform.current_stream)
    install_trace_sink(sink)

    module, output = _run_overlap_path((7, 4), timeline, platform)

    assert torch.equal(output, torch.tensor([11.0]))
    assert module.cached_fc1_input is None
    assert module.cached_fc2_input is None
    assert module.cached_fc2_output is None
    assert module.cached_output is None
    assert [record["name"] for record in sink.records] == ["moe-shared-expert"] * 5
    assert [record["attrs"] for record in sink.records] == [
        {"layer": 7, "ep_size": 4, "stage": stage} for stage in _STAGES
    ]

    lifecycle = [event for event in timeline if event[0] in {"wait_stream", "B", "E"}]
    assert lifecycle == [
        ("wait_stream", "shared", "main"),
        ("B", "pre_forward_comm", "shared"),
        ("E", "pre_forward_comm", "shared", None),
        ("B", "linear_fc1_forward_and_act", "shared"),
        ("E", "linear_fc1_forward_and_act", "shared", None),
        ("B", "linear_fc2_forward", "shared"),
        ("E", "linear_fc2_forward", "shared", None),
        ("B", "post_forward_comm", "shared"),
        ("E", "post_forward_comm", "shared", None),
        ("B", "get_output", "shared"),
        ("E", "get_output", "shared", None),
        ("wait_stream", "main", "shared"),
    ]
    assert sink.active == []
    assert platform.current_stream() is platform.main


@pytest.mark.parametrize("gate_mode", ["disabled", "suppressed"])
def test_overlap_closed_gate_skips_attrs_and_preserves_numerical_path(
    overlap_runtime, gate_mode: str
) -> None:
    timeline, platform = overlap_runtime
    sink = _RecordingSink(timeline, platform.current_stream, enabled=gate_mode != "disabled")
    install_trace_sink(sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None)

    module, output = _run_overlap_path(_UnpackFails(), timeline, platform)

    assert torch.equal(output, torch.tensor([11.0]))
    assert module.cached_output is None
    assert sink.records == []
    assert sink.active == []
    assert [event for event in timeline if event[0] == "wait_stream"] == [
        ("wait_stream", "shared", "main"),
        ("wait_stream", "main", "shared"),
    ]


def test_overlap_stage_error_closes_scope_and_preserves_exception_identity(overlap_runtime) -> None:
    timeline, platform = overlap_runtime
    sink = _RecordingSink(timeline, platform.current_stream)
    install_trace_sink(sink)
    module = _SharedExpertHarness(platform, timeline)
    error = RuntimeError("shared expert FC2 failed")

    module.pre_forward_comm(torch.tensor([2.0]))
    module.linear_fc1_forward_and_act()
    module.linear_fc2 = _FailingLinear(error)

    with pytest.raises(RuntimeError, match="shared expert FC2 failed") as raised:
        module.linear_fc2_forward()

    assert raised.value is error
    assert sink.active == []
    assert [record["attrs"]["stage"] for record in sink.records] == list(_STAGES[:3])
    assert sink.records[-1]["exit_exception"] is RuntimeError
    assert sink.records[-1]["exit_value"] is error
    assert timeline[-2:] == [
        ("E", "linear_fc2_forward", "shared", RuntimeError),
        ("stream-E", "shared", RuntimeError),
    ]
    assert platform.current_stream() is platform.main


def test_overlap_probe_keeps_the_two_native_stream_dependencies_only() -> None:
    class_source = inspect.getsource(SharedExpertMLP)
    method_sources = {name: inspect.getsource(getattr(SharedExpertMLP, name)) for name in _STAGES}

    assert class_source.count(".wait_stream(") == 2
    assert method_sources["pre_forward_comm"].count(".wait_stream(") == 1
    assert method_sources["get_output"].count(".wait_stream(") == 1
    assert all(
        ".wait_stream(" not in method_sources[name]
        for name in ("linear_fc1_forward_and_act", "linear_fc2_forward", "post_forward_comm")
    )
    assert "synchronize" not in class_source
    assert ".wait(" not in class_source
    assert "Work" not in class_source
