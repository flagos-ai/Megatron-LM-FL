# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU and structural contracts for local trainable TP Linear probes."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core import parallel_state
from megatron.core import utils as core_utils
from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.tensor_parallel import layers
from megatron.core.tensor_parallel import observability as tp_observability
from megatron.megalens.trace import BASE_TRACING_EVENTS, FULL_TRACING_EVENTS


class _FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _RecordingScope:
    def __init__(self, record: dict[str, Any], order: list[str]) -> None:
        self.record = record
        self.order = order

    def __enter__(self) -> "_RecordingScope":
        self.order.append(f"B:{self.record['name']}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.record["exit_exception"] = exc_type
        self.order.append(f"E:{self.record['name']}")
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self, order: list[str], enabled: set[str] | None = None) -> None:
        self.order = order
        self.enabled = enabled or {"tp-all-gather-first"}
        self.records: list[dict[str, Any]] = []

    def is_enabled(self, name: str) -> bool:
        return name in self.enabled

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
        return _RecordingScope(record, self.order)


class _FakeGlobalMemoryBuffer:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls: list[tuple[list[int], torch.dtype, str]] = []

    def get_tensor(self, shape, dtype, name):
        normalized_shape = list(shape)
        self.calls.append((normalized_shape, dtype, name))
        self.order.append("allocate")
        return torch.empty(normalized_shape, dtype=dtype)


class _ForwardContext(SimpleNamespace):
    def save_for_backward(self, *tensors) -> None:
        self.saved_tensors = tensors


class _Work:
    def __init__(
        self,
        order: list[str],
        *,
        result: Any = True,
        error: BaseException | None = None,
        label: str = "wait",
    ) -> None:
        self.order = order
        self.result = result
        self.error = error
        self.label = label
        self.wait_calls = 0

    def wait(self) -> Any:
        self.wait_calls += 1
        self.order.append(self.label)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def _run_sequence_parallel_forward(
    monkeypatch, *, sink_enabled: bool = True, error=None, state: dict[str, Any] | None = None
):
    order: list[str] = []
    sink = _RecordingSink(order)
    if state is not None:
        state.update({"order": order, "sink": sink})
    if sink_enabled:
        install_trace_sink(sink)

    group = _FakeGroup(2)
    input_tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    global_buffer = _FakeGlobalMemoryBuffer(order)
    collective_calls = []
    monkeypatch.setattr(layers, "get_global_memory_buffer", lambda: global_buffer)

    def all_gather(output, value, *, group):
        order.append("collective")
        collective_calls.append((output, value, group))
        if error is not None:
            raise error
        output[:2].copy_(value)
        output[2:].copy_(value + 10)

    monkeypatch.setattr(layers, "dist_all_gather_func", all_gather)

    def peer_ranks(actual_group):
        order.append("metadata")
        assert actual_group is group
        return [7]

    monkeypatch.setattr(tp_observability, "get_process_group_peer_ranks", peer_ranks)
    ctx = _ForwardContext()
    result = layers.LinearWithGradAccumulationAndAsyncCommunication.forward(
        ctx, input_tensor, weight, None, False, False, True, None, 0, group, "vendor"
    )
    return result, ctx, sink, order, global_buffer, collective_calls, input_tensor, group


def _run_backward(
    monkeypatch,
    *,
    sequence_parallel: bool,
    allreduce_dgrad: bool,
    state: dict[str, Any] | None = None,
):
    order: list[str] = []
    sink = _RecordingSink(order, {"tp-linear-async-launch", "tp-linear-async-complete"})
    install_trace_sink(sink)
    group = _FakeGroup(2)
    input_tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    grad_output_rows = 4 if sequence_parallel else 2
    grad_output = torch.arange(grad_output_rows * weight.shape[0], dtype=torch.float32).reshape(
        grad_output_rows, weight.shape[0]
    )
    global_buffer = _FakeGlobalMemoryBuffer(order)
    monkeypatch.setattr(layers, "get_global_memory_buffer", lambda: global_buffer)
    monkeypatch.setattr(layers, "cur_platform", SimpleNamespace(current_device=lambda: "cpu"))
    works: dict[str, _Work] = {}

    def all_gather(output, value, *, group, async_op):
        assert group is tp_group
        assert async_op is True
        order.append("collective:all-gather")
        output[: value.shape[0]].copy_(value)
        output[value.shape[0] :].copy_(value + 10)
        works["all-gather"] = _Work(order, label="wait:all-gather")
        return works["all-gather"]

    def all_reduce(value, *, group, async_op):
        assert group is tp_group
        assert async_op is True
        order.append("collective:all-reduce")
        works["all-reduce"] = _Work(order, label="wait:all-reduce")
        return works["all-reduce"]

    def reduce_scatter(output, value, *, group, async_op):
        assert group is tp_group
        assert async_op is True
        order.append("collective:reduce-scatter")
        output.copy_(value[: output.shape[0]])
        works["reduce-scatter"] = _Work(order, label="wait:reduce-scatter")
        return works["reduce-scatter"]

    tp_group = group
    monkeypatch.setattr(layers, "dist_all_gather_func", all_gather)
    monkeypatch.setattr(layers.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(layers, "dist_reduce_scatter_func", reduce_scatter)
    ctx = SimpleNamespace(
        saved_tensors=(input_tensor, weight),
        main_grad=None,
        use_bias=False,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=sequence_parallel,
        wgrad_deferral_limit=0,
        grad_output_buffer=None,
        tp_group=group,
        te_fl_prefer="vendor",
    )
    if state is not None:
        state.update(
            {"order": order, "sink": sink, "works": works, "ctx": ctx, "grad_output": grad_output}
        )
    backward = layers.LinearWithGradAccumulationAndAsyncCommunication.backward.__wrapped__
    result = backward(ctx, grad_output)
    return result, sink, order, works


def test_sync_forward_all_gather_emits_source_leaf_and_preserves_result(monkeypatch) -> None:
    result, ctx, sink, order, global_buffer, calls, input_tensor, group = (
        _run_sequence_parallel_forward(monkeypatch)
    )

    expected_input = torch.cat((input_tensor, input_tensor + 10), dim=0)
    expected = expected_input.matmul(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).t())
    assert torch.equal(result, expected)
    assert ctx.saved_tensors[0] is input_tensor
    assert global_buffer.calls == [([4, 2], torch.float32, "mpu")]
    assert len(calls) == 1
    assert calls[0][1] is input_tensor
    assert calls[0][2] is group
    assert order == [
        "allocate",
        "B:tp-all-gather-first",
        "collective",
        "metadata",
        "E:tp-all-gather-first",
    ]
    assert sink.records == [
        {
            "name": "tp-all-gather-first",
            "ctx": {"data_bytes": 16, "group_size": 2, "op": "all-gather", "dim": "first"},
            "slots": ("group",),
            "attrs": {},
            "values": {"group": [7]},
            "exit_exception": None,
        }
    ]


def test_sync_forward_all_gather_trace_off_skips_metadata(monkeypatch) -> None:
    result, _, sink, order, _, calls, input_tensor, _ = _run_sequence_parallel_forward(
        monkeypatch, sink_enabled=False
    )

    assert result.shape == (4, 3)
    assert calls[0][1] is input_tensor
    assert sink.records == []
    assert order == ["allocate", "collective"]


def test_sync_all_gather_trace_off_reuses_one_noop_scope() -> None:
    input_tensor = torch.ones(2, 2)
    group = _FakeGroup(2)

    first_scope = tp_observability.sync_linear_all_gather_scope(input_tensor, group)
    second_scope = tp_observability.sync_linear_all_gather_scope(input_tensor, group)

    assert first_scope is second_scope
    with first_scope:
        pass


def test_async_launch_and_wait_correlate_without_wrapping_work() -> None:
    order: list[str] = []
    sink = _RecordingSink(order, {"tp-linear-async-launch", "tp-linear-async-complete"})
    install_trace_sink(sink)
    input_tensor = torch.ones(2, 3)
    group = _FakeGroup(2)
    work = _Work(order)

    with tp_observability.async_linear_collective_launch_scope(
        input_tensor,
        group,
        collective_op="all-gather",
        dim="first",
        launch_site="linear_backward_wgrad_input_all_gather",
        payload_role="weight_gradient_input",
    ) as observation:
        order.append("collective")

    result = tp_observability.wait_async_linear_collective(
        work,
        observation,
        completion_site="linear_backward_wgrad_input_ready",
        wait_role="dependency",
    )

    assert result is True
    assert work.wait_calls == 1
    assert order == [
        "B:tp-linear-async-launch",
        "collective",
        "E:tp-linear-async-launch",
        "B:tp-linear-async-complete",
        "wait",
        "E:tp-linear-async-complete",
    ]
    assert observation is not None
    assert not hasattr(observation, "request")
    launch, completion = sink.records
    operation_id = launch["ctx"]["operation_id"]
    assert operation_id.startswith("tp-linear:")
    assert completion["ctx"]["operation_id"] == operation_id
    assert launch["ctx"] == {
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "execution_route": "local_linear_direct_async",
        "collective_op": "all-gather",
        "data_bytes": 24,
        "group_size": 2,
        "launch_site": "linear_backward_wgrad_input_all_gather",
        "pass_direction": "backward",
        "payload_role": "weight_gradient_input",
        "dim": "first",
        "async_op": True,
        "completion_included": False,
        "timing_phase": "launch_attempt",
    }
    assert launch["slots"] == ("api_returned", "error_type")
    assert launch["values"] == {"api_returned": True, "error_type": None}
    assert completion["ctx"] == {
        "operation_id": operation_id,
        "operation_id_scope": "rank_local",
        "execution_route": "local_linear_direct_async",
        "collective_op": "all-gather",
        "data_bytes": 24,
        "group_size": 2,
        "launch_site": "linear_backward_wgrad_input_all_gather",
        "pass_direction": "backward",
        "payload_role": "weight_gradient_input",
        "dim": "first",
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "work_wait",
        "completion_site": "linear_backward_wgrad_input_ready",
        "duration_attribution": "per_request",
        "global_device_completion_guaranteed": False,
        "host_blocking_guaranteed": False,
        "launch_observed": True,
        "op": "wait",
        "terminal": True,
        "timing_phase": "stream_dependency",
        "wait_role": "dependency",
    }
    assert completion["slots"] == ("completed", "error_type")
    assert completion["values"] == {"completed": True, "error_type": None}


def test_async_trace_off_reuses_noop_scope_and_skips_metadata(monkeypatch) -> None:
    def unexpected_metadata(*args, **kwargs):
        del args, kwargs
        raise AssertionError("trace-off path built async metadata")

    monkeypatch.setattr(tp_observability, "_best_effort_tensor_bytes", unexpected_metadata)
    monkeypatch.setattr(tp_observability, "_best_effort_group_size", unexpected_metadata)
    kwargs = {
        "collective_op": "all-reduce",
        "dim": None,
        "launch_site": "linear_backward_dgrad_all_reduce",
        "payload_role": "input_gradient",
    }

    first_scope = tp_observability.async_linear_collective_launch_scope(
        torch.ones(2, 2), _FakeGroup(2), **kwargs
    )
    second_scope = tp_observability.async_linear_collective_launch_scope(
        torch.ones(2, 2), _FakeGroup(2), **kwargs
    )
    order: list[str] = []
    work = _Work(order)

    assert first_scope is second_scope
    with first_scope as observation:
        order.append("collective")
    result = tp_observability.wait_async_linear_collective(
        work,
        observation,
        completion_site="linear_backward_dgrad_all_reduce_return",
        wait_role="return",
    )

    assert observation is None
    assert result is True
    assert order == ["collective", "wait"]


@pytest.mark.parametrize(
    ("enabled", "expected_names", "launch_observed"),
    [
        ({"tp-linear-async-complete"}, ["tp-linear-async-complete"], False),
        ({"tp-linear-async-launch"}, ["tp-linear-async-launch"], None),
    ],
)
def test_async_launch_and_completion_gates_are_independent(
    enabled: set[str], expected_names: list[str], launch_observed: bool | None
) -> None:
    order: list[str] = []
    sink = _RecordingSink(order, enabled)
    install_trace_sink(sink)
    work = _Work(order)

    with tp_observability.async_linear_collective_launch_scope(
        torch.ones(2, 2),
        _FakeGroup(2),
        collective_op="all-reduce",
        dim=None,
        launch_site="linear_backward_dgrad_all_reduce",
        payload_role="input_gradient",
    ) as observation:
        order.append("collective")
    result = tp_observability.wait_async_linear_collective(
        work,
        observation,
        completion_site="linear_backward_dgrad_all_reduce_return",
        wait_role="return",
    )

    assert result is True
    assert [record["name"] for record in sink.records] == expected_names
    if launch_observed is not None:
        assert sink.records[0]["ctx"]["launch_observed"] is launch_observed


def test_async_launch_and_wait_errors_preserve_identity_and_outcome() -> None:
    order: list[str] = []
    sink = _RecordingSink(order, {"tp-linear-async-launch", "tp-linear-async-complete"})
    install_trace_sink(sink)
    launch_error = RuntimeError("launch failed")

    with pytest.raises(RuntimeError) as raised:
        with tp_observability.async_linear_collective_launch_scope(
            torch.ones(2, 2),
            _FakeGroup(2),
            collective_op="reduce-scatter",
            dim="first",
            launch_site="linear_backward_dgrad_reduce_scatter",
            payload_role="input_gradient",
        ):
            raise launch_error

    assert raised.value is launch_error
    assert sink.records[0]["values"] == {"api_returned": False, "error_type": "RuntimeError"}
    assert sink.records[0]["exit_exception"] is RuntimeError

    wait_error = ValueError("wait failed")
    with tp_observability.async_linear_collective_launch_scope(
        torch.ones(2, 2),
        _FakeGroup(2),
        collective_op="all-reduce",
        dim=None,
        launch_site="linear_backward_dgrad_all_reduce",
        payload_role="input_gradient",
    ) as observation:
        pass
    work = _Work(order, error=wait_error)
    with pytest.raises(ValueError) as raised:
        tp_observability.wait_async_linear_collective(
            work,
            observation,
            completion_site="linear_backward_dgrad_all_reduce_return",
            wait_role="return",
        )

    assert raised.value is wait_error
    assert work.wait_calls == 1
    completion = sink.records[-1]
    assert completion["values"] == {"completed": False, "error_type": "ValueError"}
    assert completion["exit_exception"] is ValueError


def test_async_error_identity_survives_trace_outcome_failure(monkeypatch) -> None:
    order: list[str] = []
    sink = _RecordingSink(order, {"tp-linear-async-launch"})
    original_scope = sink.scope

    def failing_scope(*args, **kwargs):
        scope = original_scope(*args, **kwargs)

        def fail_set(key, value):
            del key, value
            raise RuntimeError("outcome sink failed")

        monkeypatch.setattr(scope, "set", fail_set)
        return scope

    monkeypatch.setattr(sink, "scope", failing_scope)
    install_trace_sink(sink)
    launch_error = ValueError("collective failed")

    with pytest.raises(ValueError) as raised:
        with tp_observability.async_linear_collective_launch_scope(
            torch.ones(2, 2),
            _FakeGroup(2),
            collective_op="all-reduce",
            dim=None,
            launch_site="linear_backward_dgrad_all_reduce",
            payload_role="input_gradient",
        ):
            raise launch_error

    assert raised.value is launch_error
    assert any("trace outcome recording also failed" in note for note in launch_error.__notes__)
    assert sink.records[0]["exit_exception"] is ValueError


def test_async_wait_false_result_is_preserved_as_incomplete_attempt() -> None:
    order: list[str] = []
    sink = _RecordingSink(order, {"tp-linear-async-complete"})
    install_trace_sink(sink)
    with tp_observability.async_linear_collective_launch_scope(
        torch.ones(2, 2),
        _FakeGroup(2),
        collective_op="all-reduce",
        dim=None,
        launch_site="linear_backward_dgrad_all_reduce",
        payload_role="input_gradient",
    ) as observation:
        pass
    work = _Work(order, result=False)

    result = tp_observability.wait_async_linear_collective(
        work,
        observation,
        completion_site="linear_backward_dgrad_all_reduce_return",
        wait_role="return",
    )

    assert result is False
    assert work.wait_calls == 1
    assert sink.records[0]["values"] == {"completed": False, "error_type": None}


def test_explicit_retry_reuses_operation_identity_and_terminal_is_per_attempt() -> None:
    order: list[str] = []
    sink = _RecordingSink(order, {"tp-linear-async-complete"})
    install_trace_sink(sink)
    with tp_observability.async_linear_collective_launch_scope(
        torch.ones(2, 2),
        _FakeGroup(2),
        collective_op="all-reduce",
        dim=None,
        launch_site="linear_backward_dgrad_all_reduce",
        payload_role="input_gradient",
    ) as observation:
        pass

    class _RetryWork:
        def __init__(self) -> None:
            self.results = [False, True]
            self.wait_calls = 0

        def wait(self) -> bool:
            self.wait_calls += 1
            return self.results.pop(0)

    work = _RetryWork()
    first = tp_observability.wait_async_linear_collective(
        work, observation, completion_site="manual_retry", wait_role="dependency", terminal=False
    )
    second = tp_observability.wait_async_linear_collective(
        work, observation, completion_site="manual_retry", wait_role="return", terminal=True
    )

    assert first is False
    assert second is True
    assert work.wait_calls == 2
    first_record, second_record = sink.records
    assert first_record["ctx"]["operation_id"] == second_record["ctx"]["operation_id"]
    assert first_record["ctx"]["terminal"] is False
    assert second_record["ctx"]["terminal"] is True
    assert first_record["values"]["completed"] is False
    assert second_record["values"]["completed"] is True


def test_async_linear_events_are_available_in_base_and_full_granularity() -> None:
    expected = {"tp-linear-async-launch", "tp-linear-async-complete"}

    assert expected <= BASE_TRACING_EVENTS
    assert expected <= FULL_TRACING_EVENTS


def test_sequence_parallel_backward_pairs_all_gather_and_reduce_scatter(monkeypatch) -> None:
    result, sink, order, works = _run_backward(
        monkeypatch, sequence_parallel=True, allreduce_dgrad=False
    )

    assert result[0].shape == (2, 2)
    assert result[1].shape == (3, 2)
    assert result[2] is None
    assert {name: work.wait_calls for name, work in works.items()} == {
        "all-gather": 1,
        "reduce-scatter": 1,
    }
    assert order == [
        "allocate",
        "B:tp-linear-async-launch",
        "collective:all-gather",
        "E:tp-linear-async-launch",
        "B:tp-linear-async-complete",
        "wait:all-gather",
        "E:tp-linear-async-complete",
        "B:tp-linear-async-launch",
        "collective:reduce-scatter",
        "E:tp-linear-async-launch",
        "B:tp-linear-async-complete",
        "wait:reduce-scatter",
        "E:tp-linear-async-complete",
    ]
    assert [record["ctx"]["collective_op"] for record in sink.records] == [
        "all-gather",
        "all-gather",
        "reduce-scatter",
        "reduce-scatter",
    ]
    assert sink.records[0]["ctx"]["operation_id"] == sink.records[1]["ctx"]["operation_id"]
    assert sink.records[2]["ctx"]["operation_id"] == sink.records[3]["ctx"]["operation_id"]
    assert sink.records[0]["ctx"]["operation_id"] != sink.records[2]["ctx"]["operation_id"]


def test_non_sequence_parallel_backward_pairs_all_reduce_return_wait(monkeypatch) -> None:
    result, sink, order, works = _run_backward(
        monkeypatch, sequence_parallel=False, allreduce_dgrad=True
    )

    assert result[0].shape == (2, 2)
    assert result[1].shape == (3, 2)
    assert works["all-reduce"].wait_calls == 1
    assert order == [
        "B:tp-linear-async-launch",
        "collective:all-reduce",
        "E:tp-linear-async-launch",
        "B:tp-linear-async-complete",
        "wait:all-reduce",
        "E:tp-linear-async-complete",
    ]
    launch, completion = sink.records
    assert launch["ctx"]["collective_op"] == "all-reduce"
    assert "dim" not in launch["ctx"]
    assert completion["ctx"]["operation_id"] == launch["ctx"]["operation_id"]
    assert completion["ctx"]["completion_site"] == ("linear_backward_dgrad_all_reduce_return")
    assert completion["ctx"]["wait_role"] == "return"


def test_invalid_sequence_parallel_all_reduce_keeps_launch_before_assert(monkeypatch) -> None:
    state: dict[str, Any] = {}

    with pytest.raises(AssertionError):
        _run_backward(monkeypatch, sequence_parallel=True, allreduce_dgrad=True, state=state)

    assert state["works"]["all-gather"].wait_calls == 1
    assert state["works"]["all-reduce"].wait_calls == 0
    assert "reduce-scatter" not in state["works"]
    assert state["order"] == [
        "allocate",
        "B:tp-linear-async-launch",
        "collective:all-gather",
        "E:tp-linear-async-launch",
        "B:tp-linear-async-complete",
        "wait:all-gather",
        "E:tp-linear-async-complete",
        "B:tp-linear-async-launch",
        "collective:all-reduce",
        "E:tp-linear-async-launch",
    ]
    assert [record["ctx"]["collective_op"] for record in state["sink"].records] == [
        "all-gather",
        "all-gather",
        "all-reduce",
    ]


def test_sync_forward_all_gather_closes_scope_and_preserves_collective_error(monkeypatch) -> None:
    error = RuntimeError("all-gather failed")
    state: dict[str, Any] = {}
    with pytest.raises(RuntimeError) as raised:
        _run_sequence_parallel_forward(monkeypatch, error=error, state=state)

    assert raised.value is error
    assert state["order"] == [
        "allocate",
        "B:tp-all-gather-first",
        "collective",
        "E:tp-all-gather-first",
    ]
    assert state["sink"].records[0]["exit_exception"] is RuntimeError


def test_deferred_wgrad_first_all_gather_uses_the_same_sync_leaf(monkeypatch) -> None:
    order: list[str] = []
    sink = _RecordingSink(
        order, {"tp-all-gather-first", "tp-linear-async-launch", "tp-linear-async-complete"}
    )
    install_trace_sink(sink)
    tp_group = _FakeGroup(2)
    global_buffer = _FakeGlobalMemoryBuffer(order)
    monkeypatch.setattr(parallel_state, "get_global_memory_buffer", lambda: global_buffer)
    monkeypatch.setattr(
        tp_observability,
        "get_process_group_peer_ranks",
        lambda actual_group: [9] if actual_group is tp_group else None,
    )
    work = _Work(order)
    collective_modes: list[bool] = []

    def all_gather(output, value, *, group, async_op):
        assert group is tp_group
        collective_modes.append(async_op)
        order.append(f"collective:{async_op}")
        output[: value.shape[0]].copy_(value)
        output[value.shape[0] :].copy_(value + 10)
        return work if async_op else None

    monkeypatch.setattr(core_utils, "dist_all_gather_func", all_gather)

    def wgrad(all_gathered_input, grad_output, main_grad):
        del all_gathered_input, grad_output, main_grad
        order.append("wgrad")

    monkeypatch.setitem(
        sys.modules,
        "fused_weight_gradient_mlp_cuda",
        SimpleNamespace(wgrad_gemm_accum_fp32=wgrad, wgrad_gemm_accum_fp16=wgrad),
    )
    activations = [torch.ones(2, 2), torch.full((2, 2), 2.0)]
    grad_outputs = [torch.ones(4, 3), torch.full((4, 3), 2.0)]
    weight = SimpleNamespace(main_grad=torch.zeros(3, 2))
    config = SimpleNamespace(sequence_parallel=True, gradient_accumulation_fusion=True)

    result = core_utils.drain_embedding_wgrad_compute(
        config, activations, grad_outputs, weight, tp_group
    )

    assert result is None
    assert activations == []
    assert grad_outputs == []
    assert collective_modes == [False, True]
    assert work.wait_calls == 1
    assert order == [
        "allocate",
        "B:tp-all-gather-first",
        "collective:False",
        "E:tp-all-gather-first",
        "allocate",
        "B:tp-linear-async-launch",
        "collective:True",
        "E:tp-linear-async-launch",
        "wgrad",
        "B:tp-linear-async-complete",
        "wait",
        "E:tp-linear-async-complete",
        "wgrad",
    ]
    assert sink.records[0]["ctx"] == {
        "data_bytes": 16,
        "group_size": 2,
        "op": "all-gather",
        "dim": "first",
    }
    assert sink.records[0]["values"] == {"group": [9]}
    async_launch, async_completion = sink.records[1:]
    assert async_launch["ctx"]["launch_site"] == "embedding_wgrad_drain_all_gather"
    assert async_launch["ctx"]["collective_op"] == "all-gather"
    assert async_completion["ctx"]["operation_id"] == async_launch["ctx"]["operation_id"]
    assert async_completion["ctx"]["completion_site"] == ("embedding_wgrad_drain_input_ready")
    assert async_completion["ctx"]["wait_role"] == "dependency"
