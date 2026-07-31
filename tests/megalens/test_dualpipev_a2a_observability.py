# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU and structural contracts for the DualPipeV asynchronous A2A lifecycle."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.megalens.trace import BASE_TRACING_EVENTS, FULL_TRACING_EVENTS
from megatron.plugin.dualpipev import observability as dualpipev_observability
from megatron.plugin.dualpipev.fb_overlap.modules import utils as dualpipev_utils

ROOT = Path(__file__).resolve().parents[2]


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.transitions.append(("B", self.record["name"], None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.sink.transitions.append(("E", self.record["name"], exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self, *enabled: str) -> None:
        self.enabled = frozenset(enabled)
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []

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
        return _RecordingScope(self, record)


class _Work:
    def __init__(self, *, result: Any = True, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.wait_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        self.wait_calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _owner() -> SimpleNamespace:
    dispatcher = SimpleNamespace(ep_size=4, tp_size=2)
    return SimpleNamespace(layer_number=7, mlp=SimpleNamespace(token_dispatcher=dispatcher))


def _event_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {**record["ctx"], **record["values"]}


def _install_collective(monkeypatch, work: Any):
    calls = []

    def all_to_all_single(output, input_, **kwargs):
        calls.append((output, input_, kwargs))
        return work

    monkeypatch.setattr(dualpipev_utils.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dualpipev_utils.dist, "all_to_all_single", all_to_all_single)
    return calls


def _launch(monkeypatch, work: Any, **metadata):
    calls = _install_collective(monkeypatch, work)
    input_tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    defaults = {
        "trace_owner": _owner(),
        "pass_direction": "forward",
        "logical_phase": "dispatch",
        "payload_role": "token_hidden_states",
    }
    defaults.update(metadata)
    original, output, request = dualpipev_utils.async_all_to_all(
        input_tensor, None, None, object(), **defaults
    )
    assert original is input_tensor
    assert output.shape == input_tensor.shape
    assert request is work
    return input_tensor, calls


@pytest.fixture(autouse=True)
def _reset_observation_state():
    reset_trace_sink()
    with dualpipev_observability._A2A_WORK_OBSERVATION_LOCK:
        dualpipev_observability._A2A_WORK_OBSERVATIONS.clear()
    yield
    reset_trace_sink()
    with dualpipev_observability._A2A_WORK_OBSERVATION_LOCK:
        dualpipev_observability._A2A_WORK_OBSERVATIONS.clear()


def test_launch_and_terminal_wait_preserve_raw_work_and_correlate_identity(monkeypatch) -> None:
    sink = _RecordingSink(
        dualpipev_observability.A2A_LAUNCH_EVENT, dualpipev_observability.A2A_COMPLETE_EVENT
    )
    install_trace_sink(sink)
    work = _Work(result="ready")

    input_tensor, calls = _launch(monkeypatch, work)
    result = dualpipev_observability.wait_async_all_to_all(
        work, completion_site="forward_dispatch_hidden_ready"
    )

    assert result == "ready"
    assert len(calls) == 1
    _, collective_input, collective_kwargs = calls[0]
    assert collective_input is input_tensor
    assert collective_kwargs == {
        "output_split_sizes": None,
        "input_split_sizes": None,
        "group": collective_kwargs["group"],
        "async_op": True,
    }
    assert [record["name"] for record in sink.records] == [
        "ep-alltoall-async-launch",
        "ep-alltoall-async-complete",
    ]

    launch = _event_fields(sink.records[0])
    completion = _event_fields(sink.records[1])
    assert launch["operation_id"] == completion["operation_id"]
    assert launch["request_id"] == launch["operation_id"]
    assert launch["operation_id_scope"] == "rank_local"
    assert launch["layer"] == 7
    assert launch["comm_type"] == "ep-alltoall"
    assert launch["dispatcher"] == "alltoall"
    assert launch["data_bytes"] == input_tensor.numel() * input_tensor.element_size()
    assert launch["group_size"] == 4
    assert launch["ep_size"] == 4
    assert launch["tp_size"] == 2
    assert launch["execution_route"] == "dualpipev_fb_overlap"
    assert launch["pass_direction"] == "forward"
    assert launch["logical_phase"] == "dispatch"
    assert launch["payload_role"] == "token_hidden_states"
    assert launch["async_op"] is True
    assert launch["completion_included"] is False
    assert launch["timing_phase"] == "async_dispatch"
    assert completion["completion_site"] == "forward_dispatch_hidden_ready"
    assert completion["terminal"] is True
    assert completion["wait_role"] == "terminal"
    assert completion["completion_guarantee"] == "current_stream_after_wait"
    assert completion["host_blocking_guaranteed"] is False
    assert completion["completed"] is True
    assert completion["error_type"] is None
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS

    assert dualpipev_observability.wait_async_all_to_all(work) == "ready"
    assert len(work.wait_calls) == 2
    assert len(sink.records) == 2


def test_dependency_wait_retains_sidecar_until_terminal_wait(monkeypatch) -> None:
    sink = _RecordingSink(dualpipev_observability.A2A_COMPLETE_EVENT)
    install_trace_sink(sink)
    work = _Work()
    _launch(monkeypatch, work)

    dualpipev_observability.wait_async_all_to_all(
        work,
        terminal=False,
        completion_site="comm_stream_dependency_before_forward_dispatch_probabilities",
    )
    assert id(work) in dualpipev_observability._A2A_WORK_OBSERVATIONS
    dualpipev_observability.wait_async_all_to_all(
        work, completion_site="backward_combine_gradient_ready"
    )

    assert len(work.wait_calls) == 2
    assert [record["name"] for record in sink.records] == [
        "ep-alltoall-async-complete",
        "ep-alltoall-async-complete",
    ]
    first = _event_fields(sink.records[0])
    second = _event_fields(sink.records[1])
    assert first["operation_id"] == second["operation_id"]
    assert (first["terminal"], first["wait_role"]) == (False, "dependency")
    assert (second["terminal"], second["wait_role"]) == (True, "terminal")
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS


def test_trace_off_skips_identity_and_preserves_collective_and_wait(monkeypatch) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        pytest.fail("trace-off path built DualPipeV A2A metadata")

    monkeypatch.setattr(dualpipev_observability, "_build_a2a_operation", fail)
    work = _Work(result=17)
    input_tensor, calls = _launch(monkeypatch, work)

    assert dualpipev_observability.wait_async_all_to_all(work, "timeout") == 17
    assert calls[0][1] is input_tensor
    assert work.wait_calls == [(('timeout',), {})]
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS


def test_failed_terminal_wait_records_error_rethrows_and_retains_sidecar(monkeypatch) -> None:
    sink = _RecordingSink(dualpipev_observability.A2A_COMPLETE_EVENT)
    install_trace_sink(sink)
    error = RuntimeError("wait failed")
    work = _Work(error=error)
    _launch(monkeypatch, work)

    with pytest.raises(RuntimeError) as raised:
        dualpipev_observability.wait_async_all_to_all(work)

    assert raised.value is error
    failed = _event_fields(sink.records[0])
    assert failed["completed"] is False
    assert failed["error_type"] == "RuntimeError"
    assert id(work) in dualpipev_observability._A2A_WORK_OBSERVATIONS

    work.error = None
    work.result = True
    dualpipev_observability.wait_async_all_to_all(work)
    succeeded = _event_fields(sink.records[1])
    assert succeeded["operation_id"] == failed["operation_id"]
    assert succeeded["completed"] is True
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS


def test_launch_exception_is_rethrown_without_registering_a_work(monkeypatch) -> None:
    sink = _RecordingSink(dualpipev_observability.A2A_LAUNCH_EVENT)
    install_trace_sink(sink)
    error = RuntimeError("launch failed")
    monkeypatch.setattr(dualpipev_utils.dist, "get_world_size", lambda group: 4)

    def fail(*args, **kwargs):
        del args, kwargs
        raise error

    monkeypatch.setattr(dualpipev_utils.dist, "all_to_all_single", fail)

    with pytest.raises(RuntimeError) as raised:
        dualpipev_utils.async_all_to_all(
            torch.ones((2, 2)),
            None,
            None,
            object(),
            trace_owner=_owner(),
            pass_direction="forward",
            logical_phase="combine",
            payload_role="expert_output",
        )

    assert raised.value is error
    assert sink.transitions == [
        ("B", "ep-alltoall-async-launch", None),
        ("E", "ep-alltoall-async-launch", RuntimeError),
    ]
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS


def test_non_weakrefable_work_fails_closed_to_raw_wait(monkeypatch) -> None:
    class WorkWithoutWeakref:
        __slots__ = ("wait_calls",)

        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self):
            self.wait_calls += 1
            return True

    sink = _RecordingSink(
        dualpipev_observability.A2A_LAUNCH_EVENT, dualpipev_observability.A2A_COMPLETE_EVENT
    )
    install_trace_sink(sink)
    work = WorkWithoutWeakref()
    _launch(monkeypatch, work)

    assert dualpipev_observability.wait_async_all_to_all(work) is True
    assert work.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["ep-alltoall-async-launch"]


def test_legacy_helper_call_without_route_metadata_remains_unobserved(monkeypatch) -> None:
    sink = _RecordingSink(
        dualpipev_observability.A2A_LAUNCH_EVENT, dualpipev_observability.A2A_COMPLETE_EVENT
    )
    install_trace_sink(sink)
    work = _Work()
    _install_collective(monkeypatch, work)
    input_tensor = torch.ones((2, 2))

    original, output, request = dualpipev_utils.async_all_to_all(input_tensor, None, None, object())

    assert original is input_tensor
    assert output.shape == input_tensor.shape
    assert request is work
    assert dualpipev_observability.wait_async_all_to_all(request) is True
    assert sink.records == []
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS


def test_reused_backend_work_disables_ambiguous_completion_pairing(monkeypatch) -> None:
    sink = _RecordingSink(dualpipev_observability.A2A_COMPLETE_EVENT)
    install_trace_sink(sink)
    work = _Work()
    _install_collective(monkeypatch, work)
    input_tensor = torch.ones((2, 2))

    for payload_role in ("token_hidden_states", "routing_probabilities"):
        _, _, request = dualpipev_utils.async_all_to_all(
            input_tensor,
            None,
            None,
            object(),
            trace_owner=_owner(),
            pass_direction="forward",
            logical_phase="dispatch",
            payload_role=payload_role,
        )
        assert request is work

    assert dualpipev_observability.wait_async_all_to_all(work) is True
    assert work.wait_calls == [((), {})]
    assert sink.records == []
    assert not dualpipev_observability._A2A_WORK_OBSERVATIONS


def test_single_rank_preserves_existing_none_work_boundary(monkeypatch) -> None:
    sink = _RecordingSink(
        dualpipev_observability.A2A_LAUNCH_EVENT, dualpipev_observability.A2A_COMPLETE_EVENT
    )
    install_trace_sink(sink)
    monkeypatch.setattr(dualpipev_utils.dist, "get_world_size", lambda group: 1)
    input_tensor = torch.ones((2, 2))

    result = dualpipev_utils.async_all_to_all(
        input_tensor,
        None,
        None,
        object(),
        trace_owner=_owner(),
        pass_direction="forward",
        logical_phase="dispatch",
        payload_role="token_hidden_states",
    )

    assert result == (input_tensor, input_tensor, None)
    assert sink.records == []


def _literal_keyword(call: ast.Call, name: str) -> Any:
    keyword = next(item for item in call.keywords if item.arg == name)
    return ast.literal_eval(keyword.value)


def test_all_dualpipev_launch_and_wait_sites_have_exact_lifecycle_metadata() -> None:
    paths = tuple(
        ROOT / "megatron/plugin/dualpipev/fb_overlap/overlap_funcs" / name
        for name in ("fwd.py", "bwd.py", "fwdbwd.py")
    )
    launch_calls = []
    wait_calls = []
    direct_waits = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "async_all_to_all":
                    launch_calls.append(node)
                elif node.func.id == "wait_async_all_to_all":
                    wait_calls.append(node)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "wait"
            ):
                direct_waits.append(node)

    assert len(launch_calls) == 22
    assert len(wait_calls) == 18
    assert direct_waits == []
    assert all(
        {keyword.arg for keyword in call.keywords}
        >= {"trace_owner", "pass_direction", "logical_phase", "payload_role"}
        for call in launch_calls
    )
    assert Counter(_literal_keyword(call, "pass_direction") for call in launch_calls) == {
        "forward": 9,
        "backward": 13,
    }
    assert Counter(_literal_keyword(call, "logical_phase") for call in launch_calls) == {
        "dispatch": 12,
        "combine": 10,
    }
    assert Counter(_literal_keyword(call, "payload_role") for call in launch_calls) == {
        "token_hidden_states": 3,
        "routing_probabilities": 3,
        "expert_output": 3,
        "expert_output_gradient": 7,
        "token_hidden_states_gradient": 3,
        "routing_probabilities_gradient": 3,
    }
    assert Counter(_literal_keyword(call, "completion_site") for call in wait_calls) == {
        "forward_dispatch_hidden_ready": 3,
        "forward_dispatch_probabilities_ready": 3,
        "forward_combine_output_ready": 3,
        "backward_combine_gradient_ready": 3,
        "backward_dispatch_hidden_gradient_ready": 3,
        "backward_dispatch_probability_gradient_ready": 3,
    }

    owners = Counter(
        ast.unparse(next(k.value for k in call.keywords if k.arg == "trace_owner"))
        for call in launch_calls
    )
    assert owners["next_bwd_layer_graph.layer"] == 4


def test_dependency_wait_and_cross_layer_handoff_remain_explicit() -> None:
    utils = (ROOT / "megatron/plugin/dualpipev/fb_overlap/modules/utils.py").read_text(
        encoding="utf-8"
    )
    transformer_block = (
        ROOT / "megatron/plugin/dualpipev/fb_overlap/transformer_block.py"
    ).read_text(encoding="utf-8")

    assert utils.count("wait_async_all_to_all(") == 1
    assert "terminal=False" in utils
    assert 'completion_site="comm_stream_dependency_before_forward_dispatch_probabilities"' in utils
    assert "event.wait()" not in utils
    assert "(bwd_layer_output_grad, bwd_unperm_a2a_handle)," in transformer_block
    assert "bwd_unperm_a2a_handle=bwd_unperm_a2a_handle" in transformer_block


def test_async_a2a_events_are_available_in_base_and_full_granularity() -> None:
    expected = {
        dualpipev_observability.A2A_LAUNCH_EVENT,
        dualpipev_observability.A2A_COMPLETE_EVENT,
    }
    assert expected <= BASE_TRACING_EVENTS
    assert expected <= FULL_TRACING_EVENTS
