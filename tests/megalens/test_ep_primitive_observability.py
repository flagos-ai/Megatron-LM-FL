# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_module
from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.megalens.core_adapter import MegaLensTraceSink
from megatron.megalens.trace import Tracer


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
        return False


class _RecordingSink:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []
        self.active: list[str] = []
        self.gate_calls: list[str] = []

    def is_enabled(self, name: str) -> bool:
        self.gate_calls.append(name)
        return self.enabled

    def scope(
        self,
        name: str,
        *,
        ctx: Mapping[str, Any] | None = None,
        slots: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> _RecordingScope:
        record = {"name": name, "ctx": dict(ctx or {})}
        self.records.append(record)
        return _RecordingScope(self, record)


class _Group:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _clean_trace_state(monkeypatch: pytest.MonkeyPatch):
    reset_trace_sink()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    yield
    reset_trace_sink()


def _alltoall_owner(
    sink: _RecordingSink, operations: list[tuple[Any, ...]], *, group_size: int = 2
) -> SimpleNamespace:
    tokens_per_expert = torch.tensor([1, 2])
    synchronized_tokens = torch.tensor([2, 1])

    def synchronize(point: str, tokens: torch.Tensor) -> torch.Tensor:
        operations.append(("synchronize", tuple(sink.active), point, tokens))
        return synchronized_tokens

    return SimpleNamespace(
        ep_group=_Group(group_size),
        ep_size=group_size,
        tp_size=3,
        tokens_per_expert=tokens_per_expert,
        synchronized_tokens=synchronized_tokens,
        output_splits=[2, 1],
        input_splits=[1, 2],
        _maybe_dtoh_and_synchronize=synchronize,
    )


def _allgather_owner(*, tp_size: int = 2, ep_size: int = 2, group_size: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        tp_ep_group=_Group(group_size),
        tp_size=tp_size,
        ep_size=ep_size,
        routing_map=torch.tensor([[True, False], [False, True]]),
        local_probs=torch.ones(2, dtype=torch.bfloat16),
    )


def test_standard_alltoall_dispatch_preserves_source_scope_fields_and_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    operations: list[tuple[Any, ...]] = []
    owner = _alltoall_owner(sink, operations)
    tokens = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    probs = torch.arange(4, dtype=torch.float16)
    dispatched_tokens = torch.tensor([[11.0]])
    dispatched_probs = torch.tensor([0.75])
    outputs = iter((dispatched_tokens, dispatched_probs))

    def all_to_all(group, tensor, output_splits, input_splits):
        operations.append(
            ("all-to-all", tuple(sink.active), group, tensor, output_splits, input_splits)
        )
        return next(outputs)

    monkeypatch.setattr(token_dispatcher_module, "all_to_all", all_to_all)

    result = MoEAlltoAllTokenDispatcher.token_dispatch(owner, tokens, probs)

    assert result == (dispatched_tokens, dispatched_probs)
    assert owner.tokens_per_expert is owner.synchronized_tokens
    assert sink.records == [
        {
            "name": "ep-alltoall-dispatch",
            "ctx": {
                "comm_type": "ep-alltoall",
                "dispatcher": "alltoall",
                "data_bytes": tokens.numel() * tokens.element_size()
                + probs.numel() * probs.element_size(),
                "group_size": 2,
                "ep_size": 2,
                "tp_size": 3,
            },
        }
    ]
    assert [operation[:3] for operation in operations] == [
        ("synchronize", ("ep-alltoall-dispatch",), "before_ep_alltoall"),
        ("all-to-all", ("ep-alltoall-dispatch",), owner.ep_group),
        ("all-to-all", ("ep-alltoall-dispatch",), owner.ep_group),
    ]
    assert operations[1][3:] == (tokens, owner.output_splits, owner.input_splits)
    assert operations[2][3:] == (probs, owner.output_splits, owner.input_splits)


def test_standard_alltoall_combine_preserves_arguments_and_ignores_legacy_async_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    operations: list[tuple[Any, ...]] = []
    owner = _alltoall_owner(sink, operations)
    hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    combined = torch.tensor([[21.0]])

    def all_to_all(group, tensor, output_splits, input_splits):
        operations.append(
            ("all-to-all", tuple(sink.active), group, tensor, output_splits, input_splits)
        )
        return combined

    monkeypatch.setattr(token_dispatcher_module, "all_to_all", all_to_all)

    result = MoEAlltoAllTokenDispatcher.token_combine(
        owner, hidden_states, async_finish=False, allocate_on_comm_stream=False
    )

    assert result is combined
    assert sink.records[0]["ctx"] == {
        "comm_type": "ep-alltoall",
        "dispatcher": "alltoall",
        "data_bytes": hidden_states.numel() * hidden_states.element_size(),
        "group_size": 2,
        "ep_size": 2,
        "tp_size": 3,
    }
    assert operations == [
        (
            "all-to-all",
            ("ep-alltoall-combine",),
            owner.ep_group,
            hidden_states,
            owner.input_splits,
            owner.output_splits,
        )
    ]


def test_standard_alltoall_group_one_still_emits_source_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    operations: list[tuple[Any, ...]] = []
    owner = _alltoall_owner(sink, operations, group_size=1)
    hidden_states = torch.ones((2, 2))
    monkeypatch.setattr(
        token_dispatcher_module,
        "all_to_all",
        lambda group, tensor, output_splits, input_splits: tensor,
    )

    result = MoEAlltoAllTokenDispatcher.token_combine(owner, hidden_states)

    assert result is hidden_states
    assert sink.records[0]["ctx"]["group_size"] == 1
    assert sink.transitions == [
        ("B", "ep-alltoall-combine", None),
        ("E", "ep-alltoall-combine", None),
    ]


@pytest.mark.parametrize("gate_mode", ["null", "disabled", "suppressed"])
def test_closed_standard_alltoall_gates_skip_context_and_preserve_calls(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    if gate_mode != "null":
        install_trace_sink(
            sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None
        )
    operations: list[tuple[Any, ...]] = []
    owner = _alltoall_owner(sink, operations)
    tokens = torch.ones((2, 2))
    probs = torch.ones(2)
    monkeypatch.setattr(
        token_dispatcher_module,
        "ep_collective_trace_context",
        lambda *args, **kwargs: pytest.fail("closed EP gate constructed context"),
    )
    monkeypatch.setattr(
        token_dispatcher_module,
        "all_to_all",
        lambda group, tensor, output_splits, input_splits: tensor,
    )

    dispatched = MoEAlltoAllTokenDispatcher.token_dispatch(owner, tokens, probs)
    combined = MoEAlltoAllTokenDispatcher.token_combine(owner, tokens)

    assert dispatched == (tokens, probs)
    assert combined is tokens
    assert sink.records == []


@pytest.mark.parametrize(
    "method,event_name,args",
    [
        (
            MoEAlltoAllTokenDispatcher.token_dispatch,
            "ep-alltoall-dispatch",
            (torch.ones((2, 2)), torch.ones(2)),
        ),
        (MoEAlltoAllTokenDispatcher.token_combine, "ep-alltoall-combine", (torch.ones((2, 2)),)),
    ],
)
def test_standard_alltoall_errors_close_scope_and_preserve_exception_identity(
    monkeypatch: pytest.MonkeyPatch, method, event_name: str, args: tuple[torch.Tensor, ...]
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    owner = _alltoall_owner(sink, [])
    error = RuntimeError("all-to-all failed")
    monkeypatch.setattr(
        token_dispatcher_module, "all_to_all", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )

    with pytest.raises(RuntimeError) as raised:
        method(owner, *args)

    assert raised.value is error
    assert sink.transitions == [("B", event_name, None), ("E", event_name, RuntimeError)]
    assert sink.active == []


def test_real_adapter_records_standard_alltoall_source_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="full")
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    ticks: list[tuple[str, str, dict[str, Any]]] = []
    tracer._tick = lambda name, phase, attrs: ticks.append((name, phase, dict(attrs)))
    install_trace_sink(MegaLensTraceSink(tracer))
    owner = _alltoall_owner(_RecordingSink(), [])
    hidden_states = torch.ones((2, 2))
    monkeypatch.setattr(
        token_dispatcher_module,
        "all_to_all",
        lambda group, tensor, output_splits, input_splits: tensor,
    )

    output = MoEAlltoAllTokenDispatcher.token_combine(owner, hidden_states)

    assert output is hidden_states
    assert ticks == [
        (
            "ep-alltoall-combine",
            "B",
            {
                "comm_type": "ep-alltoall",
                "dispatcher": "alltoall",
                "data_bytes": 16,
                "group_size": 2,
                "ep_size": 2,
                "tp_size": 3,
            },
        ),
        ("ep-alltoall-combine", "E", {}),
    ]


def test_standard_alltoall_probe_markers_and_public_signatures_remain_stable() -> None:
    assert (
        getattr(MoEAlltoAllTokenDispatcher.token_dispatch, "__megatron_trace_event__", None)
        == "ep-alltoall-dispatch"
    )
    assert (
        getattr(MoEAlltoAllTokenDispatcher.token_combine, "__megatron_trace_event__", None)
        == "ep-alltoall-combine"
    )
    assert list(inspect.signature(MoEAlltoAllTokenDispatcher.token_dispatch).parameters) == [
        "self",
        "permutated_local_input_tokens",
        "permuted_probs",
    ]
    assert list(inspect.signature(MoEAlltoAllTokenDispatcher.token_combine).parameters) == [
        "self",
        "hidden_states",
        "async_finish",
        "allocate_on_comm_stream",
    ]


def test_allgather_dispatch_preserves_hidden_only_payload_and_three_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    owner = _allgather_owner()
    routing_map = owner.routing_map
    probs = torch.arange(4, dtype=torch.float16).reshape(2, 2)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    gathered_map = torch.cat((routing_map, routing_map))
    gathered_probs = torch.cat((probs, probs))
    gathered_hidden = torch.cat((hidden_states, hidden_states))
    outputs = iter((gathered_map, gathered_probs, gathered_hidden))
    calls: list[tuple[Any, ...]] = []

    def gather(tensor, *, group, use_global_buffer=False):
        calls.append(
            (
                "gather",
                tuple(sink.active),
                tensor,
                group,
                use_global_buffer,
                torch.is_grad_enabled(),
            )
        )
        return next(outputs)

    monkeypatch.setattr(token_dispatcher_module, "gather_from_sequence_parallel_region", gather)

    output_hidden, output_probs = MoEAllGatherTokenDispatcher.token_dispatch(
        owner, hidden_states, probs
    )

    assert output_hidden is gathered_hidden
    assert output_probs is gathered_probs
    assert owner.routing_map is gathered_map
    assert sink.records == [
        {
            "name": "ep-allgather-dispatch",
            "ctx": {
                "comm_type": "ep-allgather",
                "dispatcher": "allgather",
                "data_bytes": hidden_states.numel() * hidden_states.element_size(),
                "group_size": 4,
                "ep_size": 2,
                "tp_size": 2,
            },
        }
    ]
    assert [call[:5] for call in calls] == [
        ("gather", ("ep-allgather-dispatch",), routing_map, owner.tp_ep_group, False),
        ("gather", ("ep-allgather-dispatch",), probs, owner.tp_ep_group, False),
        ("gather", ("ep-allgather-dispatch",), hidden_states, owner.tp_ep_group, True),
    ]
    assert calls[0][5] is False
    assert calls[1][5] is True
    assert calls[2][5] is True


def test_allgather_combine_preserves_reduce_scatter_cast_boundary_and_source_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    owner = _allgather_owner()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    reduced = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    calls: list[tuple[Any, ...]] = []

    def reduce_scatter(tensor, *, group):
        calls.append(("reduce-scatter", tuple(sink.active), tensor, group))
        return reduced

    monkeypatch.setattr(
        token_dispatcher_module, "reduce_scatter_to_sequence_parallel_region", reduce_scatter
    )

    output = MoEAllGatherTokenDispatcher.token_combine(owner, hidden_states)

    assert output.dtype is hidden_states.dtype
    assert torch.equal(output, reduced.float())
    assert calls[0][0:2] == ("reduce-scatter", ("ep-allgather-combine",))
    assert calls[0][2].dtype is owner.local_probs.dtype
    assert calls[0][3] is owner.tp_ep_group
    assert sink.records == [
        {
            "name": "ep-allgather-combine",
            "ctx": {
                "comm_type": "ep-reduce-scatter",
                "dispatcher": "allgather",
                "data_bytes": hidden_states.numel() * hidden_states.element_size(),
                "group_size": 4,
                "ep_size": 2,
                "tp_size": 2,
            },
        }
    ]


def test_allgather_group_one_emits_paired_source_events_without_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    owner = _allgather_owner(tp_size=1, ep_size=1, group_size=1)
    hidden_states = torch.ones((2, 2))
    probs = torch.ones((2, 2))
    monkeypatch.setattr(
        token_dispatcher_module,
        "gather_from_sequence_parallel_region",
        lambda *args, **kwargs: pytest.fail("group-one AllGather called a collective"),
    )
    monkeypatch.setattr(
        token_dispatcher_module,
        "reduce_scatter_to_sequence_parallel_region",
        lambda *args, **kwargs: pytest.fail("group-one combine called a collective"),
    )

    dispatched = MoEAllGatherTokenDispatcher.token_dispatch(owner, hidden_states, probs)
    combined = MoEAllGatherTokenDispatcher.token_combine(owner, hidden_states)

    assert dispatched[0] is hidden_states
    assert dispatched[1] is probs
    assert combined is hidden_states
    assert [record["name"] for record in sink.records] == [
        "ep-allgather-dispatch",
        "ep-allgather-combine",
    ]
    assert [record["ctx"]["group_size"] for record in sink.records] == [1, 1]


@pytest.mark.parametrize("gate_mode", ["null", "disabled", "suppressed"])
def test_closed_allgather_gates_skip_context_and_preserve_collectives(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    if gate_mode != "null":
        install_trace_sink(
            sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None
        )
    owner = _allgather_owner()
    owner.local_probs = torch.ones(2, dtype=torch.float32)
    hidden_states = torch.ones((2, 2))
    probs = torch.ones((2, 2))
    calls: list[str] = []
    monkeypatch.setattr(
        token_dispatcher_module,
        "ep_collective_trace_context",
        lambda *args, **kwargs: pytest.fail("closed EP gate constructed context"),
    )

    def gather(tensor, **kwargs):
        calls.append("gather")
        return tensor

    def reduce_scatter(tensor, **kwargs):
        calls.append("reduce-scatter")
        return tensor

    monkeypatch.setattr(token_dispatcher_module, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(
        token_dispatcher_module, "reduce_scatter_to_sequence_parallel_region", reduce_scatter
    )

    dispatched = MoEAllGatherTokenDispatcher.token_dispatch(owner, hidden_states, probs)
    combined = MoEAllGatherTokenDispatcher.token_combine(owner, hidden_states)

    assert dispatched[0] is hidden_states
    assert dispatched[1] is probs
    assert combined is hidden_states
    assert calls == ["gather", "gather", "gather", "reduce-scatter"]
    assert sink.records == []


@pytest.mark.parametrize(
    "method,event_name,args,collective_name",
    [
        (
            MoEAllGatherTokenDispatcher.token_dispatch,
            "ep-allgather-dispatch",
            (torch.ones((2, 2)), torch.ones((2, 2))),
            "gather_from_sequence_parallel_region",
        ),
        (
            MoEAllGatherTokenDispatcher.token_combine,
            "ep-allgather-combine",
            (torch.ones((2, 2)),),
            "reduce_scatter_to_sequence_parallel_region",
        ),
    ],
)
def test_allgather_errors_close_scope_and_preserve_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
    method,
    event_name: str,
    args: tuple[torch.Tensor, ...],
    collective_name: str,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    owner = _allgather_owner()
    error = RuntimeError("TPxEP collective failed")
    monkeypatch.setattr(
        token_dispatcher_module,
        collective_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError) as raised:
        method(owner, *args)

    assert raised.value is error
    assert sink.transitions == [("B", event_name, None), ("E", event_name, RuntimeError)]
    assert sink.active == []


def test_real_adapter_records_allgather_combine_source_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="full")
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    ticks: list[tuple[str, str, dict[str, Any]]] = []
    tracer._tick = lambda name, phase, attrs: ticks.append((name, phase, dict(attrs)))
    install_trace_sink(MegaLensTraceSink(tracer))
    owner = _allgather_owner()
    owner.local_probs = torch.ones(2, dtype=torch.float32)
    hidden_states = torch.ones((2, 2))
    monkeypatch.setattr(
        token_dispatcher_module,
        "reduce_scatter_to_sequence_parallel_region",
        lambda tensor, **kwargs: tensor,
    )

    output = MoEAllGatherTokenDispatcher.token_combine(owner, hidden_states)

    assert output is hidden_states
    assert ticks == [
        (
            "ep-allgather-combine",
            "B",
            {
                "comm_type": "ep-reduce-scatter",
                "dispatcher": "allgather",
                "data_bytes": 16,
                "group_size": 4,
                "ep_size": 2,
                "tp_size": 2,
            },
        ),
        ("ep-allgather-combine", "E", {}),
    ]


def test_allgather_probe_markers_and_public_signatures_remain_stable() -> None:
    assert (
        getattr(MoEAllGatherTokenDispatcher.token_dispatch, "__megatron_trace_event__", None)
        == "ep-allgather-dispatch"
    )
    assert (
        getattr(MoEAllGatherTokenDispatcher.token_combine, "__megatron_trace_event__", None)
        == "ep-allgather-combine"
    )
    assert list(inspect.signature(MoEAllGatherTokenDispatcher.token_dispatch).parameters) == [
        "self",
        "hidden_states",
        "probs",
    ]
    assert list(inspect.signature(MoEAllGatherTokenDispatcher.token_combine).parameters) == [
        "self",
        "hidden_states",
    ]
