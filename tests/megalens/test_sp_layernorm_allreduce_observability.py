# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink

finalize_grads = importlib.import_module("megatron.core.distributed.finalize_model_grads")

ROOT = Path(__file__).resolve().parents[2]


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.active.append(self.record["name"])
        self.sink.events.append(("scope-B", self.record["name"]))
        self.sink.transitions.append(("B", self.record["name"], None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self.sink.active.pop() == self.record["name"]
        self.sink.events.append(("scope-E", self.record["name"], exc_type))
        self.sink.transitions.append(("E", self.record["name"], exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.sink.events.append(("scope-set", self.record["name"], key, value))
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []
        self.events: list[tuple[Any, ...]] = []
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


class _Group:
    def __init__(self, size: int = 2) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _Param:
    def __init__(
        self,
        values: Sequence[float] | None,
        *,
        requires_grad: bool = True,
        sequence_parallel: bool = False,
        average: bool = False,
    ) -> None:
        self.requires_grad = requires_grad
        self.grad = None if values is None else torch.tensor(values, dtype=torch.float32)
        self.sequence_parallel = sequence_parallel
        self.average_gradients_across_tp_domain = average


class _Model:
    def __init__(self, named_params: Sequence[tuple[str, _Param]]) -> None:
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        self._named_params = list(named_params)

    def named_parameters(self):
        return list(self._named_params)


@pytest.fixture(autouse=True)
def _reset_sink(monkeypatch: pytest.MonkeyPatch):
    reset_trace_sink()
    monkeypatch.setattr(
        finalize_grads,
        "get_tensor_model_parallel_group_if_none",
        lambda group, *args, **kwargs: group,
    )
    yield
    reset_trace_sink()


def _config(*, sequence_parallel: bool = True, qk_layernorm: bool = True) -> SimpleNamespace:
    return SimpleNamespace(sequence_parallel=sequence_parallel, qk_layernorm=qk_layernorm)


def test_sp_layernorm_scopes_sum_and_average_buckets_with_source_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _Group()
    sp_param = _Param([1.0, 2.0], sequence_parallel=True)
    qk_param = _Param([3.0])
    avg_param = _Param([4.0, 5.0], sequence_parallel=True, average=True)
    excluded_param = _Param([6.0])
    frozen_param = _Param([7.0], requires_grad=False, sequence_parallel=True)
    missing_grad = _Param(None, sequence_parallel=True)
    model = _Model(
        [
            ("decoder.input_layernorm.weight", sp_param),
            ("decoder.q_layernorm.weight", qk_param),
            ("decoder.average.weight", avg_param),
            ("decoder.linear.weight", excluded_param),
            ("decoder.frozen_layernorm.weight", frozen_param),
            ("decoder.missing_layernorm.weight", missing_grad),
        ]
    )
    collective_calls: list[tuple[torch.Tensor, object, object, tuple[str, ...]]] = []

    monkeypatch.setattr(finalize_grads, "get_pg_size", lambda actual: actual.size())

    def peer_ranks(actual):
        assert actual is group
        sink.events.append(("peer-query", tuple(sink.active)))
        return [1, 3]

    monkeypatch.setattr(finalize_grads, "get_process_group_peer_ranks", peer_ranks)

    def all_reduce(tensor, *, op=None, group=None):
        collective_calls.append((tensor.clone(), op, group, tuple(sink.active)))
        sink.events.append(("collective", op, tuple(sink.active)))
        tensor.add_(10.0 if op == torch.distributed.ReduceOp.SUM else 20.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    result = finalize_grads._allreduce_non_tensor_model_parallel_grads([model], _config(), group)

    assert result is None
    assert len(collective_calls) == 2
    assert collective_calls[0][1:] == (
        torch.distributed.ReduceOp.SUM,
        group,
        ("sp-layernorm-allreduce",),
    )
    assert collective_calls[1][1:] == (
        torch.distributed.ReduceOp.AVG,
        group,
        ("sp-layernorm-allreduce",),
    )
    assert torch.equal(collective_calls[0][0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(collective_calls[1][0], torch.tensor([4.0, 5.0]))
    assert torch.equal(sp_param.grad, torch.tensor([11.0, 12.0]))
    assert torch.equal(qk_param.grad, torch.tensor([13.0]))
    assert torch.equal(avg_param.grad, torch.tensor([24.0, 25.0]))
    assert torch.equal(excluded_param.grad, torch.tensor([6.0]))
    assert torch.equal(frozen_param.grad, torch.tensor([7.0]))
    assert missing_grad.grad is None

    assert [record["ctx"] for record in sink.records] == [
        {"data_bytes": 12, "group_size": 2, "reduce_op": "SUM", "grad_bucket": "sum"},
        {"data_bytes": 8, "group_size": 2, "reduce_op": "AVG", "grad_bucket": "avg"},
    ]
    assert [record["slots"] for record in sink.records] == [("group",), ("group",)]
    assert [record["attrs"] for record in sink.records] == [{}, {}]
    assert [record["values"] for record in sink.records] == [{"group": [1, 3]}, {"group": [1, 3]}]
    assert sink.events == [
        ("scope-B", "sp-layernorm-allreduce"),
        ("collective", torch.distributed.ReduceOp.SUM, ("sp-layernorm-allreduce",)),
        ("peer-query", ("sp-layernorm-allreduce",)),
        ("scope-set", "sp-layernorm-allreduce", "group", [1, 3]),
        ("scope-E", "sp-layernorm-allreduce", None),
        ("scope-B", "sp-layernorm-allreduce"),
        ("collective", torch.distributed.ReduceOp.AVG, ("sp-layernorm-allreduce",)),
        ("peer-query", ("sp-layernorm-allreduce",)),
        ("scope-set", "sp-layernorm-allreduce", "group", [1, 3]),
        ("scope-E", "sp-layernorm-allreduce", None),
    ]


@pytest.mark.parametrize("gate_mode", ["disabled", "suppressed"])
def test_sp_layernorm_closed_gate_preserves_collectives_and_skips_metadata(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    install_trace_sink(sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None)
    group = _Group()
    sum_param = _Param([1.0], sequence_parallel=True)
    avg_param = _Param([2.0], average=True)
    model = _Model([("layernorm.weight", sum_param), ("average.weight", avg_param)])
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        finalize_grads,
        "get_pg_size",
        lambda group: pytest.fail("closed gate queried group size metadata"),
    )
    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("closed gate queried group peers"),
    )

    def all_reduce(tensor, *, op=None, group=None):
        calls.append((op, group))
        tensor.add_(1.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    finalize_grads._allreduce_non_tensor_model_parallel_grads([model], _config(), group)

    assert calls == [
        (torch.distributed.ReduceOp.SUM, group),
        (torch.distributed.ReduceOp.AVG, group),
    ]
    assert torch.equal(sum_param.grad, torch.tensor([2.0]))
    assert torch.equal(avg_param.grad, torch.tensor([3.0]))
    assert sink.records == []
    assert sink.transitions == []


def test_sp_layernorm_collective_error_closes_scope_and_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _Group()
    param = _Param([1.0], sequence_parallel=True)
    model = _Model([("layernorm.weight", param)])
    error = RuntimeError("collective failed")

    monkeypatch.setattr(finalize_grads, "get_pg_size", lambda actual: actual.size())
    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("failed collective queried group peers"),
    )
    monkeypatch.setattr(
        finalize_grads,
        "_unflatten_dense_tensors",
        lambda *args, **kwargs: pytest.fail("failed collective copied gradients back"),
    )

    def fail_all_reduce(tensor, *, op=None, group=None):
        sink.events.append(("collective", op, tuple(sink.active)))
        raise error

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", fail_all_reduce)

    with pytest.raises(RuntimeError) as exc_info:
        finalize_grads._allreduce_non_tensor_model_parallel_grads([model], _config(), group)

    assert exc_info.value is error
    assert sink.transitions == [
        ("B", "sp-layernorm-allreduce", None),
        ("E", "sp-layernorm-allreduce", RuntimeError),
    ]
    assert sink.records[0]["values"] == {}
    assert sink.active == []
    assert torch.equal(param.grad, torch.tensor([1.0]))


def test_sp_layernorm_prefers_main_grad_and_skips_present_none_main_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _Group()
    main_param = _Param([100.0], sequence_parallel=True)
    main_param.main_grad = torch.tensor([2.0])
    missing_main_param = _Param([3.0], sequence_parallel=True)
    missing_main_param.main_grad = None
    model = _Model(
        [
            ("main_layernorm.weight", main_param),
            ("missing_main_layernorm.weight", missing_main_param),
        ]
    )

    monkeypatch.setattr(finalize_grads, "get_pg_size", lambda actual: actual.size())
    monkeypatch.setattr(finalize_grads, "get_process_group_peer_ranks", lambda actual: [1])

    def all_reduce(tensor, *, op=None, group=None):
        assert op == torch.distributed.ReduceOp.SUM
        tensor.add_(4.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    finalize_grads._allreduce_non_tensor_model_parallel_grads([model], _config(), group)

    assert torch.equal(main_param.main_grad, torch.tensor([6.0]))
    assert torch.equal(main_param.grad, torch.tensor([100.0]))
    assert missing_main_param.main_grad is None
    assert torch.equal(missing_main_param.grad, torch.tensor([3.0]))
    assert sink.records[0]["ctx"]["data_bytes"] == 4


def test_sp_layernorm_unavailable_peer_metadata_does_not_change_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _Group()
    param = _Param([1.0], sequence_parallel=True)
    model = _Model([("layernorm.weight", param)])
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(finalize_grads, "get_pg_size", lambda actual: actual.size())
    monkeypatch.setattr(finalize_grads, "get_process_group_peer_ranks", lambda actual: None)

    def all_reduce(tensor, *, op=None, group=None):
        calls.append((op, group))
        tensor.add_(2.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    finalize_grads._allreduce_non_tensor_model_parallel_grads([model], _config(), group)

    assert calls == [(torch.distributed.ReduceOp.SUM, group)]
    assert torch.equal(param.grad, torch.tensor([3.0]))
    assert sink.records[0]["values"] == {"group": None}
    assert sink.transitions == [
        ("B", "sp-layernorm-allreduce", None),
        ("E", "sp-layernorm-allreduce", None),
    ]


def test_sp_layernorm_single_rank_and_empty_buckets_emit_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    monkeypatch.setattr(
        finalize_grads.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("empty path issued a collective"),
    )

    class _SingleRankModel:
        ddp_config = SimpleNamespace(use_megatron_fsdp=False)

        def named_parameters(self):
            pytest.fail("single-rank path inspected parameters")

    finalize_grads._allreduce_non_tensor_model_parallel_grads(
        [_SingleRankModel()], _config(), _Group(size=1)
    )
    empty_model = _Model([("linear.weight", _Param([1.0]))])
    finalize_grads._allreduce_non_tensor_model_parallel_grads(
        [empty_model], _config(), _Group(size=2)
    )

    assert sink.records == []
    assert sink.transitions == []


def test_sp_layernorm_scope_is_sibling_of_all_grads_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _Group()
    param = _Param([1.0], sequence_parallel=True)

    class _FinalizeModel(_Model):
        def __init__(self) -> None:
            super().__init__([("layernorm.weight", param)])
            self.config = SimpleNamespace(
                timers=None,
                barrier_with_L1_time=False,
                moe_router_enable_expert_bias=False,
                sequence_parallel=True,
                qk_layernorm=False,
            )

        def finish_grad_sync(self, *, force_all_reduce: bool = False) -> None:
            sink.events.append(("finish", force_all_reduce, tuple(sink.active)))

    model = _FinalizeModel()
    groups = SimpleNamespace(
        tp=group, pp=object(), embd=object(), pos_embd=object(), dp_cp=object()
    )
    helper_events: list[tuple[str, tuple[str, ...]]] = []

    for helper_name in (
        "_allreduce_conditional_embedding_grads",
        "_allreduce_word_embedding_grads",
        "_allreduce_position_embedding_grads",
        "reset_model_temporary_tensors",
    ):

        def record_helper(*args, _name=helper_name, **kwargs):
            helper_events.append((_name, tuple(sink.active)))

        monkeypatch.setattr(finalize_grads, helper_name, record_helper)

    monkeypatch.setattr(finalize_grads, "get_pg_size", lambda actual: actual.size())
    monkeypatch.setattr(finalize_grads, "get_process_group_peer_ranks", lambda actual: [1])

    def all_reduce(tensor, *, op=None, group=None):
        sink.events.append(("collective", op, tuple(sink.active)))

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    finalize_grads.finalize_model_grads([model], pg_collection=groups, force_all_reduce=True)

    assert ("finish", True, ("all-grads-sync",)) in sink.events
    assert (
        "collective",
        torch.distributed.ReduceOp.SUM,
        ("sp-layernorm-allreduce",),
    ) in sink.events
    assert sink.transitions == [
        ("B", "all-grads-sync", None),
        ("E", "all-grads-sync", None),
        ("B", "sp-layernorm-allreduce", None),
        ("E", "sp-layernorm-allreduce", None),
    ]
    assert helper_events == [
        ("_allreduce_conditional_embedding_grads", ()),
        ("_allreduce_word_embedding_grads", ()),
        ("_allreduce_position_embedding_grads", ()),
        ("reset_model_temporary_tensors", ()),
    ]


def test_sp_layernorm_tp_group_list_retains_existing_unsupported_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    model = _Model([("layernorm.weight", _Param([1.0], sequence_parallel=True))])
    monkeypatch.setattr(
        finalize_grads.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("unsupported list issued a collective"),
    )

    with pytest.raises(AttributeError, match="size"):
        finalize_grads._allreduce_non_tensor_model_parallel_grads([model], _config(), [_Group()])

    assert sink.records == []


def test_sp_layernorm_producer_is_dependency_light_and_narrow() -> None:
    source_path = ROOT / "megatron/core/distributed/finalize_model_grads.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "megatron.training" not in source
    assert "megatron.megalens" not in source

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reducer = functions["_allreduce_non_tensor_model_parallel_grads"]
    producer_calls = [
        node
        for node in ast.walk(reducer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_all_reduce_with_trace"
    ]
    assert len(producer_calls) == 1
    assert isinstance(producer_calls[0].args[2], ast.Constant)
    assert producer_calls[0].args[2].value == "sp-layernorm-allreduce"

    helper = functions["_all_reduce_with_trace"]
    helper_names = {
        node.func.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"prepare_trace_scope", "open_trace_scope"} <= helper_names
    assert "trace_scope" not in helper_names
