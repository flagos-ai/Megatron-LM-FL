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
    def __init__(self, name: str, *, size: int = 2, peers: Sequence[int] = (0, 1)) -> None:
        self.name = name
        self._size = size
        self.peers = list(peers)

    def size(self) -> int:
        return self._size


class _ConditionalParam:
    def __init__(
        self,
        values: Sequence[float] | None,
        *,
        requires_grad: bool = True,
        pipeline_parallel: bool = True,
    ) -> None:
        self.requires_grad = requires_grad
        self.pipeline_parallel = pipeline_parallel
        self.main_grad = None if values is None else torch.tensor(values, dtype=torch.float32)


class _ConditionalChunk:
    def __init__(self, named_params: Sequence[tuple[str, _ConditionalParam]]) -> None:
        self._named_params = list(named_params)

    def named_parameters(self):
        return list(self._named_params)


class _EmbeddingModel:
    def __init__(
        self,
        values: Sequence[Sequence[float]] | None,
        *,
        use_dist_opt: bool = False,
        partial: bool = False,
    ) -> None:
        grad = None if values is None else torch.tensor(values, dtype=torch.float32)
        self.weight = SimpleNamespace(grad=grad)
        self.ddp_config = SimpleNamespace(
            use_megatron_fsdp=False,
            use_distributed_optimizer=use_dist_opt,
            use_partial_reduce_for_shared_embedding=partial,
        )


@pytest.fixture(autouse=True)
def _reset_sink(monkeypatch: pytest.MonkeyPatch):
    reset_trace_sink()
    monkeypatch.setattr(
        finalize_grads,
        "get_attr_wrapped_model",
        lambda model, attr, *args, **kwargs: (
            model if kwargs.get("return_model_obj", False) else getattr(model, attr)
        ),
    )
    yield
    reset_trace_sink()


def _install_embedding_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_stage: bool = True,
    last_stage: bool = False,
    rank: int = 0,
) -> None:
    def group_size(group):
        if isinstance(group, list):
            return group[0].size()
        return group.size()

    monkeypatch.setattr(finalize_grads, "get_pg_size", group_size)
    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda group: [peer for peer in group.peers if peer != rank],
    )
    monkeypatch.setattr(finalize_grads.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        finalize_grads.torch.distributed, "get_process_group_ranks", lambda group: list(group.peers)
    )
    monkeypatch.setattr(finalize_grads, "is_pp_first_stage", lambda group: first_stage)
    monkeypatch.setattr(finalize_grads, "is_pp_last_stage", lambda group: last_stage)
    monkeypatch.setattr(finalize_grads, "get_device_type_for_comm", lambda group: "cpu")
    monkeypatch.setattr(
        finalize_grads, "cur_platform", SimpleNamespace(current_device=lambda: torch.device("cpu"))
    )


def _run_embedding_grad(
    models: Sequence[_EmbeddingModel],
    embd_group,
    *,
    config: SimpleNamespace | None = None,
    trace_kind: str = "word",
) -> None:
    finalize_grads._allreduce_embedding_grad(
        list(models),
        embd_group,
        object(),
        lambda model: model.weight,
        config=config,
        trace_kind=trace_kind,
    )


def test_conditional_embedding_scope_uses_flattened_vpp_grad_and_source_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    group = _Group("pp", peers=(0, 2))
    first = _ConditionalParam([1.0, 2.0])
    second = _ConditionalParam([3.0, 4.0])
    ignored = _ConditionalParam([7.0], pipeline_parallel=False)
    chunks = [
        _ConditionalChunk([("cond.weight", first), ("ignored.weight", ignored)]),
        _ConditionalChunk([("cond.weight", second)]),
    ]
    original_flatten = finalize_grads._flatten_dense_tensors
    original_unflatten = finalize_grads._unflatten_dense_tensors

    def flatten(grads):
        sink.events.append(("flatten", tuple(sink.active)))
        return original_flatten(grads)

    def unflatten(flattened, grads):
        sink.events.append(("unflatten", tuple(sink.active)))
        return original_unflatten(flattened, grads)

    def all_reduce(tensor, *, group=None):
        assert group is not None
        sink.events.append(("collective", tensor.clone(), group.name, tuple(sink.active)))
        tensor.add_(10.0)

    monkeypatch.setattr(finalize_grads, "_flatten_dense_tensors", flatten)
    monkeypatch.setattr(finalize_grads, "_unflatten_dense_tensors", unflatten)
    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    finalize_grads._allreduce_conditional_embedding_grads(
        chunks, SimpleNamespace(has_cond_embedder=True), group
    )

    assert torch.equal(first.main_grad, torch.tensor([14.0, 16.0]))
    assert torch.equal(second.main_grad, torch.tensor([14.0, 16.0]))
    assert torch.equal(ignored.main_grad, torch.tensor([7.0]))
    assert sink.records == [
        {
            "name": "embedding-grads-allreduce",
            "ctx": {"data_bytes": 8, "group_size": 2, "embedding_kind": "conditional"},
            "slots": ("group",),
            "attrs": {},
            "values": {"group": [2]},
        }
    ]
    assert sink.events[0] == ("flatten", ())
    assert sink.events[1] == ("scope-B", "embedding-grads-allreduce")
    assert sink.events[2][0] == "collective"
    assert torch.equal(sink.events[2][1], torch.tensor([4.0, 6.0]))
    assert sink.events[2][2:] == ("pp", ("embedding-grads-allreduce",))
    assert sink.events[-2:] == [("scope-E", "embedding-grads-allreduce", None), ("unflatten", ())]


def test_conditional_embedding_empty_and_pp_list_boundaries_emit_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    monkeypatch.setattr(
        finalize_grads.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("empty conditional path issued a collective"),
    )
    empty = _ConditionalChunk([("frozen", _ConditionalParam([1.0], requires_grad=False))])

    finalize_grads._allreduce_conditional_embedding_grads(
        [empty], SimpleNamespace(has_cond_embedder=True), _Group("pp")
    )
    finalize_grads._allreduce_conditional_embedding_grads(
        [empty], SimpleNamespace(has_cond_embedder=False), [_Group("pp")]
    )
    with pytest.raises(AssertionError, match="does not support"):
        finalize_grads._allreduce_conditional_embedding_grads(
            [empty], SimpleNamespace(has_cond_embedder=True), [_Group("pp")]
        )

    assert sink.records == []
    assert sink.transitions == []


@pytest.mark.parametrize(
    ("stage", "first_stage", "last_stage", "mtp_layers", "selected_index"),
    [
        ("first", True, False, 1, 0),
        ("last", False, True, 0, 1),
        ("mtp-middle", False, False, 1, 1),
        ("plain-middle", False, False, 0, 0),
    ],
)
def test_single_group_embedding_scope_preserves_target_stage_selection(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    first_stage: bool,
    last_stage: bool,
    mtp_layers: int,
    selected_index: int,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch, first_stage=first_stage, last_stage=last_stage)
    group = _Group("embedding", size=3, peers=(0, 2, 4))
    models = [_EmbeddingModel([[1.0, 2.0]]), _EmbeddingModel([[10.0, 20.0]])]

    def all_reduce(tensor, *, group=None):
        sink.events.append(("collective", stage, group.name, tuple(sink.active)))
        tensor.add_(5.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    _run_embedding_grad(
        models, group, config=SimpleNamespace(mtp_num_layers=mtp_layers), trace_kind="word"
    )

    expected = [torch.tensor([[1.0, 2.0]]), torch.tensor([[10.0, 20.0]])]
    expected[selected_index] = expected[selected_index] + 5.0
    assert torch.equal(models[0].weight.grad, expected[0])
    assert torch.equal(models[1].weight.grad, expected[1])
    assert sink.records[0]["ctx"] == {"data_bytes": 8, "group_size": 3, "embedding_kind": "word"}
    assert sink.records[0]["values"] == {"group": [2, 4]}
    assert sink.events[1] == ("collective", stage, "embedding", ("embedding-grads-allreduce",))


def test_word_and_position_wrappers_forward_source_kinds_without_changing_target_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    config = SimpleNamespace(mtp_num_layers=2)
    word_group = object()
    position_group = object()
    pp_group = object()

    monkeypatch.setattr(
        finalize_grads,
        "_allreduce_embedding_grad",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    finalize_grads._allreduce_word_embedding_grads([object()], config, word_group, pp_group)
    finalize_grads._allreduce_position_embedding_grads([object()], config, position_group, pp_group)

    assert calls[0][0][1:3] == (word_group, pp_group)
    assert calls[0][1] == {"config": config, "trace_kind": "word"}
    assert calls[1][0][1:3] == (position_group, pp_group)
    assert calls[1][1] == {"skip_if_none": False, "trace_kind": "position"}


@pytest.mark.parametrize("gate_mode", ["disabled", "suppressed"])
def test_single_group_closed_gate_preserves_collective_and_skips_trace_metadata(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    install_trace_sink(sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None)
    _install_embedding_runtime(monkeypatch)
    group = _Group("embedding")
    model = _EmbeddingModel([[1.0]])
    size_calls = 0

    def group_size(actual):
        nonlocal size_calls
        assert actual is group
        size_calls += 1
        if size_calls > 1:
            pytest.fail("closed trace gate queried group size metadata")
        return actual.size()

    monkeypatch.setattr(finalize_grads, "get_pg_size", group_size)
    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda actual: pytest.fail("closed trace gate queried group peers"),
    )

    def all_reduce(tensor, *, group=None):
        tensor.add_(3.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    _run_embedding_grad([model], group, config=SimpleNamespace(mtp_num_layers=0))

    assert size_calls == 1
    assert torch.equal(model.weight.grad, torch.tensor([[4.0]]))
    assert sink.records == []
    assert sink.transitions == []


def test_embedding_ineligible_and_frozen_paths_emit_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    monkeypatch.setattr(
        finalize_grads.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("ineligible embedding path issued a collective"),
    )

    _run_embedding_grad(
        [_EmbeddingModel([[1.0]])],
        _Group("single", size=1, peers=(0,)),
        config=SimpleNamespace(mtp_num_layers=0),
    )

    monkeypatch.setattr(
        finalize_grads.torch.distributed, "get_process_group_ranks", lambda group: [3, 5]
    )
    _run_embedding_grad(
        [_EmbeddingModel([[1.0]])],
        _Group("nonmember", peers=(3, 5)),
        config=SimpleNamespace(mtp_num_layers=0),
    )

    monkeypatch.setattr(
        finalize_grads.torch.distributed, "get_process_group_ranks", lambda group: list(group.peers)
    )
    frozen = _EmbeddingModel(None)
    _run_embedding_grad([frozen], _Group("active"), config=SimpleNamespace(mtp_num_layers=0))
    finalize_grads._allreduce_embedding_grad(
        [_EmbeddingModel([[1.0]])],
        _Group("active"),
        object(),
        lambda model: None,
        config=SimpleNamespace(mtp_num_layers=0),
        trace_kind="word",
    )

    assert sink.records == []
    assert sink.transitions == []


@pytest.mark.parametrize(
    ("case", "use_dist_opt", "partial", "group_count", "expected_rows"),
    [
        ("partial-one", True, True, 1, [(4, 8)]),
        ("partial-many", True, True, 2, [(4, 6), (6, 8)]),
        ("dist-full", True, False, 2, [(0, 8)]),
        ("replica-one", False, False, 1, [(0, 8)]),
        ("replica-many", False, False, 2, [(0, 8), (0, 8)]),
    ],
)
def test_flags_scale_group_list_scopes_each_existing_physical_collective(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    use_dist_opt: bool,
    partial: bool,
    group_count: int,
    expected_rows: list[tuple[int, int]],
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    groups = [_Group("group-0", size=2, peers=(0, 1)), _Group("group-1", size=3, peers=(0, 2, 4))][
        :group_count
    ]
    original = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    model = _EmbeddingModel(original.tolist(), use_dist_opt=use_dist_opt, partial=partial)
    calls: list[tuple[torch.Tensor, _Group, tuple[str, ...]]] = []

    monkeypatch.setattr(finalize_grads.parallel_state, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(finalize_grads.parallel_state, "get_data_parallel_rank", lambda: 1)

    def all_reduce(tensor, *, group=None):
        calls.append((tensor.clone(), group, tuple(sink.active)))
        tensor.add_(100.0 * len(calls))

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    _run_embedding_grad(
        [model], groups, config=SimpleNamespace(mtp_num_layers=0), trace_kind="word"
    )

    assert len(calls) == len(expected_rows)
    for index, ((start, end), (actual, group, active)) in enumerate(zip(expected_rows, calls)):
        assert torch.equal(actual, original[start:end])
        assert group is groups[index]
        assert active == ("embedding-grads-allreduce",)
        record = sink.records[index]
        assert record["ctx"] == {
            "data_bytes": (end - start) * 2 * 4,
            "group_size": groups[index].size(),
            "embedding_kind": "word",
        }
        assert record["values"] == {"group": [peer for peer in groups[index].peers if peer != 0]}

    expected = original.clone()
    if case == "partial-one":
        expected[4:8].add_(100.0)
    elif case == "partial-many":
        expected[4:6].add_(100.0)
        expected[6:8].add_(200.0)
    elif case == "replica-many":
        expected.add_(200.0)
    else:
        expected.add_(100.0)
    assert torch.equal(model.weight.grad, expected)


@pytest.mark.parametrize("gate_mode", ["disabled", "suppressed"])
@pytest.mark.parametrize(
    ("case", "use_dist_opt", "partial"),
    [("partial-many", True, True), ("replica-many", False, False)],
)
def test_flags_scale_group_list_closed_gate_preserves_multi_collective_behavior(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str, case: str, use_dist_opt: bool, partial: bool
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    install_trace_sink(sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None)
    _install_embedding_runtime(monkeypatch)
    groups = [_Group("group-0", size=2, peers=(0, 1)), _Group("group-1", size=3, peers=(0, 2, 4))]
    original = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    model = _EmbeddingModel(original.tolist(), use_dist_opt=use_dist_opt, partial=partial)
    calls: list[tuple[torch.Tensor, _Group]] = []
    size_queries: list[Any] = []

    monkeypatch.setattr(finalize_grads.parallel_state, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(finalize_grads.parallel_state, "get_data_parallel_rank", lambda: 1)

    def group_size(actual):
        size_queries.append(actual)
        if not isinstance(actual, list):
            pytest.fail("closed list trace gate queried concrete group size metadata")
        return actual[0].size()

    monkeypatch.setattr(finalize_grads, "get_pg_size", group_size)
    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("closed list trace gate queried group peers"),
    )

    def all_reduce(tensor, *, group=None):
        calls.append((tensor.clone(), group))
        tensor.add_(100.0 * len(calls))

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", all_reduce)

    _run_embedding_grad(
        [model], groups, config=SimpleNamespace(mtp_num_layers=0), trace_kind="word"
    )

    assert size_queries == [groups]
    assert [group for _, group in calls] == groups
    assert len(calls) == 2
    expected = original.clone()
    if case == "partial-many":
        assert torch.equal(calls[0][0], original[4:6])
        assert torch.equal(calls[1][0], original[6:8])
        expected[4:6].add_(100.0)
        expected[6:8].add_(200.0)
    else:
        assert torch.equal(calls[0][0], original)
        assert torch.equal(calls[1][0], original)
        expected.add_(200.0)
    assert torch.equal(model.weight.grad, expected)
    assert sink.records == []
    assert sink.transitions == []


def test_embedding_collective_error_closes_scope_and_stops_list_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    groups = [_Group("group-0"), _Group("group-1")]
    model = _EmbeddingModel([[1.0], [2.0]])
    error = RuntimeError("embedding collective failed")
    calls = 0

    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("failed collective queried group peers"),
    )
    monkeypatch.setattr(
        finalize_grads,
        "_reshard_if_dtensor",
        lambda *args, **kwargs: pytest.fail("failed collective reshared gradients"),
    )

    def fail_all_reduce(tensor, *, group=None):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", fail_all_reduce)

    with pytest.raises(RuntimeError) as exc_info:
        _run_embedding_grad(
            [model], groups, config=SimpleNamespace(mtp_num_layers=0), trace_kind="word"
        )

    assert exc_info.value is error
    assert calls == 1
    assert sink.transitions == [
        ("B", "embedding-grads-allreduce", None),
        ("E", "embedding-grads-allreduce", RuntimeError),
    ]
    assert sink.records[0]["values"] == {}
    assert sink.active == []


def test_conditional_collective_error_skips_unflatten_and_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    group = _Group("pp")
    first = _ConditionalParam([1.0])
    second = _ConditionalParam([2.0])
    chunks = [
        _ConditionalChunk([("cond.weight", first)]),
        _ConditionalChunk([("cond.weight", second)]),
    ]
    error = RuntimeError("conditional collective failed")

    monkeypatch.setattr(
        finalize_grads,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("failed collective queried group peers"),
    )
    monkeypatch.setattr(
        finalize_grads,
        "_unflatten_dense_tensors",
        lambda *args, **kwargs: pytest.fail("failed collective unflattened gradients"),
    )
    monkeypatch.setattr(
        finalize_grads.torch.distributed,
        "all_reduce",
        lambda tensor, *, group=None: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError) as exc_info:
        finalize_grads._allreduce_conditional_embedding_grads(
            chunks, SimpleNamespace(has_cond_embedder=True), group
        )

    assert exc_info.value is error
    assert torch.equal(first.main_grad, torch.tensor([3.0]))
    assert torch.equal(second.main_grad, torch.tensor([2.0]))
    assert sink.transitions == [
        ("B", "embedding-grads-allreduce", None),
        ("E", "embedding-grads-allreduce", RuntimeError),
    ]
    assert sink.records[0]["values"] == {}


def test_embedding_leaf_is_sibling_of_all_grads_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_embedding_runtime(monkeypatch)
    group = _Group("pp")
    param = _ConditionalParam([1.0])

    class _FinalizeModel(_ConditionalChunk):
        def __init__(self) -> None:
            super().__init__([("cond.weight", param)])
            self.config = SimpleNamespace(
                timers=None,
                barrier_with_L1_time=False,
                moe_router_enable_expert_bias=False,
                has_cond_embedder=True,
            )

        def finish_grad_sync(self, *, force_all_reduce: bool = False) -> None:
            sink.events.append(("finish", force_all_reduce, tuple(sink.active)))

    model = _FinalizeModel()
    groups = SimpleNamespace(
        tp=object(), pp=group, embd=object(), pos_embd=object(), dp_cp=object()
    )
    monkeypatch.setattr(finalize_grads, "get_model_config", lambda model: model.config)
    for helper_name in (
        "_allreduce_non_tensor_model_parallel_grads",
        "_allreduce_word_embedding_grads",
        "_allreduce_position_embedding_grads",
        "reset_model_temporary_tensors",
    ):
        monkeypatch.setattr(finalize_grads, helper_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        finalize_grads.torch.distributed,
        "all_reduce",
        lambda tensor, *, group=None: sink.events.append(("collective", tuple(sink.active))),
    )

    finalize_grads.finalize_model_grads([model], pg_collection=groups, force_all_reduce=True)

    assert ("finish", True, ("all-grads-sync",)) in sink.events
    assert ("collective", ("embedding-grads-allreduce",)) in sink.events
    assert sink.transitions == [
        ("B", "all-grads-sync", None),
        ("E", "all-grads-sync", None),
        ("B", "embedding-grads-allreduce", None),
        ("E", "embedding-grads-allreduce", None),
    ]


def test_embedding_producers_are_dependency_light_and_route_all_target_collectives() -> None:
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
    expected_calls = {"_allreduce_conditional_embedding_grads": 1, "_allreduce_embedding_grad": 6}
    for function_name, expected_count in expected_calls.items():
        function = functions[function_name]
        trace_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_all_reduce_with_trace"
        ]
        raw_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "all_reduce"
        ]
        assert len(trace_calls) == expected_count
        assert raw_calls == []
        assert all(
            len(call.args) >= 3
            and isinstance(call.args[2], ast.Constant)
            and call.args[2].value == "embedding-grads-allreduce"
            for call in trace_calls
        )

    helper = functions["_all_reduce_with_trace"]
    helper_names = {
        node.func.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"prepare_trace_scope", "open_trace_scope"} <= helper_names
    assert "trace_scope" not in helper_names
