from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.tensor_parallel import mappings


class _FakeGroup:
    def __init__(self, size: int, *, rank: int = 0, order: list[str] | None = None) -> None:
        self._size = size
        self._rank = rank
        self._order = order

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        if self._order is not None:
            self._order.append("rank")
        return self._rank


class _RecordingScope:
    def __init__(self, record: dict[str, Any], order: list[str]) -> None:
        self.record = record
        self.order = order

    def __enter__(self) -> "_RecordingScope":
        self.record["entered"] = True
        self.order.append(f"B:{self.record['name']}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.record["exit_exception"] = exc_type
        self.record["exit_value"] = exc_value
        self.order.append(f"E:{self.record['name']}")
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record.setdefault("values", {})[key] = value
        return True


class _RecordingSink:
    def __init__(self, *, enabled: bool | set[str] = True, order: list[str] | None = None) -> None:
        self.enabled = enabled
        self.order = order if order is not None else []
        self.queries: list[str] = []
        self.records: list[dict[str, Any]] = []

    def is_enabled(self, name: str) -> bool:
        self.queries.append(name)
        if isinstance(self.enabled, set):
            return name in self.enabled
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
        return _RecordingScope(record, self.order)


class _FakeGlobalMemoryBuffer:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order
        self.calls: list[tuple[list[int], torch.dtype, str]] = []

    def get_tensor(self, shape, dtype, name):
        normalized_shape = list(shape)
        self.calls.append((normalized_shape, dtype, name))
        if self.order is not None:
            self.order.append("allocate")
        return torch.empty(normalized_shape, dtype=dtype)


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def _configure_allocation_device(monkeypatch, *, use_global_buffer: bool) -> None:
    if use_global_buffer:
        monkeypatch.setattr(
            mappings.cur_platform,
            "current_device",
            lambda: pytest.fail("global-buffer path queried the platform device"),
        )
    else:
        monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")


def _install_peer_metadata(monkeypatch, *, ranks=(2, 5), current_rank=2, order=None) -> None:
    def _get_group_ranks(group):
        if order is not None:
            order.append("metadata")
        return list(ranks)

    monkeypatch.setattr(mappings.torch.distributed, "get_process_group_ranks", _get_group_ranks)
    monkeypatch.setattr(mappings.torch.distributed, "get_rank", lambda: current_rank)


@pytest.mark.parametrize("use_global_buffer", [False, True])
def test_tp_reduce_scatter_first_equal_preserves_contract_and_allocation(
    monkeypatch, use_global_buffer
) -> None:
    order: list[str] = []
    sink = _RecordingSink(order=order)
    install_trace_sink(sink)
    _configure_allocation_device(monkeypatch, use_global_buffer=use_global_buffer)
    _install_peer_metadata(monkeypatch, ranks=(3, 7), current_rank=3, order=order)
    group = _FakeGroup(2)
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not tensor.is_contiguous()
    expected = torch.tensor([[40, 41, 42], [43, 44, 45]], dtype=torch.float32)
    collective_calls: list[tuple[torch.Tensor, torch.Tensor, Any]] = []
    global_buffer = _FakeGlobalMemoryBuffer(order)
    monkeypatch.setattr(mappings, "get_global_memory_buffer", lambda: global_buffer)
    real_empty = torch.empty

    def _empty(*args, **kwargs):
        order.append("allocate")
        return real_empty(*args, **kwargs)

    def _reduce_scatter(output, value, *, group):
        order.append("collective")
        collective_calls.append((output, value, group))
        output.copy_(expected)

    if not use_global_buffer:
        monkeypatch.setattr(mappings.torch, "empty", _empty)
    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", _reduce_scatter)

    result = mappings._reduce_scatter_along_first_dim(
        tensor, group, use_global_buffer=use_global_buffer
    )

    assert torch.equal(result, expected)
    assert result is collective_calls[0][0]
    _, collective_input, collective_group = collective_calls[0]
    assert collective_input is not tensor
    assert collective_input.is_contiguous()
    assert torch.equal(collective_input, tensor)
    assert collective_group is group
    assert len(collective_calls) == 1
    assert global_buffer.calls == ([([2, 3], torch.float32, "mpu")] if use_global_buffer else [])
    assert order == [
        "allocate",
        "B:tp-reduce-scatter",
        "collective",
        "metadata",
        "E:tp-reduce-scatter",
    ]
    assert sink.queries == ["tp-reduce-scatter"]
    record = sink.records[0]
    assert record["ctx"] == {
        "data_bytes": 48,
        "group_size": 2,
        "op": "reduce-scatter",
        "dim": "first",
    }
    assert record["slots"] == ("group",)
    assert record["values"] == {"group": [7]}


@pytest.mark.parametrize("use_global_buffer", [False, True])
def test_tp_reduce_scatter_first_uneven_uses_full_input_bytes_and_raw_split_views(
    monkeypatch, use_global_buffer
) -> None:
    order: list[str] = []
    sink = _RecordingSink(order=order)
    install_trace_sink(sink)
    _configure_allocation_device(monkeypatch, use_global_buffer=use_global_buffer)
    _install_peer_metadata(monkeypatch, ranks=(11, 13), current_rank=13, order=order)
    group = _FakeGroup(2, rank=1, order=order)
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4).transpose(0, 1)
    assert not tensor.is_contiguous()
    split_sizes = [3, 1]
    expected = torch.tensor([[90, 91]], dtype=torch.float32)
    collective_calls: list[tuple[torch.Tensor, list[torch.Tensor], Any]] = []
    global_buffer = _FakeGlobalMemoryBuffer(order)
    monkeypatch.setattr(mappings, "get_global_memory_buffer", lambda: global_buffer)
    real_split = torch.split
    real_empty_like = torch.empty_like

    def _split(*args, **kwargs):
        order.append("input-split")
        return real_split(*args, **kwargs)

    def _empty_like(*args, **kwargs):
        order.append("allocate")
        return real_empty_like(*args, **kwargs)

    def _reduce_scatter(output, values, *, group):
        order.append("collective")
        collective_calls.append((output, values, group))
        output.copy_(expected)

    monkeypatch.setattr(mappings.torch, "split", _split)
    if not use_global_buffer:
        monkeypatch.setattr(mappings.torch, "empty_like", _empty_like)
    monkeypatch.setattr(mappings.torch.distributed, "reduce_scatter", _reduce_scatter)

    result = mappings._reduce_scatter_along_first_dim(
        tensor, group, input_split_sizes=split_sizes, use_global_buffer=use_global_buffer
    )

    assert torch.equal(result, expected)
    assert result is collective_calls[0][0]
    _, collective_inputs, collective_group = collective_calls[0]
    assert [list(value.shape) for value in collective_inputs] == [[3, 2], [1, 2]]
    assert torch.equal(collective_inputs[0], tensor[:3])
    assert torch.equal(collective_inputs[1], tensor[3:])
    assert all(not value.is_contiguous() for value in collective_inputs)
    assert collective_group is group
    assert len(collective_calls) == 1
    assert global_buffer.calls == ([([1, 2], torch.float32, "mpu")] if use_global_buffer else [])
    assert order == [
        "rank",
        "input-split",
        "allocate",
        "B:tp-reduce-scatter",
        "collective",
        "metadata",
        "E:tp-reduce-scatter",
    ]
    assert sink.queries == ["tp-reduce-scatter"]
    record = sink.records[0]
    assert record["ctx"]["split_sizes"] is split_sizes
    assert record["ctx"] == {
        "data_bytes": 32,
        "group_size": 2,
        "op": "reduce-scatter",
        "dim": "first",
        "split_sizes": [3, 1],
    }
    assert record["slots"] == ("group",)
    assert record["values"] == {"group": [11]}


def test_tp_reduce_scatter_last_preserves_composite_nesting_and_one_collective(monkeypatch) -> None:
    order: list[str] = []
    sink = _RecordingSink(order=order)
    install_trace_sink(sink)
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")
    _install_peer_metadata(monkeypatch, ranks=(17, 19), current_rank=17, order=order)
    group = _FakeGroup(2)
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2).transpose(1, 2)
    assert list(tensor.shape) == [2, 2, 6]
    assert not tensor.is_contiguous()
    expected = (torch.arange(12, dtype=torch.float32) + 100).reshape(2, 2, 3)
    collective_calls: list[tuple[torch.Tensor, torch.Tensor, Any]] = []
    original_first_dim = mappings._reduce_scatter_along_first_dim
    real_split = torch.split
    real_cat = torch.cat
    real_empty = torch.empty

    def _split(*args, **kwargs):
        if kwargs.get("dim") == 1:
            order.append("preprocess-split")
        return real_split(*args, **kwargs)

    def _cat(*args, **kwargs):
        order.append("preprocess-cat")
        return real_cat(*args, **kwargs)

    def _empty(*args, **kwargs):
        order.append("inner-allocate")
        return real_empty(*args, **kwargs)

    def _reduce_scatter(output, value, *, group):
        order.append("collective")
        collective_calls.append((output, value, group))
        output.copy_(expected.reshape(4, 3))

    class _RecordedReshape:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def reshape(self, *shape):
            order.append("final-reshape")
            return self.value.reshape(*shape)

    def _first_dim_with_recorded_reshape(*args, **kwargs):
        return _RecordedReshape(original_first_dim(*args, **kwargs))

    monkeypatch.setattr(mappings.torch, "split", _split)
    monkeypatch.setattr(mappings.torch, "cat", _cat)
    monkeypatch.setattr(mappings.torch, "empty", _empty)
    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", _reduce_scatter)
    monkeypatch.setattr(
        mappings, "_reduce_scatter_along_first_dim", _first_dim_with_recorded_reshape
    )

    result = mappings._reduce_scatter_along_last_dim(tensor, group)

    assert torch.equal(result, expected)
    assert len(collective_calls) == 1
    _, collective_input, collective_group = collective_calls[0]
    assert list(collective_input.shape) == [8, 3]
    assert collective_input.is_contiguous()
    assert collective_group is group
    assert order == [
        "preprocess-split",
        "preprocess-cat",
        "B:tp-reduce-scatter-last",
        "inner-allocate",
        "B:tp-reduce-scatter",
        "collective",
        "metadata",
        "E:tp-reduce-scatter",
        "final-reshape",
        "metadata",
        "E:tp-reduce-scatter-last",
    ]
    assert sink.queries == ["tp-reduce-scatter-last", "tp-reduce-scatter"]
    assert [record["name"] for record in sink.records] == [
        "tp-reduce-scatter-last",
        "tp-reduce-scatter",
    ]
    outer, inner = sink.records
    assert outer["ctx"] == {
        "data_bytes": 96,
        "group_size": 2,
        "op": "reduce-scatter",
        "dim": "last",
    }
    assert inner["ctx"] == {
        "data_bytes": 96,
        "group_size": 2,
        "op": "reduce-scatter",
        "dim": "first",
    }
    assert outer["values"] == inner["values"] == {"group": [19]}


@pytest.mark.parametrize(
    ("enabled_events", "expected_event"),
    [
        ({"tp-reduce-scatter-last"}, "tp-reduce-scatter-last"),
        ({"tp-reduce-scatter"}, "tp-reduce-scatter"),
    ],
)
def test_tp_reduce_scatter_composite_respects_independent_event_gates(
    monkeypatch, enabled_events, expected_event
) -> None:
    sink = _RecordingSink(enabled=enabled_events)
    install_trace_sink(sink)
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")
    _install_peer_metadata(monkeypatch, ranks=(29, 31), current_rank=29)
    group = _FakeGroup(2)
    tensor = torch.ones((2, 4), dtype=torch.float32)
    collective_calls = []

    def _reduce_scatter(output, value, *, group):
        collective_calls.append((output, value, group))

    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", _reduce_scatter)

    result = mappings._reduce_scatter_along_last_dim(tensor, group)

    assert list(result.shape) == [2, 2]
    assert len(collective_calls) == 1
    assert [record["name"] for record in sink.records] == [expected_event]
    assert sink.queries == ["tp-reduce-scatter-last", "tp-reduce-scatter"]


@pytest.mark.parametrize("gate_mode", ["null", "disabled", "suppressed"])
@pytest.mark.parametrize("variant", ["equal", "uneven", "last"])
def test_tp_reduce_scatter_closed_gate_skips_context_and_metadata(
    monkeypatch, gate_mode, variant
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    if gate_mode != "null":
        install_trace_sink(sink, suppress_scope=lambda: gate_mode == "suppressed")
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")
    group = _FakeGroup(2, rank=1)
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4).transpose(0, 1)
    collective_calls: list[str] = []

    monkeypatch.setattr(
        mappings,
        "_tp_reduce_scatter_context",
        lambda *args, **kwargs: pytest.fail("closed gate built TP reduce-scatter context"),
        raising=False,
    )
    monkeypatch.setattr(
        mappings,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("closed gate queried TP reduce-scatter peers"),
    )

    def _equal_reduce_scatter(output, value, *, group):
        collective_calls.append("equal")

    def _uneven_reduce_scatter(output, values, *, group):
        collective_calls.append("uneven")

    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", _equal_reduce_scatter)
    monkeypatch.setattr(mappings.torch.distributed, "reduce_scatter", _uneven_reduce_scatter)

    if variant == "equal":
        result = mappings._reduce_scatter_along_first_dim(tensor, group)
        assert list(result.shape) == [2, 2]
    elif variant == "uneven":
        result = mappings._reduce_scatter_along_first_dim(tensor, group, input_split_sizes=[3, 1])
        assert list(result.shape) == [1, 2]
    else:
        result = mappings._reduce_scatter_along_last_dim(tensor, group)
        assert list(result.shape) == [4, 1]

    assert collective_calls == ["uneven" if variant == "uneven" else "equal"]
    assert sink.records == []


def test_tp_reduce_scatter_single_rank_preserves_distinct_source_paths(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _install_peer_metadata(monkeypatch, ranks=(23,), current_rank=23)
    group = _FakeGroup(1)
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    monkeypatch.setattr(
        mappings,
        "dist_reduce_scatter_func",
        lambda *args, **kwargs: pytest.fail("single-rank path called equal collective"),
    )
    monkeypatch.setattr(
        mappings.torch.distributed,
        "reduce_scatter",
        lambda *args, **kwargs: pytest.fail("single-rank path called uneven collective"),
    )

    first_result = mappings._reduce_scatter_along_first_dim(
        tensor, group, input_split_sizes=[2], use_global_buffer=True
    )

    assert first_result is tensor
    assert sink.queries == []
    assert sink.records == []

    preprocess_calls = []
    real_split = torch.split
    real_cat = torch.cat

    def _split(*args, **kwargs):
        preprocess_calls.append("split")
        return real_split(*args, **kwargs)

    def _cat(*args, **kwargs):
        preprocess_calls.append("cat")
        return real_cat(*args, **kwargs)

    monkeypatch.setattr(mappings.torch, "split", _split)
    monkeypatch.setattr(mappings.torch, "cat", _cat)

    last_result = mappings._reduce_scatter_along_last_dim(tensor, group)

    assert torch.equal(last_result, tensor)
    assert preprocess_calls == ["split", "cat"]
    assert sink.queries == ["tp-reduce-scatter-last"]
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["ctx"] == {
        "data_bytes": 24,
        "group_size": 1,
        "op": "reduce-scatter",
        "dim": "last",
    }
    assert record["values"] == {"group": []}


@pytest.mark.parametrize("metadata_mode", ["missing", "error"])
@pytest.mark.parametrize("variant", ["first", "last"])
def test_tp_reduce_scatter_unavailable_peer_metadata_records_none(
    monkeypatch, metadata_mode, variant
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")
    group = _FakeGroup(2)
    tensor = torch.ones((4, 2), dtype=torch.float32)
    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", lambda output, value, *, group: None)

    if metadata_mode == "missing":
        monkeypatch.delattr(mappings.torch.distributed, "get_process_group_ranks", raising=False)
        monkeypatch.setattr(
            mappings.torch.distributed,
            "get_rank",
            lambda: pytest.fail("missing membership API queried global rank"),
        )
    else:
        monkeypatch.setattr(mappings.torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(
            mappings.torch.distributed,
            "get_process_group_ranks",
            lambda group: (_ for _ in ()).throw(RuntimeError("membership unavailable")),
        )

    if variant == "first":
        result = mappings._reduce_scatter_along_first_dim(tensor, group)
        assert list(result.shape) == [2, 2]
    else:
        result = mappings._reduce_scatter_along_last_dim(tensor, group)
        assert list(result.shape) == [4, 1]

    assert all(record["values"]["group"] is None for record in sink.records)


@pytest.mark.parametrize("variant", ["equal", "uneven", "last"])
def test_tp_reduce_scatter_collective_error_closes_scopes_and_preserves_identity(
    monkeypatch, variant
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")
    group = _FakeGroup(2, rank=1)
    tensor = torch.ones((4, 2), dtype=torch.float32)
    collective_error = RuntimeError(f"{variant} reduce-scatter failed")

    def _fail_collective(*args, **kwargs):
        raise collective_error

    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", _fail_collective)
    monkeypatch.setattr(mappings.torch.distributed, "reduce_scatter", _fail_collective)
    monkeypatch.setattr(
        mappings,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("failed collective queried peer metadata"),
    )

    with pytest.raises(RuntimeError, match="reduce-scatter failed") as raised:
        if variant == "equal":
            mappings._reduce_scatter_along_first_dim(tensor, group)
        elif variant == "uneven":
            mappings._reduce_scatter_along_first_dim(tensor, group, input_split_sizes=[3, 1])
        else:
            mappings._reduce_scatter_along_last_dim(tensor, group)

    assert raised.value is collective_error
    expected_records = 2 if variant == "last" else 1
    assert len(sink.records) == expected_records
    assert all(record["exit_exception"] is RuntimeError for record in sink.records)
    assert all(record["exit_value"] is collective_error for record in sink.records)
    assert all("group" not in record.get("values", {}) for record in sink.records)


@pytest.mark.parametrize("variant", ["equal-shape", "uneven-split"])
def test_tp_reduce_scatter_preprocess_error_happens_before_probe(monkeypatch, variant) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _FakeGroup(2, rank=0)
    tensor = torch.ones((3, 2), dtype=torch.float32)

    if variant == "equal-shape":
        with pytest.raises(AssertionError, match="divisible"):
            mappings._reduce_scatter_along_first_dim(tensor, group)
    else:
        with pytest.raises(RuntimeError):
            mappings._reduce_scatter_along_first_dim(tensor, group, input_split_sizes=[2, 2])

    assert sink.queries == []
    assert sink.records == []


def test_tp_reduce_scatter_uses_two_literal_core_producers_without_async_extension() -> None:
    source_path = Path(mappings.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    producer_names = []
    imported_modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open_trace_scope" and len(node.args) >= 2:
                event_node = node.args[1]
                if isinstance(event_node, ast.Constant) and isinstance(event_node.value, str):
                    producer_names.append(event_node.value)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    reduce_scatter_source = "\n".join(
        (
            inspect.getsource(mappings._reduce_scatter_along_first_dim),
            inspect.getsource(mappings._reduce_scatter_along_last_dim),
        )
    )
    assert producer_names.count("tp-reduce-scatter") == 1
    assert producer_names.count("tp-reduce-scatter-last") == 1
    assert "async_op" not in reduce_scatter_source
    assert ".wait(" not in reduce_scatter_source
    assert "synchronize" not in reduce_scatter_source
    assert "megatron.training.global_vars" not in imported_modules
    assert not any(
        module == "megatron.megalens" or module.startswith("megatron.megalens.")
        for module in imported_modules
    )
