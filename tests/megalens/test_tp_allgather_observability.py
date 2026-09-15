from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.tensor_parallel import mappings


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
    def __init__(self, *, enabled: bool = True, order: list[str] | None = None) -> None:
        self.enabled = enabled
        self.order = order if order is not None else []
        self.records: list[dict[str, Any]] = []

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
        return _RecordingScope(record, self.order)


class _FakeGlobalMemoryBuffer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], torch.dtype, str]] = []

    def get_tensor(self, shape, dtype, name):
        normalized_shape = list(shape)
        self.calls.append((normalized_shape, dtype, name))
        return torch.empty(normalized_shape, dtype=dtype)


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def _use_cpu_allocations(monkeypatch) -> None:
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")


def _install_peer_metadata(monkeypatch, *, ranks=(2, 5), current_rank=2, order=None) -> None:
    def _get_group_ranks(group):
        if order is not None:
            order.append("metadata")
        return list(ranks)

    monkeypatch.setattr(mappings.torch.distributed, "get_process_group_ranks", _get_group_ranks)
    monkeypatch.setattr(mappings.torch.distributed, "get_rank", lambda: current_rank)


def test_tp_allgather_last_emits_source_contract_and_keeps_postprocess_outside_scope(
    monkeypatch,
) -> None:
    order: list[str] = []
    sink = _RecordingSink(order=order)
    install_trace_sink(sink)
    _use_cpu_allocations(monkeypatch)
    _install_peer_metadata(monkeypatch, order=order)
    group = _FakeGroup(2)
    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    expected = torch.tensor([[0, 1, 2, 10, 11, 12], [3, 4, 5, 13, 14, 15]], dtype=torch.float16)
    collective_calls: list[tuple[torch.Tensor, torch.Tensor, Any]] = []
    real_empty = torch.empty
    real_cat = torch.cat

    def _empty(*args, **kwargs):
        order.append("allocate")
        return real_empty(*args, **kwargs)

    def _all_gather(output, value, *, group):
        order.append("collective")
        collective_calls.append((output, value, group))
        output[: tensor.shape[0]].copy_(tensor)
        output[tensor.shape[0] :].copy_(tensor + 10)

    def _cat(tensors, dim=0):
        order.append("postprocess-cat")
        return real_cat(tensors, dim=dim)

    monkeypatch.setattr(mappings.torch, "empty", _empty)
    monkeypatch.setattr(mappings, "dist_all_gather_func", _all_gather)
    monkeypatch.setattr(mappings.torch, "cat", _cat)

    result = mappings._gather_along_last_dim(tensor, group)

    assert torch.equal(result, expected)
    assert result.is_contiguous()
    assert len(collective_calls) == 1
    _, collective_input, collective_group = collective_calls[0]
    assert collective_input is tensor
    assert collective_group is group
    assert order == [
        "allocate",
        "B:tp-all-gather-last",
        "collective",
        "metadata",
        "E:tp-all-gather-last",
        "postprocess-cat",
    ]
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["slots"] == ("group",)
    assert record["ctx"] == {"data_bytes": 12, "group_size": 2, "op": "all-gather", "dim": "last"}
    assert record["values"] == {"group": [5]}


@pytest.mark.parametrize("use_global_buffer", [False, True])
def test_tp_allgather_first_equal_preserves_allocation_collective_and_result(
    monkeypatch, use_global_buffer
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _use_cpu_allocations(monkeypatch)
    _install_peer_metadata(monkeypatch, ranks=(3, 7), current_rank=3)
    group = _FakeGroup(2)
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    expected = torch.tensor([[0, 1], [2, 3], [20, 21], [22, 23]], dtype=torch.float32)
    collective_calls: list[tuple[torch.Tensor, torch.Tensor, Any]] = []
    global_buffer = _FakeGlobalMemoryBuffer()
    monkeypatch.setattr(mappings, "get_global_memory_buffer", lambda: global_buffer)

    def _all_gather(output, value, *, group):
        collective_calls.append((output, value, group))
        output[: tensor.shape[0]].copy_(tensor)
        output[tensor.shape[0] :].copy_(tensor + 20)

    monkeypatch.setattr(mappings, "dist_all_gather_func", _all_gather)

    result = mappings._gather_along_first_dim(tensor, group, use_global_buffer=use_global_buffer)

    assert torch.equal(result, expected)
    assert len(collective_calls) == 1
    _, collective_input, collective_group = collective_calls[0]
    assert collective_input is tensor
    assert collective_group is group
    assert global_buffer.calls == ([([4, 2], torch.float32, "mpu")] if use_global_buffer else [])
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["slots"] == ("group",)
    assert record["ctx"] == {"data_bytes": 16, "group_size": 2, "op": "all-gather", "dim": "first"}
    assert record["values"] == {"group": [7]}


@pytest.mark.parametrize("use_global_buffer", [False, True])
def test_tp_allgather_first_uneven_preserves_raw_input_and_split_contract(
    monkeypatch, use_global_buffer
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _use_cpu_allocations(monkeypatch)
    _install_peer_metadata(monkeypatch, ranks=(11, 13), current_rank=11)
    group = _FakeGroup(2)
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4).transpose(0, 1)
    assert not tensor.is_contiguous()
    split_sizes = [4, 3]
    expected_remote = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 100
    expected = torch.cat((tensor, expected_remote), dim=0)
    collective_calls: list[tuple[list[torch.Tensor], torch.Tensor, Any]] = []
    global_buffer = _FakeGlobalMemoryBuffer()
    monkeypatch.setattr(mappings, "get_global_memory_buffer", lambda: global_buffer)

    def _all_gather(output_tensors, value, *, group):
        collective_calls.append((output_tensors, value, group))
        output_tensors[0].copy_(tensor)
        output_tensors[1].copy_(expected_remote)

    monkeypatch.setattr(mappings.torch.distributed, "all_gather", _all_gather)

    result = mappings._gather_along_first_dim(
        tensor, group, output_split_sizes=split_sizes, use_global_buffer=use_global_buffer
    )

    assert torch.equal(result, expected)
    assert len(collective_calls) == 1
    output_tensors, collective_input, collective_group = collective_calls[0]
    assert [list(value.shape) for value in output_tensors] == [[4, 2], [3, 2]]
    assert collective_input is tensor
    assert not collective_input.is_contiguous()
    assert collective_group is group
    assert global_buffer.calls == ([([7, 2], torch.float32, "mpu")] if use_global_buffer else [])
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["ctx"]["split_sizes"] is split_sizes
    assert record["ctx"] == {
        "data_bytes": 32,
        "group_size": 2,
        "op": "all-gather",
        "dim": "first",
        "split_sizes": [4, 3],
    }
    assert record["values"] == {"group": [13]}


@pytest.mark.parametrize("gate_mode", ["null", "disabled", "suppressed"])
@pytest.mark.parametrize("variant", ["last", "first", "uneven"])
def test_tp_allgather_closed_gate_skips_context_and_metadata(
    monkeypatch, gate_mode, variant
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    if gate_mode != "null":
        install_trace_sink(sink, suppress_scope=lambda: gate_mode == "suppressed")
    _use_cpu_allocations(monkeypatch)
    group = _FakeGroup(2)
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not tensor.is_contiguous()
    collective_calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        mappings,
        "_tp_all_gather_context",
        lambda *args, **kwargs: pytest.fail("closed gate built TP all-gather context"),
        raising=False,
    )
    monkeypatch.setattr(
        mappings,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("closed gate queried TP all-gather peers"),
    )

    def _equal_all_gather(output, value, *, group):
        collective_calls.append(("equal", value))

    def _uneven_all_gather(output_tensors, value, *, group):
        collective_calls.append(("uneven", value))

    monkeypatch.setattr(mappings, "dist_all_gather_func", _equal_all_gather)
    monkeypatch.setattr(mappings.torch.distributed, "all_gather", _uneven_all_gather)

    if variant == "last":
        result = mappings._gather_along_last_dim(tensor, group)
        assert list(result.shape) == [4, 6]
    elif variant == "first":
        result = mappings._gather_along_first_dim(tensor, group)
        assert list(result.shape) == [8, 3]
    else:
        result = mappings._gather_along_first_dim(tensor, group, output_split_sizes=[4, 2])
        assert list(result.shape) == [6, 3]

    assert len(collective_calls) == 1
    collective_kind, collective_input = collective_calls[0]
    if variant == "uneven":
        assert collective_kind == "uneven"
        assert collective_input is tensor
    else:
        assert collective_kind == "equal"
        assert collective_input is not tensor
        assert collective_input.is_contiguous()
        assert torch.equal(collective_input, tensor)
    assert sink.records == []


@pytest.mark.parametrize("variant", ["last", "first"])
def test_tp_allgather_single_rank_returns_input_without_work(monkeypatch, variant) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _FakeGroup(1)
    tensor = torch.ones(3, dtype=torch.float32)

    monkeypatch.setattr(
        mappings.torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("single-rank path allocated output"),
    )
    monkeypatch.setattr(
        mappings,
        "get_global_memory_buffer",
        lambda: pytest.fail("single-rank path queried global buffer"),
    )
    monkeypatch.setattr(
        mappings,
        "dist_all_gather_func",
        lambda *args, **kwargs: pytest.fail("single-rank path called equal all-gather"),
    )
    monkeypatch.setattr(
        mappings.torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("single-rank path called uneven all-gather"),
    )

    if variant == "last":
        result = mappings._gather_along_last_dim(tensor, group)
    else:
        result = mappings._gather_along_first_dim(
            tensor, group, output_split_sizes=[1], use_global_buffer=True
        )

    assert result is tensor
    assert sink.records == []


@pytest.mark.parametrize("metadata_mode", ["missing", "error"])
def test_tp_allgather_unavailable_peer_metadata_records_none(monkeypatch, metadata_mode) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _use_cpu_allocations(monkeypatch)
    group = _FakeGroup(2)
    tensor = torch.ones((2, 3), dtype=torch.float32)
    monkeypatch.setattr(mappings, "dist_all_gather_func", lambda output, value, *, group: None)

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

    result = mappings._gather_along_first_dim(tensor, group)

    assert list(result.shape) == [4, 3]
    assert sink.records[0]["values"]["group"] is None


@pytest.mark.parametrize("variant", ["last", "first", "uneven"])
def test_tp_allgather_collective_error_closes_scope_and_preserves_identity(
    monkeypatch, variant
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    _use_cpu_allocations(monkeypatch)
    group = _FakeGroup(2)
    tensor = torch.ones((2, 3), dtype=torch.float32)
    collective_error = RuntimeError(f"{variant} all-gather failed")

    def _fail_collective(*args, **kwargs):
        raise collective_error

    monkeypatch.setattr(mappings, "dist_all_gather_func", _fail_collective)
    monkeypatch.setattr(mappings.torch.distributed, "all_gather", _fail_collective)
    monkeypatch.setattr(
        mappings,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("failed collective queried peer metadata"),
    )
    if variant == "last":
        monkeypatch.setattr(
            mappings.torch,
            "cat",
            lambda *args, **kwargs: pytest.fail("failed last gather ran postprocessing"),
        )

    with pytest.raises(RuntimeError, match="all-gather failed") as raised:
        if variant == "last":
            mappings._gather_along_last_dim(tensor, group)
        elif variant == "first":
            mappings._gather_along_first_dim(tensor, group)
        else:
            mappings._gather_along_first_dim(tensor, group, output_split_sizes=[2, 3])

    assert raised.value is collective_error
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["exit_exception"] is RuntimeError
    assert record["exit_value"] is collective_error
    assert "group" not in record.get("values", {})


def test_tp_allgather_uses_two_literal_core_producers_without_training_imports() -> None:
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

    assert producer_names.count("tp-all-gather-first") == 1
    assert producer_names.count("tp-all-gather-last") == 1
    assert "megatron.training.global_vars" not in imported_modules
    assert not any(
        module == "megatron.megalens" or module.startswith("megatron.megalens.")
        for module in imported_modules
    )
