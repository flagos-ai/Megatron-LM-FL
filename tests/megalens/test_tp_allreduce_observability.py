from __future__ import annotations

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
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.record["entered"] = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.record["exit_exception"] = exc_type
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record.setdefault("values", {})[key] = value
        return True


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

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
        }
        self.records.append(record)
        return _RecordingScope(record)


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def test_tp_allreduce_emits_collective_contract_and_preserves_result(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _FakeGroup(2)
    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    collective_calls: list[tuple[torch.Tensor, Any]] = []

    monkeypatch.setattr(
        mappings.torch.distributed,
        "all_reduce",
        lambda value, *, group: collective_calls.append((value, group)),
    )
    monkeypatch.setattr(mappings.torch.distributed, "get_process_group_ranks", lambda group: [2, 5])
    monkeypatch.setattr(mappings.torch.distributed, "get_rank", lambda: 2)

    result = mappings._reduce(tensor, group)

    assert result is tensor
    assert collective_calls == [(tensor, group)]
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["name"] == "tp-allreduce"
    assert record["ctx"]["data_bytes"] == 12
    assert record["slots"] == ("group",)
    assert {**record["ctx"], **record["values"]} == {
        "data_bytes": 12,
        "group_size": 2,
        "group": [5],
        "op": "all_reduce",
        "timing_phase": "collective_call",
        "payload_role": "inplace_input_output",
    }
    assert record["entered"]
    assert record["exit_exception"] is None


def test_tp_allreduce_null_sink_preserves_collective_and_return_semantics(monkeypatch) -> None:
    group = _FakeGroup(2)
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
    assert not tensor.is_contiguous()
    collective_calls: list[tuple[torch.Tensor, Any]] = []
    metadata_calls: list[str] = []

    monkeypatch.setattr(
        mappings,
        "_tp_allreduce_context",
        lambda input_, group_size: pytest.fail("trace-off path built TP all-reduce context"),
    )

    monkeypatch.setattr(
        mappings.torch.distributed,
        "all_reduce",
        lambda value, *, group: collective_calls.append((value, group)),
    )

    def _metadata_unavailable(group):
        metadata_calls.append("group-ranks")
        raise RuntimeError("process-group membership unavailable")

    def _global_rank() -> int:
        metadata_calls.append("global-rank")
        return 0

    monkeypatch.setattr(
        mappings.torch.distributed, "get_process_group_ranks", _metadata_unavailable
    )
    monkeypatch.setattr(mappings.torch.distributed, "get_rank", _global_rank)

    result = mappings._reduce(tensor, group)

    assert result is tensor
    assert len(collective_calls) == 1
    collective_tensor, collective_group = collective_calls[0]
    assert collective_group is group
    assert collective_tensor.is_contiguous()
    assert torch.equal(collective_tensor, tensor)
    assert metadata_calls == []


def test_tp_allreduce_single_rank_emits_source_noop_scope(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _FakeGroup(1)
    tensor = torch.ones(4, dtype=torch.float32)

    monkeypatch.setattr(
        mappings.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("single-rank fast path called all_reduce"),
    )
    monkeypatch.setattr(
        mappings,
        "get_process_group_peer_ranks",
        lambda group: pytest.fail("single-rank fast path queried peer metadata"),
    )

    result = mappings._reduce(tensor, group)

    assert result is tensor
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["name"] == "tp-allreduce"
    assert record["ctx"] == {
        "data_bytes": 16,
        "group_size": 1,
        "op": "all_reduce",
        "timing_phase": "collective_call",
        "payload_role": "inplace_input_output",
    }
    assert record["slots"] == ("group",)
    assert "values" not in record
    assert record["entered"]
    assert record["exit_exception"] is None


def test_tp_allreduce_tolerates_unavailable_peer_metadata(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _FakeGroup(2)
    tensor = torch.ones(4, dtype=torch.float32)
    collective_calls = []

    monkeypatch.setattr(
        mappings.torch.distributed,
        "all_reduce",
        lambda value, *, group: collective_calls.append((value, group)),
    )
    monkeypatch.setattr(
        mappings.torch.distributed,
        "get_process_group_ranks",
        lambda group: (_ for _ in ()).throw(RuntimeError("membership unavailable")),
    )

    result = mappings._reduce(tensor, group)

    assert result is tensor
    assert collective_calls == [(tensor, group)]
    assert sink.records[0]["values"]["group"] is None


def test_tp_allreduce_collective_error_closes_scope_and_propagates(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    group = _FakeGroup(2)
    tensor = torch.ones(4, dtype=torch.float32)
    collective_error = RuntimeError("all-reduce failed")

    def _fail_collective(value, *, group):
        raise collective_error

    monkeypatch.setattr(mappings.torch.distributed, "all_reduce", _fail_collective)
    monkeypatch.setattr(mappings.torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(mappings.torch.distributed, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="all-reduce failed") as raised:
        mappings._reduce(tensor, group)

    assert raised.value is collective_error
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["entered"]
    assert record["exit_exception"] is RuntimeError
