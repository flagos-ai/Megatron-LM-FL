from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.distributed import param_and_grad_buffer
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
from megatron.core.observability import install_trace_sink, reset_trace_sink


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


class _FakeCoalescingManager:
    def __init__(self, handle: object) -> None:
        self.handle = handle

    def __enter__(self) -> object:
        return self.handle

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def _make_bucket_group(*, overlap_grad_reduce: bool):
    process_group = object()
    bucket_group = _ParamAndGradBucketGroup.__new__(_ParamAndGradBucketGroup)
    bucket_group.is_first_batch = False
    bucket_group.grad_reduce_handle = None
    bucket_group.buckets = [
        SimpleNamespace(
            grad_data=torch.ones(4, dtype=torch.float32),
            gradient_scaling_factor=1.0,
            params_with_extra_main_grads=[],
        ),
        SimpleNamespace(
            grad_data=torch.ones(3, dtype=torch.float16),
            gradient_scaling_factor=1.0,
            params_with_extra_main_grads=[],
        ),
    ]
    bucket_group.ddp_config = SimpleNamespace(
        check_for_nan_in_grad=False,
        check_for_large_grads=False,
        average_in_collective=False,
        overlap_grad_reduce=overlap_grad_reduce,
        num_distributed_optimizer_instances=1,
        use_distributed_optimizer=False,
        reduce_scatter_with_fp32_accumulation=False,
    )
    bucket_group.data_parallel_group = process_group
    bucket_group.collective_group_size = 3
    bucket_group.cached_grad_buffer_shard_list = [None, None]
    return bucket_group, process_group


def test_dp_allreduce_probe_records_async_bucket_dispatch_and_preserves_handle(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, process_group = _make_bucket_group(overlap_grad_reduce=True)
    handle = object()
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(handle),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: collective_calls.append(
            (tensor, op, group, async_op)
        ),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed, "get_process_group_ranks", lambda group: [2, 5, 7]
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 5)

    result = bucket_group.start_grad_sync()

    assert result is None
    assert bucket_group.grad_reduce_handle is handle
    assert [(call[2], call[3]) for call in collective_calls] == [
        (process_group, True),
        (process_group, True),
    ]
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["name"] == "dp-allreduce"
    assert record["slots"] == ("data_bytes", "group")
    assert {**record["ctx"], **record["values"]} == {
        "api_async_op": True,
        "async_op": True,
        "completion_included": False,
        "data_bytes": 22,
        "group": [2, 7],
        "group_size": 3,
        "group_role": "data_parallel",
        "n_buckets": 2,
        "op": "all_reduce",
        "overlap_enabled": True,
        "payload_role": "gradient_bucket",
        "stage": "main_bucket_allreduce",
        "timing_phase": "async_dispatch",
    }
    assert record["entered"]
    assert record["exit_exception"] is None


def test_dp_allreduce_null_sink_skips_metadata_queries_and_preserves_sync_call(monkeypatch):
    bucket_group, process_group = _make_bucket_group(overlap_grad_reduce=False)
    payloads = [object(), object()]
    for bucket, payload in zip(bucket_group.buckets, payloads):
        bucket.grad_data = payload

    collective_calls = []
    metadata_calls = []
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_dp_allreduce_context",
        lambda **kwargs: pytest.fail("trace-off path built DP all-reduce context"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: collective_calls.append((tensor, group, async_op)),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: metadata_calls.append("group-ranks"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_rank",
        lambda: metadata_calls.append("global-rank") or 0,
    )

    result = bucket_group.start_grad_sync()

    assert result is None
    assert bucket_group.grad_reduce_handle is None
    assert collective_calls == [
        (payloads[0], process_group, False),
        (payloads[1], process_group, False),
    ]
    assert metadata_calls == []


def test_dp_allreduce_collective_error_closes_scope_and_preserves_exception(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, _ = _make_bucket_group(overlap_grad_reduce=False)
    collective_error = RuntimeError("DP all-reduce failed")

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: (_ for _ in ()).throw(collective_error),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2]
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="DP all-reduce failed") as exc_info:
        bucket_group.start_grad_sync()

    assert exc_info.value is collective_error
    assert bucket_group.grad_reduce_handle is None
    assert len(sink.records) == 1
    assert sink.records[0]["entered"]
    assert sink.records[0]["exit_exception"] is RuntimeError


def test_dp_force_allreduce_uses_intra_optimizer_group_and_emits_probe(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, process_group = _make_bucket_group(overlap_grad_reduce=False)
    bucket_group.ddp_config.use_distributed_optimizer = True
    bucket_group.intra_distributed_optimizer_instance_group = process_group
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: collective_calls.append((group, async_op)),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2]
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 1)

    result = bucket_group.start_grad_sync(force_all_reduce=True)

    assert result is None
    assert collective_calls == [(process_group, False), (process_group, False)]
    assert len(sink.records) == 1
    assert sink.records[0]["name"] == "dp-allreduce"
    assert sink.records[0]["ctx"]["group_role"] == "intra_optimizer_instance"
    assert sink.records[0]["values"]["group"] == [0, 2]


def test_dp_multi_instance_force_allreduce_is_fail_closed_until_both_stages_are_modeled(
    monkeypatch,
):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, intra_group = _make_bucket_group(overlap_grad_reduce=False)
    inter_group = object()
    bucket_group.ddp_config.use_distributed_optimizer = True
    bucket_group.ddp_config.num_distributed_optimizer_instances = 2
    bucket_group.collective_group_size = 1
    bucket_group.intra_distributed_optimizer_instance_group = intra_group
    bucket_group.intra_distributed_optimizer_instance_size = 1
    bucket_group.intra_distributed_optimizer_instance_rank = 0
    bucket_group.inter_distributed_optimizer_instance_group = inter_group
    collective_groups = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: collective_groups.append(group),
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 1)

    result = bucket_group.start_grad_sync(force_all_reduce=True)

    assert result is None
    assert collective_groups == [intra_group, intra_group, inter_group, inter_group]
    assert sink.records == []
