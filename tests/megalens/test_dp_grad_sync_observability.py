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


def _configure_distributed_optimizer(bucket_group, process_group) -> None:
    bucket_group.ddp_config.use_distributed_optimizer = True
    bucket_group.intra_distributed_optimizer_instance_group = process_group
    bucket_group.intra_distributed_optimizer_instance_size = 3
    bucket_group.intra_distributed_optimizer_instance_rank = 1
    bucket_group.buckets[0].grad_data = torch.ones(6, dtype=torch.float32)
    bucket_group.buckets[1].grad_data = torch.ones(3, dtype=torch.float16)


def _make_param_bucket_group(*, overlap_param_gather: bool):
    process_group = object()
    bucket_group = _ParamAndGradBucketGroup.__new__(_ParamAndGradBucketGroup)
    bucket_group.buckets = [
        SimpleNamespace(param_data=torch.ones(6, dtype=torch.float32)),
        SimpleNamespace(param_data=torch.ones(3, dtype=torch.float16)),
    ]
    bucket_group.ddp_config = SimpleNamespace(
        overlap_param_gather=overlap_param_gather, use_distributed_optimizer=True
    )
    bucket_group.intra_distributed_optimizer_instance_group = process_group
    bucket_group.intra_distributed_optimizer_instance_size = 3
    bucket_group.intra_distributed_optimizer_instance_rank = 1
    bucket_group.cached_param_buffer_shard_list = [None, None]
    bucket_group.param_gather_handle = None
    bucket_group.param_gather_dispatched = False
    return bucket_group, process_group


def test_dp_param_allgather_probe_records_distopt_dispatch_and_preserves_handle(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, process_group = _make_param_bucket_group(overlap_param_gather=True)
    handle = object()
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(handle),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "dist_all_gather_func",
        lambda output, input_, *, group, async_op: collective_calls.append(
            (output.numel(), input_.numel(), group, async_op)
        ),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed, "get_process_group_ranks", lambda group: [2, 5, 7]
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 5)

    result = bucket_group.start_param_sync()

    assert result is None
    assert bucket_group.param_gather_handle is handle
    assert bucket_group.param_gather_dispatched is True
    assert collective_calls == [(6, 2, process_group, True), (3, 1, process_group, True)]
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["name"] == "dp-param-all-gather"
    assert record["slots"] == ("group",)
    assert {**record["ctx"], **record["values"]} == {
        "api_async_op": True,
        "async_op": True,
        "completion_included": False,
        "data_bytes": 30,
        "group": [2, 7],
        "group_size": 3,
        "group_role": "intra_optimizer_instance",
        "n_buckets": 2,
        "op": "all_gather",
        "optimizer_kind": "distributed",
        "overlap_enabled": True,
        "payload_role": "parameter_bucket",
        "stage": "distributed_optimizer_param_allgather",
        "timing_phase": "async_dispatch",
    }


def test_dp_param_allgather_null_sink_skips_metadata_and_preserves_sync_call(monkeypatch):
    bucket_group, process_group = _make_param_bucket_group(overlap_param_gather=False)
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_dp_param_allgather_context",
        lambda **kwargs: pytest.fail("trace-off path built parameter all-gather context"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_dp_param_allgather_data_bytes",
        lambda *args, **kwargs: pytest.fail("trace-off path counted parameter bytes"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "dist_all_gather_func",
        lambda output, input_, *, group, async_op: collective_calls.append(
            (output.numel(), input_.numel(), group, async_op)
        ),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: pytest.fail("trace-off path queried parameter group ranks"),
    )

    result = bucket_group.start_param_sync()

    assert result is None
    assert bucket_group.param_gather_handle is None
    assert collective_calls == [(6, 2, process_group, False), (3, 1, process_group, False)]


def test_dp_param_allgather_probe_supports_layerwise_optimizer(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    process_group = object()
    local_param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    remote_param = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    bucket = SimpleNamespace(
        _layerwise_src_buffer=None,
        grad_data=torch.empty(0, dtype=torch.float32),
        layerwise_gather_list=None,
        layerwise_param_flat_sizes=[2, 3],
        layerwise_params_list=[[local_param], [remote_param]],
        param_data=None,
        params_list=[local_param, remote_param],
    )
    bucket_group = _ParamAndGradBucketGroup.__new__(_ParamAndGradBucketGroup)
    bucket_group.buckets = [bucket]
    bucket_group.ddp_config = SimpleNamespace(
        overlap_param_gather=True, use_distributed_optimizer=False
    )
    bucket_group.intra_distributed_optimizer_instance_group = process_group
    bucket_group.intra_distributed_optimizer_instance_size = 2
    bucket_group.intra_distributed_optimizer_instance_rank = 0
    bucket_group.param_gather_handle = None
    bucket_group.param_gather_dispatched = False
    work = object()
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_gather",
        lambda output, input_, *, group, async_op: collective_calls.append(
            ([tensor.numel() for tensor in output], input_.numel(), group, async_op)
        )
        or work,
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed, "get_process_group_ranks", lambda group: [3, 8]
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 3)

    result = bucket_group.start_param_sync()

    assert result is None
    assert bucket_group.param_gather_handle.handles == [work]
    assert collective_calls == [([2, 3], 2, process_group, True)]
    record = sink.records[0]
    assert record["name"] == "dp-param-all-gather"
    assert {**record["ctx"], **record["values"]} == {
        "api_async_op": True,
        "async_op": True,
        "completion_included": False,
        "data_bytes": 20,
        "group": [8],
        "group_size": 2,
        "group_role": "intra_optimizer_instance",
        "n_buckets": 1,
        "op": "all_gather",
        "optimizer_kind": "layerwise",
        "overlap_enabled": True,
        "payload_role": "parameter_bucket",
        "stage": "layerwise_optimizer_param_allgather",
        "timing_phase": "async_dispatch",
    }


def test_dp_param_allgather_skips_empty_layerwise_dispatch_probe(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    process_group = object()
    empty_param = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))
    bucket = SimpleNamespace(
        _layerwise_src_buffer=None,
        grad_data=torch.empty(0, dtype=torch.float32),
        layerwise_gather_list=None,
        layerwise_param_flat_sizes=[0, 0],
        layerwise_params_list=[[], []],
        param_data=None,
        params_list=[empty_param],
    )
    bucket_group = _ParamAndGradBucketGroup.__new__(_ParamAndGradBucketGroup)
    bucket_group.buckets = [bucket]
    bucket_group.ddp_config = SimpleNamespace(
        overlap_param_gather=True, use_distributed_optimizer=False
    )
    bucket_group.intra_distributed_optimizer_instance_group = process_group
    bucket_group.intra_distributed_optimizer_instance_size = 2
    bucket_group.intra_distributed_optimizer_instance_rank = 0
    bucket_group.param_gather_handle = None
    bucket_group.param_gather_dispatched = False
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("empty layerwise bucket launched all-gather"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: pytest.fail("empty layerwise probe queried group ranks"),
    )

    result = bucket_group.start_param_sync()

    assert result is None
    assert bucket_group.param_gather_handle.handles == []
    assert sink.records == []


def test_dp_param_allgather_existing_handle_wait_does_not_emit_new_dispatch(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, _ = _make_param_bucket_group(overlap_param_gather=True)
    waits = []
    bucket_group.param_gather_handle = SimpleNamespace(wait=lambda: waits.append("wait"))
    monkeypatch.setattr(
        param_and_grad_buffer,
        "prepare_trace_scope",
        lambda name: pytest.fail("wait-only path evaluated a new dispatch probe"),
    )

    result = bucket_group.start_param_sync(force_sync=True)

    assert result is None
    assert waits == ["wait"]
    assert bucket_group.param_gather_handle is None
    assert sink.records == []


def test_dp_reduce_scatter_probe_records_distopt_dispatch_and_preserves_handle(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, process_group = _make_bucket_group(overlap_grad_reduce=True)
    _configure_distributed_optimizer(bucket_group, process_group)
    handle = object()
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(handle),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "dist_reduce_scatter_func",
        lambda output, input_, *, op, group, async_op: collective_calls.append(
            (output, input_, group, async_op)
        ),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed, "get_process_group_ranks", lambda group: [2, 5, 7]
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 5)

    result = bucket_group.start_grad_sync()

    assert result is None
    assert bucket_group.grad_reduce_handle is handle
    assert [(call[0].numel(), call[1].numel(), call[2], call[3]) for call in collective_calls] == [
        (2, 6, process_group, True),
        (1, 3, process_group, True),
    ]
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["name"] == "dp-reduce-scatter"
    assert record["slots"] == ("data_bytes", "group")
    assert {**record["ctx"], **record["values"]} == {
        "api_async_op": True,
        "async_op": True,
        "completion_included": False,
        "data_bytes": 30,
        "group": [2, 7],
        "group_size": 3,
        "group_role": "intra_optimizer_instance",
        "n_buckets": 2,
        "op": "reduce_scatter",
        "overlap_enabled": True,
        "payload_role": "gradient_bucket",
        "stage": "intra_instance_reduce_scatter",
        "timing_phase": "async_dispatch",
    }
    assert record["entered"]
    assert record["exit_exception"] is None


def test_dp_reduce_scatter_null_sink_skips_metadata_and_preserves_collectives(monkeypatch):
    bucket_group, process_group = _make_bucket_group(overlap_grad_reduce=False)
    _configure_distributed_optimizer(bucket_group, process_group)
    collective_calls = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_dp_reduce_scatter_context",
        lambda **kwargs: pytest.fail("trace-off path built DP reduce-scatter context"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "dist_reduce_scatter_func",
        lambda output, input_, *, op, group, async_op: collective_calls.append(
            (output.numel(), input_.numel(), group, async_op)
        ),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: pytest.fail("trace-off path queried DP group ranks"),
    )

    result = bucket_group.start_grad_sync()

    assert result is None
    assert bucket_group.grad_reduce_handle is None
    assert collective_calls == [(2, 6, process_group, False), (1, 3, process_group, False)]


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


def test_dp_multi_instance_distopt_records_reduce_scatter_then_inter_allreduce(monkeypatch):
    sink = _RecordingSink()
    install_trace_sink(sink)
    bucket_group, intra_group = _make_bucket_group(overlap_grad_reduce=False)
    _configure_distributed_optimizer(bucket_group, intra_group)
    inter_group = object()
    bucket_group.ddp_config.num_distributed_optimizer_instances = 2
    bucket_group.inter_distributed_optimizer_instance_group = inter_group
    reduce_scatter_groups = []
    allreduce_groups = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "dist_reduce_scatter_func",
        lambda output, input_, *, op, group, async_op: reduce_scatter_groups.append(group),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: allreduce_groups.append(group),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: [2, 5, 7] if group is intra_group else [5, 9],
    )
    monkeypatch.setattr(param_and_grad_buffer.torch.distributed, "get_rank", lambda: 5)

    result = bucket_group.start_grad_sync()

    assert result is None
    assert reduce_scatter_groups == [intra_group, intra_group]
    assert allreduce_groups == [inter_group, inter_group]
    assert [record["name"] for record in sink.records] == ["dp-reduce-scatter", "dp-allreduce"]
    reduce_scatter_record, allreduce_record = sink.records
    assert reduce_scatter_record["ctx"]["stage"] == "intra_instance_reduce_scatter"
    assert reduce_scatter_record["values"] == {"data_bytes": 30, "group": [2, 7]}
    assert allreduce_record["ctx"] == {
        "api_async_op": False,
        "async_op": False,
        "completion_included": False,
        "group_size": 2,
        "group_role": "inter_optimizer_instance",
        "n_buckets": 2,
        "op": "all_reduce",
        "overlap_enabled": False,
        "payload_role": "gradient_shard",
        "stage": "inter_instance_shard_allreduce",
        "timing_phase": "collective_call",
    }
    assert allreduce_record["values"] == {"data_bytes": 10, "group": [9]}


def test_dp_multi_instance_null_sink_skips_inter_metadata(monkeypatch):
    bucket_group, intra_group = _make_bucket_group(overlap_grad_reduce=False)
    _configure_distributed_optimizer(bucket_group, intra_group)
    inter_group = object()
    bucket_group.ddp_config.num_distributed_optimizer_instances = 2
    bucket_group.inter_distributed_optimizer_instance_group = inter_group
    collective_groups = []

    monkeypatch.setattr(
        param_and_grad_buffer,
        "_dp_inter_instance_allreduce_context",
        lambda **kwargs: pytest.fail("trace-off path built inter-instance all-reduce context"),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_coalescing_manager",
        lambda group, *, async_ops: _FakeCoalescingManager(object()),
    )
    monkeypatch.setattr(
        param_and_grad_buffer,
        "dist_reduce_scatter_func",
        lambda output, input_, *, op, group, async_op: collective_groups.append(group),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group, async_op: collective_groups.append(group),
    )
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: pytest.fail("trace-off path queried inter-instance group ranks"),
    )

    result = bucket_group.start_grad_sync()

    assert result is None
    assert collective_groups == [intra_group, intra_group, inter_group, inter_group]


def test_dp_multi_instance_force_allreduce_records_both_allreduce_stages(monkeypatch):
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
    monkeypatch.setattr(
        param_and_grad_buffer.torch.distributed,
        "get_process_group_ranks",
        lambda group: [1] if group is intra_group else [1, 3],
    )

    result = bucket_group.start_grad_sync(force_all_reduce=True)

    assert result is None
    assert collective_groups == [intra_group, intra_group, inter_group, inter_group]
    assert [record["name"] for record in sink.records] == ["dp-allreduce", "dp-allreduce"]
    intra_record, inter_record = sink.records
    assert intra_record["ctx"]["stage"] == "main_bucket_allreduce"
    assert intra_record["ctx"]["group_role"] == "intra_optimizer_instance"
    assert intra_record["values"] == {"data_bytes": 22, "group": []}
    assert inter_record["ctx"]["stage"] == "inter_instance_shard_allreduce"
    assert inter_record["ctx"]["group_role"] == "inter_optimizer_instance"
    assert inter_record["values"] == {"data_bytes": 22, "group": [3]}
