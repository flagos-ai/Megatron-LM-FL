from __future__ import annotations

import gc
import sys
import weakref
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.pipeline_parallel import p2p_communication


class _FakeGroup:
    def __init__(self, *, size: int = 3, rank: int = 1) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _FakeRequest:
    def __init__(
        self, name: str, *, result: Any = None, error: BaseException | None = None
    ) -> None:
        self.name = name
        self.result = result
        self.error = error
        self.wait_calls = 0
        self.wait_invocations: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def wait(self, *args, **kwargs):
        self.wait_calls += 1
        self.wait_invocations.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class _FailOnEnterLock:
    def __enter__(self):
        raise AssertionError("trace-off wait acquired the observation lock")

    def __exit__(self, exc_type, exc_value, traceback):
        return False


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
    def __init__(self, enabled_names: set[str] | None = None) -> None:
        self.enabled_names = enabled_names
        self.records: list[dict[str, Any]] = []

    def is_enabled(self, name: str) -> bool:
        return self.enabled_names is None or name in self.enabled_names

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


def _make_communicator(
    monkeypatch,
    *,
    group_size: int = 3,
    group_rank: int = 1,
    backend: str = "nccl",
    **config_overrides,
):
    group = _FakeGroup(size=group_size, rank=group_rank)
    config_values = {
        "enable_hetero": False,
        "timers": None,
        "variable_seq_lengths": False,
        "mtp_standalone": False,
        "use_ring_exchange_p2p": False,
        "batch_p2p_comm": False,
        "batch_p2p_sync": False,
        "pipeline_dtype": torch.float32,
        "virtual_pipeline_model_parallel_size": None,
    }
    config_values.update(config_overrides)
    monkeypatch.setattr(
        p2p_communication.dist, "get_global_rank", lambda process_group, group_rank: 10 + group_rank
    )
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "get_backend", lambda process_group: backend
    )
    communicator = p2p_communication.P2PCommunicator(
        pp_group=group, config=SimpleNamespace(**config_values)
    )
    return communicator


def test_send_forward_records_launch_and_wait_without_changing_call_semantics(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    request = _FakeRequest("send-next", result=True)
    tensor = torch.arange(6, dtype=torch.float16)
    isend_calls = []

    def fake_isend(*, tensor, dst, group):
        isend_calls.append((tensor, dst, group))
        return request

    monkeypatch.setattr(p2p_communication.torch.distributed, "isend", fake_isend)

    result = communicator.send_forward(tensor, is_last_stage=False)

    assert result is None
    assert isend_calls == [(tensor, communicator.next_rank, communicator.pp_group)]
    assert request.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-launch", "send-forward"]

    launch, wait = sink.records
    assert launch["ctx"]["comm_type"] == "p2p-launch"
    assert launch["ctx"]["timing_phase"] == "launch"
    assert launch["ctx"]["backend"] == "nccl"
    assert launch["ctx"]["transport_api"] == "isend_irecv"
    assert launch["ctx"]["request_pairing"] == "key"
    assert launch["ctx"]["backends"] == ["nccl"]
    assert launch["ctx"]["operation_count"] == 1
    operation = launch["ctx"]["operations"][0]
    assert operation["direction"] == "send"
    assert operation["pipeline_direction"] == "forward"
    assert operation["peer_rank"] == communicator.next_rank
    assert operation["data_bytes"] == tensor.numel() * tensor.element_size()
    assert operation["microbatch"] is None

    assert wait["ctx"]["timing_phase"] == "stream_dependency"
    assert wait["ctx"]["comm_type"] == "p2p"
    assert wait["ctx"]["operation_id"] == operation["operation_id"]
    assert wait["ctx"]["request_id"] == operation["request_id"]
    assert wait["ctx"]["completion_kind"] == "work_wait"
    assert wait["ctx"]["completion_guarantee"] == "current_stream_after_wait"
    assert wait["ctx"]["host_blocking_guaranteed"] is False
    assert wait["ctx"]["timeout_supplied"] is False
    assert "is_blocking" not in wait["ctx"]
    assert wait["slots"] == ("completed", "error_type")
    assert wait["values"] == {"completed": True}
    assert wait["exit_exception"] is None


def test_overlap_bidirectional_requests_record_wait_with_launch_identity(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    send_request = _FakeRequest("send-next", result="send-complete")
    recv_request = _FakeRequest("recv-prev", result="recv-complete")
    output = torch.arange(4, dtype=torch.float32)
    real_empty = torch.empty

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: send_request
    )
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "irecv", lambda *, tensor, src, group: recv_request
    )

    input_tensor, requests = communicator.send_forward_recv_forward(
        output, recv_prev=True, tensor_shape=(4,), overlap_p2p_comm=True
    )

    assert tuple(input_tensor.shape) == (4,)
    assert list(requests) == ["recv_prev", "send_next"]
    assert requests["send_next"] is send_request
    assert requests["recv_prev"] is recv_request
    assert requests["send_next"].name == "send-next"
    assert requests["recv_prev"].name == "recv-prev"
    assert (
        p2p_communication.wait_p2p_request(communicator, requests["send_next"], timeout=1)
        == "send-complete"
    )
    assert (
        p2p_communication.wait_p2p_request(communicator, requests["recv_prev"]) == "recv-complete"
    )
    assert send_request.wait_calls == 1
    assert recv_request.wait_calls == 1

    assert [record["name"] for record in sink.records] == [
        "p2p-launch",
        "send-forward",
        "recv-forward",
    ]
    launch = sink.records[0]
    launched_by_direction = {
        (operation["direction"], operation["pipeline_direction"]): operation
        for operation in launch["ctx"]["operations"]
    }
    send_operation = launched_by_direction[("send", "forward")]
    recv_operation = launched_by_direction[("recv", "forward")]
    assert send_operation["peer_rank"] == communicator.next_rank
    assert recv_operation["peer_rank"] == communicator.prev_rank
    assert send_operation["data_bytes"] == output.numel() * output.element_size()
    assert recv_operation["data_bytes"] == input_tensor.numel() * input_tensor.element_size()
    assert send_operation["operation_id"] != recv_operation["operation_id"]
    assert launch["ctx"]["completion_mode"] == "external_wait"

    send_wait, recv_wait = sink.records[1:]
    assert send_wait["ctx"]["operation_id"] == send_operation["operation_id"]
    assert recv_wait["ctx"]["operation_id"] == recv_operation["operation_id"]
    assert send_wait["ctx"]["batch_id"] == launch["ctx"]["batch_id"]
    assert recv_wait["ctx"]["batch_id"] == launch["ctx"]["batch_id"]
    assert send_wait["ctx"]["timeout_supplied"] is True
    assert recv_wait["ctx"]["timeout_supplied"] is False
    assert send_wait["ctx"]["host_blocking_guaranteed"] is False
    assert recv_wait["ctx"]["host_blocking_guaranteed"] is False
    assert send_wait["ctx"]["completion_site"] == "exposed_request_wait"
    assert recv_wait["ctx"]["completion_site"] == "exposed_request_wait"
    assert send_wait["ctx"]["duration_attribution"] == "per_request"
    assert recv_wait["ctx"]["duration_attribution"] == "per_request"
    assert "completion_api" not in send_wait["ctx"]
    assert "wait_call_index" not in send_wait["ctx"]
    assert send_wait["values"] == {"completed": True}
    assert recv_wait["values"] == {"completed": True}


def test_external_wait_false_result_records_incomplete_without_changing_return(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    request = _FakeRequest("send-next", result=False)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: request
    )

    _, requests = communicator.send_forward_recv_forward(
        torch.ones(2), recv_prev=False, tensor_shape=(2,), overlap_p2p_comm=True
    )
    assert requests["send_next"] is request
    result = p2p_communication.wait_p2p_request(communicator, requests["send_next"], timeout=1)

    assert result is False
    assert request.wait_invocations == [((), {"timeout": 1})]
    wait = sink.records[1]
    assert wait["name"] == "send-forward"
    assert wait["ctx"]["timeout_supplied"] is True
    assert wait["ctx"]["completion_site"] == "exposed_request_wait"
    assert wait["ctx"]["duration_attribution"] == "per_request"
    assert "launch_observed" not in wait["ctx"]
    assert wait["values"] == {"completed": False}
    assert wait["exit_exception"] is None


def test_raw_external_work_wait_preserves_behavior_without_completion_event(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    result_token = object()
    request = _FakeRequest("send-next", result=result_token)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: request
    )

    _, requests = communicator.send_forward_recv_forward(
        torch.ones(2), recv_prev=False, tensor_shape=(2,), overlap_p2p_comm=True
    )

    assert requests["send_next"] is request
    assert requests["send_next"].wait(timeout=3) is result_token
    assert request.wait_invocations == [((), {"timeout": 3})]
    assert [record["name"] for record in sink.records] == ["p2p-launch"]


def test_wait_p2p_request_falls_back_to_raw_work_for_custom_communicator() -> None:
    custom_communicator = SimpleNamespace()
    result_token = object()
    request = _FakeRequest("custom", result=result_token)

    result = p2p_communication.wait_p2p_request(custom_communicator, request, "arg", timeout=5)

    assert result is result_token
    assert request.wait_invocations == [(("arg",), {"timeout": 5})]

    wait_error = RuntimeError("custom wait failed")
    failing_request = _FakeRequest("custom-error", error=wait_error)
    with pytest.raises(RuntimeError) as raised:
        p2p_communication.wait_p2p_request(custom_communicator, failing_request)

    assert raised.value is wait_error
    assert failing_request.wait_invocations == [((), {})]


def test_batched_forward_send_backward_recv_pairs_list_requests_with_waits(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    send_request = _FakeRequest("send-next", result=True)
    recv_request = _FakeRequest("recv-next", result=True)
    output = torch.arange(8, dtype=torch.float16)
    real_empty = torch.empty
    batched_ops = []

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_batch_isend_irecv(ops):
        batched_ops.extend(ops)
        return [send_request, recv_request]

    sync_calls = 0

    def fake_synchronize():
        nonlocal sync_calls
        assert send_request.wait_calls == 1
        assert recv_request.wait_calls == 1
        sync_calls += 1

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv
    )
    monkeypatch.setattr(p2p_communication.cur_platform, "synchronize", fake_synchronize)

    output_grad = communicator.send_forward_recv_backward(
        output, tensor_shapes=(8,), is_last_stage=False
    )

    assert tuple(output_grad.shape) == (8,)
    assert len(batched_ops) == 2
    assert batched_ops[0].op is p2p_communication.torch.distributed.isend
    assert batched_ops[0].peer == communicator.next_rank
    assert batched_ops[1].op is p2p_communication.torch.distributed.irecv
    assert batched_ops[1].peer == communicator.next_rank
    assert send_request.wait_calls == 1
    assert recv_request.wait_calls == 1
    assert sync_calls == 1

    assert [record["name"] for record in sink.records] == [
        "p2p-launch",
        "send-forward",
        "recv-backward",
        "p2p-batch-device-sync",
    ]
    launch = sink.records[0]["ctx"]
    assert launch["backend"] == "nccl"
    assert launch["transport_api"] == "batch_isend_irecv"
    assert launch["request_pairing"] == "backend_dependent"
    assert launch["backends"] == ["nccl"]
    assert launch["completion_mode"] == "internal_wait"
    assert launch["operation_count"] == 2
    for operation, wait in zip(launch["operations"], sink.records[1:3]):
        assert wait["ctx"]["operation_id"] == operation["operation_id"]
        assert wait["ctx"]["request_id"] == operation["request_id"]
        assert wait["ctx"]["request_pairing"] == "position"
        assert wait["ctx"]["completion_guarantee"] == "current_stream_after_wait"
        assert wait["ctx"]["host_blocking_guaranteed"] is False
        assert wait["values"] == {"completed": True}
    device_sync = sink.records[3]["ctx"]
    assert device_sync["operation_ids"] == [
        operation["operation_id"] for operation in launch["operations"]
    ]
    assert device_sync["physical_request_count"] == 2
    assert device_sync["request_pairing"] == "position"
    assert sink.records[3]["values"] == {
        "completed": True,
        "device_completion_guaranteed": True,
        "host_blocking_guaranteed": True,
    }


def test_batched_coalesced_work_correlates_all_logical_operations(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True)
    aggregate_request = _FakeRequest("coalesced-batch", result=True)
    real_empty = torch.empty
    batched_ops = []

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_batch_isend_irecv(ops):
        batched_ops.extend(ops)
        return [aggregate_request]

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv
    )
    monkeypatch.setattr(
        p2p_communication.cur_platform,
        "synchronize",
        lambda: pytest.fail("batch_p2p_sync=False reached device synchronize"),
    )

    recv_prev, recv_next, requests = communicator._communicate(
        tensor_send_next=torch.full((4,), 2.0),
        tensor_send_prev=torch.full((4,), 1.0),
        recv_prev=True,
        recv_next=True,
        tensor_shape=(4,),
    )

    assert tuple(recv_prev.shape) == (4,)
    assert tuple(recv_next.shape) == (4,)
    assert requests is None
    assert len(batched_ops) == 4
    assert aggregate_request.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-launch", "p2p-batch-complete"]

    launch, completion = (record["ctx"] for record in sink.records)
    operation_ids = [operation["operation_id"] for operation in launch["operations"]]
    aggregate_request_id = f'{launch["batch_id"]}:aggregate'
    assert launch["request_pairing"] == "backend_dependent"
    assert completion["batch_id"] == launch["batch_id"]
    assert completion["request_pairing"] == "aggregate"
    assert completion["physical_request_count"] == 1
    assert completion["request_id"] == aggregate_request_id
    assert completion["operation_count"] == 4
    assert completion["operation_ids"] == operation_ids
    assert completion["operations"] == launch["operations"]
    assert completion["completion_kind"] == "aggregate_work_wait"
    assert completion["completion_guarantee"] == "current_stream_after_wait"
    assert completion["host_blocking_guaranteed"] is False
    assert completion["duration_attribution"] == "shared_nonexclusive"
    assert sink.records[1]["slots"] == ("completed", "error_type")
    assert sink.records[1]["values"] == {"completed": True}


def test_batched_coalesced_wait_preserves_exception_identity() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    wait_error = RuntimeError("aggregate wait failed")
    aggregate_request = _FakeRequest("coalesced-batch", error=wait_error)
    operations = p2p_communication._build_p2p_operations(
        tensor_send_prev=None,
        tensor_recv_prev=torch.empty(2),
        tensor_send_next=torch.ones(2),
        tensor_recv_next=None,
        prev_pipeline_rank=10,
        next_pipeline_rank=12,
        backend="nccl",
        transport_api="batch_isend_irecv",
        wait_on_reqs=True,
    )

    with pytest.raises(RuntimeError) as raised:
        p2p_communication._wait_p2p_batch_request(aggregate_request, operations)

    assert raised.value is wait_error
    assert aggregate_request.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-batch-complete"]
    assert sink.records[0]["ctx"]["operation_ids"] == [
        operation.operation_id for operation in operations
    ]
    assert sink.records[0]["values"] == {"completed": False, "error_type": "RuntimeError"}
    assert sink.records[0]["exit_exception"] is RuntimeError


def test_batched_coalesced_trace_off_skips_observation_metadata(monkeypatch) -> None:
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True)
    aggregate_request = _FakeRequest("coalesced-batch", result=True)
    real_empty = torch.empty

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: [aggregate_request]
    )
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("trace-off path built P2P observation metadata"),
    )

    output_grad = communicator.send_forward_recv_backward(
        torch.ones(4), tensor_shapes=(4,), is_last_stage=False
    )

    assert tuple(output_grad.shape) == (4,)
    assert aggregate_request.wait_calls == 1


def test_batched_non_positional_trace_off_waits_every_work_and_syncs(monkeypatch) -> None:
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    requests = [_FakeRequest(f"backend-{index}") for index in range(3)]
    real_empty = torch.empty
    synchronize_calls = []

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: requests
    )
    monkeypatch.setattr(
        p2p_communication.cur_platform, "synchronize", lambda: synchronize_calls.append(True)
    )
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("trace-off path built P2P observation metadata"),
    )

    output_grad = communicator.send_forward_recv_backward(
        torch.ones(4), tensor_shapes=(4,), is_last_stage=False
    )

    assert tuple(output_grad.shape) == (4,)
    assert [request.wait_calls for request in requests] == [1, 1, 1]
    assert synchronize_calls == [True]


def test_batched_non_positional_trace_off_preserves_wait_exception_identity(monkeypatch) -> None:
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    wait_error = RuntimeError("backend wait failed")
    requests = [
        _FakeRequest("backend-0", error=wait_error),
        _FakeRequest("backend-1"),
        _FakeRequest("backend-2"),
    ]
    real_empty = torch.empty

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: requests
    )
    monkeypatch.setattr(
        p2p_communication.cur_platform,
        "synchronize",
        lambda: pytest.fail("failed wait reached batch device synchronize"),
    )
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("trace-off path built P2P observation metadata"),
    )

    with pytest.raises(RuntimeError) as raised:
        communicator.send_forward_recv_backward(
            torch.ones(4), tensor_shapes=(4,), is_last_stage=False
        )

    assert raised.value is wait_error
    assert [request.wait_calls for request in requests] == [1, 0, 0]


def test_batched_non_positional_request_count_preserves_backend_wait_behavior(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    requests = [_FakeRequest(f"unexpected-{index}") for index in range(3)]
    real_empty = torch.empty
    synchronize_calls = []

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: requests
    )
    monkeypatch.setattr(
        p2p_communication.cur_platform, "synchronize", lambda: synchronize_calls.append(True)
    )

    output_grad = communicator.send_forward_recv_backward(
        torch.ones(4), tensor_shapes=(4,), is_last_stage=False
    )

    assert tuple(output_grad.shape) == (4,)
    assert all(request.wait_calls == 1 for request in requests)
    assert synchronize_calls == [True]
    assert [record["name"] for record in sink.records] == ["p2p-launch", "p2p-batch-device-sync"]
    assert sink.records[0]["exit_exception"] is None
    sync_context = sink.records[1]["ctx"]
    assert sync_context["operation_count"] == 2
    assert sync_context["physical_request_count"] == 3
    assert sync_context["request_pairing"] == "unknown"


def test_batch_device_sync_runs_after_aggregate_wait_and_correlates_operations(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    order = []
    batched_ops = []
    real_empty = torch.empty

    class _OrderedRequest:
        def wait(self):
            order.append("wait")
            return True

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_batch_isend_irecv(ops):
        order.append("launch")
        batched_ops.extend(ops)
        return [_OrderedRequest()]

    def fake_synchronize():
        order.append("device-sync")

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv
    )
    monkeypatch.setattr(p2p_communication.cur_platform, "synchronize", fake_synchronize)

    output_grad = communicator.send_forward_recv_backward(
        torch.ones(4), tensor_shapes=(4,), is_last_stage=False
    )

    assert tuple(output_grad.shape) == (4,)
    assert len(batched_ops) == 2
    assert order == ["launch", "wait", "device-sync"]
    assert [record["name"] for record in sink.records] == [
        "p2p-launch",
        "p2p-batch-complete",
        "p2p-batch-device-sync",
    ]
    launch, wait, device_sync = (record["ctx"] for record in sink.records)
    operation_ids = [operation["operation_id"] for operation in launch["operations"]]
    assert wait["operation_ids"] == operation_ids
    assert device_sync["operation_ids"] == operation_ids
    assert device_sync["operations"] == launch["operations"]
    assert device_sync["batch_id"] == launch["batch_id"]
    assert device_sync["physical_request_count"] == 1
    assert device_sync["request_pairing"] == "aggregate"
    assert device_sync["completion_kind"] == "device_synchronize"
    assert device_sync["completion_guarantee"] == "host_after_current_device_synchronize"
    assert device_sync["completion_site"] == "batch_p2p_sync_workaround"
    assert device_sync["device_scope"] == "current_device"
    assert device_sync["duration_attribution"] == "device_wide_nonexclusive"
    assert sink.records[2]["slots"] == (
        "completed",
        "device_completion_guaranteed",
        "error_type",
        "host_blocking_guaranteed",
    )
    assert sink.records[2]["values"] == {
        "completed": True,
        "device_completion_guaranteed": True,
        "host_blocking_guaranteed": True,
    }


def test_batch_device_sync_only_gate_builds_metadata_at_sync_boundary(monkeypatch) -> None:
    sink = _RecordingSink(enabled_names={"p2p-batch-device-sync"})
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    request = _FakeRequest("single-batch-work", result=True)
    sync_calls = 0

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_synchronize():
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: [request]
    )
    monkeypatch.setattr(p2p_communication.cur_platform, "synchronize", fake_synchronize)
    monkeypatch.setattr(
        p2p_communication,
        "_p2p_launch_context",
        lambda *args, **kwargs: pytest.fail("sync-only gate built launch metadata"),
    )

    result = communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert result is None
    assert request.wait_calls == 1
    assert sync_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-batch-device-sync"]
    context = sink.records[0]["ctx"]
    assert context["operation_count"] == 1
    assert context["physical_request_count"] == 1
    assert context["request_pairing"] == "position"
    assert context["has_p2p_operations"] is True


def test_batch_device_sync_trace_off_skips_metadata_and_preserves_calls(monkeypatch) -> None:
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    request = _FakeRequest("single-batch-work", result=True)
    sync_calls = 0

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_synchronize():
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: [request]
    )
    monkeypatch.setattr(p2p_communication.cur_platform, "synchronize", fake_synchronize)
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("trace-off sync path built observation metadata"),
    )

    result = communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert result is None
    assert request.wait_calls == 1
    assert sync_calls == 1


def test_batch_device_sync_preserves_exception_identity(monkeypatch) -> None:
    sink = _RecordingSink(enabled_names={"p2p-batch-device-sync"})
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    request = _FakeRequest("single-batch-work", result=True)
    sync_error = RuntimeError("device synchronize failed")
    sync_calls = 0

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fail_synchronize():
        nonlocal sync_calls
        sync_calls += 1
        raise sync_error

    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: [request]
    )
    monkeypatch.setattr(p2p_communication.cur_platform, "synchronize", fail_synchronize)

    with pytest.raises(RuntimeError) as raised:
        communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert raised.value is sync_error
    assert request.wait_calls == 1
    assert sync_calls == 1
    assert sink.records[0]["values"] == {
        "completed": False,
        "device_completion_guaranteed": False,
        "error_type": "RuntimeError",
        "host_blocking_guaranteed": False,
    }
    assert sink.records[0]["exit_exception"] is RuntimeError


def test_batch_wait_failure_skips_device_synchronize(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    wait_error = RuntimeError("batch wait failed")
    request = _FakeRequest("single-batch-work", error=wait_error)

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    monkeypatch.setattr(p2p_communication.torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "batch_isend_irecv", lambda ops: [request]
    )
    monkeypatch.setattr(
        p2p_communication.cur_platform,
        "synchronize",
        lambda: pytest.fail("failed batch wait reached device synchronize"),
    )

    with pytest.raises(RuntimeError) as raised:
        communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert raised.value is wait_error
    assert request.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-launch", "send-forward"]


def test_batch_device_sync_records_unconditional_empty_workaround(monkeypatch) -> None:
    sink = _RecordingSink(enabled_names={"p2p-batch-device-sync"})
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, batch_p2p_comm=True, batch_p2p_sync=True)
    sync_calls = 0

    def fake_synchronize():
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setattr(p2p_communication.cur_platform, "synchronize", fake_synchronize)

    recv_prev, recv_next, requests = communicator._communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=(2,),
    )

    assert recv_prev is None
    assert recv_next is None
    assert requests == []
    assert sync_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-batch-device-sync"]
    context = sink.records[0]["ctx"]
    assert context["batch_id"] is None
    assert context["operation_count"] == 0
    assert context["operation_ids"] == []
    assert context["physical_request_count"] == 0
    assert context["request_pairing"] == "none"
    assert context["has_p2p_operations"] is False
    assert context["backend_complete"] is False
    assert sink.records[0]["values"] == {
        "completed": True,
        "device_completion_guaranteed": True,
        "host_blocking_guaranteed": True,
    }


def test_overlap_backward_wait_records_both_directions_and_preserves_exception(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    wait_error = RuntimeError("send wait failed")
    send_request = _FakeRequest("send-prev", error=wait_error)
    recv_request = _FakeRequest("recv-next", result="recv-complete")
    input_grad = torch.arange(4, dtype=torch.float32)
    real_empty = torch.empty

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: send_request
    )
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "irecv", lambda *, tensor, src, group: recv_request
    )

    output_grad, requests = communicator.send_backward_recv_backward(
        input_grad, recv_next=True, tensor_shape=(4,), overlap_p2p_comm=True
    )

    assert tuple(output_grad.shape) == (4,)
    assert list(requests) == ["recv_next", "send_prev"]
    assert requests["send_prev"] is send_request
    assert requests["recv_next"] is recv_request
    with pytest.raises(RuntimeError) as raised:
        p2p_communication.wait_p2p_request(communicator, requests["send_prev"], timeout=7)
    assert raised.value is wait_error
    assert (
        p2p_communication.wait_p2p_request(communicator, requests["recv_next"]) == "recv-complete"
    )
    assert send_request.wait_invocations == [((), {"timeout": 7})]
    assert recv_request.wait_invocations == [((), {})]

    assert [record["name"] for record in sink.records] == [
        "p2p-launch",
        "send-backward",
        "recv-backward",
    ]
    launch = sink.records[0]["ctx"]
    launched_by_direction = {
        (operation["direction"], operation["pipeline_direction"]): operation
        for operation in launch["operations"]
    }
    send_wait, recv_wait = sink.records[1:]
    assert (
        send_wait["ctx"]["operation_id"]
        == launched_by_direction[("send", "backward")]["operation_id"]
    )
    assert (
        recv_wait["ctx"]["operation_id"]
        == launched_by_direction[("recv", "backward")]["operation_id"]
    )
    assert send_wait["exit_exception"] is RuntimeError
    assert recv_wait["exit_exception"] is None
    assert send_wait["ctx"]["timeout_supplied"] is True
    assert recv_wait["ctx"]["timeout_supplied"] is False
    assert send_wait["values"] == {"completed": False, "error_type": "RuntimeError"}
    assert recv_wait["values"] == {"completed": True}


def test_launch_exception_is_recorded_and_propagated_unchanged(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    launch_error = RuntimeError("isend launch failed")

    def failing_isend(*, tensor, dst, group):
        raise launch_error

    monkeypatch.setattr(p2p_communication.torch.distributed, "isend", failing_isend)

    with pytest.raises(RuntimeError) as raised:
        communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert raised.value is launch_error
    assert [record["name"] for record in sink.records] == ["p2p-launch"]
    assert sink.records[0]["ctx"]["timing_phase"] == "launch"
    assert sink.records[0]["exit_exception"] is RuntimeError


def test_enabled_trace_skips_empty_p2p_launch_without_changing_noop_result(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)

    result = communicator._communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=(1,),
    )

    assert result == (None, None, {})
    assert sink.records == []


def test_wait_only_gate_builds_completion_metadata_without_launch_scope(monkeypatch) -> None:
    sink = _RecordingSink(enabled_names={"send-forward"})
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    request = _FakeRequest("send-next", result=True)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: request
    )

    result = communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert result is None
    assert request.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["send-forward"]
    assert sink.records[0]["ctx"]["completion_site"] == "communicate_internal_wait"
    assert sink.records[0]["values"] == {"completed": True}


def test_launch_only_gate_skips_directional_wait_metadata(monkeypatch) -> None:
    sink = _RecordingSink(enabled_names={"p2p-launch"})
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    request = _FakeRequest("send-next", result=True)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: request
    )
    monkeypatch.setattr(
        p2p_communication,
        "_p2p_wait_context",
        lambda *args, **kwargs: pytest.fail("disabled wait event built completion metadata"),
    )

    result = communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert result is None
    assert request.wait_calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-launch"]


def test_ring_exchange_records_inline_launch_without_work_wait(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    assert getattr(p2p_communication._launch_p2p, "__megatron_trace_event__", None) == (
        "p2p-launch"
    )
    communicator = _make_communicator(monkeypatch, use_ring_exchange_p2p=True)
    output = torch.arange(6, dtype=torch.float16)
    real_empty = torch.empty
    ring_calls = []

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def fake_ring_exchange(**kwargs):
        ring_calls.append(kwargs)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(
        p2p_communication.torch.distributed, "ring_exchange", fake_ring_exchange, raising=False
    )

    output_grad = communicator.send_forward_recv_backward(
        output, tensor_shapes=(6,), is_last_stage=False
    )

    assert tuple(output_grad.shape) == (6,)
    assert len(ring_calls) == 1
    assert ring_calls[0]["tensor_send_next"] is output
    assert ring_calls[0]["tensor_recv_next"] is output_grad
    assert [record["name"] for record in sink.records] == ["p2p-launch"]
    launch = sink.records[0]["ctx"]
    assert launch["backend"] == "nccl"
    assert launch["transport_api"] == "ring_exchange"
    assert launch["completion_mode"] == "inline"
    assert launch["completion_included"] is False
    assert launch["api_return_included"] is True
    assert launch["completion_kind"] == "inline_api_return"
    assert launch["completion_guarantee"] == "api_return_observed"
    assert launch["timing_phase"] == "inline_api_call"
    assert launch["host_blocking_guaranteed"] is False
    assert launch["device_completion_guaranteed"] is False
    assert launch["duration_attribution"] == "shared_nonexclusive"
    assert launch["physical_request_count"] == 0
    assert launch["operation_count"] == 2
    assert [operation["pipeline_direction"] for operation in launch["operations"]] == [
        "forward",
        "backward",
    ]
    assert all(operation["request_id"] is None for operation in launch["operations"])
    assert launch["request_pairing"] == "none"
    assert launch["backends"] == ["nccl"]
    assert launch["operation_ids"] == [
        operation["operation_id"] for operation in launch["operations"]
    ]
    assert sink.records[0]["slots"] == ("completed", "error_type")
    assert sink.records[0]["values"] == {"completed": True}


def test_ring_exchange_inline_launch_preserves_exception_identity(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, use_ring_exchange_p2p=True)
    ring_error = RuntimeError("ring exchange failed")
    calls = 0

    def fail_ring_exchange(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        raise ring_error

    monkeypatch.setattr(
        p2p_communication.torch.distributed, "ring_exchange", fail_ring_exchange, raising=False
    )

    with pytest.raises(RuntimeError) as raised:
        communicator._communicate(
            tensor_send_next=torch.ones(2),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=(2,),
        )

    assert raised.value is ring_error
    assert calls == 1
    assert [record["name"] for record in sink.records] == ["p2p-launch"]
    assert sink.records[0]["ctx"]["completion_kind"] == "inline_api_return"
    assert sink.records[0]["values"] == {"completed": False, "error_type": "RuntimeError"}
    assert sink.records[0]["exit_exception"] is RuntimeError


def test_ring_exchange_trace_off_skips_metadata_and_calls_transport_once(monkeypatch) -> None:
    communicator = _make_communicator(monkeypatch, use_ring_exchange_p2p=True)
    calls = 0

    def fake_ring_exchange(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1

    monkeypatch.setattr(
        p2p_communication.torch.distributed, "ring_exchange", fake_ring_exchange, raising=False
    )
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("trace-off ring path built observation metadata"),
    )

    recv_prev, recv_next, requests = communicator._communicate(
        tensor_send_next=torch.ones(2),
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=(2,),
    )

    assert recv_prev is None
    assert recv_next is None
    assert requests == []
    assert calls == 1


def test_ring_exchange_ignores_directional_wait_only_gate(monkeypatch) -> None:
    sink = _RecordingSink(enabled_names={"send-forward"})
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch, use_ring_exchange_p2p=True)
    calls = 0

    def fake_ring_exchange(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1

    monkeypatch.setattr(
        p2p_communication.torch.distributed, "ring_exchange", fake_ring_exchange, raising=False
    )
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("ring path built metadata for a Work-only event"),
    )

    result = communicator.send_forward(torch.ones(2), is_last_stage=False)

    assert result is None
    assert calls == 1
    assert sink.records == []


@pytest.mark.parametrize("group_rank", [0, 1])
@pytest.mark.parametrize(
    ("pipeline_backend", "world_backend", "uses_world"),
    [("gloo", "nccl", True), ("gloo", None, True), ("ucc", "nccl", False)],
)
def test_size2_unbatched_records_backend_of_actual_request_group(
    monkeypatch, group_rank, pipeline_backend, world_backend, uses_world
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(
        monkeypatch, group_size=2, group_rank=group_rank, backend=pipeline_backend
    )
    world_group = p2p_communication.torch.distributed.group.WORLD
    real_empty = torch.empty
    calls = []

    def cpu_empty(shape, *, requires_grad, device, dtype):
        del device
        return real_empty(shape, requires_grad=requires_grad, dtype=dtype)

    def backend_for(group):
        if group is communicator.pp_group:
            return pipeline_backend
        if group is world_group:
            if world_backend is None:
                raise RuntimeError("WORLD backend metadata unavailable")
            return world_backend
        raise AssertionError("unexpected P2P process group")

    def fake_isend(*, tensor, dst, group):
        calls.append(("send", tensor, group))
        return _FakeRequest(f"send-{len(calls)}", result=True)

    def fake_irecv(*, tensor, src, group):
        calls.append(("recv", tensor, group))
        return _FakeRequest(f"recv-{len(calls)}", result=True)

    monkeypatch.setattr(p2p_communication.torch, "empty", cpu_empty)
    monkeypatch.setattr(p2p_communication.torch.distributed, "get_backend", backend_for)
    monkeypatch.setattr(p2p_communication.torch.distributed, "isend", fake_isend)
    monkeypatch.setattr(p2p_communication.torch.distributed, "irecv", fake_irecv)
    send_prev = torch.full((2,), 1.0)
    send_next = torch.full((2,), 2.0)

    recv_prev, recv_next, requests = communicator._communicate(
        tensor_send_next=send_next,
        tensor_send_prev=send_prev,
        recv_prev=True,
        recv_next=True,
        tensor_shape=(2,),
    )

    assert requests is None
    group_by_key = {}
    call_keys = []
    for kind, tensor, group in calls:
        if kind == "send" and tensor is send_prev:
            key = "send_prev"
        elif kind == "send" and tensor is send_next:
            key = "send_next"
        elif kind == "recv" and tensor is recv_prev:
            key = "recv_prev"
        elif kind == "recv" and tensor is recv_next:
            key = "recv_next"
        else:
            raise AssertionError("unexpected P2P tensor")
        call_keys.append(key)
        group_by_key[key] = group
    assert set(group_by_key) == {"send_prev", "recv_prev", "send_next", "recv_next"}
    assert call_keys == (
        ["send_next", "recv_prev", "send_prev", "recv_next"]
        if group_rank % 2 == 0
        else ["recv_prev", "send_next", "recv_next", "send_prev"]
    )
    if uses_world:
        primary_keys = (
            {"send_prev", "send_next"} if group_rank % 2 == 0 else {"recv_prev", "recv_next"}
        )
        assert {
            key: communicator.pp_group if key in primary_keys else world_group
            for key in group_by_key
        } == group_by_key
    else:
        assert all(group is communicator.pp_group for group in group_by_key.values())

    launch = sink.records[0]["ctx"]
    assert launch["transport_api"] == "isend_irecv"
    assert launch["completion_mode"] == "internal_wait"
    assert launch["request_pairing"] == "key"
    metadata_complete = not uses_world or world_backend is not None
    assert launch["backend_complete"] is metadata_complete
    assert launch["backend"] == (
        None if not metadata_complete else "mixed" if uses_world else pipeline_backend
    )
    assert launch["backends"] == sorted(
        {backend for backend in (pipeline_backend, world_backend) if backend is not None}
        if uses_world
        else {pipeline_backend}
    )
    for operation in launch["operations"]:
        request_group = group_by_key[operation["operation_id"].rsplit(":", 1)[-1]]
        expected_backend = (
            pipeline_backend if request_group is communicator.pp_group else world_backend
        )
        assert operation["backend"] == expected_backend


def test_trace_off_keeps_async_request_behavior_and_does_not_import_megalens(monkeypatch) -> None:
    communicator = _make_communicator(monkeypatch)
    send_request = _FakeRequest("send-next", result="done")
    output = torch.arange(3, dtype=torch.float32)
    megalens_modules_before = {
        module_name for module_name in sys.modules if module_name.startswith("megatron.megalens")
    }

    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: send_request
    )
    monkeypatch.setattr(
        p2p_communication,
        "_build_p2p_operations",
        lambda **kwargs: pytest.fail("trace-off path built P2P observation metadata"),
    )

    _, requests = communicator.send_forward_recv_forward(
        output, recv_prev=False, tensor_shape=(3,), overlap_p2p_comm=True
    )

    assert list(requests) == ["send_next"]
    assert requests["send_next"] is send_request
    assert requests["send_next"].name == "send-next"
    assert communicator._p2p_work_observations == {}
    communicator._p2p_work_observation_lock = _FailOnEnterLock()
    assert (
        p2p_communication.wait_p2p_request(communicator, requests["send_next"], timeout=11)
        == "done"
    )
    assert send_request.wait_invocations == [((), {"timeout": 11})]
    assert {
        module_name for module_name in sys.modules if module_name.startswith("megatron.megalens")
    } == megalens_modules_before


def test_async_observation_metadata_does_not_extend_tensor_lifetime(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    communicator = _make_communicator(monkeypatch)
    send_request = _FakeRequest("send-next", result=True)
    output = torch.arange(3, dtype=torch.float32)
    output_ref = weakref.ref(output)

    monkeypatch.setattr(
        p2p_communication.torch.distributed, "isend", lambda *, tensor, dst, group: send_request
    )

    _, requests = communicator.send_forward_recv_forward(
        output, recv_prev=False, tensor_shape=(3,), overlap_p2p_comm=True
    )
    del output
    gc.collect()

    assert output_ref() is None
    assert p2p_communication.wait_p2p_request(communicator, requests["send_next"]) is True
    assert [record["name"] for record in sink.records] == ["p2p-launch", "send-forward"]


def test_async_observation_sidecar_does_not_extend_work_lifetime(monkeypatch) -> None:
    install_trace_sink(_RecordingSink())
    communicator = _make_communicator(monkeypatch)
    request = _FakeRequest("send-next", result=True)
    request_key = id(request)
    request_ref = weakref.ref(request)
    pending_requests = [request]

    def fake_isend(*, tensor, dst, group):
        del tensor, dst, group
        return pending_requests.pop()

    monkeypatch.setattr(p2p_communication.torch.distributed, "isend", fake_isend)

    _, requests = communicator.send_forward_recv_forward(
        torch.arange(3, dtype=torch.float32),
        recv_prev=False,
        tensor_shape=(3,),
        overlap_p2p_comm=True,
    )

    assert request_key in communicator._p2p_work_observations
    del requests
    del request
    gc.collect()

    assert request_ref() is None
    assert request_key not in communicator._p2p_work_observations
