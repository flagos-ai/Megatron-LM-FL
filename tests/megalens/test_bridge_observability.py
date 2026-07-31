# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import (
    install_trace_sink,
    reset_trace_sink,
)
from megatron.core.pipeline_parallel import bridge_communicator as bridge_module
from megatron.core.pipeline_parallel.bridge_communicator import (
    BridgeCommunicator,
    CommRole,
    RankCommInfo,
)


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.active.append(self.record["name"])
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        assert self.sink.active.pop() == self.record["name"]
        self.record["exit_exception"] = exc_type
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record.setdefault("values", {})[key] = value
        return True


class _RecordingSink:
    def __init__(self) -> None:
        self.active: list[str] = []
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
        return _RecordingScope(self, record)


class _FakeP2POp:
    def __init__(self, op: Any, tensor: torch.Tensor, peer: int, *args, **kwargs) -> None:
        del args, kwargs
        self.op = op
        self.tensor = tensor
        self.peer = peer


class _FakeWork:
    def __init__(
        self,
        *,
        op: _FakeP2POp,
        message_kind: str,
        event_name: str,
        sink: _RecordingSink | None,
        calls: list[tuple[Any, ...]],
        received_shape: tuple[int, ...],
        fill_value: int,
        error: BaseException | None = None,
    ) -> None:
        self.op = op
        self.message_kind = message_kind
        self.event_name = event_name
        self.sink = sink
        self.calls = calls
        self.received_shape = received_shape
        self.fill_value = fill_value
        self.error = error

    def wait(self):
        expected_active = [] if self.sink is None else [self.event_name]
        if self.sink is not None:
            assert self.sink.active == expected_active
        self.calls.append(
            ("wait", self.message_kind, self.op.peer, self.event_name)
        )
        if self.error is not None:
            raise self.error
        if self.op.op is torch.distributed.irecv:
            with torch.no_grad():
                if self.message_kind == "shape":
                    self.op.tensor.copy_(
                        torch.tensor(self.received_shape, dtype=torch.int64)
                    )
                else:
                    self.op.tensor.fill_(self.fill_value)
        return True


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def _make_bridge(rank: int) -> BridgeCommunicator:
    bridge = object.__new__(BridgeCommunicator)
    bridge.current_rank = rank
    bridge.src_grid = SimpleNamespace(rank_offset=0, size=1)
    bridge.dest_grid = SimpleNamespace(rank_offset=1, size=1)
    bridge.src_module_name = "encoder"
    bridge.dest_module_name = "language"
    bridge.comm_dtype = torch.float32
    bridge.tensor_ndim = 3
    bridge.dim_mapping = {"s": 0, "b": 1, "h": 2}
    bridge.src_local_leader_rank = 0
    bridge.dest_local_leader_rank = 1
    bridge.src_grid_broadcast_pg = object()
    bridge.dest_grid_broadcast_pg = object()
    bridge.src_grid_broadcast_ranks = [0]
    bridge.dest_grid_broadcast_ranks = [1]
    bridge.comm_map = {}
    return bridge


def _install_combined_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sink: _RecordingSink | None,
    send_event: str,
    recv_event: str,
    received_shape: tuple[int, ...],
    launch_error_kind: str | None = None,
    wait_error_at: tuple[str, int] | None = None,
    broadcast_error_kind: str | None = None,
) -> tuple[list[tuple[Any, ...]], list[list[_FakeP2POp]], dict[str, BaseException]]:
    calls: list[tuple[Any, ...]] = []
    batches: list[list[_FakeP2POp]] = []
    errors: dict[str, BaseException] = {}

    def fake_batch_isend_irecv(ops: list[_FakeP2POp]):
        message_kind = "shape" if ops[0].tensor.dtype == torch.int64 else "payload"
        expected_active = [] if sink is None else ["bridge-p2p-launch"]
        if sink is not None:
            assert sink.active == expected_active
        calls.append(("batch", message_kind))
        batches.append(list(ops))
        if launch_error_kind == message_kind:
            error = RuntimeError(f"{message_kind} launch failed")
            errors["launch"] = error
            raise error

        requests = []
        for index, op in enumerate(ops):
            event_name = send_event if op.op is torch.distributed.isend else recv_event
            wait_error = None
            if wait_error_at == (message_kind, index):
                wait_error = RuntimeError(f"{message_kind} wait failed")
                errors["wait"] = wait_error
            requests.append(
                _FakeWork(
                    op=op,
                    message_kind=message_kind,
                    event_name=event_name,
                    sink=sink,
                    calls=calls,
                    received_shape=received_shape,
                    fill_value=index + 1,
                    error=wait_error,
                )
            )
        return requests

    def fake_broadcast(tensor: torch.Tensor, *, src: int, group: Any):
        message_kind = "shape" if tensor.dtype == torch.int64 else "payload"
        expected_active = [] if sink is None else ["bridge-grid-broadcast"]
        if sink is not None:
            assert sink.active == expected_active
        calls.append(("broadcast", message_kind, src, group))
        if broadcast_error_kind == message_kind:
            error = RuntimeError(f"{message_kind} broadcast failed")
            errors["broadcast"] = error
            raise error

    monkeypatch.setattr(bridge_module.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(bridge_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(bridge_module.torch.distributed, "P2POp", _FakeP2POp)
    monkeypatch.setattr(
        bridge_module.torch.distributed,
        "batch_isend_irecv",
        fake_batch_isend_irecv,
    )
    monkeypatch.setattr(bridge_module.dist, "broadcast", fake_broadcast)
    return calls, batches, errors


def _assert_payload_context(
    record: Mapping[str, Any],
    *,
    event_name: str,
    direction: str,
    pipeline_direction: str,
    peer_rank: int,
    data_bytes: int,
) -> None:
    assert record["name"] == event_name
    assert record["slots"] == ("completed", "error_type")
    assert record["values"] == {"completed": True}
    assert record["exit_exception"] is None
    context = record["ctx"]
    assert context == {
        "backend": "nccl",
        "comm_type": "p2p",
        "communicator_kind": "bridge",
        "completion_guarantee": "api_return_observed",
        "completion_included": True,
        "completion_kind": "inline_api_return",
        "completion_mode": "inline",
        "data_bytes": data_bytes,
        "device_completion_guaranteed": False,
        "direction": direction,
        "duration_attribution": "per_operation",
        "message_kind": "payload",
        "operation_count": 1,
        "payload_role": (
            "activation" if pipeline_direction == "forward" else "gradient"
        ),
        "peer_rank": peer_rank,
        "pipeline_direction": pipeline_direction,
        "request_id": None,
        "request_pairing": "none",
        "src_module": "encoder",
        "dest_module": "language",
        "timing_phase": "inline_api_call",
        "transport_api": "send_recv",
    }


@pytest.mark.parametrize(
    (
        "method_name",
        "rank",
        "rank_info",
        "event_name",
        "pipeline_direction",
        "peer_rank",
    ),
    (
        (
            "send_forward",
            0,
            RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1]),
            "bridge-send-forward",
            "forward",
            1,
        ),
        (
            "send_backward",
            1,
            RankCommInfo(role=CommRole.RECEIVER, recv_from_ranks=[0]),
            "bridge-send-backward",
            "backward",
            0,
        ),
    ),
)
def test_bridge_blocking_send_records_only_the_payload_call(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    rank: int,
    rank_info: RankCommInfo,
    event_name: str,
    pipeline_direction: str,
    peer_rank: int,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(rank)
    bridge.comm_map[rank] = rank_info
    shape_calls = []
    send_calls = []
    bridge._communicate_shapes = lambda **kwargs: shape_calls.append(kwargs) or ([], [])

    def fake_send(tensor: torch.Tensor, *, dst: int):
        assert sink.active == [event_name]
        send_calls.append((tensor, dst))

    monkeypatch.setattr(bridge_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(bridge_module.dist, "send", fake_send)
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    getattr(bridge, method_name)(tensor)

    assert len(shape_calls) == 1
    assert len(send_calls) == 1
    sent_tensor, sent_peer = send_calls[0]
    assert sent_peer == peer_rank
    torch.testing.assert_close(sent_tensor, tensor)
    assert sink.active == []
    assert len(sink.records) == 1
    _assert_payload_context(
        sink.records[0],
        event_name=event_name,
        direction="send",
        pipeline_direction=pipeline_direction,
        peer_rank=peer_rank,
        data_bytes=tensor.numel() * tensor.element_size(),
    )


@pytest.mark.parametrize(
    (
        "method_name",
        "rank",
        "rank_info",
        "event_name",
        "pipeline_direction",
        "peer_rank",
        "requires_grad",
    ),
    (
        (
            "recv_forward",
            1,
            RankCommInfo(role=CommRole.RECEIVER, recv_from_ranks=[0]),
            "bridge-recv-forward",
            "forward",
            0,
            True,
        ),
        (
            "recv_backward",
            0,
            RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1]),
            "bridge-recv-backward",
            "backward",
            1,
            False,
        ),
    ),
)
def test_bridge_blocking_recv_keeps_broadcast_outside_the_payload_scope(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    rank: int,
    rank_info: RankCommInfo,
    event_name: str,
    pipeline_direction: str,
    peer_rank: int,
    requires_grad: bool,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(rank)
    bridge.comm_map[rank] = rank_info
    shape = (2, 3, 4)
    if pipeline_direction == "forward":
        bridge._communicate_shapes = lambda **kwargs: ([shape], [])
    else:
        bridge._communicate_shapes = lambda **kwargs: ([], [shape])
    calls = []

    def fake_recv(tensor: torch.Tensor, *, src: int):
        assert sink.active == [event_name]
        with torch.no_grad():
            tensor.fill_(7)
        calls.append(("recv", src, tensor))

    def fake_broadcast(tensor: torch.Tensor, *, src: int, group: Any):
        assert sink.active == ["bridge-grid-broadcast"]
        calls.append(("broadcast", src, tensor, group))

    monkeypatch.setattr(bridge_module.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(bridge_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(bridge_module.dist, "recv", fake_recv)
    monkeypatch.setattr(bridge_module.dist, "broadcast", fake_broadcast)

    result = getattr(bridge, method_name)()

    assert [call[0] for call in calls] == ["recv", "broadcast", "broadcast"]
    assert calls[0][1] == peer_rank
    assert tuple(result.shape) == shape
    assert result.dtype == torch.float32
    assert result.requires_grad is requires_grad
    assert torch.all(result == 7)
    assert [record["name"] for record in sink.records] == [
        event_name,
        "bridge-grid-broadcast",
        "bridge-grid-broadcast",
    ]
    _assert_payload_context(
        sink.records[0],
        event_name=event_name,
        direction="recv",
        pipeline_direction=pipeline_direction,
        peer_rank=peer_rank,
        data_bytes=result.numel() * result.element_size(),
    )
    shape_broadcast, payload_broadcast = sink.records[1:]
    assert shape_broadcast["ctx"]["message_kind"] == "shape"
    assert shape_broadcast["ctx"]["shape_of"] == (
        "activation" if pipeline_direction == "forward" else "gradient"
    )
    assert "payload_role" not in shape_broadcast["ctx"]
    assert payload_broadcast["ctx"]["message_kind"] == "payload"
    assert payload_broadcast["ctx"]["payload_role"] == (
        "activation" if pipeline_direction == "forward" else "gradient"
    )
    assert "shape_of" not in payload_broadcast["ctx"]
    assert all(
        record["values"] == {"completed": True}
        for record in (shape_broadcast, payload_broadcast)
    )


def test_bridge_trace_off_calls_transport_without_building_payload_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _make_bridge(0)
    bridge.comm_map[0] = RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1])
    bridge._communicate_shapes = lambda **kwargs: ([], [])
    send_calls = []
    monkeypatch.setattr(
        bridge_module,
        "_bridge_payload_context",
        lambda *args, **kwargs: pytest.fail("trace-off built Bridge payload metadata"),
    )
    monkeypatch.setattr(
        bridge_module.dist,
        "send",
        lambda tensor, *, dst: send_calls.append((tensor, dst)),
    )
    tensor = torch.ones(2, 3, 4)

    bridge.send_forward(tensor)

    assert len(send_calls) == 1
    assert send_calls[0][1] == 1
    torch.testing.assert_close(send_calls[0][0], tensor)


def test_bridge_payload_exception_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(0)
    bridge.comm_map[0] = RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1])
    bridge._communicate_shapes = lambda **kwargs: ([], [])
    transport_error = RuntimeError("bridge send failed")
    monkeypatch.setattr(bridge_module.dist, "get_backend", lambda: "nccl")

    def fail_send(tensor: torch.Tensor, *, dst: int):
        del tensor, dst
        raise transport_error

    monkeypatch.setattr(bridge_module.dist, "send", fail_send)

    with pytest.raises(RuntimeError) as raised:
        bridge.send_forward(torch.ones(2, 3, 4))

    assert raised.value is transport_error
    assert sink.records[0]["values"] == {
        "completed": False,
        "error_type": "RuntimeError",
    }
    assert sink.records[0]["exit_exception"] is RuntimeError


@pytest.mark.parametrize(
    (
        "method_name",
        "rank",
        "rank_info",
        "send_event",
        "recv_event",
        "shape_wait_events",
        "payload_wait_events",
        "result_direction",
        "grid_side",
        "requires_grad",
    ),
    (
        (
            "send_forward_recv_backward",
            0,
            RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1, 2]),
            "bridge-send-forward",
            "bridge-recv-backward",
            [
                "bridge-send-forward",
                "bridge-send-forward",
                "bridge-recv-backward",
                "bridge-recv-backward",
            ],
            [
                "bridge-send-forward",
                "bridge-recv-backward",
                "bridge-send-forward",
                "bridge-recv-backward",
            ],
            "backward",
            "src",
            False,
        ),
        (
            "send_backward_recv_forward",
            1,
            RankCommInfo(role=CommRole.RECEIVER, recv_from_ranks=[0, 2]),
            "bridge-send-backward",
            "bridge-recv-forward",
            [
                "bridge-recv-forward",
                "bridge-recv-forward",
                "bridge-send-backward",
                "bridge-send-backward",
            ],
            [
                "bridge-send-backward",
                "bridge-recv-forward",
                "bridge-send-backward",
                "bridge-recv-forward",
            ],
            "forward",
            "dest",
            True,
        ),
    ),
)
def test_bridge_combined_multi_peer_records_shape_payload_waits_and_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    rank: int,
    rank_info: RankCommInfo,
    send_event: str,
    recv_event: str,
    shape_wait_events: list[str],
    payload_wait_events: list[str],
    result_direction: str,
    grid_side: str,
    requires_grad: bool,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(rank)
    bridge.comm_map[rank] = rank_info
    received_shape = (2, 2, 3)
    calls, batches, _ = _install_combined_transport(
        monkeypatch,
        sink=sink,
        send_event=send_event,
        recv_event=recv_event,
        received_shape=received_shape,
    )
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    result = getattr(bridge, method_name)(tensor)

    assert tuple(result.shape) == (2, 4, 3)
    assert result.requires_grad is requires_grad
    assert torch.all(result[:, :2, :] == 2)
    assert torch.all(result[:, 2:, :] == 4)
    assert len(batches) == 2
    assert [call[:2] for call in calls] == [
        ("batch", "shape"),
        *(("wait", "shape") for _ in range(4)),
        ("batch", "payload"),
        *(("wait", "payload") for _ in range(4)),
        ("broadcast", "shape"),
        ("broadcast", "payload"),
    ]

    record_names = [record["name"] for record in sink.records]
    assert record_names == [
        "bridge-p2p-launch",
        *shape_wait_events,
        "bridge-p2p-launch",
        *payload_wait_events,
        "bridge-grid-broadcast",
        "bridge-grid-broadcast",
    ]
    shape_launch = sink.records[0]
    shape_waits = sink.records[1:5]
    payload_launch = sink.records[5]
    payload_waits = sink.records[6:10]
    broadcasts = sink.records[10:]

    for launch, waits, message_kind, batch_ops in (
        (shape_launch, shape_waits, "shape", batches[0]),
        (payload_launch, payload_waits, "payload", batches[1]),
    ):
        launch_context = launch["ctx"]
        operations = launch_context["operations"]
        assert launch_context["message_kind"] == message_kind
        assert launch_context["operation_count"] == 4
        assert launch_context["request_pairing"] == "position"
        assert launch_context["transport_api"] == "batch_isend_irecv"
        assert launch_context["completion_included"] is False
        assert len({operation["operation_id"] for operation in operations}) == 4
        for operation, wait, p2p_op in zip(operations, waits, batch_ops):
            wait_context = wait["ctx"]
            expected_direction = (
                "send" if p2p_op.op is torch.distributed.isend else "recv"
            )
            expected_pipeline_direction = (
                "forward" if wait["name"].endswith("forward") else "backward"
            )
            assert operation["message_kind"] == message_kind
            assert operation["direction"] == expected_direction
            assert operation["pipeline_direction"] == expected_pipeline_direction
            assert operation["peer_rank"] == p2p_op.peer
            assert operation["data_bytes"] == (
                p2p_op.tensor.numel() * p2p_op.tensor.element_size()
            )
            assert wait_context["operation_id"] == operation["operation_id"]
            assert wait_context["request_id"] == operation["request_id"]
            assert wait_context["batch_id"] == launch_context["batch_id"]
            assert operation["operation_id"].startswith(
                f"{launch_context['batch_id']}:"
            )
            assert wait_context["operation_id_scope"] == "rank_local"
            assert wait_context["request_pairing"] == "position"
            assert wait_context["completion_kind"] == "work_wait"
            assert wait_context["completion_guarantee"] == "current_stream_after_wait"
            assert wait_context["timing_phase"] == "stream_dependency"
            assert wait["values"] == {"completed": True}
            if message_kind == "shape":
                assert operation["data_bytes"] == bridge.tensor_ndim * 8
                assert operation["shape_of"] == (
                    "activation"
                    if expected_pipeline_direction == "forward"
                    else "gradient"
                )
                assert "payload_role" not in operation
                assert "payload_role" not in wait_context
            else:
                assert operation["data_bytes"] == 2 * 2 * 3 * 4
                assert operation["payload_role"] == (
                    "activation"
                    if expected_pipeline_direction == "forward"
                    else "gradient"
                )
                assert "shape_of" not in operation
                assert "shape_of" not in wait_context

    shape_broadcast, payload_broadcast = broadcasts
    assert shape_broadcast["ctx"]["message_kind"] == "shape"
    assert shape_broadcast["ctx"]["shape_of"] == (
        "activation" if result_direction == "forward" else "gradient"
    )
    assert shape_broadcast["ctx"]["data_bytes"] == bridge.tensor_ndim * 8
    assert payload_broadcast["ctx"]["message_kind"] == "payload"
    assert payload_broadcast["ctx"]["payload_role"] == (
        "activation" if result_direction == "forward" else "gradient"
    )
    assert payload_broadcast["ctx"]["data_bytes"] == result.numel() * result.element_size()
    assert all(
        record["ctx"]["collective_role"] == "source" for record in broadcasts
    )
    assert all(record["ctx"]["grid_side"] == grid_side for record in broadcasts)
    assert all(
        record["ctx"]["pipeline_direction"] == result_direction
        for record in broadcasts
    )
    expected_group = (
        bridge.src_grid_broadcast_pg
        if grid_side == "src"
        else bridge.dest_grid_broadcast_pg
    )
    broadcast_calls = [call for call in calls if call[0] == "broadcast"]
    assert all(call[2] == rank for call in broadcast_calls)
    assert all(call[3] is expected_group for call in broadcast_calls)
    assert all(record["ctx"]["source_rank"] == rank for record in broadcasts)
    assert all(record["ctx"]["backend"] == "nccl" for record in broadcasts)
    assert all(record["values"] == {"completed": True} for record in broadcasts)


@pytest.mark.parametrize(
    (
        "method_name",
        "rank",
        "rank_info",
        "pipeline_direction",
        "grid_side",
        "requires_grad",
    ),
    (
        (
            "send_forward_recv_backward",
            1,
            RankCommInfo(role=CommRole.MEMBER),
            "backward",
            "src",
            False,
        ),
        (
            "send_backward_recv_forward",
            3,
            RankCommInfo(role=CommRole.MEMBER),
            "forward",
            "dest",
            True,
        ),
    ),
)
def test_bridge_combined_member_records_only_grid_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    rank: int,
    rank_info: RankCommInfo,
    pipeline_direction: str,
    grid_side: str,
    requires_grad: bool,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(rank)
    if grid_side == "src":
        bridge.src_grid = SimpleNamespace(rank_offset=0, size=2)
        bridge.src_local_leader_rank = 0
        bridge.src_grid_broadcast_ranks = [0, 1]
        expected_source_rank = bridge.src_local_leader_rank
        expected_group = bridge.src_grid_broadcast_pg
    else:
        bridge.dest_grid = SimpleNamespace(rank_offset=2, size=2)
        bridge.dest_local_leader_rank = 2
        bridge.dest_grid_broadcast_ranks = [2, 3]
        expected_source_rank = bridge.dest_local_leader_rank
        expected_group = bridge.dest_grid_broadcast_pg
    bridge.comm_map[rank] = rank_info
    received_shape = (2, 4, 3)
    broadcast_calls = []

    def fail_batch(ops):
        del ops
        pytest.fail("Bridge MEMBER executed cross-grid P2P")

    def fake_broadcast(tensor: torch.Tensor, *, src: int, group: Any):
        assert sink.active == ["bridge-grid-broadcast"]
        broadcast_calls.append((tensor, src, group))
        with torch.no_grad():
            if tensor.dtype == torch.int64:
                tensor.copy_(torch.tensor(received_shape, dtype=torch.int64))
            else:
                tensor.fill_(9)

    monkeypatch.setattr(bridge_module.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(bridge_module.torch.distributed, "batch_isend_irecv", fail_batch)
    monkeypatch.setattr(bridge_module.dist, "broadcast", fake_broadcast)

    result = getattr(bridge, method_name)(torch.ones(2, 4, 3))

    assert tuple(result.shape) == received_shape
    assert result.requires_grad is requires_grad
    assert torch.all(result == 9)
    assert len(broadcast_calls) == 2
    assert all(call[1] == expected_source_rank for call in broadcast_calls)
    assert all(call[2] is expected_group for call in broadcast_calls)
    assert [record["name"] for record in sink.records] == [
        "bridge-grid-broadcast",
        "bridge-grid-broadcast",
    ]
    shape_record, payload_record = sink.records
    assert shape_record["ctx"]["message_kind"] == "shape"
    assert payload_record["ctx"]["message_kind"] == "payload"
    assert all(
        record["ctx"]["collective_role"] == "participant"
        for record in sink.records
    )
    assert all(record["ctx"]["grid_side"] == grid_side for record in sink.records)
    assert all(
        record["ctx"]["pipeline_direction"] == pipeline_direction
        for record in sink.records
    )
    assert all(
        record["ctx"]["source_rank"] == expected_source_rank
        for record in sink.records
    )


@pytest.mark.parametrize(
    ("method_name", "rank", "rank_info", "send_event", "recv_event"),
    (
        (
            "send_forward_recv_backward",
            0,
            RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1, 2]),
            "bridge-send-forward",
            "bridge-recv-backward",
        ),
        (
            "send_backward_recv_forward",
            1,
            RankCommInfo(role=CommRole.RECEIVER, recv_from_ranks=[0, 2]),
            "bridge-send-backward",
            "bridge-recv-forward",
        ),
    ),
)
def test_bridge_combined_trace_off_preserves_transport_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    rank: int,
    rank_info: RankCommInfo,
    send_event: str,
    recv_event: str,
) -> None:
    bridge = _make_bridge(rank)
    bridge.comm_map[rank] = rank_info
    calls, batches, _ = _install_combined_transport(
        monkeypatch,
        sink=None,
        send_event=send_event,
        recv_event=recv_event,
        received_shape=(2, 2, 3),
    )
    monkeypatch.setattr(
        bridge_module,
        "_build_bridge_batch_operations",
        lambda *args, **kwargs: pytest.fail("trace-off built Bridge batch metadata"),
    )
    monkeypatch.setattr(
        bridge_module,
        "_bridge_grid_broadcast_context",
        lambda *args, **kwargs: pytest.fail("trace-off built Bridge broadcast metadata"),
    )

    result = getattr(bridge, method_name)(
        torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    )

    assert tuple(result.shape) == (2, 4, 3)
    assert len(batches) == 2
    assert sum(call[0] == "wait" for call in calls) == 8
    assert sum(call[0] == "broadcast" for call in calls) == 2


def test_bridge_combined_non_positional_work_count_preserves_raw_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(0)
    bridge.comm_map[0] = RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1, 2])
    bridge._communicate_shapes = lambda **kwargs: (
        [],
        [(2, 2, 3), (2, 2, 3)],
    )
    wait_order = []

    class _RawWork:
        def __init__(self, index: int) -> None:
            self.index = index

        def wait(self):
            assert sink.active == []
            wait_order.append(self.index)
            return True

    def fake_batch(ops: list[_FakeP2POp]):
        assert sink.active == ["bridge-p2p-launch"]
        assert len(ops) == 4
        return [_RawWork(0), _RawWork(1), _RawWork(2)]

    def fake_broadcast(tensor: torch.Tensor, *, src: int, group: Any):
        del tensor, src, group
        assert sink.active == ["bridge-grid-broadcast"]

    monkeypatch.setattr(bridge_module.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(bridge_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(bridge_module.torch.distributed, "P2POp", _FakeP2POp)
    monkeypatch.setattr(
        bridge_module.torch.distributed,
        "batch_isend_irecv",
        fake_batch,
    )
    monkeypatch.setattr(bridge_module.dist, "broadcast", fake_broadcast)

    result = bridge.send_forward_recv_backward(torch.ones(2, 4, 3))

    assert tuple(result.shape) == (2, 4, 3)
    assert wait_order == [0, 1, 2]
    assert [record["name"] for record in sink.records] == [
        "bridge-p2p-launch",
        "bridge-grid-broadcast",
        "bridge-grid-broadcast",
    ]


def test_bridge_combined_payload_launch_exception_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(0)
    bridge.comm_map[0] = RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1, 2])
    calls, _, errors = _install_combined_transport(
        monkeypatch,
        sink=sink,
        send_event="bridge-send-forward",
        recv_event="bridge-recv-backward",
        received_shape=(2, 2, 3),
        launch_error_kind="payload",
    )

    with pytest.raises(RuntimeError) as raised:
        bridge.send_forward_recv_backward(torch.ones(2, 4, 3))

    assert raised.value is errors["launch"]
    assert [call[:2] for call in calls] == [
        ("batch", "shape"),
        *(("wait", "shape") for _ in range(4)),
        ("batch", "payload"),
    ]
    assert sink.records[-1]["name"] == "bridge-p2p-launch"
    assert sink.records[-1]["ctx"]["message_kind"] == "payload"
    assert sink.records[-1]["exit_exception"] is RuntimeError


def test_bridge_combined_payload_wait_exception_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(0)
    bridge.comm_map[0] = RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1, 2])
    calls, _, errors = _install_combined_transport(
        monkeypatch,
        sink=sink,
        send_event="bridge-send-forward",
        recv_event="bridge-recv-backward",
        received_shape=(2, 2, 3),
        wait_error_at=("payload", 1),
    )

    with pytest.raises(RuntimeError) as raised:
        bridge.send_forward_recv_backward(torch.ones(2, 4, 3))

    assert raised.value is errors["wait"]
    assert [call[:2] for call in calls] == [
        ("batch", "shape"),
        *(("wait", "shape") for _ in range(4)),
        ("batch", "payload"),
        ("wait", "payload"),
        ("wait", "payload"),
    ]
    failed_wait = sink.records[-1]
    assert failed_wait["name"] == "bridge-recv-backward"
    assert failed_wait["values"] == {
        "completed": False,
        "error_type": "RuntimeError",
    }
    assert failed_wait["exit_exception"] is RuntimeError


def test_bridge_combined_broadcast_exception_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    bridge = _make_bridge(0)
    bridge.comm_map[0] = RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1, 2])
    calls, _, errors = _install_combined_transport(
        monkeypatch,
        sink=sink,
        send_event="bridge-send-forward",
        recv_event="bridge-recv-backward",
        received_shape=(2, 2, 3),
        broadcast_error_kind="payload",
    )

    with pytest.raises(RuntimeError) as raised:
        bridge.send_forward_recv_backward(torch.ones(2, 4, 3))

    assert raised.value is errors["broadcast"]
    assert [call[:2] for call in calls][-2:] == [
        ("broadcast", "shape"),
        ("broadcast", "payload"),
    ]
    failed_broadcast = sink.records[-1]
    assert failed_broadcast["name"] == "bridge-grid-broadcast"
    assert failed_broadcast["ctx"]["message_kind"] == "payload"
    assert failed_broadcast["values"] == {
        "completed": False,
        "error_type": "RuntimeError",
    }
    assert failed_broadcast["exit_exception"] is RuntimeError
