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
def test_bridge_blocking_recv_excludes_shape_and_broadcast_from_payload_scope(
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
        assert sink.active == []
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
    assert len(sink.records) == 1
    _assert_payload_context(
        sink.records[0],
        event_name=event_name,
        direction="recv",
        pipeline_direction=pipeline_direction,
        peer_rank=peer_rank,
        data_bytes=result.numel() * result.element_size(),
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
