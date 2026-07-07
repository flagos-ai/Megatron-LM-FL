from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.hetero import p2p_communication


class _Group:
    def name(self):
        return "nccl"


class _Context:
    def __init__(self):
        self.group = _Group()
        self._current_process_mesh_index = 1
        self._process_meshes = [
            SimpleNamespace(_rank_generator=SimpleNamespace(pp=2)),
            SimpleNamespace(_rank_generator=SimpleNamespace(pp=3)),
        ]

    def get_pipeline_model_parallel_group(self, local_pp_group=False):
        return self.group if local_pp_group else [self.group]


@pytest.fixture
def local_communication(monkeypatch):
    context = _Context()
    monkeypatch.setattr(p2p_communication, "get_parallel_context", lambda: context)
    monkeypatch.setattr(p2p_communication, "is_inter_mesh_comm", lambda **kwargs: False)
    monkeypatch.setattr(p2p_communication.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        p2p_communication,
        "cur_platform",
        SimpleNamespace(current_device=lambda: torch.device("cpu")),
    )
    return context


def test_shape_and_inter_mesh_direction_helpers(monkeypatch):
    assert p2p_communication.is_single_shape(torch.Size([2, 3])) is True
    assert p2p_communication.is_single_shape([2, 3]) is True
    assert p2p_communication.is_single_shape([[2, 3]]) is False
    assert p2p_communication.is_single_shape([]) is False

    context = _Context()
    monkeypatch.setattr(p2p_communication, "get_pipeline_model_parallel_rank", lambda: 2)
    assert p2p_communication.is_inter_mesh_comm(context, comm_with_front_layer=True) is True
    monkeypatch.setattr(p2p_communication, "get_pipeline_model_parallel_rank", lambda: 4)
    assert p2p_communication.is_inter_mesh_comm(context, comm_with_front_layer=False) is True

    with pytest.raises(AssertionError, match="ParallelContext"):
        p2p_communication.is_inter_mesh_comm(None, False)


def test_stage_boundaries_do_not_communicate():
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)

    assert p2p_communication.recv_forward_hetero([2, 3], True, config) is None
    assert p2p_communication.recv_backward_hetero([2, 3], True, config) is None
    assert p2p_communication.send_forward_hetero(torch.ones(1), True, config) is None
    assert p2p_communication.send_backward_hetero(torch.ones(1), True, config) is None
    assert (
        p2p_communication.send_forward_recv_backward_hetero(
            torch.ones(1), [1], True, config
        )
        is None
    )
    assert (
        p2p_communication.send_backward_recv_forward_hetero(
            torch.ones(1), [1], True, config
        )
        is None
    )


def test_local_recv_and_send_wrappers(local_communication):
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    calls = []

    def communicate(**kwargs):
        calls.append(kwargs)
        received = torch.ones(kwargs.get("tensor_shape") or [1])
        return received, received * 2, None

    forward = p2p_communication.recv_forward_hetero([2, 3], False, config, communicate)
    backward = p2p_communication.recv_backward_hetero([2, 3], False, config, communicate)
    assert forward.shape == (2, 3)
    assert torch.equal(backward, torch.full((2, 3), 2.0))

    tensor = torch.ones(2, 3)
    p2p_communication.send_forward_hetero(tensor, False, config, communicate)
    p2p_communication.send_backward_hetero(tensor, False, config, communicate)
    assert calls[-2]["tensor_send_next"] is tensor
    assert calls[-1]["tensor_send_prev"] is tensor


def test_local_combined_send_recv_wrappers(local_communication):
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)

    def communicate(**kwargs):
        received = torch.full(kwargs["tensor_shape"], 3.0)
        return received, received + 1, None

    output_grad = p2p_communication.send_forward_recv_backward_hetero(
        torch.ones(2, 3), torch.Size([2, 3]), False, config, communicate
    )
    input_tensor = p2p_communication.send_backward_recv_forward_hetero(
        torch.ones(2, 3), torch.Size([2, 3]), False, config, communicate
    )

    assert torch.equal(output_grad, torch.full((2, 3), 4.0))
    assert torch.equal(input_tensor, torch.full((2, 3), 3.0))


class _CpuGroup:
    def name(self):
        return "cpu:gloo"


class _InterMeshContext:
    def __init__(self, slices, dp_coef=1.0):
        self.slices = slices
        self.dp_coef = dp_coef
        self.group = _CpuGroup()

    def get_pipeline_model_parallel_group(self, local_pp_group=False):
        return self.group if local_pp_group else [self.group]

    def get_inter_mesh_tensor_slices(self, rank, local_tensor_shape, next=True):
        return self.slices

    def get_dp_coef_when_recv_backward(self):
        return self.dp_coef


def test_inter_mesh_recv_wrappers_stitch_slices_and_scale_backward(monkeypatch):
    monkeypatch.setattr(p2p_communication.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        p2p_communication.torch.distributed,
        "get_process_group_ranks",
        lambda group: [0, 9, 10],
    )
    monkeypatch.setattr(
        p2p_communication,
        "cur_platform",
        SimpleNamespace(current_device=lambda: torch.device("cpu")),
    )
    monkeypatch.setattr(
        p2p_communication,
        "get_parallel_context",
        lambda: _InterMeshContext(
            [(9, (0, 1), (0, 2), 2), (10, (1, 2), (2, 4), 2)], dp_coef=0.5
        ),
    )
    monkeypatch.setattr(p2p_communication, "is_inter_mesh_comm", lambda **kwargs: True)

    def communicate(**kwargs):
        shape = kwargs["tensor_shape"]
        value = 4.0 if kwargs["recv_prev"] else 6.0
        tensor = torch.full(shape, value)
        return (
            tensor if kwargs["recv_prev"] else None,
            tensor if kwargs["recv_next"] else None,
            None,
        )

    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    forward = p2p_communication.recv_forward_hetero(
        (4, 2, 2), is_first_stage=False, config=config, _communicate=communicate
    )
    assert torch.equal(forward[:2, :1, :], torch.full((2, 1, 2), 4.0))
    assert torch.equal(forward[2:4, 1:2, :], torch.full((2, 1, 2), 4.0))

    backward = p2p_communication.recv_backward_hetero(
        (4, 2, 2), is_last_stage=False, config=config, _communicate=communicate
    )
    assert torch.equal(backward[:2, :1, :], torch.full((2, 1, 2), 3.0))
    assert torch.equal(backward[2:4, 1:2, :], torch.full((2, 1, 2), 3.0))


def test_inter_mesh_send_wrappers_slice_payloads(monkeypatch):
    monkeypatch.setattr(p2p_communication.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        p2p_communication.torch.distributed,
        "get_process_group_ranks",
        lambda group: [0, 9, 10],
    )
    monkeypatch.setattr(
        p2p_communication,
        "cur_platform",
        SimpleNamespace(current_device=lambda: torch.device("cpu")),
    )
    monkeypatch.setattr(
        p2p_communication,
        "get_parallel_context",
        lambda: _InterMeshContext(
            [(9, (0, 1), (0, 2), 2), (10, (1, 2), (2, 4), 2)], dp_coef=0.5
        ),
    )
    monkeypatch.setattr(p2p_communication, "is_inter_mesh_comm", lambda **kwargs: True)
    calls = []

    def communicate(**kwargs):
        calls.append(kwargs)
        shape = kwargs["tensor_shape"]
        if kwargs["recv_prev"]:
            return torch.ones(shape), None, None
        if kwargs["recv_next"]:
            return None, torch.ones(shape), None
        return None, None, None

    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    payload = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)

    p2p_communication.send_forward_hetero(
        payload, is_last_stage=False, config=config, _communicate=communicate
    )
    p2p_communication.send_backward_hetero(
        payload, is_first_stage=False, config=config, _communicate=communicate
    )
    grad = p2p_communication.send_forward_recv_backward_hetero(
        payload, payload.shape, is_last_stage=False, config=config, _communicate=communicate
    )
    act = p2p_communication.send_backward_recv_forward_hetero(
        payload, payload.shape, is_first_stage=False, config=config, _communicate=communicate
    )

    sent_next = [call["tensor_send_next"] for call in calls if call["tensor_send_next"] is not None]
    sent_prev = [call["tensor_send_prev"] for call in calls if call["tensor_send_prev"] is not None]
    assert sent_next and sent_prev
    assert all(tensor.shape == (2, 1, 2) for tensor in sent_next + sent_prev)
    assert torch.equal(grad[:2, :1, :], torch.full((2, 1, 2), 0.5))
    assert torch.equal(act[:2, :1, :], torch.ones((2, 1, 2)))
