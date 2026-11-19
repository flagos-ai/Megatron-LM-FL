# Copyright (c) 2026, BAAI. All rights reserved.
"""Hetero p2p wrappers: timer bookkeeping, list-of-shapes returns, inter-mesh slicing and dp-coef scaling."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as core_parallel_state
from megatron.plugin.hetero import p2p_communication, parallel_context

MESHES = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]


class _FakeGroup:
    def __init__(self, ranks, name):
        self.ranks = list(ranks or [])
        self._name = name

    def name(self):
        return self._name


class _RecordingTimers:
    def __init__(self):
        self.events = []

    def __call__(self, name, log_level=None):
        self.events.append(("create", name, log_level))
        return self

    def start(self):
        self.events.append(("start",))

    def stop(self):
        self.events.append(("stop",))


def _install_fake_distributed(monkeypatch, rank, world_size):
    def new_group(ranks=None, timeout=None, backend=None, pg_options=None, **kwargs):
        return _FakeGroup(ranks, kwargs.get("group_desc") or backend or "group")

    def all_gather_object(output, current):
        order = sorted(range(world_size), key=lambda r: (0 if r % 2 == 0 else 1, r))
        if isinstance(current, dict) and "process_mesh_idx" in current:
            mesh_worlds = [tp * cp * dp * pp for tp, cp, ep, dp, pp in MESHES]
            gathered = []
            for physical in range(world_size):
                logical = order.index(physical)
                mesh_idx, accumulated = 0, 0
                for index, size in enumerate(mesh_worlds):
                    if logical < accumulated + size:
                        mesh_idx = index
                        break
                    accumulated += size
                gathered.append({"rank": physical, "process_mesh_idx": mesh_idx})
            output[:] = gathered
        else:
            output[:] = [
                {"rank": physical, "device_type": "cuda" if physical % 2 == 0 else "musa"}
                for physical in range(world_size)
            ]

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda group=None: group.ranks.index(rank) if hasattr(group, "ranks") else rank,
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: len(group.ranks) if hasattr(group, "ranks") else world_size,
    )
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: list(group.ranks)
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(core_parallel_state, "_global_process_group_list", None)


def _make_context(monkeypatch, rank):
    _install_fake_distributed(monkeypatch, rank, 6)
    args = SimpleNamespace(
        hetero_process_meshes=MESHES,
        expert_tensor_parallel_size_per_process_mesh=None,
        expert_tensor_parallel_size=None,
        nccl_communicator_config_path=None,
        distributed_timeout_minutes=1,
        use_tp_pp_dp_mapping=False,
        distributed_backend="gloo",
        use_gloo_process_groups=True,
        hetero_device_types=["cuda", "musa"],
        hetero_current_device_type="cuda" if rank % 2 == 0 else "musa",
        calculate_per_token_loss=False,
    )
    context = parallel_context.ParallelContext.__new__(parallel_context.ParallelContext)
    context._is_initialized = False
    context._rank = rank
    context._current_process_mesh_index = 0
    context._process_meshes = []
    context._rank_to_process_mesh = {}
    context._inter_mesh_group_ranks = defaultdict(list)
    context._inter_mesh_process_groups_pp = {}
    context._inter_mesh_process_groups_dp = {}
    context._inter_mesh_process_groups_edp = {}
    context._inter_mesh_tensor_slices = {}
    context._inter_mesh_tensor_slices_for_embd_group = {}
    context._global_group_ranks = defaultdict(list)
    context._global_all_group_ranks = defaultdict(list)
    context._global_process_groups = defaultdict(list)
    context._global_process_group_to_ranks = {}
    context._global_parallel_world_sizes = {}
    context._global_parallel_ranks = {}
    context._timeout = None
    context._rank_mapper = parallel_context.RankMapper(args)
    context._args = args
    context.build_all_process_meshes()
    context.build_all_inter_mesh_process_groups()
    context.build_global_process_groups()
    context._is_initialized = True
    monkeypatch.setattr(parallel_context, "_GLOBAL_PARALLEL_CONTEXT", context)
    monkeypatch.setattr(
        p2p_communication,
        "cur_platform",
        SimpleNamespace(current_device=lambda: torch.device("cpu")),
    )
    return context


def _communicate_recording(sends, value):
    def communicate(**kwargs):
        shape = kwargs.get("tensor_shape") or [1]
        sends.append(
            (
                kwargs.get("tensor_send_next"),
                kwargs.get("tensor_send_prev"),
                tuple(kwargs.get("tensor_shape") or ()),
                kwargs.get("group"),
            )
        )
        return (
            torch.full(shape, value) if kwargs.get("recv_prev") else None,
            torch.full(shape, value) if kwargs.get("recv_next") else None,
            None,
        )

    return communicate


@pytest.mark.parametrize(
    "timer_name, call",
    [
        pytest.param(
            "forward-recv",
            lambda config, comm: p2p_communication.recv_forward_hetero(
                torch.Size([2, 3, 4]), False, config, comm
            ),
            id="recv-forward",
        ),
        pytest.param(
            "backward-recv",
            lambda config, comm: p2p_communication.recv_backward_hetero(
                torch.Size([2, 3, 4]), False, config, comm
            ),
            id="recv-backward",
        ),
        pytest.param(
            "forward-send",
            lambda config, comm: p2p_communication.send_forward_hetero(
                torch.ones(2, 3, 4), False, config, comm
            ),
            id="send-forward",
        ),
        pytest.param(
            "backward-send",
            lambda config, comm: p2p_communication.send_backward_hetero(
                torch.ones(2, 3, 4), False, config, comm
            ),
            id="send-backward",
        ),
        pytest.param(
            "forward-send-backward-recv",
            lambda config, comm: p2p_communication.send_forward_recv_backward_hetero(
                torch.ones(2, 3, 4), torch.Size([2, 3, 4]), False, config, comm
            ),
            id="send-forward-recv-backward",
        ),
        pytest.param(
            "backward-send-forward-recv",
            lambda config, comm: p2p_communication.send_backward_recv_forward_hetero(
                torch.ones(2, 3, 4), torch.Size([2, 3, 4]), False, config, comm
            ),
            id="send-backward-recv-forward",
        ),
    ],
)
def test_wrapper_starts_and_stops_timer_when_timers_enabled(timer_name, call, monkeypatch):
    _make_context(monkeypatch, rank=2)
    timers = _RecordingTimers()
    config = SimpleNamespace(timers=timers, pipeline_dtype=torch.float32)
    call(config, _communicate_recording([], 5.0))
    assert timers.events == [
        ("create", timer_name, 2),
        ("start",),
        ("create", timer_name, None),
        ("stop",),
    ]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda config: p2p_communication.recv_forward_hetero([[2, 3], [4, 5]], True, config),
            id="recv-forward",
        ),
        pytest.param(
            lambda config: p2p_communication.recv_backward_hetero([[2, 3], [4, 5]], True, config),
            id="recv-backward",
        ),
    ],
)
def test_recv_wrapper_returns_none_per_shape_when_boundary_stage(call):
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    assert call(config) == [None, None]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda config: p2p_communication.send_forward_recv_backward_hetero(
                [torch.ones(2, 3), torch.ones(2, 3)], [[2, 3], [2, 3]], True, config
            ),
            id="send-forward-recv-backward",
        ),
        pytest.param(
            lambda config: p2p_communication.send_backward_recv_forward_hetero(
                [torch.ones(2, 3), torch.ones(2, 3)], [[2, 3], [2, 3]], True, config
            ),
            id="send-backward-recv-forward",
        ),
    ],
)
def test_combined_wrapper_returns_none_per_shape_when_boundary_stage(call):
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    assert call(config) == [None, None]


def test_recv_forward_uses_local_pipeline_group_within_one_mesh(monkeypatch):
    context = _make_context(monkeypatch, rank=2)
    sends = []
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    received = p2p_communication.recv_forward_hetero(
        [2, 3], False, config, _communicate_recording(sends, 5.0)
    )
    local_pp = context._process_meshes[0].get_process_group("pp")
    assert torch.equal(received, torch.full((2, 3), 5.0))
    assert sends == [(None, None, (2, 3), local_pp)]


def test_recv_forward_assembles_inter_mesh_slices_from_previous_mesh(monkeypatch):
    _make_context(monkeypatch, rank=4)
    sends = []
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    received = p2p_communication.recv_forward_hetero(
        torch.Size([4, 1, 8]), False, config, _communicate_recording(sends, 3.0)
    )
    assert torch.equal(received, torch.full((4, 1, 8), 3.0))
    assert [entry[2] for entry in sends] == [(4, 1, 8)]


def test_recv_backward_assembles_inter_mesh_slices_without_scaling_when_dp_coef_is_one(monkeypatch):
    _make_context(monkeypatch, rank=2)
    sends = []
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    backward = p2p_communication.recv_backward_hetero(
        torch.Size([4, 1, 8]), False, config, _communicate_recording(sends, 3.0)
    )
    assert torch.equal(backward, torch.full((4, 1, 8), 3.0))
    assert [entry[2] for entry in sends] == [(4, 1, 8)]


def test_send_forward_slices_payload_across_inter_mesh_boundary(monkeypatch):
    _make_context(monkeypatch, rank=2)
    sends = []
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    payload = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8)
    p2p_communication.send_forward_hetero(
        payload, False, config, _communicate_recording(sends, 0.0)
    )
    assert len(sends) == 1
    sent, _, _, _ = sends[0]
    assert torch.equal(sent, payload)


def test_send_backward_uses_local_pipeline_group_within_one_mesh(monkeypatch):
    context = _make_context(monkeypatch, rank=2)
    sends = []
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    payload = torch.ones(2, 3)
    p2p_communication.send_backward_hetero(
        payload, False, config, _communicate_recording(sends, 0.0)
    )
    local_pp = context._process_meshes[0].get_process_group("pp")
    assert sends == [(None, payload, (), local_pp)]


def test_send_forward_recv_backward_exchanges_inter_mesh_slices(monkeypatch):
    _make_context(monkeypatch, rank=2)
    sends = []
    config = SimpleNamespace(timers=None, pipeline_dtype=torch.float32)
    payload = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8)
    grad = p2p_communication.send_forward_recv_backward_hetero(
        payload, payload.shape, False, config, _communicate_recording(sends, 3.0)
    )
    assert torch.equal(grad, torch.full((4, 1, 8), 3.0))
    assert [entry[2] for entry in sends] == [(4, 1, 8)]
    sent = sends[0][0]
    assert torch.equal(sent, payload)
