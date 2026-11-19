# Copyright (c) 2026, BAAI. All rights reserved.
"""Pipeline p2p exchanges: stage-pair send/recv pairing, dtype and shape guards, overlap handles."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import get_pipeline_model_parallel_group
from megatron.core.pipeline_parallel.p2p_communication import (
    P2PCommunicator,
    _p2p_ops,
    is_single_shape,
)
from megatron.core.timers import Timers
from tests.unit_tests.test_utilities import Utils

SHAPE = (4, 2, 8)


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _pp2_communicator(model_parallel, **overrides):
    model_parallel(pipeline_model_parallel_size=2, tensor_model_parallel_size=4)
    fields = {
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.float,
        "batch_p2p_comm": False,
    }
    fields.update(overrides)
    config = ModelParallelConfig(**fields)
    return P2PCommunicator(get_pipeline_model_parallel_group(), config)


def _pp4_communicator(model_parallel, **overrides):
    model_parallel(pipeline_model_parallel_size=4, tensor_model_parallel_size=2)
    fields = {
        "pipeline_model_parallel_size": 4,
        "pipeline_dtype": torch.float,
        "batch_p2p_comm": False,
    }
    fields.update(overrides)
    config = ModelParallelConfig(**fields)
    return P2PCommunicator(get_pipeline_model_parallel_group(), config)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.Size([4, 2, 8]), True),
        ((4, 2, 8), True),
        ([4, 2, 8], True),
        ([4, "micro", 8], False),
        ([], False),
        (None, False),
        ([(4, 2, 8)], False),
    ],
    ids=[
        "torch-size",
        "int-tuple",
        "int-list",
        "non-int-list",
        "empty-list",
        "none",
        "nested-list",
    ],
)
def test_is_single_shape_accepts_only_flat_integer_shapes(value, expected):
    assert is_single_shape(value) is expected


def test_p2p_ops_splits_two_rank_pipelines_between_pp_group_and_world(model_parallel):
    model_parallel(pipeline_model_parallel_size=2, tensor_model_parallel_size=4)
    group = get_pipeline_model_parallel_group()
    rank = dist.get_rank()
    peer = dist.get_global_rank(group, (group.rank() + 1) % 2)
    device = torch.device("cuda", torch.cuda.current_device())
    if group.rank() == 0:
        forward_send = torch.full((4,), 10.0 * rank + 1.0, device=device)
        recv_next = torch.empty(4, device=device)
        reqs = _p2p_ops(
            tensor_send_prev=None,
            tensor_recv_prev=None,
            tensor_send_next=forward_send,
            tensor_recv_next=recv_next,
            group=group,
            prev_pipeline_rank=peer,
            next_pipeline_rank=peer,
        )
        assert set(reqs) == {"send_next", "recv_next"}
        for req in reqs.values():
            req.wait()
        torch.testing.assert_close(
            recv_next, torch.full((4,), 10.0 * peer + 2.0, device=device), rtol=1e-5, atol=1e-5
        )
    else:
        backward_send = torch.full((4,), 10.0 * rank + 2.0, device=device)
        recv_prev = torch.empty(4, device=device)
        reqs = _p2p_ops(
            tensor_send_prev=backward_send,
            tensor_recv_prev=recv_prev,
            tensor_send_next=None,
            tensor_recv_next=None,
            group=group,
            prev_pipeline_rank=peer,
            next_pipeline_rank=peer,
        )
        assert set(reqs) == {"send_prev", "recv_prev"}
        for req in reqs.values():
            req.wait()
        torch.testing.assert_close(
            recv_prev, torch.full((4,), 10.0 * peer + 1.0, device=device), rtol=1e-5, atol=1e-5
        )


def test_p2p_ops_rings_when_pipeline_group_spans_four_ranks(model_parallel):
    model_parallel(pipeline_model_parallel_size=4, tensor_model_parallel_size=2)
    group = get_pipeline_model_parallel_group()
    stage = group.rank()
    next_rank = dist.get_global_rank(group, (stage + 1) % 4)
    prev_rank = dist.get_global_rank(group, (stage - 1) % 4)
    device = torch.device("cuda", torch.cuda.current_device())

    send_next = torch.full((3,), 10.0 * stage + 1.0, device=device)
    send_prev = torch.full((3,), 10.0 * stage + 2.0, device=device)
    recv_prev = torch.empty(3, device=device)
    recv_next = torch.empty(3, device=device)

    reqs = _p2p_ops(
        tensor_send_prev=send_prev,
        tensor_recv_prev=recv_prev,
        tensor_send_next=send_next,
        tensor_recv_next=recv_next,
        group=group,
        prev_pipeline_rank=prev_rank,
        next_pipeline_rank=next_rank,
    )

    assert set(reqs) == {"send_next", "recv_prev", "send_prev", "recv_next"}
    for req in reqs.values():
        req.wait()

    torch.testing.assert_close(
        recv_prev,
        torch.full((3,), 10.0 * ((stage - 1) % 4) + 1.0, device=device),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        recv_next,
        torch.full((3,), 10.0 * ((stage + 1) % 4) + 2.0, device=device),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("recv_prev", "recv_next", "dtype", "tensor_shape", "match"),
    [
        (True, False, None, SHAPE, "pipeline_dtype must be provided if recv_prev is True"),
        (True, False, torch.float, None, "tensor_shape must be specified if recv_prev is True"),
        (False, True, None, SHAPE, "dtype must be provided if recv_next is True"),
        (False, True, torch.float, None, "tensor_shape must be specified if recv_next is True"),
    ],
    ids=[
        "recv-prev-without-dtype",
        "recv-prev-without-shape",
        "recv-next-without-dtype",
        "recv-next-without-shape",
    ],
)
def test_communicate_rejects_recv_requests_without_dtype_or_tensor_shape(
    recv_prev, recv_next, dtype, tensor_shape, match, model_parallel
):
    model_parallel(pipeline_model_parallel_size=2, tensor_model_parallel_size=4)
    config = ModelParallelConfig(pipeline_model_parallel_size=1, pipeline_dtype=dtype)
    comm = P2PCommunicator(get_pipeline_model_parallel_group(), config)

    with pytest.raises(RuntimeError, match=match):
        comm._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )


def test_stage_pairs_exchange_forward_and_backward_tensors_with_timers(model_parallel):
    comm = _pp2_communicator(model_parallel, timers=Timers(2, "max"))
    group = comm.pp_group
    peer = dist.get_global_rank(group, (group.rank() + 1) % 2)
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())

    if group.rank() == 0:
        comm.send_forward(torch.full(SHAPE, 17.0 + rank, device=device), is_last_stage=False)
        received_grad = comm.recv_backward(SHAPE, is_last_stage=False)
        torch.testing.assert_close(
            received_grad, torch.full(SHAPE, 43.0 + peer, device=device), rtol=1e-5, atol=1e-5
        )
        received_grads = comm.send_forward_recv_backward(
            [torch.full(SHAPE, 19.0 + rank, device=device)], [SHAPE], is_last_stage=False
        )
        assert isinstance(received_grads, list)
        torch.testing.assert_close(
            received_grads[0], torch.full(SHAPE, 47.0 + peer, device=device), rtol=1e-5, atol=1e-5
        )
    else:
        received_forward = comm.recv_forward(SHAPE, is_first_stage=False)
        torch.testing.assert_close(
            received_forward, torch.full(SHAPE, 17.0 + peer, device=device), rtol=1e-5, atol=1e-5
        )
        comm.send_backward(torch.full(SHAPE, 43.0 + rank, device=device), is_first_stage=False)
        received_inputs = comm.send_backward_recv_forward(
            [torch.full(SHAPE, 47.0 + rank, device=device)], [SHAPE], is_first_stage=False
        )
        assert isinstance(received_inputs, list)
        torch.testing.assert_close(
            received_inputs[0], torch.full(SHAPE, 19.0 + peer, device=device), rtol=1e-5, atol=1e-5
        )


def test_send_forward_recv_forward_returns_wait_handles_when_p2p_overlap_enabled(model_parallel):
    comm = _pp2_communicator(model_parallel, timers=Timers(2, "max"))
    group = comm.pp_group
    peer = dist.get_global_rank(group, (group.rank() + 1) % 2)
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.full(SHAPE, 5.0 + dist.get_rank(), device=device)

    input_tensor, wait_handles = comm.send_forward_recv_forward(
        payload, True, SHAPE, overlap_p2p_comm=True
    )
    assert set(wait_handles) == {"send_next", "recv_prev"}
    for handle in wait_handles.values():
        handle.wait()
    torch.testing.assert_close(
        input_tensor, torch.full(SHAPE, 5.0 + peer, device=device), rtol=1e-5, atol=1e-5
    )

    blocking = comm.send_forward_recv_forward(payload, True, SHAPE, overlap_p2p_comm=False)
    torch.testing.assert_close(
        blocking, torch.full(SHAPE, 5.0 + peer, device=device), rtol=1e-5, atol=1e-5
    )


def test_send_backward_recv_backward_returns_wait_handles_when_p2p_overlap_enabled(model_parallel):
    comm = _pp4_communicator(model_parallel, timers=Timers(2, "max"))
    group = comm.pp_group
    stage = group.rank()
    next_rank = dist.get_global_rank(group, (stage + 1) % 4)
    device = torch.device("cuda", torch.cuda.current_device())
    grad_payload = torch.full((4, 1, 2), 9.0 + dist.get_rank(), device=device)

    output_grad, wait_handles = comm.send_backward_recv_backward(
        grad_payload, True, (4, 1, 2), overlap_p2p_comm=True
    )
    assert set(wait_handles) == {"send_prev", "recv_next"}
    for handle in wait_handles.values():
        handle.wait()
    torch.testing.assert_close(
        output_grad, torch.full((4, 1, 2), 9.0 + next_rank, device=device), rtol=1e-5, atol=1e-5
    )

    blocking = comm.send_backward_recv_backward(grad_payload, True, (4, 1, 2))
    torch.testing.assert_close(
        blocking, torch.full((4, 1, 2), 9.0 + next_rank, device=device), rtol=1e-5, atol=1e-5
    )


def test_send_forward_backward_recv_forward_backward_exchanges_both_directions(model_parallel):
    comm = _pp4_communicator(model_parallel, timers=Timers(2, "max"))
    group = comm.pp_group
    stage = group.rank()
    next_rank = dist.get_global_rank(group, (stage + 1) % 4)
    prev_rank = dist.get_global_rank(group, (stage - 1) % 4)
    device = torch.device("cuda", torch.cuda.current_device())
    forward_payload = torch.full((2, 3, 4), 23.0 + dist.get_rank(), device=device)
    backward_payload = torch.full((2, 3, 4), 29.0 + dist.get_rank(), device=device)

    input_tensor, output_grad = comm.send_forward_backward_recv_forward_backward(
        forward_payload, backward_payload, True, True, (2, 3, 4)
    )
    torch.testing.assert_close(
        input_tensor, torch.full((2, 3, 4), 23.0 + prev_rank, device=device), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        output_grad, torch.full((2, 3, 4), 29.0 + next_rank, device=device), rtol=1e-5, atol=1e-5
    )


def test_batched_p2p_comm_round_trips_tensors_via_batch_isend_irecv(model_parallel):
    comm = _pp2_communicator(model_parallel, batch_p2p_comm=True, timers=Timers(2, "max"))
    group = comm.pp_group
    peer = dist.get_global_rank(group, (group.rank() + 1) % 2)
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.full(SHAPE, 3.0 + dist.get_rank(), device=device)

    received = comm.send_forward_recv_forward(payload, True, SHAPE)
    torch.testing.assert_close(
        received, torch.full(SHAPE, 3.0 + peer, device=device), rtol=1e-5, atol=1e-5
    )
