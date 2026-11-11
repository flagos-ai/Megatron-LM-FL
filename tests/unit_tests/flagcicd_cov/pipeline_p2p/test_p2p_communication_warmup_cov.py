# Copyright (c) 2026, BAAI. All rights reserved.
"""P2P comm-group warm-up (plain and hetero) and boundary-stage return values of the p2p calls."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import get_pipeline_model_parallel_group
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from tests.unit_tests.test_utilities import Utils


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


def test_warm_up_comm_group_restores_variable_seq_lengths_after_warming(model_parallel):
    model_parallel(pipeline_model_parallel_size=4, tensor_model_parallel_size=2)
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.float,
        variable_seq_lengths=True,
        batch_p2p_comm=False,
    )
    comm = P2PCommunicator(get_pipeline_model_parallel_group(), config)

    assert comm.warm_up_comm_group() is None
    assert comm.config.variable_seq_lengths is True

    group = comm.pp_group
    stage = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.full((4, 1, 2), (stage + 1) * 11.0, device=device)
    received, _, _ = comm._communicate(
        tensor_send_next=payload,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=(4, 1, 2),
    )
    torch.testing.assert_close(
        received,
        torch.full((4, 1, 2), ((stage - 1) % 4 + 1) * 11.0, device=device),
        rtol=1e-5,
        atol=1e-5,
    )


def test_warm_up_comm_group_hetero_warms_every_pipeline_group_the_rank_joins(model_parallel):
    model_parallel(pipeline_model_parallel_size=1, tensor_model_parallel_size=1)
    groups = [
        dist.new_group(ranks=members, backend="nccl") for members in ([0, 1, 2, 3], [4, 5, 6, 7])
    ]
    config = ModelParallelConfig(
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.float,
        batch_p2p_comm=False,
        enable_hetero="on",
    )
    comm = P2PCommunicator(groups, config)

    assert comm.warm_up_comm_group() is None

    rank = dist.get_rank()
    own = next(pg for pg in groups if rank in dist.get_process_group_ranks(pg))
    stage = own.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.full((1,), (stage + 1) * 7.0, device=device)
    received, _, _ = comm._communicate(
        tensor_send_next=payload,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=(1,),
        group=own,
    )
    torch.testing.assert_close(
        received, torch.full((1,), ((stage - 1) % 4 + 1) * 7.0, device=device), rtol=1e-5, atol=1e-5
    )

    dist.barrier()
    for pg in groups:
        dist.destroy_process_group(pg)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        pytest.param(
            "recv_forward",
            {"tensor_shapes": (2, 3, 4), "is_first_stage": True},
            None,
            id="recv_forward-single-shape",
        ),
        pytest.param(
            "recv_forward",
            {"tensor_shapes": [(2, 3, 4), (5, 6, 7)], "is_first_stage": True},
            [None, None],
            id="recv_forward-shape-list",
        ),
        pytest.param(
            "recv_backward",
            {"tensor_shapes": (2, 3, 4), "is_last_stage": True},
            None,
            id="recv_backward-single-shape",
        ),
        pytest.param(
            "recv_backward",
            {"tensor_shapes": [(2, 3, 4), (5, 6, 7)], "is_last_stage": True},
            [None, None],
            id="recv_backward-shape-list",
        ),
        pytest.param(
            "send_forward",
            {"output_tensors": torch.empty(2, 3, 4), "is_last_stage": True},
            None,
            id="send_forward-single-tensor",
        ),
        pytest.param(
            "send_backward",
            {"input_tensor_grads": torch.empty(2, 3, 4), "is_first_stage": True},
            None,
            id="send_backward-single-tensor",
        ),
        pytest.param(
            "send_forward_recv_backward",
            {
                "output_tensors": torch.empty(2, 3, 4),
                "tensor_shapes": (2, 3, 4),
                "is_last_stage": True,
            },
            None,
            id="send_forward_recv_backward-single",
        ),
        pytest.param(
            "send_forward_recv_backward",
            {
                "output_tensors": [torch.empty(2, 3, 4), torch.empty(5, 6, 7)],
                "tensor_shapes": [(2, 3, 4), (5, 6, 7)],
                "is_last_stage": True,
            },
            [None, None],
            id="send_forward_recv_backward-list",
        ),
        pytest.param(
            "send_backward_recv_forward",
            {
                "input_tensor_grads": torch.empty(2, 3, 4),
                "tensor_shapes": (2, 3, 4),
                "is_first_stage": True,
            },
            None,
            id="send_backward_recv_forward-single",
        ),
        pytest.param(
            "send_backward_recv_forward",
            {
                "input_tensor_grads": [torch.empty(2, 3, 4), torch.empty(5, 6, 7)],
                "tensor_shapes": [(2, 3, 4), (5, 6, 7)],
                "is_first_stage": True,
            },
            [None, None],
            id="send_backward_recv_forward-list",
        ),
    ],
)
def test_boundary_stage_p2p_methods_dispatch_to_hetero_helpers_when_enabled(
    method, kwargs, expected, model_parallel
):
    comm = _pp2_communicator(model_parallel, enable_hetero="on")
    assert getattr(comm, method)(**kwargs) == expected


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        pytest.param(
            "recv_backward",
            {"tensor_shapes": (2, 3, 4), "is_last_stage": True},
            None,
            id="recv_backward-single-shape",
        ),
        pytest.param(
            "recv_backward",
            {"tensor_shapes": [(2, 3, 4), (5, 6, 7)], "is_last_stage": True},
            [None, None],
            id="recv_backward-shape-list",
        ),
        pytest.param(
            "send_forward_recv_backward",
            {
                "output_tensors": torch.empty(2, 3, 4),
                "tensor_shapes": (2, 3, 4),
                "is_last_stage": True,
            },
            None,
            id="send_forward_recv_backward-single",
        ),
        pytest.param(
            "send_forward_recv_backward",
            {
                "output_tensors": [torch.empty(2, 3, 4), torch.empty(5, 6, 7)],
                "tensor_shapes": [(2, 3, 4), (5, 6, 7)],
                "is_last_stage": True,
            },
            [None, None],
            id="send_forward_recv_backward-list",
        ),
        pytest.param(
            "send_backward_recv_forward",
            {
                "input_tensor_grads": torch.empty(2, 3, 4),
                "tensor_shapes": (2, 3, 4),
                "is_first_stage": True,
            },
            None,
            id="send_backward_recv_forward-single",
        ),
        pytest.param(
            "send_backward_recv_forward",
            {
                "input_tensor_grads": [torch.empty(2, 3, 4), torch.empty(5, 6, 7)],
                "tensor_shapes": [(2, 3, 4), (5, 6, 7)],
                "is_first_stage": True,
            },
            [None, None],
            id="send_backward_recv_forward-list",
        ),
    ],
)
def test_boundary_stage_p2p_methods_return_none_without_communicating(
    method, kwargs, expected, model_parallel
):
    comm = _pp2_communicator(model_parallel)
    assert getattr(comm, method)(**kwargs) == expected
