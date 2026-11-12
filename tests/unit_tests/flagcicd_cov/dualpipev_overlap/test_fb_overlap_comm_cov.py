# Copyright (c) 2026, BAAI. All rights reserved.
"""fb_overlap communication helpers: async all-to-all and pipeline p2p on 8 ranks."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.plugin.dualpipev.fb_overlap.modules import utils
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel(monkeypatch):
    monkeypatch.setattr(utils, 'COMM_STREAM', None)

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _p2p_params(direction_kwargs, pp):
    params = utils.P2PCommParams(**direction_kwargs)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=pp,
        batch_p2p_comm=False,
    )
    params.tensor_shape = (2, 3)
    params.config = config
    params.p2p_communicator = P2PCommunicator(
        pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
    )
    return params


def test_all_to_all_on_one_rank_returns_the_input(model_parallel):
    model_parallel()
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    group = parallel_state.get_tensor_model_parallel_group()
    source = torch.ones(2, device=device)
    result = utils.async_all_to_all(source, None, None, group)
    assert result[0] is source
    assert result[1] is source
    assert result[2] is None


@pytest.mark.parametrize(
    'use_stream', [pytest.param(False, id='default-stream'), pytest.param(True, id='comm-stream')]
)
def test_equal_split_all_to_all_exchanges_one_element_per_rank(model_parallel, use_stream):
    model_parallel()
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    group = torch.distributed.group.WORLD
    source = torch.tensor([100.0 * rank + j for j in range(world)], device=device)
    stream = torch.cuda.current_stream() if use_stream else None
    _, output, handle = utils.async_all_to_all(source, None, None, group, stream=stream)
    handle.wait()
    if use_stream:
        torch.cuda.current_stream().wait_stream(utils.COMM_STREAM)
    assert output.tolist() == [100.0 * j + rank for j in range(world)]
    assert (utils.COMM_STREAM is not None) is use_stream


def test_unequal_split_all_to_all_sizes_output_from_split_lists(model_parallel):
    model_parallel()
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    group = torch.distributed.group.WORLD
    source = torch.cat(
        [torch.full((j + 1,), 100.0 * rank + j, device=device) for j in range(world)]
    )
    _, output, handle = utils.async_all_to_all(
        source, [rank + 1] * world, [j + 1 for j in range(world)], group
    )
    handle.wait()
    expected = torch.cat(
        [torch.full((rank + 1,), 100.0 * j + rank, device=device) for j in range(world)]
    )
    assert torch.equal(output, expected)
    assert tuple(output.shape) == (world * (rank + 1),)


def test_p2p_helper_forwards_to_the_next_stage(model_parallel):
    model_parallel(pipeline_model_parallel_size=4)
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    params = _p2p_params({'send_next': pp_rank != 3, 'recv_prev': pp_rank != 0}, 4)
    received, handles = utils.p2p_comm_helper(
        params, torch.full((2, 3), float(pp_rank), device=device)
    )
    for handle in handles.values():
        handle.wait()
    if pp_rank == 0:
        assert received is None
    else:
        assert torch.equal(received, torch.full((2, 3), float(pp_rank - 1), device=device))
    if pp_rank == 0:
        assert sorted(handles) == ['send_next']
    elif pp_rank == 3:
        assert sorted(handles) == ['recv_prev']
    else:
        assert sorted(handles) == ['recv_prev', 'send_next']


def test_p2p_helper_sends_gradients_to_the_previous_stage(model_parallel):
    model_parallel(pipeline_model_parallel_size=4)
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    params = _p2p_params({'send_prev': pp_rank != 0, 'recv_next': pp_rank != 3}, 4)
    received, handles = utils.p2p_comm_helper(
        params, torch.full((2, 3), float(pp_rank), device=device)
    )
    for handle in handles.values():
        handle.wait()
    if pp_rank == 3:
        assert received is None
    else:
        assert torch.equal(received, torch.full((2, 3), float(pp_rank + 1), device=device))
    if pp_rank == 0:
        assert sorted(handles) == ['recv_next']
    elif pp_rank == 3:
        assert sorted(handles) == ['send_prev']
    else:
        assert sorted(handles) == ['recv_next', 'send_prev']
