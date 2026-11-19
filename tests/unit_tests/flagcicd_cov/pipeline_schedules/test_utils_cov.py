# Copyright (c) 2026, BAAI. All rights reserved.
"""Pipeline stage rank helpers, DualPipeV predicates, and schedule-node plumbing."""

import os

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.utils as pp_utils
from megatron.core.pipeline_parallel.utils import ScheduleNode, get_comp_stream
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils

cur_platform = get_platform()


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'dualpipev_stage,dualpipev_size,expected',
    [(None, None, True), (0, None, True), (0, 1, True), (0, 4, True), (1, 4, False), (3, 4, False)],
    ids=[
        'no-stage-no-size',
        'zero-stage-no-size',
        'single-stage',
        'first-of-four',
        'second-of-four',
        'last-of-four',
    ],
)
def test_is_dualpipev_first_stage_returns_whether_stage_is_first(
    dualpipev_stage, dualpipev_size, expected
):
    assert pp_utils.is_dualpipev_first_stage(dualpipev_stage, dualpipev_size) is expected


@pytest.mark.parametrize(
    'dualpipev_stage,dualpipev_size,expected',
    [(None, None, True), (0, None, True), (0, 1, True), (3, 4, True), (2, 4, False), (0, 4, False)],
    ids=[
        'no-stage-no-size',
        'zero-stage-no-size',
        'single-stage',
        'last-of-four',
        'third-of-four',
        'first-of-four',
    ],
)
def test_is_dualpipev_last_stage_returns_whether_stage_is_last(
    dualpipev_stage, dualpipev_size, expected
):
    assert pp_utils.is_dualpipev_last_stage(dualpipev_stage, dualpipev_size) is expected


@pytest.mark.parametrize(
    'predicate,args',
    [
        (pp_utils.is_dualpipev_first_stage, (1, None)),
        (pp_utils.is_dualpipev_first_stage, (2, 1)),
        (pp_utils.is_dualpipev_last_stage, (2, None)),
        (pp_utils.is_dualpipev_last_stage, (3, 1)),
    ],
    ids=['first-missing-size', 'first-single-size', 'last-missing-size', 'last-single-size'],
)
def test_dualpipev_stage_predicates_reject_nonzero_stage_without_size(predicate, args):
    with pytest.raises(AssertionError, match='Expected dualpipev_stage to be 0 or None'):
        predicate(*args)


def test_pp_rank_helpers_return_first_last_and_neighbors(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=4)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    ranks = torch.distributed.get_process_group_ranks(pp_group)
    group_rank = pp_group.rank()

    assert pp_utils.get_pp_first_rank(pp_group) == ranks[0]
    assert pp_utils.get_pp_last_rank(pp_group) == ranks[-1]

    if group_rank == len(ranks) - 1:
        assert pp_utils.get_pp_next_rank(pp_group) is None
    else:
        assert pp_utils.get_pp_next_rank(pp_group) == ranks[group_rank + 1]

    if group_rank == 0:
        assert pp_utils.get_pp_prev_rank(pp_group) is None
    else:
        assert pp_utils.get_pp_prev_rank(pp_group) == ranks[group_rank - 1]


def test_set_streams_creates_default_comm_stream_when_unset(monkeypatch):
    monkeypatch.setattr(pp_utils, '_COMM_STREAM', None)
    pp_utils.set_streams()
    stream = pp_utils.get_comm_stream()
    assert isinstance(stream, cur_platform.Stream)
    assert stream.device.type == 'cuda'
    pp_utils.set_streams()
    assert pp_utils.get_comm_stream() is stream


def test_set_streams_creates_high_priority_comm_stream_when_requested(monkeypatch):
    monkeypatch.setattr(pp_utils, '_COMM_STREAM', None)
    pp_utils.set_streams(high_priority=True)
    stream = pp_utils.get_comm_stream()
    _, high = cur_platform.Stream.priority_range()
    assert isinstance(stream, cur_platform.Stream)
    assert stream.priority == high


def test_set_streams_uses_provided_comm_stream_when_unset(monkeypatch):
    custom = cur_platform.Stream(device='cuda')
    monkeypatch.setattr(pp_utils, '_COMM_STREAM', None)
    pp_utils.set_streams(comm_stream=custom)
    assert pp_utils.get_comm_stream() is custom


def test_set_ideal_affinity_assigns_current_gpu_numa_cpus():
    assert pp_utils.set_ideal_affinity_for_current_gpu() is None
    cpus = os.sched_getaffinity(0)
    assert min(cpus) >= 0
    assert max(cpus) < os.cpu_count()


def test_schedule_node_forwards_and_backwards_with_input_grads():
    weight = torch.nn.Parameter(torch.randn(3, 4, device='cuda'))
    x = torch.randn(2, 3, device='cuda', requires_grad=True)
    event = cur_platform.Event()
    node = ScheduleNode(
        forward_func=lambda inp: inp @ weight, stream=get_comp_stream, event=event, name='matmul'
    )

    assert callable(node.stream)
    output = node.forward(x)

    assert torch.equal(node.get_output(), x @ weight)
    assert torch.equal(output, x @ weight)

    grad_out = torch.full((2, 4), 2.0, device='cuda')
    grads = node.backward(grad_out)

    assert isinstance(node.stream, cur_platform.Stream)
    assert not callable(node.stream)
    expected_input_grad = torch.full((2, 4), 1.0, device='cuda') @ weight.detach().t() * 2.0
    torch.testing.assert_close(grads, expected_input_grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        weight.grad, x.detach().t() @ torch.full((2, 4), 2.0, device='cuda'), rtol=1e-5, atol=1e-5
    )
    assert node.output is None
    assert node.inputs is None
