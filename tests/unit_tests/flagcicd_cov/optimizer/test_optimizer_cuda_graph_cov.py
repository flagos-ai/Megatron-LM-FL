# Copyright (c) 2026, BAAI. All rights reserved.
"""OptimizerCudaGraphWrapper: eager steps, capture and replay after warmup, state cleanup."""

import pytest
import torch

from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def fresh_graph_state(monkeypatch):
    monkeypatch.setattr(OptimizerCudaGraphWrapper, 'curr_iteration', 0)
    monkeypatch.setattr(OptimizerCudaGraphWrapper, 'cuda_graph', None)
    monkeypatch.setattr(OptimizerCudaGraphWrapper, 'result', None)


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def test_optimizer_cuda_graph_wrapper_steps_eagerly_before_warmup(fresh_graph_state):
    tensor = torch.zeros(4, device=_device())

    def optimizer_step():
        tensor.add_(1.0)
        return 41

    wrapper = OptimizerCudaGraphWrapper(optimizer_step, cuda_graph_warmup_steps=2)

    assert wrapper.curr_iter() == 0
    result = wrapper()

    assert result == 41
    assert OptimizerCudaGraphWrapper.result == 41
    assert OptimizerCudaGraphWrapper.cuda_graph is None
    torch.testing.assert_close(tensor, torch.ones(4, device=_device()))
    assert wrapper.curr_iter() == 1

    wrapper.next_iter()
    assert wrapper.curr_iter() == 2


@pytest.mark.parametrize(
    "step_args,step_kwargs,match",
    [
        pytest.param((1,), {}, 'positional args', id='positional'),
        pytest.param((), {'lr': 0.1}, 'keyword args', id='keyword'),
    ],
)
def test_optimizer_cuda_graph_wrapper_rejects_step_arguments(
    fresh_graph_state, step_args, step_kwargs, match
):
    wrapper = OptimizerCudaGraphWrapper(lambda: 0)

    with pytest.raises(AssertionError, match=match):
        wrapper(*step_args, **step_kwargs)


def test_optimizer_cuda_graph_wrapper_captures_and_replays_after_warmup(
    model_parallel, fresh_graph_state
):
    model_parallel()
    tensor = torch.zeros(4, device=_device())

    def optimizer_step():
        tensor.add_(1.0)
        return 42

    wrapper = OptimizerCudaGraphWrapper(optimizer_step, cuda_graph_warmup_steps=1)

    # Iteration 0: before the warmup iteration, the step runs eagerly.
    assert wrapper() == 42
    torch.testing.assert_close(tensor, torch.ones(4, device=_device()))

    # Iteration 1: the warmup iteration captures the step into a CUDA graph
    # and then replays it once.
    assert wrapper() == 42
    assert isinstance(OptimizerCudaGraphWrapper.cuda_graph, torch.cuda.CUDAGraph)
    torch.testing.assert_close(tensor, torch.full((4,), 2.0, device=_device()))

    # Iteration 2: the graph is replayed without re-running the Python step.
    assert wrapper() == 42
    torch.testing.assert_close(tensor, torch.full((4,), 3.0, device=_device()))
    assert wrapper.curr_iter() == 3


def test_optimizer_cuda_graph_wrapper_destructor_clears_shared_result(fresh_graph_state):
    wrapper = OptimizerCudaGraphWrapper(lambda: 7)

    wrapper()
    assert OptimizerCudaGraphWrapper.result == 7
    assert OptimizerCudaGraphWrapper.cuda_graph is None

    del wrapper

    assert OptimizerCudaGraphWrapper.result is None
    assert OptimizerCudaGraphWrapper.cuda_graph is None


def test_optimizer_cuda_graph_wrapper_destructor_releases_captured_graph(
    model_parallel, fresh_graph_state
):
    model_parallel()
    tensor = torch.zeros(4, device=_device())

    def optimizer_step():
        tensor.add_(1.0)
        return 42

    wrapper = OptimizerCudaGraphWrapper(optimizer_step, cuda_graph_warmup_steps=1)
    wrapper()
    wrapper()
    assert isinstance(OptimizerCudaGraphWrapper.cuda_graph, torch.cuda.CUDAGraph)

    del wrapper

    assert OptimizerCudaGraphWrapper.cuda_graph is None
    assert OptimizerCudaGraphWrapper.result is None
