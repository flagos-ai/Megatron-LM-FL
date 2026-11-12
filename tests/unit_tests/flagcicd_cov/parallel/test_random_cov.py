# Copyright (c) 2026, BAAI. All rights reserved.
'''CUDA RNG state tracking and checkpointing: tracker fork/restore, inference and cudagraphable tracker variants, checkpointed recomputation with and without retained outputs, and graph-warmup bypasses.'''

import contextlib

import pytest
import torch

import megatron.core.tensor_parallel.random as random_module
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
    checkpoint,
    convert_cuda_rng_state,
    get_cuda_rng_tracker,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)
        model_parallel_cuda_manual_seed(123)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def inference_rng_tracker():
    initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
    yield get_cuda_rng_tracker(inference_rng_tracker=True)
    initialize_rng_tracker(force_reset=True)


@pytest.fixture
def cudagraphable_rng_tracker():
    initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)
    yield get_cuda_rng_tracker(use_cudagraphable_rng=True)
    initialize_rng_tracker(force_reset=True)


@pytest.fixture
def cuda_graph_warmup():
    from megatron.core.transformer import cuda_graphs

    cuda_graphs._set_warmup_start()
    yield
    cuda_graphs._set_warmup_end()


def test_cuda_rng_tracker_fork_restores_state_after_use(model_parallel):
    model_parallel(2, 1)
    tracker = get_cuda_rng_tracker()
    assert set(tracker.get_states()) == {
        'data-parallel-rng',
        'model-parallel-rng',
        'expert-parallel-rng',
    }

    before_state = torch.cuda.get_rng_state()
    with tracker.fork():
        inside = torch.rand(4, device='cuda')
    assert torch.equal(torch.cuda.get_rng_state(), before_state)

    after = torch.rand(4, device='cuda')
    assert not torch.equal(after, inside)


def test_cuda_rng_tracker_fork_raises_for_state_that_was_not_added(model_parallel):
    model_parallel(2, 1)
    with pytest.raises(Exception, match='cuda rng state missing-rng is not added'):
        with get_cuda_rng_tracker().fork('missing-rng'):
            pass


def test_set_cuda_rng_state_accepts_string_and_integer_devices(model_parallel):
    model_parallel(1, 1)
    torch.cuda.manual_seed(99)
    state = torch.cuda.get_rng_state()
    expected = torch.rand(4, device='cuda')

    random_module._set_cuda_rng_state(state, device='cuda')
    assert torch.equal(torch.rand(4, device='cuda'), expected)
    torch.rand(4, device='cuda')

    random_module._set_cuda_rng_state(state, device=torch.cuda.current_device())
    assert torch.equal(torch.rand(4, device='cuda'), expected)


def test_get_cuda_rng_state_graph_safe_accepts_integer_device(model_parallel):
    model_parallel(1, 1)
    graphable = random_module._get_cuda_rng_state(0, clone=True, graph_safe=True)

    assert isinstance(graphable, torch.Generator)
    assert torch.equal(graphable.get_state(), torch.cuda.get_rng_state(0))
    torch.testing.assert_close(
        torch.rand(4, device='cuda:0', generator=graphable),
        torch.rand(4, device='cuda:0'),
        rtol=0,
        atol=0,
    )


def test_convert_cuda_rng_state_validates_state_type():
    generator = torch.Generator(device='cuda')
    assert convert_cuda_rng_state(generator, to_graphable=True) is generator
    state = torch.cuda.get_rng_state()
    assert convert_cuda_rng_state(state, to_graphable=False) is state
    with pytest.raises(ValueError, match='Invalid state type'):
        convert_cuda_rng_state(3, to_graphable=True)
    with pytest.raises(ValueError, match='Invalid state type'):
        convert_cuda_rng_state(3, to_graphable=False)


def test_inference_rng_tracker_set_states_and_fork_are_noops(inference_rng_tracker):
    inference_rng_tracker.set_states({'unused': None})
    assert inference_rng_tracker.get_states() == {}
    assert isinstance(inference_rng_tracker.fork(), contextlib.nullcontext)


def test_cudagraphable_rng_tracker_adds_and_forks_graph_safe_states(
    model_parallel, cudagraphable_rng_tracker
):
    model_parallel(2, 1)
    cudagraphable_rng_tracker.add('test-rng', 777)
    assert 'test-rng' in cudagraphable_rng_tracker.get_states()
    assert random_module.is_graph_safe_cuda_rng_tracker(cudagraphable_rng_tracker) is True

    with cudagraphable_rng_tracker.fork():
        torch.rand(4, device='cuda')
    assert 'test-rng' in cudagraphable_rng_tracker.get_states()


@pytest.mark.parametrize(
    'distribute_saved_activations', [False, True], ids=['plain', 'distributed']
)
def test_checkpoint_recomputes_function_and_preserves_gradients(
    model_parallel, distribute_saved_activations
):
    model_parallel(2, 1)
    torch.manual_seed(21)
    x_reference = torch.randn(4, 4, device='cuda', requires_grad=True)
    x = x_reference.detach().clone().requires_grad_(True)

    def scale_and_square(tensor):
        return (tensor * 3.0).pow(2)

    output = checkpoint(scale_and_square, distribute_saved_activations, x)
    expected = scale_and_square(x_reference)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    (output * 1.5).sum().backward()
    (expected * 1.5).sum().backward()
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=1e-4, atol=1e-5)


def test_checkpoint_raises_when_backward_triggered_by_grad(model_parallel):
    model_parallel(2, 1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)

    output = checkpoint(lambda tensor: tensor * 2.0, False, x)
    with pytest.raises(RuntimeError, match='please use .backward()'):
        torch.autograd.grad(output.sum(), x)


@pytest.mark.parametrize(
    'retain_input_tensors, run_function',
    [
        pytest.param(False, lambda tensor, scale: (tensor * scale).pow(2), id='release'),
        pytest.param(True, lambda tensor, scale: tensor, id='retain'),
    ],
)
def test_checkpoint_without_output_recomputes_and_preserves_gradients(
    model_parallel, retain_input_tensors, run_function
):
    model_parallel(2, 1)
    torch.manual_seed(31)
    x_reference = torch.randn(4, 4, device='cuda', requires_grad=True)
    x = x_reference.detach().clone().requires_grad_(True)

    ckpt = CheckpointWithoutOutput(retain_input_tensors=retain_input_tensors)
    output = ckpt.checkpoint(run_function, x, 2.5)
    expected = run_function(x_reference, 2.5)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    downstream = output * 1.5
    ckpt.discard_output_and_register_recompute(downstream)
    if retain_input_tensors:
        assert output.untyped_storage().nbytes() != 0
    else:
        assert output.untyped_storage().nbytes() == 0

    downstream.sum().backward()
    expected_grad = torch.autograd.grad((expected * 1.5).sum(), x_reference)[0]
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-5)
    assert ckpt.outputs is None


def test_checkpoint_without_output_fp8_marks_context(model_parallel, monkeypatch):
    model_parallel(2, 1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)
    fp8_autocast_calls = []
    te_fp8_autocast = random_module.fp8_autocast

    def recording_fp8_autocast(*args, **kwargs):
        fp8_autocast_calls.append(dict(kwargs))
        return te_fp8_autocast(*args, **kwargs)

    monkeypatch.setattr(random_module, 'fp8_autocast', recording_fp8_autocast, raising=False)

    ckpt = CheckpointWithoutOutput(fp8=True)
    output = ckpt.checkpoint(lambda tensor: tensor * 2.0, x)
    downstream = output * 0.5
    ckpt.discard_output_and_register_recompute(downstream)

    assert ckpt.ctx.fp8 is False
    assert ckpt.ctx.fp8_recipe is None

    downstream.sum().backward()

    torch.testing.assert_close(x.grad, torch.ones(4, 4, device='cuda'), rtol=0, atol=0)
    assert fp8_autocast_calls == [{'enabled': False, 'fp8_recipe': None}]
    assert ckpt.ctx is None


def test_checkpoint_without_output_manager_recomputes_all_checkpoints(model_parallel):
    model_parallel(2, 1)
    torch.manual_seed(33)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)

    manager = CheckpointWithoutOutputManager()
    first = CheckpointWithoutOutput(ckpt_manager=manager)
    second = CheckpointWithoutOutput(ckpt_manager=manager)
    out_first = first.checkpoint(lambda tensor: tensor * 2.0, x)
    out_second = second.checkpoint(lambda tensor: tensor * 3.0, out_first)
    assert len(manager.checkpoints) == 2

    downstream = out_second * 1.5
    manager.discard_all_outputs_and_register_unified_recompute(downstream)
    downstream.sum().backward()

    torch.testing.assert_close(x.grad, 9 * torch.ones(4, 4, device='cuda'), rtol=0, atol=0)
    assert first.outputs is None
    assert second.outputs is None


def test_cuda_rng_tracker_fork_raises_when_exited_out_of_order(model_parallel):
    model_parallel(2, 1)
    tracker = get_cuda_rng_tracker()
    tracker.add('outer-rng', 4242)
    tracker.add('inner-rng', 5353)
    outer = tracker.fork('outer-rng')
    inner = tracker.fork('inner-rng')
    outer.__enter__()
    inner.__enter__()

    with pytest.raises(Exception, match='is not the same as the desired state name'):
        outer.__exit__(None, None, None)
    inner.__exit__(None, None, None)


def test_checkpoint_bypasses_checkpointing_during_cuda_graph_warmup(
    model_parallel, cuda_graph_warmup
):
    model_parallel(2, 1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)
    invocations = []

    def triple(tensor):
        invocations.append(tensor.detach().clone())
        return tensor * 3.0

    output = checkpoint(triple, False, x)

    torch.testing.assert_close(output, 3 * x.detach(), rtol=0, atol=0)
    assert type(output.grad_fn).__name__ == 'MulBackward0'
    output.sum().backward()
    torch.testing.assert_close(x.grad, 3 * torch.ones(4, 4, device='cuda'), rtol=0, atol=0)
    assert len(invocations) == 1
    assert torch.equal(invocations[0], x.detach())


def test_checkpoint_without_output_bypasses_during_cuda_graph_warmup(
    model_parallel, cuda_graph_warmup
):
    model_parallel(2, 1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)
    ckpt = CheckpointWithoutOutput()

    output = ckpt.checkpoint(lambda tensor: tensor * 2.0, x)

    torch.testing.assert_close(output, 2 * x.detach(), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(x.grad, 2 * torch.ones(4, 4, device='cuda'), rtol=0, atol=0)
    assert ckpt.ctx is None
    assert ckpt.outputs is None


def test_checkpoint_without_output_recompute_is_noop_when_already_triggered(model_parallel):
    model_parallel(2, 1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)
    invocations = []

    def scale(tensor):
        invocations.append(tensor.detach().clone())
        return tensor * 2.0 + 1.0

    ckpt = CheckpointWithoutOutput()
    output = ckpt.checkpoint(scale, x)
    downstream = output * 0.5
    ckpt.discard_output_and_register_recompute(downstream)
    downstream.sum().backward()

    assert len(invocations) == 2
    assert torch.equal(invocations[0], invocations[1])
    torch.testing.assert_close(x.grad, torch.ones(4, 4, device='cuda'), rtol=0, atol=0)

    ckpt._recompute(None)

    assert len(invocations) == 2
    assert ckpt.ctx is None
    assert ckpt.outputs is None


def test_checkpoint_without_output_recompute_raises_for_grad_based_backward(model_parallel):
    model_parallel(2, 1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)

    ckpt = CheckpointWithoutOutput()
    output = ckpt.checkpoint(lambda tensor: tensor * 2.0, x)
    downstream = output * 0.5
    ckpt.discard_output_and_register_recompute(downstream)
    with pytest.raises(RuntimeError, match='please use .backward()'):
        torch.autograd.grad(downstream.sum(), x)
