# Copyright (c) 2026, BAAI. All rights reserved.
"""Full-iteration CUDA graphs: static input buffers, warmup/capture/replay of a training step, and per-stage graph reset."""

import pytest
import torch

from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    StaticBufferLoader,
    clone_tensors_in_struct,
    copy_tensors_in_struct,
    get_graph_pool,
    get_shared_capture_stream,
    get_shared_graph_pool,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def fresh_static_buffers(monkeypatch):
    # StaticBufferLoader.static_buffers is a process-wide cache of graph input
    # buffers; reset it to the class's initial mapping so each test starts clean.
    buffers = {'training': [], 'validation': []}
    monkeypatch.setattr(StaticBufferLoader, 'static_buffers', buffers)
    return buffers


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    FullCudaGraphWrapper.reset_cuda_graph()
    Utils.destroy_model_parallel()


def test_get_shared_capture_stream_returns_one_stream_across_calls(monkeypatch):
    import megatron.core.full_cuda_graph as full_cuda_graph

    monkeypatch.setattr(full_cuda_graph, '_shared_capture_stream', None)
    first = get_shared_capture_stream()
    assert first is get_shared_capture_stream()
    assert isinstance(first, torch.cuda.Stream)


def test_get_graph_pool_shares_one_pool_only_in_single_mempool_mode(monkeypatch):
    import megatron.core.full_cuda_graph as full_cuda_graph

    monkeypatch.setattr(full_cuda_graph, '_shared_graph_pool', None)
    shared = get_shared_graph_pool()
    assert get_shared_graph_pool() is shared
    assert get_graph_pool(True) is shared
    separate = get_graph_pool(False)
    assert separate is not shared
    assert get_graph_pool(False) is not separate


def test_copy_tensors_in_struct_deep_copies_tensors_in_nested_structures():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    src = (tensor, [tensor, 7], {'k': tensor, 'n': 'text'}, 3.5)
    out = copy_tensors_in_struct(src)
    assert isinstance(out, tuple) and isinstance(out[1], list) and isinstance(out[2], dict)
    assert out[3] == 3.5
    for copied in (out[0], out[1][0], out[2]['k']):
        assert torch.equal(copied, tensor)
        assert copied.data_ptr() != tensor.data_ptr()
        assert copied.requires_grad is False
    assert out[1][1] == 7
    assert out[2]['n'] == 'text'


def test_clone_tensors_in_struct_copies_into_existing_tensors_and_scalars():
    target_tensor = torch.zeros(2, 2)
    target = {'t': target_tensor, 's': [0, 1], 'n': None}
    source = {'t': torch.ones(2, 2), 's': [2, 3], 'n': 'new'}
    clone_tensors_in_struct(target, source)
    assert torch.equal(target['t'], torch.ones(2, 2))
    assert target['s'] == [2, 3]
    assert target['n'] == 'new'
    assert target['t'] is target_tensor


def test_clone_tensors_in_struct_rejects_tuples_at_the_top_level():
    with pytest.raises(Exception, match='Unsupported copy for tuple yet'):
        clone_tensors_in_struct({}, (torch.ones(2),))


def test_clone_tensors_in_struct_rejects_non_container_top_level():
    with pytest.raises(Exception, match='Expect top-level as container type'):
        clone_tensors_in_struct({}, 3)


def test_static_buffer_loader_appends_first_microbatch_then_updates_new_keys(
    monkeypatch, fresh_static_buffers
):
    buffers = fresh_static_buffers
    loader = StaticBufferLoader()
    tokens = torch.tensor([1.0, 2.0])
    mask = torch.tensor([True, False])

    first = loader({'tokens': tokens, 'mask': mask}, 'training', 0)
    assert buffers['training'][0] is first
    assert torch.equal(first['tokens'], tokens)
    assert first['mask'].dtype == torch.bool
    assert torch.equal(first['mask'], mask)

    tokens2 = torch.tensor([5.0, 6.0])
    extra = torch.tensor([9.0, 9.0])
    second = loader(
        ({'tokens': tokens2, 'mask': mask, 'labels': 3, 'extra': extra},), 'training', 0
    )
    assert second is first
    assert torch.equal(second['tokens'], tokens2)
    assert second['labels'] == 3
    assert isinstance(second['mask'], torch.Tensor)
    assert torch.equal(second['extra'], extra)
    assert second['extra'].shape == extra.shape

    third = loader({'tokens': tokens, 'mask': mask}, 'training', 1)
    assert buffers['training'][1] is third
    assert len(buffers['training']) == 2
    assert torch.equal(buffers['training'][1]['tokens'], tokens)


def test_static_buffer_loader_keeps_training_and_validation_stages_separate(fresh_static_buffers):
    buffers = fresh_static_buffers
    loader = StaticBufferLoader()
    train_buf = loader({'tokens': torch.ones(2)}, 'training', 0)
    valid_buf = loader({'tokens': torch.zeros(2)}, 'validation', 0)
    assert buffers['training'][0] is train_buf
    assert buffers['validation'][0] is valid_buf
    assert torch.equal(train_buf['tokens'], torch.ones(2))
    assert torch.equal(valid_buf['tokens'], torch.zeros(2))


def _sum_loss_forward_backward(*, data_iterator, model, num_microbatches, **kwargs):
    losses = []
    for it in data_iterator:
        for _ in range(num_microbatches):
            out = model[0](next(it)['x'])
            loss = out.sum()
            loss.backward()
            losses.append({'loss': loss})
    return losses


def test_full_cuda_graph_wrapper_captures_and_replays_training_step(
    monkeypatch, model_parallel, fresh_static_buffers
):
    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    model = torch.nn.Linear(4, 1, bias=False).cuda()
    weight = model.weight.detach().clone()
    num_microbatches = 2
    batch = {'x': torch.arange(8.0).reshape(2, 4).cuda()}
    wrapper = FullCudaGraphWrapper(_sum_loss_forward_backward, cuda_graph_warmup_steps=1)

    def run_iteration():
        return wrapper(
            data_iterator=[iter([dict(batch)] * num_microbatches)],
            model=[model],
            num_microbatches=num_microbatches,
            seq_length=4,
            micro_batch_size=2,
            forward_only=False,
        )

    warmup = run_iteration()
    captured = run_iteration()
    replayed = run_iteration()
    expected = (weight @ batch['x'].T).sum().item()
    assert [entry['loss'].item() for entry in warmup] == pytest.approx([expected, expected])
    assert [entry['loss'].item() for entry in captured] == pytest.approx([expected, expected])
    assert [entry['loss'].item() for entry in replayed] == pytest.approx([expected, expected])
    assert isinstance(FullCudaGraphWrapper.cuda_graph['training'], torch.cuda.CUDAGraph)

    # Static buffers are the graph inputs, so new data must flow through the replay.
    batch['x'] = torch.arange(8.0, 16.0).reshape(2, 4).cuda()
    replayed_new_data = run_iteration()
    expected_new = (weight @ batch['x'].T).sum().item()
    assert [entry['loss'].item() for entry in replayed_new_data] == pytest.approx(
        [expected_new, expected_new]
    )

    wrapper.reset_cuda_graph(stage='training')
    assert FullCudaGraphWrapper.cuda_graph['training'] is None
    assert FullCudaGraphWrapper.result['training'] is None
    assert FullCudaGraphWrapper.curr_iteration['training'] == 0


def test_full_cuda_graph_wrapper_reset_clears_only_the_requested_stage(
    monkeypatch, model_parallel, fresh_static_buffers
):
    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    model = torch.nn.Linear(4, 1, bias=False).cuda()
    batch = {'x': torch.arange(8.0).reshape(2, 4).cuda()}
    wrapper = FullCudaGraphWrapper(_sum_loss_forward_backward, cuda_graph_warmup_steps=1)

    def run_iteration(forward_only):
        return wrapper(
            data_iterator=[iter([dict(batch)] * 2)],
            model=[model],
            num_microbatches=2,
            seq_length=4,
            micro_batch_size=2,
            forward_only=forward_only,
        )

    run_iteration(forward_only=False)
    run_iteration(forward_only=False)
    run_iteration(forward_only=True)
    run_iteration(forward_only=True)
    training_graph = FullCudaGraphWrapper.cuda_graph['training']
    validation_graph = FullCudaGraphWrapper.cuda_graph['validation']
    assert isinstance(training_graph, torch.cuda.CUDAGraph)
    assert isinstance(validation_graph, torch.cuda.CUDAGraph)
    assert FullCudaGraphWrapper.curr_iteration == {'training': 2, 'validation': 2}

    wrapper.reset_cuda_graph(stage='training')
    assert FullCudaGraphWrapper.cuda_graph['training'] is None
    assert FullCudaGraphWrapper.result['training'] is None
    assert FullCudaGraphWrapper.curr_iteration['training'] == 0
    assert FullCudaGraphWrapper.cuda_graph['validation'] is validation_graph
    assert FullCudaGraphWrapper.curr_iteration['validation'] == 2

    wrapper.reset_cuda_graph()
    assert FullCudaGraphWrapper.cuda_graph == {'training': None, 'validation': None}
    assert FullCudaGraphWrapper.result == {'training': None, 'validation': None}
    assert FullCudaGraphWrapper.curr_iteration == {'training': 0, 'validation': 0}


def test_full_cuda_graph_wrapper_data_read_builds_iterators_per_model(
    monkeypatch, fresh_static_buffers
):
    wrapper = FullCudaGraphWrapper(lambda **kwargs: None)
    microbatches = 2

    data_list = wrapper.data_read(
        [iter([{'x': torch.ones(2)}] * microbatches)], [None], True, microbatches
    )
    assert isinstance(data_list[0], type(iter([])))
    assert len(next(data_list[0])['x']) == 2

    data_list = wrapper.data_read(None, [None], True, microbatches)
    assert data_list == [None]

    data_list = wrapper.data_read(
        [iter([{'x': torch.ones(2)}] * microbatches), None], [None, None], False, microbatches
    )
    assert isinstance(data_list[0], type(iter([])))
    assert data_list[1] is None
    batch = next(data_list[0])
    assert torch.equal(batch['x'], torch.ones(2))
    assert StaticBufferLoader.static_buffers['validation'][0] is batch
