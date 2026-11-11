# Copyright (c) 2026, BAAI. All rights reserved.
"""Interleaved 1F1B schedule configuration variants and argument validation."""

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 8
SEQ_LENGTH = 4
MICRO_BATCH_SIZE = 2


class _StageModule(torch.nn.Module):
    def __init__(self, config, linear, vp_stage):
        super().__init__()
        self.linear = linear
        self.config = config
        self.model_type = 'unit-test'
        self.vp_stage = vp_stage
        self._input_tensor = None

    def set_input_tensor(self, tensor):
        self._input_tensor = tensor[0] if isinstance(tensor, list) else tensor

    def forward(self, data=None):
        if self._input_tensor is not None:
            data, self._input_tensor = self._input_tensor, None
        return self.linear(data)


class _SlicingChunk(_StageModule):
    def __init__(self, config, linear, slice_sequence, tp_rank, tp_size):
        super().__init__(config, linear, 0)
        self.slice_sequence = slice_sequence
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def forward(self, data=None):
        if self._input_tensor is not None:
            data, self._input_tensor = self._input_tensor, None
        if self.slice_sequence:
            half = data.shape[0] // self.tp_size
            data = data[self.tp_rank * half : (self.tp_rank + 1) * half]
        return self.linear(data)


def _config(pp_size, vp_size, **overrides):
    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        pipeline_dtype=torch.float32,
    )
    config.hidden_size = HIDDEN_SIZE
    config.calculate_per_token_loss = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_linear(seed, in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE):
    generator = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features, generator=generator))
        linear.bias.copy_(torch.randn(out_features, generator=generator))
    return linear


def _make_input(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE, generator=generator).cuda()


def _forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
    def loss_func(output_tensor):
        loss = output_tensor.float().sum()
        return loss, {'lm loss': loss.detach().clone()}

    output = model(next(data_iterator))
    return output, loss_func


def _build_chunks(config, pp_size, num_chunks, pp_rank, seed):
    chunks = []
    for chunk_id in range(num_chunks):
        linear = _make_linear(seed + chunk_id * pp_size + pp_rank).cuda()
        chunks.append(_StageModule(config, linear, chunk_id))
    return chunks


def _reference(num_stages, seed, inputs, num_microbatches, spliced_rank=None, spliced_size=1):
    stages = []
    for index in range(num_stages):
        stage = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).cuda()
        donor = _make_linear(seed + index)
        with torch.no_grad():
            stage.weight.copy_(donor.weight)
            stage.bias.copy_(donor.bias)
        stages.append(stage)
    losses = []
    total = torch.zeros((), device='cuda')
    half = SEQ_LENGTH // spliced_size
    for data in inputs:
        hidden = (
            data if spliced_rank is None else data[spliced_rank * half : (spliced_rank + 1) * half]
        )
        for stage in stages:
            hidden = stage(hidden)
        losses.append(hidden.sum())
        total = total + hidden.sum()
    (total / num_microbatches).backward()
    return losses, total, stages


def _assert_chunk_grads_match_reference(chunks, pp_size, pp_rank, stages):
    for chunk_id, chunk in enumerate(chunks):
        position = chunk_id * pp_size + pp_rank
        torch.testing.assert_close(
            chunk.linear.weight.grad, stages[position].weight.grad, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            chunk.linear.bias.grad, stages[position].bias.grad, rtol=1e-5, atol=1e-5
        )


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def vp2_world(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    return {'rank': pp_group.rank(), 'last': pp_group.rank() == pp_group.size() - 1}


def test_interleaved_flushes_remaining_grad_sync_on_deep_pipeline(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = pp_group.rank()
    sync_calls = []

    def grad_sync_func(parameters):
        sync_calls.append([parameter.shape for parameter in parameters])

    config = _config(4, 2, grad_sync_func=grad_sync_func)
    chunks = _build_chunks(config, 4, 2, pp_rank, 34)
    inputs = [_make_input(730 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    _, _, stages = _reference(8, 34, inputs, 4)
    if pp_rank == 3:
        assert len(losses) == 4
    _assert_chunk_grads_match_reference(chunks, 4, pp_rank, stages)
    assert sync_calls == [
        [(HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE,)],
        [(HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE,)],
    ]


def test_interleaved_synchronizes_params_for_next_model_chunk(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=3,
    )
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = pp_group.rank()
    sync_calls = []
    config = _config(2, 3, param_sync_func=lambda parameters: sync_calls.append(parameters))
    chunks = _build_chunks(config, 2, 3, pp_rank, 35)
    inputs = [_make_input(740 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 3) for _ in range(3)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    _, _, stages = _reference(6, 35, inputs, 4)
    if pp_rank == 1:
        assert len(losses) == 4
    _assert_chunk_grads_match_reference(chunks, 2, pp_rank, stages)
    assert [len(list(call)) for call in sync_calls] == [2, 2, 2]


def test_interleaved_sequence_parallel_divides_tensor_shape_by_tp_size(model_parallel):
    model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = pp_group.rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    config = _config(2, 2, sequence_parallel=True)
    chunks = [
        _SlicingChunk(config, _make_linear(37 + pp_rank).cuda(), pp_rank == 0, tp_rank, 2),
        _StageModule(config, _make_linear(37 + 2 + pp_rank).cuda(), 1),
    ]
    inputs = [_make_input(745 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )

    ref_losses, _, stages = _reference(4, 37, inputs, 4, spliced_rank=tp_rank, spliced_size=2)
    if pp_rank == 1:
        assert len(losses) == 4
        for entry, ref_loss in zip(losses, ref_losses):
            assert entry['lm loss'].item() == pytest.approx(ref_loss.item(), rel=1e-5, abs=1e-5)
    _assert_chunk_grads_match_reference(chunks, 2, pp_rank, stages)


def test_interleaved_forward_only_expands_hidden_dim_for_hyper_connections(vp2_world):
    config = _config(2, 2, enable_hyper_connections=True, num_residual_streams=2)
    pp_rank = vp2_world['rank']
    chunks = [
        _StageModule(
            config,
            _make_linear(
                36 + pp_rank, HIDDEN_SIZE if pp_rank == 0 else HIDDEN_SIZE * 2, HIDDEN_SIZE * 2
            ).cuda(),
            0,
        ),
        _StageModule(
            config, _make_linear(36 + 2 + pp_rank, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2).cuda(), 1
        ),
    ]
    inputs = [_make_input(755 + index) for index in range(4)]

    losses = schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=_forward_step,
        data_iterator=[iter(inputs * 2) for _ in range(2)],
        model=chunks,
        num_microbatches=4,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=True,
    )

    stages = []
    for index in range(4):
        stage = torch.nn.Linear(
            HIDDEN_SIZE if index == 0 else HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, bias=True
        ).cuda()
        donor = _make_linear(
            36 + index, HIDDEN_SIZE if index == 0 else HIDDEN_SIZE * 2, HIDDEN_SIZE * 2
        )
        with torch.no_grad():
            stage.weight.copy_(donor.weight)
            stage.bias.copy_(donor.bias)
        stages.append(stage)
    ref_losses = []
    for data in inputs:
        hidden = data
        for stage in stages:
            hidden = stage(hidden)
        ref_losses.append(hidden.sum())
    if vp2_world['last']:
        assert len(losses) == 4
        for entry, ref_loss in zip(losses, ref_losses):
            assert entry['lm loss'].item() == pytest.approx(ref_loss.item(), rel=1e-5, abs=1e-5)
    else:
        assert losses == []


def test_interleaved_rejects_both_p2p_overlap_modes(vp2_world):
    config = _config(2, 2, overlap_p2p_comm=True, batch_p2p_comm=True)
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 39)
    with pytest.raises(ValueError, match='Can not use both overlap_p2p_comm and batch_p2p_comm'):
        schedule.forward_backward_pipelining_with_interleaving(
            forward_step_func=_forward_step,
            data_iterator=[iter([]), iter([])],
            model=chunks,
            num_microbatches=4,
            seq_length=SEQ_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            forward_only=True,
        )


def test_interleaved_rejects_microbatch_group_outside_pipeline_range(vp2_world):
    config = _config(2, 2, microbatch_group_size_per_vp_stage=1)
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 40)
    with pytest.raises(ValueError, match='should range in'):
        schedule.forward_backward_pipelining_with_interleaving(
            forward_step_func=_forward_step,
            data_iterator=[iter([]), iter([])],
            model=chunks,
            num_microbatches=4,
            seq_length=SEQ_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            forward_only=True,
        )


def test_interleaved_rejects_oversized_final_microbatch_group(vp2_world):
    config = _config(2, 2)
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 42)
    with pytest.raises(RuntimeError, match='remainder of M'):
        schedule.forward_backward_pipelining_with_interleaving(
            forward_step_func=_forward_step,
            data_iterator=[iter([]), iter([])],
            model=chunks,
            num_microbatches=3,
            seq_length=SEQ_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            forward_only=True,
        )


def test_interleaved_rejects_partial_process_groups(vp2_world):
    config = _config(2, 2)
    chunks = _build_chunks(config, 2, 2, vp2_world['rank'], 43)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    communicator = P2PCommunicator(pp_group=pp_group, config=config)
    with pytest.raises(ValueError, match='Invalid combination of p2p_communicator'):
        schedule.forward_backward_pipelining_with_interleaving(
            forward_step_func=_forward_step,
            data_iterator=[iter([]), iter([])],
            model=chunks,
            num_microbatches=4,
            seq_length=SEQ_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            forward_only=True,
            p2p_communicator=communicator,
        )
