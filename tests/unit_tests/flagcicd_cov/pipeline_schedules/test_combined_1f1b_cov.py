# Copyright (c) 2026, BAAI. All rights reserved.
"""Combined (EP-overlap) 1F1B schedule entry points driven by small GPT models."""

from collections import Counter
from functools import partial

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.pipeline_parallel.combined_1f1b as combined
import megatron.core.pipeline_parallel.schedules as schedule
import megatron.core.pipeline_parallel.utils as pp_utils
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

SEQ_LENGTH = 32
VOCAB_SIZE = 128
HIDDEN_SIZE = 64


class _RecordingTimers:
    def __init__(self):
        self.started = Counter()
        self.stopped = Counter()

    def __call__(self, name, log_level=None):
        timers = self

        class _Timer:
            def start(self, barrier=False):
                timers.started[name] += 1

            def stop(self):
                timers.stopped[name] += 1

        return _Timer()


def _make_data(seed):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, VOCAB_SIZE, (1, SEQ_LENGTH), dtype=torch.int64, generator=generator
    ).cuda()
    labels = torch.randint(
        0, VOCAB_SIZE, (1, SEQ_LENGTH), dtype=torch.int64, generator=generator
    ).cuda()
    position_ids = torch.arange(SEQ_LENGTH, dtype=torch.int64).unsqueeze(0).cuda()
    attention_mask = torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=bool).cuda()
    return {
        'input_ids': input_ids,
        'labels': labels,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }


def _summed_loss_func(output_tensor):
    loss = output_tensor.float().sum()
    return loss, {'lm loss': loss.detach().clone()}


def _shaped_loss_func(output_tensor):
    loss = output_tensor * 1.0
    return loss, {'lm loss': output_tensor.float().sum().detach().clone()}


def _make_plan_forward_step(loss_func):
    def forward_step(data_iterator, model, return_schedule_plan=False):
        data = next(data_iterator)
        if return_schedule_plan:
            return model.build_schedule_plan(**data), loss_func
        return model(**data), loss_func

    return forward_step


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_combined_no_pipelining_overlap_moe_matches_eager_reference(model_parallel):
    model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=2
    )
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    num_microbatches = 2
    timers = _RecordingTimers()
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=4,
        moe_token_dispatcher_type='alltoall',
        moe_grouped_gemm=False,
        expert_model_parallel_size=2,
        moe_router_dtype='fp32',
        attention_backend='unfused',
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        overlap_moe_expert_parallel_comm=True,
        timers=timers,
        grad_scale_func=lambda loss: loss * 2.0,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(
            config, use_transformer_engine=False, vp_stage=0, pp_rank=0
        ),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LENGTH,
        pre_process=True,
        post_process=True,
    )
    model.bfloat16().cuda()
    model.train()
    data = [_make_data(810), _make_data(811)]
    reference_losses = [model(**entry).float().sum() for entry in data]

    losses = schedule.forward_backward_no_pipelining(
        forward_step_func=_make_plan_forward_step(_summed_loss_func),
        data_iterator=iter(data),
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=SEQ_LENGTH,
        micro_batch_size=1,
        forward_only=False,
    )

    schedule_grads = {
        'embedding': model.embedding.weight.grad.detach().clone(),
        'output_layer': model.output_layer.weight.grad.detach().clone(),
    }
    for parameter in model.parameters():
        parameter.grad = None
    total = torch.zeros((), dtype=torch.float32, device='cuda')
    for entry in data:
        total = total + model(**entry).float().sum()
    (total / num_microbatches).backward()

    assert len(losses) == num_microbatches
    for entry, reference_loss in zip(losses, reference_losses):
        assert entry['lm loss'].item() == pytest.approx(reference_loss.item(), rel=1.6e-2, abs=1e-2)
    torch.testing.assert_close(
        schedule_grads['embedding'].float(),
        2.0 * model.embedding.weight.grad.float(),
        rtol=1.6e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        schedule_grads['output_layer'].float(),
        2.0 * model.output_layer.weight.grad.float(),
        rtol=1.6e-2,
        atol=1e-2,
    )
    assert timers.started['forward-compute'] == num_microbatches
    assert timers.stopped['forward-compute'] == num_microbatches


def test_combined_interleaved_pipelining_overlaps_forward_and_backward(model_parallel, monkeypatch):
    model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    timers = _RecordingTimers()
    config = TransformerConfig(
        num_layers=4,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=128,
        attention_backend='unfused',
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        enable_autocast=True,
        autocast_dtype=torch.float32,
        timers=timers,
    )
    chunks = [
        GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config, use_transformer_engine=False, vp_stage=vp_stage, pp_rank=pp_rank
            ),
            vocab_size=VOCAB_SIZE,
            max_sequence_length=SEQ_LENGTH,
            pre_process=vp_stage == 0,
            post_process=vp_stage == 1,
            vp_stage=vp_stage,
        )
        for vp_stage in range(2)
    ]
    for chunk in chunks:
        chunk.cuda()
        chunk.train()
    data = [_make_data(820), _make_data(821)]
    forward_data_store = []
    pending_inputs = []
    consumed_inputs = []
    consumed_outputs = []
    backward_grads = []

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        if model_chunk_id == 0:
            return None
        hidden = pending_inputs.pop(0)
        consumed_inputs.append(hidden)
        return [hidden]

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        if model_chunk_id == 0:
            pending_inputs.append(output_tensor)
        else:
            consumed_outputs.append(output_tensor[0])
            backward_grads.append(torch.ones_like(output_tensor[0]))

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        return (consumed_inputs.pop(0), consumed_outputs.pop(0), backward_grads.pop(0))

    def backward_step_helper_postprocess(virtual_microbatch_id):
        return None

    def get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward):
        assert forward
        return virtual_microbatch_id // 2

    def get_model_chunk_id(virtual_microbatch_id, forward):
        assert forward or virtual_microbatch_id == 0
        return virtual_microbatch_id % 2 if forward else 1

    schedule_args = (
        config,
        _make_plan_forward_step(_shaped_loss_func),
        [iter(data), iter(data)],
        chunks,
        1,
        forward_data_store,
        forward_step_helper_preprocess,
        forward_step_helper_postprocess,
        backward_step_helper_preprocess,
        backward_step_helper_postprocess,
        get_microbatch_id_in_model_chunk,
        get_model_chunk_id,
        partial(schedule.check_first_val_step, None, False),
        lambda virtual_microbatch_id: virtual_microbatch_id // 2 == 0,
        False,
    )
    monkeypatch.setattr(pp_utils, '_COMM_STREAM', None)
    combined.combined_1f1b_schedule_for_interleaved_pipelining(
        *schedule_args, f_virtual_microbatch_id=0
    )
    first_output = combined.combined_1f1b_schedule_for_interleaved_pipelining(
        *schedule_args, f_virtual_microbatch_id=1
    )
    backward_output = consumed_outputs[0]
    _, input_tensor_grad = combined.combined_1f1b_schedule_for_interleaved_pipelining(
        *schedule_args, f_virtual_microbatch_id=2, b_virtual_microbatch_id=0
    )

    schedule_grads = {
        'embedding': chunks[0].embedding.weight.grad.detach().clone(),
        'output_layer': chunks[1].output_layer.weight.grad.detach().clone(),
    }
    for chunk in chunks:
        for parameter in chunk.parameters():
            parameter.grad = None

    with torch.autocast('cuda', dtype=torch.float32):
        hidden = chunks[0](**data[0])
        chunks[1].set_input_tensor([hidden])
        reference_loss = chunks[1](**data[0])
    hidden.retain_grad()
    reference_loss.float().sum().backward()

    assert input_tensor_grad.shape == hidden.shape
    torch.testing.assert_close(input_tensor_grad, hidden.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        schedule_grads['embedding'], chunks[0].embedding.weight.grad, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        schedule_grads['output_layer'], chunks[1].output_layer.weight.grad, rtol=1e-5, atol=1e-5
    )
    if pp_rank == 1:
        assert len(forward_data_store) == 1
        assert forward_data_store[0]['lm loss'].item() == pytest.approx(
            reference_loss.float().sum().item(), rel=1e-5, abs=1e-5
        )
    else:
        assert forward_data_store == []
    assert backward_output.schedule_plan is None
    assert backward_output.loss_func is None
    assert first_output[0][0] is backward_output
    assert timers.started['forward-compute'] == 3
    assert timers.stopped['forward-compute'] == 3
