# Copyright (c) 2026, BAAI. All rights reserved.
"""NCCL inference dispatcher: CUDA-graph steps gather equal token counts with summed valid-token metadata; eager steps pad to the max rank count, allgather, compact, and expand/truncate on combine."""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NCCLAllGatherDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup

    Utils.destroy_model_parallel()


@pytest.fixture
def dispatcher_class_state():
    yield
    NCCLAllGatherDispatcher._use_allgather_v = False
    NCCLAllGatherDispatcher._local_tokens_per_rank = None
    InferenceAllGatherDispatcherBase._valid_tokens_tensor = None
    InferenceAllGatherDispatcherBase._host_valid_tokens_estimate = None


@pytest.fixture
def eager_prefill_step(model_parallel, dispatcher_class_state):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    context = DynamicInferenceContext(
        model_config=_inference_config(),
        inference_config=InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.02,
            block_size_tokens=128,
            max_requests=64,
            max_tokens=512,
        ),
    )
    context.add_request(
        DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(0, 8, dtype=torch.long, device="cuda"),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
    )
    context.initialize_attention_state()
    yield context, ProcessGroupCollection.use_mpu_process_groups()


@pytest.fixture
def cuda_graph_step(model_parallel, dispatcher_class_state):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    context = DynamicInferenceContext(
        model_config=_inference_config(),
        inference_config=InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.02,
            block_size_tokens=128,
            num_cuda_graphs=4,
            max_requests=64,
            max_tokens=512,
        ),
    )
    context.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
    yield context, ProcessGroupCollection.use_mpu_process_groups()


def _inference_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="allgather",
        inference_moe_token_dispatcher_type="nccl",
    )


def _build_dispatcher(model_parallel, ep_size, **kwargs):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=ep_size)
    return _dispatcher_from_groups(
        ProcessGroupCollection.use_mpu_process_groups(), ep_size, **kwargs
    )


def _dispatcher_from_groups(pg_collection, ep_size, **kwargs):
    num_local_experts = 8 // ep_size
    first_local_expert = pg_collection.ep.rank() * num_local_experts
    return NCCLAllGatherDispatcher(
        num_local_experts,
        list(range(first_local_expert, first_local_expert + num_local_experts)),
        config=_inference_config(),
        pg_collection=pg_collection,
        **kwargs,
    )


def _local_rank_parts(rank, tokens_per_rank):
    tokens = torch.arange(tokens_per_rank)
    cols = torch.arange(8)
    hidden = (rank * 1000 + tokens[:, None] * 10 + cols[None, :]).to(torch.float32)
    k = torch.arange(2)
    probs = (rank * 10 + tokens[:, None] * 2 + k[None, :]).to(torch.float32)
    routing_map = (tokens[:, None] * 2 + k[None, :] + rank) % 8
    return hidden, probs, routing_map


def test_dispatcher_stores_metadata_sync_flag_when_disabled(model_parallel):
    dispatcher = _build_dispatcher(model_parallel, 1, runs_metadata_sync=False)
    assert isinstance(dispatcher, InferenceAllGatherDispatcherBase)
    assert dispatcher._runs_metadata_sync is False
    assert dispatcher.topk == 2


def test_base_update_metadata_is_abstract_and_raises(model_parallel):
    dispatcher = _build_dispatcher(model_parallel, 1)
    with pytest.raises(NotImplementedError) as caught:
        InferenceAllGatherDispatcherBase.update_metadata(dispatcher, 4)
    assert str(caught.value) == ""


def test_token_dispatch_returns_local_tensors_when_single_ep_rank(model_parallel):
    dispatcher = _build_dispatcher(model_parallel, 1)
    hidden = torch.randn(4, 2, 8)
    routing_map, probs = torch.zeros(8, 2, dtype=torch.int64), torch.rand(8, 2)
    flat, out_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
    gathered_hidden, gathered_probs = dispatcher.token_dispatch(flat, out_probs)
    assert gathered_hidden is flat
    assert gathered_probs is out_probs
    assert dispatcher.routing_map is routing_map


def test_token_combine_casts_to_bfloat16_when_single_ep_rank(model_parallel):
    dispatcher = _build_dispatcher(model_parallel, 1)
    expert_output = torch.randn(8, 8)
    local_output = dispatcher.token_combine(expert_output)
    assert local_output.dtype == torch.bfloat16
    assert torch.equal(local_output, expert_output.to(torch.bfloat16))


def test_dispatch_postprocess_passes_gathered_tensors_through(model_parallel):
    dispatcher = _build_dispatcher(model_parallel, 1)
    flat, probs = torch.randn(8, 8), torch.rand(8, 2)
    result = dispatcher.dispatch_postprocess(flat, probs)
    assert result[0] is flat
    assert result[1] is None
    assert result[2] is probs


def test_combine_preprocess_passes_expert_output_through(model_parallel):
    dispatcher = _build_dispatcher(model_parallel, 1)
    expert_output = torch.randn(8, 8)
    assert dispatcher.combine_preprocess(expert_output) is expert_output


def test_allocate_buffers_zeros_valid_tokens_on_current_device(
    model_parallel, dispatcher_class_state
):
    dispatcher = _build_dispatcher(model_parallel, 2)
    NCCLAllGatherDispatcher.allocate_buffers()
    valid_tokens = InferenceAllGatherDispatcherBase._valid_tokens()
    assert valid_tokens.shape == (1,)
    assert valid_tokens.dtype == torch.int32
    assert valid_tokens.item() == 0
    assert valid_tokens.device.type == "cuda"


def test_update_metadata_fills_equal_token_total_when_cuda_graph_step(cuda_graph_step):
    context, pg_collection = cuda_graph_step
    assert context.using_cuda_graph_this_step() is True
    assert NCCLAllGatherDispatcher._use_allgather_v is False
    dispatcher = _dispatcher_from_groups(pg_collection, 2)
    dispatcher.update_metadata(7)
    assert InferenceAllGatherDispatcherBase._valid_tokens().item() == 14
    assert InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate() == 14


def test_update_metadata_allgathers_unequal_counts_when_eager_prefill_step(eager_prefill_step):
    context, pg_collection = eager_prefill_step
    assert context.using_cuda_graph_this_step() is False
    assert NCCLAllGatherDispatcher._use_allgather_v is True
    dispatcher = _dispatcher_from_groups(pg_collection, 2)
    dispatcher.update_metadata(4 + 2 * dispatcher.ep_rank)
    assert NCCLAllGatherDispatcher._local_tokens_per_rank == [4, 6]
    assert InferenceAllGatherDispatcherBase._valid_tokens().item() == 10
    assert InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate() == 10


def test_token_dispatch_gathers_equal_tokens_when_cuda_graph_step(cuda_graph_step):
    context, pg_collection = cuda_graph_step
    assert context.using_cuda_graph_this_step() is True
    assert NCCLAllGatherDispatcher._use_allgather_v is False
    dispatcher = _dispatcher_from_groups(pg_collection, 2)
    hidden, probs, routing_map = (part.cuda() for part in _local_rank_parts(dispatcher.ep_rank, 4))
    flat, out_probs = dispatcher.dispatch_preprocess(hidden.view(2, 2, 8), routing_map, probs)
    gathered_hidden, gathered_probs = dispatcher.token_dispatch(flat, out_probs)
    expected_hidden, expected_probs, expected_routing = (
        torch.cat([part_zero, part_one]).cuda()
        for part_zero, part_one in zip(_local_rank_parts(0, 4), _local_rank_parts(1, 4))
    )
    assert gathered_hidden.shape == (8, 8)
    assert torch.equal(gathered_hidden, expected_hidden)
    assert torch.equal(gathered_probs, expected_probs)
    assert torch.equal(dispatcher.routing_map, expected_routing)


def test_token_dispatch_pads_and_compacts_unequal_tokens_when_eager_prefill_step(
    eager_prefill_step,
):
    context, pg_collection = eager_prefill_step
    assert context.using_cuda_graph_this_step() is False
    assert NCCLAllGatherDispatcher._use_allgather_v is True
    dispatcher = _dispatcher_from_groups(pg_collection, 2)
    tokens_per_rank = 4 + 2 * dispatcher.ep_rank
    hidden, probs, routing_map = (
        part.cuda() for part in _local_rank_parts(dispatcher.ep_rank, tokens_per_rank)
    )
    flat, out_probs = dispatcher.dispatch_preprocess(
        hidden.view(tokens_per_rank // 2, 2, 8), routing_map, probs
    )
    gathered_hidden, gathered_probs = dispatcher.token_dispatch(flat, out_probs)
    expected_hidden, expected_probs, expected_routing = (
        torch.cat([part_zero, part_one]).cuda()
        for part_zero, part_one in zip(_local_rank_parts(0, 4), _local_rank_parts(1, 6))
    )
    assert gathered_hidden.shape == (10, 8)
    assert torch.equal(gathered_hidden, expected_hidden)
    assert torch.equal(gathered_probs, expected_probs)
    assert torch.equal(dispatcher.routing_map, expected_routing)


def test_token_combine_reduce_scatters_equal_tokens_when_cuda_graph_step(cuda_graph_step):
    context, pg_collection = cuda_graph_step
    assert context.using_cuda_graph_this_step() is True
    assert NCCLAllGatherDispatcher._use_allgather_v is False
    dispatcher = _dispatcher_from_groups(pg_collection, 2)
    rows = torch.arange(8)
    cols = torch.arange(8)
    base = (rows[:, None] * 8 + cols[None, :] + 1).to(torch.float32)
    expert_output = ((dispatcher.ep_rank + 1) * base).cuda()
    local_output = dispatcher.token_combine(expert_output)
    expected = (
        (3 * base[4 * dispatcher.ep_rank : 4 * dispatcher.ep_rank + 4]).to(torch.bfloat16).cuda()
    )
    assert local_output.shape == (4, 8)
    assert local_output.dtype == torch.bfloat16
    assert torch.equal(local_output, expected)


def test_token_combine_expands_pads_and_truncates_when_eager_prefill_step(eager_prefill_step):
    context, pg_collection = eager_prefill_step
    assert context.using_cuda_graph_this_step() is False
    assert NCCLAllGatherDispatcher._use_allgather_v is True
    dispatcher = _dispatcher_from_groups(pg_collection, 2)
    dispatcher.update_metadata(4 + 2 * dispatcher.ep_rank)
    tokens_per_rank = 4 + 2 * dispatcher.ep_rank
    rows = torch.arange(10)
    cols = torch.arange(8)
    base = (rows[:, None] * 8 + cols[None, :] + 1).to(torch.float32)
    expert_output = ((dispatcher.ep_rank + 1) * base).cuda()
    local_output = dispatcher.token_combine(expert_output)
    expected = (
        (3 * base[4 * dispatcher.ep_rank : 4 * dispatcher.ep_rank + tokens_per_rank])
        .to(torch.bfloat16)
        .cuda()
    )
    assert local_output.shape == (tokens_per_rank, 8)
    assert local_output.dtype == torch.bfloat16
    assert torch.equal(local_output, expected)
