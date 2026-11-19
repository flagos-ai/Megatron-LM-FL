# Copyright (c) 2026, BAAI. All rights reserved.
"""NVLS inference dispatcher local paths: flattening dispatch_preprocess with shared-expert side-stream launch, single-EP-rank pass-throughs, and shape-restoring combine_postprocess."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup

    Utils.destroy_model_parallel()


@pytest.fixture
def single_rank_groups(model_parallel):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    return ProcessGroupCollection.use_mpu_process_groups()


def _build_nvls_dispatcher(single_rank_groups, **kwargs):
    return NVLSAllGatherVDispatcher(
        8,
        list(range(8)),
        config=TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
        ),
        pg_collection=single_rank_groups,
        **kwargs,
    )


def test_nvls_dispatcher_stores_topk_and_shared_expert_defaults(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    assert dispatcher.topk == 2
    assert dispatcher._runs_metadata_sync is True
    assert dispatcher._local_tokens == 0
    assert dispatcher._shared_expert_output is None
    assert dispatcher._external_shared_expert_launch is False


def test_nvls_dispatch_preprocess_flattens_and_counts_local_tokens(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    hidden = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    routing_map = torch.arange(12).reshape(6, 2) % 8
    probs = torch.rand(6, 2)

    flat, returned_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)

    assert torch.equal(flat, hidden.view(6, 8))
    assert returned_probs is probs
    assert dispatcher._local_tokens == 6
    assert dispatcher.routing_map is routing_map


def test_nvls_token_dispatch_returns_local_tensors_when_single_ep_rank(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    flat = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    probs = torch.rand(6, 2)
    gathered_hidden, gathered_probs = dispatcher.token_dispatch(flat, probs)
    assert gathered_hidden is flat
    assert gathered_probs is probs


def test_nvls_token_combine_casts_to_bfloat16_when_single_ep_rank(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    expert_output = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    local_output = dispatcher.token_combine(expert_output)
    assert local_output.dtype == torch.bfloat16
    assert torch.equal(local_output, expert_output.to(torch.bfloat16))


def test_nvls_dispatch_postprocess_returns_tensors_unchanged(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    flat, probs = torch.randn(6, 8), torch.rand(6, 2)
    result = dispatcher.dispatch_postprocess(flat, probs)
    assert result[0] is flat
    assert result[1] is None
    assert result[2] is probs


def test_nvls_combine_preprocess_returns_expert_output_unchanged(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    expert_output = torch.randn(6, 8)
    assert dispatcher.combine_preprocess(expert_output) is expert_output


def test_nvls_combine_postprocess_restores_input_shape(single_rank_groups):
    dispatcher = _build_nvls_dispatcher(single_rank_groups)
    hidden = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    routing_map = torch.arange(12).reshape(6, 2) % 8
    flat, _ = dispatcher.dispatch_preprocess(hidden, routing_map, torch.rand(6, 2))
    output = dispatcher.combine_postprocess(flat.clone())
    assert output.shape == (3, 2, 8)
    assert torch.equal(output, hidden)
    assert dispatcher._shared_expert_output is None
