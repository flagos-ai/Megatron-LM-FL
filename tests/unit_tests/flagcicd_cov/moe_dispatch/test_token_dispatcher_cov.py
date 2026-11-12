# Copyright (c) 2026, BAAI. All rights reserved.
"""Dispatcher base contracts: interface NotImplementedError's, shared-expert registration, alltoall cudagraph metadata points, and flex dispatcher backend wiring around the absent deep_ep kernels."""

import pytest
import torch

import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
    _DeepepManager,
    _DispatchManager,
    _HybridEPManager,
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
def single_rank_groups(model_parallel):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    return ProcessGroupCollection.use_mpu_process_groups()


@pytest.fixture
def flex_deepep_stubs(monkeypatch):
    calls = {"dispatch": [], "combine": [], "num_sms": []}
    handle = object()
    dispatched_indices = torch.tensor([[0, 1], [2, -1], [3, 0], [1, 2]])
    dispatched_probs = torch.tensor([[0.5, 0.25], [0.75, 0.0], [0.1, 0.9], [0.3, 0.7]])
    dispatched_tokens_per_expert = torch.tensor([2, 2, 2, 1])

    def _fake_fused_dispatch(
        hidden_states,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        calls["dispatch"].append(
            dict(
                hidden_states=hidden_states,
                token_indices=token_indices,
                token_probs=token_probs,
                num_experts=num_experts,
                group=group,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        return (
            hidden_states * 2,
            dispatched_indices,
            dispatched_probs,
            dispatched_tokens_per_expert,
            handle,
        )

    def _fake_fused_combine(
        hidden_states, group, handle, async_finish=False, allocate_on_comm_stream=False
    ):
        calls["combine"].append(
            dict(
                hidden_states=hidden_states,
                group=group,
                handle=handle,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        return hidden_states * 3, None

    def _fake_set_deepep_num_sms(num_sms):
        calls["num_sms"].append(num_sms)

    monkeypatch.setattr(token_dispatcher, "fused_dispatch", _fake_fused_dispatch)
    monkeypatch.setattr(token_dispatcher, "fused_combine", _fake_fused_combine)
    monkeypatch.setattr(token_dispatcher, "set_deepep_num_sms", _fake_set_deepep_num_sms)
    return calls


def _moe_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _flex_config(**overrides):
    return _moe_config(moe_token_dispatcher_type="flex", **overrides)


@pytest.mark.parametrize(
    "method_name,args",
    [
        pytest.param("dispatch_preprocess", (None, None, None), id="dispatch_preprocess"),
        pytest.param("token_dispatch", (None, None), id="token_dispatch"),
        pytest.param("dispatch_postprocess", (None, None), id="dispatch_postprocess"),
        pytest.param("combine_preprocess", (None,), id="combine_preprocess"),
        pytest.param("token_combine", (None,), id="token_combine"),
        pytest.param("combine_postprocess", (None,), id="combine_postprocess"),
    ],
)
def test_base_dispatcher_interface_methods_raise_not_implemented(
    single_rank_groups, method_name, args
):
    dispatcher = MoETokenDispatcher(config=_moe_config(), pg_collection=single_rank_groups)

    with pytest.raises(NotImplementedError, match=f"{method_name} function not implemented"):
        getattr(dispatcher, method_name)(*args)


def test_set_shared_experts_stores_module_and_enables_nccl_stream(single_rank_groups):
    dispatcher = MoETokenDispatcher(
        config=_moe_config(moe_shared_expert_overlap=True), pg_collection=single_rank_groups
    )
    shared_experts = torch.nn.Identity()
    assert dispatcher.use_nccl_stream is False

    dispatcher.set_shared_experts(shared_experts)

    assert dispatcher.shared_experts is shared_experts
    assert dispatcher.use_nccl_stream is True


def test_set_shared_experts_rejects_dispatcher_without_overlap(single_rank_groups):
    dispatcher = MoETokenDispatcher(config=_moe_config(), pg_collection=single_rank_groups)
    with pytest.raises(AssertionError) as caught:
        dispatcher.set_shared_experts(torch.nn.Identity())
    assert str(caught.value) == ""


def test_alltoall_dispatcher_keeps_default_dtoh_point_without_cuda_graph(
    single_rank_groups, monkeypatch
):
    monkeypatch.setattr(
        MoEAlltoAllTokenDispatcher, "cuda_dtoh_stream", MoEAlltoAllTokenDispatcher.cuda_dtoh_stream
    )
    dispatcher = MoEAlltoAllTokenDispatcher(
        8, list(range(8)), config=_moe_config(), pg_collection=single_rank_groups
    )
    assert dispatcher.cuda_dtoh_point == "before_permutation_1"
    assert dispatcher.cuda_sync_point == "no_sync"


def test_alltoall_dispatcher_moves_dtoh_point_before_ep_alltoall_with_cuda_graph(
    single_rank_groups, monkeypatch
):
    monkeypatch.setattr(
        MoEAlltoAllTokenDispatcher, "cuda_dtoh_stream", MoEAlltoAllTokenDispatcher.cuda_dtoh_stream
    )
    dispatcher = MoEAlltoAllTokenDispatcher(
        8,
        list(range(8)),
        config=_moe_config(cuda_graph_impl="local"),
        pg_collection=single_rank_groups,
    )
    assert dispatcher.cuda_dtoh_point == "before_ep_alltoall"


def test_hybrid_ep_manager_raises_import_error_when_hybrid_ep_missing(single_rank_groups):
    with pytest.raises(ImportError, match="HybridEP is not installed"):
        _HybridEPManager(
            group=single_rank_groups.tp_ep, num_local_experts=8, num_experts=8, config=_moe_config()
        )


def test_hybrid_ep_manager_passes_drop_and_pad_check_with_capacity_factor(single_rank_groups):
    config = _moe_config(moe_pad_expert_input_to_capacity=True, moe_expert_capacity_factor=1.0)
    with pytest.raises(ImportError, match="HybridEP is not installed"):
        _HybridEPManager(
            group=single_rank_groups.tp_ep, num_local_experts=8, num_experts=8, config=config
        )


def test_deepep_manager_raises_import_error_when_deepep_missing(single_rank_groups):
    with pytest.raises(ImportError, match="DeepEP is not installed"):
        _DeepepManager(
            group=single_rank_groups.tp_ep,
            num_local_experts=8,
            router_topk=2,
            num_experts=8,
            config=_moe_config(),
        )


def test_flex_dispatcher_rejects_unknown_backend(single_rank_groups):
    with pytest.raises(ValueError, match="Invalid backend"):
        MoEFlexTokenDispatcher(
            8,
            list(range(8)),
            config=_moe_config(
                moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="bogus"
            ),
            pg_collection=single_rank_groups,
        )


def test_flex_dispatcher_deepep_backend_requires_tpxep_larger_than_one(single_rank_groups):
    with pytest.raises(AssertionError, match="DeepEP dispatcher requires TPxEP > 1"):
        MoEFlexTokenDispatcher(
            8,
            list(range(8)),
            config=_moe_config(moe_token_dispatcher_type="flex"),
            pg_collection=single_rank_groups,
        )


def test_flex_dispatcher_deepep_backend_raises_import_error_when_deepep_missing(model_parallel):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    with pytest.raises(ImportError, match="DeepEP is not installed"):
        MoEFlexTokenDispatcher(
            4,
            list(range(4)),
            config=_flex_config(),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )


def test_flex_dispatcher_hybridep_backend_raises_import_error_when_hybrid_ep_missing(
    model_parallel,
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    with pytest.raises(ImportError, match="HybridEP is not installed"):
        MoEFlexTokenDispatcher(
            4,
            list(range(4)),
            config=_flex_config(moe_flex_dispatcher_backend="hybridep"),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )


def test_flex_deepep_backend_wires_manager_and_cudagraph_attrs(model_parallel, flex_deepep_stubs):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    dispatcher = MoEFlexTokenDispatcher(
        4,
        list(range(4)),
        config=_flex_config(),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    assert isinstance(dispatcher._comm_manager, _DeepepManager)
    assert dispatcher.cudagraph_attrs == [
        '_comm_manager.token_probs',
        '_comm_manager.token_indices',
    ]
    assert dispatcher.check_over_budget() is None


def test_flex_dispatch_preprocess_expands_routing_map_over_tpxep_ranks(
    model_parallel, flex_deepep_stubs
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    dispatcher = MoEFlexTokenDispatcher(
        4,
        list(range(4)),
        config=_flex_config(),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    hidden = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8)
    routing_map = torch.zeros(4, 8, dtype=torch.bool)
    routing_map[torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 4, 6])] = True
    routing_map[torch.tensor([0, 1, 2, 3]), torch.tensor([2, 3, 0, 3])] = True
    probs = torch.tensor(
        [
            [0.1, 0.6, 0.2, 0.05, 0.03, 0.02, 0.0, 0.0],
            [0.05, 0.05, 0.7, 0.1, 0.05, 0.0, 0.05, 0.0],
            [0.2, 0.1, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05],
            [0.0, 0.15, 0.05, 0.25, 0.05, 0.05, 0.35, 0.05],
        ]
    )

    flat, token_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)

    assert torch.equal(flat, hidden.view(4, 8))
    expected_probs = torch.tensor([[0.6, 0.2], [0.7, 0.1], [0.4, 0.2], [0.35, 0.25]])
    assert torch.equal(token_probs, expected_probs)
    expected_indices = torch.tensor([[1, 2], [2, 3], [4, 0], [6, 3]])
    assert torch.equal(dispatcher._comm_manager.token_indices, expected_indices)


def test_flex_deepep_roundtrip_permutes_experts_and_combines_back(
    model_parallel, flex_deepep_stubs
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    dispatcher = MoEFlexTokenDispatcher(
        4,
        list(range(4)),
        config=_flex_config(),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    hidden = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8)
    routing_map = torch.zeros(4, 8, dtype=torch.bool)
    routing_map[torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 4, 6])] = True
    probs = torch.tensor(
        [
            [0.1, 0.6, 0.2, 0.05, 0.03, 0.02, 0.0, 0.0],
            [0.05, 0.05, 0.7, 0.1, 0.05, 0.0, 0.05, 0.0],
            [0.2, 0.1, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05],
            [0.0, 0.15, 0.05, 0.25, 0.05, 0.05, 0.35, 0.05],
        ]
    )

    flat, _ = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
    dispatched, dispatched_probs = dispatcher.token_dispatch(flat, None)
    assert torch.equal(dispatched, flat * 2)
    assert torch.equal(
        dispatched_probs, torch.tensor([[0.5, 0.25], [0.75, 0.0], [0.1, 0.9], [0.3, 0.7]])
    )

    sorted_tokens, tokens_per_expert, sorted_probs = dispatcher.dispatch_postprocess(
        dispatched, dispatched_probs
    )
    assert torch.equal(tokens_per_expert, torch.tensor([2, 2, 2, 1]))
    assert torch.equal(sorted_tokens, dispatched[[0, 2, 0, 3, 1, 3, 2]])
    assert torch.equal(sorted_probs, torch.tensor([0.5, 0.9, 0.25, 0.3, 0.75, 0.7, 0.1]))

    expert_output = sorted_tokens * 2
    restored = dispatcher.combine_preprocess(expert_output)
    assert torch.equal(restored, flat * torch.tensor([[8.0], [4.0], [8.0], [8.0]]))

    combined = dispatcher.token_combine(restored)
    assert torch.equal(combined, restored * 3)

    output = dispatcher.combine_postprocess(combined)
    assert output.shape == (2, 2, 8)
    assert torch.equal(output, (restored * 3).view(2, 2, 8))
    assert len(flex_deepep_stubs["combine"]) == 1
    assert flex_deepep_stubs["combine"][0]["async_finish"] is True
