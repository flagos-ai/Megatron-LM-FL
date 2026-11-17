# Copyright (c) 2026, BAAI. All rights reserved.
"""DeepEP manager metadata: topk routing conversion, capacity-prob masking, multihot restoration, quantization padding and permute/unpermute bookkeeping around the absent deep_ep kernels."""

import logging

import pytest
import torch

import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager, _DispatchManager
from megatron.core.transformer.transformer_config import TransformerConfig


class _RecordingGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _manager_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="flex",
        moe_deepep_num_sms=24,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.fixture
def deepep_stubs(monkeypatch):
    calls = {"dispatch": [], "combine": [], "num_sms": []}
    handle = object()
    state = {
        "indices": torch.tensor([[0, 1], [2, -1], [3, 0], [1, 2]]),
        "probs": torch.tensor([[0.5, 0.25], [0.75, 0.0], [0.1, 0.9], [0.3, 0.7]]),
        "tokens_per_expert": torch.tensor([2, 2, 2, 1]),
        "handle": handle,
    }

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
            state["indices"],
            state["probs"],
            state["tokens_per_expert"],
            state["handle"],
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
    calls["state"] = state
    return calls


def _make_manager(deepep_stubs, **config_overrides):
    return _DeepepManager(
        group=_RecordingGroup(2),
        num_local_experts=4,
        router_topk=2,
        num_experts=8,
        config=_manager_config(**config_overrides),
    )


def _topk_probs():
    return torch.tensor(
        [
            [0.1, 0.6, 0.2, 0.05, 0.03, 0.02, 0.0, 0.0],
            [0.05, 0.05, 0.7, 0.1, 0.05, 0.0, 0.05, 0.0],
            [0.2, 0.1, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05],
            [0.0, 0.15, 0.05, 0.25, 0.05, 0.05, 0.35, 0.05],
        ]
    )


def test_deepep_manager_init_registers_deep_ep_sm_count_and_metadata_defaults(deepep_stubs):
    manager = _make_manager(deepep_stubs)
    assert deepep_stubs["num_sms"] == [24]
    assert manager.router_dtype is None
    assert manager.capacity_factor is None
    assert manager.permute_fusion is False
    assert manager.handle is None
    assert manager.token_indices is None
    assert manager.token_probs is None


def test_deepep_setup_metadata_selects_topk_probs_and_indices(deepep_stubs):
    manager = _make_manager(deepep_stubs)
    routing_map = torch.zeros(4, 8, dtype=torch.bool)
    probs = _topk_probs()

    manager.setup_metadata(routing_map, probs)

    assert torch.equal(
        manager.token_probs, torch.tensor([[0.6, 0.2], [0.7, 0.1], [0.4, 0.2], [0.35, 0.25]])
    )
    assert torch.equal(manager.token_indices, torch.tensor([[1, 2], [2, 3], [4, 0], [6, 3]]))


def test_deepep_setup_metadata_masks_dropped_tokens_with_capacity_factor(deepep_stubs):
    manager = _make_manager(deepep_stubs, moe_expert_capacity_factor=1.0)
    routing_map = torch.zeros(2, 8, dtype=torch.bool)
    probs = torch.tensor(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.2, 0.0, 0.0, 0.0, 0.15, 0.0]]
    )

    manager.setup_metadata(routing_map, probs)

    assert torch.equal(manager.token_indices, torch.tensor([[0, -1], [2, 6]]))
    assert torch.equal(manager.token_probs, torch.tensor([[0.5, 0.0], [0.2, 0.15]]))


def test_deepep_dispatch_records_handle_and_returns_kernel_output(deepep_stubs):
    manager = _make_manager(deepep_stubs)
    manager.setup_metadata(torch.zeros(4, 8, dtype=torch.bool), _topk_probs())
    hidden = torch.randn(4, 8)

    dispatched = manager.dispatch(hidden)

    call = deepep_stubs["dispatch"][0]
    assert call["hidden_states"] is hidden
    assert call["num_experts"] == 8
    assert call["group"].size() == 2
    assert call["async_finish"] is False
    assert call["allocate_on_comm_stream"] is False
    assert torch.equal(
        call["token_probs"], torch.tensor([[0.6, 0.2], [0.7, 0.1], [0.4, 0.2], [0.35, 0.25]])
    )
    assert torch.equal(call["token_indices"], torch.tensor([[1, 2], [2, 3], [4, 0], [6, 3]]))
    assert torch.equal(dispatched, hidden * 2)
    assert manager.handle is deepep_stubs["state"]["handle"]
    assert torch.equal(manager.tokens_per_expert, torch.tensor([2, 2, 2, 1]))
    assert torch.equal(manager.dispatched_indices, torch.tensor([[0, 1], [2, -1], [3, 0], [1, 2]]))


def test_deepep_dispatch_casts_bf16_probs_to_float32_with_warning(deepep_stubs, caplog):
    manager = _make_manager(deepep_stubs)
    manager.setup_metadata(torch.zeros(4, 8, dtype=torch.bool), _topk_probs().to(torch.bfloat16))
    hidden = torch.randn(4, 8)

    with caplog.at_level(logging.WARNING, logger="megatron.core.transformer.moe.token_dispatcher"):
        manager.dispatch(hidden)

    assert "DeepEP only supports float32 probs" in caplog.text
    assert deepep_stubs["dispatch"][0]["token_probs"].dtype == torch.float32
    assert manager.token_probs.dtype == torch.float32


def test_deepep_indices_to_multihot_builds_routing_map_and_probs(deepep_stubs):
    manager = _make_manager(deepep_stubs)
    indices = torch.tensor([[0, 1], [2, -1], [3, 0], [1, 2]])
    probs = torch.tensor([[0.5, 0.25], [0.75, 0.0], [0.1, 0.9], [0.3, 0.7]])

    routing_map, multihot_probs = manager._indices_to_multihot(indices, probs)

    assert torch.equal(
        routing_map,
        torch.tensor(
            [
                [True, True, False, False],
                [False, False, True, False],
                [True, False, False, True],
                [False, True, True, False],
            ]
        ),
    )
    assert torch.equal(
        multihot_probs,
        torch.tensor(
            [
                [0.5, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.75, 0.0],
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 0.3, 0.7, 0.0],
            ]
        ),
    )


def test_deepep_pad_routing_map_rounds_each_expert_to_quantization_alignment(deepep_stubs):
    manager = _make_manager(deepep_stubs, use_transformer_engine_op_fuser=True)
    routing_map = torch.zeros(256, 2, dtype=torch.bool)
    routing_map[:128, 0] = True
    routing_map[128:, 1] = True

    padded_map, padded_tokens_per_expert = manager._pad_routing_map(
        routing_map, torch.tensor([128, 128])
    )

    assert padded_tokens_per_expert.tolist() == [256, 256]
    assert padded_map.shape == (256, 2)
    assert padded_map.sum().item() == 512
    assert padded_map[:, 0].all()
    assert padded_map[:, 1].all()


def test_deepep_pad_routing_map_keeps_counts_when_tokens_insufficient(deepep_stubs, caplog):
    manager = _make_manager(deepep_stubs, use_transformer_engine_op_fuser=True)
    routing_map = torch.zeros(256, 2, dtype=torch.bool)
    routing_map[:128, 0] = True
    routing_map[128:, 1] = True

    with caplog.at_level(logging.WARNING, logger="megatron.core.transformer.moe.token_dispatcher"):
        padded_map, padded_tokens_per_expert = manager._pad_routing_map(
            routing_map, torch.tensor([300, 10])
        )

    assert "Not enough tokens to pad" in caplog.text
    assert padded_map is routing_map
    assert padded_tokens_per_expert.tolist() == [300, 10]


def test_deepep_get_permuted_hidden_states_reorders_rows_by_expert(deepep_stubs):
    manager = _make_manager(deepep_stubs, moe_router_dtype="fp64")
    manager.setup_metadata(torch.zeros(4, 8, dtype=torch.bool), _topk_probs())
    hidden = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

    dispatched = manager.dispatch(hidden)
    permuted, permuted_probs = manager.get_permuted_hidden_states_by_experts(dispatched)

    assert torch.equal(permuted, dispatched[[0, 2, 0, 3, 1, 3, 2]])
    assert permuted_probs.dtype == torch.float64
    assert torch.equal(
        permuted_probs,
        torch.tensor([0.5, 0.9, 0.25, 0.3, 0.75, 0.7, 0.1], dtype=torch.float32).to(torch.float64),
    )
    assert manager.hidden_shape_before_permute == (4, 8)
    assert torch.equal(manager.tokens_per_expert, torch.tensor([2, 2, 2, 1]))


def test_deepep_get_restored_hidden_states_unpermutes_expert_outputs(deepep_stubs):
    manager = _make_manager(deepep_stubs)
    manager.setup_metadata(torch.zeros(4, 8, dtype=torch.bool), _topk_probs())
    hidden = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

    dispatched = manager.dispatch(hidden)
    permuted, _ = manager.get_permuted_hidden_states_by_experts(dispatched)
    expert_output = permuted * 2

    restored = manager.get_restored_hidden_states_by_experts(expert_output)

    assert torch.equal(restored, hidden * torch.tensor([[8.0], [4.0], [8.0], [8.0]]))


def test_deepep_combine_releases_handle_and_metadata(deepep_stubs):
    manager = _make_manager(deepep_stubs)
    manager.setup_metadata(torch.zeros(4, 8, dtype=torch.bool), _topk_probs())
    hidden = torch.randn(4, 8)
    dispatched = manager.dispatch(hidden)

    combined = manager.combine(dispatched)

    call = deepep_stubs["combine"][0]
    assert call["hidden_states"] is dispatched
    assert call["handle"] is deepep_stubs["state"]["handle"]
    assert call["async_finish"] is False
    assert torch.equal(combined, dispatched * 3)
    assert manager.handle is None
    assert manager.dispatched_indices is None
    assert manager.dispatched_probs is None


@pytest.mark.parametrize(
    "method_name,args",
    [
        pytest.param("setup_metadata", (None, None), id="setup_metadata"),
        pytest.param("dispatch", (None,), id="dispatch"),
        pytest.param("combine", (None,), id="combine"),
        pytest.param("get_permuted_hidden_states_by_experts", (None,), id="permuted-by-experts"),
        pytest.param("get_restored_hidden_states_by_experts", (None,), id="restored-by-experts"),
    ],
)
def test_dispatch_manager_abstract_bodies_are_documented_noops(deepep_stubs, method_name, args):
    manager = _make_manager(deepep_stubs)
    assert getattr(_DispatchManager, method_name)(manager, *args) is None
