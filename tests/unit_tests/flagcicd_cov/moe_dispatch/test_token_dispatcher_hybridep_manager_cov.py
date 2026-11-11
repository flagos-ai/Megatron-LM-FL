# Copyright (c) 2026, BAAI. All rights reserved.
"""HybridEP manager: rank-budget sizing in setup_metadata, drop-and-pad capacity, overflow accumulation from dispatch handles, handle release in combine, and flex dispatcher wiring."""

import logging

import pytest
import torch

import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher, _HybridEPManager
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RecordingGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup

    Utils.destroy_model_parallel()


@pytest.fixture
def hybridep_stubs(monkeypatch):
    calls = {"dispatch": [], "combine": []}
    state = {"handle": torch.tensor([0.0, 1.0])}

    def _fake_hybrid_ep_dispatch(**kwargs):
        calls["dispatch"].append(kwargs)
        return (
            kwargs["x"] * 2,
            kwargs["probs"] * 0.5,
            None,
            torch.tensor([2.0, 4.0]),
            state["handle"],
        )

    def _fake_hybrid_ep_combine(**kwargs):
        calls["combine"].append(kwargs)
        return kwargs["x"] * 3

    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", _fake_hybrid_ep_dispatch)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_combine", _fake_hybrid_ep_combine)
    calls["state"] = state
    return calls


def _hybridep_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _make_manager(hybridep_stubs, **config_overrides):
    return _HybridEPManager(
        group=_RecordingGroup(2),
        num_local_experts=4,
        num_experts=8,
        config=_hybridep_config(**config_overrides),
    )


def _routing_inputs():
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
    return routing_map, probs


def test_hybridep_manager_init_registers_capacity_settings_and_zero_budget(hybridep_stubs):
    manager = _make_manager(hybridep_stubs)
    assert manager.capacity_factor is None
    assert manager.drop_and_pad is False
    assert manager.num_permuted_tokens is None
    assert manager.handle is None
    assert manager.pad_multiple is None
    assert manager.moe_expert_rank_capacity_factor is None
    assert manager.over_budget.device.type == "cuda"
    assert manager.over_budget.item() == 0


def test_hybridep_setup_metadata_rounds_rank_budget_up_to_alignment(hybridep_stubs):
    manager = _make_manager(
        hybridep_stubs, use_transformer_engine_op_fuser=True, moe_expert_rank_capacity_factor=1.5
    )
    routing_map, probs = _routing_inputs()

    manager.setup_metadata(routing_map, probs)

    assert manager.num_permuted_tokens == 256
    assert torch.equal(manager.routing_map, routing_map)
    assert torch.equal(manager.token_probs, probs)


def test_hybridep_setup_metadata_computes_capacity_when_drop_and_pad(hybridep_stubs):
    manager = _make_manager(
        hybridep_stubs, moe_pad_expert_input_to_capacity=True, moe_expert_capacity_factor=1.0
    )
    routing_map, probs = _routing_inputs()

    manager.setup_metadata(routing_map, probs)

    assert manager.capacity == 1
    assert manager.num_permuted_tokens == 1 * 2 * 4
    assert torch.equal(manager.tokens_per_expert, torch.full((4,), 2, dtype=torch.long))


def test_hybridep_dispatch_accumulates_over_budget_from_overflow_handle(hybridep_stubs):
    manager = _make_manager(
        hybridep_stubs, use_transformer_engine_op_fuser=True, moe_expert_rank_capacity_factor=1.5
    )
    routing_map, probs = _routing_inputs()
    manager.setup_metadata(routing_map, probs)
    hidden = torch.randn(4, 8)

    dispatched = manager.dispatch(hidden)

    assert torch.equal(dispatched, hidden * 2)
    assert manager.over_budget.item() == 1
    assert manager.num_permuted_tokens == 256
    assert manager.tokens_per_expert.dtype == torch.int64
    assert torch.equal(manager.tokens_per_expert, torch.tensor([2, 4]))
    call = hybridep_stubs["dispatch"][0]
    assert call["x"] is hidden
    assert call["group"].size() == 2
    assert call["num_local_experts"] == 4
    assert call["num_permuted_tokens"] == 256
    assert call["pad_multiple"] is None
    assert call["fused"] is False
    assert call["num_sms_preprocessing_api"] == 108


def test_hybridep_dispatch_resolves_num_permuted_tokens_when_unbounded(hybridep_stubs):
    manager = _make_manager(hybridep_stubs)
    routing_map, probs = _routing_inputs()
    manager.setup_metadata(routing_map, probs)
    hidden = torch.randn(4, 8)

    manager.dispatch(hidden)

    assert manager.num_permuted_tokens.item() == 6
    assert torch.equal(manager.tokens_per_expert, torch.tensor([2, 4], dtype=torch.int64))


def test_hybridep_dispatch_casts_bf16_probs_to_float32_with_warning(hybridep_stubs, caplog):
    manager = _make_manager(hybridep_stubs)
    routing_map, probs = _routing_inputs()
    manager.setup_metadata(routing_map, probs.to(torch.bfloat16))
    hidden = torch.randn(4, 8)

    with caplog.at_level(logging.WARNING, logger="megatron.core.transformer.moe.token_dispatcher"):
        dispatched = manager.dispatch(hidden)

    assert "HybridEP only supports float32 probs" in caplog.text
    assert hybridep_stubs["dispatch"][0]["probs"].dtype == torch.float32
    assert torch.equal(hybridep_stubs["dispatch"][0]["probs"], probs.to(torch.bfloat16).float())
    assert manager.token_probs.dtype == torch.float32
    assert torch.equal(dispatched, hidden * 2)


def test_hybridep_dispatch_sets_pad_multiple_when_fp8_enabled(hybridep_stubs):
    manager = _make_manager(hybridep_stubs, fp8=True)
    routing_map, probs = _routing_inputs()
    manager.setup_metadata(routing_map, probs)
    hidden = torch.randn(4, 8)

    manager.dispatch(hidden)

    assert manager.pad_multiple == 16
    assert hybridep_stubs["dispatch"][0]["pad_multiple"] == 16


def test_hybridep_combine_releases_handle_and_budget_between_steps(hybridep_stubs):
    manager = _make_manager(hybridep_stubs)
    routing_map, probs = _routing_inputs()
    manager.setup_metadata(routing_map, probs)
    hidden = torch.randn(4, 8)
    manager.dispatch(hidden)

    combined = manager.combine(hidden)

    assert torch.equal(combined, hidden * 3)
    assert manager.handle is None
    assert manager.num_permuted_tokens is None
    call = hybridep_stubs["combine"][0]
    assert call["x"] is hidden
    assert call["handle"] is hybridep_stubs["state"]["handle"]
    assert call["num_permuted_tokens"].item() == 6
    assert call["pad_multiple"] is None
    assert call["fused"] is False


def test_flex_hybridep_backend_wires_manager_cudagraph_attrs_and_budget(
    model_parallel, hybridep_stubs
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    dispatcher = MoEFlexTokenDispatcher(
        4,
        list(range(4)),
        config=_hybridep_config(
            use_transformer_engine_op_fuser=True, moe_expert_rank_capacity_factor=1.5
        ),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    assert isinstance(dispatcher._comm_manager, _HybridEPManager)
    assert dispatcher.cudagraph_attrs == ['_comm_manager.token_probs', '_comm_manager.routing_map']
    assert dispatcher.check_over_budget().item() == 0

    routing_map = torch.zeros(4, 8, dtype=torch.bool)
    routing_map[torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 4, 6])] = True
    dispatcher._comm_manager.setup_metadata(routing_map, _routing_inputs()[1])
    dispatcher._comm_manager.dispatch(torch.ones(4, 8))
    assert dispatcher.check_over_budget().item() == 1

    dispatcher.reset_over_budget()
    assert dispatcher.check_over_budget().item() == 0
