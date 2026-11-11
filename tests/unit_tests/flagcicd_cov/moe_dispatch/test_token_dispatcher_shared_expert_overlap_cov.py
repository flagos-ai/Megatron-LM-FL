# Copyright (c) 2026, BAAI. All rights reserved.
"""Shared-expert overlap: alltoall and flex dispatcher overlap hooks across dispatch and combine, plus the NVLS dispatch_preprocess side-stream launch joined in combine_postprocess."""

import pytest
import torch

import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP, SharedExpertState
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def overlap_class_state(monkeypatch):
    monkeypatch.setattr(
        MoEAlltoAllTokenDispatcher, "cuda_dtoh_stream", MoEAlltoAllTokenDispatcher.cuda_dtoh_stream
    )
    monkeypatch.setattr(SharedExpertMLP, "stream", SharedExpertMLP.stream)


@pytest.fixture
def flex_deepep_stubs(monkeypatch):
    handle = object()
    state = {
        # DeepEP returns its outputs on the current device.
        "indices": torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]], device="cuda"
        ),
        "probs": torch.full((8, 2), 0.25, device="cuda"),
        "tokens_per_expert": torch.tensor([4, 4, 4, 4], device="cuda"),
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
        return hidden_states * 3, None

    monkeypatch.setattr(token_dispatcher, "fused_dispatch", _fake_fused_dispatch)
    monkeypatch.setattr(token_dispatcher, "fused_combine", _fake_fused_combine)
    # DeepEP is absent on the CI image, so fused_a2a also leaves its SM-count setter as None.
    monkeypatch.setattr(token_dispatcher, "set_deepep_num_sms", lambda num_sms: None)
    return state


def _overlap_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_shared_expert_intermediate_size=32,
        moe_shared_expert_overlap=True,
        use_cpu_initialization=True,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _build_overlap_layer(gate, moe_token_dispatcher_type="alltoall", **config_overrides):
    config = _overlap_config(
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        moe_shared_expert_gate=gate,
        **config_overrides,
    )
    submodules = get_submodules(
        get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
    )
    layer = MoELayer(config, submodules).cuda()
    layer.set_layer_number(0)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return layer, pg_collection


def _routing_and_probs(num_tokens, num_experts):
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    probs = torch.zeros(num_tokens, num_experts)
    for token in range(num_tokens):
        for expert in ((2 * token) % num_experts, (2 * token + 1) % num_experts):
            routing_map[token, expert] = True
            probs[token, expert] = 0.25
    return routing_map, probs


def _shared_expert_reference(shared_experts, hidden_flat):
    fc1_out = torch.nn.functional.linear(hidden_flat, shared_experts.linear_fc1.weight)
    x_glu, x_linear = fc1_out.chunk(2, dim=-1)
    intermediate = torch.nn.functional.silu(x_glu) * x_linear
    output = torch.nn.functional.linear(intermediate, shared_experts.linear_fc2.weight)
    if shared_experts.use_shared_expert_gate:
        gate = torch.sigmoid(torch.nn.functional.linear(hidden_flat, shared_experts.gate_weight))
        output = output * gate
    return output


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(False, id="without-shared-expert-gate"),
        pytest.param(True, id="with-shared-expert-gate"),
    ],
)
def test_alltoall_set_shared_experts_appends_shared_expert_cudagraph_attrs(
    model_parallel, overlap_class_state, gate
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    torch.manual_seed(1234)
    layer, _ = _build_overlap_layer(gate=gate)

    dispatcher = layer.token_dispatcher
    assert dispatcher.shared_experts is layer.shared_experts
    assert dispatcher.use_nccl_stream is True
    assert "shared_experts.cached_fc1_input" in dispatcher.cudagraph_attrs
    assert (
        dispatcher.cudagraph_attrs.index("shared_experts.cached_fc1_input")
        == len(dispatcher.cudagraph_attrs) - 1
    )
    if gate:
        assert "shared_experts.gate_score" in dispatcher.cudagraph_attrs
    else:
        assert "shared_experts.gate_score" not in dispatcher.cudagraph_attrs


def test_alltoall_overlap_roundtrip_adds_shared_expert_output(model_parallel, overlap_class_state):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    torch.manual_seed(1234)
    layer, _ = _build_overlap_layer(gate=False)
    dispatcher = layer.token_dispatcher
    hidden = torch.randn(4, 2, 32, device="cuda")
    routing_map, probs = _routing_and_probs(8, 8)
    routing_map, probs = routing_map.cuda(), probs.cuda()

    with torch.no_grad():
        permuted, permuted_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
        assert permuted.shape == (16, 32)
        assert permuted_probs.shape == (16,)

        dispatched, dispatched_probs = dispatcher.token_dispatch(permuted, permuted_probs)
        sorted_tokens, tokens_per_expert, sorted_probs = dispatcher.dispatch_postprocess(
            dispatched, dispatched_probs
        )
        assert torch.equal(tokens_per_expert, torch.full((8,), 2))
        assert sorted_tokens.shape == (16, 32)

        expert_output = sorted_tokens * sorted_probs.unsqueeze(-1)
        restored = dispatcher.combine_preprocess(expert_output)
        combined = dispatcher.token_combine(restored)
        output = dispatcher.combine_postprocess(combined)

    reference = _shared_expert_reference(layer.shared_experts, hidden.view(8, 32)).view(4, 2, 32)
    torch.testing.assert_close(output, hidden * 0.5 + reference, rtol=1e-5, atol=1e-5)
    assert layer.shared_experts._overlap_state is SharedExpertState.IDLE


def test_flex_overlap_roundtrip_adds_shared_expert_output(
    model_parallel, overlap_class_state, flex_deepep_stubs
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    torch.manual_seed(1234)
    layer, _ = _build_overlap_layer(gate=False, moe_token_dispatcher_type="flex")
    dispatcher = layer.token_dispatcher
    assert dispatcher.use_nccl_stream is True
    hidden = torch.randn(4, 2, 32, device="cuda")
    routing_map, probs = _routing_and_probs(8, 8)
    routing_map, probs = routing_map.cuda(), probs.cuda()

    with torch.no_grad():
        flat, token_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
        assert flat.shape == (8, 32)
        assert token_probs.shape == (8, 2)

        dispatched, dispatched_probs = dispatcher.token_dispatch(flat, None)
        sorted_tokens, tokens_per_expert, sorted_probs = dispatcher.dispatch_postprocess(
            dispatched, dispatched_probs
        )
        assert tokens_per_expert.tolist() == [4, 4, 4, 4]
        assert sorted_tokens.shape == (16, 32)

        expert_output = sorted_tokens * sorted_probs.unsqueeze(-1)
        restored = dispatcher.combine_preprocess(expert_output)
        assert torch.equal(restored, flat)
        combined = dispatcher.token_combine(restored)
        output = dispatcher.combine_postprocess(combined)

    reference = _shared_expert_reference(layer.shared_experts, hidden.view(8, 32)).view(4, 2, 32)
    torch.testing.assert_close(output, hidden * 3 + reference, rtol=1e-5, atol=1e-5)
    assert layer.shared_experts._overlap_state is SharedExpertState.IDLE


def test_nvls_shared_expert_output_launched_on_side_stream_and_joined_in_postprocess(
    model_parallel, overlap_class_state
):
    model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    torch.manual_seed(1234)
    layer, pg_collection = _build_overlap_layer(gate=False)
    nvls_dispatcher = NVLSAllGatherVDispatcher(
        8, list(range(8)), config=layer.config, pg_collection=pg_collection
    )
    nvls_dispatcher.set_shared_experts(layer.shared_experts)
    hidden = torch.randn(4, 2, 32, device="cuda")
    routing_map, probs = _routing_and_probs(8, 8)
    routing_map, probs = routing_map.cuda(), probs.cuda()

    flat, returned_probs = nvls_dispatcher.dispatch_preprocess(hidden, routing_map, probs)
    assert flat.shape == (8, 32)
    assert returned_probs is probs
    assert nvls_dispatcher._local_tokens == 8
    assert nvls_dispatcher._shared_expert_output.shape == (4, 2, 32)
    torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
    reference = _shared_expert_reference(layer.shared_experts, hidden.view(8, 32)).view(4, 2, 32)
    torch.testing.assert_close(
        nvls_dispatcher._shared_expert_output, reference, rtol=1e-5, atol=1e-6
    )

    combined = nvls_dispatcher.token_combine(flat)
    assert combined.dtype == torch.bfloat16
    output = nvls_dispatcher.combine_postprocess(combined)
    assert nvls_dispatcher._shared_expert_output is None
    torch.testing.assert_close(
        output, combined.view(4, 2, 32).float() + reference, rtol=1e-5, atol=1e-6
    )
