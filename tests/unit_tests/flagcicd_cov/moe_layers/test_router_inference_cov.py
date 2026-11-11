# Copyright (c) 2026, BAAI. All rights reserved.
"""InferenceTopKRouter behaviour: constructor guards, normalized dense top-k probabilities and
fallback to the training router when the inference engine is inactive."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.router as router_module
from megatron.core.inference.utils import InferenceMode
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _cuda_or_cpu_device():
    """cuda on the A100 CI node, cpu on a laptop: what the real platform reports."""
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _router_config(**overrides):
    """Router config following the developer test conventions."""
    defaults = dict(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.0,
        moe_router_dtype='fp32',
        add_bias_linear=False,
        use_cpu_initialization=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def pinned_device(monkeypatch):
    """Pin the router's platform object to one device for both run environments."""
    device = _cuda_or_cpu_device()
    monkeypatch.setattr(
        router_module, 'cur_platform', SimpleNamespace(current_device=lambda: device)
    )
    return device


class TestInferenceTopKRouterConstructor:

    def test_init_rejects_group_limited_routing(self, model_parallel, pinned_device):
        model_parallel()
        config = _router_config(moe_router_group_topk=2, moe_router_num_groups=2)

        with pytest.raises(
            AssertionError, match='InferenceTopKRouter requires moe_router_num_groups=None'
        ):
            InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection())

    def test_init_rejects_score_functions_outside_sigmoid_softmax(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        config = _router_config(moe_router_score_function='sqrtsoftplus')

        with pytest.raises(AssertionError, match='requires moe_router_score_function in'):
            InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection())


class TestInferenceTopKRouterForward:

    def test_forward_dense_returns_normalized_sigmoid_topk_probabilities(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        router = InferenceTopKRouter(
            config=_router_config(moe_router_score_function='sigmoid'),
            pg_collection=get_default_pg_collection(),
        )
        router.load_state_dict({'weight': torch.eye(4, 8)}, strict=False)
        hidden = (torch.arange(32, dtype=torch.float32) % 8).reshape(4, 1, 8).to(pinned_device)

        probs, top_indices = router._forward(hidden)

        logits = hidden.squeeze(1)[:, :4]
        scores = torch.sigmoid(logits)
        ref_values, ref_indices = torch.topk(scores, k=2, dim=1)
        expected_probs = ref_values / ref_values.sum(dim=-1, keepdim=True)
        assert probs.shape == (4, 2)
        assert top_indices.dtype == torch.int64
        assert torch.equal(top_indices, ref_indices)
        torch.testing.assert_close(probs, expected_probs, rtol=1e-6, atol=1e-7)

    def test_forward_delegates_to_training_router_when_inference_mode_is_inactive(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        config = _router_config(moe_router_score_function='sigmoid')
        pg = get_default_pg_collection()
        router = InferenceTopKRouter(config=config, pg_collection=pg)
        reference_router = TopKRouter(config=config, pg_collection=pg)
        weight = torch.eye(4, 8)
        router.load_state_dict({'weight': weight}, strict=False)
        reference_router.load_state_dict({'weight': weight}, strict=False)
        hidden = (torch.arange(64, dtype=torch.float32) % 8).reshape(4, 2, 8).to(pinned_device)

        probs, routing_map = router.forward(hidden)
        reference_probs, reference_map = reference_router(hidden)

        assert probs.shape == (8, 4)
        assert routing_map.shape == (8, 4)
        assert (routing_map.sum(dim=1) == 2).all()
        torch.testing.assert_close(probs, reference_probs, rtol=1e-6, atol=1e-7)
        assert torch.equal(routing_map, reference_map)

    def test_forward_returns_dense_topk_when_inference_mode_is_active(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        router = InferenceTopKRouter(
            config=_router_config(moe_router_score_function='sigmoid'),
            pg_collection=get_default_pg_collection(),
        )
        router.load_state_dict({'weight': torch.eye(4, 8)}, strict=False)
        hidden = (torch.arange(32, dtype=torch.float32) % 8).reshape(4, 1, 8).to(pinned_device)

        with InferenceMode.active():
            probs, top_indices = router.forward(hidden)

        logits = hidden.squeeze(1)[:, :4]
        scores = torch.sigmoid(logits)
        ref_values, ref_indices = torch.topk(scores, k=2, dim=1)
        expected_probs = ref_values / ref_values.sum(dim=-1, keepdim=True)
        assert probs.shape == (4, 2)
        assert top_indices.shape == (4, 2)
        assert torch.equal(top_indices, ref_indices)
        torch.testing.assert_close(probs, expected_probs, rtol=1e-6, atol=1e-7)


class TestForceBiasedForward:
    """Needs the CUDA RNG tracker: runs on the A100 CI node."""

    def test_forward_replaces_deterministic_routing_with_seeded_noise_when_force_biased(
        self, model_parallel
    ):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.device('cuda', torch.cuda.current_device())
        router = TopKRouter(
            config=_router_config(
                moe_router_force_biased=0.1,
                moe_router_score_function='sigmoid',
                moe_router_load_balancing_type='none',
            ),
            pg_collection=get_default_pg_collection(),
        )
        router.load_state_dict({'weight': torch.eye(4, 8)}, strict=False)
        router = router.to(device)
        router.eval()
        hidden = (torch.arange(128, dtype=torch.float32) % 8).reshape(8, 2, 8).to(device)

        with torch.no_grad():
            probs_biased, map_biased = router(hidden)

        selected = probs_biased.masked_select(map_biased).reshape(-1, 2)
        assert (map_biased.sum(dim=1) == 2).all()
        torch.testing.assert_close(
            selected.sum(dim=1), torch.ones(16, device=device), rtol=1e-5, atol=1e-6
        )

        router.config.moe_router_force_biased = None
        with torch.no_grad():
            probs_plain, map_plain = router(hidden)

        expected_map = torch.zeros(16, 4, dtype=torch.bool, device=device)
        expected_map[:, 2] = True
        expected_map[:, 3] = True
        assert torch.equal(map_plain, expected_map)
        assert not torch.equal(map_biased, map_plain)
