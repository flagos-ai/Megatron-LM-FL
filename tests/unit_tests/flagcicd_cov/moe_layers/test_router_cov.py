# Copyright (c) 2026, BAAI. All rights reserved.
"""TopKRouter behaviour: biased gating projection, eval-mode sinkhorn top-k, MTP-scaled aux and z
losses, expert-bias token counting, seeded input jitter and hash-table routing."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.router as router_module
from megatron.core.transformer.moe import moe_logging
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, get_default_pg_collection
from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay
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


class _DelegatingRouter(Router):
    """Concrete router that calls the abstract base implementations."""

    def routing(self, logits):
        return super().routing(logits)

    def forward(self, input):
        return super().forward(input)


@pytest.fixture
def replay_registry():
    """Snapshot of the global RouterReplay registry; new instances are removed afterwards."""
    original = list(RouterReplay.global_router_replay_instances)
    yield original
    RouterReplay.global_router_replay_instances[:] = original


class TestRouterBaseContract:

    def test_routing_raises_not_implemented_on_base_router(self, model_parallel, pinned_device):
        model_parallel()
        router = _DelegatingRouter(
            config=_router_config(), pg_collection=get_default_pg_collection()
        )

        with pytest.raises(NotImplementedError, match='Routing function not implemented.'):
            router.routing(torch.zeros(3, 4))

    def test_forward_raises_not_implemented_on_base_router(self, model_parallel, pinned_device):
        model_parallel()
        router = _DelegatingRouter(
            config=_router_config(), pg_collection=get_default_pg_collection()
        )

        with pytest.raises(NotImplementedError, match='Forward function not implemented.'):
            router.forward(torch.zeros(3, 2, 8))


class TestGatingAndJitter:

    def test_gating_adds_bias_to_linear_projection(self, model_parallel, pinned_device):
        model_parallel()
        router = TopKRouter(
            config=_router_config(add_bias_linear=True), pg_collection=get_default_pg_collection()
        )
        weight = torch.eye(4, 8)
        bias = torch.tensor([0.25, -0.5, 0.125, 0.0])
        router.load_state_dict({'weight': weight, 'bias': bias})
        hidden = (torch.arange(32, dtype=torch.float32) % 8).reshape(2, 2, 8).to(pinned_device)

        logits = router.gating(hidden)

        expected = hidden[:, :, :4] + bias.to(pinned_device)
        assert logits.shape == (2, 2, 4)
        torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-6)
        assert router.weight.device.type == pinned_device.type

    def test_apply_input_jitter_multiplies_input_by_seeded_uniform_noise(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        eps = 0.1
        router = TopKRouter(
            config=_router_config(moe_input_jitter_eps=eps),
            pg_collection=get_default_pg_collection(),
        )
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) + 1.0

        torch.manual_seed(1234)
        first = router.apply_input_jitter(hidden)

        torch.manual_seed(1234)
        low = torch.tensor(1.0 - eps, dtype=hidden.dtype)
        high = torch.tensor(1.0 + eps, dtype=hidden.dtype)
        expected_first = hidden * (
            low + torch.rand(hidden.shape, dtype=hidden.dtype) * (high - low)
        )
        torch.testing.assert_close(first, expected_first, rtol=1e-6, atol=1e-6)
        sampler = router.input_jitter

        torch.manual_seed(4321)
        second = router.apply_input_jitter(hidden)

        torch.manual_seed(4321)
        expected_second = hidden * (
            low + torch.rand(hidden.shape, dtype=hidden.dtype) * (high - low)
        )
        torch.testing.assert_close(second, expected_second, rtol=1e-6, atol=1e-6)
        assert not torch.equal(first, second)
        assert router.input_jitter is sampler


class TestConstructor:

    def test_constructor_registers_replay_instance_when_routing_replay_enabled(
        self, model_parallel, pinned_device, replay_registry
    ):
        model_parallel()
        router = TopKRouter(
            config=_router_config(moe_enable_routing_replay=True),
            pg_collection=get_default_pg_collection(),
        )

        new_instances = [
            instance
            for instance in RouterReplay.global_router_replay_instances
            if instance not in replay_registry
        ]

        assert new_instances == [router.router_replay]
        assert router.router_replay.router_replay_action is None


class TestSinkhornLoadBalancing:

    def test_sinkhorn_load_balancing_softmaxes_scores_and_masks_to_topk_in_eval_mode(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        router = TopKRouter(
            config=_router_config(
                moe_router_load_balancing_type='sinkhorn', moe_router_score_function='sigmoid'
            ),
            pg_collection=get_default_pg_collection(),
        )
        router.eval()
        logits = torch.tensor([[2.0, 1.0, 0.5, -0.5], [0.0, 3.0, 1.0, -1.0]])

        scores, routing_map = router.sinkhorn_load_balancing(logits)

        expected_map = torch.tensor([[True, True, False, False], [False, True, True, False]])
        assert torch.equal(routing_map, expected_map)
        assert torch.equal(scores * routing_map, scores)
        activated = torch.softmax(logits, dim=-1, dtype=torch.float32)
        expected_scores = activated * expected_map
        torch.testing.assert_close(scores, expected_scores, rtol=1e-6, atol=1e-7)


class TestAuxLossAttachment:

    def test_attach_and_log_load_balancing_loss_scales_mtp_loss_by_valid_token_count(
        self, model_parallel, pinned_device, monkeypatch
    ):
        model_parallel()
        monkeypatch.setattr(moe_logging, '_MOE_METRICS_TRACKER', None)
        monkeypatch.setattr(MoEAuxLossAutoScaler, 'main_loss_backward_scale', None)
        router = TopKRouter(
            config=_router_config(
                calculate_per_token_loss=True, mtp_num_layers=2, mtp_use_repeated_layer=True
            ),
            pg_collection=get_default_pg_collection(),
            is_mtp_layer=True,
            layer_number=1,
        )
        activation = torch.ones(3, requires_grad=True)
        aux_loss = torch.tensor(4.0, requires_grad=True)

        output = router.attach_and_log_load_balancing_loss(
            activation,
            0.5,
            aux_loss,
            'load_balancing_loss',
            reduce_group=get_default_pg_collection().tp_cp,
            valid_token_count=torch.tensor(5.0),
        )

        output.sum().backward()
        entry = moe_logging.get_moe_metrics_tracker().metrics['load_balancing_loss']
        assert torch.equal(entry.values, torch.tensor([0.0, 0.0, 0.0, 0.0, 4.0, 0.0]))
        assert entry.needs_dp_avg is True
        assert torch.equal(activation.grad, torch.ones(3))
        assert torch.equal(aux_loss.grad, torch.tensor(2.5))

    def test_apply_z_loss_uses_valid_token_count_and_records_divided_mtp_loss(
        self, model_parallel, pinned_device, monkeypatch
    ):
        model_parallel()
        monkeypatch.setattr(moe_logging, '_MOE_METRICS_TRACKER', None)
        router = TopKRouter(
            config=_router_config(
                calculate_per_token_loss=True,
                moe_z_loss_coeff=1e-3,
                mtp_num_layers=2,
                mtp_use_repeated_layer=True,
            ),
            pg_collection=get_default_pg_collection(),
            is_mtp_layer=True,
            layer_number=1,
        )
        logits = torch.tensor([[2.0, 1.0], [0.5, -1.0], [3.0, 0.0]])
        padding_mask = torch.tensor([False, True, False])

        returned = router.apply_z_loss(logits, padding_mask=padding_mask)

        assert torch.equal(returned, logits)
        logsum = torch.logsumexp(logits, dim=-1)
        valid = ~padding_mask
        expected_z_loss = torch.square(logsum) * valid
        expected_z_loss = expected_z_loss.sum() / valid.sum() * 1e-3
        entry = moe_logging.get_moe_metrics_tracker().metrics['z_loss']
        torch.testing.assert_close(
            entry.values[4], expected_z_loss / 2 / 1e-3, rtol=1e-6, atol=1e-7
        )


class TestExpertBias:

    def test_apply_expert_bias_counts_routed_tokens_per_expert(self, model_parallel, pinned_device):
        model_parallel()
        router = TopKRouter(
            config=_router_config(
                moe_router_enable_expert_bias=True, moe_router_score_function='sigmoid'
            ),
            pg_collection=get_default_pg_collection(),
        )
        routing_map = torch.tensor(
            [
                [True, True, False, False],
                [False, True, True, False],
                [True, False, False, True],
                [False, False, True, True],
            ],
            device=pinned_device,
        )

        router._apply_expert_bias(routing_map)

        assert torch.equal(
            router.local_tokens_per_expert,
            torch.tensor([2.0, 2.0, 2.0, 2.0], device=router.local_tokens_per_expert.device),
        )

    def test_forward_restores_expert_bias_to_float32_after_half_precision_cast(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        router = TopKRouter(
            config=_router_config(
                moe_router_enable_expert_bias=True, moe_router_score_function='sigmoid'
            ),
            pg_collection=get_default_pg_collection(),
        )
        router.load_state_dict({'weight': torch.eye(4, 8)}, strict=False)
        router = router.to(torch.bfloat16)
        hidden = (
            (torch.arange(32, dtype=torch.float32) % 8)
            .reshape(2, 2, 8)
            .to(pinned_device, torch.bfloat16)
        )

        probs, routing_map = router(hidden)

        assert router.weight.dtype == torch.bfloat16
        assert router.expert_bias.dtype == torch.float32
        expected_counts = torch.tensor([0.0, 0.0, 4.0, 4.0], device=pinned_device)
        torch.testing.assert_close(
            router.local_tokens_per_expert.float(), expected_counts, rtol=0.0, atol=0.0
        )
        assert torch.equal(routing_map, probs > 0)


class TestHashRouting:

    def test_hash_routing_rejects_unsupported_score_function(self, model_parallel, pinned_device):
        model_parallel()
        router = TopKRouter(
            config=_router_config(
                moe_n_hash_layers=1, actual_vocab_size=128, moe_router_score_function='gelu'
            ),
            pg_collection=get_default_pg_collection(),
            layer_number=1,
        )

        with pytest.raises(ValueError, match='Invalid score_function: gelu'):
            router._hash_routing(
                torch.zeros(4, 4, device=pinned_device),
                torch.zeros(1, 4, dtype=torch.long, device=pinned_device),
            )

    def test_hash_routing_scales_normalized_probs_by_topk_scaling_factor(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        router = TopKRouter(
            config=_router_config(
                moe_n_hash_layers=1,
                actual_vocab_size=16,
                moe_router_score_function='sigmoid',
                moe_router_topk_scaling_factor=4.0,
            ),
            pg_collection=get_default_pg_collection(),
            layer_number=1,
        )
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.5, -0.5],
                [0.0, 3.0, 1.0, -1.0],
                [1.5, -0.5, 2.5, 0.0],
                [-1.0, 0.0, 0.5, 3.0],
            ],
            device=pinned_device,
        )
        input_ids = torch.tensor([[0], [3], [5], [7]], dtype=torch.long, device=pinned_device)

        routing_probs, routing_map = router._hash_routing(logits, input_ids)

        scores = torch.sigmoid(logits)
        flat_ids = input_ids.T.reshape(-1)
        top_indices = router.tid2eid[flat_ids].long()
        probs = scores.gather(1, top_indices)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20) * 4.0
        expected_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        expected_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        assert torch.equal(routing_map, expected_map)
        torch.testing.assert_close(routing_probs, expected_probs, rtol=1e-6, atol=1e-7)
        assert torch.equal(
            top_indices, torch.tensor([[0, 1], [3, 0], [1, 2], [3, 0]], device=pinned_device)
        )
