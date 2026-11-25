# Copyright (c) 2026, BAAI. All rights reserved.
"""MoE router math and bookkeeping: switch load-balancing loss, expert capacity, top-k scoring, aux-loss tracker reduction, random logits bias, and quantization alignment."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import moe_logging, moe_utils


class _CpGroup:
    """Stand-in for a process group of the given size and rank."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.fixture
def single_rank_parallel_state(tmp_path):
    dist.init_process_group(
        backend='gloo', init_method=f'file://{tmp_path}/rendezvous', world_size=1, rank=0
    )
    parallel_state.initialize_model_parallel(1, 1)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture
def isolated_moe_tracker(monkeypatch):
    monkeypatch.setattr(moe_logging, '_MOE_METRICS_TRACKER', None)


@pytest.fixture
def identity_all_reduce(monkeypatch):
    # monkeypatch torch.distributed so the single-process result is the identity reduction
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor, group=None, op=None: tensor)


@pytest.fixture
def deterministic_algorithms():
    """Enable torch deterministic algorithms, restoring the previous mode afterwards."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def test_switch_load_balancing_loss_func_matches_closed_form():
    probs = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    tokens_per_expert = torch.tensor([3.0, 1.0])
    loss = moe_utils.switch_load_balancing_loss_func(
        probs, tokens_per_expert, total_num_tokens=2, topk=1, num_experts=2, moe_aux_loss_coeff=0.1
    )
    expected = 2 * 0.1 / (1 * 2 * 2) * ((0.6 + 0.3) * 3.0 + (0.4 + 0.7) * 1.0)
    torch.testing.assert_close(loss, torch.tensor(expected))


def test_switch_load_balancing_loss_func_zeroes_padding_tokens():
    probs = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
    tokens_per_expert = torch.tensor([1.0, 1.0])
    padding_mask = torch.tensor([True, False])
    loss = moe_utils.switch_load_balancing_loss_func(
        probs,
        tokens_per_expert,
        total_num_tokens=2,
        topk=1,
        num_experts=2,
        moe_aux_loss_coeff=0.1,
        padding_mask=padding_mask,
    )
    expected = 2 * 0.1 / (1 * 2 * 2) * (0.5 * 1.0 + 0.5 * 1.0)
    torch.testing.assert_close(loss, torch.tensor(expected))


def test_switch_load_balancing_loss_func_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_moe_aux_loss', None, raising=False)
    with pytest.raises(ValueError, match='fused_moe_aux_loss is not available'):
        moe_utils.switch_load_balancing_loss_func(
            torch.ones(2, 2), torch.ones(2), 2, 1, 2, 0.1, fused=True
        )


@pytest.mark.parametrize(
    ('num_tokens', 'num_experts', 'capacity_factor', 'min_capacity', 'expected'),
    [
        pytest.param(10, 4, 1.0, None, 3, id='rounded-up'),
        pytest.param(4, 8, 1.0, 2, 2, id='minimum-applied'),
        pytest.param(5, 2, 2.0, None, 5, id='exact'),
    ],
)
def test_get_capacity_ceil_divides_and_applies_minimum(
    num_tokens, num_experts, capacity_factor, min_capacity, expected
):
    assert (
        moe_utils.get_capacity(num_tokens, num_experts, capacity_factor, min_capacity) == expected
    )


def test_aux_loss_auto_scaler_backward_defaults_scale_to_one(monkeypatch):
    monkeypatch.setattr(moe_utils.MoEAuxLossAutoScaler, 'main_loss_backward_scale', None)
    output = torch.ones(2, requires_grad=True)
    aux_loss = torch.full((2,), 3.0, requires_grad=True)
    scaled = moe_utils.MoEAuxLossAutoScaler.apply(output, aux_loss)
    scaled.sum().backward()
    assert output.grad.tolist() == [1.0, 1.0]
    assert aux_loss.grad.tolist() == [1.0, 1.0]


def test_aux_loss_auto_scaler_backward_uses_configured_scale(monkeypatch):
    monkeypatch.setattr(moe_utils.MoEAuxLossAutoScaler, 'main_loss_backward_scale', None)
    moe_utils.MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(2.5))
    output = torch.ones(2, requires_grad=True)
    aux_loss = torch.full((2,), 3.0, requires_grad=True)
    moe_utils.MoEAuxLossAutoScaler.apply(output, aux_loss).sum().backward()
    assert output.grad.tolist() == [1.0, 1.0]
    assert aux_loss.grad.tolist() == [2.5, 2.5]


def test_topk_routing_returns_dense_output_when_requested():
    logits = torch.tensor([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])
    probs, top_indices = moe_utils.topk_routing_with_score_function(
        logits, topk=2, dense_output=True
    )
    assert top_indices.tolist() == [[2, 1], [0, 1]]
    expected = torch.softmax(torch.tensor([[3.0, 2.0], [6.0, 5.0]]), dim=-1)
    torch.testing.assert_close(probs, expected)


def test_topk_routing_raises_for_unknown_score_function():
    with pytest.raises(ValueError, match='Invalid score_function: nonsense'):
        moe_utils.topk_routing_with_score_function(
            torch.ones(2, 3), topk=1, score_function='nonsense'
        )


def test_topk_routing_builds_sparse_maps_deterministically(deterministic_algorithms):
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
    probs, routing_map = moe_utils.topk_routing_with_score_function(logits, topk=1)
    assert probs.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert routing_map.tolist() == [[False, True, False], [False, False, True]]


def test_topk_routing_fused_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_topk_with_score_function', None, raising=False)
    with pytest.raises(ValueError, match='fused_topk_with_score_function is not available'):
        moe_utils.topk_routing_with_score_function(torch.ones(2, 3), topk=1, fused=True)


def test_compute_routing_scores_for_aux_loss_softmax_normalizes_logits():
    logits = torch.tensor([[1.0, 2.0]])
    routing_map, scores = moe_utils.compute_routing_scores_for_aux_loss(
        logits, topk=1, score_function='softmax'
    )
    torch.testing.assert_close(scores, torch.softmax(logits, dim=-1, dtype=torch.float32))
    assert routing_map.tolist() == [[False, True]]


def test_compute_routing_scores_for_aux_loss_sigmoid_normalizes_scores():
    logits = torch.tensor([[0.0, 1.0]])
    routing_map, scores = moe_utils.compute_routing_scores_for_aux_loss(
        logits, topk=1, score_function='sigmoid'
    )
    sigmoid = torch.sigmoid(logits.float())
    torch.testing.assert_close(scores, sigmoid / sigmoid.sum(dim=-1, keepdim=True))
    assert routing_map.tolist() == [[False, True]]


def test_compute_routing_scores_for_aux_loss_sqrtsoftplus_normalizes_scores():
    logits = torch.tensor([[0.0, 1.0]])
    routing_map, scores = moe_utils.compute_routing_scores_for_aux_loss(
        logits, topk=1, score_function='sqrtsoftplus'
    )
    softplus = torch.nn.functional.softplus(logits.float()).sqrt()
    torch.testing.assert_close(scores, softplus / softplus.sum(dim=-1, keepdim=True))
    assert routing_map.tolist() == [[False, True]]


def test_compute_routing_scores_for_aux_loss_raises_for_unknown_score_function():
    with pytest.raises(ValueError, match='Invalid score_function: polygamma'):
        moe_utils.compute_routing_scores_for_aux_loss(
            torch.ones(2, 2), topk=1, score_function='polygamma'
        )


def test_compute_routing_scores_for_aux_loss_fused_raises_without_fused_kernel(monkeypatch):
    monkeypatch.setattr(moe_utils, 'fused_compute_score_for_moe_aux_loss', None, raising=False)
    with pytest.raises(ValueError, match='fused_compute_score_for_moe_aux_loss is not available'):
        moe_utils.compute_routing_scores_for_aux_loss(
            torch.ones(2, 2), topk=1, score_function='softmax', fused=True
        )


def test_apply_router_token_dropping_keeps_all_tokens_when_capacity_exceeds_tokens():
    routing_probs = torch.tensor([[0.9, 0.0], [0.0, 0.8]])
    routing_map = torch.tensor([[True, False], [False, True]])
    final_probs, final_map = moe_utils.apply_router_token_dropping(
        routing_probs, routing_map, router_topk=1, capacity_factor=10.0
    )
    assert final_probs.tolist() == routing_probs.tolist()
    assert final_map.tolist() == routing_map.tolist()


def test_apply_router_token_dropping_rejects_unknown_policy():
    routing_probs = torch.ones(4, 2)
    routing_map = torch.ones(4, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='Invalid drop_policy: nonsense'):
        moe_utils.apply_router_token_dropping(
            routing_probs, routing_map, router_topk=1, capacity_factor=0.5, drop_policy='nonsense'
        )


def test_save_to_aux_losses_tracker_records_value_per_layer(
    isolated_moe_tracker, identity_all_reduce
):
    moe_utils.save_to_aux_losses_tracker(
        'load_balancing_loss', torch.tensor(2.0), layer_number=1, num_layers=2
    )
    entry = moe_logging.get_moe_metrics_tracker().metrics['load_balancing_loss']
    assert entry.values.tolist() == [2.0, 0.0]
    assert entry.needs_dp_avg is True


def test_reduce_aux_losses_tracker_across_ranks_reduces_named_metrics(
    isolated_moe_tracker, identity_all_reduce
):
    moe_utils.save_to_aux_losses_tracker(
        'load_balancing_loss', torch.tensor(2.0), layer_number=1, num_layers=2
    )
    pg_collection = ProcessGroupCollection(pp=_CpGroup(1, 0), dp=_CpGroup(1, 0))
    moe_utils.reduce_aux_losses_tracker_across_ranks(
        track_names=['load_balancing_loss'], pg_collection=pg_collection
    )
    entry = moe_logging.get_moe_metrics_tracker().metrics['load_balancing_loss']
    assert entry.values.tolist() == [2.0, 0.0]


def test_track_moe_metrics_reports_aggregated_loss(isolated_moe_tracker, identity_all_reduce):
    moe_utils.save_to_aux_losses_tracker(
        'load_balancing_loss', torch.tensor(2.0), layer_number=1, num_layers=2
    )
    pg_collection = ProcessGroupCollection(pp=_CpGroup(1, 0), dp=_CpGroup(1, 0))
    log_string = moe_utils.track_moe_metrics(
        loss_scale=0.5, iteration=3, num_layers=2, pg_collection=pg_collection
    )
    assert log_string == ' load_balancing_loss: 0.50 |'


def test_apply_biased_logits_adds_layer_shared_bias(single_rank_parallel_state):
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    logits = torch.zeros(3, 4)
    first = moe_utils.apply_biased_logits(logits, std=0.5, layer_number=5)
    assert first.shape == logits.shape
    assert not torch.equal(first, logits)
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    second = moe_utils.apply_biased_logits(torch.zeros(3, 4), std=0.5, layer_number=5)
    torch.testing.assert_close(first, second)


def test_random_stes_shared_reuses_cached_bias_per_layer(single_rank_parallel_state, monkeypatch):
    monkeypatch.setattr(moe_utils.RandomSTEShared, '_cache', {})
    logits = torch.zeros(3, 4)
    first = moe_utils.RandomSTEShared.apply(logits, -0.5, 3)
    torch.testing.assert_close(moe_utils.RandomSTEShared._cache[3], (first - logits)[0])
    second = moe_utils.RandomSTEShared.apply(logits, -0.5, 3)
    torch.testing.assert_close(second, first)


def test_random_ste_backward_passes_gradient_through(single_rank_parallel_state):
    logits = torch.zeros(3, 4, requires_grad=True)
    randomized = moe_utils.RandomSTE.apply(logits)
    randomized.sum().backward()
    assert logits.grad.tolist() == [[1.0] * 4] * 3


def test_random_stes_shared_backward_returns_none_for_extra_inputs(single_rank_parallel_state):
    logits = torch.zeros(3, 4, requires_grad=True)
    out = moe_utils.RandomSTEShared.apply(logits, 0.5, 2)
    out.sum().backward()
    assert logits.grad.tolist() == [[1.0] * 4] * 3


def test_router_gating_linear_computes_and_backprops_in_float64():
    inp = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(
        [[0.5, 1.5], [-0.5, 0.5], [1.0, -1.0]], dtype=torch.float64, requires_grad=True
    )
    bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    out = moe_utils.router_gating_linear(inp, weight, bias, torch.float64)
    torch.testing.assert_close(
        out,
        torch.tensor(
            [[[1.0 * 0.5 + 2.0 * 1.5 + 0.1, -0.5 + 2.0 * 0.5 + 0.2, 1.0 - 2.0 + 0.3]]],
            dtype=torch.float64,
        ),
    )
    out.sum().backward()
    assert inp.grad.tolist() == [[[1.0, 1.0]]]
    assert weight.grad.tolist() == [[1.0, 2.0]] * 3
    assert bias.grad.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ('config_kwargs', 'expected'),
    [
        pytest.param({'use_transformer_engine_op_fuser': True}, 256, id='op-fuser'),
        pytest.param({'fp8': 'e4m3', 'fp8_recipe': 'mxfp8'}, 32, id='fp8-mxfp8'),
        pytest.param({'fp8': 'e4m3', 'fp8_recipe': 'tensorwise'}, 16, id='fp8-tensorwise'),
        pytest.param({'fp4': 'e2m1', 'fp4_recipe': 'nvfp4'}, 128, id='fp4-nvfp4'),
        pytest.param({}, 0, id='no-quantization'),
    ],
)
def test_get_align_size_for_quantization_follows_quantization_mode(config_kwargs, expected):
    config_values = dict(
        use_transformer_engine_op_fuser=False, fp8=None, fp8_recipe=None, fp4=None, fp4_recipe=None
    )
    config_values.update(config_kwargs)
    assert moe_utils.get_align_size_for_quantization(SimpleNamespace(**config_values)) == expected


@pytest.mark.parametrize(
    ('config_kwargs', 'expected'),
    [
        pytest.param({'moe_router_padding_for_quantization': True}, True, id='router-padded'),
        pytest.param(
            {'moe_token_dispatcher_type': 'flex', 'moe_flex_dispatcher_backend': 'hybridep'},
            True,
            id='hybridep-flex',
        ),
        pytest.param({}, False, id='no-padding'),
    ],
)
def test_skip_routed_expert_padding_reports_router_or_dispatcher_padding(config_kwargs, expected):
    config_values = dict(
        moe_router_padding_for_quantization=False,
        moe_token_dispatcher_type='alltoall',
        moe_flex_dispatcher_backend='deepep',
    )
    config_values.update(config_kwargs)
    assert moe_utils.skip_routed_expert_padding(SimpleNamespace(**config_values)) is expected
