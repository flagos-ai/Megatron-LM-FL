# Copyright (c) 2026, BAAI. All rights reserved.
"""Vocab-parallel cross entropy: per-partition logits max, predicted logits, loss and gradients."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_cross_entropy
from megatron.core.parallel_state import get_tensor_model_parallel_group
from tests.unit_tests.test_utilities import Utils


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_calculate_logits_max_returns_float_logits_and_per_token_max():
    torch.manual_seed(1234)
    logits = torch.randn(2, 3, 8, dtype=torch.bfloat16, device=DEVICE)

    float_logits, logits_max = _eager(fused_cross_entropy.calculate_logits_max)(logits)

    assert float_logits.dtype == torch.float32
    torch.testing.assert_close(float_logits, logits.float(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(logits_max, logits.float().amax(dim=-1), rtol=1.0e-6, atol=1.0e-6)


def test_calculate_predicted_logits_shifts_by_max_and_packs_sum_exp_with_predicted():
    torch.manual_seed(2345)
    seq, batch, vocab = 2, 3, 8
    logits = torch.randn(seq, batch, vocab, dtype=torch.float32, device=DEVICE)
    logits_max = logits.amax(dim=-1)
    target = torch.randint(0, vocab, (seq, batch), device=DEVICE)

    (target_mask, masked_target_1d, predicted_sum_exp, exp_logits) = _eager(
        fused_cross_entropy.calculate_predicted_logits
    )(logits.clone(), target, logits_max, 0, vocab)

    shifted = logits - logits_max.unsqueeze(-1)
    predicted = shifted.view(-1, vocab)[torch.arange(seq * batch), target.view(-1)]
    sum_exp = shifted.exp().sum(dim=-1)
    assert not target_mask.any()
    torch.testing.assert_close(masked_target_1d, target.view(-1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        predicted_sum_exp, torch.cat((predicted.view_as(target), sum_exp)), rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(exp_logits, shifted.exp(), rtol=1.0e-6, atol=1.0e-6)


def test_calculate_predicted_logits_masks_targets_outside_the_vocab_partition():
    torch.manual_seed(3456)
    seq, batch, vocab = 2, 3, 8
    logits = torch.randn(seq, batch, vocab, dtype=torch.float32, device=DEVICE)
    logits_max = logits.amax(dim=-1)
    target = torch.tensor([[0, 5, 7], [3, 9, 6]], device=DEVICE)

    (target_mask, masked_target_1d, predicted_sum_exp, exp_logits) = _eager(
        fused_cross_entropy.calculate_predicted_logits
    )(logits.clone(), target, logits_max, 4, 8)

    expected_mask = (target < 4) | (target >= 8)
    assert torch.equal(target_mask, expected_mask)
    expected_masked = target - 4
    expected_masked[expected_mask] = 0
    torch.testing.assert_close(masked_target_1d, expected_masked.view(-1), rtol=0.0, atol=0.0)
    shifted = logits - logits_max.unsqueeze(-1)
    predicted = shifted.view(-1, vocab)[torch.arange(seq * batch), expected_masked.view(-1)]
    predicted[expected_mask.view(-1)] = 0.0
    sum_exp = shifted.exp().sum(dim=-1)
    torch.testing.assert_close(
        predicted_sum_exp, torch.cat((predicted.view_as(target), sum_exp)), rtol=1.0e-6, atol=1.0e-6
    )


def test_calculate_cross_entropy_loss_normalizes_exp_logits_and_returns_log_sum_exp_minus_predicted():
    torch.manual_seed(4567)
    logits = torch.randn(4, 8, dtype=torch.float32, device=DEVICE)
    predicted = torch.randn(4, dtype=torch.float32, device=DEVICE)
    sum_exp = logits.exp().sum(dim=-1)
    packed = torch.cat((predicted, sum_exp))

    exp_logits, loss = _eager(fused_cross_entropy.calculate_cross_entropy_loss)(
        logits.exp().clone(), packed
    )

    torch.testing.assert_close(loss, torch.log(sum_exp) - predicted, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        exp_logits, logits.exp() / sum_exp.unsqueeze(-1), rtol=1.0e-6, atol=1.0e-6
    )


def test_calculate_gradients_returns_bf16_softmax_minus_one_hot_times_grad_output():
    torch.manual_seed(5678)
    seq, batch, vocab = 2, 3, 8
    softmax = torch.softmax(torch.randn(seq, batch, vocab, device=DEVICE), dim=-1)
    grad_output = torch.randn(seq, batch, device=DEVICE)
    target = torch.randint(0, vocab, (seq, batch), device=DEVICE)
    target_mask = torch.zeros(seq, batch, dtype=torch.bool, device=DEVICE)
    target_mask[0, 0] = True
    masked_target_1d = target.clone().view(-1)
    masked_target_1d[0] = 0

    grad = _eager(fused_cross_entropy.calculate_gradients)(
        softmax.clone(), grad_output, target_mask, masked_target_1d
    )

    softmax_update = 1.0 - target_mask.view(-1).float()
    update = F.one_hot(masked_target_1d, vocab).float() * softmax_update.unsqueeze(-1)
    expected = ((softmax.view(-1, vocab) - update) * grad_output.view(-1, 1)).to(torch.bfloat16)
    assert grad.dtype == torch.bfloat16
    assert torch.equal(grad, expected.view(seq, batch, vocab))


def test_fused_vocab_parallel_cross_entropy_matches_unfused_reference_with_single_rank(
    model_parallel,
):
    model_parallel()
    torch.manual_seed(6789)
    tp_group = get_tensor_model_parallel_group()
    seq, batch, vocab = 4, 2, 8
    logits = torch.randn(seq, batch, vocab, dtype=torch.bfloat16, device=DEVICE)
    target = torch.randint(0, vocab, (seq, batch), device=DEVICE)
    logits_fused = logits.detach().clone().requires_grad_(True)

    loss = fused_cross_entropy.fused_vocab_parallel_cross_entropy(logits_fused, target, tp_group)
    loss.backward(torch.ones_like(loss))

    logits_ref = logits.float().requires_grad_(True)
    ref_loss = F.cross_entropy(logits_ref.view(-1, vocab), target.view(-1), reduction="none")
    ref_loss.backward(torch.ones_like(ref_loss))

    assert loss.shape == (seq, batch)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, ref_loss.view(seq, batch), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(logits_fused.grad.float(), logits_ref.grad, rtol=1.6e-2, atol=1.0e-2)


def test_fused_vocab_parallel_cross_entropy_shards_vocab_across_tp_ranks(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    torch.manual_seed(7890)
    tp_group = get_tensor_model_parallel_group()
    tp_size, tp_rank = tp_group.size(), tp_group.rank()
    seq, batch, vocab = 4, 2, 16
    partition = vocab // tp_size
    logits = torch.randn(seq, batch, vocab, dtype=torch.bfloat16, device=DEVICE)
    target = torch.randint(0, vocab, (seq, batch), device=DEVICE)
    shard = logits[:, :, tp_rank * partition : (tp_rank + 1) * partition]
    logits_fused = shard.detach().clone().requires_grad_(True)

    loss = fused_cross_entropy.fused_vocab_parallel_cross_entropy(logits_fused, target, tp_group)
    loss.backward(torch.ones_like(loss))

    logits_ref = logits.float().requires_grad_(True)
    ref_loss = F.cross_entropy(logits_ref.view(-1, vocab), target.view(-1), reduction="none")
    ref_loss.backward(torch.ones_like(ref_loss))

    torch.testing.assert_close(loss, ref_loss.view(seq, batch), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        logits_fused.grad.float(),
        logits_ref.grad[:, :, tp_rank * partition : (tp_rank + 1) * partition],
        rtol=1.6e-2,
        atol=1.0e-2,
    )


def test_public_fused_cross_entropy_helpers_match_unfused_reference_with_single_rank():
    torch.manual_seed(2468)
    seq, batch, vocab = 4, 2, 8
    logits = torch.randn(seq, batch, vocab, dtype=torch.bfloat16, device=DEVICE)
    target = torch.randint(0, vocab, (seq, batch), device=DEVICE)
    grad_output = torch.randn(seq, batch, device=DEVICE)

    float_logits, logits_max = fused_cross_entropy.calculate_logits_max(logits)
    (target_mask, masked_target_1d, predicted_sum_exp, exp_logits) = (
        fused_cross_entropy.calculate_predicted_logits(
            float_logits.clone(), target, logits_max, 0, vocab
        )
    )
    softmax, loss = fused_cross_entropy.calculate_cross_entropy_loss(exp_logits, predicted_sum_exp)
    grad_input = fused_cross_entropy.calculate_gradients(
        softmax, grad_output, target_mask, masked_target_1d
    )

    ref_loss = F.cross_entropy(logits.float().view(-1, vocab), target.view(-1), reduction="none")
    torch.testing.assert_close(
        loss.view(seq, batch), ref_loss.view(seq, batch), rtol=1.0e-5, atol=1.0e-5
    )

    assert not target_mask.any()
    softmax_ref = torch.softmax(logits.float(), dim=-1).view(-1, vocab)
    update = F.one_hot(target.view(-1), vocab).float()
    expected_grad = ((softmax_ref - update) * grad_output.view(-1, 1)).to(torch.bfloat16)
    assert grad_input.dtype == torch.bfloat16
    torch.testing.assert_close(
        grad_input.view(seq, batch, vocab),
        expected_grad.view(seq, batch, vocab),
        rtol=1.6e-2,
        atol=1.0e-2,
    )
