# Copyright (c) 2026, BAAI. All rights reserved.
'''Vocab-parallel cross entropy: sharded-softmax loss and input gradients, with label smoothing and chunked logits, against a full-vocab PyTorch reference.'''

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy_chunked,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)

    yield setup
    Utils.destroy_model_parallel()


def _reference_cross_entropy(full_logits, target, vocab_start, vocab_end, label_smoothing=0.0):
    log_probs = F.log_softmax(full_logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if label_smoothing == 0.0:
        return nll
    partition_size = vocab_end - vocab_start
    smoothing = label_smoothing * partition_size / (partition_size - 1)
    mean_log_probs = log_probs[..., vocab_start:vocab_end].mean(dim=-1)
    return (1.0 - smoothing) * nll - smoothing * mean_log_probs


def test_vocab_parallel_cross_entropy_matches_torch_reference(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    torch.manual_seed(17)
    full_logits = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    target = torch.randint(0, 16, (4, 2), device='cuda')
    logits = full_logits.detach()[:, :, rank * 8 : (rank + 1) * 8].clone().requires_grad_(True)

    loss = vocab_parallel_cross_entropy(logits, target)
    expected = _reference_cross_entropy(full_logits, target, rank * 8, (rank + 1) * 8)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    grad_output = torch.randn(4, 2, device='cuda')
    (loss * grad_output).sum().backward()
    (expected * grad_output).sum().backward()
    torch.testing.assert_close(
        logits.grad, full_logits.grad[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-5, atol=1e-6
    )


def test_vocab_parallel_cross_entropy_label_smoothing_matches_reference(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    torch.manual_seed(18)
    full_logits = torch.randn(4, 2, 16, device='cuda', requires_grad=True)
    target = torch.randint(0, 16, (4, 2), device='cuda')
    logits = full_logits.detach()[:, :, rank * 8 : (rank + 1) * 8].clone().requires_grad_(True)

    loss = vocab_parallel_cross_entropy(logits, target, label_smoothing=0.1)
    expected = _reference_cross_entropy(
        full_logits, target, rank * 8, (rank + 1) * 8, label_smoothing=0.1
    )
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    grad_output = torch.randn(4, 2, device='cuda')
    (loss * grad_output).sum().backward()
    (expected * grad_output).sum().backward()
    torch.testing.assert_close(
        logits.grad, full_logits.grad[:, :, rank * 8 : (rank + 1) * 8], rtol=1e-4, atol=1e-6
    )


def test_vocab_parallel_cross_entropy_chunked_matches_reference(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    torch.manual_seed(19)
    # The chunked kernel is specified for bf16 logits: it upcasts a copy and works on it in place.
    full_bf16 = torch.randn(4, 2, 16, device='cuda').to(torch.bfloat16)
    full_logits = full_bf16.float().requires_grad_(True)
    target = torch.randint(0, 16, (4, 2), device='cuda')
    logits = full_bf16[:, :, rank * 8 : (rank + 1) * 8].clone().requires_grad_(True)

    loss = vocab_parallel_cross_entropy_chunked(logits, target, chunk_size=2)
    expected = _reference_cross_entropy(full_logits, target, rank * 8, (rank + 1) * 8)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    grad_output = torch.randn(4, 2, device='cuda')
    (loss * grad_output).sum().backward()
    (expected * grad_output).sum().backward()
    # The gradient is returned in the input dtype (bf16): compare with the fp32 reference rounded to bf16.
    torch.testing.assert_close(
        logits.grad, full_logits.grad[:, :, rank * 8 : (rank + 1) * 8].to(torch.bfloat16)
    )


def test_vocab_parallel_cross_entropy_chunked_label_smoothing_matches_reference(model_parallel):
    model_parallel(2, 1)
    rank = parallel_state.get_tensor_model_parallel_rank()
    torch.manual_seed(20)
    # The chunked kernel is specified for bf16 logits: it upcasts a copy and works on it in place.
    full_bf16 = torch.randn(4, 2, 16, device='cuda').to(torch.bfloat16)
    full_logits = full_bf16.float().requires_grad_(True)
    target = torch.randint(0, 16, (4, 2), device='cuda')
    logits = full_bf16[:, :, rank * 8 : (rank + 1) * 8].clone().requires_grad_(True)

    loss = vocab_parallel_cross_entropy_chunked(logits, target, label_smoothing=0.1, chunk_size=3)
    expected = _reference_cross_entropy(
        full_logits, target, rank * 8, (rank + 1) * 8, label_smoothing=0.1
    )
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    grad_output = torch.randn(4, 2, device='cuda')
    (loss * grad_output).sum().backward()
    (expected * grad_output).sum().backward()
    # The gradient is returned in the input dtype (bf16): compare with the fp32 reference rounded to bf16.
    torch.testing.assert_close(
        logits.grad, full_logits.grad[:, :, rank * 8 : (rank + 1) * 8].to(torch.bfloat16)
    )
