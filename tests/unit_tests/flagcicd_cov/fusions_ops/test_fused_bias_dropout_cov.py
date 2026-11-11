# Copyright (c) 2026, BAAI. All rights reserved.
"""Bias-dropout-add: residual addition with broadcast bias, dropout scaling, in-place inference."""

import torch

from megatron.core.fusions import fused_bias_dropout


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_bias_dropout_add_fused_train_returns_residual_plus_bias_when_dropout_disabled():
    torch.manual_seed(1234)
    x = torch.randn(4, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    residual = torch.randn(4, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    bias = torch.randn(16, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_dropout.bias_dropout_add_fused_train)((x, bias), residual, 0.0)

    torch.testing.assert_close(out, residual + (x + bias), rtol=1.0e-6, atol=1.0e-6)
    out.backward(torch.ones_like(out))
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual.grad, torch.ones_like(residual), rtol=0.0, atol=0.0)


def test_bias_dropout_add_fused_train_drops_half_the_elements_when_prob_is_half():
    torch.manual_seed(4321)
    x = torch.full((8, 64), 2.0, dtype=torch.float32, device=DEVICE)
    residual = torch.zeros(8, 64, dtype=torch.float32, device=DEVICE)
    bias = torch.zeros(64, dtype=torch.float32, device=DEVICE)

    out = _eager(fused_bias_dropout.bias_dropout_add_fused_train)((x, bias), residual, 0.5)

    dropped_fraction = (out == 0.0).float().mean().item()
    assert 0.4 <= dropped_fraction <= 0.6
    kept = out[out != 0.0]
    torch.testing.assert_close(kept, torch.full_like(kept, 4.0), rtol=0.0, atol=0.0)


def test_bias_dropout_add_fused_inference_writes_into_input_storage_when_no_grad_required():
    torch.manual_seed(5678)
    x = torch.randn(4, 16, dtype=torch.float32, device=DEVICE)
    residual = torch.randn(4, 16, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(16, dtype=torch.float32, device=DEVICE)
    expected = residual + (x + bias)
    x_ptr = x.data_ptr()

    out = _eager(fused_bias_dropout.bias_dropout_add_fused_inference)((x, bias), residual, 0.0)

    torch.testing.assert_close(out, expected, rtol=1.0e-6, atol=1.0e-6)
    assert out.data_ptr() == x_ptr


def test_bias_dropout_add_fused_train_public_call_matches_reference_when_dropout_disabled():
    torch.manual_seed(1357)
    x = torch.randn(4, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    residual = torch.randn(4, 16, dtype=torch.float32, device=DEVICE, requires_grad=True)
    bias = torch.randn(16, dtype=torch.float32, device=DEVICE)

    out = fused_bias_dropout.bias_dropout_add_fused_train((x, bias), residual, 0.0)
    out.backward(torch.ones_like(out))

    torch.testing.assert_close(out, residual + (x + bias), rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual.grad, torch.ones_like(residual), rtol=0.0, atol=0.0)
