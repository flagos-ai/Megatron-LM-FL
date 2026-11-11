# Copyright (c) 2026, BAAI. All rights reserved.
"""Scaled masked softmax: fused CUDA kernels per mask type and the torch fallback selection."""

import pytest
import torch

from megatron.core.fusions import fused_softmax
from megatron.core.fusions.fused_softmax import (
    FusedScaleMaskSoftmax,
    ScaledMaskedSoftmax,
    ScaledSoftmax,
    ScaledUpperTriangMaskedSoftmax,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP16_TOLS = dict(rtol=1.0e-3, atol=1.0e-3)


def _masked_softmax_fp32(scores, mask, scale):
    scores = scores.float() * scale
    if mask is not None:
        scores = scores.masked_fill(mask, -float("inf"))
    return torch.softmax(scores, dim=-1)


def test_scaled_upper_triang_masked_softmax_matches_reference_for_forward_and_backward():
    torch.manual_seed(1234)
    attn_batches, sq, sk, scale = 8, 64, 64, 0.5
    x = torch.randn(attn_batches, sq, sk, dtype=torch.float16, device=DEVICE)
    mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool, device=DEVICE), diagonal=1)
    x_fused = x.detach().clone().requires_grad_(True)
    out = ScaledUpperTriangMaskedSoftmax.apply(x_fused, scale)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)

    x_ref = x.detach().float().requires_grad_(True)
    ref = _masked_softmax_fp32(x_ref, mask, scale)
    ref.backward(grad_output.float())

    assert out.dtype == torch.float16
    assert out.shape == (attn_batches, sq, sk)
    torch.testing.assert_close(out.float(), ref.detach(), **FP16_TOLS)
    torch.testing.assert_close(x_fused.grad.float(), x_ref.grad, **FP16_TOLS)


def test_scaled_masked_softmax_matches_reference_with_padding_mask_for_backward():
    torch.manual_seed(2345)
    b, np_, sq, sk, scale = 2, 4, 64, 64, 0.5
    x = torch.randn(b, np_, sq, sk, dtype=torch.float16, device=DEVICE)
    mask = torch.zeros(b, 1, sq, sk, dtype=torch.bool, device=DEVICE)
    mask[:, :, :, ::4] = True

    x_fused = x.detach().clone().requires_grad_(True)
    out = ScaledMaskedSoftmax.apply(x_fused, mask, scale)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)

    x_ref = x.detach().float().requires_grad_(True)
    ref = _masked_softmax_fp32(x_ref, mask, scale)
    ref.backward(grad_output.float())

    assert not mask.all(dim=-1).any()
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.detach(), **FP16_TOLS)
    torch.testing.assert_close(x_fused.grad.float(), x_ref.grad, **FP16_TOLS)


def test_scaled_softmax_matches_reference_without_mask_for_forward_and_backward():
    torch.manual_seed(3456)
    b, np_, sq, sk, scale = 2, 4, 64, 64, 2.0
    x = torch.randn(b, np_, sq, sk, dtype=torch.float16, device=DEVICE)

    x_fused = x.detach().clone().requires_grad_(True)
    out = ScaledSoftmax.apply(x_fused, scale)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)

    x_ref = x.detach().float().requires_grad_(True)
    ref = _masked_softmax_fp32(x_ref, None, scale)
    ref.backward(grad_output.float())

    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.detach(), **FP16_TOLS)
    torch.testing.assert_close(x_fused.grad.float(), x_ref.grad, **FP16_TOLS)


def _fused_module(attn_mask_type, fusion, window_size=None):
    return FusedScaleMaskSoftmax(
        input_in_fp16=True,
        input_in_bf16=False,
        attn_mask_type=attn_mask_type,
        scaled_masked_softmax_fusion=fusion,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=window_size,
    )


@pytest.mark.parametrize(
    "fusion,sq,sk,b,np,expected",
    [
        pytest.param(True, 128, 128, 4, 16, True, id="all_constraints_hold"),
        pytest.param(False, 128, 128, 4, 16, False, id="fusion_flag_off"),
        pytest.param(True, 128, 16, 4, 16, False, id="sk_not_above_16"),
        pytest.param(True, 126, 128, 4, 16, False, id="sq_not_multiple_of_4"),
        pytest.param(True, 128, 130, 4, 16, False, id="sk_not_multiple_of_4"),
        pytest.param(True, 128, 128, 3, 5, False, id="attn_batches_not_multiple_of_4"),
    ],
)
def test_is_kernel_available_selects_padding_branch_when_constraints_hold(
    fusion, sq, sk, b, np, expected
):
    module = _fused_module(AttnMaskType.padding, fusion)

    assert module.is_kernel_available(None, b, np, sq, sk) is expected


def test_is_kernel_available_rejects_float32_input_when_fusion_is_on():
    module = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=False,
        attn_mask_type=AttnMaskType.padding,
        scaled_masked_softmax_fusion=True,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
    )

    assert module.is_kernel_available(None, 4, 16, 128, 128) is False


def test_fused_scale_mask_softmax_uses_masked_kernel_when_mask_is_provided():
    torch.manual_seed(4567)
    b, np_, sq, sk = 4, 16, 128, 128
    module = _fused_module(AttnMaskType.padding, True)
    x = torch.randn(b, np_, sq, sk, dtype=torch.float16, device=DEVICE)
    mask = torch.zeros(b, 1, sq, sk, dtype=torch.bool, device=DEVICE)
    mask[:, :, :, ::4] = True

    assert module.is_kernel_available(mask, *x.size()) is True
    out = module(x, mask)

    expected = _masked_softmax_fp32(x, mask, 1.0)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), expected, **FP16_TOLS)


def test_fused_scale_mask_softmax_uses_plain_kernel_when_mask_is_none():
    torch.manual_seed(5678)
    b, np_, sq, sk = 4, 16, 128, 128
    module = _fused_module(AttnMaskType.padding, True)
    x = torch.randn(b, np_, sq, sk, dtype=torch.float16, device=DEVICE)

    assert module.is_kernel_available(None, *x.size()) is True
    out = module(x, None)

    expected = _masked_softmax_fp32(x, None, 1.0)
    torch.testing.assert_close(out.float(), expected, **FP16_TOLS)


def test_forward_torch_softmax_applies_sliding_window_mask_when_window_size_is_set():
    torch.manual_seed(6789)
    b, np_, sq, sk, window = 2, 4, 16, 16, (2, 3)
    module = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=False,
        attn_mask_type=AttnMaskType.causal,
        scaled_masked_softmax_fusion=False,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=window,
    )
    x = torch.randn(b, np_, sq, sk, dtype=torch.float32, device=DEVICE)

    out = module(x, None)

    idx = torch.arange(sq, device=DEVICE)
    row, col = idx[:, None], idx[None, :]
    attends = (col >= row - window[0]) & (col <= row + window[1])
    expected = torch.softmax(x.masked_fill(~attends, -float("inf")), dim=-1)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)
