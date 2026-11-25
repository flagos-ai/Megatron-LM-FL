# Copyright (c) 2026, BAAI. All rights reserved.
"""MLA YaRN RoPE: in-place pair rotation of the q tail, KV split sharing one position embedding."""

import importlib.util
import sys
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.fusions import fused_mla_yarn_rope_apply
from megatron.core.utils import null_decorator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CU_SEQLENS = [0, 5, 13, 21]
HEADS = 4
NOPE_DIM = 32
EMB_DIM = 16
K_DIM = 16
V_DIM = 16
MAX_SEQLEN = 8

TOLS = dict(rtol=1.0e-5, atol=1.0e-5)


def _cos_sin(max_seqlen, emb_dim):
    pos = torch.arange(max_seqlen, dtype=torch.float64) + 1.0
    dims_left = torch.arange(emb_dim // 2, dtype=torch.float64) * 0.3 + 0.1
    dims_right = torch.arange(emb_dim // 2, dtype=torch.float64) * 0.6 + 0.05
    theta_left = torch.outer(pos, dims_left)
    theta_right = torch.outer(pos, dims_right)
    cos = torch.cat([theta_left.cos(), theta_right.cos()], dim=-1)
    sin = torch.cat([theta_left.sin(), theta_right.sin()], dim=-1)
    return (
        cos.to(device=DEVICE, dtype=torch.float32).view(max_seqlen, 1, 1, emb_dim),
        sin.to(device=DEVICE, dtype=torch.float32).view(max_seqlen, 1, 1, emb_dim),
    )


def _token_positions(layout):
    if layout == "sbhd":
        return torch.arange(MAX_SEQLEN).repeat_interleave(2)
    positions = []
    for seq in range(len(CU_SEQLENS) - 1):
        positions.extend(range(CU_SEQLENS[seq + 1] - CU_SEQLENS[seq]))
    return torch.tensor(positions)


def _rotate_pairs(x1, x2, cos, sin, inverse):
    half = x1.shape[-1]
    cos_left, cos_right = cos[..., :half], cos[..., half:]
    sin_left, sin_right = sin[..., :half], sin[..., half:]
    if inverse:
        sin_left, sin_right = -sin_left, -sin_right
    return x1 * cos_left - x2 * sin_left, x2 * cos_right + x1 * sin_right


def _pe_from_pairs(pair_left, pair_right, remove_interleaving):
    if remove_interleaving:
        return torch.cat([pair_left.unsqueeze(-1), pair_right.unsqueeze(-1)], dim=-1).flatten(-2)
    return torch.cat([pair_left, pair_right], dim=-1)


@pytest.mark.parametrize(
    "layout,inverse,remove_interleaving",
    [("sbhd", False, False), ("thd", False, False), ("thd", False, True), ("thd", True, False)],
    ids=["sbhd", "thd", "thd_remove_interleaving", "thd_inverse"],
)
def test_fused_mla_rope_inplace_matches_reference_for_forward_and_backward(
    layout, inverse, remove_interleaving
):
    torch.manual_seed(1234 + (layout == "thd") + 2 * int(inverse) + 4 * int(remove_interleaving))
    if layout == "sbhd":
        cu_seqlens = None
        total = MAX_SEQLEN * 2
        q = torch.randn(MAX_SEQLEN, 2, HEADS, NOPE_DIM + EMB_DIM, device=DEVICE)
    else:
        cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device=DEVICE)
        total = CU_SEQLENS[-1]
        q = torch.randn(total, HEADS, NOPE_DIM + EMB_DIM, device=DEVICE)
    cos, sin = _cos_sin(MAX_SEQLEN, EMB_DIM)
    q_fused = q.detach().clone().requires_grad_(True)

    out = fused_mla_yarn_rope_apply.fused_mla_rope_inplace(
        q_fused,
        cos,
        sin,
        NOPE_DIM,
        EMB_DIM,
        cu_seqlens_q=cu_seqlens,
        inverse=inverse,
        remove_interleaving=remove_interleaving,
    )
    grad_output = torch.randn_like(out)
    out.backward(grad_output.clone())

    q_ref = q.detach().clone().requires_grad_(True)
    q_flat = q_ref.reshape(total, HEADS, NOPE_DIM + EMB_DIM)
    x1 = q_flat[..., NOPE_DIM : NOPE_DIM + EMB_DIM : 2]
    x2 = q_flat[..., NOPE_DIM + 1 : NOPE_DIM + EMB_DIM : 2]
    token_pos = _token_positions(layout)
    cos_tok = cos[token_pos, 0, 0][:, None, :]
    sin_tok = sin[token_pos, 0, 0][:, None, :]
    pair_left, pair_right = _rotate_pairs(x1, x2, cos_tok, sin_tok, inverse)
    pe_out = _pe_from_pairs(pair_left, pair_right, remove_interleaving)
    out_ref = torch.cat([q_flat[..., :NOPE_DIM], pe_out], dim=-1)
    if layout == "sbhd":
        out_ref = out_ref.view(MAX_SEQLEN, 2, HEADS, NOPE_DIM + EMB_DIM)
    out_ref.backward(grad_output)

    assert out.shape == q.shape
    torch.testing.assert_close(out, out_ref.detach(), **TOLS)
    torch.testing.assert_close(q_fused.grad, q_ref.grad, **TOLS)


@pytest.mark.parametrize(
    "layout,remove_interleaving",
    [("sbhd", False), ("thd", False), ("thd", True)],
    ids=["sbhd", "thd", "thd_remove_interleaving"],
)
def test_fused_mla_rope_kv_split_matches_reference_for_forward_and_backward(
    layout, remove_interleaving
):
    torch.manual_seed(4321 + (layout == "thd") + 2 * int(remove_interleaving))
    if layout == "sbhd":
        cu_seqlens = None
        total = MAX_SEQLEN * 2
        kv = torch.randn(MAX_SEQLEN, 2, HEADS, K_DIM + V_DIM, device=DEVICE)
        k_pos_emb = torch.randn(MAX_SEQLEN, 2, 1, EMB_DIM, device=DEVICE)
    else:
        cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device=DEVICE)
        total = CU_SEQLENS[-1]
        kv = torch.randn(total, HEADS, K_DIM + V_DIM, device=DEVICE)
        k_pos_emb = torch.randn(total, 1, EMB_DIM, device=DEVICE)
    cos, sin = _cos_sin(MAX_SEQLEN, EMB_DIM)
    kv_fused = kv.detach().clone().requires_grad_(True)
    emb_fused = k_pos_emb.detach().clone().requires_grad_(True)

    o_key, o_value = fused_mla_yarn_rope_apply.fused_mla_rope_kv_split(
        kv_fused,
        emb_fused,
        cos,
        sin,
        EMB_DIM,
        K_DIM,
        V_DIM,
        cu_seqlens_kv=cu_seqlens,
        remove_interleaving=remove_interleaving,
    )
    grad_key = torch.randn_like(o_key)
    grad_value = torch.randn_like(o_value)
    torch.autograd.backward((o_key, o_value), (grad_key, grad_value))

    kv_ref = kv.detach().clone().requires_grad_(True)
    emb_ref = k_pos_emb.detach().clone().requires_grad_(True)
    kv_flat = kv_ref.reshape(total, HEADS, K_DIM + V_DIM)
    emb_flat = emb_ref.reshape(total, EMB_DIM)
    k, v = kv_flat[..., :K_DIM], kv_flat[..., K_DIM:]
    token_pos = _token_positions(layout)
    cos_tok = cos[token_pos, 0, 0]
    sin_tok = sin[token_pos, 0, 0]
    pair_left, pair_right = _rotate_pairs(
        emb_flat[:, 0::2], emb_flat[:, 1::2], cos_tok, sin_tok, False
    )
    pe_out = _pe_from_pairs(pair_left, pair_right, remove_interleaving)
    key_ref = torch.cat([k, pe_out[:, None, :].expand(total, HEADS, EMB_DIM)], dim=-1)
    value_ref = v
    if layout == "sbhd":
        key_ref = key_ref.view(MAX_SEQLEN, 2, HEADS, K_DIM + EMB_DIM)
        value_ref = value_ref.view(MAX_SEQLEN, 2, HEADS, V_DIM)
    torch.autograd.backward((key_ref, value_ref), (grad_key, grad_value))

    assert o_key.shape == key_ref.shape
    assert o_value.shape == value_ref.shape
    torch.testing.assert_close(o_key, key_ref.detach(), **TOLS)
    torch.testing.assert_close(o_value, value_ref.detach(), **TOLS)
    torch.testing.assert_close(kv_fused.grad, kv_ref.grad, **TOLS)
    torch.testing.assert_close(
        emb_fused.grad.reshape(total, EMB_DIM), emb_ref.grad.reshape(total, EMB_DIM), **TOLS
    )


def test_fused_mla_yarn_module_falls_back_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    spec = importlib.util.spec_from_file_location(
        "fused_mla_yarn_no_triton", fused_mla_yarn_rope_apply.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.HAVE_TRITON is False
    assert module.triton.jit is null_decorator
    assert module.triton.autotune is null_decorator
    assert isinstance(module.tl, MagicMock)
