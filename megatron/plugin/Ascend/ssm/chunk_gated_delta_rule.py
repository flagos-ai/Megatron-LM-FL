# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Megatron GDN adaptation and Ascend forward/backward implementation.

Optional FLA-NPU kernels are loaded only for supported NPU inputs.
Unsupported inputs retain the original FLA path.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

# Head dim the AscendC GDN kernels are built for.
_SUPPORTED_HEAD_DIM = 128
# Chunk size the AscendC GDN kernels are built for.
_SUPPORTED_CHUNK_SIZE = 64


def _optional_module(name):
    """Return None only when the optional FLA-NPU package is absent."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != "fla_npu":
            raise
        return None


def l2norm(x, eps=1e-6, output_dtype=None):
    """Normalize NPU inputs with FLA-NPU and retain FLA elsewhere."""
    # FLA-NPU's small-width backward processes complete tiles of 16 rows only.
    if x.device.type == "npu" and (x.shape[-1] > 512 or x.numel() // x.shape[-1] % 16 == 0):
        ops = _optional_module("fla_npu.ops.triton")
        if ops is not None:
            return ops.l2norm(x, eps=eps, output_dtype=output_dtype)
    from fla.modules.l2norm import l2norm as original

    return original(x, eps=eps, output_dtype=output_dtype)


def _npu_kernels_support(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
) -> bool:
    """Check the dense, head-128 training contract of the AscendC kernels."""
    if q.device.type != "npu":
        return False
    if any(t.device != q.device for t in (k, v, g, beta)):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if g.ndim != 3 or beta.ndim != 3:
        return False
    if v.shape[:3] != q.shape[:3] or g.shape != q.shape[:3] or beta.shape != g.shape:
        return False
    if any(size == 0 for size in q.shape[:3]):
        return False
    if q.shape[-1] != _SUPPORTED_HEAD_DIM or v.shape[-1] != _SUPPORTED_HEAD_DIM:
        return False
    if g.dtype != torch.float32 or beta.dtype != q.dtype:
        return False
    if initial_state is not None or output_final_state:
        return False
    if cu_seqlens is not None:
        return False
    return True


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run supported dense inputs on Ascend; delegate other inputs to FLA."""
    supported = not head_first and _npu_kernels_support(
        q, k, v, g, beta, initial_state, output_final_state, cu_seqlens
    )

    if supported:
        supported = _optional_module("fla_npu") is not None
    if not supported:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as original

        return original(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
        )

    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    # BSHD -> BHSD for q/k/v; g and beta stay BSH.
    return _get_gdn_function().apply(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        g.contiguous(),
        beta.contiguous(),
        float(scale) if scale is not None else k.shape[-1] ** -0.5,
    )


def normalize_qk(self, x):
    return l2norm(x)


def gated_delta_rule(self, q, k, v, **kwargs):
    if self.config.deterministic_mode:
        return self.gated_delta_rule(q, k, v, **kwargs)
    return chunk_gated_delta_rule(q, k, v, **kwargs)


@lru_cache(maxsize=1)
def _load_kernels():
    """Load optional kernels on first use; failed loads remain retryable."""
    return SimpleNamespace(
        ascendc=importlib.import_module("fla_npu.ops.ascendc"),
        triton=importlib.import_module("fla_npu.ops.triton"),
    )


def solve_tri(A: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Solve triangular system with backend selection."""
    ops = _load_kernels()
    backend = os.getenv("FLA_NPU_GDN_SOLVE_TRI_BACKEND", "ascendc").strip().lower()
    if backend == "ascendc":
        return ops.ascendc.solve_tri(A.to(output_dtype).contiguous(), layout="bsnd")
    if backend == "triton":
        return ops.triton.solve_tril_npu(
            A=A, cu_seqlens=None, chunk_indices_out=None, output_dtype=output_dtype
        )
    raise ValueError(
        f"FLA_NPU_GDN_SOLVE_TRI_BACKEND must be 'ascendc' or 'triton', got {backend!r}"
    )


def recompute_w_u(
    k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute W and U from K, V, beta, A."""
    ops = _load_kernels()
    return ops.ascendc.recompute_w_u_fwd(
        k, v, beta, A, _SUPPORTED_CHUNK_SIZE, g=g, gk=None, cu_seqlens=None, chunk_indices=None
    )


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense recurrence: cumulative gates, WY representation, state, then output."""
    ops = _load_kernels()
    g = ops.triton.chunk_local_cumsum(
        g,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        cu_seqlens=None,
        chunk_indices_out=None,
        head_first=False,
    )
    A = ops.triton.chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        output_dtype=torch.float32,
    )
    A = solve_tri(A, output_dtype=k.dtype)
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()
    w, u = recompute_w_u(k, v, beta, A, g)
    h, v_new, _ = ops.ascendc.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        cu_seqlens=None,
        chunk_indices=None,
    )
    o = ops.ascendc.chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        transpose_state_layout=False,
    )
    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    return g, o, A


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute intermediates and accumulate gradients through the WY factors."""
    ops = _load_kernels()
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    w, u = recompute_w_u(k, v, beta, A, g)
    do = do.transpose(1, 2).contiguous()
    h, v_new, _ = ops.ascendc.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        cu_seqlens=None,
        chunk_indices=None,
    )
    dv = ops.ascendc.chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        _SUPPORTED_CHUNK_SIZE,
        g_gamma=None,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
    )
    dh, _, dv = ops.ascendc.chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        _SUPPORTED_CHUNK_SIZE,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dq, dk, dw, dg = ops.ascendc.chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        _SUPPORTED_CHUNK_SIZE,
        cu_seqlens=None,
        chunk_indices=None,
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dA = ops.ascendc.prepare_wy_repr_bwd_da(
        k,
        v,
        beta,
        A,
        dw,
        dv,
        g,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        cu_seqlens=None,
        chunk_indices=None,
    )
    dk2, dv, db, dg2 = ops.ascendc.prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, dv, g, _SUPPORTED_CHUNK_SIZE, cu_seqlens=None, chunk_indices=None
    )
    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()
    dk.add_(dk2)
    dg.add_(dg2)
    dg = ops.triton.chunk_local_cumsum(
        dg,
        chunk_size=_SUPPORTED_CHUNK_SIZE,
        reverse=True,
        cu_seqlens=None,
        chunk_indices_out=None,
        head_first=False,
    )
    return dq, dk, dv, db, dg


@lru_cache(maxsize=1)
def _get_gdn_function():
    ops = _load_kernels()

    class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
        """Autograd function for GDN with complete forward/backward."""

        @staticmethod
        @ops.triton.input_guard
        @ops.triton.autocast_custom_fwd
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
        ):
            g, o, A = flash_chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=scale)
            ctx.save_for_backward(q, k, v, g, beta, A)
            ctx.scale = scale
            return (o.to(q.dtype), None)

        @staticmethod
        @ops.triton.input_guard
        @ops.triton.autocast_custom_bwd
        def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
            q, k, v, g, beta, A = ctx.saved_tensors
            dq, dk, dv, db, dg = flash_chunk_gated_delta_rule_bwd(
                q=q, k=k, v=v, g=g, beta=beta, A=A, scale=ctx.scale, do=do
            )
            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                dg.to(g.dtype),
                db.to(beta.dtype),
                None,
            )

    return ChunkGatedDeltaRuleFunction
