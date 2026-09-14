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
import logging
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Head dim the AscendC GDN kernels are built for.
_SUPPORTED_HEAD_DIM = 128
# Chunk size the AscendC GDN kernels are built for.
_SUPPORTED_CHUNK_SIZE = 64


def _fla_original_chunk_gated_delta_rule():
    """Return the original FLA implementation for unsupported inputs."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as original

        return original
    except ImportError as e:  # pragma: no cover - depends on site-packages
        logger.warning(f"[GDN] FLA fallback unavailable: {e}")
        return None


def _fla_original_l2norm():
    """Return FLA's own ``l2norm``, or None when unavailable."""
    try:
        from fla.modules.l2norm import l2norm as original

        return original
    except ImportError as e:  # pragma: no cover - depends on site-packages
        logger.warning(f"[GDN] FLA l2norm fallback unavailable: {e}")
        return None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """L2 normalization over the last axis, matching FLA's signature exactly.

    ``fla_npu.ops.triton.l2norm`` takes the same ``(x, eps, output_dtype)``
    arguments as ``fla.modules.l2norm.l2norm``, so this is a straight swap on
    NPU tensors and a delegation to FLA everywhere else.
    """
    if x.device.type == "npu":
        try:
            from fla_npu.ops.triton import l2norm as npu_l2norm

        except ImportError as e:
            logger.warning(f"[GDN] fla_npu l2norm unavailable, using FLA: {e}")
        else:
            return npu_l2norm(x, eps=eps, output_dtype=output_dtype)

    original = _fla_original_l2norm()
    if original is None:
        raise RuntimeError(
            "l2norm requires either fla_npu (on NPU) or FLA, but neither is importable."
        )
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
    """Report whether the AscendC kernels cover this input.

    Checked before any kernel launches so an unsupported case costs nothing but
    a fallback.  Variable-length (``cu_seqlens``) input is deliberately excluded
    for now: the kernels accept it but that path is unverified on this stack.
    """
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


def _runtime_available() -> bool:
    """Check the fla_npu operator set is loaded, without launching kernels."""
    try:
        _load_kernels()
        return hasattr(torch, "npu") and torch.npu.is_available()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning(f"[GDN] Runtime validation failed: {e}")
        return False


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
    """Chunked gated delta rule on Ascend NPU, matching FLA's signature.

    Falls back to FLA for anything the AscendC kernels do not cover.  Kernel
    execution errors propagate rather than silently becoming a fallback result,
    so a broken kernel is visible instead of being masked by a slow path.

    ``head_first=True`` is deprecated in FLA and means BHSD input; the AscendC
    path assumes BSHD, so that case is handed back to FLA unchanged.
    """
    supported = not head_first and _npu_kernels_support(
        q, k, v, g, beta, initial_state, output_final_state, cu_seqlens
    )

    if not supported or not _runtime_available():
        original = _fla_original_chunk_gated_delta_rule()
        if original is None:
            raise RuntimeError(
                "chunk_gated_delta_rule: input is unsupported by the Ascend kernels "
                "and FLA is not importable for fallback."
            )
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

    # BSHD -> BHSD for q/k/v; g and beta stay BSH.
    output, final_state = flash_gated_delta_rule(
        q=q.transpose(1, 2).contiguous(),
        k=k.transpose(1, 2).contiguous(),
        v=v.transpose(1, 2).contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return output.contiguous(), final_state


def normalize_qk(self, x):
    return l2norm(x)


def gated_delta_rule(self, q, k, v, **kwargs):
    if self.config.deterministic_mode:
        return self.gated_delta_rule(q, k, v, **kwargs)
    return chunk_gated_delta_rule(q, k, v, **kwargs)


@lru_cache(maxsize=1)
def _load_kernels():
    """Load optional kernels on first use; failed loads remain retryable."""
    kernels = {}
    module = importlib.import_module("fla_npu.ops.ascendc")
    for alias, name in (
        ("ascendc_chunk_bwd_dqkwg", "chunk_bwd_dqkwg"),
        ("ascendc_chunk_bwd_dv_local", "chunk_bwd_dv_local"),
        ("ascendc_chunk_fwd_o", "chunk_fwd_o"),
        ("ascendc_chunk_gated_delta_rule_bwd_dhu", "chunk_gated_delta_rule_bwd_dhu"),
        ("ascendc_chunk_gated_delta_rule_fwd_h", "chunk_gated_delta_rule_fwd_h"),
        ("ascendc_prepare_wy_repr_bwd_da", "prepare_wy_repr_bwd_da"),
        ("ascendc_prepare_wy_repr_bwd_full", "prepare_wy_repr_bwd_full"),
        ("ascendc_recompute_w_u_fwd", "recompute_w_u_fwd"),
        ("ascendc_solve_tri", "solve_tri"),
    ):
        kernel = getattr(module, name, None)
        if not callable(kernel):
            raise ImportError(f"Missing GDN kernel: {module.__name__}.{name}")
        kernels[alias] = kernel
    module = importlib.import_module("fla_npu.ops.triton")
    for alias, name in (
        ("autocast_custom_bwd", "autocast_custom_bwd"),
        ("autocast_custom_fwd", "autocast_custom_fwd"),
        ("chunk_local_cumsum", "chunk_local_cumsum"),
        ("chunk_scaled_dot_kkt_fwd", "chunk_scaled_dot_kkt_fwd"),
        ("input_guard", "input_guard"),
        ("l2norm_bwd", "l2norm_bwd"),
        ("l2norm_fwd", "l2norm_fwd"),
        ("solve_tril_npu", "solve_tril_npu"),
    ):
        kernel = getattr(module, name, None)
        if not callable(kernel):
            raise ImportError(f"Missing GDN kernel: {module.__name__}.{name}")
        kernels[alias] = kernel
    return SimpleNamespace(**kernels)


def solve_tri_ascendc(A: torch.Tensor, output_dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Solve triangular system using AscendC."""
    ops = _load_kernels()
    A_in = A.to(output_dtype).contiguous()
    return ops.ascendc_solve_tri(A_in, layout="bsnd")


def solve_tri(A: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Solve triangular system with backend selection."""
    ops = _load_kernels()
    backend = os.getenv("FLA_NPU_GDN_SOLVE_TRI_BACKEND", "ascendc").strip().lower()
    if backend == "ascendc":
        return solve_tri_ascendc(A, output_dtype=output_dtype)
    elif backend == "triton":
        return ops.solve_tril_npu(
            A=A, cu_seqlens=None, chunk_indices_out=None, output_dtype=output_dtype
        )
    else:
        raise ValueError(
            f"FLA_NPU_GDN_SOLVE_TRI_BACKEND must be 'ascendc' or 'triton', got {backend!r}"
        )


def recompute_w_u(
    k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute W and U from K, V, beta, A."""
    ops = _load_kernels()
    w, u = ops.ascendc_recompute_w_u_fwd(
        k, v, beta, A, _SUPPORTED_CHUNK_SIZE, g=g, gk=None, cu_seqlens=None, chunk_indices=None
    )
    return (w, u)


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
    g = ops.chunk_local_cumsum(
        g, chunk_size=_SUPPORTED_CHUNK_SIZE, cu_seqlens=None, chunk_indices_out=None, head_first=False
    )
    A = ops.chunk_scaled_dot_kkt_fwd(
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
    h, v_new, _ = ops.ascendc_chunk_gated_delta_rule_fwd_h(
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
    o = ops.ascendc_chunk_fwd_o(
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
    return (g, o, A)


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
    h, v_new, _ = ops.ascendc_chunk_gated_delta_rule_fwd_h(
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
    dv = ops.ascendc_chunk_bwd_dv_local(
        q, k, do, g, scale, _SUPPORTED_CHUNK_SIZE, g_gamma=None, A=A, cu_seqlens=None, chunk_indices=None
    )
    dh, _, dv = ops.ascendc_chunk_gated_delta_rule_bwd_dhu(
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
    dq, dk, dw, dg = ops.ascendc_chunk_bwd_dqkwg(
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
    dA = ops.ascendc_prepare_wy_repr_bwd_da(
        k, v, beta.float(), A, dw, dv, g.float(), chunk_size=_SUPPORTED_CHUNK_SIZE, cu_seqlens=None, chunk_indices=None
    )
    dk2, dv, db, dg2 = ops.ascendc_prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, dv, g, _SUPPORTED_CHUNK_SIZE, cu_seqlens=None, chunk_indices=None
    )
    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()
    dk.add_(dk2)
    dg.add_(dg2)
    dg = ops.chunk_local_cumsum(
        dg, chunk_size=_SUPPORTED_CHUNK_SIZE, reverse=True, cu_seqlens=None, chunk_indices_out=None, head_first=False
    )
    return (dq, dk, dv, db, dg)


@lru_cache(maxsize=1)
def _get_gdn_function():
    ops = _load_kernels()

    class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
        """Autograd function for GDN with complete forward/backward."""

        @staticmethod
        @ops.input_guard
        @ops.autocast_custom_fwd
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            use_qk_l2norm_in_kernel: bool = False,
        ):
            if use_qk_l2norm_in_kernel:
                q, q_rstd = ops.l2norm_fwd(q)
                k, k_rstd = ops.l2norm_fwd(k)
            else:
                q_rstd, k_rstd = (None, None)
            g, o, A = flash_chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=scale)
            ctx.save_for_backward(q, k, v, g, beta, A)
            ctx.q_rstd = q_rstd
            ctx.k_rstd = k_rstd
            ctx.scale = scale
            ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
            return (o.to(q.dtype), None)

        @staticmethod
        @ops.input_guard
        @ops.autocast_custom_bwd
        def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
            q, k, v, g, beta, A = ctx.saved_tensors
            dq, dk, dv, db, dg = flash_chunk_gated_delta_rule_bwd(
                q=q, k=k, v=v, g=g, beta=beta, A=A, scale=ctx.scale, do=do
            )
            if ctx.use_qk_l2norm_in_kernel:
                dq = ops.l2norm_bwd(q, ctx.q_rstd, dq)
                dk = ops.l2norm_bwd(k, ctx.k_rstd, dk)
            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                dg.to(g.dtype),
                db.to(beta.dtype),
                None,
                None,
            )

    return ChunkGatedDeltaRuleFunction


def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Main entry point for GDN.

    Internal dense-only entry; the adapter validates inputs before calling.
    Expects BHSD layout for q, k, v and BSH for g, beta.
    Returns output in BSHD layout.
    """
    if scale is None:
        scale = k.shape[-1] ** (-0.5)
    o, final_state = _get_gdn_function().apply(
        q, k, v, g, beta, float(scale), use_qk_l2norm_in_kernel
    )
    return (o, final_state)
