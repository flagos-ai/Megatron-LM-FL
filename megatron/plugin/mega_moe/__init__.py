# Copyright (c) FlagOS Team, BAAI Corporation.
#
# SM90 MegaMoE plugin for Megatron-LM.
#
# This module provides fused MoE forward (SM90 FP8) and backward kernels.
# It requires:
#   1. `deep_gemm` package installed (for JIT kernel compilation infrastructure)
#   2. `_mega_moe_C` extension compiled (host runtime for TMA, heuristics, launch)
#
# Build instructions:
#   cd megatron/plugin/mega_moe/
#   export DEEP_GEMM_ROOT=/path/to/DeepGEMM
#   python setup_ext.py build_ext --inplace

import types
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from ._check import require_mega_moe_ext
from ._install_kernels import install_kernel_sources

# Validate dependencies before proceeding
require_mega_moe_ext()

# Install our .cuh kernel sources into deep_gemm's include path so the JIT
# compiler can find them at runtime (e.g. #include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>)
install_kernel_sources()

# Now safe to import
import _mega_moe_C as _C  # noqa: E402

# noinspection PyProtectedMember
import torch.distributed._symmetric_memory as symm_mem

from .utils import align, uneven_all_gather


# ============================================================================
# Helpers
# ============================================================================

def _is_sm90() -> bool:
    return torch.cuda.get_device_capability()[0] == 9


# ============================================================================
# SymmBuffer
# ============================================================================

class SymmBuffer:
    """Symmetric-memory buffer for MegaMoE forward and backward."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = 'swiglu',
        with_backward: bool = False,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.with_backward = with_backward

        # Allocate forward symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )

        # Optionally extend with backward buffer regions
        backward_num_bytes = 0
        self._slice_backward = None
        if with_backward:
            backward_num_bytes, self._slice_backward = \
                _C.get_symm_buffer_size_for_sm90_mega_moe_backward(
                    group.size(), num_experts,
                    num_max_tokens_per_rank, num_topk,
                    hidden, intermediate_hidden,
                    use_fp8_dispatch, activation
                )

        total_bytes = num_bytes + backward_num_bytes
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(total_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create forward input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

        # Create backward buffer views
        self.dy = None
        self.dx_combine = None
        self.d_o_pool = None
        self.d_a_pool = None
        self.recomp_h = None
        self.recomp_a = None
        if with_backward and self._slice_backward is not None:
            backward_base = self.buffer[num_bytes:]
            (self.dy, self.dx_combine,
             self.d_o_pool, self.d_a_pool,
             self.recomp_h, self.recomp_a) = self._slice_backward(backward_base)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


# ============================================================================
# Buffer factory
# ============================================================================

def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = 'swiglu',
) -> SymmBuffer:
    """Create a SymmBuffer with token count aligned to kernel requirements."""
    num_max_tokens_per_rank = align(
        num_max_tokens_per_rank,
        _C.get_token_alignment_for_sm90_mega_moe()
    )
    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


# ============================================================================
# Weight transform utilities
# ============================================================================

def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """Interleave gate/up: [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...]"""
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(
        torch.stack([gate, up], dim=2).reshape(g, n, *rest)
    )


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Transform weights for SM100 (Blackwell) FP8/FP4 MegaMoE forward kernel.

    L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    L2: only transpose SF for UTCCP.
    """
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """SM90 (Hopper) variant of weight transform.

    SM90 has no TMEM / UTCCP path, so the SF tensors are consumed directly by
    WGMMA and don't need the 4x32 transpose. Only L1's gate/up FP8 weight
    interleave is preserved.
    """
    l1_fp8, l1_sf = l1_weights

    def _interleave_one(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(
            torch.stack([gate, up], dim=2).reshape(g, n, *rest)
        )

    return (_interleave_one(l1_fp8), l1_sf), l2_weights


# ============================================================================
# Forward kernel wrapper
# ============================================================================

def fp8_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (128, 128, 128),
    activation: str = 'swiglu',
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
):
    """SM90 FP8 MegaMoE forward kernel."""
    _C.fp8_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


# ============================================================================
# Backward kernel wrapper
# ============================================================================

def fp8_mega_moe_backward(
    dx: torch.Tensor,
    dW1: torch.Tensor,
    dW2: torch.Tensor,
    dy: torch.Tensor,
    l1_weights_bf16: torch.Tensor,
    l2_weights_bf16: torch.Tensor,
    l1_weights_fp8: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recompute: bool = True,
    activation: str = 'swiglu',
):
    """Compute MegaMoE backward pass in-place.

    Args:
        dx: [T, H] BF16 output — input gradient
        dW1: [E_local, 2*IH, H] FP32 — accumulated L1 weight gradient
        dW2: [E_local, H, IH] FP32 — accumulated L2 weight gradient
        dy: [T, H] BF16 — incoming gradient of loss w.r.t. MoE output
        l1_weights_bf16: [E_local, 2*IH, H] BF16 master weights
        l2_weights_bf16: [E_local, H, IH] BF16 master weights
        l1_weights_fp8: (FP8 tensor, SF tensor) for recomputation
        sym_buffer: SymmBuffer with with_backward=True
        cumulative_local_expert_recv_stats: optional precomputed expert counts
        recompute: if True, recompute forward activations; else use checkpointed
        activation: activation function name ('swiglu')
    """
    assert sym_buffer.with_backward, \
        "SymmBuffer must be created with with_backward=True for backward pass"

    group = sym_buffer.group
    num_ranks = group.size()
    rank = group.rank()
    num_experts_per_rank = sym_buffer.num_experts // num_ranks
    num_tokens = dy.shape[0]

    # FULL per-local-expert counts across all ranks.
    # The pool layout needs total counts from all ranks, not just local ones.
    local_topk = sym_buffer.topk_idx[:num_tokens].contiguous()
    global_topk = (
        uneven_all_gather(local_topk, group=group) if num_ranks > 1 else local_topk
    )
    flat = global_topk.reshape(-1)
    valid = flat >= 0
    routed_to_us = valid & ((flat // num_experts_per_rank) == rank)
    local_eids = (flat % num_experts_per_rank).to(torch.long)
    expert_counts = torch.bincount(
        local_eids[routed_to_us], minlength=num_experts_per_rank
    ).to(torch.int32)

    _C.fp8_mega_moe_backward(
        dx, dW1, dW2, dy,
        l1_weights_bf16, l2_weights_bf16,
        l1_weights_fp8,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, rank,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recompute, activation, expert_counts
    )


# ============================================================================
# Autograd Function
# ============================================================================

class MegaMoEFunction(torch.autograd.Function):
    """Autograd wrapper for fused MegaMoE forward + backward."""

    @staticmethod
    def forward(
        ctx, x, x_sf, topk_idx, topk_weights,
        l1_w_fp8, l1_w_sf, l1_w_bf16,
        l2_w_fp8, l2_w_sf, l2_w_bf16,
        sym_buffer, cumulative_local_expert_recv_stats=None,
        recipe=(128, 128, 128), activation='swiglu',
        activation_clamp=None, fast_math=True
    ):
        T, H = x.shape[0], l1_w_bf16.shape[2]
        y = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
        fp8_mega_moe(
            y, (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf),
            sym_buffer, cumulative_local_expert_recv_stats,
            recipe, activation, activation_clamp, fast_math
        )

        ctx.save_for_backward(
            topk_idx, topk_weights,
            l1_w_fp8, l1_w_sf, l1_w_bf16,
            l2_w_fp8, l2_w_sf, l2_w_bf16
        )
        ctx.sym_buffer = sym_buffer
        ctx.cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats
        ctx.activation = activation
        return y

    @staticmethod
    def backward(ctx, dy):
        (topk_idx, topk_weights,
         l1_w_fp8, l1_w_sf, l1_w_bf16,
         l2_w_fp8, l2_w_sf, l2_w_bf16) = ctx.saved_tensors

        T = dy.shape[0]
        H = l1_w_bf16.shape[2]
        E_local = l1_w_bf16.shape[0]
        IH_2 = l1_w_bf16.shape[1]
        IH = l2_w_bf16.shape[2]

        dx = torch.empty(T, H, dtype=torch.bfloat16, device=dy.device)
        dW1 = torch.zeros(E_local, IH_2, H, dtype=torch.float32, device=dy.device)
        dW2 = torch.zeros(E_local, H, IH, dtype=torch.float32, device=dy.device)

        fp8_mega_moe_backward(
            dx, dW1, dW2, dy,
            l1_w_bf16, l2_w_bf16,
            (l1_w_fp8, l1_w_sf),
            ctx.sym_buffer,
            ctx.cumulative_local_expert_recv_stats,
            recompute=True,
            activation=ctx.activation
        )

        # Returns: x, x_sf, topk_idx, topk_weights,
        #          l1_w_fp8, l1_w_sf, l1_w_bf16,
        #          l2_w_fp8, l2_w_sf, l2_w_bf16,
        #          sym_buffer, cumulative_local_expert_recv_stats,
        #          recipe, activation, activation_clamp, fast_math
        return (dx, None, None, None,
                None, None, dW1,
                None, None, dW2,
                None, None,
                None, None, None, None)
