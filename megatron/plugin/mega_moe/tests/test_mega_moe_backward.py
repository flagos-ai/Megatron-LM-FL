"""SM90 (Hopper) MegaMoE backward-pass correctness test.

Phase 1 (--accuracy): Reference self-consistency check using FP32 PyTorch.
Phase 2 (--fused): Fused NVLink kernel vs FP32 reference comparison.
Phase 3 (--search-tol): Empirical diff tolerance sweep across shapes.
Phase 4 (--speedup): Fused kernel vs PyTorch baseline timing comparison.
Phase 5 (--validate): Fused-vs-FP32 correctness validation with per-output thresholds.

Validates:
  * dx  — input gradient (BF16, [T, H])
  * dW1 — layer-1 weight gradient (FP32, [E_local, 2*IH, H])
  * dW2 — layer-2 weight gradient (FP32, [E_local, H, IH])

Expected numerical error sources for the fused kernel:
  * BF16 intermediate storage vs FP32 reference
  * FP8 quantized inputs/weights (per-128 and block-128 SF)
  * WGMMA hardware rounding (BF16 dot product)
  * atomicAdd accumulation order for weight gradients
  * NVLink communication ordering (scatter/gather)

Architecture note (post Bug-1..6 fixes):
  The L2 kernel fuses SwiGLU backward into its epilogue — d_a = d_o @ W2 is computed
  in registers and immediately passed through SwiGLU BW, writing d_h directly to
  d_h_buffer. The sym_buffer.d_a_pool is a LEGACY buffer (always zero).

Recommended default --diff-tol: 0.10 (based on forward 0.07 + backward chaining)

Usage (single-node multi-GPU):
  torchrun --nproc_per_node=NUM_GPUS tests/test_mega_moe_backward.py --accuracy
  torchrun --nproc_per_node=NUM_GPUS tests/test_mega_moe_backward.py --fused
  torchrun --nproc_per_node=NUM_GPUS tests/test_mega_moe_backward.py --search-tol
  torchrun --nproc_per_node=NUM_GPUS tests/test_mega_moe_backward.py --speedup
  torchrun --nproc_per_node=NUM_GPUS tests/test_mega_moe_backward.py --validate
"""

import argparse
import math
import os
import sys
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Tuple, Optional, Dict, Any

try:
    import deep_ep as _deep_ep
    _deep_ep_import_error = None
except Exception as ex:
    _deep_ep = None
    _deep_ep_import_error = ex

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major

FP8_E4M3_MAX = 448.0
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)
WEIGHT_SF_GRAN_MN = 128
WEIGHT_SF_GRAN_K = 128


# ============================================================================
# Section 0: DeepEP helpers for baseline dispatch/combine communication
# ============================================================================

def _import_deep_ep():
    if _deep_ep is None:
        dist_print(f"Failed to import deep_ep: {_deep_ep_import_error}", once_in_node=True)
        return None
    return _deep_ep


class _DeepEPHandle:
    def __init__(self, raw_handle, psum_num_recv_tokens_per_expert: torch.Tensor):
        self.raw_handle = raw_handle
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert


class _DeepEPBufferCompat:
    """Compatibility shim for DeepEP Buffer used in backward baseline."""

    def __init__(self, deep_ep, group, num_nvl_bytes: int):
        self.buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=0,
            explicitly_destroy=True,
        )

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_experts: int,
        expert_alignment: int,
        **_,
    ):
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = (
            self.buffer.get_dispatch_layout(topk_idx, num_experts)
        )
        recv_x, _, recv_topk_weights, num_recv_tokens_per_expert, raw_handle, event = self.buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
        )
        psum = torch.tensor(
            num_recv_tokens_per_expert, dtype=torch.int, device=topk_idx.device
        ).cumsum(dim=0, dtype=torch.int)
        return recv_x, recv_topk_weights, _DeepEPHandle(raw_handle, psum), event

    def combine(self, x, *, handle):
        raw_handle = handle.raw_handle if isinstance(handle, _DeepEPHandle) else handle
        return self.buffer.combine(x, handle=raw_handle)

    def destroy(self):
        self.buffer.destroy()


def _make_deep_ep_buffer_for_backward(deep_ep, group, num_max_tokens_per_rank, hidden, num_topk):
    """Create a DeepEP buffer for the backward baseline (BF16 dispatch)."""
    # Estimate NVLink buffer size needed for BF16 dy dispatch + BF16 dx combine.
    # Each token dispatched is hidden*2 bytes (BF16); with topk expansion and
    # per-rank buffering we need generous headroom.
    num_nvl_bytes_estimate = (
        num_max_tokens_per_rank * num_topk * hidden * 2  # BF16 tokens
        * group.size() * 4  # headroom for alignment/metadata
    )
    nvl_alignment = 2 * 1024 * 1024
    num_nvl_bytes = ((num_nvl_bytes_estimate + nvl_alignment - 1) // nvl_alignment) * nvl_alignment
    num_nvl_bytes = max(num_nvl_bytes, 64 * 1024 * 1024)  # at least 64 MB
    return _DeepEPBufferCompat(deep_ep, group, num_nvl_bytes=num_nvl_bytes)


# ============================================================================
# Section 1: Triton SwiGLU backward kernel
# ============================================================================

@triton.jit
def _swiglu_backward_kernel(
    d_a_ptr,       # [M, IH] BF16 — gradient of post-activation
    gate_ptr,      # [M, IH] BF16 — pre-activation gate (h[:, :IH])
    up_ptr,        # [M, IH] BF16 — pre-activation up (h[:, IH:])
    topk_w_ptr,    # [M] FP32 — topk weights (optional)
    d_gate_ptr,    # [M, IH] BF16 — output: gradient for gate
    d_up_ptr,      # [M, IH] BF16 — output: gradient for up
    M, IH,
    stride_da_m, stride_da_n,
    stride_g_m, stride_g_n,
    stride_u_m, stride_u_n,
    stride_dg_m, stride_dg_n,
    stride_du_m, stride_du_n,
    HAS_TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward of SwiGLU: a = silu(gate) * up * topk_w

    Given d_a (gradient of a):
      d_swiglu = d_a * topk_w  (chain rule through scalar multiply)
      sig = sigmoid(gate)
      silu_gate = gate * sig
      d_up   = d_swiglu * silu_gate
      d_gate = d_swiglu * up * sig * (1 + gate * (1 - sig))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask = mask_m[:, None] & (offs_n[None, :] < IH)

    # Load inputs
    da = tl.load(d_a_ptr + offs_m[:, None] * stride_da_m + offs_n[None, :] * stride_da_n,
                 mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offs_m[:, None] * stride_g_m + offs_n[None, :] * stride_g_n,
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs_m[:, None] * stride_u_m + offs_n[None, :] * stride_u_n,
                 mask=mask, other=0.0).to(tl.float32)

    # Apply topk weight (chain rule: d_swiglu = d_a * topk_w)
    if HAS_TOPK:
        w = tl.load(topk_w_ptr + offs_m, mask=mask_m, other=1.0)
        d_swiglu = da * w[:, None]
    else:
        d_swiglu = da

    # SwiGLU backward
    sig = tl.sigmoid(gate)
    silu_gate = gate * sig
    d_up_val = d_swiglu * silu_gate
    d_gate_val = d_swiglu * up * sig * (1.0 + gate * (1.0 - sig))

    # Store
    tl.store(d_gate_ptr + offs_m[:, None] * stride_dg_m + offs_n[None, :] * stride_dg_n,
             d_gate_val.to(tl.bfloat16), mask=mask)
    tl.store(d_up_ptr + offs_m[:, None] * stride_du_m + offs_n[None, :] * stride_du_n,
             d_up_val.to(tl.bfloat16), mask=mask)


def swiglu_backward_triton(
    d_a: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    topk_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton SwiGLU backward. Returns (d_gate, d_up) in BF16."""
    assert d_a.is_cuda and d_a.dtype == torch.bfloat16
    M, IH = d_a.shape
    assert gate.shape == (M, IH) and up.shape == (M, IH)

    d_gate = torch.empty_like(gate)
    d_up = torch.empty_like(up)

    BLOCK_M, BLOCK_N = 16, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(IH, BLOCK_N))

    topk_ptr = topk_weights if topk_weights is not None else d_a

    _swiglu_backward_kernel[grid](
        d_a, gate, up, topk_ptr,
        d_gate, d_up,
        M, IH,
        d_a.stride(0), d_a.stride(1),
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        d_gate.stride(0), d_gate.stride(1),
        d_up.stride(0), d_up.stride(1),
        HAS_TOPK=(topk_weights is not None),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return d_gate, d_up


# ============================================================================
# Section 2: PyTorch FP32 reference backward
# ============================================================================

def _swiglu_fp32(gate_up: torch.Tensor, clamp: float = float('inf')) -> torch.Tensor:
    """Forward SwiGLU in FP32."""
    half = gate_up.size(-1) // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _swiglu_backward_fp32(
    d_a: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    topk_w: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU backward in FP32. Returns (d_gate, d_up).

    Forward: a = silu(gate) * up * topk_w
    Chain rule: d_swiglu = d_a * topk_w (gradient passes through the scalar multiply)
    """
    if topk_w is not None:
        d_swiglu = d_a * topk_w.unsqueeze(-1)
    else:
        d_swiglu = d_a
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    d_up = d_swiglu * silu_gate
    d_gate = d_swiglu * up * sig * (1.0 + gate * (1.0 - sig))
    return d_gate, d_up


def _dequant_per_token_per_128k(x_fp8: torch.Tensor, x_sf: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 with per-token-per-128K scale factors to FP32."""
    M, K = x_fp8.shape
    num_groups = K // 128
    x_f32 = x_fp8.float().view(M, num_groups, 128)
    sf = x_sf.view(M, num_groups, 1)
    return (x_f32 * sf).view(M, K)


def _dequant_block_128_128(w_fp8: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight with block (128,128) scale factors to FP32."""
    N, K = w_fp8.shape
    n_blocks = N // WEIGHT_SF_GRAN_MN
    k_blocks = K // WEIGHT_SF_GRAN_K
    w_f32 = w_fp8.float().view(n_blocks, WEIGHT_SF_GRAN_MN, k_blocks, WEIGHT_SF_GRAN_K)
    sf = w_sf.view(n_blocks, 1, k_blocks, 1)
    return (w_f32 * sf).view(N, K)


def reference_mega_moe_backward(
    dy_local: torch.Tensor,
    x_fp8_local: torch.Tensor,
    x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor,
    topk_weights_local: torch.Tensor,
    l1_w_fp8: torch.Tensor,
    l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor,
    l2_w_sf: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float = float('inf'),
    l1_w_bf16: Optional[torch.Tensor] = None,
    l2_w_bf16: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch FP32 reference backward for MegaMoE.

    The fused kernel uses:
      - FP8 weights for forward recomputation (matching the forward pass)
      - BF16 master weights for backward GEMMs (d_a = d_o @ W2_bf16, dx = d_h @ W1_bf16)

    If l1_w_bf16/l2_w_bf16 are provided, they are used for backward GEMMs to match
    the kernel's behavior. Otherwise falls back to dequantized FP8 for both.

    Returns:
        dx_local: [T_local, H] FP32 — input gradient for this rank
        dW1_local: [E_local, 2*IH, H] FP32 — weight gradient for L1
        dW2_local: [E_local, H, IH] FP32 — weight gradient for L2
    """
    num_experts_per_rank = num_experts // num_ranks

    # All-gather inputs across ranks
    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)
    dy_g = uneven_all_gather(dy_local, group=group)
    T_global = x_fp8_g.size(0)

    # Get per-rank sizes
    local_size = torch.tensor([x_fp8_local.size(0)], device="cuda", dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device="cuda")
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()

    # Dequantize inputs
    x_fp32 = _dequant_per_token_per_128k(x_fp8_g, x_sf_g)

    # Accumulate gradients
    dx_global = torch.zeros(T_global, hidden, dtype=torch.float32, device="cuda")
    dW1_local = torch.zeros(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                            dtype=torch.float32, device="cuda")
    dW2_local = torch.zeros(num_experts_per_rank, hidden, intermediate_hidden,
                            dtype=torch.float32, device="cuda")

    # Process per-expert, per-topk-slot
    for k in range(num_topk):
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        eids = topk_idx_g[sel_idx, k]
        weights = topk_w_g[sel_idx, k]

        dst_rank = (eids // num_experts_per_rank).long()
        dst_local = (eids % num_experts_per_rank).long()

        # Only process experts owned by this rank
        local_mask = dst_rank == rank_idx
        if not local_mask.any():
            continue
        local_sel = sel_idx[local_mask]
        local_expert_idx = dst_local[local_mask]
        local_weights = weights[local_mask]

        x_sel = x_fp32[local_sel]       # [S, H]
        dy_sel = dy_g[local_sel].float() # [S, H]

        # Forward recomputation per expert
        for e in range(num_experts_per_rank):
            e_mask = local_expert_idx == e
            if not e_mask.any():
                continue
            e_sel = e_mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x_sel[e_sel]           # [Se, H]
            dy_e = dy_sel[e_sel]         # [Se, H]
            w_e = local_weights[e_sel]   # [Se]
            global_idx = local_sel[e_sel]

            # The kernel uses BF16 master weights for ALL operations in backward:
            #   - Forward recompute: h = x @ W1_bf16^T (Phase 1, line 288-289 in API)
            #   - L2 backward GEMM: d_a = d_o @ W2_bf16 (L2 kernel TMA)
            #   - L1 backward GEMM: dx = d_h @ W1_bf16 (L1 kernel TMA)
            # FP8 weights are only used for the FORWARD PASS (storing l1_acts/l2_acts).
            if l1_w_bf16 is not None:
                w1 = l1_w_bf16[e].float()  # [2*IH, H]
            else:
                w1 = _dequant_block_128_128(l1_w_fp8[e], l1_w_sf[e])
            if l2_w_bf16 is not None:
                w2 = l2_w_bf16[e].float()  # [H, IH]
            else:
                w2 = _dequant_block_128_128(l2_w_fp8[e], l2_w_sf[e])

            # Forward recompute: h = x @ W1^T
            h = x_e @ w1.t()  # [Se, 2*IH] FP32
            gate_e = h[:, :intermediate_hidden]
            up_e = h[:, intermediate_hidden:]

            if math.isfinite(activation_clamp):
                gate_e = gate_e.clamp(max=activation_clamp)
                up_e = up_e.clamp(min=-activation_clamp, max=activation_clamp)

            # Forward recompute: a = silu(gate) * up * topk_w
            sig = torch.sigmoid(gate_e)
            silu_gate = gate_e * sig
            a = silu_gate * up_e * w_e.unsqueeze(-1)  # [Se, IH]

            # --- Backward L2 ---
            # d_o = dy (combine backward is identity fan-out)
            d_o = dy_e  # [Se, H]

            # d_a = d_o @ W2  (kernel uses BF16 master weights)
            d_a = d_o @ w2   # [Se, IH]

            # dW2 += d_o^T @ a
            dW2_local[e] += d_o.t() @ a  # [H, IH]

            # --- SwiGLU backward ---
            d_gate, d_up = _swiglu_backward_fp32(d_a, gate_e, up_e, w_e)

            # d_h = concat(d_gate, d_up)
            d_h = torch.cat([d_gate, d_up], dim=-1)  # [Se, 2*IH]

            # --- Backward L1 ---
            # d_x = d_h @ W1  (kernel uses BF16 master weights)
            d_x = d_h @ w1   # [Se, H]

            # dW1 += d_h^T @ x
            dW1_local[e] += d_h.t() @ x_e  # [2*IH, H]

            # Accumulate dx
            dx_global[global_idx] += d_x

    # Reduce dx across ranks (each rank has contributions from its experts)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    dist.all_reduce(dx_global, op=dist.ReduceOp.SUM, group=group)
    dx_local = dx_global[start:end].contiguous()

    return dx_local, dW1_local, dW2_local


def reference_d_a_and_dh(
    dy_local: torch.Tensor,
    x_fp8_local: torch.Tensor,
    x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor,
    topk_weights_local: torch.Tensor,
    l1_w_fp8: torch.Tensor,
    l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor,
    l2_w_sf: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reference d_a (= dy @ W2) and d_h (= SwiGLU backward output) per
    local expert, concatenated across experts. Order-independent (norm/NaN only).

    Used by the fused-intermediate diagnostic to bisect whether the L2 GEMM
    (produces d_a) or the SwiGLU backward (produces d_h from d_a) is the bug.

    Returns (d_a_cat [S, IH], d_h_cat [S, 2*IH]) in FP32.
    """
    num_experts_per_rank = num_experts // num_ranks
    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)
    dy_g = uneven_all_gather(dy_local, group=group)
    x_fp32 = _dequant_per_token_per_128k(x_fp8_g, x_sf_g)

    da_pe = [[] for _ in range(num_experts_per_rank)]
    dh_pe = [[] for _ in range(num_experts_per_rank)]
    for k in range(num_topk):
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        eids = topk_idx_g[sel_idx, k]
        weights = topk_w_g[sel_idx, k]
        dst_rank = (eids // num_experts_per_rank).long()
        dst_local = (eids % num_experts_per_rank).long()
        local_mask = dst_rank == rank_idx
        if not local_mask.any():
            continue
        local_sel = sel_idx[local_mask]
        local_expert_idx = dst_local[local_mask]
        local_weights = weights[local_mask]
        x_sel = x_fp32[local_sel]
        dy_sel = dy_g[local_sel].float()
        for e in range(num_experts_per_rank):
            e_mask = local_expert_idx == e
            if not e_mask.any():
                continue
            e_sel = e_mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x_sel[e_sel]
            dy_e = dy_sel[e_sel]
            w_e = local_weights[e_sel]
            w1 = _dequant_block_128_128(l1_w_fp8[e], l1_w_sf[e])
            w2 = _dequant_block_128_128(l2_w_fp8[e], l2_w_sf[e])
            h = x_e @ w1.t()
            gate_e = h[:, :intermediate_hidden]
            up_e = h[:, intermediate_hidden:]
            d_a = dy_e @ w2                                   # L2 activation gradient
            d_gate, d_up = _swiglu_backward_fp32(d_a, gate_e, up_e, w_e)
            d_h = torch.cat([d_gate, d_up], dim=-1)
            da_pe[e].append(d_a)
            dh_pe[e].append(d_h)

    dev = x_fp8_local.device
    da_list = [torch.cat(da_pe[e], 0) if da_pe[e]
               else torch.zeros(0, intermediate_hidden, device=dev)
               for e in range(num_experts_per_rank)]
    dh_list = [torch.cat(dh_pe[e], 0) if dh_pe[e]
               else torch.zeros(0, 2 * intermediate_hidden, device=dev)
               for e in range(num_experts_per_rank)]
    return da_list, dh_list


# ============================================================================
# Section 3: Test scenarios
# ============================================================================

def _run_backward_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    diff_tol: float,
):
    """Run a single backward correctness scenario."""
    num_max = cfg["num_max_tokens_per_rank"]
    num_tokens = cfg.get("num_tokens", num_max)
    hidden = cfg["hidden"]
    intermediate_hidden = cfg["intermediate_hidden"]
    num_experts = cfg["num_experts"]
    num_topk = cfg["num_topk"]
    activation_clamp = cfg.get("activation_clamp", float('inf'))

    assert num_experts % num_ranks == 0
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    # Generate random inputs
    x_bf16 = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)

    # Random routing
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                             dtype=torch.int64, device="cuda")
    topk_weights = torch.rand(num_tokens, num_topk, dtype=torch.float32,
                              device="cuda") * 0.5 + 0.5

    # Random weights (FP8 quantized)
    l1_w_bf16 = torch.randn(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                            dtype=torch.bfloat16, device="cuda") * 0.01
    l2_w_bf16 = torch.randn(num_experts_per_rank, hidden, intermediate_hidden,
                            dtype=torch.bfloat16, device="cuda") * 0.01

    # Quantize weights to FP8 with block (128,128) SF
    # Process per-expert to avoid OOM on large expert counts (e.g. 256 experts)
    def _quantize_weight_block(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        E, N, K = w.shape
        n_blocks = N // WEIGHT_SF_GRAN_MN
        k_blocks = K // WEIGHT_SF_GRAN_K
        w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
        w_sf_out = torch.empty(E, n_blocks, k_blocks, dtype=torch.float32, device=w.device)
        for i in range(E):
            wi = w[i].float().view(n_blocks, WEIGHT_SF_GRAN_MN, k_blocks, WEIGHT_SF_GRAN_K)
            amax = wi.abs().amax(dim=(1, 3))  # [n_blocks, k_blocks]
            sf = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
            wi_scaled = wi / sf[:, None, :, None]
            w_fp8[i] = wi_scaled.reshape(N, K).to(torch.float8_e4m3fn)
            w_sf_out[i] = sf
        return w_fp8, w_sf_out

    l1_w_fp8, l1_w_sf = _quantize_weight_block(l1_w_bf16)
    l2_w_fp8, l2_w_sf = _quantize_weight_block(l2_w_bf16)
    del l1_w_bf16, l2_w_bf16  # Free BF16 master weights to save memory

    # Random dy
    dy = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.01

    # Run reference backward
    dx_ref, dW1_ref, dW2_ref = reference_mega_moe_backward(
        dy, x_fp8, x_sf, topk_idx, topk_weights,
        l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk, hidden, intermediate_hidden,
        activation_clamp,
    )

    # Validate (self-consistency check — the reference is the golden truth)
    dx_norm = dx_ref.norm().item()
    dW1_norm = dW1_ref.norm().item()
    dW2_norm = dW2_ref.norm().item()

    passed = dx_norm > 0 and dW1_norm > 0 and dW2_norm > 0
    status = "PASS" if passed else "FAIL"
    dist_print(f"  [{status}] {name}: "
               f"|dx|={dx_norm:.4f}, |dW1|={dW1_norm:.4f}, |dW2|={dW2_norm:.4f}",
               rank_idx == 0)
    return passed


# ============================================================================
# Section 4: Fused kernel backward test
# ============================================================================

def _run_fused_backward_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    diff_tol: float,
) -> Tuple[bool, Dict[str, float]]:
    """Run a fused kernel backward scenario and compare against FP32 reference.

    Returns:
        (passed, diffs): whether all diffs are within tolerance, and the diff dict
    """
    import deep_gemm
    from deep_gemm.mega import SymmBuffer, fp8_mega_moe_backward
    from deep_gemm.utils.math import align

    num_max = cfg["num_max_tokens_per_rank"]
    num_tokens = cfg.get("num_tokens", num_max)
    hidden = cfg["hidden"]
    intermediate_hidden = cfg["intermediate_hidden"]
    num_experts = cfg["num_experts"]
    num_topk = cfg["num_topk"]
    activation_clamp = cfg.get("activation_clamp", float('inf'))

    assert num_experts % num_ranks == 0
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    # Align num_max_tokens_per_rank
    alignment = deep_gemm._C.get_token_alignment_for_sm90_mega_moe()
    num_max_aligned = align(num_max, alignment)

    # Generate random inputs
    x_bf16 = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)

    # Random routing
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                             dtype=torch.int64, device="cuda")
    topk_weights = torch.rand(num_tokens, num_topk, dtype=torch.float32,
                              device="cuda") * 0.5 + 0.5

    # Random BF16 master weights
    l1_w_bf16 = torch.randn(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                            dtype=torch.bfloat16, device="cuda") * 0.01
    l2_w_bf16 = torch.randn(num_experts_per_rank, hidden, intermediate_hidden,
                            dtype=torch.bfloat16, device="cuda") * 0.01

    # Quantize weights to FP8 with block (128,128) SF
    # Process per-expert to avoid OOM on large expert counts (e.g. 256 experts)
    def _quantize_weight_block(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        E, N, K = w.shape
        n_blocks = N // WEIGHT_SF_GRAN_MN
        k_blocks = K // WEIGHT_SF_GRAN_K
        w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
        w_sf_out = torch.empty(E, n_blocks, k_blocks, dtype=torch.float32, device=w.device)
        for i in range(E):
            wi = w[i].float().view(n_blocks, WEIGHT_SF_GRAN_MN, k_blocks, WEIGHT_SF_GRAN_K)
            amax = wi.abs().amax(dim=(1, 3))  # [n_blocks, k_blocks]
            sf = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
            wi_scaled = wi / sf[:, None, :, None]
            w_fp8[i] = wi_scaled.reshape(N, K).to(torch.float8_e4m3fn)
            w_sf_out[i] = sf
        return w_fp8, w_sf_out

    l1_w_fp8, l1_w_sf = _quantize_weight_block(l1_w_bf16)
    l2_w_fp8, l2_w_sf = _quantize_weight_block(l2_w_bf16)

    # Random dy
    dy = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.01

    # ─── FP32 Reference ─────────────────────────────────────────────────
    dx_ref, dW1_ref, dW2_ref = reference_mega_moe_backward(
        dy, x_fp8, x_sf, topk_idx, topk_weights,
        l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk, hidden, intermediate_hidden,
        activation_clamp,
        l1_w_bf16=l1_w_bf16, l2_w_bf16=l2_w_bf16,
    )

    # ─── Fused Kernel ───────────────────────────────────────────────────
    # Create SymmBuffer with backward support
    sym_buffer = SymmBuffer(
        group, num_experts,
        num_max_aligned, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch=True,
        activation='swiglu',
        with_backward=True,
    )

    # Fill the forward checkpoint buffers (simulates what the forward pass stores)
    sym_buffer.x[:num_tokens].copy_(x_fp8)
    sym_buffer.x_sf[:num_tokens].copy_(x_sf)
    sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
    sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

    # Run forward to populate l1_acts, l2_acts in the sym_buffer
    # (we need these for recomputation in backward)
    y_fwd = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    l1_transformed, l2_transformed = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))

    cum_stats_fwd = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")

    deep_gemm.fp8_mega_moe(
        y_fwd,
        l1_transformed,
        l2_transformed,
        sym_buffer,
        cumulative_local_expert_recv_stats=cum_stats_fwd,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=True,
    )
    torch.cuda.synchronize()

    # Now run backward (forward pass left l1_acts, l2_acts, x, topk in sym_buffer)
    # Backward kernel expects numel >= num_experts_per_rank + 1 (cumulative prefix)
    cum_stats_bwd = torch.zeros((num_experts_per_rank + 1,), dtype=torch.int, device="cuda")

    dx_fused = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    dW1_fused = torch.zeros(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                            dtype=torch.float32, device="cuda")
    dW2_fused = torch.zeros(num_experts_per_rank, hidden, intermediate_hidden,
                            dtype=torch.float32, device="cuda")

    fp8_mega_moe_backward(
        dx_fused, dW1_fused, dW2_fused, dy,
        l1_w_bf16, l2_w_bf16,
        (l1_w_fp8, l1_w_sf),
        sym_buffer,
        cumulative_local_expert_recv_stats=cum_stats_bwd,
        recompute=True,
        activation='swiglu',
    )
    torch.cuda.synchronize()

    # ─── Intermediate diagnostics ──────────────────────────────────────────
    # Run with DG_BWD_DEBUG=1. Reads the kernel's internal buffers directly
    # from sym_buffer (no recompile needed) to localize failures.
    #
    # Architecture note (post Bug-1..6 fixes):
    #   The L2 kernel fuses SwiGLU backward into its epilogue — it computes
    #   d_a = d_o @ W2 in registers, immediately applies SwiGLU BW, and writes
    #   d_h directly to d_h_buffer (a C++ local, not in sym_buffer). The
    #   sym_buffer.d_a_pool is a LEGACY buffer kept zeroed for TMA descriptor
    #   ABI compatibility — it is never written by the kernel.
    #
    # What IS observable from Python:
    #   * d_o_pool  — dispatch-filled dy gather (NVLink pull). NaN here → dispatch bug.
    #   * recomp_h  — forward gate/up checkpoint (fwd order, used as SwiGLU BW input).
    #   * recomp_a  — dequantized l2_acts (forward activations, reordered to bwd layout).
    #   * dx_fused  — final output (L1 GEMM + combine scatter result).
    #   * dx/dW1/dW2 cosine similarity — end-to-end correctness.
    def _bwd_diag(d_a_ref_list, d_h_ref_list, global_counts):
        def _diag_stats(label: str, t: Optional[torch.Tensor]) -> Tuple[bool, float]:
            if t is None:
                return False, 0.0
            f = t.float()
            has_nan = bool(torch.isnan(f).any().item())
            n = float(torch.nan_to_num(f, nan=0.0).norm().item())
            print(f"    [diag] {label:35s}: nan={int(has_nan)} "
                  f"norm={n:.4f} shape={tuple(t.shape)} dtype={t.dtype}")
            return has_nan, n

        print(f"  >>> DG_BWD_DEBUG intermediates for '{name}'")
        do_nan, _ = _diag_stats("d_o_pool (dispatch=dy gather)", sym_buffer.d_o_pool)
        _diag_stats("recomp_h (fwd gate/up checkpoint)", sym_buffer.recomp_h)
        _diag_stats("recomp_a (fwd l2_acts, reordered)", sym_buffer.recomp_a)
        _diag_stats("d_a_pool (LEGACY, always zero)", sym_buffer.d_a_pool)
        _diag_stats("dx_fused (final output)", dx_fused)

        # Reference d_a / d_h for norm comparison
        d_a_ref_cat = torch.cat([d for d in d_a_ref_list if d.numel() > 0], dim=0)
        d_h_ref_cat = torch.cat([d for d in d_h_ref_list if d.numel() > 0], dim=0)
        print(f"    [diag] {'ref d_a (dy@W2, FP32)':35s}: "
              f"nan={int(bool(torch.isnan(d_a_ref_cat).any().item()))} "
              f"norm={d_a_ref_cat.norm().item():.4f}")
        print(f"    [diag] {'ref d_h (SwiGLU bw, FP32)':35s}: "
              f"nan={int(bool(torch.isnan(d_h_ref_cat).any().item()))} "
              f"norm={d_h_ref_cat.norm().item():.4f}")

        # Diagnosis based on observable state
        if do_nan:
            print("    [diag] >>> d_o_pool has NaN: dispatch (NVLink dy pull) bug.")
        else:
            print("    [diag] >>> d_o_pool clean. L2 dispatch working correctly.")
            print("    [diag]     (d_a is fused into d_h in L2 epilogue — not observable)")

        # ── End-to-end validation: dx / dW1 / dW2 ──
        diff_dx = calc_diff(dx_fused.float(), dx_ref.float())
        diff_dW1 = calc_diff(dW1_fused, dW1_ref)
        diff_dW2 = calc_diff(dW2_fused, dW2_ref)
        print(f"    [diag] end-to-end cosine: dx={diff_dx:.5f} "
              f"dW1={diff_dW1:.5f} dW2={diff_dW2:.5f}")
        if diff_dx > 0.5 or diff_dW1 > 0.5 or diff_dW2 > 0.5:
            print("    [diag] >>> LARGE deviation detected — possible kernel bug.")
            if diff_dW2 > 0.5 and diff_dx < 0.2:
                print("    [diag]     dW2 bad but dx OK → Phase-3 cuBLASLt / "
                      "recomp_a reorder issue.")
            elif diff_dx > 0.5 and diff_dW1 < 0.2:
                print("    [diag]     dx bad but dW1 OK → L1 GEMM epilogue / "
                      "combine scatter issue.")
        else:
            print("    [diag] >>> All outputs within expected BF16/FP8 tolerance.")

        # ── Validate d_o_pool integrity per-expert using global counts ──
        # Use the GLOBAL expert counts (all ranks' tokens routed to this rank's
        # experts) for correct pool layout, matching the kernel's actual layout.
        kPoolBlockM = 64
        num_max_pool = int(sym_buffer.d_o_pool.shape[0])

        # global_counts was computed outside this function (on ALL ranks) to avoid
        # calling collectives inside a rank-0-only gate.
        counts = global_counts

        offsets = [0] * num_experts_per_rank
        for e in range(1, num_experts_per_rank):
            offsets[e] = offsets[e - 1] + \
                ((counts[e - 1] + kPoolBlockM - 1) // kPoolBlockM) * kPoolBlockM

        # Check d_o_pool per-expert: non-zero valid rows indicate dispatch worked
        do_pool_f = sym_buffer.d_o_pool.float()
        print("    [deep] d_o_pool per-expert (dispatch verification):")
        for e in range(min(num_experts_per_rank, 8)):
            if counts[e] == 0:
                continue
            sl = slice(offsets[e], offsets[e] + counts[e])
            do_e = do_pool_f[sl]
            e_nan = int(torch.isnan(do_e).any(dim=1).sum().item())
            e_zero_rows = int((do_e.norm(dim=1) == 0).sum().item())
            print(f"      expert {e}: count={counts[e]} nan_rows={e_nan} "
                  f"zero_rows={e_zero_rows} norm={do_e.norm().item():.4f}")

        # ── Validate recomp_a (dequantized l2_acts) ──
        # Forward checkpoint (l1_acts/l2_acts) is 128-padded.
        fwd_offsets = [0] * num_experts_per_rank
        for e in range(1, num_experts_per_rank):
            fwd_offsets[e] = fwd_offsets[e - 1] + \
                ((counts[e - 1] + 128 - 1) // 128) * 128

        ra_f = sym_buffer.recomp_a.float()
        valid_row = torch.zeros(num_max_pool, dtype=torch.bool, device="cuda")
        for e in range(num_experts_per_rank):
            if counts[e] > 0 and offsets[e] + counts[e] <= num_max_pool:
                valid_row[offsets[e]:offsets[e] + counts[e]] = True
        ra_nan_valid = int((torch.isnan(ra_f).any(dim=1) & valid_row).sum().item())
        if ra_nan_valid > 0:
            print(f"    [deep] recomp_a has {ra_nan_valid} NaN at valid rows "
                  f"(reorder or forward checkpoint issue)")
        else:
            print(f"    [deep] recomp_a: no NaN at valid rows (OK)")

    # Compute reference d_a/d_h on ALL ranks when debugging — the all-gathers
    # inside reference_d_a_and_dh are collectives and must run on every rank,
    # not just rank 0, or multi-rank deadlocks.
    if os.environ.get("DG_BWD_DEBUG"):
        try:
            _diag_d_a_ref, _diag_d_h_ref = reference_d_a_and_dh(
                dy, x_fp8, x_sf, topk_idx, topk_weights,
                l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
                rank_idx, num_ranks, group,
                num_experts, num_topk, hidden, intermediate_hidden)
        except Exception as _ref_e:
            _diag_d_a_ref = _diag_d_h_ref = None
            dist_print(f"  [diag] (reference_d_a_and_dh raised: {_ref_e})",
                       rank_idx == 0)

        # Compute global expert counts on ALL ranks (uneven_all_gather is collective).
        # Must run before the rank==0 gate to avoid deadlock.
        _diag_local_topk = topk_idx[:num_tokens].contiguous()
        if num_ranks > 1:
            _diag_global_topk = uneven_all_gather(_diag_local_topk, group=group)
        else:
            _diag_global_topk = _diag_local_topk
        _diag_flat = _diag_global_topk.reshape(-1)
        _diag_valid = _diag_flat >= 0
        _diag_routed = _diag_valid & ((_diag_flat // num_experts_per_rank) == rank_idx)
        _diag_eids = (_diag_flat % num_experts_per_rank).to(torch.long)
        _diag_global_counts = torch.bincount(
            _diag_eids[_diag_routed], minlength=num_experts_per_rank).cpu().tolist()
    else:
        _diag_d_a_ref = _diag_d_h_ref = None
        _diag_global_counts = None

    if os.environ.get("DG_BWD_DEBUG") and rank_idx == 0:
        try:
            _bwd_diag(_diag_d_a_ref, _diag_d_h_ref, _diag_global_counts)
        except Exception as _diag_e:
            dist_print(f"  [diag] (diagnostic raised, ignored): {_diag_e}",
                       rank_idx == 0)

    # ─── Compare ────────────────────────────────────────────────────────
    diff_dx = calc_diff(dx_fused.float(), dx_ref.float())
    diff_dW1 = calc_diff(dW1_fused, dW1_ref)
    diff_dW2 = calc_diff(dW2_fused, dW2_ref)

    # Debug: compare actual values to find where the mismatch is
    if os.environ.get("DG_BWD_DEBUG") and rank_idx == 0:
        dx_f = dx_fused.float()
        dx_r = dx_ref.float()
        # Per-token norm comparison
        fn = dx_f.norm(dim=1)
        rn = dx_r.norm(dim=1)
        # Find token with max diff
        per_tok_diff = (dx_f - dx_r).norm(dim=1)
        worst_tok = int(per_tok_diff.argmax().item())
        print(f"    [debug] dx shapes: fused={tuple(dx_f.shape)} ref={tuple(dx_r.shape)}")
        print(f"    [debug] dx norms: fused={dx_f.norm():.6f} ref={dx_r.norm():.6f}")
        print(f"    [debug] dx[0,:5] fused={dx_f[0,:5].tolist()}")
        print(f"    [debug] dx[0,:5] ref  ={dx_r[0,:5].tolist()}")
        print(f"    [debug] worst token {worst_tok}: "
              f"fused_norm={fn[worst_tok]:.6f} ref_norm={rn[worst_tok]:.6f} "
              f"diff_norm={per_tok_diff[worst_tok]:.6f}")
        print(f"    [debug] dx[{worst_tok},:5] fused={dx_f[worst_tok,:5].tolist()}")
        print(f"    [debug] dx[{worst_tok},:5] ref  ={dx_r[worst_tok,:5].tolist()}")
        # Check if ref is all zeros or very small
        ref_nonzero = int((dx_r.norm(dim=1) > 1e-8).sum().item())
        fused_nonzero = int((dx_f.norm(dim=1) > 1e-8).sum().item())
        print(f"    [debug] nonzero rows: fused={fused_nonzero}/{dx_f.shape[0]} "
              f"ref={ref_nonzero}/{dx_r.shape[0]}")
        # dW1 comparison
        print(f"    [debug] dW1 norms: fused={dW1_fused.norm():.6f} ref={dW1_ref.norm():.6f}")
        print(f"    [debug] dW1[0,0,:5] fused={dW1_fused[0,0,:5].tolist()}")
        print(f"    [debug] dW1[0,0,:5] ref  ={dW1_ref[0,0,:5].tolist()}")

    diffs = {"dx": diff_dx, "dW1": diff_dW1, "dW2": diff_dW2}
    passed = all(d < diff_tol for d in diffs.values())
    status = "PASS" if passed else "FAIL"

    dist_print(
        f"  [{status}] {name}: "
        f"dx={diff_dx:.5f}, dW1={diff_dW1:.5f}, dW2={diff_dW2:.5f} "
        f"(tol={diff_tol:.3f})",
        rank_idx == 0)

    sym_buffer.destroy()
    del dx_fused, dW1_fused, dW2_fused, dx_ref, dW1_ref, dW2_ref
    del l1_w_bf16, l2_w_bf16, l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf
    del l1_transformed, l2_transformed
    del x_bf16, x_fp8, x_sf, dy, y_fwd, topk_idx, topk_weights
    torch.cuda.empty_cache()
    return passed, diffs


def _search_diff_tolerance(
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
) -> Dict[str, float]:
    """Run a sweep of shapes to determine the empirical max diff ratio.

    This helps establish an acceptable tolerance for the fused backward kernel
    vs the FP32 reference. The key sources of numerical difference:
      1. BF16 intermediate storage (vs FP32 reference)
      2. FP8 quantized x and weights (vs full-precision)
      3. WGMMA instruction rounding
      4. atomicAdd accumulation order for dW1/dW2
      5. Multi-rank communication (scatter/gather order)

    Expected ranges (from forward path analogy):
      - dx: 0.01 ~ 0.05 (single GEMM + SwiGLU → another GEMM)
      - dW1: 0.02 ~ 0.08 (outer product accumulated via atomicAdd)
      - dW2: 0.02 ~ 0.08 (same)
    """
    SEARCH_SCENARIOS = [
        {"name": "search_small",
         "num_max_tokens_per_rank": 384, "num_tokens": 384,
         "hidden": 512, "intermediate_hidden": 512,
         "num_experts": 8, "num_topk": 2},
        {"name": "search_medium",
         "num_max_tokens_per_rank": 384, "num_tokens": 384,
         "hidden": 1024, "intermediate_hidden": 512,
         "num_experts": 8, "num_topk": 2},
        {"name": "search_large",
         "num_max_tokens_per_rank": 768, "num_tokens": 768,
         "hidden": 1024, "intermediate_hidden": 512,
         "num_experts": 16, "num_topk": 4},
        {"name": "search_dsv3_like",
         "num_max_tokens_per_rank": 384, "num_tokens": 384,
         "hidden": 7168, "intermediate_hidden": 2048,
         "num_experts": 16, "num_topk": 8},
        {"name": "search_many_experts",
         "num_max_tokens_per_rank": 384, "num_tokens": 384,
         "hidden": 4096, "intermediate_hidden": 2048,
         "num_experts": 64, "num_topk": 4},
        {"name": "search_high_topk",
         "num_max_tokens_per_rank": 768, "num_tokens": 768,
         "hidden": 4096, "intermediate_hidden": 2048,
         "num_experts": 32, "num_topk": 8},
    ]

    max_diffs = {"dx": 0.0, "dW1": 0.0, "dW2": 0.0}

    dist_print("\n--- Tolerance search sweep ---", rank_idx == 0)
    for cfg in SEARCH_SCENARIOS:
        if cfg["num_experts"] % num_ranks != 0:
            continue
        try:
            _, diffs = _run_fused_backward_scenario(
                cfg["name"], cfg, rank_idx, num_ranks, group,
                diff_tol=1.0)  # large tol to always "pass"
            for key in max_diffs:
                max_diffs[key] = max(max_diffs[key], diffs[key])
        except Exception as e:
            dist_print(f"  [SKIP] {cfg['name']}: {e}", rank_idx == 0)

    dist_print(f"\n  Observed max diffs: "
               f"dx={max_diffs['dx']:.5f}, "
               f"dW1={max_diffs['dW1']:.5f}, "
               f"dW2={max_diffs['dW2']:.5f}", rank_idx == 0)

    # Recommend 2× observed max as safe tolerance (with floor of 0.05)
    recommended = {k: max(v * 2.0, 0.05) for k, v in max_diffs.items()}
    overall_rec = max(recommended.values())
    dist_print(f"  Recommended --diff-tol: {overall_rec:.3f} "
               f"(2× max observed, floor=0.05)", rank_idx == 0)
    return max_diffs


# ============================================================================
# Section 4b: Fused-vs-FP32 correctness validation (post Bug-6 fix)
# ============================================================================

VALIDATE_SCENARIOS = [
    {"name": "val_small",
     "num_max_tokens_per_rank": 384, "num_tokens": 384,
     "hidden": 512, "intermediate_hidden": 512,
     "num_experts": 8, "num_topk": 2},
    {"name": "val_medium",
     "num_max_tokens_per_rank": 384, "num_tokens": 384,
     "hidden": 1024, "intermediate_hidden": 512,
     "num_experts": 8, "num_topk": 2},
    {"name": "val_large",
     "num_max_tokens_per_rank": 768, "num_tokens": 768,
     "hidden": 1024, "intermediate_hidden": 512,
     "num_experts": 16, "num_topk": 4},
    {"name": "val_dsv3_like",
     "num_max_tokens_per_rank": 384, "num_tokens": 384,
     "hidden": 7168, "intermediate_hidden": 2048,
     "num_experts": 16, "num_topk": 8},
    {"name": "val_large_experts",
     "num_max_tokens_per_rank": 384, "num_tokens": 384,
     "hidden": 4096, "intermediate_hidden": 2048,
     "num_experts": 64, "num_topk": 4},
    {"name": "val_wide_ih",
     "num_max_tokens_per_rank": 768, "num_tokens": 768,
     "hidden": 4096, "intermediate_hidden": 4096,
     "num_experts": 32, "num_topk": 4},
]

# Per-output tolerance thresholds for fused kernel vs FP32 reference.
# These account for: BF16 intermediate storage, FP8 quantized activations/weights,
# WGMMA rounding, and NVLink communication ordering.
# Smaller shapes have proportionally larger BF16 rounding impact.
VALIDATE_THRESHOLDS = {
    "dx": {"small": 0.10, "large": 0.05},
    "dW1": {"small": 0.15, "large": 0.05},
    "dW2": {"small": 0.15, "large": 0.05},
}


def _run_fused_backward_validation(
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
) -> bool:
    """Validate fused backward kernel outputs against FP32 reference.

    This is the post-fix correctness test that replaces the stale L2 GEMM isolation
    diagnostic. It validates:
      1. No NaN in any output (dx, dW1, dW2)
      2. d_o_pool (dispatch result) has no NaN
      3. Per-output cosine similarity (1 - calc_diff) exceeds threshold
      4. Relative error is bounded for each output independently

    Returns True if all scenarios pass.
    """
    import deep_gemm
    from deep_gemm.mega import SymmBuffer, fp8_mega_moe_backward
    from deep_gemm.utils.math import align

    dist_print("\n--- Fused-vs-FP32 correctness validation ---", rank_idx == 0)
    dist_print(f"{'scenario':<18} {'dx':>8} {'dW1':>8} {'dW2':>8} "
               f"{'dx_nan':>7} {'dW1_nan':>8} {'dW2_nan':>8} {'status':>7}",
               rank_idx == 0)
    dist_print("-" * 80, rank_idx == 0)

    all_passed = True

    for cfg in VALIDATE_SCENARIOS:
        if cfg["num_experts"] % num_ranks != 0:
            continue

        name = cfg["name"]
        num_max = cfg["num_max_tokens_per_rank"]
        num_tokens = cfg.get("num_tokens", num_max)
        hidden = cfg["hidden"]
        intermediate_hidden = cfg["intermediate_hidden"]
        num_experts = cfg["num_experts"]
        num_topk = cfg["num_topk"]
        num_experts_per_rank = num_experts // num_ranks

        # Determine if this is a "large" scenario (more tokens → less rounding noise)
        is_large = (num_tokens * num_topk >= 2048) or (hidden >= 4096)
        thr_key = "large" if is_large else "small"

        torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)

        alignment = deep_gemm._C.get_token_alignment_for_sm90_mega_moe()
        num_max_aligned = align(num_max, alignment)

        # Generate random inputs
        x_bf16 = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        x_fp8, x_sf = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128,
                                             use_packed_ue8m0=False)
        topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                                 dtype=torch.int64, device="cuda")
        topk_weights = torch.rand(num_tokens, num_topk, dtype=torch.float32,
                                  device="cuda") * 0.5 + 0.5

        l1_w_bf16 = torch.randn(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                                dtype=torch.bfloat16, device="cuda") * 0.01
        l2_w_bf16 = torch.randn(num_experts_per_rank, hidden, intermediate_hidden,
                                dtype=torch.bfloat16, device="cuda") * 0.01

        def _quantize_weight_block(w):
            E, N, K = w.shape
            n_blocks = N // WEIGHT_SF_GRAN_MN
            k_blocks = K // WEIGHT_SF_GRAN_K
            w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
            w_sf = torch.empty(E, n_blocks, k_blocks, dtype=torch.float32, device=w.device)
            for i in range(E):
                wi = w[i].float().view(n_blocks, WEIGHT_SF_GRAN_MN, k_blocks, WEIGHT_SF_GRAN_K)
                amax = wi.abs().amax(dim=(1, 3))
                sf = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
                wi_scaled = wi / sf[:, None, :, None]
                w_fp8[i] = wi_scaled.reshape(N, K).to(torch.float8_e4m3fn)
                w_sf[i] = sf
            return w_fp8, w_sf

        l1_w_fp8, l1_w_sf = _quantize_weight_block(l1_w_bf16)
        l2_w_fp8, l2_w_sf = _quantize_weight_block(l2_w_bf16)

        dy = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.01

        # ─── FP32 Reference ─────────────────────────────────────────────
        dx_ref, dW1_ref, dW2_ref = reference_mega_moe_backward(
            dy, x_fp8, x_sf, topk_idx, topk_weights,
            l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
            rank_idx, num_ranks, group,
            num_experts, num_topk, hidden, intermediate_hidden,
            l1_w_bf16=l1_w_bf16, l2_w_bf16=l2_w_bf16,
        )

        # ─── Fused Kernel ───────────────────────────────────────────────
        sym_buffer = SymmBuffer(
            group, num_experts,
            num_max_aligned, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch=True,
            activation='swiglu',
            with_backward=True,
        )

        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_sf)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        # Run forward to populate l1_acts, l2_acts
        y_fwd = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        l1_transformed, l2_transformed = deep_gemm.transform_weights_for_mega_moe_sm90(
            (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))

        deep_gemm.fp8_mega_moe(
            y_fwd, l1_transformed, l2_transformed, sym_buffer,
            cumulative_local_expert_recv_stats=torch.zeros(
                (num_experts_per_rank,), dtype=torch.int, device="cuda"),
            recipe=(128, 128, 128), activation="swiglu", fast_math=True,
        )
        torch.cuda.synchronize()

        # Run backward
        dx_fused = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        dW1_fused = torch.zeros(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                                dtype=torch.float32, device="cuda")
        dW2_fused = torch.zeros(num_experts_per_rank, hidden, intermediate_hidden,
                                dtype=torch.float32, device="cuda")

        fp8_mega_moe_backward(
            dx_fused, dW1_fused, dW2_fused, dy,
            l1_w_bf16, l2_w_bf16, (l1_w_fp8, l1_w_sf),
            sym_buffer,
            cumulative_local_expert_recv_stats=torch.zeros(
                (num_experts_per_rank + 1,), dtype=torch.int, device="cuda"),
            recompute=True, activation='swiglu',
        )
        torch.cuda.synchronize()

        # ─── Validate ───────────────────────────────────────────────────
        diff_dx = calc_diff(dx_fused.float(), dx_ref.float())
        diff_dW1 = calc_diff(dW1_fused, dW1_ref)
        diff_dW2 = calc_diff(dW2_fused, dW2_ref)

        dx_has_nan = bool(torch.isnan(dx_fused).any().item())
        dW1_has_nan = bool(torch.isnan(dW1_fused).any().item())
        dW2_has_nan = bool(torch.isnan(dW2_fused).any().item())

        # Check d_o_pool (dispatch correctness)
        do_nan = bool(torch.isnan(sym_buffer.d_o_pool.float()).any().item())

        # Pass criteria
        passed = (
            not dx_has_nan and not dW1_has_nan and not dW2_has_nan and
            not do_nan and
            diff_dx < VALIDATE_THRESHOLDS["dx"][thr_key] and
            diff_dW1 < VALIDATE_THRESHOLDS["dW1"][thr_key] and
            diff_dW2 < VALIDATE_THRESHOLDS["dW2"][thr_key]
        )

        status = "PASS" if passed else "FAIL"
        dist_print(
            f"  {name:<18} {diff_dx:>8.5f} {diff_dW1:>8.5f} {diff_dW2:>8.5f} "
            f"{int(dx_has_nan):>7} {int(dW1_has_nan):>8} {int(dW2_has_nan):>8} "
            f"[{status}]",
            rank_idx == 0)

        if not passed and rank_idx == 0:
            thr_dx = VALIDATE_THRESHOLDS["dx"][thr_key]
            thr_dW1 = VALIDATE_THRESHOLDS["dW1"][thr_key]
            thr_dW2 = VALIDATE_THRESHOLDS["dW2"][thr_key]
            details = []
            if dx_has_nan:
                details.append("dx has NaN")
            if dW1_has_nan:
                details.append("dW1 has NaN")
            if dW2_has_nan:
                details.append("dW2 has NaN")
            if do_nan:
                details.append("d_o_pool has NaN (dispatch bug)")
            if diff_dx >= thr_dx:
                details.append(f"dx diff {diff_dx:.5f} >= thr {thr_dx:.3f}")
            if diff_dW1 >= thr_dW1:
                details.append(f"dW1 diff {diff_dW1:.5f} >= thr {thr_dW1:.3f}")
            if diff_dW2 >= thr_dW2:
                details.append(f"dW2 diff {diff_dW2:.5f} >= thr {thr_dW2:.3f}")
            print(f"    FAIL reasons: {'; '.join(details)}")

        all_passed = all_passed and passed

        sym_buffer.destroy()
        del dx_fused, dW1_fused, dW2_fused, dx_ref, dW1_ref, dW2_ref
        del l1_w_bf16, l2_w_bf16, l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf
        del l1_transformed, l2_transformed
        del x_bf16, x_fp8, x_sf, dy, y_fwd, topk_idx, topk_weights
        torch.cuda.empty_cache()

    dist_print("-" * 80, rank_idx == 0)
    dist_print(f"  Validation {'PASSED' if all_passed else 'FAILED'} "
               f"(thresholds: small={VALIDATE_THRESHOLDS['dx']['small']}, "
               f"large={VALIDATE_THRESHOLDS['dx']['large']})",
               rank_idx == 0)
    return all_passed


# ============================================================================
# Section 5: Test scenarios and main driver
# ============================================================================

SMOKE_SCENARIOS = [
    {
        "name": "smoke_small",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 512,
        "intermediate_hidden": 512,
        "num_experts": 8,
        "num_topk": 2,
    },
    {
        "name": "smoke_medium",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 8,
        "num_topk": 2,
    },
    {
        "name": "smoke_dsv3_decode",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 2048,
        "intermediate_hidden": 1024,
        "num_experts": 16,
        "num_topk": 4,
    },
    {
        "name": "smoke_large_experts",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 64,
        "num_topk": 4,
    },
    {
        "name": "smoke_dsv3_full",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 7168,
        "intermediate_hidden": 2048,
        "num_experts": 128,
        "num_topk": 8,
    },
]

# Multi-GPU scenarios — require multiple ranks to fit in memory
MULTI_GPU_SCENARIOS = [
    {
        "name": "mgpu_dsv3_full",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 7168,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 8,
    },
    {
        "name": "mgpu_large_experts",
        "num_max_tokens_per_rank": 768,
        "num_tokens": 768,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 128,
        "num_topk": 4,
    },
    {
        "name": "mgpu_wide_hidden",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 7168,
        "intermediate_hidden": 4096,
        "num_experts": 128,
        "num_topk": 8,
    },
    {
        "name": "mgpu_high_tokens",
        "num_max_tokens_per_rank": 1536,
        "num_tokens": 1536,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 8,
    },
]

SHAPE_SCENARIOS = [
    {
        "name": f"shape_T{t}_H{h}_IH{ih}_E{e}_K{k}",
        "num_max_tokens_per_rank": t,
        "num_tokens": t,
        "hidden": h,
        "intermediate_hidden": ih,
        "num_experts": e,
        "num_topk": k,
    }
    for t, h, ih, e, k in [
        (384, 512, 512, 8, 2),
        (384, 1024, 512, 8, 2),
        (384, 1024, 1024, 16, 4),
        (384, 2048, 1024, 16, 4),
        (384, 4096, 2048, 32, 4),
        (384, 4096, 2048, 64, 8),
        (768, 4096, 2048, 32, 4),
        (768, 7168, 2048, 64, 8),
        (384, 7168, 4096, 32, 8),
        (1536, 4096, 2048, 64, 4),
    ]
]

SPEEDUP_SCENARIOS = [
    {
        "name": "perf_dsv3_decode",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 7168,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 8,
    },
    {
        "name": "perf_large",
        "num_max_tokens_per_rank": 768,
        "num_tokens": 768,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 128,
        "num_topk": 4,
    },
    {
        "name": "perf_medium",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 2048,
        "intermediate_hidden": 1024,
        "num_experts": 16,
        "num_topk": 4,
    },
    {
        "name": "perf_wide_ih",
        "num_max_tokens_per_rank": 384,
        "num_tokens": 384,
        "hidden": 7168,
        "intermediate_hidden": 4096,
        "num_experts": 256,
        "num_topk": 8,
    },
    {
        "name": "perf_high_tokens",
        "num_max_tokens_per_rank": 1536,
        "num_tokens": 1536,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 128,
        "num_topk": 4,
    },
]


def _run_speedup_benchmark(
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
):
    """Benchmark fused backward kernel vs PyTorch baseline and report speedup."""
    import deep_gemm
    from deep_gemm.mega import SymmBuffer, fp8_mega_moe_backward
    from deep_gemm.utils.math import align
    import time

    num_warmup = 3
    num_iters = 10

    dist_print(f"\n{'scenario':<22} {'fused(ms)':>10} {'baseline(ms)':>12} {'speedup':>8}",
               rank_idx == 0)
    dist_print("-" * 56, rank_idx == 0)

    for cfg in SPEEDUP_SCENARIOS:
        if cfg["num_experts"] % num_ranks != 0:
            continue

        name = cfg["name"]
        num_max = cfg["num_max_tokens_per_rank"]
        num_tokens = cfg.get("num_tokens", num_max)
        hidden = cfg["hidden"]
        intermediate_hidden = cfg["intermediate_hidden"]
        num_experts = cfg["num_experts"]
        num_topk = cfg["num_topk"]
        num_experts_per_rank = num_experts // num_ranks

        torch.manual_seed(rank_idx * 1000 + 42)

        alignment = deep_gemm._C.get_token_alignment_for_sm90_mega_moe()
        num_max_aligned = align(num_max, alignment)

        # Generate random inputs
        x_bf16 = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        x_fp8, x_sf = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128,
                                             use_packed_ue8m0=False)
        topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                                 dtype=torch.int64, device="cuda")
        topk_weights = torch.rand(num_tokens, num_topk, dtype=torch.float32,
                                  device="cuda") * 0.5 + 0.5
        l1_w_bf16 = torch.randn(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                                dtype=torch.bfloat16, device="cuda") * 0.01
        l2_w_bf16 = torch.randn(num_experts_per_rank, hidden, intermediate_hidden,
                                dtype=torch.bfloat16, device="cuda") * 0.01

        # Quantize weights to FP8
        def _quantize_weight_block(w):
            E, N, K = w.shape
            n_blocks = N // WEIGHT_SF_GRAN_MN
            k_blocks = K // WEIGHT_SF_GRAN_K
            w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
            w_sf = torch.empty(E, n_blocks, k_blocks, dtype=torch.float32, device=w.device)
            for i in range(E):
                wi = w[i].float().view(n_blocks, WEIGHT_SF_GRAN_MN, k_blocks, WEIGHT_SF_GRAN_K)
                amax = wi.abs().amax(dim=(1, 3))
                sf = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
                wi_scaled = wi / sf[:, None, :, None]
                w_fp8[i] = wi_scaled.reshape(N, K).to(torch.float8_e4m3fn)
                w_sf[i] = sf
            return w_fp8, w_sf

        l1_w_fp8, l1_w_sf = _quantize_weight_block(l1_w_bf16)
        l2_w_fp8, l2_w_sf = _quantize_weight_block(l2_w_bf16)
        dy = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.01

        # ─── Setup fused kernel ────────────────────────────────────────
        sym_buffer = SymmBuffer(
            group, num_experts,
            num_max_aligned, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch=True,
            activation='swiglu',
            with_backward=True,
        )

        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_sf)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        # Run forward to populate l1_acts/l2_acts
        y_fwd = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        l1_transformed, l2_transformed = deep_gemm.transform_weights_for_mega_moe_sm90(
            (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))
        cum_stats_fwd = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")
        deep_gemm.fp8_mega_moe(
            y_fwd, l1_transformed, l2_transformed, sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats_fwd,
            recipe=(128, 128, 128), activation="swiglu", fast_math=True,
        )
        torch.cuda.synchronize()

        cum_stats_bwd = torch.zeros((num_experts_per_rank + 1,), dtype=torch.int, device="cuda")

        def run_fused():
            dx_f = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
            dW1_f = torch.zeros(num_experts_per_rank, 2 * intermediate_hidden, hidden,
                                dtype=torch.float32, device="cuda")
            dW2_f = torch.zeros(num_experts_per_rank, hidden, intermediate_hidden,
                                dtype=torch.float32, device="cuda")
            fp8_mega_moe_backward(
                dx_f, dW1_f, dW2_f, dy,
                l1_w_bf16, l2_w_bf16, (l1_w_fp8, l1_w_sf),
                sym_buffer,
                cumulative_local_expert_recv_stats=cum_stats_bwd,
                recompute=True, activation='swiglu',
            )

        # ─── Setup baseline (DeepEP dispatch + per-expert BF16 GEMMs + combine) ──
        # This mirrors the real non-fused EP backward pipeline:
        #   1. Dispatch dy to expert-owning ranks (combine backward)
        #   2. Per-expert: recompute forward + backward GEMMs in BF16
        #   3. Combine dx back to source ranks (dispatch backward)
        deep_ep = _import_deep_ep()
        ep_buffer = None
        if deep_ep is not None:
            ep_buffer = _make_deep_ep_buffer_for_backward(
                deep_ep, group, num_max, hidden, num_topk)

        if ep_buffer is not None:
            # DeepEP-based baseline with real NVLink communication.
            # Pre-dispatch x outside the timing loop (in real training, x was
            # dispatched during forward and is available locally for recomputation).
            recv_x_pre, _, handle_x_pre, _ = ep_buffer.dispatch(
                x_bf16,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=num_experts,
                expert_alignment=alignment,
            )
            recv_x_local = recv_x_pre if isinstance(recv_x_pre, torch.Tensor) else recv_x_pre[0]
            n_recv = recv_x_local.shape[0]
            # Release the handle (combine with dummy to free internal state)
            ep_buffer.combine(torch.zeros(n_recv, hidden, dtype=torch.bfloat16, device="cuda"),
                              handle=handle_x_pre)

            def run_baseline():
                # Phase 1: Dispatch dy to expert-owning ranks (combine backward)
                recv_dy, _, handle, _ = ep_buffer.dispatch(
                    dy,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_experts=num_experts,
                    expert_alignment=alignment,
                )
                n = recv_dy.shape[0] if isinstance(recv_dy, torch.Tensor) else recv_dy[0].shape[0]
                recv_dy_t = recv_dy if isinstance(recv_dy, torch.Tensor) else recv_dy[0]

                # Phase 2: Per-expert backward GEMMs
                psum = handle.psum_num_recv_tokens_per_expert
                dx_pool = torch.zeros(n, hidden, dtype=torch.bfloat16, device="cuda")

                for e_local in range(num_experts_per_rank):
                    start = 0 if e_local == 0 else psum[e_local - 1].item()
                    end = psum[e_local].item()
                    count = end - start
                    if count == 0:
                        continue

                    x_e = recv_x_local[start:end]      # [count, H] BF16 (pre-dispatched)
                    dy_e = recv_dy_t[start:end]        # [count, H] BF16
                    w1 = l1_w_bf16[e_local]            # [2*IH, H] BF16
                    w2 = l2_w_bf16[e_local]            # [H, IH] BF16

                    # Recompute forward
                    h = x_e @ w1.t()                   # [count, 2*IH]
                    gate = h[:, :intermediate_hidden]
                    up = h[:, intermediate_hidden:]
                    a = torch.nn.functional.silu(gate.float()).to(torch.bfloat16) * up

                    # L2 backward: d_a = dy @ W2
                    d_a = dy_e @ w2                    # [count, IH]

                    # Weight gradients (computed for fair timing, not stored)
                    dy_e.float().t() @ a.float()            # dW2
                    # SwiGLU backward
                    gate_f = gate.float()
                    sig = torch.sigmoid(gate_f)
                    silu_gate = (gate_f * sig).to(torch.bfloat16)
                    d_up = d_a * silu_gate
                    d_gate = (d_a.float() * up.float() * sig * (1.0 + gate_f * (1.0 - sig))).to(torch.bfloat16)
                    d_h = torch.cat([d_gate, d_up], dim=1)

                    d_h.float().t() @ x_e.float()          # dW1

                    # L1 backward: dx = d_h @ W1
                    dx_pool[start:end] = d_h @ w1

                # Phase 3: Combine dx back to source ranks (dispatch backward)
                ep_buffer.combine(dx_pool, handle=handle)

        else:
            # Fallback: all-gather baseline (no DeepEP available)
            dist_print("  [WARN] deep_ep not available, baseline uses all-gather "
                       "(missing dispatch/combine communication cost)", rank_idx == 0)
            x_bf16_g = uneven_all_gather(x_bf16, group=group)
            topk_idx_g = uneven_all_gather(topk_idx.view(-1), group=group).view(-1, num_topk)
            num_tokens_g = topk_idx_g.shape[0]

            def run_baseline():
                dx_base = torch.zeros(num_tokens_g, hidden, dtype=torch.bfloat16, device="cuda")
                dy_g = uneven_all_gather(dy, group=group)

                for e_local in range(num_experts_per_rank):
                    e_global = rank_idx * num_experts_per_rank + e_local
                    mask = (topk_idx_g == e_global).any(dim=1)
                    indices = mask.nonzero(as_tuple=True)[0]
                    if indices.numel() == 0:
                        continue

                    x_e = x_bf16_g[indices]
                    w1 = l1_w_bf16[e_local]
                    w2 = l2_w_bf16[e_local]

                    h = x_e @ w1.t()
                    gate, up = h[:, :intermediate_hidden], h[:, intermediate_hidden:]
                    a = torch.nn.functional.silu(gate.float()).to(torch.bfloat16) * up

                    dy_e = dy_g[indices]
                    d_a = dy_e @ w2

                    gate_f = gate.float()
                    sig = torch.sigmoid(gate_f)
                    silu_gate = (gate_f * sig).to(torch.bfloat16)
                    d_up = d_a * silu_gate
                    d_gate = (d_a.float() * up.float() * sig * (1.0 + gate_f * (1.0 - sig))).to(torch.bfloat16)
                    d_h = torch.cat([d_gate, d_up], dim=1)

                    # Weight gradients (included for fair timing)
                    _ = dy_e.float().t() @ a.float()   # dW2
                    _ = d_h.float().t() @ x_e.float()  # dW1

                    dx_base[indices] += d_h @ w1

        # ─── Benchmark fused ───────────────────────────────────────────
        # Warmup
        for _ in range(num_warmup):
            run_fused()
        torch.cuda.synchronize()
        dist.barrier(group)

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            run_fused()
            torch.cuda.synchronize()
        dist.barrier(group)
        t_fused_ms = (time.perf_counter() - t0) / num_iters * 1000

        # ─── Benchmark baseline ────────────────────────────────────────
        # Warmup
        for _ in range(num_warmup):
            run_baseline()
        torch.cuda.synchronize()
        dist.barrier(group)

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            run_baseline()
            torch.cuda.synchronize()
        dist.barrier(group)
        t_baseline_ms = (time.perf_counter() - t0) / num_iters * 1000

        speedup = t_baseline_ms / t_fused_ms if t_fused_ms > 0 else float('inf')
        dist_print(
            f"{name:<22} {t_fused_ms:>9.2f}  {t_baseline_ms:>11.2f}  {speedup:>7.2f}x",
            rank_idx == 0)

        # Cleanup
        sym_buffer.destroy()
        if ep_buffer is not None:
            ep_buffer.destroy()
        torch.cuda.empty_cache()

    dist_print("", rank_idx == 0)


def test(args: argparse.Namespace):
    """Main test entry point (torchrun launch)."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    torch.set_default_device('cuda')
    dist.init_process_group(backend='nccl')

    # Set module-level _local_rank for dist_print
    import deep_gemm.utils.dist as _dist_mod
    _dist_mod._local_rank = local_rank

    rank_idx = global_rank
    num_ranks = world_size
    group = dist.new_group(list(range(world_size)))
    dist_print(f"SM90 MegaMoE Backward Test — {num_ranks} ranks", rank_idx == 0)

    all_passed = True

    if args.accuracy:
        dist_print("=" * 60, rank_idx == 0)
        dist_print("Running accuracy tests (Phase 1: reference self-check)", rank_idx == 0)
        dist_print("=" * 60, rank_idx == 0)

        dist_print("\n--- Smoke tests (reference) ---", rank_idx == 0)
        for scenario in SMOKE_SCENARIOS:
            if scenario["num_experts"] % num_ranks != 0:
                continue
            passed = _run_backward_scenario(
                scenario["name"], scenario,
                rank_idx, num_ranks, group, args.diff_tol)
            all_passed = all_passed and passed
            if not passed and args.fail_fast:
                break

        if all_passed and not args.fail_fast:
            dist_print("\n--- Shape sweep tests (reference) ---", rank_idx == 0)
            for scenario in SHAPE_SCENARIOS:
                if scenario["num_experts"] % num_ranks != 0:
                    continue
                passed = _run_backward_scenario(
                    scenario["name"], scenario,
                    rank_idx, num_ranks, group, args.diff_tol)
                all_passed = all_passed and passed
                if not passed and args.fail_fast:
                    break

    if args.fused:
        dist_print("=" * 60, rank_idx == 0)
        dist_print("Running fused kernel tests (Phase 2: kernel vs FP32 reference)",
                   rank_idx == 0)
        dist_print("=" * 60, rank_idx == 0)

        dist_print("\n--- Smoke tests (fused kernel) ---", rank_idx == 0)
        for scenario in SMOKE_SCENARIOS:
            if scenario["num_experts"] % num_ranks != 0:
                continue
            try:
                passed, _ = _run_fused_backward_scenario(
                    scenario["name"], scenario,
                    rank_idx, num_ranks, group, args.diff_tol)
                all_passed = all_passed and passed
                if not passed and args.fail_fast:
                    break
            except Exception as e:
                dist_print(f"  [ERROR] {scenario['name']}: {e}", rank_idx == 0)
                all_passed = False
                if args.fail_fast:
                    break
            finally:
                torch.cuda.empty_cache()

        if all_passed and not args.fail_fast:
            dist_print("\n--- Shape sweep tests (fused kernel) ---", rank_idx == 0)
            for scenario in SHAPE_SCENARIOS:
                if scenario["num_experts"] % num_ranks != 0:
                    continue
                try:
                    passed, _ = _run_fused_backward_scenario(
                        scenario["name"], scenario,
                        rank_idx, num_ranks, group, args.diff_tol)
                    all_passed = all_passed and passed
                    if not passed and args.fail_fast:
                        break
                except Exception as e:
                    dist_print(f"  [ERROR] {scenario['name']}: {e}", rank_idx == 0)
                    all_passed = False
                    if args.fail_fast:
                        break
                finally:
                    torch.cuda.empty_cache()

        if all_passed and not args.fail_fast and num_ranks > 1:
            dist_print("\n--- Multi-GPU tests (fused kernel) ---", rank_idx == 0)
            for scenario in MULTI_GPU_SCENARIOS:
                if scenario["num_experts"] % num_ranks != 0:
                    continue
                try:
                    passed, _ = _run_fused_backward_scenario(
                        scenario["name"], scenario,
                        rank_idx, num_ranks, group, args.diff_tol)
                    all_passed = all_passed and passed
                    if not passed and args.fail_fast:
                        break
                except Exception as e:
                    dist_print(f"  [ERROR] {scenario['name']}: {e}", rank_idx == 0)
                    all_passed = False
                    if args.fail_fast:
                        break
                finally:
                    torch.cuda.empty_cache()

    if args.search_tol:
        dist_print("=" * 60, rank_idx == 0)
        dist_print("Running tolerance search (Phase 3: empirical diff measurement)",
                   rank_idx == 0)
        dist_print("=" * 60, rank_idx == 0)
        _search_diff_tolerance(rank_idx, num_ranks, group)

    if args.speedup:
        dist_print("=" * 60, rank_idx == 0)
        dist_print("Running speedup benchmark (Phase 4: fused vs baseline timing)",
                   rank_idx == 0)
        dist_print("=" * 60, rank_idx == 0)
        _run_speedup_benchmark(rank_idx, num_ranks, group)

    if args.validate:
        dist_print("=" * 60, rank_idx == 0)
        dist_print("Running fused-vs-FP32 correctness validation (Phase 5)",
                   rank_idx == 0)
        dist_print("=" * 60, rank_idx == 0)
        val_passed = _run_fused_backward_validation(rank_idx, num_ranks, group)
        all_passed = all_passed and val_passed

    if args.accuracy or args.fused or args.validate:
        dist_print("=" * 60, rank_idx == 0)
        dist_print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}",
                   rank_idx == 0)
        dist_print("=" * 60, rank_idx == 0)

    if not (args.accuracy or args.fused or args.search_tol or args.speedup
            or args.validate):
        dist_print("Use --accuracy, --fused, --search-tol, --speedup, or --validate "
                   "to run tests.", rank_idx == 0)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SM90 MegaMoE backward test")
    parser.add_argument("--accuracy", action="store_true",
                        help="Run reference self-consistency tests (Phase 1)")
    parser.add_argument("--fused", action="store_true",
                        help="Run fused kernel vs reference tests (Phase 2)")
    parser.add_argument("--search-tol", action="store_true",
                        help="Run tolerance search sweep (Phase 3)")
    parser.add_argument("--speedup", action="store_true",
                        help="Run speedup benchmark: fused kernel vs PyTorch baseline (Phase 4)")
    parser.add_argument("--validate", action="store_true",
                        help="Run fused-vs-FP32 correctness validation with per-output "
                             "thresholds (Phase 5)")
    parser.add_argument("--diff-tol", type=float, default=0.10,
                        help="calc_diff tolerance; default: 0.10 "
                             "(backward has more numerical error than forward 0.07)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first failure")
    args = parser.parse_args()

    # torchrun sets LOCAL_RANK, LOCAL_WORLD_SIZE, WORLD_SIZE, RANK
    test(args)
