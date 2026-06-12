# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused Triton kernel for CSA Compressor post-GEMM operations.

Fuses: cutoff+reshape, add APE, overlap_transform, softmax+weighted_sum,
       discard_halo, RMSNorm.

Input:  kv, score from GEMM outputs [sq_with_halo, b, coff * head_dim]
Output: normalized compressed KV [n_out, b, head_dim] ready for RoPE

Uses torch.autograd.Function: forward runs Triton kernel, backward uses
PyTorch ops for gradient computation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_compress_no_overlap_kernel(
    kv_ptr,
    score_ptr,
    ape_ptr,
    norm_weight_ptr,
    out_ptr,
    ratio: tl.constexpr,
    head_dim: tl.constexpr,
    n_groups,
    has_halo: tl.constexpr,
    eps,
    stride_s,
    stride_b,
    out_stride_s,
    out_stride_b,
    BLOCK_D: tl.constexpr,
    RATIO: tl.constexpr,
):
    """Fused compressor kernel for non-overlap mode (ratio=128).

    Grid: (n_out, batch_size)
    Each program handles one output compressed position for one batch element.
    Processes BLOCK_D elements of head_dim at a time.
    """
    pid_out = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Map output index to group index (skip halo group)
    group_idx = pid_out + (1 if has_halo else 0)

    # Offsets for head_dim
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    # Accumulator for weighted sum
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # --- Softmax + weighted sum over ratio positions ---
    # For each position in the group, compute score + APE, then softmax
    # softmax is per-element along ratio dim

    # First pass: find max score per element
    max_scores = tl.full([BLOCK_D], value=float("-inf"), dtype=tl.float32)

    for r in range(RATIO):
        src_pos = group_idx * ratio + r
        # Load score[src_pos, b, d] + ape[r, d]
        score_offset = src_pos * stride_s + pid_b * stride_b + d_offs
        ape_offset = r * head_dim + d_offs
        s = tl.load(score_ptr + score_offset, mask=d_mask, other=0.0).to(tl.float32)
        a = tl.load(ape_ptr + ape_offset, mask=d_mask, other=0.0).to(tl.float32)
        s = s + a
        max_scores = tl.maximum(max_scores, s)

    # Second pass: compute exp sum
    exp_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r in range(RATIO):
        src_pos = group_idx * ratio + r
        score_offset = src_pos * stride_s + pid_b * stride_b + d_offs
        ape_offset = r * head_dim + d_offs
        s = tl.load(score_ptr + score_offset, mask=d_mask, other=0.0).to(tl.float32)
        a = tl.load(ape_ptr + ape_offset, mask=d_mask, other=0.0).to(tl.float32)
        s = s + a
        exp_sum += tl.exp(s - max_scores)

    # Third pass: weighted sum
    for r in range(RATIO):
        src_pos = group_idx * ratio + r
        score_offset = src_pos * stride_s + pid_b * stride_b + d_offs
        ape_offset = r * head_dim + d_offs
        s = tl.load(score_ptr + score_offset, mask=d_mask, other=0.0).to(tl.float32)
        a = tl.load(ape_ptr + ape_offset, mask=d_mask, other=0.0).to(tl.float32)
        s = s + a
        weight = tl.exp(s - max_scores) / exp_sum

        kv_offset = src_pos * stride_s + pid_b * stride_b + d_offs
        kv_val = tl.load(kv_ptr + kv_offset, mask=d_mask, other=0.0).to(tl.float32)
        acc += weight * kv_val

    # --- RMSNorm ---
    var = tl.sum(acc * acc) / head_dim
    rms = tl.rsqrt(var + eps)
    norm_w = tl.load(norm_weight_ptr + d_offs, mask=d_mask, other=1.0).to(tl.float32)
    out_val = acc * rms * norm_w

    # Store
    out_offset = pid_out * out_stride_s + pid_b * out_stride_b + d_offs
    tl.store(out_ptr + out_offset, out_val.to(tl.bfloat16), mask=d_mask)


@triton.jit
def _fused_compress_overlap_kernel(
    kv_ptr,
    score_ptr,
    ape_ptr,
    norm_weight_ptr,
    out_ptr,
    ratio: tl.constexpr,
    head_dim: tl.constexpr,
    n_groups,
    has_halo: tl.constexpr,
    eps,
    stride_s,
    stride_b,
    out_stride_s,
    out_stride_b,
    BLOCK_D: tl.constexpr,
    RATIO: tl.constexpr,
):
    """Fused compressor kernel for overlap mode (ratio=4).

    In overlap mode, coff=2, so kv/score have shape [sq, b, 2*head_dim].
    The overlap transform creates a 2*ratio window:
      - positions [0..ratio-1]: first half (dim_offset=0) of previous group
      - positions [ratio..2*ratio-1]: second half (dim_offset=head_dim) of current group

    APE mapping:
      - position r < ratio: ape[r, :head_dim] (first half)
      - position r >= ratio: ape[r-ratio, head_dim:] (second half)

    Grid: (n_out, batch_size)
    """
    pid_out = tl.program_id(0)
    pid_b = tl.program_id(1)

    group_idx = pid_out + (1 if has_halo else 0)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_scores = tl.full([BLOCK_D], value=float("-inf"), dtype=tl.float32)

    start_pos = group_idx * ratio
    coff_head_dim = 2 * head_dim  # coff=2 in overlap mode

    # First pass: find max score
    for r in range(2 * RATIO):
        if r < ratio:
            src_pos = (group_idx - 1) * ratio + r
            dim_offset = 0
            ape_offset = r * coff_head_dim + d_offs
            valid = group_idx > 0
        else:
            src_pos = start_pos + (r - ratio)
            dim_offset = head_dim
            ape_offset = (r - ratio) * coff_head_dim + head_dim + d_offs
            valid = True

        if valid:
            score_offset = src_pos * stride_s + pid_b * stride_b + dim_offset + d_offs
            s = tl.load(score_ptr + score_offset, mask=d_mask, other=0.0).to(tl.float32)
            a = tl.load(ape_ptr + ape_offset, mask=d_mask, other=0.0).to(tl.float32)
            s = s + a
            max_scores = tl.maximum(max_scores, s)

    # Second pass: exp sum
    exp_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r in range(2 * RATIO):
        if r < ratio:
            src_pos = (group_idx - 1) * ratio + r
            dim_offset = 0
            ape_offset = r * coff_head_dim + d_offs
            valid = group_idx > 0
        else:
            src_pos = start_pos + (r - ratio)
            dim_offset = head_dim
            ape_offset = (r - ratio) * coff_head_dim + head_dim + d_offs
            valid = True

        if valid:
            score_offset = src_pos * stride_s + pid_b * stride_b + dim_offset + d_offs
            s = tl.load(score_ptr + score_offset, mask=d_mask, other=0.0).to(tl.float32)
            a = tl.load(ape_ptr + ape_offset, mask=d_mask, other=0.0).to(tl.float32)
            s = s + a
            exp_sum += tl.exp(s - max_scores)

    # Third pass: weighted sum
    for r in range(2 * RATIO):
        if r < ratio:
            src_pos = (group_idx - 1) * ratio + r
            dim_offset = 0
            ape_offset = r * coff_head_dim + d_offs
            valid = group_idx > 0
        else:
            src_pos = start_pos + (r - ratio)
            dim_offset = head_dim
            ape_offset = (r - ratio) * coff_head_dim + head_dim + d_offs
            valid = True

        if valid:
            score_offset = src_pos * stride_s + pid_b * stride_b + dim_offset + d_offs
            s = tl.load(score_ptr + score_offset, mask=d_mask, other=0.0).to(tl.float32)
            a = tl.load(ape_ptr + ape_offset, mask=d_mask, other=0.0).to(tl.float32)
            s = s + a
            weight = tl.exp(s - max_scores) / exp_sum

            kv_offset = src_pos * stride_s + pid_b * stride_b + dim_offset + d_offs
            kv_val = tl.load(kv_ptr + kv_offset, mask=d_mask, other=0.0).to(tl.float32)
            acc += weight * kv_val

    # --- RMSNorm ---
    var = tl.sum(acc * acc) / head_dim
    rms = tl.rsqrt(var + eps)
    norm_w = tl.load(norm_weight_ptr + d_offs, mask=d_mask, other=1.0).to(tl.float32)
    out_val = acc * rms * norm_w

    out_offset = pid_out * out_stride_s + pid_b * out_stride_b + d_offs
    tl.store(out_ptr + out_offset, out_val.to(tl.bfloat16), mask=d_mask)


def _triton_forward(kv, score, ape, norm_weight, ratio, head_dim, has_halo, use_overlap, eps):
    """Run the Triton kernel forward pass (no autograd)."""
    sq_with_halo, b, _ = kv.shape
    cutoff = (sq_with_halo // ratio) * ratio
    n_groups = cutoff // ratio
    n_out = n_groups - (1 if has_halo else 0)

    out = torch.empty(n_out, b, head_dim, dtype=torch.bfloat16, device=kv.device)

    stride_s = kv.stride(0)
    stride_b = kv.stride(1)
    out_stride_s = out.stride(0)
    out_stride_b = out.stride(1)

    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (n_out, b)

    if use_overlap:
        _fused_compress_overlap_kernel[grid](
            kv, score, ape, norm_weight, out,
            ratio=ratio,
            head_dim=head_dim,
            n_groups=n_groups,
            has_halo=has_halo,
            eps=eps,
            stride_s=stride_s,
            stride_b=stride_b,
            out_stride_s=out_stride_s,
            out_stride_b=out_stride_b,
            BLOCK_D=BLOCK_D,
            RATIO=ratio,
        )
    else:
        _fused_compress_no_overlap_kernel[grid](
            kv, score, ape, norm_weight, out,
            ratio=ratio,
            head_dim=head_dim,
            n_groups=n_groups,
            has_halo=has_halo,
            eps=eps,
            stride_s=stride_s,
            stride_b=stride_b,
            out_stride_s=out_stride_s,
            out_stride_b=out_stride_b,
            BLOCK_D=BLOCK_D,
            RATIO=ratio,
        )

    return out


def _pytorch_forward(kv, score, ape, norm_weight, ratio, head_dim, has_halo, use_overlap, eps):
    """PyTorch reference forward (differentiable, used for backward)."""
    sq_with_halo, b, _ = kv.shape
    cutoff = (sq_with_halo // ratio) * ratio
    kv_cut = kv[:cutoff]
    score_cut = score[:cutoff]
    n_compressed = cutoff // ratio

    coff = 2 if use_overlap else 1

    kv_r = kv_cut.view(n_compressed, ratio, b, -1)
    score_r = score_cut.view(n_compressed, ratio, b, -1)

    score_r = score_r + ape.view(1, ratio, 1, -1)

    if use_overlap:
        d = head_dim
        n_groups = n_compressed

        new_kv = kv_r.new_full((n_groups, 2 * ratio, b, d), 0)
        new_kv[:, ratio:] = kv_r[:, :, :, d:]
        new_kv[1:, :ratio] = kv_r[:-1, :, :, :d]
        kv_r = new_kv

        new_score = score_r.new_full((n_groups, 2 * ratio, b, d), float("-inf"))
        new_score[:, ratio:] = score_r[:, :, :, d:]
        new_score[1:, :ratio] = score_r[:-1, :, :, :d]
        score_r = new_score

    kv_r = (kv_r * torch.softmax(score_r.float(), dim=1)).sum(dim=1)

    if has_halo:
        kv_r = kv_r[1:]

    # RMSNorm
    kv_f = kv_r.float()
    var = (kv_f * kv_f).mean(dim=-1, keepdim=True)
    kv_f = kv_f * torch.rsqrt(var + eps)
    kv_f = kv_f * norm_weight.float()
    return kv_f.to(torch.bfloat16)


class FusedCompressorPostGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, score, ape, norm_weight, ratio, head_dim, has_halo, use_overlap, eps):
        ctx.save_for_backward(kv, score, ape, norm_weight)
        ctx.ratio = ratio
        ctx.head_dim = head_dim
        ctx.has_halo = has_halo
        ctx.use_overlap = use_overlap
        ctx.eps = eps
        return _triton_forward(kv, score, ape, norm_weight, ratio, head_dim, has_halo, use_overlap, eps)

    @staticmethod
    def backward(ctx, grad_output):
        kv, score, ape, norm_weight = ctx.saved_tensors
        ratio = ctx.ratio
        head_dim = ctx.head_dim
        has_halo = ctx.has_halo
        use_overlap = ctx.use_overlap
        eps = ctx.eps

        # Re-run PyTorch forward with autograd to get gradients
        kv_d = kv.detach().requires_grad_(True)
        score_d = score.detach().requires_grad_(True)
        ape_d = ape.detach().requires_grad_(True)
        norm_weight_d = norm_weight.detach().requires_grad_(True)

        with torch.enable_grad():
            out = _pytorch_forward(kv_d, score_d, ape_d, norm_weight_d, ratio, head_dim, has_halo, use_overlap, eps)
            out.backward(grad_output)

        return kv_d.grad, score_d.grad, ape_d.grad, norm_weight_d.grad, None, None, None, None, None


def fused_compressor_post_gemm(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    ratio: int,
    head_dim: int,
    has_halo: bool,
    use_overlap: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused compressor forward pass (post-GEMM operations).

    Args:
        kv: [sq_with_halo, b, coff * head_dim] from linear_wkv
        score: [sq_with_halo, b, coff * head_dim] from linear_wgate
        ape: [ratio, coff * head_dim] absolute position embedding (fp32)
        norm_weight: [head_dim] RMSNorm weight
        ratio: compression ratio (4 or 128)
        head_dim: output head dimension
        has_halo: whether to discard first group (halo)
        use_overlap: whether overlap mode is active (ratio=4)
        eps: RMSNorm epsilon

    Returns:
        [n_out, b, head_dim] normalized compressed KV (bf16)
    """
    return FusedCompressorPostGEMM.apply(kv, score, ape, norm_weight, ratio, head_dim, has_halo, use_overlap, eps)
