"""Unit test for fused_compressor_kernel correctness."""

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/share/project/lixianduo/context_parallel/Megatron-LM-FL")

from megatron.core.transformer.experimental_attention_variant.fused_compressor_kernel import (
    fused_compressor_post_gemm,
)


def reference_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, has_halo, use_overlap, eps):
    """Reference implementation matching the original PyTorch code."""
    sq_with_halo, b, _ = kv.shape
    cutoff = (sq_with_halo // ratio) * ratio
    kv = kv[:cutoff]
    score = score[:cutoff]
    n_compressed = cutoff // ratio

    coff = 2 if use_overlap else 1

    # Reshape: [n_compressed, ratio, b, coff * head_dim]
    kv = kv.view(n_compressed, ratio, b, -1)
    score = score.view(n_compressed, ratio, b, -1)

    # APE: [ratio, coff * head_dim] -> [1, ratio, 1, coff * head_dim]
    score = score + ape.view(1, ratio, 1, -1)

    if use_overlap:
        # _overlap_transform
        d = head_dim
        n_groups = n_compressed

        # kv transform
        new_kv = kv.new_full((n_groups, 2 * ratio, b, d), 0)
        new_kv[:, ratio:] = kv[:, :, :, d:]
        new_kv[1:, :ratio] = kv[:-1, :, :, :d]
        kv = new_kv

        # score transform
        new_score = score.new_full((n_groups, 2 * ratio, b, d), float("-inf"))
        new_score[:, ratio:] = score[:, :, :, d:]
        new_score[1:, :ratio] = score[:-1, :, :, :d]
        score = new_score

    # softmax + weighted sum
    kv = (kv * torch.softmax(score.float(), dim=1)).sum(dim=1)  # [n_compressed, b, head_dim]

    # Discard halo
    if has_halo:
        kv = kv[1:]
        n_compressed = n_compressed - 1

    # RMSNorm
    kv = kv.float()
    var = (kv * kv).mean(dim=-1, keepdim=True)
    kv = kv * torch.rsqrt(var + eps)
    kv = kv * norm_weight.float()
    kv = kv.to(torch.bfloat16)

    return kv


def test_no_overlap():
    """Test non-overlap mode (ratio=128)."""
    torch.manual_seed(42)
    ratio = 128
    head_dim = 128
    coff = 1
    sq_with_halo = 2048 + ratio  # with halo
    b = 1

    kv = torch.randn(sq_with_halo, b, coff * head_dim, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(sq_with_halo, b, coff * head_dim, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(ratio, coff * head_dim, dtype=torch.float32, device="cuda")
    norm_weight = torch.ones(head_dim, dtype=torch.float32, device="cuda")

    ref = reference_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, True, False, 1e-6)
    out = fused_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, True, False, 1e-6)

    print(f"[no_overlap] ref shape: {ref.shape}, out shape: {out.shape}")
    print(f"[no_overlap] max diff: {(ref.float() - out.float()).abs().max().item():.6f}")
    print(f"[no_overlap] mean diff: {(ref.float() - out.float()).abs().mean().item():.6f}")

    # Allow some tolerance for bf16 + triton
    assert ref.shape == out.shape, f"Shape mismatch: {ref.shape} vs {out.shape}"
    assert (ref.float() - out.float()).abs().max().item() < 0.05, "Too large difference!"
    print("[no_overlap] PASSED\n")


def test_overlap():
    """Test overlap mode (ratio=4)."""
    torch.manual_seed(42)
    ratio = 4
    head_dim = 128
    coff = 2
    sq_with_halo = 2048 + ratio  # with halo
    b = 1

    kv = torch.randn(sq_with_halo, b, coff * head_dim, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(sq_with_halo, b, coff * head_dim, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(ratio, coff * head_dim, dtype=torch.float32, device="cuda")
    norm_weight = torch.ones(head_dim, dtype=torch.float32, device="cuda")

    ref = reference_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, True, True, 1e-6)
    out = fused_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, True, True, 1e-6)

    print(f"[overlap] ref shape: {ref.shape}, out shape: {out.shape}")
    print(f"[overlap] max diff: {(ref.float() - out.float()).abs().max().item():.6f}")
    print(f"[overlap] mean diff: {(ref.float() - out.float()).abs().mean().item():.6f}")

    assert ref.shape == out.shape, f"Shape mismatch: {ref.shape} vs {out.shape}"
    assert (ref.float() - out.float()).abs().max().item() < 0.05, "Too large difference!"
    print("[overlap] PASSED\n")


def test_no_halo():
    """Test without halo (cp_size=1 scenario)."""
    torch.manual_seed(42)
    ratio = 128
    head_dim = 128
    coff = 1
    sq = 2048
    b = 1

    kv = torch.randn(sq, b, coff * head_dim, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(sq, b, coff * head_dim, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(ratio, coff * head_dim, dtype=torch.float32, device="cuda")
    norm_weight = torch.ones(head_dim, dtype=torch.float32, device="cuda")

    ref = reference_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, False, False, 1e-6)
    out = fused_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, False, False, 1e-6)

    print(f"[no_halo] ref shape: {ref.shape}, out shape: {out.shape}")
    print(f"[no_halo] max diff: {(ref.float() - out.float()).abs().max().item():.6f}")
    print(f"[no_halo] mean diff: {(ref.float() - out.float()).abs().mean().item():.6f}")

    assert ref.shape == out.shape, f"Shape mismatch: {ref.shape} vs {out.shape}"
    assert (ref.float() - out.float()).abs().max().item() < 0.05, "Too large difference!"
    print("[no_halo] PASSED\n")


def test_backward():
    """Test that backward pass produces valid gradients."""
    torch.manual_seed(42)
    ratio = 4
    head_dim = 128
    coff = 2
    sq_with_halo = 64 + ratio
    b = 1

    kv = torch.randn(sq_with_halo, b, coff * head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    score = torch.randn(sq_with_halo, b, coff * head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ape = torch.randn(ratio, coff * head_dim, dtype=torch.float32, device="cuda", requires_grad=True)
    norm_weight = torch.ones(head_dim, dtype=torch.float32, device="cuda", requires_grad=True)

    out = fused_compressor_post_gemm(kv, score, ape, norm_weight, ratio, head_dim, True, True, 1e-6)
    loss = out.float().sum()
    loss.backward()

    assert kv.grad is not None, "kv.grad is None"
    assert score.grad is not None, "score.grad is None"
    assert ape.grad is not None, "ape.grad is None"
    assert norm_weight.grad is not None, "norm_weight.grad is None"

    print(f"[backward] kv.grad norm: {kv.grad.float().norm().item():.6f}")
    print(f"[backward] score.grad norm: {score.grad.float().norm().item():.6f}")
    print(f"[backward] ape.grad norm: {ape.grad.float().norm().item():.6f}")
    print(f"[backward] norm_weight.grad norm: {norm_weight.grad.float().norm().item():.6f}")

    assert kv.grad.float().norm().item() > 0, "kv.grad is zero"
    assert score.grad.float().norm().item() > 0, "score.grad is zero"
    print("[backward] PASSED\n")


if __name__ == "__main__":
    test_no_overlap()
    test_overlap()
    test_no_halo()
    test_backward()
    print("All tests passed!")
