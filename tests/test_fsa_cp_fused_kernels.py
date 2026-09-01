"""Unit tests for fused FSA CP Triton kernels.

Tests the fused zigzag+transpose kernels against the original separate operations
to ensure numerical equivalence.

Run:  python -m pytest tests/test_fsa_cp_fused_kernels.py -v
  or: python tests/test_fsa_cp_fused_kernels.py
"""

import torch
import pytest


def _build_undo_order(cp_size: int) -> list:
    """Reproduce the undo-zigzag order from mamba_context_parallel."""
    num_chunks = cp_size * 2
    # undo order: maps output_chunk_idx -> input_chunk_idx
    # Original code: order = [2*i for i in range(cp_size)] + [num_chunks-2*i-1 for i in range(cp_size)]
    # That gives the mapping: "output chunk j should come from input chunk order[j]"
    order = [2 * i for i in range(cp_size)] + [
        num_chunks - 2 * i - 1 for i in range(cp_size)
    ]
    return order


def _build_redo_order(cp_size: int) -> list:
    """Reproduce the redo-zigzag order from mamba_context_parallel."""
    num_chunks = cp_size * 2
    order = [None] * num_chunks
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, num_chunks))
    return order


def reference_undo_zigzag(x: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Reference: torch.chunk + reorder + cat (same as original code)."""
    num_chunks = cp_size * 2
    chunks = torch.chunk(x, num_chunks, dim=0)
    order = _build_undo_order(cp_size)
    return torch.cat([chunks[i] for i in order], dim=0)


def reference_redo_zigzag(x: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Reference: torch.chunk + reorder + cat (same as original code)."""
    num_chunks = cp_size * 2
    chunks = torch.chunk(x, num_chunks, dim=0)
    order = _build_redo_order(cp_size)
    return torch.cat([chunks[i] for i in order], dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedZigzagTranspose:
    """Test fused_zigzag_undo_sbhd_to_bshd and fused_bshd_to_sbhd_zigzag_redo."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import the kernels (skip if triton not available)."""
        try:
            from megatron.core.extensions.fsa_cp_fused_kernels import (
                fused_zigzag_undo_sbhd_to_bshd,
                fused_bshd_to_sbhd_zigzag_redo,
            )
            self.fused_undo = fused_zigzag_undo_sbhd_to_bshd
            self.fused_redo = fused_bshd_to_sbhd_zigzag_redo
        except ImportError:
            pytest.skip("fsa_cp_fused_kernels or triton not available")

    @pytest.mark.parametrize("cp_size", [2, 4])
    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("num_heads", [1, 2, 4])
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_fused_undo_matches_reference(self, cp_size, batch, num_heads, head_dim, dtype):
        """fused_zigzag_undo_sbhd_to_bshd should match: undo_zigzag + transpose(0,1).contiguous()"""
        sq_global = 1024 * cp_size  # must be divisible by 2*cp_size
        # Create input in SBHD layout with zigzag order
        x_sbhd = torch.randn(sq_global, batch, num_heads, head_dim, device="cuda", dtype=dtype)

        # Reference path: undo zigzag on 3d, then reshape to 4d, then transpose
        x_3d = x_sbhd.reshape(sq_global, batch, num_heads * head_dim)
        x_undone_3d = reference_undo_zigzag(x_3d, cp_size)
        x_undone_4d = x_undone_3d.reshape(sq_global, batch, num_heads, head_dim)
        ref_bshd = x_undone_4d.transpose(0, 1).contiguous()

        # Fused path
        fused_bshd = self.fused_undo(x_sbhd, cp_size)

        assert fused_bshd.shape == ref_bshd.shape, f"{fused_bshd.shape} != {ref_bshd.shape}"
        assert fused_bshd.is_contiguous(), "fused output should be contiguous"
        torch.testing.assert_close(fused_bshd, ref_bshd, atol=0, rtol=0)

    @pytest.mark.parametrize("cp_size", [2, 4])
    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("num_heads", [1, 2, 4])
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_fused_redo_matches_reference(self, cp_size, batch, num_heads, head_dim, dtype):
        """fused_bshd_to_sbhd_zigzag_redo should match: transpose(0,1).contiguous() + redo_zigzag"""
        sq_global = 1024 * cp_size
        # Create input in BSHD layout, sequential order
        x_bshd = torch.randn(batch, sq_global, num_heads, head_dim, device="cuda", dtype=dtype)

        # Reference: transpose to SBHD, then redo zigzag
        x_sbhd = x_bshd.transpose(0, 1).contiguous()
        x_3d = x_sbhd.reshape(sq_global, batch, num_heads * head_dim)
        ref_3d = reference_redo_zigzag(x_3d, cp_size)
        ref_sbhd = ref_3d.reshape(sq_global, batch, num_heads, head_dim)

        # Fused path
        fused_sbhd = self.fused_redo(x_bshd, cp_size)

        assert fused_sbhd.shape == ref_sbhd.shape, f"{fused_sbhd.shape} != {ref_sbhd.shape}"
        torch.testing.assert_close(fused_sbhd, ref_sbhd, atol=0, rtol=0)

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_roundtrip(self, cp_size):
        """undo then redo should recover the original data (up to layout)."""
        sq_global = 2048 * cp_size
        batch, num_heads, head_dim = 1, 2, 256
        dtype = torch.bfloat16

        # Start in SBHD zigzag
        x_sbhd_zigzag = torch.randn(sq_global, batch, num_heads, head_dim, device="cuda", dtype=dtype)

        # undo -> BSHD sequential
        bshd = self.fused_undo(x_sbhd_zigzag, cp_size)
        # redo -> SBHD zigzag
        recovered = self.fused_redo(bshd, cp_size)

        torch.testing.assert_close(recovered, x_sbhd_zigzag, atol=0, rtol=0)

    def test_qwen36_config(self):
        """Test with the actual Qwen3.6-35B-A3B dimensions."""
        # TP=4, CP=2: num_q_per_rank=2, num_kv=1, head_dim=256
        cp_size = 2
        sq_global = 32768
        batch = 1
        dtype = torch.bfloat16

        # Q: 2 heads after a2a split
        q_sbhd = torch.randn(sq_global, batch, 2, 256, device="cuda", dtype=dtype)
        q_bshd = self.fused_undo(q_sbhd, cp_size)
        assert q_bshd.shape == (batch, sq_global, 2, 256)
        assert q_bshd.is_contiguous()

        # K/V: 1 head after AllGather
        k_sbhd = torch.randn(sq_global, batch, 1, 256, device="cuda", dtype=dtype)
        k_bshd = self.fused_undo(k_sbhd, cp_size)
        assert k_bshd.shape == (batch, sq_global, 1, 256)
        assert k_bshd.is_contiguous()

        # Output: back from BSHD to SBHD zigzag
        out_bshd = torch.randn(batch, sq_global, 2, 256, device="cuda", dtype=dtype)
        out_sbhd = self.fused_redo(out_bshd, cp_size)
        assert out_sbhd.shape == (sq_global, batch, 2, 256)
        assert out_sbhd.is_contiguous()

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_gradient_flow_undo(self, cp_size):
        """Ensure gradients flow through the fused undo kernel."""
        sq_global = 1024 * cp_size
        batch, num_heads, head_dim = 1, 2, 128

        x = torch.randn(sq_global, batch, num_heads, head_dim,
                         device="cuda", dtype=torch.float32, requires_grad=True)

        y = self.fused_undo(x, cp_size)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Every element contributes exactly once to the sum, so grad should be all 1s
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_gradient_flow_redo(self, cp_size):
        """Ensure gradients flow through the fused redo kernel."""
        sq_global = 1024 * cp_size
        batch, num_heads, head_dim = 1, 2, 128

        x = torch.randn(batch, sq_global, num_heads, head_dim,
                         device="cuda", dtype=torch.float32, requires_grad=True)

        y = self.fused_redo(x, cp_size)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
