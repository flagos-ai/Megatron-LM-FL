# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Accuracy and performance tests for FusedIndexerSparseAttnFunc.

Validates that ``fused_indexer_sparse_attn`` from the Triton plugin produces
numerically equivalent attention output to ``unfused_compressed_sparse_attn``
from ``megatron.core.transformer.experimental_attention_variant.csa``.

The fused path:
1. Scores + top-K selection via indexer (q_indexer, k_indexer, weights)
2. Combines compressed top-K indices with window indices
3. Runs sparse attention over combined indices
4. Computes indexer KL loss (sparse or dense variant)

The unfused path takes pre-computed indices and runs only the sparse attention.
We compare the attention output (not the loss) between them.

Run with: pytest tests/unit_tests/plugin/dsa_kernel/test_fused_indexer_sparse_attn.py -v -s
Requires: CUDA GPU with Triton support.
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import pytest
import torch
from torch import Tensor

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
    fused_indexer_sparse_attn,
    _sbhd_to_bshd_indexer_inputs,
    _indexer_topk_bshd,
    _kl_loss_from_target_predict,
    _kl_loss_from_dense_scores,
)
from megatron.plugin.dsa_kernel.triton_indexer_kernels import (
    sparse_indexer_score_recompute,
    sparse_attn_score_recompute,
    dense_indexer_score_recompute,
    dense_attn_score_recompute,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    unfused_compressed_sparse_attn,
    get_window_topk_idxs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fused_inputs(
    sq: int,
    b: int,
    np_: int,
    hn: int,
    n_comp: int,
    win_topk: int,
    idx_nh: int,
    idx_hd: int,
    indexer_topk: int,
    ratio: int,
    kv_offset: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Generate random inputs for fused_indexer_sparse_attn.

    KV layout: [n_kv, b, hn] where n_kv = kv_offset + n_comp (total KV length
    including original tokens and compressed tokens).

    Convention:
    - Positions [0, kv_offset) are original tokens
    - Positions [kv_offset, kv_offset + n_comp) are compressed tokens
    - window_idxs refer to original token positions [0, kv_offset)
    - k_indexer has n_comp positions (compressed tokens)
    - indexer top-K indices are offsets into k_indexer [0, n_comp), later
      shifted by kv_offset for the combined index set
    """
    torch.manual_seed(seed)

    n_kv = kv_offset + n_comp

    query = torch.randn(sq, b, np_, hn, device=device, dtype=torch.bfloat16)
    kv_full = torch.randn(n_kv, b, hn, device=device, dtype=torch.bfloat16)
    attn_sink = torch.randn(np_, device=device, dtype=torch.float32) * 0.1

    # Window indices: point to original token positions [0, kv_offset)
    # Use a simple sliding window pattern
    window_idxs = get_window_topk_idxs(win_topk, b, sq, device).int()
    # Clamp to valid range [0, kv_offset) and mark out-of-range as -1
    invalid_win = window_idxs >= kv_offset
    window_idxs = window_idxs.clamp(max=kv_offset - 1)
    window_idxs[invalid_win] = -1

    # Indexer inputs
    q_indexer = torch.randn(sq, b, idx_nh, idx_hd, device=device, dtype=torch.bfloat16)
    k_indexer = torch.randn(n_comp, b, idx_hd, device=device, dtype=torch.bfloat16)
    weights = torch.randn(sq, b, idx_nh, device=device, dtype=torch.bfloat16).abs()

    softmax_scale = 1.0 / math.sqrt(hn)
    indexer_softmax_scale = 1.0 / math.sqrt(idx_hd)

    return {
        "query": query,
        "kv_full": kv_full,
        "attn_sink": attn_sink,
        "window_idxs": window_idxs,
        "q_indexer": q_indexer,
        "k_indexer": k_indexer,
        "weights": weights,
        "indexer_topk": indexer_topk,
        "ratio": ratio,
        "softmax_scale": softmax_scale,
        "indexer_softmax_scale": indexer_softmax_scale,
        "kv_offset": kv_offset,
        "n_kv": n_kv,
        "sq": sq,
        "b": b,
        "np": np_,
        "hn": hn,
        "n_comp": n_comp,
        "win_topk": win_topk,
    }


def _run_fused(inputs: dict, sparse_loss: bool, loss_coeff: float = 0.1) -> Tuple[Tensor, Tensor]:
    """Run fused_indexer_sparse_attn and return (output, indexer_loss)."""
    return fused_indexer_sparse_attn(
        inputs["query"],
        inputs["kv_full"],
        inputs["attn_sink"],
        inputs["window_idxs"],
        inputs["q_indexer"],
        inputs["k_indexer"],
        inputs["weights"],
        inputs["indexer_topk"],
        inputs["ratio"],
        inputs["softmax_scale"],
        inputs["indexer_softmax_scale"],
        loss_coeff,
        sparse_loss,
        inputs["kv_offset"],
        calculate_per_token_loss=False,
    )


def _run_unfused_with_same_indices(inputs: dict) -> Tuple[Tensor, Tensor]:
    """Run unfused_compressed_sparse_attn with the same indices that the fused
    path would compute.

    Steps:
    1. Replicate the indexer scoring + top-K selection from the fused path
    2. Combine indices (compressed + window) — same order as fused
    3. Run unfused_compressed_sparse_attn (ground truth)

    Returns:
        (output, combined_topk_idxs)
    """
    from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
        _sbhd_to_bshd_indexer_inputs,
        _indexer_topk_bshd,
    )

    n_comp = inputs["n_comp"]
    kv_offset = inputs["kv_offset"]
    effective_topk = min(inputs["indexer_topk"], n_comp)

    # Replicate indexer scoring (same as fused path steps 1-3)
    q_idx_bshd, k_idx_bsd, w_bsh, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
        inputs["q_indexer"], inputs["k_indexer"],
        inputs["weights"], inputs["indexer_softmax_scale"]
    )
    topk_indices_cmp, _, _ = _indexer_topk_bshd(
        q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, inputs["ratio"]
    )

    # Add kv_offset to compressed indices
    topk_indices_global = topk_indices_cmp.clone()
    valid_cmp = topk_indices_global >= 0
    topk_indices_global[valid_cmp] += kv_offset

    # Combine: compressed first, then window (same order as fused)
    combined_idxs = torch.cat(
        [topk_indices_global, inputs["window_idxs"]], dim=-1
    )  # (b, sq, total_topk)

    # Run unfused sparse attention (ground truth)
    output = unfused_compressed_sparse_attn(
        inputs["query"],
        inputs["kv_full"],
        inputs["attn_sink"],
        combined_idxs,
        inputs["softmax_scale"],
    )

    return output, combined_idxs


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


class TestFusedIndexerSparseAttnAccuracy:
    """Compare fused_indexer_sparse_attn output against unfused_compressed_sparse_attn.

    The fused path uses triton_sparse_attn_forward (online softmax) while the
    unfused path uses standard materialized softmax. Both include the attention
    sink. We verify the attention outputs are numerically close.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            # Small shapes — tighter tolerance
            (32, 1, 4, 128, 16, 8, 2, 64, 8, 4),
            (32, 2, 4, 128, 32, 16, 2, 64, 16, 4),
            # Medium shapes
            (64, 2, 4, 128, 64, 32, 2, 64, 32, 4),
            (128, 1, 8, 128, 32, 16, 4, 64, 16, 4),
            # Larger topk (closer to production)
            (64, 1, 4, 128, 128, 32, 2, 64, 64, 4),
            (128, 1, 4, 128, 256, 64, 2, 64, 128, 4),
        ],
        ids=[
            "small_B1",
            "small_B2",
            "medium_B2",
            "medium_8heads",
            "large_topk64",
            "large_topk128",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_output_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device,
    ):
        """Fused attention output matches unfused path (both with attn_sink)."""
        kv_offset = sq  # original tokens occupy [0, sq)
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        # Fused path
        out_fused, indexer_loss = _run_fused(inputs, sparse_loss=sparse_loss)

        # Unfused path (same indices)
        out_unfused, _ = _run_unfused_with_same_indices(inputs)

        # Compare attention output
        out_fused_f = out_fused.float()
        out_unfused_f = out_unfused.float()

        abs_diff = (out_fused_f - out_unfused_f).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # Tolerance: bf16 accumulation in triton online softmax vs PyTorch materialized
        atol = 5e-2
        rtol = 5e-2
        assert torch.allclose(out_fused_f, out_unfused_f, atol=atol, rtol=rtol), (
            f"Output mismatch (sparse_loss={sparse_loss}): "
            f"max abs diff = {max_diff:.4e}, mean abs diff = {mean_diff:.4e}"
        )

        # Cosine similarity should be very high
        cos_sim = torch.nn.functional.cosine_similarity(
            out_fused_f.reshape(-1).unsqueeze(0),
            out_unfused_f.reshape(-1).unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, (
            f"Cosine similarity too low: {cos_sim:.6f}"
        )

        # Indexer loss should be finite and non-negative
        assert torch.isfinite(indexer_loss), f"Indexer loss is not finite: {indexer_loss.item()}"
        assert indexer_loss.item() >= 0, f"Indexer loss is negative: {indexer_loss.item()}"

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (64, 2, 4, 128, 64, 32, 2, 64, 32, 4),
        ],
    )
    def test_loss_nonzero(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Indexer loss is non-trivially > 0 (not degenerate)."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        _, loss_sparse = _run_fused(inputs, sparse_loss=True, loss_coeff=1.0)
        _, loss_dense = _run_fused(inputs, sparse_loss=False, loss_coeff=1.0)

        # With random inputs, KL divergence should be > 0
        assert loss_sparse.item() > 1e-6, (
            f"Sparse loss is suspiciously close to zero: {loss_sparse.item()}"
        )
        assert loss_dense.item() > 1e-6, (
            f"Dense loss is suspiciously close to zero: {loss_dense.item()}"
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (32, 2, 4, 128, 32, 16, 2, 64, 16, 4),
        ],
    )
    def test_deterministic(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Multiple calls with same input produce identical results."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        out1, loss1 = _run_fused(inputs, sparse_loss=True)
        out2, loss2 = _run_fused(inputs, sparse_loss=True)

        assert torch.equal(out1, out2), "Fused output is non-deterministic"
        assert torch.equal(loss1, loss2), "Fused loss is non-deterministic"

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (32, 1, 4, 128, 16, 8, 2, 64, 8, 4),
        ],
    )
    def test_loss_coeff_scaling(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Indexer loss scales linearly with loss_coeff."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        _, loss_1x = _run_fused(inputs, sparse_loss=True, loss_coeff=1.0)
        _, loss_2x = _run_fused(inputs, sparse_loss=True, loss_coeff=2.0)

        ratio_actual = loss_2x.item() / max(loss_1x.item(), 1e-12)
        assert abs(ratio_actual - 2.0) < 0.01, (
            f"Loss does not scale linearly: ratio = {ratio_actual:.4f}, expected 2.0"
        )


# ---------------------------------------------------------------------------
# Backward accuracy tests
# ---------------------------------------------------------------------------


class TestFusedIndexerSparseAttnBackward:
    """Validate backward-pass gradient accuracy of fused_indexer_sparse_attn.

    Compares gradients (d_query, d_kv_full, d_attn_sink) from the fused Triton
    path against PyTorch-autograd gradients from unfused_compressed_sparse_attn.

    The indexer branch (q_indexer, k_indexer, weights) has a simplified backward
    in the fused path, so we only validate gradients that flow through the sparse
    attention portion.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @staticmethod
    def _run_fused_with_grad(inputs: dict, sparse_loss: bool) -> dict:
        """Run fused path with gradients enabled, return output + grads."""
        query = inputs["query"].clone().detach().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().detach().requires_grad_(True)
        attn_sink = inputs["attn_sink"].clone().detach().requires_grad_(True)

        out, loss = fused_indexer_sparse_attn(
            query, kv_full, attn_sink, inputs["window_idxs"],
            inputs["q_indexer"], inputs["k_indexer"], inputs["weights"],
            inputs["indexer_topk"], inputs["ratio"],
            inputs["softmax_scale"], inputs["indexer_softmax_scale"],
            0.1, sparse_loss, inputs["kv_offset"], False,
        )

        # Backward with a simple scalar objective
        scalar = out.float().sum() + loss
        scalar.backward()

        return {
            "output": out,
            "loss": loss,
            "grad_query": query.grad,
            "grad_kv_full": kv_full.grad,
            "grad_attn_sink": attn_sink.grad,
        }

    @staticmethod
    def _run_unfused_with_grad(inputs: dict) -> dict:
        """Run unfused path with gradients enabled, return output + grads."""
        from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
            _sbhd_to_bshd_indexer_inputs,
            _indexer_topk_bshd,
        )

        query = inputs["query"].clone().detach().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().detach().requires_grad_(True)
        attn_sink = inputs["attn_sink"].clone().detach().requires_grad_(True)

        n_comp = inputs["n_comp"]
        kv_offset = inputs["kv_offset"]
        effective_topk = min(inputs["indexer_topk"], n_comp)

        # Replicate indexer to get same indices (no grad needed for indices)
        q_idx_bshd, k_idx_bsd, _, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
            inputs["q_indexer"], inputs["k_indexer"],
            inputs["weights"], inputs["indexer_softmax_scale"]
        )
        topk_indices_cmp, _, _ = _indexer_topk_bshd(
            q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, inputs["ratio"]
        )

        topk_indices_global = topk_indices_cmp.clone()
        valid_cmp = topk_indices_global >= 0
        topk_indices_global[valid_cmp] += kv_offset

        combined_idxs = torch.cat(
            [topk_indices_global, inputs["window_idxs"]], dim=-1
        )

        out = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, combined_idxs, inputs["softmax_scale"],
        )

        scalar = out.float().sum()
        scalar.backward()

        return {
            "output": out,
            "grad_query": query.grad,
            "grad_kv_full": kv_full.grad,
            "grad_attn_sink": attn_sink.grad,
        }

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (32, 1, 4, 128, 16, 8, 2, 64, 8, 4),
            (32, 2, 4, 128, 32, 16, 2, 64, 16, 4),
            (64, 2, 4, 128, 64, 32, 2, 64, 32, 4),
            (128, 1, 8, 128, 32, 16, 4, 64, 16, 4),
            (64, 1, 4, 128, 128, 32, 2, 64, 64, 4),
        ],
        ids=[
            "small_B1",
            "small_B2",
            "medium_B2",
            "medium_8heads",
            "large_topk64",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_query_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device,
    ):
        """Gradient w.r.t. query matches unfused reference."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_with_grad(inputs, sparse_loss)
        unfused_res = self._run_unfused_with_grad(inputs)

        g_fused = fused_res["grad_query"].float()
        g_unfused = unfused_res["grad_query"].float()

        abs_diff = (g_fused - g_unfused).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # bf16 backward through online softmax vs materialized — use cosine similarity
        # as primary metric since element-wise tolerance is too strict for bf16 numerics
        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        assert cos_sim > 0.95, (
            f"grad_query cosine similarity too low: {cos_sim:.6f} "
            f"(max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e})"
        )
        print(
            f"\n  grad_query: cos_sim={cos_sim:.6f}, "
            f"max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}"
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (32, 1, 4, 128, 16, 8, 2, 64, 8, 4),
            (32, 2, 4, 128, 32, 16, 2, 64, 16, 4),
            (64, 2, 4, 128, 64, 32, 2, 64, 32, 4),
            (128, 1, 8, 128, 32, 16, 4, 64, 16, 4),
        ],
        ids=[
            "small_B1",
            "small_B2",
            "medium_B2",
            "medium_8heads",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_kv_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device,
    ):
        """Gradient w.r.t. kv_full matches unfused reference."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_with_grad(inputs, sparse_loss)
        unfused_res = self._run_unfused_with_grad(inputs)

        g_fused = fused_res["grad_kv_full"].float()
        g_unfused = unfused_res["grad_kv_full"].float()

        abs_diff = (g_fused - g_unfused).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        assert cos_sim > 0.95, (
            f"grad_kv_full cosine similarity too low: {cos_sim:.6f} "
            f"(max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e})"
        )
        print(
            f"\n  grad_kv_full: cos_sim={cos_sim:.6f}, "
            f"max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}"
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (32, 1, 4, 128, 16, 8, 2, 64, 8, 4),
            (64, 2, 4, 128, 64, 32, 2, 64, 32, 4),
            (128, 1, 8, 128, 32, 16, 4, 64, 16, 4),
        ],
        ids=[
            "small_B1",
            "medium_B2",
            "medium_8heads",
        ],
    )
    def test_grad_attn_sink_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Gradient w.r.t. attn_sink matches unfused reference."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_with_grad(inputs, sparse_loss=True)
        unfused_res = self._run_unfused_with_grad(inputs)

        g_fused = fused_res["grad_attn_sink"].float()
        g_unfused = unfused_res["grad_attn_sink"].float()

        abs_diff = (g_fused - g_unfused).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # attn_sink is a small vector (np,), use relative tolerance
        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        assert cos_sim > 0.90, (
            f"grad_attn_sink cosine similarity too low: {cos_sim:.6f} "
            f"(max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e})"
        )
        print(
            f"\n  grad_attn_sink: cos_sim={cos_sim:.6f}, "
            f"max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}"
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (32, 1, 4, 128, 16, 8, 2, 64, 8, 4),
            (64, 2, 4, 128, 64, 32, 2, 64, 32, 4),
        ],
        ids=["small", "medium"],
    )
    def test_backward_no_nan_inf(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Ensure backward pass produces no NaN or Inf in any gradient."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_with_grad(inputs, sparse_loss=True)

        for name in ("grad_query", "grad_kv_full", "grad_attn_sink"):
            g = fused_res[name]
            assert not torch.isnan(g).any(), f"{name} contains NaN"
            assert not torch.isinf(g).any(), f"{name} contains Inf"


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


class TestFusedIndexerSparseAttnPerformance:
    """Performance benchmarks: Fused Triton path vs unfused PyTorch CSA path.

    Measures:
    - Forward time for both paths
    - Speedup from fused operation
    - Reports per-shape timing
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @staticmethod
    def _benchmark(fn, warmup: int = 10, iters: int = 50) -> float:
        """Benchmark a CUDA function. Returns median elapsed ms."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times.sort()
        return times[len(times) // 2]

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            # Production-scale shapes with large topk
            (512, 1, 16, 128, 128, 64, 4, 64, 64, 4),
            (512, 2, 16, 128, 256, 64, 4, 64, 128, 4),
            (1024, 1, 16, 128, 256, 128, 4, 64, 128, 4),
            (1024, 1, 16, 128, 512, 128, 4, 64, 256, 4),
            (2048, 1, 8, 128, 512, 128, 4, 64, 256, 4),
            (2048, 1, 8, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "sq512_b1_topk64",
            "sq512_b2_topk128",
            "sq1024_b1_topk128",
            "sq1024_b1_topk256",
            "sq2048_b1_topk256",
            "sq2048_b1_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_performance_forward(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device,
    ):
        """Measure forward performance: fused Triton vs unfused CSA + indexer loss."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        # Fused path benchmark (attention + indexer loss in one call)
        time_fused = self._benchmark(
            lambda: _run_fused(inputs, sparse_loss=sparse_loss)
        )

        # Unfused path benchmark: attention + indexer loss separately (fair comparison)
        # Pre-compute indices (not timed — same for both paths)
        with torch.no_grad():
            q_idx_bshd, k_idx_bsd, _, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
                inputs["q_indexer"], inputs["k_indexer"],
                inputs["weights"], inputs["indexer_softmax_scale"]
            )
            effective_topk = min(inputs["indexer_topk"], n_comp)
            topk_indices_cmp, _, _ = _indexer_topk_bshd(
                q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, ratio
            )
            topk_indices_global = topk_indices_cmp.clone()
            valid_cmp = topk_indices_global >= 0
            topk_indices_global[valid_cmp] += kv_offset
            combined_idxs = torch.cat(
                [topk_indices_global, inputs["window_idxs"]], dim=-1
            )

        d = hn
        idx_nh_val = inputs["q_indexer"].shape[2]

        def run_unfused_forward_with_loss():
            # 1. Unfused sparse attention forward
            out = unfused_compressed_sparse_attn(
                inputs["query"], inputs["kv_full"], inputs["attn_sink"],
                combined_idxs, inputs["softmax_scale"],
            )

            # 2. Compute LSE (needed for indexer loss target)
            kv_t = inputs["kv_full"].permute(1, 0, 2)  # (b, n_kv, hn)
            safe_idx = combined_idxs.clamp(min=0).long()
            safe_idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, -1, hn)
            kv_g = torch.gather(
                kv_t.unsqueeze(1).expand(-1, sq, -1, -1), dim=2, index=safe_idx_exp
            ).float()
            q_perm = inputs["query"].permute(1, 2, 0, 3).float()  # (b, np, sq, hn)
            scores = torch.einsum("bnsh,bskh->bnsk", q_perm, kv_g) * inputs["softmax_scale"]
            invalid_mask = (combined_idxs < 0).unsqueeze(1)
            scores = scores.masked_fill(invalid_mask, float("-inf"))
            sink_val = inputs["attn_sink"].view(1, np_, 1, 1).float()
            scores_max = torch.max(scores.max(dim=-1, keepdim=True).values, sink_val)
            exp_scores = torch.exp(scores - scores_max)
            exp_sink_val = torch.exp(sink_val - scores_max)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink_val
            lse_full = (scores_max + torch.log(sum_exp)).squeeze(-1)  # (b, np, sq)
            lse_bsh = lse_full.permute(0, 2, 1).contiguous()  # (b, sq, np)

            # 3. Indexer loss computation
            qi_bshd = inputs["q_indexer"].permute(1, 0, 2, 3).contiguous()
            ki_bsd = inputs["k_indexer"].permute(1, 0, 2).contiguous()
            w_raw = inputs["weights"].permute(1, 0, 2).contiguous()
            w_scaled = w_raw * inputs["indexer_softmax_scale"]

            if sparse_loss:
                # Partial LSE over compressed indices only
                safe_cmp_global = topk_indices_cmp.clone()
                valid_m = safe_cmp_global >= 0
                safe_cmp_global[valid_m] += kv_offset
                safe_cmp_idx = safe_cmp_global.clamp(min=0).long()
                safe_cmp_idx_exp = safe_cmp_idx.unsqueeze(-1).expand(-1, -1, -1, hn)
                kv_cmp = torch.gather(
                    kv_t.unsqueeze(1).expand(-1, sq, -1, -1), dim=2, index=safe_cmp_idx_exp
                ).float()
                scores_cmp = torch.einsum("bnsh,bskh->bnsk", q_perm, kv_cmp) * inputs["softmax_scale"]
                inv_cmp_mask = (~valid_m).unsqueeze(1)
                scores_cmp = scores_cmp.masked_fill(inv_cmp_mask, float("-inf"))
                sc_max = torch.max(scores_cmp.max(dim=-1, keepdim=True).values, sink_val)
                ec = torch.exp(scores_cmp - sc_max)
                es = torch.exp(sink_val - sc_max)
                se = ec.sum(dim=-1, keepdim=True) + es
                lse_indexer_bsh = (sc_max + torch.log(se)).squeeze(-1).permute(0, 2, 1).contiguous()

                predict_result = sparse_indexer_score_recompute(
                    qi_bshd, ki_bsd, w_scaled, topk_indices_cmp,
                    qhead_per_kv_head=idx_nh_val,
                )
                q_attn_bshd = inputs["query"].permute(1, 0, 2, 3).contiguous()
                k_attn_bsd = inputs["kv_full"][:, :, :d].permute(1, 0, 2).contiguous()
                target_result = sparse_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_indexer_bsh,
                    topk_indices_cmp + kv_offset,
                    inputs["softmax_scale"], qhead_per_kv_head=np_,
                )
                _loss = _kl_loss_from_target_predict(
                    target_result["target"], predict_result["predict"],
                    topk_indices_cmp, 0.1, False,
                )
            else:
                q_attn_bshd = inputs["query"].permute(1, 0, 2, 3).contiguous()
                k_attn_bsd = inputs["kv_full"][kv_offset:kv_offset + n_comp, :, :d].permute(1, 0, 2).contiguous()
                dense_idx_result = dense_indexer_score_recompute(
                    qi_bshd, ki_bsd, w_scaled,
                    qhead_per_kv_head=idx_nh_val,
                    sm_scale=inputs["indexer_softmax_scale"],
                    ratio=ratio,
                )
                dense_attn_result = dense_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_bsh,
                    qhead_per_kv_head=np_,
                    softmax_scale=inputs["softmax_scale"],
                    ratio=ratio,
                )
                _loss = _kl_loss_from_dense_scores(
                    dense_attn_result["out"], dense_attn_result["denom"],
                    dense_idx_result["out"], dense_idx_result["denom"],
                    topk_indices_cmp, 0.1, False,
                )
            return out, _loss

        time_unfused = self._benchmark(run_unfused_forward_with_loss)

        speedup = time_unfused / max(time_fused, 1e-6)

        print(
            f"\n  [sq={sq}, b={b}, np={np_}, topk={indexer_topk}, "
            f"win={win_topk}, sparse_loss={sparse_loss}]"
        )
        print(f"    Fused (Triton):  {time_fused:.3f} ms")
        print(f"    Unfused (CSA):   {time_unfused:.3f} ms")
        print(f"    Speedup:         {speedup:.2f}x")

        # The fused path should not be catastrophically slower
        # (it may be slower at very small shapes due to kernel launch overhead)
        assert speedup > 0.3, (
            f"Fused path is too slow compared to unfused: {speedup:.2f}x"
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (512, 1, 16, 128, 128, 64, 4, 64, 64, 4),
            (1024, 1, 16, 128, 256, 128, 4, 64, 128, 4),
            (2048, 1, 8, 128, 512, 128, 4, 64, 256, 4),
        ],
        ids=[
            "sq512_topk64",
            "sq1024_topk128",
            "sq2048_topk256",
        ],
    )
    def test_performance_backward(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Measure backward pass performance for the fused path."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        # Make inputs require grad for backward
        query = inputs["query"].clone().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().requires_grad_(True)

        def run_fwd_bwd():
            out, loss = fused_indexer_sparse_attn(
                query, kv_full, inputs["attn_sink"], inputs["window_idxs"],
                inputs["q_indexer"], inputs["k_indexer"], inputs["weights"],
                inputs["indexer_topk"], inputs["ratio"],
                inputs["softmax_scale"], inputs["indexer_softmax_scale"],
                0.1, True, inputs["kv_offset"], False,
            )
            total = out.float().sum() + loss
            total.backward()
            # Zero grads for next iteration
            query.grad = None
            kv_full.grad = None

        time_bwd = self._benchmark(run_fwd_bwd, warmup=5, iters=20)

        print(
            f"\n  [sq={sq}, b={b}, np={np_}, topk={indexer_topk}, win={win_topk}]"
        )
        print(f"    Fwd+Bwd (Triton): {time_bwd:.3f} ms")

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (1024, 1, 16, 128, 256, 128, 4, 64, 128, 4),
        ],
    )
    def test_peak_memory(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device,
    ):
        """Measure peak GPU memory usage for fused vs unfused paths."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        # Measure fused path memory
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)
        _run_fused(inputs, sparse_loss=True)
        torch.cuda.synchronize()
        mem_fused_peak = torch.cuda.max_memory_allocated(device) - mem_before

        # Clear
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)
        _run_unfused_with_same_indices(inputs)
        torch.cuda.synchronize()
        mem_unfused_peak = torch.cuda.max_memory_allocated(device) - mem_before

        print(
            f"\n  [sq={sq}, b={b}, np={np_}, topk={indexer_topk}]"
        )
        print(f"    Fused peak memory:   {mem_fused_peak / 1024**2:.1f} MB")
        print(f"    Unfused peak memory: {mem_unfused_peak / 1024**2:.1f} MB")

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (512, 1, 16, 128, 128, 64, 4, 64, 64, 4),
            (512, 2, 16, 128, 256, 64, 4, 64, 128, 4),
            (1024, 1, 16, 128, 256, 128, 4, 64, 128, 4),
            (2048, 1, 8, 128, 512, 128, 4, 64, 256, 4),
        ],
        ids=[
            "sq512_b1_topk64",
            "sq512_b2_topk128",
            "sq1024_b1_topk128",
            "sq2048_b1_topk256",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_performance_end_to_end(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device,
    ):
        """End-to-end (forward + backward) performance: fused Triton vs unfused CSA."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        # --- Fused path (forward + backward) ---
        fused_query = inputs["query"].clone().requires_grad_(True)
        fused_kv = inputs["kv_full"].clone().requires_grad_(True)
        fused_sink = inputs["attn_sink"].clone().requires_grad_(True)

        def run_fused_e2e():
            out, loss = fused_indexer_sparse_attn(
                fused_query, fused_kv, fused_sink, inputs["window_idxs"],
                inputs["q_indexer"], inputs["k_indexer"], inputs["weights"],
                inputs["indexer_topk"], inputs["ratio"],
                inputs["softmax_scale"], inputs["indexer_softmax_scale"],
                0.1, sparse_loss, inputs["kv_offset"], False,
            )
            total = out.float().sum() + loss
            total.backward()
            fused_query.grad = None
            fused_kv.grad = None
            fused_sink.grad = None

        # --- Unfused path (forward + backward + indexer loss) ---
        # Fair comparison: unfused path also computes indexer loss (score_recompute
        # + KL divergence) and backpropagates through indexer parameters.
        unfused_query = inputs["query"].clone().requires_grad_(True)
        unfused_kv = inputs["kv_full"].clone().requires_grad_(True)
        unfused_sink = inputs["attn_sink"].clone().requires_grad_(True)
        unfused_q_indexer = inputs["q_indexer"].clone().requires_grad_(True)
        unfused_k_indexer = inputs["k_indexer"].clone().requires_grad_(True)
        unfused_weights = inputs["weights"].clone().requires_grad_(True)

        # Pre-compute indices (same as fused path would — not timed)
        with torch.no_grad():
            q_idx_bshd, k_idx_bsd, _, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
                inputs["q_indexer"], inputs["k_indexer"],
                inputs["weights"], inputs["indexer_softmax_scale"]
            )
            effective_topk = min(inputs["indexer_topk"], inputs["n_comp"])
            topk_indices_cmp, _, _ = _indexer_topk_bshd(
                q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, inputs["ratio"]
            )
            topk_indices_global = topk_indices_cmp.clone()
            valid_cmp = topk_indices_global >= 0
            topk_indices_global[valid_cmp] += kv_offset
            combined_idxs = torch.cat(
                [topk_indices_global, inputs["window_idxs"]], dim=-1
            )

        n_comp = inputs["n_comp"]
        idx_nh = inputs["q_indexer"].shape[2]
        np_ = inputs["np"]
        d = inputs["hn"]
        loss_coeff = 0.1

        def run_unfused_e2e():
            # 1. Unfused sparse attention (forward + backward)
            out = unfused_compressed_sparse_attn(
                unfused_query, unfused_kv, unfused_sink,
                combined_idxs, inputs["softmax_scale"],
            )

            # 2. Indexer loss computation (mirrors fused path step 6)
            # Transpose indexer inputs for score_recompute
            qi_bshd = unfused_q_indexer.permute(1, 0, 2, 3).contiguous()
            ki_bsd = unfused_k_indexer.permute(1, 0, 2).contiguous()
            w_raw = unfused_weights.permute(1, 0, 2).contiguous()
            w_scaled = w_raw * inputs["indexer_softmax_scale"]

            if sparse_loss:
                # Sparse indexer loss needs LSE from attention forward.
                # Recompute LSE inline from scores (same as unfused_compressed_sparse_attn).
                sq_val, _, np_val, hn_val = unfused_query.shape
                kv_t = unfused_kv.permute(1, 0, 2)  # (b, n_kv, hn)
                safe_idx = combined_idxs.clamp(min=0).long()
                safe_idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, -1, hn_val)
                kv_g = torch.gather(
                    kv_t.unsqueeze(1).expand(-1, sq_val, -1, -1), dim=2, index=safe_idx_exp
                ).float()
                q_perm = unfused_query.permute(1, 2, 0, 3).float()  # (b, np, sq, hn)
                scores = torch.einsum("bnsh,bskh->bnsk", q_perm, kv_g) * inputs["softmax_scale"]
                invalid_mask = (combined_idxs < 0).unsqueeze(1)
                scores = scores.masked_fill(invalid_mask, float("-inf"))
                sink_val = unfused_sink.view(1, np_val, 1, 1).float()
                scores_max = torch.max(scores.max(dim=-1, keepdim=True).values, sink_val)
                exp_scores = torch.exp(scores - scores_max)
                exp_sink = torch.exp(sink_val - scores_max)
                sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
                # lse: (b, np, sq, 1) -> (b, sq, np)
                lse_full = (scores_max + torch.log(sum_exp)).squeeze(-1)  # (b, np, sq)
                lse_bsh = lse_full.permute(0, 2, 1).contiguous()  # (b, sq, np)

                # Only need LSE from first effective_topk positions (indexer subset)
                # Recompute with only compressed indices for the partial LSE
                safe_cmp_global = (topk_indices_cmp.clone())
                valid_cmp_mask = safe_cmp_global >= 0
                safe_cmp_global[valid_cmp_mask] += kv_offset
                safe_cmp_idx = safe_cmp_global.clamp(min=0).long()
                safe_cmp_idx_exp = safe_cmp_idx.unsqueeze(-1).expand(-1, -1, -1, hn_val)
                kv_cmp = torch.gather(
                    kv_t.unsqueeze(1).expand(-1, sq_val, -1, -1), dim=2, index=safe_cmp_idx_exp
                ).float()
                scores_cmp = torch.einsum("bnsh,bskh->bnsk", q_perm, kv_cmp) * inputs["softmax_scale"]
                invalid_cmp_mask = (~valid_cmp_mask).unsqueeze(1)
                scores_cmp = scores_cmp.masked_fill(invalid_cmp_mask, float("-inf"))
                scores_cmp_max = torch.max(scores_cmp.max(dim=-1, keepdim=True).values, sink_val)
                exp_cmp = torch.exp(scores_cmp - scores_cmp_max)
                exp_sink_cmp = torch.exp(sink_val - scores_cmp_max)
                sum_exp_cmp = exp_cmp.sum(dim=-1, keepdim=True) + exp_sink_cmp
                lse_cmp = (scores_cmp_max + torch.log(sum_exp_cmp)).squeeze(-1)
                lse_indexer_bsh = lse_cmp.permute(0, 2, 1).contiguous()  # (b, sq, np)

                predict_result = sparse_indexer_score_recompute(
                    qi_bshd, ki_bsd, w_scaled, topk_indices_cmp,
                    qhead_per_kv_head=idx_nh,
                )
                predict = predict_result["predict"]

                q_attn_bshd = unfused_query.permute(1, 0, 2, 3).contiguous()
                k_attn_bsd = unfused_kv[:, :, :d].permute(1, 0, 2).contiguous()

                target_result = sparse_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_indexer_bsh,
                    topk_indices_cmp + kv_offset,
                    inputs["softmax_scale"], qhead_per_kv_head=np_,
                )
                target = target_result["target"]

                indexer_loss = _kl_loss_from_target_predict(
                    target, predict, topk_indices_cmp, loss_coeff, False
                )
            else:
                # Dense indexer loss
                q_attn_bshd = unfused_query.permute(1, 0, 2, 3).contiguous()
                k_attn_bsd = unfused_kv[kv_offset:kv_offset + n_comp, :, :d].permute(1, 0, 2).contiguous()

                # Need full LSE for dense path
                sq_val, _, np_val, hn_val = unfused_query.shape
                kv_t = unfused_kv.permute(1, 0, 2)
                safe_idx = combined_idxs.clamp(min=0).long()
                safe_idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, -1, hn_val)
                kv_g = torch.gather(
                    kv_t.unsqueeze(1).expand(-1, sq_val, -1, -1), dim=2, index=safe_idx_exp
                ).float()
                q_perm = unfused_query.permute(1, 2, 0, 3).float()
                scores = torch.einsum("bnsh,bskh->bnsk", q_perm, kv_g) * inputs["softmax_scale"]
                invalid_mask = (combined_idxs < 0).unsqueeze(1)
                scores = scores.masked_fill(invalid_mask, float("-inf"))
                sink_val = unfused_sink.view(1, np_val, 1, 1).float()
                scores_max = torch.max(scores.max(dim=-1, keepdim=True).values, sink_val)
                exp_scores = torch.exp(scores - scores_max)
                exp_sink = torch.exp(sink_val - scores_max)
                sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
                lse_full = (scores_max + torch.log(sum_exp)).squeeze(-1)  # (b, np, sq)
                lse_bsh = lse_full.permute(0, 2, 1).contiguous()  # (b, sq, np)

                dense_idx_result = dense_indexer_score_recompute(
                    qi_bshd, ki_bsd, w_scaled,
                    qhead_per_kv_head=idx_nh,
                    sm_scale=inputs["indexer_softmax_scale"],
                    ratio=inputs["ratio"],
                )
                index_score = dense_idx_result["out"]
                index_lse = dense_idx_result["denom"]

                dense_attn_result = dense_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_bsh,
                    qhead_per_kv_head=np_,
                    softmax_scale=inputs["softmax_scale"],
                    ratio=inputs["ratio"],
                )
                attn_score = dense_attn_result["out"]
                attn_l1norm = dense_attn_result["denom"]

                indexer_loss = _kl_loss_from_dense_scores(
                    attn_score, attn_l1norm, index_score, index_lse,
                    topk_indices_cmp, loss_coeff, False,
                )

            # 3. Combined backward
            total = out.float().sum() + indexer_loss
            total.backward()
            unfused_query.grad = None
            unfused_kv.grad = None
            unfused_sink.grad = None
            unfused_q_indexer.grad = None
            unfused_k_indexer.grad = None
            unfused_weights.grad = None

        time_fused = self._benchmark(run_fused_e2e, warmup=5, iters=20)
        time_unfused = self._benchmark(run_unfused_e2e, warmup=5, iters=20)
        speedup = time_unfused / max(time_fused, 1e-6)

        print(
            f"\n  [sq={sq}, b={b}, np={np_}, topk={indexer_topk}, "
            f"win={win_topk}, sparse_loss={sparse_loss}]"
        )
        print(f"    Fused E2E (Triton):  {time_fused:.3f} ms")
        print(f"    Unfused E2E (CSA):   {time_unfused:.3f} ms")
        print(f"    E2E Speedup:         {speedup:.2f}x")


# ---------------------------------------------------------------------------
# Module-level tests: CompressedSparseAttention unfused vs fused (triton)
# ---------------------------------------------------------------------------

from unittest.mock import patch

from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    _HAVE_HADAMARD = True
except ImportError:
    _HAVE_HADAMARD = False


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


def _make_test_mla_config(
    num_attention_heads=8,
    v_head_dim=128,
    hidden_size=256,
    csa_window_size=64,
    dsa_indexer_topk=64,
    dsa_indexer_n_heads=4,
    dsa_indexer_head_dim=64,
    dsa_indexer_loss_coeff=0.1,
    dsa_indexer_use_sparse_loss=True,
):
    """Helper to create MLATransformerConfig for module-level CSA tests."""
    qk_pos_emb_head_dim = 32
    return MLATransformerConfig(
        num_layers=4,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=[4, 4, 4, 4],
        csa_window_size=csa_window_size,
        csa_dense_mode=False,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
    )


def _make_test_compressor_submodules():
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressorSubmodules(
        linear_wkv=ModuleSpec(module=TELinear),
        linear_wgate=ModuleSpec(module=TELinear),
        norm=ModuleSpec(module=TENorm),
    )


def _make_test_csa_indexer_submodules():
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_weights_proj=ModuleSpec(module=TELinear),
        compressor=ModuleSpec(module=Compressor, submodules=_make_test_compressor_submodules()),
    )


def _make_test_csa_submodules():
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(module=Compressor, submodules=_make_test_compressor_submodules()),
        indexer=ModuleSpec(module=CSAIndexer, submodules=_make_test_csa_indexer_submodules()),
    )


def _build_csa_module(config, compress_ratio=4):
    """Build a CompressedSparseAttention module ready for testing."""
    from megatron.core.models.common.embeddings import RotaryEmbedding
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.enums import AttnMaskType

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    rotary_pos_emb = RotaryEmbedding(
        config.qk_pos_emb_head_dim,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rotary_base,
        cp_group=pg_collection.cp,
    )
    csa = CompressedSparseAttention(
        config=config,
        submodules=_make_test_csa_submodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type='self',
        pg_collection=pg_collection,
        rotary_pos_emb=rotary_pos_emb,
        compress_ratio=compress_ratio,
    ).cuda()
    return csa


def _make_csa_inputs(sq, b, config, device="cuda"):
    """Generate inputs matching CompressedSparseAttention.forward signature."""
    np_ = config.num_attention_heads
    hn = config.v_head_dim

    query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16, device=device)
    key = torch.randn(sq, b, 1, hn, dtype=torch.bfloat16, device=device)
    x = torch.randn(sq, b, config.hidden_size, dtype=torch.bfloat16, device=device)
    qr = torch.randn(sq, b, config.q_lora_rank, dtype=torch.bfloat16, device=device)

    return {"query": query, "key": key, "value": key.clone(), "x": x, "qr": qr}


# Patch target for fused_indexer_sparse_attn in the csa module
_PATCH_TARGET = (
    'megatron.core.transformer.experimental_attention_variant.csa.fused_indexer_sparse_attn'
)

# Reuse the already-imported triton fused_indexer_sparse_attn
_triton_fused_indexer_sparse_attn = fused_indexer_sparse_attn


def _hadamard_patches():
    """Context manager patches for hadamard transform if not installed."""
    if _HAVE_HADAMARD:
        from contextlib import nullcontext
        return nullcontext()
    else:
        from contextlib import ExitStack
        stack = ExitStack()
        # Enter patches immediately; ExitStack.__enter__ returns self and
        # __exit__ will undo all entered contexts.
        stack.enter_context(patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            _mock_hadamard_transform,
        ))
        stack.enter_context(patch(
            'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
            lambda x: x * (x.size(-1) ** -0.5),
        ))
        return stack


class TestCSAFusedVsUnfusedAccuracy:
    """Module-level accuracy: CompressedSparseAttention unfused path vs fused (triton).

    Instantiates the full CSA module and switches between _forward_unfused_csa
    and _forward_fused_indexer_training (with triton plugin intercepting the call).
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_class(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (64, 1, 8, 128, 16, 16),
            (128, 1, 8, 128, 32, 32),
            (256, 2, 8, 128, 64, 64),
        ],
        ids=["sq64_topk16", "sq128_topk32", "sq256_topk64"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_output_accuracy(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device,
    ):
        """Fused (triton) output matches unfused CSA path at module level."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                csa_window_size=window_size,
                dsa_indexer_topk=indexer_topk,
                dsa_indexer_loss_coeff=0.1,
                dsa_indexer_use_sparse_loss=sparse_loss,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=4)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused path ---
            csa.apply_dsa_kernel_fusion = False
            torch.manual_seed(7)
            out_unfused = csa(
                query=inputs["query"].clone(),
                key=inputs["key"].clone(),
                value=inputs["value"].clone(),
                attention_mask=None,
                x=inputs["x"].clone(),
                qr=inputs["qr"].clone(),
            )

            # --- Fused path (triton) ---
            csa.apply_dsa_kernel_fusion = True
            with patch(_PATCH_TARGET, _triton_fused_indexer_sparse_attn):
                torch.manual_seed(7)
                out_fused = csa(
                    query=inputs["query"].clone(),
                    key=inputs["key"].clone(),
                    value=inputs["value"].clone(),
                    attention_mask=None,
                    x=inputs["x"].clone(),
                    qr=inputs["qr"].clone(),
                )

            # Compare
            out_f = out_fused.float()
            out_u = out_unfused.float()

            cos_sim = torch.nn.functional.cosine_similarity(
                out_f.reshape(-1).unsqueeze(0),
                out_u.reshape(-1).unsqueeze(0),
            ).item()

            abs_diff = (out_f - out_u).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()

            print(
                f"\n  [sq={sq}, b={b}, np={num_heads}, topk={indexer_topk}, "
                f"sparse_loss={sparse_loss}]"
            )
            print(f"    cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")

            assert cos_sim > 0.95, (
                f"Module-level output mismatch: cos_sim={cos_sim:.6f}"
            )

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (128, 1, 8, 128, 32, 32),
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_accuracy(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device,
    ):
        """Gradients from fused (triton) path match unfused CSA path."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                csa_window_size=window_size,
                dsa_indexer_topk=indexer_topk,
                dsa_indexer_loss_coeff=0.1,
                dsa_indexer_use_sparse_loss=sparse_loss,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=4)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused path with grad ---
            csa.apply_dsa_kernel_fusion = False
            query_u = inputs["query"].clone().requires_grad_(True)
            key_u = inputs["key"].clone().requires_grad_(True)
            out_u = csa(
                query=query_u, key=key_u, value=key_u.clone(),
                attention_mask=None, x=inputs["x"].clone(), qr=inputs["qr"].clone(),
            )
            out_u.sum().backward()
            grad_q_unfused = query_u.grad.float().clone()
            grad_k_unfused = key_u.grad.float().clone()

            csa.zero_grad()

            # --- Fused path with grad ---
            csa.apply_dsa_kernel_fusion = True
            query_f = inputs["query"].clone().requires_grad_(True)
            key_f = inputs["key"].clone().requires_grad_(True)
            with patch(_PATCH_TARGET, _triton_fused_indexer_sparse_attn):
                out_f = csa(
                    query=query_f, key=key_f, value=key_f.clone(),
                    attention_mask=None, x=inputs["x"].clone(), qr=inputs["qr"].clone(),
                )
                out_f.sum().backward()
            grad_q_fused = query_f.grad.float().clone()
            grad_k_fused = key_f.grad.float().clone()

            # Compare grad_query
            cos_q = torch.nn.functional.cosine_similarity(
                grad_q_fused.reshape(-1).unsqueeze(0),
                grad_q_unfused.reshape(-1).unsqueeze(0),
            ).item()
            # Compare grad_key
            cos_k = torch.nn.functional.cosine_similarity(
                grad_k_fused.reshape(-1).unsqueeze(0),
                grad_k_unfused.reshape(-1).unsqueeze(0),
            ).item()

            print(
                f"\n  grad_query cos_sim={cos_q:.6f}, "
                f"grad_key cos_sim={cos_k:.6f}"
            )

            assert cos_q > 0.90, f"grad_query cos_sim too low: {cos_q:.6f}"
            assert cos_k > 0.90, f"grad_key cos_sim too low: {cos_k:.6f}"


class TestCSAFusedVsUnfusedPerformance:
    """Module-level performance: CompressedSparseAttention unfused vs fused (triton).

    End-to-end forward+backward timing comparison using the full module pipeline.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_class(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @staticmethod
    def _benchmark(fn, warmup: int = 5, iters: int = 20) -> float:
        """Benchmark a CUDA function. Returns median elapsed ms."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times.sort()
        return times[len(times) // 2]

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (512, 1, 16, 128, 64, 64),
            (1024, 1, 16, 128, 128, 128),
            (2048, 1, 8, 128, 128, 256),
        ],
        ids=["sq512_topk64", "sq1024_topk128", "sq2048_topk256"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_e2e_performance(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device,
    ):
        """End-to-end (forward + backward) module-level performance comparison."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                hidden_size=256,
                csa_window_size=window_size,
                dsa_indexer_topk=indexer_topk,
                dsa_indexer_loss_coeff=0.1,
                dsa_indexer_use_sparse_loss=sparse_loss,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=4)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused benchmark ---
            csa.apply_dsa_kernel_fusion = False
            query_u = inputs["query"].clone().requires_grad_(True)
            key_u = inputs["key"].clone().requires_grad_(True)

            def run_unfused():
                out = csa(
                    query=query_u, key=key_u, value=key_u.clone(),
                    attention_mask=None, x=inputs["x"], qr=inputs["qr"],
                )
                out.sum().backward()
                query_u.grad = None
                key_u.grad = None
                csa.zero_grad()

            time_unfused = self._benchmark(run_unfused)

            # --- Fused benchmark (triton) ---
            csa.apply_dsa_kernel_fusion = True
            query_f = inputs["query"].clone().requires_grad_(True)
            key_f = inputs["key"].clone().requires_grad_(True)

            def run_fused():
                out = csa(
                    query=query_f, key=key_f, value=key_f.clone(),
                    attention_mask=None, x=inputs["x"], qr=inputs["qr"],
                )
                out.sum().backward()
                query_f.grad = None
                key_f.grad = None
                csa.zero_grad()

            with patch(_PATCH_TARGET, _triton_fused_indexer_sparse_attn):
                time_fused = self._benchmark(run_fused)

            speedup = time_unfused / max(time_fused, 1e-6)

            print(
                f"\n  [sq={sq}, b={b}, np={num_heads}, topk={indexer_topk}, "
                f"win={window_size}, sparse_loss={sparse_loss}]"
            )
            print(f"    Unfused CSA E2E:    {time_unfused:.3f} ms")
            print(f"    Fused Triton E2E:   {time_fused:.3f} ms")
            print(f"    Speedup:            {speedup:.2f}x")

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (512, 1, 16, 128, 64, 64),
            (1024, 1, 16, 128, 128, 128),
            (2048, 1, 8, 128, 128, 256),
        ],
        ids=["sq512_topk64", "sq1024_topk128", "sq2048_topk256"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_e2e_memory(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device,
    ):
        """Peak GPU memory comparison: fused vs unfused E2E (forward + backward)."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                hidden_size=256,
                csa_window_size=window_size,
                dsa_indexer_topk=indexer_topk,
                dsa_indexer_loss_coeff=0.1,
                dsa_indexer_use_sparse_loss=sparse_loss,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=4)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused memory measurement ---
            csa.apply_dsa_kernel_fusion = False
            query_u = inputs["query"].clone().requires_grad_(True)
            key_u = inputs["key"].clone().requires_grad_(True)

            # Warmup
            out = csa(
                query=query_u, key=key_u, value=key_u.clone(),
                attention_mask=None, x=inputs["x"], qr=inputs["qr"],
            )
            out.sum().backward()
            query_u.grad = None
            key_u.grad = None
            csa.zero_grad()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            mem_before_unfused = torch.cuda.memory_allocated(device)

            out = csa(
                query=query_u, key=key_u, value=key_u.clone(),
                attention_mask=None, x=inputs["x"], qr=inputs["qr"],
            )
            out.sum().backward()
            torch.cuda.synchronize()
            mem_unfused_peak = torch.cuda.max_memory_allocated(device) - mem_before_unfused

            query_u.grad = None
            key_u.grad = None
            csa.zero_grad()
            torch.cuda.empty_cache()

            # --- Fused memory measurement ---
            csa.apply_dsa_kernel_fusion = True
            query_f = inputs["query"].clone().requires_grad_(True)
            key_f = inputs["key"].clone().requires_grad_(True)

            # Warmup
            with patch(_PATCH_TARGET, _triton_fused_indexer_sparse_attn):
                out = csa(
                    query=query_f, key=key_f, value=key_f.clone(),
                    attention_mask=None, x=inputs["x"], qr=inputs["qr"],
                )
                out.sum().backward()
            query_f.grad = None
            key_f.grad = None
            csa.zero_grad()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            mem_before_fused = torch.cuda.memory_allocated(device)

            with patch(_PATCH_TARGET, _triton_fused_indexer_sparse_attn):
                out = csa(
                    query=query_f, key=key_f, value=key_f.clone(),
                    attention_mask=None, x=inputs["x"], qr=inputs["qr"],
                )
                out.sum().backward()
            torch.cuda.synchronize()
            mem_fused_peak = torch.cuda.max_memory_allocated(device) - mem_before_fused

            ratio = mem_unfused_peak / max(mem_fused_peak, 1)

            print(
                f"\n  [sq={sq}, b={b}, np={num_heads}, topk={indexer_topk}, "
                f"win={window_size}, sparse_loss={sparse_loss}]"
            )
            print(f"    Unfused peak memory: {mem_unfused_peak / 1024**2:.1f} MB")
            print(f"    Fused peak memory:   {mem_fused_peak / 1024**2:.1f} MB")
            print(f"    Memory ratio:        {ratio:.2f}x")
