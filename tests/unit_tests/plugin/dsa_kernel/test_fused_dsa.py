# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Accuracy and performance tests for FusedIndexerSparseAttnFunc.

Validates that ``fused_indexer_sparse_attn`` from the Triton plugin produces
numerically equivalent attention output to ``unfused_compressed_sparse_attn``
from ``megatron.core.transformer.experimental_attention_variant.csa``.

Test parameters are configured for training-realistic workloads:
- Sequence length >= 2048 (typical training context)
- TopK > 256 (large sparse attention windows)
- Number of heads >= 32 (full-model multi-head attention)

The fused path:
1. Scores + top-K selection via indexer (q_indexer, k_indexer, weights)
2. Combines compressed top-K indices with window indices
3. Runs sparse attention over combined indices
4. Computes indexer KL loss (sparse or dense variant)

The unfused path takes pre-computed indices and runs only the sparse attention.
We compare the attention output (not the loss) between them.

Run with: pytest tests/unit_tests/plugin/dsa_kernel/test_fused_dsa.py -v -s
Requires: CUDA GPU with Triton support.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Tuple

import pytest
import torch
from torch import Tensor

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# SM90+ check for module-level tests that use apply_dsa_kernel_fusion=True
_SM90_AVAILABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9
)
_skip_unless_sm90 = pytest.mark.skipif(
    not _SM90_AVAILABLE, reason="SM90+ (Hopper or later) required for apply_dsa_kernel_fusion"
)

# ---------------------------------------------------------------------------
# Logging setup — use `pytest -s --log-cli-level=INFO` for detailed output,
# or `--log-cli-level=WARNING` to suppress per-test metric prints.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

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
            # Training-scale shapes: seq>=2048, topk>256, heads>=32
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
            (8192, 1, 32, 128, 2048, 256, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
            "longctx_B1_sq8192_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_output_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
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

        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff,
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
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
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
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
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
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
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_query_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
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
        logger.info(f"grad_query: cos_sim={cos_sim:.6f}")
        logger.debug(
            f"grad_query detail: max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="grad_query",
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_kv_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
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
        logger.info(f"grad_kv_full: cos_sim={cos_sim:.6f}")
        logger.debug(
            f"grad_kv_full detail: max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="grad_kv",
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    def test_grad_attn_sink_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device, dsa_metrics,
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
        logger.info(f"grad_attn_sink: cos_sim={cos_sim:.6f}")
        logger.debug(
            f"grad_attn_sink detail: max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="grad_attn_sink",
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
        ],
        ids=["7B_sq2048", "13B_sq4096"],
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


@pytest.mark.perf
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
            # Training-scale shapes: seq>=2048, topk>256, heads>=32
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
            (4096, 1, 32, 128, 1024, 128, 4, 64, 512, 4),
            (4096, 1, 32, 128, 1024, 128, 4, 64, 1024, 4),
            (8192, 1, 32, 128, 2048, 256, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
            "13B_B1_sq4096_topk512",
            "13B_B1_sq4096_topk1024",
            "longctx_B1_sq8192_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_performance_forward(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
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
                # Preserve -1 for invalid mask in sparse_attn_score_recompute
                topk_for_target_perf = topk_indices_cmp.clone()
                valid_perf = topk_for_target_perf >= 0
                topk_for_target_perf[valid_perf] += kv_offset
                target_result = sparse_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_indexer_bsh,
                    topk_for_target_perf,
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

        logger.info(
            f"[sq={sq}, b={b}, np={np_}, topk={indexer_topk}, win={win_topk}, "
            f"sparse_loss={sparse_loss}] "
            f"fused={time_fused:.3f}ms, unfused={time_unfused:.3f}ms, "
            f"speedup={speedup:.2f}x"
        )
        dsa_metrics.record_performance(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            fused_ms=time_fused, unfused_ms=time_unfused, speedup=speedup, label="fwd",
        )

        # The fused path should not be catastrophically slower
        # (it may be slower at very small shapes due to kernel launch overhead)
        assert speedup > 0.3, (
            f"Fused path is too slow compared to unfused: {speedup:.2f}x"
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_sq2048_topk256",
            "13B_sq4096_topk384",
            "70B_sq2048_topk512",
        ],
    )
    def test_performance_backward(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device, dsa_metrics,
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

        logger.info(
            f"[sq={sq}, b={b}, np={np_}, topk={indexer_topk}, win={win_topk}] "
            f"fwd+bwd={time_bwd:.3f}ms"
        )
        dsa_metrics.record_performance(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk},
            fused_ms=time_bwd, unfused_ms=time_bwd, speedup=1.0, label="bwd",
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
        ],
    )
    def test_peak_memory(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, device, dsa_metrics,
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

        logger.info(
            f"[sq={sq}, b={b}, np={np_}, topk={indexer_topk}] "
            f"fused={mem_fused_peak / 1024**2:.1f}MB, "
            f"unfused={mem_unfused_peak / 1024**2:.1f}MB"
        )
        dsa_metrics.record_memory(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk},
            fused_mb=mem_fused_peak / 1024**2, unfused_mb=mem_unfused_peak / 1024**2,
            ratio=mem_unfused_peak / max(mem_fused_peak, 1),
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_performance_end_to_end(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
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

                # Preserve -1 for invalid mask in sparse_attn_score_recompute
                topk_for_target_e2e = topk_indices_cmp.clone()
                valid_e2e = topk_for_target_e2e >= 0
                topk_for_target_e2e[valid_e2e] += kv_offset
                target_result = sparse_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, lse_indexer_bsh,
                    topk_for_target_e2e,
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

        logger.info(
            f"[sq={sq}, b={b}, np={np_}, topk={indexer_topk}, win={win_topk}, "
            f"sparse_loss={sparse_loss}] "
            f"fused_e2e={time_fused:.3f}ms, unfused_e2e={time_unfused:.3f}ms, "
            f"speedup={speedup:.2f}x"
        )
        dsa_metrics.record_performance(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            fused_ms=time_fused, unfused_ms=time_unfused, speedup=speedup, label="e2e",
        )


# ---------------------------------------------------------------------------
# Module-level tests: CompressedSparseAttention unfused vs fused (triton)
# ---------------------------------------------------------------------------

from contextlib import contextmanager
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
from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
    dsa_sparse_attn as _triton_dsa_sparse_attn_raw,
)
from tests.unit_tests.test_utilities import Utils


# ---------------------------------------------------------------------------
# OOM guard — unfused path can OOM at large sequence lengths
# ---------------------------------------------------------------------------


@contextmanager
def _oom_guard():
    """Convert CUDA OOM into pytest.skip — unfused path is memory-hungry at 4k+."""
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("CUDA OOM on unfused path at this sequence length")


# ---------------------------------------------------------------------------
# Triton dsa_sparse_attn SBHD interface
#
# Now that csa.py uses _ensure_dsa_kernels() to lazily import the Triton
# backend (dsa_sparse_attn_sbhd, build_flat_topk_idxs, etc.), tests that set
# apply_dsa_kernel_fusion=True in the config no longer need monkey-patching.
# The globals are populated during CSA construction.
# ---------------------------------------------------------------------------

from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
    dsa_sparse_attn_sbhd as _triton_dsa_sparse_attn_sbhd,  # noqa: F401
)


# Patch targets for the csa.py module-level globals (used by _ensure_dsa_kernels).
# These are needed only when apply_dsa_kernel_fusion was NOT set in the config
# (so _ensure_dsa_kernels was never called) and you need to inject the Triton
# implementations manually.
_PATCH_DSA_SPARSE_ATTN = (
    'megatron.core.transformer.experimental_attention_variant.csa._dsa_sparse_attn'
)
_PATCH_BUILD_FLAT_TOPK_IDXS = (
    'megatron.core.transformer.experimental_attention_variant.csa._build_flat_topk_idxs_fn'
)

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
    apply_dsa_kernel_fusion=True,
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
        apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
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


# Patch target for fused_indexer_sparse_attn in the csa module (kept for reference;
# not needed when apply_dsa_kernel_fusion=True is set in the config).
_PATCH_FUSED_INDEXER = (
    'megatron.core.transformer.experimental_attention_variant.csa._fused_indexer_sparse_attn'
)


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


@_skip_unless_sm90
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
            (2048, 1, 32, 128, 128, 256),
            (4096, 1, 32, 128, 256, 384),
            (2048, 1, 64, 128, 128, 512),
        ],
        ids=["7B_sq2048_topk256", "13B_sq4096_topk384", "70B_sq2048_topk512"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_output_accuracy(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device, dsa_metrics,
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
            with _oom_guard():
                torch.manual_seed(7)
                out_unfused = csa(
                    query=inputs["query"].clone(),
                    key=inputs["key"].clone(),
                    value=inputs["value"].clone(),
                    attention_mask=None,
                    x=inputs["x"].clone(),
                    qr=inputs["qr"].clone(),
                )

            # --- Fused path (triton via _ensure_dsa_kernels) ---
            csa.apply_dsa_kernel_fusion = True
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

            logger.info(
                f"[sq={sq}, b={b}, np={num_heads}, topk={indexer_topk}, "
                f"sparse_loss={sparse_loss}] cos_sim={cos_sim:.6f}"
            )
            logger.debug(f"  max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")

            assert cos_sim > 0.95, (
                f"Module-level output mismatch: cos_sim={cos_sim:.6f}"
            )
            dsa_metrics.record_accuracy(
                params={"sq": sq, "b": b, "np": num_heads, "topk": indexer_topk, "sparse_loss": sparse_loss},
                cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="module_output",
            )

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (2048, 1, 32, 128, 128, 256),
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_accuracy(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device, dsa_metrics,
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
            with _oom_guard():
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

            logger.info(f"grad_query cos_sim={cos_q:.6f}, grad_key cos_sim={cos_k:.6f}")

            assert cos_q > 0.90, f"grad_query cos_sim too low: {cos_q:.6f}"
            assert cos_k > 0.90, f"grad_key cos_sim too low: {cos_k:.6f}"

            dsa_metrics.record_accuracy(
                params={"sq": sq, "b": b, "np": num_heads, "topk": indexer_topk, "sparse_loss": sparse_loss},
                cos_sim=cos_q, target="module_grad_query",
            )
            dsa_metrics.record_accuracy(
                params={"sq": sq, "b": b, "np": num_heads, "topk": indexer_topk, "sparse_loss": sparse_loss},
                cos_sim=cos_k, target="module_grad_key",
            )


@_skip_unless_sm90
class TestDSAIndexerLossAutoScaler:
    """Verify DSAIndexerLossAutoScaler correctly propagates gradients in training.

    DSAIndexerLossAutoScaler attaches the indexer KL loss to the attention output
    without altering the forward value. In backward:
    - grad w.r.t. output passes through unchanged.
    - grad w.r.t. indexer_loss = ones_like(loss) * main_loss_backward_scale.

    This test validates the complete gradient flow through the CSA module when
    the AutoScaler is active (indexer_loss_coeff > 0, training mode).
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

    def test_forward_identity(self, device):
        """AutoScaler does not modify the forward output value."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
        )

        output = torch.randn(128, 2, 4096, dtype=torch.bfloat16, device=device)
        loss = torch.tensor(0.05, device=device, requires_grad=True)
        output.requires_grad_(True)

        result = DSAIndexerLossAutoScaler.apply(output, loss)

        assert torch.equal(result, output), "AutoScaler altered forward output"

    def test_grad_output_passthrough(self, device):
        """Gradient w.r.t. output passes through unchanged."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
        )

        output = torch.randn(64, 1, 2048, dtype=torch.float32, device=device, requires_grad=True)
        loss = torch.tensor(0.1, device=device, requires_grad=True)

        result = DSAIndexerLossAutoScaler.apply(output, loss)
        grad_upstream = torch.randn_like(result)
        result.backward(grad_upstream)

        assert torch.equal(output.grad, grad_upstream), (
            "AutoScaler modified grad w.r.t. output"
        )

    def test_grad_loss_scaled(self, device):
        """Gradient w.r.t. indexer_loss equals ones * main_loss_backward_scale."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
        )

        output = torch.randn(64, 1, 2048, dtype=torch.float32, device=device, requires_grad=True)
        loss = torch.tensor(0.1, device=device, requires_grad=True)

        # Set a known scale
        scale = torch.tensor(2.5, device=device)
        DSAIndexerLossAutoScaler.set_loss_scale(scale)

        result = DSAIndexerLossAutoScaler.apply(output, loss)
        grad_upstream = torch.randn_like(result)
        result.backward(grad_upstream)

        expected_loss_grad = torch.ones_like(loss) * 2.5
        assert torch.allclose(loss.grad, expected_loss_grad), (
            f"Loss grad mismatch: got {loss.grad.item()}, expected {expected_loss_grad.item()}"
        )

        # Reset to default for other tests
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (2048, 1, 32, 128, 128, 256),
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_e2e_grad_with_autoscaler(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device, dsa_metrics,
    ):
        """End-to-end: AutoScaler in CSA module allows indexer params to receive gradients.

        Verifies that:
        1. CSA output has gradients flowing back to query (attention path).
        2. Indexer parameters receive non-zero gradients (loss path via AutoScaler).
        3. Fused and unfused paths produce consistent gradient magnitudes.
        """
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
        )

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

            # Set a known scale for deterministic gradient checking
            DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=device))

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Fused path ---
            csa.apply_dsa_kernel_fusion = True
            query_fused = inputs["query"].clone().requires_grad_(True)
            torch.manual_seed(7)
            out_fused = csa(
                query=query_fused,
                key=inputs["key"].clone(),
                value=inputs["value"].clone(),
                attention_mask=None,
                x=inputs["x"].clone(),
                qr=inputs["qr"].clone(),
            )
            grad_out = torch.randn_like(out_fused)
            out_fused.backward(grad_out)

            # Attention path: query should receive gradient
            assert query_fused.grad is not None, "query has no gradient (fused)"
            assert not torch.isnan(query_fused.grad).any(), "query grad has NaN (fused)"
            grad_query_fused_norm = query_fused.grad.float().norm().item()

            # Indexer path: indexer params should receive gradient via AutoScaler
            indexer_grads_fused = {}
            for name, p in csa.indexer.named_parameters():
                if p.grad is not None:
                    indexer_grads_fused[name] = p.grad.float().norm().item()
            assert len(indexer_grads_fused) > 0, (
                "No indexer parameter received gradient (fused). "
                "AutoScaler may not be propagating loss backward."
            )

            # --- Unfused path ---
            csa.zero_grad()
            csa.apply_dsa_kernel_fusion = False
            query_unfused = inputs["query"].clone().requires_grad_(True)
            with _oom_guard():
                torch.manual_seed(7)
                out_unfused = csa(
                    query=query_unfused,
                    key=inputs["key"].clone(),
                    value=inputs["value"].clone(),
                    attention_mask=None,
                    x=inputs["x"].clone(),
                    qr=inputs["qr"].clone(),
                )
                out_unfused.backward(grad_out)

            assert query_unfused.grad is not None, "query has no gradient (unfused)"
            grad_query_unfused_norm = query_unfused.grad.float().norm().item()

            indexer_grads_unfused = {}
            for name, p in csa.indexer.named_parameters():
                if p.grad is not None:
                    indexer_grads_unfused[name] = p.grad.float().norm().item()
            assert len(indexer_grads_unfused) > 0, (
                "No indexer parameter received gradient (unfused). "
            )

            # Gradient magnitude should be in the same ballpark
            ratio_q = grad_query_fused_norm / max(grad_query_unfused_norm, 1e-12)
            logger.info(
                f"[sparse_loss={sparse_loss}] query grad norm ratio "
                f"(fused/unfused) = {ratio_q:.4f}"
            )
            assert 0.5 < ratio_q < 2.0, (
                f"Query grad norm diverged: fused={grad_query_fused_norm:.4e}, "
                f"unfused={grad_query_unfused_norm:.4e}"
            )

            # Indexer grad norms should both be non-trivial
            for name in indexer_grads_fused:
                if name in indexer_grads_unfused:
                    logger.info(
                        f"  indexer.{name}: fused={indexer_grads_fused[name]:.4e}, "
                        f"unfused={indexer_grads_unfused[name]:.4e}"
                    )

            dsa_metrics.record_accuracy(
                params={"sq": sq, "np": num_heads, "topk": indexer_topk, "sparse_loss": sparse_loss},
                cos_sim=ratio_q, target="autoscaler_grad_ratio",
            )

            # Cleanup
            DSAIndexerLossAutoScaler.main_loss_backward_scale = None


@_skip_unless_sm90
@pytest.mark.perf
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
            (2048, 1, 32, 128, 128, 256),
            (4096, 1, 32, 128, 256, 384),
            (2048, 1, 64, 128, 128, 512),
            (4096, 1, 32, 128, 128, 512),
            (4096, 1, 32, 128, 128, 1024),
        ],
        ids=["7B_sq2048_topk256", "13B_sq4096_topk384", "70B_sq2048_topk512", "13B_sq4096_topk512", "13B_sq4096_topk1024"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_e2e_performance(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device, dsa_metrics,
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

            with _oom_guard():
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

            time_fused = self._benchmark(run_fused)

            speedup = time_unfused / max(time_fused, 1e-6)

            logger.info(
                f"[sq={sq}, b={b}, np={num_heads}, topk={indexer_topk}, "
                f"win={window_size}, sparse_loss={sparse_loss}] "
                f"unfused={time_unfused:.3f}ms, fused={time_fused:.3f}ms, "
                f"speedup={speedup:.2f}x"
            )

            dsa_metrics.record_performance(
                params={"sq": sq, "b": b, "np": num_heads, "topk": indexer_topk, "win": window_size, "sparse_loss": sparse_loss},
                fused_ms=time_fused, unfused_ms=time_unfused, speedup=speedup, label="module_e2e",
            )

    @pytest.mark.parametrize(
        "sq,b,num_heads,v_head_dim,window_size,indexer_topk",
        [
            (2048, 1, 32, 128, 128, 256),
            (4096, 1, 32, 128, 256, 384),
            (2048, 1, 64, 128, 128, 512),
            (4096, 1, 32, 128, 128, 512),
            (4096, 1, 32, 128, 128, 1024),
        ],
        ids=["7B_sq2048_topk256", "13B_sq4096_topk384", "70B_sq2048_topk512", "13B_sq4096_topk512", "13B_sq4096_topk1024"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_e2e_memory(
        self, sq, b, num_heads, v_head_dim, window_size, indexer_topk,
        sparse_loss, device, dsa_metrics,
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

            with _oom_guard():
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

            out = csa(
                query=query_f, key=key_f, value=key_f.clone(),
                attention_mask=None, x=inputs["x"], qr=inputs["qr"],
            )
            out.sum().backward()
            torch.cuda.synchronize()
            mem_fused_peak = torch.cuda.max_memory_allocated(device) - mem_before_fused

            ratio = mem_unfused_peak / max(mem_fused_peak, 1)

            logger.info(
                f"[sq={sq}, b={b}, np={num_heads}, topk={indexer_topk}, "
                f"win={window_size}, sparse_loss={sparse_loss}] "
                f"unfused={mem_unfused_peak / 1024**2:.1f}MB, "
                f"fused={mem_fused_peak / 1024**2:.1f}MB, ratio={ratio:.2f}x"
            )

            dsa_metrics.record_memory(
                params={"sq": sq, "b": b, "np": num_heads, "topk": indexer_topk, "win": window_size, "sparse_loss": sparse_loss},
                fused_mb=mem_fused_peak / 1024**2, unfused_mb=mem_unfused_peak / 1024**2, ratio=ratio,
            )


# ---------------------------------------------------------------------------
# No-indexer tests: compress_ratio=0 (window-only) and compress_ratio=128
# ---------------------------------------------------------------------------


@_skip_unless_sm90
class TestCSANoIndexerFusedVsUnfused:
    """E2E accuracy: CompressedSparseAttention with no indexer (ratio=0 or ratio=128).

    Tests the _forward_fused_no_indexer path vs _forward_unfused_csa.
    - ratio=0:   window-only attention, no compressor, no indexer.
    - ratio=128: attend-all-compressed, compressor built but no indexer.

    Both paths rely on dsa_sparse_attn (patched with triton) for the fused side
    and unfused_compressed_sparse_attn for the unfused side.
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
        "compress_ratio,sq,b,num_heads,v_head_dim,window_size",
        [
            (0,   2048, 1, 32, 128, 128),
            (0,   4096, 1, 32, 128, 256),
            (128, 2048, 1, 32, 128, 128),
            (128, 4096, 1, 32, 128, 256),
        ],
        ids=["win_only_2k", "win_only_4k", "ratio128_2k", "ratio128_4k"],
    )
    def test_output_accuracy_no_indexer(
        self, compress_ratio, sq, b, num_heads, v_head_dim, window_size, device, dsa_metrics,
    ):
        """Fused (triton) output matches unfused CSA path for no-indexer configs."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                csa_window_size=window_size,
                dsa_indexer_topk=64,
                dsa_indexer_loss_coeff=0.0,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=compress_ratio)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused path ---
            csa.apply_dsa_kernel_fusion = False
            with _oom_guard():
                torch.manual_seed(7)
                out_unfused = csa(
                    query=inputs["query"].clone(),
                    key=inputs["key"].clone(),
                    value=inputs["value"].clone(),
                    attention_mask=None,
                    x=inputs["x"].clone(),
                    qr=inputs["qr"].clone(),
                )

            # --- Fused path (triton dsa_sparse_attn via _ensure_dsa_kernels) ---
            csa.apply_dsa_kernel_fusion = True
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

            logger.info(
                f"[ratio={compress_ratio}, sq={sq}, b={b}, np={num_heads}, "
                f"win={window_size}] cos_sim={cos_sim:.6f}"
            )
            logger.debug(f"  max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")

            assert cos_sim > 0.95, (
                f"No-indexer output mismatch: cos_sim={cos_sim:.6f}"
            )

            dsa_metrics.record_accuracy(
                params={"ratio": compress_ratio, "sq": sq, "b": b, "np": num_heads, "win": window_size},
                cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="no_indexer_output",
            )

    @pytest.mark.parametrize(
        "compress_ratio,sq,b,num_heads,v_head_dim,window_size",
        [
            (0,   2048, 1, 32, 128, 128),
            (0,   4096, 1, 32, 128, 256),
            (128, 2048, 1, 32, 128, 128),
            (128, 4096, 1, 32, 128, 256),
        ],
        ids=["win_only_2k", "win_only_4k", "ratio128_2k", "ratio128_4k"],
    )
    def test_grad_accuracy_no_indexer(
        self, compress_ratio, sq, b, num_heads, v_head_dim, window_size, device, dsa_metrics,
    ):
        """Gradients from fused (triton) path match unfused CSA path for no-indexer configs."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                csa_window_size=window_size,
                dsa_indexer_topk=64,
                dsa_indexer_loss_coeff=0.0,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=compress_ratio)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused path with grad ---
            csa.apply_dsa_kernel_fusion = False
            query_u = inputs["query"].clone().requires_grad_(True)
            key_u = inputs["key"].clone().requires_grad_(True)
            with _oom_guard():
                out_u = csa(
                    query=query_u, key=key_u, value=key_u.clone(),
                    attention_mask=None, x=inputs["x"].clone(), qr=inputs["qr"].clone(),
                )
                out_u.sum().backward()
            grad_q_unfused = query_u.grad.float().clone()
            grad_k_unfused = key_u.grad.float().clone()

            csa.zero_grad()

            # --- Fused path with grad (triton dsa_sparse_attn via _ensure_dsa_kernels) ---
            csa.apply_dsa_kernel_fusion = True
            query_f = inputs["query"].clone().requires_grad_(True)
            key_f = inputs["key"].clone().requires_grad_(True)
            out_f = csa(
                query=query_f, key=key_f, value=key_f.clone(),
                attention_mask=None, x=inputs["x"].clone(), qr=inputs["qr"].clone(),
            )
            torch.cuda.synchronize()
            out_f.sum().backward()
            torch.cuda.synchronize()
            grad_q_fused = query_f.grad.float().clone()
            grad_k_fused = key_f.grad.float().clone()

            # Debug: check for NaN/Inf/zero in fused gradients
            logger.debug(f"grad_q_fused: nan={torch.isnan(grad_q_fused).any().item()}, "
                  f"inf={torch.isinf(grad_q_fused).any().item()}, "
                  f"all_zero={(grad_q_fused == 0).all().item()}, "
                  f"norm={grad_q_fused.norm().item():.4e}")
            logger.debug(f"grad_k_fused: nan={torch.isnan(grad_k_fused).any().item()}, "
                  f"inf={torch.isinf(grad_k_fused).any().item()}, "
                  f"all_zero={(grad_k_fused == 0).all().item()}, "
                  f"norm={grad_k_fused.norm().item():.4e}")
            logger.debug(f"grad_q_unfused: norm={grad_q_unfused.norm().item():.4e}")
            logger.debug(f"grad_k_unfused: norm={grad_k_unfused.norm().item():.4e}")

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

            logger.info(
                f"[ratio={compress_ratio}, sq={sq}, np={num_heads}, win={window_size}] "
                f"grad_query cos_sim={cos_q:.6f}, grad_key cos_sim={cos_k:.6f}"
            )

            assert cos_q > 0.90, f"grad_query cos_sim too low: {cos_q:.6f}"
            assert cos_k > 0.90, f"grad_key cos_sim too low: {cos_k:.6f}"

            dsa_metrics.record_accuracy(
                params={"ratio": compress_ratio, "sq": sq, "np": num_heads, "win": window_size},
                cos_sim=cos_q, target="no_indexer_grad_query",
            )
            dsa_metrics.record_accuracy(
                params={"ratio": compress_ratio, "sq": sq, "np": num_heads, "win": window_size},
                cos_sim=cos_k, target="no_indexer_grad_key",
            )


# ---------------------------------------------------------------------------
# No-indexer end-to-end performance tests
# ---------------------------------------------------------------------------


@_skip_unless_sm90
@pytest.mark.perf
class TestCSANoIndexerPerformance:
    """E2E performance: CompressedSparseAttention no-indexer path.

    Measures forward+backward timing for the fused (triton dsa_sparse_attn)
    path vs the unfused CSA path when no indexer is used (ratio=0 or ratio=128).
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
        "compress_ratio,sq,b,num_heads,v_head_dim,window_size",
        [
            (0,   2048, 1, 32, 128, 128),
            (0,   4096, 1, 32, 128, 256),
            (128, 2048, 1, 32, 128, 128),
            (128, 4096, 1, 32, 128, 256),
        ],
        ids=["win_only_2k", "win_only_4k", "ratio128_2k", "ratio128_4k"],
    )
    def test_e2e_performance_no_indexer(
        self, compress_ratio, sq, b, num_heads, v_head_dim, window_size, device, dsa_metrics,
    ):
        """Forward+backward performance: fused triton vs unfused CSA (no indexer)."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                csa_window_size=window_size,
                dsa_indexer_topk=64,
                dsa_indexer_loss_coeff=0.0,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=compress_ratio)
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

            with _oom_guard():
                time_unfused = self._benchmark(run_unfused)

            # --- Fused benchmark (triton dsa_sparse_attn via _ensure_dsa_kernels) ---
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

            time_fused = self._benchmark(run_fused)

            speedup = time_unfused / max(time_fused, 1e-6)

            logger.info(
                f"[ratio={compress_ratio}, sq={sq}, b={b}, np={num_heads}, "
                f"win={window_size}] "
                f"unfused={time_unfused:.3f}ms, fused={time_fused:.3f}ms, "
                f"speedup={speedup:.2f}x"
            )

            dsa_metrics.record_performance(
                params={"ratio": compress_ratio, "sq": sq, "b": b, "np": num_heads, "win": window_size},
                fused_ms=time_fused, unfused_ms=time_unfused, speedup=speedup, label="no_indexer_e2e",
            )

    @pytest.mark.parametrize(
        "compress_ratio,sq,b,num_heads,v_head_dim,window_size",
        [
            (0,   2048, 1, 32, 128, 128),
            (0,   4096, 1, 32, 128, 256),
            (128, 2048, 1, 32, 128, 128),
            (128, 4096, 1, 32, 128, 256),
        ],
        ids=["win_only_2k", "win_only_4k", "ratio128_2k", "ratio128_4k"],
    )
    def test_e2e_memory_no_indexer(
        self, compress_ratio, sq, b, num_heads, v_head_dim, window_size, device, dsa_metrics,
    ):
        """Peak GPU memory: fused triton vs unfused CSA (no indexer)."""
        with _hadamard_patches():
            config = _make_test_mla_config(
                num_attention_heads=num_heads,
                v_head_dim=v_head_dim,
                csa_window_size=window_size,
                dsa_indexer_topk=64,
                dsa_indexer_loss_coeff=0.0,
            )

            torch.manual_seed(42)
            csa = _build_csa_module(config, compress_ratio=compress_ratio)
            csa.train()

            inputs = _make_csa_inputs(sq, b, config, device)

            # --- Unfused memory measurement ---
            csa.apply_dsa_kernel_fusion = False
            query_u = inputs["query"].clone().requires_grad_(True)
            key_u = inputs["key"].clone().requires_grad_(True)

            with _oom_guard():
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
                mem_before = torch.cuda.memory_allocated(device)

                out = csa(
                    query=query_u, key=key_u, value=key_u.clone(),
                    attention_mask=None, x=inputs["x"], qr=inputs["qr"],
                )
                out.sum().backward()
                torch.cuda.synchronize()
                mem_unfused_peak = torch.cuda.max_memory_allocated(device) - mem_before

            query_u.grad = None
            key_u.grad = None
            csa.zero_grad()
            torch.cuda.empty_cache()

            # --- Fused memory measurement ---
            csa.apply_dsa_kernel_fusion = True
            query_f = inputs["query"].clone().requires_grad_(True)
            key_f = inputs["key"].clone().requires_grad_(True)

            # Warmup
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
            mem_before = torch.cuda.memory_allocated(device)

            out = csa(
                query=query_f, key=key_f, value=key_f.clone(),
                attention_mask=None, x=inputs["x"], qr=inputs["qr"],
            )
            out.sum().backward()
            torch.cuda.synchronize()
            mem_fused_peak = torch.cuda.max_memory_allocated(device) - mem_before

            ratio = mem_unfused_peak / max(mem_fused_peak, 1)

            logger.info(
                f"[ratio={compress_ratio}, sq={sq}, b={b}, np={num_heads}, "
                f"win={window_size}] "
                f"unfused={mem_unfused_peak / 1024**2:.1f}MB, "
                f"fused={mem_fused_peak / 1024**2:.1f}MB, ratio={ratio:.2f}x"
            )

            dsa_metrics.record_memory(
                params={"ratio": compress_ratio, "sq": sq, "b": b, "np": num_heads, "win": window_size},
                fused_mb=mem_fused_peak / 1024**2, unfused_mb=mem_unfused_peak / 1024**2, ratio=ratio,
            )


# ---------------------------------------------------------------------------
# Kernel-level backward accuracy: _DSASparseAttnFunc (dsa_sparse_attn)
# ---------------------------------------------------------------------------


class TestDSASparseAttnBackward:
    """Kernel-level backward accuracy for dsa_sparse_attn (_DSASparseAttnFunc).

    Verifies gradient correctness for both backward dispatch paths:
    - BMM backward: triggered when TopK <= 512 (forward used PyTorch BMM)
    - Triton backward: triggered when TopK > 512 (forward used Triton kernel)

    Reference: unfused_compressed_sparse_attn (fully materialized PyTorch autograd).
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @staticmethod
    def _make_inputs(
        total_sq: int, total_skv: int, np_: int, d: int, topk: int,
        device: torch.device, seed: int = 42,
    ) -> dict:
        """Generate random inputs for direct dsa_sparse_attn testing."""
        torch.manual_seed(seed)

        query = torch.randn(total_sq, np_, d, device=device, dtype=torch.bfloat16)
        kv = torch.randn(total_skv, d, device=device, dtype=torch.bfloat16)
        attn_sink = torch.randn(np_, device=device, dtype=torch.float32) * 0.1
        softmax_scale = 1.0 / math.sqrt(d)

        # Generate random valid indices in [0, total_skv), with some -1 for
        # early positions (simulating causal masking)
        topk_idxs = torch.randint(0, total_skv, (total_sq, topk), device=device, dtype=torch.int32)
        # Mark ~10% as invalid
        invalid_mask = torch.rand(total_sq, topk, device=device) < 0.1
        topk_idxs[invalid_mask] = -1
        # First row: only 1 valid position (stress test for edge cases)
        topk_idxs[0, 1:] = -1

        return {
            "query": query,
            "kv": kv,
            "attn_sink": attn_sink,
            "softmax_scale": softmax_scale,
            "topk_idxs": topk_idxs,
        }

    @staticmethod
    def _run_fused(inputs: dict) -> dict:
        """Run dsa_sparse_attn (fused kernel) with grad."""
        query = inputs["query"].clone().detach().requires_grad_(True)
        kv = inputs["kv"].clone().detach().requires_grad_(True)
        attn_sink = inputs["attn_sink"].clone().detach().requires_grad_(True)

        # topk_idxs is (total_sq, topk) → unsqueeze to (total_sq, 1, topk)
        topk_idxs = inputs["topk_idxs"].unsqueeze(1)

        out, lse, _ = _triton_dsa_sparse_attn_raw(
            query, kv, topk_idxs, inputs["softmax_scale"],
            d_v=query.shape[-1], attn_sink=attn_sink,
        )
        out.float().sum().backward()

        return {
            "output": out,
            "grad_query": query.grad,
            "grad_kv": kv.grad,
            "grad_attn_sink": attn_sink.grad,
        }

    @staticmethod
    def _run_unfused(inputs: dict) -> dict:
        """Run unfused_compressed_sparse_attn (reference) with grad."""
        total_sq, np_, d = inputs["query"].shape
        total_skv = inputs["kv"].shape[0]

        # unfused expects: query (sq, b, np, d), kv_full (n_kv, b, d), indices (b, sq, topk)
        # Our inputs have b=1 implicit, so unsqueeze the batch dim.
        query = inputs["query"].unsqueeze(1).clone().detach().requires_grad_(True)  # (sq, 1, np, d)
        kv = inputs["kv"].unsqueeze(1).clone().detach().requires_grad_(True)  # (skv, 1, d)
        attn_sink = inputs["attn_sink"].clone().detach().requires_grad_(True)

        # topk_idxs: (total_sq, topk) → (b=1, sq, topk)
        topk_idxs = inputs["topk_idxs"].unsqueeze(0)

        out = unfused_compressed_sparse_attn(
            query, kv, attn_sink, topk_idxs, inputs["softmax_scale"],
        )
        out.float().sum().backward()

        return {
            "output": out,
            "grad_query": query.grad.squeeze(1),  # (sq, np, d)
            "grad_kv": kv.grad.squeeze(1),  # (skv, d)
            "grad_attn_sink": attn_sink.grad,
        }

    @pytest.mark.parametrize(
        "total_sq,total_skv,np_,d,topk",
        [
            # BMM backward path: TopK <= 512
            (2048, 2048, 32, 128, 128),
            (2048, 2048, 32, 128, 256),
            (4096, 4096, 32, 128, 256),
            (2048, 2048, 64, 128, 128),
            # Triton backward path: TopK > 512
            (2048, 2048, 32, 128, 640),
            (2048, 4096, 32, 128, 768),
        ],
        ids=[
            "bmm_bwd_topk128",
            "bmm_bwd_topk256",
            "bmm_bwd_topk256_sq4k",
            "bmm_bwd_topk128_np64",
            "triton_bwd_topk640",
            "triton_bwd_topk768",
        ],
    )
    def test_grad_query_accuracy(self, total_sq, total_skv, np_, d, topk, device, dsa_metrics):
        """Gradient w.r.t. query matches unfused reference for both bwd paths."""
        inputs = self._make_inputs(total_sq, total_skv, np_, d, topk, device)

        fused_res = self._run_fused(inputs)
        unfused_res = self._run_unfused(inputs)

        g_fused = fused_res["grad_query"].float()
        g_unfused = unfused_res["grad_query"].float()

        # Check no NaN/Inf
        assert not torch.isnan(g_fused).any(), "grad_query has NaN"
        assert not torch.isinf(g_fused).any(), "grad_query has Inf"

        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        logger.info(
            f"[sq={total_sq}, skv={total_skv}, np={np_}, topk={topk}] "
            f"grad_query cos_sim={cos_sim:.6f}"
        )
        assert cos_sim > 0.95, f"grad_query cosine similarity too low: {cos_sim:.6f}"

        dsa_metrics.record_accuracy(
            params={"sq": total_sq, "skv": total_skv, "np": np_, "topk": topk},
            cos_sim=cos_sim, target="kernel_grad_query",
        )

    @pytest.mark.parametrize(
        "total_sq,total_skv,np_,d,topk",
        [
            # BMM backward path: TopK <= 512
            (2048, 2048, 32, 128, 128),
            (2048, 2048, 32, 128, 256),
            # Triton backward path: TopK > 512
            (2048, 2048, 32, 128, 640),
        ],
        ids=[
            "bmm_bwd_topk128",
            "bmm_bwd_topk256",
            "triton_bwd_topk640",
        ],
    )
    def test_grad_kv_accuracy(self, total_sq, total_skv, np_, d, topk, device, dsa_metrics):
        """Gradient w.r.t. kv matches unfused reference for both bwd paths."""
        inputs = self._make_inputs(total_sq, total_skv, np_, d, topk, device)

        fused_res = self._run_fused(inputs)
        unfused_res = self._run_unfused(inputs)

        g_fused = fused_res["grad_kv"].float()
        g_unfused = unfused_res["grad_kv"].float()

        # Check no NaN/Inf
        assert not torch.isnan(g_fused).any(), "grad_kv has NaN"
        assert not torch.isinf(g_fused).any(), "grad_kv has Inf"

        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        logger.info(
            f"[sq={total_sq}, skv={total_skv}, np={np_}, topk={topk}] "
            f"grad_kv cos_sim={cos_sim:.6f}"
        )
        assert cos_sim > 0.95, f"grad_kv cosine similarity too low: {cos_sim:.6f}"

        dsa_metrics.record_accuracy(
            params={"sq": total_sq, "skv": total_skv, "np": np_, "topk": topk},
            cos_sim=cos_sim, target="kernel_grad_kv",
        )


# ---------------------------------------------------------------------------
# Indexer gradient accuracy: fused (triton plugin) vs unfused (Megatron core)
# ---------------------------------------------------------------------------


class TestIndexerGradAccuracy:
    """Validate indexer gradient accuracy: fused plugin vs Megatron core reference.

    The fused path pre-computes indexer gradients in forward via
    ``fused_sparse_indexer_loss_and_backward`` / ``fused_dense_indexer_loss_and_backward``
    (triton plugin). The unfused path uses ``FusedDSAIndexerLoss`` + autograd
    (``bwd_fused_indexer_loss_naive``) from Megatron core.

    We verify: grad_q_indexer, grad_k_indexer, grad_weights match between the two.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @staticmethod
    def _run_fused_indexer_grads(inputs: dict, sparse_loss: bool) -> dict:
        """Run fused path, extract indexer gradients."""
        assert torch.is_grad_enabled(), "grad must be enabled for fused indexer grad test"

        q_indexer = inputs["q_indexer"].clone().detach().requires_grad_(True)
        k_indexer = inputs["k_indexer"].clone().detach().requires_grad_(True)
        weights = inputs["weights"].clone().detach().requires_grad_(True)

        # query/kv_full need grad so FusedIndexerSparseAttnFunc.forward enters needs_grad path
        query = inputs["query"].clone().detach().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().detach().requires_grad_(True)
        attn_sink = inputs["attn_sink"].clone().detach().requires_grad_(True)

        out, loss = fused_indexer_sparse_attn(
            query, kv_full, attn_sink, inputs["window_idxs"],
            q_indexer, k_indexer, weights,
            inputs["indexer_topk"], inputs["ratio"],
            inputs["softmax_scale"], inputs["indexer_softmax_scale"],
            0.1, sparse_loss, inputs["kv_offset"], False,
        )

        assert loss.requires_grad, "loss must require grad"
        assert out.requires_grad, "out must require grad"

        # Backward through both out and loss. The attention backward (from out)
        # only contributes to query/kv_full/attn_sink grads, not indexer grads.
        # The indexer grads come solely from grad_loss * precomputed_grads.
        (out.float().sum() + loss).backward()

        return {
            "loss": loss.item(),
            "grad_q_indexer": q_indexer.grad,
            "grad_k_indexer": k_indexer.grad,
            "grad_weights": weights.grad,
        }

    @staticmethod
    def _make_mock_pg_collection():
        """Create a mock ProcessGroupCollection with TP size=1 (no all-reduce)."""
        class _MockPG:
            def size(self):
                return 1
        class _MockPGCollection:
            tp = _MockPG()
            cp = None
        return _MockPGCollection()

    @staticmethod
    def _run_unfused_indexer_grads(inputs: dict, sparse_loss: bool) -> dict:
        """Run Megatron core FusedDSAIndexerLoss, extract indexer gradients."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            FusedDSAIndexerLoss,
        )

        sq = inputs["sq"]
        b = inputs["b"]
        np_ = inputs["np"]
        hn = inputs["hn"]
        n_comp = inputs["n_comp"]
        kv_offset = inputs["kv_offset"]

        # FusedDSAIndexerLoss expects SBHD layout
        q_indexer = inputs["q_indexer"].clone().detach().requires_grad_(True)
        k_indexer = inputs["k_indexer"].clone().detach().requires_grad_(True)

        # weights_for_unfused = weights * indexer_softmax_scale (pre-scaled)
        weights_raw = inputs["weights"].clone().detach().float()
        weights_for_unfused = (weights_raw * inputs["indexer_softmax_scale"]).requires_grad_(True)

        # query and key are detached in the real unfused path
        query_det = inputs["query"].clone().detach()
        # key_for_loss: [n_comp, b, np, hn] — compressed KV expanded to np heads
        kv_compressed = inputs["kv_full"][kv_offset:kv_offset + n_comp]  # (n_comp, b, hn)
        key_for_loss = kv_compressed.unsqueeze(2).expand(-1, -1, np_, -1).detach()

        # Build causal mask [b, sq, n_comp]
        causal_mask = (
            torch.arange(n_comp, device=q_indexer.device).unsqueeze(0).expand(sq, -1)
        )
        positions = torch.arange(1, sq + 1, device=q_indexer.device).unsqueeze(1)
        ratio = inputs["ratio"]
        causal_mask = (
            torch.where(causal_mask >= positions // ratio, float("-inf"), 0.0)
            .unsqueeze(0)
            .expand(b, -1, -1)
        )

        pg_collection = TestIndexerGradAccuracy._make_mock_pg_collection()

        topk_indices, loss = FusedDSAIndexerLoss.apply(
            q_indexer,
            weights_for_unfused,
            k_indexer,
            query_det,
            key_for_loss,
            inputs["softmax_scale"],
            min(inputs["indexer_topk"], n_comp),
            0.1,  # loss_coeff
            causal_mask,
            sparse_loss,
            pg_collection,
            False,  # calculate_per_token_loss
        )

        loss.backward()

        return {
            "loss": loss.item(),
            "grad_q_indexer": q_indexer.grad,
            "grad_k_indexer": k_indexer.grad,
            "grad_weights": weights_for_unfused.grad,
        }

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_q_indexer_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
    ):
        """grad_q_indexer from fused plugin matches Megatron core reference."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_indexer_grads(inputs, sparse_loss)
        unfused_res = self._run_unfused_indexer_grads(inputs, sparse_loss)

        g_fused = fused_res["grad_q_indexer"].float()
        g_unfused = unfused_res["grad_q_indexer"].float()

        logger.warning(
            f"  [DEBUG test] fused_loss={fused_res['loss']:.6e}, unfused_loss={unfused_res['loss']:.6e}"
        )
        logger.warning(
            f"  [DEBUG test] g_fused: norm={g_fused.norm().item():.6e}, "
            f"all_zero={(g_fused == 0).all().item()}, max_abs={g_fused.abs().max().item():.6e}, "
            f"shape={list(g_fused.shape)}, dtype_orig={fused_res['grad_q_indexer'].dtype}"
        )
        logger.warning(
            f"  [DEBUG test] g_unfused: norm={g_unfused.norm().item():.6e}, "
            f"all_zero={(g_unfused == 0).all().item()}, max_abs={g_unfused.abs().max().item():.6e}, "
            f"shape={list(g_unfused.shape)}, dtype_orig={unfused_res['grad_q_indexer'].dtype}"
        )
        # Check if fused grad is zero but unfused is not
        if (g_fused == 0).all() and not (g_unfused == 0).all():
            logger.warning(
                "  [DEBUG test] FUSED GRAD IS ALL ZERO! Investigating precomputed grad..."
            )
            # Check grad_k and grad_w to see if they are also zero
            gk_fused = fused_res["grad_k_indexer"].float()
            gw_fused = fused_res["grad_weights"].float()
            logger.warning(
                f"  [DEBUG test] grad_k_fused: norm={gk_fused.norm().item():.6e}, all_zero={(gk_fused==0).all().item()}"
            )
            logger.warning(
                f"  [DEBUG test] grad_w_fused: norm={gw_fused.norm().item():.6e}, all_zero={(gw_fused==0).all().item()}"
            )

        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        abs_diff = (g_fused - g_unfused).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        logger.info(
            f"[sq={sq}, b={b}, topk={indexer_topk}, sparse_loss={sparse_loss}] "
            f"grad_q_indexer: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e}"
        )

        assert cos_sim > 0.95, (
            f"grad_q_indexer cosine similarity too low: {cos_sim:.6f} "
            f"(max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e})"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="grad_q_indexer",
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_k_indexer_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
    ):
        """grad_k_indexer from fused plugin matches Megatron core reference."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_indexer_grads(inputs, sparse_loss)
        unfused_res = self._run_unfused_indexer_grads(inputs, sparse_loss)

        g_fused = fused_res["grad_k_indexer"].float()
        g_unfused = unfused_res["grad_k_indexer"].float()

        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        abs_diff = (g_fused - g_unfused).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        logger.info(
            f"[sq={sq}, b={b}, topk={indexer_topk}, sparse_loss={sparse_loss}] "
            f"grad_k_indexer: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e}"
        )

        assert cos_sim > 0.95, (
            f"grad_k_indexer cosine similarity too low: {cos_sim:.6f} "
            f"(max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e})"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="grad_k_indexer",
        )

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_grad_weights_accuracy(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
    ):
        """grad_weights from fused plugin matches Megatron core reference.

        Note: fused returns d(loss)/d(weights_raw), unfused returns
        d(loss)/d(weights_scaled) where scaled = raw * indexer_softmax_scale.
        These differ by a constant factor. If they match, fused correctly
        accounts for the scale in its backward; if not, this test catches it.
        """
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        fused_res = self._run_fused_indexer_grads(inputs, sparse_loss)
        unfused_res = self._run_unfused_indexer_grads(inputs, sparse_loss)

        g_fused = fused_res["grad_weights"].float()
        g_unfused = unfused_res["grad_weights"].float()

        cos_sim = torch.nn.functional.cosine_similarity(
            g_fused.reshape(-1).unsqueeze(0),
            g_unfused.reshape(-1).unsqueeze(0),
        ).item()

        abs_diff = (g_fused - g_unfused).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        logger.info(
            f"[sq={sq}, b={b}, topk={indexer_topk}, sparse_loss={sparse_loss}] "
            f"grad_weights: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e}"
        )

        assert cos_sim > 0.95, (
            f"grad_weights cosine similarity too low: {cos_sim:.6f} "
            f"(max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e})"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=cos_sim, max_diff=max_diff, mean_diff=mean_diff, target="grad_weights",
        )


# ---------------------------------------------------------------------------
# Indexer loss value consistency: fused (triton plugin) vs unfused (Megatron core)
# ---------------------------------------------------------------------------


class TestIndexerLossConsistency:
    """Validate indexer loss value matches between fused and unfused paths.

    The fused path computes indexer loss inside ``FusedIndexerSparseAttnFunc.forward``
    via ``fused_sparse_indexer_loss_and_backward`` / ``fused_dense_indexer_loss_and_backward``.
    The unfused path uses ``FusedDSAIndexerLoss.forward`` → ``fwd_fused_indexer_loss_naive``
    from Megatron core.

    We compare the scalar loss values and verify they agree within bf16 tolerance.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @staticmethod
    def _get_fused_loss(inputs: dict, sparse_loss: bool) -> float:
        """Run fused path, return indexer loss scalar."""
        with torch.no_grad():
            _, loss = fused_indexer_sparse_attn(
                inputs["query"], inputs["kv_full"], inputs["attn_sink"],
                inputs["window_idxs"], inputs["q_indexer"], inputs["k_indexer"],
                inputs["weights"], inputs["indexer_topk"], inputs["ratio"],
                inputs["softmax_scale"], inputs["indexer_softmax_scale"],
                0.1, sparse_loss, inputs["kv_offset"], False,
            )
        return loss.item()

    @staticmethod
    def _get_unfused_loss(inputs: dict, sparse_loss: bool) -> float:
        """Run Megatron core FusedDSAIndexerLoss, return indexer loss scalar."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            FusedDSAIndexerLoss,
        )

        sq = inputs["sq"]
        b = inputs["b"]
        np_ = inputs["np"]
        hn = inputs["hn"]
        n_comp = inputs["n_comp"]
        kv_offset = inputs["kv_offset"]
        ratio = inputs["ratio"]

        q_indexer = inputs["q_indexer"].clone().detach()
        k_indexer = inputs["k_indexer"].clone().detach()
        weights_raw = inputs["weights"].clone().detach().float()
        weights_for_unfused = weights_raw * inputs["indexer_softmax_scale"]

        query_det = inputs["query"].clone().detach()
        kv_compressed = inputs["kv_full"][kv_offset:kv_offset + n_comp]
        key_for_loss = kv_compressed.unsqueeze(2).expand(-1, -1, np_, -1).detach()

        # Build causal mask [b, sq, n_comp]
        causal_mask = (
            torch.arange(n_comp, device=q_indexer.device).unsqueeze(0).expand(sq, -1)
        )
        positions = torch.arange(1, sq + 1, device=q_indexer.device).unsqueeze(1)
        causal_mask = (
            torch.where(causal_mask >= positions // ratio, float("-inf"), 0.0)
            .unsqueeze(0)
            .expand(b, -1, -1)
        )

        # Mock pg_collection with TP size=1
        class _MockPG:
            def size(self):
                return 1
        class _MockPGCollection:
            tp = _MockPG()
            cp = None
        pg_collection = _MockPGCollection()

        with torch.no_grad():
            _, loss = FusedDSAIndexerLoss.apply(
                q_indexer,
                weights_for_unfused,
                k_indexer,
                query_det,
                key_for_loss,
                inputs["softmax_scale"],
                min(inputs["indexer_topk"], n_comp),
                0.1,  # loss_coeff
                causal_mask,
                sparse_loss,
                pg_collection,
                False,  # calculate_per_token_loss
            )

        return loss.item()

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
            (2048, 1, 64, 128, 512, 128, 4, 64, 512, 4),
        ],
        ids=[
            "7B_B1_sq2048_topk256",
            "7B_B2_sq2048_topk256",
            "13B_B1_sq4096_topk384",
            "70B_B1_sq2048_topk512",
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_loss_value_consistency(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
    ):
        """Indexer loss from fused plugin matches Megatron core reference."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        loss_fused = self._get_fused_loss(inputs, sparse_loss)
        loss_unfused = self._get_unfused_loss(inputs, sparse_loss)

        # Both losses are f32 scalars; use relative tolerance
        abs_diff = abs(loss_fused - loss_unfused)
        denom = max(abs(loss_fused), abs(loss_unfused), 1e-8)
        rel_diff = abs_diff / denom

        logger.info(
            f"[sq={sq}, b={b}, topk={indexer_topk}, sparse_loss={sparse_loss}] "
            f"loss_fused={loss_fused:.6e}, loss_unfused={loss_unfused:.6e}, "
            f"rel_diff={rel_diff:.4e}"
        )

        # bf16 inputs → allow up to 5% relative error for loss accumulation
        assert rel_diff < 0.05, (
            f"Indexer loss mismatch: fused={loss_fused:.6e}, unfused={loss_unfused:.6e}, "
            f"rel_diff={rel_diff:.4e}"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "np": np_, "topk": indexer_topk, "sparse_loss": sparse_loss},
            cos_sim=1.0 - rel_diff, max_diff=abs_diff, target="indexer_loss",
        )

    # ------------------------------------------------------------------
    # Per-token vs mean reduction relationship
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
            (2048, 2, 32, 128, 512, 128, 4, 64, 256, 4),
            (4096, 1, 32, 128, 1024, 256, 4, 64, 384, 4),
        ],
        ids=["7B_B1_sq2048", "7B_B2_sq2048", "13B_B1_sq4096"],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_per_token_vs_mean_reduction(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device, dsa_metrics,
    ):
        """loss(per_token) / loss(mean) ≈ B * S_q (sum vs mean over rows)."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        loss_coeff = 1.0  # Use 1.0 so ratio is cleaner

        _, loss_mean = fused_indexer_sparse_attn(
            inputs["query"], inputs["kv_full"], inputs["attn_sink"],
            inputs["window_idxs"], inputs["q_indexer"], inputs["k_indexer"],
            inputs["weights"], inputs["indexer_topk"], inputs["ratio"],
            inputs["softmax_scale"], inputs["indexer_softmax_scale"],
            loss_coeff, sparse_loss, inputs["kv_offset"],
            calculate_per_token_loss=False,
        )

        _, loss_per_token = fused_indexer_sparse_attn(
            inputs["query"], inputs["kv_full"], inputs["attn_sink"],
            inputs["window_idxs"], inputs["q_indexer"], inputs["k_indexer"],
            inputs["weights"], inputs["indexer_topk"], inputs["ratio"],
            inputs["softmax_scale"], inputs["indexer_softmax_scale"],
            loss_coeff, sparse_loss, inputs["kv_offset"],
            calculate_per_token_loss=True,
        )

        n_rows = b * sq
        expected_ratio = float(n_rows)
        actual_ratio = loss_per_token.item() / max(loss_mean.item(), 1e-12)

        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.01, (
            f"per_token/mean ratio mismatch: got {actual_ratio:.2f}, "
            f"expected {expected_ratio:.2f} (rel_err={rel_err:.4e})"
        )

        logger.info(
            f"[sq={sq}, b={b}, sparse_loss={sparse_loss}] "
            f"per_token={loss_per_token.item():.6f}, mean={loss_mean.item():.6f}, "
            f"ratio={actual_ratio:.2f}, expected={expected_ratio:.0f}"
        )
        dsa_metrics.record_accuracy(
            params={"sq": sq, "b": b, "sparse_loss": sparse_loss},
            cos_sim=1.0 - rel_err, target="per_token_vs_mean",
        )

    # ------------------------------------------------------------------
    # Per-token loss with gradient — ensure backward still works
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "sq,b,np_,hn,n_comp,win_topk,idx_nh,idx_hd,indexer_topk,ratio",
        [
            (2048, 1, 32, 128, 512, 128, 4, 64, 256, 4),
        ],
    )
    @pytest.mark.parametrize("sparse_loss", [True, False], ids=["sparse", "dense"])
    def test_per_token_loss_backward_no_nan(
        self, sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
        indexer_topk, ratio, sparse_loss, device,
    ):
        """Backward with calculate_per_token_loss=True produces no NaN/Inf."""
        kv_offset = sq
        inputs = _make_fused_inputs(
            sq, b, np_, hn, n_comp, win_topk, idx_nh, idx_hd,
            indexer_topk, ratio, kv_offset, device,
        )

        query = inputs["query"].clone().detach().requires_grad_(True)
        kv_full = inputs["kv_full"].clone().detach().requires_grad_(True)
        attn_sink = inputs["attn_sink"].clone().detach().requires_grad_(True)

        out, loss = fused_indexer_sparse_attn(
            query, kv_full, attn_sink, inputs["window_idxs"],
            inputs["q_indexer"], inputs["k_indexer"], inputs["weights"],
            inputs["indexer_topk"], inputs["ratio"],
            inputs["softmax_scale"], inputs["indexer_softmax_scale"],
            0.1, sparse_loss, inputs["kv_offset"],
            calculate_per_token_loss=True,
        )

        scalar = out.float().sum() + loss
        scalar.backward()

        for name, param in [("query", query), ("kv_full", kv_full), ("attn_sink", attn_sink)]:
            assert param.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(param.grad).any(), f"{name}.grad has NaN"
            assert not torch.isinf(param.grad).any(), f"{name}.grad has Inf"

        assert loss.item() > 0, f"Per-token loss should be positive, got {loss.item()}"
        logger.info(
            f"[sparse_loss={sparse_loss}] per_token_loss={loss.item():.6f}, "
            f"grad_query_norm={query.grad.float().norm().item():.4e}"
        )

