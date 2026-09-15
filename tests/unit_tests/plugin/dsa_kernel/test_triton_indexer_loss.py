# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Numerical parity tests for the fused Triton DSA indexer-loss kernels.

``megatron.plugin.dsa_kernel.triton_indexer_loss`` is an implementation
optimisation only: it reduces over the attention/indexer head dimension inside
the kernel so the ``[b, np, sq, sk]`` and ``[sq, b, index_n_heads, sk]`` fp32
intermediates are never materialised, but the maths is unchanged. These tests
pin that down against the unfused reference in ``dsa.py``.

Run with::

    pytest tests/unit_tests/plugin/dsa_kernel/test_triton_indexer_loss.py -v
"""

from __future__ import annotations

import logging

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _cached_causal_neg_inf_mask,
    _compute_index_scores,
    bwd_fused_indexer_loss_naive,
    fwd_fused_indexer_loss_naive,
)
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)

pytest.importorskip("triton", reason="Triton is required for the fused kernels")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _sm90_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_or_newer(), reason="fused indexer-loss kernels require SM90+"
)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two tensors flattened to vectors."""
    return torch.nn.functional.cosine_similarity(
        a.detach().float().reshape(1, -1), b.detach().float().reshape(1, -1)
    ).item()


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error normalised by the reference's scale."""
    denom = ref.detach().float().abs().max().clamp(min=1e-12)
    return ((got.detach().float() - ref.detach().float()).abs().max() / denom).item()


def _make_inputs(
    seqlen: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """Indexer + attention tensors in the SBHD layouts ``DSAttention`` passes."""
    torch.manual_seed(seed)
    q = torch.randn(
        seqlen, batch_size, index_n_heads, index_head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(seqlen, batch_size, index_head_dim, dtype=dtype, device="cuda")
    # Keep weights positive-ish and modest: this mirrors the scaled projection
    # output and keeps the reference's fp32 softmax well conditioned.
    weights = (
        torch.randn(seqlen, batch_size, index_n_heads, dtype=dtype, device="cuda")
        * (index_n_heads**-0.5)
    )
    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    key = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    return q, k, weights, query, key


@requires_sm90
class TestIndexScoreKernel:
    """``indexer_index_score`` vs ``dsa._compute_index_scores``."""

    @pytest.mark.parametrize("seqlen", [64, 128])
    @pytest.mark.parametrize("index_n_heads", [8, 32])
    def test_matches_reference(self, seqlen, index_n_heads, dsa_metrics):
        from megatron.plugin.dsa_kernel.triton_indexer_loss import indexer_index_score

        q, k, weights, _, _ = _make_inputs(
            seqlen, 2, 4, 64, index_n_heads, 128
        )
        got = indexer_index_score(q, k, weights)
        ref = _compute_index_scores(q, weights, k)

        cos = _cos_sim(got, ref)
        rel = _rel_err(got, ref)
        dsa_metrics.record_accuracy(
            {"seqlen": seqlen, "index_n_heads": index_n_heads},
            cos_sim=cos,
            max_diff=(got - ref).abs().max().item(),
            target="index_score",
        )
        assert got.shape == ref.shape
        assert cos > 0.9999, f"cos_sim={cos}"
        assert rel < 1e-3, f"rel_err={rel}"


@requires_sm90
class TestAttnHeadSumKernel:
    """``attn_head_sum`` vs an explicit softmax-then-sum over heads."""

    @staticmethod
    def _reference(query, key, softmax_scale, mask):
        """The ``[b, np, sq, sk]`` path from ``dsa.compute_dsa_indexer_loss``."""
        sq, b, np_, hn = query.shape
        sk = key.shape[0]
        scores = torch.bmm(
            query.permute(1, 2, 0, 3).reshape(b * np_, sq, hn).float(),
            key.permute(1, 2, 3, 0).reshape(b * np_, hn, sk).float(),
        ).reshape(b, np_, sq, sk) * softmax_scale
        scores = scores + (
            mask.unsqueeze(1) if mask.dim() == 3 else mask.view(1, 1, sq, sk)
        )
        row_valid = (mask > float("-inf")).any(dim=-1)
        row_mask = (
            row_valid.view(b, 1, sq, 1) if row_valid.dim() == 2
            else row_valid.view(1, 1, sq, 1)
        )
        scores = scores.masked_fill(~row_mask, 0.0)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32) * row_mask.float()
        return probs.sum(dim=1)

    @pytest.mark.parametrize("mask_kind", ["2d_causal", "3d_batched"])
    @pytest.mark.parametrize("num_heads", [4, 16])
    def test_matches_reference(self, mask_kind, num_heads, dsa_metrics):
        from megatron.plugin.dsa_kernel.triton_indexer_loss import attn_head_sum

        seqlen, batch_size, head_dim = 128, 2, 64
        softmax_scale = head_dim**-0.5
        _, _, _, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, 8, 128
        )

        if mask_kind == "2d_causal":
            mask = _cached_causal_neg_inf_mask(seqlen, seqlen, query.device)
        else:
            mask = _cached_causal_neg_inf_mask(
                seqlen, seqlen, query.device
            ).unsqueeze(0).repeat(batch_size, 1, 1).contiguous()
            # Mask a trailing chunk for one batch element so the two batch rows
            # genuinely differ (a batch-shared mask would not exercise strides).
            mask[1, :, seqlen // 2 :] = float("-inf")

        got = attn_head_sum(query, key, softmax_scale, mask)
        ref = self._reference(query, key, softmax_scale, mask)

        cos = _cos_sim(got, ref)
        rel = _rel_err(got, ref)
        dsa_metrics.record_accuracy(
            {"mask": mask_kind, "num_heads": num_heads},
            cos_sim=cos,
            max_diff=(got - ref).abs().max().item(),
            target="head_sum",
        )
        assert cos > 0.9999, f"cos_sim={cos}"
        assert rel < 1e-3, f"rel_err={rel}"

    def test_fully_masked_rows_are_zero_not_nan(self):
        """A row with no valid KV position must give zeros, never NaN.

        ``exp(-inf - -inf)`` is the natural failure mode here, so this guards the
        online-softmax ``safe_max`` handling.
        """
        from megatron.plugin.dsa_kernel.triton_indexer_loss import attn_head_sum

        seqlen, batch_size, num_heads, head_dim = 64, 1, 4, 64
        _, _, _, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, 8, 128
        )
        mask = torch.zeros(
            (batch_size, seqlen, seqlen), dtype=torch.float32, device="cuda"
        )
        mask[:, :8, :] = float("-inf")  # first 8 query rows have nothing to attend to

        got = attn_head_sum(query, key, head_dim**-0.5, mask)
        assert torch.isfinite(got).all(), "fused head sum produced NaN/Inf"
        assert torch.equal(
            got[:, :8, :], torch.zeros_like(got[:, :8, :])
        ), "fully-masked rows must be zero"


@requires_sm90
class TestIndexerLossParity:
    """End-to-end fwd/bwd parity against the unfused dense reference."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_class_pg(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(42)
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp"]
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("calculate_per_token_loss", [False, True])
    @pytest.mark.parametrize("mask_kind", ["none", "2d_causal", "3d_batched"])
    def test_forward_loss_matches_reference(
        self, calculate_per_token_loss, mask_kind, dsa_metrics
    ):
        from megatron.plugin.dsa_kernel.triton_indexer_loss import (
            fwd_indexer_loss_triton,
        )

        seqlen, batch_size, num_heads, head_dim = 128, 2, 8, 64
        index_topk = 32
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0
        q, k, weights, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, 8, 128
        )
        mask = self._build_mask(mask_kind, seqlen, batch_size, q.device)

        _, loss_ref = fwd_fused_indexer_loss_naive(
            q, weights, k, query, key, index_topk, softmax_scale, loss_coeff,
            mask, False, self.pg_collection, calculate_per_token_loss,
        )
        _, loss_got, _ = fwd_indexer_loss_triton(
            q, weights, k, query, key, index_topk, softmax_scale, loss_coeff,
            mask, self.pg_collection, calculate_per_token_loss,
        )

        rel = _rel_err(loss_got, loss_ref)
        dsa_metrics.record_accuracy(
            {"mask": mask_kind, "per_token": calculate_per_token_loss},
            cos_sim=1.0 - rel,
            max_diff=(loss_got - loss_ref).abs().item(),
            target="loss",
        )
        assert torch.isfinite(loss_got), "fused loss is not finite"
        assert rel < 1e-3, (
            f"loss mismatch: fused={loss_got.item()} ref={loss_ref.item()} rel={rel}"
        )

    @pytest.mark.parametrize("calculate_per_token_loss", [False, True])
    @pytest.mark.parametrize("mask_kind", ["none", "2d_causal", "3d_batched"])
    def test_backward_grads_match_reference(
        self, calculate_per_token_loss, mask_kind, dsa_metrics
    ):
        from megatron.plugin.dsa_kernel.triton_indexer_loss import (
            bwd_indexer_loss_triton,
        )

        seqlen, batch_size, num_heads, head_dim = 128, 2, 8, 64
        index_topk = 32
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0
        q, k, weights, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, 8, 128
        )
        mask = self._build_mask(mask_kind, seqlen, batch_size, q.device)
        grad_loss = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        topk_indices, _ = fwd_fused_indexer_loss_naive(
            q, weights, k, query, key, index_topk, softmax_scale, loss_coeff,
            mask, False, self.pg_collection, calculate_per_token_loss,
        )
        ref = bwd_fused_indexer_loss_naive(
            q, weights, k, query, key, topk_indices, softmax_scale, loss_coeff,
            False, grad_loss, self.pg_collection,
            causal_mask_override=mask,
            calculate_per_token_loss=calculate_per_token_loss,
        )
        got = bwd_indexer_loss_triton(
            q, weights, k, query, key, softmax_scale, loss_coeff, grad_loss,
            self.pg_collection, mask, calculate_per_token_loss,
        )

        for name, g, r in zip(("grad_q", "grad_weights", "grad_k"), got, ref):
            assert g.shape == r.shape, f"{name} shape {g.shape} != {r.shape}"
            assert torch.isfinite(g.float()).all(), f"{name} has NaN/Inf"
            cos = _cos_sim(g, r)
            dsa_metrics.record_accuracy(
                {"mask": mask_kind, "per_token": calculate_per_token_loss},
                cos_sim=cos,
                max_diff=(g.float() - r.float()).abs().max().item(),
                target=name,
            )
            # bf16 gradient outputs quantise to ~3 decimal digits, so compare by
            # direction (cos_sim) rather than elementwise magnitude.
            assert cos > 0.999, f"{name} cos_sim={cos}"

    def test_backward_matches_autograd(self):
        """Cross-check the manual backward against autograd on the reference fwd.

        Guards against fused and unfused sharing a wrong derivative. fp32 inputs
        and a small shape keep autograd's own error well below the tolerance.
        """
        from megatron.plugin.dsa_kernel.triton_indexer_loss import (
            bwd_indexer_loss_triton,
        )

        seqlen, batch_size, num_heads, head_dim = 32, 1, 4, 64
        index_n_heads, index_head_dim = 4, 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0
        q, k, weights, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, index_n_heads,
            index_head_dim, dtype=torch.float32,
        )
        mask = _cached_causal_neg_inf_mask(seqlen, seqlen, q.device)

        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        w_ref = weights.clone().requires_grad_(True)
        _, loss = fwd_fused_indexer_loss_naive(
            q_ref, w_ref, k_ref, query, key, 16, softmax_scale, loss_coeff,
            mask, False, self.pg_collection, False,
        )
        loss.backward()

        grad_q, grad_w, grad_k = bwd_indexer_loss_triton(
            q, weights, k, query, key, softmax_scale, loss_coeff,
            torch.tensor(1.0, device="cuda"), self.pg_collection, mask, False,
        )

        for name, got, ref in (
            ("grad_q", grad_q, q_ref.grad),
            ("grad_weights", grad_w, w_ref.grad),
            ("grad_k", grad_k, k_ref.grad),
        ):
            cos = _cos_sim(got, ref)
            assert cos > 0.999, f"{name} vs autograd cos_sim={cos}"

    @staticmethod
    def _build_mask(kind, seqlen, batch_size, device):
        if kind == "none":
            return None
        if kind == "2d_causal":
            return _cached_causal_neg_inf_mask(seqlen, seqlen, device)
        mask = (
            _cached_causal_neg_inf_mask(seqlen, seqlen, device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .contiguous()
        )
        mask[1, :, seqlen // 2 :] = float("-inf")
        return mask


@requires_sm90
class TestFusedDSAIndexerLossDispatch:
    """``FusedDSAIndexerLoss`` must agree whether or not the kernels are used."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_class_pg(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(42)
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp"]
        )
        yield
        Utils.destroy_model_parallel()

    def test_use_triton_flag_preserves_loss_and_grads(self):
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            FusedDSAIndexerLoss,
        )

        seqlen, batch_size, num_heads, head_dim = 128, 2, 8, 64
        softmax_scale = head_dim**-0.5
        q, k, weights, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, 8, 128
        )
        mask = _cached_causal_neg_inf_mask(seqlen, seqlen, q.device)

        results = {}
        for use_triton in (False, True):
            qq = q.clone().requires_grad_(True)
            kk = k.clone().requires_grad_(True)
            ww = weights.clone().requires_grad_(True)
            _, loss = FusedDSAIndexerLoss.apply(
                qq, ww, kk, query, key, softmax_scale, 32, 1.0, mask, False,
                self.pg_collection, False, use_triton,
            )
            loss.backward()
            results[use_triton] = (loss.detach(), qq.grad, ww.grad, kk.grad)

        loss_ref, *grads_ref = results[False]
        loss_got, *grads_got = results[True]
        assert _rel_err(loss_got, loss_ref) < 1e-3, (
            f"loss differs: {loss_got.item()} vs {loss_ref.item()}"
        )
        for name, g, r in zip(("grad_q", "grad_weights", "grad_k"), grads_got, grads_ref):
            assert _cos_sim(g, r) > 0.999, f"{name} cos_sim={_cos_sim(g, r)}"

    def test_sparse_loss_falls_back_to_unfused(self):
        """``sparse_loss=True`` is unimplemented in Triton and must not be used."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            FusedDSAIndexerLoss,
        )

        seqlen, batch_size, num_heads, head_dim = 64, 1, 4, 64
        q, k, weights, query, key = _make_inputs(
            seqlen, batch_size, num_heads, head_dim, 8, 128
        )
        mask = _cached_causal_neg_inf_mask(seqlen, seqlen, q.device)

        losses = []
        for use_triton in (False, True):
            _, loss = FusedDSAIndexerLoss.apply(
                q, weights, k, query, key, head_dim**-0.5, 16, 1.0, mask,
                True,  # sparse_loss
                self.pg_collection, False, use_triton,
            )
            losses.append(loss.detach())
        assert torch.equal(losses[0], losses[1]), (
            "sparse_loss must take the identical unfused path regardless of the flag"
        )


@pytest.mark.perf
@pytest.mark.skipif(not _sm90_or_newer(), reason="requires SM90+")
class TestIndexerLossPerformance:
    """Fused vs unfused indexer-loss latency and peak memory.

    Reported through ``dsa_metrics`` so ``--dsa-report`` collects these next to the
    sparse-attention numbers. Shapes follow the GLM5.2 744B layer (``np=64`` before
    TP, ``index_n_heads=32``); the unfused reference materialises fp32
    ``[b, np, sq, sk]`` and ``[sq, b, index_n_heads, sk]`` tensors, so it runs out of
    memory well before the fused path does. Those cases are recorded as OOM rather
    than skipped, since "unfused cannot run this shape" is the result.
    """

    NUM_HEADS = 64
    HEAD_DIM = 192
    INDEX_N_HEADS = 32
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 2048

    @pytest.fixture(scope="class", autouse=True)
    def setup_class_pg(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(42)
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp"]
        )
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _benchmark(fn, warmup: int = 3, iters: int = 10):
        """Median elapsed ms and peak MiB, or (nan, nan) if the shape OOMs."""
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            times.sort()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            return times[len(times) // 2], peak
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan"), float("nan")

    def _run_case(self, seqlen, backward, dsa_metrics):
        from megatron.plugin.dsa_kernel.triton_indexer_loss import (
            bwd_indexer_loss_triton,
            fwd_indexer_loss_triton,
        )

        batch_size = 1
        softmax_scale = self.HEAD_DIM**-0.5
        q, k, weights, query, key = _make_inputs(
            seqlen, batch_size, self.NUM_HEADS, self.HEAD_DIM,
            self.INDEX_N_HEADS, self.INDEX_HEAD_DIM,
        )
        mask = _cached_causal_neg_inf_mask(seqlen, seqlen, q.device)
        topk = min(self.INDEX_TOPK, seqlen)
        grad_loss = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        def unfused():
            topk_idx, _ = fwd_fused_indexer_loss_naive(
                q, weights, k, query, key, topk, softmax_scale, 1.0, mask,
                False, self.pg_collection, False,
            )
            if backward:
                bwd_fused_indexer_loss_naive(
                    q, weights, k, query, key, topk_idx, softmax_scale, 1.0,
                    False, grad_loss, self.pg_collection,
                    causal_mask_override=mask, calculate_per_token_loss=False,
                )

        def fused():
            fwd_indexer_loss_triton(
                q, weights, k, query, key, topk, softmax_scale, 1.0, mask,
                self.pg_collection, False,
            )
            if backward:
                bwd_indexer_loss_triton(
                    q, weights, k, query, key, softmax_scale, 1.0, grad_loss,
                    self.pg_collection, mask, False,
                )

        unfused_ms, unfused_mb = self._benchmark(unfused)
        torch.cuda.empty_cache()
        fused_ms, fused_mb = self._benchmark(fused)

        label = "fwd+bwd" if backward else "fwd"
        params = {"sq": seqlen, "b": batch_size, "np": self.NUM_HEADS, "topk": topk}
        # nan propagates through these ratios, which is what marks an OOM row.
        dsa_metrics.record_performance(
            params, fused_ms=fused_ms, unfused_ms=unfused_ms,
            speedup=unfused_ms / fused_ms if fused_ms > 0 else float("nan"),
            label=label,
        )
        dsa_metrics.record_memory(
            params, fused_mb=fused_mb, unfused_mb=unfused_mb,
            ratio=unfused_mb / fused_mb if fused_mb > 0 else float("nan"),
        )
        logger.info(
            "[sq=%d, np=%d, topk=%d] %s: unfused=%.3fms fused=%.3fms speedup=%.2fx | "
            "unfused=%.1fMB fused=%.1fMB",
            seqlen, self.NUM_HEADS, topk, label,
            unfused_ms, fused_ms, unfused_ms / fused_ms if fused_ms > 0 else float("nan"),
            unfused_mb, fused_mb,
        )

        assert fused_ms == fused_ms, f"fused path OOMed at seqlen={seqlen}"

    @pytest.mark.parametrize("seqlen", [1024, 2048, 4096, 8192])
    def test_performance_forward(self, seqlen, dsa_metrics):
        self._run_case(seqlen, backward=False, dsa_metrics=dsa_metrics)

    @pytest.mark.parametrize("seqlen", [1024, 2048, 4096, 8192])
    def test_performance_forward_backward(self, seqlen, dsa_metrics):
        self._run_case(seqlen, backward=True, dsa_metrics=dsa_metrics)
