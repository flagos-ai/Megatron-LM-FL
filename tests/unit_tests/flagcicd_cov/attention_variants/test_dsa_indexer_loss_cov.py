# Copyright (c) 2026, BAAI. All rights reserved.
"""DSA indexer loss: naive backward gradient parity with autograd, fused autograd equivalence, tensor-parallel head-partial score reduction, and main-loss backward scaling."""

import pytest
import torch

import megatron.core.parallel_state as parallel_state
import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def _assert_close_at_tf32(actual, expected, msg):
    """Compare fp32 gradients whose computation runs einsums through cuBLAS.

    The CI image sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, so fp32 matmuls use TF32 (10-bit
    mantissa) regardless of torch.backends flags; bound the error relative to the largest
    gradient instead of per element.
    """
    torch.testing.assert_close(
        actual, expected, rtol=1e-2, atol=1e-3 * expected.abs().max().item(), msg=msg
    )


class TestBwdFusedIndexerLossNaive:
    """bwd_fused_indexer_loss_naive must be the exact backward of fwd_fused_indexer_loss_naive."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'mask_kind', ['default', 'override_2d', 'override_3d'], ids=['default', '2d', '3d']
    )
    @pytest.mark.parametrize('sparse_loss', [False, True], ids=['dense', 'sparse'])
    @pytest.mark.parametrize(
        'calculate_per_token_loss', [False, True], ids=['mean_loss', 'per_token_loss']
    )
    def test_backward_matches_autograd_of_naive_forward(
        self, mask_kind, sparse_loss, calculate_per_token_loss
    ):
        torch.manual_seed(42)
        device = torch.cuda.current_device()
        sq, b, n_heads, head_dim = 16, 2, 4, 32
        idx_heads, idx_dim, index_topk, loss_coeff = 4, 64, 8, 2.0
        softmax_scale = head_dim**-0.5

        q = torch.randn(sq, b, idx_heads, idx_dim, dtype=torch.float32, device=device)
        weights = torch.randn(sq, b, idx_heads, dtype=torch.float32, device=device)
        k = torch.randn(sq, b, idx_dim, dtype=torch.float32, device=device)
        query = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        key = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        causal = torch.triu(
            torch.full((sq, sq), float('-inf'), dtype=torch.float32, device=device), diagonal=1
        )
        if mask_kind == 'default':
            bwd_mask = None
            fwd_mask = causal
        elif mask_kind == 'override_2d':
            bwd_mask = causal
            fwd_mask = causal
        else:
            bwd_mask = causal.unsqueeze(0).expand(b, sq, sq).contiguous()
            fwd_mask = bwd_mask

        q_ref = q.clone().requires_grad_(True)
        weights_ref = weights.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        topk_ref, loss_ref = dsa_module.fwd_fused_indexer_loss_naive(
            q_ref,
            weights_ref,
            k_ref,
            query,
            key,
            index_topk,
            softmax_scale,
            loss_coeff,
            fwd_mask,
            sparse_loss,
            self.pg,
            calculate_per_token_loss,
        )
        loss_ref.backward()

        grad_q, grad_weights, grad_k = dsa_module.bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_ref,
            softmax_scale,
            loss_coeff,
            sparse_loss,
            torch.ones((), device=device),
            self.pg,
            causal_mask_override=bwd_mask,
            calculate_per_token_loss=calculate_per_token_loss,
        )

        assert topk_ref.shape == (b, sq, min(index_topk, sq))
        assert torch.all(topk_ref >= 0) and torch.all(topk_ref < sq)
        _assert_close_at_tf32(grad_q, q_ref.grad, 'grad_q mismatch')
        torch.testing.assert_close(
            grad_weights, weights_ref.grad, rtol=1e-2, atol=1e-4, msg='grad_weights mismatch'
        )
        _assert_close_at_tf32(grad_k, k_ref.grad, 'grad_k mismatch')

    def test_backward_returns_zero_gradients_when_loss_coeff_is_zero(self):
        torch.manual_seed(7)
        device = torch.cuda.current_device()
        sq, b, n_heads, head_dim = 16, 2, 4, 32
        idx_heads, idx_dim, index_topk = 4, 64, 8
        softmax_scale = head_dim**-0.5

        q = torch.randn(sq, b, idx_heads, idx_dim, dtype=torch.float32, device=device)
        weights = torch.randn(sq, b, idx_heads, dtype=torch.float32, device=device)
        k = torch.randn(sq, b, idx_dim, dtype=torch.float32, device=device)
        query = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        key = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        causal = torch.triu(
            torch.full((sq, sq), float('-inf'), dtype=torch.float32, device=device), diagonal=1
        )
        topk = (dsa_module._compute_index_scores(q, weights, k) + causal).topk(
            min(index_topk, sq), dim=-1
        )[1]

        grad_q, grad_weights, grad_k = dsa_module.bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            0.0,
            False,
            torch.ones((), device=device),
            self.pg,
            causal_mask_override=causal,
        )

        assert torch.count_nonzero(grad_q) == 0
        assert torch.count_nonzero(grad_weights) == 0
        assert torch.count_nonzero(grad_k) == 0


class TestFusedDSAIndexerLoss:
    """FusedDSAIndexerLoss.apply must reproduce the naive forward and manual backward."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('sparse_loss', [False, True], ids=['dense', 'sparse'])
    def test_apply_matches_naive_forward_and_backward(self, sparse_loss):
        torch.manual_seed(11)
        device = torch.cuda.current_device()
        sq, b, n_heads, head_dim = 16, 2, 4, 32
        idx_heads, idx_dim, index_topk, loss_coeff = 4, 64, 8, 1.5
        softmax_scale = head_dim**-0.5

        q = torch.randn(sq, b, idx_heads, idx_dim, dtype=torch.float32, device=device)
        weights = torch.randn(sq, b, idx_heads, dtype=torch.float32, device=device)
        k = torch.randn(sq, b, idx_dim, dtype=torch.float32, device=device)
        query = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        key = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        mask = torch.triu(
            torch.full((sq, sq), float('-inf'), dtype=torch.float32, device=device), diagonal=1
        )

        q_ref = q.clone().requires_grad_(True)
        weights_ref = weights.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        topk_ref, loss_ref = dsa_module.fwd_fused_indexer_loss_naive(
            q_ref,
            weights_ref,
            k_ref,
            query,
            key,
            index_topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            self.pg,
            False,
        )
        loss_ref.backward()

        q_fused = q.clone().requires_grad_(True)
        weights_fused = weights.clone().requires_grad_(True)
        k_fused = k.clone().requires_grad_(True)
        topk_fused, loss_fused = dsa_module.FusedDSAIndexerLoss.apply(
            q_fused,
            weights_fused,
            k_fused,
            query,
            key,
            softmax_scale,
            index_topk,
            loss_coeff,
            mask,
            sparse_loss,
            self.pg,
            False,
        )
        loss_fused.backward()

        assert torch.equal(topk_fused, topk_ref)
        torch.testing.assert_close(loss_fused, loss_ref.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            q_fused.grad, q_ref.grad, rtol=1e-2, atol=1e-4, msg='grad_q mismatch'
        )
        torch.testing.assert_close(
            weights_fused.grad, weights_ref.grad, rtol=1e-2, atol=1e-4, msg='grad_weights mismatch'
        )
        torch.testing.assert_close(
            k_fused.grad, k_ref.grad, rtol=1e-2, atol=1e-4, msg='grad_k mismatch'
        )


class TestBwdFusedIndexerLossTensorParallel:
    """With TP > 1 the manual backward all-reduces head-partial attention scores."""

    @pytest.fixture
    def model_parallel_tp(self):
        def initialize(tensor_model_parallel_size):
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])

        yield initialize
        Utils.destroy_model_parallel()

    def test_backward_all_reduces_head_partial_attention_scores_across_tp(self, model_parallel_tp):
        sq, b, n_heads, head_dim = 16, 2, 4, 32
        idx_heads, idx_dim, index_topk, loss_coeff = 4, 64, 8, 1.0
        softmax_scale = head_dim**-0.5

        pg_tp1 = model_parallel_tp(1)
        torch.manual_seed(42)
        device = torch.cuda.current_device()
        q = torch.randn(sq, b, idx_heads, idx_dim, dtype=torch.float32, device=device)
        weights = torch.randn(sq, b, idx_heads, dtype=torch.float32, device=device)
        k = torch.randn(sq, b, idx_dim, dtype=torch.float32, device=device)
        query = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        key = torch.randn(sq, b, n_heads, head_dim, dtype=torch.float32, device=device)
        mask = torch.triu(
            torch.full((sq, sq), float('-inf'), dtype=torch.float32, device=device), diagonal=1
        )
        q_ref = q.clone().requires_grad_(True)
        weights_ref = weights.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        topk_tp1, loss_tp1 = dsa_module.fwd_fused_indexer_loss_naive(
            q_ref,
            weights_ref,
            k_ref,
            query,
            key,
            index_topk,
            softmax_scale,
            loss_coeff,
            mask,
            False,
            pg_tp1,
            False,
        )
        loss_tp1.backward()
        baseline = (
            topk_tp1.clone(),
            loss_tp1.detach().clone(),
            q_ref.grad.clone(),
            weights_ref.grad.clone(),
            k_ref.grad.clone(),
        )
        Utils.destroy_model_parallel()

        pg_tp2 = model_parallel_tp(2)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        heads_per_rank = n_heads // 2
        query_shard = query[
            :, :, tp_rank * heads_per_rank : (tp_rank + 1) * heads_per_rank, :
        ].contiguous()
        key_shard = key[
            :, :, tp_rank * heads_per_rank : (tp_rank + 1) * heads_per_rank, :
        ].contiguous()
        q_tpn = q.clone().requires_grad_(True)
        weights_tpn = weights.clone().requires_grad_(True)
        k_tpn = k.clone().requires_grad_(True)
        topk_tpn, loss_tpn = dsa_module.FusedDSAIndexerLoss.apply(
            q_tpn,
            weights_tpn,
            k_tpn,
            query_shard,
            key_shard,
            softmax_scale,
            index_topk,
            loss_coeff,
            mask,
            False,
            pg_tp2,
            False,
        )
        loss_tpn.backward()

        assert torch.equal(topk_tpn, baseline[0])
        torch.testing.assert_close(loss_tpn, baseline[1], rtol=1e-5, atol=1e-5)
        _assert_close_at_tf32(q_tpn.grad, baseline[2], 'grad_q mismatch across TP')
        torch.testing.assert_close(
            weights_tpn.grad,
            baseline[3],
            rtol=1e-5,
            atol=1e-5,
            msg='grad_weights mismatch across TP',
        )
        _assert_close_at_tf32(k_tpn.grad, baseline[4], 'grad_k mismatch across TP')


class TestDSAIndexerLossAutoScaler:

    @pytest.fixture(autouse=True)
    def reset_loss_scale(self, monkeypatch):
        monkeypatch.setattr(dsa_module.DSAIndexerLossAutoScaler, 'main_loss_backward_scale', None)

    def test_backward_defaults_to_unit_scale_when_scale_is_unset(self):
        device = torch.cuda.current_device()
        output = torch.randn(4, 3, 2, dtype=torch.float32, device=device).requires_grad_(True)
        proxy = torch.randn(8, dtype=torch.float32, device=device).requires_grad_(True)
        indexer_loss = proxy.square().mean()
        result = dsa_module.DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        result.sum().backward()

        assert torch.equal(output.grad, torch.ones_like(output))
        expected = 2.0 * proxy.detach() / proxy.numel()
        torch.testing.assert_close(proxy.grad, expected, rtol=1e-6, atol=1e-8)

    def test_backward_scales_indexer_loss_gradient_with_configured_scale(self):
        device = torch.cuda.current_device()
        scale = torch.tensor(2.5, device=device)
        dsa_module.DSAIndexerLossAutoScaler.set_loss_scale(scale)
        output = torch.randn(4, 3, 2, dtype=torch.float32, device=device).requires_grad_(True)
        proxy = torch.randn(8, dtype=torch.float32, device=device).requires_grad_(True)
        indexer_loss = proxy.square().mean()
        result = dsa_module.DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        result.sum().backward()

        assert torch.equal(output.grad, torch.ones_like(output))
        expected = 2.5 * 2.0 * proxy.detach() / proxy.numel()
        torch.testing.assert_close(proxy.grad, expected, rtol=1e-6, atol=1e-8)
