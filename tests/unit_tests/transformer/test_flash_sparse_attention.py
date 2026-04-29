# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_func

    HAVE_FLASH_SPARSE = True
except ImportError:
    HAVE_FLASH_SPARSE = False

pytestmark = pytest.mark.skipif(
    not HAVE_FLASH_SPARSE, reason="flash-sparse-attn is not installed"
)


def _local_attn_submodules():
    return get_gpt_layer_local_spec().submodules.self_attention.submodules


def _make_config(**overrides):
    defaults = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash_sparse,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_packed_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max().item()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )


class TestFlashSparseAttentionForward:
    """Test _flash_sparse_attention standard (training/prefill) path."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = _make_config()
        self.attention = SelfAttention(
            self.config,
            _local_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.attention, SelfAttention)
        assert self.config.attention_backend == AttnBackend.flash_sparse

    def test_gpu_forward_shape(self):
        seq_len = 32
        batch_size = 2
        self.attention.cuda()

        hidden_states = torch.randn(
            seq_len, batch_size, self.config.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )
        attention_mask = torch.ones(
            batch_size, 1, 1, seq_len, dtype=bool, device='cuda',
        )

        output, bias = self.attention(hidden_states, attention_mask)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16

    def test_gpu_forward_deterministic(self):
        seq_len = 32
        batch_size = 2
        self.attention.cuda()

        hidden_states = torch.randn(
            seq_len, batch_size, self.config.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )
        attention_mask = torch.ones(
            batch_size, 1, 1, seq_len, dtype=bool, device='cuda',
        )

        out1, _ = self.attention(hidden_states, attention_mask)
        out2, _ = self.attention(hidden_states, attention_mask)
        torch.testing.assert_close(out1, out2)

    def test_gpu_forward_backward(self):
        seq_len = 32
        batch_size = 2
        self.attention.cuda()
        self.attention.train()

        hidden_states = torch.randn(
            seq_len, batch_size, self.config.hidden_size,
            dtype=torch.bfloat16, device='cuda', requires_grad=True,
        )
        attention_mask = torch.ones(
            batch_size, 1, 1, seq_len, dtype=bool, device='cuda',
        )

        output, _ = self.attention(hidden_states, attention_mask)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape


class TestFlashSparseAttentionPackedSeq:
    """Test _flash_sparse_attention varlen (thd) path."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = _make_config()
        self.attention = SelfAttention(
            self.config,
            _local_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward_packed(self):
        seq_len = 32
        batch_size = 1
        self.attention.cuda()

        hidden_states = torch.randn(
            seq_len, batch_size, self.config.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )
        packed_seq_params = _make_packed_seq_params(seq_len)

        output, bias = self.attention(
            hidden_states, attention_mask=None, packed_seq_params=packed_seq_params,
        )

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16


class TestFlashSparseAttentionDecode:
    """Test _flash_sparse_attention static decode path (sq == 1)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = _make_config()
        self.attention = SelfAttention(
            self.config,
            _local_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_decode_kernel_directly(self):
        """Directly test _flash_sparse_attention with sq=1 to exercise the kvcache path."""
        batch_size = 4
        num_heads = self.config.num_attention_heads
        head_dim = self.config.kv_channels
        kv_seq_len = 64

        self.attention.cuda()
        self.attention.eval()

        query = torch.randn(
            1, batch_size, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda',
        )
        key = torch.randn(
            kv_seq_len, batch_size, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda',
        )
        value = torch.randn(
            kv_seq_len, batch_size, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda',
        )

        with torch.no_grad():
            output = self.attention._flash_sparse_attention(
                query, key, value,
                attn_mask_type=AttnMaskType.causal,
            )

        # Output should be [1, batch_size, num_heads * head_dim]
        assert output.shape == (1, batch_size, num_heads * head_dim)
        assert output.dtype == torch.bfloat16


class TestFlashSparseDecodeAndPrefill:
    """Test flash_sparse_decode_and_prefill for dynamic batching."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = _make_config()
        self.attention = SelfAttention(
            self.config,
            _local_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_prefill_path(self):
        """Test prefill branch (max_seqlen_q > 1)."""
        num_heads = self.config.num_attention_heads
        head_dim = self.config.kv_channels
        total_q = 20
        total_k = 20

        self.attention.cuda()
        self.attention.eval()

        q = torch.randn(total_q, 1, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(total_k, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(total_k, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

        cu_seqlens_q = torch.tensor([0, 8, 20], dtype=torch.int32, device='cuda')
        cu_seqlens_k = torch.tensor([0, 8, 20], dtype=torch.int32, device='cuda')
        seqlens_k = torch.tensor([8, 12], dtype=torch.int32, device='cuda')

        with torch.no_grad():
            output = self.attention.flash_sparse_decode_and_prefill(
                q, k, v,
                max_seqlen_q=12,
                max_seqlen_k=12,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqlens_k=seqlens_k,
            )

        assert output.shape[0] == total_q
        assert output.dtype == torch.bfloat16

    def test_decode_path(self):
        """Test decode branch (max_seqlen_q == 1)."""
        num_heads = self.config.num_attention_heads
        head_dim = self.config.kv_channels
        batch_size = 3
        total_k = 30

        self.attention.cuda()
        self.attention.eval()

        q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(total_k, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(total_k, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

        cu_seqlens_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device='cuda')
        cu_seqlens_k = torch.tensor([0, 10, 20, 30], dtype=torch.int32, device='cuda')
        seqlens_k = torch.tensor([10, 10, 10], dtype=torch.int32, device='cuda')

        with torch.no_grad():
            output = self.attention.flash_sparse_decode_and_prefill(
                q, k, v,
                max_seqlen_q=1,
                max_seqlen_k=10,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqlens_k=seqlens_k,
            )

        assert output.shape[0] == batch_size
        assert output.dtype == torch.bfloat16


class TestFlashSparseNumerics:
    """Compare flash_sparse output against unfused DotProductAttention."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sparse_vs_unfused(self):
        seq_len = 32
        batch_size = 2

        # Build sparse attention
        sparse_config = _make_config(attention_backend=AttnBackend.flash_sparse)
        sparse_attn = SelfAttention(
            sparse_config,
            _local_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        # Build unfused attention
        unfused_config = _make_config(attention_backend=AttnBackend.unfused)
        unfused_attn = SelfAttention(
            unfused_config,
            _local_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        # Copy weights so both have identical parameters
        unfused_attn.load_state_dict(sparse_attn.state_dict())

        hidden_states = torch.randn(
            seq_len, batch_size, sparse_config.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )
        attention_mask = torch.ones(
            batch_size, 1, 1, seq_len, dtype=bool, device='cuda',
        )

        sparse_attn.eval()
        unfused_attn.eval()

        with torch.no_grad():
            out_sparse, _ = sparse_attn(hidden_states, attention_mask)
            out_unfused, _ = unfused_attn(hidden_states, attention_mask)

        torch.testing.assert_close(out_sparse, out_unfused, atol=5e-2, rtol=1e-1)


class TestFlashSparseWithThreshold:
    """Test that sparse_softmax_threshold config is respected."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_different_thresholds_produce_different_output(self):
        seq_len = 512
        batch_size = 2

        config_low = _make_config(sparse_softmax_threshold=0.0)
        config_high = _make_config(sparse_softmax_threshold=0.9)

        attn_low = SelfAttention(
            config_low, _local_attn_submodules(),
            layer_number=1, attn_mask_type=AttnMaskType.causal,
        ).cuda()
        attn_high = SelfAttention(
            config_high, _local_attn_submodules(),
            layer_number=1, attn_mask_type=AttnMaskType.causal,
        ).cuda()

        # Same weights
        attn_high.load_state_dict(attn_low.state_dict())

        hidden_states = torch.randn(
            seq_len, batch_size, config_low.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )
        attention_mask = torch.ones(
            batch_size, 1, 1, seq_len, dtype=bool, device='cuda',
        )

        attn_low.eval()
        attn_high.eval()

        with torch.no_grad():
            out_low, _ = attn_low(hidden_states, attention_mask)
            out_high, _ = attn_high(hidden_states, attention_mask)

        # threshold=0.0 keeps all tiles, threshold=0.9 aggressively prunes
        # At seq_len=512 this should produce measurably different outputs
        assert not torch.allclose(out_low, out_high, atol=1e-3)

    def test_long_seq_speedup_and_accuracy(self):
        """At 131072 seq length, threshold=0.0 (sparse) should be faster than
        threshold=-999 (effectively dense, no tiles pruned) while staying within atol=0.01."""
        seq_len = 131072
        batch_size = 1
        num_warmup = 3
        num_iters = 10

        config_dense = _make_config(sparse_softmax_threshold=-999.0)
        config_sparse = _make_config(sparse_softmax_threshold=0.0)

        attn_dense = SelfAttention(
            config_dense, _local_attn_submodules(),
            layer_number=1, attn_mask_type=AttnMaskType.causal,
        ).cuda()
        attn_sparse = SelfAttention(
            config_sparse, _local_attn_submodules(),
            layer_number=1, attn_mask_type=AttnMaskType.causal,
        ).cuda()

        attn_sparse.load_state_dict(attn_dense.state_dict())
        attn_dense.eval()
        attn_sparse.eval()

        hidden_states = torch.randn(
            seq_len, batch_size, config_dense.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )
        attention_mask = torch.ones(
            batch_size, 1, 1, seq_len, dtype=bool, device='cuda',
        )

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                attn_dense(hidden_states, attention_mask)
                attn_sparse(hidden_states, attention_mask)
        torch.cuda.synchronize()

        # Benchmark dense (threshold=-999, no pruning)
        torch.cuda.synchronize()
        start_dense = torch.cuda.Event(enable_timing=True)
        end_dense = torch.cuda.Event(enable_timing=True)
        start_dense.record()
        with torch.no_grad():
            for _ in range(num_iters):
                out_dense, _ = attn_dense(hidden_states, attention_mask)
        end_dense.record()
        torch.cuda.synchronize()
        time_dense = start_dense.elapsed_time(end_dense) / num_iters

        # Benchmark sparse (threshold=0.0)
        torch.cuda.synchronize()
        start_sparse = torch.cuda.Event(enable_timing=True)
        end_sparse = torch.cuda.Event(enable_timing=True)
        start_sparse.record()
        with torch.no_grad():
            for _ in range(num_iters):
                out_sparse, _ = attn_sparse(hidden_states, attention_mask)
        end_sparse.record()
        torch.cuda.synchronize()
        time_sparse = start_sparse.elapsed_time(end_sparse) / num_iters

        speedup = time_dense / time_sparse
        print(f"\n[131072 seq] dense={time_dense:.2f}ms  sparse={time_sparse:.2f}ms  "
              f"speedup={speedup:.2f}x")

        # Accuracy: outputs should be close
        max_diff = (out_dense - out_sparse).abs().max().item()
        print(f"[131072 seq] max abs diff = {max_diff:.6f}")
        assert max_diff < 0.01, f"max abs diff {max_diff} exceeds 0.01"

        # Speedup: sparse should be faster
        assert speedup > 1.0, f"no speedup: dense={time_dense:.2f}ms sparse={time_sparse:.2f}ms"
