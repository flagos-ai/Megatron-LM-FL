# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Flash Sparse Attention Headwise Context Parallel."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.flash_sparse_attention import _fsa_headwise_cp_forward
from tests.unit_tests.test_utilities import Utils


def _gather_tensor_across_cp_group(tensor, cp_group):
    """Gather tensors from all CP ranks for comparison."""
    world_size = torch.distributed.get_world_size(cp_group)

    # Use all_gather so every rank gets the full result
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor.contiguous(), group=cp_group)

    # Concatenate along sequence dimension
    return torch.cat(gathered, dim=0)


@pytest.mark.internal
class TestFSAHeadwiseCP:
    """Test FSA headwise context parallel with hybrid communication mode."""

    @pytest.mark.parametrize(
        "tp_size, cp_size, num_q_heads_global, num_kv_heads_global, seq_len, batch_size, head_dim",
        [
            # Case 1: CP only (tp_size=1), sufficient KV heads
            (1, 2, 16, 8, 128, 2, 64),   # cp_size < num_kv_heads_per_tp (8)
            (1, 4, 32, 16, 256, 1, 64),  # cp_size < num_kv_heads_per_tp (16)

            # Case 2: CP only (tp_size=1), insufficient KV heads
            (1, 4, 16, 2, 128, 2, 64),   # cp_size (4) > num_kv_heads_per_tp (2)
            (1, 8, 32, 4, 256, 1, 128),  # cp_size (8) > num_kv_heads_per_tp (4)

            # Case 3: TP + CP, sufficient KV heads
            (2, 2, 16, 8, 128, 2, 64),   # num_kv_heads_per_tp (4) >= cp_size (2)
            (4, 2, 32, 16, 256, 1, 64),  # num_kv_heads_per_tp (4) >= cp_size (2)

            # Case 4: TP + CP, insufficient KV heads
            (2, 4, 16, 4, 128, 2, 64),   # num_kv_heads_per_tp (2) < cp_size (4)
            (4, 4, 32, 8, 256, 1, 128),  # num_kv_heads_per_tp (2) < cp_size (4)

            # Case 5: 35B model config variants
            # Real config: Q=16, KV=2, tp=4 → per TP rank: 4 Q heads, 1 KV head
            # seq=32768, batch=1, head_dim=256 (scaled down for test)
            (2, 1, 16, 2, 16384, 1, 64), # CP=1, no CP baseline, 4 Q per TP rank
            (2, 2, 16, 2, 16384, 1, 64), # CP=2, kv_per_tp(1) < cp(2), 2 Q per CP rank
            (2, 4, 16, 2, 16384, 1, 64), # TP=2, CP=4, kv_per_tp(1) < cp(4), 2 Q per CP rank
        ],
    )
    def test_cp_vs_no_cp_numerical_equivalence(
        self, tp_size, cp_size, num_q_heads_global, num_kv_heads_global,
        seq_len, batch_size, head_dim
    ):
        """
        Test numerical equivalence between CP and no-CP execution.

        This is the critical correctness test:
        - Case 1 (no CP): Run FSA on full sequence without CP
        - Case 2 (with CP): Run FSA with CP enabled
        - Compare: Results should be numerically identical

        Covers:
        1. CP only (tp_size=1)
        2. TP + CP (tp_size>1)
        3. cp_size > num_kv_heads_per_tp (AllGather mode)
        4. cp_size <= num_kv_heads_per_tp (all-to-all mode)
        """
        world_size = tp_size * cp_size

        # Skip if not enough GPUs
        if torch.cuda.device_count() < world_size:
            pytest.skip(
                f"Requires {world_size} GPUs (tp={tp_size}, cp={cp_size}), "
                f"but only {torch.cuda.device_count()} available"
            )

        dtype = torch.bfloat16
        device = torch.device("cuda")

        # Validate configuration
        assert num_q_heads_global % tp_size == 0
        assert num_kv_heads_global % tp_size == 0

        num_q_heads_per_tp = num_q_heads_global // tp_size
        num_kv_heads_per_tp = num_kv_heads_global // tp_size

        assert num_q_heads_per_tp % cp_size == 0, (
            f"Q heads per TP ({num_q_heads_per_tp}) must be divisible by CP size ({cp_size})"
        )

        # ==================================================================
        # Case 1: No CP (baseline) - cp_size=1
        # ==================================================================
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=1
        )

        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # Create full sequence inputs
        torch.manual_seed(42 + tp_rank)  # Same seed per TP rank
        query_full = torch.randn(
            seq_len, batch_size, num_q_heads_per_tp, head_dim,
            device=device, dtype=dtype
        )
        key_full = torch.randn(
            seq_len, batch_size, num_kv_heads_per_tp, head_dim,
            device=device, dtype=dtype
        )
        value_full = torch.randn(
            seq_len, batch_size, num_kv_heads_per_tp, head_dim,
            device=device, dtype=dtype
        )

        window_sizes = torch.tensor(
            [[32, 0, 32, 0]] * num_kv_heads_per_tp,
            device=device, dtype=torch.int32
        )

        # Run without CP
        try:
            from flash_sparse_attn import flash_sparse_attn_func
        except ImportError as e:
            pytest.skip(f"flash_sparse_attn library not available: {e}")

        # Batch processing (matching real training scenario)
        output_no_cp = flash_sparse_attn_func(
            query_full,
            key_full,
            value_full,
            window_sizes=window_sizes,
            softmax_threshold=0.5,
            pack_gqa=False,  # Support GQA (Q heads != KV heads)
        )  # [seq, batch, num_q_heads_per_tp, head_dim]

        Utils.destroy_model_parallel()
        torch.cuda.synchronize()

        # ==================================================================
        # Case 2: With CP
        # ==================================================================
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size
        )

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_group = parallel_state.get_context_parallel_group()

        # Split sequence across CP ranks
        seq_len_local = seq_len // cp_size
        seq_start = cp_rank * seq_len_local
        seq_end = seq_start + seq_len_local

        # Use same seed to ensure same inputs
        torch.manual_seed(42 + tp_rank)
        query_full_cp = torch.randn(
            seq_len, batch_size, num_q_heads_per_tp, head_dim,
            device=device, dtype=dtype
        )
        key_full_cp = torch.randn(
            seq_len, batch_size, num_kv_heads_per_tp, head_dim,
            device=device, dtype=dtype
        )
        value_full_cp = torch.randn(
            seq_len, batch_size, num_kv_heads_per_tp, head_dim,
            device=device, dtype=dtype
        )

        # Extract local sequence chunk
        query_local = query_full_cp[seq_start:seq_end].contiguous()
        key_local = key_full_cp[seq_start:seq_end].contiguous()
        value_local = value_full_cp[seq_start:seq_end].contiguous()

        # Run with CP
        output_cp_local = _fsa_headwise_cp_forward(
            query=query_local,
            key=key_local,
            value=value_local,
            window_sizes=window_sizes,
            cp_group=cp_group,
            cp_size=cp_size,
            num_q_heads_per_tp=num_q_heads_per_tp,
            num_kv_heads_per_tp=num_kv_heads_per_tp,
            softmax_threshold=0.5,
        )

        # Gather output from all CP ranks
        output_cp_full = _gather_tensor_across_cp_group(output_cp_local, cp_group)

        # ==================================================================
        # Numerical comparison (on CP rank 0 only)
        # ==================================================================
        if cp_rank == 0:
            # Check shapes
            assert output_no_cp.shape == output_cp_full.shape, (
                f"Shape mismatch: no_cp {output_no_cp.shape} vs cp {output_cp_full.shape}"
            )

            # Compute differences
            abs_diff = torch.abs(output_no_cp - output_cp_full)
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()

            rel_diff = abs_diff / (torch.abs(output_no_cp) + 1e-8)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            # Print comparison stats
            print(f"\n{'='*60}")
            print(f"Numerical Equivalence Test")
            print(f"TP={tp_size}, CP={cp_size}")
            print(f"Q heads: {num_q_heads_global} global, {num_q_heads_per_tp} per TP")
            print(f"KV heads: {num_kv_heads_global} global, {num_kv_heads_per_tp} per TP")
            print(f"Communication mode: {'AllGather' if num_kv_heads_per_tp < cp_size else 'all-to-all'}")
            print(f"{'-'*60}")
            print(f"Max absolute diff: {max_abs_diff:.6e}")
            print(f"Mean absolute diff: {mean_abs_diff:.6e}")
            print(f"Max relative diff: {max_rel_diff:.6e}")
            print(f"Mean relative diff: {mean_rel_diff:.6e}")
            print(f"{'='*60}\n")

            # Assert numerical equivalence
            # For bfloat16, use 5e-3 tolerance (matching Megatron GDN tests)
            atol, rtol = 5e-3, 5e-3
            assert max_abs_diff < atol, (
                f"Max absolute difference {max_abs_diff} exceeds threshold {atol}"
            )
            assert max_rel_diff < rtol, (
                f"Max relative difference {max_rel_diff} exceeds threshold {rtol}"
            )

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "cp_size, num_q_heads_global, num_kv_heads_global, seq_len, batch_size, head_dim",
        [
            # Long sequence scenario to highlight memory savings
            (2, 16, 8, 4096, 2, 128),
            (4, 32, 8, 8192, 1, 128),
            (8, 64, 8, 16384, 1, 128),
        ],
    )
    def test_cp_memory_reduction(
        self, cp_size, num_q_heads_global, num_kv_heads_global,
        seq_len, batch_size, head_dim
    ):
        """
        Test that enabling CP reduces per-GPU memory usage compared to no-CP.

        In no-CP mode, each GPU holds the full sequence for all Q/KV heads.
        In CP mode (headwise), each GPU only processes a subset of Q heads
        on the full sequence, so the activation memory should be smaller.

        Expected memory relationship:
        - No CP: each GPU holds [seq_len, b, num_q_heads, hn] for Q/K/V/output
        - With CP: each GPU holds [seq_len/cp_size, b, num_q_heads, hn] as input,
          and intermediate activations are [seq_len, b, 1, hn] (single head)
        """
        if torch.cuda.device_count() < cp_size:
            pytest.skip(
                f"Requires {cp_size} GPUs, but only {torch.cuda.device_count()} available"
            )

        try:
            from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_func
        except ImportError as e:
            pytest.skip(f"flash_sparse_attn library not available: {e}")

        dtype = torch.bfloat16
        device = torch.device("cuda")
        num_q_heads_per_tp = num_q_heads_global  # tp_size=1 for simplicity
        num_kv_heads_per_tp = num_kv_heads_global

        def _get_peak_memory_no_cp():
            """Measure peak memory for no-CP (full sequence, all heads)."""
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            mem_before = torch.cuda.memory_allocated(device)

            # Simulate no-CP: full sequence, all heads on one GPU
            query = torch.randn(seq_len, batch_size, num_q_heads_per_tp, head_dim,
                               device=device, dtype=dtype)
            key = torch.randn(seq_len, batch_size, num_kv_heads_per_tp, head_dim,
                             device=device, dtype=dtype)
            value = torch.randn(seq_len, batch_size, num_kv_heads_per_tp, head_dim,
                               device=device, dtype=dtype)
            window_sizes = torch.tensor(
                [[64, 0, 64, 0]] * num_kv_heads_per_tp,
                device=device, dtype=torch.int32
            )

            output = flash_sparse_attn_func(
                query, key, value,
                window_sizes=window_sizes,
                softmax_threshold=0.5,
                pack_gqa=False,
            )

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device)

            # Cleanup
            del query, key, value, output, window_sizes
            torch.cuda.empty_cache()

            return peak_mem - mem_before

        def _get_peak_memory_with_cp():
            """Measure peak memory for CP (local sequence, headwise split)."""
            Utils.initialize_model_parallel(context_parallel_size=cp_size)
            cp_group = parallel_state.get_context_parallel_group()

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            mem_before = torch.cuda.memory_allocated(device)

            # With CP: each rank holds local sequence slice
            seq_len_local = seq_len // cp_size
            query = torch.randn(seq_len_local, batch_size, num_q_heads_per_tp, head_dim,
                               device=device, dtype=dtype)
            key = torch.randn(seq_len_local, batch_size, num_kv_heads_per_tp, head_dim,
                             device=device, dtype=dtype)
            value = torch.randn(seq_len_local, batch_size, num_kv_heads_per_tp, head_dim,
                               device=device, dtype=dtype)
            window_sizes = torch.tensor(
                [[64, 0, 64, 0]] * num_kv_heads_per_tp,
                device=device, dtype=torch.int32
            )

            output = _fsa_headwise_cp_forward(
                query, key, value, window_sizes,
                cp_group, cp_size,
                num_q_heads_per_tp, num_kv_heads_per_tp
            )

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device)

            # Cleanup
            del query, key, value, output, window_sizes
            torch.cuda.empty_cache()
            Utils.destroy_model_parallel()

            return peak_mem - mem_before

        # Measure memory for both cases
        peak_mem_no_cp = _get_peak_memory_no_cp()
        peak_mem_with_cp = _get_peak_memory_with_cp()

        # Report
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            mem_no_cp_mb = peak_mem_no_cp / (1024 * 1024)
            mem_with_cp_mb = peak_mem_with_cp / (1024 * 1024)
            reduction_pct = (1 - peak_mem_with_cp / peak_mem_no_cp) * 100

            print(f"\n{'='*60}")
            print(f"Memory Usage Comparison (cp_size={cp_size})")
            print(f"  seq_len={seq_len}, batch={batch_size}, "
                  f"q_heads={num_q_heads_per_tp}, kv_heads={num_kv_heads_per_tp}, "
                  f"head_dim={head_dim}")
            print(f"{'='*60}")
            print(f"  No CP peak memory:   {mem_no_cp_mb:.2f} MB")
            print(f"  With CP peak memory: {mem_with_cp_mb:.2f} MB")
            print(f"  Memory reduction:    {reduction_pct:.1f}%")
            print(f"{'='*60}")

        # Assert that CP reduces memory
        assert peak_mem_with_cp < peak_mem_no_cp, (
            f"CP should reduce per-GPU memory, but got "
            f"no_cp={peak_mem_no_cp / (1024**2):.2f}MB vs "
            f"with_cp={peak_mem_with_cp / (1024**2):.2f}MB"
        )

    def test_fsa_headwise_cp_q_heads_not_divisible(self):
        """Test that error is raised when Q heads are not divisible by CP size."""
        cp_size = 4
        num_q_heads_per_tp = 7  # Not divisible by 4
        num_kv_heads_per_tp = 2

        if torch.cuda.device_count() < cp_size:
            pytest.skip(f"Requires at least {cp_size} GPUs")

        Utils.initialize_model_parallel(context_parallel_size=cp_size)

        dtype = torch.bfloat16
        device = torch.device("cuda")
        seq_len_local = 32
        batch_size = 1
        head_dim = 64

        query = torch.randn(seq_len_local, batch_size, num_q_heads_per_tp, head_dim,
                           device=device, dtype=dtype)
        key = torch.randn(seq_len_local, batch_size, num_kv_heads_per_tp, head_dim,
                         device=device, dtype=dtype)
        value = torch.randn(seq_len_local, batch_size, num_kv_heads_per_tp, head_dim,
                           device=device, dtype=dtype)
        window_sizes = torch.tensor([[32, 0, 32, 0]] * num_kv_heads_per_tp,
                                    device=device, dtype=torch.int32)

        cp_group = parallel_state.get_context_parallel_group()

        with pytest.raises(ValueError, match="Q heads per TP.*must be divisible"):
            _fsa_headwise_cp_forward(
                query, key, value, window_sizes,
                cp_group, cp_size,
                num_q_heads_per_tp, num_kv_heads_per_tp
            )

        Utils.destroy_model_parallel()
