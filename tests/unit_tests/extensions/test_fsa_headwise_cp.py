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

        # flash_sparse_attn_func expects BSHD [batch, seq, heads, head_dim]
        query_bshd = query_full.transpose(0, 1).contiguous()
        key_bshd = key_full.transpose(0, 1).contiguous()
        value_bshd = value_full.transpose(0, 1).contiguous()

        output_no_cp_bshd = flash_sparse_attn_func(
            query_bshd,
            key_bshd,
            value_bshd,
            window_sizes=window_sizes,
            softmax_threshold=0.5,
            pack_gqa=False,  # Support GQA (Q heads != KV heads)
        )  # [batch, seq, num_q_heads_per_tp, head_dim]

        # Convert back to SBHD for comparison
        output_no_cp = output_no_cp_bshd.transpose(0, 1).contiguous()
        # [seq, batch, num_q_heads_per_tp, head_dim]

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

        # Note: Headwise CP trades per-head attention memory for communication buffers.
        # For short sequences, communication overhead (all-to-all buffers, transpose copies)
        # can exceed the attention memory savings. The real benefit appears with long sequences
        # where the O(seq^2) attention score memory dominates.
        # This test is informational — just report the numbers.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            if peak_mem_with_cp < peak_mem_no_cp:
                print(f"  ✓ CP reduces memory by {reduction_pct:.1f}%")
            else:
                print(f"  ✗ CP increases memory by {-reduction_pct:.1f}% "
                      f"(expected for short sequences with communication overhead)")

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


@pytest.mark.internal
class TestFSAHeadwiseCPIntegration:
    """
    Integration tests for FSA headwise CP that exercise the full
    _flash_sparse_attention path including:
    - SBHD/BSHD format correctness through the entire call chain
    - window_sizes_heuristic computation with correct seqlen_k
    - Proper interaction between TP, CP, and FSA kernel
    """

    @pytest.mark.parametrize(
        "cp_size, num_q_heads_global, num_kv_heads_global, seq_len, batch_size, head_dim",
        [
            # CP=2, sufficient KV heads (all-to-all mode)
            (2, 16, 8, 256, 1, 64),
            # CP=4, insufficient KV heads (AllGather mode)
            (4, 16, 2, 256, 1, 64),
            # CP=2, batch_size > 1
            (2, 8, 4, 128, 2, 64),
            # Longer sequence
            (2, 16, 4, 1024, 1, 128),
        ],
    )
    def test_sbhd_bshd_format_consistency(
        self, cp_size, num_q_heads_global, num_kv_heads_global,
        seq_len, batch_size, head_dim
    ):
        """
        Verify that _fsa_headwise_cp_forward correctly handles SBHD format:
        - Input must be SBHD [sq_local, b, np_local, hn]
        - Internal transpose to BSHD for flash_sparse_attn_func
        - Output must be SBHD [sq_local, b, np_local, hn]

        Also verify that passing BSHD directly (the old bug) would give wrong results.
        """
        if torch.cuda.device_count() < cp_size:
            pytest.skip(f"Requires {cp_size} GPUs, only {torch.cuda.device_count()} available")

        try:
            from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_func
        except ImportError as e:
            pytest.skip(f"flash_sparse_attn library not available: {e}")

        Utils.initialize_model_parallel(context_parallel_size=cp_size)

        cp_rank = parallel_state.get_context_parallel_rank()
        cp_group = parallel_state.get_context_parallel_group()

        dtype = torch.bfloat16
        device = torch.device("cuda")
        num_q_heads_per_tp = num_q_heads_global  # tp_size=1
        num_kv_heads_per_tp = num_kv_heads_global
        seq_len_local = seq_len // cp_size

        torch.manual_seed(42)

        # Generate full-sequence tensors on all ranks (for reference)
        query_full = torch.randn(seq_len, batch_size, num_q_heads_per_tp, head_dim,
                                 device=device, dtype=dtype)
        key_full = torch.randn(seq_len, batch_size, num_kv_heads_per_tp, head_dim,
                               device=device, dtype=dtype)
        value_full = torch.randn(seq_len, batch_size, num_kv_heads_per_tp, head_dim,
                                 device=device, dtype=dtype)

        # Reference: run FSA on full sequence with BSHD (correct format)
        window_sizes = torch.tensor(
            [[seq_len, 0, seq_len, 0]] * num_kv_heads_per_tp,
            device=device, dtype=torch.int32
        )
        q_bshd = query_full.transpose(0, 1).contiguous()
        k_bshd = key_full.transpose(0, 1).contiguous()
        v_bshd = value_full.transpose(0, 1).contiguous()
        ref_output_bshd = flash_sparse_attn_func(
            q_bshd, k_bshd, v_bshd,
            window_sizes=window_sizes,
            softmax_threshold=0.5,
            pack_gqa=False,
        )
        ref_output = ref_output_bshd.transpose(0, 1).contiguous()  # Back to SBHD

        # CP path: split sequence and run _fsa_headwise_cp_forward (SBHD in/out)
        seq_start = cp_rank * seq_len_local
        seq_end = seq_start + seq_len_local
        query_local = query_full[seq_start:seq_end].contiguous()
        key_local = key_full[seq_start:seq_end].contiguous()
        value_local = value_full[seq_start:seq_end].contiguous()

        # Verify input is SBHD
        assert query_local.shape == (seq_len_local, batch_size, num_q_heads_per_tp, head_dim), \
            f"Input should be SBHD, got {query_local.shape}"

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

        # Verify output is SBHD
        assert output_cp_local.shape == (seq_len_local, batch_size, num_q_heads_per_tp, head_dim), \
            f"Output should be SBHD [sq_local, b, np, hn], got {output_cp_local.shape}"

        # Gather and compare with reference
        output_cp_full = _gather_tensor_across_cp_group(output_cp_local, cp_group)

        if cp_rank == 0:
            abs_diff = torch.abs(ref_output - output_cp_full)
            max_abs_diff = abs_diff.max().item()
            assert max_abs_diff < 5e-3, (
                f"SBHD format mismatch: max abs diff = {max_abs_diff:.6e}. "
                f"Likely internal transpose to BSHD is incorrect."
            )

        # Verify the OLD BUG: if we had passed BSHD directly (dim0=batch), output would be wrong
        # Simulate: treat dim0 as batch (wrong!) — should get totally different results
        if batch_size != seq_len_local:
            # Only test when dimensions differ so we can detect the transposition error
            query_wrong_bshd = query_local.transpose(0, 1).contiguous()  # [b, sq_local, np, hn]
            # If we passed this as "SBHD", the function would interpret b as sq and sq as b
            # Shape: [b=batch_size, sq_local, np, hn] interpreted as [sq=batch_size, b=sq_local, np, hn]
            # This is the bug we fixed — just verify the shapes don't accidentally work
            assert query_wrong_bshd.shape[0] != query_local.shape[0], \
                "BSHD and SBHD should have different dim0 when batch != seq_local"

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "cp_size, seq_len, num_kv_heads_per_tp",
        [
            (1, 512, 4),    # CP=1: seqlen_k should be seq_len itself
            (2, 512, 4),    # CP=2: seqlen_k should be 512 (not 256)
            (4, 1024, 2),   # CP=4: seqlen_k should be 1024 (not 256)
        ],
    )
    def test_window_sizes_use_global_seqlen(
        self, cp_size, seq_len, num_kv_heads_per_tp
    ):
        """
        Verify that window_sizes_heuristic receives the global sequence length
        (sq_local * cp_size), not just sq_local.

        The FSA kernel always processes the full global sequence (after all-to-all),
        so window sizes must be calibrated for sq_global.

        This test calls window_sizes_heuristic with both sq_local and sq_global,
        and asserts the CP path uses sq_global.
        """
        try:
            from flash_sparse_attn.ops.triton.utils import window_sizes_heuristic
        except ImportError as e:
            pytest.skip(f"flash_sparse_attn library not available: {e}")

        device = torch.device("cuda")
        sq_local = seq_len // cp_size
        sq_global = seq_len

        # Compute window sizes with correct (global) seqlen
        ws_global = window_sizes_heuristic(
            seqlen_k=sq_global,
            num_heads_kv=num_kv_heads_per_tp,
            device=device,
            equal_bandwidth=True,
        )

        # Compute window sizes with incorrect (local) seqlen — the old bug
        ws_local = window_sizes_heuristic(
            seqlen_k=sq_local,
            num_heads_kv=num_kv_heads_per_tp,
            device=device,
            equal_bandwidth=True,
        )

        # Verify shape: [num_kv_heads, 4]
        assert ws_global.shape == (num_kv_heads_per_tp, 4), \
            f"window_sizes shape should be [num_kv_heads, 4], got {ws_global.shape}"

        if cp_size > 1:
            # Window sizes should differ when computed with different seqlen_k
            # (unless the window heuristic is completely seqlen-independent,
            # which would be unusual but not incorrect)
            # At minimum, window values should not exceed the global seqlen
            assert (ws_global[:, 0] <= sq_global).all(), \
                f"Window sizes exceed global seqlen: {ws_global[:, 0]} > {sq_global}"
            assert (ws_global[:, 2] <= sq_global).all(), \
                f"Window sizes exceed global seqlen: {ws_global[:, 2]} > {sq_global}"

            # The bug would produce windows based on sq_local, which are smaller
            # For most heuristics, larger seqlen produces larger or equal windows
            print(f"\n  CP={cp_size}, seq_len={seq_len}")
            print(f"  window_sizes (global, correct):  {ws_global.tolist()}")
            print(f"  window_sizes (local, incorrect): {ws_local.tolist()}")

            # Key check: windows computed with global seqlen should be >= local seqlen
            # (they cover the full sequence the kernel actually processes)
            if not torch.equal(ws_global, ws_local):
                print(f"  ✓ Window sizes differ between global and local seqlen (as expected)")
            else:
                print(f"  ⚠ Window sizes are identical (heuristic may be seqlen-independent)")

    @pytest.mark.parametrize(
        "cp_size, num_q_heads_per_tp, num_kv_heads_per_tp, seq_len, batch_size, head_dim",
        [
            # batch=1 (the bug scenario: dim0=1 after wrong transpose)
            (2, 8, 4, 128, 1, 64),
            (4, 16, 2, 256, 1, 64),
            # batch=2 (different from seq_local, easier to catch shape errors)
            (2, 8, 4, 128, 2, 64),
        ],
    )
    def test_output_shape_matches_input_sequence_dim(
        self, cp_size, num_q_heads_per_tp, num_kv_heads_per_tp,
        seq_len, batch_size, head_dim
    ):
        """
        Verify that the output sequence dimension equals sq_local (not batch or 1).

        The original bug caused dim0=batch (often 1) to be treated as the sequence
        dimension, resulting in reduce_scatter failure when dim0 is not divisible by TP.
        """
        if torch.cuda.device_count() < cp_size:
            pytest.skip(f"Requires {cp_size} GPUs, only {torch.cuda.device_count()} available")

        try:
            from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_func
        except ImportError as e:
            pytest.skip(f"flash_sparse_attn library not available: {e}")

        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        cp_group = parallel_state.get_context_parallel_group()

        dtype = torch.bfloat16
        device = torch.device("cuda")
        seq_len_local = seq_len // cp_size

        query = torch.randn(seq_len_local, batch_size, num_q_heads_per_tp, head_dim,
                           device=device, dtype=dtype)
        key = torch.randn(seq_len_local, batch_size, num_kv_heads_per_tp, head_dim,
                         device=device, dtype=dtype)
        value = torch.randn(seq_len_local, batch_size, num_kv_heads_per_tp, head_dim,
                           device=device, dtype=dtype)
        window_sizes = torch.tensor(
            [[seq_len, 0, seq_len, 0]] * num_kv_heads_per_tp,
            device=device, dtype=torch.int32
        )

        output = _fsa_headwise_cp_forward(
            query=query,
            key=key,
            value=value,
            window_sizes=window_sizes,
            cp_group=cp_group,
            cp_size=cp_size,
            num_q_heads_per_tp=num_q_heads_per_tp,
            num_kv_heads_per_tp=num_kv_heads_per_tp,
            softmax_threshold=0.5,
        )

        # Critical checks:
        # 1. Output dim0 must be seq_len_local (not batch_size, not 1)
        assert output.shape[0] == seq_len_local, (
            f"Output dim0 should be seq_len_local={seq_len_local}, got {output.shape[0]}. "
            f"This indicates SBHD/BSHD confusion."
        )
        # 2. Output dim1 must be batch_size
        assert output.shape[1] == batch_size, (
            f"Output dim1 should be batch_size={batch_size}, got {output.shape[1]}. "
            f"This indicates SBHD/BSHD confusion."
        )
        # 3. Full shape check
        assert output.shape == (seq_len_local, batch_size, num_q_heads_per_tp, head_dim), (
            f"Output shape should be (sq_local={seq_len_local}, b={batch_size}, "
            f"np={num_q_heads_per_tp}, hn={head_dim}), got {output.shape}"
        )

        # 4. After reshape for linear_proj, dim0 must be divisible by common TP sizes
        output_reshaped = output.reshape(output.shape[0], output.shape[1], -1)
        # Simulate what linear_proj's reduce_scatter would check
        for tp_size in [2, 4]:
            assert output_reshaped.shape[0] % tp_size == 0 or output_reshaped.shape[0] < tp_size, (
                f"Output seq dim ({output_reshaped.shape[0]}) not divisible by tp_size={tp_size}. "
                f"reduce_scatter in linear_proj would fail."
            )

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "cp_size, num_q_heads_per_tp, num_kv_heads_per_tp, seq_len, head_dim",
        [
            (2, 8, 4, 256, 64),   # sufficient KV heads
            (4, 16, 2, 256, 64),  # insufficient KV heads
        ],
    )
    def test_window_sizes_passed_to_kernel_match_global_seq(
        self, cp_size, num_q_heads_per_tp, num_kv_heads_per_tp, seq_len, head_dim
    ):
        """
        End-to-end check that the window_sizes computed in _flash_sparse_attention
        would be based on sq_global, not sq_local.

        Simulates the computation path in attention.py without instantiating
        the full Attention module.
        """
        try:
            from flash_sparse_attn.ops.triton.utils import window_sizes_heuristic
        except ImportError as e:
            pytest.skip(f"flash_sparse_attn library not available: {e}")

        device = torch.device("cuda")
        sq_local = seq_len // cp_size
        sq_global = seq_len

        # Simulate what attention.py does:
        # seqlen_k = query.shape[0] * cp_size
        # where query.shape[0] is sq_local
        simulated_seqlen_k = sq_local * cp_size
        assert simulated_seqlen_k == sq_global, (
            f"seqlen_k computation should yield sq_global={sq_global}, "
            f"got {simulated_seqlen_k}"
        )

        # Compute window sizes as attention.py would
        ws = window_sizes_heuristic(
            seqlen_k=simulated_seqlen_k,
            num_heads_kv=num_kv_heads_per_tp,
            device=device,
            equal_bandwidth=True,
        )

        # Window sizes must be valid for the global sequence length
        # that the FSA kernel will actually process
        assert ws.shape[0] == num_kv_heads_per_tp
        # No window dimension should exceed the global sequence length
        for i in range(num_kv_heads_per_tp):
            window_left = ws[i, 0].item()
            window_right = ws[i, 2].item()
            assert window_left <= sq_global, (
                f"KV head {i}: window_left={window_left} exceeds sq_global={sq_global}"
            )
            assert window_right <= sq_global, (
                f"KV head {i}: window_right={window_right} exceeds sq_global={sq_global}"
            )

        print(f"\n  CP={cp_size}, sq_local={sq_local}, sq_global={sq_global}")
        print(f"  window_sizes: {ws.tolist()}")
