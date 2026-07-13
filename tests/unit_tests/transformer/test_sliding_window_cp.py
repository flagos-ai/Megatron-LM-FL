"""
Unit test for sliding window attention + context parallel (contiguous split).

Verifies that:
1. _exchange_kv_halo correctly exchanges boundary tokens between ranks
2. get_window_topk_idxs_cp generates correct indices with halo offset
3. The full CP sliding window path produces the same output as the non-CP baseline

Usage:
    torchrun --nproc_per_node=2 test_sliding_window_cp.py
"""

import os
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def teardown():
    dist.destroy_process_group()


def test_exchange_kv_halo():
    """Test that halo exchange correctly transfers boundary tokens."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    local_seq_len = 2048
    batch_size = 2
    head_dim = 128
    halo_size = 127  # window_size - 1

    # Each rank has KV with values = rank * 1000 + position
    kv = torch.arange(local_seq_len, device=device, dtype=torch.float32)
    kv = kv.unsqueeze(1).unsqueeze(2).expand(-1, batch_size, head_dim).clone()
    kv += rank * 10000  # make values distinguishable per rank

    from megatron.core.transformer.experimental_attention_variant.csa_utils import _exchange_halo

    cp_group = dist.group.WORLD
    kv_with_halo = _exchange_halo(kv, halo_size, cp_group)

    if rank == 0:
        # rank 0 should not have halo (no predecessor)
        assert (
            kv_with_halo.shape[0] == local_seq_len
        ), f"rank 0 should have no halo, got shape {kv_with_halo.shape[0]}"
        assert torch.equal(kv_with_halo, kv)
        print(f"[rank {rank}] halo exchange: PASS (no halo)")
    else:
        # rank > 0 should have halo prepended
        assert (
            kv_with_halo.shape[0] == halo_size + local_seq_len
        ), f"rank {rank} expected {halo_size + local_seq_len}, got {kv_with_halo.shape[0]}"
        # The halo should be the last halo_size tokens from the previous rank
        expected_halo_start = (rank - 1) * 10000 + (local_seq_len - halo_size)
        actual_halo_start = kv_with_halo[0, 0, 0].item()
        assert (
            abs(actual_halo_start - expected_halo_start) < 1e-5
        ), f"rank {rank} halo start mismatch: expected {expected_halo_start}, got {actual_halo_start}"
        # Local part should be unchanged
        assert torch.equal(kv_with_halo[halo_size:], kv)
        print(f"[rank {rank}] halo exchange: PASS")


def test_window_indices_cp():
    """Test that CP-aware window indices are correct."""
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    from megatron.core.transformer.experimental_attention_variant.csa import get_window_topk_idxs
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        get_window_topk_idxs_cp,
    )

    window_size = 128
    local_seq_len = 2048
    batch_size = 1

    # rank 0: halo_size = 0 (same as original)
    if rank == 0:
        idxs_orig = get_window_topk_idxs(window_size, batch_size, local_seq_len, device)
        idxs_cp = get_window_topk_idxs_cp(window_size, batch_size, local_seq_len, 0, device)
        assert torch.equal(idxs_orig, idxs_cp), "rank 0 CP indices should match original"
        print(f"[rank {rank}] window indices: PASS (matches original)")

    # rank 1: halo_size = window_size - 1
    if rank == 1:
        halo_size = window_size - 1
        idxs_cp = get_window_topk_idxs_cp(window_size, batch_size, local_seq_len, halo_size, device)

        # For query position 0 on rank 1 (global position = local_seq_len):
        # Its position in kv_with_halo is 0 + halo_size = 127
        # Its window should be [127 - 127, ..., 127] = [0, 1, ..., 127]
        expected_first_row = torch.arange(window_size, device=device)
        actual_first_row = idxs_cp[0, 0, :]
        assert torch.equal(actual_first_row, expected_first_row), (
            f"rank 1 first query window mismatch:\n"
            f"  expected: {expected_first_row[:5]}...{expected_first_row[-5:]}\n"
            f"  actual:   {actual_first_row[:5]}...{actual_first_row[-5:]}"
        )

        # For query position 1 on rank 1:
        # Its position in kv_with_halo is 1 + halo_size = 128
        # Its window should be [1, 2, ..., 128]
        expected_second_row = torch.arange(1, window_size + 1, device=device)
        actual_second_row = idxs_cp[0, 1, :]
        assert torch.equal(
            actual_second_row, expected_second_row
        ), f"rank 1 second query window mismatch"

        # No -1 values should exist (halo provides full history)
        assert (idxs_cp >= 0).all(), "rank 1 should have no -1 padding with full halo"
        print(f"[rank {rank}] window indices: PASS")


def test_full_sliding_window_cp():
    """End-to-end test: CP output should match non-CP baseline."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    from megatron.core.transformer.experimental_attention_variant.csa import (
        get_window_topk_idxs,
        unfused_compressed_sparse_attn,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        _exchange_halo,
        get_window_topk_idxs_cp,
    )

    torch.manual_seed(42)

    seq_len = 4096
    local_seq_len = seq_len // world_size
    batch_size = 1
    num_heads = 16
    head_dim = 128
    window_size = 128

    # Generate full-sequence Q and KV on all ranks (for baseline comparison)
    query_full = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device)
    kv_full = torch.randn(seq_len, batch_size, head_dim, device=device)
    attn_sink = torch.zeros(num_heads, device=device)
    softmax_scale = head_dim**-0.5

    # --- Baseline: non-CP (full sequence) ---
    window_idxs_full = get_window_topk_idxs(window_size, batch_size, seq_len, device).int()
    output_baseline = unfused_compressed_sparse_attn(
        query_full, kv_full, attn_sink, window_idxs_full, softmax_scale
    )

    # --- CP path: each rank processes its local chunk ---
    start = rank * local_seq_len
    end = start + local_seq_len

    query_local = query_full[start:end].contiguous()
    kv_local = kv_full[start:end].contiguous()

    # Halo exchange
    cp_group = dist.group.WORLD
    halo_size = window_size - 1 if rank > 0 else 0
    kv_with_halo = _exchange_halo(kv_local, window_size - 1, cp_group)

    # CP-aware window indices
    window_idxs_cp = get_window_topk_idxs_cp(
        window_size, batch_size, local_seq_len, halo_size, device
    ).int()

    # Sparse attention on local chunk with halo
    output_cp_local = unfused_compressed_sparse_attn(
        query_local, kv_with_halo, attn_sink, window_idxs_cp, softmax_scale
    )

    # Compare with baseline slice
    output_baseline_local = output_baseline[start:end]

    max_diff = (output_cp_local - output_baseline_local).abs().max().item()
    mean_diff = (output_cp_local - output_baseline_local).abs().mean().item()

    passed = max_diff < 1e-4
    status = "PASS" if passed else "FAIL"
    print(
        f"[rank {rank}] full sliding window CP: {status} "
        f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
    )
    assert passed, f"rank {rank} output mismatch: max_diff={max_diff}"


if __name__ == "__main__":
    setup()
    try:
        test_exchange_kv_halo()
        dist.barrier()
        test_window_indices_cp()
        dist.barrier()
        test_full_sliding_window_cp()
        dist.barrier()
        if dist.get_rank() == 0:
            print("\n=== All tests PASSED ===")
    finally:
        teardown()
