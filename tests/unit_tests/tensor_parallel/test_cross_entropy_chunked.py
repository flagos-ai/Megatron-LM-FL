# Copyright (c) 2026, BAAI. All rights reserved.
#
# Unit tests for chunked vocab parallel cross entropy.
#
# Usage (4+ GPUs required, TP=4, supports 4 or 8 GPUs):
#   torchrun --nproc_per_node 8 tests/unit_tests/tensor_parallel/test_cross_entropy_chunked.py
#
# Or via pytest with distributed launcher:
#   pytest -xvs tests/unit_tests/tensor_parallel/test_cross_entropy_chunked.py


import sys
import os
import time
from typing import Tuple

import torch

# Add project root to path so 'tests' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from megatron.core import parallel_state
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy_chunked,
)
from tests.unit_tests.test_utilities import Utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_inputs(
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    tp_size: int,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic logits and target tensors.

    Returns:
        logits: [seq_len, batch_size, vocab_size / tp_size], dtype
        target: [seq_len, batch_size], int64
    """
    device = torch.cuda.current_device()
    # Use TP rank (not global rank) to slice vocab correctly when DP > 1
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    # Use same seed on all ranks to generate consistent full logits,
    # then slice by TP rank to simulate TP partition.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    full_logits = torch.randn(
        seq_len, batch_size, vocab_size, generator=gen, dtype=dtype
    )
    target = torch.randint(
        0, vocab_size, (seq_len, batch_size), generator=gen
    )

    # Slice vocab dimension for this TP rank
    partition_size = vocab_size // tp_size
    start = tp_rank * partition_size
    end = start + partition_size
    logits = full_logits[:, :, start:end].contiguous().cuda()
    target = target.cuda()

    return logits, target


def _measure_peak_memory(
    fn,
    *args,
    warmup: int = 2,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """Run fn and measure peak GPU memory allocated during execution.

    Returns:
        result: output of fn
        peak_memory_mb: peak memory in MiB
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if result.requires_grad:
            result.sum().backward()

    # Reset peak stats
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    # Forward
    result = fn(*args, **kwargs)
    # Backward
    if result.requires_grad:
        result.sum().backward()

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() - mem_before

    return result, peak_memory / (1024 * 1024)  # Convert to MiB


def _measure_time(
    fn,
    *args,
    warmup: int = 3,
    repeats: int = 10,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """Measure average execution time (forward + backward) in milliseconds.

    Returns:
        result: output of fn (from last run)
        avg_time_ms: average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if result.requires_grad:
            result.sum().backward()

    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if result.requires_grad:
            result.sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_time_ms = sum(times) / len(times)
    return result, avg_time_ms


# ---------------------------------------------------------------------------
# Test 1: Correctness — chunked output matches baseline
# ---------------------------------------------------------------------------

def test_chunked_cross_entropy_correctness():
    """Verify chunked cross entropy produces identical results to baseline."""
    tp_size = 4
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    seq_len = 8192
    batch_size = 1
    vocab_size = 248320

    logits, target = _generate_inputs(seq_len, batch_size, vocab_size, tp_size)

    # --- Forward correctness ---
    logits_baseline = logits.clone().requires_grad_(True)
    logits_chunked = logits.clone().requires_grad_(True)

    loss_baseline = vocab_parallel_cross_entropy(logits_baseline, target)
    loss_chunked = vocab_parallel_cross_entropy_chunked(
        logits_chunked, target, chunk_size=2048
    )

    # Check forward outputs match
    forward_max_diff = (loss_baseline - loss_chunked).abs().max().item()
    forward_passed = forward_max_diff < 1e-5

    # --- Backward correctness ---
    loss_baseline.sum().backward()
    loss_chunked.sum().backward()

    grad_baseline = logits_baseline.grad
    grad_chunked = logits_chunked.grad

    backward_max_diff = (grad_baseline - grad_chunked).abs().max().item()
    backward_passed = backward_max_diff < 1e-4

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("TEST 1: Correctness")
        print("=" * 70)
        print(f"  Forward  max diff: {forward_max_diff:.2e}  "
              f"{'PASSED' if forward_passed else 'FAILED'}")
        print(f"  Backward max diff: {backward_max_diff:.2e}  "
              f"{'PASSED' if backward_passed else 'FAILED'}")
        print("=" * 70)

    assert forward_passed, f"Forward mismatch: max_diff={forward_max_diff}"
    assert backward_passed, f"Backward mismatch: max_diff={backward_max_diff}"

    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Test 2: Correctness with label smoothing
# ---------------------------------------------------------------------------

def test_chunked_cross_entropy_label_smoothing():
    """Verify chunked cross entropy matches baseline with label smoothing."""
    tp_size = 4
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    seq_len = 4096
    batch_size = 2
    vocab_size = 248320
    label_smoothing = 0.1

    logits, target = _generate_inputs(seq_len, batch_size, vocab_size, tp_size)

    logits_baseline = logits.clone().requires_grad_(True)
    logits_chunked = logits.clone().requires_grad_(True)

    loss_baseline = vocab_parallel_cross_entropy(
        logits_baseline, target, label_smoothing=label_smoothing
    )
    loss_chunked = vocab_parallel_cross_entropy_chunked(
        logits_chunked, target, label_smoothing=label_smoothing, chunk_size=1024
    )

    forward_max_diff = (loss_baseline - loss_chunked).abs().max().item()
    forward_passed = forward_max_diff < 1e-5

    loss_baseline.sum().backward()
    loss_chunked.sum().backward()

    backward_max_diff = (logits_baseline.grad - logits_chunked.grad).abs().max().item()
    backward_passed = backward_max_diff < 1e-4

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("TEST 2: Correctness with label smoothing")
        print("=" * 70)
        print(f"  Forward  max diff: {forward_max_diff:.2e}  "
              f"{'PASSED' if forward_passed else 'FAILED'}")
        print(f"  Backward max diff: {backward_max_diff:.2e}  "
              f"{'PASSED' if backward_passed else 'FAILED'}")
        print("=" * 70)

    assert forward_passed, f"Forward mismatch: max_diff={forward_max_diff}"
    assert backward_passed, f"Backward mismatch: max_diff={backward_max_diff}"

    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Test 3: Correctness with different chunk sizes
# ---------------------------------------------------------------------------

def test_chunked_cross_entropy_various_chunk_sizes():
    """Verify different chunk sizes all produce correct results."""
    tp_size = 4
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    seq_len = 8192
    batch_size = 1
    vocab_size = 248320
    chunk_sizes = [1024, 2048, 4096, 8192]  # Including chunk == seq_len

    logits, target = _generate_inputs(seq_len, batch_size, vocab_size, tp_size)

    # Baseline
    logits_baseline = logits.clone().requires_grad_(True)
    loss_baseline = vocab_parallel_cross_entropy(logits_baseline, target)
    loss_baseline.sum().backward()

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("TEST 3: Various chunk sizes")
        print("=" * 70)

    all_passed = True
    for chunk_size in chunk_sizes:
        logits_chunked = logits.clone().requires_grad_(True)
        loss_chunked = vocab_parallel_cross_entropy_chunked(
            logits_chunked, target, chunk_size=chunk_size
        )
        loss_chunked.sum().backward()

        fwd_diff = (loss_baseline - loss_chunked).abs().max().item()
        bwd_diff = (logits_baseline.grad - logits_chunked.grad).abs().max().item()
        passed = fwd_diff < 1e-5 and bwd_diff < 1e-4

        if rank == 0:
            status = "PASSED" if passed else "FAILED"
            print(f"  chunk_size={chunk_size:5d}: "
                  f"fwd_diff={fwd_diff:.2e}, bwd_diff={bwd_diff:.2e}  {status}")

        all_passed = all_passed and passed

    if rank == 0:
        print("=" * 70)

    assert all_passed, "Some chunk sizes failed correctness check"

    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Test 4: Performance — execution time comparison
# ---------------------------------------------------------------------------

def test_chunked_cross_entropy_performance():
    """Compare execution time of baseline vs chunked implementations."""
    tp_size = 4
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    seq_len = 28672
    batch_size = 1
    vocab_size = 248320

    logits, target = _generate_inputs(seq_len, batch_size, vocab_size, tp_size)

    # Baseline timing
    logits_baseline = logits.clone().requires_grad_(True)
    _, baseline_time = _measure_time(
        vocab_parallel_cross_entropy,
        logits_baseline, target,
        warmup=3, repeats=10,
    )

    # Chunked timing (various chunk sizes)
    chunk_sizes = [2048, 4096, 8192]

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("TEST 4: Performance (forward + backward)")
        print("=" * 70)
        print(f"  Config: seq_len={seq_len}, batch={batch_size}, "
              f"vocab/TP={vocab_size // tp_size}, TP={tp_size}")
        print(f"  Baseline:               {baseline_time:8.2f} ms")

    for chunk_size in chunk_sizes:
        logits_chunked = logits.clone().requires_grad_(True)
        _, chunked_time = _measure_time(
            vocab_parallel_cross_entropy_chunked,
            logits_chunked, target,
            warmup=3, repeats=10,
            chunk_size=chunk_size,
        )
        slowdown = chunked_time / baseline_time

        if rank == 0:
            print(f"  Chunked (chunk={chunk_size:5d}): {chunked_time:8.2f} ms  "
                  f"(slowdown: {slowdown:.2f}x)")

    if rank == 0:
        print("=" * 70)

    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Test 5: Memory — peak memory comparison
# ---------------------------------------------------------------------------

def test_chunked_cross_entropy_memory():
    """Compare peak memory usage of baseline vs chunked implementations."""
    tp_size = 4
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    seq_len = 28672
    batch_size = 1
    vocab_size = 248320
    partition_vocab_size = vocab_size // tp_size

    logits, target = _generate_inputs(seq_len, batch_size, vocab_size, tp_size)

    # Expected sizes for reference
    bf16_input_mib = seq_len * batch_size * partition_vocab_size * 2 / (1024 * 1024)
    fp32_full_mib = seq_len * batch_size * partition_vocab_size * 4 / (1024 * 1024)

    # Baseline memory
    logits_baseline = logits.clone().requires_grad_(True)
    _, baseline_peak_mib = _measure_peak_memory(
        vocab_parallel_cross_entropy,
        logits_baseline, target,
        warmup=2,
    )

    # Chunked memory (various chunk sizes)
    chunk_sizes = [2048, 4096, 8192]

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("TEST 5: Peak Memory (forward + backward)")
        print("=" * 70)
        print(f"  Config: seq_len={seq_len}, batch={batch_size}, "
              f"vocab/TP={partition_vocab_size}, TP={tp_size}")
        print(f"  Reference: bf16 input = {bf16_input_mib:.1f} MiB, "
              f"fp32 full = {fp32_full_mib:.1f} MiB")
        print(f"  Baseline:               {baseline_peak_mib:10.1f} MiB")

    for chunk_size in chunk_sizes:
        logits_chunked = logits.clone().requires_grad_(True)
        _, chunked_peak_mib = _measure_peak_memory(
            vocab_parallel_cross_entropy_chunked,
            logits_chunked, target,
            warmup=2,
            chunk_size=chunk_size,
        )
        saved_mib = baseline_peak_mib - chunked_peak_mib
        saved_pct = (saved_mib / baseline_peak_mib * 100) if baseline_peak_mib > 0 else 0

        if rank == 0:
            print(f"  Chunked (chunk={chunk_size:5d}): {chunked_peak_mib:10.1f} MiB  "
                  f"(saved {saved_mib:.1f} MiB / {saved_pct:.1f}%)")

    if rank == 0:
        print("=" * 70)

    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

def test_chunked_cross_entropy_edge_cases():
    """Test edge cases: seq_len not divisible by chunk_size, chunk > seq_len."""
    tp_size = 4
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    test_cases = [
        # (seq_len, chunk_size, description)
        (100, 4096, "chunk_size > seq_len"),
        (5000, 4096, "seq_len not divisible by chunk_size"),
        (4096, 4096, "seq_len == chunk_size"),
        (1, 4096, "seq_len == 1"),
    ]

    vocab_size = 248320
    batch_size = 1

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("TEST 6: Edge cases")
        print("=" * 70)

    all_passed = True
    for seq_len, chunk_size, desc in test_cases:
        logits, target = _generate_inputs(seq_len, batch_size, vocab_size, tp_size)

        logits_baseline = logits.clone().requires_grad_(True)
        logits_chunked = logits.clone().requires_grad_(True)

        loss_baseline = vocab_parallel_cross_entropy(logits_baseline, target)
        loss_chunked = vocab_parallel_cross_entropy_chunked(
            logits_chunked, target, chunk_size=chunk_size
        )

        fwd_diff = (loss_baseline - loss_chunked).abs().max().item()

        loss_baseline.sum().backward()
        loss_chunked.sum().backward()

        bwd_diff = (logits_baseline.grad - logits_chunked.grad).abs().max().item()
        passed = fwd_diff < 1e-5 and bwd_diff < 1e-4

        if rank == 0:
            status = "PASSED" if passed else "FAILED"
            print(f"  {desc:40s}: fwd={fwd_diff:.2e}, bwd={bwd_diff:.2e}  {status}")

        all_passed = all_passed and passed

    if rank == 0:
        print("=" * 70)

    assert all_passed, "Some edge cases failed"

    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_chunked_cross_entropy_correctness()
    test_chunked_cross_entropy_label_smoothing()
    test_chunked_cross_entropy_various_chunk_sizes()
    test_chunked_cross_entropy_performance()
    test_chunked_cross_entropy_memory()
    test_chunked_cross_entropy_edge_cases()

    rank = torch.distributed.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
