from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _online_stats(
    X, Y, Stats, N: tl.constexpr, V: tl.constexpr, START: tl.constexpr, C: tl.constexpr
):

    chunk = tl.cdiv(N, tl.num_programs(0))
    begin = tl.program_id(0) * chunk
    end = tl.minimum(begin + chunk, N)
    offsets = tl.arange(0, C)
    for row in tl.range(begin, end):
        ptr = X + row.to(tl.int64) * V
        label = tl.load(Y + row) - START
        local = (label >= 0) & (label < V)
        predicted = tl.load(ptr + tl.maximum(0, tl.minimum(label, V - 1)), local, other=0.0).to(
            tl.float32
        )
        maximum = -float("inf")
        total = 0.0

        full = (V // C) * C
        for col in tl.range(0, full, C):
            values = tl.load(ptr + col + offsets).to(tl.float32)
            updated = tl.maximum(maximum, tl.max(values, 0))
            total = total * tl.exp(maximum - updated) + tl.sum(tl.exp(values - updated), 0)
            maximum = updated
        if full < V:
            indices = full + offsets
            values = tl.load(ptr + indices, indices < V, other=-float("inf")).to(tl.float32)
            updated = tl.maximum(maximum, tl.max(values, 0))
            total = total * tl.exp(maximum - updated) + tl.sum(tl.exp(values - updated), 0)
            maximum = updated
        tl.store(Stats + row, maximum)
        tl.store(Stats + N + row, total)
        tl.store(Stats + 2 * N + row, predicted)


@triton.jit
def _global_stats(
    Stats,
    Y,
    Norm,
    Loss,
    N: tl.constexpr,
    WORLD: tl.constexpr,
    V: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
):
    rows = tl.program_id(0) * B + tl.arange(0, B)
    ranks = tl.arange(0, P)
    offsets = ranks[:, None] * 3 * N + rows[None, :]
    mask = (ranks[:, None] < WORLD) & (rows[None, :] < N)
    m = tl.load(Stats + offsets, mask, other=-float("inf"))
    s = tl.load(Stats + N + offsets, mask, other=0.0)
    p = tl.load(Stats + 2 * N + offsets, mask, other=0.0)
    maximum = tl.max(m, 0)
    maximum = tl.where(rows < N, maximum, 0.0)
    denominator = tl.sum(s * tl.exp(m - maximum[None, :]), 0)
    labels = tl.load(Y + rows, rows < N, other=-100)
    valid = (labels >= 0) & (labels < V * WORLD)

    # Native CE masks shifted target logits, but keeps loss for out-of-vocabulary labels.
    loss = tl.log(denominator) + tl.where(valid, maximum - tl.sum(p, 0), 0.0)
    tl.store(Norm + rows, maximum, rows < N)
    tl.store(Norm + N + rows, denominator, rows < N)
    tl.store(Loss + rows, loss, rows < N)


@triton.jit
def _backward_logits(
    X, D, Y, Norm, Grad, N: tl.constexpr, V: tl.constexpr, START: tl.constexpr, C: tl.constexpr
):
    chunk = tl.cdiv(N, tl.num_programs(0))
    begin = tl.program_id(0) * chunk
    end = tl.minimum(begin + chunk, N)
    offsets = tl.arange(0, C)
    for row in tl.range(begin, end):
        ptr = X + row.to(tl.int64) * V
        out = D + row.to(tl.int64) * V
        maximum = tl.load(Norm + row)
        denominator = tl.load(Norm + N + row)
        label = tl.load(Y + row)
        grad = tl.load(Grad + row).to(tl.float32)
        local_label = label - START
        has_target = (local_label >= 0) & (local_label < V)
        safe_label = tl.maximum(0, tl.minimum(local_label, V - 1))
        target_logit = tl.load(ptr + safe_label, has_target, other=0.0).to(tl.float32)

        full = (V // C) * C
        for col in tl.range(0, full, C):
            values = tl.load(ptr + col + offsets).to(tl.float32)
            tl.store(out + col + offsets, (tl.exp(values - maximum) / denominator) * grad)
        if full < V:
            indices = full + offsets
            values = tl.load(ptr + indices, indices < V, other=0.0).to(tl.float32)
            tl.store(out + indices, (tl.exp(values - maximum) / denominator) * grad, indices < V)
        tl.store(
            out + safe_label,
            (tl.exp(target_logit - maximum) / denominator - 1.0) * grad,
            has_target,
        )


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return max(1, int(props["num_vectorcore"]))


def local_statistics(logits: torch.Tensor, labels: torch.Tensor, rank_start: int) -> torch.Tensor:
    """Compute FP32 row maxima, exponential sums and target logits per TP rank."""
    n, v = logits.shape
    cols = min(4096, triton.next_power_of_2(v))
    stats = torch.empty((3, n), device=logits.device, dtype=torch.float32)
    if n:
        _online_stats[(min(_vector_cores(logits.device.index), n),)](
            logits, labels, stats, n, v, rank_start, cols
        )
    return stats


def global_statistics(
    stats: torch.Tensor, labels: torch.Tensor, world: int, vocab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine gathered row statistics into a global normalizer and per-token loss."""
    n = labels.numel()
    norm = torch.empty((2, n), device=stats.device, dtype=torch.float32)
    loss = torch.empty((n,), device=stats.device, dtype=torch.float32)
    if not n:
        return norm, loss
    _global_stats[(triton.cdiv(n, 256),)](
        stats, labels, norm, loss, n, world, vocab, 256, triton.next_power_of_2(world)
    )
    return norm, loss


def backward_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    norm: torch.Tensor,
    grad: torch.Tensor,
    rank_start: int,
    out: torch.Tensor,
) -> None:
    """Write weighted logits gradients; out may alias temporary, unsaved logits."""
    n, v = logits.shape
    cols = min(4096, triton.next_power_of_2(v))
    if n:
        _backward_logits[(min(_vector_cores(logits.device.index), n),)](
            logits, out, labels, norm, grad, n, v, rank_start, cols
        )
