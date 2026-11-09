# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reference replay helpers used by packed CSA kernel tests."""

import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch


def capture_rng_state() -> dict:
    """Snapshot every RNG that the framework consumes during a fwd+bwd pass.

    Mirrors ``RerunStateMachine._save_state`` in
    ``megatron/core/rerun_state_machine.py``. Also captures Megatron's own
    ``CudaRNGStatesTracker`` (used for TP-aware dropout), which advances
    independently of ``torch.cuda``'s RNG when any layer calls
    ``get_cuda_rng_tracker().fork()``.
    """
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "mpu_tracker": get_cuda_rng_tracker().get_states(),
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of ``capture_rng_state``."""
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    if "mpu_tracker" in state:
        get_cuda_rng_tracker().set_states(state["mpu_tracker"])


class RacingStreams:
    """Run side-stream GEMMs in parallel with the model to perturb scheduling.

    Goal: force the CUDA scheduler to keep making different choices across
    runs so that any kernel whose bit-output depends on dispatch order
    surfaces as a bit-exact failure.

    Three things make scheduling vary more than the default "all streams
    do the same work" pattern:

    * ranks open DIFFERENT numbers of side streams (rank N opens N % 4
      more streams), so each rank presents different SM pressure → ranks
      finish model fwd/bwd at different wall-clock times → NCCL
      collectives race in different orders across runs.
    * each side stream runs GEMMs of MIXED SIZES (1024 / 2048 / 3072),
      picked from a per-rank-seeded CPU generator. Mixed sizes create
      more scheduling decision points than a uniform chain.
    * half the side streams have HIGHER priority than the default
      (priority=-1 vs 0). The scheduler must arbitrate priority classes
      under contention; the arbitration is hardware-state-dependent.

    Notes:

    * Effective only when ``CUDA_DEVICE_MAX_CONNECTIONS > 1``. Under ``=1``
      (Hopper determinism default) all streams serialise into a single
      hardware queue and this helper is a no-op. The
      ``test_bit_exact_under_racing_streams`` test skips on ``=1``.
    * Side-stream RNG uses a CPU ``torch.Generator`` — does NOT touch the
      per-device CUDA RNG, so the model's input RNG state stays clean.
    * Matrix contents are ``torch.ones`` because GEMM scheduling depends
      on shape/dtype/launch order, not values.
    """

    # GEMM sizes the stream rotates through. Mixed → adjacent kernels have
    # different completion latencies → more scheduling decision points.
    _SIZE_CHOICES = (1024, 2048, 3072)
    # Rank N opens this-many-extra streams, cycling. Smaller than world size
    # so within a node neighbouring ranks differ but bounded.
    _RANK_STREAM_CYCLE = 4
    # Alternating priorities — half HIGH, half DEFAULT. Scheduler must
    # arbitrate priority classes under contention.
    _STREAM_PRIORITIES = (-1, 0)
    # Per-rank generator seed = _BASE_SEED + rank. CPU generator, not CUDA,
    # so the model's per-device RNG state is untouched.
    _BASE_SEED = 0xC0FFEE

    def __init__(self, num_streams: int = 4, num_iters: int = 200):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.num_streams = num_streams + (rank % self._RANK_STREAM_CYCLE)
        self.num_iters = num_iters
        self._gen = torch.Generator()
        self._gen.manual_seed(self._BASE_SEED + rank)
        self.streams: list[torch.cuda.Stream] = []
        self._noise: list[torch.Tensor] = []

    def __enter__(self):
        self.streams = [
            torch.cuda.Stream(priority=self._STREAM_PRIORITIES[i % len(self._STREAM_PRIORITIES)])
            for i in range(self.num_streams)
        ]
        for stream in self.streams:
            with torch.cuda.stream(stream):
                # One randint call instead of num_iters calls.
                size_indices = torch.randint(
                    0, len(self._SIZE_CHOICES), (self.num_iters,), generator=self._gen
                ).tolist()
                # Keep only the last matmul result alive — earlier ones get
                # GC'd (PyTorch's caching allocator preserves storage while
                # the in-flight kernel still references it). Otherwise we'd
                # retain num_iters * num_streams tensors → multi-GB.
                result = None
                for idx in size_indices:
                    sz = self._SIZE_CHOICES[idx]
                    a = torch.ones(sz, sz, device="cuda", dtype=torch.bfloat16)
                    b = torch.ones(sz, sz, device="cuda", dtype=torch.bfloat16)
                    result = a @ b
                if result is not None:
                    self._noise.append(result)
        return self

    def __exit__(self, *args):
        for stream in self.streams:
            stream.synchronize()
        self.streams.clear()
        self._noise.clear()


def _clone_preserving_layout(t: torch.Tensor) -> torch.Tensor:
    """Copy ``t`` keeping its exact strides, including expanded (stride-0) views.

    ``Tensor.clone`` only preserves the strides of dense tensors; an ``expand``ed operand
    would come back as a dense copy and steer a kernel into a different specialisation
    than production uses (e.g. ``selective_state_update``'s ``TIE_HDIM`` path).
    """
    src = t.detach()
    if src.is_contiguous():
        return src.clone()
    extent = (
        src.storage_offset()
        + 1
        + sum((n - 1) * s for n, s in zip(src.shape, src.stride()) if n > 1)
    )
    flat = torch.empty(0, dtype=src.dtype, device=src.device).set_(src.untyped_storage())
    return flat[:extent].clone().as_strided(src.shape, src.stride(), src.storage_offset())


def clone_inputs(inputs: Any) -> Any:
    """Deep-copy tensors (preserving layout and ``requires_grad``); pass everything else through."""
    if isinstance(inputs, torch.Tensor):
        clone = _clone_preserving_layout(inputs)
        if inputs.requires_grad:
            clone.requires_grad_(True)
        return clone
    if isinstance(inputs, dict):
        return {k: clone_inputs(v) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        cloned = [clone_inputs(v) for v in inputs]
        return type(inputs)(cloned) if isinstance(inputs, tuple) else cloned
    return inputs


def flatten_tensors(obj: Any, prefix: str = "out") -> Dict[str, torch.Tensor]:
    """Flatten nested outputs into ``{name: tensor}``; non-tensors are ignored."""
    found: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        found[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.update(flatten_tensors(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.update(flatten_tensors(v, f"{prefix}[{i}]"))
    return found


def _leaf_inputs(inputs: Any, prefix: str = "in") -> Dict[str, torch.Tensor]:
    return {name: t for name, t in flatten_tensors(inputs, prefix).items() if t.requires_grad}


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """The logical contents of ``t`` as a flat ``uint8`` tensor (layout-independent)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff ``a`` and ``b`` have the same shape and dtype and identical bit patterns.

    Stricter than ``torch.equal``: ``+0.0`` and ``-0.0`` differ, and NaNs only match when
    their payloads match. Strides are not compared -- both operands are read in logical
    order -- so a kernel may return a differently laid out tensor with the same contents.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return bool(torch.equal(_as_bytes(a), _as_bytes(b)))


def _describe_mismatch(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape or a.dtype != b.dtype:
        return (
            f"{name}: shape/dtype differ ({tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype})"
        )
    bits_differ = (_as_bytes(a) != _as_bytes(b)).view(a.numel(), a.element_size()).any(dim=1)
    count = int(bits_differ.sum().item())
    msg = f"{name}: {count}/{a.numel()} elements differ"
    if not a.is_complex():
        af, bf = a.detach().reshape(-1).float(), b.detach().reshape(-1).float()
        value_differs = (af != bf) & ~(af.isnan() & bf.isnan())
        diff = torch.where(value_differs, (af - bf).abs(), torch.zeros_like(af))
        max_diff = float(diff.max().item()) if count else 0.0
        msg += f" (max |diff| {max_diff:.3e}"
        bit_only = count - int(value_differs.sum().item())
        if bit_only:
            msg += f"; {bit_only} differ only in bit pattern, e.g. signed zero or NaN payload"
        msg += ")"
    return msg


def run_once(
    fn: Callable[..., Any],
    inputs: Any,
    grad_outputs: Optional[Dict[str, torch.Tensor]] = None,
    backward: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run ``fn(**inputs)`` (or ``fn(*inputs)``) once and return outputs and input grads.

    ``grad_outputs`` maps flattened output names to the upstream gradient to use; outputs
    without an entry get ``ones_like``. Returns detached clones so later replays cannot
    alias them.
    """
    local = clone_inputs(inputs)
    result = fn(**local) if isinstance(local, dict) else fn(*local)
    outputs = {k: v for k, v in flatten_tensors(result).items()}
    grads: Dict[str, torch.Tensor] = {}
    if backward:
        leaves = _leaf_inputs(local)
        diff_outputs = [
            (name, t) for name, t in outputs.items() if t.requires_grad and t.is_floating_point()
        ]
        if leaves and diff_outputs:
            gos = [(grad_outputs or {}).get(name, torch.ones_like(t)) for name, t in diff_outputs]
            computed = torch.autograd.grad(
                [t for _, t in diff_outputs],
                list(leaves.values()),
                grad_outputs=gos,
                allow_unused=True,
            )
            for name, g in zip(leaves, computed):
                if g is not None:
                    grads[name] = g.detach().clone()
    torch.cuda.synchronize()
    return {k: v.detach().clone() for k, v in outputs.items()}, grads


def assert_replays_bit_exact(
    fn: Callable[..., Any],
    inputs: Any,
    *,
    replays: int = 3,
    backward: bool = True,
    grad_outputs: Optional[Dict[str, torch.Tensor]] = None,
    contention: bool = False,
    restore_rng: bool = False,
    what: str = "kernel",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Assert that ``replays`` runs of ``fn`` on identical inputs are byte-identical.

    Every output tensor and input gradient is compared with ``bytes_equal``.

    Args:
        fn: kernel entry point; called with ``**inputs`` if ``inputs`` is a dict, else
            ``*inputs``.
        inputs: materialised inputs. Tensors with ``requires_grad`` get their gradient
            compared too (when ``backward``).
        replays: number of runs; the first is the reference.
        backward: also run autograd and compare input gradients.
        grad_outputs: fixed upstream gradients keyed by flattened output name (``out``,
            ``out[0]``, ``out.name``). Defaults to ones, so the same value feeds every replay.
        contention: run replays 2.. under ``RacingStreams`` side-stream GEMM pressure.
        restore_rng: snapshot every RNG before the reference run and restore it before
            each replay -- for kernels that consume random numbers (dropout).
        what: label for error messages.

    Returns:
        The reference outputs and gradients (for follow-up assertions).
    """
    if replays < 2:
        raise ValueError("replays must be >= 2")
    rng = capture_rng_state() if restore_rng else None
    ref_out, ref_grad = run_once(fn, inputs, grad_outputs, backward)
    if not ref_out:
        raise AssertionError(f"{what} produced no tensor outputs; nothing to compare")
    for i in range(1, replays):
        if rng is not None:
            torch.cuda.synchronize()
            restore_rng_state(rng)
        if contention:
            with RacingStreams():
                out, grad = run_once(fn, inputs, grad_outputs, backward)
        else:
            out, grad = run_once(fn, inputs, grad_outputs, backward)
        _assert_replay_matches(i, ref_out, ref_grad, out, grad, what)
    return ref_out, ref_grad


def _assert_replay_matches(i, ref_out, ref_grad, out, grad, what) -> None:
    """Raise if replay ``i`` differs from the reference in any output or gradient tensor."""
    if out.keys() != ref_out.keys() or grad.keys() != ref_grad.keys():
        raise AssertionError(
            f"{what}: replay {i} returned different tensors "
            f"({sorted(out)}/{sorted(grad)} vs {sorted(ref_out)}/{sorted(ref_grad)})"
        )
    mismatches = [
        _describe_mismatch(name, ref_out[name], out[name])
        for name in ref_out
        if not bytes_equal(ref_out[name], out[name])
    ] + [
        _describe_mismatch(f"grad({name})", ref_grad[name], grad[name])
        for name in ref_grad
        if not bytes_equal(ref_grad[name], grad[name])
    ]
    if mismatches:
        raise AssertionError(
            f"{what} is not bit-exact across replays (replay {i} vs 1):\n  "
            + "\n  ".join(mismatches)
        )


def seeded(seed: int = 1234) -> None:
    """Seed every RNG the kernels may consume so input construction is repeatable."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
