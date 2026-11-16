# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA top-k state retained by a single checkpointed stack invocation."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

from torch import Tensor


@dataclass
class DSAIndexSharingState:
    """Keep shared indices and valid lengths together for one microbatch."""

    topk_indices: dict[int, Tensor] = field(default_factory=dict)
    topk_lengths: dict[int, Tensor] = field(default_factory=dict)


_current_state: ContextVar[DSAIndexSharingState | None] = ContextVar(
    "dsa_index_sharing_state", default=None
)


def get_dsa_index_sharing_state() -> DSAIndexSharingState | None:
    """Return the state owned by the currently executing checkpointed stack."""
    return _current_state.get()


@contextmanager
def dsa_index_sharing_context(state: DSAIndexSharingState) -> Iterator[None]:
    """Activate a stack's state for either its initial forward or recomputation.

    The checkpoint closures own ``state``; this context only selects it while a
    chunk executes. Restoring the previous context also supports nested stacks
    and exceptions without leaving another microbatch's state active.
    """
    token = _current_state.set(state)
    try:
        yield
    finally:
        _current_state.reset(token)
