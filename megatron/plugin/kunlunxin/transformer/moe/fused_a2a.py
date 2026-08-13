"""KunLunXin plugin implementations for megatron.core.transformer.moe.fused_a2a."""

import torch

from megatron.core.transformer.moe.fused_a2a import (
    FusedCombine as _CoreFusedCombine,
    FusedDispatch as _CoreFusedDispatch,
)
from megatron.plugin.platform import get_platform
from megatron.plugin.kunlunxin.debug import debug_patch, log_patch

try:
    from deep_ep import BufferV2
except ImportError:
    BufferV2 = None

_buffer_v2 = None
cur_platform = get_platform()


@debug_patch("transformer.moe.fused_a2a.get_buffer")
def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a KunLunXin DeepEP BufferV2 communication buffer.

    Args:
        group: Distributed process group for communication.
        hidden_bytes: Hidden size in bytes (unused, BufferV2 uses 0).
    """
    if BufferV2 is None:
        return None

    global _buffer_v2
    if _buffer_v2 is None or _buffer_v2.group != group:
        _buffer_v2 = BufferV2(group, 0, 0)
    return _buffer_v2


class FusedDispatchKunlunxin(_CoreFusedDispatch):
    """KunLunXin fused dispatch with stream synchronization guards."""

    @staticmethod
    def forward(*args, **kwargs):
        """Forward pass of fused dispatch."""
        log_patch("transformer.moe.fused_a2a.FusedDispatchKunlunxin.forward")
        cur_platform.synchronize()
        ret = _CoreFusedDispatch.forward(*args, **kwargs)
        cur_platform.synchronize()
        return ret

    @staticmethod
    def backward(*args, **kwargs):
        """Backward pass of fused dispatch."""
        log_patch("transformer.moe.fused_a2a.FusedDispatchKunlunxin.backward")
        cur_platform.synchronize()
        ret = _CoreFusedDispatch.backward(*args, **kwargs)
        cur_platform.synchronize()
        return ret


@debug_patch("transformer.moe.fused_a2a.fused_dispatch")
def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Perform fused dispatch operation with KunLunXin synchronization guards.

    Args:
        x: Input tensor.
        token_indices: Token routing indices.
        token_probs: Token routing probabilities.
        num_experts: Number of experts.
        group: Distributed process group.
        async_finish: Whether to finish asynchronously.
        allocate_on_comm_stream: Whether to allocate on comm stream.
    """
    return FusedDispatchKunlunxin.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
    )


class FusedCombineKunlunxin(_CoreFusedCombine):
    """KunLunXin fused combine with stream synchronization guards."""

    @staticmethod
    def forward(*args, **kwargs):
        """Forward pass of fused combine."""
        log_patch("transformer.moe.fused_a2a.FusedCombineKunlunxin.forward")
        cur_platform.synchronize()
        ret = _CoreFusedCombine.forward(*args, **kwargs)
        cur_platform.synchronize()
        return ret

    @staticmethod
    def backward(*args, **kwargs):
        """Backward pass of fused combine."""
        log_patch("transformer.moe.fused_a2a.FusedCombineKunlunxin.backward")
        cur_platform.synchronize()
        ret = _CoreFusedCombine.backward(*args, **kwargs)
        cur_platform.synchronize()
        return ret


@debug_patch("transformer.moe.fused_a2a.fused_combine")
def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
    """Perform fused combine operation with KunLunXin synchronization guards.

    Args:
        x: Input tensor.
        group: Distributed process group.
        handle: Dispatch handle from fused_dispatch.
        async_finish: Whether to finish asynchronously.
        allocate_on_comm_stream: Whether to allocate on comm stream.
    """
    return FusedCombineKunlunxin.apply(x, group, handle, async_finish, allocate_on_comm_stream)
