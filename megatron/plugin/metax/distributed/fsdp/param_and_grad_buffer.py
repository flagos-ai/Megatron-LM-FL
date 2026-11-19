# Copyright (c) 2026, BAAI. All rights reserved.
"""MetaX stream policy for FSDP CUDA graph capture."""

import torch

from megatron.plugin.platform import get_platform


def _get_communication_stream(stream, ddp_config):
    """Use the active stream for MACA graph collectives, including warmup."""
    # Resolve at each operation: warmup and capture may use different streams.
    # MACA graph finalization can crash when collectives use extra user streams.
    # Retain the runtime guard because the registry can select its only vendor
    # implementation when MG_FL_PREFER is empty (no vendor preference).
    if stream is None or (
        ddp_config.megatron_fsdp_cuda_graph_mode and getattr(torch.version, "maca", None)
    ):
        return get_platform().current_stream()
    return stream
