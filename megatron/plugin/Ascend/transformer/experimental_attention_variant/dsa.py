# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Ascend Hadamard rotation for DSA and CSA indexers."""

import torch

# At dim=128, six FP32 scratch stages use 192 MiB per chunk.
_HADAMARD_CHUNK_ROWS = 65536


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension, bounding the Ascend kernel's FP32 scratch."""
    if x.device.type != "npu":
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            rotate_activation as original,
        )

        return original.__wrapped__(x)

    from flag_gems import hadamard_transform

    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    if x.numel() == 0:
        return x.clone()
    hidden_size = x.size(-1)
    flat = x.reshape(-1, hidden_size)
    outputs = [
        hadamard_transform(chunk, scale=hidden_size**-0.5)
        for chunk in flat.split(_HADAMARD_CHUNK_ROWS, dim=0)
    ]
    return torch.cat(outputs, dim=0).reshape(x.shape)
