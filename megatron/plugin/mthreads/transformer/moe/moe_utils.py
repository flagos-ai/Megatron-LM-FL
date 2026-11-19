# Copyright (c) BAAI Corporation.
"""Avoid the muDNN bool-sort failure in unfused MoE token dispatch."""

import torch

from megatron.core.transformer.moe import moe_utils


def permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor | None = None,
    num_out_tokens: int | None = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    tokens_per_expert: torch.Tensor | None = None,
    align_size: int = 0,
) -> tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None
]:
    """Use integer routing keys because bool stable argsort fails on MUSA.

    In the validated MUSA runtime, the core bool argsort raises
    ``SortCall MUDNN failed`` before expert computation. Its int8 path works,
    and False/True become 0/1 without changing the order or equality of keys.
    Stable sorting must still preserve token order within each expert.
    """
    # Fused kernels bypass this bool sort; capacity padding already uses int8.
    # Keep those paths and other devices on the core implementation. Calling
    # __wrapped__ avoids dispatching back into this override recursively.
    if fused or drop_and_pad or tokens.device.type != "musa":
        return moe_utils.permute.__wrapped__(
            tokens,
            routing_map,
            probs=probs,
            num_out_tokens=num_out_tokens,
            fused=fused,
            drop_and_pad=drop_and_pad,
            tokens_per_expert=tokens_per_expert,
            align_size=align_size,
        )

    assert num_out_tokens is not None, "num_out_tokens is required for the argsort-based permute"
    # Normalize truth values before casting: directly casting a non-bool mask
    # to int8 could truncate nonzero values and change which routes are selected.
    routing_keys = routing_map.bool().to(torch.int8).T.contiguous()
    # Keep argsort and a fixed output length for graph capture; masked_select
    # would make the output shape depend on device-side routing values.
    flat_sorted = routing_keys.reshape(-1).argsort(descending=True, stable=True)[:num_out_tokens]
    sorted_indices = flat_sorted % tokens.shape[0]
    permuted_probs = None
    if probs is not None:
        permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert
