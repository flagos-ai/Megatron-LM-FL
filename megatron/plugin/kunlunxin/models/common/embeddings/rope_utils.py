# Copyright (c) 2025 KUNLUNXIN CORPORATION. All Rights Reserved.
import warnings
from typing import Optional

import torch
from torch import Tensor

from megatron.plugin.kunlunxin.debug import debug_patch


class RotaryPositionalEmbeddingWithFreqFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, freqs):
        output = t.new_empty(t.shape)
        torch.ops.custom_ops.rotary_pos_emb(t, freqs, out=output)
        ctx.freqs = freqs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        freqs_ = ctx.freqs
        grad_t = grad_output.new_empty(grad_output.shape)
        torch.ops.custom_ops.rotary_pos_emb_backward(grad_output, freqs_, out=grad_t)
        return grad_t, None


@debug_patch("models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd")
def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
    multi_latent_attention: Optional[bool] = None,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated. Please use mla_rotary_interleaved instead.",
            DeprecationWarning,
        )
        mla_rotary_interleaved = multi_latent_attention

    # Match the upstream shape guard before choosing the XPU fused path.
    if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
        freqs = freqs.squeeze(-2)

    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    if (
        mscale == 1.0
        and not rotary_interleaved
        and not inverse
        and not mla_output_remove_interleaving
    ):
        t = RotaryPositionalEmbeddingWithFreqFunction.apply(t, freqs)
    else:
        from megatron.core.models.common.embeddings.rope_utils import _rotate_half

        # first part is cosine component
        # second part is sine component, need to change signs with _rotate_half method
        cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
        sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
        if inverse:
            sin_ = -sin_
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    if mla_rotary_interleaved and mla_output_remove_interleaving:
        x1, x2 = torch.chunk(t, 2, dim=-1)
        t = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)

    return torch.cat((t, t_pass), dim=-1)
