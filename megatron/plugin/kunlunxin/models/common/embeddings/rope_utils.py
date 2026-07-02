# Copyright (c) 2025 KUNLUNXIN CORPORATION. All Rights Reserved.
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
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    if mscale == 1.0:
        t = RotaryPositionalEmbeddingWithFreqFunction.apply(t, freqs)
    else:
        from megatron.core.models.common.embeddings.rope_utils import _rotate_half

        # first part is cosine component
        # second part is sine component, need to change signs with _rotate_half method
        cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
        sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)
