# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

"""Dense NPU convolution compatibility path for Megatron GatedDeltaNet."""

import torch.nn.functional as F


def causal_conv1d(
    x,
    weight=None,
    bias=None,
    residual=None,
    initial_state=None,
    output_final_state=False,
    activation=None,
    backend="triton",
    cu_seqlens=None,
    **kwargs,
):
    # Stateful and variable-length inputs retain the upstream implementation.
    if (
        backend != "triton"
        or kwargs
        or x.device.type != "npu"
        or x.ndim != 3
        or weight is None
        or initial_state is not None
        or output_final_state
        or cu_seqlens is not None
        or activation not in (None, "silu", "swish")
    ):
        from fla.modules.convolution import causal_conv1d as original

        return original(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            backend=backend,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

    output = F.conv1d(
        x.transpose(1, 2).contiguous(),
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[-1] - 1,
        groups=x.shape[-1],
    )
    output = output[..., :x.shape[1]]
    if activation in ("silu", "swish"):
        output = F.silu(output)
    output = output.transpose(1, 2)
    if residual is not None:
        output = output + residual
    return output, None


def gated_delta_net_conv1d(self, **kwargs):
    return causal_conv1d(**kwargs)
