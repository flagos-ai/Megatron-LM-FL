"""KunLunXin fused SwiGLU overrides matching XME v0.17 behavior."""

import torch

from megatron.plugin.kunlunxin.debug import debug_patch


@debug_patch("fusions.fused_bias_swiglu.bias_swiglu_impl")
def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Run XPU fused bias SwiGLU through torch_xmlir."""
    from torch_xmlir.nn.swiglu import SwiGLUFunction

    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert fp8_input_store is False
    input = input.view(-1, ori_shape[-1])

    if bias is not None:
        input = input + bias

    if cpu_offload_input:
        input.activation_offloading = True
        if bias is not None:
            bias.activation_offloading = True

    output = SwiGLUFunction.apply(input)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


@debug_patch("fusions.fused_bias_swiglu.swiglu")
def swiglu(y):
    """Run XPU fused SwiGLU through torch_xmlir."""
    from torch_xmlir.nn.swiglu import SwiGLUFunction

    ori_shape = y.shape
    y = y.view(-1, ori_shape[-1])
    output = SwiGLUFunction.apply(y)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


class WeightedSwiGLUFunction(torch.autograd.Function):
    """Weighted SwiGLU autograd function backed by XPU custom ops."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor, axis: int = -1, turn: bool = True):
        """Run weighted SwiGLU forward."""
        if axis < -input.ndim or axis >= input.ndim:
            raise IndexError("for swiglu, input axis out of range.")
        if input.shape[axis] % 2 != 0:
            raise ValueError("for swiglu, input.shape[axis] must be even.")

        output_shape = list(input.shape)
        output_shape[axis] = input.shape[axis] // 2
        output = input.new_empty(output_shape)
        ctx.weights_dtype = weights.dtype

        if input.numel() == 0:
            ctx.save_for_backward(input)
            return output

        ctx.turn = turn
        ctx.axis = axis
        torch.ops.custom_ops.swiglu_forward(input, axis, turn, out=output)
        dtype = input.dtype
        output = output * weights
        output = output.to(dtype)
        ctx.save_for_backward(input, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Run weighted SwiGLU backward."""
        input = ctx.saved_tensors[0]
        d_input = grad_output.new_empty(input.shape).to(ctx.weights_dtype)

        if input.numel() == 0:
            dw_shape = list(input.shape)
            dw_shape[-1] = 1
            d_weights = grad_output.new_empty(dw_shape).to(ctx.weights_dtype)
            return d_input.to(input.dtype), d_weights, None, None

        weights = ctx.saved_tensors[1]
        torch.ops.custom_ops.swiglu_backward(
            input.to(weights.dtype), grad_output * weights, ctx.axis, ctx.turn, dx=d_input
        )

        output_shape = list(input.shape)
        output_shape[ctx.axis] = input.shape[ctx.axis] // 2
        d_weights = input.new_empty(output_shape)
        torch.ops.custom_ops.swiglu_forward(input, ctx.axis, ctx.turn, out=d_weights)
        d_weights = d_weights * grad_output.to(weights.dtype)
        d_weights = torch.sum(d_weights, dim=-1, keepdim=True)
        return d_input.to(input.dtype), d_weights.to(weights.dtype), None, None


@debug_patch("fusions.fused_bias_swiglu.weighted_bias_swiglu_impl")
def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False, clamp_value=None):
    """Run XPU token-wise weighted SwiGLU."""
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert not fp8_input_store, "Not support fp8 now!"
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    output = WeightedSwiGLUFunction.apply(input, weights)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
