"""KunLunXin tensor-parallel layer overrides matching XME v0.17 behavior."""

from functools import wraps
from typing import List, Optional, Tuple

import os
import torch
import torch.distributed as dist

from megatron.plugin.kunlunxin.debug import log_patch


def _get_bool_env(env_name: str, default: bool = False) -> bool:
    """Parse booleans with the same accepted values as XME."""
    default_value = "1" if default else "0"
    return os.getenv(env_name, default_value).upper() in ("TRUE", "1", "YES", "Y")


try:
    from hydrax.xaccelerator.fc_fusion import fc_fusion
    from hydrax.xaccelerator.linear import linear_bwd_dgrad, linear_bwd_wgrad, linear_fwd
    from hydrax.xaccelerator.linear import linear_with_all_gather_x as _linear_with_all_gather_x
    from hydrax.xaccelerator.stateful import StatefulConfig, state_of
    from torch_xmlir.config import config

    stateful_config = StatefulConfig(retain_state_for_parameter=not config.disable_cast_cache)

    class LinearWithGradAccumulationAndAsyncCommunicationKunlunxin(torch.autograd.Function):
        """XPU fused linear with gradient accumulation and async communication."""

        @staticmethod
        @stateful_config.make_callable_stateful
        def forward(
            ctx,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            gradient_accumulation_fusion: bool,
            allreduce_dgrad: bool,
            sequence_parallel: bool,
            grad_output_buffer: Optional[List[torch.Tensor]] = None,
            wgrad_deferral_limit: Optional[int] = 0,
            tp_group: Optional[torch.distributed.ProcessGroup] = None,
            te_fl_prefer: Optional[str] = None,
        ) -> torch.Tensor:
            """Forward pass of XPU fused tensor-parallel linear."""
            log_patch("tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunicationKunlunxin.forward")
            from megatron.core.parallel_state import (
                get_tensor_model_parallel_group,
                get_tensor_model_parallel_world_size,
            )

            if tp_group is None:
                tp_group = get_tensor_model_parallel_group()
            ctx.tp_group = tp_group

            input = input.contiguous()
            ctx.use_bias = bias is not None
            ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
            ctx.allreduce_dgrad = allreduce_dgrad
            ctx.sequence_parallel = sequence_parallel
            ctx.wgrad_deferral_limit = wgrad_deferral_limit
            ctx.grad_output_buffer = grad_output_buffer

            state_of(weight).memory_efficient = not any(
                (
                    config.disable_cast_cache,
                    getattr(weight, "_is_shared_weight", False),
                )
            )

            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size
                total_input = input.new_empty(dim_size)
                if not _get_bool_env("XMLIR_BATCH_PARALLEL"):
                    dist.all_gather_into_tensor(total_input, input, group=tp_group)
                    output = linear_fwd(total_input, weight, bias)
                    ctx.input_shape = input.shape
                else:
                    output = _linear_with_all_gather_x(total_input, weight, input, bias, group=tp_group)
                    ctx.input_shape = input.shape
            else:
                total_input = input
                output = linear_fwd(total_input, weight, bias)

            ctx.save_for_backward(total_input, weight)
            return output

        @classmethod
        @stateful_config.make_callable_stateful
        def backward(cls, ctx, grad_output):
            """Backward pass of XPU fused tensor-parallel linear."""
            log_patch("tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunicationKunlunxin.backward")
            state_of(grad_output).memory_efficient = not config.disable_cast_cache
            grad_input, handle = cls._async_backward_input(ctx, grad_output)
            grad_weight = cls._backward_weight(ctx, grad_output)
            grad_bias = cls._backward_bias(ctx, grad_output)
            if handle is not None:
                handle.wait()
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        @classmethod
        def _async_backward_input(
            cls, ctx, grad_output: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
            """Compute input gradient and launch async communication when needed."""
            total_input, weight = ctx.saved_tensors
            grad_total_input = torch.empty_like(total_input)
            linear_bwd_dgrad(grad_output, weight, grad_total_input)

            if ctx.sequence_parallel:
                grad_input = total_input.new_empty(ctx.input_shape, requires_grad=False)
                handle = dist.reduce_scatter_tensor(
                    grad_input, grad_total_input, group=ctx.tp_group, async_op=True
                )
            elif ctx.allreduce_dgrad:
                grad_input = grad_total_input
                handle = dist.all_reduce(grad_input, group=ctx.tp_group, async_op=True)
            else:
                grad_input = grad_total_input
                handle = None
            return grad_input, handle

        @classmethod
        def _backward_weight(cls, ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
            """Compute or defer weight gradient."""
            from megatron.core.utils import prepare_input_tensors_for_wgrad_compute

            total_input, weight = ctx.saved_tensors
            grad_output_buffer = ctx.grad_output_buffer
            wgrad_deferral_limit = ctx.wgrad_deferral_limit

            wgrad_compute = True
            if grad_output_buffer is not None and (
                wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit
            ):
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

            if ctx.gradient_accumulation_fusion:
                if wgrad_compute:
                    grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                        grad_output, total_input
                    )
                    fc_fusion(grad_output, total_input, lhs_trans=True, beta=1.0, out=weight.main_grad)

                if hasattr(weight, "grad_added_to_main_grad"):
                    if getattr(weight, "zero_out_wgrad", False):
                        grad_weight = total_input.new_zeros(weight.main_grad.shape, requires_grad=False)
                    else:
                        grad_weight = total_input.new_empty(weight.main_grad.shape, requires_grad=False)
                    weight.grad_added_to_main_grad = True
                else:
                    grad_weight = None
            else:
                grad_weight = torch.empty_like(weight)
                linear_bwd_wgrad(grad_output, total_input, grad_weight)
            return grad_weight

        @classmethod
        def _backward_bias(cls, ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
            """Compute bias gradient if bias is enabled."""
            return grad_output.sum(0) if ctx.use_bias else None

    def column_parallel_linear_forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """Disable inplace cast for shared weights before original ColumnParallelLinear forward."""
        log_patch("tensor_parallel.layers.column_parallel_linear_forward")
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        @wraps(ColumnParallelLinear.forward.__wrapped__)
        @stateful_config.make_callable_stateful
        def _call_original(module, input_arg, weight_arg=None, runtime_gather_output_arg=None):
            if weight_arg is not None:
                state_of(weight_arg).memory_efficient = False
                weight_arg._is_shared_weight = True  # pyright: ignore[reportAttributeAccessIssue]
            return ColumnParallelLinear.forward.__wrapped__(
                module, input_arg, weight_arg, runtime_gather_output_arg
            )

        return _call_original(self, input_, weight, runtime_gather_output)

except ImportError:
    LinearWithGradAccumulationAndAsyncCommunicationKunlunxin = None
    column_parallel_linear_forward = None
