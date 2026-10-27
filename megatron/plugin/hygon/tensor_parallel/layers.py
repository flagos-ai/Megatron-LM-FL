# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 BAAI. All rights reserved.

"""BW1100 vocabulary GEMMs using the existing tensor-parallel autograd protocol.

The forward/backward protocol below mirrors core.tensor_parallel.layers;
only the vocabulary forward and dgrad GEMMs use different layouts.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch

from megatron.core.tensor_parallel import layers

# Measured for the shapes admitted below; this is a maximum, not a divisibility requirement.
_VOCAB_FORWARD_CHUNK_SIZE = 16384


@lru_cache(maxsize=None)
def _is_bw1100(device_index: int) -> bool:
    properties = torch.cuda.get_device_properties(device_index)
    return (
        properties.name.upper().startswith("BW11")
        and getattr(properties, "gcnArchName", "").split(":")[0] == "gfx938"
    )


def _supports_vocab_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sequence_parallel: bool,
    tp_group: torch.distributed.ProcessGroup | None,
) -> bool:
    # Keep the optimization within the measured BF16 TP2 vocabulary shapes.
    if (
        os.environ.get("MEGATRON_HYGON_VOCAB_GEMM", "1") == "0"
        or not getattr(torch.version, "hip", None)
        or not input.is_cuda
        or input.device != weight.device
        or input.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or input.ndim not in (2, 3)
        or weight.shape != (124160, 5120)
        or input.shape[-1] != weight.shape[1]
        or not input.is_contiguous()
        or not weight.is_contiguous()
        or hasattr(weight, "__fsdp_param__")
        or torch.compiler.is_compiling()
    ):
        return False
    rows = input.numel() // weight.shape[1]
    if sequence_parallel:
        if tp_group is None:
            return False
        rows *= tp_group.size()
    if rows not in (8192, 16384) or torch.cuda.is_current_stream_capturing():
        return False
    return _is_bw1100(weight.device.index)


def _vocab_forward(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = input.view(-1, weight.shape[1])
    output = torch.empty((flat.shape[0], weight.shape[0]), dtype=input.dtype, device=input.device)
    for start in range(0, weight.shape[0], _VOCAB_FORWARD_CHUNK_SIZE):
        end = min(start + _VOCAB_FORWARD_CHUNK_SIZE, weight.shape[0])
        # Split output channels, retaining the complete reduction dimension.
        torch.mm(flat, weight[start:end].t(), out=output[:, start:end])
    return output.view(*input.shape[:-1], weight.shape[0])


def _vocab_dgrad(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = grad_output.reshape(-1, weight.shape[0])
    grad_input = torch.mm(weight.t(), flat.t()).t().contiguous()
    return grad_input.view(*grad_output.shape[:-1], weight.shape[1])


class LinearWithGradAccumulationAndAsyncCommunicationHygon(torch.autograd.Function):
    """Use BW1100 vocabulary GEMMs, falling back to the core linear elsewhere."""

    @staticmethod
    @layers.custom_fwd
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: list[torch.Tensor] | None,
        wgrad_deferral_limit: int | None,
        tp_group: torch.distributed.ProcessGroup | None,
        te_fl_prefer: str | None,
    ) -> torch.Tensor:
        """Gather sequence shards and compute the local vocabulary projection."""
        ctx.hygon_vocab_gemm = _supports_vocab_gemm(input, weight, sequence_parallel, tp_group)
        if not ctx.hygon_vocab_gemm:
            return layers.LinearWithGradAccumulationAndAsyncCommunication.forward(
                ctx,
                input,
                weight,
                bias,
                gradient_accumulation_fusion,
                allreduce_dgrad,
                sequence_parallel,
                grad_output_buffer,
                wgrad_deferral_limit,
                tp_group,
                te_fl_prefer,
            )
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        ctx.save_for_backward(input, weight)
        # We can't save main_grad in save_for_backward as this module would be
        # reused across layers like MTP logits. So, to prevent in-place modification
        # checks we save the tensor in ctx.
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.tp_group = tp_group
        ctx.te_fl_prefer = te_fl_prefer

        if sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = layers.get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            layers.dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output = _vocab_forward(total_input, weight)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @layers.custom_bwd
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Compute transposed dgrad while preserving weight gradients and collectives."""
        if not ctx.hygon_vocab_gemm:
            return layers.LinearWithGradAccumulationAndAsyncCommunication.backward(ctx, grad_output)
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        handle = None
        tp_group = ctx.tp_group

        if ctx.gradient_accumulation_fusion:
            weight.main_grad = main_grad

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * tp_group.size()

                all_gather_buffer = layers.get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = layers.dist_all_gather_func(
                    all_gather_buffer, input, group=tp_group, async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = _vocab_dgrad(grad_output, weight)

        if ctx.sequence_parallel and wgrad_compute:
            # pylint: disable=possibly-used-before-assignment
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = layers.prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size,
                dtype=input.dtype,
                device=layers.cur_platform.current_device(),
                requires_grad=False,
            )
            # reduce_scatter
            handle = layers.dist_reduce_scatter_func(
                sub_grad_input, grad_input, group=tp_group, async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                # In case of Megatron-FSDP, need to create main grad buffers in-place
                if hasattr(weight, "__fsdp_param__"):
                    weight.main_grad = weight.get_main_grad()
                    # Import here to avoid circular import
                    from megatron.core.extensions.transformer_engine import te_general_gemm

                    if te_general_gemm is not None:
                        # Use TE general_gemm to support mixed-precision output
                        # (e.g. bf16 input -> fp32 main_grad) which torch.matmul
                        # does not support via the out= parameter.
                        te_general_gemm(
                            total_input,
                            grad_output,
                            out_dtype=weight.main_grad.dtype,
                            layout="NT",
                            out=weight.main_grad,
                            grad=True,
                        )
                    else:
                        torch.matmul(grad_output.t(), total_input, out=weight.main_grad)
                else:
                    if not (ctx.te_fl_prefer == 'flagos' or ctx.te_fl_prefer == 'reference'):
                        if weight.main_grad.dtype == torch.float32:
                            layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                                total_input, grad_output, weight.main_grad
                            )
                        elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                            layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                                total_input, grad_output, weight.main_grad
                            )
                        else:
                            raise RuntimeError(
                                "Unsupported gradient type for gradient accumulation fusion"
                            )
                    else:
                        if weight.main_grad.dtype in (torch.float32, torch.float16, torch.bfloat16):
                            grad_weight = torch.matmul(grad_output.t(), total_input)
                            weight.main_grad += grad_weight.view_as(weight.main_grad)
                        else:
                            raise RuntimeError(
                                "Unsupported gradient type for gradient accumulation fusion"
                            )

            if hasattr(weight, "grad_added_to_main_grad"):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, "zero_out_wgrad", False):
                    if layers.HAVE_TE:
                        # layers.get_dummy_wgrad function in TE enables reuse of single dummy wgrad buffer
                        # across different layers/microbatches. The function accepts shape as list.
                        grad_weight = layers.get_dummy_wgrad(
                            list(weight.main_grad.shape), input.dtype, zero=True
                        )
                    else:
                        grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=layers.cur_platform.current_device(),
                            requires_grad=False,
                        )
                else:
                    if layers.HAVE_TE:
                        grad_weight = layers.get_dummy_wgrad(
                            list(weight.main_grad.shape), input.dtype
                        )
                    else:
                        grad_weight = torch.empty(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=layers.cur_platform.current_device(),
                            requires_grad=False,
                        )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return (
                sub_grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None
