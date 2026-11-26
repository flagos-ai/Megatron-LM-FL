import os
from typing import Callable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

# Bound saved logits and recompute larger projections one chunk at a time.
_LOGITS_LIMIT_BYTES = 4 * 1024**3
_CHUNK_LOGITS_BYTES = 1024**3


def _logits_save_limit_bytes(device):
    total = torch.npu.get_device_properties(device).total_memory
    return min(_LOGITS_LIMIT_BYTES, total // 8)


def _chunk_size(weight):
    budget = min(_CHUNK_LOGITS_BYTES, max(1, _logits_save_limit_bytes(weight.device)))
    rows = max(1, budget // (weight.shape[0] * weight.element_size()))
    return max(1, rows // 256 * 256) if rows >= 256 else rows


def _project(hidden, weight, bias, scale):
    logits = torch.mm(hidden, weight.t())
    if bias is not None:
        logits = logits + bias
    if scale != 1.0:
        logits = logits * scale
    return logits


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, tp_group, bias, scale, chunked):
        from .linear_cross_entropy_kernels import global_statistics, local_statistics

        x = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        labels = target.transpose(0, 1).reshape(-1).contiguous()
        world = tp_group.size() if tp_group is not None else 1
        rank = tp_group.rank() if tp_group is not None else 0
        vocab = weight.shape[0]
        ctx.chunked = chunked
        ctx.scale = scale
        if chunked:
            ctx.chunk = _chunk_size(weight)
            local = torch.empty((3, x.shape[0]), device=x.device, dtype=torch.float32)
            for start in range(0, x.shape[0], ctx.chunk):
                end = min(start + ctx.chunk, x.shape[0])
                logits = _project(x[start:end], weight, bias, scale)
                local[:, start:end] = local_statistics(logits, labels[start:end], rank * vocab)
                del logits
        else:
            logits = _project(x, weight, bias, scale)
            local = local_statistics(logits, labels, rank * vocab)
        if world > 1 and x.shape[0]:
            gathered = torch.empty((world * 3, x.shape[0]), device=x.device, dtype=torch.float32)
            dist.all_gather_into_tensor(gathered, local, group=tp_group)
        else:
            gathered = local
        norm, loss = global_statistics(gathered, labels, world, vocab)
        ctx.save_for_backward(x, weight, labels, norm, bias, None if chunked else logits)
        ctx.hidden_shape = hidden.shape
        ctx.rank_start = rank * vocab
        return loss.reshape(target.shape[1], target.shape[0]).transpose(0, 1).contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        from .linear_cross_entropy_kernels import backward_logits

        x, weight, labels, norm, bias, saved_logits = ctx.saved_tensors
        grad = grad_loss.transpose(0, 1).reshape(-1).float().contiguous()
        needs_bias = bias is not None and ctx.needs_input_grad[4]
        db = torch.zeros_like(bias, dtype=torch.float32) if needs_bias else None
        if ctx.chunked:
            dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
            dw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
            for start in range(0, x.shape[0], ctx.chunk):
                end = min(start + ctx.chunk, x.shape[0])
                logits = _project(x[start:end], weight, bias, ctx.scale)
                backward_logits(
                    logits,
                    labels[start:end],
                    norm[:, start:end].contiguous(),
                    grad[start:end],
                    ctx.rank_start,
                    logits,
                )
                if ctx.scale != 1.0:
                    logits.mul_(ctx.scale)
                if db is not None:
                    db.add_(logits.float().sum(dim=0))
                dlinear = logits.to(weight.dtype)
                if dx is not None:
                    torch.mm(dlinear, weight, out=dx[start:end])
                if dw is not None:
                    partial = torch.mm(dlinear.t(), x[start:end])
                    dw.add_(partial)
                    del partial
                del logits, dlinear
            dw = dw.to(weight.dtype) if dw is not None else None
        else:
            # Saved logits stay read-only so retained graphs can run backward again.
            logits = torch.empty_like(saved_logits)
            backward_logits(saved_logits, labels, norm, grad, ctx.rank_start, logits)
            if ctx.scale != 1.0:
                logits.mul_(ctx.scale)
            if db is not None:
                db.add_(logits.float().sum(dim=0))
            dlinear = logits.to(weight.dtype)
            dx = torch.mm(dlinear, weight) if ctx.needs_input_grad[0] else None
            dw = None
            if ctx.needs_input_grad[1]:
                dw = torch.mm(dlinear.t(), x)
        return (
            dx.reshape(ctx.hidden_shape) if dx is not None else None,
            dw,
            None,
            None,
            db.to(bias.dtype) if db is not None else None,
            None,
            None,
        )


def compute_output_loss(
    self,
    hidden_states: torch.Tensor,
    output_layer: torch.nn.Module,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor | Callable[[], torch.Tensor],
    runtime_gather_output: bool | None = None,
    scale_logits_fn: Callable | None = None,
) -> torch.Tensor:
    """Fuse the Ascend output projection with vocabulary-parallel per-token CE."""
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
    )

    def fallback():
        return LanguageModule.compute_output_loss.__wrapped__(
            self,
            hidden_states,
            output_layer,
            output_weight,
            labels,
            runtime_gather_output,
            scale_logits_fn,
        )

    config = self.config
    if (
        os.getenv("MEGATRON_FUSE_CE_MM", "0") != "1"
        or hidden_states.device.type != "npu"
        or not isinstance(output_layer, ColumnParallelLinear)
        or not config.cross_entropy_loss_fusion
        or config.cross_entropy_fusion_impl != "native"
    ):
        return fallback()
    gather = output_layer.gather_output if runtime_gather_output is None else runtime_gather_output
    context = config._cpu_offloading_context
    if (
        gather
        or output_layer.gradient_accumulation_fusion
        or config.defer_embedding_wgrad_compute
        or config.cpu_offloading
        or (context is not None and context.inside_context)
    ):
        return fallback()
    if scale_logits_fn is not None and (
        getattr(scale_logits_fn, "__self__", None) is not self
        or getattr(scale_logits_fn, "__func__", None) is not LanguageModule._scale_logits
    ):
        return fallback()
    if callable(labels):
        labels = labels()
    scale = config.mup_output_mult if scale_logits_fn is not None and config.use_mup else 1.0
    group = output_layer.tp_group

    def processor(hidden, live_weight):
        if output_layer.sequence_parallel and not output_layer.explicit_expert_comm:
            hidden = gather_from_sequence_parallel_region(
                hidden, group=group, tensor_parallel_output_grad=True
            )
        elif not output_layer.disable_grad_reduce and not output_layer.explicit_expert_comm:
            hidden = copy_to_tensor_model_parallel_region(hidden, group=group)
        bias = output_layer.bias if not output_layer.skip_bias_add else None
        element_size = max(
            live_weight.element_size(), bias.element_size() if bias is not None else 0
        )
        chunked = labels.numel() * live_weight.shape[0] * element_size > _logits_save_limit_bytes(
            live_weight.device
        )
        return _LinearCrossEntropy.apply(hidden, live_weight, labels, group, bias, scale, chunked)

    loss, _ = output_layer(
        hidden_states,
        weight=output_weight,
        runtime_gather_output=runtime_gather_output,
        output_processor=processor,
    )
    return loss


def column_parallel_linear_forward(
    self,
    input_: torch.Tensor,
    weight: torch.Tensor | None = None,
    runtime_gather_output: bool | None = None,
    *,
    output_processor: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run an optional joint loss processor within the output module's hooks."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    if output_processor is None:
        return ColumnParallelLinear.forward.__wrapped__(
            self, input_, weight=weight, runtime_gather_output=runtime_gather_output
        )
    if weight is None:
        weight = self.weight
        if weight is None:
            raise RuntimeError("weight was not supplied to ColumnParallelLinear forward pass")
    elif tuple(weight.shape) != (self.output_size_per_partition, self.input_size):
        raise RuntimeError(f"supplied weight has unexpected shape {tuple(weight.shape)}")
    return output_processor(input_, weight), self.bias if self.skip_bias_add else None
