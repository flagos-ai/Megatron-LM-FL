# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import VocabUtility


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates logits_max."""

        vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates predicted logits."""

        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor, predicted_logits: torch.Tensor, sum_exp_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates cross entropy loss."""

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare gradient calculation operands."""

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates gradients."""

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
        """Vocab parallel cross entropy forward function."""

        vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            vocab_parallel_logits
        )
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """  # pylint: disable=line-too-long
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities.
            # So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Vocab parallel cross entropy backward function."""

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
        )

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad

            # Finally elementwise multiplication with the output gradients.
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = VocabParallelCrossEntropy.calculate_gradients(
                grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
            )

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
            dimension is [sequence_length, batch_size, vocab_size/num_parallel_ranks]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        label_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)


class _VocabParallelCrossEntropyChunked(torch.autograd.Function):
    """Chunked vocab parallel cross entropy that reduces peak memory usage.

    Processes the sequence dimension in chunks so that only one chunk's worth
    of fp32 logits exists in memory at a time instead of the full
    [seq_len, batch, vocab/TP] tensor.

    Forward peak memory:
        Original : bf16_input + fp32_logits = S*B*V*2 + S*B*V*4  (~10.5 GiB)
        Chunked  : bf16_input + fp32_chunk  = S*B*V*2 + C*B*V*4  (~4.4 GiB, C=4096)

    Backward uses recomputation (same pattern as activation recompute): the
    fp32 softmax is NOT saved from forward; instead the forward pass is re-run
    per chunk during backward to recover the softmax needed for gradient
    computation.

    The computation logic is 100% delegated to VocabParallelCrossEntropy static
    methods — no numerical differences from the original implementation.
    """

    @staticmethod
    def _run_chunk_forward(
        chunk_logits_bf16,
        chunk_target,
        vocab_start_index,
        vocab_end_index,
        label_smoothing,
    ):
        """Run one chunk through the full forward pass.

        Returns:
            softmax   : [chunk_size, batch, vocab/TP], fp32
            chunk_loss: [chunk_size, batch], fp32
            target_mask      : [chunk_size, batch], bool
            masked_target_1d : [chunk_size * batch], int64
        """
        # Step 1 — convert to fp32 and find per-position max
        chunk_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            chunk_logits_bf16
        )
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group(),
        )

        # Step 2 — subtract max, extract ground-truth logit, compute exp
        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                chunk_logits, chunk_target, logits_max, vocab_start_index, vocab_end_index
            )
        )
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Step 3 — compute loss and normalise exp_logits into softmax (in-place)
        softmax, chunk_loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        # Step 4 — optional label smoothing (matches original logic exactly)
        vocab_size = softmax.size(-1)  # partition_vocab_size
        if label_smoothing > 0:
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            log_probs = torch.log(softmax)
            mean_log_probs = log_probs.mean(dim=-1)
            chunk_loss = (1.0 - smoothing) * chunk_loss - smoothing * mean_log_probs

        return softmax, chunk_loss, target_mask, masked_target_1d

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0, chunk_size=4096):
        """Chunked forward pass.

        Args:
            vocab_parallel_logits: [seq_len, batch, vocab/TP], bf16
            target              : [seq_len, batch], int64
            label_smoothing     : float in [0.0, 1.0)
            chunk_size          : number of sequence positions per chunk

        Returns:
            loss: [seq_len, batch], fp32
        """
        seq_len, batch, partition_vocab_size = vocab_parallel_logits.shape
        device = vocab_parallel_logits.device

        # Allocate output buffer — small (seq_len * batch * 4 bytes)
        loss = torch.empty((seq_len, batch), dtype=torch.float32, device=device)

        # Vocab range is the same for every chunk; compute once
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        # Process sequence in chunks
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            # fp32 chunk lives only inside this iteration (~1 GiB for chunk=4096)
            _, chunk_loss, _, _ = _VocabParallelCrossEntropyChunked._run_chunk_forward(
                vocab_parallel_logits[start:end],
                target[start:end],
                vocab_start_index,
                vocab_end_index,
                label_smoothing,
            )

            loss[start:end] = chunk_loss
            # chunk tensors go out of scope here and are freed

        # Save only the bf16 input for backward recomputation (not fp32 softmax)
        ctx.save_for_backward(vocab_parallel_logits, target)
        ctx.chunk_size = chunk_size
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        ctx.label_smoothing = label_smoothing
        ctx.vocab_size = partition_vocab_size  # matches original ctx.vocab_size semantics

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Chunked backward pass with recomputation.

        Recomputes the forward pass per chunk (recovers softmax) instead of
        reading a stored fp32 softmax tensor, keeping peak memory at ~4.4 GiB.
        """
        vocab_parallel_logits, target = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        vocab_start_index = ctx.vocab_start_index
        vocab_end_index = ctx.vocab_end_index
        label_smoothing = ctx.label_smoothing
        vocab_size = ctx.vocab_size  # partition_vocab_size

        seq_len = vocab_parallel_logits.size(0)

        # Gradient tensor — same shape and dtype as the input logits (bf16)
        grad_input = torch.zeros_like(vocab_parallel_logits)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            # Recompute softmax for this chunk (~1 GiB fp32, freed at end of iteration)
            softmax, _, target_mask, masked_target_1d = (
                _VocabParallelCrossEntropyChunked._run_chunk_forward(
                    vocab_parallel_logits[start:end],
                    target[start:end],
                    vocab_start_index,
                    vocab_end_index,
                    label_smoothing,
                )
            )

            # Compute gradients — reuses original VocabParallelCrossEntropy logic
            (grad_2d, arange_1d, softmax_update, chunk_grad_input) = (
                VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                    softmax, target_mask
                )
            )

            if label_smoothing > 0:
                smoothing = label_smoothing * vocab_size / (vocab_size - 1)
                grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
                average_grad = 1 / vocab_size
                grad_2d[arange_1d, :] -= smoothing * average_grad
                chunk_grad_input.mul_(grad_output[start:end].unsqueeze(dim=-1))
            else:
                chunk_grad_input = VocabParallelCrossEntropy.calculate_gradients(
                    grad_2d,
                    arange_1d,
                    masked_target_1d,
                    softmax_update,
                    chunk_grad_input,
                    grad_output[start:end],
                )

            grad_input[start:end] = chunk_grad_input
            # softmax and intermediate tensors freed here

        # Return gradients for (vocab_parallel_logits, target, label_smoothing, chunk_size)
        return grad_input, None, None, None


def vocab_parallel_cross_entropy_chunked(
    vocab_parallel_logits,
    target,
    label_smoothing=0.0,
    chunk_size=4096,
):
    """Chunked version of vocab_parallel_cross_entropy with lower peak memory.

    Splits the sequence dimension into chunks so that only one chunk's fp32
    logits exists at a time. The backward pass recomputes the softmax per chunk
    instead of storing it from the forward pass.

    Args:
        vocab_parallel_logits: [seq_len, batch, vocab/TP], typically bf16
        target               : [seq_len, batch], int64
        label_smoothing      : float in [0.0, 1.0), default 0.0
        chunk_size           : sequence positions per chunk (default 4096).
                               Smaller values reduce peak memory further but
                               increase the number of all-reduce calls.

    Returns:
        loss: [seq_len, batch], fp32

    Memory (seq=28672, batch=1, vocab/TP=62080):
        Original : ~10.5 GiB peak
        chunk=4096: ~4.4 GiB peak  (58% reduction)
        chunk=8192: ~5.4 GiB peak  (49% reduction)
    """
    return _VocabParallelCrossEntropyChunked.apply(
        vocab_parallel_logits, target, label_smoothing, chunk_size
    )
