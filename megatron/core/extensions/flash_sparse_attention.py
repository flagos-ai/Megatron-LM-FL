# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Flash Sparse Attention Context Parallel Support."""

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup

from megatron.core.ssm.mamba_context_parallel import _all_to_all_cp2hp, _all_to_all_hp2cp
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region


def _fsa_headwise_cp_forward(
    query: Tensor,           # [sq_local, b, np_local, hn]
    key: Tensor,             # [sq_local, b, nkv_local, hn]
    value: Tensor,           # [sq_local, b, nkv_local, hn]
    window_sizes: Tensor,    # [nkv_local, 4] — TP-local complete config
    cp_group: ProcessGroup,
    cp_size: int,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    softmax_threshold: float = 0.5,
) -> Tensor:
    """
    Headwise CP for FSA with hybrid communication mode:
    - Q heads: all-to-all (head dimension split)
    - K/V heads: all-to-all if sufficient, else AllGather sequence dimension

    Key design:
    - window_sizes only needs TP-awareness (guaranteed by window_sizes_heuristic)
    - When num_kv_heads_per_tp < cp_size, K/V use AllGather sequence dimension
    - When num_kv_heads_per_tp >= cp_size, K/V use all-to-all (same as Q)

    Args:
        query: [sq_local, b, np_local, hn] - Q tensor
        key: [sq_local, b, nkv_local, hn] - K tensor
        value: [sq_local, b, nkv_local, hn] - V tensor
        window_sizes: [nkv_local, 4] - TP-local window configuration
        cp_group: Context parallel process group
        cp_size: Context parallel size
        num_q_heads_per_tp: Number of Q heads per TP rank
        num_kv_heads_per_tp: Number of KV heads per TP rank
        softmax_threshold: Threshold for sparse attention

    Returns:
        output: [sq_local, b, np_local, hn] - Attention output
    """
    try:
        from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_func
    except ImportError as exc:
        raise ImportError(
            "flash_sparse_attn library is required for FSA headwise CP. "
            "Please install it from https://github.com/FlagOpen/flash_sparse_attn"
        ) from exc

    cp_rank = dist.get_rank(cp_group)
    sq_local, b, np_local, hn = query.shape

    # Step 1: Q heads - all-to-all (sequence -> head dimension)
    # _all_to_all_cp2hp expects 3-d input [seq, batch, hidden]
    # Reshape: [sq_local, b, np_local, hn] -> [sq_local, b, np_local * hn]
    if num_q_heads_per_tp % cp_size != 0:
        raise ValueError(
            f"Q heads per TP ({num_q_heads_per_tp}) must be divisible by "
            f"CP size ({cp_size})"
        )
    q_3d = query.reshape(sq_local, b, np_local * hn)
    q_full_seq_3d = _all_to_all_cp2hp(q_3d, cp_group)
    # [sq_global, b, np_local * hn / cp_size] = [sq_global, b, (np_local/cp_size) * hn]
    sq_global = sq_local * cp_size
    num_q_heads_per_rank = np_local // cp_size
    q_full_seq = q_full_seq_3d.reshape(sq_global, b, num_q_heads_per_rank, hn)
    # [sq_global, b, num_q_heads_per_rank, hn]

    # Global Q head index for this rank's first Q head
    first_q_head_global_idx = cp_rank * num_q_heads_per_rank

    # Step 2: K/V heads - choose communication mode based on head count
    nkv_local = key.shape[2]
    if num_kv_heads_per_tp >= cp_size and num_kv_heads_per_tp % cp_size == 0:
        # Scenario 1: Sufficient KV heads, use all-to-all
        # After all-to-all, Q and KV heads are naturally aligned:
        #   rank r gets Q heads [r*nq_per_rank : (r+1)*nq_per_rank]
        #   rank r gets KV heads [r*nkv_per_rank : (r+1)*nkv_per_rank]
        # GQA mapping is preserved within each rank.
        k_3d = key.reshape(sq_local, b, nkv_local * hn)
        v_3d = value.reshape(sq_local, b, nkv_local * hn)
        k_full_seq_3d = _all_to_all_cp2hp(k_3d, cp_group)
        v_full_seq_3d = _all_to_all_cp2hp(v_3d, cp_group)
        num_kv_heads_per_rank = nkv_local // cp_size
        k_full_seq = k_full_seq_3d.reshape(sq_global, b, num_kv_heads_per_rank, hn)
        v_full_seq = v_full_seq_3d.reshape(sq_global, b, num_kv_heads_per_rank, hn)
        # [sq_global, b, num_kv_heads_per_rank, hn]

        # Window sizes for this rank's KV heads
        kv_start = cp_rank * num_kv_heads_per_rank
        kv_end = kv_start + num_kv_heads_per_rank
        window_sizes_local = window_sizes[kv_start:kv_end, :]

    else:
        # Scenario 2: Insufficient KV heads, AllGather sequence dimension
        # All Q heads on this rank map to the same KV head
        # (because num_q_per_kv = np_local/nkv_local >= np_local/cp_size = num_q_per_rank)
        k_3d = key.reshape(sq_local, b, nkv_local * hn)
        v_3d = value.reshape(sq_local, b, nkv_local * hn)
        k_full_seq_3d = gather_from_sequence_parallel_region(k_3d, group=cp_group)
        v_full_seq_3d = gather_from_sequence_parallel_region(v_3d, group=cp_group)
        k_full_seq = k_full_seq_3d.reshape(sq_global, b, nkv_local, hn)
        v_full_seq = v_full_seq_3d.reshape(sq_global, b, nkv_local, hn)
        # [sq_global, b, nkv_local, hn] — all KV heads on full sequence

        # Select the KV head corresponding to this rank's Q heads
        num_q_per_kv = num_q_heads_per_tp // num_kv_heads_per_tp
        kv_head_idx = first_q_head_global_idx // num_q_per_kv

        k_full_seq = k_full_seq[:, :, kv_head_idx:kv_head_idx+1, :]
        v_full_seq = v_full_seq[:, :, kv_head_idx:kv_head_idx+1, :]
        # [sq_global, b, 1, hn]

        window_sizes_local = window_sizes[kv_head_idx:kv_head_idx+1, :]

    # Step 3: FSA kernel (pack_gqa=False handles GQA natively)
    output = flash_sparse_attn_func(
        q_full_seq, k_full_seq, v_full_seq,
        window_sizes=window_sizes_local,
        softmax_threshold=softmax_threshold,
        pack_gqa=False,
    )  # [sq_global, b, num_q_heads_per_rank, hn]

    # Step 4: All-to-all backward (head -> sequence dimension)
    # Reshape to 3-d: [sq_global, b, num_q_heads_per_rank * hn]
    output_3d = output.reshape(sq_global, b, num_q_heads_per_rank * hn)
    output_sp_3d = _all_to_all_hp2cp(output_3d, cp_group)
    # [sq_local, b, np_local * hn]
    output_sp = output_sp_3d.reshape(sq_local, b, np_local, hn)
    return output_sp
