# Copyright (c) 2026, BAAI. All rights reserved.
"""Distributed engram embeddings: seeded shard init, all-to-all/all-reduce lookups, sharded state."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import engram
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _gather_full_table(weight, group):
    world_size = torch.distributed.get_world_size(group)
    shards = [torch.empty_like(weight) for _ in range(world_size)]
    torch.distributed.all_gather(shards, weight, group=group)
    return torch.cat(shards, dim=0)


def test_engram_embedding_gpu_init_repeats_for_same_seed_and_varies_with_engram_seed(
    model_parallel,
):
    model_parallel(engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    group = parallel_state.get_engram_embedding_parallel_group()

    def make_config(engram_seed):
        return TransformerConfig(
            num_layers=1,
            num_attention_heads=1,
            hidden_size=8,
            use_cpu_initialization=False,
            perform_initialization=True,
            engram_seed=engram_seed,
        )

    first = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=make_config(0),
        embedding_parallel_group=group,
    )
    second = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=make_config(0),
        embedding_parallel_group=group,
    )
    third = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=make_config(7),
        embedding_parallel_group=group,
    )
    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    assert not torch.allclose(first.weight, third.weight)
    assert first.embedding_parallel_size == 2
    assert first.num_embeddings_per_partition == 8
    assert first.vocab_start_index == 8 * first.embedding_parallel_rank
    assert first.vocab_end_index == first.vocab_start_index + 8
    assert first.weight.device.type == "cuda"
    assert first.weight.tensor_model_parallel is True
    assert first.weight.partition_dim == 0
    assert first.weight.partition_stride == 1


def test_engram_embedding_alltoall_round_trips_lookups_and_gradient_counts(model_parallel):
    model_parallel(engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    group = parallel_state.get_engram_embedding_parallel_group()
    embedding = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=False, perform_initialization=True),
        embedding_parallel_group=group,
    )
    full_table = _gather_full_table(embedding.weight.data, group)
    rank = torch.distributed.get_rank()
    torch.manual_seed(1000 + rank)
    input_ids = torch.randint(0, 16, (2, 6), device="cuda")
    output = embedding(input_ids)
    assert output.shape == (2, 6, 8)
    torch.testing.assert_close(output, full_table[input_ids], rtol=0, atol=0)
    output.sum().backward()
    gathered_ids = [torch.empty_like(input_ids) for _ in range(2)]
    torch.distributed.all_gather(gathered_ids, input_ids, group=group)
    counts = torch.bincount(torch.cat(gathered_ids).flatten(), minlength=16)
    expected_grad = (
        counts[embedding.vocab_start_index : embedding.vocab_end_index]
        .to(torch.float32)
        .unsqueeze(1)
        .expand(-1, 8)
    )
    torch.testing.assert_close(embedding.weight.grad, expected_grad, rtol=0, atol=0)


def test_engram_embedding_slices_input_columns_per_sp_rank_when_reduce_scatter_enabled(
    model_parallel,
):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    embedding = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        reduce_scatter_embeddings=True,
        config=ModelParallelConfig(use_cpu_initialization=False, perform_initialization=True),
    )
    torch.manual_seed(19)
    input_ids = torch.randint(0, 16, (2, 8), device="cuda")
    output = embedding(input_ids)
    assert output.shape == (4, 2, 8)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_rank == 0:
        columns = input_ids[:, :4]
    else:
        columns = input_ids[:, 4:]
    expected = F.embedding(columns, embedding.weight.data).transpose(0, 1).contiguous()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_engram_embedding_sharded_state_dict_marks_vocab_shard_and_dp_replica(model_parallel):
    model_parallel(engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    group = parallel_state.get_engram_embedding_parallel_group()
    embedding = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=False, perform_initialization=True),
        embedding_parallel_group=group,
    )
    sharded = embedding.sharded_state_dict(prefix="module.")
    assert list(sharded) == ["module.weight"]
    tensor = sharded["module.weight"]
    assert tensor.local_shape == (8, 8)
    assert tensor.global_shape == (16, 8)
    assert tensor.global_offset == (8 * embedding.embedding_parallel_rank, 0)
    assert tensor.axis_fragmentations == (2, 1)
    dp_rank = torch.distributed.get_rank(group=parallel_state.get_engram_data_parallel_group())
    assert tensor.replica_id == (0, 0, dp_rank)


@pytest.mark.parametrize(
    "offload_optimizer_states", [pytest.param(True, id="offload"), pytest.param(False, id="keep")]
)
def test_multi_head_embedding_alltoall_flags_and_matches_full_table(
    model_parallel, offload_optimizer_states
):
    model_parallel(engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        hidden_size=8,
        use_cpu_initialization=False,
        perform_initialization=True,
        engram_embedding_parallel_method="alltoall",
        engram_offload_embedding_optimizer_states=offload_optimizer_states,
    )
    multi_head = engram.MultiHeadEmbedding(config, list_of_N=[7, 5], D=4).cuda()
    assert multi_head.embedding.weight.is_engram_embedding is True
    assert multi_head.embedding.weight.allreduce is False
    if offload_optimizer_states:
        assert multi_head.embedding.weight.is_offloading_candidate is True
    else:
        assert getattr(multi_head.embedding.weight, "is_offloading_candidate", False) is False
    group = parallel_state.get_engram_embedding_parallel_group()
    full_table = _gather_full_table(multi_head.embedding.weight.data, group)
    torch.manual_seed(23)
    head_zero = torch.randint(0, 7, (2, 4))
    head_one = torch.randint(0, 5, (2, 4))
    input_ids = torch.stack([head_zero, head_one], dim=-1).cuda()
    output = multi_head(input_ids)
    assert output.shape == (4, 2, 2, 4)
    shifted = input_ids + multi_head.offsets
    expected = F.embedding(shifted, full_table).transpose(0, 1).contiguous()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_multi_head_embedding_allreduce_matches_full_tp_table(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        hidden_size=8,
        use_cpu_initialization=False,
        perform_initialization=True,
        engram_embedding_parallel_method="allreduce",
    )
    multi_head = engram.MultiHeadEmbedding(config, list_of_N=[7, 5], D=4).cuda()
    tp_group = multi_head.tp_group
    full_table = _gather_full_table(multi_head.embedding.weight.data, tp_group)
    torch.manual_seed(29)
    head_zero = torch.randint(0, 7, (2, 4))
    head_one = torch.randint(0, 5, (2, 4))
    input_ids = torch.stack([head_zero, head_one], dim=-1).cuda()
    output = multi_head(input_ids)
    assert output.shape == (4, 2, 2, 4)
    shifted = input_ids + multi_head.offsets
    expected = F.embedding(shifted, full_table).transpose(0, 1).contiguous()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
