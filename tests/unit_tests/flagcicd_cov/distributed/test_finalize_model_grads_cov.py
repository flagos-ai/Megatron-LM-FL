# Copyright (c) 2026, BAAI. All rights reserved.
"""finalize_model_grads scales grads by global tokens and all-reduces embedding, router, and TP-parallel grads."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_router_grads,
    _allreduce_word_embedding_grads,
    _get_main_grad_attr,
    _get_position_embedding_weight,
    _reshard_if_dtensor,
    _unshard_if_dtensor,
    _update_router_expert_bias,
    finalize_model_grads,
    reset_model_temporary_tensors,
)
from megatron.core.timers import Timers
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RecordingChunk(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.finish_grad_sync_calls = 0
        self.recorded_scalings = []

    def finish_grad_sync(self, force_all_reduce=False):
        self.finish_grad_sync_calls += 1

    def scale_gradients(self, scaling_factor):
        self.recorded_scalings.append(scaling_factor)


class _WordEmbeddingChunk(torch.nn.Module):
    def __init__(self, use_distributed_optimizer=False, partial_reduce=False):
        super().__init__()
        self.pre_process = True
        self.share_embeddings_and_output_weights = True
        self.ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=use_distributed_optimizer,
            use_partial_reduce_for_shared_embedding=partial_reduce,
        )
        self.embedding = torch.nn.Embedding(8, 4)

    def shared_embedding_or_output_weight(self):
        return self.embedding.weight


class _PositionEmbeddingChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = True
        self.ddp_config = DistributedDataParallelConfig()
        self.position_embeddings = torch.nn.Embedding(8, 4)


class _RouterChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(4, 4, bias=False)
        self.router.weight.flextron_router_pp_sync = True


class _ExpertBiasRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_bias = torch.zeros(4)
        self.local_tokens_per_expert = torch.tensor([3.0, 1.0, 4.0, 1.0])


class _AuxLossModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tracker_calls = 0

    def reset_global_aux_loss_tracker(self):
        self.tracker_calls += 1


class _SequenceParallelChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ddp_config = DistributedDataParallelConfig()
        self.average = torch.nn.Linear(4, 4, bias=False)
        self.average.weight.average_gradients_across_tp_domain = True
        self.seq_parallel = torch.nn.Linear(4, 4, bias=False)
        self.seq_parallel.weight.sequence_parallel = True
        self.frozen_avg = torch.nn.Linear(4, 4, bias=False)
        self.frozen_avg.weight.average_gradients_across_tp_domain = True


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _pp_partner_rank():
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    ranks = torch.distributed.get_process_group_ranks(pp_group)
    return ranks[1 - parallel_state.get_pipeline_model_parallel_rank()]


def _tp_partner_rank():
    tp_group = parallel_state.get_tensor_model_parallel_group()
    ranks = torch.distributed.get_process_group_ranks(tp_group)
    return [r for r in ranks if r != torch.distributed.get_rank()][0]


def test_get_main_grad_attr_prefers_main_grad_over_grad():
    param = torch.nn.Parameter(torch.zeros(2))
    assert _get_main_grad_attr(param) == "grad"
    param.main_grad = torch.zeros(2)
    assert _get_main_grad_attr(param) == "main_grad"


def test_unshard_and_reshard_dtensor_round_trip(model_parallel):
    model_parallel()
    mesh = init_device_mesh("cuda", mesh_shape=(torch.distributed.get_world_size(),))
    global_tensor = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

    unsharded = _unshard_if_dtensor(dtensor)
    torch.testing.assert_close(unsharded, global_tensor, rtol=0, atol=0)

    resharded = _reshard_if_dtensor(unsharded, dtensor)
    assert isinstance(resharded, DTensor)
    assert resharded.placements == dtensor.placements
    torch.testing.assert_close(resharded.to_local(), dtensor.to_local(), rtol=0, atol=0)


def test_unshard_and_reshard_pass_plain_tensors_through():
    tensor = torch.arange(4.0)
    assert _unshard_if_dtensor(tensor) is tensor
    reference = torch.arange(4.0)
    assert _reshard_if_dtensor(tensor, reference) is reference


def test_get_position_embedding_weight_returns_submodule_weight():
    chunk = _PositionEmbeddingChunk()
    assert _get_position_embedding_weight(chunk) is chunk.position_embeddings.weight


def test_finalize_model_grads_scales_grads_by_global_num_tokens(model_parallel):
    model_parallel()
    config = TransformerConfig(num_layers=1, num_attention_heads=1, timers=Timers(2, "max"))
    chunk = _RecordingChunk(config)
    num_tokens = torch.tensor([float(torch.distributed.get_rank() + 1)], device="cuda")

    finalize_model_grads([chunk], num_tokens=num_tokens)

    assert chunk.finish_grad_sync_calls == 1
    assert chunk.recorded_scalings == [1.0 / 36.0]
    assert num_tokens.item() == 36.0


def _pp_pair_embedding_groups():
    """One process group per PP pair; each rank lists only its own pair.

    The list form of the embedding group carries the groups the rank belongs
    to (mirroring FSDPDistributedIndex), because torch.distributed.new_group
    returns a non-member sentinel on the other ranks.
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    groups = [dist.new_group([r, r + world_size // 2]) for r in range(world_size // 2)]
    return [groups[rank % (world_size // 2)]]


def test_word_embedding_grads_allreduce_with_single_group_list(model_parallel):
    model_parallel(pipeline_model_parallel_size=2)
    config = TransformerConfig(num_layers=1, num_attention_heads=1)
    chunk = _WordEmbeddingChunk().cuda()
    weight = chunk.embedding.weight
    rank = torch.distributed.get_rank()
    weight.main_grad = torch.full((8, 4), float(rank), device="cuda")
    embd_groups = _pp_pair_embedding_groups()
    pp_group = parallel_state.get_pipeline_model_parallel_group()

    _allreduce_word_embedding_grads([chunk], config, embd_groups, pp_group)

    partner = _pp_partner_rank()
    torch.testing.assert_close(
        weight.main_grad, torch.full((8, 4), float(rank + partner), device="cuda"), rtol=0, atol=0
    )


def test_word_embedding_grads_allreduce_with_parallel_state_embd_group(model_parallel):
    model_parallel(pipeline_model_parallel_size=2)
    config = TransformerConfig(num_layers=1, num_attention_heads=1)
    chunk = _WordEmbeddingChunk().cuda()
    weight = chunk.embedding.weight
    rank = torch.distributed.get_rank()
    weight.main_grad = torch.full((8, 4), float(rank), device="cuda")

    _allreduce_word_embedding_grads([chunk], config, None, None)

    # With pp=2 every rank's embedding group is its PP pair {r, r+4}, so each
    # rank reduces its own grad with its partner's.
    partner = _pp_partner_rank()
    torch.testing.assert_close(
        weight.main_grad, torch.full((8, 4), float(rank + partner), device="cuda"), rtol=0, atol=0
    )


def test_word_embedding_grads_partial_reduce_shards_rows_by_dp_rank(model_parallel):
    model_parallel(pipeline_model_parallel_size=2)
    config = TransformerConfig(num_layers=1, num_attention_heads=1)
    chunk = _WordEmbeddingChunk(use_distributed_optimizer=True, partial_reduce=True).cuda()
    weight = chunk.embedding.weight
    rank = torch.distributed.get_rank()
    weight.main_grad = torch.full((8, 4), float(rank), device="cuda")
    embd_groups = _pp_pair_embedding_groups()
    pp_group = parallel_state.get_pipeline_model_parallel_group()

    _allreduce_word_embedding_grads([chunk], config, embd_groups, pp_group)

    # Only the rows owned by this rank's DP group are reduced with the PP pair.
    partner = _pp_partner_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    per_partition = weight.main_grad.shape[0] // dp_world_size
    expected = torch.full((8, 4), float(rank), device="cuda")
    expected[per_partition * dp_rank : per_partition * (dp_rank + 1)] = float(rank + partner)
    torch.testing.assert_close(weight.main_grad, expected, rtol=0, atol=0)


def test_router_grads_sum_across_pp_and_vpp_chunks(model_parallel):
    model_parallel(pipeline_model_parallel_size=2)
    config = TransformerConfig(num_layers=1, num_attention_heads=1)
    config.flextron = True
    rank = torch.distributed.get_rank()
    chunks = [_RouterChunk().cuda() for _ in range(2)]
    for i, chunk in enumerate(chunks):
        chunk.router.weight.main_grad = torch.full((4, 4), float(rank * 10 + i), device="cuda")

    _allreduce_router_grads(chunks, config)

    partner = _pp_partner_rank()
    expected = float(rank * 10 * 2 + 1) + float(partner * 10 * 2 + 1)
    for chunk in chunks:
        torch.testing.assert_close(
            chunk.router.weight.main_grad,
            torch.full((4, 4), expected, device="cuda"),
            rtol=0,
            atol=0,
        )


def test_non_tensor_model_parallel_grads_average_and_sum(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    config = TransformerConfig(
        num_layers=1, num_attention_heads=2, sequence_parallel=True, tensor_model_parallel_size=2
    )
    chunk = _SequenceParallelChunk().cuda()
    rank = torch.distributed.get_rank()
    chunk.average.weight.grad = torch.full((4, 4), float(rank + 1), device="cuda")
    chunk.seq_parallel.weight.grad = torch.full((4, 4), float(rank + 1), device="cuda")

    _allreduce_non_tensor_model_parallel_grads([chunk], config)

    partner = _tp_partner_rank()
    expected_avg = (float(rank + 1) + float(partner + 1)) / 2
    expected_sum = float(rank + 1) + float(partner + 1)
    torch.testing.assert_close(
        chunk.average.weight.grad, torch.full((4, 4), expected_avg, device="cuda"), rtol=0, atol=0
    )
    torch.testing.assert_close(
        chunk.seq_parallel.weight.grad,
        torch.full((4, 4), expected_sum, device="cuda"),
        rtol=0,
        atol=0,
    )
    assert chunk.frozen_avg.weight.grad is None


def test_non_tensor_model_parallel_grads_with_megatron_fsdp_local_tensor(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    config = TransformerConfig(
        num_layers=1, num_attention_heads=2, sequence_parallel=True, tensor_model_parallel_size=2
    )
    chunk = _SequenceParallelChunk().cuda()
    chunk.ddp_config.use_megatron_fsdp = True
    rank = torch.distributed.get_rank()
    seq_grad = torch.full((4, 4), float(rank + 1), device="cuda")
    seq_grad._local_tensor = torch.full((4, 4), float(rank + 1), device="cuda")
    chunk.seq_parallel.weight.grad = seq_grad

    _allreduce_non_tensor_model_parallel_grads([chunk], config)

    partner = _tp_partner_rank()
    expected_sum = float(rank + 1) + float(partner + 1)
    torch.testing.assert_close(
        chunk.seq_parallel.weight.grad._local_tensor,
        torch.full((4, 4), expected_sum, device="cuda"),
        rtol=0,
        atol=0,
    )


def test_non_tensor_model_parallel_grads_noop_for_single_tp_rank(model_parallel):
    model_parallel()
    # Sequence parallelism requires TP>1 (model_parallel_config.py), and this
    # test only covers the TP=1 early return, which is independent of it.
    config = TransformerConfig(num_layers=1, num_attention_heads=1)
    chunk = _SequenceParallelChunk().cuda()
    chunk.seq_parallel.weight.grad = torch.full((4, 4), 7.0, device="cuda")

    _allreduce_non_tensor_model_parallel_grads([chunk], config)

    torch.testing.assert_close(
        chunk.seq_parallel.weight.grad, torch.full((4, 4), 7.0, device="cuda"), rtol=0, atol=0
    )


def test_reset_model_temporary_tensors_resets_global_aux_loss_tracker():
    config = TransformerConfig(
        num_layers=1, num_attention_heads=1, moe_router_load_balancing_type="global_aux_loss"
    )
    module = _AuxLossModule()

    reset_model_temporary_tensors(config, [module])

    assert module.tracker_calls == 1


def test_update_router_expert_bias_leaves_eval_mode_routers_unchanged():
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    )
    router = _ExpertBiasRouter()
    router.eval()
    expert_bias_before = router.expert_bias.clone()
    dense = torch.nn.Linear(4, 4)

    _update_router_expert_bias([dense, router], config)

    torch.testing.assert_close(router.expert_bias, expert_bias_before, rtol=0, atol=0)
    torch.testing.assert_close(
        router.local_tokens_per_expert, torch.tensor([3.0, 1.0, 4.0, 1.0]), rtol=0, atol=0
    )
