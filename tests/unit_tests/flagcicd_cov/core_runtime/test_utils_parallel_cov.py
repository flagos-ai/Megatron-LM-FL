# Copyright (c) 2026, BAAI. All rights reserved.
"""Process-group helpers, sharded-tensor construction and TP-rank batch broadcasting."""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

import megatron.core.parallel_state
import megatron.core.utils as utils
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_batch_on_this_tp_rank,
    get_data_parallel_group_if_dtensor,
    get_full_tensor_if_necessary,
    get_model_xattn,
    get_pg_rank,
    get_pg_size,
    get_pg_src_rank,
    get_tensor_model_parallel_group_if_none,
    is_submodule,
    log_on_each_pipeline_stage,
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
    prepare_input_tensors_for_wgrad_compute,
)


class _FakeGroup:
    """Stand-in for a torch.distributed process group."""

    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture
def distributed(monkeypatch):
    """Make the group helpers consult the fake groups instead of an unstarted default group."""
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)


def _replicated_dtensor():
    """Build a real DTensor over a single-rank mesh that skips process-group setup."""
    mesh = DeviceMesh('cpu', torch.arange(1), _init_backend=False, _rank=0)
    local = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return DTensor.from_local(local, mesh, [Replicate()]), local


def test_tp_group_helper_returns_none_when_distributed_is_not_initialized(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    assert get_tensor_model_parallel_group_if_none(None) is None


def test_tp_group_helper_passes_the_group_through_when_parallel_state_is_not_initialized(
    distributed, monkeypatch
):
    monkeypatch.setattr(megatron.core.parallel_state, 'is_initialized', lambda: False)
    group = _FakeGroup(0, 1)
    assert get_tensor_model_parallel_group_if_none(group) is group


def test_tp_group_helper_falls_back_to_the_expert_group(distributed, monkeypatch):
    expert_group = _FakeGroup(0, 1)
    monkeypatch.setattr(megatron.core.parallel_state, 'is_initialized', lambda: True)
    monkeypatch.setattr(
        megatron.core.parallel_state,
        'get_expert_tensor_parallel_group',
        lambda check_initialized=True: expert_group,
    )
    with pytest.warns(DeprecationWarning, match='tp_group is None, using default tp group'):
        assert get_tensor_model_parallel_group_if_none(None, is_expert=True) is expert_group


def test_pg_size_and_rank_use_the_first_group_of_a_list(distributed):
    groups = [_FakeGroup(2, 4), _FakeGroup(3, 8)]
    assert get_pg_size(groups) == 4
    assert get_pg_rank(groups) == 2
    assert get_pg_size(None) == 1
    assert get_pg_rank(None) == 0


def test_pg_src_rank_returns_the_first_global_rank_of_the_group(distributed, monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda group: [5, 6, 7, 8])
    assert get_pg_src_rank(_FakeGroup(3, 4)) == 5
    assert get_pg_src_rank(None) == 0


def test_attr_wrapped_model_rejects_a_list_of_models():
    with pytest.raises(RuntimeError, match='given a list of models'):
        get_attr_wrapped_model([torch.nn.Linear(2, 2)], 'model_type')


def test_model_xattn_is_false_when_the_model_has_no_xattn_flag():
    assert get_model_xattn(torch.nn.Linear(2, 2)) is False


def test_model_xattn_returns_the_flag_from_the_wrapped_model():
    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = torch.nn.Linear(2, 2)
            self.module.xattn_needed = True

    assert get_model_xattn(Wrapped()) is True


def test_log_on_each_pipeline_stage_rejects_half_supplied_groups(distributed):
    with pytest.raises(
        ValueError, match='tp_group and dp_cp_group must be provided or not provided together'
    ):
        log_on_each_pipeline_stage(
            logging.getLogger('test_pipeline_stage'),
            logging.INFO,
            'message',
            tp_group=_FakeGroup(0, 1),
        )


def test_param_hash_check_passes_when_there_are_no_parameters(monkeypatch):
    monkeypatch.setattr(
        megatron.core.parallel_state, 'get_data_parallel_group', lambda: _FakeGroup(0, 1)
    )
    monkeypatch.setattr(
        megatron.core.parallel_state, 'get_expert_data_parallel_group', lambda: _FakeGroup(0, 1)
    )
    assert check_param_hashes_across_dp_replicas([torch.nn.Sequential()], cross_check=True)


def test_tp_sharded_tensor_for_checkpoint_combines_fsdp_and_tp_shards(distributed):
    dtensor, local = _replicated_dtensor()
    sharded = make_tp_sharded_tensor_for_checkpoint(
        dtensor,
        'layers.0.weight',
        tp_axis=0,
        tp_group=_FakeGroup(1, 2),
        dp_cp_group=_FakeGroup(2, 4),
    )
    assert sharded.data is dtensor._local_tensor
    assert torch.equal(sharded.data, local)
    assert sharded.replica_id == (0, 0, 0)
    assert sharded.axis_fragmentations == (8, 1)
    assert sharded.global_offset == (12, 0)


def test_tp_sharded_tensor_for_checkpoint_appends_an_fsdp_offset_for_other_axes(distributed):
    dtensor, local = _replicated_dtensor()
    sharded = make_tp_sharded_tensor_for_checkpoint(
        dtensor,
        'layers.0.weight',
        tp_axis=1,
        tp_group=_FakeGroup(1, 2),
        dp_cp_group=_FakeGroup(2, 4),
    )
    assert torch.equal(sharded.data, local)
    assert sharded.replica_id == (0, 0, 0)
    assert sharded.axis_fragmentations == (4, 2)
    assert sharded.global_offset == (4, 3)


def test_sharded_tensor_for_checkpoint_falls_back_to_parallel_state_groups(monkeypatch):
    tp_group, dp_group = _FakeGroup(3, 4), _FakeGroup(1, 2)
    monkeypatch.setattr(
        megatron.core.parallel_state, 'get_tensor_model_parallel_group', lambda: tp_group
    )
    monkeypatch.setattr(
        megatron.core.parallel_state,
        'get_data_parallel_group',
        lambda with_context_parallel=False, partial_data_parallel=False: dp_group,
    )
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    tensor = torch.zeros(2, 2)
    sharded = make_sharded_tensor_for_checkpoint(tensor, 'embeddings.weight')
    assert sharded.data is tensor
    assert sharded.replica_id == (0, 3, 1)
    assert sharded.axis_fragmentations == (1, 1)


def test_sharded_tensor_for_checkpoint_treats_fsdp_shards_as_replicated(distributed):
    dtensor, local = _replicated_dtensor()
    sharded = make_sharded_tensor_for_checkpoint(
        dtensor, 'layers.0.weight', tp_group=_FakeGroup(1, 2), dp_cp_group=_FakeGroup(2, 4)
    )
    assert torch.equal(sharded.data, local)
    assert sharded.replica_id == (0, 1, 0)
    assert sharded.axis_fragmentations == (4, 1)
    assert sharded.global_offset == (4, 0)


def test_full_tensor_if_necessary_returns_the_local_shard_when_replicated():
    dtensor, local = _replicated_dtensor()
    assert torch.equal(get_full_tensor_if_necessary(dtensor), local)


def test_full_tensor_if_necessary_returns_the_local_shard_when_the_mesh_is_not_split():
    mesh = DeviceMesh('cpu', torch.arange(1), _init_backend=False, _rank=0)
    local = torch.arange(2, dtype=torch.float32).reshape(1, 2)
    dtensor = DTensor.from_local(local, mesh, [Shard(0)])
    assert torch.equal(get_full_tensor_if_necessary(dtensor), local)


def test_data_parallel_group_is_none_for_plain_tensors():
    group = _FakeGroup(0, 1)
    assert get_data_parallel_group_if_dtensor(torch.zeros(2)) is None
    assert get_data_parallel_group_if_dtensor(torch.zeros(2), data_parallel_group=group) is None


def test_wgrad_input_tensors_are_made_contiguous_and_flattened_to_2d():
    grad_output = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).transpose(0, 1)
    all_gathered_input = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    grad, gathered = prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input)
    assert grad.is_contiguous()
    assert gathered.is_contiguous()
    assert grad.shape == (4, 2)
    assert gathered.shape == (4, 2)
    assert torch.equal(grad, torch.tensor([[0.0, 1.0], [4.0, 5.0], [2.0, 3.0], [6.0, 7.0]]))


def test_embedding_wgrad_drain_requires_matching_buffer_lengths():
    with pytest.raises(
        AssertionError, match='Length of activation and gradient buffers need to be equal'
    ):
        drain_embedding_wgrad_compute(
            config=None,
            embedding_activation_buffer=[torch.zeros(1)],
            grad_output_buffer=[],
            weight=None,
            tp_group=None,
        )


def test_is_submodule_rejects_a_module_compared_with_itself_in_strict_mode():
    module = torch.nn.Linear(2, 2)
    assert is_submodule(module, module) is False
    assert is_submodule(module, module, strict=False) is True


def test_is_submodule_detects_nested_and_unrelated_modules():
    parent = torch.nn.Sequential(torch.nn.Linear(2, 2))
    nested = torch.nn.Sequential(torch.nn.Linear(2, 2))
    parent.add_module('nested', nested)
    assert is_submodule(parent[0], parent) is True
    assert is_submodule(nested[0], parent) is True
    assert is_submodule(torch.nn.Linear(2, 2), parent) is False


@pytest.fixture
def tp_broadcast(monkeypatch):
    """Record every broadcast a TP group makes and pin the device to the CPU."""
    calls = []

    def fake_broadcast(tensor, src, group=None):
        calls.append((tensor, src, group))
        if tensor.dim() == 0:
            tensor.fill_(0)

    monkeypatch.setattr(torch.distributed, 'broadcast', fake_broadcast)
    monkeypatch.setattr(
        utils,
        'cur_platform',
        SimpleNamespace(device=lambda value: torch.device('cpu'), current_device=lambda: 'cpu'),
    )
    return calls


def _tensor_shapes(calls):
    return [(tensor.dtype, tuple(tensor.shape)) for tensor, _, _ in calls]


def test_tp_rank_zero_on_first_stage_broadcasts_dualpipev_tensors(tp_broadcast):
    batch = {
        'tokens': torch.ones(2, 3, dtype=torch.int64),
        'labels': torch.full((2, 3), 5, dtype=torch.int64),
        'loss_mask': torch.ones(2, 3),
        'position_ids': torch.arange(6).reshape(2, 3),
    }
    result = get_batch_on_this_tp_rank(
        batch,
        is_sft=False,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=False,
        broadcast_src_rank=0,
        broadcast_group='tp-group',
        cp_size=1,
        tp_rank=0,
        micro_batch_size=2,
        seq_length=3,
        mtp_on_this_rank=False,
        pipeline_model_parallel_size=2,
        is_pipeline_first_stage=True,
        is_dualpipev=True,
    )
    assert result is batch
    assert result['labels'] is batch['labels']
    assert result['loss_mask'] is batch['loss_mask']
    assert _tensor_shapes(tp_broadcast) == [
        (torch.int64, (2, 3)),
        (torch.float32, (2, 3)),
        (torch.int64, (2, 3)),
        (torch.int64, (2, 3)),
    ]
    assert torch.equal(tp_broadcast[0][0], batch['labels'])
    assert torch.equal(tp_broadcast[1][0], batch['loss_mask'])
    assert all(call[1:] == (0, 'tp-group') for call in tp_broadcast)


def test_non_zero_tp_rank_on_dualpipev_first_stage_receives_label_tensors(tp_broadcast):
    result = get_batch_on_this_tp_rank(
        {},
        is_sft=False,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=False,
        broadcast_src_rank=0,
        broadcast_group='tp-group',
        cp_size=1,
        tp_rank=1,
        micro_batch_size=2,
        seq_length=3,
        mtp_on_this_rank=False,
        pipeline_model_parallel_size=2,
        is_pipeline_first_stage=True,
        is_dualpipev=True,
    )
    assert _tensor_shapes(tp_broadcast) == [
        (torch.int64, (2, 3)),
        (torch.float32, (2, 3)),
        (torch.int64, (2, 3)),
        (torch.int64, (2, 3)),
    ]
    assert result['tokens'].dtype == torch.int64
    assert result['labels'].dtype == torch.int64
    assert result['loss_mask'].dtype == torch.float32
    assert result['position_ids'].dtype == torch.int64
    assert result['attention_mask'] is None
    assert result['cu_seqlens'] is None


def test_non_zero_tp_rank_on_intermediate_stage_receives_sft_metadata(tp_broadcast):
    result = get_batch_on_this_tp_rank(
        {},
        is_sft=True,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=False,
        broadcast_src_rank=4,
        broadcast_group='tp-group',
        cp_size=2,
        tp_rank=1,
        micro_batch_size=1,
        seq_length=5,
        mtp_on_this_rank=False,
        pipeline_model_parallel_size=3,
    )
    assert _tensor_shapes(tp_broadcast) == [
        (torch.int64, ()),
        (torch.int32, (1,)),
        (torch.int64, ()),
    ]
    assert result['tokens'] is None
    assert result['labels'] is None
    assert result['loss_mask'] is None
    assert result['position_ids'] is None
    assert result['cu_seqlens'] is None
    assert result['cu_seqlens_padded'] is None
    assert result['max_seqlen'].dtype == torch.int32
    assert tuple(result['max_seqlen'].shape) == (1,)
    assert all(call[1] == 4 for call in tp_broadcast)
