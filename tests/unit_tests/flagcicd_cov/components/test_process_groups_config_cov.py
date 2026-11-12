# Copyright (c) 2026, BAAI. All rights reserved.
"""Process-group collection setup: optimizer and DDP group validation, fallbacks for missing ddp_config and provided collections, and helper metaclass attribute guards."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupHelperMeta


class _Group:
    def __init__(self, name):
        self.name = name

    def size(self):
        return 2

    def __repr__(self):
        return f'Group({self.name})'


@pytest.fixture
def single_rank_parallel_state(tmp_path):
    dist.init_process_group(
        backend='gloo', init_method=f'file://{tmp_path}/rendezvous', world_size=1, rank=0
    )
    parallel_state.initialize_model_parallel(1, 1)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_helper_meta_blocks_direct_virtual_pipeline_size_assignment():
    class _Sample(metaclass=ProcessGroupHelperMeta):
        pass

    with pytest.raises(
        AttributeError, match="Cannot set 'virtual_pipeline_model_parallel_size' directly"
    ):
        _Sample.virtual_pipeline_model_parallel_size = 2


def test_helper_meta_allows_setting_other_class_attributes():
    class _Sample(metaclass=ProcessGroupHelperMeta):
        pass

    _Sample.allowed_field = 3
    assert _Sample.allowed_field == 3


def test_process_group_collection_rejects_unknown_attribute():
    with pytest.raises(ValueError, match='Unknown attribute: unexpected'):
        ProcessGroupCollection(unexpected=_Group('x'))


def test_process_group_collection_repr_marks_none_groups():
    collection = ProcessGroupCollection()
    collection.dp = None
    assert 'dp(None)' in repr(collection)


def test_setup_process_groups_for_optimizer_without_collection_disables_gloo(
    single_rank_parallel_state,
):
    model = SimpleNamespace(
        ddp_config=SimpleNamespace(
            num_distributed_optimizer_instances=1, use_distributed_optimizer=False
        )
    )
    groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
        None, [model], use_gloo_process_groups=False
    )
    assert groups['intra_dp_cp_group_gloo'] is None
    assert groups['intra_expt_dp_group_gloo'] is None
    assert groups['engram_dp_group_gloo'] is None
    assert groups['inter_dist_opt_group'] is None
    assert groups['dp_group'].size() == 1
    assert groups['mp_group'].size() == 1


def test_setup_process_groups_for_optimizer_without_collection_without_ddp_config(
    single_rank_parallel_state,
):
    model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=1))
    groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
        None, [model], use_gloo_process_groups=False
    )
    assert groups['inter_dist_opt_group'] is None
    assert groups['intra_dist_opt_group'].size() == 1
    assert groups['dp_group'].size() == 1
    assert groups['dp_cp_group'].size() == 1
    assert groups['intra_dp_cp_group'].size() == 1
    assert groups['expt_dp_group'].size() == 1
    assert groups['intra_expt_dp_group'].size() == 1
    assert groups['mp_group'].size() == 1
    assert groups['expt_tp_pp_group'].size() == 1


def test_setup_process_groups_for_optimizer_requires_dp_cp_for_context_parallel():
    model = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=2),
        ddp_config=SimpleNamespace(
            num_distributed_optimizer_instances=1, use_distributed_optimizer=False
        ),
    )
    collection = ProcessGroupCollection(
        dp=_Group('dp'), expt_dp=None, mp=_Group('mp'), tp_ep_pp=_Group('tp_ep_pp')
    )
    with pytest.raises(ValueError, match='dp_cp process group is required'):
        ProcessGroupCollection.setup_process_groups_for_optimizer(
            collection, [model], use_gloo_process_groups=False
        )


def test_setup_process_groups_for_optimizer_requires_intra_groups_for_multiple_instances():
    model = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1),
        ddp_config=SimpleNamespace(
            num_distributed_optimizer_instances=2, use_distributed_optimizer=False
        ),
    )
    collection = ProcessGroupCollection(
        dp=_Group('dp'),
        dp_cp=_Group('dp_cp'),
        expt_dp=_Group('expt_dp'),
        mp=_Group('mp'),
        tp_ep_pp=_Group('tp_ep_pp'),
    )
    with pytest.raises(
        ValueError, match='intra_dp_cp, intra_expt_dp, inter_dist_opt, and intra_dist_opt'
    ):
        ProcessGroupCollection.setup_process_groups_for_optimizer(
            collection, [model], use_gloo_process_groups=False
        )


def test_setup_process_groups_for_optimizer_falls_back_without_ddp_config():
    model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=1))
    collection = ProcessGroupCollection(
        dp=_Group('dp'),
        dp_cp=_Group('dp_cp'),
        expt_dp=_Group('expt_dp'),
        mp=_Group('mp'),
        tp_ep_pp=_Group('tp_ep_pp'),
    )
    groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
        collection, [model], use_gloo_process_groups=False
    )
    assert groups['intra_dp_cp_group'] is collection.dp_cp
    assert groups['intra_expt_dp_group'] is collection.expt_dp
    assert groups['inter_dist_opt_group'] is None
    assert groups['intra_dist_opt_group'] is None


def test_setup_process_groups_for_ddp_requires_dp_group():
    collection = ProcessGroupCollection(expt_dp=None)
    with pytest.raises(ValueError, match='dp process group is required'):
        ProcessGroupCollection.setup_process_groups_for_ddp(
            collection,
            SimpleNamespace(context_parallel_size=1),
            SimpleNamespace(num_distributed_optimizer_instances=1),
        )


def test_setup_process_groups_for_ddp_requires_intra_groups_for_multiple_instances():
    collection = ProcessGroupCollection(
        dp=_Group('dp'), dp_cp=_Group('dp_cp'), expt_dp=_Group('expt_dp')
    )
    with pytest.raises(ValueError, match='intra_dp_cp, intra_expt_dp, and inter_dist_opt'):
        ProcessGroupCollection.setup_process_groups_for_ddp(
            collection,
            SimpleNamespace(context_parallel_size=1),
            SimpleNamespace(num_distributed_optimizer_instances=2),
        )


def test_setup_process_groups_for_ddp_uses_provided_intra_groups_for_multiple_instances():
    collection = ProcessGroupCollection(
        dp=_Group('dp'),
        dp_cp=_Group('dp_cp'),
        expt_dp=_Group('expt_dp'),
        intra_dp_cp=_Group('intra_dp_cp'),
        intra_expt_dp=_Group('intra_expt_dp'),
        inter_dist_opt=_Group('inter_dist_opt'),
        tp=_Group('tp'),
        pp=_Group('pp'),
        ep=_Group('ep'),
    )
    groups = ProcessGroupCollection.setup_process_groups_for_ddp(
        collection,
        SimpleNamespace(context_parallel_size=1),
        SimpleNamespace(num_distributed_optimizer_instances=2),
    )
    assert groups['intra_dp_cp_group'] is collection.intra_dp_cp
    assert groups['intra_expt_dp_group'] is collection.intra_expt_dp
    assert groups['inter_dist_opt_group'] is collection.inter_dist_opt
    assert groups['engram_dp_group'] is None


def test_setup_process_groups_for_ddp_requires_tp_pp_ep_groups():
    collection = ProcessGroupCollection(
        dp=_Group('dp'), dp_cp=_Group('dp_cp'), expt_dp=_Group('expt_dp')
    )
    with pytest.raises(ValueError, match='tp, pp and ep process groups are required'):
        ProcessGroupCollection.setup_process_groups_for_ddp(
            collection,
            SimpleNamespace(context_parallel_size=1),
            SimpleNamespace(num_distributed_optimizer_instances=1),
        )
