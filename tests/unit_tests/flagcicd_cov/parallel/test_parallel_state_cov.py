# Copyright (c) 2026, BAAI. All rights reserved.
'''Model-parallel process-group layout: rank generation for tp/pp/cp/ep combinations and order strings, hierarchical context parallel, NCCL options, all-gather groups, and destroy/re-init cycles.'''

import os
from datetime import timedelta

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)

    yield setup
    Utils.destroy_model_parallel()


def test_rank_generator_offsets_ranks_when_rank_offset_is_positive():
    generator = ps.RankGenerator(
        tp=2, ep=1, dp=2, pp=2, cp=1, order='tp-cp-ep-dp-pp', rank_offset=8
    )
    assert generator.get_ranks('tp') == [[8, 9], [10, 11], [12, 13], [14, 15]]
    assert generator.get_ranks('pp') == [[8, 12], [9, 13], [10, 14], [11, 15]]
    assert generator.get_ranks('dp') == [[8, 10], [9, 11], [12, 14], [13, 15]]


def test_rank_generator_raises_when_sized_parallelism_missing_from_order():
    with pytest.raises(RuntimeError, match=r'The size of \(pp\) is \(2\)'):
        ps.RankGenerator(tp=2, ep=1, dp=1, pp=2, cp=1, order='tp-dp')


def test_get_nccl_options_returns_none_without_matching_config():
    assert ps.get_nccl_options('dp', {}) is None
    assert ps.get_nccl_options('dp', {'tp': {'max_ctas': 8}}) is None
    assert ps.get_nccl_options('dp', {'dp_cp': {'max_ctas': 8}}) is None


def test_get_nccl_options_applies_configured_nccl_fields():
    options = ps.get_nccl_options(
        'dp_cp',
        {
            'dp_cp': {
                'cga_cluster_size': 4,
                'max_ctas': 16,
                'min_ctas': 2,
                'net_name': 'IB',
                'is_high_priority_stream': True,
            }
        },
    )
    assert options.is_high_priority_stream is True
    assert options.config.cga_cluster_size == 4
    assert options.config.max_ctas == 16
    assert options.config.min_ctas == 2
    assert options.config.net_name == 'IB'


def test_get_nccl_options_rejects_unsupported_net_name():
    with pytest.raises(RuntimeError, match=r'net_name \(tcp\) is not supported'):
        ps.get_nccl_options('dp', {'dp': {'net_name': 'tcp'}})


def test_initialize_model_parallel_tp2_pp2_builds_expected_groups(model_parallel):
    model_parallel(2, 2)
    rank = torch.distributed.get_rank()
    tp_rank = rank % 2
    pp_rank = rank // 4

    assert ps.get_tensor_model_parallel_world_size() == 2
    assert ps.get_pipeline_model_parallel_world_size() == 2
    assert ps.get_data_parallel_world_size() == 2
    assert ps.get_tensor_model_parallel_group().size() == 2
    assert ps.get_model_parallel_group().size() == 4
    assert ps.get_tensor_model_parallel_rank() == tp_rank
    assert ps.get_pipeline_model_parallel_rank() == pp_rank
    assert ps.get_tensor_model_parallel_src_rank() == rank - tp_rank
    assert ps.get_model_parallel_src_rank() == 2 * ((rank // 2) % 2)
    assert ps.get_data_parallel_src_rank() == (rank % 2) + 4 * (rank // 4)

    pipeline_first = rank - 4 * pp_rank
    assert ps.get_pipeline_model_parallel_first_rank() == pipeline_first
    assert ps.get_pipeline_model_parallel_last_rank() == pipeline_first + 4
    assert ps.get_pipeline_model_parallel_next_rank() == (
        pipeline_first + 4 if pp_rank == 0 else pipeline_first
    )
    assert ps.get_pipeline_model_parallel_prev_rank() == (
        pipeline_first + 4 if pp_rank == 0 else pipeline_first
    )
    assert ps.get_last_rank_when_using_pipeline() == 7
    assert ps.get_all_ranks() == f'{tp_rank}_{(rank // 2) % 2}_0_{pp_rank}_0'
    assert ps.is_pipeline_first_stage() == (pp_rank == 0)
    assert ps.is_pipeline_last_stage() == (pp_rank == 1)
    assert ps.is_rank_in_embedding_group() is True
    assert ps.is_rank_in_position_embedding_group() == (rank < 4)
    assert ps.get_embedding_group().size() == 2
    # Process groups are only stored on their member ranks: the first stage holds the
    # position-embedding group, the last stage does not.
    if rank < 4:
        assert ps.get_position_embedding_group().size() == 1
    assert ps.get_virtual_pipeline_model_parallel_world_size() is None
    assert ps.get_amax_reduction_group().size() == 4
    assert ps.get_amax_reduction_group(tp_only_amax_red=True).size() == 2
    assert ps.get_tensor_and_data_parallel_group().size() == 4


def test_initialize_model_parallel_with_virtual_pipeline_tracks_stages(model_parallel):
    model_parallel(2, 2, virtual_pipeline_model_parallel_size=2)
    rank = torch.distributed.get_rank()

    assert ps.get_virtual_pipeline_model_parallel_rank() == 0
    assert ps.get_virtual_pipeline_model_parallel_world_size() == 2
    assert ps.is_rank_in_embedding_group(ignore_virtual=False, vp_stage=0) == (rank < 4)
    assert ps.is_rank_in_embedding_group(ignore_virtual=False, vp_stage=1) == (rank >= 4)


def test_initialize_model_parallel_dualpipev_reports_two_virtual_stages(model_parallel):
    model_parallel(2, 2, create_dualpipev_parallel_size=True)
    rank = torch.distributed.get_rank()
    pp_rank = rank // 4

    assert ps.get_dualpipev_pipeline_model_parallel_world_size() == 2
    assert ps.is_pipeline_first_stage(ignore_dualpipev=False, dualpipev_stage=0) == (pp_rank == 0)
    assert ps.is_pipeline_first_stage(ignore_dualpipev=False, dualpipev_stage=1) is False
    assert ps.is_pipeline_last_stage(ignore_dualpipev=False, dualpipev_stage=0) is False
    assert ps.is_pipeline_last_stage(ignore_dualpipev=False, dualpipev_stage=1) == (pp_rank == 0)


def test_initialize_model_parallel_with_tp_pp_dp_order_changes_pipeline_groups(model_parallel):
    model_parallel(2, 2, order='tp-pp-dp')
    rank = torch.distributed.get_rank()
    tp_rank = rank % 2
    pp_rank = (rank // 2) % 2

    assert ps.get_tensor_model_parallel_world_size() == 2
    assert ps.get_data_parallel_world_size() == 2
    assert ps.get_pipeline_model_parallel_first_rank() == rank - 2 * pp_rank
    assert ps.get_pipeline_model_parallel_last_rank() == rank - 2 * pp_rank + 2
    assert ps.get_tensor_model_parallel_src_rank() == rank - tp_rank


def test_initialize_model_parallel_rejects_indivisible_model_size(model_parallel):
    with pytest.raises(RuntimeError, match=r'world_size \(8\) is not divisible by 3'):
        model_parallel(3, 1)


def test_initialize_model_parallel_rejects_virtual_pipeline_without_pp(model_parallel):
    with pytest.raises(RuntimeError, match='pipeline-model-parallel size should be greater than 1'):
        model_parallel(1, 1, virtual_pipeline_model_parallel_size=2)


def test_initialize_model_parallel_rejects_indivisible_expert_sizes(model_parallel):
    with pytest.raises(RuntimeError, match='not divisible by expert_tensor_model_pipeline'):
        model_parallel(1, 1, expert_model_parallel_size=3)


def test_initialize_model_parallel_rejects_indivisible_engram_embedding_size(model_parallel):
    with pytest.raises(RuntimeError, match='engram_embedding_parallel_size'):
        model_parallel(1, 1, engram_embedding_parallel_size=3)


def test_initialize_model_parallel_clears_stale_collnet_environment(model_parallel, monkeypatch):
    monkeypatch.setenv('NCCL_COLLNET_ENABLE', '1')
    model_parallel(2, 2)
    assert 'NCCL_COLLNET_ENABLE' not in os.environ


def test_initialize_model_parallel_reads_nccl_communicator_config(model_parallel, tmp_path):
    config_file = tmp_path / 'nccl_communicator_config.yaml'
    config_file.write_text(
        'dp_cp:\n' '  max_ctas: 16\n' '  min_ctas: 2\n' 'tp:\n' '  net_name: socket\n'
    )
    model_parallel(
        2, 2, nccl_communicator_config_path=str(config_file), high_priority_stream_groups=['dp_cp']
    )
    assert ps.get_data_parallel_world_size(with_context_parallel=True) == 2
    assert ps.get_tensor_model_parallel_group().size() == 2

    rejected_file = tmp_path / 'nccl_communicator_config_rejected.yaml'
    rejected_file.write_text('dp_cp:\n' '  net_name: tcp\n')
    with pytest.raises(RuntimeError, match=r'net_name \(tcp\) is not supported'):
        model_parallel(2, 2, nccl_communicator_config_path=str(rejected_file))


def test_initialize_model_parallel_hierarchical_context_parallel_builds_levels(model_parallel):
    model_parallel(2, 1, context_parallel_size=4, hierarchical_context_parallel_sizes=[2, 2])
    rank = torch.distributed.get_rank()
    cp_members = [(rank % 2) + 2 * k for k in range(4)]
    cp_rank = (rank // 2) % 4

    assert ps.get_context_parallel_world_size() == 4
    assert ps.get_context_parallel_group().size() == 4
    assert ps.get_context_parallel_rank() == cp_rank
    assert ps.get_context_parallel_global_ranks() == cp_members

    groups = ps.get_hierarchical_context_parallel_groups()
    assert len(groups) == 2
    expected_level0 = cp_members[:2] if cp_rank < 2 else cp_members[2:]
    expected_level1 = cp_members[::2] if cp_rank % 2 == 0 else cp_members[1::2]
    assert torch.distributed.get_process_group_ranks(groups[0]) == expected_level0
    assert torch.distributed.get_process_group_ranks(groups[1]) == expected_level1


def test_initialize_model_parallel_hybrid_context_parallel_builds_power_of_two_groups(
    model_parallel,
):
    model_parallel(2, 1, context_parallel_size=2, hybrid_context_parallel=True)
    assert ps.get_data_parallel_world_size(with_context_parallel=True) == 4
    assert ps.get_hybrid_data_context_parallel_groups(group_size=2).size() == 2
    assert ps.get_hybrid_data_context_parallel_groups(group_size=4).size() == 4


def test_is_rank_in_embedding_group_reports_intermediate_members(model_parallel):
    model_parallel(
        1, 4, virtual_pipeline_model_parallel_size=2, get_embedding_ranks=lambda ranks: list(ranks)
    )
    rank = torch.distributed.get_rank()
    pp_rank = rank // 2

    assert ps.is_rank_in_embedding_group() is True
    assert ps.is_rank_in_embedding_group(ignore_virtual=False, vp_stage=0) == (pp_rank != 3)
    assert ps.is_rank_in_embedding_group(ignore_virtual=False, vp_stage=1) == (pp_rank != 0)


def test_initialize_model_parallel_partial_distributed_optimizer_groups(model_parallel):
    model_parallel(2, 1, num_distributed_optimizer_instances=2)
    rank = torch.distributed.get_rank()
    assert ps.get_data_parallel_world_size(with_context_parallel=True) == 4
    assert ps.get_data_parallel_group(with_context_parallel=True).size() == 4
    assert (
        ps.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True).size()
        == 2
    )
    assert ps.get_data_parallel_group_gloo(with_context_parallel=True).size() == 4
    assert ps.get_data_parallel_group_gloo().size() == 4
    assert torch.distributed.get_process_group_ranks(ps.get_data_parallel_group_gloo()) == [
        (rank % 2) + 2 * k for k in range(4)
    ]
    assert (
        ps.get_data_parallel_group_gloo(
            with_context_parallel=True, partial_data_parallel=True
        ).size()
        == 2
    )


def test_initialize_model_parallel_expert_partial_optimizer_groups(model_parallel):
    model_parallel(1, 2, expert_model_parallel_size=2, num_distributed_optimizer_instances=2)
    rank = torch.distributed.get_rank()
    ep_rank = rank % 2

    assert ps.get_expert_model_parallel_world_size() == 2
    assert ps.get_expert_model_parallel_rank() == ep_rank
    assert ps.get_expert_model_parallel_src_rank() == rank - ep_rank
    assert ps.get_expert_tensor_parallel_world_size() == 1
    assert ps.get_expert_tensor_parallel_rank() == 0
    assert ps.get_expert_data_parallel_world_size() == 2
    assert ps.get_expert_tensor_and_model_parallel_world_size() == 2
    assert ps.get_expert_tensor_and_model_parallel_rank() == ep_rank
    assert ps.get_expert_data_parallel_group().size() == 2
    assert ps.get_expert_data_parallel_group_gloo().size() == 2
    assert ps.get_expert_data_parallel_group(partial_expert_data_parallel=True).size() == 1
    assert ps.get_expert_data_parallel_group_gloo(partial_expert_data_parallel=True).size() == 1
    assert ps.get_inter_distributed_optimizer_instance_group().size() == 2
    assert ps.get_intra_distributed_optimizer_instance_group().size() == 4
    expert_tensor_and_model = ps.get_expert_tensor_and_model_parallel_group()
    assert expert_tensor_and_model.size() == 2
    assert torch.distributed.get_process_group_ranks(expert_tensor_and_model) == [
        rank - ep_rank,
        rank - ep_rank + 1,
    ]
    assert ps.get_expert_tensor_model_pipeline_parallel_group().size() == 4


def test_initialize_model_parallel_engram_groups(model_parallel):
    model_parallel(1, 2, engram_embedding_parallel_size=2)
    rank = torch.distributed.get_rank()
    e_tp_rank = rank % 2
    e_dp_rank = (rank // 2) % 2

    assert ps.get_engram_embedding_parallel_world_size() == 2
    assert ps.get_engram_embedding_parallel_rank() == e_tp_rank
    assert ps.get_engram_data_parallel_world_size() == 2
    assert ps.get_engram_data_parallel_group().size() == 2
    assert ps.get_engram_data_parallel_group_gloo().size() == 2
    assert ps.get_engram_embedding_parallel_group_gloo().size() == 2
    assert torch.distributed.get_process_group_ranks(ps.get_engram_data_parallel_group()) == [
        rank - 2 * e_dp_rank,
        rank - 2 * e_dp_rank + 2,
    ]
    assert torch.distributed.get_process_group_ranks(ps.get_engram_model_parallel_group()) == (
        [0, 1, 4, 5] if e_dp_rank == 0 else [2, 3, 6, 7]
    )


def test_initialize_model_parallel_use_sharp_dp_group(model_parallel):
    model_parallel(2, 2, use_sharp=True)
    assert ps.get_data_parallel_world_size() == 2
    assert ps.get_data_parallel_group(with_context_parallel=True).size() == 2


def test_initialize_model_parallel_use_sharp_dp_replica_group(model_parallel):
    model_parallel(
        1,
        2,
        expert_model_parallel_size=2,
        num_distributed_optimizer_instances=2,
        use_sharp=True,
        sharp_enabled_group='dp_replica',
    )
    assert ps.get_expert_data_parallel_world_size() == 2
    assert ps.get_inter_distributed_optimizer_instance_group().size() == 2


def test_update_pg_timeout_updates_all_and_single_process_groups(model_parallel):
    model_parallel(2, 2)
    ps.update_pg_timeout(timedelta(minutes=10))
    ps.update_pg_timeout(timedelta(minutes=11), pg=ps.get_data_parallel_group())
    assert ps.is_initialized() is True
    assert ps.get_data_parallel_group().size() == 2


def test_create_all_gather_groups_requires_initialized_parallel_state():
    with pytest.raises(RuntimeError, match='requires parallel state to be initialized'):
        ps.create_all_gather_groups()


def test_create_all_gather_groups_returns_matching_dp_groups(model_parallel):
    model_parallel(2, 2)
    rank = torch.distributed.get_rank()
    dp_cp_ag_group, expt_dp_ag_group = ps.create_all_gather_groups()

    assert expt_dp_ag_group is None
    assert dp_cp_ag_group.size() == 2
    base = (rank % 2) + 4 * (rank // 4)
    assert torch.distributed.get_process_group_ranks(dp_cp_ag_group) == [base, base + 2]


def test_create_all_gather_groups_with_expert_parallelism(model_parallel):
    model_parallel(1, 2, expert_model_parallel_size=2)
    rank = torch.distributed.get_rank()
    dp_cp_ag_group, expt_dp_ag_group = ps.create_all_gather_groups(for_expert_parallelism=True)

    assert dp_cp_ag_group.size() == 4
    assert torch.distributed.get_process_group_ranks(dp_cp_ag_group) == [
        4 * (rank // 4) + k for k in range(4)
    ]
    assert expt_dp_ag_group.size() == 2
    base = 4 * (rank // 4) + (rank % 2)
    assert torch.distributed.get_process_group_ranks(expt_dp_ag_group) == [base, base + 2]


def test_initialize_model_parallel_destroys_and_reinitializes(model_parallel):
    model_parallel(2, 2)
    assert ps.is_initialized() is True
    assert ps.model_parallel_is_initialized() is True

    ps.destroy_model_parallel()
    assert ps.is_initialized() is False
    assert ps.model_parallel_is_initialized() is False
    # is_rank_in_embedding_group() is not asserted here: destroy_model_parallel() does not reset
    # _EMBEDDING_GLOBAL_RANKS (reported as a source bug), so it still answers for the old layout.
    assert ps.is_rank_in_position_embedding_group() is False

    model_parallel(1, 1)
    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_pipeline_model_parallel_world_size() == 1
    assert ps.get_data_parallel_world_size() == 8
