# Copyright (c) 2026, BAAI. All rights reserved.
"""Balanced CP scheduler bucket balancing and HDP group planning; hybrid context-parallel forward/backward runs each rank's assigned subsamples against a single-model reference."""

import contextlib

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.hybrid_cp_schedule import (
    BalancedCPScheduler,
    hybrid_context_parallel_forward_backward,
)
from megatron.core.pipeline_parallel.schedules import (
    check_first_val_step,
    forward_backward_no_pipelining,
    get_forward_backward_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_MAX_SEQ_LEN_PER_CP_RANK = 512
_HIDDEN = 16
_SAMPLE_SEQLENS = [
    (0, 4096),
    (1, 1024),
    (2, 1024),
    (3, 2048),
    (4, 512),
    (5, 512),
    (6, 512),
    (7, 512),
    (8, 512),
    (9, 512),
]


@pytest.fixture
def cp2_parallel():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    yield
    Utils.destroy_model_parallel()


def _make_scheduler(max_seq_len_per_rank=_MAX_SEQ_LEN_PER_CP_RANK):
    return BalancedCPScheduler(
        max_seq_len_per_rank=max_seq_len_per_rank,
        dp_cp_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )


@pytest.mark.parametrize(
    "seq_len,expected",
    [(256, 1), (512, 1), (513, 2), (1024, 2), (2048, 4), (3000, 8), (4096, 8)],
    ids=[
        "below-max",
        "at-max",
        "just-above-max",
        "2x-max",
        "4x-max",
        "between-4x-and-8x",
        "8x-max",
    ],
)
def test_gpus_needed_rounds_sequence_length_up_to_next_power_of_two(
    cp2_parallel, seq_len, expected
):
    scheduler = _make_scheduler()
    assert scheduler.gpus_needed(seq_len) == expected


def test_get_total_workload_divides_squared_length_by_cp_size(cp2_parallel):
    scheduler = _make_scheduler()
    assert scheduler.get_total_workload(2048) == 2048 * 2048 / 4
    assert scheduler.get_total_workload(2048, 8) == 2048 * 2048 / 8


def test_next_hdp_group_returns_empty_plan_when_no_samples(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [], scheduler.get_total_workload, 8
    )
    assert micro_batches == [[] for _ in range(8)]
    assert leftovers == []
    assert exec_times == [0.0 for _ in range(8)]
    assert sample_ids == [[] for _ in range(8)]


def test_next_hdp_group_returns_unplaceable_sample_as_leftover(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 2048), (1, 2048), (2, 512)], scheduler.get_total_workload, 8
    )
    assert micro_batches == [[2048] for _ in range(8)]
    assert leftovers == [(2, 512)]
    assert exec_times == [1048576.0 for _ in range(8)]
    assert sample_ids == [[0], [0], [0], [0], [1], [1], [1], [1]]


def test_next_hdp_group_trims_last_sample_when_trimming_reduces_slack(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 2048), (1, 2048), (2, 2048), (3, 1024)], scheduler.get_total_workload, 8
    )
    assert micro_batches == [[2048] for _ in range(8)]
    assert leftovers == [(3, 1024), (2, 2048)]
    assert exec_times == [1048576.0 for _ in range(8)]
    assert sample_ids == [[0], [0], [0], [0], [1], [1], [1], [1]]


def test_next_hdp_group_stops_scheduling_once_group_is_balanced(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 2048), (1, 1024), (2, 1024), (3, 1024), (4, 1024), (5, 1024)],
        scheduler.get_total_workload,
        8,
    )
    assert micro_batches == [[2048]] * 4 + [[1024, 1024]] * 4
    assert leftovers == [(5, 1024)]
    assert exec_times == [1048576.0 for _ in range(8)]
    assert sample_ids == [[0], [0], [0], [0], [1, 3], [1, 3], [2, 4], [2, 4]]


def test_next_hdp_group_expands_cp_size_to_fill_empty_gpus(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 1024), (1, 1024), (2, 1024)], scheduler.get_total_workload, 8
    )
    assert micro_batches == [[1024] for _ in range(8)]
    assert leftovers == []
    assert exec_times == [262144.0] * 4 + [524288.0] * 4
    assert sample_ids == [[0], [0], [0], [0], [1], [1], [2], [2]]


def test_next_hdp_group_skips_empty_bucket_when_scanning_for_placeable_sample(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 2048), (1, 4096), (2, 512)], scheduler.get_total_workload, 8
    )
    assert micro_batches == [[2048]] * 4 + [[512]] * 4
    assert leftovers == [(1, 4096)]
    assert exec_times == [1048576.0] * 4 + [65536.0] * 4
    assert sample_ids == [[0]] * 4 + [[2]] * 4


def test_next_hdp_group_keeps_overloaded_sample_when_trimming_would_not_reduce_slack(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 512), (1, 512), (2, 512), (3, 512), (4, 1024), (5, 1024), (6, 1024), (7, 1024)],
        scheduler.get_total_workload,
        8,
    )
    assert micro_batches == [[512]] * 4 + [[1024, 1024]] * 4
    assert leftovers == []
    assert exec_times == [262144.0] * 4 + [1048576.0] * 4
    assert sample_ids == [[0], [1], [2], [3], [4, 6], [4, 6], [5, 7], [5, 7]]


def test_next_hdp_group_keeps_single_sample_when_most_loaded_gpu_holds_one_sample(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 2048), (1, 512), (2, 512), (3, 512), (4, 512), (5, 512)],
        scheduler.get_total_workload,
        8,
    )
    assert micro_batches == [[2048]] * 4 + [[512, 512]] + [[512]] * 3
    assert leftovers == []
    assert exec_times == [1048576.0] * 4 + [524288.0] + [262144.0] * 3
    assert sample_ids == [[0]] * 4 + [[1, 5]] + [[2], [3], [4]]


def test_next_hdp_group_with_pp_strategy_rotates_bucket_scan(cp2_parallel):
    scheduler = _make_scheduler()
    micro_batches, leftovers, exec_times, sample_ids = scheduler.next_hdp_group(
        [(0, 2048), (1, 2048), (2, 512)], scheduler.get_total_workload, 8, strategy="pp"
    )
    assert micro_batches == [[2048, 2048]] * 4 + [[512]] * 4
    assert leftovers == []
    assert exec_times == [2097152.0] * 4 + [65536.0] * 4
    assert sample_ids == [[0, 1]] * 4 + [[2]] * 4


def test_get_groups_and_subsamples_partitions_every_sample_across_groups(cp2_parallel):
    scheduler = _make_scheduler()
    groups, sample_id_groups = scheduler.get_groups_and_subsamples(_SAMPLE_SEQLENS, None)
    assert len(groups) == 3
    assert sample_id_groups[0] == [[0] for _ in range(8)]
    assert sample_id_groups[1] == [[3], [3], [3], [3], [1], [1], [2], [2]]
    assert sample_id_groups[2] == [[4], [4], [5], [5], [6], [7], [8], [9]]
    assigned = sorted(
        sample_id for group_ids in sample_id_groups for ids in group_ids for sample_id in ids
    )
    assert assigned == [0] * 8 + [1] * 2 + [2] * 2 + [3] * 4 + [4, 4] + [5, 5] + [6, 7, 8, 9]


class _DenseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(_HIDDEN, _HIDDEN)
        self.config = config
        self.model_type = "unit-test"

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, hidden_states):
        return self.linear(hidden_states)


def _hidden_for_sample(sample_id):
    generator = torch.Generator().manual_seed(1000 + sample_id)
    return torch.randn(2, _HIDDEN, generator=generator).cuda()


def _hybrid_forward_step(data_iterator, model):
    sample = next(data_iterator)
    output = model(sample["hidden"])
    num_tokens = torch.tensor(sample["seqlen"], dtype=torch.int64)

    def loss_func(output_tensor):
        return (
            output_tensor.sum(),
            num_tokens,
            {
                "sample_id": sample["sample_id"],
                "local_cp_size": int(sample["local_cp_size"]),
                "loss": output_tensor.sum().detach(),
            },
        )

    return output, loss_func


def _hybrid_e2e_setup(sample_seqlens):
    scheduler = _make_scheduler()
    _, sample_id_groups = scheduler.get_groups_and_subsamples(sample_seqlens, None)
    hidden = {sample_id: _hidden_for_sample(sample_id) for sample_id, _ in sample_seqlens}
    batch = [
        {"hidden": hidden[sample_id], "seqlen": seqlen, "sample_id": sample_id}
        for sample_id, seqlen in sample_seqlens
    ]
    config = TransformerConfig(
        num_layers=2,
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hybrid_context_parallel=True,
        calculate_per_token_loss=True,
    )
    torch.manual_seed(2024)
    model = _DenseModel(config).cuda()
    return sample_id_groups, hidden, batch, config, model


def test_hybrid_context_parallel_forward_backward_runs_subsamples_assigned_to_rank(cp2_parallel):
    sample_id_groups, hidden, batch, config, model = _hybrid_e2e_setup(_SAMPLE_SEQLENS)
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    expected_ids = [sid for group_ids in sample_id_groups for sid in group_ids[hdp_rank]]

    forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
        forward_step_func=_hybrid_forward_step,
        data_iterator=iter([(batch, sample_id_groups)]),
        model=model,
        num_microbatches=1,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=[],
        config=config,
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=True,
        no_sync_func=contextlib.nullcontext,
        total_num_tokens=0,
        check_first_val_step=check_first_val_step,
        model_type="unit-test",
    )

    assert [entry["sample_id"] for entry in forward_data_store] == expected_ids
    assert [entry["local_cp_size"] for entry in forward_data_store] == [
        8,
        4 if hdp_rank < 4 else 2,
        2 if hdp_rank < 4 else 1,
    ]
    with torch.no_grad():
        for entry in forward_data_store:
            reference = model(hidden[entry["sample_id"]]).sum()
            torch.testing.assert_close(entry["loss"], reference, rtol=1e-6, atol=1e-6)
    assert total_num_tokens == 4096 + (2048 if hdp_rank < 4 else 1024) + 512
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hybrid_context_parallel_forward_backward_runs_two_subsamples_per_rank_in_last_group(
    cp2_parallel,
):
    sample_seqlens = [(0, 4096)] + [(sample_id, 512) for sample_id in range(1, 17)]
    sample_id_groups, hidden, batch, config, model = _hybrid_e2e_setup(sample_seqlens)
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    expected_ids = [sid for group_ids in sample_id_groups for sid in group_ids[hdp_rank]]
    assert expected_ids == [0, hdp_rank + 1, hdp_rank + 9]

    forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
        forward_step_func=_hybrid_forward_step,
        data_iterator=iter([(batch, sample_id_groups)]),
        model=model,
        num_microbatches=1,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=[],
        config=config,
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=True,
        no_sync_func=contextlib.nullcontext,
        total_num_tokens=0,
        check_first_val_step=check_first_val_step,
        model_type="unit-test",
    )

    assert [entry["sample_id"] for entry in forward_data_store] == expected_ids
    assert [entry["local_cp_size"] for entry in forward_data_store] == [8, 1, 1]
    with torch.no_grad():
        for entry in forward_data_store:
            reference = model(hidden[entry["sample_id"]]).sum()
            torch.testing.assert_close(entry["loss"], reference, rtol=1e-6, atol=1e-6)
    assert total_num_tokens == 4096 + 512 + 512
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hybrid_context_parallel_forward_backward_accumulates_gradients_when_not_forward_only(
    cp2_parallel,
):
    sample_seqlens = [(0, 4096)] + [(sample_id, 512) for sample_id in range(1, 17)]
    sample_id_groups, hidden, batch, config, model = _hybrid_e2e_setup(sample_seqlens)
    torch.manual_seed(2024)
    reference_model = _DenseModel(config).cuda()
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    expected_ids = [sid for group_ids in sample_id_groups for sid in group_ids[hdp_rank]]

    forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
        forward_step_func=_hybrid_forward_step,
        data_iterator=iter([(batch, sample_id_groups)]),
        model=model,
        num_microbatches=1,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=[],
        config=config,
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=False,
        no_sync_func=contextlib.nullcontext,
        total_num_tokens=0,
        check_first_val_step=check_first_val_step,
        model_type="unit-test",
    )

    assert [entry["sample_id"] for entry in forward_data_store] == expected_ids
    assert total_num_tokens == 4096 + 512 + 512
    expected_loss = sum(reference_model(hidden[sid]).sum() for sid in expected_ids)
    expected_loss.backward()
    for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6)


def test_forward_backward_func_dispatches_hybrid_cp_schedule_for_hybrid_config(cp2_parallel):
    sample_id_groups, hidden, batch, config, model = _hybrid_e2e_setup(_SAMPLE_SEQLENS)
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    expected_ids = [sid for group_ids in sample_id_groups for sid in group_ids[hdp_rank]]

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func is forward_backward_no_pipelining

    losses = forward_backward_func(
        forward_step_func=_hybrid_forward_step,
        data_iterator=iter([(batch, sample_id_groups)]),
        model=[model],
        num_microbatches=1,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )

    assert [entry["sample_id"] for entry in losses] == expected_ids
    with torch.no_grad():
        for entry in losses:
            reference = model(hidden[entry["sample_id"]]).sum()
            torch.testing.assert_close(entry["loss"], reference, rtol=1e-6, atol=1e-6)
