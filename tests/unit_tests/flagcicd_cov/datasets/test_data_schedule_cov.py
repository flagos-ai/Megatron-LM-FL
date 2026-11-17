# Copyright (c) 2026, BAAI. All rights reserved.
"""Hybrid context-parallel data loading: group selection, packed-batch unpacking, HDP rerouting."""

from types import SimpleNamespace

import torch

from megatron.core import parallel_state
from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import HybridCPDataLoaderWrapper
from megatron.core.process_groups_config import ProcessGroupCollection


class _Group:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _install_cpu_device(monkeypatch):
    monkeypatch.setattr(
        data_schedule, "cur_platform", SimpleNamespace(current_device=lambda: torch.device("cpu"))
    )


def _wrapper(monkeypatch, data_iterator=None):
    _install_cpu_device(monkeypatch)
    config = SimpleNamespace(max_seqlen_per_dp_cp_rank=16)
    return HybridCPDataLoaderWrapper(
        data_iterator,
        config,
        pg_collection=ProcessGroupCollection(tp=_Group(1, 0), dp=_Group(2, 0), dp_cp=_Group(2, 0)),
    )


def _loopback_all_to_all(
    output, input, output_split_sizes=None, input_split_sizes=None, group=None
):
    own = output_split_sizes[0]
    output[:own] = input[:own]
    if len(output) > own:
        output[own:] = 0


def test_hybrid_cp_dataloader_takes_groups_from_the_process_group_collection(monkeypatch):
    wrapper = _wrapper(monkeypatch)

    assert iter(wrapper) is wrapper
    assert wrapper.total_hdp_gpus == 2
    assert wrapper.dp_cp_group.size() == 2
    assert wrapper.dp_group.rank() == 0
    assert wrapper.tp_group.size() == 1
    assert wrapper.cp_balancing_scheduler.max_seq_len_per_rank == 16
    assert wrapper.cp_balancing_scheduler.total_hdp_gpus == 2


def test_hybrid_cp_dataloader_falls_back_to_parallel_state_groups(monkeypatch):
    _install_cpu_device(monkeypatch)
    dp_cp = _Group(2, 0)
    dp = _Group(2, 0)
    tp = _Group(1, 0)
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel=False: dp_cp if with_context_parallel else dp,
    )
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: tp)
    config = SimpleNamespace(max_seqlen_per_dp_cp_rank=16)

    wrapper = HybridCPDataLoaderWrapper(None, config, pg_collection=None)

    assert wrapper.dp_cp_group is dp_cp
    assert wrapper.dp_group is dp
    assert wrapper.tp_group is tp
    assert wrapper.total_hdp_gpus == 2


def test_hybrid_cp_dataloader_unpack_batch_splits_packed_samples_at_the_boundaries(monkeypatch):
    wrapper = _wrapper(monkeypatch)
    sample = {
        "cu_seqlens": torch.tensor([0, 2, 2, 5], dtype=torch.int32),
        "tokens": torch.tensor([10, 20, 30, 40, 50]),
        "positions": torch.tensor([1, 2, 3, 4, 5]),
        "batch_idx": 1,
        "max_seqlen": 5,
    }

    unpacked = wrapper.unpack_batch([sample])

    assert len(unpacked) == 2
    torch.testing.assert_close(unpacked[0]["tokens"], torch.tensor([10, 20]))
    torch.testing.assert_close(unpacked[0]["positions"], torch.tensor([1, 2]))
    torch.testing.assert_close(unpacked[1]["tokens"], torch.tensor([30, 40, 50]))
    torch.testing.assert_close(unpacked[1]["positions"], torch.tensor([3, 4, 5]))
    assert "cu_seqlens" not in unpacked[0]
    assert "batch_idx" not in unpacked[0]
    assert "max_seqlen" not in unpacked[0]


def test_hybrid_cp_dataloader_maps_global_ids_to_their_owning_rank(monkeypatch):
    wrapper = _wrapper(monkeypatch)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])

    assert wrapper._gid_to_src_rank(2, torch.tensor([0, 3, 5], dtype=torch.int32)) == 0
    assert wrapper._gid_to_src_rank(4, torch.tensor([0, 3, 5], dtype=torch.int32)) == 1


def test_hybrid_cp_dataloader_reroutes_scheduled_samples_to_hdp_ranks(monkeypatch):
    wrapper = _wrapper(monkeypatch)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(torch.distributed, "all_to_all_single", _loopback_all_to_all)
    batch = [
        {"tokens": torch.tensor([10, 20, 30, 40])},
        {"tokens": torch.tensor([50, 60, 70, 80, 90, 100])},
    ]

    received = wrapper.reroute_samples_to_hdp_ranks(
        batch,
        torch.tensor([0, 1], dtype=torch.int32),
        [(0, 4), (1, 6)],
        [[[0], [1]]],
        torch.tensor([0, 2], dtype=torch.int32),
    )

    assert list(received) == [0]
    torch.testing.assert_close(received[0]["tokens"], torch.tensor([10, 20, 30, 40]))


def test_hybrid_cp_dataloader_sends_nothing_to_ranks_outside_the_dp_group(monkeypatch):
    _install_cpu_device(monkeypatch)
    config = SimpleNamespace(max_seqlen_per_dp_cp_rank=16)
    wrapper = HybridCPDataLoaderWrapper(
        None,
        config,
        pg_collection=ProcessGroupCollection(tp=_Group(1, 0), dp=_Group(2, 0), dp_cp=_Group(4, 0)),
    )
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    sent_splits = []

    def recording_all_to_all(
        output, input, output_split_sizes=None, input_split_sizes=None, group=None
    ):
        sent_splits.append((list(input_split_sizes), list(output_split_sizes)))
        own = output_split_sizes[0]
        output[:own] = input[:own]
        if len(output) > own:
            output[own:] = 0

    monkeypatch.setattr(torch.distributed, "all_to_all_single", recording_all_to_all)
    batch = [
        {"tokens": torch.tensor([10, 20, 30, 40])},
        {"tokens": torch.tensor([50, 60, 70, 80, 90, 100])},
    ]

    received = wrapper.reroute_samples_to_hdp_ranks(
        batch,
        torch.tensor([0, 1], dtype=torch.int32),
        [(0, 4), (1, 6)],
        [[[0, 1], [], [], []]],
        torch.tensor([0, 2], dtype=torch.int32),
    )

    assert list(received) == [0, 1]
    torch.testing.assert_close(received[0]["tokens"], torch.tensor([10, 20, 30, 40]))
    torch.testing.assert_close(received[1]["tokens"], torch.tensor([50, 60, 70, 80, 90, 100]))
    assert sent_splits == [([10, 0, 0, 0], [10, 0, 0, 0])]


def test_hybrid_cp_dataloader_next_returns_none_without_a_data_iterator(monkeypatch):
    wrapper = _wrapper(monkeypatch)

    assert next(wrapper) == (None, None)
