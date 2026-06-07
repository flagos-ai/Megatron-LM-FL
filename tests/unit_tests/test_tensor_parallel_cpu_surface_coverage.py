# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.tensor_parallel.mappings as mappings
import megatron.core.tensor_parallel.random as rng


class _Group:
    def __init__(self, size=2, rank=1):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _patch_cpu_collectives(monkeypatch):
    monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")

    def _all_gather_into_tensor(output, input_, group=None):
        chunks = output.chunk(group.size(), dim=0)
        for idx, chunk in enumerate(chunks):
            chunk.copy_(input_ + idx)

    def _reduce_scatter_tensor(output, input_, group=None):
        output.copy_(input_.chunk(group.size(), dim=0)[group.rank()])

    def _all_gather(output_list, input_, group=None):
        for idx, out in enumerate(output_list):
            out.copy_(input_ + idx)

    def _reduce_scatter(output, input_list, group=None):
        output.copy_(input_list[group.rank()])

    def _all_reduce(input_, group=None):
        input_.add_(group.rank() + 1)

    def _all_to_all_single(
        output, input_, output_split_sizes=None, input_split_sizes=None, group=None
    ):
        if output_split_sizes is None:
            output.copy_(input_.flip(0))
            return
        parts = torch.split(input_, input_split_sizes, dim=0)
        output.copy_(torch.cat([part + idx for idx, part in enumerate(parts)], dim=0))

    monkeypatch.setattr(mappings, "dist_all_gather_func", _all_gather_into_tensor)
    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", _reduce_scatter_tensor)
    monkeypatch.setattr(mappings.torch.distributed, "all_gather", _all_gather)
    monkeypatch.setattr(mappings.torch.distributed, "reduce_scatter", _reduce_scatter)
    monkeypatch.setattr(mappings.torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setattr(mappings.torch.distributed, "all_to_all_single", _all_to_all_single)


def test_tensor_parallel_mapping_collective_helpers_cpu_paths(monkeypatch):
    _patch_cpu_collectives(monkeypatch)
    group = _Group(size=2, rank=1)
    single = _Group(size=1, rank=0)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    assert mappings._reduce(x.clone(), single).shape == x.shape
    reduced = mappings._reduce(x.clone(), group)
    assert torch.equal(reduced, x + 2)

    assert torch.equal(mappings._split_along_last_dim(x, single), x)
    assert torch.equal(mappings._split_along_last_dim(x, group), x[:, 2:])
    assert torch.equal(mappings._split_along_first_dim(x[:2], group), x[1:2])
    with pytest.raises(AssertionError, match="First dimension"):
        mappings._split_along_first_dim(x, group)

    gathered_last = mappings._gather_along_last_dim(x[:, :2], group)
    assert gathered_last.shape == x.shape
    gathered_first = mappings._gather_along_first_dim(x[:1], group)
    assert gathered_first.shape == (2, 4)
    gathered_split = mappings._gather_along_first_dim(x[:1], group, output_split_sizes=[1, 1])
    assert gathered_split.shape == (2, 4)

    scattered_first = mappings._reduce_scatter_along_first_dim(x[:2], group)
    assert torch.equal(scattered_first, x[1:2])
    scattered_split = mappings._reduce_scatter_along_first_dim(
        x[:2], group, input_split_sizes=[1, 1]
    )
    assert torch.equal(scattered_split, x[1:2])
    scattered_last = mappings._reduce_scatter_along_last_dim(x, group)
    assert scattered_last.shape == (3, 2)


def test_tensor_parallel_mapping_autograd_wrappers_cpu_paths(monkeypatch):
    _patch_cpu_collectives(monkeypatch)
    group = _Group(size=2, rank=1)
    monkeypatch.setattr(mappings, "get_tensor_model_parallel_group_if_none", lambda group=None: group)

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_(True)
    copied = mappings.copy_to_tensor_model_parallel_region(x, group)
    assert copied is not x or torch.equal(copied, x)
    reduced_expected = x.detach().clone() + 2
    reduced = mappings.reduce_from_tensor_model_parallel_region(x, group)
    assert torch.allclose(reduced.detach(), reduced_expected)
    scattered = mappings.scatter_to_tensor_model_parallel_region(x, group)
    assert scattered.shape == (2, 2)
    gathered = mappings.gather_from_tensor_model_parallel_region(scattered, group)
    assert gathered.shape == x.shape

    seq_scattered = mappings.scatter_to_sequence_parallel_region(x, group)
    assert seq_scattered.shape == (1, 4)
    seq_gathered = mappings.gather_from_sequence_parallel_region(seq_scattered, group=group)
    assert seq_gathered.shape == x.shape
    seq_gathered_dup = mappings.gather_from_sequence_parallel_region(
        seq_scattered, tensor_parallel_output_grad=False, group=group
    )
    assert seq_gathered_dup.shape == x.shape

    rs_seq = mappings.reduce_scatter_to_sequence_parallel_region(x, group=group)
    assert rs_seq.shape == (1, 4)
    ag_last = mappings.all_gather_last_dim_from_tensor_parallel_region(scattered, group=group)
    assert ag_last.shape == x.shape
    rs_last = mappings.reduce_scatter_last_dim_to_tensor_parallel_region(x, group=group)
    assert rs_last.shape == (2, 2)

    exchanged = mappings.all_to_all(group, x.detach())
    assert exchanged.shape == x.shape
    uneven = mappings.all_to_all(
        group,
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
        output_split_sizes_=[1, 2],
        input_split_sizes=[1, 2],
    )
    assert uneven.shape == (3, 2)

    sp2hp = mappings.all_to_all_sp2hp(x.detach(), group=group)
    hp2sp = mappings.all_to_all_hp2sp(x.detach(), group=group)
    assert sp2hp.shape == (4, 2)
    assert hp2sp.shape == (1, 8)
    assert sp2hp.numel() == hp2sp.numel() == x.numel()


def test_tensor_parallel_rng_tracker_cpu_state_paths(monkeypatch):
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER", None)
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", False)

    states = {}
    current = {"state": torch.tensor([0], dtype=torch.uint8)}
    seeds = []

    monkeypatch.setattr(rng, "_set_cuda_rng_state", lambda state, **kwargs: current.update(state=state.clone()))
    monkeypatch.setattr(
        rng,
        "_get_cuda_rng_state",
        lambda **kwargs: current["state"].clone(),
    )
    monkeypatch.setattr(
        rng.cur_platform,
        "get_rng_state",
        lambda device=None: current["state"].clone(),
    )

    def _manual_seed(seed):
        seeds.append(seed)
        current["state"] = torch.tensor([seed % 256], dtype=torch.uint8)

    monkeypatch.setattr(rng.cur_platform, "manual_seed", _manual_seed)
    monkeypatch.setattr(rng, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(rng, "get_expert_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(rng, "get_expert_tensor_parallel_rank", lambda: 4)

    tracker = rng.CudaRNGStatesTracker()
    assert tracker.is_initialized() is False
    tracker.add("first", 123)
    assert tracker.is_initialized() is True
    assert "first" in tracker.get_states()
    with pytest.raises(Exception, match="already exists"):
        tracker.add("second", 123)
    with pytest.raises(Exception, match="already exists"):
        tracker.add("first", 124)
    with pytest.raises(Exception, match="not added"):
        with tracker.fork("missing"):
            pass

    with tracker.fork("first"):
        torch.manual_seed(999)
    assert tracker._current_state_name == "default-rng"

    tracker.set_states({"manual": torch.tensor([9], dtype=torch.uint8)})
    assert tracker.get_states()["manual"].item() == 9

    rng.initialize_rng_tracker(force_reset=True)
    assert rng.get_cuda_rng_tracker() is rng._CUDA_RNG_STATE_TRACKER
    assert isinstance(rng.get_all_rng_states(), dict)

    rng.initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
    inference_tracker = rng.get_cuda_rng_tracker(inference_rng_tracker=True)
    inference_tracker.add("ignored", 1)
    inference_tracker.set_states({"ignored": torch.tensor([1], dtype=torch.uint8)})
    with inference_tracker.fork("anything"):
        states["forked"] = True
    assert states["forked"] is True

    seeds.clear()
    rng.model_parallel_cuda_manual_seed(11, force_reset_rng=True)
    assert seeds[:1] == [11]
    assert rng.get_data_parallel_rng_tracker_name() in rng._CUDA_RNG_STATE_TRACKER.states_
    assert rng._MODEL_PARALLEL_RNG_TRACKER_NAME in rng._CUDA_RNG_STATE_TRACKER.states_
    assert rng.get_expert_parallel_rng_tracker_name() in rng._CUDA_RNG_STATE_TRACKER.states_

    assert rng.convert_cuda_rng_state(torch.tensor([1], dtype=torch.uint8), to_graphable=False).item() == 1
    with pytest.raises(ValueError, match="Invalid state type"):
        rng.convert_cuda_rng_state(object(), to_graphable=False)
    with pytest.raises(ValueError, match="Invalid state type"):
        rng.convert_cuda_rng_state(object(), to_graphable=True)

    assert rng.is_checkpointing() is False
    rng._set_checkpointing()
    assert rng.is_checkpointing() is True
    rng._unset_checkpointing()
    assert rng.is_checkpointing() is False
