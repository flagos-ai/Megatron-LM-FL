# Copyright (c) 2026, BAAI. All rights reserved.

"""MCoreTensorAwareStateDict: algorithm validation, cached and empty shard distributions, the nvrx import guard, and copying tensors to CPU."""

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, tensor_aware_state_dict
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict


def test_validate_params_rejects_unknown_algorithm():
    with pytest.raises(NotImplementedError, match='Only "atomic" and "fully_parallel"'):
        MCoreTensorAwareStateDict._validate_params('atomic_with_side_effects')


def test_get_distribution_uses_cached_distribution_without_recomputation():
    cached = ('main_rank_for_shard', 'shards', 'metadata', 'all_ranks')

    distribution = MCoreTensorAwareStateDict._get_distribution(
        True, {'k': 'unused'}, None, cached_distribution=cached
    )

    assert distribution == cached


def test_get_distribution_returns_empty_distribution_for_atomic_algorithm():
    distribution = MCoreTensorAwareStateDict._get_distribution(False, {'k': 'unused'}, None)

    assert distribution == (None, None, None, None)


def test_from_state_dict_raises_import_error_without_nvrx(monkeypatch):
    monkeypatch.setattr(tensor_aware_state_dict, 'HAVE_NVRX', False)

    with pytest.raises(ImportError, match='nvidia_resiliency_ext is not installed'):
        MCoreTensorAwareStateDict.from_state_dict({'k': 'unused'})


def test_copy_tensors_to_cpu_clones_cpu_tensor_data():
    sh_ten = ShardedTensor.from_rank_offsets('cpu_ten', torch.ones(2, 3), replica_id=0)
    original_data = sh_ten.data
    tasd = MCoreTensorAwareStateDict(common={}, sharded_state_dict={'k': sh_ten})

    tasd.copy_tensors_to_cpu()

    assert sh_ten.data is not original_data
    assert torch.equal(sh_ten.data, original_data)
    assert sh_ten.data.device.type == 'cpu'
