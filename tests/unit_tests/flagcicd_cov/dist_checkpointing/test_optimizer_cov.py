# Copyright (c) 2026, BAAI. All rights reserved.

"""Optimizer state sharding: param-id to ShardedTensor mapping, optimizer state wrapping with key exclusion, and errors for unknown param ids."""

import logging

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    optim_state_to_sharding_state,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_get_param_id_to_sharded_param_map_warns_when_empty(model_parallel, caplog):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    sh_ten = ShardedTensor.from_rank_offsets('model_param', torch.ones(2, 3), replica_id=0)
    untracked_param = torch.nn.Parameter(torch.ones(4, 4))

    with caplog.at_level(logging.WARNING):
        param_map = get_param_id_to_sharded_param_map({'param': sh_ten}, [untracked_param])

    assert param_map == {}
    if torch.distributed.get_rank() == 0:
        assert 'Sharded parameters mapping is empty' in caplog.text


def test_get_param_id_to_sharded_param_map_matches_optimizer_parameters(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    param = torch.nn.Parameter(torch.arange(6, dtype=torch.float).view(2, 3))
    # Model state dicts built with keep_vars=True expose the Parameter itself as data.
    sh_ten = ShardedTensor.from_rank_offsets('model_param', param, replica_id=0)
    other = torch.nn.Parameter(torch.ones(2, 3))

    param_map = get_param_id_to_sharded_param_map({'param': sh_ten}, [other, param])

    assert set(param_map.keys()) == {1}
    assert param_map[1] is sh_ten


def test_optim_state_to_sharding_state_excludes_requested_keys(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    sh_ten = ShardedTensor.from_rank_offsets('model_param', torch.ones(2, 3), replica_id=0)
    optim_state_dict = {
        'state': {0: {'exp_avg': torch.ones(2, 3), 'step': 10}},
        'param_groups': [{'lr': 0.1, 'params': [0]}],
    }

    optim_state_to_sharding_state(optim_state_dict, {0: sh_ten}, exclude_keys=('step',))

    assert set(optim_state_dict['state'][0].keys()) == {'exp_avg'}
    exp_avg = optim_state_dict['state'][0]['exp_avg']
    assert isinstance(exp_avg, ShardedTensor)
    assert exp_avg.key == 'optimizer.state.exp_avg.model_param'
    assert torch.equal(exp_avg.data, torch.ones(2, 3))
    wrapped_params = optim_state_dict['param_groups'][0]['params']
    assert isinstance(wrapped_params, LocalNonpersistentObject)


def test_optim_state_to_sharding_state_raises_for_unknown_param_id(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    sh_ten = ShardedTensor.from_rank_offsets('model_param', torch.ones(2, 3), replica_id=0)
    optim_state_dict = {
        'state': {5: {'exp_avg': torch.ones(2, 3)}},
        'param_groups': [{'params': [5]}],
    }

    with pytest.raises(ValueError, match='Param id 5 does not match any model sharded param'):
        optim_state_to_sharding_state(optim_state_dict, {0: sh_ten})
