# Copyright (c) 2026, BAAI. All rights reserved.
"""Checkpoint state-dict helpers: sharded-tensor extraction, checkpoint key prefix rewriting, scoped debug logging, and serialization-safe metadata cleanup."""

import logging

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.utils import (
    _clean_metadata_for_serialization,
    apply_prefix_mapping,
    debug_msg,
    debug_time,
    extract_sharded_tensors,
    extract_sharded_tensors_or_nonpersistent,
    fallback_logger,
    logger_stack,
    replace_prefix_for_sharding,
)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def captured_debug_logs():
    handler = _ListHandler()
    original_level = fallback_logger.level
    fallback_logger.setLevel(logging.DEBUG)
    fallback_logger.addHandler(handler)
    yield handler.messages
    fallback_logger.removeHandler(handler)
    fallback_logger.setLevel(original_level)


def test_extract_sharded_tensors_splits_sharded_from_other_values():
    sharded = ShardedTensor.from_rank_offsets('k', torch.arange(2.0), (0, 0, 2))
    obj = ShardedObject('o', object(), (1,), (0,))
    tensors, rest = extract_sharded_tensors({'tensor': sharded, 'object': obj, 'plain': 3})
    assert tensors == {'tensor': sharded}
    assert rest == {'object': obj, 'plain': 3}


def test_extract_sharded_tensors_or_nonpersistent_keeps_factories_and_local_objects():
    sharded = ShardedTensor.from_rank_offsets('k', torch.arange(2.0), (0, 0, 2))
    factory = ShardedTensorFactory('f', torch.arange(2.0), lambda *args: {}, lambda sd: sd)
    local = LocalNonpersistentObject(object())
    kept, dropped = extract_sharded_tensors_or_nonpersistent(
        {'sharded': sharded, 'factory': factory, 'local': local, 'plain': 3}
    )
    assert kept == {'sharded': sharded, 'factory': factory, 'local': local}
    assert dropped == {'plain': 3}


def test_replace_prefix_for_sharding_rewrites_all_sharded_keys():
    tensor = ShardedTensor.from_rank_offsets('old/module.weight', torch.arange(2.0), (0, 0, 2))
    obj = ShardedObject('old/extra', object(), (1,), (0,))
    replace_prefix_for_sharding({'t': tensor, 'o': obj}, 'old/', 'new/')
    assert tensor.key == 'new/module.weight'
    assert obj.key == 'new/extra'


def test_replace_prefix_for_sharding_rejects_mismatched_prefix():
    tensor = ShardedTensor.from_rank_offsets('other.key', torch.arange(2.0), (0, 0, 2))
    with pytest.raises(ValueError, match='Expected other.key to begin with prefix old/'):
        replace_prefix_for_sharding({'t': tensor}, 'old/', 'new/')


def test_apply_prefix_mapping_rewrites_only_matching_keys():
    tensor = ShardedTensor.from_rank_offsets('module.weight', torch.arange(2.0), (0, 0, 2))
    obj = ShardedObject('other.extra', object(), (1,), (0,))
    state = {'t': tensor, 'o': obj, 'plain': 5}
    apply_prefix_mapping(state, {'module.': 'renamed.', 'nothing': 'x'})
    assert tensor.key == 'renamed.weight'
    assert obj.key == 'other.extra'
    assert state['plain'] == 5


def test_logger_stack_yields_dotted_scope_and_fallback_logger():
    with logger_stack('outer') as (name_outer, logger_outer):
        with logger_stack('inner') as (name_inner, logger_inner):
            assert name_inner == 'outer.inner'
            assert logger_inner is logger_outer
    assert name_outer == 'outer'
    assert logger_outer is fallback_logger


def test_debug_time_logs_elapsed_time_for_block(captured_debug_logs):
    with debug_time('timed block'):
        torch.zeros(3)
    assert [message.startswith('timed block took ') for message in captured_debug_logs] == [True]


def test_debug_time_skips_logging_when_faster_than_threshold(captured_debug_logs):
    with debug_time('quiet block', threshold=3600.0):
        pass
    assert captured_debug_logs == []


def test_debug_msg_logs_scoped_message(captured_debug_logs):
    with logger_stack('scope'):
        debug_msg('Checkpoint initialized')
    assert captured_debug_logs[-1] == 'scope Checkpoint initialized'


def test_clean_metadata_for_serialization_drops_dp_cp_group():
    cleaned = _clean_metadata_for_serialization({'dp_cp_group': object(), 'other': 1})
    assert cleaned == {'other': 1}


def test_clean_metadata_for_serialization_returns_none_for_none():
    assert _clean_metadata_for_serialization(None) is None
