# Copyright (c) 2026, BAAI. All rights reserved.

"""Strategy base interfaces: default-strategy backend fallback, NotImplemented contracts, the sharded-object handling default, and the synchronous fallback of async saves."""

import logging

import pytest

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    LoadStrategyBase,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from tests.unit_tests.test_utilities import Utils


class _DelegatingLoad(LoadShardedStrategy):
    """Concrete load strategy that exercises the base-class contracts."""

    def check_backend_compatibility(self, loaded_backend):
        return super().check_backend_compatibility(loaded_backend)

    def check_version_compatibility(self, loaded_version):
        return super().check_version_compatibility(loaded_version)

    def load(self, sharded_state_dict, checkpoint_dir):
        return super().load(sharded_state_dict, checkpoint_dir)

    def load_tensors_metadata(self, checkpoint_dir):
        return super().load_tensors_metadata(checkpoint_dir)

    def remove_sharded_tensors(self, checkpoint_dir, key_prefix):
        return super().remove_sharded_tensors(checkpoint_dir, key_prefix)


class _ObjectHandlingLoad(_DelegatingLoad):
    @property
    def can_handle_sharded_objects(self):
        return True


class _DelegatingSave(SaveShardedStrategy):
    """Concrete save strategy that exercises the base-class contracts."""

    def save(self, sharded_state_dict, checkpoint_dir):
        return super().save(sharded_state_dict, checkpoint_dir)


class _RaisingAsyncSave(AsyncSaveShardedStrategy):
    def async_save(self, sharded_state_dict, checkpoint_dir):
        return super().async_save(sharded_state_dict, checkpoint_dir)


class _SyncAsyncSave(AsyncSaveShardedStrategy):
    def __init__(self, finalize_calls):
        super().__init__('mock_async', 1)
        self.finalize_calls = finalize_calls

    def async_save(self, sharded_state_dict, checkpoint_dir):
        def finalize_fn():
            self.finalize_calls.append('finalized')

        return AsyncRequest(None, (), [finalize_fn])


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_get_default_strategy_warns_for_unsupported_backend(caplog):
    with caplog.at_level(logging.WARNING):
        strategy = get_default_strategy(StrategyAction.LOAD_SHARDED, 'bad_backend', 1)

    assert isinstance(strategy, TorchDistLoadShardedStrategy)
    assert 'bad_backend is not supported. `torch_dist` backend will be used' in caplog.text
    save_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, 'torch_dist', 1)
    assert isinstance(save_strategy, TorchDistSaveShardedStrategy)


def test_get_default_strategy_rejects_unknown_action():
    with pytest.raises(AssertionError, match='LOAD_COMMON is not supported'):
        get_default_strategy(StrategyAction.LOAD_COMMON, 'torch_dist', 1)


def test_load_strategy_base_reports_cannot_handle_sharded_objects():
    strategy = _DelegatingLoad()

    assert strategy.can_handle_sharded_objects is False
    assert _ObjectHandlingLoad().can_handle_sharded_objects is True


@pytest.mark.parametrize(
    ('method', 'args'),
    [
        pytest.param('check_backend_compatibility', ('torch_dist',), id='backend'),
        pytest.param('check_version_compatibility', (1,), id='version'),
        pytest.param('load', ({'key': 1}, '/unused'), id='load'),
        pytest.param('remove_sharded_tensors', ('/unused', 'prefix'), id='remove'),
    ],
)
def test_load_strategy_base_contract_raises_not_implemented(method, args):
    strategy = _DelegatingLoad()

    with pytest.raises(NotImplementedError) as caught:
        getattr(strategy, method)(*args)

    assert str(caught.value) == ''


def test_load_tensors_metadata_contract_message_names_the_strategy():
    strategy = _DelegatingLoad()

    with pytest.raises(
        NotImplementedError, match='Loading only tensors metadata not implemented for'
    ):
        strategy.load_tensors_metadata('/unused')


def test_load_sharded_metadata_delegates_when_not_handling_objects():
    strategy = _DelegatingLoad()

    with pytest.raises(NotImplementedError, match='Loading only tensors metadata'):
        strategy.load_sharded_metadata('/unused')


def test_load_sharded_metadata_raises_when_handling_objects():
    strategy = _ObjectHandlingLoad()

    with pytest.raises(NotImplementedError, match='Loading only sharded metadata not implemented'):
        strategy.load_sharded_metadata('/unused')


def test_save_strategy_base_reports_backend_and_version():
    strategy = _DelegatingSave('mock', 2)

    assert str(strategy) == '_DelegatingSave(mock, 2)'
    assert strategy.can_handle_sharded_objects is False


def test_save_sharded_strategy_save_contract_raises_not_implemented():
    strategy = _DelegatingSave('mock', 1)

    with pytest.raises(NotImplementedError) as caught:
        strategy.save({'key': 1}, '/unused')

    assert str(caught.value) == ''


def test_async_save_strategy_async_save_contract_raises_not_implemented():
    strategy = _RaisingAsyncSave('mock', 1)

    with pytest.raises(NotImplementedError) as caught:
        strategy.async_save({'key': 1}, '/unused')

    assert str(caught.value) == ''


def test_async_save_strategy_save_executes_request_synchronously(model_parallel, caplog):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    finalize_calls = []
    strategy = _SyncAsyncSave(finalize_calls)

    with caplog.at_level(logging.WARNING):
        strategy.save({'key': 1}, '/unused')

    assert finalize_calls == ['finalized']
    assert 'AsyncSaveShardedStrategy is deprecated' in caplog.text
