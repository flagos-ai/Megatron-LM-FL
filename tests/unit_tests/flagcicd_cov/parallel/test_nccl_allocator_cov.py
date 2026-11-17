# Copyright (c) 2026, BAAI. All rights reserved.
'''nccl_mem allocator context: function argument-name inspection plus the disabled no-op path and enabled device/process-group resolution of the symmetric-memory context.'''

import contextlib

import pytest
import torch

from megatron.core.nccl_allocator import get_func_args, nccl_mem
from tests.unit_tests.test_utilities import Utils


def test_get_func_args_returns_parameter_names_in_order():
    def sample(first, second=2, **extra):
        return first

    assert get_func_args(sample) == ['first', 'second', 'extra']


def test_nccl_mem_context_is_noop_when_disabled():
    memory = nccl_mem('cov-pool', enabled=False)

    assert memory.pool == 'cov-pool'
    assert memory.device is None
    assert memory.group is None
    assert memory.symmetric is True
    assert isinstance(memory.mem_context, contextlib.nullcontext)
    with memory as entered:
        assert entered is None
    assert memory.group is None


@pytest.fixture
def model_parallel():
    def setup(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)

    yield setup
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'device', [None, 0, 'cuda:0'], ids=['current-device', 'device-index', 'device-name']
)
def test_nccl_mem_enabled_context_resolves_device_and_default_group(model_parallel, device):
    model_parallel(1, 1)
    default_group = torch.distributed.distributed_c10d._get_default_group()

    memory = nccl_mem(None, enabled=True, device=device)

    if device is None:
        expected = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        expected = torch.device('cuda:0')
    assert memory.device == expected
    assert memory.group is default_group
    assert memory.symmetric is True
