# Copyright (c) 2026, BAAI. All rights reserved.
"""TorchFullyShardedDataParallel wrapping by transformer impl; load_state_dict leaves params unchanged."""

import pytest
import torch
from torch.distributed.fsdp import FSDPModule

from megatron.core.distributed import DistributedDataParallelConfig, TorchFullyShardedDataParallel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils


class _WrappableModel(MegatronModule):
    _fsdp_modules = [torch.nn.Linear]

    def __init__(self, config, with_column_parallel=False):
        super().__init__(config)
        self.linear0 = torch.nn.Linear(4, 4)
        self.linear1 = torch.nn.Linear(4, 4)
        if with_column_parallel:
            self.column_parallel_linear = ColumnParallelLinear(
                input_size=4, output_size=4, config=config, init_method=init_method_normal(0.02)
            )

    def forward(self, x):
        return self.linear1(torch.relu(self.linear0(x)))


def _is_fsdp_wrapped(module):
    return module.__class__.__name__.startswith("FSDP")


@pytest.fixture
def init_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


def _make_model(with_column_parallel=True, **config_kwargs):
    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True, **config_kwargs)
    ddp_config = DistributedDataParallelConfig()
    module = _WrappableModel(config, with_column_parallel=with_column_parallel)
    return TorchFullyShardedDataParallel(config, ddp_config, module)


def test_local_transformer_impl_excludes_column_parallel_linear(init_model_parallel):
    fsdp_model = _make_model(with_column_parallel=True, transformer_impl="local")
    module = fsdp_model.module
    assert _is_fsdp_wrapped(module.linear0)
    assert _is_fsdp_wrapped(module.linear1)
    assert not _is_fsdp_wrapped(module.column_parallel_linear)


def test_te_transformer_impl_wraps_column_parallel_linear(init_model_parallel):
    torch.manual_seed(123)
    reference = _WrappableModel(
        TransformerConfig(num_layers=1, kv_channels=1, bf16=True), with_column_parallel=True
    ).cuda()
    torch.manual_seed(123)
    fsdp_model = _make_model(with_column_parallel=True, transformer_impl="transformer_engine")
    module = fsdp_model.module
    assert _is_fsdp_wrapped(module.linear0)
    assert _is_fsdp_wrapped(module.column_parallel_linear)

    x = torch.randn(2, 4, device="cuda")
    torch.testing.assert_close(fsdp_model(x), reference(x), rtol=1.6e-2, atol=1e-2)


def test_recompute_granularity_sets_backward_prefetch_chain(init_model_parallel, monkeypatch):
    recorded = []
    original = FSDPModule.set_modules_to_backward_prefetch

    def spy(self, modules):
        recorded.append((self, list(modules)))
        original(self, modules)

    monkeypatch.setattr(FSDPModule, "set_modules_to_backward_prefetch", spy)

    fsdp_model = _make_model(
        with_column_parallel=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    module = fsdp_model.module

    assert [modules for _, modules in recorded] == [[], [module.linear0]]
    assert recorded[0][0] is module.linear0
    assert recorded[1][0] is module.linear1


def test_recompute_granularity_none_skips_backward_prefetch(init_model_parallel, monkeypatch):
    recorded = []
    original = FSDPModule.set_modules_to_backward_prefetch

    def spy(self, modules):
        recorded.append((self, list(modules)))
        original(self, modules)

    monkeypatch.setattr(FSDPModule, "set_modules_to_backward_prefetch", spy)

    _make_model(with_column_parallel=False, recompute_granularity=None)
    assert recorded == []


def test_load_state_dict_is_a_no_op(init_model_parallel):
    fsdp_model = _make_model()
    weight_before = fsdp_model.module.linear0.weight.detach().clone()

    result = fsdp_model.load_state_dict(
        {"linear0.weight": torch.zeros_like(weight_before)}, strict=True
    )

    assert result is None
    torch.testing.assert_close(
        fsdp_model.module.linear0.weight.detach(), weight_before, rtol=0, atol=0
    )
