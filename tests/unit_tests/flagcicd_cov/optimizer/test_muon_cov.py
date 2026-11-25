# Copyright (c) 2026, BAAI. All rights reserved.
"""Muon shim: config placement, layer-wise validation, delegation to get_megatron_optimizer."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def _config(**overrides):
    defaults = dict(
        optimizer='adam',
        lr=0.01,
        adam_eps=1e-8,
        weight_decay=0.0,
        bf16=True,
        use_distributed_optimizer=False,
        clip_grad=0.0,
    )
    return OptimizerConfig(**{**defaults, **overrides})


def _build_ddp_model():
    linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=_device())
    linear.weight.data.fill_(1.0)
    return DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        linear,
    )


def test_get_megatron_muon_optimizer_rejects_layer_wise_without_dist_prefix():
    with pytest.raises(
        ValueError, match='Layer-wise distributed optimizer is enabled by dist_ prefix'
    ):
        get_megatron_muon_optimizer(
            _config(), model_chunks=[], layer_wise_distributed_optimizer=True
        )


@pytest.mark.parametrize(
    "pass_config_as_keyword",
    [pytest.param(False, id='config-positional'), pytest.param(True, id='config-keyword')],
)
def test_get_megatron_muon_optimizer_delegates_to_get_megatron_optimizer(
    model_parallel, pass_config_as_keyword
):
    model_parallel()
    ddp_model = _build_ddp_model()
    config = _config()

    if pass_config_as_keyword:
        optimizer = get_megatron_muon_optimizer(config=config, model_chunks=[ddp_model])
    else:
        optimizer = get_megatron_muon_optimizer(config, [ddp_model])

    assert isinstance(optimizer, ChainedOptimizer)
    assert len(optimizer.chained_optimizers) == 1
    sub_optimizer = optimizer.chained_optimizers[0]
    assert isinstance(sub_optimizer, Float16OptimizerWithFloat16Params)
    assert isinstance(sub_optimizer.optimizer, torch.optim.Optimizer)
    assert len(sub_optimizer.get_parameters()) == 1
    assert sub_optimizer.get_parameters()[0].dtype == torch.float32
