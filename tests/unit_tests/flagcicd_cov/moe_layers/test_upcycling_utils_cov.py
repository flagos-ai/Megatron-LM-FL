# Copyright (c) 2026, BAAI. All rights reserved.
"""Dense-to-MoE upcycling: state-dict key and value conversion with activation scaling, and
checkpoint loading into expert weights."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import megatron.core.transformer.moe.router as router_module
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.upcycling_utils import _find_submodule, upcycle_state_dict
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _cuda_or_cpu_device():
    """cuda on the A100 CI node, cpu on a laptop: what the real platform reports."""
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


class _MlpModel(nn.Module):
    """Smallest model that exposes its mlp the way upcycling expects."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


def _dense_model():
    """Dense [8, 16] MLP with deterministic weights and biases."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        ffn_hidden_size=16,
        use_cpu_initialization=True,
        add_bias_linear=True,
        perform_initialization=False,
    )
    mlp = MLP(
        config,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
        is_expert=False,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    model = _MlpModel(mlp)
    model.load_state_dict(
        {
            'mlp.linear_fc1.weight': torch.arange(128, dtype=torch.float32).reshape(16, 8),
            'mlp.linear_fc1.bias': torch.arange(16, dtype=torch.float32),
            'mlp.linear_fc2.weight': torch.arange(128, dtype=torch.float32).reshape(8, 16),
            'mlp.linear_fc2.bias': torch.arange(8, dtype=torch.float32),
        },
        strict=False,
    )
    return model


def _moe_model(pre_softmax=False, activation_func=torch.nn.functional.gelu):
    """4-expert MoE layer over half the dense ffn size, with deterministic parameters."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=8,
        moe_router_topk=2,
        moe_router_pre_softmax=pre_softmax,
        use_cpu_initialization=True,
        add_bias_linear=True,
        perform_initialization=False,
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type='allgather',
        moe_router_load_balancing_type='aux_loss',
        activation_func=activation_func,
    )
    submodules = get_submodules(
        get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False).mlp
    )
    model = _MlpModel(MoELayer(config, submodules))
    model.load_state_dict(
        {
            'mlp.router.weight': torch.arange(32, dtype=torch.float32).reshape(4, 8),
            'mlp.router.bias': torch.arange(4, dtype=torch.float32),
        },
        strict=False,
    )
    return model


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def pinned_device(monkeypatch):
    """Pin the router's platform object to one device for both run environments."""
    device = _cuda_or_cpu_device()
    monkeypatch.setattr(
        router_module, 'cur_platform', SimpleNamespace(current_device=lambda: device)
    )
    return device


class TestFindSubmodule:

    def test_find_submodule_returns_none_when_no_module_matches(self):
        assert _find_submodule(torch.nn.Sequential(), 'mlp') is None
        assert _find_submodule(_MlpModel(torch.nn.Identity()), 'missing') is None


class TestGetConfig:

    def test_get_config_rejects_unsupported_activation_func(self, model_parallel, pinned_device):
        model_parallel()

        with pytest.raises(ValueError, match='The activation func is not supported by upcycling'):
            upcycling_utils._get_config(
                _moe_model(activation_func=torch.nn.functional.relu), _dense_model()
            )

    def test_get_config_rejects_unsupported_experts_type(self, model_parallel, pinned_device):
        model_parallel()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_ffn_hidden_size=8,
            moe_router_topk=2,
            use_cpu_initialization=True,
            add_bias_linear=True,
            perform_initialization=False,
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type='allgather',
        )
        submodules = MoESubmodules(
            experts=lambda num_local_experts, cfg, **kwargs: torch.nn.Identity()
        )
        moe_model = _MlpModel(MoELayer(config, submodules))

        with pytest.raises(TypeError, match='The experts type'):
            upcycling_utils._get_config(moe_model, _dense_model())


class TestUpcycleStateDict:

    def test_upcycle_state_dict_splits_dense_weights_and_biases_into_experts(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        dense_model = _dense_model()
        moe_model = _moe_model()

        converted = upcycle_state_dict([moe_model], [dense_model])['model']

        scale = torch.sqrt(torch.tensor(2.0))
        dense_fc1 = torch.arange(128, dtype=torch.float32).reshape(16, 8)
        dense_fc2 = torch.arange(128, dtype=torch.float32).reshape(8, 16)
        fc1_first, fc1_second = dense_fc1[:8] * scale, dense_fc1[8:] * scale
        fc2_first, fc2_second = dense_fc2[:, :8] * scale, dense_fc2[:, 8:] * scale
        fc1_bias_first = torch.arange(8, dtype=torch.float32) * scale
        fc1_bias_second = torch.arange(8, 16, dtype=torch.float32) * scale
        fc2_bias = torch.arange(8, dtype=torch.float32)

        assert torch.equal(converted['mlp.experts.local_experts.0.linear_fc1.weight'], fc1_first)
        assert torch.equal(converted['mlp.experts.local_experts.1.linear_fc1.weight'], fc1_second)
        assert torch.equal(converted['mlp.experts.local_experts.2.linear_fc1.weight'], fc1_first)
        assert torch.equal(converted['mlp.experts.local_experts.3.linear_fc1.weight'], fc1_second)
        assert torch.equal(converted['mlp.experts.local_experts.0.linear_fc1.bias'], fc1_bias_first)
        assert torch.equal(
            converted['mlp.experts.local_experts.3.linear_fc1.bias'], fc1_bias_second
        )
        assert torch.equal(converted['mlp.experts.local_experts.0.linear_fc2.weight'], fc2_first)
        assert torch.equal(converted['mlp.experts.local_experts.1.linear_fc2.weight'], fc2_second)
        assert torch.equal(converted['mlp.experts.local_experts.0.linear_fc2.bias'], fc2_bias)
        assert torch.equal(converted['mlp.experts.local_experts.3.linear_fc2.bias'], fc2_bias)
        assert torch.equal(
            converted['mlp.router.weight'],
            torch.arange(32, dtype=torch.float32).reshape(4, 8)[:2].repeat_interleave(2, dim=0),
        )

    def test_upcycle_state_dict_uses_pre_softmax_activation_scale(
        self, model_parallel, pinned_device
    ):
        model_parallel()
        dense_model = _dense_model()
        moe_model = _moe_model(pre_softmax=True)

        converted = upcycle_state_dict([moe_model], [dense_model])['model']

        scale = torch.tensor(2.0)
        dense_fc1 = torch.arange(128, dtype=torch.float32).reshape(16, 8)
        assert torch.equal(
            converted['mlp.experts.local_experts.0.linear_fc1.weight'], dense_fc1[:8] * scale
        )
        assert torch.equal(
            converted['mlp.experts.local_experts.1.linear_fc1.weight'], dense_fc1[8:] * scale
        )


def test_load_and_upcycle_model_loads_expert_weights_and_returns_iteration(
    model_parallel, pinned_device
):
    model_parallel()
    dense_model = _dense_model()
    moe_model = _moe_model()

    iteration, num_floating_point_operations = upcycling_utils.load_and_upcycle_model(
        lambda: (1000, 42.0), [moe_model], [dense_model], strict=True
    )

    assert iteration == 1000
    assert num_floating_point_operations == 42.0
    scale = torch.sqrt(torch.tensor(2.0))
    dense_fc1 = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    torch.testing.assert_close(
        moe_model.mlp.experts.local_experts[0].linear_fc1.weight,
        dense_fc1[:8] * scale,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        moe_model.mlp.experts.local_experts[3].linear_fc1.weight,
        dense_fc1[8:] * scale,
        rtol=1e-6,
        atol=1e-6,
    )
