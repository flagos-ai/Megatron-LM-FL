# Copyright (c) 2026, BAAI. All rights reserved.
"""TE spec provider module selection: layer norms, linears, attention and MoE experts."""

import pytest

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions import transformer_engine_spec_provider as te_spec
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize(
    ('method', 'expected'),
    [
        ('linear', te_ext.TELinear),
        ('column_parallel_linear', te_ext.TEColumnParallelLinear),
        ('row_parallel_linear', te_ext.TERowParallelLinear),
        ('core_attention', te_ext.TEDotProductAttention),
        ('column_parallel_layer_norm_linear', te_ext.TELayerNormColumnParallelLinear),
    ],
    ids=['linear', 'column', 'row', 'attention', 'fused-layernorm-linear'],
)
def test_spec_provider_selects_te_modules(method, expected):
    assert getattr(TESpecProvider(), method)() is expected


def test_spec_provider_fuses_layernorm_and_linear():
    assert TESpecProvider().fuse_layernorm_and_linear() is True


def test_layer_norm_returns_apex_fused_layer_norm_for_qk_on_old_te(monkeypatch):
    monkeypatch.setattr(te_spec, 'is_te_min_version', lambda version, **kwargs: False)
    assert TESpecProvider().layer_norm(for_qk=True) is FusedLayerNorm


def test_layer_norm_uses_te_norm_for_qk_on_new_te():
    assert TESpecProvider().layer_norm(for_qk=True) is te_ext.TENorm


def test_layer_norm_selects_residual_adapter_when_requested():
    provider = TESpecProvider()
    assert provider.layer_norm(has_residual=True).__name__ == '_TENormWithResidual'
    assert provider.layer_norm(has_residual=False) is te_ext.TENorm


def test_layer_norm_residual_adapter_forwards_has_residual_to_te_norm():
    fused_config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization='RMSNorm',
        fused_residual_rmsnorm=True,
    )
    fused_norm = te_spec._TENormWithResidual(config=fused_config, hidden_size=16, eps=1e-5)
    assert isinstance(fused_norm, te_ext.TEFusedResidualRMSNorm)

    plain_config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    plain_norm = te_spec._TENormWithResidual(config=plain_config, hidden_size=16, eps=1e-5)
    assert type(plain_norm).__name__ == 'LayerNorm'
    assert plain_norm.weight.shape == (16,)


def test_grouped_mlp_modules_warns_and_uses_local_linears_on_old_te(monkeypatch):
    monkeypatch.setattr(te_spec, 'is_te_min_version', lambda version, **kwargs: False)
    with pytest.warns(UserWarning, match='Only transformer-engine>=1.7.0 supports MoE experts'):
        builder = TESpecProvider().grouped_mlp_modules(moe_use_grouped_gemm=False)
    assert builder.func is SequentialMLP
    submodules = builder.keywords['submodules']
    assert submodules.linear_fc1 is ColumnParallelLinear
    assert submodules.linear_fc2 is RowParallelLinear


def test_grouped_mlp_modules_uses_te_linears_on_new_te():
    builder = TESpecProvider().grouped_mlp_modules(moe_use_grouped_gemm=False)
    assert builder.func is SequentialMLP
    submodules = builder.keywords['submodules']
    assert submodules.linear_fc1 is te_ext.TEColumnParallelLinear
    assert submodules.linear_fc2 is te_ext.TERowParallelLinear


def test_grouped_mlp_modules_selects_grouped_linears_when_grouped_gemm_enabled():
    builder = TESpecProvider().grouped_mlp_modules(moe_use_grouped_gemm=True)
    assert builder.func is TEGroupedMLP
    submodules = builder.keywords['submodules']
    assert submodules.linear_fc1 is te_ext.TEColumnParallelGroupedLinear
    assert submodules.linear_fc2 is te_ext.TERowParallelGroupedLinear


def test_activation_func_returns_te_activation_op():
    assert TESpecProvider().activation_func() is te_ext.TEActivationOp
