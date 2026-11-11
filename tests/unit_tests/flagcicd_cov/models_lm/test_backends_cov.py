# Copyright (c) 2026, BAAI. All rights reserved.
"""Spec providers: local layer-norm fallback and inference TE module selection."""

import importlib
import importlib.util
import sys

import pytest

import megatron.core.models.backends as backends_module
from megatron.core.extensions.transformer_engine import TEDotProductAttention, TELinear, TENorm
from megatron.core.models.backends import InferenceSpecProvider, LocalSpecProvider
from megatron.core.tensor_parallel.inference_layers import InferenceLayerNormColumnParallelLinear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.torch_norm import WrappedTorchNorm


@pytest.fixture
def restore_backends_module():
    yield
    importlib.reload(backends_module)


def test_local_spec_provider_has_no_fused_layernorm_linear_module():
    provider = LocalSpecProvider()

    assert provider.fuse_layernorm_and_linear() is False
    assert provider.column_parallel_layer_norm_linear() is None


def test_local_spec_provider_selects_megatron_core_modules():
    provider = LocalSpecProvider()

    assert provider.column_parallel_linear() is ColumnParallelLinear
    assert provider.row_parallel_linear() is RowParallelLinear
    assert provider.core_attention() is DotProductAttention


def test_local_spec_provider_layer_norm_switches_to_torch_norm_when_rms_requested(monkeypatch):
    with monkeypatch.context() as snapshot:
        snapshot.setattr(backends_module, 'LNImpl', backends_module.LNImpl)
        returned = LocalSpecProvider().layer_norm(rms_norm=True)
        assert returned is WrappedTorchNorm
        assert backends_module.LNImpl is WrappedTorchNorm


def test_inference_spec_provider_fuses_layernorm_and_linear_into_te_module():
    provider = InferenceSpecProvider()

    assert provider.fuse_layernorm_and_linear() is True
    assert provider.column_parallel_layer_norm_linear() is (InferenceLayerNormColumnParallelLinear)


def test_inference_spec_provider_returns_te_linear_modules():
    provider = InferenceSpecProvider()

    assert provider.linear() is TELinear
    assert provider.core_attention() is TEDotProductAttention


def test_inference_spec_provider_layer_norm_returns_te_norm_for_qk():
    assert InferenceSpecProvider().layer_norm(for_qk=True) is TENorm


def test_backends_fall_back_to_torch_norm_when_apex_is_unavailable(
    monkeypatch, restore_backends_module
):
    with monkeypatch.context() as blocker:
        blocker.setitem(sys.modules, "apex", None)
        with pytest.warns(UserWarning, match="Apex is not installed. Falling back to Torch Norm"):
            importlib.reload(backends_module)
        assert backends_module.HAVE_APEX is False
        assert backends_module.LNImpl is WrappedTorchNorm
        assert backends_module.FusedLayerNorm is None

    importlib.reload(backends_module)
    assert backends_module.HAVE_APEX is (importlib.util.find_spec("apex") is not None)
