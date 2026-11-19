# Copyright (c) 2026, BAAI. All rights reserved.
"""Heterogeneous layer specs: qk layernorm selection by backend and normalization."""

import importlib
import importlib.util
import sys

import pytest

import megatron.core.extensions.transformer_engine as te_extensions
import megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs as hetero_specs
from megatron.core.transformer.torch_norm import WrappedTorchNorm


@pytest.fixture
def restore_hetero_specs():
    yield
    importlib.reload(hetero_specs)


def test_get_qk_layernorm_returns_te_norm_when_te_at_least_1_9():
    qk_norm = hetero_specs._get_qk_layernorm(use_te=True, normalization='LayerNorm')

    assert qk_norm is hetero_specs.TENorm


@pytest.mark.parametrize('normalization', ['LayerNorm', 'RMSNorm'], ids=['layernorm', 'rmsnorm'])
def test_get_qk_layernorm_returns_ln_impl_when_not_using_te(normalization):
    expected_impl = hetero_specs.LNImpl if normalization == 'LayerNorm' else WrappedTorchNorm

    qk_norm = hetero_specs._get_qk_layernorm(use_te=False, normalization=normalization)

    assert qk_norm is expected_impl


def test_heterogeneous_layer_specs_fall_back_when_te_and_apex_are_unavailable(
    monkeypatch, restore_hetero_specs
):
    with monkeypatch.context() as blocker:
        blocker.setattr(te_extensions, 'HAVE_TE', False)
        blocker.setitem(sys.modules, 'apex', None)
        with pytest.warns(UserWarning, match='Apex is not installed. Falling back to Torch Norm'):
            importlib.reload(hetero_specs)
        assert hetero_specs.TEDotProductAttention is None
        assert hetero_specs.TENorm is None
        assert hetero_specs.HAVE_APEX is False
        assert hetero_specs.LNImpl is WrappedTorchNorm
        assert (
            hetero_specs._get_qk_layernorm(use_te=False, normalization='LayerNorm')
            is WrappedTorchNorm
        )

    importlib.reload(hetero_specs)
    assert hetero_specs.HAVE_TE == te_extensions.HAVE_TE
    assert hetero_specs.HAVE_APEX is (importlib.util.find_spec('apex') is not None)
