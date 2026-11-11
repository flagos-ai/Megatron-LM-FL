# Copyright (c) 2026, BAAI. All rights reserved.
"""T5 block specs: per-layer repetition, cross-attention layout, import fallbacks."""

import importlib
import importlib.util
import sys

import pytest

import megatron.core.extensions.transformer_engine as te_extensions
import megatron.core.models.T5.t5_spec as t5_spec
from megatron.core.transformer.torch_norm import WrappedTorchNorm


@pytest.fixture
def restore_t5_spec():
    yield
    importlib.reload(t5_spec)


def test_t5_encoder_with_local_block_spec_repeats_layer_spec_per_layer():
    block_spec = t5_spec.get_t5_encoder_with_local_block_spec(3)

    assert len(block_spec.layer_specs) == 3
    assert all(spec is block_spec.layer_specs[0] for spec in block_spec.layer_specs)
    assert block_spec.layer_specs[0].module is t5_spec.TransformerLayer
    assert block_spec.layer_norm is t5_spec.TENorm


def test_t5_decoder_with_local_block_spec_repeats_layer_spec_per_layer():
    block_spec = t5_spec.get_t5_decoder_with_local_block_spec(2)

    assert len(block_spec.layer_specs) == 2
    assert all(spec is block_spec.layer_specs[0] for spec in block_spec.layer_specs)
    assert block_spec.layer_specs[0].submodules.cross_attention.module is t5_spec.CrossAttention


def test_t5_spec_falls_back_when_te_and_apex_are_unavailable(monkeypatch, restore_t5_spec):
    with monkeypatch.context() as blocker:
        blocker.setattr(te_extensions, "HAVE_TE", False)
        blocker.setitem(sys.modules, "apex", None)
        with pytest.warns(UserWarning, match="Apex is not installed. Falling back to Torch Norm"):
            importlib.reload(t5_spec)
        assert t5_spec.HAVE_TE is False
        assert t5_spec.TEDotProductAttention is None
        assert t5_spec.TELayerNormColumnParallelLinear is None
        assert t5_spec.HAVE_APEX is False
        assert t5_spec.LNImpl is WrappedTorchNorm

    importlib.reload(t5_spec)
    assert t5_spec.HAVE_TE == te_extensions.HAVE_TE
    assert t5_spec.HAVE_APEX is (importlib.util.find_spec("apex") is not None)
