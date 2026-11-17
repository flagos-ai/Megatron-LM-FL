# Copyright (c) 2026, BAAI. All rights reserved.
"""bert_lm_head LNImpl fallback when fused layer norm is unavailable."""

import importlib

import pytest

import megatron.core.fusions.fused_layer_norm as fused_layer_norm
import megatron.core.models.bert.bert_lm_head as bert_lm_head
from megatron.core.transformer.torch_norm import WrappedTorchNorm


@pytest.fixture
def restore_bert_lm_head():
    yield
    importlib.reload(bert_lm_head)


def test_bert_lm_head_falls_back_to_torch_norm_without_fused_layer_norm(
    monkeypatch, restore_bert_lm_head
):
    with monkeypatch.context() as blocker:
        blocker.setattr(fused_layer_norm, "HAVE_FUSED_LAYER_NORM", False)
        with pytest.warns(UserWarning, match="Apex is not installed. Falling back to Torch Norm"):
            importlib.reload(bert_lm_head)
        assert bert_lm_head.LNImpl is WrappedTorchNorm

    importlib.reload(bert_lm_head)
    expected_impl = (
        fused_layer_norm.FusedLayerNorm
        if fused_layer_norm.HAVE_FUSED_LAYER_NORM
        else WrappedTorchNorm
    )
    assert bert_lm_head.LNImpl is expected_impl
