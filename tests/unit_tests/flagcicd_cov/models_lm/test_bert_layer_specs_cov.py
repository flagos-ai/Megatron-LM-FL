# Copyright (c) 2026, BAAI. All rights reserved.
"""bert_layer_specs import fallbacks: torch norm without apex, ImportError without TE."""

import importlib
import importlib.util
import sys

import pytest

import megatron.core.extensions.transformer_engine as te_extensions
import megatron.core.models.bert.bert_layer_specs as bert_layer_specs
from megatron.core.transformer.torch_norm import WrappedTorchNorm


@pytest.fixture
def restore_bert_layer_specs():
    yield
    importlib.reload(bert_layer_specs)


def test_bert_layer_specs_raise_import_error_without_te_when_building_te_spec(
    monkeypatch, restore_bert_layer_specs
):
    with monkeypatch.context() as blocker:
        blocker.setattr(te_extensions, "HAVE_TE", False)
        blocker.setitem(sys.modules, "apex", None)
        with pytest.warns(UserWarning, match="Apex is not installed. Falling back to Torch Norm"):
            importlib.reload(bert_layer_specs)
        assert bert_layer_specs.TEDotProductAttention is None
        assert bert_layer_specs.HAVE_APEX is False
        assert bert_layer_specs.LNImpl is WrappedTorchNorm
        with pytest.raises(ImportError, match="Transformer Engine is not installed"):
            bert_layer_specs.get_bert_layer_with_transformer_engine_submodules()

    importlib.reload(bert_layer_specs)
    assert bert_layer_specs.HAVE_TE == te_extensions.HAVE_TE
    assert bert_layer_specs.HAVE_APEX is (importlib.util.find_spec("apex") is not None)
