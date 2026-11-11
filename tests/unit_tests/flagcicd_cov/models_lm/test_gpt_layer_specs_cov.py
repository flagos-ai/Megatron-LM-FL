# Copyright (c) 2026, BAAI. All rights reserved.
"""GPT layer specs: inference/MLA submodules, fp8 deprecations, decoder layer patterns, MTP block specs."""

import importlib
import importlib.util
import sys

import pytest

import megatron.core.extensions.transformer_engine as te_extensions
import megatron.core.models.backends as backends_module
import megatron.core.models.gpt.gpt_layer_specs as gpt_layer_specs
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import (
    _get_mlp_module_spec,
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_layer_local_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_inference_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
    get_mlp_module_spec,
)
from megatron.core.tensor_parallel.inference_layers import InferenceColumnParallelLinear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.torch_norm import L2Norm, WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    values = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.fixture
def restore_gpt_layer_specs():
    yield
    importlib.reload(gpt_layer_specs)


def test_gpt_layer_with_inference_submodules_uses_inference_linear_modules():
    submodules = get_gpt_layer_with_inference_submodules()

    assert submodules.self_attention.module is SelfAttention
    assert submodules.self_attention.submodules.linear_qkv is (
        backends_module.InferenceLayerNormColumnParallelLinear
    )
    assert submodules.self_attention.submodules.core_attention is (
        backends_module.TEDotProductAttention
    )
    assert submodules.self_attention.submodules.q_layernorm is IdentityOp
    assert submodules.pre_mlp_layernorm is IdentityOp
    assert submodules.sharded_state_dict_keys_map == {
        'mlp.0.weight': 'mlp.linear_fc1.layer_norm_weight',
        'mlp.0.bias': 'mlp.linear_fc1.layer_norm_bias',
        'mlp.1.basic_ops.0.weight': 'mlp.linear_fc1.weight',
        'mlp.1.basic_ops.1.bias': 'mlp.linear_fc1.bias',
        'mlp.3.basic_ops.0.weight': 'mlp.linear_fc2.weight',
        'mlp.3.basic_ops.1.bias': 'mlp.linear_fc2.bias',
    }


def test_gpt_layer_with_inference_submodules_wraps_qk_norms_when_requested():
    submodules = get_gpt_layer_with_inference_submodules(qk_layernorm=True, qk_l2_norm=True)

    assert submodules.self_attention.submodules.q_layernorm is L2Norm
    assert submodules.self_attention.submodules.k_layernorm is L2Norm


def test_gpt_layer_with_inference_submodules_builds_mla_attention():
    submodules = get_gpt_layer_with_inference_submodules(multi_latent_attention=True)

    assert submodules.self_attention.module is MLASelfAttention
    assert submodules.self_attention.submodules.linear_q_proj is InferenceColumnParallelLinear
    assert submodules.self_attention.submodules.q_layernorm is IdentityOp


def test_gpt_layer_with_inference_spec_wraps_transformer_layer():
    spec = get_gpt_layer_with_inference_spec()

    assert spec.module is TransformerLayer
    assert spec.submodules.self_attention.module is SelfAttention


def test_get_gpt_layer_with_transformer_engine_submodules_warns_on_deprecated_fp8():
    with pytest.warns(UserWarning, match='has been deprecated'):
        get_gpt_layer_with_transformer_engine_submodules(fp8='e4m3')


def test_get_gpt_layer_local_submodules_builds_mla_attention(monkeypatch):
    with monkeypatch.context() as snapshot:
        snapshot.setattr(backends_module, 'LNImpl', backends_module.LNImpl)
        submodules = get_gpt_layer_local_submodules(multi_latent_attention=True)

    assert submodules.self_attention.module is MLASelfAttention
    assert submodules.self_attention.submodules.linear_q_proj is ColumnParallelLinear
    assert submodules.input_layernorm is gpt_layer_specs.LNImpl
    assert submodules.pre_mlp_layernorm is gpt_layer_specs.LNImpl


def test_get_gpt_layer_local_submodules_switches_ln_impl_for_rms_norm(monkeypatch):
    with monkeypatch.context() as snapshot:
        snapshot.setattr(backends_module, 'LNImpl', backends_module.LNImpl)
        submodules = get_gpt_layer_local_submodules(normalization='RMSNorm', qk_layernorm=True)

    assert submodules.input_layernorm is WrappedTorchNorm
    assert submodules.self_attention.submodules.q_layernorm is WrappedTorchNorm


def test_get_gpt_layer_local_submodules_warns_on_deprecated_fp8():
    with pytest.warns(UserWarning, match='has been deprecated'):
        get_gpt_layer_local_submodules(fp8='e4m3')


def test_deprecated_private_mlp_spec_helper_warns_and_delegates():
    with pytest.warns(UserWarning, match='deprecation track'):
        builder = _get_mlp_module_spec(use_te=False, num_experts=None)

    submodules = builder.keywords['submodules']
    assert builder.func.__self__ is MLP
    assert submodules.linear_fc1 is ColumnParallelLinear
    assert submodules.linear_fc2 is RowParallelLinear
    assert submodules.activation_func is None


def test_get_mlp_module_spec_warns_on_deprecated_fp8():
    with pytest.warns(UserWarning, match='has been deprecated'):
        get_mlp_module_spec(use_te=False, fp8='e4m3')


def test_get_mlp_module_spec_rejects_te_op_fuser_with_experts():
    with pytest.raises(ValueError, match='does not support mixture-of-experts'):
        get_mlp_module_spec(use_te_op_fuser=True, num_experts=4)


def test_get_mlp_module_spec_builds_te_op_fuser_mlp():
    builder = get_mlp_module_spec(use_te_op_fuser=True, num_experts=None)

    submodules = builder.keywords['submodules']
    assert builder.func.__self__ is gpt_layer_specs.TEFusedMLP
    assert submodules.linear_fc1 is TESpecProvider().column_parallel_layer_norm_linear()
    assert submodules.linear_fc2 is TESpecProvider().row_parallel_linear()


def test_get_gpt_decoder_layer_specs_rejects_non_int_list_moe_layer_freq(model_parallel):
    model_parallel()
    config = _config(moe_layer_freq=1.5)

    with pytest.raises(ValueError, match='Invalid moe_layer_freq'):
        get_gpt_decoder_layer_specs(config, use_transformer_engine=False)


def test_get_gpt_decoder_layer_specs_rejects_invalid_layer_pattern_entry(model_parallel):
    model_parallel()
    config = _config(moe_layer_freq=[2, 2])

    with pytest.raises(ValueError, match='Invalid layer pattern'):
        get_gpt_decoder_layer_specs(config, use_transformer_engine=False)


def test_get_gpt_mtp_block_spec_copies_last_layer_from_block_spec(model_parallel):
    model_parallel()
    config = _config(mtp_num_layers=2)
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)

    mtp_block_spec = get_gpt_mtp_block_spec(config, block_spec, False)

    assert len(mtp_block_spec.layer_specs) == 2
    assert all(spec.module is MultiTokenPredictionLayer for spec in mtp_block_spec.layer_specs)
    for layer_spec in mtp_block_spec.layer_specs:
        copied_layer = layer_spec.submodules.mtp_model_layer
        assert copied_layer.submodules.self_attention.module is SelfAttention
        assert copied_layer is not block_spec.layer_specs[-1]


def test_get_gpt_mtp_block_spec_accepts_single_layer_module_spec(model_parallel):
    model_parallel()
    config = _config(mtp_num_layers=1)
    layer_spec = get_gpt_layer_local_spec()

    mtp_block_spec = get_gpt_mtp_block_spec(config, layer_spec, False)

    assert len(mtp_block_spec.layer_specs) == 1
    assert mtp_block_spec.layer_specs[0].module is MultiTokenPredictionLayer


def test_get_gpt_mtp_block_spec_repeats_layer_when_mtp_use_repeated_layer(model_parallel):
    model_parallel()
    config = _config(mtp_num_layers=2, mtp_use_repeated_layer=True)
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)

    mtp_block_spec = get_gpt_mtp_block_spec(config, block_spec, False)

    assert len(mtp_block_spec.layer_specs) == 1
    layer_spec = mtp_block_spec.layer_specs[0]
    assert layer_spec.module is MultiTokenPredictionLayer
    assert layer_spec.submodules.mtp_model_layer.module is TransformerLayer
    assert layer_spec.submodules.mtp_model_layer.submodules.self_attention.module is SelfAttention


def test_get_gpt_mtp_block_spec_rejects_invalid_spec(model_parallel):
    model_parallel()
    config = _config(mtp_num_layers=1)

    with pytest.raises(ValueError, match='Invalid spec'):
        get_gpt_mtp_block_spec(config, 'not-a-spec', False)


def test_gpt_layer_specs_fall_back_when_te_kitchen_and_apex_are_unavailable(
    monkeypatch, restore_gpt_layer_specs
):
    with monkeypatch.context() as blocker:
        blocker.setattr(te_extensions, 'HAVE_TE', False)
        blocker.setitem(sys.modules, 'megatron.core.extensions.kitchen', None)
        blocker.setitem(sys.modules, 'apex', None)
        with pytest.warns(UserWarning, match='Apex is not installed. Falling back to Torch Norm'):
            importlib.reload(gpt_layer_specs)
        assert gpt_layer_specs.TEFusedMLP is None
        assert gpt_layer_specs.TESpecProvider is None
        assert gpt_layer_specs.HAVE_KITCHEN is False
        assert gpt_layer_specs.HAVE_APEX is False
        assert gpt_layer_specs.LNImpl is WrappedTorchNorm

    importlib.reload(gpt_layer_specs)
    assert gpt_layer_specs.HAVE_TE == te_extensions.HAVE_TE
    assert gpt_layer_specs.HAVE_APEX is (importlib.util.find_spec('apex') is not None)
    assert gpt_layer_specs.HAVE_KITCHEN is False
