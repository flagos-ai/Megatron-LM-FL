# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.quantization.quant_config import MatchContext, QuantizationConfig


def _quant_config(config):
    return QuantizationConfig(
        config,
        MatchContext(module_path="decoder.layers.0.mlp.linear_fc1", layer_number=0),
        "recipe",
    )


def test_te_quantization_recipe_and_params_parse_validation_paths():
    keys = te_ext.TEQuantizationRecipe.get_config_keys()
    assert "fp8_quantization_recipe" in keys
    assert "override_quantized_autocast" in keys

    recipe = te_ext.TEQuantizationRecipe.parse_from_config(
        {
            "fp8_quantization_recipe": Fp8Recipe.tensorwise,
            "fp8_format": "hybrid",
            "override_nonquantized_autocast": True,
            "tp_only_amax_red": True,
        }
    )
    assert recipe.fp8_quantization_recipe == Fp8Recipe.tensorwise
    assert recipe.fp8_format == "hybrid"
    assert recipe.tp_only_amax_red is True

    with pytest.raises(ValueError, match="not valid"):
        te_ext.TEQuantizationRecipe.parse_from_config({"unexpected": 1})
    with pytest.raises(ValueError, match="Delayed scaling"):
        te_ext.TEQuantizationRecipe.parse_from_config(
            {"fp8_quantization_recipe": Fp8Recipe.delayed}
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        te_ext.TEQuantizationRecipe.parse_from_config(
            {
                "fp8_quantization_recipe": Fp8Recipe.tensorwise,
                "fp4_quantization_recipe": Fp4Recipe.nvfp4,
            }
        )
    with pytest.raises(ValueError, match="requires custom_recipe_factory"):
        te_ext.TEQuantizationRecipe.parse_from_config(
            {"fp8_quantization_recipe": Fp8Recipe.custom}
        )

    params = te_ext.TEQuantizationParams.parse_from_config(
        _quant_config(
            {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp8_quantization_recipe": Fp8Recipe.mxfp8},
            }
        )
    )
    assert params.training_recipe.fp8_quantization_recipe == Fp8Recipe.mxfp8
    assert params.evaluation_recipe is None

    params = te_ext.TEQuantizationParams.parse_from_config(
        _quant_config(
            {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp8_quantization_recipe": Fp8Recipe.blockwise},
                "evaluation_recipe": {"fp4_quantization_recipe": Fp4Recipe.nvfp4},
            }
        )
    )
    assert params.evaluation_recipe.fp4_quantization_recipe == Fp4Recipe.nvfp4

    with pytest.raises(ValueError, match="must have 'transformer_engine_config_type'"):
        te_ext.TEQuantizationParams.parse_from_config(_quant_config({}))
    with pytest.raises(ValueError, match="Unsupported config type"):
        te_ext.TEQuantizationParams.parse_from_config(
            _quant_config({"transformer_engine_config_type": "Other"})
        )
    with pytest.raises(ValueError, match="must have 'training_recipe'"):
        te_ext.TEQuantizationParams.parse_from_config(
            _quant_config({"transformer_engine_config_type": "TEQuantizationParams"})
        )


def test_te_quantization_autocast_and_context_decision_paths(monkeypatch):
    autocast_calls = []

    class _Autocast:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Recipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _State:
        enabled = False

        @classmethod
        def is_fp8_enabled(cls):
            return cls.enabled

    def _fp8_autocast(**kwargs):
        autocast_calls.append(kwargs)
        return _Autocast(**kwargs)

    fake_recipe_mod = SimpleNamespace(
        Format=SimpleNamespace(E4M3="e4m3", HYBRID="hybrid"),
        Float8CurrentScaling=_Recipe,
        Float8BlockScaling=_Recipe,
        MXFP8BlockScaling=_Recipe,
        NVFP4BlockScaling=_Recipe,
    )
    monkeypatch.setattr(te_ext, "FP8GlobalStateManager", _State, raising=False)
    monkeypatch.setattr(te_ext, "fp8_autocast", _fp8_autocast, raising=False)
    monkeypatch.setattr(te_ext, "te", SimpleNamespace(common=SimpleNamespace(recipe=fake_recipe_mod)))
    monkeypatch.setattr(te_ext, "model_parallel_is_initialized", lambda: False)

    disabled = te_ext.TEQuantizationRecipe(override_nonquantized_autocast=True)
    ctx = te_ext._get_fp8_autocast_for_quant_recipe(disabled)
    assert isinstance(ctx, _Autocast)
    assert autocast_calls[-1] == {"enabled": False}

    no_override = te_ext.TEQuantizationRecipe(override_nonquantized_autocast=False)
    assert isinstance(te_ext._get_fp8_autocast_for_quant_recipe(no_override), nullcontext)

    _State.enabled = True
    quantized_no_override = te_ext.TEQuantizationRecipe(
        fp8_quantization_recipe=Fp8Recipe.tensorwise,
        override_quantized_autocast=False,
    )
    assert isinstance(
        te_ext._get_fp8_autocast_for_quant_recipe(quantized_no_override), nullcontext
    )

    for recipe, fmt in [
        (Fp8Recipe.tensorwise, "e4m3"),
        (Fp8Recipe.blockwise, "hybrid"),
        (Fp8Recipe.mxfp8, "e4m3"),
    ]:
        ctx = te_ext._get_fp8_autocast_for_quant_recipe(
            te_ext.TEQuantizationRecipe(fp8_quantization_recipe=recipe, fp8_format=fmt)
        )
        assert isinstance(ctx, _Autocast)
        assert autocast_calls[-1]["enabled"] is True
        assert autocast_calls[-1]["fp8_group"] is None
        assert isinstance(autocast_calls[-1]["fp8_recipe"], _Recipe)

    fp4_ctx = te_ext._get_fp8_autocast_for_quant_recipe(
        te_ext.TEQuantizationRecipe(fp4_quantization_recipe=Fp4Recipe.nvfp4)
    )
    assert isinstance(fp4_ctx, _Autocast)
    assert isinstance(autocast_calls[-1]["fp8_recipe"], _Recipe)

    with pytest.raises(ValueError, match="Unhandled fp8_format"):
        te_ext._get_fp8_autocast_for_quant_recipe(
            te_ext.TEQuantizationRecipe(
                fp8_quantization_recipe=Fp8Recipe.tensorwise, fp8_format="bad"
            )
        )
    with pytest.raises(ValueError, match="Unhandled fp8 recipe"):
        te_ext._get_fp8_autocast_for_quant_recipe(
            te_ext.TEQuantizationRecipe(fp8_quantization_recipe="bad")
        )
    with pytest.raises(ValueError, match="Unhandled fp4 recipe"):
        te_ext._get_fp8_autocast_for_quant_recipe(
            te_ext.TEQuantizationRecipe(fp4_quantization_recipe="bad")
        )

    training = te_ext.TEQuantizationRecipe(fp8_quantization_recipe=Fp8Recipe.tensorwise)
    evaluation = te_ext.TEQuantizationRecipe(fp4_quantization_recipe=Fp4Recipe.nvfp4)
    params = te_ext.TEQuantizationParams(training_recipe=training, evaluation_recipe=evaluation)
    assert isinstance(te_ext._get_fp8_autocast_for_quant_params(None, True), nullcontext)
    assert te_ext._get_fp8_autocast_for_quant_params(params, True).kwargs["enabled"] is True
    assert te_ext._get_fp8_autocast_for_quant_params(params, False).kwargs["enabled"] is True

    assert te_ext._get_should_context_be_quantized_recipe(disabled, True) is False
    assert te_ext._get_should_context_be_quantized_recipe(no_override, False) is False
    assert (
        te_ext._get_should_context_be_quantized_recipe(
            te_ext.TEQuantizationRecipe(
                fp8_quantization_recipe=Fp8Recipe.tensorwise,
                override_nonquantized_autocast=True,
            ),
            False,
        )
        is True
    )
    assert te_ext._get_should_context_be_quantized_params(None, True, True) is True
    assert te_ext._get_should_context_be_quantized_params(params, False, False) is True


def test_te_extra_kwargs_condition_init_and_import_error_paths(monkeypatch):
    monkeypatch.setattr(te_ext, "is_te_min_version", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        te_ext,
        "cur_platform",
        SimpleNamespace(current_device=lambda: "accelerator"),
    )

    config = SimpleNamespace(
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        init_model_with_meta_device=False,
        perform_initialization=True,
        fused_residual_rmsnorm=False,
        normalization="LayerNorm",
        sequence_parallel=False,
        layernorm_zero_centered_gamma=False,
    )
    assert te_ext._get_extra_te_kwargs(config) == {
        "params_dtype": torch.bfloat16,
        "device": "cpu",
    }
    config.use_cpu_initialization = False
    config.init_model_with_meta_device = True
    assert te_ext._get_extra_te_kwargs(config)["device"] == "meta"
    config.init_model_with_meta_device = False
    assert te_ext._get_extra_te_kwargs(config)["device"] == "accelerator"

    weight = torch.empty(2)
    init = lambda tensor: tensor.fill_(3.0)
    te_ext.condition_init_method(config, init)(weight)
    torch.testing.assert_close(weight, torch.full_like(weight, 3.0))
    config.perform_initialization = False
    te_ext.condition_init_method(config, init)(weight.fill_(0.0))
    torch.testing.assert_close(weight, torch.zeros_like(weight))

    monkeypatch.setattr(te_ext, "HAVE_TE", False)
    with pytest.raises(ImportError, match="Transformer Engine is not installed"):
        te_ext.TENorm(config, hidden_size=4)
