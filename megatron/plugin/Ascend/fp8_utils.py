from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TEDelayedScaling

from megatron.core.utils import is_te_min_version
from megatron.plugin.utils.parser_utils import add_field_literal_choice
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

def get_fp8_recipe(config: TransformerConfig):
    """Return fp8 recipe.

    Arguments:
        config (TransformerConfig): Configuration object.

    Returns:
        FP8 recipe.
    """
    if not HAVE_TE:
        return None

    if config.fp8 == "e4m3":
        fp8_format = transformer_engine.common.recipe.Format.E4M3
    elif config.fp8 == "hybrid":
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
    elif config.fp8 == "hif8":
        fp8_format = transformer_engine.common.recipe.Format.HIF8
    else:
        raise ValueError("E4M3, HYBRID and HIF8 are the only supported FP8 formats.")

    # Select fp8 recipe (TE version >= 2.1.0).
    fp8_recipe = None
    if is_te_min_version("2.1.0"):
        if config.fp8_recipe == Fp8Recipe.delayed:
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version(
            "2.2.0.dev0"
        ):
            fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
            )
        elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version(
            "2.3.0.dev0"
        ):
            fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.mxfp8:
            fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.custom:
            assert config.fp8_quantizer_factory is not None
            from megatron.core.fp8_utils import _get_custom_recipe

            fp8_recipe = _get_custom_recipe(config.fp8_quantizer_factory)
        else:
            raise ValueError(
                "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                "you are using a compatible TE version."
            )
    else:
        # Assert that the user is using delayed scaling.
        assert config.fp8_recipe == Fp8Recipe.delayed, (
            "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
            "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
            "Float8BlockScaling."
        )
        fp8_recipe = TEDelayedScaling(
            config=config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, not config.fp8_wgrad),
        )
    return fp8_recipe