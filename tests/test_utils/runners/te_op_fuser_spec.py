"""Transformer Engine op-fuser spec for the controlled MegaLens GPU profile."""

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)


te_op_fuser_spec = get_gpt_layer_with_transformer_engine_spec(
    num_experts=None,
    qk_layernorm=False,
    multi_latent_attention=False,
    use_te_op_fuser=True,
    use_kitchen=False,
)
