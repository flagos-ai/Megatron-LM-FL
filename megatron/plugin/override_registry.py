"""
Centralized override registry for FlagScale plugin system.

All override mappings are declared here using :func:`register`. The plugin
implementation modules are lazily imported only when the corresponding
``@overridable`` function is first called at runtime.

To add a new override, simply add a ``register(...)`` call below.
The ``@override`` decorator on the implementation function is no longer needed.
"""

from megatron.plugin.decorators import register


# =============================================================================
# Optimizer - clip_grads
# =============================================================================
register(
    target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
    impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32",
)

register(
    target="megatron.core.tensor_parallel.random._set_cuda_rng_state",
    impl="megatron.plugin.tensor_parallel.random._set_cuda_rng_state",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax",
    impl="megatron.plugin.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax",
    impl="megatron.plugin.fusions.fused_softmax.ScaledMaskedSoftmax",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.ScaledSoftmax",
    impl="megatron.plugin.fusions.fused_softmax.ScaledSoftmax",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available",
    impl="megatron.plugin.fusions.fused_softmax.is_kernel_available",
    vendor="npu",
)

register(
    target="megatron.core.fp8_utils.get_fp8_recipe",
    impl="megatron.plugin.fp8_utils.get_fp8_recipe",
    vendor="npu",
)

register(
    target="megatron.core.transformer.transformer_config.TransformerConfig",
    impl="megatron.plugin.transformer.transformer_config.NPUTransformerConfig",
    vendor="npu",
)
