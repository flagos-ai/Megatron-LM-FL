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

# =============================================================================
# Transformer - multi_token_prediction
# =============================================================================
register(
    target="megatron.core.transformer.multi_token_prediction.reduce_loss_in_tracker",
    impl="megatron.plugin.kunlunxin.transformer.multi_token_prediction.reduce_loss_in_tracker",
    vendor="kunlunxin",
)

# =============================================================================
# Transformer - moe.fused_a2a
# =============================================================================
register(
    target="megatron.core.transformer.moe.fused_a2a.get_buffer",
    impl="megatron.plugin.kunlunxin.transformer.moe.fused_a2a.get_buffer",
    vendor="kunlunxin",
)
register(
    target="megatron.core.transformer.moe.fused_a2a.FusedDispatch",
    impl="megatron.plugin.kunlunxin.transformer.moe.fused_a2a.FusedDispatchKunlunxin",
    vendor="kunlunxin",
)
register(
    target="megatron.core.transformer.moe.fused_a2a.FusedCombine",
    impl="megatron.plugin.kunlunxin.transformer.moe.fused_a2a.FusedCombineKunlunxin",
    vendor="kunlunxin",
)
register(
    target="megatron.core.transformer.moe.fused_a2a.fused_dispatch",
    impl="megatron.plugin.kunlunxin.transformer.moe.fused_a2a.fused_dispatch",
    vendor="kunlunxin",
)
register(
    target="megatron.core.transformer.moe.fused_a2a.fused_combine",
    impl="megatron.plugin.kunlunxin.transformer.moe.fused_a2a.fused_combine",
    vendor="kunlunxin",
)

# =============================================================================
# Tensor parallel - random
# =============================================================================
register(
    target="megatron.core.tensor_parallel.random.CudaRNGStatesTracker",
    impl="megatron.plugin.kunlunxin.tensor_parallel.random.CudaRNGStatesTrackerKunlunxin",
    vendor="kunlunxin",
)

# =============================================================================
# Dist checkpointing - filesystem_async
# =============================================================================
register(
    target="megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.prepare_write_data",
    impl="megatron.plugin.kunlunxin.dist_checkpointing.strategies.filesystem_async.prepare_write_data",
    vendor="kunlunxin",
)
register(
    target="megatron.core.dist_checkpointing.strategies.filesystem_async.preload_tensors",
    impl="megatron.plugin.kunlunxin.dist_checkpointing.strategies.filesystem_async.preload_tensors_kunlunxin",
    vendor="kunlunxin",
)
register(
    target="megatron.core.dist_checkpointing.strategies.filesystem_async.write_preloaded_data",
    impl="megatron.plugin.kunlunxin.dist_checkpointing.strategies.filesystem_async.write_preloaded_data",
    vendor="kunlunxin",
)

register(
    target="megatron.core.tensor_parallel.random._set_cuda_rng_state",
    impl="megatron.plugin.Ascend.tensor_parallel.random._set_cuda_rng_state",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax",
    impl="megatron.plugin.Ascend.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax",
    impl="megatron.plugin.Ascend.fusions.fused_softmax.ScaledMaskedSoftmax",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.ScaledSoftmax",
    impl="megatron.plugin.Ascend.fusions.fused_softmax.ScaledSoftmax",
    vendor="npu",
)

register(
    target="megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available",
    impl="megatron.plugin.Ascend.fusions.fused_softmax.is_kernel_available",
    vendor="npu",
)

register(
    target="megatron.core.fp8_utils.get_fp8_recipe",
    impl="megatron.plugin.Ascend.fp8_utils.get_fp8_recipe",
    vendor="npu",
)

register(
    target="megatron.core.transformer.transformer_config.TransformerConfig",
    impl="megatron.plugin.Ascend.transformer.transformer_config.NPUTransformerConfig",
    vendor="npu",
)
