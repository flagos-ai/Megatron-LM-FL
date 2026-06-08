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