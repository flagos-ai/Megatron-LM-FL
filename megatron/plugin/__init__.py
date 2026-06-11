# Load the centralized override registry so that all lazy mappings are
# available before any @overridable function is called.
import megatron.plugin.override_registry  # noqa: F401
