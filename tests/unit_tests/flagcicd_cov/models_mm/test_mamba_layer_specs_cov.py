# Copyright (c) 2026, BAAI. All rights reserved.
"""Mamba layer specs re-export the hybrid stack specs unchanged for backward compatibility."""
import megatron.core.models.hybrid.hybrid_layer_specs as hybrid_layer_specs
import megatron.core.models.mamba.mamba_layer_specs as mamba_layer_specs


def test_mamba_layer_specs_reexports_hybrid_stack_specs():
    assert mamba_layer_specs.hybrid_stack_spec is hybrid_layer_specs.hybrid_stack_spec
    assert (
        mamba_layer_specs.hybrid_inference_stack_spec
        is hybrid_layer_specs.hybrid_inference_stack_spec
    )
    assert mamba_layer_specs.mamba_stack_spec is hybrid_layer_specs.mamba_stack_spec
    assert (
        mamba_layer_specs.mamba_inference_stack_spec
        is hybrid_layer_specs.mamba_inference_stack_spec
    )
    assert mamba_layer_specs.mamba_stack_spec.module is hybrid_layer_specs.HybridStack
