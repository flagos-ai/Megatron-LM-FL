# Copyright (c) 2026, BAAI. All rights reserved.
"""Backward-compatible re-exports in the ModelOpt mamba model_specs module."""

from megatron.core.post_training.modelopt.hybrid import model_specs as hybrid_model_specs
from megatron.core.post_training.modelopt.mamba import model_specs as mamba_model_specs


def test_mamba_model_specs_reexports_the_hybrid_spec_builders():
    assert mamba_model_specs.get_hybrid_stack_modelopt_spec is (
        hybrid_model_specs.get_hybrid_stack_modelopt_spec
    )
    assert mamba_model_specs.get_mamba_stack_modelopt_spec is (
        hybrid_model_specs.get_mamba_stack_modelopt_spec
    )
