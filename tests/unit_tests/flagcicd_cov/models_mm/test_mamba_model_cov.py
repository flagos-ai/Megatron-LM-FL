# Copyright (c) 2026, BAAI. All rights reserved.
"""MambaModel forwards the deprecated mamba_stack_spec argument to HybridModel construction."""
import pytest

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _mamba_config():
    return TransformerConfig(
        num_layers=2, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
    )


class TestMambaModelDeprecatedSpec:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_mamba_stack_spec_is_forwarded_as_hybrid_stack_spec(self):
        model = MambaModel(
            config=_mamba_config(),
            mamba_stack_spec=hybrid_stack_spec,
            vocab_size=96,
            max_sequence_length=8,
            hybrid_layer_pattern='M*',
        )

        assert model.hybrid_stack_spec is hybrid_stack_spec
        assert model.hybrid_layer_pattern == 'M*'
        assert [type(layer).__name__ for layer in model.decoder.layers] == [
            'MambaLayer',
            'TransformerLayer',
        ]
        assert model.decoder.layers[0].layer_number == 1
        assert model.decoder.layers[1].layer_number == 2

    @pytest.mark.parametrize(
        "extra_kwargs",
        [{'hybrid_stack_spec': hybrid_stack_spec}, {}],
        ids=["hybrid_stack_spec_kwarg", "positional_hybrid_stack_spec"],
    )
    def test_mamba_stack_spec_conflicting_with_hybrid_stack_spec_raises(self, extra_kwargs):
        with pytest.raises(
            ValueError, match="Cannot specify both hybrid_stack_spec and mamba_stack_spec"
        ):
            MambaModel(
                _mamba_config(),
                hybrid_stack_spec,
                mamba_stack_spec=hybrid_stack_spec,
                max_sequence_length=8,
                hybrid_layer_pattern='M*',
                **extra_kwargs,
            )
