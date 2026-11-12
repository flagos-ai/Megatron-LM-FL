# Copyright (c) 2026, BAAI. All rights reserved.
"""Multimodal projector: unknown projector types rejected; MLP projection adds the encoder bias to vision features."""
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestMultimodalProjectorUnsupportedType:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        self.mlp_layer_spec = get_mlp_module_spec(use_te=False).keywords['submodules']

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor_rejects_unknown_projector_type(self):
        with pytest.raises(Exception, match="Unsupported multimodal projection type conv"):
            MultimodalProjector(
                config=self.transformer_config,
                submodules=self.mlp_layer_spec,
                projector_type="conv",
                input_size=64,
            )

    def test_mlp_projector_maps_input_to_language_hidden_size(self):
        projector = MultimodalProjector(
            config=self.transformer_config,
            submodules=self.mlp_layer_spec,
            projector_type="mlp",
            input_size=32,
        )
        projector.to(torch.cuda.current_device())
        hidden_states = torch.randn(4, 32, device=torch.cuda.current_device())

        projected = projector(hidden_states)

        encoder_output, encoder_output_bias = projector.encoder(hidden_states)
        expected = encoder_output + encoder_output_bias
        assert projected.shape == torch.Size([4, self.transformer_config.hidden_size])
        assert projected.dtype == torch.float32
        torch.testing.assert_close(projected, expected, rtol=1e-5, atol=1e-5)

        # Independent MLP computation from the module weights: the fc1 bias
        # is added before the activation and the fc2 bias after the projection
        # (the zero-initialized fc2 bias made the old inequality assert vacuous).
        linear_fc1, linear_fc2 = projector.encoder.linear_fc1, projector.encoder.linear_fc2
        intermediate = (
            torch.nn.functional.linear(hidden_states, linear_fc1.weight) + linear_fc1.bias
        )
        activated = self.transformer_config.activation_func(intermediate)
        weights_expected = (
            torch.nn.functional.linear(activated, linear_fc2.weight) + linear_fc2.bias
        )
        torch.testing.assert_close(projected, weights_expected, rtol=1e-5, atol=1e-5)
