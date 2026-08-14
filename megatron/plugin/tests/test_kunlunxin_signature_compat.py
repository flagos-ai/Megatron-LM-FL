"""Tests for KunLunXin override signature compatibility with current ME-FL callers."""

import unittest
from unittest import mock

import torch

from megatron.plugin.kunlunxin.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.plugin.kunlunxin.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd


class TestKunLunXinSignatureCompatibility(unittest.TestCase):
    """Validate override implementations accept current upstream caller arguments."""

    def test_rope_accepts_current_megatron_mla_keywords(self):
        """Accept MLA keyword arguments passed by apply_rotary_pos_emb."""
        t = torch.randn(2, 1, 1, 4)
        freqs = torch.randn(2, 1, 1, 4)

        output = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=False,
            mla_rotary_interleaved=True,
            mscale=1.0,
            inverse=False,
            mla_output_remove_interleaving=True,
        )

        self.assertEqual(output.shape, t.shape)

    def test_weighted_swiglu_accepts_clamp_value_positional_argument(self):
        """Accept clamp_value passed positionally by the MoE experts path."""
        input_tensor = torch.randn(2, 8)
        weights = torch.randn(2, 1)

        with mock.patch(
            "megatron.plugin.kunlunxin.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=torch.randn(2, 4),
        ) as apply_mock:
            output = weighted_bias_swiglu_impl(input_tensor, None, weights, False, 0.5)

        self.assertEqual(output.shape, (2, 4))
        apply_mock.assert_called_once()
        args, kwargs = apply_mock.call_args
        self.assertEqual(args[0].shape, input_tensor.shape)
        self.assertTrue(torch.equal(args[0], input_tensor))
        self.assertIs(args[1], weights)
        self.assertEqual(kwargs, {})

    def test_weighted_swiglu_accepts_clamp_value_keyword_argument(self):
        """Accept clamp_value passed as a keyword argument by future callers."""
        input_tensor = torch.randn(2, 8)
        weights = torch.randn(2, 1)

        with mock.patch(
            "megatron.plugin.kunlunxin.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=torch.randn(2, 4),
        ) as apply_mock:
            output = weighted_bias_swiglu_impl(
                input_tensor,
                None,
                weights,
                fp8_input_store=False,
                clamp_value=0.5,
            )

        self.assertEqual(output.shape, (2, 4))
        apply_mock.assert_called_once()
        args, kwargs = apply_mock.call_args
        self.assertEqual(args[0].shape, input_tensor.shape)
        self.assertTrue(torch.equal(args[0], input_tensor))
        self.assertIs(args[1], weights)
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
