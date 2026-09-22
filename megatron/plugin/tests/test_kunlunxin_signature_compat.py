"""Tests for KunLunXin override signature compatibility with current ME-FL callers."""

import inspect
import sys
import types
import unittest
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.plugin.kunlunxin.fusions.fused_bias_swiglu import (
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
)
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


@pytest.mark.parametrize("shape,with_bias", [((2, 8), False), ((2, 3, 8), True)])
@pytest.mark.parametrize("clamp_value", [None, 0.0, -1.0])
def test_bias_swiglu_five_arguments_keep_vendor_path(shape, with_bias, clamp_value):
    source = torch.randn(shape)
    bias = torch.randn(shape[-1]) if with_bias else None
    vendor = types.ModuleType("torch_xmlir.nn.swiglu")
    vendor.SwiGLUFunction = mock.Mock()
    vendor.SwiGLUFunction.apply.side_effect = (
        lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1]
    )
    with mock.patch.dict(sys.modules, {"torch_xmlir.nn.swiglu": vendor}):
        output = bias_swiglu_impl(source, bias, False, False, clamp_value)
    expected_input = source.view(-1, shape[-1])
    if bias is not None:
        expected_input = expected_input + bias
    vendor.SwiGLUFunction.apply.assert_called_once()
    torch.testing.assert_close(vendor.SwiGLUFunction.apply.call_args.args[0], expected_input)
    gate, linear = expected_input.chunk(2, -1)
    torch.testing.assert_close(output, (F.silu(gate) * linear).view(*shape[:-1], shape[-1] // 2))


def test_bias_swiglu_override_signature_matches_core():
    from megatron.core.fusions import fused_bias_swiglu as core

    assert inspect.signature(bias_swiglu_impl) == inspect.signature(core.bias_swiglu_impl)


@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_clamp_forward_and_backward(with_bias):
    # Exercise saturated and unsaturated values, avoiding clamp boundaries.
    source = torch.tensor(
        [[-2.0, -0.2, 0.2, 2.0, -2.0, -0.2, 0.2, 2.0]], requires_grad=True
    )
    reference = source.detach().clone().requires_grad_(True)
    bias = torch.full((8,), 0.1, requires_grad=True) if with_bias else None
    ref_bias = bias.detach().clone().requires_grad_(True) if with_bias else None
    if with_bias:
        output = bias_swiglu_impl(source, bias, False, False, 0.5)
    else:
        output = bias_swiglu_impl(source, None, clamp_value=0.5)
    ref_input = reference + ref_bias if with_bias else reference
    gate, linear = ref_input.chunk(2, -1)
    expected = F.silu(gate.clamp(max=0.5)) * linear.clamp(-0.5, 0.5)
    torch.testing.assert_close(output, expected)
    grad = torch.linspace(-1.0, 1.0, output.numel()).view_as(output)
    output.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(source.grad, reference.grad)
    if with_bias:
        torch.testing.assert_close(bias.grad, ref_bias.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
