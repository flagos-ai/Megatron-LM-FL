# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import argparse
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.moe.experts as experts_module
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import _add_network_size_args, parse_args
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def test_op_fuser_transformer_config_args_are_exposed():
    parser = argparse.ArgumentParser()
    _add_network_size_args(parser)

    args = parser.parse_args(
        ["--use-transformer-engine-op-fuser", "--moe-mlp-glu-interleave-size", "16"]
    )

    assert args.use_transformer_engine_op_fuser is True
    assert args.moe_mlp_glu_interleave_size == 16


def test_remove_glu_interleaving_restores_contiguous_gate_and_linear_halves():
    interleaved = torch.tensor([[1, 2, 5, 6, 3, 4, 7, 8], [11, 12, 15, 16, 13, 14, 17, 18]])
    expected = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18]])

    output = TEGroupedMLP._remove_glu_interleaving(interleaved, interleave_size=2)

    torch.testing.assert_close(output, expected)


def test_make_fused_ops_reuses_grouped_linear_weights_on_meta_device(monkeypatch):
    class FakeGroupedLinear(torch.nn.Module):
        def __init__(
            self,
            num_gemms,
            in_features,
            out_features,
            *,
            bias,
            device,
            dtype,
            accumulate_into_main_grad,
            single_grouped_weight,
            single_grouped_bias=False,
            delay_wgrad_compute=False,
        ):
            super().__init__()
            self.num_gemms = num_gemms
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            self.device = device
            self.dtype = dtype
            self.fuse_wgrad_accumulation = accumulate_into_main_grad
            self.single_grouped_weight = single_grouped_weight
            self.single_grouped_bias = single_grouped_bias
            self.delay_wgrad_compute = delay_wgrad_compute

        def need_backward_dw(self):
            return False

    class FakeScaledSwiGLU(torch.nn.Module):
        def __init__(self, glu_interleave_size, *, activation_recompute_in_mlp=False):
            super().__init__()
            self.glu_interleave_size = glu_interleave_size
            self.activation_recompute_in_mlp = activation_recompute_in_mlp

    class FakeSequential(list):
        def register_forward_pre_hook(self, hook):
            self.forward_pre_hook = hook

    fake_te = SimpleNamespace(
        pytorch=SimpleNamespace(
            GroupedLinear=FakeGroupedLinear,
            ops=SimpleNamespace(
                GroupedLinear=FakeGroupedLinear,
                ScaledSwiGLU=FakeScaledSwiGLU,
                Sequential=FakeSequential,
            ),
        )
    )
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        moe_mlp_glu_interleave_size=16,
        delay_wgrad_compute=False,
        activation_func_clamp_value=None,
        activation_func=F.silu,
    )
    module.activation_func = F.silu
    module.config.gated_linear_unit = True
    module.linear_fc1 = FakeGroupedLinear(
        2,
        4,
        8,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=True,
        single_grouped_weight=False,
    )
    module.linear_fc2 = FakeGroupedLinear(
        2,
        8,
        4,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=True,
    )
    module.linear_fc1.weight0 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.weight1 = torch.nn.Parameter(torch.ones(8, 4) * 2)
    module.linear_fc1.bias0 = torch.nn.Parameter(torch.zeros(8))
    module.linear_fc1.bias1 = torch.nn.Parameter(torch.ones(8))
    module.linear_fc2.weight = torch.nn.Parameter(torch.ones(4, 8))

    ops = module._make_fused_ops()

    assert len(ops) == 3
    assert ops[0].device == "meta"
    assert ops[0].weight0 is module.linear_fc1.weight0
    assert ops[0].weight1 is module.linear_fc1.weight1
    assert ops[0].bias0 is module.linear_fc1.bias0
    assert ops[0].bias1 is module.linear_fc1.bias1
    assert ops[1].glu_interleave_size == 16
    assert ops[2].device == "meta"
    assert ops[2].weight is module.linear_fc2.weight
    assert hasattr(ops, "forward_pre_hook")


def test_fused_forward_caches_ops_and_forwards_expected_arguments():
    class FakeFusedOps:
        def __call__(self, hidden_states, fc1_tokens, probs, fc2_tokens):
            self.args = (hidden_states, fc1_tokens, probs, fc2_tokens)
            return hidden_states + 1

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    # `_fused_forward` calls `skip_routed_expert_padding(config)` (added by PR 4071), which
    # reads `moe_token_dispatcher_type` and `moe_flex_dispatcher_backend` after the
    # `moe_router_padding_for_quantization` short-circuit fails.
    module.config = SimpleNamespace(
        fp8=False,
        fp4=False,
        moe_router_padding_for_quantization=False,
        moe_token_dispatcher_type=None,
        moe_flex_dispatcher_backend=None,
        moe_paged_stash=False,
    )
    module._fused_ops = None
    fused_ops = FakeFusedOps()
    module._make_fused_ops = lambda: fused_ops
    hidden_states = torch.zeros(2, 4)
    tokens_per_expert = torch.tensor([1, 1])
    probs = torch.ones(2)

    output = module._fused_forward(hidden_states, tokens_per_expert, probs)

    torch.testing.assert_close(output, torch.ones_like(hidden_states))
    assert module._fused_ops[0] is fused_ops
    assert fused_ops.args[0] is hidden_states
    assert fused_ops.args[1] is tokens_per_expert
    assert fused_ops.args[2] is probs
    assert fused_ops.args[3] is tokens_per_expert


def test_apply_bias_returns_input_unchanged_when_bias_is_none():
    intermediate = torch.arange(6, dtype=torch.float32).view(3, 2)

    output = TEGroupedMLP._apply_bias(
        intermediate, bias_parallel=None, tokens_per_expert=[2, 1], permuted_probs=torch.ones(3)
    )

    assert output is intermediate


def test_apply_bias_combines_per_expert_bias_and_probs():
    intermediate = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    bias_parallel = [torch.tensor([10.0, 20.0]), torch.tensor([100.0, 200.0])]
    tokens_per_expert = [2, 1]
    permuted_probs = torch.tensor([0.5, 0.5, 1.0])
    expected = torch.tensor([[6.0, 12.0], [8.0, 14.0], [105.0, 206.0]], dtype=torch.float32)

    output = TEGroupedMLP._apply_bias(
        intermediate, bias_parallel, tokens_per_expert, permuted_probs
    )

    torch.testing.assert_close(output, expected)
    assert output.dtype == intermediate.dtype


def test_make_fused_impl_pre_forward_hook_dispatches_submodule_hooks():
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    fc1_child = torch.nn.Linear(2, 2)
    fc2_child = torch.nn.Linear(2, 2)
    module.linear_fc1 = torch.nn.Sequential(fc1_child)
    module.linear_fc2 = torch.nn.Sequential(fc2_child)

    calls = []

    def fc1_hook(submodule, _inp):
        calls.append(("fc1", submodule))
        return None

    def fc2_hook(submodule, _inp):
        calls.append(("fc2", submodule))
        return None

    fc1_child.register_forward_pre_hook(fc1_hook)
    fc2_child.register_forward_pre_hook(fc2_hook)

    hook = module._make_fused_impl_pre_forward_hook()
    hook(object())

    visited = {label for label, _ in calls}
    assert visited == {"fc1", "fc2"}


def test_make_fused_impl_pre_forward_hook_rejects_input_modifying_hook():
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    fc1_child = torch.nn.Linear(2, 2)
    module.linear_fc1 = torch.nn.Sequential(fc1_child)
    module.linear_fc2 = torch.nn.Sequential(torch.nn.Linear(2, 2))

    fc1_child.register_forward_pre_hook(lambda submodule, _inp: torch.zeros(1))

    hook = module._make_fused_impl_pre_forward_hook()

    with pytest.raises(RuntimeError, match="modifies the input tensor"):
        hook(object())


def test_make_fused_ops_handles_single_grouped_weight_for_fc1(monkeypatch):
    class FakeGroupedLinear(torch.nn.Module):
        def __init__(
            self,
            num_gemms,
            in_features,
            out_features,
            *,
            bias,
            device,
            dtype,
            accumulate_into_main_grad,
            single_grouped_weight,
            single_grouped_bias=False,
            delay_wgrad_compute=False,
        ):
            super().__init__()
            self.num_gemms = num_gemms
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            self.device = device
            self.dtype = dtype
            self.fuse_wgrad_accumulation = accumulate_into_main_grad
            self.single_grouped_weight = single_grouped_weight
            self.single_grouped_bias = single_grouped_bias
            self.delay_wgrad_compute = delay_wgrad_compute

        def need_backward_dw(self):
            return False

    class FakeScaledSwiGLU(torch.nn.Module):
        def __init__(self, glu_interleave_size, *, activation_recompute_in_mlp=False):
            super().__init__()
            self.glu_interleave_size = glu_interleave_size
            self.activation_recompute_in_mlp = activation_recompute_in_mlp

    class FakeSequential(list):
        def register_forward_pre_hook(self, hook):
            self.forward_pre_hook = hook

    fake_te = SimpleNamespace(
        pytorch=SimpleNamespace(
            GroupedLinear=FakeGroupedLinear,
            ops=SimpleNamespace(
                GroupedLinear=FakeGroupedLinear,
                ScaledSwiGLU=FakeScaledSwiGLU,
                Sequential=FakeSequential,
            ),
        )
    )
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        moe_mlp_glu_interleave_size=8,
        delay_wgrad_compute=False,
        activation_func_clamp_value=None,
        activation_func=F.silu,
        gated_linear_unit=True,
    )
    module.activation_func = F.silu
    module.activation_recompute = True
    module.linear_fc1 = FakeGroupedLinear(
        2,
        4,
        8,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=True,
    )
    module.linear_fc2 = FakeGroupedLinear(
        2,
        8,
        4,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=True,
        single_grouped_weight=False,
    )
    module.linear_fc1.weight = torch.nn.Parameter(torch.ones(2, 8, 4))
    module.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(4, 8))
    module.linear_fc2.weight1 = torch.nn.Parameter(torch.ones(4, 8) * 2)
    module.linear_fc2.bias0 = torch.nn.Parameter(torch.zeros(4))
    module.linear_fc2.bias1 = torch.nn.Parameter(torch.ones(4))

    ops = module._make_fused_ops()

    assert ops[0].weight is module.linear_fc1.weight
    assert ops[1].glu_interleave_size == 8
    assert ops[1].activation_recompute_in_mlp is True
    assert ops[2].weight0 is module.linear_fc2.weight0
    assert ops[2].weight1 is module.linear_fc2.weight1
    assert ops[2].bias0 is module.linear_fc2.bias0
    assert ops[2].bias1 is module.linear_fc2.bias1


def _make_fake_te_namespace():
    """Build a fake TE namespace with the activation classes _make_fused_ops uses."""

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(
            self,
            num_gemms,
            in_features,
            out_features,
            *,
            bias,
            device,
            dtype,
            accumulate_into_main_grad,
            single_grouped_weight,
            single_grouped_bias=False,
            delay_wgrad_compute=False,
        ):
            super().__init__()
            self.num_gemms = num_gemms
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            self.device = device
            self.dtype = dtype
            self.fuse_wgrad_accumulation = accumulate_into_main_grad
            self.single_grouped_weight = single_grouped_weight
            self.single_grouped_bias = single_grouped_bias
            self.delay_wgrad_compute = delay_wgrad_compute

        def need_backward_dw(self):
            return False

    class FakeScaledSwiGLU(torch.nn.Module):
        def __init__(self, glu_interleave_size, *, activation_recompute_in_mlp=False):
            super().__init__()
            self.glu_interleave_size = glu_interleave_size
            self.activation_recompute_in_mlp = activation_recompute_in_mlp

    class FakeScaledClampedQGeGLU(torch.nn.Module):
        def __init__(self, glu_interleave_size, *, activation_recompute_in_mlp=False, limit=None):
            super().__init__()
            self.glu_interleave_size = glu_interleave_size
            self.activation_recompute_in_mlp = activation_recompute_in_mlp
            self.limit = limit

    class FakeScaledSReLU(torch.nn.Module):
        def __init__(self, *, activation_recompute_in_mlp=False):
            super().__init__()
            self.activation_recompute_in_mlp = activation_recompute_in_mlp

    class FakeSequential(list):
        def register_forward_pre_hook(self, hook):
            self.forward_pre_hook = hook

    return (
        SimpleNamespace(
            pytorch=SimpleNamespace(
                GroupedLinear=FakeGroupedLinear,
                ops=SimpleNamespace(
                    GroupedLinear=FakeGroupedLinear,
                    ScaledSwiGLU=FakeScaledSwiGLU,
                    ScaledClampedQGeGLU=FakeScaledClampedQGeGLU,
                    ScaledSReLU=FakeScaledSReLU,
                    Sequential=FakeSequential,
                ),
            )
        ),
        FakeGroupedLinear,
    )


def test_make_fused_ops_uses_clamped_qgeglu_for_quick_gelu(monkeypatch):
    """quick_gelu + clamp value → ScaledClampedQGeGLU(limit=clamp)."""
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        moe_mlp_glu_interleave_size=4,
        delay_wgrad_compute=False,
        activation_func_clamp_value=7.0,
        activation_func=quick_gelu,
        gated_linear_unit=True,
    )
    module.activation_func = quick_gelu
    module.activation_recompute = True
    common = dict(
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=False,
    )
    module.linear_fc1 = FakeGroupedLinear(2, 4, 8, bias=False, **common)
    module.linear_fc2 = FakeGroupedLinear(2, 8, 4, bias=False, **common)
    module.linear_fc1.weight0 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.weight1 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(4, 8))
    module.linear_fc2.weight1 = torch.nn.Parameter(torch.ones(4, 8))

    ops = module._make_fused_ops()

    activation = ops[1]
    assert type(activation).__name__ == "FakeScaledClampedQGeGLU"
    assert activation.glu_interleave_size == 4
    assert activation.activation_recompute_in_mlp is True
    assert activation.limit == 7.0


def test_make_fused_ops_uses_scaled_srelu_for_weighted_squared_relu(monkeypatch):
    """squared_relu + weighted squared-ReLU fusion -> ScaledSReLU."""
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        moe_mlp_glu_interleave_size=None,
        delay_wgrad_compute=False,
        activation_func_clamp_value=None,
        activation_func=squared_relu,
        gated_linear_unit=False,
        use_fused_weighted_squared_relu=True,
    )
    module.activation_func = squared_relu
    module.activation_recompute = True
    common = dict(
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=False,
    )
    module.linear_fc1 = FakeGroupedLinear(2, 4, 8, bias=False, **common)
    module.linear_fc2 = FakeGroupedLinear(2, 8, 4, bias=False, **common)
    module.linear_fc1.weight0 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.weight1 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(4, 8))
    module.linear_fc2.weight1 = torch.nn.Parameter(torch.ones(4, 8))

    ops = module._make_fused_ops()

    assert type(ops[1]).__name__ == "FakeScaledSReLU"
    assert ops[1].activation_recompute_in_mlp is True


def test_make_fused_ops_rejects_scaled_srelu_with_gated_linear_unit(monkeypatch):
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        moe_mlp_glu_interleave_size=None,
        delay_wgrad_compute=False,
        activation_func_clamp_value=None,
        activation_func=squared_relu,
        gated_linear_unit=True,
        use_fused_weighted_squared_relu=True,
    )
    module.activation_func = squared_relu
    common = dict(
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=False,
    )
    module.linear_fc1 = FakeGroupedLinear(2, 4, 8, bias=False, **common)
    module.linear_fc2 = FakeGroupedLinear(2, 8, 4, bias=False, **common)
    module.linear_fc1.weight0 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.weight1 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(4, 8))
    module.linear_fc2.weight1 = torch.nn.Parameter(torch.ones(4, 8))

    with pytest.raises(RuntimeError, match="expected SwiGLU, quick_gelu"):
        module._make_fused_ops()


def _install_fake_te_ops_modules(
    monkeypatch, fake_te, *, include_clamped_qgeglu=True, include_scaled_srelu=True
):
    transformer_engine = ModuleType("transformer_engine")
    pytorch = ModuleType("transformer_engine.pytorch")
    ops = ModuleType("transformer_engine.pytorch.ops")
    ops.GroupedLinear = fake_te.pytorch.GroupedLinear
    ops.ScaledSwiGLU = fake_te.pytorch.ops.ScaledSwiGLU
    if include_clamped_qgeglu:
        ops.ScaledClampedQGeGLU = fake_te.pytorch.ops.ScaledClampedQGeGLU
    if include_scaled_srelu:
        ops.ScaledSReLU = fake_te.pytorch.ops.ScaledSReLU
    pytorch.ops = ops
    transformer_engine.pytorch = pytorch
    monkeypatch.setitem(sys.modules, "transformer_engine", transformer_engine)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.ops", ops)


def _make_fused_impl_support_module(
    FakeGroupedLinear, *, activation_func, gated_linear_unit, use_fused_weighted_squared_relu=False
):
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        activation_func=activation_func,
        gated_linear_unit=gated_linear_unit,
        use_fused_weighted_squared_relu=use_fused_weighted_squared_relu,
        moe_apply_probs_on_input=False,
    )
    module.activation_func = object()
    module.tp_group = SimpleNamespace(size=lambda: 1)
    module.offload_expert_fc1 = False
    module.offload_moe_act = False
    common = dict(
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=False,
    )
    module.linear_fc1 = FakeGroupedLinear(2, 4, 8, bias=False, **common)
    module.linear_fc2 = FakeGroupedLinear(2, 8, 4, bias=False, **common)
    return module


def test_is_fused_impl_supported_uses_config_activation_for_swiglu(monkeypatch):
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)
    monkeypatch.setattr(experts_module, "HAVE_TE", True)
    monkeypatch.setattr(experts_module, "is_te_min_version", lambda _: True)
    _install_fake_te_ops_modules(monkeypatch, fake_te)

    module = _make_fused_impl_support_module(
        FakeGroupedLinear, activation_func=F.silu, gated_linear_unit=True
    )

    assert module._is_fused_impl_supported() is True


@pytest.mark.parametrize(
    ("use_fused_weighted_squared_relu", "gated_linear_unit", "expected"),
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_is_fused_impl_supported_gates_scaled_srelu_on_weighted_flag_and_non_glu(
    monkeypatch, use_fused_weighted_squared_relu, gated_linear_unit, expected
):
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)
    monkeypatch.setattr(experts_module, "HAVE_TE", True)
    monkeypatch.setattr(experts_module, "is_te_min_version", lambda _: True)
    _install_fake_te_ops_modules(monkeypatch, fake_te)

    module = _make_fused_impl_support_module(
        FakeGroupedLinear,
        activation_func=squared_relu,
        gated_linear_unit=gated_linear_unit,
        use_fused_weighted_squared_relu=use_fused_weighted_squared_relu,
    )

    assert module._is_fused_impl_supported() is expected


def test_is_fused_impl_supported_requires_scaled_srelu_op(monkeypatch):
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)
    monkeypatch.setattr(experts_module, "HAVE_TE", True)
    monkeypatch.setattr(experts_module, "is_te_min_version", lambda _: True)
    _install_fake_te_ops_modules(monkeypatch, fake_te, include_scaled_srelu=False)

    module = _make_fused_impl_support_module(
        FakeGroupedLinear,
        activation_func=squared_relu,
        gated_linear_unit=False,
        use_fused_weighted_squared_relu=True,
    )

    assert module._is_fused_impl_supported() is False


def test_make_fused_ops_attaches_single_grouped_bias_for_fc1(monkeypatch):
    """single_grouped_bias=True → bias attached as `bias` (not `bias{idx}`)."""
    fake_te, FakeGroupedLinear = _make_fake_te_namespace()
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        moe_mlp_glu_interleave_size=2,
        delay_wgrad_compute=False,
        activation_func_clamp_value=None,
        activation_func=F.silu,
        gated_linear_unit=True,
    )
    module.activation_func = F.silu
    common = dict(device="cuda", dtype=torch.bfloat16, accumulate_into_main_grad=False)
    # FC1 stores bias as a single grouped tensor; weights still per-GEMM.
    module.linear_fc1 = FakeGroupedLinear(
        2, 4, 8, bias=True, single_grouped_weight=False, single_grouped_bias=True, **common
    )
    module.linear_fc2 = FakeGroupedLinear(
        2, 8, 4, bias=False, single_grouped_weight=False, single_grouped_bias=False, **common
    )
    module.linear_fc1.weight0 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.weight1 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.bias = torch.nn.Parameter(torch.zeros(2, 8))
    module.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(4, 8))
    module.linear_fc2.weight1 = torch.nn.Parameter(torch.ones(4, 8))

    ops = module._make_fused_ops()

    assert ops[0].weight0 is module.linear_fc1.weight0
    assert ops[0].weight1 is module.linear_fc1.weight1
    assert ops[0].bias is module.linear_fc1.bias  # ← single grouped bias attached at "bias"
    assert not hasattr(
        ops[0], "bias0"
    ), "bias should not be split into bias{idx} when single_grouped_bias=True"


def test_backward_dw_dispatches_fused_children_in_fc2_then_fc1_order():
    """delay_wgrad_compute=True (via wrapper) → backward_dw calls fused [2] then [0],
    then triggers the wrapper-side wgrad/reduce hooks (PR 4311) so DDP's reduce-scatter
    fires on the original linear_fc1/fc2 modules."""

    class _Recorder:
        def __init__(self, label):
            self.label = label
            self.calls = []

        def backward_dw(self):
            self.calls.append("backward_dw")

    class _FakeWrapper:
        def __init__(self, label):
            self.label = label
            self.delay_wgrad_compute = True
            self.backward_dw_calls = 0
            self.trigger_calls = 0

        def backward_dw(self):
            self.backward_dw_calls += 1

        def _trigger_wgrad_accumulation_and_reduce_hooks(self):
            self.trigger_calls += 1

    class _FakeSequential:
        def __init__(self, children):
            self._children = children

        def children(self):
            return iter(self._children)

    fc1_op = _Recorder("fc1")
    activation_op = _Recorder("activation")
    fc2_op = _Recorder("fc2")
    fake_seq = _FakeSequential([fc1_op, activation_op, fc2_op])

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module._with_fused_impl = True
    module._fused_ops = (fake_seq,)
    module.linear_fc1 = _FakeWrapper("fc1_wrapper")
    module.linear_fc2 = _FakeWrapper("fc2_wrapper")
    module.config = SimpleNamespace(delay_wgrad_compute=False)  # config flag is irrelevant

    module.backward_dw()

    assert fc2_op.calls == ["backward_dw"]
    assert fc1_op.calls == ["backward_dw"]
    assert activation_op.calls == [], "activation op must not be invoked"
    assert module.linear_fc1.backward_dw_calls == 0, "wrapper backward_dw must be skipped"
    assert module.linear_fc2.backward_dw_calls == 0
    # PR 4311: DDP reduce-scatter hooks live on the original wrappers and must be triggered
    # explicitly because the fused path's backward_dw runs on the new GroupedLinear instances.
    assert module.linear_fc2.trigger_calls == 1
    assert module.linear_fc1.trigger_calls == 1


def test_backward_dw_falls_back_to_wrappers_when_delay_wgrad_off():
    """delay_wgrad_compute=False on wrappers → use the original wrapper backward_dw path."""

    class _FakeWrapper:
        def __init__(self):
            self.delay_wgrad_compute = False
            self.backward_dw_calls = 0

        def backward_dw(self):
            self.backward_dw_calls += 1

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module._with_fused_impl = True
    module._fused_ops = None
    module.linear_fc1 = _FakeWrapper()
    module.linear_fc2 = _FakeWrapper()
    module.config = SimpleNamespace(delay_wgrad_compute=False)

    module.backward_dw()

    assert module.linear_fc1.backward_dw_calls == 1
    assert module.linear_fc2.backward_dw_calls == 1


@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TE Grouped MLP is only supported in TE 1.9.0.dev0 and later.",
)
class TestTEGroupedMLP:

    def setup_method(self, method, use_cpu_initialization=False, swiglu=True):
        Utils.initialize_model_parallel(1, 1)
        num_layers = 1
        self.hidden_size = 16
        self.num_experts = 2
        self.gated_linear_unit = swiglu
        self.activation_func = F.silu if swiglu else F.gelu
        self.use_cpu_initialization = use_cpu_initialization

        tf_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=self.use_cpu_initialization,
            add_bias_linear=False,
            gated_linear_unit=self.gated_linear_unit,
            activation_func=self.activation_func,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
        )

        self.fc1_ffn_hidden_size = tf_config.ffn_hidden_size
        self.fc2_ffn_hidden_size = tf_config.ffn_hidden_size
        # If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.gated_linear_unit:
            self.fc1_ffn_hidden_size *= 2

        ## Vanilla sequential GEMM
        # Set random seed for reproducability
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        sequential_submodules = get_submodules(
            get_gpt_layer_local_submodules(self.num_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(sequential_submodules, MoESubmodules)
        self.sequential_mlp = MoELayer(tf_config, sequential_submodules)

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16 = True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear = False
        self.sequential_mlp = Float16Module(self.sequential_mlp.config, self.sequential_mlp).module

        ## Grouped GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        tf_config.moe_grouped_gemm = True
        grouped_submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                self.num_experts, moe_grouped_gemm=True
            ).mlp
        )
        assert isinstance(grouped_submodules, MoESubmodules)
        self.grouped_mlp = MoELayer(tf_config, grouped_submodules)
        assert isinstance(self.grouped_mlp.experts, TEGroupedMLP)
        self.grouped_mlp = Float16Module(self.grouped_mlp.config, self.grouped_mlp).module

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.sequential_mlp, MoELayer)
        assert isinstance(self.grouped_mlp, MoELayer)

        num_weights_smm = sum([p.numel() for p in self.sequential_mlp.parameters()])
        num_weights_gmm = sum([p.numel() for p in self.grouped_mlp.parameters()])

        # For the same hyper-parm model configs except the `moe_grouped_gemm`,
        # GroupedGEMM and sequential GEMMs should hold the same number of parms.
        assert num_weights_smm == num_weights_gmm
        # expected num weights: router linear weights+bias + MLP weights(no bias) of all experts
        expected_num_weights = (
            self.hidden_size * self.num_experts
            + self.hidden_size
            * (self.fc1_ffn_hidden_size + self.fc2_ffn_hidden_size)
            * self.num_experts
        )
        assert num_weights_smm == expected_num_weights

        assert torch.equal(self.sequential_mlp.router.weight, self.grouped_mlp.router.weight)

        # weights of linear_fc1: [fc1_ffn_hidden_size, hidden_size]
        # weights of linear_fc2: [hidden_size, fc2_ffn_hidden_size]
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").shape == (
                self.fc1_ffn_hidden_size,
                self.hidden_size,
            )
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").shape == (
                self.hidden_size,
                self.fc2_ffn_hidden_size,
            )

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "dense_dtype",
        [None, torch.bfloat16, torch.float16],
        ids=["indexed", "dense_bf16", "dense_fp16"],
    )
    def test_gpu_forward_backward(self, monkeypatch, dense_dtype):
        device = "cuda" if dense_dtype is None else "npu"
        if dense_dtype is None and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if dense_dtype is not None:
            if not hasattr(torch, "npu") or not torch.npu.is_available():
                pytest.skip("Dense grouped weights require NPU")
            monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
            monkeypatch.setenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "1")
            config = replace(
                self.grouped_mlp.config,
                moe_single_grouped_weight=True,
                params_dtype=dense_dtype,
                bf16=dense_dtype == torch.bfloat16,
                fp16=dense_dtype == torch.float16,
                gradient_accumulation_fusion=False,
            )
            self.grouped_mlp = MoELayer(config, self.grouped_mlp.submodules)
            self.sequential_mlp.to(dtype=dense_dtype)
            self.grouped_mlp.router.load_state_dict(self.sequential_mlp.router.state_dict())
        self.sequential_mlp.to(device)
        self.grouped_mlp.to(device)
        # Copy the weights to ensure the same init value
        with torch.no_grad():
            for i in range(self.num_experts):
                self.sequential_mlp.experts.local_experts[i].linear_fc1.weight.copy_(
                    self.grouped_mlp.experts.linear_fc1._get_weight_tensors()[i]
                )
                self.sequential_mlp.experts.local_experts[i].linear_fc2.weight.copy_(
                    self.grouped_mlp.experts.linear_fc2._get_weight_tensors()[i]
                )
        # [sequence length, batch size, hidden size]
        seq_len = 32
        batch_size = 2
        hidden_states = torch.rand(
            (seq_len, batch_size, self.hidden_size),
            dtype=dense_dtype or torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        hidden_states.retain_grad()

        output_smm, _ = self.sequential_mlp(hidden_states)
        output_smm.mean().backward()
        smm_results = [output_smm, hidden_states.grad]
        for i in range(self.num_experts):
            smm_results.append(self.sequential_mlp.experts.local_experts[i].linear_fc1.weight.grad)
            smm_results.append(self.sequential_mlp.experts.local_experts[i].linear_fc2.weight.grad)

        hidden_states.grad = None
        if dense_dtype is not None:
            from transformer_engine.plugin.core.manager import OpManager, get_default_manager

            op = "te_general_grouped_gemm_for_grouped_tensor"
            manager, dispatch = get_default_manager(), OpManager.call
            assert manager.get_selected_impl_id(op) == "vendor.npu"
            routed = set()

            def record_dispatch(self, op_name, *args, **kwargs):
                result = dispatch(self, op_name, *args, **kwargs)
                if op_name == op:
                    routed.add((args[1], args[3], self._get_last_impl_id(op)))
                return result

            monkeypatch.setattr(OpManager, "call", record_dispatch)
        output_gmm, _ = self.grouped_mlp(hidden_states)
        output_gmm.mean().backward()
        if dense_dtype is not None:
            assert hidden_states.device.type == output_gmm.device.type == "npu"
            assert routed == {
                (True, False, "vendor.npu"),
                (False, False, "vendor.npu"),
                (False, True, "vendor.npu"),
            }
        gmm_results = [output_gmm, hidden_states.grad]
        fc1, fc2 = self.grouped_mlp.experts.linear_fc1, self.grouped_mlp.experts.linear_fc2
        for i in range(self.num_experts):
            gmm_results.append(
                fc1.weight.grad[i] if dense_dtype else fc1._get_weight_tensors()[i].grad
            )
            gmm_results.append(
                fc2.weight.grad[i] if dense_dtype else fc2._get_weight_tensors()[i].grad
            )

        for smm_result, gmm_result in zip(smm_results, gmm_results):
            if dense_dtype is None:
                torch.testing.assert_close(smm_result, gmm_result)
            else:
                tolerance = (
                    dict(atol=2e-3, rtol=2e-2)
                    if dense_dtype == torch.bfloat16
                    else dict(atol=2e-4, rtol=2e-3)
                )
                torch.testing.assert_close(smm_result, gmm_result, **tolerance)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward_backward_with_no_tokens_allocated(self):
        """Test the case when no token is allocated for groupedGEMM kernels."""
        self.grouped_mlp.cuda()
        num_allocated_tokens = 0
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.int32)
        hidden_states = torch.rand((num_allocated_tokens, self.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()
        probs = torch.rand((num_allocated_tokens,), dtype=torch.float32)
        probs = probs.cuda()
        output, _ = self.grouped_mlp.experts(
            hidden_states, tokens_per_expert=tokens_per_expert, permuted_probs=probs
        )
        assert torch.equal(output, torch.zeros_like(output))
        assert output.shape == (num_allocated_tokens, self.hidden_size)

        output.mean().backward()
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").grad is not None
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_make_fused_ops_constructs_with_real_te(self):
        """Verify `_make_fused_ops` builds a working TE op-fuser pipeline.

        Regression guard for the `device="meta"` shell + weight-reattachment
        sequence — a TE-side change that allocates state in the constructor
        would break this without any of the mock-based unit tests catching it.
        """
        try:
            from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU
        except ImportError:
            pytest.skip("TE op fuser API not available")
        import inspect

        if "single_grouped_weight" not in inspect.signature(GroupedLinear.__init__).parameters:
            pytest.skip(
                "Installed TE op fuser GroupedLinear lacks `single_grouped_weight` kwarg; "
                "_make_fused_ops requires a TE build that exposes it."
            )

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)

        tf_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=False,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                self.num_experts, moe_grouped_gemm=True
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        layer = MoELayer(tf_config, submodules)
        layer = Float16Module(layer.config, layer).module
        layer.cuda()
        experts = layer.experts
        assert isinstance(experts, TEGroupedMLP)
        assert experts._with_fused_impl

        ops = experts._make_fused_ops()

        assert len(ops) == 3
        assert isinstance(ops[0], GroupedLinear)
        assert isinstance(ops[1], ScaledSwiGLU)
        assert isinstance(ops[2], GroupedLinear)
        # Weights of the wrapper ops must alias the underlying GroupedLinear
        # parameters so optimizer updates are visible to the fused path.
        for idx in range(experts.linear_fc1.num_gemms):
            if not getattr(experts.linear_fc1, "single_grouped_weight", False):
                assert getattr(ops[0], f"weight{idx}") is getattr(
                    experts.linear_fc1, f"weight{idx}"
                )
        for idx in range(experts.linear_fc2.num_gemms):
            if not getattr(experts.linear_fc2, "single_grouped_weight", False):
                assert getattr(ops[2], f"weight{idx}") is getattr(
                    experts.linear_fc2, f"weight{idx}"
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_fused_path_loss_decreases(self):
        """End-to-end signal that the Megatron op-fuser wrapper actually trains.

        Builds a small grouped-MLP MoE layer with `use_transformer_engine_op_fuser=True`,
        runs a few Adam steps driving the output toward a fixed target, and asserts the
        loss decreases and stays finite. This catches regressions in the Megatron-side
        wrapper plumbing — `_make_fused_ops` op-sequence construction, the fused-impl
        pre-forward hook, weight aliasing into the TE op wrappers, and (delayed-)wgrad
        backward — failure modes that pass our mock-based unit tests but break real
        training.

        IMPORTANT: this test does NOT verify that TE's underlying cuDSL fused grouped
        MLP kernel is actually invoked. That kernel additionally requires SM100
        (Blackwell) compute capability, plus `NVTE_CUTEDSL_FUSED_GROUPED_MLP` set, plus
        a fusion-registration that happens at TE module import. On H100/A100 hardware
        (which is the current CI matrix) the kernel is unavailable regardless and TE
        falls back to its basic-op path — the Megatron wrapper still runs end-to-end
        and the loss still decreases. So this test covers the wrapper code this PR
        adds; it does not cover the cuDSL kernel selection. The functional test
        `moe/gpt3_mcore_te_tp1_pp1_te_4experts_groupedGEMM_op_fuser` sets the env var
        in its recipe as Blackwell forward-compat, but on current CI hardware it also
        runs basic-op.
        """
        try:
            from transformer_engine.pytorch.ops import GroupedLinear  # noqa: F401
        except ImportError:
            pytest.skip("TE op fuser API not available")
        import inspect

        if "single_grouped_weight" not in inspect.signature(GroupedLinear.__init__).parameters:
            pytest.skip("Installed TE op fuser GroupedLinear lacks `single_grouped_weight` kwarg")

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)

        tf_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=False,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                self.num_experts, moe_grouped_gemm=True
            ).mlp
        )
        layer = MoELayer(tf_config, submodules)
        layer = Float16Module(layer.config, layer).module
        layer.cuda()
        assert isinstance(layer.experts, TEGroupedMLP)
        assert layer.experts._with_fused_impl

        seq_len = 32
        batch_size = 2
        torch.manual_seed(7)
        hidden_states = torch.rand(
            (seq_len, batch_size, self.hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        # MoE blocks with small-init weights produce near-zero output in bf16,
        # so a `||output||^2`-style loss starts at zero and never moves the
        # weights. Drive the output toward a fixed non-zero target so the
        # gradient is non-trivial even when the initial output is tiny.
        with torch.no_grad():
            init_output, _ = layer(hidden_states)
        target = torch.full_like(init_output, 0.5, dtype=torch.float32)
        # Adam (not SGD) because bf16 weights quantize SGD updates of tiny
        # gradients down to zero. Adam's per-parameter adaptive step keeps
        # the update large enough to move bf16 weights regardless of grad
        # magnitude — without it, even a working fused path produces only
        # ~1e-6 loss movement per step, which is below trustable noise.
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)

        losses = []
        for _ in range(16):
            optimizer.zero_grad(set_to_none=True)
            output, _ = layer(hidden_states)
            loss = (output.float() - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())

        assert all(
            l == l and l != float("inf") for l in losses
        ), f"Fused-path produced non-finite loss: {losses}"
        # Require a meaningful relative drop. If the fused path silently no-ops
        # weight updates (the historical wgrad-hook bug PR #4311 fixed) the
        # loss is constant and this fires.
        assert losses[-1] < losses[0] * 0.5, (
            f"Fused-path training did not reduce loss enough: "
            f"initial={losses[0]:.6f}, final={losses[-1]:.6f}, full={losses}"
        )

    @pytest.mark.parametrize("parallel_mode", ["column", "row"])
    def test_expert_parallel_attributes_survive_reset(self, parallel_mode):
        """Megatron's parallel attributes must outlive TE's parameter rebuilds."""
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            expert_model_parallel_size=1,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        # explicit_expert_comm, which clears TE's parallel_mode, needs EP or expert TP.
        config.expert_model_parallel_size = 2
        linear = TEGroupedLinear(
            self.num_experts,
            self.hidden_size,
            self.hidden_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            is_expert=True,
        )
        assert linear.explicit_expert_comm

        expected_dim = 0 if parallel_mode == "column" else 1
        # Once after construction and once more after a reset: TE rebuilds its
        # parameters on reset, so attributes set only in __init__ are lost.
        for _ in range(2):
            for i in range(linear.num_gemms):
                weight = getattr(linear, f"weight{i}")
                assert weight.partition_dim == expected_dim
                assert weight.partition_stride == 1
                assert weight.allreduce is False
            linear.reset_parameters()

    @pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()), reason="NPU required"
    )
    def test_npu_indexed_expert_weights_count_in_grad_norm(self):
        """Indexed experts must not be dropped from gradient statistics as TP duplicates."""
        from megatron.core.extensions.transformer_engine import TEGroupedLinear
        from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate

        if Utils.world_size % 2:
            pytest.skip("This parallel layout requires an even number of ranks")
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(2, 1, expert_model_parallel_size=1)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            tensor_model_parallel_size=2,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        linear = TEGroupedLinear(
            self.num_experts,
            self.hidden_size,
            self.hidden_size,
            parallel_mode="column",
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            is_expert=True,
        )
        assert linear.explicit_expert_comm

        # Expert TP is 2 here, so each rank holds its own half of every expert weight
        # and must contribute it to the gradient norm, clipping and the zero count.
        for i in range(linear.num_gemms):
            weight = getattr(linear, f"weight{i}")
            assert param_is_not_tensor_parallel_duplicate(weight)

    @pytest.fixture
    def npu_dense_config(self, monkeypatch):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("NPU required")
        from transformer_engine.plugin.core.backends.vendor.npu.patches import apply_patch

        # Megatron's transfer_to_npu import can replace TE's compatibility hooks.
        apply_patch()
        monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
        monkeypatch.setenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "1")
        return replace(
            self.grouped_mlp.config,
            moe_single_grouped_weight=True,
            gradient_accumulation_fusion=False,
            use_cpu_initialization=False,
        )

    def test_npu_dense_replicated_weights_use_tp_deduplication(self, npu_dense_config):
        """Replicated experts must contribute to gradient statistics only once per TP group."""
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate

        if Utils.world_size % 4:
            pytest.skip("This parallel layout requires a multiple of four ranks")
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            4, 1, expert_model_parallel_size=1, expert_tensor_parallel_size=1
        )
        config = replace(
            npu_dense_config,
            tensor_model_parallel_size=4,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            sequence_parallel=True,
        )
        experts = MoELayer(config, self.grouped_mlp.submodules).experts
        for linear in (experts.linear_fc1, experts.linear_fc2):
            assert not linear.explicit_expert_comm
            assert not linear.weight.tensor_model_parallel
            assert param_is_not_tensor_parallel_duplicate(linear.weight) == (
                parallel_state.get_tensor_model_parallel_rank() == 0
            )

    @pytest.mark.parametrize(
        "unsupported",
        [
            "single_param_off",
            "fused_gemm_off",
            "fp32",
            "offload",
            "input_fp32",
            "backend_denied",
            "backend_denied_at_runtime",
            "backend_error",
        ],
    )
    def test_npu_dense_grouped_unsupported(self, npu_dense_config, monkeypatch, unsupported):
        from transformer_engine.plugin.core.policy import with_denied_vendors

        rejected = pytest.raises(
            ValueError, match="moe_single_grouped_weight is currently supported only"
        )
        if unsupported.endswith("off"):
            flag = (
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM"
                if unsupported == "single_param_off"
                else "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"
            )
            monkeypatch.setenv(flag, "0")
        if unsupported == "input_fp32":
            layer = MoELayer(npu_dense_config, self.grouped_mlp.submodules).experts
            with pytest.raises(TypeError, match="requires BF16 or FP16 input"):
                layer.linear_fc1(torch.randn(8, 16, device="npu"), [2] * 4)
        elif unsupported in ("backend_denied_at_runtime", "backend_error"):
            import torch_npu

            linear = MoELayer(npu_dense_config, self.grouped_mlp.submodules).experts.linear_fc1
            tokens = torch.randn(2 * linear.num_gemms, 16, device="npu", dtype=torch.bfloat16)
            if unsupported == "backend_denied_at_runtime":
                with (
                    with_denied_vendors("NPU"),
                    pytest.raises(RuntimeError, match="No available implementation"),
                ):
                    linear(tokens, [2] * linear.num_gemms)
            else:

                def fail_grouped_matmul(*args, **kwargs):
                    raise RuntimeError("injected grouped GEMM failure")

                monkeypatch.setattr(torch_npu, "npu_grouped_matmul", fail_grouped_matmul)
                with pytest.raises(RuntimeError, match="injected grouped GEMM failure"):
                    linear(tokens, [2] * linear.num_gemms)
        elif unsupported == "backend_denied":
            # With the NPU vendor denied, TE-FL selects another implementation of the
            # grouped GEMM, so the dense layout must not be offered at all.
            with with_denied_vendors("NPU"), rejected:
                replace(npu_dense_config)
        else:
            with rejected:
                replace(
                    npu_dense_config,
                    params_dtype=torch.float32 if unsupported == "fp32" else torch.bfloat16,
                    bf16=unsupported != "fp32",
                    cpu_offloading=unsupported == "offload",
                )

    @pytest.mark.parametrize("change", ["single_param", "fused_gemm", "backend"])
    @pytest.mark.parametrize("meta_init", [False, True])
    def test_npu_dense_grouped_initialization_changes(
        self, npu_dense_config, monkeypatch, change, meta_init
    ):
        """Reject changes after config validation or before meta materialization."""
        from contextlib import nullcontext

        from transformer_engine.plugin.core.policy import with_denied_vendors

        config = replace(npu_dense_config, init_model_with_meta_device=meta_init)
        if meta_init:
            experts = MoELayer(config, self.grouped_mlp.submodules).experts
        if change != "backend":
            flag = (
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM"
                if change == "single_param"
                else "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"
            )
            monkeypatch.setenv(flag, "0")
        policy = with_denied_vendors("NPU") if change == "backend" else nullcontext()
        with policy, pytest.raises(RuntimeError, match="NPU dense grouped weights require"):
            if meta_init:
                experts.linear_fc1.reset_parameters()
            else:
                MoELayer(config, self.grouped_mlp.submodules)

    @staticmethod
    def _npu_expert_tensors(layer, optimizer, state=None):
        """Read per-expert weights or gathered optimizer shards in either parameter layout."""
        from megatron.core import parallel_state

        values = []
        for linear in (layer.linear_fc1, layer.linear_fc2):
            for parameter in linear.parameters():
                value = parameter.detach()
                if state is not None:
                    value = torch.zeros_like(parameter, dtype=torch.float32).view(-1)
                    if parameter in optimizer.model_param_gbuf_map:
                        bound = optimizer._get_model_param_range_map(parameter)["param"]
                        master = parameter.main_param
                        shard = (
                            master
                            if state == "param"
                            else (
                                master.grad
                                if state == "grad"
                                else optimizer.optimizer.state[master][state]
                            )
                        )
                        value[bound.start : bound.end].copy_(shard)
                    torch.distributed.all_reduce(
                        value, group=parallel_state.get_expert_data_parallel_group()
                    )
                    value = value.view(parameter.shape)
                values.extend(value.unbind(0) if value.ndim == 3 else [value])
        return [value.detach().cpu().double() for value in values]

    @pytest.mark.parametrize(
        "save_dense,load_dense,sharding",
        [
            (True, True, "fully_sharded_model_space"),
            (True, True, "fully_reshardable"),
            (False, True, "fully_reshardable"),
            (True, False, "fully_reshardable"),
        ],
        ids=["dense_sharded", "dense_reshardable", "indexed_to_dense", "dense_to_indexed"],
    )
    def test_npu_dense_grouped_checkpoint(
        self, npu_dense_config, tmp_path_dist_ckpt, monkeypatch, sharding, save_dense, load_dense
    ):
        """Recover weights and Adam state, then continue training across parameter layouts."""
        from copy import deepcopy

        from megatron.core import parallel_state
        from megatron.core.dist_checkpointing import load, save
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from tests.unit_tests.dist_checkpointing import TempNamedDir

        if Utils.world_size != 8:
            pytest.skip("This parallel layout requires eight ranks")
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            2, 1, expert_model_parallel_size=2, expert_tensor_parallel_size=2
        )
        # Native indexed FC2 metadata rejects flattened_range in model-space format.
        # Cross-layout recovery therefore uses its supported fully_reshardable format.
        metadata = {
            "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True),
            "distrib_optim_sharding_type": sharding,
        }

        def create(dense):
            for flag in (
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
                "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM",
            ):
                monkeypatch.setenv(flag, str(int(dense)))
            config = replace(
                npu_dense_config,
                moe_single_grouped_weight=dense,
                init_model_with_meta_device=True,
                num_moe_experts=8,
                tensor_model_parallel_size=2,
                expert_model_parallel_size=2,
                expert_tensor_parallel_size=2,
            )
            layer = TEGroupedMLP(
                4,
                config,
                get_submodules(self.grouped_mlp.submodules.experts),
                get_default_pg_collection(),
            )
            for linear, axis in ((layer.linear_fc1, 0), (layer.linear_fc2, 1)):
                assert all(p.is_meta for p in linear.parameters())
                for _ in range(2):
                    linear.reset_parameters()
                    for parameter in linear.parameters():
                        assert parameter.device.type == "npu"
                        assert (parameter.partition_dim, parameter.partition_stride) == (
                            axis + int(dense),
                            1,
                        )
                        assert parameter.tensor_model_parallel and not parameter.allreduce
            ddp = DistributedDataParallel(
                layer.config, DistributedDataParallelConfig(use_distributed_optimizer=True), layer
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    bf16=True,
                    params_dtype=torch.bfloat16,
                    use_distributed_optimizer=True,
                    lr=0.001,
                    clip_grad=0.0,
                ),
                [ddp],
            )
            return layer, ddp, optimizer.chained_optimizers[-1]

        layer, ddp, optimizer = create(save_dense)
        x = torch.randn(8, 16, device="npu", dtype=torch.bfloat16) * 0.1

        def step(module, model_optimizer):
            module.zero_grad_buffer()
            model_optimizer.zero_grad()
            output, _ = module(x, torch.tensor([3, 0, 2, 3]), torch.ones(8, device="npu"))
            output.float().square().sum().backward()
            module.finish_grad_sync()
            assert model_optimizer.step()[0]

        size = sum(p.numel() for p in layer.linear_fc1.parameters())
        cut = size // 2 + 1
        begin, end = (
            (0, cut) if parallel_state.get_expert_data_parallel_rank() == 0 else (cut, size)
        )
        # Flat ranges of the parent weight owned by this rank: one that crosses expert
        # rows, and empty ones at the start, the middle and the end.
        probe_ranges = {
            "cross_row": (begin, end),
            **{
                f"empty_{index}": (0, size) if begin == 0 else (index, index)
                for index in (0, cut, size)
            },
        }
        if not (save_dense and load_dense):
            probe_ranges = {}

        def probe_values(first, last, is_loading=False):
            # A checkpoint probe stores its own flat index. The load target starts as
            # NaN, so only values read back from the checkpoint can match.
            values = torch.arange(first, last, dtype=torch.float32, device="npu")
            return values.fill_(float("nan")) if is_loading else values

        def state(module, model_optimizer, is_loading=False):
            model = module.sharded_state_dict(metadata=metadata)
            parent = model.get("linear_fc1.weight")
            return {
                "model": model,
                "optimizer": model_optimizer.sharded_state_dict(
                    model, metadata=metadata, is_loading=is_loading
                ),
                "probes": {
                    name: replace(
                        parent,
                        key=name + "." + parent.key,
                        data=probe_values(first, last, is_loading),
                        flattened_range=slice(first, last),
                        replica_id=(*parent.replica_id[:2], 0),
                    )
                    for name, (first, last) in probe_ranges.items()
                },
            }

        def values(module, model_optimizer):
            return {
                "tensors": {
                    name: self._npu_expert_tensors(module, model_optimizer, name)
                    for name in (None, "param", "exp_avg", "exp_avg_sq")
                },
                "param_groups": deepcopy(
                    [
                        {key: value for key, value in group.items() if key != "params"}
                        for group in model_optimizer.optimizer.param_groups
                    ]
                ),
            }

        step(ddp, optimizer)
        saved = values(layer, optimizer)
        with TempNamedDir(tmp_path_dist_ckpt / "grouped_checkpoint") as path:
            save(state(layer, optimizer), path)
            step(ddp, optimizer)
            restored_layer, restored_ddp, restored_optimizer = create(load_dense)
            target = state(restored_layer, restored_optimizer, is_loading=True)
            expected_probes = {name: probe_values(*span) for name, span in probe_ranges.items()}
            restored = load(target, path)
            torch.testing.assert_close(restored["probes"], expected_probes, rtol=0, atol=0)
            restored_layer.load_state_dict(restored["model"])
            torch.testing.assert_close(
                self._npu_expert_tensors(restored_layer, restored_optimizer),
                saved["tensors"][None],
                rtol=0,
                atol=0,
            )
            restored_optimizer.load_state_dict(restored["optimizer"])
            torch.testing.assert_close(
                values(restored_layer, restored_optimizer), saved, rtol=0, atol=0
            )
            step(restored_ddp, restored_optimizer)
            tolerance = (
                dict(rtol=0, atol=0) if save_dense == load_dense else dict(rtol=2e-2, atol=2e-3)
            )
            torch.testing.assert_close(
                values(restored_layer, restored_optimizer), values(layer, optimizer), **tolerance
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("expert_tp", [1, 2])
    def test_npu_grouped_optimizer_alignment(self, npu_dense_config, monkeypatch, dtype, expert_tp):
        """Compare indexed/dense expert gradients, clipping and distributed Adam updates."""
        from megatron.core import parallel_state
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.resharding.utils import extract_param_metadata
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

        if Utils.world_size % 8:
            pytest.skip("This comparison requires a multiple of eight ranks")
        Utils.destroy_model_parallel()
        tp, ep = (4, 8) if expert_tp == 1 else (2, 2)
        Utils.initialize_model_parallel(
            tp, 1, expert_model_parallel_size=ep, expert_tensor_parallel_size=expert_tp
        )
        config = replace(
            npu_dense_config,
            hidden_size=64,
            ffn_hidden_size=128,
            moe_ffn_hidden_size=128,
            num_moe_experts=ep * 2,
            tensor_model_parallel_size=tp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=expert_tp,
            sequence_parallel=True,
            init_model_with_meta_device=True,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
        )

        def set_mode(enabled):
            for name in (
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
                "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM",
            ):
                monkeypatch.setenv(name, str(int(enabled)))

        models, optimizers = [], []
        for enabled in (False, True):
            set_mode(enabled)
            _set_random_seed(seed_=233, data_parallel_random_init=False)
            layer = TEGroupedMLP(
                2,
                replace(config, moe_single_grouped_weight=enabled),
                get_submodules(self.grouped_mlp.submodules.experts),
                get_default_pg_collection(),
            )
            for name, axis in (("linear_fc1", 0), ("linear_fc2", 1)):
                linear = getattr(layer, name)
                assert all(p.is_meta for p in linear.parameters())
                for _ in range(2):
                    linear.reset_parameters()
                    for parameter in linear.parameters():
                        assert parameter.device.type == "npu"
                        assert parameter.tensor_model_parallel and not parameter.allreduce
                        assert parameter.partition_dim == axis + int(enabled)
                        assert parameter.partition_stride == 1
                if enabled:
                    groups = ProcessGroupCollection.use_mpu_process_groups(
                        ["tp", "pp", "dp", "ep", "expt_tp"]
                    )
                    meta = extract_param_metadata(
                        linear.weight,
                        name + ".weight",
                        torch.distributed.get_rank(),
                        groups,
                        num_experts=config.num_moe_experts,
                    )
                    assert meta.is_tp and meta.is_ep and meta.partition_dim == axis + 1
                    assert (
                        meta.tensor_parallel_group_ranks
                        == torch.distributed.get_process_group_ranks(groups.expt_tp)
                    )
            if enabled:
                # Repeated reset can consume RNG differently for stacked/indexed storage.
                with torch.no_grad():
                    for name in ("linear_fc1", "linear_fc2"):
                        reference = getattr(models[0].module, name)
                        getattr(layer, name).weight.copy_(
                            torch.stack([getattr(reference, f"weight{i}") for i in range(2)])
                        )
            ddp = DistributedDataParallel(
                layer.config, DistributedDataParallelConfig(use_distributed_optimizer=True), layer
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    bf16=config.bf16,
                    fp16=config.fp16,
                    params_dtype=dtype,
                    use_distributed_optimizer=True,
                    loss_scale=1.0,
                    lr=1e-4,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    clip_grad=0.1,
                ),
                [ddp],
            )
            models.append(ddp)
            optimizers.append(optimizer)

        def tensors(index, state=None):
            return self._npu_expert_tensors(
                models[index].module, optimizers[index].chained_optimizers[-1], state
            )

        tolerance = dict(atol=2e-3, rtol=2e-2) if config.bf16 else dict(atol=2e-4, rtol=2e-3)

        def assert_aligned(left, right):
            torch.testing.assert_close(left, right, **tolerance)
            left = torch.cat([value.reshape(-1) for value in left])
            right = torch.cat([value.reshape(-1) for value in right])
            assert (left - right).norm() <= (1e-2 if config.bf16 else 2e-3) * left.norm()

        torch.testing.assert_close(tensors(0), tensors(1), rtol=0, atol=0)
        # The layout is committed when the parameters are built, so the switches
        # must not change how an existing layer executes.
        set_mode(False)
        for iteration in range(3):
            before = [tensors(i, "param") for i in range(2)]
            results = []
            for index, (ddp, optimizer) in enumerate(zip(models, optimizers)):
                ddp.zero_grad_buffer()
                optimizer.zero_grad()
                values = []
                for microbatch, counts in enumerate(([3, 5], [8, 0])):
                    torch.manual_seed(
                        71
                        + 2 * iteration
                        + microbatch
                        + parallel_state.get_expert_model_parallel_rank()
                    )
                    x = torch.randn(8, 64, device="npu", dtype=dtype, requires_grad=True)
                    probabilities = torch.rand(8, device="npu", requires_grad=True)
                    output, _ = ddp(x, torch.tensor(counts), probabilities)
                    (output.float() * (torch.randn_like(output) * 32).float()).sum().backward()
                    values.extend(
                        v.detach().cpu().double() for v in (output, x.grad, probabilities.grad)
                    )
                ddp.finish_grad_sync()
                assert not optimizer.prepare_grads()
                gradient = tensors(index, "grad")
                norm = torch.tensor(
                    sum(value.square().sum().item() for value in gradient), device="npu"
                )
                torch.distributed.all_reduce(
                    norm, group=parallel_state.get_expert_tensor_model_pipeline_parallel_group()
                )
                reported = float(optimizer.get_grad_norm())
                assert reported == pytest.approx(norm.sqrt().item(), rel=2e-6, abs=1e-7)
                assert reported > optimizer.config.clip_grad
                old_moments = (
                    [[torch.zeros_like(v) for v in gradient]] * 2
                    if iteration == 0
                    else [tensors(index, s) for s in ("exp_avg", "exp_avg_sq")]
                )
                success, step_norm, _ = optimizer.step()
                assert success and float(step_norm) == pytest.approx(reported, rel=1e-7)
                clipped = [v * min(1.0, 0.1 / (reported + 1e-6)) for v in gradient]
                # Adam may reuse its gradient buffer; validate clipping through both moments.
                for state, previous, beta, power in zip(
                    ("exp_avg", "exp_avg_sq"), old_moments, (0.9, 0.95), (1, 2)
                ):
                    torch.testing.assert_close(
                        tensors(index, state),
                        [
                            beta * old + (1 - beta) * g.pow(power)
                            for old, g in zip(previous, clipped)
                        ],
                        rtol=2e-6,
                        atol=1e-8,
                    )
                results.append(values + gradient)
            assert_aligned(*results)
            for state in (None, "param", "exp_avg", "exp_avg_sq"):
                assert_aligned(tensors(0, state), tensors(1, state))
            updates = [
                [old - new for old, new in zip(before[i], tensors(i, "param"))] for i in range(2)
            ]
            assert_aligned(*updates)
