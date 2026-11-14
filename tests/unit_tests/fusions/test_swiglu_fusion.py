# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_clamped_weighted_bias_swiglu(input_dtype):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    # Reference: manual clamp + silu + weight
    y_1, y_2 = torch.chunk(x, 2, -1)
    y_1c = y_1.clamp(min=None, max=clamp_value)
    y_2c = y_2.clamp(min=-clamp_value, max=clamp_value)
    y = (F.silu(y_1c) * y_2c * weights).to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach().clone()
    x_2.requires_grad = True
    weights_2 = weights.detach().clone()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach().clone()

    # Fused implementation
    y_2_out = weighted_bias_swiglu_impl(x_2, None, weights_2, clamp_value=clamp_value)
    y_2_out.backward(bwd_input_2)

    assert y_2_out.dtype == y.dtype
    assert torch.allclose(y, y_2_out, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


class TestAscendClampedSwiGLU:
    """Compare both expert paths directly with Core, including probability gradients."""

    @pytest.fixture(autouse=True)
    def setup_npu(self, monkeypatch):
        from megatron.plugin.platform import get_platform

        if get_platform().device_name() != "npu":
            pytest.skip("Requires the Ascend fusion implementation")
        from megatron.plugin.Ascend.fusions import weighted_swiglu_kernels as kernels

        self.device = get_platform().device(get_platform().current_device())
        self.calls = []
        forward, backward = kernels.clamped_swiglu_forward, kernels.clamped_swiglu_backward

        def record_forward(x, weights, limit):
            result = forward(x, weights, limit)
            self.calls.append(("forward", weights is not None))
            return result

        def record_backward(grad, x, weights, limit):
            result = backward(grad, x, weights, limit)
            self.calls.append(("backward", weights is not None))
            return result

        monkeypatch.setattr(kernels, "clamped_swiglu_forward", record_forward)
        monkeypatch.setattr(kernels, "clamped_swiglu_backward", record_backward)
        with torch.random.fork_rng(devices=[get_platform().current_device()], device_type="npu"):
            torch.manual_seed(1234)
            yield

    @staticmethod
    def reference(x, weights, limit):
        if weights is None:
            return bias_swiglu_impl.__wrapped__(x, None, clamp_value=limit)
        return weighted_bias_swiglu_impl.__wrapped__(x, None, weights, clamp_value=limit)

    def compare(self, x, weights, limit, recompute=False):
        from torch.utils.checkpoint import checkpoint

        x = x.detach().requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        weights = None if weights is None else weights.detach().requires_grad_()
        reference_weights = None if weights is None else weights.detach().clone().requires_grad_()

        def candidate(inp, probs):
            if probs is None:
                return bias_swiglu_impl(inp, None, clamp_value=limit)
            return weighted_bias_swiglu_impl(inp, None, probs, clamp_value=limit)

        output = (
            checkpoint(candidate, x, weights, use_reentrant=True)
            if recompute
            else candidate(x, weights)
        )
        expected = self.reference(reference_x, reference_weights, limit)
        grad = torch.randn_like(output) / 8
        output.backward(grad)
        expected.backward(grad)
        pairs = [(output, expected), (x.grad, reference_x.grad)]
        if weights is not None:
            pairs.append((weights.grad, reference_weights.grad))
        for actual, reference in pairs:
            assert actual.dtype == reference.dtype
            assert torch.isfinite(actual).all()
            # Probability reductions may associate across column tiles differently.
            rtol = 2e-5 if actual.dtype == torch.float32 else 2e-3
            torch.testing.assert_close(actual, reference, rtol=rtol, atol=2e-5)
        weighted = weights is not None
        assert ("forward", weighted) in self.calls
        assert ("backward", weighted) in self.calls
        return output, x, weights, grad

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("width", [65, 4352])
    def test_output_input_and_probability_gradients(self, dtype, weighted, width):
        x = (torch.randn(17, 2 * width, device=self.device) * 8).to(dtype)
        weights = torch.rand(17, 1, device=self.device) if weighted else None
        self.compare(x, weights, 10.0)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("limit", [0.3, 0.8, 10.0])
    def test_clamp_boundaries(self, dtype, weighted, limit):
        values = torch.tensor([-12, -10, -0.8, 0, 0.8, 10, 12], device=self.device, dtype=dtype)
        x = torch.stack((values.repeat(2), values.flip(0).repeat(2)))
        weights = torch.tensor([[0.0], [0.371]], device=self.device) if weighted else None
        self.compare(x, weights, limit)

    @pytest.mark.parametrize("probability", ["none", "scalar", "per_token"])
    @pytest.mark.parametrize("rows", [0, 7])
    def test_strided_broadcast_and_recompute(self, probability, rows):
        x = torch.randn(rows, 3, 260, device=self.device, dtype=torch.bfloat16)[..., ::2]
        if rows == 0:
            x = x.reshape(0, 130)
        weights = None
        if probability == "scalar":
            weights = torch.tensor(0.317, device=self.device, dtype=torch.bfloat16)
        elif probability == "per_token":
            weights = torch.rand(x.shape[:-1].numel(), 1, device=self.device)
        self.compare(x, weights, 10.0, recompute=True)

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_independent_fp64_reference(self, weighted, dtype):
        x = (torch.randn(13, 130, device=self.device) * 3).to(dtype)
        weights = torch.rand(13, 1, device=self.device) if weighted else None
        output, inp, probs, grad = self.compare(x, weights, 10.0)
        cpu_x = inp.detach().cpu().double().requires_grad_()
        cpu_p = None if probs is None else probs.detach().cpu().double().requires_grad_()
        assert cpu_x.device.type == "cpu" and cpu_x.dtype == torch.float64
        assert cpu_p is None or (cpu_p.device.type == "cpu" and cpu_p.dtype == torch.float64)
        gate, up = cpu_x.chunk(2, dim=-1)
        reference = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        if weighted:
            reference = reference * cpu_p
        reference.backward(grad.detach().cpu().double())
        pairs = [(output, reference), (inp.grad, cpu_x.grad)]
        if weighted:
            pairs.append((probs.grad, cpu_p.grad))
        for actual, expected in pairs:
            actual = actual.detach().cpu().double()
            relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
            assert relative.item() < (0.01 if dtype == torch.bfloat16 else 0.001)

    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("prob_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_scalar_probability_promotion(self, scalar, dtype, prob_dtype):
        x = torch.randn(7, 130, device=self.device, dtype=dtype)
        shape = () if scalar else (1, 1)
        weights = torch.full(shape, 0.317, device=self.device, dtype=prob_dtype)
        self.compare(x, weights, 10.0)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_nonfinite_values(self, weighted):
        values = [float("nan"), float("inf"), -float("inf"), 0.0]
        x = torch.tensor([values + [1.0] * 4, [1.0] * 4 + values], device=self.device)
        x.requires_grad_()
        xr = x.detach().clone().requires_grad_()
        p = torch.full((2, 1), 0.317, device=self.device, requires_grad=True) if weighted else None
        pr = None if p is None else p.detach().clone().requires_grad_()
        actual = (
            weighted_bias_swiglu_impl(x, None, p, clamp_value=10.0)
            if weighted
            else bias_swiglu_impl(x, None, clamp_value=10.0)
        )
        expected = self.reference(xr, pr, 10.0)
        actual.sum().backward()
        expected.sum().backward()
        pairs = [(actual, expected), (x.grad, xr.grad)]
        if weighted:
            pairs.append((p.grad, pr.grad))
        for value, baseline in pairs:
            torch.testing.assert_close(value, baseline, rtol=2e-5, atol=2e-6, equal_nan=True)
        assert ("forward", weighted) in self.calls and ("backward", weighted) in self.calls

    @pytest.mark.parametrize("weighted", [False, True])
    def test_higher_order_gradients(self, weighted):
        x = torch.randn(3, 16, device=self.device, requires_grad=True)
        weights = torch.rand(3, 1, device=self.device, requires_grad=True) if weighted else None
        reference_x = x.detach().clone().requires_grad_()
        reference_w = None if weights is None else weights.detach().clone().requires_grad_()
        if weighted:
            output = weighted_bias_swiglu_impl(x, None, weights, clamp_value=10.0)
        else:
            output = bias_swiglu_impl(x, None, clamp_value=10.0)
        expected = self.reference(reference_x, reference_w, 10.0)
        inputs = (x,) if weights is None else (x, weights)
        refs = (reference_x,) if weights is None else (reference_x, reference_w)
        first = torch.autograd.grad(output.sum(), inputs, create_graph=True)
        first_ref = torch.autograd.grad(expected.sum(), refs, create_graph=True)
        second = torch.autograd.grad(sum(g.sum() for g in first), inputs)
        second_ref = torch.autograd.grad(sum(g.sum() for g in first_ref), refs)
        for actual, reference in zip((*first, *second), (*first_ref, *second_ref)):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_cpu_fallback(self, weighted):
        x = torch.randn(3, 16, device="cpu", requires_grad=True)
        p = torch.rand(3, 1, device="cpu", requires_grad=True)
        actual = (
            weighted_bias_swiglu_impl(x, None, p, clamp_value=10.0)
            if weighted
            else bias_swiglu_impl(x, None, clamp_value=10.0)
        )
        expected = self.reference(x, p if weighted else None, 10.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert self.calls == []

    @pytest.mark.parametrize("shared", [False, True])
    def test_real_mlp_and_shared_expert_with_recompute(self, shared, monkeypatch):
        from torch.utils.checkpoint import checkpoint

        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer import mlp as mlp_module
        from megatron.core.transformer.mlp import MLP
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
        from megatron.core.transformer.spec_utils import get_submodules
        from megatron.core.transformer.transformer_config import TransformerConfig
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(31)
        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=64,
                num_attention_heads=4,
                num_moe_experts=2,
                moe_ffn_hidden_size=128,
                moe_shared_expert_intermediate_size=128,
                ffn_hidden_size=128,
                add_bias_linear=False,
                gated_linear_unit=True,
                activation_func=F.silu,
                activation_func_clamp_value=10.0,
                bias_activation_fusion=True,
                params_dtype=torch.bfloat16,
                bf16=True,
                use_cpu_initialization=True,
                gradient_accumulation_fusion=False,
            )
            submodules = get_submodules(get_gpt_layer_local_submodules().mlp)

            def build(cfg):
                if shared:
                    return SharedExpertMLP(
                        cfg,
                        submodules,
                        gate=False,
                        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                    ).to(self.device)
                return MLP(cfg, submodules).to(self.device)

            candidate, reference = build(config), build(config)
            with torch.no_grad():
                candidate.linear_fc1.weight.mul_(100)
            reference.load_state_dict(candidate.state_dict())
            shape = (11, 2, 64) if shared else (22, 64)
            x = torch.randn(shape, device=self.device, dtype=torch.bfloat16).requires_grad_()
            xr = x.detach().clone().requires_grad_()
            probs = (
                None if shared else torch.rand(shape[:-1], device=self.device, requires_grad=True)
            )
            pr = None if shared else probs.detach().clone().requires_grad_()

            def forward(module, inp, scale):
                return module(inp) if shared else module(inp, per_token_scale=scale)[0]

            actual = checkpoint(lambda a, b: forward(candidate, a, b), x, probs, use_reentrant=True)
            grad = torch.randn_like(actual) / 32
            actual.backward(grad)
            with monkeypatch.context() as core_ops:
                core_ops.setattr(mlp_module, "bias_swiglu_impl", bias_swiglu_impl.__wrapped__)
                core_ops.setattr(
                    mlp_module, "weighted_bias_swiglu_impl", weighted_bias_swiglu_impl.__wrapped__
                )
                expected = forward(reference, xr, pr)
                expected.backward(grad)
            pairs = [(actual, expected), (x.grad, xr.grad)]
            pairs += [
                (p.grad, q.grad) for p, q in zip(candidate.parameters(), reference.parameters())
            ]
            if not shared:
                pairs.append((probs.grad, pr.grad))
            for value, baseline in pairs:
                torch.testing.assert_close(value, baseline, rtol=2e-2, atol=2e-4)
            assert ("forward", not shared) in self.calls
            assert ("backward", not shared) in self.calls
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "weighted, option", [(False, "bias"), (False, "fp8"), (True, "fp8"), (False, "offload")]
    )
    def test_positive_clamp_core_fallback(self, weighted, option, record_property):
        """Fallback preserves saved inputs, offload tags and every input gradient."""
        x = torch.randn(5, 130, device=self.device, dtype=torch.bfloat16) * 3
        grad = torch.randn(5, 65, device=self.device, dtype=x.dtype) / 8
        bias = torch.randn(130, device=self.device, dtype=x.dtype)
        probs = torch.rand(5, 1, device=self.device)

        def evaluate(reference):
            inp = x.detach().clone().requires_grad_()
            has_bias = option in ("bias", "offload")
            b = bias.detach().clone().requires_grad_() if has_bias else None
            p = probs.detach().clone().requires_grad_() if weighted else None
            saved = []

            def pack(tensor):
                saved.append(tensor)
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                if weighted:
                    fn = (
                        weighted_bias_swiglu_impl.__wrapped__
                        if reference
                        else weighted_bias_swiglu_impl
                    )
                    output = fn(inp, b, p, fp8_input_store=option == "fp8", clamp_value=10.0)
                else:
                    fn = bias_swiglu_impl.__wrapped__ if reference else bias_swiglu_impl
                    output = fn(
                        inp,
                        b,
                        fp8_input_store=option == "fp8",
                        cpu_offload_input=option == "offload",
                        clamp_value=10.0,
                    )
            assert saved[0].dtype == (torch.float8_e4m3fn if option == "fp8" else inp.dtype)
            if option != "fp8":
                assert saved[0].data_ptr() == inp.data_ptr()
            if option == "offload":
                assert len(saved) == 2
                assert all(getattr(tensor, "activation_offloading", False) for tensor in saved)
            output.backward(grad)
            leaves = [inp] + ([b] if b is not None else []) + ([p] if p is not None else [])
            return [output.detach(), *(leaf.grad for leaf in leaves)]

        try:
            expected = evaluate(True)
        except RuntimeError as error:
            if option != "fp8":
                raise
            # Some NPU runtimes cannot store float8; preserve the Core error in that case.
            with pytest.raises(RuntimeError) as actual_error:
                evaluate(False)
            core_error = str(error).splitlines()[0]
            assert str(actual_error.value).splitlines()[0] == core_error
            record_property("npu_fp8_core_error", core_error)
            # Still exercise real FP8 saved tensors and backward on the supported CPU path.
            x, grad, bias, probs = (tensor.cpu() for tensor in (x, grad, bias, probs))
            expected = evaluate(True)
        for actual, baseline in zip(evaluate(False), expected):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        assert self.calls == []

    @pytest.mark.parametrize("shape, invalid", [((3, 2), True), ((1, 5, 1), False)])
    def test_unsupported_probability_shape_uses_core(self, shape, invalid, monkeypatch):
        """Unsupported and invalid probability shapes reach Core before errors propagate."""
        from megatron.core.fusions import fused_bias_swiglu as core

        x = torch.randn(5, 130, device=self.device, dtype=torch.bfloat16)
        weights = torch.rand(shape, device=self.device)
        original = core.weighted_bias_swiglu_impl.__wrapped__
        calls = []

        def fallback(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(core.weighted_bias_swiglu_impl, "__wrapped__", fallback)
        if invalid:
            with pytest.raises(RuntimeError) as expected:
                original(x, None, weights, clamp_value=10.0)
            with pytest.raises(RuntimeError) as actual:
                weighted_bias_swiglu_impl(x, None, weights, clamp_value=10.0)
            assert str(actual.value) == str(expected.value)
        else:
            inp = x.detach().clone().requires_grad_()
            reference_x = x.detach().clone().requires_grad_()
            probs = weights.detach().clone().requires_grad_()
            reference_p = weights.detach().clone().requires_grad_()
            actual = weighted_bias_swiglu_impl(inp, None, probs, clamp_value=10.0)
            expected = original(reference_x, None, reference_p, clamp_value=10.0)
            actual.sum().backward()
            expected.sum().backward()
            for value, baseline in (
                (actual, expected),
                (inp.grad, reference_x.grad),
                (probs.grad, reference_p.grad),
            ):
                torch.testing.assert_close(value, baseline, rtol=0, atol=0)
        assert calls == [1]
        assert self.calls == []
