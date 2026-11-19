# Copyright (c) 2026, BAAI. All rights reserved.
"""HyperConnectionModule: op wiring, mHC mappings, expansion, residual mixing, checkpointing."""

import pytest
import torch

import megatron.core.transformer.hyper_connection as hyper_connection
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig


def _eager(fn):
    # torch.compile hides the decorated body from coverage tools; run the original bytecode.
    return torch.compiler.disable(fn)


def _make_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_hyper_connections=True,
        num_residual_streams=2,
        mhc_sinkhorn_iterations=3,
        mhc_init_gating_factor=0.05,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_hyper_connection_module_wires_fused_public_api_when_use_fused_mhc_enabled():
    from megatron.core.fusions import fused_mhc_kernels

    module = HyperConnectionModule(
        config=_make_config(use_fused_mhc=True, mhc_fused_backend="native"), layer_number=1
    )
    assert module._sinkhorn_op.func is fused_mhc_kernels.fused_sinkhorn
    assert module._h_aggregate_op.func is fused_mhc_kernels.fused_h_aggregate
    assert module._h_post_bda_op.func is fused_mhc_kernels.fused_h_post_bda
    assert module._proj_rms_compute_h_op.func is fused_mhc_kernels.fused_proj_rms_compute_h
    for op in (
        module._sinkhorn_op,
        module._h_aggregate_op,
        module._h_post_bda_op,
        module._proj_rms_compute_h_op,
    ):
        assert op.keywords == {"backend": "native"}


def test_hyper_connection_module_keeps_native_ops_when_fused_mhc_disabled():
    module = HyperConnectionModule(config=_make_config(), layer_number=1)
    assert module._sinkhorn_op is hyper_connection.native_sinkhorn
    assert module._h_aggregate_op is hyper_connection.native_h_aggregate
    assert module._h_post_bda_op is hyper_connection.native_h_post_bda
    assert module._proj_rms_compute_h_op is None


def test_compute_mappings_fused_path_matches_native_module_when_use_fused_mhc_enabled():
    native_module = HyperConnectionModule(config=_make_config(), layer_number=1).cuda()
    fused_module = HyperConnectionModule(
        config=_make_config(use_fused_mhc=True, mhc_fused_backend="native"), layer_number=1
    ).cuda()
    native_state = native_module.state_dict()
    fused_module.load_state_dict(native_state)
    for name, parameter in fused_module.state_dict().items():
        assert torch.equal(parameter, native_state[name])

    torch.manual_seed(1234)
    s, b = 4, 2
    x = torch.randn(s, b, native_module.n * native_module.hidden_size, device="cuda")

    h_pre_ref, h_post_ref, h_res_ref = native_module.compute_mappings(x)
    h_pre, h_post, h_res = fused_module.compute_mappings(x)

    assert h_res.shape == (s, b, native_module.n, native_module.n)
    torch.testing.assert_close(h_pre, h_pre_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h_post, h_post_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h_res, h_res_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        h_res.sum(dim=-1), torch.ones_like(h_res.sum(dim=-1)), rtol=1e-3, atol=1e-3
    )


def test_compute_h_applies_gating_and_bias_when_run_eagerly():
    module = HyperConnectionModule(config=_make_config(num_residual_streams=3), layer_number=1)
    torch.manual_seed(1234)
    s, b = 2, 4
    proj = torch.randn(s, b, module.n * module.n + 2 * module.n)
    r = torch.rand(s, b, 1) + 0.5

    h_pre, h_post, h_res = _eager(HyperConnectionModule._compute_h)(module, proj, r)

    alpha = torch.cat(
        [
            module.alpha_pre.detach().expand(module.n),
            module.alpha_post.detach().expand(module.n),
            module.alpha_res.detach().expand(module.n * module.n),
        ]
    )
    h = r * proj * alpha + module.bias.detach()
    torch.testing.assert_close(
        h_pre, h[..., : module.n].sigmoid() + module.compute_h_eps, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        h_post, h[..., module.n : 2 * module.n].sigmoid() * 2, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(h_res, h[..., 2 * module.n :], rtol=1e-6, atol=1e-7)


def test_compute_h_applies_gating_and_bias_when_compiled_on_gpu():
    module = HyperConnectionModule(
        config=_make_config(num_residual_streams=3), layer_number=1
    ).cuda()
    torch.manual_seed(1234)
    s, b = 2, 4
    proj = torch.randn(s, b, module.n * module.n + 2 * module.n, device="cuda")
    r = torch.rand(s, b, 1, device="cuda") + 0.5

    h_pre, h_post, h_res = module._compute_h(proj, r)

    alpha = torch.cat(
        [
            module.alpha_pre.detach().expand(module.n),
            module.alpha_post.detach().expand(module.n),
            module.alpha_res.detach().expand(module.n * module.n),
        ]
    )
    h = r * proj * alpha + module.bias.detach()
    torch.testing.assert_close(
        h_pre, h[..., : module.n].sigmoid() + module.compute_h_eps, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        h_post, h[..., module.n : 2 * module.n].sigmoid() * 2, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(h_res, h[..., 2 * module.n :], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("x_kind", ["hidden_states", "bias"], ids=["hidden_states", "bias_vector"])
def test_apply_h_post_expands_tensor_when_run_eagerly(x_kind):
    module = HyperConnectionModule(config=_make_config(), layer_number=1)
    torch.manual_seed(1234)
    s, b, C = 3, 2, module.hidden_size
    h_post = torch.rand(s, b, module.n, requires_grad=True)
    if x_kind == "bias_vector":
        x = torch.randn(C, requires_grad=True)
    else:
        x = torch.randn(s, b, C, requires_grad=True)

    expanded = _eager(HyperConnectionModule._apply_h_post)(module, x, h_post)

    assert expanded.shape == (s, b, module.n * C)
    if x_kind == "bias_vector":
        reference = h_post.unsqueeze(-1) * x.view(1, 1, 1, C)
    else:
        reference = h_post.unsqueeze(-1) * x.unsqueeze(2)
    torch.testing.assert_close(expanded, reference.reshape(s, b, module.n * C))

    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    x_ref = x.detach().clone().requires_grad_(True)
    h_post_ref = h_post.detach().clone().requires_grad_(True)
    if x_kind == "bias_vector":
        reference_ref = h_post_ref.unsqueeze(-1) * x_ref.view(1, 1, 1, C)
    else:
        reference_ref = h_post_ref.unsqueeze(-1) * x_ref.unsqueeze(2)
    reference_ref.reshape(s, b, module.n * C).backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(h_post.grad, h_post_ref.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("x_kind", ["hidden_states", "bias"], ids=["hidden_states", "bias_vector"])
def test_apply_h_post_expands_tensor_when_compiled_on_gpu(x_kind):
    module = HyperConnectionModule(config=_make_config(), layer_number=1).cuda()
    torch.manual_seed(1234)
    s, b, C = 3, 2, module.hidden_size
    h_post = torch.rand(s, b, module.n, device="cuda")
    if x_kind == "bias_vector":
        x = torch.randn(C, device="cuda")
    else:
        x = torch.randn(s, b, C, device="cuda")

    expanded = module._apply_h_post(x, h_post)

    assert expanded.shape == (s, b, module.n * C)
    if x_kind == "bias_vector":
        reference = h_post.unsqueeze(-1) * x.view(1, 1, 1, C)
    else:
        reference = h_post.unsqueeze(-1) * x.unsqueeze(2)
    torch.testing.assert_close(
        expanded, reference.reshape(s, b, module.n * C), rtol=1e-5, atol=1e-6
    )


def test_apply_h_res_transposes_h_res_when_run_eagerly():
    module = HyperConnectionModule(config=_make_config(), layer_number=1)
    torch.manual_seed(1234)
    s, b, n, C = 3, 2, module.n, module.hidden_size
    h_res = torch.rand(s, b, n, n, requires_grad=True)
    residual = torch.randn(s, b, n * C, requires_grad=True)

    mixed = _eager(HyperConnectionModule.apply_h_res)(module, h_res, residual)

    reference = torch.einsum("sbji,sbjc->sbic", h_res, residual.view(s, b, n, C)).reshape(
        s, b, n * C
    )
    torch.testing.assert_close(mixed, reference, rtol=1e-6, atol=1e-6)

    grad = torch.randn_like(mixed)
    mixed.backward(grad)
    h_res_ref = h_res.detach().clone().requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)
    torch.einsum("sbji,sbjc->sbic", h_res_ref, residual_ref.view(s, b, n, C)).reshape(
        s, b, n * C
    ).backward(grad)
    torch.testing.assert_close(h_res.grad, h_res_ref.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(residual.grad, residual_ref.grad, rtol=1e-5, atol=1e-7)


def test_apply_h_res_transposes_h_res_when_compiled_on_gpu():
    module = HyperConnectionModule(config=_make_config(), layer_number=1).cuda()
    torch.manual_seed(1234)
    s, b, n, C = 3, 2, module.n, module.hidden_size
    h_res = torch.rand(s, b, n, n, device="cuda")
    residual = torch.randn(s, b, n * C, device="cuda")

    mixed = module.apply_h_res(h_res, residual)

    reference = torch.einsum("sbji,sbjc->sbic", h_res, residual.view(s, b, n, C)).reshape(
        s, b, n * C
    )
    torch.testing.assert_close(mixed, reference, rtol=1e-5, atol=1e-5)


def test_output_contract_averages_residual_streams_when_contracting():
    torch.manual_seed(1234)
    s, b, n, C = 2, 3, 4, 5
    x = torch.randn(s, b, n * C)
    contracted = HyperConnectionModule.output_contract(x, n)
    assert contracted.shape == (s, b, C)
    torch.testing.assert_close(contracted, x.view(s, b, n, C).mean(dim=2))


class TestFusedHResHPostBdaCheckpoint:
    @staticmethod
    def _make_module():
        return HyperConnectionModule(config=_make_config(), layer_number=1).cuda()

    @staticmethod
    def _make_inputs(module, with_bias):
        torch.manual_seed(1234)
        s, b, n, C = 4, 2, module.n, module.hidden_size
        h_res = torch.rand(s, b, n, n, device="cuda", requires_grad=True)
        residual = torch.randn(s, b, n * C, device="cuda", requires_grad=True)
        h_post = torch.rand(s, b, n, device="cuda", requires_grad=True)
        x = torch.randn(s, b, C, device="cuda", requires_grad=True)
        bias = torch.randn(C, device="cuda") if with_bias else None
        return h_res, residual, h_post, x, bias

    @staticmethod
    def _clone_inputs(h_res, residual, h_post, x, bias):
        clones = [
            tensor.detach().clone().requires_grad_(True) for tensor in (h_res, residual, h_post, x)
        ]
        return clones + [bias]

    @staticmethod
    def _weighted_loss(output, seed=1234):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = torch.randn(output.shape, generator=generator).to(output.device)
        return (output * weights).sum()

    @pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
    def test_fast_path_matches_uncheckpointed_when_dropout_zero(self, with_bias):
        module = self._make_module()
        h_res, residual, h_post, x, bias = self._make_inputs(module, with_bias)

        out_ref = module.fused_h_res_h_post_bda(
            h_res, residual, h_post, (x, bias), 0.0, True, False, manager=None
        )
        loss_ref = self._weighted_loss(out_ref)
        loss_ref.backward()

        h_res2, residual2, h_post2, x2, bias = self._clone_inputs(h_res, residual, h_post, x, bias)
        manager = CheckpointWithoutOutputManager()
        out_ckpt = module.fused_h_res_h_post_bda(
            h_res2, residual2, h_post2, (x2, bias), 0.0, True, False, manager=manager
        )
        torch.testing.assert_close(out_ckpt, out_ref, rtol=1e-6, atol=1e-6)
        loss_ckpt = self._weighted_loss(out_ckpt)
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()

        torch.testing.assert_close(h_res2.grad, h_res.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(residual2.grad, residual.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h_post2.grad, h_post.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x2.grad, x.grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
    def test_dropout_path_matches_uncheckpointed_when_training(self, with_bias):
        module = self._make_module()
        h_res, residual, h_post, x, bias = self._make_inputs(module, with_bias)
        dropout_prob = 0.25

        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        out_ref = module.fused_h_res_h_post_bda(
            h_res, residual, h_post, (x, bias), dropout_prob, True, False, manager=None
        )
        loss_ref = self._weighted_loss(out_ref)
        loss_ref.backward()

        h_res2, residual2, h_post2, x2, bias = self._clone_inputs(h_res, residual, h_post, x, bias)
        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        manager = CheckpointWithoutOutputManager()
        out_ckpt = module.fused_h_res_h_post_bda(
            h_res2, residual2, h_post2, (x2, bias), dropout_prob, True, False, manager=manager
        )
        torch.testing.assert_close(out_ckpt, out_ref, rtol=1e-6, atol=1e-6)
        loss_ckpt = self._weighted_loss(out_ckpt)
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()

        torch.testing.assert_close(h_res2.grad, h_res.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(residual2.grad, residual.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h_post2.grad, h_post.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x2.grad, x.grad, rtol=1e-5, atol=1e-6)
