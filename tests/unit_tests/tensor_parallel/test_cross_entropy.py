import numpy as np
import pytest
import torch

from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 2)
    vocab_parallel_logits = torch.range(0, 7).repeat(16, 4).cuda()
    target = torch.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    assert torch.equal(torch.round(expected_output), torch.round(output))
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_npu_linear_cross_entropy(monkeypatch, chunked, sequence_parallel):
    """Compare native CE loss, retained-graph gradients, fallback and minimal MTP."""
    if get_platform().device_name() != "npu":
        pytest.skip("Ascend NPU is required")
    from types import SimpleNamespace

    from megatron.core import parallel_state
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer import multi_token_prediction as mtp
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossLoggingHelper,
        process_mtp_loss,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.plugin.Ascend.fusions import linear_cross_entropy as ce

    world = min(4, Utils.world_size)
    if sequence_parallel and world == 1:
        pytest.skip("Sequence parallelism requires multiple ranks")
    Utils.initialize_model_parallel(world)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=world,
            tensor_model_parallel_size=world,
            sequence_parallel=sequence_parallel,
            bf16=True,
            params_dtype=torch.bfloat16,
            perform_initialization=False,
            gradient_accumulation_fusion=False,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="native",
        )

        class Model(LanguageModule):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.config = config
                self.pg_collection = SimpleNamespace(tp=group)

        model = Model()
        torch.manual_seed(233)
        tokens, batch, hidden, vocab = world * 5, 2, 64, 259
        full_x = torch.randn(tokens, batch, hidden, device="npu", dtype=torch.bfloat16) / 8
        x = (full_x.chunk(world)[group.rank()] if sequence_parallel else full_x).clone()
        x.requires_grad_()
        layer = ColumnParallelLinear(
            hidden,
            world * vocab,
            config=config,
            init_method=lambda value: value,
            bias=False,
            tp_group=group,
        )
        with torch.no_grad():
            layer.weight.normal_(std=0.125).add_(group.rank() / 32)
        labels = torch.arange(tokens * batch, device="npu").reshape(batch, tokens) * 257
        labels = labels % (world * vocab)
        labels[0, 1], labels[0, 2], labels[0, 4] = -100, -1, world * vocab
        upstream = torch.linspace(0.01, 0.9, tokens * batch, device="npu").reshape(batch, tokens)
        upstream[:, ::3] = 0
        fused_calls, projected_rows = [], []
        apply, project = ce._LinearCrossEntropy.apply, ce._project

        def fused_apply(*args):
            fused_calls.append(args[-1])
            return apply(*args)

        def tracked_project(hidden, *args):
            projected_rows.append(hidden.shape[0])
            return project(hidden, *args)

        monkeypatch.setattr(ce._LinearCrossEntropy, "apply", fused_apply)
        monkeypatch.setattr(ce, "_project", tracked_project)
        monkeypatch.setattr(ce, "_chunk_size", lambda weight: 3)
        monkeypatch.setattr(ce, "_LOGITS_LIMIT_BYTES", 1 if chunked else 4 * 1024**3)
        monkeypatch.setenv("MEGATRON_FUSE_CE_MM", "1")
        actual = model.compute_output_loss(x, layer, None, labels)
        assert fused_calls == [chunked]
        assert set(projected_rows) == ({3, tokens * batch % 3} if chunked else {tokens * batch})
        xr = x.detach().clone().requires_grad_()
        wr = layer.weight.detach().clone().requires_grad_()
        for repeat, enabled in enumerate(("0", None)):
            if enabled is None:
                monkeypatch.delenv("MEGATRON_FUSE_CE_MM")
            else:
                monkeypatch.setenv("MEGATRON_FUSE_CE_MM", enabled)
            # Native CE mutates its own saved softmax, so rebuild the reference each time.
            expected = model.compute_output_loss(xr, layer, wr, labels)
            assert fused_calls == [chunked]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            actual.backward(upstream * (repeat + 1), retain_graph=repeat == 0)
            expected.backward(upstream * (repeat + 1))
            for value, reference in ((x.grad, xr.grad), (layer.weight.grad, wr.grad)):
                error = (value.float() - reference.float()).norm()
                assert error / reference.float().norm().clamp_min(1e-12) < 0.01

        monkeypatch.setenv("MEGATRON_FUSE_CE_MM", "1")
        config.cpu_offloading = True
        fallback = model.compute_output_loss(xr, layer, wr, labels)
        assert fused_calls == [chunked]
        torch.testing.assert_close(fallback, expected, rtol=1e-5, atol=2e-6)
        config.cpu_offloading = False

        if not sequence_parallel:
            config.mtp_num_layers = 1
            config.mtp_loss_scaling_factor = 0.3
            loss_mask = torch.ones_like(labels)
            loss_mask[:, ::3] = 0
            hidden_states = torch.cat((x.detach(), x.detach() / 2))
            events = []
            roll, native_loss = mtp.roll_tensor, model.compute_language_model_loss

            def tracked_roll(*args, **kwargs):
                events.append("roll")
                return roll(*args, **kwargs)

            def tracked_loss(*args):
                events.append("loss")
                return native_loss(*args)

            monkeypatch.setattr(mtp, "roll_tensor", tracked_roll)
            monkeypatch.setattr(model, "compute_language_model_loss", tracked_loss)
            hook = layer.register_forward_pre_hook(lambda *args: events.append("output"))
            for per_token in (False, True):
                config.calculate_per_token_loss = per_token
                results = []
                for enabled, offloading, callback in (
                    ("0", False, None),
                    ("0", False, model.compute_output_loss),
                    ("1", True, model.compute_output_loss),
                    ("1", False, model.compute_output_loss),
                ):
                    monkeypatch.setenv("MEGATRON_FUSE_CE_MM", enabled)
                    config.cpu_offloading = offloading
                    events.clear()
                    calls_before = len(fused_calls)
                    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
                    h = hidden_states.clone().requires_grad_()
                    w = layer.weight.detach().clone().requires_grad_()
                    output = process_mtp_loss(
                        h,
                        labels,
                        loss_mask,
                        layer,
                        w,
                        False,
                        True,
                        model.compute_language_model_loss,
                        config,
                        scale_logits_fn=model._scale_logits,
                        compute_output_loss=callback,
                    )
                    fused = enabled == "1" and not offloading
                    assert len(fused_calls) - calls_before == int(fused)
                    assert events == (
                        ["roll", "roll", "output"] if fused else ["output", "roll", "roll", "loss"]
                    )
                    loss = model.compute_output_loss(output, layer, w, labels).mean()
                    loss.backward()
                    results.append(
                        (
                            MTPLossLoggingHelper.tracker["values"].clone(),
                            h.grad.float(),
                            w.grad.float(),
                        )
                    )
                for result in results[1:]:
                    torch.testing.assert_close(result[0], results[0][0], rtol=1e-5, atol=2e-6)
                    for actual_value, reference in zip(result[1:], results[0][1:]):
                        error = (actual_value - reference).norm()
                        relative_error = error / reference.norm().clamp_min(1e-12)
                        # Chunked BF16 GEMMs round partial dW before FP32 accumulation.
                        assert relative_error < 0.01
            hook.remove()
    finally:
        Utils.destroy_model_parallel()
