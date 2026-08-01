from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.models.common.embeddings import rope_utils
from megatron.plugin.slideformer import (
    MegatronSlideFormerConfig,
    MegatronSlideFormerEngine,
    MegatronSlideFormerEngineConfig,
    apply_true_megatron_slideformer,
    kernels,
)
from megatron.plugin.slideformer.kernels import (
    apply_kernel_policy,
    legacy_fused_linear_cross_entropy,
    liger_fused_linear_cross_entropy,
    prepare_kernel_policy,
)
from megatron.plugin.slideformer.layout import (
    assert_trainable_parameter_coverage,
    resolve_megatron_decoder_layout,
)


class ToyTELinear(nn.Linear):
    """Linear-shaped test double exposing the TE fused-wgrad interface."""

    weight_names = ("weight",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fuse_wgrad_accumulation = False


class ToyMegatronModel(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 2, *, te_like: bool = False) -> None:
        super().__init__()
        linear_cls = ToyTELinear if te_like else nn.Linear
        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    linear_cls(hidden_size, hidden_size),
                    nn.GELU(),
                    linear_cls(hidden_size, hidden_size),
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder.final_layernorm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.decoder.layers:
            x = layer(x)
        x = self.decoder.final_layernorm(x)
        return self.output_layer(x)


class ToySharedOutput(nn.Module):
    def forward(self, inputs: torch.Tensor, *, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, weight)


class ToyMegatronTiedModel(nn.Module):
    """MCore-style tying passes embedding weight into a weightless output layer."""

    share_embeddings_and_output_weights = True

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(hidden_size, hidden_size)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])
        self.decoder.final_layernorm = nn.LayerNorm(hidden_size)
        self.output_layer = ToySharedOutput()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        for layer in self.decoder.layers:
            hidden = layer(hidden)
        hidden = self.decoder.final_layernorm(hidden)
        return self.output_layer(hidden, weight=self.embedding.weight)


def test_config_reads_megatron_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ENABLE", "1")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD", "true")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_PARAM_PREFETCH", "0")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_CPU_GRAD_BUFFER_COUNT", "3")

    config = MegatronSlideFormerConfig.from_env()

    assert config.enable is True
    assert config.activation_offload is True
    assert config.param_prefetch is False
    assert config.activation_backend == "slideformer-slot"
    assert config.unified_h2d_scheduler is True
    assert config.activation_slot_prefetch is True
    assert config.max_outstanding_h2d == 3
    assert config.shared_cpu_buffers is True
    assert config.cpu_grad_buffer_count == 3
    assert config.cpu_param_staging_buffer_count == 1
    assert config.te_fused_main_grad is True
    assert config.kernel_policy == "auto"
    assert config.attention_backend == "auto"
    assert config.mlp_backend == "auto"
    assert config.split_swiglu_threshold_gib == 0.5
    assert config.loss_backend == "auto"
    assert config.norm_backend == "auto"
    assert config.rope_backend == "auto"


def test_config_reads_kernel_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_KERNEL_POLICY", "auto")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ATTENTION_BACKEND", "megatron")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_MLP_BACKEND", "liger")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_LOSS_BACKEND", "megatron")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_NORM_BACKEND", "liger")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ROPE_BACKEND", "flash")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_STRICT_KERNELS", "0")

    config = MegatronSlideFormerConfig.from_env()

    assert config.attention_backend == "megatron"
    assert config.mlp_backend == "liger"
    assert config.loss_backend == "megatron"
    assert config.norm_backend == "liger"
    assert config.rope_backend == "flash"
    assert config.strict_kernels is False


def test_config_rejects_invalid_shared_cpu_buffer_counts() -> None:
    config = replace(
        MegatronSlideFormerConfig(enable=True), activation_offload=True, cpu_grad_buffer_count=-1
    )

    with pytest.raises(ValueError, match="CPU grad buffer count"):
        config.validate(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


def test_prebuild_kernel_policy_enables_native_swiglu_and_flash_rope(monkeypatch) -> None:
    class Args:
        swiglu = True
        bias_swiglu_fusion = False
        position_embedding_type = "rope"
        apply_rope_fusion = True

    monkeypatch.setattr(kernels, "flash_attn_func", object())
    from megatron.core.models.common.embeddings import rope_utils

    monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb", None)
    monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb_thd", None)
    monkeypatch.setattr(rope_utils, "apply_rotary_emb_flash", object())
    args = Args()

    config = replace(
        MegatronSlideFormerConfig(),
        attention_backend="flash",
        mlp_backend="liger",
        norm_backend="liger",
        rope_backend="flash",
    )
    report = prepare_kernel_policy(args, config)

    assert args.bias_swiglu_fusion is True
    assert args.apply_rope_fusion is False
    assert report == {"megatron_fused_swiglu": True, "rope_prebuild_backend": "flash_attention"}


def test_default_policy_splits_oversized_te_swiglu(monkeypatch) -> None:
    from megatron.core.extensions import transformer_engine

    monkeypatch.setattr(transformer_engine, "HAVE_TE", True)
    monkeypatch.setattr(kernels, "flash_attn_func", object())
    args = SimpleNamespace(
        swiglu=True,
        bias_swiglu_fusion=False,
        position_embedding_type="rope",
        apply_rope_fusion=False,
        reset_attention_mask=False,
        add_bias_linear=False,
        normalization="RMSNorm",
        micro_batch_size=64,
        seq_length=1024,
        ffn_hidden_size=17408,
        bf16=True,
        fp16=False,
    )

    report = prepare_kernel_policy(args, MegatronSlideFormerConfig())

    assert args._slideformer_split_te_swiglu is True
    assert report["mlp_prebuild_backend"] == "split_te_swiglu"
    args.micro_batch_size = 8
    report = prepare_kernel_policy(args, MegatronSlideFormerConfig())
    assert args._slideformer_split_te_swiglu is True
    assert report["mlp_prebuild_backend"] == "split_te_swiglu"
    args.micro_batch_size = 4
    report = prepare_kernel_policy(args, MegatronSlideFormerConfig())
    assert args._slideformer_split_te_swiglu is False
    assert "mlp_prebuild_backend" not in report
    args.micro_batch_size = 64
    args.normalization = "LayerNorm"
    report = prepare_kernel_policy(args, MegatronSlideFormerConfig())
    assert args._slideformer_split_te_swiglu is False
    assert "mlp_prebuild_backend" not in report


def test_split_te_swiglu_preserves_concatenated_weight_numerics(monkeypatch) -> None:
    class FakeLigerSiLUMulFunction:
        @staticmethod
        def apply(gate, up):
            return F.silu(gate) * up

    class FusedNormFC1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(16, 8))
            self.layer_norm_weight = nn.Parameter(torch.randn(8))
            self.bias = None
            self.normalization = "RMSNorm"
            self.zero_centered_gamma = False
            self.eps = 1e-6

    class TupleLinear(nn.Linear):
        def forward(self, inputs):
            return super().forward(inputs), None

    class DenseSwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                gated_linear_unit=True,
                activation_func=F.silu,
                add_bias_linear=False,
                tensor_model_parallel_size=1,
                layernorm_epsilon=1e-6,
                hidden_dropout=0.0,
            )
            self.linear_fc1 = FusedNormFC1()
            self.linear_fc2 = TupleLinear(8, 8, bias=False)

        def forward(self, hidden_states, **_):
            normalized = F.rms_norm(
                hidden_states,
                (8,),
                self.linear_fc1.layer_norm_weight,
                self.config.layernorm_epsilon,
            )
            intermediate = F.linear(normalized, self.linear_fc1.weight)
            gate, up = torch.chunk(intermediate, 2, dim=-1)
            output, _ = self.linear_fc2(F.silu(gate) * up)
            return output, None

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = DenseSwiGLU()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                transformer_impl="transformer_engine",
                attention_backend=kernels.AttnBackend.auto,
                gated_linear_unit=True,
                normalization="RMSNorm",
                mtp_num_layers=None,
                use_mup=False,
            )
            self.embedding = nn.Embedding(8, 8)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([Layer()])
            self.decoder.final_layernorm = nn.RMSNorm(8)
            self.output_layer = nn.Linear(8, 8, bias=False)
            self.position_embedding_type = "rope"

    monkeypatch.setattr(kernels, "LigerSiLUMulFunction", FakeLigerSiLUMulFunction)
    model = Model()
    inputs = torch.randn(3, 2, 8, requires_grad=True)
    expected = model.decoder.layers[0].mlp(inputs)[0]
    runtime_args = SimpleNamespace(_slideformer_split_te_swiglu=True)

    report = apply_kernel_policy(
        model, replace(MegatronSlideFormerConfig(), strict_kernels=False), runtime_args=runtime_args
    )
    actual = model.decoder.layers[0].mlp(inputs)[0]

    assert report["mlp"] == {
        "requested": "auto",
        "effective": "split_te_swiglu",
        "patched_layers": 1,
    }
    torch.testing.assert_close(actual, expected)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected_input_grad = inputs.grad.detach().clone()
    expected_fc1_grad = model.decoder.layers[0].mlp.linear_fc1.weight.grad.detach().clone()
    fc2_weight = model.decoder.layers[0].mlp.linear_fc2.weight
    expected_fc2_grad = fc2_weight.grad.detach().clone()

    inputs.grad = None
    model.decoder.layers[0].mlp.linear_fc1.weight.grad = None
    fc2_weight.grad = None
    fc2_weight.main_grad = torch.empty_like(fc2_weight)
    with kernels.split_te_recompute_early_stop():
        recomputed = model.decoder.layers[0].mlp(inputs)[0]
    assert torch.count_nonzero(recomputed) == 0
    recomputed.backward(grad_output)

    torch.testing.assert_close(inputs.grad, expected_input_grad)
    torch.testing.assert_close(
        model.decoder.layers[0].mlp.linear_fc1.weight.grad, expected_fc1_grad
    )
    torch.testing.assert_close(fc2_weight.main_grad, expected_fc2_grad)
    assert fc2_weight.grad is None


def test_default_prebuild_kernel_policy_selects_transformer_engine(monkeypatch) -> None:
    from megatron.core.extensions import transformer_engine
    from megatron.core.transformer.enums import AttnBackend

    monkeypatch.setattr(transformer_engine, "HAVE_TE", True)
    monkeypatch.setattr(kernels, "flash_attn_func", object())
    args = SimpleNamespace(
        swiglu=True,
        bias_swiglu_fusion=False,
        position_embedding_type="rope",
        apply_rope_fusion=False,
        reset_attention_mask=False,
        add_bias_linear=False,
        attention_backend=AttnBackend.auto,
    )

    report = prepare_kernel_policy(args, MegatronSlideFormerConfig())

    assert args.transformer_impl == "transformer_engine"
    assert args.use_te_activation_func is True
    assert args.bias_swiglu_fusion is False
    assert args.apply_rope_fusion is True
    assert args.attention_backend == AttnBackend.auto
    assert report["structural_backend"] == "transformer_engine"
    assert report["attention_prebuild_backend"] == "transformer_engine_auto"


def test_default_te_policy_keeps_bias_activation_fusion_for_biased_mlp(monkeypatch) -> None:
    from megatron.core.extensions import transformer_engine

    monkeypatch.setattr(transformer_engine, "HAVE_TE", True)
    args = SimpleNamespace(
        swiglu=True,
        bias_swiglu_fusion=True,
        position_embedding_type="rope",
        apply_rope_fusion=False,
        reset_attention_mask=False,
        add_bias_linear=True,
    )

    prepare_kernel_policy(args, MegatronSlideFormerConfig())

    assert args.transformer_impl == "transformer_engine"
    assert args.use_te_activation_func is False
    assert args.bias_swiglu_fusion is True


def test_default_te_policy_rejects_missing_flash_attention(monkeypatch) -> None:
    from megatron.core.extensions import transformer_engine

    monkeypatch.setattr(transformer_engine, "HAVE_TE", True)
    monkeypatch.setattr(kernels, "flash_attn_func", None)
    args = SimpleNamespace(reset_attention_mask=False, add_bias_linear=False)

    with pytest.raises(RuntimeError, match="requires flash-attn"):
        prepare_kernel_policy(args, MegatronSlideFormerConfig())


def test_prebuild_kernel_policy_rejects_reset_attention_mask() -> None:
    args = SimpleNamespace(reset_attention_mask=True)

    with pytest.raises(RuntimeError, match="reset-attention-mask"):
        prepare_kernel_policy(args, MegatronSlideFormerConfig())


def test_flash_attention_preserves_megatron_softmax_scaling(monkeypatch) -> None:
    call = {}

    def fake_flash_attention(query, key, value, **kwargs):
        call.update(kwargs)
        return query

    monkeypatch.setattr(kernels, "flash_attn_func", fake_flash_attention)
    config = SimpleNamespace(
        attention_dropout=0.0,
        softmax_scale=None,
        kv_channels=64,
        apply_query_key_layer_scaling=True,
        softmax_type="vanilla",
    )
    core = kernels.FlashAttentionCore(
        config, layer_number=2, attn_mask_type=kernels.AttnMaskType.causal
    )
    query = torch.randn(4, 2, 3, 64)

    core(query, query, query, attention_mask=None)

    assert call["softmax_scale"] == pytest.approx(1 / 16)


def test_disabled_kernel_policy_does_not_patch_model() -> None:
    model = ToyMegatronModel()
    config = replace(MegatronSlideFormerConfig(), kernel_policy="off")

    report = apply_kernel_policy(model, config)

    assert report["attention"]["effective"] == "megatron"
    assert report["mlp"]["effective"] == "megatron"
    assert report["loss"]["effective"] == "megatron"


def test_liger_norm_policy_discovers_qwen3_qk_norms(monkeypatch) -> None:
    class FakeLigerRMSNormFunction:
        @staticmethod
        def apply(hidden_states, weight, eps, *_):
            return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight, eps)

    class Qwen3LikeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = nn.RMSNorm(8)
            self.pre_mlp_layernorm = nn.RMSNorm(8)
            self.self_attention = nn.Module()
            self.self_attention.q_layernorm = nn.RMSNorm(4)
            self.self_attention.k_layernorm = nn.RMSNorm(4)

    class Qwen3LikeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([Qwen3LikeLayer()])
            self.decoder.final_layernorm = nn.RMSNorm(8)

    monkeypatch.setattr(kernels, "LigerRMSNormFunction", FakeLigerRMSNormFunction)
    model = Qwen3LikeModel()
    config = replace(
        MegatronSlideFormerConfig(),
        attention_backend="megatron",
        mlp_backend="megatron",
        loss_backend="megatron",
        norm_backend="liger",
        rope_backend="megatron",
    )

    report = apply_kernel_policy(model, config)

    assert report["norm"] == {
        "requested": "liger",
        "effective": "liger_rms_norm",
        "patched_modules": 5,
    }
    for module in model.modules():
        if isinstance(module, nn.RMSNorm):
            assert module._slideformer_liger_backend == "rms_norm"


@pytest.mark.skipif(
    not torch.cuda.is_available() or rope_utils.apply_rotary_emb_flash is None,
    reason="FlashAttention rotary requires CUDA and flash-attn",
)
def test_flash_rope_adapter_matches_megatron_reference() -> None:
    sequence, batch, heads, head_dim = 16, 2, 4, 32
    hidden_states = torch.randn(
        sequence, batch, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    half_freqs = torch.randn(sequence, 1, 1, head_dim // 2, device="cuda", dtype=torch.float32)
    freqs = torch.cat((half_freqs, half_freqs), dim=-1)
    reference_config = SimpleNamespace(
        apply_rope_fusion=False, rotary_interleaved=False, _slideformer_flash_rope=False
    )
    flash_config = SimpleNamespace(
        apply_rope_fusion=False, rotary_interleaved=False, _slideformer_flash_rope=True
    )
    cp_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)

    reference = rope_utils.apply_rotary_pos_emb(
        hidden_states, freqs, reference_config, cp_group=cp_group
    )
    actual = rope_utils.apply_rotary_pos_emb(hidden_states, freqs, flash_config, cp_group=cp_group)

    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)


def test_liger_lce_preserves_megatron_loss_layout_and_gradients(monkeypatch) -> None:
    class FakeLigerLoss:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, weight, inputs, target, bias=None):
            return F.cross_entropy(
                F.linear(inputs, weight, bias),
                target,
                ignore_index=self.kwargs["ignore_index"],
                reduction=self.kwargs["reduction"],
            )

    class OutputLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(7, 4))
            self.bias = None

        def forward(self, inputs, weight=None, **kwargs):
            del kwargs
            return F.linear(inputs, weight if weight is not None else self.weight), None

    monkeypatch.setattr(kernels, "LigerFusedLinearCrossEntropyLoss", FakeLigerLoss)
    output_layer = OutputLayer()
    hidden_states = torch.randn(3, 2, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2], [3, -100, 4]])

    actual = liger_fused_linear_cross_entropy(
        hidden_states=hidden_states,
        output_layer=output_layer,
        output_weight=None,
        labels=labels,
        runtime_gather_output=True,
    )
    expected = F.cross_entropy(
        F.linear(hidden_states, output_layer.weight).transpose(0, 1).reshape(-1, 7),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)

    assert actual.shape == labels.shape
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert hidden_states.grad is not None
    assert output_layer.weight.grad is not None


def test_legacy_lce_preserves_megatron_masked_loss_layout(monkeypatch) -> None:
    class FakeLegacyLoss:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, weight, inputs, target, bias=None):
            return F.cross_entropy(
                F.linear(inputs, weight, bias),
                target,
                ignore_index=self.kwargs["ignore_index"],
                reduction=self.kwargs["reduction"],
            )

    class OutputLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(7, 4))
            self.bias = None

        def forward(self, inputs, weight=None, **kwargs):
            del kwargs
            return F.linear(inputs, weight if weight is not None else self.weight), None

    monkeypatch.setattr(kernels, "LegacyFusedLinearCrossEntropyLoss", FakeLegacyLoss)
    output_layer = OutputLayer()
    hidden_states = torch.randn(3, 2, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2], [3, 4, 5]])
    loss_mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)

    actual = legacy_fused_linear_cross_entropy(
        hidden_states=hidden_states,
        output_layer=output_layer,
        output_weight=None,
        labels=labels,
        loss_mask=loss_mask,
        runtime_gather_output=True,
    )
    expected = F.cross_entropy(
        F.linear(hidden_states, output_layer.weight).transpose(0, 1).reshape(-1, 7),
        labels.masked_fill(~loss_mask.bool(), -100).reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    assert actual.shape == labels.shape
    torch.testing.assert_close((actual * loss_mask).sum(), expected)
    (actual * loss_mask).sum().backward()
    assert hidden_states.grad is not None
    assert output_layer.weight.grad is not None


def test_legacy_lce_fully_masked_batch_remains_backward_safe(monkeypatch) -> None:
    monkeypatch.setattr(kernels, "LegacyFusedLinearCrossEntropyLoss", object())
    hidden_states = torch.randn(3, 2, 4, requires_grad=True)

    actual = legacy_fused_linear_cross_entropy(
        hidden_states=hidden_states,
        output_layer=torch.nn.Linear(4, 7, bias=False),
        output_weight=None,
        labels=torch.zeros(2, 3, dtype=torch.long),
        loss_mask=torch.zeros(2, 3),
        runtime_gather_output=True,
    )

    assert actual.shape == (2, 3)
    assert actual.sum() == 0
    actual.sum().backward()
    assert hidden_states.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or kernels.LegacyFusedLinearCrossEntropyLoss is None,
    reason="Legacy LCE numerical test requires CUDA and liger-kernel",
)
def test_legacy_lce_matches_cross_entropy_on_cuda() -> None:
    class OutputLayer(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.bias = None

        def forward(self, inputs, weight=None, **kwargs):
            del kwargs
            return F.linear(inputs, weight if weight is not None else self.weight), None

    torch.manual_seed(7)
    hidden = torch.randn(5, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(31, 16, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 31, (2, 5), device="cuda")
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    loss_mask[0, -1] = 0
    output_layer = OutputLayer(weight.clone())
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.clone().requires_grad_(True)

    actual = legacy_fused_linear_cross_entropy(
        hidden_states=hidden,
        output_layer=output_layer,
        output_weight=None,
        labels=labels,
        loss_mask=loss_mask,
        runtime_gather_output=True,
    ).sum()
    reference = F.cross_entropy(
        F.linear(reference_hidden, reference_weight).transpose(0, 1).reshape(-1, 31).float(),
        labels.masked_fill(~loss_mask.bool(), -100).reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    actual.backward()
    reference.backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(
        output_layer.weight.grad, reference_weight.grad, atol=2e-2, rtol=2e-2
    )


def test_engine_checkpoint_round_trips_scheduler_state(tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self):
            self.loaded = None

        def state_dict(self):
            return {"engine_step": 4}

        def load_state_dict(self, state):
            self.loaded = state

    class FakeScheduler:
        def __init__(self):
            self.loaded = None

        def state_dict(self):
            return {"num_steps": 256}

        def load_state_dict(self, state):
            self.loaded = state

    engine = FakeEngine()
    scheduler = FakeScheduler()

    path = MegatronSlideFormerEngine.save_checkpoint(
        engine, tmp_path, 4, opt_param_scheduler=scheduler
    )
    iteration = MegatronSlideFormerEngine.load_checkpoint(
        engine, tmp_path, opt_param_scheduler=scheduler
    )

    assert path == tmp_path / "slideformer" / "iter_0000004.pt"
    assert iteration == 4
    assert engine.loaded == {"engine_step": 4}
    assert scheduler.loaded == {"num_steps": 256}


def test_layout_resolves_gpt_like_structure() -> None:
    model = ToyMegatronModel()

    layout = resolve_megatron_decoder_layout(model)

    assert layout.embedding is model.embedding
    assert layout.decoder is model.decoder
    assert list(layout.layers) == list(model.decoder.layers)
    assert layout.final_norm is model.decoder.final_layernorm
    assert layout.output_layer is model.output_layer
    assert_trainable_parameter_coverage(model, layout)


def test_layout_fails_on_uncovered_trainable_parameter() -> None:
    model = ToyMegatronModel()
    model.extra_projection = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="extra_projection"):
        assert_trainable_parameter_coverage(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SlideFormer engine requires CUDA")
def test_mcore_style_tied_embedding_stays_resident_through_output() -> None:
    model = ToyMegatronTiedModel().cuda()
    engine = apply_true_megatron_slideformer(
        model,
        config=MegatronSlideFormerEngineConfig(
            activation_offload=False, offload_after_forward=True, prefetch=True
        ),
    )
    try:
        embedding_owner = next(
            owner for owner in engine.managed_layers if owner.layer is model.embedding
        )
        assert engine._tied_embedding_output is True
        assert embedding_owner.keep_loaded_after_forward is True
        assert embedding_owner.gpu_param_pool is not None
        assert len(embedding_owner.gpu_param_pool.tensors) == 3

        engine.zero_unmanaged_grads()
        loss = model(torch.arange(8, device="cuda").view(2, 4)).float().pow(2).mean()
        assert model.embedding.weight.is_cuda
        loss.backward()
        engine.step_unmanaged_params()
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SlideFormer engine requires CUDA")
def test_slideformer_engine_accepts_cpu_initialized_model() -> None:
    model = ToyMegatronModel()
    assert all(not parameter.is_cuda for parameter in model.parameters())
    engine = apply_true_megatron_slideformer(
        model,
        config=MegatronSlideFormerEngineConfig(
            activation_offload=False, offload_after_forward=True, prefetch=True
        ),
    )
    try:
        engine.zero_unmanaged_grads()
        loss = model(torch.randn(2, 4, 8, device="cuda")).float().pow(2).mean()
        loss.backward()
        engine.step_unmanaged_params()
        assert engine.traffic_counters.get("parameter_sync_h2d_count", 0) == 0
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SlideFormer engine requires CUDA")
def test_slideformer_engine_matches_baseline_single_step() -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    baseline = ToyMegatronModel().to(device)
    slideformer = ToyMegatronModel(te_like=True).to(device)
    slideformer.load_state_dict(baseline.state_dict())
    x = torch.randn(2, 4, 8, device=device)

    baseline_optimizer = torch.optim.Adam(
        baseline.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_loss = baseline(x).float().pow(2).mean()
    baseline_loss.backward()
    baseline_optimizer.step()

    engine = apply_true_megatron_slideformer(
        slideformer,
        config=MegatronSlideFormerEngineConfig(
            lr=2e-4,
            weight_decay=0.0,
            activation_offload=True,
            offload_after_forward=True,
            prefetch=True,
            te_fused_main_grad=True,
        ),
    )
    try:
        memory = engine.cpu_memory_summary()["bytes"]
        max_owner_numel = max(
            sum(param.numel() for param in owner.params) for owner in engine.managed_layers
        )
        assert memory["gradients"] == 2 * max_owner_numel * torch.float32.itemsize
        assert memory["execution_parameter_staging"] == max_owner_numel * torch.float32.itemsize
        transformer_owners = [
            owner for owner in engine.managed_layers if owner.is_transformer_layer
        ]
        assert all(owner._te_fused_params for owner in transformer_owners)
        assert all(
            param.overwrite_main_grad
            for owner in transformer_owners
            for param in owner._te_fused_params
        )
        owners_with_params = [owner for owner in engine.managed_layers if owner.params]
        parameter_pools = {id(owner.gpu_param_pool) for owner in owners_with_params}
        assert len(parameter_pools) == 1
        parameter_pool = owners_with_params[0].gpu_param_pool
        assert parameter_pool is not None
        assert len(parameter_pool.tensors) == 2
        assert all(tensor.numel() == max_owner_numel for tensor in parameter_pool.tensors)
        output_owner = next(
            owner for owner in engine.managed_layers if owner.layer is slideformer.output_layer
        )
        assert output_owner.keep_loaded_after_forward is True

        engine.zero_unmanaged_grads()
        assert engine.managed_layers[0].prefetched is True
        slideformer_loss = slideformer(x).float().pow(2).mean()
        assert output_owner.loaded is True
        assert output_owner._gpu_param_pool_slot is not None
        slideformer_loss.backward()
        engine.step_unmanaged_params()

        assert engine.traffic_counters.get("parameter_sync_h2d_count", 0) == 0
        assert torch.allclose(slideformer_loss, baseline_loss, atol=1e-6, rtol=1e-6)
        authoritative_params = {
            id(param): owner.cpu_params[param]
            for owner in engine.managed_layers
            for param in owner.params
        }
        for name, baseline_param in baseline.named_parameters():
            slideformer_param = dict(slideformer.named_parameters())[name]
            slideformer_value = authoritative_params.get(id(slideformer_param), slideformer_param)
            assert torch.allclose(
                slideformer_value.detach().cpu(),
                baseline_param.detach().cpu(),
                atol=5e-6,
                rtol=5e-6,
            )
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SlideFormer engine requires CUDA")
def test_async_grad_copy_completes_before_parameter_storage_is_released() -> None:
    device = torch.device("cuda")
    model = ToyMegatronModel().to(device)
    engine = apply_true_megatron_slideformer(
        model,
        config=MegatronSlideFormerEngineConfig(
            activation_offload=True, offload_after_forward=True, overlap_grad_d2h_cpu_adam=True
        ),
    )
    owner = next(item for item in engine.managed_layers if item.params)
    original_event = owner.grad_copy_event
    original_offload = owner.offload_params
    calls: list[str] = []

    class RecordingEvent:
        def record(self, stream: torch.cuda.Stream) -> None:
            del stream
            calls.append("record")

        def synchronize(self) -> None:
            calls.append("synchronize")

    def record_offload(*, force: bool = False) -> None:
        assert force
        calls.append("offload")

    try:
        owner.grad_copy_event = RecordingEvent()
        owner.offload_params = record_offload
        owner._grad_seen = set(owner.params)
        owner._grad_copy_submitted = True

        owner.prepare_grads_for_async_step()

        assert calls == ["record", "synchronize", "offload"]
    finally:
        owner.grad_copy_event = original_event
        owner.offload_params = original_offload
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SlideFormer engine requires CUDA")
def test_slideformer_checkpoint_round_trip() -> None:
    torch.manual_seed(5678)
    device = torch.device("cuda")
    model = ToyMegatronModel().to(device)
    restored = ToyMegatronModel().to(device)
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 4, 8, device=device)

    engine = apply_true_megatron_slideformer(
        model,
        config=MegatronSlideFormerEngineConfig(
            lr=2e-4,
            weight_decay=0.0,
            activation_offload=True,
            offload_after_forward=True,
            prefetch=True,
        ),
    )
    restored_engine = None
    try:
        engine.zero_unmanaged_grads()
        loss = model(x).float().pow(2).mean()
        loss.backward()
        engine.step_unmanaged_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = engine.save_checkpoint(tmpdir, iteration=3)
            assert path.exists()

            restored_engine = apply_true_megatron_slideformer(
                restored,
                config=MegatronSlideFormerEngineConfig(
                    lr=2e-4,
                    weight_decay=0.0,
                    activation_offload=True,
                    offload_after_forward=True,
                    prefetch=True,
                ),
            )
            assert restored_engine.load_checkpoint(tmpdir) == 3

        for name, param in model.named_parameters():
            restored_param = dict(restored.named_parameters())[name]
            assert torch.allclose(
                param.detach().cpu(), restored_param.detach().cpu(), atol=0, rtol=0
            )
    finally:
        engine.close()
        if restored_engine is not None:
            restored_engine.close()
