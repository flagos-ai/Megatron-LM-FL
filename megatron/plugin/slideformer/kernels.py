from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.plugin.slideformer.layout import resolve_megatron_decoder_layout

try:
    from megatron.plugin.slideformer.legacy_lce import LegacyFusedLinearCrossEntropyLoss
except ImportError:  # pragma: no cover - deployment dependent
    LegacyFusedLinearCrossEntropyLoss = None

try:
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - deployment dependent
    flash_attn_func = None

try:
    from liger_kernel.ops import LigerRMSNormFunction, LigerSiLUMulFunction
except ImportError:  # pragma: no cover - deployment dependent
    LigerRMSNormFunction = None
    LigerSiLUMulFunction = None

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:  # pragma: no cover - deployment dependent
    LigerFusedLinearCrossEntropyLoss = None


class FlashAttentionCore(nn.Module):
    """FlashAttention-2 adapter for dense single-GPU Megatron self-attention."""

    def __init__(self, config, layer_number: int, attn_mask_type: AttnMaskType) -> None:
        super().__init__()
        if flash_attn_func is None:
            raise RuntimeError("SlideFormer FlashAttention requires the flash-attn package")
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.softmax_scale = config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = config.kv_channels**-0.5
        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number
        if config.softmax_type != "vanilla":
            raise RuntimeError(
                "SlideFormer FlashAttention currently supports softmax_type='vanilla' only"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params=None,
        **_: Any,
    ) -> torch.Tensor:
        if attention_bias is not None or packed_seq_params is not None:
            raise NotImplementedError(
                "SlideFormer FlashAttention currently supports dense attention without bias"
            )
        mask_type = attn_mask_type or self.attn_mask_type
        causal = mask_type == AttnMaskType.causal
        if attention_mask is not None and not causal:
            if attention_mask.dtype != torch.bool or bool(attention_mask.any()):
                raise NotImplementedError(
                    "SlideFormer FlashAttention does not support arbitrary attention masks"
                )
        output = flash_attn_func(
            query.permute(1, 0, 2, 3),
            key.permute(1, 0, 2, 3),
            value.permute(1, 0, 2, 3),
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )
        output = output.permute(1, 0, 2, 3).contiguous()
        return output.view(output.shape[0], output.shape[1], -1)


def _validate_dense_swiglu(model: nn.Module) -> tuple[bool, str, int]:
    layout = resolve_megatron_decoder_layout(model)
    for layer in layout.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "linear_fc1") or not hasattr(mlp, "linear_fc2"):
            return False, "decoder contains a non-dense or unsupported MLP", 0
        config = mlp.config
        if not config.gated_linear_unit or config.activation_func is not F.silu:
            return False, "decoder MLP is not SwiGLU", 0
        if config.add_bias_linear:
            return False, "Liger fused SwiGLU requires add_bias_linear=False", 0
        if getattr(config, "tensor_model_parallel_size", 1) != 1:
            return False, "Liger fused SwiGLU currently requires TP=1", 0
    return True, "compatible dense SwiGLU", len(layout.layers)


def _liger_swiglu_compute(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    intermediate, fc1_bias = mlp.linear_fc1(hidden_states)
    if fc1_bias is not None:
        raise RuntimeError("SlideFormer Liger SwiGLU does not support FC1 bias")
    gate, up = torch.chunk(intermediate, 2, dim=-1)
    activated = LigerSiLUMulFunction.apply(gate, up)
    output, fc2_bias = mlp.linear_fc2(activated)
    if fc2_bias is not None:
        raise RuntimeError("SlideFormer Liger SwiGLU does not support FC2 bias")
    return output


def _liger_mlp_forward(self, hidden_states, per_token_scale=None, **kwargs):
    del kwargs
    if per_token_scale is not None:
        raise RuntimeError("SlideFormer Liger SwiGLU does not support per-token scaling")
    return _liger_swiglu_compute(self, hidden_states), None


def _split_te_mlp_forward(self, hidden_states, per_token_scale=None, **kwargs):
    del kwargs
    if per_token_scale is not None:
        raise RuntimeError("SlideFormer split SwiGLU does not support per-token scaling")
    fc1 = self.linear_fc1
    norm_weight = getattr(fc1, "layer_norm_weight", None)
    if norm_weight is None:
        raise RuntimeError("SlideFormer split SwiGLU requires TE fused FC1 RMSNorm")
    if getattr(fc1, "normalization", None) != "RMSNorm":
        raise RuntimeError("SlideFormer split SwiGLU requires TE fused FC1 RMSNorm")
    if getattr(fc1, "zero_centered_gamma", False):
        norm_weight = norm_weight + 1
    normalized = F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_weight, fc1.eps)
    fc1_weight = fc1.weight
    gate_weight, up_weight = torch.chunk(fc1_weight, 2, dim=0)
    fc1_bias = getattr(fc1, "bias", None)
    gate_bias = up_bias = None
    if fc1_bias is not None and fc1_bias.numel() > 0:
        gate_bias, up_bias = torch.chunk(fc1_bias, 2, dim=0)
    gate = F.linear(normalized, gate_weight, gate_bias)
    up = F.linear(normalized, up_weight, up_bias)
    activated = LigerSiLUMulFunction.apply(gate, up)
    output, fc2_bias = self.linear_fc2(activated)
    return output, fc2_bias


def apply_split_te_swiglu(model: nn.Module) -> int:
    """Split TE's concatenated Qwen FC1 into gate/up GEMMs without changing weights."""

    if LigerSiLUMulFunction is None:
        raise RuntimeError("SlideFormer split SwiGLU requires the liger-kernel package")
    compatible, reason, _ = _validate_dense_swiglu(model)
    if not compatible:
        raise RuntimeError(reason)
    patched = 0
    for layer in resolve_megatron_decoder_layout(model).layers:
        mlp = layer.mlp
        if (
            not hasattr(mlp.linear_fc1, "layer_norm_weight")
            or getattr(mlp.linear_fc1, "normalization", None) != "RMSNorm"
        ):
            raise RuntimeError("SlideFormer split SwiGLU requires TE fused FC1 RMSNorm")
        if hasattr(mlp, "_slideformer_liger_original_forward"):
            continue
        mlp._slideformer_liger_original_forward = mlp.forward
        mlp._slideformer_liger_backend = "split_te"
        mlp.forward = MethodType(_split_te_mlp_forward, mlp)
        patched += 1
    return patched


def _should_split_te_swiglu(args, config) -> bool:
    threshold_gib = config.split_swiglu_threshold_gib
    if threshold_gib <= 0 or config.mlp_backend != "auto":
        return False
    if not getattr(args, "swiglu", False) or getattr(args, "add_bias_linear", True):
        return False
    if getattr(args, "normalization", None) != "RMSNorm" or getattr(args, "fp8", None):
        return False
    micro_batch_size = getattr(args, "micro_batch_size", None)
    seq_length = getattr(args, "seq_length", None)
    ffn_hidden_size = getattr(args, "ffn_hidden_size", None)
    if not micro_batch_size or not seq_length or not ffn_hidden_size:
        return False
    element_size = 2 if getattr(args, "bf16", False) or getattr(args, "fp16", False) else 4
    fc1_output_bytes = micro_batch_size * seq_length * 2 * ffn_hidden_size * element_size
    return fc1_output_bytes >= threshold_gib * 1024**3


def apply_liger_swiglu(model: nn.Module) -> int:
    if LigerSiLUMulFunction is None:
        raise RuntimeError("SlideFormer fused SwiGLU requires the liger-kernel package")
    compatible, reason, _ = _validate_dense_swiglu(model)
    if not compatible:
        raise RuntimeError(reason)
    patched = 0
    for layer in resolve_megatron_decoder_layout(model).layers:
        mlp = layer.mlp
        if hasattr(mlp, "_slideformer_liger_original_forward"):
            continue
        mlp._slideformer_liger_original_forward = mlp.forward
        mlp._slideformer_liger_backend = "fused"
        mlp.forward = MethodType(_liger_mlp_forward, mlp)
        patched += 1
    return patched


def _iter_rms_norms(model: nn.Module):
    layout = resolve_megatron_decoder_layout(model)
    roots = [*layout.layers]
    if layout.final_norm is not None:
        roots.append(layout.final_norm)
    seen: set[int] = set()
    for root in roots:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if isinstance(module, nn.RMSNorm):
                yield module


def _liger_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return LigerRMSNormFunction.apply(
        hidden_states, self.weight, self.eps, 0.0, "llama", True, None
    )


def apply_liger_rms_norm(model: nn.Module) -> int:
    """Replace local PyTorch RMSNorm modules without changing model structure."""

    if LigerRMSNormFunction is None:
        raise RuntimeError("SlideFormer fused RMSNorm requires the liger-kernel package")
    patched = 0
    for module in _iter_rms_norms(model):
        if hasattr(module, "_slideformer_liger_original_forward"):
            continue
        module._slideformer_liger_original_forward = module.forward
        module._slideformer_liger_backend = "rms_norm"
        module.forward = MethodType(_liger_rms_norm_forward, module)
        patched += 1
    return patched


def apply_flash_attention(model: nn.Module) -> int:
    if flash_attn_func is None:
        raise RuntimeError("SlideFormer FlashAttention requires the flash-attn package")
    patched = 0
    for index, layer in enumerate(resolve_megatron_decoder_layout(model).layers):
        attention = getattr(layer, "self_attention", None)
        core_attention = getattr(attention, "core_attention", None)
        if attention is None or core_attention is None:
            raise RuntimeError("decoder layer has no replaceable self-attention core")
        if isinstance(core_attention, FlashAttentionCore):
            continue
        mask_type = getattr(
            core_attention,
            "attn_mask_type",
            getattr(attention, "attn_mask_type", AttnMaskType.causal),
        )
        replacement = FlashAttentionCore(
            getattr(attention, "config", layer.config),
            getattr(attention, "layer_number", index + 1),
            mask_type,
        )
        replacement.train(core_attention.training)
        attention.core_attention = replacement
        patched += 1
    return patched


def prepare_kernel_policy(args, config) -> dict[str, Any]:
    """Apply kernel choices that must be decided before the model spec is built."""

    report: dict[str, Any] = {}
    if config.kernel_policy == "off":
        return report

    structural_backends = {
        config.attention_backend,
        config.mlp_backend,
        config.norm_backend,
        config.rope_backend,
    }
    uses_te_prebuild = structural_backends == {"auto"}
    if uses_te_prebuild:
        from megatron.core.extensions.transformer_engine import HAVE_TE

        if not HAVE_TE:
            raise RuntimeError(
                "SlideFormer auto kernel policy requires Transformer Engine; "
                "install the Megatron-pinned TE build or select explicit local backends"
            )
        args.transformer_impl = "transformer_engine"
        # TransformerConfig rejects TE activation together with Megatron's
        # bias-activation fusion. Qwen-family SwiGLU is bias-free and uses the
        # TE activation; biased MLPs keep Megatron's fused bias activation.
        args.use_te_activation_func = not getattr(args, "add_bias_linear", True)
        if args.use_te_activation_func:
            args.bias_swiglu_fusion = False
        if flash_attn_func is None and config.strict_kernels:
            raise RuntimeError(
                "SlideFormer auto kernel policy requires flash-attn for the default "
                "Transformer Engine attention backend"
            )
        if flash_attn_func is not None:
            # Leave TE dispatch on auto: it selects FlashAttention 2 for the
            # supported Qwen shape while retaining TE's compatibility checks.
            report["attention_prebuild_backend"] = "transformer_engine_auto"
        if getattr(args, "position_embedding_type", None) == "rope":
            args.apply_rope_fusion = True
        report["structural_backend"] = "transformer_engine"
        split_te_swiglu = _should_split_te_swiglu(args, config)
        setattr(args, "_slideformer_split_te_swiglu", split_te_swiglu)
        if split_te_swiglu:
            report["mlp_prebuild_backend"] = "split_te_swiglu"

    if config.attention_backend != "megatron" and getattr(args, "reset_attention_mask", False):
        raise RuntimeError(
            "SlideFormer FlashAttention does not support --reset-attention-mask; "
            "set attention_backend=megatron explicitly"
        )

    if not uses_te_prebuild and config.mlp_backend != "megatron" and getattr(args, "swiglu", False):
        args.bias_swiglu_fusion = True
        report["megatron_fused_swiglu"] = True

    if (
        config.rope_backend != "megatron"
        and getattr(args, "position_embedding_type", None) == "rope"
    ):
        from megatron.core.models.common.embeddings import rope_utils

        te_rope_available = (
            rope_utils.fused_apply_rotary_pos_emb is not None
            or rope_utils.fused_apply_rotary_pos_emb_thd is not None
        )
        flash_rope_available = rope_utils.apply_rotary_emb_flash is not None
        if not te_rope_available and flash_rope_available:
            # TransformerConfig validates TE RoPE during construction. Disable
            # that unavailable backend now; the post-build policy enables the
            # FlashAttention rotary kernel on the model config.
            args.apply_rope_fusion = False
            report["rope_prebuild_backend"] = "flash_attention"
    return report


def liger_fused_linear_cross_entropy(
    *,
    hidden_states: torch.Tensor,
    output_layer: nn.Module,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor,
    runtime_gather_output: bool | None,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    del loss_mask
    if LigerFusedLinearCrossEntropyLoss is None:
        raise RuntimeError("SlideFormer fused linear cross entropy requires liger-kernel")
    if runtime_gather_output is False:
        raise RuntimeError("SlideFormer Liger LCE currently requires TP=1 gathered output")
    labels_t = labels.transpose(0, 1).contiguous()
    loss_fn = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="none", accum_dtype=torch.float32
    )
    original_forward = output_layer.forward

    def fused_forward(inputs, *args, weight=None, **kwargs):
        del args, kwargs
        linear_weight = weight if weight is not None else output_weight
        if linear_weight is None:
            linear_weight = output_layer.weight
        loss = loss_fn(
            linear_weight,
            inputs.reshape(-1, inputs.shape[-1]),
            labels_t.reshape(-1),
            bias=getattr(output_layer, "bias", None),
        ).view_as(labels_t)
        # GPTModel returns token losses as [batch, sequence], matching labels.
        return loss.transpose(0, 1).contiguous()

    output_layer.forward = fused_forward
    try:
        return output_layer(hidden_states, weight=output_weight)
    finally:
        output_layer.forward = original_forward


def legacy_fused_linear_cross_entropy(
    *,
    hidden_states: torch.Tensor,
    output_layer: nn.Module,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor,
    runtime_gather_output: bool | None,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run scalar Legacy LCE while preserving Megatron's per-token loss API.

    The loss mask is converted to ignore labels before LCE.  The resulting
    scalar sum is placed at one valid token, so Megatron's existing loss
    reduction and token-count reporting remain unchanged.
    """

    if runtime_gather_output is False:
        raise RuntimeError("SlideFormer Legacy LCE currently requires TP=1 gathered output")
    if LegacyFusedLinearCrossEntropyLoss is None:
        raise RuntimeError("SlideFormer Legacy LCE requires the liger-kernel package")
    labels_t = labels.transpose(0, 1).contiguous()
    if loss_mask is not None:
        mask_t = loss_mask.transpose(0, 1).contiguous().bool()
        labels_t = labels_t.masked_fill(~mask_t, -100)
    valid = labels_t != -100
    if not bool(valid.any()):
        # Keep the zero loss connected to the graph so a fully masked local
        # batch remains backward-safe.
        return hidden_states[..., 0].transpose(0, 1).to(torch.float32) * 0.0

    loss_fn = LegacyFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum")
    original_forward = output_layer.forward

    def fused_forward(inputs, *args, weight=None, **kwargs):
        del args, kwargs
        linear_weight = weight if weight is not None else output_weight
        if linear_weight is None:
            linear_weight = output_layer.weight
        return loss_fn(
            linear_weight,
            inputs.reshape(-1, inputs.shape[-1]),
            labels_t.reshape(-1),
            bias=getattr(output_layer, "bias", None),
        )

    output_layer.forward = fused_forward
    try:
        scalar_loss = output_layer(hidden_states, weight=output_weight)
    finally:
        output_layer.forward = original_forward

    anchor = valid.reshape(-1).to(torch.int64).argmax().reshape(1)
    token_losses = torch.zeros(
        labels_t.numel(), dtype=scalar_loss.dtype, device=scalar_loss.device
    ).scatter(0, anchor, scalar_loss.reshape(1))
    return token_losses.view_as(labels_t).transpose(0, 1).contiguous()


def apply_kernel_policy(model: nn.Module, config, *, runtime_args=None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "policy": config.kernel_policy,
        "attention": {"requested": config.attention_backend, "effective": "megatron"},
        "mlp": {"requested": config.mlp_backend, "effective": "megatron"},
        "loss": {"requested": config.loss_backend, "effective": "megatron"},
        "norm": {"requested": config.norm_backend, "effective": "megatron"},
        "rope": {"requested": config.rope_backend, "effective": "megatron"},
    }
    if config.kernel_policy == "off":
        return report

    layout = resolve_megatron_decoder_layout(model)
    model_config = getattr(layout.model, "config", None)
    uses_transformer_engine = (
        getattr(model_config, "transformer_impl", None) == "transformer_engine"
    )
    split_te_swiglu = bool(
        runtime_args is not None and getattr(runtime_args, "_slideformer_split_te_swiglu", False)
    )
    if split_te_swiglu:
        count = apply_split_te_swiglu(model)
        report["mlp"].update(effective="split_te_swiglu", patched_layers=count)
    elif uses_transformer_engine:
        te_attention_backend = getattr(model_config, "attention_backend", None)
        report["attention"]["effective"] = (
            "transformer_engine_flash_attention"
            if te_attention_backend == AttnBackend.flash
            else "transformer_engine_auto_attention"
        )
        report["mlp"]["effective"] = (
            "transformer_engine_swiglu"
            if getattr(model_config, "gated_linear_unit", False)
            else "transformer_engine_mlp"
        )
        report["norm"]["effective"] = (
            "transformer_engine_rmsnorm"
            if getattr(model_config, "normalization", None) == "RMSNorm"
            else "transformer_engine_layernorm"
        )
        report["rope"]["effective"] = (
            "transformer_engine_fused_rope"
            if getattr(layout.model, "position_embedding_type", None) == "rope"
            else "not_applicable"
        )

    explicit_flash = config.attention_backend == "flash"
    if uses_transformer_engine:
        pass
    elif config.attention_backend != "megatron" and flash_attn_func is not None:
        try:
            count = apply_flash_attention(model)
            report["attention"].update(effective="flash_attention_2", patched_layers=count)
        except RuntimeError as error:
            report["attention"]["reason"] = str(error)
            if explicit_flash and config.strict_kernels:
                raise
    elif explicit_flash and config.strict_kernels:
        raise RuntimeError("FlashAttention was explicitly requested but flash-attn is unavailable")
    elif config.attention_backend != "megatron" and flash_attn_func is None:
        report["attention"]["reason"] = "flash-attn is unavailable"
        if config.strict_kernels:
            raise RuntimeError(report["attention"]["reason"])

    explicit_liger_mlp = config.mlp_backend == "liger"
    compatible, reason, _ = _validate_dense_swiglu(model)
    if uses_transformer_engine:
        pass
    elif config.mlp_backend != "megatron" and compatible and LigerSiLUMulFunction is not None:
        count = apply_liger_swiglu(model)
        report["mlp"].update(effective="liger_fused_swiglu", patched_layers=count)
    else:
        layout = resolve_megatron_decoder_layout(model)
        model_config = getattr(layout.model, "config", None)
        native_fused = bool(compatible and getattr(model_config, "bias_activation_fusion", False))
        if native_fused:
            report["mlp"]["effective"] = "megatron_fused_swiglu"
        else:
            report["mlp"]["reason"] = reason if not compatible else "liger-kernel is unavailable"
        if (
            config.strict_kernels
            and compatible
            and not native_fused
            and (explicit_liger_mlp or config.mlp_backend == "auto")
        ):
            raise RuntimeError(report["mlp"]["reason"])

    layout = resolve_megatron_decoder_layout(model)
    explicit_liger_loss = config.loss_backend == "liger"
    explicit_legacy_loss = config.loss_backend == "legacy"
    model_config = getattr(layout.model, "config", None)
    loss_compatible = (
        layout.output_layer is not None
        and hasattr(layout.model, "_postprocess")
        and not getattr(model_config, "mtp_num_layers", None)
        and not getattr(model_config, "use_mup", False)
    )
    if (
        config.loss_backend in {"auto", "legacy"}
        and loss_compatible
        and LegacyFusedLinearCrossEntropyLoss is not None
    ):
        layout.model._slideformer_output_processor = legacy_fused_linear_cross_entropy
        report["loss"]["effective"] = "slideformer_legacy_linear_cross_entropy"
    elif (
        config.loss_backend == "liger"
        and loss_compatible
        and LigerFusedLinearCrossEntropyLoss is not None
    ):
        layout.model._slideformer_output_processor = liger_fused_linear_cross_entropy
        report["loss"]["effective"] = "liger_fused_linear_cross_entropy"
    elif config.loss_backend == "megatron":
        pass
    else:
        report["loss"][
            "reason"
        ] = "model/runtime is incompatible with the requested fused linear cross entropy"
        loss_structure_compatible = (
            layout.output_layer is not None
            and hasattr(layout.model, "_postprocess")
            and not getattr(model_config, "mtp_num_layers", None)
            and not getattr(model_config, "use_mup", False)
        )
        if (
            config.strict_kernels
            and loss_structure_compatible
            and (explicit_liger_loss or explicit_legacy_loss or config.loss_backend == "auto")
        ):
            raise RuntimeError(report["loss"]["reason"])

    explicit_liger_norm = config.norm_backend == "liger"
    rms_norms = list(_iter_rms_norms(model))
    if uses_transformer_engine:
        pass
    elif config.norm_backend != "megatron" and rms_norms and LigerRMSNormFunction is not None:
        count = apply_liger_rms_norm(model)
        report["norm"].update(effective="liger_rms_norm", patched_modules=count)
    elif rms_norms:
        report["norm"]["reason"] = "liger-kernel RMSNorm is unavailable"
        if config.strict_kernels and (explicit_liger_norm or config.norm_backend == "auto"):
            raise RuntimeError(report["norm"]["reason"])
    else:
        report["norm"]["reason"] = "model has no local PyTorch RMSNorm modules"
        if explicit_liger_norm and config.strict_kernels:
            raise RuntimeError(report["norm"]["reason"])

    explicit_flash_rope = config.rope_backend == "flash"
    model_uses_rope = getattr(layout.model, "position_embedding_type", None) == "rope"
    model_config = getattr(layout.model, "config", None)
    from megatron.core.models.common.embeddings import rope_utils

    flash_rope_available = rope_utils.apply_rotary_emb_flash is not None
    if uses_transformer_engine:
        pass
    elif config.rope_backend != "megatron" and model_uses_rope and flash_rope_available:
        setattr(model_config, "_slideformer_flash_rope", True)
        report["rope"]["effective"] = "flash_attention_rotary"
    elif model_uses_rope:
        report["rope"]["reason"] = "FlashAttention rotary kernel is unavailable"
        if config.strict_kernels and (explicit_flash_rope or config.rope_backend == "auto"):
            raise RuntimeError(report["rope"]["reason"])
    else:
        report["rope"]["reason"] = "model does not use standard RoPE"
        if explicit_flash_rope and config.strict_kernels:
            raise RuntimeError(report["rope"]["reason"])

    layout.model._slideformer_kernel_report = report
    return report
