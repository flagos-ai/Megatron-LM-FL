# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch.nn as nn


@dataclass(frozen=True)
class MegatronDecoderLayout:
    model: nn.Module
    embedding: nn.Module | None
    decoder: nn.Module
    layers: nn.ModuleList | list[nn.Module]
    final_norm: nn.Module | None
    output_layer: nn.Module | None


def unwrap_megatron_module(module: nn.Module) -> nn.Module:
    """Unwrap common Megatron/DDP/precision wrappers to reach the model graph."""

    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr in ("module", "module_", "model"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module) and child is not current:
                current = child
                break
        else:
            return current
    return current


def _get_path(module: nn.Module, path: str) -> Any | None:
    current: Any = module
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _first_module(module: nn.Module, paths: Iterable[str]) -> nn.Module | None:
    for path in paths:
        value = _get_path(module, path)
        if isinstance(value, nn.Module):
            return value
    return None


def _first_layers(
    module: nn.Module, paths: Iterable[str]
) -> tuple[nn.Module, nn.ModuleList | list[nn.Module]]:
    for path in paths:
        value = _get_path(module, path)
        if (
            isinstance(value, (nn.ModuleList, list))
            and value
            and all(isinstance(layer, nn.Module) for layer in value)
        ):
            decoder_path = path.rsplit(".", 1)[0] if "." in path else ""
            decoder = _get_path(module, decoder_path) if decoder_path else module
            if not isinstance(decoder, nn.Module):
                continue
            return decoder, value
    raise ValueError(
        "Cannot resolve Megatron decoder layers. Expected one of: "
        "decoder.layers, transformer.layers, language_model.encoder.layers, encoder.layers, layers"
    )


def resolve_megatron_decoder_layout(model: nn.Module) -> MegatronDecoderLayout:
    """Resolve Megatron GPT-like modules into SlideFormer-managed groups."""

    unwrapped = unwrap_megatron_module(model)
    decoder, layers = _first_layers(
        unwrapped,
        (
            "decoder.layers",
            "transformer.layers",
            "language_model.encoder.layers",
            "encoder.layers",
            "layers",
        ),
    )
    embedding = _first_module(
        unwrapped, ("embedding", "language_model.embedding", "word_embeddings", "tok_embeddings")
    )
    final_norm = _first_module(
        unwrapped,
        (
            "decoder.final_layernorm",
            "decoder.final_norm",
            "decoder.norm",
            "transformer.final_layernorm",
            "transformer.final_norm",
            "transformer.norm",
            "language_model.encoder.final_layernorm",
            "language_model.encoder.final_norm",
            "language_model.encoder.norm",
            "encoder.final_layernorm",
            "encoder.final_norm",
            "encoder.norm",
            "norm",
            "ln_f",
        ),
    )
    output_layer = _first_module(
        unwrapped, ("output_layer", "lm_head", "language_model.output_layer")
    )
    return MegatronDecoderLayout(
        model=unwrapped,
        embedding=embedding,
        decoder=decoder,
        layers=layers,
        final_norm=final_norm,
        output_layer=output_layer,
    )


def _managed_param_ids(layout: MegatronDecoderLayout) -> set[int]:
    modules: list[nn.Module] = []
    if layout.embedding is not None:
        modules.append(layout.embedding)
    modules.extend(layer for layer in layout.layers if isinstance(layer, nn.Module))
    if layout.final_norm is not None:
        modules.append(layout.final_norm)
    if layout.output_layer is not None:
        modules.append(layout.output_layer)

    managed: set[int] = set()
    for module in modules:
        for param in module.parameters(recurse=True):
            managed.add(id(param))
    return managed


def find_unmanaged_trainable_parameters(
    model: nn.Module, layout: MegatronDecoderLayout | None = None
) -> list[str]:
    layout = layout or resolve_megatron_decoder_layout(model)
    managed = _managed_param_ids(layout)
    return [
        name
        for name, param in layout.model.named_parameters(remove_duplicate=False)
        if param.requires_grad and id(param) not in managed
    ]


def assert_trainable_parameter_coverage(
    model: nn.Module, layout: MegatronDecoderLayout | None = None
) -> None:
    missing = find_unmanaged_trainable_parameters(model, layout)
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing[:50])
        suffix = "" if len(missing) <= 50 else f"\n  ... and {len(missing) - 50} more"
        raise ValueError(
            "Megatron SlideFormer layout does not cover all trainable parameters:\n"
            f"{formatted}{suffix}"
        )
