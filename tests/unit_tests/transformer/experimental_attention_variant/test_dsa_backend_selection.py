# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Backend selection must not load the retired DSA plugin provider."""

from types import SimpleNamespace

import pytest

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa_backend


def _config(backend=None, fusion=False, attention_backend=AttnBackend.auto):
    return SimpleNamespace(
        dsa_kernel_backend=backend,
        apply_dsa_kernel_fusion=fusion,
        attention_backend=attention_backend,
    )


@pytest.mark.parametrize(
    'backend,fusion,expected',
    [(None, False, 'none'), ('none', True, 'none'), ('cudnn', False, 'cudnn')],
)
def test_explicit_backend_overrides_compatibility_flag(backend, fusion, expected):
    assert dsa_backend._get_dsa_kernel_backend(_config(backend, fusion)) == expected


@pytest.mark.parametrize('major', [9, 10])
def test_compatibility_flag_selects_core_cudnn_on_supported_gpus(monkeypatch, major):
    import torch

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (major, 0))
    assert dsa_backend._get_dsa_kernel_backend(_config(fusion=True)) == 'cudnn'


def test_removed_backend_is_rejected_before_import(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f'Unexpected backend import: {name}')

    monkeypatch.setattr(dsa_backend, 'import_module', unexpected_import)
    with pytest.raises(ValueError, match='none, cudnn'):
        dsa_backend._load_backend(_config('triton'))


def test_disabled_backend_does_not_import_and_clears_cache(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f'Unexpected backend import: {name}')

    monkeypatch.setattr(dsa_backend, 'import_module', unexpected_import)
    monkeypatch.setattr(dsa_backend, '_BACKEND', object())
    monkeypatch.setattr(dsa_backend, '_BACKEND_SELECTION', 'old')
    assert dsa_backend._load_backend(_config('none')) is None
    assert dsa_backend._BACKEND is None
    assert dsa_backend._BACKEND_SELECTION is None


def test_cudnn_loads_core_provider_and_reuses_cache(monkeypatch):
    provider = object()
    imports = []

    def import_provider(name):
        imports.append(name)
        assert name == 'megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels'
        return provider

    monkeypatch.setattr(dsa_backend, '_BACKEND', None)
    monkeypatch.setattr(dsa_backend, '_BACKEND_SELECTION', None)
    monkeypatch.setattr(dsa_backend, 'import_module', import_provider)
    assert dsa_backend._load_backend(_config('cudnn')) is provider
    assert dsa_backend._load_backend(_config('cudnn')) is provider
    assert len(imports) == 1


def test_unfused_attention_bypasses_fused_backend():
    assert not dsa_backend.use_fused_dsa_kernels(
        _config('cudnn', attention_backend=AttnBackend.unfused)
    )
