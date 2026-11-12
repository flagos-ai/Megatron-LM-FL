# Copyright (c) 2026, BAAI. All rights reserved.
"""JIT fuser selection per platform and MetaX detection."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import jit
from megatron.plugin.platform import platform_manager


def probe_torch(monkeypatch, version='2.4.0', available=False, device_name='NVIDIA A100'):
    """Drive the MetaX probe through the real torch: version string and CUDA device query."""
    monkeypatch.setattr(torch, '__version__', version)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: device_name)


@pytest.fixture
def fuser(monkeypatch):
    """Isolate the module-level fuser and let a test choose the platform."""
    monkeypatch.setattr(jit, 'jit_fuser', 'unset')
    monkeypatch.setattr(jit, 'is_torch_min_version', lambda version: True)
    probe_torch(monkeypatch)

    def select(device_name):
        monkeypatch.setattr(
            platform_manager,
            'get_platform',
            lambda: SimpleNamespace(device_name=lambda *args: device_name),
        )

    return select


def test_noop_decorator_returns_the_function_unchanged():
    def function():
        return 1

    assert jit.noop_decorator(function) is function


def test_cuda_enables_torch_compile(fuser):
    fuser('cuda')
    jit.enable_jit_fuser()
    assert jit.jit_fuser is torch.compile


def test_other_platforms_keep_eager_functions(fuser):
    fuser('musa')
    jit.enable_jit_fuser()
    assert jit.jit_fuser is jit.noop_decorator


def test_metax_keeps_eager_functions_even_on_cuda(fuser, monkeypatch):
    fuser('cuda')
    probe_torch(monkeypatch, available=True, device_name='MetaX C500')
    jit.enable_jit_fuser()
    assert jit.jit_fuser is jit.noop_decorator


def test_import_error_falls_back_to_noop(fuser, monkeypatch):
    fuser('cuda')

    def unavailable(version):
        raise ImportError('no torch version helper')

    monkeypatch.setattr(jit, 'is_torch_min_version', unavailable)
    jit.enable_jit_fuser()
    assert jit.jit_fuser is jit.noop_decorator


def test_old_torch_leaves_the_fuser_untouched(fuser, monkeypatch):
    fuser('cuda')
    monkeypatch.setattr(jit, 'is_torch_min_version', lambda version: False)
    jit.enable_jit_fuser()
    assert jit.jit_fuser == 'unset'


def test_disable_installs_the_noop(fuser):
    jit.disable_jit_fuser()
    assert jit.jit_fuser is jit.noop_decorator


@pytest.mark.parametrize(
    ('probe', 'expected'),
    [
        ({'version': '2.1.0+metax', 'available': False}, True),
        ({'available': True, 'device_name': 'MetaX C500'}, True),
        ({'available': True, 'device_name': 'MACA dev'}, True),
        ({'available': True, 'device_name': None}, False),
        ({'available': True}, False),
        ({'available': False}, False),
    ],
    ids=['version', 'metax-device', 'maca-device', 'unnamed-device', 'nvidia-device', 'no-cuda'],
)
def test_metax_detection(monkeypatch, probe, expected):
    probe_torch(monkeypatch, **probe)
    assert jit._is_metax_torch() is expected


def test_metax_detection_survives_a_failing_cuda_query(monkeypatch):
    def failing():
        raise RuntimeError('driver error')

    probe_torch(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'is_available', failing)
    assert jit._is_metax_torch() is False
