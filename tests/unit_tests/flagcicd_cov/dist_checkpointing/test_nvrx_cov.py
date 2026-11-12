# Copyright (c) 2026, BAAI. All rights reserved.

"""nvidia-resiliency-ext availability: async-support and minimum-version probes when the extension or its packaging dependency is absent."""

import sys

import pytest

from megatron.core.dist_checkpointing.strategies import nvrx
from megatron.core.dist_checkpointing.strategies.nvrx import (
    has_nvrx_async_support,
    is_nvrx_min_version,
)


def test_has_nvrx_async_support_returns_false_when_core_module_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'nvidia_resiliency_ext.checkpointing.async_ckpt.core', None)

    assert has_nvrx_async_support() is False


def test_has_nvrx_async_support_returns_false_when_saver_module_unavailable(monkeypatch):
    monkeypatch.setitem(
        sys.modules, 'nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver', None
    )

    assert has_nvrx_async_support() is False


def test_is_nvrx_min_version_raises_without_packaging(monkeypatch):
    monkeypatch.setattr(nvrx, 'HAVE_PACKAGING', False)

    with pytest.raises(ImportError, match='packaging is not installed'):
        is_nvrx_min_version()


def test_is_nvrx_min_version_treats_missing_nvrx_as_version_0_0_0(monkeypatch):
    monkeypatch.setitem(sys.modules, 'nvidia_resiliency_ext', None)

    assert is_nvrx_min_version() is False
    assert is_nvrx_min_version('999.0') is False
    assert is_nvrx_min_version('0.0.0') is True
