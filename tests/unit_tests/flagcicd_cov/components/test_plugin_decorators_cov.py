# Copyright (c) 2026, BAAI. All rights reserved.
"""Plugin override registry: preferred-vendor selection, lazy implementation resolution and import, registry key shortening, and overridable-function fallbacks."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import megatron.plugin.decorators as decorators
import megatron.plugin.platform
from megatron.core import parallel_state
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.plugin.decorators import (
    _get_preferred_vendor,
    _resolve_lazy_impl,
    _target_to_method_key,
    get_override_method,
    overridable,
    register,
    register_override_method,
)


@pytest.fixture
def fresh_registry(monkeypatch):
    monkeypatch.setattr(decorators, '_plugin_registry', {})
    monkeypatch.setattr(decorators, '_lazy_registry', {})
    monkeypatch.setattr(decorators, '_plugin_impl_cache', {})
    monkeypatch.setattr(decorators, '_original_impl_cache', set())


@pytest.fixture
def single_rank_parallel_state(tmp_path):
    dist.init_process_group(
        backend='gloo', init_method=f'file://{tmp_path}/rendezvous', world_size=1, rank=0
    )
    parallel_state.initialize_model_parallel(1, 1)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_get_preferred_vendor_returns_none_when_platform_query_fails(monkeypatch):
    monkeypatch.delenv('MG_FL_PREFER', raising=False)

    class _BrokenManager:
        @property
        def cur_platform(self):
            raise RuntimeError('platform unavailable')

    monkeypatch.setattr(megatron.plugin.platform, 'platform_manager', _BrokenManager())
    assert _get_preferred_vendor() is None


def test_resolve_lazy_impl_warns_on_impl_path_without_module(fresh_registry):
    register(target='megatron.core.pkg.clip_grads.plain', impl='no-module-dot')
    assert _resolve_lazy_impl('clip_grads.plain') is None


def test_resolve_lazy_impl_warns_when_function_missing_from_module(fresh_registry):
    register(target='megatron.core.pkg.clip_grads.missing', impl='math.nope')
    assert _resolve_lazy_impl('clip_grads.missing') is None


def test_resolve_lazy_impl_returns_none_when_impl_module_is_missing(fresh_registry):
    register(target='megatron.core.pkg.clip_grads.absent', impl='missing_module_xyz.func')
    assert _resolve_lazy_impl('clip_grads.absent') is None


def test_resolve_lazy_impl_returns_none_when_impl_module_name_is_empty(fresh_registry):
    register(target='megatron.core.pkg.clip_grads.broken', impl='.func')
    assert _resolve_lazy_impl('clip_grads.broken') is None


def test_resolve_lazy_impl_imports_and_registers_default_vendor_impl(fresh_registry):
    register(target='megatron.core.pkg.clip_grads.sqrt', impl='math.sqrt')
    assert _resolve_lazy_impl('clip_grads.sqrt') is math.sqrt
    assert decorators._plugin_registry['clip_grads.sqrt'] == {'default': math.sqrt}


def test_get_override_method_resolves_lazy_impl_for_preferred_vendor(monkeypatch, fresh_registry):
    monkeypatch.setenv('MG_FL_PREFER', 'metax')
    register_override_method('module.key', lambda: 'eager', vendor='other')
    register(target='megatron.core.pkg.module.key', impl='math.sqrt', vendor='metax')
    assert get_override_method('module.key') is math.sqrt


def test_target_to_method_key_returns_short_target_unchanged():
    assert _target_to_method_key('solitary') == 'solitary'


def test_target_to_method_key_keeps_last_two_segments():
    assert (
        _target_to_method_key('megatron.core.optimizer.clip_grads.get_grad_norm_fp32')
        == 'clip_grads.get_grad_norm_fp32'
    )


def test_overridable_function_falls_back_to_unknown_key_for_self_lambda(fresh_registry):
    wrapped = overridable(lambda self: (self, 'original'))
    assert wrapped(7) == (7, 'original')


def test_overridable_function_lazy_imports_plugin_mirror_module(
    monkeypatch, fresh_registry, single_rank_parallel_state
):
    # monkeypatch torch.distributed so the single-process result is the identity reduction
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor, op=None, group=None: tensor)
    norm = get_grad_norm_fp32(torch.tensor([3.0, -4.0]), norm_type=float('inf'))
    assert norm == 4.0
