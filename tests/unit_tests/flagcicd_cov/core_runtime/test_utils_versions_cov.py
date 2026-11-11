# Copyright (c) 2026, BAAI. All rights reserved.
"""Version getters, minimum-version checks and experimental-API decorators in megatron.core.utils."""

import logging
import sys
from types import SimpleNamespace

import pytest
import torch

import megatron.core.utils as utils
from megatron.core.utils import (
    get_causal_conv1d_version,
    get_fa_version,
    get_flashinfer_version,
    get_mamba_version,
    get_te_version,
    is_causal_conv1d_min_version,
    is_fa_min_version,
    is_flashinfer_min_version,
    is_mamba_min_version,
    is_te_min_version,
    is_torch_min_version,
)

PACKAGING_HELPERS = [
    pytest.param(get_te_version, id='get_te_version'),
    pytest.param(lambda: is_te_min_version('1.0'), id='is_te_min_version'),
    pytest.param(lambda: is_torch_min_version('1.0'), id='is_torch_min_version'),
    pytest.param(get_fa_version, id='get_fa_version'),
    pytest.param(lambda: is_fa_min_version('1.0'), id='is_fa_min_version'),
    pytest.param(get_mamba_version, id='get_mamba_version'),
    pytest.param(lambda: is_mamba_min_version('1.0'), id='is_mamba_min_version'),
    pytest.param(get_causal_conv1d_version, id='get_causal_conv1d_version'),
    pytest.param(lambda: is_causal_conv1d_min_version('1.0'), id='is_causal_conv1d_min_version'),
    pytest.param(get_flashinfer_version, id='get_flashinfer_version'),
    pytest.param(lambda: is_flashinfer_min_version('1.0'), id='is_flashinfer_min_version'),
]


@pytest.mark.parametrize('helper', PACKAGING_HELPERS)
def test_version_helpers_refuse_to_run_without_packaging(helper, monkeypatch):
    monkeypatch.setattr(utils, 'HAVE_PACKAGING', False)
    with pytest.raises(ImportError, match='packaging is not installed'):
        helper()


def test_te_version_falls_back_to_a_zero_version_when_te_is_missing(monkeypatch):
    monkeypatch.setattr(utils, '_te_version', None)
    monkeypatch.setitem(sys.modules, 'transformer_engine', None)
    assert str(get_te_version()) == '0.0.0'


def test_te_version_strips_the_transformer_engine_build_suffix(monkeypatch):
    monkeypatch.setattr(utils, '_te_version', None)
    monkeypatch.setitem(
        sys.modules, 'transformer_engine', SimpleNamespace(__version__='0.1.0+te2.9.0')
    )
    assert str(get_te_version()) == '2.9.0'
    assert is_te_min_version('2.9.0')
    assert not is_te_min_version('2.10.0')
    assert not is_te_min_version('2.9.0', check_equality=False)
    assert is_te_min_version('2.8.0', check_equality=False)


def test_te_version_uses_the_installed_distribution_without_a_module_attribute(monkeypatch):
    monkeypatch.setattr(utils, '_te_version', None)
    monkeypatch.setitem(sys.modules, 'transformer_engine', SimpleNamespace())
    monkeypatch.setattr(utils, 'version', lambda name: '1.8.0')
    assert str(get_te_version()) == '1.8.0'


def test_torch_version_checks_compare_against_the_running_torch():
    assert str(utils.get_torch_version()) == torch.__version__
    assert is_torch_min_version('1.13.0')
    assert not is_torch_min_version('99.0')
    assert is_torch_min_version('1.13.0', check_equality=False)
    assert not is_torch_min_version('99.0', check_equality=False)


def test_fa_version_reads_the_flash_attn_module(monkeypatch):
    monkeypatch.setattr(utils, '_fa_version', None)
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(__version__='2.7.4'))
    assert str(get_fa_version()) == '2.7.4'
    assert is_fa_min_version('2.7.0')
    assert not is_fa_min_version('2.8.0')
    assert not is_fa_min_version('2.7.4', check_equality=False)


def test_fa_version_uses_the_installed_distribution_without_a_module_attribute(monkeypatch):
    monkeypatch.setattr(utils, '_fa_version', None)
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace())
    monkeypatch.setattr(utils, 'version', lambda name: '2.5.6')
    assert str(get_fa_version()) == '2.5.6'


def test_mamba_version_reads_the_module_and_compares(monkeypatch):
    monkeypatch.setattr(utils, '_mamba_ssm_version', None)
    monkeypatch.setitem(sys.modules, 'mamba_ssm', SimpleNamespace(__version__='2.2.2'))
    assert str(get_mamba_version()) == '2.2.2'
    assert is_mamba_min_version('2.2.0')
    assert not is_mamba_min_version('2.3.0')
    assert not is_mamba_min_version('2.2.2', check_equality=False)
    assert is_mamba_min_version('2.1.9', check_equality=False)


def test_mamba_version_uses_the_installed_distribution_without_a_module_attribute(monkeypatch):
    monkeypatch.setattr(utils, '_mamba_ssm_version', None)
    monkeypatch.setitem(sys.modules, 'mamba_ssm', SimpleNamespace())
    monkeypatch.setattr(utils, 'version', lambda name: '1.2.0')
    assert str(get_mamba_version()) == '1.2.0'


def test_causal_conv1d_version_reads_the_module_and_compares(monkeypatch):
    monkeypatch.setattr(utils, '_causal_conv1d_version', None)
    monkeypatch.setitem(sys.modules, 'causal_conv1d', SimpleNamespace(__version__='1.5.0'))
    assert str(get_causal_conv1d_version()) == '1.5.0'
    assert is_causal_conv1d_min_version('1.4.0')
    assert not is_causal_conv1d_min_version('1.6.0')
    assert not is_causal_conv1d_min_version('1.5.0', check_equality=False)


def test_causal_conv1d_version_uses_the_installed_distribution_when_attribute_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(utils, '_causal_conv1d_version', None)
    monkeypatch.setitem(sys.modules, 'causal_conv1d', SimpleNamespace())
    monkeypatch.setattr(utils, 'version', lambda name: '1.4.2')
    assert str(get_causal_conv1d_version()) == '1.4.2'


def test_flashinfer_min_version_fails_only_while_the_package_is_missing(monkeypatch):
    monkeypatch.setattr(utils, '_flashinfer_version', None)
    monkeypatch.setitem(sys.modules, 'flashinfer', None)
    assert get_flashinfer_version() is None
    assert is_flashinfer_min_version('0.1.0') is False
    monkeypatch.setitem(sys.modules, 'flashinfer', SimpleNamespace(__version__='0.2.5'))
    assert is_flashinfer_min_version('0.1.0') is True


def test_flashinfer_version_reads_the_module_and_compares(monkeypatch):
    monkeypatch.setattr(utils, '_flashinfer_version', None)
    monkeypatch.setitem(sys.modules, 'flashinfer', SimpleNamespace(__version__='0.2.5'))
    assert str(get_flashinfer_version()) == '0.2.5'
    assert is_flashinfer_min_version('0.2.0')
    assert not is_flashinfer_min_version('0.2.6')


def test_flashinfer_version_uses_the_installed_distribution_without_a_module_attribute(monkeypatch):
    monkeypatch.setattr(utils, '_flashinfer_version', None)
    monkeypatch.setitem(sys.modules, 'flashinfer', SimpleNamespace())
    monkeypatch.setattr(utils, 'version', lambda name: '0.2.1')
    assert str(get_flashinfer_version()) == '0.2.1'


def test_experimental_fn_refuses_to_run_without_packaging(monkeypatch):
    monkeypatch.setattr(utils, 'HAVE_PACKAGING', False)
    with pytest.raises(ImportError, match='packaging is not installed'):
        utils.experimental_fn('0.17.0')(lambda value: value)


@pytest.fixture
def experimental_flag():
    """Flip the user-facing experimental flag with its public setter; restore it afterwards."""
    original = utils.config.ENABLE_EXPERIMENTAL
    yield utils.config.set_experimental_flag
    utils.config.set_experimental_flag(original)


def test_experimental_fn_rejects_calls_until_the_experimental_flag_is_set(experimental_flag):
    experimental_flag(False)

    @utils.experimental_fn('0.17.0')
    def feature(value):
        return value * 2

    with pytest.raises(utils.ExperimentalNotEnabledError, match='ENABLE_EXPERIMENTAL not enabled'):
        feature(21)


def test_experimental_fn_runs_and_logs_once_when_the_experimental_flag_is_set(
    experimental_flag, monkeypatch, caplog
):
    experimental_flag(True)
    monkeypatch.setenv('RANK', '0')

    @utils.experimental_fn('0.17.0')
    def feature(value):
        return value * 2

    with caplog.at_level(logging.INFO, logger='megatron.core.utils'):
        assert feature(21) == 42
        assert feature(22) == 44
    assert caplog.text.count('ENABLE_EXPERIMENTAL is True, running experimental code.') == 1


def test_experimental_fn_warns_when_the_feature_is_past_end_of_life(monkeypatch, caplog):
    monkeypatch.setenv('RANK', '0')
    with caplog.at_level(logging.WARNING, logger='megatron.core.utils'):

        @utils.experimental_fn('0.14.0')
        def stale_feature():
            return 1

    assert 'stale_feature has reached end of life' in caplog.text


def test_experimental_cls_refuses_to_run_without_packaging(monkeypatch):
    monkeypatch.setattr(utils, 'HAVE_PACKAGING', False)
    with pytest.raises(ImportError, match='packaging is not installed'):

        @utils.experimental_cls('0.17.0')
        class ExperimentalLayer:
            pass
