# Copyright (c) 2026, BAAI. All rights reserved.
"""Module-spec resolution: classes, functions, import paths and construction errors."""

import logging
from collections import OrderedDict

import pytest

from megatron.core.transformer import spec_utils
from megatron.core.transformer.spec_utils import ModuleSpec, build_module, get_module, import_module


def provider():
    return 'provided'


class Block:
    def __init__(self, width, depth=0, submodules=None):
        self.width = width
        self.depth = depth
        self.submodules = submodules


class Broken:
    def __init__(self):
        raise ValueError('boom')


def test_import_module_resolves_a_named_object():
    assert import_module(('collections', 'OrderedDict')) is OrderedDict


def test_import_module_logs_and_returns_none_for_a_missing_package(caplog):
    caplog.set_level(logging.ERROR, logger=spec_utils.logger.name)
    assert import_module(('flagcicd_cov_missing_package', 'Thing')) is None
    assert "couldn't import module due to" in caplog.text


def test_get_module_returns_classes_and_functions_as_is():
    assert get_module(Block) is Block
    assert get_module(provider) is provider
    assert get_module(ModuleSpec(module=Block)) is Block
    assert get_module(ModuleSpec(module=provider)) is provider


def test_get_module_imports_from_a_path_spec():
    assert get_module(ModuleSpec(module=('collections', 'OrderedDict'))) is OrderedDict


def test_build_module_merges_spec_params_submodules_and_call_arguments():
    built = build_module(ModuleSpec(module=Block, params={'depth': 2}, submodules='children'), 1)
    assert (built.width, built.depth, built.submodules) == (1, 2, 'children')
    called = ModuleSpec(module=Block, params={'depth': 3})(width=4)
    assert (called.width, called.depth, called.submodules) == (4, 3, None)


def test_build_module_returns_functions_without_calling_them():
    assert build_module(provider) is provider
    assert build_module(ModuleSpec(module=provider)) is provider


def test_build_module_builds_from_an_import_path():
    built = build_module(ModuleSpec(module=('collections', 'OrderedDict')))
    assert built == OrderedDict()
    assert isinstance(built, OrderedDict)


@pytest.mark.parametrize(
    ('target', 'error', 'message'),
    [
        (ModuleSpec(module=Broken), ValueError, 'boom when instantiating Broken'),
        (
            Block,
            TypeError,
            "missing 1 required positional argument: 'width' when instantiating Block",
        ),
    ],
)
def test_construction_errors_name_the_module(target, error, message):
    with pytest.raises(error, match=message):
        build_module(target)
