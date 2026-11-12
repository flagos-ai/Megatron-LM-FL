# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformBase defaults: __init__ name, abstract is_available, graph pool handle."""

import pytest

from megatron.plugin.platform.platform_base import PlatformBase
from megatron.plugin.platform.platform_cpu import PlatformCPU


class _BaseInitProbe(PlatformCPU):
    def __init__(self):
        PlatformBase.__init__(self)


def test_base_init_sets_name_to_none():
    assert _BaseInitProbe()._name is None


def test_base_is_available_raises_not_implemented():
    with pytest.raises(NotImplementedError) as caught:
        PlatformBase.is_available(PlatformCPU())
    assert not caught.value.args


def test_base_graph_pool_handle_returns_none():
    assert PlatformBase.graph_pool_handle(PlatformCPU()) is None
