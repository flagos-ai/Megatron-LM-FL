# Copyright (c) 2026, BAAI. All rights reserved.
"""Multi-storage-client feature flag: package-availability guard, enable gating of import_package, and pickling of the enabled state."""

import pickle

import pytest

import megatron.core.msc_utils as msc_utils
from megatron.core.msc_utils import _FeatureFlag


def test_import_package_raises_when_multistorageclient_missing(monkeypatch):
    monkeypatch.setattr(msc_utils, 'msc', None)
    flag = _FeatureFlag()
    flag.enable()
    with pytest.raises(RuntimeError, match='The multistorageclient package is not available'):
        flag.import_package()


def test_import_package_raises_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(msc_utils, 'msc', object())
    with pytest.raises(RuntimeError, match='The MSC feature is disabled'):
        _FeatureFlag().import_package()


def test_import_package_returns_package_when_enabled(monkeypatch):
    package = object()
    monkeypatch.setattr(msc_utils, 'msc', package)
    flag = _FeatureFlag()
    flag.enable()
    assert flag.import_package() is package


def test_feature_flag_pickling_preserves_enabled_state(monkeypatch):
    monkeypatch.setattr(msc_utils, 'msc', object())
    flag = _FeatureFlag(default=False)
    flag.enable()
    restored = pickle.loads(pickle.dumps(flag))
    assert restored.is_enabled() is True
    assert restored.__getstate__() == {'_enabled': True}
