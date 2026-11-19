# Copyright (c) 2026, BAAI. All rights reserved.
"""SafeUnpickler class whitelist: builtins resolve to their classes and disallowed modules raise on unpickling."""

import io
import pickle

import pytest

from megatron.core.safe_globals import SafeUnpickler


def test_safe_unpickler_find_class_allows_whitelisted_classes():
    unpickler = SafeUnpickler(io.BytesIO(b''))
    assert unpickler.find_class('builtins', 'dict') is dict
    assert unpickler.find_class('builtins', 'int') is int


def test_safe_unpickler_find_class_rejects_disallowed_class():
    unpickler = SafeUnpickler(io.BytesIO(b''))
    expected_msg = (
        "Refusing to unpickle disallowed class 'os.system' in FP8 extra-state checkpoint."
    )
    with pytest.raises(pickle.UnpicklingError, match=expected_msg):
        unpickler.find_class('os', 'system')
