# Copyright (c) 2026, BAAI. All rights reserved.
"""apply_module subclass check and pass-through behaviour from megatron.core.typed_torch."""

import pytest
import torch

from megatron.core.typed_torch import apply_module


def test_apply_module_returns_the_module_unchanged():
    linear = torch.nn.Linear(2, 3)

    assert apply_module(linear) is linear


def test_apply_module_passes_through_non_modules_when_the_check_is_disabled():
    value = object()

    assert apply_module(value, check_subclass=False) is value


def test_apply_module_rejects_objects_that_are_not_torch_modules():
    with pytest.raises(TypeError, match="<class 'object'> is not a subclass of torch.nn.Module"):
        apply_module(object())
