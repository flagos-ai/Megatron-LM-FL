# Copyright (c) 2026, BAAI. All rights reserved.
"""Routing-map padding: each expert's token count raised to a multiple by flipping leading zeros."""

import importlib.util
import sys
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.fusions import fused_pad_routing_map
from megatron.core.utils import null_decorator


def _eager(fn):
    # On GPU images jit_fuser is torch.compile; call the original Python body so
    # its lines execute (compiled graphs are invisible to line coverage).
    return getattr(fn, "_torchdynamo_orig_callable", fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("num_tokens", [5, 6], ids=["block8_from5", "block8_from6"])
def test_fused_pad_routing_map_flips_first_zeros_to_reach_pad_multiple(num_tokens):
    routing_map = torch.zeros(num_tokens, 3, dtype=torch.bool, device=DEVICE)
    routing_map[0, 0] = True
    routing_map[0, 1] = True
    routing_map[1, 1] = True

    out = _eager(fused_pad_routing_map.fused_pad_routing_map)(routing_map, 2)

    expected = routing_map.int().clone()
    expected[1, 0] = 1
    assert out.shape == (num_tokens, 3)
    assert out.dtype == torch.int32
    assert torch.equal(out, expected)
    assert int(out[:, 0].sum()) % 2 == 0
    assert int(out[:, 1].sum()) % 2 == 0


def test_fused_pad_routing_map_pads_every_expert_column_and_keeps_input_untouched():
    routing_map = torch.tensor(
        [[1, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.bool, device=DEVICE
    )
    original = routing_map.clone()

    out = _eager(fused_pad_routing_map.fused_pad_routing_map)(routing_map, 4)

    expected = torch.tensor(
        [[1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.int32, device=DEVICE
    )
    assert torch.equal(out, expected)
    assert int(out[:, 0].sum()) % 4 == 0
    assert int(out[:, 1].sum()) % 4 == 0
    assert torch.equal(routing_map, original)


def test_fused_pad_routing_map_returns_input_unchanged_when_no_tokens():
    routing_map = torch.zeros(0, 3, dtype=torch.bool, device=DEVICE)

    out = _eager(fused_pad_routing_map.fused_pad_routing_map)(routing_map, 4)

    assert out is routing_map


def test_fused_pad_routing_map_module_falls_back_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    spec = importlib.util.spec_from_file_location(
        "fused_pad_routing_map_no_triton", fused_pad_routing_map.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.HAVE_TRITON is False
    assert module.triton.jit is null_decorator
    assert module.triton.autotune is null_decorator
    assert isinstance(module.tl, MagicMock)


def test_fused_pad_routing_map_public_call_flips_leading_zeros_to_reach_pad_multiple():
    routing_map = torch.tensor(
        [[1, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.bool, device=DEVICE
    )
    original = routing_map.clone()

    out = fused_pad_routing_map.fused_pad_routing_map(routing_map, 4)

    expected = torch.tensor(
        [[1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.int32, device=DEVICE
    )
    assert out.shape == (7, 2)
    assert out.dtype == torch.int32
    assert torch.equal(out, expected)
    assert int(out[:, 0].sum()) % 4 == 0
    assert int(out[:, 1].sum()) % 4 == 0
    assert torch.equal(routing_map, original)
