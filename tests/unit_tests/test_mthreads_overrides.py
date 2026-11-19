# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MUSA override dispatch and collective compatibility regressions."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import uneven_dtensor as core_dtensor
from megatron.core.transformer.moe import moe_utils as core_moe
from megatron.plugin import decorators
from megatron.plugin.mthreads.distributed.fsdp import uneven_dtensor as musa_dtensor
from megatron.plugin.mthreads.transformer.moe import moe_utils as musa_moe
from megatron.plugin.platform import get_platform, platform_manager
from tests.unit_tests.test_utilities import Utils, get_current_device


@pytest.mark.parametrize("backend", ["mccl", "gloo"])
@pytest.mark.parametrize("local_rank", [0, 1, 2])
@pytest.mark.parametrize("equal_numel", [False, True])
def test_uneven_gather_preserves_values_and_outputs(backend, local_rank, equal_numel):
    inputs = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t(),
        torch.empty(0, 2),
        torch.tensor([[5.0, 6.0, 7.0]]),
    ]
    if equal_numel:
        inputs = [
            inputs[0],
            torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
            torch.tensor([[9.0], [10.0], [11.0], [12.0]]),
        ]
    outputs = [torch.full_like(value, -1) for value in inputs]
    original_outputs = list(outputs)
    group = object()

    def gather_padded(gathered, source, group=None):
        assert source.numel() == max(value.numel() for value in inputs)
        local_numel = inputs[local_rank].numel()
        torch.testing.assert_close(source.flatten()[:local_numel], inputs[local_rank].flatten())
        torch.testing.assert_close(
            source.flatten()[local_numel:], source.new_zeros(source.numel() - local_numel)
        )
        assert len(gathered) == len(inputs)
        for destination, value in zip(gathered, inputs):
            assert destination.shape == source.shape
            destination.fill_(-99)
            destination.flatten()[: value.numel()].copy_(value.flatten())

    with (
        patch("torch.distributed.get_backend", return_value=backend),
        patch("torch.distributed.all_gather", side_effect=gather_padded) as gather,
    ):
        result = musa_dtensor._all_gather_uneven(outputs, inputs[local_rank], group=group)

    assert result is None
    assert gather.call_count == 1
    assert gather.call_args.kwargs["group"] is group
    for actual, original, expected in zip(outputs, original_outputs, inputs):
        assert actual is original
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["mccl", "gloo"])
def test_all_empty_gather_skips_collective(backend):
    outputs = [torch.empty(0, 2), torch.empty(3, 0)]
    original_outputs = list(outputs)
    with (
        patch("torch.distributed.get_backend", return_value=backend),
        patch("torch.distributed.all_gather") as gather,
    ):
        assert musa_dtensor._all_gather_uneven(outputs, torch.empty(0, 2)) is None

    gather.assert_not_called()
    assert all(actual is original for actual, original in zip(outputs, original_outputs))


@pytest.mark.parametrize(
    "backend,shapes",
    [
        ("nccl", [(2, 2), (1, 2)]),
        ("mccl", [(2, 2), (2, 2)]),
        ("gloo", [(2, 2), (2, 2)]),
        ("mccl", []),
    ],
)
def test_gather_passthrough(backend, shapes):
    outputs = [torch.empty(shape) for shape in shapes]
    source = torch.ones(2, 2)
    group = object()
    expected = object()
    with (
        patch("torch.distributed.get_backend", return_value=backend),
        patch("torch.distributed.all_gather", return_value=expected) as gather,
    ):
        assert musa_dtensor._all_gather_uneven(outputs, source, group=group) is expected

    assert gather.call_count == 1
    assert gather.call_args.args[0] is outputs
    assert gather.call_args.args[1] is source
    assert gather.call_args.kwargs["group"] is group


@pytest.mark.parametrize("shape_pattern", ["equal", "uneven", "empty"])
def test_gather_distributed(shape_pattern):
    if get_platform().platform_name() != "musa":
        pytest.skip("Exercises the collective compatibility implementation on MUSA.")
    Utils.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    sizes = [2 if shape_pattern == "equal" else r for r in range(world_size)]
    if shape_pattern == "empty":
        sizes = [0] * world_size
    expected = [
        torch.arange(size * 2, dtype=torch.float32).reshape(size, 2) + r * 10
        for r, size in enumerate(sizes)
    ]
    source = expected[rank].to(get_current_device())
    outputs = [torch.empty_like(value, device=source.device) for value in expected]
    original_outputs = list(outputs)

    result = musa_dtensor._all_gather_uneven(outputs, source, group=torch.distributed.group.WORLD)

    assert result is None
    for actual, original, value in zip(outputs, original_outputs, expected):
        assert actual is original
        torch.testing.assert_close(actual.cpu(), value)


@pytest.mark.parametrize(
    "device_type,fused,drop_and_pad",
    [("musa", True, False), ("musa", False, True), ("musa", True, True), ("cpu", False, False)],
)
def test_permute_delegates_unmodified_paths(device_type, fused, drop_and_pad):
    tokens = SimpleNamespace(device=SimpleNamespace(type=device_type))
    arguments = {
        "tokens": tokens,
        "routing_map": object(),
        "probs": object(),
        "num_out_tokens": 7,
        "fused": fused,
        "drop_and_pad": drop_and_pad,
        "tokens_per_expert": object(),
        "align_size": 16,
    }
    signature = inspect.signature(core_moe.permute.__wrapped__)
    expected = object()
    with patch.object(core_moe.permute, "__wrapped__", return_value=expected) as original:
        assert musa_moe.permute(**arguments) is expected

    original.assert_called_once()
    forwarded = signature.bind(*original.call_args.args, **original.call_args.kwargs).arguments
    assert forwarded.keys() == arguments.keys()
    for name, value in arguments.items():
        assert forwarded[name] is value


_OVERRIDE_CASES = [
    pytest.param(core_moe, musa_moe, "permute", id="permute"),
    pytest.param(core_dtensor, musa_dtensor, "uneven_dtensor_to_full_tensor", id="uneven-dtensor"),
]


@pytest.fixture
def isolated_override_registry(monkeypatch):
    """Keep per-process vendor dispatch caches unchanged after each probe."""
    monkeypatch.setattr(decorators, "_plugin_registry", {})
    monkeypatch.setattr(decorators, "_plugin_impl_cache", {})
    monkeypatch.setattr(decorators, "_original_impl_cache", set())
    monkeypatch.setattr(
        decorators,
        "_lazy_registry",
        {key: dict(vendors) for key, vendors in decorators._lazy_registry.items()},
    )


@pytest.mark.parametrize("core_module,plugin_module,function_name", _OVERRIDE_CASES)
@pytest.mark.parametrize("selector", ["explicit", "inferred"])
def test_core_dispatches_to_registered_musa_override(
    monkeypatch, isolated_override_registry, core_module, plugin_module, function_name, selector
):
    if selector == "explicit":
        monkeypatch.setenv("MG_FL_PREFER", "musa")
        platform_name = "cuda"
    else:
        monkeypatch.delenv("MG_FL_PREFER", raising=False)
        platform_name = "musa"
    monkeypatch.setattr(
        platform_manager, "cur_platform", SimpleNamespace(platform_name=lambda: platform_name)
    )
    method_key = f"{core_module.__name__.rsplit('.', 1)[-1]}.{function_name}"
    assert decorators._lazy_registry[method_key]["musa"] == (
        f"{plugin_module.__name__}.{function_name}"
    )
    expected = object()
    implementation = Mock(return_value=expected)
    monkeypatch.setattr(plugin_module, function_name, implementation)
    core_function = getattr(core_module, function_name)
    arguments = (object(), object()) if function_name == "permute" else (object(),)

    assert core_function(*arguments) is expected

    implementation.assert_called_once_with(*arguments)
    assert decorators._plugin_impl_cache[core_function.__wrapped__] is implementation


@pytest.mark.parametrize("core_module,plugin_module,function_name", _OVERRIDE_CASES)
@pytest.mark.parametrize("platform_name", ["cpu", "cuda", "npu"])
def test_non_musa_platform_keeps_core_implementation(
    monkeypatch,
    isolated_override_registry,
    core_module,
    plugin_module,
    function_name,
    platform_name,
):
    monkeypatch.delenv("MG_FL_PREFER", raising=False)
    monkeypatch.setattr(
        platform_manager, "cur_platform", SimpleNamespace(platform_name=lambda: platform_name)
    )
    method_key = f"{core_module.__name__.rsplit('.', 1)[-1]}.{function_name}"
    assert "musa" in decorators._lazy_registry[method_key]
    implementation = Mock(
        side_effect=AssertionError("MUSA implementation selected on another platform")
    )
    monkeypatch.setattr(plugin_module, function_name, implementation)

    assert decorators.get_override_method(method_key) is None

    implementation.assert_not_called()
    assert method_key not in decorators._plugin_registry
