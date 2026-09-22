# Copyright (c) 2026, BAAI. All rights reserved.
"""Exercise FSDP plugin dispatch and compatibility policies without GPU kernels."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    _configure_optimizer_for_dtensor_meshes,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    _get_communication_stream,
)
from megatron.plugin import decorators, override_registry
from megatron.plugin.distributed.fsdp import fully_shard as optimizer_plugin
from megatron.plugin.metax.distributed.fsdp import param_and_grad_buffer as stream_plugin
from megatron.plugin.platform import platform_manager


@pytest.fixture(autouse=True)
def isolated_dispatch(monkeypatch):
    # Existing override-manager tests clear the registries. Reload the real
    # registrations into isolated dictionaries so test order cannot affect us.
    for name in ("_plugin_registry", "_lazy_registry", "_plugin_impl_cache"):
        monkeypatch.setattr(decorators, name, {})
    monkeypatch.setattr(decorators, "_original_impl_cache", set())
    monkeypatch.delenv("MG_FL_PREFER", raising=False)
    importlib.reload(override_registry)


def select_platform(monkeypatch, name, maca):
    platform = Mock()
    platform.platform_name.return_value = name
    monkeypatch.setattr(platform_manager, "cur_platform", platform)
    monkeypatch.setattr(torch.version, "maca", maca, raising=False)
    return platform


@pytest.mark.parametrize(
    "platform,maca,preferred,expected",
    [
        ("cuda", "3.3.0.2", None, "metax"),
        ("cuda", None, None, "cuda"),
        ("kunlunxin", "3.3.0.2", None, "kunlunxin"),
        ("musa", None, None, "musa"),
        ("cpu", "3.3.0.2", None, "cpu"),
        ("cuda", "3.3.0.2", " CUDA ", "cuda"),
        ("cuda", "3.3.0.2", "musa", "musa"),
        ("cuda", "3.3.0.2", "", None),
    ],
)
def test_vendor_selection(monkeypatch, platform, maca, preferred, expected):
    select_platform(monkeypatch, platform, maca)
    if preferred is not None:
        monkeypatch.setenv("MG_FL_PREFER", preferred)
    assert decorators._get_preferred_vendor() == expected


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize(
    "maca,preferred", [("3.3.0.2", None), (None, None), (None, ""), ("3.3.0.2", "cuda")]
)
def test_stream_policy_dispatch(monkeypatch, graph_mode, maca, preferred):
    platform = select_platform(monkeypatch, "cuda", maca)
    if preferred is not None:
        monkeypatch.setenv("MG_FL_PREFER", preferred)
    # The core module retains its selected platform; both views must agree.
    core = importlib.import_module(_get_communication_stream.__module__)
    monkeypatch.setattr(core, "cur_platform", platform)
    config = SimpleNamespace(megatron_fsdp_cuda_graph_mode=graph_mode)
    communication_stream = object()
    warmup_stream, capture_stream = object(), object()
    uses_current_stream = bool(maca) and graph_mode and preferred != "cuda"
    for current_stream in (warmup_stream, capture_stream):
        platform.current_stream.return_value = current_stream
        result = _get_communication_stream(communication_stream, config)
        assert result is (current_stream if uses_current_stream else communication_stream)
        assert _get_communication_stream(None, config) is current_stream
    if maca and preferred is None:
        assert decorators._plugin_impl_cache[_get_communication_stream.__wrapped__] is (
            stream_plugin._get_communication_stream
        )


@pytest.mark.parametrize("modern_torch", [False, True])
@pytest.mark.parametrize("foreach", [None, True])
def test_optimizer_policy_dispatch_and_checkpoint_reload(monkeypatch, modern_torch, foreach):
    # Policy coverage uses fake DTensor shards; the eight-rank regression in
    # test_optimizer_mesh_compatibility.py checks actual tensors and Adam updates.
    select_platform(monkeypatch, "cuda", None)
    monkeypatch.setattr(optimizer_plugin, "is_torch_min_version", lambda version: modern_torch)
    mesh_a, mesh_b = object(), object()
    first = Mock(spec=DTensor, device_mesh=mesh_a)
    second = Mock(spec=DTensor, device_mesh=mesh_b)
    same_mesh = Mock(spec=DTensor, device_mesh=mesh_a)
    groups = [
        {"params": [first, second], "foreach": foreach, "lr": 0.01},
        {"params": [first, same_mesh], "foreach": foreach, "lr": 0.02},
        {"params": [first, second], "foreach": False},
        {"params": [first, second], "foreach": foreach, "fused": True},
        {"params": [first, second]},
        {"params": [torch.nn.Parameter(torch.ones(1))], "foreach": foreach},
    ]
    group_ids = [id(group) for group in groups]
    param_ids = [[id(param) for param in group["params"]] for group in groups]
    optimizer = Mock(param_groups=groups)
    _configure_optimizer_for_dtensor_meshes(optimizer)
    assert decorators._plugin_impl_cache[_configure_optimizer_for_dtensor_meshes.__wrapped__] is (
        optimizer_plugin._configure_optimizer_for_dtensor_meshes
    )
    expected = foreach if modern_torch else False
    assert groups[0]["foreach"] is expected
    assert groups[1]["foreach"] is foreach
    assert groups[2]["foreach"] is False
    assert groups[3]["foreach"] is foreach
    assert "foreach" not in groups[4]
    assert groups[5]["foreach"] is foreach
    assert [id(group) for group in groups] == group_ids
    assert [[id(param) for param in group["params"]] for group in groups] == param_ids
    assert [groups[i]["lr"] for i in (0, 1)] == [0.01, 0.02]
    if modern_torch:
        optimizer.register_load_state_dict_post_hook.assert_not_called()
    else:
        optimizer.register_load_state_dict_post_hook.assert_called_once()
        hook = optimizer.register_load_state_dict_post_hook.call_args.args[0]
        groups[0]["foreach"] = foreach
        hook(optimizer)
        assert groups[0]["foreach"] is False
