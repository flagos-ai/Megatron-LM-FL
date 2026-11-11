# Copyright (c) 2026, BAAI. All rights reserved.
"""clip_grads import fallbacks (TE/apex absent) and inf-norm reduction across model-parallel groups."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.utils import local_multi_tensor_applier, local_multi_tensor_l2_norm
from megatron.plugin import utils as plugin_utils
from megatron.plugin.optimizer import clip_grads


def _block_backend_imports(context, *names):
    for name in names:
        context.setitem(sys.modules, name, None)


def test_import_falls_back_to_local_multi_tensor_impl_when_backends_absent():
    with pytest.MonkeyPatch.context() as context:
        _block_backend_imports(context, "transformer_engine", "apex", "amp_C")
        importlib.reload(clip_grads)
        assert clip_grads.l2_norm_impl is local_multi_tensor_l2_norm
        assert clip_grads.multi_tensor_applier is local_multi_tensor_applier
        assert clip_grads.multi_tensor_scale_tensor_impl is None
        assert clip_grads.get_device_type_for_comm is plugin_utils.get_device_type_for_comm
    importlib.reload(clip_grads)


def test_import_uses_apex_multi_tensor_impl_when_amp_c_present():
    fake_l2norm = object()
    fake_applier = object()
    with pytest.MonkeyPatch.context() as context:
        _block_backend_imports(context, "transformer_engine")
        context.setitem(sys.modules, "amp_C", SimpleNamespace(multi_tensor_l2norm=fake_l2norm))
        context.setitem(sys.modules, "apex", SimpleNamespace())
        context.setitem(
            sys.modules,
            "apex.multi_tensor_apply",
            SimpleNamespace(multi_tensor_applier=fake_applier),
        )
        importlib.reload(clip_grads)
        assert clip_grads.l2_norm_impl is fake_l2norm
        assert clip_grads.multi_tensor_applier is fake_applier
        assert clip_grads.multi_tensor_scale_tensor_impl is None
    importlib.reload(clip_grads)


def test_import_defines_local_device_type_helper_when_plugin_utils_missing(monkeypatch):
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "megatron.plugin.utils", None)
        importlib.reload(clip_grads)
        monkeypatch.setattr(clip_grads, "cur_platform", SimpleNamespace(device_name=lambda: "cpu"))
        assert clip_grads.get_device_type_for_comm("model") == "cpu"
    importlib.reload(clip_grads)


def _patch_communication_boundaries(monkeypatch):
    reductions = []

    def all_reduce(tensor, op=None, group=None):
        reductions.append((op, group, tensor.item()))

    cpu_platform = SimpleNamespace(device_name=lambda: "cpu")
    monkeypatch.setattr(clip_grads, "cur_platform", cpu_platform)
    monkeypatch.setattr(plugin_utils, "cur_platform", cpu_platform)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    return reductions


def test_grad_norm_reduces_inf_norm_over_single_model_parallel_group(monkeypatch):
    reductions = _patch_communication_boundaries(monkeypatch)
    norm = clip_grads.get_grad_norm_fp32([torch.tensor([3.0, -4.0])], float("inf"), "model")
    assert norm == 4.0
    assert reductions == [(torch.distributed.ReduceOp.MAX, "model", 4.0)]


def test_grad_norm_reduces_inf_norm_over_every_group_in_list(monkeypatch):
    reductions = _patch_communication_boundaries(monkeypatch)
    model_groups = [SimpleNamespace(name="mesh-0"), SimpleNamespace(name="mesh-1")]
    norm = clip_grads.get_grad_norm_fp32(torch.tensor([2.0, -5.0]), float("inf"), model_groups)
    assert norm == 5.0
    assert reductions == [
        (torch.distributed.ReduceOp.MAX, model_groups[0], 5.0),
        (torch.distributed.ReduceOp.MAX, model_groups[1], 5.0),
    ]
