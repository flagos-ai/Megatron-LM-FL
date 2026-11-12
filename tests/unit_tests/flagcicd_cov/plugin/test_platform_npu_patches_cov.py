# Copyright (c) 2026, BAAI. All rights reserved.
"""platform_npu patch machinery: wrappers, Patch/PatchesManager apply/remove, registry_patch round trip."""

import importlib.metadata
import sys
import types

import pytest
import torch

from megatron.plugin.platform import platform_npu
from megatron.plugin.platform.platform_npu import (
    Patch,
    dummy_compile,
    ensure_contiguous_wrapper,
    get_func_name,
    lcm,
    torch_all_reduce_double_dtype_bypass_wrapper,
    type_wrapper,
    version_wrapper,
)


class _WaitHandle:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def test_type_wrapper_rewrites_npu_to_cuda_in_string_results():
    @type_wrapper
    def describe(value):
        return value

    assert describe("npu:0") == "cuda:0"
    assert describe("plain") == "plain"
    tensor = torch.tensor([1.0])
    assert torch.equal(describe(tensor), tensor)


def test_ensure_contiguous_wrapper_passes_contiguous_tensor():
    received = []

    @ensure_contiguous_wrapper
    def consumer(tensor):
        received.append(tensor)
        return tensor

    strided = torch.arange(4)[::2]
    assert consumer(strided).is_contiguous()
    assert received[0].is_contiguous()
    contiguous = torch.ones(2)
    assert consumer(contiguous) is contiguous
    assert received[1] is contiguous


def test_lcm_matches_math_definition():
    assert lcm(4, 6) == 12
    assert lcm(3, 5) == 15


def test_dummy_function_returns_none():
    assert platform_npu.dummy_function("anything", key="value") is None


def test_all_reduce_bypass_converts_double_tensors_and_copies_back():
    calls = []
    handle = _WaitHandle()

    def inner(tensor, tag):
        calls.append((tag, tensor.dtype))
        tensor.add_(1.0)
        return handle

    wrapped = torch_all_reduce_double_dtype_bypass_wrapper(inner)
    payload = torch.ones(3, dtype=torch.double)
    result = wrapped(payload, "reduce")
    assert result is handle
    assert handle.waited is True
    assert calls == [("reduce", torch.float32)]
    assert payload.dtype is torch.double
    assert torch.equal(payload, torch.full((3,), 2.0, dtype=torch.double))


def test_all_reduce_bypass_passes_non_double_tensors_through():
    calls = []

    def inner(tensor):
        calls.append(tensor)
        return None

    wrapped = torch_all_reduce_double_dtype_bypass_wrapper(inner)
    payload = torch.ones(2, dtype=torch.float32)
    assert wrapped(payload) is None
    assert calls[0] is payload


def test_dummy_compile_returns_callable_or_identity_wrapper():
    def original(value):
        return value + 1

    assert dummy_compile(original) is original
    wrapper = dummy_compile()
    assert wrapper(original) is original


def test_version_wrapper_forces_transformer_engine_version():
    @version_wrapper
    def version(name, *args, **kwargs):
        return f"real-{name}"

    assert version("transformer-engine") == "2.2.0"
    assert version("torch") == "real-torch"


def test_get_func_name_accepts_strings_and_functions():
    assert get_func_name("some.name") == "some.name"
    assert get_func_name(test_get_func_name_accepts_strings_and_functions) == (
        f"{test_get_func_name_accepts_strings_and_functions.__module__}."
        f"{test_get_func_name_accepts_strings_and_functions.__qualname__}"
    )


def test_dummy_function_wrapper_raises_for_missing_function():
    missing = platform_npu.dummy_function_wrapper("missing_fn")
    with pytest.raises(RuntimeError, match="function missing_fn no exist"):
        missing()


def test_patch_init_splits_names_and_defaults_to_dummy():
    module_only = Patch("bare_module", None, False)
    assert module_only.orig_module_name == "bare_module"
    assert module_only.orig_func_name is None

    dotted = Patch("some.module.function", None, False)
    assert dotted.orig_module_name == "some.module"
    assert dotted.orig_func_name == "function"
    with pytest.raises(RuntimeError, match="function some.module.function no exist"):
        dotted.patch_func()


def test_apply_patch_propagates_replacement_across_modules(monkeypatch, request):
    holder = types.ModuleType("flagcicd_cov_patch_holder")
    holder.view = torch.Tensor.view
    monkeypatch.setitem(sys.modules, "flagcicd_cov_patch_holder", holder)
    original = torch.Tensor.view

    patch = Patch("torch.Tensor.view", ensure_contiguous_wrapper, False)
    patch.apply_patch()
    request.addfinalizer(patch.remove_patch)
    assert patch.is_applied is True
    assert patch.orig_func_id == id(original)
    assert patch.patch_func_id == id(patch.patch_func)
    assert torch.Tensor.view is not original
    assert holder.view is torch.Tensor.view
    patch.remove_patch()
    assert patch.is_applied is False
    assert torch.Tensor.view is original
    assert holder.view is original


def test_parse_path_raises_for_missing_module_without_dummy():
    with pytest.raises(ModuleNotFoundError, match="flagcicd_cov_missing_module"):
        Patch.parse_path("flagcicd_cov_missing_module", "function", False)


def test_parse_path_creates_dummy_module_with_parent_link():
    with pytest.MonkeyPatch.context() as context:
        parent = types.ModuleType("flagcicd_cov_dummy_parent")
        context.setitem(sys.modules, "flagcicd_cov_dummy_parent", parent)
        module, func = Patch.parse_path("flagcicd_cov_dummy_parent.child", "function", True)
        context.setitem(sys.modules, "flagcicd_cov_dummy_parent.child", module)
        assert module.__file__ == "mindspeed.dummy_module.py"
        assert parent.child is module
        assert func is None


def test_parse_path_resolves_class_attributes():
    module, func = Patch.parse_path("torch.Tensor", "view", False)
    assert module is torch.Tensor
    assert func is torch.Tensor.view

    module, wrapper = Patch.parse_path("torch.Tensor", "missing_method", True)
    assert module is torch.Tensor
    with pytest.raises(RuntimeError, match="function missing_method no exist"):
        wrapper()

    with pytest.raises(RuntimeError, match="no exist missing_method"):
        Patch.parse_path("torch.Tensor", "missing_method", False)


def test_remove_wrappers_raises_for_unregistered_function(monkeypatch):
    monkeypatch.setattr(platform_npu.PatchesManager, "patches_info", {})
    with pytest.raises(ValueError, match="The function <un.registered> not exist."):
        platform_npu.PatchesManager.remove_wrappers("un.registered", None)


def test_registry_patch_applies_and_restores_npu_replacements(monkeypatch, request):
    monkeypatch.setattr(platform_npu.PatchesManager, "patches_info", {})
    request.addfinalizer(platform_npu.PatchesManager.remove_patches)
    original_compile = torch.compile
    original_view = torch.Tensor.view
    real_torch_version = importlib.metadata.version("torch")

    platform_npu.registry_patch()
    registered = set(platform_npu.PatchesManager.patches_info)
    assert registered == {
        "torch.compile",
        "torch.jit.script",
        "torch.nn.parameter.Parameter.type",
        "torch.Tensor.type",
        "torch.Tensor.view",
        "torch.distributed.all_reduce",
        "torch._C._jit_set_nvfuser_enabled",
        "importlib.metadata.version",
    }
    assert torch.compile is not original_compile

    def callback(value):
        return value + 1

    assert torch.compile(callback) is callback
    assert torch.Tensor.view is not original_view
    assert importlib.metadata.version("transformer-engine") == "2.2.0"
    assert importlib.metadata.version("torch") == real_torch_version
    assert platform_npu.PatchesManager.get_patch("torch.compile").is_applied is True
    platform_npu.PatchesManager.remove_patches()
    assert torch.compile is original_compile
    assert torch.Tensor.view is original_view
