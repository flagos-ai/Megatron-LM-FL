"""Version boundaries and unsupported combinations must fail before GPU work."""
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.slideformer import compat
from megatron.plugin.slideformer.engine import MegatronSlideFormerEngineConfig
from megatron.plugin.slideformer.session import validate_native_config


def test_upstream_tp_selection_includes_explicit_sum_and_average_precedence(monkeypatch):
    native = import_module("megatron.core.distributed.finalize_model_grads")
    monkeypatch.delattr(native, "get_tensor_parallel_gradient_reduction", raising=False)
    config = SimpleNamespace(sequence_parallel=True, qk_layernorm=True)
    param = torch.nn.Parameter(torch.ones(2))
    assert compat.get_tensor_parallel_gradient_reduction("plain", param, config) is None
    assert compat.get_tensor_parallel_gradient_reduction("q_layernorm.weight", param, config) == "sum"
    param.sum_gradients_across_tp_domain = True
    assert compat.get_tensor_parallel_gradient_reduction("plain", param, config) == "sum"
    param.average_gradients_across_tp_domain = True
    assert compat.get_tensor_parallel_gradient_reduction("plain", param, config) == "avg"


def test_fl_helpers_remain_authoritative(monkeypatch):
    native = import_module("megatron.core.distributed.finalize_model_grads")
    monkeypatch.setattr(native, "get_tensor_parallel_gradient_reduction", lambda *a: "native", raising=False)
    assert compat.get_tensor_parallel_gradient_reduction("", None, None) == "native"


def test_local_token_count_does_not_alias_input(monkeypatch):
    native = import_module("megatron.core.distributed.finalize_model_grads")
    monkeypatch.delattr(native, "get_global_num_tokens", raising=False)
    original = torch.tensor(17)
    count = compat.get_global_num_tokens(original, None, None)
    count.add_(1)
    assert original.item() == 17
    with pytest.raises(NotImplementedError):
        compat.get_global_num_tokens(original, [], None)


@pytest.mark.parametrize("field,value", [
    ("pipeline_model_parallel_size", 2), ("gtp_weight_remat_size", 2),
    ("expert_gtp_weight_remat_size", 2), ("virtual_pipeline_model_parallel_size", 2),
    ("fp16", True), ("fp8", "hybrid"), ("moe_router_load_balancing_type", "quantile_balancing"),
])
def test_unsupported_native_modes_fail_before_group_or_cuda_access(field, value):
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(**{field: value})
    with pytest.raises(NotImplementedError):
        validate_native_config(model, None, MegatronSlideFormerEngineConfig(defer_optimizer_step=True))


def test_frozen_weights_rejected_before_group_or_cuda_access():
    model = torch.nn.Linear(2, 2)
    model.weight.requires_grad_(False)
    model.config = SimpleNamespace()
    with pytest.raises(ValueError, match="full fine-tuning"):
        validate_native_config(model, None, MegatronSlideFormerEngineConfig(defer_optimizer_step=True))


def test_te_grouped_experts_all_receive_fused_gradient_targets():
    module = torch.nn.Module()
    module.num_gemms = 3
    for i in range(module.num_gemms):
        module.register_parameter(f"weight{i}", torch.nn.Parameter(torch.ones(2, 2)))
    assert compat.te_fused_weight_parameters(module) == tuple(module.parameters())
    del module.weight1
    with pytest.raises(ValueError, match="Unsupported TE fused weight layout"):
        compat.te_fused_weight_parameters(module)


def test_te_dense_and_named_weight_layouts_remain_supported():
    module = torch.nn.Linear(2, 2)
    assert compat.te_fused_weight_parameters(module) == (module.weight,)
    module.weight_names = ("weight",)
    assert compat.te_fused_weight_parameters(module) == (module.weight,)
