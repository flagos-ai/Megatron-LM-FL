# Copyright (c) 2026, BAAI. All rights reserved.
"""get_moe_module_spec local backend: SequentialMLP experts with column/row parallels."""

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.experts import SequentialMLP


def test_get_moe_module_spec_uses_local_backend_when_te_disabled():
    builder = get_moe_module_spec(use_te=False, num_experts=4, moe_grouped_gemm=False)

    submodules = builder.keywords['submodules']
    assert submodules.experts.func is SequentialMLP
    assert submodules.experts.keywords['submodules'].linear_fc1 is ColumnParallelLinear
    assert submodules.experts.keywords['submodules'].linear_fc2 is RowParallelLinear
    assert submodules.shared_experts.func.__name__ == 'SharedExpertMLP'
