from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.slideformer.gpu_gradient_pipeline import GPUGradientPipeline


def test_accumulator_size_is_fp32_and_excludes_empty_owners():
    owners = [
        SimpleNamespace(params=[object()], cpu_grad_flat=torch.empty(7)),
        SimpleNamespace(params=[], cpu_grad_flat=torch.empty(100)),
        SimpleNamespace(params=[object()], cpu_grad_flat=torch.empty(3)),
    ]

    assert GPUGradientPipeline.estimate_accumulator_bytes(owners) == 10 * 4


def test_accumulator_budget_rejects_oversized_allocation():
    with pytest.raises(MemoryError, match="exceeding the configured 1 MiB budget"):
        GPUGradientPipeline.validate_memory_budget(1024**2 + 1, 1)


def test_accumulator_budget_requires_a_positive_limit():
    with pytest.raises(ValueError, match="at least 1 MiB"):
        GPUGradientPipeline.validate_memory_budget(0, 0)
