# Copyright (c) 2026, BAAI. All rights reserved.
"""Megatron-FSDP mixed precision: policy defaults, TE version bounds, and FP8 guards on plain tensors."""

from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import pytest
import torch
from packaging.version import Version as PkgVersion

import megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision as mixed_precision
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import (
    MixedPrecisionPolicy,
    fp8_dequantize,
    fp8_discard_transpose_cache,
    fp8_get_raw_data,
    fp8_need_transpose_data,
    fp8_quantize,
    fp8_set_raw_data,
    get_quantized_model_init_context_cls,
    is_blockwise_float8tensor,
    is_float8tensor,
    is_te_min_version,
)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16, torch.int32],
    ids=["fp32", "bf16", "fp16", "int32"],
)
def test_is_float8tensor_false_for_plain_tensors(dtype):
    # The classifiers are boolean predicates: a plain tensor of any dtype is not
    # a Float8Tensor, a blockwise FP8 tensor, or an MXFP8 tensor.
    tensor = torch.zeros(4, dtype=dtype)
    assert is_float8tensor(tensor) is False
    assert is_blockwise_float8tensor(tensor) is False
    assert fp8_need_transpose_data(tensor) is False


def test_is_te_min_version_bounds_against_installed_transformer_engine():
    if mixed_precision.TE_VERSION is None:
        # No TransformerEngine installed: no minimum version can be satisfied.
        assert is_te_min_version("0.0") is False
        assert is_te_min_version("0.0", check_equality=False) is False
        return
    import transformer_engine

    installed = PkgVersion(str(transformer_engine.__version__))
    assert mixed_precision.TE_VERSION == installed
    assert is_te_min_version("1.0") is True
    assert is_te_min_version("1000.0") is False
    assert is_te_min_version("1000.0", check_equality=False) is False
    assert is_te_min_version(str(installed)) is True
    assert is_te_min_version(f"{installed.major}.{installed.minor + 1}") is False
    assert (
        is_te_min_version(f"{installed.major}.{installed.minor + 1}", check_equality=False) is False
    )


def test_fp8_helpers_reject_plain_tensors():
    tensor = torch.zeros(4)
    with pytest.raises(AssertionError, match="is not a FP8 tensor"):
        fp8_discard_transpose_cache(tensor)
    with pytest.raises(AssertionError, match="is not a FP8 tensor"):
        fp8_get_raw_data(tensor)
    with pytest.raises(AssertionError, match="is not a FP8 tensor"):
        fp8_set_raw_data(tensor, torch.zeros(4))


def test_get_quantized_model_init_context_cls_matches_te_availability():
    context_cls = get_quantized_model_init_context_cls()
    if context_cls is nullcontext:
        assert is_te_min_version("1.0") is False
    else:
        assert context_cls.__name__ == "quantized_model_init"
        assert is_te_min_version("1.0") is True


def test_mixed_precision_policy_defaults_and_frozen():
    policy = MixedPrecisionPolicy()
    assert policy.main_params_dtype == torch.float32
    assert policy.main_grads_dtype is None
    assert policy.grad_comm_dtype is None
    casted = MixedPrecisionPolicy(
        main_params_dtype=torch.bfloat16,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.bfloat16,
    )
    assert casted.main_params_dtype == torch.bfloat16
    assert casted.main_grads_dtype == torch.float32
    assert casted.grad_comm_dtype == torch.bfloat16
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'main_params_dtype'"):
        policy.main_params_dtype = torch.bfloat16


def test_fp8_dequantize_rejects_plain_tensors():
    tensor = torch.zeros(4)
    with pytest.raises(AssertionError, match="is not a FP8 tensor"):
        fp8_dequantize(tensor)


def test_fp8_quantize_returns_early_without_model_params():
    # fp8_quantize is documented to quantize the given sharded parameters; with
    # none, it returns without touching the optimizer or any collective.
    result = fp8_quantize(
        model_params=[],
        main_params=[],
        start_offsets=[],
        data_parallel_group=None,
        fsdp_shard_model_params=[],
    )
    assert result is None
