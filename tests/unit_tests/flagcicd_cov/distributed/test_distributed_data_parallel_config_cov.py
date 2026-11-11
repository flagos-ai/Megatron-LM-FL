# Copyright (c) 2026, BAAI. All rights reserved.
"""DistributedDataParallelConfig rejects incompatible nccl_ub, bucketing, and fp32-accumulation options."""

import pytest

import megatron.core.distributed.distributed_data_parallel_config as core_ddp_config
import megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config as fsdp_ddp_config
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig as FsdpDistributedDataParallelConfig,
)


def _pin_torch_below_2_11(monkeypatch, config_module):
    # The nccl_ub allocator restriction only exists on torch < 2.11; the CI image
    # pins torch 2.7.0a0, so pin the version probe to that branch in every environment.
    monkeypatch.setattr(config_module, "is_torch_min_version", lambda version, **kwargs: False)


def test_reuse_grad_buf_for_mxfp8_param_ag_requires_fp8_param_gather():
    with pytest.raises(AssertionError, match="Reuse grad buffer only when keeping params in MXFP8"):
        DistributedDataParallelConfig(reuse_grad_buf_for_mxfp8_param_ag=True)
    config = DistributedDataParallelConfig(
        reuse_grad_buf_for_mxfp8_param_ag=True, fp8_param_gather=True
    )
    assert config.reuse_grad_buf_for_mxfp8_param_ag is True


@pytest.mark.parametrize(
    "alloc_conf", ["expandable_segments:True", "max_split_size_mb:512,expandable_segments:True"]
)
def test_nccl_ub_rejects_expandable_segments_allocator(monkeypatch, alloc_conf):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
    _pin_torch_below_2_11(monkeypatch, core_ddp_config)
    with pytest.raises(
        ValueError,
        match="expandable_segments:True is currently not supported "
        "with nccl_ub due to compatibility issue",
    ):
        DistributedDataParallelConfig(nccl_ub=True)


@pytest.mark.parametrize(
    "alloc_conf",
    ["max_split_size_mb:512", "expandable_segments:False", ""],
    ids=["other_option", "expandable_segments_disabled", "empty"],
)
def test_nccl_ub_accepts_other_allocator_configs(monkeypatch, alloc_conf):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
    _pin_torch_below_2_11(monkeypatch, core_ddp_config)
    config = DistributedDataParallelConfig(nccl_ub=True)
    assert config.nccl_ub is True


def test_nccl_ub_skips_allocator_check_on_torch_at_least_2_11(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(core_ddp_config, "is_torch_min_version", lambda version, **kwargs: True)
    config = DistributedDataParallelConfig(nccl_ub=True)
    assert config.nccl_ub is True


def test_param_name_patterns_for_fp32_local_accumulation_rejects_fp32_reduction():
    with pytest.raises(
        AssertionError,
        match="Only need to explicitly specify param_name patterns for FP32 local accumulation",
    ):
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, param_name_patterns_for_fp32_local_accumulation=('all',)
        )
    config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False, param_name_patterns_for_fp32_local_accumulation=('all',)
    )
    assert config.param_name_patterns_for_fp32_local_accumulation == ('all',)


def test_num_buckets_rejects_combined_bucket_size():
    with pytest.raises(AssertionError, match="Cannot specify both num_buckets and bucket_size"):
        DistributedDataParallelConfig(num_buckets=2, bucket_size=1000)


@pytest.mark.parametrize("num_buckets", [0, -1])
def test_num_buckets_must_be_positive(num_buckets):
    with pytest.raises(AssertionError, match="num_buckets must be greater than 0"):
        DistributedDataParallelConfig(num_buckets=num_buckets)


def test_num_buckets_without_bucket_size_is_accepted():
    config = DistributedDataParallelConfig(num_buckets=4)
    assert config.num_buckets == 4
    assert config.bucket_size is None


def test_fsdp_nccl_ub_rejects_expandable_segments_allocator(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _pin_torch_below_2_11(monkeypatch, fsdp_ddp_config)
    with pytest.raises(
        ValueError,
        match="expandable_segments:True is currently not supported "
        "with nccl_ub due to compatibility issue",
    ):
        FsdpDistributedDataParallelConfig(nccl_ub=True)


def test_fsdp_nccl_ub_accepts_unset_allocator_config(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    _pin_torch_below_2_11(monkeypatch, fsdp_ddp_config)
    config = FsdpDistributedDataParallelConfig(nccl_ub=True)
    assert config.nccl_ub is True
