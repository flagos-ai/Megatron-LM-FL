# Copyright (c) 2026, BAAI. All rights reserved.
"""ModelParallelConfig.__post_init__ validation rules for parallelism and overlap settings."""

from dataclasses import replace

import pytest
import torch

from megatron.core.model_parallel_config import ModelParallelConfig


def test_sequence_parallelism_without_tensor_parallelism_is_rejected():
    with pytest.raises(
        ValueError, match="Cannot use sequence parallelism without tensor parallelism"
    ):
        ModelParallelConfig(sequence_parallel=True, tensor_model_parallel_size=1)


def test_pipeline_parallelism_without_pipeline_dtype_is_rejected():
    with pytest.raises(
        ValueError, match="When using pipeline parallelism, pipeline_dtype must be specified"
    ):
        ModelParallelConfig(pipeline_model_parallel_size=2, pipeline_dtype=None)


def test_deferred_embedding_wgrad_without_pipeline_parallelism_is_rejected():
    with pytest.raises(
        ValueError,
        match="Cannot defer embedding wgrad compute when pipeline model parallel is not used",
    ):
        ModelParallelConfig(defer_embedding_wgrad_compute=True, pipeline_model_parallel_size=1)


def test_deferred_embedding_wgrad_without_gradient_accumulation_fusion_is_rejected():
    with pytest.raises(
        ValueError,
        match="Cannot defer embedding wgrad compute when gradient accumulation fusion is not used",
    ):
        ModelParallelConfig(
            defer_embedding_wgrad_compute=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            gradient_accumulation_fusion=False,
        )


def test_deferred_embedding_wgrad_with_negative_deferral_limit_is_rejected():
    with pytest.raises(
        ValueError,
        match="Wgrad deferral limit should be greater than or equal to 0 when it is enabled!",
    ):
        ModelParallelConfig(
            defer_embedding_wgrad_compute=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            gradient_accumulation_fusion=True,
            wgrad_deferral_limit=-1,
        )


@pytest.mark.parametrize(
    ('overlap_p2p_comm', 'batch_p2p_comm'),
    [(False, True), (True, True)],
    ids=['no_overlap_comm', 'batch_p2p_comm'],
)
def test_warmup_flush_overlap_without_unbatched_p2p_comm_is_rejected(
    overlap_p2p_comm, batch_p2p_comm
):
    with pytest.raises(ValueError, match="compatible with overlap_p2p_comm but not batch_p2p_comm"):
        ModelParallelConfig(
            overlap_p2p_comm_warmup_flush=True,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
        )


def test_warmup_flush_overlap_is_accepted_with_unbatched_overlapped_p2p_comm():
    baseline = ModelParallelConfig(overlap_p2p_comm=True, batch_p2p_comm=False)
    accepted = ModelParallelConfig(
        overlap_p2p_comm_warmup_flush=True, overlap_p2p_comm=True, batch_p2p_comm=False
    )

    assert accepted.overlap_p2p_comm_warmup_flush is True
    assert accepted == replace(baseline, overlap_p2p_comm_warmup_flush=True)
