# Copyright (c) 2026, BAAI. All rights reserved.
"""ParallelContext accessors: default world sizes/ranks when distributed is not initialized, pipeline split predicates, set_parallel_context guard."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.hetero import parallel_context


def _bare_context():
    context = parallel_context.ParallelContext.__new__(parallel_context.ParallelContext)
    context._is_initialized = True
    context._rank = 0
    context._global_parallel_world_sizes = {}
    context._global_parallel_ranks = {}
    context._global_process_groups = defaultdict(list)
    context._global_group_ranks = defaultdict(list)
    context._global_all_group_ranks = defaultdict(list)
    context._global_process_group_to_ranks = {}
    return context


@pytest.mark.parametrize(
    "accessor, expected",
    [
        pytest.param("get_data_parallel_world_size", 0, id="dp-world-size"),
        pytest.param("get_data_parallel_rank", 0, id="dp-rank"),
        pytest.param("get_context_parallel_world_size", 0, id="cp-world-size"),
        pytest.param("get_context_parallel_rank", 0, id="cp-rank"),
        pytest.param("get_tensor_and_context_parallel_world_size", 0, id="tp-cp-world-size"),
        pytest.param("get_tensor_and_context_parallel_rank", 0, id="tp-cp-rank"),
        pytest.param("get_expert_model_parallel_world_size", 0, id="ep-world-size"),
        pytest.param("get_expert_model_parallel_rank", 0, id="ep-rank"),
        pytest.param("get_expert_tensor_and_model_parallel_world_size", 0, id="tp-ep-world-size"),
        pytest.param("get_expert_tensor_and_model_parallel_rank", 0, id="tp-ep-rank"),
        pytest.param("get_expert_data_parallel_rank", 0, id="edp-rank"),
        pytest.param("get_expert_data_parallel_world_size", 1, id="edp-world-size"),
    ],
)
def test_parallel_accessor_returns_default_when_distributed_not_initialized(
    monkeypatch, accessor, expected
):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    context = _bare_context()
    assert getattr(context, accessor)() == expected


def test_pipeline_stage_before_split_uses_rank_and_split_boundaries():
    context = _bare_context()
    context.set_pipeline_model_parallel_world_size(1)
    assert context.is_pipeline_stage_before_split() is True
    assert context.is_pipeline_stage_after_split() is True
    assert context.is_inside_encoder() is True
    assert context.is_inside_decoder() is True

    context.set_pipeline_model_parallel_world_size(4)
    context.set_pipeline_model_parallel_rank(1)
    assert context.is_pipeline_stage_before_split() is True
    assert context.is_pipeline_stage_after_split() is True

    context.set_pipeline_model_parallel_split_rank(2)
    context.set_pipeline_model_parallel_rank(1)
    assert context.is_pipeline_stage_before_split() is True
    assert context.is_pipeline_stage_after_split() is False
    context.set_pipeline_model_parallel_rank(3)
    assert context.is_pipeline_stage_before_split() is False
    assert context.is_pipeline_stage_after_split() is True
    assert context.is_pipeline_stage_at_split() is False


def test_set_parallel_context_rejects_initialization_twice(monkeypatch):
    monkeypatch.setattr(parallel_context, "_GLOBAL_PARALLEL_CONTEXT", SimpleNamespace())
    with pytest.raises(AssertionError, match="parallel context is already initialized"):
        parallel_context.set_parallel_context(
            SimpleNamespace(context_parallel_size=1, num_distributed_optimizer_instances=1)
        )
