# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from datetime import timedelta
from types import SimpleNamespace

import pytest

from megatron.core import parallel_state as ps


def test_parallel_state_nccl_options_create_group_and_rank_generators(monkeypatch):
    class _Options:
        def __init__(self, is_high_priority_stream=False):
            self.is_high_priority_stream = is_high_priority_stream
            self.config = SimpleNamespace()

    monkeypatch.setattr(
        ps.torch.distributed,
        "ProcessGroupNCCL",
        SimpleNamespace(Options=_Options),
        raising=False,
    )
    options = ps.get_nccl_options(
        "dp",
        {
            "dp": {
                "is_high_priority_stream": True,
                "cga_cluster_size": 4,
                "max_ctas": 8,
                "min_ctas": 1,
                "net_name": "IB",
            }
        },
    )
    assert options.is_high_priority_stream is True
    assert options.config.cga_cluster_size == 4
    assert options.config.net_name == "IB"
    assert ps.get_nccl_options("tp", {}) is None
    with pytest.raises(RuntimeError, match="net_name"):
        ps.get_nccl_options("dp", {"dp": {"net_name": "ethernet"}})

    created = []
    monkeypatch.setattr(ps, "is_torch_min_version", lambda version: True)
    monkeypatch.setattr(ps.torch.distributed, "new_group", lambda **kwargs: created.append(kwargs) or "group")
    monkeypatch.setattr(ps.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(ps, "_global_process_group_list", None)
    group = ps.create_group(ranks=[0, 1], timeout=timedelta(seconds=5), group_desc="unit")
    assert group == "group"
    assert created[-1]["group_desc"] == "unit"
    assert ps._global_process_group_list == [None, "group"]

    created.clear()
    monkeypatch.setattr(ps, "is_torch_min_version", lambda version: False)
    ps.create_group(ranks=[2], timeout=None, group_desc="old")
    assert "timeout" not in created[-1]
    assert "group_desc" not in created[-1]

    groups = ps.generate_masked_orthogonal_rank_groups(
        world_size=8, parallel_size=[2, 2, 2], mask=[True, False, True]
    )
    assert groups == [[0, 1, 4, 5], [2, 3, 6, 7]]

    rank_generator = ps.RankGenerator(tp=2, ep=1, dp=2, pp=2, cp=1, order="tp-dp-pp")
    assert rank_generator.get_mask("tp-dp-pp-cp-ep", "tp-pp") == [True, False, True, False, False]
    assert rank_generator.get_ranks("tp") == [[0, 1], [2, 3], [4, 5], [6, 7]]
    offset_generator = ps.RankGenerator(tp=1, ep=1, dp=2, pp=1, cp=1, order="dp", rank_offset=10)
    assert offset_generator.get_ranks("dp") == [[10, 11]]
    with pytest.raises(AssertionError, match="EP and CP"):
        ps.RankGenerator(tp=1, ep=2, dp=1, pp=1, cp=2, order="ep-cp")
    with pytest.raises(RuntimeError, match="specified the order"):
        ps.RankGenerator(tp=2, ep=1, dp=1, pp=1, cp=1, order="dp")

    assert ps.default_embedding_ranks([3]) == [3]
    assert ps.default_embedding_ranks([0, 1, 2]) == [0, 2]
    assert ps.default_position_embedding_ranks([5, 6]) == [5]

    cfg = {}
    ps.overwrite_nccl_comm_cfgs(cfg, "tp", ("max_ctas", 16))
    assert cfg == {"tp": {"max_ctas": 16}}


def test_parallel_state_hierarchical_hybrid_groups_and_timeout(monkeypatch):
    monkeypatch.setattr(ps, "HAVE_EINOPS", False)
    with pytest.raises(ImportError, match="einops"):
        ps.create_hierarchical_groups(0, [0, 1], [2])

    created = []
    monkeypatch.setattr(ps, "HAVE_EINOPS", True)
    monkeypatch.setattr(ps, "create_group", lambda ranks, **kwargs: created.append((list(ranks), kwargs)) or tuple(ranks))
    groups, gloo_groups = ps.create_hierarchical_groups(
        rank=0,
        ranks=list(range(8)),
        hierarchical_group_sizes=[2, 2, 2],
        create_gloo_process_groups=True,
        group_desc="CP",
    )
    assert len(groups) == 3
    assert len(gloo_groups) == 3
    assert any(call[1]["backend"] == "gloo" for call in created if "backend" in call[1])

    created.clear()
    hybrid = ps.create_hybrid_dp_cp_groups(rank=1, ranks=list(range(8)), pg_options="opts")
    assert set(hybrid) == {2, 4}
    assert all(group for group in hybrid.values())

    calls = []
    monkeypatch.setattr(ps.torch.distributed, "barrier", lambda pg=None: calls.append(("barrier", pg)))
    monkeypatch.setattr(ps.cur_platform, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        ps.torch.distributed,
        "distributed_c10d",
        SimpleNamespace(_set_pg_timeout=lambda timeout, group: calls.append((timeout, group))),
    )
    monkeypatch.setattr(ps, "_global_process_group_list", [None, "g1"])
    ps.update_pg_timeout(timedelta(seconds=3))
    assert ("barrier", None) in calls
    assert (timedelta(seconds=3), "g1") in calls
    calls.clear()
    ps.update_pg_timeout(timedelta(seconds=4), pg="specific")
    assert (timedelta(seconds=4), "specific") in calls


def test_parallel_state_global_rank_size_getters_and_stage_helpers(monkeypatch):
    class _ProcessGroup:
        def __init__(self, name, size, rank):
            self.name = name
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    tp_group = _ProcessGroup("tp", size=2, rank=1)
    pp_group = _ProcessGroup("pp", size=3, rank=2)
    monkeypatch.setattr(ps, "get_parallel_context", lambda: None)
    monkeypatch.setattr(ps, "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(ps, "_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(ps, "_MPU_TENSOR_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(ps, "_MPU_PIPELINE_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(ps, "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(ps, "_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(ps, "_DUALPIPEV_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(ps, "_TENSOR_MODEL_PARALLEL_GROUP", tp_group)
    monkeypatch.setattr(ps, "_PIPELINE_MODEL_PARALLEL_GROUP", pp_group)
    monkeypatch.setattr(ps, "_MODEL_PARALLEL_GROUP", _ProcessGroup("mp", 1, 0))
    monkeypatch.setattr(ps, "_EMBEDDING_GROUP", _ProcessGroup("embedding", 2, 1))
    monkeypatch.setattr(ps, "_POSITION_EMBEDDING_GROUP", _ProcessGroup("position", 1, 0))
    monkeypatch.setattr(ps, "_PIPELINE_GLOBAL_RANKS", [2, 3, 4])
    monkeypatch.setattr(ps, "_EMBEDDING_GLOBAL_RANKS", [2, 4])
    monkeypatch.setattr(ps, "_POSITION_EMBEDDING_GLOBAL_RANKS", [99])
    monkeypatch.setattr(
        ps.torch.distributed,
        "get_world_size",
        lambda group=None: group.size() if group is not None else 1,
    )
    monkeypatch.setattr(
        ps.torch.distributed,
        "get_rank",
        lambda group=None: group.rank() if group is not None else 2,
    )

    ps.set_tensor_model_parallel_world_size(8)
    ps.set_pipeline_model_parallel_world_size(4)
    ps.set_tensor_model_parallel_rank(6)
    ps.set_pipeline_model_parallel_rank(3)
    ps.set_virtual_pipeline_model_parallel_world_size(2)
    ps.set_virtual_pipeline_model_parallel_rank(1)
    assert ps.get_tensor_model_parallel_world_size() == 8
    assert ps.get_pipeline_model_parallel_world_size() == 4
    assert ps.get_tensor_model_parallel_rank() == 6
    assert ps.get_pipeline_model_parallel_rank() == 3
    assert ps.get_virtual_pipeline_model_parallel_world_size() == 2
    assert ps.get_virtual_pipeline_model_parallel_rank() == 1

    monkeypatch.setattr(ps, "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(ps, "_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(ps, "_MPU_TENSOR_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(ps, "_MPU_PIPELINE_MODEL_PARALLEL_RANK", None)
    assert ps.get_tensor_model_parallel_world_size() == 2
    assert ps.get_pipeline_model_parallel_world_size() == 3
    assert ps.get_tensor_model_parallel_rank() == 1
    assert ps.get_pipeline_model_parallel_rank() == 2

    assert ps.is_pipeline_first_stage() is False
    assert ps.is_pipeline_last_stage() is True
    assert ps.is_rank_in_embedding_group() is True
    assert ps.is_rank_in_position_embedding_group() is False
    assert ps.get_pipeline_model_parallel_first_rank() == 2
    assert ps.get_pipeline_model_parallel_last_rank() == 4
    assert ps.get_pipeline_model_parallel_next_rank() == 2
    assert ps.get_pipeline_model_parallel_prev_rank() == 3

    monkeypatch.setattr(ps, "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", 2)
    monkeypatch.setattr(ps, "_MPU_PIPELINE_MODEL_PARALLEL_RANK", 0)
    assert ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=0) is True
    assert ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=1) is False

    monkeypatch.setattr(ps, "_MPU_PIPELINE_MODEL_PARALLEL_RANK", 2)
    assert ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=0) is False
    assert ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=1) is True
    with pytest.raises(AssertionError, match="vp_stage"):
        ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=None)
    with pytest.raises(AssertionError, match="vp_stage"):
        ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=None)
