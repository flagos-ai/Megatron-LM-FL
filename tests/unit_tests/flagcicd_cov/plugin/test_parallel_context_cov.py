# Copyright (c) 2026, BAAI. All rights reserved.
"""ProcessMesh construction and group bookkeeping: virtual pipeline state, nccl configs, distopt/gloo groups, logical-coord mapping."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as core_parallel_state
from megatron.plugin.hetero import parallel_context


class _FakeGroup:
    def __init__(self, ranks, desc):
        self.ranks = list(ranks or [])
        self.desc = desc

    def name(self):
        return self.desc


def _install_fake_distributed(monkeypatch, rank=0, world_size=4):
    created = []

    def new_group(ranks=None, timeout=None, backend=None, pg_options=None, **kwargs):
        group = _FakeGroup(ranks, kwargs.get("group_desc") or backend or "group")
        created.append(group)
        return group

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda group=None: group.ranks.index(rank) if hasattr(group, "ranks") else rank,
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: len(group.ranks) if hasattr(group, "ranks") else world_size,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, current: output.__setitem__(
            slice(None), [{"rank": r, "device_type": "cpu"} for r in range(world_size)]
        ),
    )
    monkeypatch.setattr(core_parallel_state, "_global_process_group_list", None)
    return created


def _mesh_args(**overrides):
    args = SimpleNamespace(
        distributed_backend="gloo",
        use_gloo_process_groups=True,
        hetero_device_types=None,
        hetero_current_device_type="cpu",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_mesh(monkeypatch, **kwargs):
    _install_fake_distributed(monkeypatch)
    args = _mesh_args()
    arguments = dict(
        data_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        order="tp-cp-ep-dp-pp",
        offset=0,
        rank_mapper=parallel_context.RankMapper(args),
        args=args,
        distributed_timeout_minutes=1,
    )
    arguments.update(kwargs)
    return parallel_context.ProcessMesh(**arguments)


def test_ensure_var_is_initialized_raises_when_missing():
    with pytest.raises(AssertionError, match="parallel context is not initialized"):
        parallel_context._ensure_var_is_initialized(None, "parallel context")
    parallel_context._ensure_var_is_initialized("value", "parallel context")


def test_ensure_var_is_not_initialized_raises_when_present():
    with pytest.raises(AssertionError, match="parallel context is already initialized"):
        parallel_context._ensure_var_is_not_initialized("value", "parallel context")
    parallel_context._ensure_var_is_not_initialized(None, "parallel context")


def test_get_nccl_options_returns_none_when_name_absent_from_configs():
    assert parallel_context.get_nccl_options("tp", {}) is None


def test_create_group_forwards_arguments_to_distributed_new_group(monkeypatch):
    created = _install_fake_distributed(monkeypatch, rank=1, world_size=4)
    group = parallel_context.create_group(
        ranks=[1, 3], timeout="timeout", backend="gloo", group_desc="pp"
    )
    assert group.ranks == [1, 3]
    assert created == [group]


def test_process_mesh_rejects_interleaved_schedule_without_enough_stages(monkeypatch):
    with pytest.raises(RuntimeError, match="pipeline-model-parallel size should be greater than 2"):
        _make_mesh(
            monkeypatch, virtual_pipeline_model_parallel_size=1, pipeline_model_parallel_size=2
        )


def test_process_mesh_initializes_virtual_pipeline_state(monkeypatch):
    mesh = _make_mesh(
        monkeypatch, virtual_pipeline_model_parallel_size=2, pipeline_model_parallel_size=4
    )
    assert mesh._virtual_pipeline_model_parallel_rank == 0
    assert mesh._virtual_pipeline_model_parallel_world_size == 2


def test_process_mesh_loads_nccl_communicator_configs_from_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "nccl.yaml"
    config_path.write_text("unused_group:\n  max_ctas: 8\n")
    mesh = _make_mesh(monkeypatch, nccl_communicator_config_path=str(config_path))
    assert mesh._nccl_comm_cfgs == {"unused_group": {"max_ctas": 8}}


def test_process_mesh_rejects_nccl_configs_when_yaml_is_absent(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError, match="Cannot import `yaml`"):
        _make_mesh(monkeypatch, nccl_communicator_config_path=str(tmp_path / "nccl.yaml"))


def test_process_mesh_rejects_world_not_divisible_by_expert_pipeline(monkeypatch):
    with pytest.raises(RuntimeError, match="is not divisible by"):
        _make_mesh(monkeypatch, pipeline_model_parallel_size=2, expert_tensor_parallel_size=3)


def test_process_mesh_skips_gloo_groups_when_disabled(monkeypatch):
    mesh = _make_mesh(monkeypatch, args=_mesh_args(use_gloo_process_groups=False))
    assert mesh.get_process_group("dp", gloo=True) is None
    assert mesh.get_process_group("dp").ranks == [0]


def test_process_mesh_skips_partial_distopt_gloo_groups_when_disabled(monkeypatch):
    mesh = _make_mesh(
        monkeypatch,
        data_parallel_size=2,
        num_distributed_optimizer_instances=2,
        args=_mesh_args(use_gloo_process_groups=False),
    )
    assert mesh.get_process_group("intra-dp-cp", gloo=True) is None
    assert mesh.get_process_group("intra-dp-cp").ranks == [0, 1]


def test_process_mesh_builds_distopt_groups_for_single_optimizer_instance(monkeypatch):
    mesh = _make_mesh(monkeypatch, data_parallel_size=2, pipeline_model_parallel_size=1)
    assert mesh.get_process_group("intra-dp-cp").ranks == [0, 1]
    assert mesh.get_process_group("intra-dp-cp", gloo=True).desc == "dp-cp_gloo"


def test_process_mesh_applies_sharp_to_dp_groups(monkeypatch):
    barriers = []
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: barriers.append(kwargs))
    monkeypatch.setenv("NCCL_COLLNET_ENABLE", "unset-by-monkeypatch")
    mesh = _make_mesh(monkeypatch, data_parallel_size=2, use_sharp=True)
    assert len(barriers) == 1
    assert barriers[0]["group"].ranks == [0, 1]
    assert os.environ.get("NCCL_COLLNET_ENABLE") == "0"


def test_process_group_size_uses_matching_backend_group(monkeypatch):
    mesh = _make_mesh(monkeypatch, data_parallel_size=2, pipeline_model_parallel_size=2)
    assert mesh.get_process_group_size("dp") == 2
    assert mesh.get_process_group_size("dp", gloo=True) == 2
    assert mesh.get_process_group_size("pp") == 2


def test_process_group_ranks_assert_initialized(monkeypatch):
    mesh = _make_mesh(monkeypatch)
    assert mesh.get_process_group_ranks("pp", check_initialized=True) == [0, 1]
    with pytest.raises(AssertionError, match="Process group exp_pp is not initialized"):
        mesh.get_process_group_ranks("pp", is_expert=True, check_initialized=True)


def test_logical_coords_to_physical_ranks_for_expert_and_dense_meshes(monkeypatch):
    mesh = _make_mesh(
        monkeypatch,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_tensor_parallel_size=2,
    )
    assert mesh.logical_coords_to_physical_ranks([[0, 0, 0, 0]]) == [0]
    assert mesh.logical_coords_to_physical_ranks([[1, 0, 0, 1]]) == [3]
    assert mesh.logical_coords_to_physical_ranks([[1, 0, 0, 1]], is_expert=True) == [3]
    assert mesh.logical_coords_to_physical_ranks([[0, 0, 0, 1]], is_expert=True) == [2]
