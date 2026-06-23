from collections import defaultdict
from types import SimpleNamespace

import pytest

from megatron.plugin.hetero import parallel_context


def test_group_names_and_overlap_mapping():
    assert parallel_context.get_group_name("tp") == "tp"
    assert parallel_context.get_group_name("tp", is_expert=True) == "exp_tp"
    assert parallel_context.get_nccl_option_name("dp-cp") == "dp_cp"
    assert parallel_context.get_nccl_option_name("tp", is_expert=True) == "ep_tp"

    with pytest.raises(ValueError, match="Invalid token"):
        parallel_context.get_nccl_option_name("invalid")
    with pytest.raises(ValueError, match="Invalid token"):
        parallel_context.get_nccl_option_name("invalid", is_expert=True)

    assert parallel_context.find_overlapped_mapping(2, 4) == {
        0: [(0, 0, 1), (1, 1, 2)],
        1: [(2, 0, 1), (3, 1, 2)],
    }
    assert parallel_context.find_overlapped_mapping(2, 3, global_size=6) == {
        0: [(0, 0, 2), (1, 2, 3)],
        1: [(1, 0, 1), (2, 1, 3)],
    }


def test_rank_mapper_orders_devices_and_translates_ranks(monkeypatch):
    rank_infos = [
        {"rank": 0, "device_type": "cuda"},
        {"rank": 1, "device_type": "musa"},
        {"rank": 2, "device_type": "cuda"},
        {"rank": 3, "device_type": "musa"},
    ]
    monkeypatch.setattr(parallel_context.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_context.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_context.torch.distributed, "get_rank", lambda: 1)

    def all_gather_object(output, current):
        output[:] = rank_infos

    monkeypatch.setattr(parallel_context.torch.distributed, "all_gather_object", all_gather_object)
    mapper = parallel_context.RankMapper(
        SimpleNamespace(
            hetero_device_types=["cuda", "musa"],
            hetero_current_device_type="musa",
        )
    )

    assert mapper.to_physical_ranks([0, 1, 2, 3]) == [0, 2, 1, 3]
    assert mapper.to_logical_ranks([0, 2, 1, 3]) == [0, 1, 2, 3]


def _bare_parallel_context():
    context = parallel_context.ParallelContext.__new__(parallel_context.ParallelContext)
    context._is_initialized = True
    context._rank = 1
    context._current_process_mesh_index = 0
    context._global_parallel_world_sizes = {}
    context._global_parallel_ranks = {}
    context._global_process_groups = defaultdict(list)
    context._global_group_ranks = defaultdict(list)
    context._global_all_group_ranks = defaultdict(list)
    context._global_process_group_to_ranks = {}
    return context


class _Group:
    def __init__(self, ranks, desc):
        self.ranks = list(ranks)
        self.desc = desc

    def name(self):
        return self.desc


def _patch_distributed(monkeypatch, rank=0, world_size=4):
    monkeypatch.setattr(parallel_context.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_context.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        parallel_context.torch.distributed,
        "get_world_size",
        lambda group=None: len(group.ranks) if hasattr(group, "ranks") else world_size,
    )
    monkeypatch.setattr(
        parallel_context.torch.distributed,
        "get_rank",
        lambda group=None: group.ranks.index(rank) if hasattr(group, "ranks") else rank,
    )


class _IdentityRankMapper:
    def to_physical_ranks(self, logical_ranks):
        return list(logical_ranks)

    def to_logical_ranks(self, physical_ranks):
        return list(physical_ranks)


def test_parallel_context_rank_world_size_and_pipeline_stage_accessors(monkeypatch):
    context = _bare_parallel_context()
    context.set_tensor_model_parallel_world_size(2)
    context.set_pipeline_model_parallel_world_size(4)
    context.set_virtual_pipeline_model_parallel_world_size(2)
    context.set_tensor_model_parallel_rank(1)
    context.set_pipeline_model_parallel_rank(0)
    context.set_pipeline_model_parallel_split_rank(2)
    context.set_virtual_pipeline_model_parallel_rank(0)

    assert context.is_initialized() is True
    assert context.get_tensor_model_parallel_world_size() == 2
    assert context.get_pipeline_model_parallel_world_size() == 4
    assert context.get_virtual_pipeline_model_parallel_world_size() == 2
    assert context.get_tensor_model_parallel_rank() == 1
    assert context.get_pipeline_model_parallel_rank() == 0
    assert context.get_pipeline_model_parallel_split_rank() == 2
    assert context.is_pipeline_first_stage() is True
    assert context.is_pipeline_last_stage() is False
    assert context.is_pipeline_stage_before_split(rank=1) is True
    assert context.is_pipeline_stage_after_split(rank=2) is True
    assert context.is_pipeline_stage_at_split() is False

    context.set_virtual_pipeline_model_parallel_rank(1)
    context.set_pipeline_model_parallel_rank(3)
    assert context.is_pipeline_first_stage() is False
    assert context.is_pipeline_last_stage() is True

    context.set_data_parallel_rank(3)
    context.set_expert_model_parallel_world_size(4)
    context.set_expert_model_parallel_rank(2)
    context.set_expert_tensor_parallel_world_size(2)
    context.set_expert_tensor_parallel_rank(1)
    assert context.get_data_parallel_rank() == 3
    assert context.get_expert_model_parallel_world_size() == 4
    assert context.get_expert_model_parallel_rank() == 2
    assert context.get_expert_tensor_parallel_world_size() == 2
    assert context.get_expert_tensor_parallel_rank() == 1


def test_parallel_context_global_group_accessors_and_dp_coefficient():
    context = _bare_parallel_context()
    group = object()
    context._global_process_groups["tp-pp"] = group
    context._global_group_ranks["tp-pp"] = [0, 1]
    context._global_all_group_ranks["tp-pp"] = [[0, 1], [2, 3]]

    assert context.get_global_process_group("tp-pp", check_initialized=True) is group
    assert context.get_global_group_ranks("tp-pp", check_initialized=True) == [0, 1]
    assert context.get_global_all_group_ranks("tp-pp", check_initialized=True) == [
        [0, 1],
        [2, 3],
    ]
    assert context.get_model_parallel_group() is group
    assert context.get_model_parallel_src_rank() == 0

    with pytest.raises(AssertionError, match="not initialized"):
        context.get_global_process_group("missing", check_initialized=True)

    class Mesh:
        def __init__(self, dp):
            self.dp = dp

        def get_parallel_size(self, token, is_expert=False):
            assert token == "dp"
            return self.dp

    context._args = SimpleNamespace(calculate_per_token_loss=False)
    context._process_meshes = [Mesh(2), Mesh(4)]
    assert context.get_dp_coef_when_recv_backward() == pytest.approx(0.5)
    context._args.calculate_per_token_loss = True
    assert context.get_dp_coef_when_recv_backward() == 1.0


def test_process_mesh_builds_rank_groups_and_partial_dist_optimizer_paths(monkeypatch):
    _patch_distributed(monkeypatch, rank=0, world_size=4)
    created_groups = []

    def create_group(ranks, timeout=None, backend=None, pg_options=None, group_desc=None, **kwargs):
        group = _Group(ranks, group_desc or backend or "group")
        created_groups.append((group, backend, pg_options, kwargs))
        return group

    monkeypatch.setattr(parallel_context, "create_group", create_group)
    monkeypatch.setattr(parallel_context, "get_nccl_options", lambda name, cfg: {"name": name})

    mesh = parallel_context.ProcessMesh(
        data_parallel_size=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_distributed_optimizer_instances=2,
        expert_tensor_parallel_size=1,
        distributed_timeout_minutes=1,
        order="tp-cp-ep-dp-pp",
        offset=0,
        rank_mapper=_IdentityRankMapper(),
        args=SimpleNamespace(distributed_backend="fake", use_gloo_process_groups=True),
    )

    assert mesh.get_parallel_size("dp") == 2
    assert mesh.get_parallel_size("pp") == 2
    assert mesh.get_parallel_size("tp", is_expert=True) == 1
    dp_ranks = mesh.get_process_group("dp", check_initialized=True).ranks
    assert 0 in dp_ranks and len(dp_ranks) == 2
    assert mesh.get_process_group("dp", gloo=True, check_initialized=True).desc == "dp_gloo"
    assert mesh.get_process_group("intra-dp-cp", check_initialized=True).ranks
    assert mesh.get_all_process_group_ranks("dp-cp", check_initialized=True)
    assert mesh.logical_coords_to_physical_ranks([[0, 0, 0, 0]]) == [0]
    assert any(group.desc == "exp_dp_gloo" for group, _, _, _ in created_groups)

    with pytest.raises(ValueError, match="Invalid token"):
        mesh.get_parallel_size("bad")


class _Mesh:
    def __init__(self, offset, sizes, expert_sizes=None):
        self.offset = offset
        self.sizes = sizes
        self.expert_sizes = expert_sizes or sizes
        self._rank_generator = SimpleNamespace(pp=sizes["pp"])
        self.groups = {
            "tp-pp": [[offset, offset + 1]],
            "pp": [[offset, offset + 1]],
            "exp_tp-ep-pp": [[offset, offset + 1]],
        }

    def get_parallel_size(self, token, is_expert=False):
        return (self.expert_sizes if is_expert else self.sizes)[token]

    def get_all_process_group_ranks(self, token, is_expert=False, check_initialized=False):
        return self.groups[parallel_context.get_group_name(token, is_expert=is_expert)]

    def logical_coords_to_physical_ranks(self, coords, is_expert=False):
        ranks = []
        for tp, cp, dp, pp in coords:
            ranks.append(self.offset + pp * 100 + dp * 10 + cp * 2 + tp)
        return ranks


def test_parallel_context_inter_mesh_slices_and_global_group_building(monkeypatch):
    created_groups = []
    monkeypatch.setattr(
        parallel_context,
        "create_group",
        lambda ranks, timeout=None, group_desc=None, **kwargs: created_groups.append(
            _Group(ranks, group_desc or "group")
        )
        or created_groups[-1],
    )

    context = _bare_parallel_context()
    context._rank = 100
    context._timeout = None
    context._current_process_mesh_index = 0
    first = _Mesh(0, {"tp": 2, "cp": 1, "dp": 2, "pp": 2}, {"tp": 1, "ep": 1, "dp": 4, "pp": 2})
    second = _Mesh(1000, {"tp": 1, "cp": 1, "dp": 1, "pp": 2}, {"tp": 1, "ep": 1, "dp": 1, "pp": 2})
    first.groups = {"tp-pp": [[100, 101]], "pp": [[100, 101]], "exp_tp-ep-pp": [[100, 101]]}
    second.groups = {
        "tp-pp": [[1000, 1001]],
        "pp": [[1000, 1001]],
        "exp_tp-ep-pp": [[1000, 1001]],
    }
    context._process_meshes = [first, second]

    context.build_inter_mesh_process_groups(first, second)
    assert context._inter_mesh_process_groups_pp
    assert context._inter_mesh_process_groups_dp
    assert context._inter_mesh_process_groups_edp
    assert context.get_inter_mesh_process_group(100, 1000) is True
    with pytest.raises(RuntimeError, match="does not exist"):
        context.get_inter_mesh_process_group(1, 9999)

    forward_slices = context.get_inter_mesh_tensor_slices(100, (4, 2, 8), next=True)
    assert forward_slices
    assert context.get_inter_mesh_tensor_slices(100, (4, 2, 8), next=True) is forward_slices

    context._current_process_mesh_index = 1
    backward_slices = context.get_inter_mesh_tensor_slices(1000, (4, 2, 8), next=False)
    assert backward_slices

    context._current_process_mesh_index = 0
    context._global_parallel_ranks["pp_split"] = 1
    context.build_global_process_groups()
    assert context.get_global_all_group_ranks("pp", check_initialized=True)
    assert context.get_embedding_group(check_initialized=True)
    assert context.get_position_embedding_group(check_initialized=True)
    assert any(group.desc == "embd" for group in created_groups)
