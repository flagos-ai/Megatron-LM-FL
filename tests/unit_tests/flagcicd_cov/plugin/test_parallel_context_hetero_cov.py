# Copyright (c) 2026, BAAI. All rights reserved.
"""ParallelContext multi-mesh building: rank placement per device type, inter-mesh links, global pp chains, tensor slices, memory buffer lifecycle."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as core_parallel_state
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import GlobalMemoryBuffer
from megatron.plugin.hetero import parallel_context


class _FakeGroup:
    def __init__(self, ranks, desc):
        self.ranks = list(ranks or [])
        self.desc = desc

    def name(self):
        return self.desc


def _install_fake_distributed(monkeypatch, rank, world_size, meshes):
    def new_group(ranks=None, timeout=None, backend=None, pg_options=None, **kwargs):
        return _FakeGroup(ranks, kwargs.get("group_desc") or backend or "group")

    def all_gather_object(output, current):
        order = sorted(range(world_size), key=lambda r: (0 if r % 2 == 0 else 1, r))
        if isinstance(current, dict) and "process_mesh_idx" in current:
            mesh_worlds = [tp * cp * dp * pp for tp, cp, ep, dp, pp in meshes]
            gathered = []
            for physical in range(world_size):
                logical = order.index(physical)
                mesh_idx, accumulated = 0, 0
                for index, size in enumerate(mesh_worlds):
                    if logical < accumulated + size:
                        mesh_idx = index
                        break
                    accumulated += size
                gathered.append({"rank": physical, "process_mesh_idx": mesh_idx})
            output[:] = gathered
        else:
            output[:] = [
                {"rank": physical, "device_type": "cuda" if physical % 2 == 0 else "musa"}
                for physical in range(world_size)
            ]

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
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(core_parallel_state, "_global_process_group_list", None)


def _mesh_args(meshes, current_device_type, **overrides):
    args = SimpleNamespace(
        hetero_process_meshes=meshes,
        expert_tensor_parallel_size_per_process_mesh=None,
        expert_tensor_parallel_size=None,
        nccl_communicator_config_path=None,
        distributed_timeout_minutes=1,
        use_tp_pp_dp_mapping=False,
        distributed_backend="gloo",
        use_gloo_process_groups=True,
        hetero_device_types=["cuda", "musa"],
        hetero_current_device_type=current_device_type,
        calculate_per_token_loss=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_context(monkeypatch, rank, meshes, world_size, **args_overrides):
    _install_fake_distributed(monkeypatch, rank, world_size, meshes)
    current_device_type = "cuda" if rank % 2 == 0 else "musa"
    args = _mesh_args(meshes, current_device_type, **args_overrides)
    context = parallel_context.ParallelContext.__new__(parallel_context.ParallelContext)
    context._is_initialized = False
    context._rank = rank
    context._current_process_mesh_index = 0
    context._process_meshes = []
    context._rank_to_process_mesh = {}
    context._inter_mesh_group_ranks = defaultdict(list)
    context._inter_mesh_process_groups_pp = {}
    context._inter_mesh_process_groups_dp = {}
    context._inter_mesh_process_groups_edp = {}
    context._inter_mesh_tensor_slices = {}
    context._inter_mesh_tensor_slices_for_embd_group = {}
    context._global_group_ranks = defaultdict(list)
    context._global_all_group_ranks = defaultdict(list)
    context._global_process_groups = defaultdict(list)
    context._global_process_group_to_ranks = {}
    context._global_parallel_world_sizes = {}
    context._global_parallel_ranks = {}
    context._timeout = None
    context._rank_mapper = parallel_context.RankMapper(args)
    context._args = args
    context.build_all_process_meshes()
    context.build_all_inter_mesh_process_groups()
    context.build_global_process_groups()
    context._global_memory_buffer = GlobalMemoryBuffer()
    context._tranformer_config = None
    context._ddp_config = None
    context._optimizer_config = None
    context._optimizer_config_overrides = None
    context._dataset_config = None
    context._is_initialized = True
    return context


def test_init_rejects_context_parallelism(monkeypatch):
    _install_fake_distributed(monkeypatch, 0, 2, [(1, 1, 1, 1, 2)])
    args = SimpleNamespace(context_parallel_size=2, num_distributed_optimizer_instances=1)
    with pytest.raises(AssertionError, match="Context parallelism is not supported"):
        parallel_context.ParallelContext(args)


def test_init_rejects_distributed_optimizer_instances(monkeypatch):
    _install_fake_distributed(monkeypatch, 0, 2, [(1, 1, 1, 1, 2)])
    args = SimpleNamespace(context_parallel_size=1, num_distributed_optimizer_instances=2)
    with pytest.raises(AssertionError, match="Distributed optimizer is not supported"):
        parallel_context.ParallelContext(args)


def test_init_requires_initialized_distributed(monkeypatch):
    _install_fake_distributed(monkeypatch, 0, 2, [(1, 1, 1, 1, 2)])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    args = SimpleNamespace(context_parallel_size=1, num_distributed_optimizer_instances=1)
    with pytest.raises(AssertionError):
        parallel_context.ParallelContext(args)


def test_build_all_process_meshes_rejects_world_size_mismatch(monkeypatch):
    with pytest.raises(RuntimeError, match="World size mismatch. Expected 6, but got 2"):
        _make_context(monkeypatch, 0, [(1, 1, 1, 1, 2)], 6)


def test_build_all_process_meshes_places_ranks_per_device_type(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    first, second = context._process_meshes
    assert first.get_all_process_group_ranks("pp", check_initialized=True) == [[0, 2]]
    assert first.get_all_process_group_ranks("dp", check_initialized=True) == [[0], [2]]
    assert second.get_all_process_group_ranks("pp", check_initialized=True) == [[4, 1, 3, 5]]
    assert second.get_all_process_group_ranks("dp", check_initialized=True) == [[4], [1], [3], [5]]
    assert context._rank_to_process_mesh[0] is first
    assert context._rank_to_process_mesh[5] is second
    assert context._current_process_mesh_index == 0


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"expert_tensor_parallel_size_per_process_mesh": [1, 1]}, id="per-mesh"),
        pytest.param({"expert_tensor_parallel_size": 1}, id="global"),
    ],
)
def test_build_all_process_meshes_reads_expert_tensor_parallel_size(monkeypatch, overrides):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6, **overrides)
    assert context._process_meshes[0]._expert_tensor_parallel_size == 1
    assert context._process_meshes[1]._expert_tensor_parallel_size == 1


def test_build_inter_mesh_process_groups_links_boundary_ranks(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    assert dict(context._inter_mesh_process_groups_pp) == {(2, 4): True}
    assert dict(context._inter_mesh_process_groups_dp) == {(2, 4): True}
    assert dict(context._inter_mesh_process_groups_edp) == {(2, 4): True}
    assert context.get_inter_mesh_process_group(2, 4) is True
    assert context.get_inter_mesh_process_group(4, 2) is True
    with pytest.raises(RuntimeError, match=r"ProcessGroup \[0, 5\] does not exist"):
        context.get_inter_mesh_process_group(0, 5)


def test_build_inter_mesh_process_groups_skipped_for_single_mesh(monkeypatch):
    context = _make_context(monkeypatch, 0, [(1, 1, 1, 2, 1)], 2)
    assert context._process_meshes[0].get_all_process_group_ranks("pp") == [[0], [1]]
    assert context._inter_mesh_process_groups_pp == {}
    assert context._global_all_group_ranks["pp"] == [[0], [1]]
    assert context._global_group_ranks["embd"] == [[0]]
    assert context._global_group_ranks["embd_pos"] == [[0]]
    assert context._global_parallel_ranks["last_rank"] == 1


def test_build_global_process_groups_chains_pipeline_ranks(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    assert context.get_global_all_group_ranks("pp", check_initialized=True) == [[0, 2, 4, 1, 3, 5]]
    assert context.get_global_all_group_ranks("tp-pp", check_initialized=True) == [
        [0, 2, 4, 1, 3, 5]
    ]
    assert context.get_global_all_group_ranks(
        "tp-ep-pp", is_expert=True, check_initialized=True
    ) == [[0, 2, 4, 1, 3, 5]]
    assert context._global_group_ranks["embd"] == [[0, 5]]
    assert context._global_group_ranks["embd_pos"] == [[0]]
    assert context._global_parallel_ranks["last_rank"] == 5


def test_pipeline_accessors_fall_back_to_distributed_group_queries(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    assert context.get_tensor_model_parallel_world_size() == 1
    assert context.get_tensor_model_parallel_rank() == 0
    assert context.get_pipeline_model_parallel_world_size() == 6
    assert context.get_pipeline_model_parallel_rank() == 0
    local_pp = context._process_meshes[0].get_process_group("pp")
    assert local_pp.ranks == [0, 2]
    assert context.get_pipeline_model_parallel_rank(group=local_pp) == 0
    assert context.get_position_embedding_group() is context._global_process_groups["embd_pos"][0]


def test_position_embedding_group_returns_none_when_uninitialized(monkeypatch):
    _install_fake_distributed(monkeypatch, 0, 2, [(1, 1, 1, 1, 2)])
    context = parallel_context.ParallelContext.__new__(parallel_context.ParallelContext)
    context._rank = 0
    context._global_process_groups = defaultdict(list)
    context._global_process_group_to_ranks = {}
    assert context.get_position_embedding_group(check_initialized=False) is None


def test_inter_mesh_tensor_slices_forward_and_backward(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 2, meshes, 6)
    forward = context.get_inter_mesh_tensor_slices(2, (4, 1, 8), next=True)
    assert forward == [(4, (0, 1), (0, 4), 8)]
    assert context.get_inter_mesh_tensor_slices(2, (4, 1, 8), next=True) is forward

    backward_context = _make_context(monkeypatch, 4, meshes, 6)
    backward = backward_context.get_inter_mesh_tensor_slices(4, (4, 1, 8), next=False)
    assert backward == [(2, (0, 1), (0, 4), 8)]


def test_get_dp_coef_returns_one_when_mesh_dp_sizes_match(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    assert context.get_dp_coef_when_recv_backward() == 1.0


@pytest.mark.parametrize(
    "rank, expected",
    [
        pytest.param(0, True, id="first-stage-member"),
        pytest.param(2, False, id="middle-rank-nonmember"),
    ],
)
def test_is_rank_in_embedding_group_follows_embd_membership(monkeypatch, rank, expected):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, rank, meshes, 6)
    assert context.is_rank_in_embedding_group() is expected


def test_get_expert_data_parallel_world_size_queries_distributed_group(monkeypatch):
    context = _make_context(monkeypatch, 0, [(1, 1, 1, 2, 1)], 2)
    assert context.get_expert_data_parallel_world_size() == 2


def test_build_intra_dist_opt_process_groups_is_a_noop(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    groups_before = dict(context._global_process_groups)
    assert context.build_intra_dist_opt_process_groups() is None
    assert dict(context._global_process_groups) == groups_before


def test_config_getters_return_the_built_configs(monkeypatch):
    context = _make_context(monkeypatch, 0, [(1, 1, 1, 1, 2)], 2)
    tokenizer = build_tokenizer(
        SimpleNamespace(
            tokenizer_type="NullTokenizer",
            metadata_path=None,
            pad_vocab_size=False,
            vocab_size=32,
            null_tokenizer_eod_id=None,
            null_tokenizer_pad_id=None,
        )
    )
    transformer_config = TransformerConfig(
        num_layers=2, hidden_size=8, num_attention_heads=1, ffn_hidden_size=16
    )
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True)
    optimizer_config = OptimizerConfig(lr=0.1)
    dataset_config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=16,
        blend=None,
        blend_per_split=None,
        split="1,1,1",
        num_dataset_builder_threads=1,
        path_to_cache=None,
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        object_storage_cache_path=None,
        mid_level_dataset_surplus=None,
    )
    context._transformer_config = transformer_config
    context._ddp_config = ddp_config
    context._optimizer_config = optimizer_config
    context._optimizer_config_overrides = {}
    context._dataset_config = dataset_config
    assert context.get_transformer_config() is transformer_config
    assert context.get_transformer_config().num_layers == 2
    assert context.get_ddp_config() is ddp_config
    assert context.get_ddp_config().grad_reduce_in_fp32 is True
    assert context.get_optimizer_config() == (optimizer_config, {})
    assert context.get_dataset_config() is dataset_config
    assert context.get_dataset_config().sequence_length == 16


def test_global_memory_buffer_lifecycle(monkeypatch):
    meshes = [(1, 1, 1, 1, 2), (1, 1, 1, 1, 4)]
    context = _make_context(monkeypatch, 0, meshes, 6)
    assert isinstance(context.get_global_memory_buffer(), GlobalMemoryBuffer)
    with pytest.raises(AssertionError, match="global memory buffer is already initialized"):
        context.set_global_memory_buffer()
    context.destroy_global_memory_buffer()
    with pytest.raises(AssertionError, match="global memory buffer is not initialized"):
        context.get_global_memory_buffer()
