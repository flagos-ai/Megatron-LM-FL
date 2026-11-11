# Copyright (c) 2026, BAAI. All rights reserved.
"""Megatron-FSDP utils: TE/torch version bounds, RNG tracker lifecycle, and distributed-index groups."""

import contextlib
import inspect

import pytest
import torch
from packaging.version import Version as PkgVersion
from torch.distributed.device_mesh import init_device_mesh

import megatron.core.distributed.fsdp.src.megatron_fsdp.utils as fsdp_utils
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (
    FSDPDistributedIndex,
    GlobalMemoryBuffer,
    create_updated_function_signature,
    find_megatron_fsdp,
    get_cuda_rng_tracker,
    get_global_memory_buffer,
    get_mesh_names,
    get_te_version,
    initialize_rng_tracker,
    is_submodule,
    is_te_min_version,
    is_torch_min_version,
    safe_get_rank,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def fresh_rng_tracker(monkeypatch):
    monkeypatch.setattr(fsdp_utils, "_CUDA_RNG_STATE_TRACKER", None, raising=False)
    monkeypatch.setattr(fsdp_utils, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", False, raising=False)


def test_get_te_version_and_is_te_min_version():
    version = get_te_version()
    if version is None:
        # No TransformerEngine installed: no minimum version can be satisfied.
        assert is_te_min_version("0.0") is False
        assert is_te_min_version("0.0", check_equality=False) is False
    else:
        import transformer_engine

        assert version == PkgVersion(str(transformer_engine.__version__))
        assert is_te_min_version("1.0") is True
        assert is_te_min_version("1000.0") is False
        assert is_te_min_version("1000.0", check_equality=False) is False
        assert is_te_min_version(str(version)) is True
        assert is_te_min_version(f"{version.major}.{version.minor + 1}") is False
        assert (
            is_te_min_version(f"{version.major}.{version.minor + 1}", check_equality=False) is False
        )


def test_is_torch_min_version_bounds():
    assert is_torch_min_version("2.0.0") is True
    assert is_torch_min_version("1000.0.0") is False
    assert is_torch_min_version("2.0.0", check_equality=False) is True
    assert is_torch_min_version("1000.0.0", check_equality=False) is False
    installed = PkgVersion(torch.__version__)
    assert is_torch_min_version(str(installed)) is True
    assert is_torch_min_version(f"{installed.major}.{installed.minor + 1}") is False
    assert (
        is_torch_min_version(f"{installed.major}.{installed.minor + 1}", check_equality=False)
        is False
    )


def test_is_submodule_strict_and_non_strict():
    root = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    child = root[0]
    container = torch.nn.ModuleDict({"root": root})
    other = torch.nn.Linear(2, 2)
    assert is_submodule(child, root) is True
    # Containment is recursive through nested containers.
    assert is_submodule(child, container) is True
    assert is_submodule(other, root) is False
    assert is_submodule(root, root) is False
    assert is_submodule(root, root, strict=False) is True


def test_find_megatron_fsdp_walks_module_chain():
    inner = MegatronFSDP.__new__(MegatronFSDP)
    assert find_megatron_fsdp(inner) is inner
    wrapper = torch.nn.Sequential()
    wrapper.module = inner
    assert find_megatron_fsdp(wrapper) is inner
    assert find_megatron_fsdp(torch.nn.Linear(2, 2)) is None


def test_safe_get_rank_returns_distributed_rank_when_initialized(model_parallel):
    model_parallel()
    assert safe_get_rank() == torch.distributed.get_rank()


def test_get_mesh_names_reports_root_and_flattened_dims(model_parallel):
    model_parallel()
    mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp_shard", "tp"))
    assert get_mesh_names(None) == []
    assert get_mesh_names(mesh, only_submesh_dims=True) == ["dp_shard", "tp"]
    submesh_names = get_mesh_names(mesh)
    assert submesh_names[:2] == ["dp_shard", "tp"]


def test_fsdp_distributed_index_group_accessors(model_parallel):
    model_parallel()
    mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp_shard", "tp"))
    dist_index = FSDPDistributedIndex(device_mesh=mesh, dp_shard_dim="dp_shard", tp_dim="tp")

    assert dist_index.get_dp_group() is dist_index.fsdp_group
    assert dist_index.get_fsdp_group() is dist_index.fsdp_group
    assert dist_index.get_outer_fsdp_group() is None
    assert dist_index.get_root_mesh() is mesh
    assert dist_index.get_submesh("dp_shard") is mesh["dp_shard"]


def test_fsdp_distributed_index_requires_fsdp_group(model_parallel):
    model_parallel()
    mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp_shard", "tp"))
    with pytest.raises(ValueError, match="requires an FSDP process group"):
        FSDPDistributedIndex(device_mesh=mesh, dp_shard_dim="missing_dim", tp_dim="tp")


def test_fsdp_distributed_index_get_submesh_rejects_unknown_names(model_parallel):
    model_parallel()
    mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp_shard", "tp"))
    dist_index = FSDPDistributedIndex(device_mesh=mesh, dp_shard_dim="dp_shard", tp_dim="tp")
    with pytest.raises(ValueError, match="No submesh with"):
        dist_index.get_submesh("missing_dim")


def test_fsdp_distributed_index_hybrid_group_accessors(model_parallel):
    model_parallel()
    mesh = init_device_mesh("cuda", mesh_shape=(2, 4), mesh_dim_names=("dp_outer", "dp_shard"))
    dist_index = FSDPDistributedIndex(
        device_mesh=mesh,
        dp_shard_dim="dp_shard",
        dp_outer_dim="dp_outer",
        hybrid_fsdp_group=torch.distributed.group.WORLD,
    )
    assert dist_index.use_hybrid_fsdp is True
    assert dist_index.get_dp_group() is torch.distributed.group.WORLD
    assert dist_index.get_outer_fsdp_group() is mesh["dp_outer"].get_group()


def test_cuda_rng_tracker_reports_uninitialized_then_initialized(fresh_rng_tracker):
    tracker = get_cuda_rng_tracker()

    assert tracker.is_initialized() is False
    assert tracker.get_states() == {}

    tracker.set_states({"cov-rng": torch.zeros(1, dtype=torch.uint8)})

    assert tracker.is_initialized() is True
    assert set(tracker.get_states()) == {"cov-rng"}

    tracker.reset()

    assert tracker.is_initialized() is False
    assert tracker.get_states() == {}


def test_cuda_rng_tracker_fork_rejects_unadded_state(fresh_rng_tracker):
    tracker = get_cuda_rng_tracker()
    with pytest.raises(Exception, match="cuda rng state cov-missing is not added"):
        with tracker.fork("cov-missing"):
            pass


def test_cuda_rng_tracker_is_cached_across_calls(fresh_rng_tracker):
    tracker = get_cuda_rng_tracker()
    assert get_cuda_rng_tracker() is tracker


def test_initialize_rng_tracker_force_reset_rebuilds_tracker(fresh_rng_tracker):
    first = get_cuda_rng_tracker()

    initialize_rng_tracker(force_reset=True)
    second = fsdp_utils._CUDA_RNG_STATE_TRACKER

    assert second is not first
    assert second.is_initialized() is False


def test_te_rng_tracker_add_and_lifecycle(fresh_rng_tracker):
    tracker = get_cuda_rng_tracker(use_te_rng_tracker=True)

    assert tracker.is_initialized() is False

    tracker.add("cov-rng", 4321)

    assert tracker.is_initialized() is True
    assert set(tracker.get_states()) == {"cov-rng"}

    tracker.reset()
    assert tracker.is_initialized() is False

    tracker.set_states({"cov-rng": torch.zeros(1, dtype=torch.uint8)})
    assert tracker.is_initialized() is True


def test_inference_rng_tracker_add_and_set_states_are_noops(fresh_rng_tracker):
    tracker = get_cuda_rng_tracker(inference_rng_tracker=True)

    tracker.add("cov-rng", 4321)
    assert tracker.get_states() == {}
    assert tracker.is_initialized() is False

    tracker.set_states({"cov-rng": torch.zeros(1, dtype=torch.uint8)})
    assert tracker.get_states() == {}

    fork_context = tracker.fork()
    assert isinstance(fork_context, contextlib.nullcontext)
    with fork_context:
        pass


def test_global_memory_buffer_reuses_and_reallocates():
    buffer = GlobalMemoryBuffer()
    small = buffer.get_tensor((2, 3), torch.float32, "cov-small")
    assert small.shape == (2, 3)
    assert small.dtype == torch.float32

    smaller = buffer.get_tensor((2, 2), torch.float32, "cov-small")
    assert smaller.data_ptr() == small.data_ptr()

    larger = buffer.get_tensor((2, 5), torch.float32, "cov-small")
    assert larger.numel() == 10
    assert larger.data_ptr() != small.data_ptr()

    other = buffer.get_tensor((2, 3), torch.float32, "cov-other")
    assert other.data_ptr() != small.data_ptr()


def test_get_global_memory_buffer_returns_singleton():
    assert get_global_memory_buffer() is get_global_memory_buffer()


def test_create_updated_function_signature_adds_keyword_only_args():
    def _sample(a, b, *, c=1):
        return a + b + c

    signature = create_updated_function_signature(_sample, sync_grad=True, install=False)
    parameter_names = list(signature.parameters)
    assert parameter_names == ["a", "b", "c", "sync_grad", "install"]
    assert signature.parameters["sync_grad"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sync_grad"].default is True
    assert signature.parameters["install"].default is False
