# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed upstream manager tests; excludes unsupported Muon and training jobs."""

import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import (
    ChunkedOptimizerStateOffloader,
)

pytestmark = pytest.mark.experimental


def test_offloader_rejects_two_master_storage_schemes():
    """One wrapped optimizer cannot alias explicit and optimizer-owned master slots."""

    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.master_weights = True
    with pytest.raises(ValueError, match="both MCore-owned master parameters"):
        ChunkedOptimizerStateOffloader(
            optimizer=optimizer,
            master_params=[param],
            chunk_size_bytes=0,
            offload_fraction=1.0,
            state_dtypes=(torch.float32,),
        )


def test_oversized_atomic_state_chunks_log_once():
    """A soft chunk target overflow is visible without splitting optimizer parameters."""

    manager = object.__new__(ChunkedOptimizerStateOffloader)
    manager.chunk_size_bytes = 1024**2
    manager._state_bytes_per_param = torch.float32.itemsize
    manager.optimizer = SimpleNamespace()
    params = tuple(nn.Parameter(torch.empty(numel)) for numel in (300_000, 400_000, 10))

    with mock.patch(
        "megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload.log_single_rank"
    ) as log:
        chunks = manager._build_chunks(params)

    assert [[id(param) for param in chunk.params] for chunk in chunks] == [
        [id(param)] for param in params
    ]
    log.assert_called_once()
    _, level, _, target_mib, oversized_count, largest_mib, optimizer_name = log.call_args.args
    assert level == logging.WARNING
    assert target_mib == 1.0
    assert oversized_count == 2
    assert largest_mib == pytest.approx(400_000 * torch.float32.itemsize / 1024**2)
    assert optimizer_name == "SimpleNamespace"


def test_h2d_targets_record_the_transfer_stream():
    """State and master allocations remain live if an update exits before its H2D completes."""

    class FakeTensor:
        def __init__(self, *, is_cuda=False, pinned=False):
            self.is_cuda = is_cuda
            self.pinned = pinned
            self.recorded_streams = []

        def is_pinned(self):
            return self.pinned

        def copy_(self, source, non_blocking=False):
            return self

        def record_stream(self, stream):
            self.recorded_streams.append(stream)

    class FakeEvent:
        def __init__(self):
            self.recorded_stream = None

        def record(self, stream):
            self.recorded_stream = stream

    h2d_stream = object()
    state_param = object()
    state_cpu = FakeTensor(pinned=True)
    state_gpu = FakeTensor(is_cuda=True)
    state_manager = object.__new__(ChunkedOptimizerStateOffloader)
    state_manager._h2d_stream = h2d_stream
    state_manager._state_staging_views = lambda chunk: [
        (state_param, "moment", state_cpu, state_gpu)
    ]
    state_manager._order_h2d_after_source_streams = lambda: None
    state_manager.optimizer = SimpleNamespace(state={state_param: {}})

    class FakeParam:
        pass

    master_param = FakeParam()
    master_cpu = FakeTensor(pinned=True)
    master_gpu = FakeTensor(is_cuda=True)
    master_param.data = master_cpu
    master_manager = object.__new__(ChunkedOptimizerStateOffloader)
    master_manager._selected_params = (master_param,)
    master_manager._master_h2d_event = None
    master_manager._explicit_master_param_ids = {id(master_param)}
    master_manager._master_weights_resident = True
    master_manager._param_devices = {master_param: object()}
    master_manager._cpu_master = {}
    master_manager._h2d_stream = h2d_stream
    master_manager._validate_master_storage = lambda: None
    master_manager._order_h2d_after_source_streams = lambda: None
    master_manager.optimizer = SimpleNamespace(state={master_param: {}})

    with (
        mock.patch.object(torch.cuda, "stream", side_effect=lambda stream: nullcontext()),
        mock.patch.object(torch.cuda, "Event", side_effect=FakeEvent),
        mock.patch.object(torch, "empty_like", return_value=master_gpu),
    ):
        state_manager._prefetch_state(object())
        master_manager._schedule_master_h2d()

    assert state_gpu.recorded_streams == [h2d_stream]
    assert master_gpu.recorded_streams == [h2d_stream]


def test_initialize_state_for_loading_requires_an_initializer():
    """A missing external optimizer initializer must fail instead of silently skipping state."""

    manager = object.__new__(ChunkedOptimizerStateOffloader)

    with pytest.raises(RuntimeError, match="requires an optimizer state initializer"):
        manager.initialize_state_for_loading(None, object())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_transfer_stream_adoption_is_idempotent_after_first_use():
    """Re-adopting the current stream pair is safe even after canonical CPU state exists."""

    param = nn.Parameter(torch.zeros(8, device='cuda'))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    d2h_stream, h2d_stream = manager.transfer_streams
    manager._cpu_state[param]['moment'] = torch.empty(param.shape, device='cpu', pin_memory=True)

    manager.use_transfer_streams(d2h_stream, h2d_stream)

    with pytest.raises(RuntimeError, match="cannot change after first use"):
        manager.use_transfer_streams(torch.cuda.Stream(), h2d_stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_chunked_step_releases_staging_slots_when_optimizer_raises():
    """An optimizer failure must not leave reusable staging buffers owned by the manager."""

    class FailingOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, defaults={})

        def step(self, closure=None):
            raise RuntimeError("optimizer failure")

    param = nn.Parameter(torch.zeros(8, device='cuda'))
    optimizer = FailingOptimizer([param])
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    cpu_state = torch.empty(param.shape, device='cpu', pin_memory=True)
    optimizer.state[param]['moment'] = cpu_state
    manager._cpu_state[param]['moment'] = cpu_state

    with pytest.raises(RuntimeError, match="optimizer failure"):
        manager.step()

    assert all(not slot.buffers for slot in manager._state_staging_slots)
    manager.synchronize_for_checkpoint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_chunked_step_propagates_group_writes_and_clones_tensor_step():
    """Group metadata is committed once without rescanning original group parameters."""

    class ParamsThatMustNotBeIterated(list):
        def __iter__(self):
            raise AssertionError("chunk execution rescanned the full optimizer parameter group")

    class GroupStateOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(
                params,
                defaults={"step": torch.tensor(0), "updates": 0, "removed_during_step": True},
            )

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                group["step"].add_(1)
                group["updates"] += 1
                group["new_field"] = "preserved"
                group.pop("removed_during_step")
                for param in group["params"]:
                    param.add_(1)

    params = [nn.Parameter(torch.zeros(256, device="cuda")) for _ in range(3)]
    optimizer = GroupStateOptimizer(params)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=1024,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    assert len(manager.chunks) == 3
    optimizer.param_groups[0]["params"] = ParamsThatMustNotBeIterated(
        optimizer.param_groups[0]["params"]
    )

    manager.step()

    group = optimizer.param_groups[0]
    assert group["step"].item() == 1
    assert group["updates"] == 1
    assert group["new_field"] == "preserved"
    assert "removed_during_step" not in group
    assert all(torch.equal(param, torch.ones_like(param)) for param in params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_state_staging_views_are_aligned_and_non_overlapping():
    """Flat staging pools must preserve a distinct aligned region for every state tensor."""

    params = [nn.Parameter(torch.zeros(size, device="cuda")) for size in (3, 5, 7)]
    optimizer = torch.optim.SGD(params, lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    for param in params:
        cpu_state = torch.empty(param.shape, dtype=torch.float32, device="cpu", pin_memory=True)
        optimizer.state[param]["moment"] = cpu_state
        manager._cpu_state[param]["moment"] = cpu_state

    staging_views = manager._state_staging_views(manager.chunks[0])
    byte_ranges = []
    for _, _, _, gpu_tensor in staging_views:
        start = gpu_tensor.data_ptr()
        assert start % 256 == 0
        byte_ranges.append((start, start + gpu_tensor.numel() * gpu_tensor.element_size()))

    byte_ranges.sort()
    assert all(left[1] <= right[0] for left, right in zip(byte_ranges, byte_ranges[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_state_h2d_waits_for_compute_stream_before_reusing_staging_buffer():
    """H2D must not overwrite compute-stream storage while prior work still uses it."""

    param = nn.Parameter(torch.zeros(4096, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    cpu_state = torch.zeros(param.shape, dtype=torch.float32, device="cpu", pin_memory=True)
    optimizer.state[param]["moment"] = cpu_state
    manager._cpu_state[param]["moment"] = cpu_state

    compute_stream = torch.cuda.Stream()
    with torch.cuda.stream(compute_stream):
        # Allocate the reusable slot on the compute stream, then make its last compute use
        # finish well after the tiny CPU-to-GPU copy would finish without stream ordering.
        staging_views = manager._state_staging_views(manager.chunks[0])
        manager._next_state_staging_slot = 0
        torch.cuda._sleep(50_000_000)
        staging_views[0][3].fill_(7)
        prefetch = manager._prefetch_state(manager.chunks[0])

    assert prefetch.event is not None
    prefetch.event.synchronize()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        optimizer.state[param]["moment"], torch.zeros_like(optimizer.state[param]["moment"])
    )
    manager.synchronize_for_checkpoint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_master_h2d_allocates_on_compute_stream():
    """A restored master window must return to the compute stream's allocator pool."""

    param = nn.Parameter(torch.ones(4096, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[param],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    manager.synchronize_for_checkpoint()
    assert param.device.type == "cpu"

    allocation_streams = []
    compute_stream = torch.cuda.Stream()
    empty_like = torch.empty_like

    def record_allocation_stream(*args, **kwargs):
        allocation_streams.append(torch.cuda.current_stream().cuda_stream)
        tensor = empty_like(*args, **kwargs)
        # If master H2D does not wait for this allocation/owner stream, the delayed
        # sentinel write can race after the CPU master copy and remain visible.
        torch.cuda._sleep(50_000_000)
        tensor.fill_(7)
        return tensor

    with (
        torch.cuda.stream(compute_stream),
        mock.patch.object(torch, "empty_like", side_effect=record_allocation_stream),
    ):
        manager.prefetch_master_for_step()

    manager.ensure_master_for_param_sync()
    torch.cuda.synchronize()
    assert allocation_streams == [compute_stream.cuda_stream]
    assert param.is_cuda
    torch.testing.assert_close(param, torch.ones_like(param))
    manager.synchronize_for_checkpoint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_state_eligibility_is_identical_for_d2h_and_cpu_adoption():
    """A scalar tensor is not state for a one-element parameter unless its shape matches."""

    param = nn.Parameter(torch.zeros(1, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    optimizer.state[param]["moment"] = torch.ones_like(param)
    optimizer.state[param]["step"] = torch.ones((), device="cuda")
    optimizer.state[param]["found_inf"] = torch.ones_like(param)

    manager.synchronize_for_checkpoint()

    assert optimizer.state[param]["moment"].device.type == "cpu"
    assert optimizer.state[param]["step"].is_cuda
    assert optimizer.state[param]["found_inf"].is_cuda
    assert "moment" in manager._cpu_state.get(param, {})
    assert "step" not in manager._cpu_state.get(param, {})
    assert "found_inf" not in manager._cpu_state.get(param, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_load_releases_cpu_buffers_from_an_old_state_schema():
    """Replacing checkpoint state must not retain obsolete pinned canonical buffers."""

    param = nn.Parameter(torch.zeros(8, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    old_state = torch.empty_like(param, device="cpu", pin_memory=True)
    new_state = torch.empty_like(param, device="cpu", pin_memory=True)
    optimizer.state[param]["old_moment"] = old_state
    manager._cpu_state[param]["old_moment"] = old_state
    state_dict = optimizer.state_dict()
    state_dict["state"][0] = {"new_moment": new_state}

    manager.load_state_dict_without_device_cast(state_dict)

    assert "old_moment" not in manager._cpu_state[param]
    assert manager._cpu_state[param]["new_moment"] is new_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_partial_load_moves_resident_tensor_state_back_to_cuda():
    """Fully-parallel CPU loads keep selected state canonical but restore resident moments."""

    params = [nn.Parameter(torch.zeros(8, device="cuda")) for _ in range(2)]
    optimizer = torch.optim.SGD(params, lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=0.5,
        state_dtypes=(torch.float32,),
    )
    state_dict = optimizer.state_dict()
    state_dict["state"] = {
        param_id: {
            "moment": torch.full(param.shape, index + 1.0, dtype=param.dtype, device="cpu"),
            "step": torch.tensor(index + 1),
        }
        for index, (param_id, param) in enumerate(
            zip(state_dict["param_groups"][0]["params"], params)
        )
    }

    manager.load_state_dict_without_device_cast(state_dict)

    selected_param = manager.selected_params[0]
    resident_param = manager._resident_params[0]
    assert optimizer.state[selected_param]["moment"].device.type == "cpu"
    assert optimizer.state[selected_param]["moment"].is_pinned()
    assert optimizer.state[resident_param]["moment"].is_cuda
    assert optimizer.state[resident_param]["step"].device.type == "cpu"
    torch.testing.assert_close(
        optimizer.state[resident_param]["moment"], torch.full_like(resident_param, 2.0)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_byte_planning_uses_state_dtypes_and_prioritizes_master_bundles():
    """A partial budget uses real moment widths and spends bytes on removable masters first."""

    native_param = nn.Parameter(torch.zeros(128, device="cuda"))
    master_param = nn.Parameter(torch.zeros(128, device="cuda"))
    optimizer = torch.optim.SGD([native_param, master_param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[master_param],
        chunk_size_bytes=0,
        offload_fraction=0.5,
        state_dtypes=(torch.bfloat16, torch.bfloat16),
    )

    assert manager._estimated_state_bytes(native_param) == 128 * 2 * torch.bfloat16.itemsize
    assert len(manager.selected_params) == 1
    assert manager.selected_params[0] is master_param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_precision_aware_master_planning_uses_real_storage():
    """Precision-aware planner accounts for every declared master using its real dtype."""

    low_precision_param = nn.Parameter(torch.zeros(128, dtype=torch.bfloat16, device="cuda"))
    native_fp32_param = nn.Parameter(torch.zeros(128, dtype=torch.float32, device="cuda"))
    optimizer = torch.optim.SGD([native_fp32_param, low_precision_param], lr=0.1)
    optimizer.master_weights = True
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.bfloat16, torch.bfloat16),
        optimizer_owned_master_dtypes={
            native_fp32_param: torch.float32,
            low_precision_param: torch.int16,
        },
    )

    state_bytes = 128 * 2 * torch.bfloat16.itemsize
    assert manager._estimated_bundle_bytes(native_fp32_param) == (
        state_bytes + 128 * torch.float32.itemsize
    )
    assert manager._estimated_bundle_bytes(low_precision_param) == (
        state_bytes + 128 * torch.int16.itemsize
    )


def test_group_metadata_comparison_accepts_bool_convertible_equality():
    """Non-bool scalar equality results are interpreted through their truth value."""

    class TruthValue:
        def __bool__(self):
            return True

    class Metadata:
        def __eq__(self, other):
            return TruthValue()

    assert ChunkedOptimizerStateOffloader._group_values_equal(Metadata(), Metadata())


@pytest.mark.parametrize("fractions", [(0.0, 0.5, 1.0), (0.0, 0.0, 0.0)])
def test_pipeline_stage_fractions_preserve_global_config(fractions):
    """Physical-stage selection is shared by virtual chunks and does not mutate its caller."""
    from megatron.core.optimizer import OptimizerConfig

    config = OptimizerConfig(
        optimizer="adam",
        use_distributed_optimizer=True,
        chunked_optimizer_state_offload=True,
        optimizer_state_offload_fraction=0.75,
        optimizer_state_offload_fraction_per_pipeline_stage=fractions,
    )
    for rank, expected in enumerate(fractions):
        local = config.for_pipeline_rank(len(fractions), rank)
        assert local.optimizer_state_offload_fraction == expected
        assert local.optimizer_state_offload_fraction_per_pipeline_stage is None
        assert config.optimizer_state_offload_fraction == 0.75
        assert config.optimizer_state_offload_fraction_per_pipeline_stage == fractions
    with pytest.raises(ValueError, match="one value"):
        config.for_pipeline_rank(2, 0)


@pytest.mark.parametrize("fractions", [(), (float("nan"),), (True,), (-0.1,), (1.1,)])
def test_pipeline_stage_fractions_reject_invalid_values(fractions):
    from megatron.core.optimizer import OptimizerConfig

    with pytest.raises(ValueError, match="finite number"):
        OptimizerConfig(
            optimizer="adam",
            use_distributed_optimizer=True,
            chunked_optimizer_state_offload=True,
            optimizer_state_offload_fraction_per_pipeline_stage=fractions,
        )
