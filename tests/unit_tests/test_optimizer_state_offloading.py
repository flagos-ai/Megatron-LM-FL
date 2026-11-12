# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for chunked optimizer state and master-weight offload."""

import logging
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import (
    ChunkedOptimizerStateOffloader,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training import arguments, checkpointing
from tests.unit_tests.test_utilities import Utils

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


def _run_training_forward_backward(model, optimizer, config, forward_backward_func, *, mxfp8=False):
    """Exercise train_step's real lifecycle and stop before unrelated training bookkeeping."""
    from megatron.training import training

    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=mxfp8,
        overlap_param_gather=mxfp8,
        seq_length=1,
        micro_batch_size=1,
        decoder_seq_length=None,
    )

    class ForwardBackwardComplete(Exception):
        pass

    def run_and_stop(**kwargs):
        forward_backward_func(**kwargs)
        raise ForwardBackwardComplete

    with (
        mock.patch.object(training, "get_args", return_value=args),
        mock.patch.object(training, "get_timers"),
        mock.patch.object(training, "get_rerun_state_machine"),
        mock.patch.object(training, "get_num_microbatches", return_value=1),
        mock.patch.object(training, "has_nvidia_modelopt", False),
        mock.patch.object(training.FullCudaGraphWrapper, "cuda_graph", {}),
        pytest.raises(ForwardBackwardComplete),
    ):
        training.train_step(None, None, [model], optimizer, None, config, run_and_stop)


@pytest.mark.parametrize("master_storage", ["mcore", "te", "remainders"])
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_training_offloads_masters_and_preserves_next_updates(master_storage, fraction):
    """Forward/backward runs with CPU masters and matches the resident Adam updates."""
    Utils.initialize_model_parallel()
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2, bf16=True)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.weights = torch.nn.ParameterList(
                    torch.nn.Parameter(
                        torch.full(
                            (256, 256), 0.01 * (index + 1), device="cuda", dtype=torch.bfloat16
                        )
                    )
                    for index in range(8 * Utils.world_size)
                )

            def forward(self):
                return sum(weight.float().square().mean() for weight in self.weights)

        def build(offload):
            model = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True, grad_reduce_in_fp32=True
                ),
                Model(),
                pg_collection=pg,
            )
            precision_aware = master_storage != "mcore"
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    bf16=True,
                    params_dtype=torch.bfloat16,
                    use_distributed_optimizer=True,
                    lr=0.001,
                    clip_grad=0.0,
                    use_precision_aware_optimizer=precision_aware,
                    store_param_remainders=master_storage == "remainders",
                    exp_avg_dtype=torch.bfloat16 if precision_aware else torch.float32,
                    exp_avg_sq_dtype=torch.bfloat16 if precision_aware else torch.float32,
                    chunked_optimizer_state_offload=offload,
                    optimizer_state_offload_chunk_size_mb=1,
                    optimizer_state_offload_fraction=fraction,
                ),
                [model],
            )
            assert isinstance(optimizer, ChainedOptimizer)
            active = [
                child for child in optimizer.chained_optimizers if not child.is_stub_optimizer
            ]
            assert len(active) == 1
            return model, optimizer, active[0]

        reference, reference_optimizer, _ = build(False)
        model, optimizer, child = build(True)
        manager = child._optimizer_state_offloader
        reference_config = SimpleNamespace(finalize_model_grads_func=reference.finish_grad_sync)
        training_config = SimpleNamespace(finalize_model_grads_func=model.finish_grad_sync)

        def check_master_residency():
            for param in manager.selected_params:
                if master_storage == "mcore":
                    assert param.device.type == "cpu"
                elif "master_param" in child.optimizer.state[param]:
                    assert child.optimizer.state[param]["master_param"].device.type == "cpu"
            for param in manager._resident_params:
                assert param.is_cuda
                master = child.optimizer.state.get(param, {}).get("master_param")
                if master is not None:
                    assert master.is_cuda

        for _ in range(3):
            for current_model, current_optimizer, current_config in (
                (reference, reference_optimizer, reference_config),
                (model, optimizer, training_config),
            ):

                def forward_backward(**kwargs):
                    if current_model is model:
                        check_master_residency()
                    current_model().backward()
                    current_config.finalize_model_grads_func()

                _run_training_forward_backward(
                    current_model, current_optimizer, current_config, forward_backward
                )
                assert current_optimizer.step()[0]
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Also release a prefetched window when a prepared update is skipped.
        optimizer.prepare_grads()
        _run_training_forward_backward(
            model, optimizer, training_config, lambda **kwargs: check_master_residency()
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "fraction,mxfp8,staging_enabled",
    [
        (1.0, False, False),
        (1.0, True, False),
        (1.0, True, True),
        (0.0, False, False),
        (0.0, True, True),
    ],
)
def test_training_drives_optimizer_state_offload_lifecycle(fraction, mxfp8, staging_enabled):
    """Training orders offload/staging/prefetch and treats zero fraction as disabled."""
    events = []
    child = mock.Mock(spec=DistributedOptimizer)
    child._copy_main_params_to_param_buffer.side_effect = lambda: events.append("stage")
    optimizer = mock.Mock()
    optimizer.config = SimpleNamespace(
        chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=fraction
    )
    optimizer.chained_optimizers = [child]
    optimizer.optimizer_state_offload_requires_pre_forward_param_sync.return_value = False
    optimizer.zero_grad.side_effect = lambda: events.append("zero_grad")
    optimizer.offload_optimizer_state_for_forward.side_effect = (
        lambda offload_master=True: events.append(f"offload:{offload_master}")
    )
    optimizer.prefetch_optimizer_state_for_gradient_finalization.side_effect = (
        lambda: events.append("prefetch")
    )
    model = mock.Mock()
    model.zero_grad_buffer.side_effect = lambda: events.append("zero_grad_buffer")
    model.remove_forward_pre_hook_handles = [object()] if staging_enabled else []
    config = SimpleNamespace(finalize_model_grads_func=lambda: events.append("finalize"))

    def forward_backward_func(**kwargs):
        events.append("forward_backward")
        config.finalize_model_grads_func()

    _run_training_forward_backward(model, optimizer, config, forward_backward_func, mxfp8=mxfp8)
    expected = []
    if fraction > 0.0:
        expected.append(f"offload:{not mxfp8}")
    expected.extend(["zero_grad_buffer", "zero_grad"])
    if staging_enabled:
        expected.append("stage")
    if fraction > 0.0 and mxfp8:
        expected.append("offload:True")
    expected.append("forward_backward")
    if fraction > 0.0:
        expected.append("prefetch")
    expected.append("finalize")
    assert events == expected


@pytest.mark.parametrize("custom_finalize", [False, True])
def test_training_rebinds_prefetch_without_nesting_finalize_hooks(custom_finalize):
    """Repeated iterations and optimizer replacement preserve exactly one prefetch hook."""
    from megatron.training import training

    events = []
    base_finalize = lambda: events.append("finalize")
    config = SimpleNamespace(finalize_model_grads_func=base_finalize if custom_finalize else None)
    model = mock.Mock()
    first = mock.Mock()
    second = mock.Mock()
    for name, optimizer in (("a", first), ("b", second)):
        optimizer.config = SimpleNamespace(chunked_optimizer_state_offload=True)
        optimizer.optimizer_state_offload_requires_pre_forward_param_sync.return_value = False
        optimizer.prefetch_optimizer_state_for_gradient_finalization.side_effect = (
            lambda name=name: events.append(f"prefetch:{name}")
        )

    with mock.patch.object(training, "finalize_model_grads", base_finalize):
        for optimizer in (first, first, second):
            _run_training_forward_backward(
                model, optimizer, config, lambda **kwargs: config.finalize_model_grads_func()
            )

    assert events == ["prefetch:a", "finalize", "prefetch:a", "finalize", "prefetch:b", "finalize"]
    wrapper = config.finalize_model_grads_func
    assert wrapper._chunked_optimizer_state_offload_wrapped_optimizer is second
    assert wrapper._chunked_optimizer_state_offload_base_finalize_model_grads_func is base_finalize


@pytest.mark.parametrize(
    "flag_name,fraction,use_fsdp",
    [
        (flag, fraction, use_fsdp)
        for flag in ("chunked_optimizer_state_offload", "offload_optimizer_states")
        for fraction, use_fsdp in ((1.0, True), (0.5, True), (0.0, True), (1.0, False))
    ]
    + [(None, 1.0, True)],
)
def test_training_args_reject_fsdp_with_active_state_offload(flag_name, fraction, use_fsdp):
    """CLI validation rejects active FSDP offload, including the deprecated spelling."""
    from megatron.training import arguments

    with mock.patch("sys.argv", ["test_optimizer_state_offloading.py"]):
        args = arguments.parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.optimizer = 'adam'
    args.use_distributed_optimizer = True
    args.use_megatron_fsdp = use_fsdp
    args.ckpt_format = 'fsdp_dtensor' if use_fsdp else 'torch_dist'
    if flag_name is not None:
        setattr(args, flag_name, True)
        args.optimizer_state_offload_fraction = fraction

    # Isolate startup validation from rank-local logging and unrelated FSDP CUDA tuning.
    with (
        mock.patch.dict("os.environ", {"CUDA_DEVICE_MAX_CONNECTIONS": "8"}),
        mock.patch.object(arguments, '_print_args'),
    ):
        if flag_name is not None and fraction > 0.0 and use_fsdp:
            with pytest.raises(AssertionError, match="does not support Megatron-FSDP"):
                arguments.validate_args(args)
        else:
            assert arguments.validate_args(args) is args


def _parse_args(options):
    with mock.patch('sys.argv', ['test_optimizer_state_offloading.py', *options]):
        args = arguments.parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.use_distributed_optimizer = True
    return args


def _validate_args(args):
    with mock.patch.object(arguments, '_print_args'):
        return arguments.validate_args(args)


@pytest.mark.parametrize(
    'flag', ['--chunked-optimizer-state-offload', '--offload-optimizer-states']
)
def test_offload_cli_reaches_optimizer_config(flag):
    from megatron.training.training import get_megatron_optimizer_config

    args = _parse_args(
        [
            flag,
            '--optimizer-state-offload-chunk-size-mb',
            '8',
            '--optimizer-state-offload-fraction',
            '0.5',
        ]
    )
    args.ckpt_format = 'torch_dist'
    _validate_args(args)
    if flag == '--offload-optimizer-states':
        with pytest.warns(FutureWarning, match='deprecated'):
            config, _ = get_megatron_optimizer_config(args)
    else:
        config, _ = get_megatron_optimizer_config(args)
    assert config.chunked_optimizer_state_offload
    assert not config.offload_optimizer_states
    assert config.optimizer_state_offload_chunk_size_mb == 8
    assert config.optimizer_state_offload_fraction == 0.5


def test_deprecated_alias_defaults_survive_config_copy():
    with pytest.warns(FutureWarning, match='deprecated'):
        config = OptimizerConfig(use_distributed_optimizer=True, offload_optimizer_states=True)
    copied = replace(config)
    assert copied.chunked_optimizer_state_offload
    assert not copied.offload_optimizer_states
    assert copied.optimizer_state_offload_chunk_size_mb == 0
    assert copied.optimizer_state_offload_fraction == 1.0


def test_existing_mxfp8_warning_still_works():
    with pytest.warns(UserWarning, match='mxfp8 without'):
        OptimizerConfig(fp8_recipe='mxfp8')


@pytest.mark.parametrize(
    'flag', ['--chunked-optimizer-state-offload', '--offload-optimizer-states']
)
@pytest.mark.parametrize(
    'ckpt_format,async_save,no_save,no_load,fraction,full_graph,error',
    [
        ('torch', False, False, False, 1.0, False, 'requires --ckpt-format torch_dist'),
        ('torch', False, True, False, 1.0, False, 'requires --ckpt-format torch_dist'),
        ('torch_dist', True, False, False, 1.0, False, 'does not support --async-save'),
        ('torch_dist', False, False, False, 1.0, True, 'does not support full-iteration'),
        ('torch_dist', False, False, False, 1.0, False, None),
        ('torch', True, True, True, 1.0, False, None),
        ('torch_dist', True, True, False, 1.0, False, None),
        ('torch', True, False, False, 0.0, False, None),
    ],
)
def test_cli_checkpoint_restrictions_apply_only_to_active_optimizer_io(
    flag, ckpt_format, async_save, no_save, no_load, fraction, full_graph, error
):
    args = _parse_args([flag, '--optimizer-state-offload-fraction', str(fraction)])
    args.ckpt_format = ckpt_format
    args.async_save = async_save
    args.use_persistent_ckpt_worker = True
    args.no_save_optim = no_save
    args.no_load_optim = no_load
    if full_graph:
        args.cuda_graph_impl = 'full_iteration'
        args.check_for_nan_in_loss_and_grad = False
    if error:
        with pytest.raises(AssertionError, match=error):
            _validate_args(args)
    else:
        assert _validate_args(args) is args


@pytest.mark.parametrize(
    'fractions,flag,error',
    [
        (['0.0', '0.5'], '--chunked-optimizer-state-offload', 'does not support --async-save'),
        (['0.0', '0.0'], '--chunked-optimizer-state-offload', None),
        (['0.0'], '--chunked-optimizer-state-offload', 'one finite value'),
        (['0.0', 'nan'], '--chunked-optimizer-state-offload', 'one finite value'),
        (['0.0', '1.1'], '--chunked-optimizer-state-offload', 'one finite value'),
        (['0.0', '0.5'], None, 'Per-stage fractions require'),
    ],
)
def test_per_stage_cli_validates_every_stage_before_optimizer_io(fractions, flag, error):
    options = ['--optimizer-state-offload-fraction-per-pipeline-stage', *fractions]
    if flag:
        options.insert(0, flag)
    args = _parse_args(options)
    args.pipeline_model_parallel_size = 2
    args.optimizer_state_offload_fraction = 0.0
    args.ckpt_format = 'torch_dist'
    args.async_save = True
    args.use_persistent_ckpt_worker = True
    if error:
        with pytest.raises((AssertionError, ValueError), match=error):
            _validate_args(args)
    else:
        assert _validate_args(args) is args


def test_per_stage_cli_reaches_optimizer_config():
    from megatron.training.training import get_megatron_optimizer_config

    args = _parse_args(
        [
            '--chunked-optimizer-state-offload',
            '--optimizer-state-offload-fraction-per-pipeline-stage',
            '0.0',
            '0.5',
        ]
    )
    args.pipeline_model_parallel_size = 2
    args.ckpt_format = 'torch_dist'
    _validate_args(args)
    config, _ = get_megatron_optimizer_config(args)
    assert config.optimizer_state_offload_fraction_per_pipeline_stage == (0.0, 0.5)
    assert config.for_pipeline_rank(2, 0).optimizer_state_offload_fraction == 0.0
    assert config.for_pipeline_rank(2, 1).optimizer_state_offload_fraction == 0.5


class _PastOffloadGuard(Exception):
    pass


@pytest.mark.parametrize(
    'fraction,no_save,ckpt_format,async_save,use_dist,error',
    [
        (1.0, False, 'torch_dist', True, True, 'does not support --async-save'),
        (1.0, False, 'torch', False, False, 'requires --ckpt-format torch_dist'),
        (1.0, False, 'torch_dist', False, False, 'requires --ckpt-format torch_dist'),
        (1.0, False, 'torch_dist', False, True, None),
        (0.0, False, 'torch', True, False, None),
        (1.0, True, 'torch', True, False, None),
    ],
)
def test_save_entry_rechecks_optimizer_checkpoint_safety(
    fraction, no_save, ckpt_format, async_save, use_dist, error
):
    args = SimpleNamespace(
        no_save_optim=no_save,
        ckpt_format=ckpt_format,
        async_save=async_save,
        use_dist_ckpt=use_dist,
    )
    optimizer = SimpleNamespace(
        config=SimpleNamespace(
            chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=fraction
        )
    )
    with (
        mock.patch.object(checkpointing, 'get_args', return_value=args),
        mock.patch.object(checkpointing, 'is_empty_async_queue', return_value=True),
        mock.patch.object(checkpointing, 'on_save_checkpoint_start', side_effect=_PastOffloadGuard),
    ):
        if error:
            with pytest.raises(RuntimeError, match=error):
                checkpointing.save_checkpoint(0, [], optimizer, None, 0)
        else:
            with pytest.raises(_PastOffloadGuard):
                checkpointing.save_checkpoint(0, [], optimizer, None, 0)


@pytest.mark.parametrize(
    'declared_format,detected_type',
    [
        (declared_format, detected_type)
        for declared_format in ('torch', 'torch_dist')
        for detected_type in (
            checkpointing.CheckpointType.LEGACY,
            checkpointing.CheckpointType.TORCH_DCP,
            checkpointing.CheckpointType.FSDP_DTENSOR,
        )
    ]
    + [('torch', checkpointing.CheckpointType.GLOBAL)],
)
def test_load_entry_rejects_detected_unsupported_optimizer_format(declared_format, detected_type):
    args = SimpleNamespace(
        load='/unused',
        pretrained_checkpoint=None,
        ckpt_format=declared_format,
        auto_detect_ckpt_format=False,
        no_load_optim=False,
        finetune=False,
    )
    optimizer = mock.Mock()
    optimizer.config = SimpleNamespace(chunked_optimizer_state_offload=True)
    with (
        mock.patch.object(checkpointing, 'get_args', return_value=args),
        mock.patch.object(checkpointing, 'unwrap_model', return_value=[]),
        mock.patch.object(
            checkpointing,
            '_load_base_checkpoint',
            return_value=({}, '/unused', False, detected_type),
        ) as load,
        pytest.raises(RuntimeError, match='requires --ckpt-format torch_dist'),
    ):
        checkpointing.load_checkpoint([], optimizer, None)
    assert load.call_count == 1
    optimizer.load_state_dict.assert_not_called()


@pytest.mark.parametrize(
    'no_load,finetune,release,fraction',
    [
        (True, False, False, 1.0),
        (False, True, False, 1.0),
        (False, False, True, 1.0),
        (False, False, False, 0.0),
    ],
)
def test_load_entry_preserves_model_only_and_disabled_paths(no_load, finetune, release, fraction):
    args = SimpleNamespace(
        load='/unused',
        pretrained_checkpoint=None,
        ckpt_format='torch',
        auto_detect_ckpt_format=True,
        no_load_optim=no_load,
        finetune=finetune,
    )
    optimizer = SimpleNamespace(
        config=SimpleNamespace(
            chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=fraction
        )
    )
    with (
        mock.patch.object(checkpointing, 'get_args', return_value=args),
        mock.patch.object(checkpointing, 'unwrap_model', return_value=[]),
        mock.patch.object(
            checkpointing,
            '_load_base_checkpoint',
            return_value=(None, '/unused', release, checkpointing.CheckpointType.LEGACY),
        ),
    ):
        assert checkpointing.load_checkpoint([], optimizer, None) == (0, 0)
