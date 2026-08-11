from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.pipeline_parallel import combined_1f1b, schedules
from megatron.core.process_groups_config import ProcessGroupCollection


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.transitions.append(("B", self.record["name"], None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.sink.transitions.append(("E", self.record["name"], exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []

    def is_enabled(self, name: str) -> bool:
        return True

    def scope(
        self,
        name: str,
        *,
        ctx: Mapping[str, Any] | None = None,
        slots: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> _RecordingScope:
        slot_names = tuple(slots or ())
        record = {
            "name": name,
            "ctx": dict(ctx or {}),
            "slots": slot_names,
            "attrs": dict(attrs or {}),
            "values": {slot: None for slot in slot_names},
        }
        self.records.append(record)
        return _RecordingScope(self, record)


class _PhaseModel(torch.nn.Module):
    def __init__(self, config, *, vp_stage: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.vp_stage = vp_stage
        self.model_type = "unit-test"
        self.input_tensor = None

    def set_input_tensor(self, input_tensor) -> None:
        self.input_tensor = input_tensor


class _ObservedMicrobatch(int):
    format_calls = 0

    def __format__(self, format_spec: str) -> str:
        type(self).format_calls += 1
        return super().__format__(format_spec)


def _phase_config(**overrides):
    values = {
        "timers": None,
        "enable_autocast": False,
        "autocast_dtype": torch.float32,
        "calculate_per_token_loss": True,
        "grad_scale_func": None,
        "deallocate_pipeline_outputs": False,
        "num_moe_experts": None,
        "mtp_num_layers": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _event_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {**record["ctx"], **record["values"]}


class _FakeGroup:
    def __init__(self, size: int = 1, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _NonInterleavedP2P:
    def __init__(self, config) -> None:
        self.config = config
        self.pp_group = _FakeGroup()
        self.is_pp_first_stage = True
        self.is_pp_last_stage = True
        self.total_stages = 1
        self.current_stage = 0

    def recv_forward(self, *args, **kwargs):
        return None

    def send_forward_recv_backward(self, *args, **kwargs):
        return None

    def send_backward_recv_forward(self, *args, **kwargs):
        return None

    def send_backward(self, *args, **kwargs) -> None:
        return None

    def warm_up_comm_group(self) -> None:
        return None


class _PP2NonInterleavedP2P:
    def __init__(self, config, *, stage: int, sequence_lengths: Sequence[int]) -> None:
        self.config = config
        self.pp_group = _FakeGroup(size=2, rank=stage)
        self.is_pp_first_stage = stage == 0
        self.is_pp_last_stage = stage == 1
        self.total_stages = 2
        self.current_stage = stage
        self._sequence_lengths = iter(sequence_lengths)
        self._sent_outputs: list[torch.Tensor] = []

    def _next_activation(self) -> torch.Tensor:
        sequence_length = next(self._sequence_lengths)
        return torch.ones((sequence_length, 2, 1), requires_grad=True)

    def recv_forward(self, *args, **kwargs):
        return None if self.is_pp_first_stage else self._next_activation()

    def send_forward(self, output_tensor, *args, **kwargs) -> None:
        if not self.is_pp_last_stage:
            self._sent_outputs.append(output_tensor)

    def send_forward_recv_backward(self, output_tensor, *args, **kwargs):
        if self.is_pp_last_stage:
            return None
        self._sent_outputs.append(output_tensor)
        return torch.ones_like(self._sent_outputs.pop(0))

    def send_backward_recv_forward(self, *args, **kwargs):
        return None if self.is_pp_first_stage else self._next_activation()

    def recv_backward(self, *args, **kwargs):
        return torch.ones_like(self._sent_outputs.pop(0))

    def send_backward(self, *args, **kwargs) -> None:
        return None

    def warm_up_comm_group(self) -> None:
        return None


class _InterleavedP2P:
    def __init__(self, config) -> None:
        self.config = config
        self.pp_group = _FakeGroup()
        self.virtual_pipeline_model_parallel_size = 2
        self.is_pp_last_stage = True

    def recv_forward(self, *args, **kwargs):
        return None

    def send_forward_recv_forward(self, output_tensor, *, recv_prev, **kwargs):
        return output_tensor if recv_prev else None


class _TrainingInterleavedP2P:
    def __init__(self, config) -> None:
        self.config = config
        self.pp_group = _FakeGroup(size=2, rank=1)
        self.virtual_pipeline_model_parallel_size = 2
        self.is_pp_last_stage = True

    @staticmethod
    def _activation():
        return torch.tensor(0.0, requires_grad=True)

    def recv_forward(self, *args, **kwargs):
        return self._activation()

    def send_forward_recv_forward(self, output_tensor, *, recv_prev, **kwargs):
        return self._activation() if recv_prev else None

    def send_forward_backward_recv_forward_backward(
        self, output_tensor, input_tensor_grad, *, recv_prev, recv_next, **kwargs
    ):
        input_tensor = self._activation() if recv_prev else None
        output_tensor_grad = torch.tensor(1.0) if recv_next else None
        return input_tensor, output_tensor_grad

    def send_backward_recv_backward(self, input_tensor_grad, *, recv_next, **kwargs):
        return torch.tensor(1.0) if recv_next else None


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def test_forward_phase_emits_nested_loss_with_shared_microbatch_identity() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    model = _PhaseModel(config, vp_stage=1)
    forward_data_store: list[Any] = []

    def forward_step_func(data_iterator, active_model):
        output = torch.tensor(2.0, requires_grad=True)

        def loss_func(value):
            return value, 7, {"loss": value.detach()}

        return output, loss_func

    output, num_tokens = schedules.forward_step(
        forward_step_func,
        data_iterator=None,
        model=model,
        num_microbatches=4,
        input_tensor=None,
        forward_data_store=forward_data_store,
        config=config,
        cp_group_size=1,
        current_microbatch=3,
        vp_stage=1,
        is_first_microbatch=False,
        is_last_stage=True,
    )

    assert output.requires_grad
    assert num_tokens == 7
    assert [record["name"] for record in sink.records] == ["forward-step", "forward-step-calc-loss"]
    assert _event_fields(sink.records[0]) == {
        "current_microbatch": 3,
        "vp_stage": 1,
        "is_first_microbatch": False,
        "is_last_stage": True,
        "timing_phase": "framework_phase",
        "operation_id": "pp:microbatch=3:vp=1",
        "num_tokens": 7,
        "sum_sq_seq_len": None,
    }
    assert _event_fields(sink.records[1]) == {
        "current_microbatch": 3,
        "vp_stage": 1,
        "is_first_microbatch": False,
        "is_last_stage": True,
        "timing_phase": "framework_phase",
        "operation_id": "pp:microbatch=3:vp=1",
    }
    assert sink.transitions == [
        ("B", "forward-step", None),
        ("B", "forward-step-calc-loss", None),
        ("E", "forward-step-calc-loss", None),
        ("E", "forward-step", None),
    ]


def test_forward_workload_uses_pre_loss_shape_and_preserves_tensor_token_count() -> None:
    class _NoItemTensor(torch.Tensor):
        def item(self):
            pytest.fail("pipeline workload read the loss token tensor")

    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    model = _PhaseModel(config)
    token_count = torch.tensor(6).as_subclass(_NoItemTensor)
    backward_workloads = []

    def forward_step_func(data_iterator, active_model):
        output = torch.ones((3, 2, 1), requires_grad=True)
        return output, lambda value: (value.sum(), token_count, {"loss": value.detach()})

    output, num_tokens = schedules.forward_step(
        forward_step_func,
        data_iterator=None,
        model=model,
        num_microbatches=1,
        input_tensor=None,
        forward_data_store=[],
        config=config,
        cp_group_size=1,
        current_microbatch=0,
        is_last_stage=True,
        record_pipeline_workload=True,
        backward_workload_queue=backward_workloads,
        workload_fallback_num_tokens=8,
        workload_tp_group_size=1,
    )

    assert output.ndim == 0
    assert num_tokens is token_count
    assert backward_workloads == [(6, 36.0)]
    assert _event_fields(sink.records[0])["num_tokens"] == 6
    assert _event_fields(sink.records[0])["sum_sq_seq_len"] == 36.0


def test_pipeline_workload_restores_parallel_shape_and_source_fallback_formula() -> None:
    config = _phase_config(sequence_parallel=True)

    assert schedules._pipeline_workload_from_output(
        torch.empty((3, 2, 1)), config, cp_group_size=2, tp_group_size=4, fallback_num_tokens=99
    ) == (48, 2304.0)
    assert schedules._pipeline_workload_from_output(
        torch.empty(()), config, cp_group_size=2, tp_group_size=4, fallback_num_tokens=6
    ) == (6, 36.0)


def test_backward_phase_preserves_gradient_and_forward_operation_identity() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    input_tensor = torch.tensor(3.0, requires_grad=True)
    output_tensor = input_tensor * 2

    input_grad = schedules.backward_step(
        input_tensor,
        output_tensor,
        output_tensor_grad=None,
        config=config,
        current_microbatch=3,
        vp_stage=1,
        is_first_microbatch=False,
        is_last_stage=True,
    )

    assert input_grad is input_tensor.grad
    assert input_grad.item() == 2.0
    assert [record["name"] for record in sink.records] == ["backward-step"]
    assert _event_fields(sink.records[0]) == {
        "current_microbatch": 3,
        "vp_stage": 1,
        "is_first_microbatch": False,
        "is_last_stage": True,
        "timing_phase": "framework_phase",
        "operation_id": "pp:microbatch=3:vp=1",
        "num_tokens": None,
        "sum_sq_seq_len": None,
    }
    assert sink.transitions == [("B", "backward-step", None), ("E", "backward-step", None)]


def test_backward_default_schedule_metadata_remains_unknown() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    input_tensor = torch.tensor(3.0, requires_grad=True)
    output_tensor = input_tensor * 2

    input_grad = schedules.backward_step(
        input_tensor, output_tensor, output_tensor_grad=None, config=config
    )

    assert input_grad.item() == 2.0
    fields = _event_fields(sink.records[0])
    assert fields["current_microbatch"] is None
    assert fields["vp_stage"] is None
    assert fields["is_first_microbatch"] is None
    assert fields["is_last_stage"] is None
    assert fields["operation_id"] is None


def test_non_interleaved_1f1b_pairs_forward_and_backward_operation_ids() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config(
        overlap_p2p_comm=False,
        finalize_model_grads_func=None,
        barrier_with_L1_time=False,
        no_sync_func=None,
        num_microbatches_with_partial_activation_checkpoints=None,
        variable_seq_lengths=False,
        sequence_parallel=False,
        hidden_size=1,
        pipeline_dtype=torch.float32,
        grad_sync_func=None,
        calculate_per_token_loss=False,
        defer_embedding_wgrad_compute=False,
        fine_grained_activation_offloading=False,
    )
    model = _PhaseModel(config)
    p2p = _NonInterleavedP2P(config)
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = _FakeGroup()
    pg_collection.cp = _FakeGroup()

    def forward_step_func(data_iterator, active_model):
        leaf = torch.tensor(float(next(data_iterator)), requires_grad=True)
        output = leaf * 1.0
        return output, lambda value: (value, {"loss": value.detach()})

    losses = schedules.forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step_func,
        data_iterator=iter([1, 2]),
        model=model,
        num_microbatches=2,
        seq_length=1,
        micro_batch_size=1,
        forward_only=False,
        p2p_communicator=p2p,
        pg_collection=pg_collection,
    )

    assert len(losses) == 2
    forward_ids = [
        _event_fields(record)["operation_id"]
        for record in sink.records
        if record["name"] == "forward-step"
    ]
    backward_ids = [
        _event_fields(record)["operation_id"]
        for record in sink.records
        if record["name"] == "backward-step"
    ]
    assert forward_ids == ["pp:microbatch=0:vp=none", "pp:microbatch=1:vp=none"]
    assert backward_ids == forward_ids
    forward_workloads = [
        (_event_fields(record)["num_tokens"], _event_fields(record)["sum_sq_seq_len"])
        for record in sink.records
        if record["name"] == "forward-step"
    ]
    backward_workloads = [
        (_event_fields(record)["num_tokens"], _event_fields(record)["sum_sq_seq_len"])
        for record in sink.records
        if record["name"] == "backward-step"
    ]
    assert forward_workloads == [(1, 1.0), (1, 1.0)]
    assert backward_workloads == forward_workloads
    open_scopes: list[str] = []
    for phase, name, error_type in sink.transitions:
        if phase == "B":
            open_scopes.append(name)
        else:
            assert open_scopes.pop() == name
            assert error_type is None
    assert open_scopes == []


@pytest.mark.parametrize(
    ("stage", "expected_phase_order"),
    [
        (0, [("F", 0), ("F", 1), ("B", 0), ("F", 2), ("B", 1), ("B", 2)]),
        (1, [("F", 0), ("B", 0), ("F", 1), ("B", 1), ("F", 2), ("B", 2)]),
    ],
)
def test_non_interleaved_pp2_workload_fifo_follows_activation_fifo(
    stage: int, expected_phase_order: list[tuple[str, int]]
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config(
        overlap_p2p_comm=False,
        finalize_model_grads_func=None,
        barrier_with_L1_time=False,
        no_sync_func=None,
        num_microbatches_with_partial_activation_checkpoints=None,
        variable_seq_lengths=True,
        sequence_parallel=False,
        hidden_size=1,
        pipeline_dtype=torch.float32,
        grad_sync_func=None,
        calculate_per_token_loss=True,
        defer_embedding_wgrad_compute=False,
        fine_grained_activation_offloading=False,
    )
    model = _PhaseModel(config)
    sequence_lengths = (2, 3, 4)
    p2p = _PP2NonInterleavedP2P(config, stage=stage, sequence_lengths=sequence_lengths)
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = _FakeGroup()
    pg_collection.cp = _FakeGroup()
    stage_zero_lengths = iter(sequence_lengths)

    def forward_step_func(data_iterator, active_model):
        input_tensor = active_model.input_tensor[0]
        if input_tensor is None:
            input_tensor = torch.ones((next(stage_zero_lengths), 2, 1), requires_grad=True)
        output = input_tensor * 1.0
        token_count = torch.tensor(output.shape[0] * output.shape[1])
        return output, lambda value: (value.sum(), token_count, {"loss": value.detach()})

    schedules.forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step_func,
        data_iterator=None,
        model=model,
        num_microbatches=3,
        seq_length=4,
        micro_batch_size=2,
        forward_only=False,
        p2p_communicator=p2p,
        pg_collection=pg_collection,
    )

    phase_records = [
        record for record in sink.records if record["name"] in {"forward-step", "backward-step"}
    ]
    assert [
        (
            "F" if record["name"] == "forward-step" else "B",
            _event_fields(record)["current_microbatch"],
        )
        for record in phase_records
    ] == expected_phase_order

    expected_workloads = [(0, 4, 16.0), (1, 6, 36.0), (2, 8, 64.0)]
    for event_name in ("forward-step", "backward-step"):
        assert [
            (
                _event_fields(record)["current_microbatch"],
                _event_fields(record)["num_tokens"],
                _event_fields(record)["sum_sq_seq_len"],
            )
            for record in sink.records
            if record["name"] == event_name
        ] == expected_workloads


def test_grad_sync_error_closes_phase_and_propagates_from_no_pipeline_schedule() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    finalize_error = RuntimeError("gradient finalization failed")

    def fail_finalize(*args, **kwargs):
        raise finalize_error

    config = _phase_config(
        overlap_moe_expert_parallel_comm=False,
        hybrid_context_parallel=False,
        finalize_model_grads_func=fail_finalize,
        barrier_with_L1_time=False,
        no_sync_func=None,
        calculate_per_token_loss=False,
        fine_grained_activation_offloading=False,
    )
    model = _PhaseModel(config)
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = _FakeGroup()
    pg_collection.cp = _FakeGroup()

    def forward_step_func(data_iterator, active_model):
        leaf = torch.tensor(1.0, requires_grad=True)
        output = leaf * 1.0
        return output, lambda value: (value, {"loss": value.detach()})

    with pytest.raises(RuntimeError, match="gradient finalization failed") as raised:
        schedules.forward_backward_no_pipelining(
            forward_step_func=forward_step_func,
            data_iterator=None,
            model=model,
            num_microbatches=1,
            seq_length=1,
            micro_batch_size=1,
            forward_only=False,
            pg_collection=pg_collection,
        )

    assert raised.value is finalize_error
    backward_record = next(record for record in sink.records if record["name"] == "backward-step")
    assert _event_fields(backward_record)["operation_id"] == "pp:microbatch=0:vp=none"
    assert _event_fields(backward_record)["is_last_stage"] is True
    assert (
        _event_fields(backward_record)["num_tokens"],
        _event_fields(backward_record)["sum_sq_seq_len"],
    ) == (1, 1.0)
    forward_record = next(record for record in sink.records if record["name"] == "forward-step")
    assert (
        _event_fields(forward_record)["num_tokens"],
        _event_fields(forward_record)["sum_sq_seq_len"],
    ) == (1, 1.0)
    grad_record = next(record for record in sink.records if record["name"] == "grad-sync")
    assert _event_fields(grad_record) == {
        "schedule": "no-pipelining",
        "timing_phase": "framework_phase",
    }
    assert ("E", "grad-sync", RuntimeError) in sink.transitions


def test_interleaved_forward_only_identifies_each_virtual_stage_without_backward() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config(
        overlap_p2p_comm=False,
        batch_p2p_comm=False,
        overlap_p2p_comm_warmup_flush=False,
        overlap_moe_expert_parallel_comm=True,
        finalize_model_grads_func=None,
        barrier_with_L1_time=False,
        no_sync_func=None,
        grad_sync_func=None,
        param_sync_func=None,
        microbatch_group_size_per_vp_stage=1,
        num_microbatches_with_partial_activation_checkpoints=None,
        sequence_parallel=False,
        hidden_size=1,
        virtual_pipeline_model_parallel_size=2,
        calculate_per_token_loss=False,
        fine_grained_activation_offloading=False,
    )
    models = [_PhaseModel(config, vp_stage=0), _PhaseModel(config, vp_stage=1)]
    p2p = _InterleavedP2P(config)
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = _FakeGroup()
    pg_collection.cp = _FakeGroup()

    def forward_step_func(data_iterator, active_model):
        leaf = torch.tensor(float(next(data_iterator)), requires_grad=True)
        output = leaf * 1.0
        return output, lambda value: (value, {"loss": value.detach()})

    losses = schedules.forward_backward_pipelining_with_interleaving(
        forward_step_func=forward_step_func,
        data_iterator=[iter([1]), iter([2])],
        model=models,
        num_microbatches=1,
        seq_length=1,
        micro_batch_size=1,
        forward_only=True,
        p2p_communicator=p2p,
        pg_collection=pg_collection,
    )

    assert len(losses) == 1
    forward_records = [record for record in sink.records if record["name"] == "forward-step"]
    assert [
        (_event_fields(record)["current_microbatch"], _event_fields(record)["vp_stage"])
        for record in forward_records
    ] == [(0, 0), (0, 1)]
    assert [_event_fields(record)["operation_id"] for record in forward_records] == [
        "pp:microbatch=0:vp=0",
        "pp:microbatch=0:vp=1",
    ]
    assert [_event_fields(record)["is_last_stage"] for record in forward_records] == [False, True]
    assert [
        (
            _event_fields(record)["num_tokens"],
            _event_fields(record)["sum_sq_seq_len"],
        )
        for record in forward_records
    ] == [(1, 1.0), (1, 1.0)]
    assert all(record["name"] != "backward-step" for record in sink.records)
    assert all(record["name"] != "grad-sync" for record in sink.records)


def test_interleaved_1f1b_pairs_backward_with_virtual_stage_operation(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config(
        overlap_p2p_comm=False,
        batch_p2p_comm=False,
        overlap_p2p_comm_warmup_flush=False,
        overlap_moe_expert_parallel_comm=False,
        finalize_model_grads_func=None,
        barrier_with_L1_time=False,
        no_sync_func=None,
        grad_sync_func=None,
        param_sync_func=None,
        microbatch_group_size_per_vp_stage=2,
        num_microbatches_with_partial_activation_checkpoints=None,
        sequence_parallel=False,
        hidden_size=1,
        virtual_pipeline_model_parallel_size=2,
        calculate_per_token_loss=False,
        fine_grained_activation_offloading=False,
    )
    models = [_PhaseModel(config, vp_stage=0), _PhaseModel(config, vp_stage=1)]
    p2p = _TrainingInterleavedP2P(config)
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = _FakeGroup()
    pg_collection.cp = _FakeGroup()

    def workload_from_value(output_tensor, config, **kwargs):
        value = int(output_tensor.detach())
        return value, float(value**2)

    monkeypatch.setattr(schedules, "_pipeline_workload_from_output", workload_from_value)

    def forward_step_func(data_iterator, active_model):
        leaf = torch.tensor(float(next(data_iterator)), requires_grad=True)
        output = leaf * 1.0
        return output, lambda value: (value, {"loss": value.detach()})

    losses = schedules.forward_backward_pipelining_with_interleaving(
        forward_step_func=forward_step_func,
        data_iterator=[iter([1, 2]), iter([3, 4])],
        model=models,
        num_microbatches=2,
        seq_length=1,
        micro_batch_size=1,
        forward_only=False,
        p2p_communicator=p2p,
        pg_collection=pg_collection,
    )

    assert len(losses) == 2
    forward_ids = {
        _event_fields(record)["operation_id"]
        for record in sink.records
        if record["name"] == "forward-step"
    }
    backward_ids = {
        _event_fields(record)["operation_id"]
        for record in sink.records
        if record["name"] == "backward-step"
    }
    assert forward_ids == {
        "pp:microbatch=0:vp=0",
        "pp:microbatch=1:vp=0",
        "pp:microbatch=0:vp=1",
        "pp:microbatch=1:vp=1",
    }
    assert backward_ids == forward_ids
    expected_workloads = {
        "pp:microbatch=0:vp=0": (1, 1.0),
        "pp:microbatch=1:vp=0": (2, 4.0),
        "pp:microbatch=0:vp=1": (3, 9.0),
        "pp:microbatch=1:vp=1": (4, 16.0),
    }
    for event_name in ("forward-step", "backward-step"):
        assert {
            _event_fields(record)["operation_id"]: (
                _event_fields(record)["num_tokens"],
                _event_fields(record)["sum_sq_seq_len"],
            )
            for record in sink.records
            if record["name"] == event_name
        } == expected_workloads


def test_multimodule_backward_accepts_schedule_identity_and_emits_phase() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    input_tensor = torch.tensor(2.0, requires_grad=True)
    output_tensor = input_tensor * 3.0

    input_grads = schedules.backward_step_multimodule(
        {"language": input_tensor},
        {"language": output_tensor},
        output_tensor_grad=None,
        config=config,
        language_model_module_name="language",
        current_microbatch=1,
        vp_stage=0,
        is_first_microbatch=False,
        is_last_stage=True,
    )

    assert input_grads["language"] is input_tensor.grad
    assert input_grads["language"].item() == 3.0
    backward_record = next(record for record in sink.records if record["name"] == "backward-step")
    assert _event_fields(backward_record)["operation_id"] == "pp:microbatch=1:vp=0"


def test_null_sink_preserves_phase_results_without_building_trace_context(monkeypatch) -> None:
    config = _phase_config()
    model = _PhaseModel(config)
    microbatch = _ObservedMicrobatch(2)
    _ObservedMicrobatch.format_calls = 0
    monkeypatch.setattr(
        schedules,
        "_pipeline_phase_context",
        lambda **kwargs: pytest.fail("trace-off path built pipeline phase context"),
    )
    monkeypatch.setattr(
        schedules,
        "_pipeline_grad_sync_context",
        lambda schedule: pytest.fail("trace-off path built grad-sync context"),
    )
    monkeypatch.setattr(
        schedules,
        "_pipeline_workload_from_output",
        lambda *args, **kwargs: pytest.fail("trace-off path extracted pipeline workload"),
    )
    token_count = torch.tensor(5)
    backward_workloads = []

    def forward_step_func(data_iterator, active_model):
        output = torch.ones((2, 1), requires_grad=True)
        return output, lambda value: (value.sum(), token_count, {"loss": value.detach()})

    output, num_tokens = schedules.forward_step(
        forward_step_func,
        data_iterator=None,
        model=model,
        num_microbatches=3,
        input_tensor=None,
        forward_data_store=[],
        config=config,
        cp_group_size=1,
        current_microbatch=microbatch,
        is_last_stage=True,
        record_pipeline_workload=True,
        backward_workload_queue=backward_workloads,
        workload_fallback_num_tokens=2,
        workload_tp_group_size=1,
    )

    assert output.item() == 2.0
    assert num_tokens is token_count
    assert backward_workloads == [None]
    assert _ObservedMicrobatch.format_calls == 0

    input_grad = schedules.backward_step(
        input_tensor=None,
        output_tensor=output,
        output_tensor_grad=None,
        config=config,
        current_microbatch=microbatch,
    )
    with schedules._pipeline_grad_sync_scope("unit-test"):
        pass

    assert input_grad is None
    assert _ObservedMicrobatch.format_calls == 0


def test_loss_error_closes_nested_phases_and_propagates_exact_exception() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    config = _phase_config()
    model = _PhaseModel(config)
    loss_error = RuntimeError("loss calculation failed")

    def forward_step_func(data_iterator, active_model):
        output = torch.tensor(2.0, requires_grad=True)

        def fail_loss(value):
            raise loss_error

        return output, fail_loss

    with pytest.raises(RuntimeError, match="loss calculation failed") as raised:
        schedules.forward_step(
            forward_step_func,
            data_iterator=None,
            model=model,
            num_microbatches=1,
            input_tensor=None,
            forward_data_store=[],
            config=config,
            cp_group_size=1,
            current_microbatch=0,
            is_first_microbatch=True,
            is_last_stage=True,
        )

    assert raised.value is loss_error
    assert sink.transitions == [
        ("B", "forward-step", None),
        ("B", "forward-step-calc-loss", None),
        ("E", "forward-step-calc-loss", RuntimeError),
        ("E", "forward-step", RuntimeError),
    ]


def test_combined_no_pipeline_schedule_records_both_operation_identities(monkeypatch) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    model = object()
    monkeypatch.setattr(combined_1f1b, "set_streams", lambda: None)

    def fake_combined_step(*args, **kwargs):
        del kwargs
        f_model = args[2]
        return object() if f_model is not None else None, 1 if f_model is not None else None, None

    monkeypatch.setattr(combined_1f1b, "_combined_forward_backward_step_impl", fake_combined_step)

    forward_data_store, total_num_tokens = combined_1f1b.combined_1f1b_schedule_for_no_pipelining(
        forward_step_func=lambda *args, **kwargs: None,
        data_iterator=None,
        model=model,
        num_microbatches=3,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=[],
        config=SimpleNamespace(),
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=False,
        no_sync_func=contextlib.nullcontext,
        total_num_tokens=0,
        check_first_val_step=lambda value: value,
    )

    assert forward_data_store == []
    assert total_num_tokens == 3
    records = [
        record for record in sink.records if record["name"] == "combined-forward-backward-step"
    ]
    assert [record["ctx"]["execution_mode"] for record in records] == [
        "forward",
        "combined",
        "combined",
        "backward",
    ]
    assert [record["ctx"]["forward_microbatch"] for record in records] == [0, 1, 2, None]
    assert [record["ctx"]["backward_microbatch"] for record in records] == [None, 0, 1, 2]
    assert [record["ctx"]["forward_operation_id"] for record in records] == [
        "pp:microbatch=0:vp=none",
        "pp:microbatch=1:vp=none",
        "pp:microbatch=2:vp=none",
        None,
    ]
    assert [record["ctx"]["backward_operation_id"] for record in records] == [
        None,
        "pp:microbatch=0:vp=none",
        "pp:microbatch=1:vp=none",
        "pp:microbatch=2:vp=none",
    ]


def test_combined_schedule_null_sink_skips_identity_context(monkeypatch) -> None:
    monkeypatch.setattr(combined_1f1b, "set_streams", lambda: None)
    monkeypatch.setattr(
        combined_1f1b,
        "_build_combined_step_context",
        lambda **kwargs: pytest.fail("trace-off path built combined schedule context"),
    )
    monkeypatch.setattr(
        combined_1f1b,
        "_combined_forward_backward_step_impl",
        lambda *args, **kwargs: (object() if args[2] is not None else None, 1, None),
    )

    _, total_num_tokens = combined_1f1b.combined_1f1b_schedule_for_no_pipelining(
        forward_step_func=lambda *args, **kwargs: None,
        data_iterator=None,
        model=object(),
        num_microbatches=1,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=[],
        config=SimpleNamespace(),
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=False,
        no_sync_func=contextlib.nullcontext,
        total_num_tokens=0,
        check_first_val_step=lambda value: value,
    )

    assert total_num_tokens == 1


def test_combined_interleaved_schedule_records_forward_and_backward_vp_identity(
    monkeypatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    monkeypatch.setattr(combined_1f1b, "set_streams", lambda: None)
    monkeypatch.setattr(
        combined_1f1b,
        "_combined_forward_backward_step_impl",
        lambda *args, **kwargs: (object(), 1, None),
    )
    forward_post_calls = []
    backward_post_calls = []

    combined_1f1b.combined_1f1b_schedule_for_interleaved_pipelining(
        config=SimpleNamespace(),
        forward_step_func=lambda *args, **kwargs: None,
        data_iterator=[None, None],
        model=[object(), object()],
        num_microbatches=5,
        forward_data_store=[],
        forward_step_helper_preprocess=lambda *args: None,
        forward_step_helper_postprocess=lambda *args: forward_post_calls.append(args),
        backward_step_helper_preprocess=lambda *args: (None, None, None),
        backward_step_helper_postprocess=lambda *args: backward_post_calls.append(args),
        get_microbatch_id_in_model_chunk=lambda virtual_id, forward: 4 if forward else 3,
        get_model_chunk_id=lambda virtual_id, forward: 0 if forward else 1,
        check_first_val_step=lambda value: value,
        is_first_microbatch_for_model_chunk=lambda virtual_id: False,
        collect_non_loss_data=False,
        f_virtual_microbatch_id=7,
        b_virtual_microbatch_id=8,
    )

    record = next(
        record for record in sink.records if record["name"] == "combined-forward-backward-step"
    )
    assert record["ctx"]["execution_mode"] == "combined"
    assert record["ctx"]["forward_microbatch"] == 4
    assert record["ctx"]["backward_microbatch"] == 3
    assert record["ctx"]["forward_vp_stage"] == 0
    assert record["ctx"]["backward_vp_stage"] == 1
    assert record["ctx"]["forward_operation_id"] == "pp:microbatch=4:vp=0"
    assert record["ctx"]["backward_operation_id"] == "pp:microbatch=3:vp=1"
    assert len(forward_post_calls) == 1
    assert backward_post_calls == [(8,)]
