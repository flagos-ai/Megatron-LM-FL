# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

import megatron.core.transformer.moe.moe_layer as moe_layer_module
import megatron.core.transformer.moe.moe_utils as moe_utils_module
import megatron.core.transformer.moe.router as router_module
from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.observability import (
    EXPERT_WORKLOAD_SLOTS,
    ROUTER_WORKLOAD_SLOTS,
    collect_router_loss_fields,
    expert_workload,
    observe_router_loss,
    router_trace_context,
    router_workload,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.megalens.core_adapter import MegaLensTraceSink
from megatron.megalens.trace import Tracer

ROOT = Path(__file__).resolve().parents[2]


class _RecordingScope:
    def __init__(self, sink: "_RecordingSink", record: dict[str, Any]) -> None:
        self.sink = sink
        self.record = record

    def __enter__(self) -> "_RecordingScope":
        self.sink.active.append(self.record["name"])
        self.sink.transitions.append(("B", self.record["name"], None))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self.sink.active.pop() == self.record["name"]
        self.sink.transitions.append(("E", self.record["name"], exc_type))
        return False

    def get(self, key: str) -> Any | None:
        return self.record["ctx"].get(key)

    def set(self, key: str, value: Any) -> bool:
        if key not in self.record["values"]:
            return False
        self.record["values"][key] = value
        return True


class _RecordingSink:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, type[BaseException] | None]] = []
        self.active: list[str] = []
        self.gate_calls: list[str] = []

    def is_enabled(self, name: str) -> bool:
        self.gate_calls.append(name)
        return self.enabled

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


class _Group:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _Dispatcher:
    def __init__(
        self,
        sink: _RecordingSink,
        events: list[tuple[Any, ...]],
        *,
        dispatch_result: tuple[torch.Tensor, torch.Tensor] | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        dispatch_error: BaseException | None = None,
        postprocess_error: BaseException | None = None,
        combine_error: BaseException | None = None,
    ) -> None:
        self.sink = sink
        self.events = events
        self.dispatch_result = dispatch_result
        self.tokens_per_expert = (
            tokens_per_expert if tokens_per_expert is not None else torch.tensor([1, 3])
        )
        self.dispatch_error = dispatch_error
        self.postprocess_error = postprocess_error
        self.combine_error = combine_error
        self.dispatched_input = torch.tensor([[11.0], [12.0], [13.0], [14.0]])
        self.permuted_probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.combined_output = torch.tensor([[21.0], [22.0]])

    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        self.events.append(("dispatch", tuple(self.sink.active), hidden_states, probs))
        if self.dispatch_error is not None:
            raise self.dispatch_error
        return self.dispatch_result or (hidden_states, probs)

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        self.events.append(("dispatch-postprocess", tuple(self.sink.active)))
        if self.postprocess_error is not None:
            raise self.postprocess_error
        return self.dispatched_input, self.tokens_per_expert, self.permuted_probs

    def combine_preprocess(self, expert_output: torch.Tensor):
        self.events.append(("combine-preprocess", tuple(self.sink.active), expert_output))
        return self.combined_output

    def token_combine(self, output: torch.Tensor):
        self.events.append(("combine", tuple(self.sink.active), output))
        if self.combine_error is not None:
            raise self.combine_error
        return self.combined_output


class _Experts(torch.nn.Module):
    def __init__(
        self,
        sink: _RecordingSink,
        events: list[tuple[Any, ...]],
        *,
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.sink = sink
        self.events = events
        self.error = error
        self.output = torch.tensor([[31.0], [32.0]])

    def forward(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        **kwargs: Any,
    ):
        self.events.append(
            (
                "experts",
                tuple(self.sink.active),
                dispatched_input,
                tokens_per_expert,
                permuted_probs,
                kwargs,
            )
        )
        if self.error is not None:
            raise self.error
        return self.output, None


class _SharedExperts(torch.nn.Module):
    def __init__(
        self,
        sink: _RecordingSink,
        events: list[tuple[Any, ...]],
        *,
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.sink = sink
        self.events = events
        self.error = error
        self.output = torch.tensor([[41.0], [42.0]])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.events.append(("shared-experts", tuple(self.sink.active), hidden_states))
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture(autouse=True)
def _reset_sink_and_group_size(monkeypatch: pytest.MonkeyPatch):
    reset_trace_sink()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    yield
    reset_trace_sink()


def _router_fixture(
    sink: _RecordingSink,
    events: list[tuple[Any, ...]],
    *,
    routing_error: BaseException | None = None,
    loss_observations: Sequence[tuple[str, torch.Tensor]] = (),
    losses_require_grad: bool = False,
) -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    logits = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    probs = torch.tensor(
        [[0.8, 0.2, 0.0, 0.0], [0.6, 0.0, 0.4, 0.0], [0.0, 0.7, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]]
    )
    routing_map = probs > 0

    def maintain() -> None:
        events.append(("maintain-bias", tuple(sink.active)))

    def jitter(actual: torch.Tensor) -> torch.Tensor:
        events.append(("jitter", tuple(sink.active), actual))
        return actual

    def gating(actual: torch.Tensor) -> torch.Tensor:
        events.append(("gating", tuple(sink.active), actual))
        return logits

    def routing(actual_logits: torch.Tensor, *, padding_mask=None, input_ids=None):
        events.append(("routing", tuple(sink.active), actual_logits, padding_mask, input_ids))
        if not losses_require_grad or torch.is_grad_enabled():
            for name, value in loss_observations:
                observe_router_loss(name, value)
        if routing_error is not None:
            raise routing_error
        return probs, routing_map

    config = SimpleNamespace(
        num_moe_experts=4,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=False,
    )
    router = SimpleNamespace(
        config=config,
        ep_group=[_Group(2)],
        layer_number=7,
        topk=2,
        _maintain_float32_expert_bias=maintain,
        apply_input_jitter=jitter,
        gating=gating,
        routing=routing,
    )
    return router, input_tensor, probs, routing_map


def _layer_fixture(
    sink: _RecordingSink,
    events: list[tuple[Any, ...]],
    *,
    dispatch_error: BaseException | None = None,
    postprocess_error: BaseException | None = None,
    expert_error: BaseException | None = None,
    combine_error: BaseException | None = None,
    tokens_per_expert: torch.Tensor | None = None,
) -> tuple[SimpleNamespace, _Dispatcher, _Experts]:
    dispatcher = _Dispatcher(
        sink,
        events,
        dispatch_error=dispatch_error,
        postprocess_error=postprocess_error,
        combine_error=combine_error,
        tokens_per_expert=tokens_per_expert,
    )
    experts = _Experts(sink, events, error=expert_error)
    layer = SimpleNamespace(
        config=SimpleNamespace(
            num_moe_experts=4,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
            moe_expert_capacity_factor=1.25,
        ),
        ep_group=[_Group(2)],
        layer_number=7,
        num_local_experts=2,
        token_dispatcher=dispatcher,
        experts=experts,
    )
    return layer, dispatcher, experts


def _shared_expert_fixture(
    sink: _RecordingSink,
    events: list[tuple[Any, ...]],
    *,
    use_shared_expert: bool = True,
    shared_expert_overlap: bool = False,
    shared_experts_recompute: bool = False,
    fp8: bool = False,
    fp4: bool = False,
    error: BaseException | None = None,
) -> tuple[SimpleNamespace, _SharedExperts, torch.Tensor]:
    shared_experts = _SharedExperts(sink, events, error=error)
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    layer = SimpleNamespace(
        config=SimpleNamespace(cuda_graph_impl="none", fp8=fp8, fp4=fp4),
        ep_group=[_Group(2)],
        layer_number=7,
        use_shared_expert=use_shared_expert,
        shared_expert_overlap=shared_expert_overlap,
        shared_experts_recompute=shared_experts_recompute,
        shared_experts=shared_experts,
    )
    return layer, shared_experts, hidden_states


def _event_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {**record["ctx"], **record["values"]}


def test_router_emits_target_workload_and_uses_routing_boundary() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    router, input_tensor, probs, routing_map = _router_fixture(sink, events)

    actual_probs, actual_map = TopKRouter.forward(router, input_tensor)

    assert actual_probs is probs
    assert actual_map is routing_map
    assert sink.gate_calls == ["moe-router"]
    assert sink.transitions == [("B", "moe-router", None), ("E", "moe-router", None)]
    assert [event[:2] for event in events] == [
        ("maintain-bias", ()),
        ("jitter", ()),
        ("gating", ()),
        ("routing", ("moe-router",)),
    ]

    record = sink.records[0]
    assert record["ctx"] == {
        "layer": 7,
        "num_experts": 4,
        "num_local_experts": 2,
        "ep_size": 2,
        "router_topk": 2,
    }
    assert record["slots"] == ROUTER_WORKLOAD_SLOTS
    fields = _event_fields(record)
    assert fields["aux_loss"] is None
    assert fields["z_loss"] is None
    assert fields["num_tokens"] == 4
    assert fields["routed_tokens"] == 6
    assert fields["dropped_tokens"] is None
    assert fields["drop_rate"] is None
    assert fields["expert_cv"] == pytest.approx(math.sqrt(1.25) / 1.5)
    assert fields["top1_expert_share"] == pytest.approx(0.5)
    expected_entropy = (
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
        - (0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        - (0.7 * math.log(0.7))
        - (0.5 * math.log(0.5))
    ) / 4
    assert fields["routing_entropy"] == pytest.approx(expected_entropy)


@pytest.mark.parametrize(
    "aux_name", ["load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss"]
)
def test_router_emits_one_normalized_aux_subtype_and_z_loss(aux_name: str) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    router, input_tensor, probs, routing_map = _router_fixture(
        sink, [], loss_observations=[(aux_name, torch.tensor(1.25)), ("z_loss", torch.tensor(0.75))]
    )

    actual_probs, actual_map = TopKRouter.forward(router, input_tensor)

    assert actual_probs is probs
    assert actual_map is routing_map
    fields = _event_fields(sink.records[0])
    assert fields["aux_loss"] == pytest.approx(1.25)
    assert fields["z_loss"] == pytest.approx(0.75)


def test_multiple_aux_subtypes_fail_closed_without_hiding_z_loss() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    router, input_tensor, _, _ = _router_fixture(
        sink,
        [],
        loss_observations=[
            ("load_balancing_loss", torch.tensor(1.0)),
            ("seq_load_balancing_loss", torch.tensor(2.0)),
            ("z_loss", torch.tensor(3.0)),
        ],
    )

    TopKRouter.forward(router, input_tensor)

    fields = _event_fields(sink.records[0])
    assert fields["aux_loss"] is None
    assert fields["z_loss"] == pytest.approx(3.0)


def test_loss_scalarization_is_deferred_and_nonfinite_values_fail_closed() -> None:
    class _ObservedScalar:
        def __init__(self, value: float) -> None:
            self.value = value
            self.detach_calls = 0
            self.item_calls = 0

        def detach(self):
            self.detach_calls += 1
            return self

        def __truediv__(self, coefficient: float):
            return self

        def item(self) -> float:
            self.item_calls += 1
            return self.value

    first = _ObservedScalar(1.0)
    second = _ObservedScalar(2.0)
    with collect_router_loss_fields() as ambiguous:
        observe_router_loss("load_balancing_loss", first)  # type: ignore[arg-type]
        observe_router_loss("seq_load_balancing_loss", second)  # type: ignore[arg-type]

    assert ambiguous.fields() == {"aux_loss": None, "z_loss": None}
    assert (first.detach_calls, first.item_calls) == (1, 0)
    assert (second.detach_calls, second.item_calls) == (1, 0)

    with collect_router_loss_fields() as invalid:
        observe_router_loss("load_balancing_loss", torch.tensor(float("nan")))
        observe_router_loss("z_loss", torch.tensor(float("inf")))
    assert invalid.fields() == {"aux_loss": None, "z_loss": None}

    with collect_router_loss_fields() as nonscalar:
        observe_router_loss("load_balancing_loss", torch.tensor([1.0, 2.0]))
    assert nonscalar.fields() == {"aux_loss": None, "z_loss": None}


def test_router_loss_collection_is_nested_and_exception_safe() -> None:
    with collect_router_loss_fields() as outer:
        observe_router_loss("load_balancing_loss", torch.tensor(1.0))
        with collect_router_loss_fields() as inner:
            observe_router_loss("global_load_balancing_loss", torch.tensor(2.0))
            observe_router_loss("z_loss", torch.tensor(3.0))
        observe_router_loss("z_loss", torch.tensor(4.0))

    assert inner.fields() == {"aux_loss": 2.0, "z_loss": 3.0}
    assert outer.fields() == {"aux_loss": 1.0, "z_loss": 4.0}

    error = RuntimeError("loss collection failed")
    with pytest.raises(RuntimeError) as raised:
        with collect_router_loss_fields():
            observe_router_loss("load_balancing_loss", torch.tensor(5.0))
            raise error
    assert raised.value is error

    # The failed collector was reset; an observation outside a scope is ignored.
    observe_router_loss("z_loss", torch.tensor(6.0))
    with collect_router_loss_fields() as fresh:
        pass
    assert fresh.fields() == {"aux_loss": None, "z_loss": None}


def test_routing_exception_discards_loss_without_scalar_sync() -> None:
    class _DeferredLoss:
        def __init__(self) -> None:
            self.item_calls = 0

        def detach(self):
            return self

        def __truediv__(self, coefficient: float):
            return self

        def item(self) -> float:
            self.item_calls += 1
            return 1.0

    sink = _RecordingSink()
    install_trace_sink(sink)
    error = RuntimeError("routing failed after loss")
    loss = _DeferredLoss()
    router, input_tensor, _, _ = _router_fixture(
        sink,
        [],
        routing_error=error,
        loss_observations=[("load_balancing_loss", loss)],  # type: ignore[list-item]
    )

    with pytest.raises(RuntimeError) as raised:
        TopKRouter.forward(router, input_tensor)

    assert raised.value is error
    assert loss.item_calls == 0
    assert sink.records[0]["values"]["aux_loss"] is None
    assert sink.active == []


@pytest.mark.parametrize(
    "aux_name", ["load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss"]
)
def test_aux_tracker_and_mtp_training_scaling_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch, aux_name: str
) -> None:
    tracker_calls: list[tuple[Any, ...]] = []
    attached_losses: list[torch.Tensor] = []

    def save_tracker(*args: Any, **kwargs: Any) -> None:
        tracker_calls.append((*args, kwargs))

    def attach(activation: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        attached_losses.append(loss)
        return activation

    monkeypatch.setattr(router_module, "save_to_aux_losses_tracker", save_tracker)
    monkeypatch.setattr(router_module.MoEAuxLossAutoScaler, "apply", staticmethod(attach))

    group = object()
    owner = SimpleNamespace(
        is_mtp_layer=True,
        config=SimpleNamespace(mtp_use_repeated_layer=True, mtp_num_layers=4, num_layers=8),
        layer_number=2,
        calculate_per_token_loss=True,
    )
    activation = torch.ones((2, 3))
    coefficient = 0.2
    coefficient_scaled_loss = torch.tensor(0.8)

    with collect_router_loss_fields() as collector:
        result = TopKRouter.attach_and_log_load_balancing_loss(
            owner,
            activation,
            coefficient,
            coefficient_scaled_loss,
            aux_name,
            group,
            valid_token_count=torch.tensor(3),
        )

    assert result is activation
    assert len(tracker_calls) == 1
    tracker_name, tracker_value, layer_number, num_layers, tracker_kwargs = tracker_calls[0]
    assert tracker_name == aux_name
    torch.testing.assert_close(tracker_value, torch.tensor(1.0))
    assert layer_number == 10
    assert num_layers == 12
    assert tracker_kwargs == {"reduce_group": group, "reduce_group_has_dp": False}
    torch.testing.assert_close(attached_losses[0], torch.tensor(0.6))
    # Source-compatible observation happens at each _apply_* producer before
    # this target-specific repeated-MTP scaling point.
    assert collector.fields() == {"aux_loss": None, "z_loss": None}


@pytest.mark.parametrize("aux_kind", ["aux_loss", "seq_aux_loss", "global_aux_loss"])
def test_aux_producers_emit_source_base_before_target_mtp_scaling(
    monkeypatch: pytest.MonkeyPatch, aux_kind: str
) -> None:
    coefficient = 0.2
    base_loss = 5.0
    seq_length = 2
    batch_size = 2
    attached: list[tuple[Any, ...]] = []

    def loss_func(**kwargs: Any) -> torch.Tensor:
        multiplier = batch_size if aux_kind == "seq_aux_loss" else 1
        return torch.tensor(coefficient * base_loss * multiplier)

    def attach(
        activation: torch.Tensor,
        actual_coefficient: float,
        loss: torch.Tensor,
        name: str,
        group: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        attached.append((activation, actual_coefficient, loss, name, group, kwargs))
        return activation

    monkeypatch.setattr(router_module, "switch_load_balancing_loss_func", loss_func)
    monkeypatch.setattr(
        router_module,
        "get_tokens_per_expert_and_token_count",
        lambda **kwargs: (torch.tensor([1.0, 1.0]), torch.tensor(2.0), torch.tensor(4.0)),
    )

    tp_cp_group = _Group(2)
    tp_dp_cp_group = _Group(4)
    owner = SimpleNamespace(
        get_aux_loss_coeff=lambda name: coefficient if name == aux_kind else 0.0,
        tp_cp_group=tp_cp_group,
        tp_dp_cp_group=tp_dp_cp_group,
        topk=2,
        config=SimpleNamespace(num_moe_experts=4, moe_router_fusion=False),
        attach_and_log_load_balancing_loss=attach,
        global_tokens_per_expert=torch.zeros(2),
        ga_steps=torch.tensor(0.0),
    )
    probs = torch.ones((4, 4))
    scores = torch.ones((4, 4))
    routing_map = torch.ones((4, 4), dtype=torch.bool)

    with collect_router_loss_fields() as collector:
        if aux_kind == "aux_loss":
            result = TopKRouter._apply_aux_loss(owner, probs, scores, routing_map)
        elif aux_kind == "seq_aux_loss":
            result = TopKRouter._apply_seq_aux_loss(
                owner, probs, scores, routing_map, seq_length, batch_size
            )
        else:
            result = TopKRouter._apply_global_aux_loss(owner, probs, scores, routing_map)

    assert result is probs
    assert collector.fields() == {"aux_loss": pytest.approx(base_loss), "z_loss": None}
    assert len(attached) == 1
    assert attached[0][0] is probs
    assert attached[0][1] == coefficient
    torch.testing.assert_close(attached[0][2], torch.tensor(coefficient * base_loss))


def test_z_loss_observation_uses_source_base_and_preserves_existing_mtp_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker_calls: list[tuple[Any, ...]] = []
    attached_losses: list[torch.Tensor] = []
    z_loss_calls: list[tuple[torch.Tensor, float, torch.Tensor | None]] = []

    def z_loss_func(
        logits: torch.Tensor, coefficient: float, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        z_loss_calls.append((logits, coefficient, padding_mask))
        return torch.tensor(0.8)

    def save_tracker(*args: Any, **kwargs: Any) -> None:
        tracker_calls.append((*args, kwargs))

    def attach(logits: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        attached_losses.append(loss)
        return logits

    monkeypatch.setattr(router_module, "z_loss_func", z_loss_func)
    monkeypatch.setattr(router_module, "save_to_aux_losses_tracker", save_tracker)
    monkeypatch.setattr(router_module.MoEAuxLossAutoScaler, "apply", staticmethod(attach))

    owner = SimpleNamespace(
        config=SimpleNamespace(
            moe_z_loss_coeff=0.4, mtp_use_repeated_layer=True, mtp_num_layers=4, num_layers=8
        ),
        training=True,
        tp_cp_group=_Group(2),
        calculate_per_token_loss=True,
        is_mtp_layer=True,
        layer_number=2,
    )
    logits = torch.ones((3, 4))
    padding_mask = torch.tensor([False, True, False])

    with collect_router_loss_fields() as collector:
        result = TopKRouter.apply_z_loss(owner, logits, padding_mask=padding_mask)

    assert result is logits
    assert len(z_loss_calls) == 1
    assert z_loss_calls[0][0] is logits
    assert z_loss_calls[0][1] == pytest.approx(0.2)
    assert z_loss_calls[0][2] is padding_mask
    # Existing target training semantics attach the token-scaled loss before
    # repeated-MTP tracker normalization. The probe keeps the source base.
    torch.testing.assert_close(attached_losses[0], torch.tensor(1.6))
    tracker_name, tracker_value, layer_number, num_layers, tracker_kwargs = tracker_calls[0]
    assert tracker_name == "z_loss"
    torch.testing.assert_close(tracker_value, torch.tensor(1.0))
    assert layer_number == 10
    assert num_layers == 12
    assert tracker_kwargs == {}
    assert collector.fields() == {"aux_loss": None, "z_loss": pytest.approx(4.0)}


def test_disabled_z_loss_paths_and_trace_off_skip_scalar_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.ones((2, 2))
    owner = SimpleNamespace(config=SimpleNamespace(moe_z_loss_coeff=0.1), training=False)

    def fail_z_loss(*args: Any, **kwargs: Any):
        pytest.fail("disabled z-loss path performed loss computation")

    monkeypatch.setattr(router_module, "z_loss_func", fail_z_loss)
    with collect_router_loss_fields() as eval_collector:
        assert TopKRouter.apply_z_loss(owner, logits) is logits

    owner.training = True
    with torch.no_grad(), collect_router_loss_fields() as no_grad_collector:
        assert TopKRouter.apply_z_loss(owner, logits) is logits

    assert eval_collector.fields() == {"aux_loss": None, "z_loss": None}
    assert no_grad_collector.fields() == {"aux_loss": None, "z_loss": None}

    class _NoMaterialize:
        def detach(self):
            pytest.fail("trace-off observation materialized a scalar")

    observe_router_loss("load_balancing_loss", _NoMaterialize())  # type: ignore[arg-type]

    with collect_router_loss_fields() as compiler_collector:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        observe_router_loss("z_loss", _NoMaterialize())  # type: ignore[arg-type]
    assert compiler_collector.fields() == {"aux_loss": None, "z_loss": None}


def test_enabled_disabled_enabled_router_calls_do_not_reuse_loss_values() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    observations = [("load_balancing_loss", torch.tensor(1.5))]
    router, input_tensor, _, _ = _router_fixture(sink, [], loss_observations=observations)

    TopKRouter.forward(router, input_tensor)
    observations.clear()
    TopKRouter.forward(router, input_tensor)
    observations.append(("load_balancing_loss", torch.tensor(2.5)))
    TopKRouter.forward(router, input_tensor)

    assert len(sink.records) == 3
    assert [_event_fields(record)["aux_loss"] for record in sink.records] == [1.5, None, 2.5]
    assert [_event_fields(record)["z_loss"] for record in sink.records] == [None, None, None]


def test_checkpoint_style_no_grad_then_recompute_uses_per_invocation_loss() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    router, input_tensor, _, _ = _router_fixture(
        sink,
        [],
        loss_observations=[
            ("load_balancing_loss", torch.tensor(1.5)),
            ("z_loss", torch.tensor(0.5)),
        ],
        losses_require_grad=True,
    )

    with torch.no_grad():
        TopKRouter.forward(router, input_tensor)
    TopKRouter.forward(router, input_tensor)

    assert len(sink.records) == 2
    assert _event_fields(sink.records[0])["aux_loss"] is None
    assert _event_fields(sink.records[0])["z_loss"] is None
    assert _event_fields(sink.records[1])["aux_loss"] == pytest.approx(1.5)
    assert _event_fields(sink.records[1])["z_loss"] == pytest.approx(0.5)


@pytest.mark.parametrize("group_container", ["direct", "list"])
def test_router_topology_uses_production_process_group_adapter(group_container: str) -> None:
    group = _Group(2)
    router = SimpleNamespace(
        config=SimpleNamespace(num_moe_experts=4),
        ep_group=group if group_container == "direct" else [group],
        layer_number=None,
        topk=2,
    )

    assert router_trace_context(router) == {
        "layer": None,
        "num_experts": 4,
        "num_local_experts": 2,
        "ep_size": 2,
        "router_topk": 2,
    }


def test_pad_to_capacity_keeps_assignment_workload_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    probs_before_drop = torch.tensor(
        [[0.9, 0.8, 0.0, 0.0], [0.7, 0.6, 0.0, 0.0], [0.5, 0.4, 0.0, 0.0], [0.3, 0.2, 0.0, 0.0]]
    )
    routing_map_before_drop = probs_before_drop > 0
    monkeypatch.setattr(
        router_module,
        "topk_routing_with_score_function",
        lambda *args, **kwargs: (probs_before_drop, routing_map_before_drop),
    )
    config = SimpleNamespace(
        num_moe_experts=4,
        moe_router_pre_softmax=False,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_topk_scaling_factor=None,
        moe_router_fusion=False,
        moe_expert_capacity_factor=1.0,
        moe_token_drop_policy="probs",
        moe_pad_expert_input_to_capacity=True,
    )
    router = SimpleNamespace(
        config=config,
        topk=2,
        score_function="softmax",
        expert_bias=None,
        router_replay=None,
        is_hash_layer=False,
        routing_type="none",
        training=False,
        apply_z_loss=lambda logits, padding_mask=None: logits,
        _apply_expert_bias=lambda routing_map, padding_mask=None: None,
    )

    probs, padded_routing_map = TopKRouter.routing(router, torch.ones((4, 1, 4)))
    fields = router_workload(probs, padded_routing_map, capacity_factor=1.0, pad_to_capacity=True)

    assert int(padded_routing_map.sum().item()) == 8
    assert fields["routed_tokens"] is None
    assert fields["dropped_tokens"] is None
    assert fields["drop_rate"] is None
    assert fields["expert_cv"] is None
    assert fields["top1_expert_share"] is None


def test_shared_expert_emits_source_fields_around_only_the_module_call() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, shared_experts, hidden_states = _shared_expert_fixture(sink, events)

    output = MoELayer.shared_experts_compute(layer, hidden_states)

    assert output is shared_experts.output
    assert sink.gate_calls == ["moe-shared-expert"]
    assert sink.transitions == [("B", "moe-shared-expert", None), ("E", "moe-shared-expert", None)]
    assert sink.records[0]["ctx"] == {"layer": 7, "ep_size": 2}
    assert sink.records[0]["slots"] == ()
    assert events == [("shared-experts", ("moe-shared-expert",), hidden_states)]


@pytest.mark.parametrize("mode", ["torch", "fp8", "fp4"])
def test_shared_expert_checkpoint_branches_keep_source_scope_and_arguments(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, shared_experts, hidden_states = _shared_expert_fixture(
        sink, events, shared_experts_recompute=True, fp8=mode == "fp8", fp4=mode == "fp4"
    )
    checkpoint_calls: list[tuple[Any, ...]] = []

    if mode == "torch":

        def checkpoint(function, distribute_saved_activations, actual_hidden_states):
            checkpoint_calls.append(
                ("torch", tuple(sink.active), distribute_saved_activations, actual_hidden_states)
            )
            return function(actual_hidden_states)

        monkeypatch.setattr(moe_layer_module.tensor_parallel, "checkpoint", checkpoint)
    else:
        tp_group = object()

        def te_checkpoint(
            function,
            distribute_saved_activations,
            rng_tracker_getter,
            actual_tp_group,
            actual_hidden_states,
        ):
            checkpoint_calls.append(
                (
                    "te",
                    tuple(sink.active),
                    distribute_saved_activations,
                    rng_tracker_getter,
                    actual_tp_group,
                    actual_hidden_states,
                )
            )
            return function(actual_hidden_states)

        monkeypatch.setattr(moe_layer_module, "te_checkpoint", te_checkpoint)
        monkeypatch.setattr(
            moe_layer_module.parallel_state, "get_tensor_model_parallel_group", lambda: tp_group
        )

    output = MoELayer.shared_experts_compute(layer, hidden_states)

    assert output is shared_experts.output
    assert checkpoint_calls[0][0] == ("torch" if mode == "torch" else "te")
    assert checkpoint_calls[0][1] == ("moe-shared-expert",)
    assert checkpoint_calls[0][2] is False
    assert checkpoint_calls[0][-1] is hidden_states
    assert len(checkpoint_calls) == 1
    if mode != "torch":
        assert (
            checkpoint_calls[0][3] is moe_layer_module.tensor_parallel.random.get_cuda_rng_tracker
        )
        assert checkpoint_calls[0][4] is tp_group
    assert events == [("shared-experts", ("moe-shared-expert",), hidden_states)]


@pytest.mark.parametrize("use_shared_expert,shared_expert_overlap", [(False, False), (True, True)])
def test_shared_expert_ineligible_paths_emit_nothing(
    use_shared_expert: bool, shared_expert_overlap: bool
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, _, hidden_states = _shared_expert_fixture(
        sink,
        events,
        use_shared_expert=use_shared_expert,
        shared_expert_overlap=shared_expert_overlap,
    )

    assert MoELayer.shared_experts_compute(layer, hidden_states) is None
    assert sink.gate_calls == []
    assert sink.records == []
    assert events == []


@pytest.mark.parametrize("gate_mode", ["disabled", "suppressed"])
def test_shared_expert_closed_gate_preserves_compute_and_skips_context(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    install_trace_sink(sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None)
    events: list[tuple[Any, ...]] = []
    layer, shared_experts, hidden_states = _shared_expert_fixture(sink, events)
    monkeypatch.setattr(
        moe_layer_module,
        "shared_experts_trace_context",
        lambda owner: pytest.fail("closed shared-expert gate constructed context"),
    )

    output = MoELayer.shared_experts_compute(layer, hidden_states)

    assert output is shared_experts.output
    assert events == [("shared-experts", (), hidden_states)]
    assert sink.gate_calls == ["moe-shared-expert"]
    assert sink.records == []


def test_shared_expert_null_sink_preserves_compute_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_trace_sink()
    events: list[tuple[Any, ...]] = []
    local_sink = _RecordingSink()
    layer, shared_experts, hidden_states = _shared_expert_fixture(local_sink, events)
    monkeypatch.setattr(
        moe_layer_module,
        "shared_experts_trace_context",
        lambda owner: pytest.fail("Null sink constructed shared-expert context"),
    )

    output = MoELayer.shared_experts_compute(layer, hidden_states)

    assert output is shared_experts.output
    assert events == [("shared-experts", (), hidden_states)]
    assert local_sink.gate_calls == []
    assert local_sink.records == []


def test_shared_expert_cuda_graph_replay_uses_cached_output_without_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, _, hidden_states = _shared_expert_fixture(sink, events)
    cached_output = torch.tensor([[51.0]])
    layer.config.cuda_graph_impl = "transformer_engine"
    layer.cudagraph_tensor_store = SimpleNamespace(
        is_empty=lambda: False, shared_expert_output=cached_output
    )
    monkeypatch.setattr(moe_utils_module, "is_graph_capturing", lambda: False)

    output = MoELayer.shared_experts_compute(layer, hidden_states)

    assert output is cached_output
    assert sink.gate_calls == []
    assert sink.records == []
    assert events == []


def test_shared_expert_error_closes_scope_and_preserves_exception_identity() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    error = RuntimeError("shared expert failed")
    layer, _, hidden_states = _shared_expert_fixture(sink, events, error=error)

    with pytest.raises(RuntimeError) as raised:
        MoELayer.shared_experts_compute(layer, hidden_states)

    assert raised.value is error
    assert sink.transitions == [
        ("B", "moe-shared-expert", None),
        ("E", "moe-shared-expert", RuntimeError),
    ]
    assert sink.active == []


def test_dispatch_and_combine_emit_target_static_fields_and_preserve_results() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, dispatcher, _ = _layer_fixture(sink, events)
    hidden_states = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    probs = torch.arange(6, dtype=torch.float32)

    dispatch_result = MoELayer.dispatch(layer, hidden_states, probs)
    combined = MoELayer.combine(layer, hidden_states)

    assert dispatch_result[0] is hidden_states
    assert dispatch_result[1] is probs
    assert combined is dispatcher.combined_output
    assert [record["name"] for record in sink.records] == ["moe-dispatch", "moe-combine"]
    assert _event_fields(sink.records[0]) == {
        "layer": 7,
        "ep_size": 2,
        "num_experts": 4,
        "num_local_experts": 2,
        "router_topk": 2,
        "dispatcher": "alltoall",
        "num_tokens": 6,
        "capacity_factor": 1.25,
    }
    assert _event_fields(sink.records[1]) == {
        "layer": 7,
        "ep_size": 2,
        "num_experts": 4,
        "num_local_experts": 2,
        "dispatcher": "alltoall",
        "num_tokens": 6,
    }
    assert [event[:2] for event in events] == [
        ("dispatch", ("moe-dispatch",)),
        ("combine", ("moe-combine",)),
    ]


def test_experts_emit_local_workload_around_only_the_expert_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, dispatcher, experts = _layer_fixture(sink, events)
    hidden_states = torch.tensor([[1.0], [2.0]])
    probs = torch.tensor([0.25, 0.75])
    original_expert_workload = moe_layer_module.expert_workload

    def checked_workload(tokens_per_expert: torch.Tensor):
        events.append(("expert-workload", tuple(sink.active)))
        return original_expert_workload(tokens_per_expert)

    monkeypatch.setattr(moe_layer_module, "expert_workload", checked_workload)

    output, bias = MoELayer.routed_experts_compute(layer, hidden_states, probs)

    assert output is dispatcher.combined_output
    assert bias is None
    assert [event[:2] for event in events] == [
        ("dispatch-postprocess", ()),
        ("expert-workload", ()),
        ("experts", ("moe-experts",)),
        ("combine-preprocess", ()),
    ]
    expert_call = events[2]
    assert expert_call[2] is dispatcher.dispatched_input
    assert expert_call[3] is dispatcher.tokens_per_expert
    assert expert_call[4] is dispatcher.permuted_probs
    assert expert_call[5] == {}
    assert experts.output is events[3][2]

    record = sink.records[0]
    assert record["ctx"] == {"layer": 7, "ep_size": 2, "num_experts": 4, "num_local_experts": 2}
    assert record["slots"] == EXPERT_WORKLOAD_SLOTS
    assert record["values"] == {
        "routed_tokens": 4,
        "expert_cv": pytest.approx(0.5),
        "top1_expert_share": pytest.approx(0.75),
        "expert_max_over_mean": pytest.approx(1.5),
        "tokens_per_expert": [1, 3],
    }


def test_zero_workloads_are_finite_and_preserve_the_full_expert_array() -> None:
    router_fields = router_workload(
        torch.empty((0, 4)),
        torch.empty((0, 4), dtype=torch.bool),
        capacity_factor=None,
        pad_to_capacity=False,
    )
    expert_fields = expert_workload(torch.zeros(3, dtype=torch.int64))
    inference_fields = expert_workload(None)
    large_count_fields = expert_workload(torch.tensor([2**24 + 1, 0], dtype=torch.int64))
    unknown_drop_fields = router_workload(
        torch.ones((1, 1)),
        torch.ones((1, 1), dtype=torch.bool),
        capacity_factor=1.0,
        pad_to_capacity=False,
    )

    assert router_fields == {
        "num_tokens": 0,
        "routed_tokens": 0,
        "dropped_tokens": 0,
        "drop_rate": 0.0,
        "expert_cv": 0.0,
        "top1_expert_share": 0.0,
        "routing_entropy": 0.0,
    }
    assert expert_fields == {
        "routed_tokens": 0,
        "expert_cv": 0.0,
        "top1_expert_share": 0.0,
        "expert_max_over_mean": 0.0,
        "tokens_per_expert": [0, 0, 0],
    }
    assert inference_fields == {}
    assert large_count_fields["routed_tokens"] == 2**24 + 1
    assert large_count_fields["tokens_per_expert"] == [2**24 + 1, 0]
    assert unknown_drop_fields["dropped_tokens"] is None
    assert unknown_drop_fields["drop_rate"] is None


def test_inference_tokens_per_expert_none_preserves_null_workload_slots() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    layer, dispatcher, _ = _layer_fixture(sink, events)
    dispatcher.tokens_per_expert = None
    dispatcher.routing_map = torch.tensor([[True, False], [False, True]])
    layer._inference_token_dispatcher = object()
    layer.is_inference_cuda_graphed_iteration = True

    output, bias = MoELayer.routed_experts_compute(layer, torch.ones((2, 2)), torch.ones(2))

    assert output is dispatcher.combined_output
    assert bias is None
    assert sink.records[0]["values"] == {slot: None for slot in EXPERT_WORKLOAD_SLOTS}
    expert_call = next(event for event in events if event[0] == "experts")
    assert expert_call[3] is None
    assert expert_call[5] == {"routing_map": dispatcher.routing_map}


@pytest.mark.parametrize("gate_mode", ["disabled", "suppressed"])
def test_closed_gate_skips_all_moe_metadata_and_workload(
    monkeypatch: pytest.MonkeyPatch, gate_mode: str
) -> None:
    sink = _RecordingSink(enabled=gate_mode != "disabled")
    install_trace_sink(sink, suppress_scope=(lambda: True) if gate_mode == "suppressed" else None)
    events: list[tuple[Any, ...]] = []

    class _NoMaterialize:
        def detach(self):
            pytest.fail("closed gate detached a loss tensor")

    router, input_tensor, probs, routing_map = _router_fixture(
        sink,
        events,
        loss_observations=[
            ("load_balancing_loss", _NoMaterialize()),  # type: ignore[list-item]
            ("z_loss", _NoMaterialize()),  # type: ignore[list-item]
        ],
    )
    layer, dispatcher, _ = _layer_fixture(sink, events)

    def fail(*args: Any, **kwargs: Any):
        pytest.fail("closed MoE gate constructed metadata or workload")

    for name in ("router_trace_context", "router_workload"):
        monkeypatch.setattr(router_module, name, fail)
    monkeypatch.setattr(router_module, "collect_router_loss_fields", fail)
    for name in (
        "dispatch_trace_context",
        "experts_trace_context",
        "expert_workload",
        "combine_trace_context",
    ):
        monkeypatch.setattr(moe_layer_module, name, fail)

    actual_probs, actual_map = TopKRouter.forward(router, input_tensor)
    dispatch_result = MoELayer.dispatch(layer, input_tensor, probs)
    expert_result = MoELayer.routed_experts_compute(layer, input_tensor, probs)
    combine_result = MoELayer.combine(layer, input_tensor)

    assert actual_probs is probs
    assert actual_map is routing_map
    assert dispatch_result[0] is input_tensor
    assert dispatch_result[1] is probs
    assert expert_result[0] is dispatcher.combined_output
    assert expert_result[1] is None
    assert combine_result is dispatcher.combined_output
    assert sink.records == []
    assert sink.transitions == []
    assert sink.gate_calls == ["moe-router", "moe-dispatch", "moe-experts", "moe-combine"]


@pytest.mark.parametrize("phase", ["router", "dispatch", "experts", "combine"])
def test_moe_phase_errors_close_active_scope_and_preserve_exception_identity(phase: str) -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    error = RuntimeError(f"{phase} failed")

    if phase == "router":
        owner, input_tensor, _, _ = _router_fixture(sink, events, routing_error=error)
        call = lambda: TopKRouter.forward(owner, input_tensor)
        event_name = "moe-router"
    else:
        kwargs = {f"{phase[:-1] if phase == 'experts' else phase}_error": error}
        if phase == "experts":
            kwargs = {"expert_error": error}
        owner, _, _ = _layer_fixture(sink, events, **kwargs)
        input_tensor = torch.ones((2, 2))
        probs = torch.ones(2)
        if phase == "dispatch":
            call = lambda: MoELayer.dispatch(owner, input_tensor, probs)
            event_name = "moe-dispatch"
        elif phase == "experts":
            call = lambda: MoELayer.routed_experts_compute(owner, input_tensor, probs)
            event_name = "moe-experts"
        else:
            call = lambda: MoELayer.combine(owner, input_tensor)
            event_name = "moe-combine"

    with pytest.raises(RuntimeError) as raised:
        call()

    assert raised.value is error
    assert sink.transitions == [("B", event_name, None), ("E", event_name, RuntimeError)]
    assert sink.active == []


def test_dispatch_postprocess_error_precedes_the_expert_scope() -> None:
    sink = _RecordingSink()
    install_trace_sink(sink)
    events: list[tuple[Any, ...]] = []
    error = RuntimeError("postprocess failed")
    layer, _, _ = _layer_fixture(sink, events, postprocess_error=error)

    with pytest.raises(RuntimeError) as raised:
        MoELayer.routed_experts_compute(layer, torch.ones((2, 2)), torch.ones(2))

    assert raised.value is error
    assert sink.gate_calls == []
    assert sink.records == []
    assert sink.transitions == []


def test_real_adapter_places_static_fields_on_begin_and_workload_on_end() -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="full")
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    ticks: list[tuple[str, str, dict[str, Any]]] = []
    tracer._tick = lambda name, phase, attrs: ticks.append((name, phase, dict(attrs)))
    install_trace_sink(MegaLensTraceSink(tracer))
    sink = _RecordingSink()
    events: list[tuple[Any, ...]] = []
    router, input_tensor, _, _ = _router_fixture(
        sink,
        events,
        loss_observations=[
            ("load_balancing_loss", torch.tensor(1.25)),
            ("z_loss", torch.tensor(0.75)),
        ],
    )

    TopKRouter.forward(router, input_tensor)

    assert [tick[:2] for tick in ticks] == [("moe-router", "B"), ("moe-router", "E")]
    assert ticks[0][2] == {
        "layer": 7,
        "num_experts": 4,
        "num_local_experts": 2,
        "ep_size": 2,
        "router_topk": 2,
    }
    assert ticks[1][2]["aux_loss"] == pytest.approx(1.25)
    assert ticks[1][2]["z_loss"] == pytest.approx(0.75)
    assert ticks[1][2]["num_tokens"] == 4
    assert ticks[1][2]["routed_tokens"] == 6
    assert ticks[1][2]["dropped_tokens"] is None


def test_real_adapter_records_shared_expert_source_fields() -> None:
    tracer = Tracer()
    tracer.global_args = SimpleNamespace(trace=True, trace_mode=1, trace_granularity="full")
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    ticks: list[tuple[str, str, dict[str, Any]]] = []
    tracer._tick = lambda name, phase, attrs: ticks.append((name, phase, dict(attrs)))
    install_trace_sink(MegaLensTraceSink(tracer))
    events: list[tuple[Any, ...]] = []
    layer, shared_experts, hidden_states = _shared_expert_fixture(_RecordingSink(), events)

    output = MoELayer.shared_experts_compute(layer, hidden_states)

    assert output is shared_experts.output
    assert ticks == [
        ("moe-shared-expert", "B", {"layer": 7, "ep_size": 2}),
        ("moe-shared-expert", "E", {}),
    ]


def test_moe_probe_markers_and_public_signatures_remain_stable() -> None:
    assert getattr(TopKRouter.forward, "__megatron_trace_event__", None) == "moe-router"
    assert getattr(MoELayer.dispatch, "__megatron_trace_event__", None) == "moe-dispatch"
    assert (
        getattr(MoELayer.shared_experts_compute, "__megatron_trace_event__", None)
        == "moe-shared-expert"
    )
    assert (
        getattr(MoELayer.routed_experts_compute, "__megatron_trace_event__", None) == "moe-experts"
    )
    assert getattr(MoELayer.combine, "__megatron_trace_event__", None) == "moe-combine"
    assert list(inspect.signature(TopKRouter.forward).parameters) == [
        "self",
        "input",
        "padding_mask",
        "input_ids",
    ]
    assert list(inspect.signature(MoELayer.dispatch).parameters) == [
        "self",
        "hidden_states",
        "probs",
    ]
    assert list(inspect.signature(MoELayer.shared_experts_compute).parameters) == [
        "self",
        "hidden_states",
    ]
    assert list(inspect.signature(MoELayer.routed_experts_compute).parameters) == [
        "self",
        "hidden_states",
        "probs",
    ]
    assert list(inspect.signature(MoELayer.combine).parameters) == ["self", "output"]

    helper_source = (ROOT / "megatron/core/transformer/moe/observability.py").read_text()
    for forbidden in (
        "megatron.training",
        "megatron.megalens",
        "all_reduce(",
        ".wait(",
        "synchronize(",
    ):
        assert forbidden not in helper_source


def test_shared_expert_routes_use_the_canonical_method_and_exclude_dualpipev() -> None:
    canonical_callers = {
        "megatron/core/transformer/moe/moe_layer.py": "self.shared_experts_compute(hidden_states)",
        "megatron/core/models/gpt/fine_grained_callables.py": (
            "layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)"
        ),
        "megatron/core/transformer/transformer_layer.py": (
            "self.mlp.shared_experts_compute(hidden_states)"
        ),
    }
    for relative_path, call in canonical_callers.items():
        source = (ROOT / relative_path).read_text()
        assert source.count(call) == 1

    dualpipev_root = ROOT / "megatron/plugin/dualpipev/fb_overlap/overlap_funcs"
    dualpipev_sources = [path.read_text() for path in dualpipev_root.glob("*.py")]
    assert all(".shared_experts_compute(" not in source for source in dualpipev_sources)
    assert sum(source.count(".shared_experts(") for source in dualpipev_sources) == 3
