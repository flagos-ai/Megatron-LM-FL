# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import importlib
from importlib.machinery import ModuleSpec
import json
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core import utils as core_utils
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common import model_chunk_schedule_plan as schedule_plan
from megatron.core.models.gpt import fine_grained_callables as fine_callables
from megatron.core import rerun_state_machine as rerun
from megatron.core import timers
from megatron.core.inference.contexts import dynamic_context
from megatron.core.inference import inference_request as inference_request_module
from megatron.core.inference import utils as inference_utils
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference import batch_dimensions_utils
from megatron.core.inference.engines import dynamic_engine
from megatron.core.dist_checkpointing import tensor_aware_state_dict as tensor_aware
from megatron.core.dist_checkpointing import exchange_utils
from megatron.core.dist_checkpointing import mapping as checkpoint_mapping
from megatron.core.dist_checkpointing import dict_utils as checkpoint_dict_utils
from megatron.core.dist_checkpointing import state_dict_utils as checkpoint_state_utils
from megatron.core.dist_checkpointing import utils as checkpoint_utils
from megatron.core.distributed.fsdp.src.megatron_fsdp import (
    param_and_grad_buffer as fsdp_param_buffer,
)
from megatron.core.pipeline_parallel import fine_grained_activation_offload as offload
from megatron.core.datasets import utils as dataset_utils
from megatron.core.quantization import utils as quantization_utils
from megatron.core.quantization.quant_config import GlobMatcher, MatchContext, RecipeConfig
from megatron.core import fp4_utils
from megatron.core.pipeline_parallel import utils as pipeline_utils
from megatron.core.tensor_parallel import utils as tensor_parallel_utils
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer import spec_utils
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer import utils as transformer_utils
from megatron.core import _rank_utils as rank_utils
from megatron.core.resharding import utils as reshard_utils


class _ReusableNullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def record(self, stream):
        self.calls.append(("event-record", stream))

    def wait(self, stream):
        self.calls.append(("event-wait", stream))


class _FakePlatform:
    def __init__(self, calls):
        self.calls = calls

    def current_stream(self):
        return "current-stream"

    def current_device(self):
        return 0

    def stream(self, stream):
        self.calls.append(("stream", stream))
        return nullcontext()

    def range_push(self, name):
        self.calls.append(("range-push", name))

    def range_pop(self):
        self.calls.append(("range-pop",))

    def Event(self):
        return _FakeEvent(self.calls)

    def Stream(self):
        return _FakeStream(self.calls, "platform-stream")

    def is_available(self):
        return False

    def empty_cache(self):
        self.calls.append(("empty-cache",))


class _FakeStream:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def wait_event(self, event):
        self.calls.append((self.name, "wait-event", event))

    def wait_stream(self, stream):
        self.calls.append((self.name, "wait-stream", stream))

    def record_event(self, event):
        self.calls.append((self.name, "record-event", event))


class _FakeBlueprint:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


class _FakeQuartResponse:
    def __init__(self, body="", status=200, mimetype=None):
        self.body = body
        self.status = status
        self.status_code = status
        self.mimetype = mimetype

    def get_data(self, as_text=False):
        if as_text and isinstance(self.body, bytes):
            return self.body.decode("utf-8")
        return self.body


class _AsyncJsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def get_json(self, *args, **kwargs):
        return self.payload


def _load_dynamic_endpoint(monkeypatch, module_name):
    quart = ModuleType("quart")
    quart.__spec__ = ModuleSpec("quart", loader=None)
    quart.Blueprint = _FakeBlueprint
    quart.Response = _FakeQuartResponse
    quart.current_app = SimpleNamespace(config={})
    quart.jsonify = lambda obj: ("json", obj)
    quart.request = _AsyncJsonRequest({})
    monkeypatch.setitem(sys.modules, "quart", quart)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class _ScheduleNode:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.model_chunk_state = SimpleNamespace()

    def forward(self, value=None):
        self.calls.append((self.name, "forward", value))
        return f"{self.name}.forward({value})"

    def backward(self, value=None):
        self.calls.append((self.name, "backward", value))
        return f"{self.name}.backward({value})"

    def backward_dw(self):
        self.calls.append((self.name, "backward_dw"))


class _LayerPlan:
    def __init__(self, prefix, calls, early_release=False):
        self.prefix = prefix
        self.calls = calls
        self.config = SimpleNamespace(ep_overlap_early_attn_memory_release=early_release)
        self.attn = _ScheduleNode(f"{prefix}.attn", calls)
        self.moe_dispatch = _ScheduleNode(f"{prefix}.dispatch", calls)
        self.mlp = _ScheduleNode(f"{prefix}.mlp", calls)
        self.moe_combine = _ScheduleNode(f"{prefix}.combine", calls)
        self.mtp_post_process = _ScheduleNode(f"{prefix}.mtp", calls)

    def get_fp8_context(self):
        self.calls.append((self.prefix, "fp8-context"))
        return nullcontext()

    def release_state(self):
        self.calls.append((self.prefix, "release"))


class _ChunkPlan:
    def __init__(self, prefix, calls, layers, post_process=True):
        self.prefix = prefix
        self.calls = calls
        self.vp_stage = prefix
        self.pre_process = _ScheduleNode(f"{prefix}.pre", calls)
        self.post_process = _ScheduleNode(f"{prefix}.post", calls) if post_process else None
        self._transformer_layers = list(layers)
        self._model_chunk_state = SimpleNamespace(model=SimpleNamespace(name=prefix))
        self._event = _FakeEvent(calls)

    def __bool__(self):
        return True

    def record_current_stream(self):
        self.calls.append((self.prefix, "record"))

    def wait_current_stream(self):
        self.calls.append((self.prefix, "wait"))

    def get_layer(self, index):
        return self._transformer_layers[index]

    def pop_layer(self):
        return self._transformer_layers.pop()

    def num_layers(self):
        return len(self._transformer_layers)

    def release_state(self):
        self.calls.append((self.prefix, "release_state"))
        self._model_chunk_state.model = None
        self.pre_process.model_chunk_state = None
        if self.post_process is not None:
            self.post_process.model_chunk_state = None


def test_transformer_layer_schedule_run_covers_overlap_orders():
    calls = []
    forward = _LayerPlan("forward", calls)
    backward = _LayerPlan("backward", calls)

    f_output, b_output = schedule_plan.TransformerLayerSchedulePlan.run(
        forward,
        backward,
        f_input="hidden",
        b_grad="grad",
        is_last_layer_in_bwd=False,
    )

    assert f_output == "forward.mtp.forward(forward.combine.forward(forward.mlp.forward(forward.dispatch.forward(forward.attn.forward(hidden)))))"
    assert b_output == "backward.attn.backward(backward.dispatch.backward(backward.mlp.backward(backward.combine.backward(backward.mtp.backward(grad)))))"
    assert calls[:4] == [
        ("backward.mtp", "backward", "grad"),
        ("backward.combine", "backward", "backward.mtp.backward(grad)"),
        ("forward", "fp8-context"),
        ("forward.attn", "forward", "hidden"),
    ]
    assert ("backward.mlp", "backward_dw") in calls
    assert ("backward.attn", "backward_dw") in calls

    early_calls = []
    early = _LayerPlan("early", early_calls, early_release=True)
    _, early_grad = schedule_plan.TransformerLayerSchedulePlan.run(
        None,
        early,
        b_grad="grad",
        is_last_layer_in_bwd=True,
    )

    assert early_grad.endswith("early.attn.backward(early.dispatch.backward(early.mlp.backward(early.combine.backward(early.mtp.backward(grad)))))")
    assert ("early.attn", "backward_dw") not in early_calls
    attn_backward_index = early_calls.index(
        (
            "early.attn",
            "backward",
            "early.dispatch.backward(early.mlp.backward(early.combine.backward(early.mtp.backward(grad))))",
        )
    )
    assert attn_backward_index < len(early_calls)


def test_transformer_layer_schedule_plan_builds_dense_moe_mtp_and_releases(monkeypatch):
    calls = []

    class _FakeMoE:
        def __init__(self):
            self.num_local_experts = 2

    class _FakeMTP:
        pass

    class _FakeTransformerLayerNode:
        def __init__(
            self,
            stream,
            event,
            layer_state,
            chunk_state,
            module,
            name,
            bwd_dw_callables=None,
            extra_args=None,
        ):
            self.stream = stream
            self.event = event
            self.layer_state = layer_state
            self.chunk_state = chunk_state
            self.module = module
            self.name = name
            self.bwd_dw_callables = bwd_dw_callables
            self.extra_args = extra_args
            calls.append(("node", name, module, bwd_dw_callables, dict(extra_args or {})))

    def fake_build_layer_callables(layer):
        return (
            ["attn-fn", "dispatch-fn", "mlp-fn", "combine-fn", "mtp-fn"],
            {"attn": "attn-dw", "mlp": "mlp-dw", "mtp_post_process": "mtp-dw"},
        )

    from megatron.core.transformer.moe import moe_layer as moe_layer_module
    from megatron.core.transformer import multi_token_prediction as mtp_module

    monkeypatch.setattr(fine_callables, "TransformerLayerNode", _FakeTransformerLayerNode)
    monkeypatch.setattr(fine_callables, "build_layer_callables", fake_build_layer_callables)
    monkeypatch.setattr(moe_layer_module, "MoELayer", _FakeMoE)
    monkeypatch.setattr(mtp_module, "MultiTokenPredictionLayer", _FakeMTP)

    dense_layer = SimpleNamespace(
        config=SimpleNamespace(delay_wgrad_compute=False, fp8=None, fp8_recipe=Fp8Recipe.delayed),
        mlp=object(),
        layer_number=1,
    )
    dense_plan = schedule_plan.TransformerLayerSchedulePlan(
        dense_layer,
        event="event",
        chunk_state="chunk",
        comp_stream="comp",
        comm_stream="comm",
        extra_args={"is_first_layer": True},
    )
    assert isinstance(dense_plan.moe_dispatch, schedule_plan.NoopScheduleNode)
    assert isinstance(dense_plan.moe_combine, schedule_plan.NoopScheduleNode)
    assert isinstance(dense_plan.mtp_post_process, schedule_plan.NoopScheduleNode)
    assert dense_plan.attn.name == "attn"
    assert dense_plan.mlp.bwd_dw_callables == "mlp-dw"

    moe_layer = SimpleNamespace(
        config=SimpleNamespace(delay_wgrad_compute=True, fp8=None, fp8_recipe=Fp8Recipe.delayed),
        mlp=_FakeMoE(),
        layer_number=2,
    )
    moe_plan = schedule_plan.TransformerLayerSchedulePlan(
        moe_layer,
        event="event",
        chunk_state="chunk",
        comp_stream="comp",
        comm_stream="comm",
        extra_args={},
    )
    assert moe_plan.moe_dispatch.name == "moe_dispatch"
    assert moe_plan.moe_combine.name == "moe_combine"
    assert moe_plan.moe_dispatch.extra_args["is_moe"] is True
    assert moe_plan.moe_dispatch.extra_args["num_local_experts"] == 2

    mtp_model_layer = SimpleNamespace(mlp=_FakeMoE())
    mtp_layer = _FakeMTP()
    mtp_layer.config = SimpleNamespace(delay_wgrad_compute=True, fp8=None, fp8_recipe=Fp8Recipe.delayed)
    mtp_layer.mtp_model_layer = mtp_model_layer
    mtp_layer.layer_number = 3
    mtp_layer.engram = "engram"
    mtp_layer.engram_hash_layer_id = 4
    mtp_plan = schedule_plan.TransformerLayerSchedulePlan(
        mtp_layer,
        event="event",
        chunk_state="chunk",
        comp_stream="comp",
        comm_stream="comm",
        extra_args={"is_engram": True},
    )
    assert mtp_plan.mtp_post_process.name == "mtp_post_process"
    assert mtp_plan.layer_state.engram == "engram"
    assert mtp_plan.layer_state.is_engram is True

    context_calls = []

    class _Context:
        def __enter__(self):
            context_calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            context_calls.append("exit")

    monkeypatch.setattr(schedule_plan, "get_fp8_context", lambda config, layer_number: _Context())
    fp8_layer = SimpleNamespace(
        config=SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise),
        layer_number=5,
    )
    fp8_plan = object.__new__(schedule_plan.TransformerLayerSchedulePlan)
    fp8_plan.layer = fp8_layer
    with fp8_plan.get_fp8_context():
        context_calls.append("body")
    assert context_calls == ["enter", "body", "exit"]

    mtp_plan.release_state()
    assert mtp_plan.attn is None
    assert mtp_plan.moe_dispatch is None
    assert mtp_plan.mlp is None
    assert mtp_plan.moe_combine is None
    assert mtp_plan.mtp_post_process is None
    assert mtp_plan.layer_state is None
    assert not hasattr(mtp_plan, "layer")


def test_transformer_model_chunk_schedule_run_and_manual_plan_methods(monkeypatch):
    calls = []
    monkeypatch.setattr(schedule_plan, "cur_platform", _FakePlatform(calls))
    monkeypatch.setattr(schedule_plan, "get_comm_stream", lambda: "comm-stream")

    forward_layers = [_LayerPlan("f0", calls), _LayerPlan("f1", calls), _LayerPlan("f2", calls)]
    backward_layers = [_LayerPlan("b0", calls), _LayerPlan("b1", calls)]
    forward_plan = _ChunkPlan("forward-chunk", calls, forward_layers)
    backward_plan = _ChunkPlan("backward-chunk", calls, backward_layers)

    result = schedule_plan.TransformerModelChunkSchedulePlan.run(
        forward_plan,
        backward_plan,
        b_grad="loss-grad",
        pre_forward=lambda vp: calls.append(("pre-forward", vp)),
        pre_backward=lambda vp: calls.append(("pre-backward", vp)),
        post_forward=lambda value, vp: calls.append(("post-forward", value, vp)),
        post_backward=lambda value, vp: calls.append(("post-backward", value, vp)),
    )

    assert result.startswith("forward-chunk.post.forward(")
    assert ("pre-forward", "forward-chunk") in calls
    assert ("pre-backward", "backward-chunk") in calls
    assert any(call[0] == "post-forward" for call in calls)
    assert any(call[0] == "post-backward" for call in calls)
    assert ("b1", "release") in calls
    assert ("b0", "release") in calls
    assert ("backward-chunk", "release_state") in calls
    assert calls.count(("range-pop",)) == 3

    plan = object.__new__(schedule_plan.TransformerModelChunkSchedulePlan)
    plan._model_chunk_state = SimpleNamespace(model=SimpleNamespace())
    plan._transformer_layers = ["layer-0", "layer-1"]
    plan._event = _FakeEvent(calls)
    plan.pre_process = SimpleNamespace(model_chunk_state=plan._model_chunk_state)
    plan.post_process = SimpleNamespace(model_chunk_state=plan._model_chunk_state)

    assert plan.event is plan._event
    assert plan.state is plan._model_chunk_state
    assert plan.num_layers() == 2
    assert plan.get_layer(1) == "layer-1"
    assert plan.pop_layer() == "layer-1"
    plan.record_current_stream()
    plan.wait_current_stream()
    plan.release_state()

    assert plan._model_chunk_state.model is None
    assert plan.pre_process is None
    assert plan.post_process is None


def test_fine_grained_callable_helpers_and_nodes(monkeypatch):
    calls = []
    config = SimpleNamespace(
        fp8=None,
        fp4=None,
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend=None,
        cuda_graph_scope=[],
    )

    assert fine_callables.should_free_input("mlp", False, config, None) is False
    assert fine_callables.should_free_input("mlp", True, config, 1) is False
    assert fine_callables.should_free_input("moe_dispatch", True, config, 1) is True
    assert fine_callables.should_free_input("moe_combine", True, config, 1) is True
    assert fine_callables.should_free_input("unknown", True, config, 1) is False

    config.fp8 = "enabled"
    assert fine_callables.should_free_input("mlp", True, config, 1) is True
    config.fp8 = None
    config.moe_token_dispatcher_type = "flex"
    config.moe_flex_dispatcher_backend = "deepep"
    assert fine_callables.should_free_input("moe_dispatch", True, config, 2) is False
    config.moe_flex_dispatcher_backend = "hybridep"
    assert fine_callables.should_free_input("mlp", True, config, 2) is False

    class _Owner:
        def method(self, value):
            return f"wrapped-{value}"

    owner = _Owner()
    assert fine_callables.weak_method(owner.method)("value") == "wrapped-value"

    chunk_state = SimpleNamespace(
        input_ids="ids",
        position_ids="positions",
        decoder_input="decoder",
        packed_seq_params="packed",
        padding_mask="padding",
        labels="labels",
        loss_mask="loss-mask",
        attention_mask="attention",
        runtime_gather_output=True,
        extra_block_kwargs={"extra": "kwarg"},
    )

    gpt_model = SimpleNamespace(
        pre_process=False,
        decoder=SimpleNamespace(
            input_tensor="decoder-from-model",
            layers=[object()],
            final_layernorm=None,
        ),
        config=SimpleNamespace(mtp_num_layers=0),
    )

    def preprocess(**kwargs):
        calls.append(("preprocess", kwargs))
        return ("decoder-out", "rotary", "cos", "sin", "offset", "padding-out")

    def postprocess(**kwargs):
        calls.append(("postprocess", kwargs))
        return "loss"

    gpt_model._preprocess = preprocess
    gpt_model._postprocess = postprocess
    monkeypatch.setattr(fine_callables, "float16_to_fp32", lambda value: f"fp32-{value}")

    pre_node = fine_callables.PreProcessNode(gpt_model, chunk_state, _FakeEvent(calls), lambda: "stream")
    assert pre_node.forward_impl() == "decoder-out"
    assert chunk_state.decoder_input == "decoder-out"
    assert chunk_state.rotary_pos_emb == "rotary"
    assert chunk_state.sequence_len_offset == "offset"

    post_node = fine_callables.PostProcessNode(gpt_model, chunk_state, _FakeEvent(calls), lambda: "stream")
    assert post_node.forward_impl("hidden") == "fp32-loss"
    post_kwargs = calls[-1][1]
    assert post_kwargs["hidden_states"] == "hidden"
    assert post_kwargs["decoder_input"] == "decoder-out"
    assert post_kwargs["runtime_gather_output"] is True


def test_transformer_layer_node_backward_and_dw_without_real_autograd(monkeypatch):
    calls = []
    monkeypatch.setattr(fine_callables, "cur_platform", _FakePlatform(calls))
    monkeypatch.setattr(fine_callables, "make_viewless", lambda value: value)

    config = SimpleNamespace(
        fp8=None,
        fp4=None,
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend=None,
        cuda_graph_scope=[],
        delay_wgrad_compute=True,
    )
    layer_state = SimpleNamespace()
    chunk_state = SimpleNamespace()

    class _BackwardDW:
        def backward_dw(self):
            calls.append(("module", "backward_dw"))

    node = fine_callables.TransformerLayerNode(
        stream=lambda: "node-stream",
        event=_FakeEvent(calls),
        layer_state=layer_state,
        chunk_state=chunk_state,
        submodule=lambda this_node, value: f"forward-{value}",
        name="mlp",
        bwd_dw_callables=_BackwardDW(),
        extra_args={"config": config, "delay_wgrad_compute": True},
    )

    assert node.forward_impl("input") == "forward-input"
    node.detached = (SimpleNamespace(grad="detached-grad"),)
    node.before_detached = ("before-detached",)

    def fake_backward(outputs, grads):
        calls.append(("default-backward", outputs, grads))
        return grads

    node.default_backward_func = fake_backward
    grads = node.backward_impl(("output",), ("output-grad",))
    assert grads == ("output-grad", "detached-grad")
    assert node.output_grads == ("output-grad", "detached-grad")
    assert node.delay_grads_release is True

    node.backward_dw()
    assert ("module", "backward_dw") in calls
    assert node.output_grads is None
    assert node.bwd_dw_callables is None

    no_delay = fine_callables.TransformerLayerNode(
        stream=lambda: "node-stream",
        event=_FakeEvent(calls),
        layer_state=layer_state,
        chunk_state=chunk_state,
        submodule=lambda this_node, value: value,
        name="attn",
        extra_args={"config": config, "delay_wgrad_compute": False},
    )
    assert no_delay.backward_dw() is None


def test_timers_start_stop_elapsed_and_log_level_filtering(monkeypatch):
    from itertools import chain, count

    calls = []
    time_values = chain([0.0, 10.0, 11.5, 20.0, 22.0, 30.0, 31.0], count(40.0))
    monkeypatch.setattr(timers.time, "time", lambda: next(time_values))
    monkeypatch.setattr(timers, "cur_platform", SimpleNamespace(synchronize=lambda: calls.append("sync")))

    timer = timers.Timer("unit")
    timer.start()
    timer.stop()
    assert timer.elapsed(reset=False) == 1.5
    assert timer.active_time() == 1.5
    timer.set_elapsed(4.0)
    assert timer.elapsed(reset=True) == 4.0
    assert timer.elapsed(reset=False) == 0.0

    timer.start()
    assert timer.elapsed(reset=True) == 2.0
    assert timer._started is True
    timer.stop()
    assert timer.active_time() == 4.5
    assert calls.count("sync") >= 5

    group = timers.Timers(log_level=1, log_option="minmax")
    enabled = group("enabled", log_level=1)
    assert isinstance(enabled, timers.Timer)
    assert group("enabled", log_level=1) is enabled
    with pytest.raises(AssertionError, match="does not match"):
        group("enabled", log_level=2)

    dummy = group("too-detailed", log_level=2)
    assert isinstance(dummy, timers.DummyTimer)
    with pytest.raises(Exception, match="dummy timer"):
        dummy.elapsed()
    with pytest.raises(Exception, match="active timer"):
        dummy.active_time()

    with pytest.raises(AssertionError, match="invalid"):
        timers.Timers(log_level=1, log_option="average")
    with pytest.raises(AssertionError, match="larger"):
        group("unsupported", log_level=3)


def test_resharding_metadata_name_resolution_and_balanced_selection():
    assert reshard_utils._get_rank_in_group(3, [1, 3, 5]) == 1
    with pytest.raises(ValueError, match="not found"):
        reshard_utils._get_rank_in_group(4, [1, 3, 5])

    assert reshard_utils._detect_expert_index_from_param_name("experts.weight12") == 12
    assert reshard_utils._detect_expert_index_from_param_name("experts.bias7") == 7
    assert reshard_utils._detect_expert_index_from_param_name("experts.weight") is None

    non_ep = reshard_utils.ParameterMetadata(
        name="decoder.layers.0.mlp.weight",
        shape=(4, 4),
        dtype="bf16",
        element_size=2,
        owner_rank=0,
    )
    reshard_utils.assign_ep_resolved_name_inplace(non_ep)
    assert non_ep.resolved_name == "decoder.layers.0.mlp.weight"
    assert non_ep.global_expert_index is None

    ep_meta = reshard_utils.ParameterMetadata(
        name="decoder.layers.0.experts.weight1",
        shape=(4, 4),
        dtype="bf16",
        element_size=2,
        is_ep=True,
        num_experts=8,
        owner_rank=3,
        expert_parallel_group_ranks=[2, 3],
    )
    reshard_utils.assign_ep_resolved_name_inplace(ep_meta)
    assert ep_meta.global_expert_index == 5
    assert ep_meta.resolved_name == "decoder.layers.0.experts.weight5"

    prefix_map = {"decoder.layers.0": "decoder.layers.8"}
    reshard_utils.assign_resolved_name_inplace(
        ep_meta,
        layer_module_prefix_map=prefix_map,
        base_name="decoder.layers.0.experts.bias0",
    )
    assert ep_meta.global_expert_index == 4
    assert ep_meta.resolved_name == "decoder.layers.8.experts.bias4"
    assert (
        reshard_utils._resolve_global_layer_number_in_name(
            "decoder.layers.0.self_attention.weight", prefix_map
        )
        == "decoder.layers.8.self_attention.weight"
    )
    assert reshard_utils._resolve_global_layer_number_in_name("embedding.weight", prefix_map) == "embedding.weight"

    class _Module:
        def named_modules(self):
            return [
                ("", self),
                ("decoder.layers.0", SimpleNamespace(layer_number=9)),
                ("decoder.layers.0.self_attention", SimpleNamespace(layer_number=None)),
                ("decoder.layers.final", SimpleNamespace(layer_number=10)),
            ]

    assert reshard_utils._build_layer_module_prefix_map(_Module()) == {
        "decoder.layers.0": "decoder.layers.8"
    }

    src = [
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=0,
            data_parallel_group_ranks=[0, 1],
        ),
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=2,
            data_parallel_group_ranks=[2, 3],
        ),
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=3,
            data_parallel_group_ranks=[2, 3],
        ),
    ]
    dst = reshard_utils.ParameterMetadata(
        name="w",
        shape=(1,),
        dtype="bf16",
        element_size=2,
        owner_rank=99,
        data_parallel_group_ranks=[99],
    )
    assert reshard_utils.select_src_metadata_balanced(src, dst, dst_rank=3).owner_rank == 3
    assert reshard_utils.select_src_metadata_balanced(src, dst, dst_rank=0).owner_rank == 0

    ep_src = [
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=0,
            expert_parallel_group_ranks=[0, 1],
            data_parallel_group_ranks=[0],
        ),
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=1,
            expert_parallel_group_ranks=[0, 1],
            data_parallel_group_ranks=[1],
        ),
    ]
    ep_dst = reshard_utils.ParameterMetadata(
        name="w",
        shape=(1,),
        dtype="bf16",
        element_size=2,
        owner_rank=9,
        expert_parallel_group_ranks=[8, 9],
        data_parallel_group_ranks=[9],
    )
    assert reshard_utils.select_src_metadata_balanced(ep_src, ep_dst, dst_rank=9).owner_rank == 1

    bad_dst = reshard_utils.ParameterMetadata(
        name="w",
        shape=(1,),
        dtype="bf16",
        element_size=2,
        owner_rank=10,
        expert_parallel_group_ranks=[8, 9, 10],
        data_parallel_group_ranks=[10],
    )
    assert reshard_utils.select_src_metadata_balanced(ep_src, bad_dst, dst_rank=10).owner_rank in {0, 1}
    with pytest.raises(ValueError, match="non-empty"):
        reshard_utils.select_src_metadata_balanced([], dst, dst_rank=0)


class _ModuleTree:
    def __init__(self, **attrs):
        self._modules = {}
        for key, value in attrs.items():
            setattr(self, key, value)

    def add(self, name, child):
        self._modules[name] = child
        return child

    def modules(self):
        yield self
        for child in self._modules.values():
            yield from child.modules()


def test_transformer_utils_cpu_helpers_and_attribute_caches(monkeypatch):
    scores = torch.zeros(2, 2)
    mask = torch.tensor([[False, True], [False, False]])
    masked = transformer_utils.attention_mask_func(scores, mask)
    assert masked[0, 1].item() == -10000.0
    assert masked[1, 1].item() == 0.0

    x = torch.tensor([-1.0, 0.0, 1.0])
    monkeypatch.setattr(
        transformer_utils,
        "gelu_impl",
        lambda value: 0.5
        * value
        * (1.0 + torch.tanh(0.7978845608028654 * value * (1.0 + 0.044715 * value * value))),
    )
    monkeypatch.setattr(
        transformer_utils,
        "erf_gelu",
        lambda value: value
        * 0.5
        * (torch.erf(value / 1.41421).to(dtype=value.dtype) + torch.ones_like(value).to(dtype=value.dtype)),
    )
    assert torch.allclose(transformer_utils.openai_gelu(x), transformer_utils.gelu_impl(x))
    assert transformer_utils.erf_gelu(x).shape == x.shape

    linear = transformer_utils.get_linear_layer(
        2,
        3,
        init_method=lambda weight: weight.data.fill_(2.0),
        perform_initialization=True,
    )
    assert torch.all(linear.weight == 2.0)
    assert torch.all(linear.bias == 0.0)

    assert transformer_utils._get_extra_state_offsets(()) == ((1,), (0,))
    assert transformer_utils._get_extra_state_offsets(((1, 4, 8), (0, 2, 6))) == ((6, 8), (2, 4))
    with pytest.raises(AssertionError, match="contiguous"):
        transformer_utils._get_extra_state_offsets(((2, 4, 8),))

    sharded_calls = []
    monkeypatch.setattr(transformer_utils, "get_pg_rank", lambda group: {"tp": 1, "dp": 2}[group])

    class _FakeShardedObject:
        def __init__(self, key, obj, shape, offset, replica_id, **kwargs):
            self.key = key
            self.obj = obj
            self.shape = shape
            self.offset = offset
            self.replica_id = replica_id
            self.kwargs = kwargs

    monkeypatch.setattr(transformer_utils, "ShardedObject", _FakeShardedObject)
    monkeypatch.setattr(
        transformer_utils,
        "make_tp_sharded_tensor_for_checkpoint",
        lambda tensor, key, axis, prepend_offsets=(), tp_group=None, dp_cp_group=None: sharded_calls.append(
            ("tp", key, axis, tuple(prepend_offsets), tp_group, dp_cp_group)
        )
        or ("tp", key),
    )
    monkeypatch.setattr(
        transformer_utils,
        "make_sharded_tensor_for_checkpoint",
        lambda tensor, key, prepend_offsets=(), tp_group=None, dp_cp_group=None: sharded_calls.append(
            ("dp", key, tuple(prepend_offsets), tp_group, dp_cp_group)
        )
        or ("dp", key),
    )
    sharded = transformer_utils.make_sharded_tensors_for_checkpoint(
        {
            "weight": torch.ones(2, 2),
            "bias": torch.zeros(2),
            "layer_extra_state": {"rng": 1},
        },
        prefix="module.",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=((0, 3, 5),),
        tp_group="tp",
        dp_cp_group="dp",
    )
    assert sharded["module.weight"] == ("tp", "module.weight")
    assert sharded["module.bias"] == ("dp", "module.bias")
    assert isinstance(sharded["module.layer_extra_state"], _FakeShardedObject)
    assert sharded["module.layer_extra_state"].replica_id == (0, 1, 2)
    assert ("tp", "module.weight", 0, ((0, 3, 5),), "tp", "dp") in sharded_calls

    class _ModuleWithCustomShard:
        def sharded_state_dict(self, prefix, sharded_offsets, metadata):
            return {"custom": (prefix, sharded_offsets, metadata["dp_cp_group"])}

    assert transformer_utils.sharded_state_dict_default(
        _ModuleWithCustomShard(),
        prefix="x.",
        sharded_offsets=((0, 1, 2),),
        metadata={"dp_cp_group": "dp"},
    ) == {"custom": ("x.", ((0, 1, 2),), "dp")}

    class _PlainModule:
        def state_dict(self, prefix="", keep_vars=True):
            assert prefix == ""
            assert keep_vars is True
            return {"plain": torch.ones(1)}

    assert transformer_utils.sharded_state_dict_default(
        _PlainModule(),
        prefix="plain.",
        metadata={"dp_cp_group": "dp"},
        tp_group="tp",
    ) == {"plain.plain": ("dp", "plain.plain")}

    monkeypatch.setattr(
        transformer_utils.parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel=True: "dp-cp-group",
    )
    assert transformer_utils.ensure_metadata_has_dp_cp_group(None) == {"dp_cp_group": "dp-cp-group"}
    metadata = {}
    assert transformer_utils.ensure_metadata_has_dp_cp_group(metadata) is metadata
    assert metadata["dp_cp_group"] == "dp-cp-group"
    with pytest.raises(AssertionError, match="metadata must be"):
        transformer_utils.ensure_metadata_has_dp_cp_group("bad")

    transformer_utils._sequence_parallel_attr_cache = None
    model = _ModuleTree(
        config=SimpleNamespace(sequence_parallel=False),
        position_embedding_type="rope",
    )
    child = model.add(
        "child",
        _ModuleTree(
            sequence_parallel=False,
            scatter_to_sequence_parallel=False,
            reduce_scatter_embeddings=False,
        ),
    )
    excluded = model.add("excluded", _ModuleTree(sequence_parallel=False))
    transformer_utils.set_model_to_sequence_parallel(model, set_to=True, exclude_modules=[excluded])
    assert model.config.sequence_parallel is True
    assert child.sequence_parallel is True
    assert child.scatter_to_sequence_parallel is True
    assert child.reduce_scatter_embeddings is True
    assert excluded.sequence_parallel is False

    transformer_utils.set_model_to_sequence_parallel(model, set_to=False, exclude_modules=[excluded])
    assert child.sequence_parallel is False

    transformer_utils.cuda_graph_attr_cache = None
    graph_model = _ModuleTree(
        config=SimpleNamespace(cuda_graph_impl="none", recompute_granularity="full"),
    )
    graph_child = graph_model.add(
        "graph_child",
        _ModuleTree(
            cuda_graph_impl=False,
            flash_decode=False,
            cudagraph_manager="cached-manager",
            recompute_granularity="selective",
            config=SimpleNamespace(cuda_graph_impl="none", flash_decode=False),
        ),
    )
    transformer_utils.toggle_cuda_graphs(graph_model, set_to="local")
    assert graph_model.config.cuda_graph_impl == "local"
    assert graph_child.cuda_graph_impl == "local"
    assert graph_child.config.cuda_graph_impl == "local"
    assert graph_child.recompute_granularity is None
    assert graph_child.cudagraph_manager == "cached-manager"

    transformer_utils.toggle_cuda_graphs(graph_model, set_to="none")
    assert graph_child.cuda_graph_impl == "none"
    assert graph_child.config.cuda_graph_impl == "none"
    assert graph_child.recompute_granularity == "selective"
    assert not hasattr(graph_child, "cudagraph_manager")
    with pytest.raises(AssertionError, match="Invalid CUDA graph"):
        transformer_utils.toggle_cuda_graphs(graph_model, set_to="global")

    assert transformer_utils.is_layer_window_attention(None, None, 1) is False
    assert transformer_utils.is_layer_window_attention((4, 4), None, 1) is True
    assert transformer_utils.is_layer_window_attention((4, 4), 2, 2) is False
    assert transformer_utils.is_layer_window_attention((4, 4), 2, 3) is True
    assert transformer_utils.is_layer_window_attention((4, 4), [False, True], 2) is True
    with pytest.raises(ValueError, match="Invalid"):
        transformer_utils.is_layer_window_attention((4, 4), "bad", 1)


def test_cuda_graphs_cpu_metadata_tree_pool_and_global_record_paths(monkeypatch):
    @dataclass
    class _Payload:
        tensor: object
        nested: object

    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_WARMUP", False)
    cuda_graphs._set_capture_start()
    assert cuda_graphs.is_graph_capturing() is True
    cuda_graphs._set_capture_end()
    assert cuda_graphs.is_graph_capturing() is False
    cuda_graphs._set_warmup_start()
    assert cuda_graphs.is_graph_warmup() is True
    cuda_graphs._set_warmup_end()
    assert cuda_graphs.is_graph_warmup() is False

    tensor = torch.randn(2, 3, requires_grad=True)
    tensor.cg_buffer_metadata = cuda_graphs.CudagraphBufferMetadata(
        is_cudagraph_input=True,
        input_use_count=2,
    )
    meta = cuda_graphs.ArgMetadata(tensor)
    assert meta.type is torch.Tensor
    assert meta.shape == tensor.shape
    assert meta.dtype == tensor.dtype
    assert meta.device == tensor.device
    assert meta.value == tensor.data_ptr()
    assert meta.requires_grad is True
    assert meta.cg_buffer_metadata is tensor.cg_buffer_metadata
    zero = meta.zeros_like()
    assert zero.shape == tensor.shape
    assert zero.requires_grad is True

    scalar_meta = cuda_graphs.ArgMetadata("token")
    assert scalar_meta.type is str
    assert scalar_meta.value == "token"

    monkeypatch.setattr(cuda_graphs.TensorReusePool, "tensor_strong_refs", [])
    monkeypatch.setattr(cuda_graphs.TensorReusePool, "tensor_strong_refs_dataptrs", set())
    monkeypatch.setattr(cuda_graphs.TensorReusePool, "pool", [])
    pool = cuda_graphs.TensorReusePool()
    first = pool.get(meta)
    assert first.shape == tensor.shape
    assert pool.owns(first) is True
    pool.insert(first)
    assert pool.get(meta) is first

    payload = _Payload(torch.tensor([1]), {"items": [torch.tensor([2]), "kept"]})
    mapped = cuda_graphs.tree_map(
        lambda value: value + 10 if torch.is_tensor(value) else value,
        payload,
    )
    assert torch.equal(mapped.tensor, torch.tensor([11]))
    assert torch.equal(mapped.nested["items"][0], torch.tensor([12]))
    assert mapped.nested["items"][1] == "kept"

    cuda_graphs._check_supported_type(cuda_graphs.ArgMetadata(None))
    cuda_graphs._check_supported_type(cuda_graphs.ArgMetadata(payload))
    with pytest.raises(AssertionError, match="not supported"):
        cuda_graphs._check_supported_type(cuda_graphs.ArgMetadata(object()))

    cloned = cuda_graphs._clone_nested_tensors(
        {"a": tensor, "b": [torch.tensor([3]), (torch.tensor([4]),)]}
    )
    assert torch.equal(cloned["a"], tensor)
    assert cloned["a"].data_ptr() != tensor.data_ptr()
    assert torch.equal(cloned["b"][0], torch.tensor([3]))
    with pytest.raises(TypeError, match="Sets of tensors"):
        cuda_graphs._clone_nested_tensors({torch.tensor(1)})

    class _FakeGenerator:
        def __init__(self):
            self.state = torch.tensor([1, 2, 3], dtype=torch.uint8)

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = state

    generator = _FakeGenerator()
    original_state = generator.state
    assert cuda_graphs._ensure_generator_state_is_cudagraph_safe(generator) is generator
    assert torch.equal(generator.state, original_state)
    assert generator.state.data_ptr() != original_state.data_ptr()

    record = cuda_graphs._CudagraphGlobalRecord
    monkeypatch.setattr(record, "cudagraph_created", False)
    monkeypatch.setattr(record, "cudagraph_record", [])
    monkeypatch.setattr(record, "cudagraph_inference_record", [])
    assert record.create_cudagraphs() is None
    runner = SimpleNamespace(base_module=SimpleNamespace(modules=lambda: []))
    record.record_fwd_graph(runner, (tensor,), {"hidden_states": tensor}, tensor)
    record.record_bwd_graph(runner)
    assert record.cudagraph_record[0][1] == "fwd"
    assert record.cudagraph_record[1][1] == "bwd"
    monkeypatch.setattr(record, "cudagraph_created", True)
    with pytest.raises(AssertionError, match="already created"):
        record.create_cudagraphs()


def test_cuda_graphs_record_and_replay_nodes_validate_status_and_surfaces(monkeypatch):
    record = cuda_graphs._CudagraphGlobalRecord
    monkeypatch.setattr(record, "cudagraph_record", [])
    runner = SimpleNamespace(
        status=cuda_graphs._GraphStatus.FWD_READY,
        bwd_graph_recorded=False,
    )
    ctx = SimpleNamespace()
    inputs = torch.tensor([1.0], requires_grad=True)
    assert cuda_graphs._CudagraphRecordNode.forward(ctx, runner, inputs) is inputs
    assert ctx.runner is runner
    runner.status = cuda_graphs._GraphStatus.BWD_READY
    grads = torch.tensor([2.0])
    assert cuda_graphs._CudagraphRecordNode.backward(ctx, grads) == (None, grads)
    assert runner.status == cuda_graphs._GraphStatus.FWD_READY
    assert runner.bwd_graph_recorded is True
    assert record.cudagraph_record == [(runner, "bwd")]

    with pytest.raises(AssertionError, match="bwd cudagraph was expected"):
        cuda_graphs._CudagraphRecordNode.forward(
            SimpleNamespace(),
            SimpleNamespace(status=cuda_graphs._GraphStatus.BWD_READY),
            inputs,
        )

    replay_runner = SimpleNamespace(
        fwd_graph=None,
        status=cuda_graphs._GraphStatus.FWD_READY,
        fwd_graph_input_surface=[],
    )
    with pytest.raises(AssertionError, match="before calling"):
        cuda_graphs._CudagraphReplayNode.forward(SimpleNamespace(), replay_runner, True, inputs)


def test_cuda_graph_runner_record_metadata_and_mismatch_cpu_paths(monkeypatch):
    @dataclass
    class _RunnerPayload:
        tensor: object
        label: str

    record = cuda_graphs._CudagraphGlobalRecord
    monkeypatch.setattr(record, "cudagraph_record", [])
    monkeypatch.setattr(record, "cudagraph_created", False)
    monkeypatch.setattr(cuda_graphs, "HAVE_TE_GRAPHS", False)

    runner = object.__new__(cuda_graphs._CudaGraphRunner)
    runner.func = lambda hidden_states, payload=None, scale=1: hidden_states * scale + payload.tensor
    runner.fwd_graph_recorded = False
    runner.status = cuda_graphs._GraphStatus.FWD_READY
    runner.fp8_runtime_enabled = False
    runner.fp4_runtime_enabled = False

    hidden_states = torch.ones(2, requires_grad=True)
    payload = _RunnerPayload(torch.tensor([3.0, 4.0]), "same")
    output = runner.record_graph_capture(
        (hidden_states,),
        {"payload": payload, "scale": 2},
    )

    assert torch.equal(output, torch.tensor([5.0, 6.0]))
    assert runner.fwd_graph_recorded is True
    assert hidden_states.cg_buffer_metadata.is_cudagraph_input is True
    assert hidden_states.cg_buffer_metadata.input_use_count == 1
    assert output.cg_buffer_metadata.is_cudagraph_output is True
    assert record.cudagraph_record[-1][1] == "fwd"

    arg_metas = runner.get_arg_metas(record.cudagraph_record[-1][2], record.cudagraph_record[-1][3])
    assert any(meta.type is torch.Tensor for meta in arg_metas)
    flattened_tensors = runner.get_tensors(
        (hidden_states, payload), {"extra": torch.tensor([7.0])}
    )
    assert flattened_tensors[0] is hidden_states
    assert torch.equal(flattened_tensors[1], payload.tensor)
    assert torch.equal(flattened_tensors[2], torch.tensor([7.0]))

    runner.fwd_graph_input_arg_metas = [cuda_graphs.ArgMetadata(hidden_states)]
    runner.fwd_graph_input_kwarg_metas = {
        "payload": cuda_graphs.ArgMetadata(payload),
        "scale": cuda_graphs.ArgMetadata(2),
    }
    assert runner.get_mismatch_errors((hidden_states,), {"payload": payload, "scale": 2}) == []
    assert "Argument count mismatch" in "\n".join(
        runner.get_mismatch_errors((), {"payload": payload, "scale": 2})
    )
    assert "Tensor mismatch" in "\n".join(
        runner.get_mismatch_errors(
            (torch.ones(3),),
            {"payload": payload, "scale": 2},
        )
    )
    assert "Missing kwargs" in "\n".join(
        runner.get_mismatch_errors((hidden_states,), {"payload": payload})
    )
    assert "Unexpected kwargs" in "\n".join(
        runner.get_mismatch_errors(
            (hidden_states,),
            {"payload": payload, "scale": 2, "extra": 1},
        )
    )
    assert "Value mismatch" in "\n".join(
        runner.get_mismatch_errors(
            (hidden_states,),
            {"payload": _RunnerPayload(payload.tensor, "different"), "scale": 2},
        )
    )

    runner.grad_enabled = False
    with pytest.raises(AssertionError, match="argument mismatch"):
        runner.replay_graph_capture(False, (torch.ones(3),), {"payload": payload, "scale": 2})

    assert runner.to_list(torch.tensor([1])).pop().item() == 1
    assert runner.to_list((1, 2)) == [1, 2]


def test_fsdp_param_and_grad_buffer_index_and_allocator_cpu_paths(monkeypatch):
    assert fsdp_param_buffer._pad(5, 4) == 8
    policy = fsdp_param_buffer.BucketingPolicy(
        suggested_bucket_size=8,
        fsdp_unit_modules=[torch.nn.Linear],
        data_parallel_sharding_strategy="optim_grads_params",
    )
    assert policy.suggested_bucket_size == 8
    assert policy.fsdp_unit_modules == [torch.nn.Linear]

    no_shard = fsdp_param_buffer.DistributedDataParallelConfig(
        data_parallel_sharding_strategy="no_shard"
    )
    elements = [
        torch.Size([5]),
        torch.Size([3]),
        torch.Size([1]),
        torch.Size([2]),
        torch.Size([4]),
    ]
    item_map, bucket_index, shard_index = fsdp_param_buffer.build_data_parallel_buffer_index(
        elements,
        data_parallel_rank=1,
        data_parallel_world_size=3,
        is_data_distributed=False,
        ddp_config=no_shard,
        bucket_id=2,
        chunk_size_factor=4,
    )
    assert bucket_index.bucket_id == 2
    assert bucket_index.size == 15
    assert item_map[0].global_data_index == 0
    assert item_map[1].global_data_index == 5
    assert item_map[4].global_data_index == 8
    assert item_map[2].global_data_index == 12
    assert item_map[3].global_data_index == 13
    assert shard_index == fsdp_param_buffer.ShardBucketIndex(
        bucket_id=2,
        global_data_index=5,
        local_data_index=5,
        bucket_data_index=5,
        size=5,
    )

    sharded = fsdp_param_buffer.DistributedDataParallelConfig(
        data_parallel_sharding_strategy="optim_grads_params"
    )
    _, padded_bucket, distributed_shard = fsdp_param_buffer.build_data_parallel_buffer_index(
        elements,
        data_parallel_rank=2,
        data_parallel_world_size=4,
        is_data_distributed=True,
        ddp_config=sharded,
        bucket_id=4,
        chunk_size_factor=4,
    )
    assert padded_bucket.size == 16
    assert distributed_shard == fsdp_param_buffer.ShardBucketIndex(
        bucket_id=4,
        global_data_index=8,
        local_data_index=0,
        bucket_data_index=8,
        size=4,
    )

    direct_shard = fsdp_param_buffer._get_dp_buffer_shard_bucket_index(
        fsdp_param_buffer.BucketIndex(7, 12, 24, []),
        is_data_distributed=False,
        data_parallel_world_size=4,
        data_parallel_rank=3,
    )
    assert direct_shard.global_data_index == 30
    assert direct_shard.local_data_index == 30
    assert direct_shard.size == 6

    temp_allocator = fsdp_param_buffer.TemporaryBucketAllocator()
    bucket = temp_allocator.allocate(0, 3, torch.float32, torch.device("cpu"))
    assert bucket.data.shape == torch.Size([3])
    assert 0 in temp_allocator.buckets
    temp_allocator.free(0)
    assert 0 not in temp_allocator.buckets

    resize_allocator = fsdp_param_buffer.StorageResizeBasedBucketAllocator()
    resize_bucket = resize_allocator.allocate(1, 4, torch.float32, torch.device("cpu"))
    assert resize_bucket.data.numel() == 4
    resize_allocator.free(1)
    assert 1 in resize_allocator.buckets
    assert resize_allocator.buckets[1].data._typed_storage()._size() == 0
    resized_again = resize_allocator.allocate(1, 6, torch.float32, torch.device("cpu"))
    assert resized_again.data._typed_storage()._size() == 6

    buffer_calls = []

    class _FakeGlobalBuffer:
        def get_tensor(self, shape, dtype, name, mem_alloc_context=None):
            buffer_calls.append((tuple(shape), dtype, name, mem_alloc_context))
            return torch.zeros(*shape, dtype=dtype)

    monkeypatch.setattr(fsdp_param_buffer, "get_global_memory_buffer", lambda: _FakeGlobalBuffer())
    rotary = fsdp_param_buffer.RotaryBucketAllocator("rotary")
    first = rotary.allocate(3, 5, torch.float32, torch.device("cpu"))
    second = rotary.allocate(3, 5, torch.float32, torch.device("cpu"))
    assert torch.equal(first.data, second.data)
    assert buffer_calls[-1][2] == "rotary_0"
    rotary.free(3)
    rotary.allocate(4, 5, torch.float32, torch.device("cpu"))
    assert rotary.using_buffer[4] == 0

    monkeypatch.setattr(fsdp_param_buffer.torch.distributed, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(fsdp_param_buffer.cur_platform, "synchronize", lambda: buffer_calls.append("sync"))
    fsdp_groups = [
        SimpleNamespace(fsdp_unit_id=1, params=[torch.nn.Parameter(torch.ones(2))], dtype=torch.float32),
        SimpleNamespace(fsdp_unit_id=1, params=[torch.nn.Parameter(torch.ones(3))], dtype=torch.float32),
        SimpleNamespace(fsdp_unit_id=2, params=[torch.nn.Parameter(torch.ones(2))], dtype=torch.float32),
        SimpleNamespace(fsdp_unit_id=2, params=[torch.nn.Parameter(torch.ones(3))], dtype=torch.float32),
        SimpleNamespace(fsdp_unit_id=None, params=[torch.nn.Parameter(torch.ones(1))], dtype=torch.float32),
    ]
    fixed = fsdp_param_buffer.FixedPoolAllocator("fixed", fsdp_groups, size=2)
    assert fixed._is_two_bucket_group_equal([0, 1], [2, 3]) is True
    fixed_bucket = fixed.allocate(0, 2, torch.float32, torch.device("cpu"), mem_alloc_context=object())
    assert fixed_bucket.data.shape == torch.Size([2])
    assert fixed.using_buffer[0] == (0, 0)
    fixed.free(0)
    assert (0, 0) in fixed.idle_buffer
    fallback_bucket = fixed.allocate(4, 1, torch.float32, torch.device("cpu"))
    assert fallback_bucket.data.shape == torch.Size([1])


def test_dynamic_inference_request_record_checkpoint_merge_and_serialization(monkeypatch):
    def _as_msgpack_round_trip(value):
        if isinstance(value, tuple):
            return [_as_msgpack_round_trip(item) for item in value]
        if isinstance(value, list):
            return [_as_msgpack_round_trip(item) for item in value]
        if isinstance(value, dict):
            return {key: _as_msgpack_round_trip(item) for key, item in value.items()}
        return value

    nvtx_calls = []
    fake_nvtx = SimpleNamespace(
        range_push=lambda name: nvtx_calls.append(("push", name)),
        range_pop=lambda: nvtx_calls.append(("pop", None)),
    )
    monkeypatch.setattr(inference_request_module.torch.cuda, "nvtx", fake_nvtx, raising=False)
    monkeypatch.setattr(inference_request_module.torch.distributed, "is_initialized", lambda: False)

    request = DynamicInferenceRequest(
        request_id=11,
        prompt="hello",
        prompt_tokens=torch.tensor([1, 2, 3, 4]),
        sampling_params=SamplingParams(num_tokens_to_generate=5, top_k=3),
        block_size_tokens=2,
        enable_prefix_caching=True,
        status=Status.ACTIVE_AND_GENERATING_TOKENS,
    )
    assert request.remaining_prompt_length == 4
    assert len(request.precomputed_block_hashes) == 2
    assert "id 11" in str(request)

    with pytest.warns(UserWarning, match="Defaulting to -1"):
        metadata = request.tracked_metadata
    assert metadata[3] == -1
    assert request.sampling_params.termination_id == -1

    request.add_event_add_engine()
    request.add_event_add_context()
    generated_event = request.add_event_generated_token(
        9,
        blocks_total=8,
        blocks_hashed_total=4,
        blocks_hashed_active=2,
        blocks_ref_count=3,
        pre_fwd_active_token_count=6,
        pre_fwd_step_count=7,
    )
    request.add_event_pause()
    request.add_event_evict()
    request.add_event_finish()
    request.add_event_fail()
    assert generated_event.payload["token_id"] == 9
    assert generated_event.payload["pre_fwd_step_count"] == 7

    request.generated_tokens = [5, 6]
    request.routing_indices = torch.zeros(5, 1, 1, dtype=torch.int64)
    serialized = request.serialize()
    restored = DynamicInferenceRequest.deserialize(_as_msgpack_round_trip(serialized))
    assert restored.events[2].payload["blocks_ref_count"] == 3
    assert torch.equal(restored.routing_indices, request.routing_indices)
    assert any(call == ("push", "DynamicInferenceRequest.serialize") for call in nvtx_calls)

    request.routing_indices = torch.zeros(1, 1, 1, dtype=torch.int64)
    with pytest.raises(AssertionError, match="routing_indices first dimension"):
        request.serialize()

    request.routing_indices = torch.zeros(5, 1, 1, dtype=torch.int64)
    request.generated_text = "alpha"
    request.generated_log_probs = [0.1, 0.2]
    request.generated_top_n_logprobs = [{"a": -1.0}, {"b": -2.0}]
    request.tpot = [10, 11]
    request.status = Status.COMPLETED
    assert request.succeeded() is True
    request.status = Status.FAILED
    assert request.failed() is True
    request.status = Status.COMPLETED

    record = DynamicInferenceRequestRecord.from_request(request)
    assert record.request_id == 11
    record.checkpoint()
    checkpointed = record[-1]
    assert torch.equal(checkpointed.prompt_tokens, torch.tensor([1, 2, 3, 4, 5, 6]))
    assert checkpointed.sampling_params.num_tokens_to_generate == 3
    assert checkpointed.event_add_engine is request.event_add_engine

    checkpointed.generated_tokens = [7]
    checkpointed.generated_text = "beta"
    checkpointed.generated_log_probs = [0.3]
    checkpointed.generated_top_n_logprobs = [{"c": -3.0}]
    checkpointed.tpot = [12]
    checkpointed.events = [checkpointed.add_event_finish()]
    checkpointed.status = Status.COMPLETED
    checkpointed.policy_epoch = [(1, 2)]
    checkpointed.kv_cache_epoch = [(3, 4)]
    checkpointed.routing_indices = torch.zeros(6, 1, 1, dtype=torch.int64)
    record.latency = 1.25

    merged = record.merge()
    assert merged.generated_tokens == [5, 6, 7]
    assert merged.generated_text == "alphabeta"
    assert merged.generated_length == 3
    assert merged.latency == 1.25
    assert merged.policy_epoch == [(1, 2)]
    assert merged.kv_cache_epoch == [(3, 4)]
    assert len(merged.events) == len(request.events) + 1

    serialized_record = record.serialize()
    restored_record = DynamicInferenceRequestRecord.deserialize(_as_msgpack_round_trip(serialized_record))
    assert restored_record.request_id == 11
    assert len(restored_record.requests) == 2


def test_text_generation_controller_lightweight_tokenization_detokenization_and_sampling(monkeypatch):
    class _TokenizerWithSkip:
        bos = 101
        eod = 102

        def tokenize(self, prompt):
            return [self.bos, 1, 2] if prompt == "with-bos" else [1, 2]

        def detokenize(self, tokens, skip_special_tokens=True):
            suffix = ":skip" if skip_special_tokens else ":keep"
            return "-".join(map(str, tokens)) + suffix

        def offsets(self, tokens, text):
            offsets = []
            current = 0
            for token in tokens:
                offsets.append(current)
                current += len(str(token)) + 1
            return offsets

    class _TokenizerNoSkip(_TokenizerWithSkip):
        def detokenize(self, tokens):
            return "|".join(map(str, tokens))

    tokenizer = _TokenizerWithSkip()
    assert TextGenerationController.tokenize_prompt(tokenizer, "with-bos") == [1, 2]
    assert TextGenerationController.tokenize_prompt(tokenizer, "with-bos", add_BOS=True) == [
        101,
        1,
        2,
    ]
    no_bos = _TokenizerWithSkip()
    no_bos.bos = None
    with pytest.raises(AssertionError):
        TextGenerationController.tokenize_prompt(no_bos, "prompt", add_BOS=True)

    assert TextGenerationController.detokenize(tokenizer, [1, 2, 102, 102]) == "1-2:skip"
    assert (
        TextGenerationController.detokenize(
            tokenizer, [1, 2, 102], remove_EOD=False, skip_special_tokens=False
        )
        == "1-2-102:keep"
    )
    assert TextGenerationController.detokenize(_TokenizerNoSkip(), [1, 2, 102]) == "1|2"

    controller = object.__new__(TextGenerationController)
    controller.tokenizer = tokenizer
    text, segments = controller.detokenize_generations(
        torch.tensor([1, 2, 102]), torch.tensor([3]), detokenize_segments=False
    )
    assert text == "1-2:skip"
    assert segments is None

    text, segments = controller.detokenize_generations(
        torch.tensor([1, 2, 3]), torch.tensor([3]), detokenize_segments=True
    )
    assert text == "1-2-3:skip"
    assert segments == [["1-", "2-", "3:skip"]]

    generation_done, generation_lengths = controller.update_generation_status(
        torch.tensor([[1, 102, 9], [3, 4, 5]]),
        torch.tensor([True, True]),
        1,
        torch.tensor([False, False]),
        torch.tensor([0, 0]),
    )
    assert torch.equal(generation_done, torch.tensor([True, False]))
    assert torch.equal(generation_lengths, torch.tensor([0, 1], dtype=torch.int32))

    generation_done, generation_lengths = controller.update_generation_status(
        torch.tensor([[1, 7], [3, 99]]),
        torch.tensor([False, True]),
        1,
        generation_done,
        generation_lengths,
        termination_id=99,
    )
    assert torch.equal(generation_done, torch.tensor([True, True]))
    assert torch.equal(generation_lengths, torch.tensor([0, 1], dtype=torch.int32))

    controller.sampling_rng = torch.Generator()
    controller.sampling_rng.manual_seed(123)
    logits = torch.tensor([[0.1, 2.0, 0.3], [3.0, 0.2, 0.1]])
    assert torch.equal(
        controller._torch_sampling_func(logits, temperature=1.0, top_k=1, top_p=0.0),
        torch.tensor([1, 0]),
    )
    top_k_sample = controller._torch_sampling_func(
        logits, temperature=0.5, top_k=2, top_p=0.0, vocab_size=3
    )
    assert top_k_sample.shape == torch.Size([2])
    top_p_sample = controller._torch_sampling_func(
        logits, temperature=1.0, top_k=0, top_p=0.8, vocab_size=3
    )
    assert top_p_sample.shape == torch.Size([2])

    with pytest.raises(AssertionError, match="top-p"):
        controller._torch_sampling_func(logits, temperature=1.0, top_k=2, top_p=0.5)
    with pytest.raises(AssertionError, match="top-p should"):
        controller._torch_sampling_func(logits, temperature=1.0, top_k=0, top_p=1.5)
    with pytest.raises(AssertionError, match="top-k is larger"):
        controller._torch_sampling_func(logits, temperature=1.0, top_k=4, top_p=0.0)
    with pytest.raises(AssertionError, match="top-k is larger than vocab size"):
        controller._torch_sampling_func(logits, temperature=1.0, top_k=2, top_p=0.0, vocab_size=2)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    padded = controller.pad_input_prompt_tokens([[1, 2], [3]], 3, 4)
    assert torch.equal(
        padded,
        torch.tensor(
            [
                [1, 2, 102, 102],
                [3, 102, 102, 102],
                [102, 102, 102, 102],
            ]
        ),
    )
    assert torch.equal(controller.unpad_input_prompt_tokens(padded, 2), padded[:2])

    top_n_logprobs = {0: [], 1: []}
    sampled = controller.sample_from_logits(
        torch.tensor([[0.1, 2.0, 0.3], [3.0, 0.2, 0.1]]),
        SamplingParams(top_k=1, top_n_logprobs=2, skip_prompt_log_probs=True),
        generation_started=torch.tensor([True, False]),
        top_n_logprobs_dict=top_n_logprobs,
    )
    assert torch.equal(sampled, torch.tensor([1, 0]))
    assert len(top_n_logprobs[0]) == 1
    assert top_n_logprobs[1] == []
    assert {"1:skip", "2:skip"} == set(top_n_logprobs[0][0])

    prompt_top_n = {0: [], 1: []}
    controller.sample_from_logits(
        torch.tensor([[0.1, 2.0, 0.3], [3.0, 0.2, 0.1]]),
        SamplingParams(top_k=1, top_n_logprobs=1, skip_prompt_log_probs=False),
        generation_started=torch.tensor([False, False]),
        top_n_logprobs_dict=prompt_top_n,
        logits=torch.tensor(
            [
                [[0.0, 1.0, 0.5], [2.0, 0.0, 0.1]],
                [[1.0, 0.2, 0.3], [0.0, 1.0, 2.0]],
            ]
        ),
    )
    assert [len(prompt_top_n[idx]) for idx in (0, 1)] == [2, 2]


def test_text_generation_controller_dynamic_internal_paths(monkeypatch):
    calls = []
    context = SimpleNamespace(
        paused_request_count=0,
        total_request_count=3,
        active_token_count=8,
        request_in_prefill_status_tensor=torch.tensor([0, 0, 1], dtype=torch.int32),
        request_query_lengths=torch.tensor([3, 3, 2], dtype=torch.int32),
        request_kv_length_offsets=torch.tensor([4, 4, 0], dtype=torch.int32),
        request_last_kv_block_offset=torch.tensor([0, 1, 2], dtype=torch.int32),
        request_kv_block_counts=torch.tensor([2, 2, 1], dtype=torch.int32),
        request_last_kv_block_id=torch.tensor([5, 6, 7], dtype=torch.int32),
        request_to_kv_block_ids=torch.tensor(
            [[4, 5, -1], [3, 6, -1], [7, -1, -1]], dtype=torch.int32
        ),
        request_ids=torch.tensor([101, 102, 103], dtype=torch.int32),
        token_to_input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
        request_metadata_types=[
            ("temperature", torch.float32, False),
            ("top_k", torch.int32, False),
            ("top_p", torch.float32, False),
            ("return_log_probs", torch.bool, False),
            ("top_n_logprobs", torch.int32, False),
        ],
        request_metadata={
            "temperature": torch.tensor([1.0, 1.0, 0.5]),
            "top_k": torch.tensor([1, 1, 2], dtype=torch.int32),
            "top_p": torch.tensor([0.0, 0.0, 0.0]),
            "return_log_probs": torch.tensor([False, True, False]),
            "top_n_logprobs": torch.tensor([0, 2, 0], dtype=torch.int32),
        },
        block_size_tokens=4,
        is_hybrid_model=False,
        mamba_metadata=None,
        mamba_conv_states=None,
        mamba_ssm_states=None,
        mamba_intermediate_conv_states=None,
        mamba_intermediate_ssm_states=None,
        kv_block_allocator=SimpleNamespace(
            release_memory_blocks=lambda blocks: calls.append(("release", blocks.clone()))
        ),
        config=SimpleNamespace(materialize_only_last_token_logits=False),
    )
    context.initialize_attention_state = lambda **kwargs: calls.append(("init-attn", kwargs))
    context.using_cuda_graph_this_step = lambda: True
    context.is_decode_only = lambda: False
    context.current_input_and_position_ids = lambda **kwargs: (
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[0, 1, 2]]),
    )
    context.last_token_logits = lambda logits: logits.squeeze(0)[torch.tensor([2, 5, 7])]

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(
                symmetric_ar_type="sym",
                nccl_all_reduce_for_prefill=True,
                moe_pad_experts_for_cuda_graph_inference=True,
                transformer_impl="transformer_engine",
                num_moe_experts=4,
                moe_router_topk=2,
                params_dtype=torch.float32,
                sequence_parallel=False,
                mtp_num_layers=2,
                moe_enable_routing_replay=True,
            )
            self.vocab_size = 5
            self.symmetric_calls = []

        def set_symmetric_ar(self, value):
            self.symmetric_calls.append(value)

    model = _Model()
    wrapper = SimpleNamespace(
        inference_context=context,
        model=model,
        tp_group="tp",
        run_one_forward_step=lambda batch: calls.append(("forward", batch))
        or torch.ones(1, batch["tokens"].shape[1], 5),
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = wrapper
    controller.model_config = model.config
    controller.model_is_pipeline_parallel = True
    controller.pp_group = "pp"
    controller.vocab_size = 5
    controller.num_speculative_tokens = 2
    controller.num_mtp_heads = 2
    controller._sampling_backend = "torch"
    controller._sampled_tokens_cuda = torch.zeros(3, dtype=torch.long)
    controller._sampled_mtp_tokens_cuda = torch.zeros(2, 3, dtype=torch.long)
    controller._accepted_tokens_per_request = torch.full((3, 2), -1, dtype=torch.long)
    controller._accepted_token_counts_per_request = torch.tensor([1, 0, 0], dtype=torch.long)
    controller._request_metadata = {
        key: value.clone() for key, value in context.request_metadata.items()
    }
    controller._torch_sampling_buckets = []
    controller._torch_sampling_func = (
        lambda logits, temp, top_k, top_p, vocab_size=None: torch.argmax(logits, dim=-1)
    )

    monkeypatch.setattr(
        "megatron.core.inference.text_generation_controllers.text_generation_controller.unwrap_model",
        lambda value: value,
    )
    monkeypatch.setattr(
        "megatron.core.inference.text_generation_controllers.text_generation_controller.get_model_config",
        lambda value: value.config,
    )
    monkeypatch.setattr(
        "megatron.core.inference.text_generation_controllers.text_generation_controller.set_decode_expert_padding",
        lambda model, set_to, capacity_factor=None: calls.append(
            ("padding", set_to, capacity_factor)
        ),
    )
    monkeypatch.setattr(
        "megatron.core.inference.text_generation_controllers.text_generation_controller.is_pipeline_last_stage",
        lambda group: True,
    )
    monkeypatch.setattr(
        "megatron.core.inference.text_generation_controllers.text_generation_controller.broadcast_from_last_pipeline_stage",
        lambda shape, dtype, tensor, pp_group: calls.append(("broadcast", shape, dtype, pp_group))
        or tensor,
    )
    monkeypatch.setattr(
        "megatron.core.inference.text_generation_controllers.text_generation_controller.get_pg_size",
        lambda group: 1,
    )

    dims = batch_dimensions_utils.InferenceBatchDimensions(
        token_count=3, prefill_req_count=1, decode_req_count=1
    )
    input_ids, position_ids = controller._dynamic_step_context_init(
        construct_graph_dimensions=dims
    )
    assert torch.equal(input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(position_ids, torch.tensor([[0, 1, 2]]))
    assert ("padding", True, 2.0) in calls
    assert model.symmetric_calls[-1] is None

    context.using_cuda_graph_this_step = lambda: False
    context.is_decode_only = lambda: True
    controller._dynamic_step_context_init()
    assert ("padding", False, None) in calls
    assert model.symmetric_calls[-1] == "sym"

    logits = controller._dynamic_step_forward_logits(
        torch.tensor([[1, 2, 3]]), torch.tensor([[0, 1, 2]])
    )
    assert logits.shape == (1, 3, 5)
    assert any(item[0] == "broadcast" for item in calls if isinstance(item, tuple))

    controller._dynamic_step_sample_bookkeeping()
    assert controller._torch_sampling_buckets == [([0, 1], 1.0, 1, 0.0), ([2], 0.5, 2, 0.0)]
    return_log_probs, top_n_log_probs = controller._dynamic_step_log_probs_bookkeeping()
    assert return_log_probs.item() is True
    assert top_n_log_probs.item() is True

    required_indices = controller._get_required_logit_indices(
        context.request_in_prefill_status_tensor,
        context.request_query_lengths,
        num_decode_requests=2,
        num_prefill_requests=1,
        device=torch.device("cpu"),
    )
    assert torch.equal(required_indices, torch.tensor([0, 1, 2, 3, 4, 5, 7]))
    with pytest.raises(AssertionError, match="Expected length"):
        controller._get_required_logit_indices(
            context.request_in_prefill_status_tensor,
            context.request_query_lengths,
            num_decode_requests=1,
            num_prefill_requests=2,
            device=torch.device("cpu"),
        )

    required_logits = torch.eye(7, 7)
    sampled_tokens, repeats = controller._sample_speculative_logits(
        required_logits, context.request_in_prefill_status_tensor
    )
    assert sampled_tokens.shape == (7,)
    assert torch.equal(repeats, torch.tensor([3, 3, 1]))

    last_indices, accepted_mask, flattened_inputs = controller._verify_speculative_tokens(
        output_tokens=torch.tensor([9, 0, 3, 8, 9, 1, 7]),
        input_tokens_required=torch.tensor([[1, 9, 2, 4, 8, 0, 7]]),
        request_in_prefill_status_tensor=context.request_in_prefill_status_tensor,
        repeats=repeats,
        num_decode_requests=2,
        num_prefill_requests=1,
        active_request_count=3,
    )
    assert torch.equal(last_indices, torch.tensor([1, 4, 6]))
    assert accepted_mask.tolist() == [True, True, False, True, True, False, True]
    assert flattened_inputs.ndim == 1

    logits_for_sample = torch.zeros(1, 8, 10)
    for index in range(8):
        logits_for_sample[0, index, (index + 1) % 10] = 10.0
    controller._dynamic_step_sample_logits_and_verify_tokens(
        logits_for_sample,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
    )
    assert controller._sampled_tokens_cuda[:3].shape == (3,)
    assert controller._accepted_tokens_per_request[:2].shape == (2, 2)
    assert controller._last_accepted_seq_indices.numel() == 3

    controller._accepted_token_counts_per_request = torch.tensor([1, 0, 0], dtype=torch.long)
    controller._rewind_kv_cache()
    assert calls[-1][0] == "release"
    assert torch.equal(calls[-1][1], torch.tensor([5, 6], dtype=torch.int32))
    assert torch.equal(context.request_last_kv_block_id[:2], torch.tensor([4, 3], dtype=torch.int32))
    assert torch.equal(context.request_to_kv_block_ids[:2, 1], torch.tensor([-1, -1], dtype=torch.int32))

    controller._torch_sampling_buckets = [([0, 1, 2], 1.0, 1, 0.0)]
    controller._sampled_tokens_cuda.zero_()
    controller._dynamic_step_sample_logits(torch.arange(1 * 8 * 5, dtype=torch.float32).reshape(1, 8, 5))
    assert torch.equal(controller._sampled_tokens_cuda[:3], torch.tensor([4, 4, 4]))

    context.moe_routing_metadata = SimpleNamespace(
        get_routing_indices=lambda: torch.arange(8 * 2 * 1).reshape(8, 2, 1)
    )
    routing = controller._router_record_bookkeeping()
    assert set(routing) == {101, 102, 103}
    assert routing[101].shape == (3, 2, 1)
    model.config.moe_enable_routing_replay = False
    assert controller._router_record_bookkeeping() is None


def test_dynamic_context_errors_factory_and_memory_size_strings():
    base_error = dynamic_context.ContextOverflowError(7, "full")
    assert str(base_error) == "request 7 | full"
    assert base_error.is_transient is True

    permanent_error = dynamic_context.MaxSequenceLengthOverflowError(3, "too long")
    serialized = dynamic_context.ContextErrorFactory.serialize(permanent_error)
    assert serialized == {
        "type": "MaxSequenceLengthOverflowError",
        "request_id": 3,
        "message": "too long",
        "is_transient": False,
    }
    restored = dynamic_context.ContextErrorFactory.deserialize(serialized)
    assert isinstance(restored, dynamic_context.MaxSequenceLengthOverflowError)
    assert restored.request_id == 3
    assert restored.message == "too long"
    assert restored.is_transient is False

    for error_type, expected_cls in (
        ("ContextOverflowError", dynamic_context.ContextOverflowError),
        ("RequestOverflowError", dynamic_context.RequestOverflowError),
        ("TokenOverflowError", dynamic_context.TokenOverflowError),
        ("BlockOverflowError", dynamic_context.BlockOverflowError),
        ("ActiveRequestCountOverflowError", dynamic_context.ActiveRequestCountOverflowError),
    ):
        error = dynamic_context.ContextErrorFactory.deserialize(
            {
                "type": error_type,
                "request_id": None,
                "message": "overflow",
                "is_transient": True,
            }
        )
        assert isinstance(error, expected_cls)
        assert error.message == "overflow"

    active_error = dynamic_context.ActiveRequestCountOverflowError(2, 3)
    assert "active_request_count (3) > max_request_count (2)" in str(active_error)
    assert active_error.request_id is None

    with pytest.raises(AssertionError):
        dynamic_context.ContextErrorFactory.serialize(ValueError("not a context error"))
    with pytest.raises(KeyError):
        dynamic_context.ContextErrorFactory.deserialize({"type": "UnknownContextError"})

    assert dynamic_context.get_mem_size_str(1) == "1 bytes"
    assert dynamic_context.get_mem_size_str(1024**2) == "1 MB"
    assert dynamic_context.get_mem_size_str(1024**3) == "1 GB"
    assert dynamic_context.get_mem_size_str(1024**4) == "1 TB"


def test_dynamic_context_cpu_state_management_prefix_and_cache_paths(monkeypatch):
    monkeypatch.setattr(dynamic_context.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(dynamic_context.DynamicInferenceContext, "TOKEN_ROUNDER", 4)
    monkeypatch.setattr(dynamic_context.DynamicInferenceContext, "REQUEST_ROUNDER", 2)
    monkeypatch.setattr(dynamic_context.parallel_state, "is_initialized", lambda: False)

    class _FakeAllocator:
        def __init__(self):
            self.total_count = 8
            self.active_count = 7
            self.paused_count = 0
            self.dummy_block_idx = 7
            self.total_avail = 6
            self.kv_hash_to_block_id = {111: 1, 222: 2}
            self.hash_to_block_id = {111: 1, 222: 2}
            self.block_ref_counts = torch.zeros(8, dtype=torch.int32)
            self.released = []
            self.registered = []
            self.timestamps = []
            self.reset_count = 0
            self.on_blocks_deregistered = None

        def reset(self):
            self.reset_count += 1

        def is_memory_available(self, count):
            return count <= self.total_avail

        def get_active_avail(self):
            return self.total_avail

        def allocate_memory_blocks(self, count):
            return torch.arange(3, 3 + count, dtype=torch.int32)

        def release_memory_blocks(self, blocks):
            self.released.append(blocks.clone())

        def update_timestamps(self, blocks):
            self.timestamps.append(blocks.clone())

        def register_kv_block_hashes(self, block_ids, block_hashes):
            self.registered.append((list(block_ids), list(block_hashes)))

    class _FakeMhaMetadata:
        def __init__(self):
            self.reset_count = 0
            self.update_kwargs = None
            self.state_data = {
                "block_table": torch.full((4, 4), -1, dtype=torch.int32),
                "cu_query_seq_lengths": torch.tensor([0, 3, 5], dtype=torch.int32),
                "cu_kv_seq_lengths": torch.tensor([0, 3, 5], dtype=torch.int32),
                "kv_seq_lengths": torch.tensor([3, 2], dtype=torch.int32),
                "max_seqlen_q": 3,
                "max_seqlen_k": 5,
            }

        def reset(self):
            self.reset_count += 1

        def update(self, **kwargs):
            self.update_kwargs = kwargs
            self.state_data["block_table"] = kwargs["request_to_kv_block_ids"][
                : kwargs["padded_batch_dimensions"].req_count
            ]
            self.state_data["cu_query_seq_lengths"] = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32),
                    torch.cumsum(kwargs["request_query_lengths"].to(torch.int32), dim=0),
                ]
            )
            self.state_data["cu_kv_seq_lengths"] = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32),
                    torch.cumsum(
                        (
                            kwargs["request_query_lengths"]
                            + kwargs["request_kv_length_offsets"]
                        ).to(torch.int32),
                        dim=0,
                    ),
                ]
            )
            self.state_data["kv_seq_lengths"] = (
                kwargs["request_query_lengths"] + kwargs["request_kv_length_offsets"]
            )
            self.state_data["max_seqlen_q"] = int(kwargs["request_query_lengths"].max().item())
            self.state_data["max_seqlen_k"] = int(self.state_data["kv_seq_lengths"].max().item())

    def _new_context(enable_prefix_caching=False):
        ctx = object.__new__(dynamic_context.DynamicInferenceContext)
        ctx.config = SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
        ctx.enable_prefix_caching = enable_prefix_caching
        ctx.prefix_caching_eviction_policy = dynamic_context.PrefixCachingEvictionPolicy.LRU
        ctx.prefix_cache_hits = 0
        ctx.prefix_cache_blocks_matched = 0
        ctx.prefix_cache_lru_clock = 0
        ctx.step_count = 0
        ctx.cache_mla_latent = False
        ctx.num_attention_layers = 1
        ctx.num_attention_heads_per_partition = 1
        ctx.hidden_size_per_attention_head = 2
        ctx.layer_map = {0: 0}
        ctx.params_dtype = torch.float32
        ctx.block_size_tokens = 4
        ctx.max_sequence_length = 16
        ctx.max_kv_block_count = 4
        ctx.max_requests = 4
        ctx.max_tokens = 16
        ctx.num_speculative_tokens = 0
        ctx.is_hybrid_model = False
        ctx.mamba_slot_allocator = None
        ctx.mamba_metadata = None
        ctx.moe_enable_routing_replay = False
        ctx.use_cuda_graphs_for_non_decode_steps = True
        ctx.smallest_non_decode_cuda_graph_size = 1
        ctx.cuda_graph_batch_dimensions_list = []
        ctx.cuda_graph_token_counts = []
        ctx.expert_model_parallel_group = None
        ctx.pipeline_parallel_group = None
        ctx.kv_cache_management_mode = dynamic_context.KVCacheManagementMode.PERSIST
        ctx.static_kv_memory_pointers = False
        ctx.unified_memory_level = 0
        ctx._uses_torch_memory_saver = False
        ctx._offloadable_tensor_names = set()
        ctx._offloadable_cpu_backups = {}
        ctx._offloadable_storage_sizes = {}
        ctx.kv_block_allocator = _FakeAllocator()
        ctx.request_metadata_types = DynamicInferenceRequest.get_metadata_types()
        ctx.request_ids = torch.full((4,), -1, dtype=torch.int32)
        ctx.request_query_lengths = torch.zeros(4, dtype=torch.int32)
        ctx.request_in_prefill_status_tensor = torch.zeros(4, dtype=torch.int32)
        ctx.request_output_lengths = torch.zeros(4, dtype=torch.int32)
        ctx.request_kv_length_offsets = torch.zeros(4, dtype=torch.int32)
        ctx.request_kv_block_counts = torch.zeros(4, dtype=torch.int32)
        ctx.request_last_kv_block_id = torch.full((4,), -1, dtype=torch.int32)
        ctx.request_last_kv_block_offset = torch.zeros(4, dtype=torch.int32)
        ctx.request_to_kv_block_ids = torch.full((4, 4), -1, dtype=torch.int32)
        ctx.request_metadata = {
            label: torch.zeros(4, dtype=dtype) for label, dtype, _ in ctx.request_metadata_types
        }
        ctx.token_to_input_ids = torch.zeros(16, dtype=torch.long)
        ctx.token_to_pos_ids = torch.zeros(16, dtype=torch.long)
        ctx.token_to_request_idx = torch.full((16,), -1, dtype=torch.long)
        ctx.token_to_block_idx = torch.full((16,), -1, dtype=torch.long)
        ctx.token_to_position_in_request = torch.zeros(16, dtype=torch.long)
        ctx.token_to_local_position_within_kv_block = torch.zeros(16, dtype=torch.long)
        ctx.memory_buffer = torch.zeros(2, 1, 8, 4, 1, 2)
        ctx.graph_attn_metadata = {"mha_metadata": _FakeMhaMetadata()}
        ctx.non_graph_attn_metadata = {"mha_metadata": _FakeMhaMetadata()}
        ctx.active_attn_metadata = None
        ctx.is_tensor_state_allocated = True
        ctx.total_request_count = 0
        ctx.active_token_count = 0
        ctx.lifetime_prefill_token_count = 0
        ctx.paused_request_count = 0
        ctx.num_prefill_requests = 0
        ctx.chunked_prefill_request_id = -1
        ctx.paused_tokens = None
        ctx.paused_speculative_tokens = None
        ctx._using_cuda_graph_this_step = False
        ctx.is_creating_cuda_graphs = False
        ctx.batch_dimensions = batch_dimensions_utils.InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
        )
        ctx.padded_batch_dimensions = batch_dimensions_utils.InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
        )
        ctx.padded_active_token_count = 0
        ctx.padded_active_request_count = 0
        ctx.padding_slice = slice(0, 0)
        return ctx

    ctx = _new_context()
    assert dynamic_context.DynamicInferenceContext.round_up_tokens(5) == 8
    assert dynamic_context.DynamicInferenceContext.round_up_requests(3) == 4
    assert ctx.is_static_batching() is False
    assert ctx.has_unfinished_requests() is False

    requests = [
        DynamicInferenceRequest(
            request_id=10,
            prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            sampling_params=SamplingParams(num_tokens_to_generate=2, termination_id=99),
        ),
        DynamicInferenceRequest(
            request_id=11,
            prompt_tokens=torch.tensor([4, 5], dtype=torch.long),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
        ),
    ]
    ctx.add_dummy_requests_parallel(requests)
    assert ctx.total_request_count == 2
    assert ctx.active_token_count == 5
    assert ctx.num_prefill_requests == 2
    assert torch.equal(ctx.token_to_input_ids[:5], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(ctx.token_to_request_idx[:5], torch.tensor([0, 0, 0, 1, 1]))
    assert ctx.has_unfinished_requests() is True
    with pytest.raises(dynamic_context.RequestOverflowError):
        ctx.add_dummy_requests_parallel(requests + requests + requests)

    monkeypatch.setattr(
        dynamic_context.CUDAGraphBatchDimensionBuilder,
        "match_graph_config",
        lambda *args, **kwargs: None,
    )
    ctx.initialize_attention_state()
    assert ctx.using_cuda_graph_this_step() is False
    assert ctx.active_attn_metadata is ctx.non_graph_attn_metadata
    assert ctx.padded_active_token_count == 8
    assert ctx.cu_query_lengths()[1] == 3
    assert ctx.cu_kv_lengths()[2] == 3
    assert torch.equal(ctx.get_active_sequence_lengths(), torch.tensor([3, 2], dtype=torch.int32))
    assert torch.equal(ctx.get_max_sequence_lengths(), torch.tensor([5, 3], dtype=torch.int32))
    input_ids, pos_ids = ctx.current_input_and_position_ids()
    assert input_ids.shape == (1, 8)
    assert pos_ids.shape == (1, 8)
    logits = torch.arange(1 * 8 * 3, dtype=torch.float32).reshape(1, 8, 3)
    assert torch.equal(ctx.last_token_logits(logits), logits[0, torch.tensor([2, 4])])

    key = torch.arange(8 * 1 * 1 * 2, dtype=torch.float32).reshape(8, 1, 1, 2)
    value = key + 100
    ctx.append_key_value_cache(1, key, value)
    cached_key, cached_value, block_table = ctx.key_value_cache(1)
    assert cached_key.shape == (8, 4, 1, 2)
    assert cached_value.shape == (8, 4, 1, 2)
    assert block_table.shape[0] >= 2

    query = torch.arange(8 * 1 * 1 * 2, dtype=torch.float32).reshape(8, 1, 1, 2)
    emb = torch.ones(16, 1, 1, 2)
    monkeypatch.setattr(
        dynamic_context,
        "apply_rotary_pos_emb",
        lambda t, freqs, config, **kwargs: t + freqs[: t.shape[0]],
    )
    rotated_query = ctx.apply_rotary_emb_query(
        query.clone(),
        emb,
        SimpleNamespace(rotary_interleaved=False),
        torch.tensor([0, 3, 5], dtype=torch.int32),
        None,
    )
    assert torch.equal(rotated_query[:8], query + 1)
    ctx.num_prefill_requests = 0
    rotated_key = ctx.apply_rotary_emb_key(
        query.clone(),
        emb,
        SimpleNamespace(rotary_interleaved=False),
        None,
    )
    assert torch.equal(rotated_key, query + 1)
    with pytest.raises(AssertionError, match="key.shape"):
        ctx.apply_rotary_emb_key(query[:7].clone(), emb, SimpleNamespace(), None)

    next_tokens = torch.arange(4)
    new_speculative_tokens = torch.arange(8).reshape(2, 4)
    ctx._move_book_keeping_tensors(
        torch.tensor([0]),
        torch.tensor([2]),
        next_tokens,
        new_speculative_tokens,
    )
    assert ctx.request_ids[2] == ctx.request_ids[0]
    before_swap = ctx.request_ids.clone()
    ctx._swap_book_keeping_tensors(torch.tensor([0]), torch.tensor([1]), next_tokens)
    assert ctx.request_ids[0] == before_swap[1]
    assert ctx.request_ids[1] == before_swap[0]
    ctx.chunked_prefill_request_id = int(ctx.request_ids[1])
    assert ctx.get_index_of_chunked_prefill_request() == 1
    ctx.chunked_prefill_request_id = 123456
    assert ctx.get_index_of_chunked_prefill_request(safe=False) == -1
    assert ctx.is_chunked_prefill_enabled() is False

    ctx.release_memory_blocks_from_request_indexes(torch.tensor([0, 1]))
    assert len(ctx.kv_block_allocator.released) == 1
    assert torch.all(ctx.request_to_kv_block_ids[torch.tensor([0, 1])] == -1)
    ctx.reset()
    assert ctx.total_request_count == 0
    assert ctx.active_token_count == 0
    assert ctx.kv_block_allocator.reset_count >= 1

    prefix_ctx = _new_context(enable_prefix_caching=True)
    prefix_req = DynamicInferenceRequest(
        request_id=12,
        prompt_tokens=torch.tensor([9, 8, 7, 6, 5], dtype=torch.long),
        sampling_params=SamplingParams(num_tokens_to_generate=2, termination_id=99),
        block_size_tokens=4,
        enable_prefix_caching=True,
    )
    prefix_req.precomputed_block_hashes = [111, 222, 333]
    prefix_ctx.kv_block_allocator.kv_hash_to_block_id = {111: 1}
    matched, parent_hash = prefix_ctx._find_kv_match_count(prefix_req, 0, 3)
    assert matched == [1]
    assert parent_hash == 111
    prefix_ctx.kv_block_allocator.kv_hash_to_block_id = {}
    prefix_ctx.kv_block_allocator.total_avail = 0
    can_add, tokens_fit, blocks_fit = prefix_ctx.check_availability(prefix_req)
    assert can_add is True
    assert tokens_fit is True
    assert blocks_fit is False
    prefix_ctx.kv_block_allocator.kv_hash_to_block_id = {111: 1}
    prefix_ctx.kv_block_allocator.total_avail = 6
    prefix_ctx.add_request(prefix_req, prefill_chunk_length=5)
    assert prefix_ctx.prefix_cache_hits == 1
    assert prefix_ctx.prefix_cache_blocks_matched == 1
    assert prefix_ctx.total_request_count == 1
    assert prefix_ctx.lifetime_prefill_token_count == 1
    assert prefix_ctx.kv_block_allocator.timestamps

    register_ctx = _new_context(enable_prefix_caching=True)
    register_req = DynamicInferenceRequest(
        request_id=13,
        prompt_tokens=torch.tensor([1, 1, 1, 1], dtype=torch.long),
        sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
        block_size_tokens=4,
        enable_prefix_caching=True,
    )
    register_req.precomputed_block_hashes = [333]
    register_ctx.add_request(register_req, prefill_chunk_length=4)
    assert register_ctx.kv_block_allocator.registered

    offload_ctx = object.__new__(dynamic_context.DynamicInferenceContext)
    offload_ctx.is_tensor_state_allocated = True
    offload_ctx.kv_cache_management_mode = dynamic_context.KVCacheManagementMode.OFFLOAD
    offload_ctx.unified_memory_level = 0
    offload_ctx._uses_torch_memory_saver = False
    offload_ctx.memory_buffer = torch.tensor([1.0, 2.0])
    offload_ctx._offloadable_tensor_names = {"memory_buffer"}
    offload_ctx._offloadable_cpu_backups = {"memory_buffer": torch.empty(2)}
    offload_ctx._offloadable_storage_sizes = {}
    offload_ctx.deallocate_inference_state_buffers()
    assert offload_ctx.is_tensor_state_allocated is False
    offload_ctx.reinitialize_inference_state_buffers()
    assert offload_ctx.is_tensor_state_allocated is True
    assert torch.equal(offload_ctx.memory_buffer, torch.tensor([1.0, 2.0]))

    recompute_ctx = object.__new__(dynamic_context.DynamicInferenceContext)
    recompute_ctx.is_tensor_state_allocated = True
    recompute_ctx.kv_cache_management_mode = dynamic_context.KVCacheManagementMode.RECOMPUTE
    recompute_ctx.unified_memory_level = 0
    recompute_ctx._uses_torch_memory_saver = False
    recompute_ctx.memory_buffer = torch.tensor([3.0])
    recompute_calls = []
    recompute_ctx.initialize_all_tensors = lambda: recompute_calls.append("init")
    recompute_ctx.deallocate_inference_state_buffers()
    assert not hasattr(recompute_ctx, "memory_buffer")
    recompute_ctx.reinitialize_inference_state_buffers()
    assert recompute_calls == ["init"]

    mamba_ctx = object.__new__(dynamic_context.DynamicInferenceContext)
    mamba_ctx.mamba_conv_states_shape = (2,)
    mamba_ctx.mamba_ssm_states_shape = (2,)
    mamba_ctx.mamba_conv_states_dtype = torch.float32
    mamba_ctx.mamba_ssm_states_dtype = torch.float32
    mamba_ctx.num_mamba_layers = 2
    mamba_ctx.kv_block_allocator = SimpleNamespace(on_blocks_deregistered=None)
    mamba_allocations = []
    monkeypatch.setattr(
        dynamic_context,
        "MambaSlotAllocator",
        lambda **kwargs: mamba_allocations.append(kwargs)
        or SimpleNamespace(on_kv_blocks_deregistered=lambda blocks: blocks),
    )
    mamba_ctx._allocate_mamba_cache(0.000001)
    assert mamba_allocations
    assert mamba_ctx.kv_block_allocator.on_blocks_deregistered is not None


def test_dynamic_engine_request_failure_suspend_resume_and_context_manager_paths(monkeypatch):
    assert dynamic_engine.format_mem_bytes(1) == "1 bytes"
    assert dynamic_engine.format_mem_bytes(2048) == "2.0 kb"
    assert dynamic_engine.format_mem_bytes(3 * 1024**2) == "3.0 mb"
    assert dynamic_engine.format_mem_bytes(4 * 1024**3) == "4.0 gb"

    loop = asyncio.new_event_loop()
    calls = []

    class _Tokenizer:
        eod = 99

        def detokenize(self, tokens):
            return "|".join(str(token) for token in tokens)

    class _Controller:
        tokenizer = _Tokenizer()

        def tokenize_prompt(self, tokenizer, prompt, add_BOS=False):
            calls.append(("tokenize", prompt, add_BOS))
            return [1, 2, 3] if add_BOS else [2, 3]

    def _new_engine(max_sequence_length=8, max_tokens=8, materialize_last=False):
        engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
        engine.controller = _Controller()
        engine.context = SimpleNamespace(
            max_sequence_length=max_sequence_length,
            max_tokens=max_tokens,
            block_size_tokens=2,
            enable_prefix_caching=False,
            has_unfinished_requests=lambda: False,
            kv_cache_management_mode=dynamic_engine.KVCacheManagementMode.PERSIST,
            static_kv_memory_pointers=False,
            deallocate_inference_state_buffers=lambda: calls.append("deallocate"),
            reinitialize_inference_state_buffers=lambda: calls.append("reinitialize"),
            chunked_prefill_request_id=-1,
        )
        engine._loop = loop
        engine.requests = {}
        engine.waiting_request_ids = dynamic_engine.deque()
        engine.failed_request_ids = []
        engine._generation_epoch = None
        engine.rank = 0
        engine.materialize_only_last_token_logits = materialize_last
        engine.enable_chunked_prefill = False
        engine.track_paused_request_events = True
        engine.track_generated_token_events = True
        engine.use_coordinator = False
        engine.is_mp_coordinator = False
        engine.state = dynamic_engine.EngineState.RUNNING
        engine.unified_memory_level = 0
        engine.resume_request_ids = None
        engine.stop_word_finished_request_ids = set()
        engine.stop_word_being_finished_ids = set()
        return engine

    try:
        monkeypatch.setattr(dynamic_engine.torch.cuda, "current_device", lambda: "cpu")
        monkeypatch.setattr(dynamic_engine.torch.distributed, "get_rank", lambda: 0)

        engine = _new_engine()
        valid_request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([4, 5], dtype=torch.long),
            sampling_params=SamplingParams(
                num_tokens_total=5,
                termination_id=None,
                add_BOS=True,
                stop_words=["stop"],
            ),
        )
        future = engine._add_request(valid_request)
        assert future.done() is False
        assert engine.waiting_request_ids[0] == 1
        assert valid_request.sampling_params.num_tokens_to_generate == 3
        assert valid_request.sampling_params.termination_id == 99
        assert valid_request.stop_word_ids == [[2, 3]]
        assert valid_request.status == Status.ACTIVE_AND_GENERATING_TOKENS
        assert valid_request.event_add_engine is not None
        assert engine.get_request(1) is valid_request
        assert engine.has_unfinished_requests() is True

        engine._generation_epoch = 7
        epoch_request = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=torch.tensor([6], dtype=torch.long),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
        )
        engine._add_request(epoch_request)
        assert epoch_request.policy_epoch == [(0, 7)]
        assert epoch_request.kv_cache_epoch == [(0, 7)]

        failed_engine = _new_engine(max_sequence_length=3, max_tokens=2)
        too_long = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            sampling_params=SamplingParams(num_tokens_to_generate=2, termination_id=99),
            generated_tokens=[8, 9],
        )
        failed_future = failed_engine._add_request(too_long)
        assert failed_future.done() is True
        assert too_long.status == Status.FAILED
        assert 3 in failed_engine.failed_request_ids
        assert too_long.prompt == "1|2|3"
        assert too_long.generated_text == "8|9"
        assert any(event.type == DynamicInferenceEventType.ERROR_NONTRANSIENT for event in too_long.events)
        assert any(event.type == DynamicInferenceEventType.FAIL for event in too_long.events)

        logprob_engine = _new_engine(materialize_last=True)
        logprob_request = DynamicInferenceRequest(
            request_id=4,
            prompt_tokens=torch.tensor([1], dtype=torch.long),
            sampling_params=SamplingParams(
                num_tokens_to_generate=1,
                return_log_probs=True,
                skip_prompt_log_probs=False,
                termination_id=99,
            ),
        )
        with pytest.raises(AssertionError, match="Prompt log probs"):
            logprob_engine._add_request(logprob_request)

        topn_engine = _new_engine()
        topn_request = DynamicInferenceRequest(
            request_id=5,
            prompt_tokens=torch.tensor([1], dtype=torch.long),
            sampling_params=SamplingParams(
                num_tokens_to_generate=1,
                return_log_probs=False,
                top_n_logprobs=1,
                termination_id=99,
            ),
        )
        with pytest.raises(AssertionError, match="top_n_logprobs"):
            topn_engine._add_request(topn_request)

        add_engine = _new_engine()
        text_future = add_engine.add_request(
            6,
            "hello",
            SamplingParams(num_tokens_to_generate=1, termination_id=99, add_BOS=True),
        )
        assert text_future.done() is False
        assert calls[-1] == ("tokenize", "hello", True)
        list_future = add_engine.add_request(
            7,
            [7, 8],
            SamplingParams(num_tokens_to_generate=1, termination_id=99),
        )
        assert list_future.done() is False
        with pytest.raises(Exception, match="specialize"):
            add_engine.add_request(8, object(), SamplingParams(num_tokens_to_generate=1))

        async def _wait_until_paths():
            wait_engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
            wait_engine._state_events = {
                state: asyncio.Event() for state in dynamic_engine.DynamicInferenceEngine._STATE_EVENTS
            }
            wait_engine._state_events[dynamic_engine.EngineState.RUNNING].set()
            await wait_engine.wait_until(dynamic_engine.EngineState.RUNNING)
            with pytest.raises(ValueError, match="transient"):
                await wait_engine.wait_until(dynamic_engine.EngineState.PAUSING)

        loop.run_until_complete(_wait_until_paths())

        monkeypatch.setattr(dynamic_engine, "range_push", lambda name: calls.append(("push", name)))
        monkeypatch.setattr(dynamic_engine, "range_pop", lambda: calls.append("pop"))
        monkeypatch.setattr(dynamic_engine.torch.cuda, "memory_stats", lambda: {
            "allocated_bytes.all.current": 1024,
            "reserved_bytes.all.current": 2048,
        })
        monkeypatch.setattr(dynamic_engine.torch.cuda, "synchronize", lambda: calls.append("sync"))
        monkeypatch.setattr(dynamic_engine, "HAVE_PSUTIL", False)
        with dynamic_engine.DynamicInferenceEngine.suspend_resume_ctx(
            "unit", unified_memory_level=0
        ):
            calls.append("inside")
        assert ("push", "unit-inference-context") in calls
        assert "pop" in calls
        assert "inside" in calls

        suspend_engine = _new_engine()
        suspend_engine.context.kv_cache_management_mode = dynamic_engine.KVCacheManagementMode.RECOMPUTE
        suspend_engine.waiting_request_ids.extend([10])
        active_request = DynamicInferenceRequest(
            request_id=9,
            prompt_tokens=torch.tensor([1, 2], dtype=torch.long),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
            finished_chunk_token_count=1,
        )
        waiting_request = DynamicInferenceRequest(
            request_id=10,
            prompt_tokens=torch.tensor([3, 4], dtype=torch.long),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
            finished_chunk_token_count=1,
        )
        suspend_engine.requests = {
            9: dynamic_engine.RequestEntry(
                record=DynamicInferenceRequestRecord.from_request(active_request),
                future=loop.create_future(),
            ),
            10: dynamic_engine.RequestEntry(
                record=DynamicInferenceRequestRecord.from_request(waiting_request),
                future=loop.create_future(),
            ),
        }
        monkeypatch.setattr(dynamic_engine, "delete_cuda_graphs", lambda: calls.append("delete-graphs"))
        suspend_engine.suspend()
        assert suspend_engine.state == dynamic_engine.EngineState.SUSPENDED
        assert set(suspend_engine.resume_request_ids) == {9, 10}
        assert list(suspend_engine.waiting_request_ids) == []
        assert active_request.finished_chunk_token_count == 0
        assert waiting_request.finished_chunk_token_count == 0
        assert "deallocate" in calls
        assert "delete-graphs" in calls

        resume_engine = _new_engine()
        resume_engine.state = dynamic_engine.EngineState.SUSPENDED
        resume_engine.context.kv_cache_management_mode = dynamic_engine.KVCacheManagementMode.RECOMPUTE
        resume_engine.resume_request_ids = [21, 22]
        resume_engine._add_request = lambda request: calls.append(("readd", request.request_id))
        resume_engine.get_request = lambda request_id: SimpleNamespace(request_id=request_id)
        resume_engine.create_cuda_graphs = lambda: calls.append("create-graphs")
        resume_engine._loop = SimpleNamespace(
            call_soon_threadsafe=lambda create_task, coro: (
                coro.close(),
                calls.append(("notify", create_task)),
            )
        )
        resume_engine.resume()
        assert resume_engine.state == dynamic_engine.EngineState.RUNNING
        assert ("readd", 21) in calls
        assert ("readd", 22) in calls
        assert "reinitialize" in calls
        assert "create-graphs" in calls

        stopped_engine = _new_engine()
        stopped_engine.state = dynamic_engine.EngineState.SUSPENDED
        stopped_engine.suspend()
        assert stopped_engine.state == dynamic_engine.EngineState.SUSPENDED
        stopped_engine.state = dynamic_engine.EngineState.RUNNING
        stopped_engine.resume()
        assert stopped_engine.state == dynamic_engine.EngineState.RUNNING
    finally:
        loop.close()


def test_dynamic_engine_reset_and_cuda_graph_creation_cpu_paths(monkeypatch):
    calls = []

    class _FakeEvent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    reset_engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
    reset_engine.context = SimpleNamespace(reset=lambda: calls.append("context-reset"))
    monkeypatch.setattr(dynamic_engine.torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(dynamic_engine.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(dynamic_engine, "get_asyncio_loop", lambda existing=None: "loop")

    dynamic_engine.DynamicInferenceEngine.reset(reset_engine)

    assert calls == ["context-reset"]
    assert reset_engine.rank == 3
    assert reset_engine.state == dynamic_engine.EngineState.RUNNING
    assert reset_engine._state_events[dynamic_engine.EngineState.RUNNING].is_set()
    assert reset_engine.requests == {}
    assert list(reset_engine.waiting_request_ids) == []
    assert reset_engine._prefix_cache_hits == 0
    assert reset_engine.use_coordinator is False

    graph_engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
    graph_engine.cuda_graph_impl = "none"
    graph_engine.cuda_graph_scope = []
    assert dynamic_engine.DynamicInferenceEngine.create_cuda_graphs(graph_engine) is None

    model_config = SimpleNamespace(
        transformer_impl="inference_optimized",
        expert_model_parallel_size=2,
        moe_enable_routing_replay=True,
    )
    model = SimpleNamespace(config=model_config)

    class _Controller:
        inference_wrapped_model = SimpleNamespace(model=model)

        def _dynamic_step_context_init(self, construct_graph_dimensions=None):
            calls.append(("init-graph", construct_graph_dimensions))
            return "input-ids", "position-ids"

        def _dynamic_step_forward_logits(self, input_ids, position_ids):
            calls.append(("forward-logits", input_ids, position_ids))
            return torch.ones(1)

    graph_context = SimpleNamespace(
        cuda_graph_batch_dimensions_list=["graph-a", "graph-b"],
        reset=lambda: calls.append("graph-context-reset"),
    )
    graph_engine.cuda_graph_impl = "local"
    graph_engine.cuda_graph_scope = [dynamic_engine.CudaGraphScope.full_iteration]
    graph_engine.context = graph_context
    graph_engine.controller = _Controller()
    graph_engine.capture_stats = None

    memory_stats = iter(
        [
            {
                "allocated_bytes.all.current": 1024,
                "reserved_bytes.all.current": 2048,
            },
            {
                "allocated_bytes.all.current": 3 * 1024,
                "reserved_bytes.all.current": 6 * 1024,
            },
        ]
    )
    time_values = iter([10.0, 12.5])
    monkeypatch.setattr(dynamic_engine, "HAVE_TQDM", False)
    monkeypatch.setattr(dynamic_engine.time, "time", lambda: next(time_values))
    monkeypatch.setattr(dynamic_engine.torch.cuda, "memory_stats", lambda: next(memory_stats))
    monkeypatch.setattr(
        dynamic_engine,
        "set_inference_cuda_graphed_iteration_for_ep_inference",
        lambda unwrapped_model: calls.append(("set-ep-graph", unwrapped_model)),
    )
    monkeypatch.setattr(
        dynamic_engine,
        "unset_inference_cuda_graphed_iteration_for_ep_inference",
        lambda unwrapped_model: calls.append(("unset-ep-graph", unwrapped_model)),
    )
    monkeypatch.setattr(
        dynamic_engine.RouterReplay,
        "set_global_router_replay_action",
        lambda action: calls.append(("router-action", action)),
    )

    with pytest.warns(UserWarning, match="full_iteration"):
        dynamic_engine.DynamicInferenceEngine.create_cuda_graphs(graph_engine)

    assert ("set-ep-graph", model) in calls
    assert ("unset-ep-graph", model) in calls
    assert ("init-graph", "graph-a") in calls
    assert ("init-graph", "graph-b") in calls
    assert calls.count("graph-context-reset") == 2
    assert (
        "router-action",
        dynamic_engine.RouterReplayAction.RECORD,
    ) in calls
    assert graph_engine.capture_stats == {
        "time": 2.5,
        "allocated_bytes": 2 * 1024,
        "reserved_bytes": 4 * 1024,
    }


def test_dynamic_engine_postprocess_schedule_and_shutdown_paths(monkeypatch):
    loop = asyncio.new_event_loop()
    calls = []

    class _Tokenizer:
        def detokenize(self, tokens):
            return "|".join(str(token) for token in tokens)

    class _Socket:
        def __init__(self, messages=None):
            self.messages = list(messages or [])
            self.sent = []
            self.closed = False

        def recv(self, flags=None):
            if not self.messages:
                raise dynamic_engine.zmq.Again()
            return self.messages.pop(0)

        def recv_multipart(self):
            return self.messages.pop(0)

        def send(self, payload):
            self.sent.append(payload)

        def send_multipart(self, payload):
            self.sent.append(("multipart", payload))

        def close(self, linger=0):
            self.closed = True
            self.linger = linger

    class _Context:
        chunked_prefill_request_id = -1
        enable_prefix_caching = True
        prefix_cache_hits = 2
        prefix_cache_blocks_matched = 3
        step_count = 1
        batch_dimensions = "real"
        padded_batch_dimensions = "padded"
        padded_active_token_count = 4

        def using_cuda_graph_this_step(self):
            return True

    fake_zmq = SimpleNamespace(NOBLOCK=1, Again=type("Again", (Exception,), {}))
    fake_msgpack = SimpleNamespace(
        packb=lambda obj, use_bin_type=True: obj,
        unpackb=lambda obj, raw=False: obj,
    )
    monkeypatch.setattr(dynamic_engine, "zmq", fake_zmq, raising=False)
    monkeypatch.setattr(dynamic_engine, "msgpack", fake_msgpack, raising=False)
    monkeypatch.setattr(dynamic_engine, "range_push", lambda name: calls.append(("push", name)))
    monkeypatch.setattr(dynamic_engine, "range_pop", lambda: calls.append("pop"))
    monkeypatch.setattr(dynamic_engine.torch.cuda, "memory_stats", lambda: {
        "allocation.all.current": 2,
        "allocated_bytes.all.current": 1024,
        "reserved_bytes.all.current": 2048,
    })

    engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
    engine.controller = SimpleNamespace(
        tokenizer=_Tokenizer(),
        detokenize=lambda tokenizer, tokens, remove_EOD=True: tokenizer.detokenize(tokens),
    )
    engine.context = _Context()
    engine.context.kv_block_allocator = SimpleNamespace(
        total_count=10,
        total_avail=7,
        enable_prefix_caching=True,
        block_ref_counts=torch.tensor([1, 0, 2], dtype=torch.int32),
    )
    engine.requests = {}
    engine.waiting_request_ids = dynamic_engine.deque()
    engine.finished_request_count = 0
    engine.evicted_request_count = 0
    engine.track_generated_token_events = True
    engine.num_speculative_tokens = 1
    engine._spec_tokens_proposed = 0
    engine._spec_tokens_accepted = 0
    engine._spec_steps = 0
    engine.stop_word_finished_request_ids = set()
    engine.stop_word_being_finished_ids = set()
    engine.use_coordinator = False
    engine.is_mp_coordinator = False
    engine.rank = 0

    def _entry(request):
        request.add_event_add_engine()
        future = loop.create_future()
        return dynamic_engine.RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(request),
            future=future,
        )

    req1 = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2], dtype=torch.long),
        sampling_params=SamplingParams(
            num_tokens_to_generate=3,
            skip_prompt_log_probs=True,
            termination_id=99,
        ),
    )
    req2 = DynamicInferenceRequest(
        request_id=2,
        prompt_tokens=torch.tensor([3, 4], dtype=torch.long),
        sampling_params=SamplingParams(
            num_tokens_to_generate=2,
            skip_prompt_log_probs=False,
            termination_id=99,
        ),
    )
    engine.requests[1] = _entry(req1)
    engine.requests[2] = _entry(req2)
    req2_future = engine.requests[2].future

    active_ids, finished_records = dynamic_engine.DynamicInferenceEngine.post_process_requests(
        engine,
        request_ids=torch.tensor([1, 2], dtype=torch.int64),
        finished_request_ids=torch.tensor([2], dtype=torch.int64),
        evict_request_ids=torch.tensor([1], dtype=torch.int64),
        step_time=0.4,
        sample=torch.tensor([7, 9], dtype=torch.long),
        accepted_tokens=torch.tensor([[11, -1], [12, -1]], dtype=torch.long),
        log_probs=[torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
        top_n_logprobs={
            0: [(torch.tensor([0.5]), torch.tensor([7]))],
            1: [(torch.tensor([0.6]), torch.tensor([9]))],
        },
        routing_indices_per_request={1: torch.ones(2, 1, 1, dtype=torch.long)},
        pre_fwd_active_token_count=4,
        pre_fwd_step_count=8,
    )

    assert active_ids == [1]
    assert len(finished_records) == 1
    assert finished_records[0].request_id == 2
    assert engine.finished_request_count == 1
    assert engine.evicted_request_count == 1
    assert list(engine.waiting_request_ids) == [1]
    assert req1.generated_tokens == [11, 7]
    assert len(req1.generated_log_probs) == 1
    assert req1.routing_indices.shape == (2, 1, 1)
    assert req2.status == Status.COMPLETED
    assert req2_future.done() is True

    engine.failed_request_ids = [3]
    failed = DynamicInferenceRequest(
        request_id=3,
        prompt_tokens=torch.tensor([5], dtype=torch.long),
        sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
        status=Status.FAILED,
    )
    failed_entry = _entry(failed)
    failed_entry.future.set_result(failed_entry.record)
    engine.requests[3] = failed_entry
    engine.logging_step_interval = 1
    engine.metrics_writer = None
    engine._prefix_cache_hits = 0
    engine._prefix_cache_blocks_matched = 0
    bookkeep = loop.run_until_complete(
        dynamic_engine.DynamicInferenceEngine.async_bookkeep(
            engine,
            None,
            {
                "kv_stats": None,
                "is_decode_only": True,
                "total_request_count": 1,
                "paused_request_count": 0,
                "max_requests": 4,
                "waiting_request_count": 1,
                "finished_request_count": 0,
                "evicted_request_count": 0,
                "total_active_used_blocks": 1,
                "total_active_block_count": 4,
                "total_paused_used_blocks": 0,
                "total_paused_block_count": 0,
            },
            0.2,
        )
    )
    assert bookkeep["finished_request_records"][0].request_id == 3
    assert engine.failed_request_ids == []
    assert engine._prefix_cache_hits == 0

    submit = [
        dynamic_engine.Headers.SUBMIT_REQUEST.value,
        10,
        [1, 2],
        SamplingParams(num_tokens_to_generate=1, termination_id=99).serialize(),
    ]
    epoch = [dynamic_engine.Headers.SET_GENERATION_EPOCH.value, 5]
    pause = [dynamic_engine.Headers.PAUSE.value]
    schedule_engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
    schedule_engine.is_mp_coordinator = True
    schedule_engine.socket_for_receiving_requests = _Socket([submit, epoch, pause])
    schedule_engine.model_parallel_num_msgs_publisher_socket = _Socket()
    schedule_engine.model_parallel_publisher_socket = _Socket()
    schedule_engine._pending_signals = dynamic_engine.deque()
    schedule_engine.state = dynamic_engine.EngineState.RUNNING
    schedule_engine._state_events = {
        state: asyncio.Event() for state in dynamic_engine.DynamicInferenceEngine._STATE_EVENTS
    }
    schedule_engine._state_events[dynamic_engine.EngineState.RUNNING].set()
    schedule_engine.requests = {
        9: dynamic_engine.RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(
                DynamicInferenceRequest(
                    request_id=9,
                    prompt_tokens=torch.tensor([1, 2], dtype=torch.long),
                    generated_tokens=[3],
                    sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=99),
                    policy_epoch=[(0, 1)],
                    kv_cache_epoch=[(0, 1)],
                )
            ),
            future=loop.create_future(),
        )
    }
    schedule_engine.add_request = lambda request_id, prompt, sampling_params: calls.append(
        ("scheduled-add", request_id, prompt, sampling_params.num_tokens_to_generate)
    )

    assert dynamic_engine.DynamicInferenceEngine.schedule_requests(schedule_engine) == 3
    assert ("scheduled-add", 10, [1, 2], 1) in calls
    assert schedule_engine.state == dynamic_engine.EngineState.PAUSING
    assert schedule_engine._generation_epoch == 5
    stamped = schedule_engine.requests[9].record[-1]
    assert stamped.policy_epoch[-1] == (2, 5)
    assert stamped.kv_cache_epoch[-1] == (2, 5)

    shutdown_engine = object.__new__(dynamic_engine.DynamicInferenceEngine)
    pending_future = loop.create_future()
    shutdown_engine.requests = {
        1: dynamic_engine.RequestEntry(record=engine.requests[1].record, future=pending_future)
    }
    shutdown_engine.socket_for_receiving_requests = _Socket()
    shutdown_sockets = [_Socket(), _Socket()]
    shutdown_engine.zmq_sockets = shutdown_sockets
    shutdown_engine.expert_parallel_zmq_communicator = SimpleNamespace(
        close=lambda: calls.append("close-ep")
    )
    shutdown_engine.world_zmq_communicator = SimpleNamespace(
        close=lambda: calls.append("close-world")
    )
    shutdown_engine.zmq_context = SimpleNamespace(
        closed=False,
        term=lambda: calls.append("term-zmq"),
    )
    shutdown_engine._state_events = {
        state: asyncio.Event() for state in dynamic_engine.DynamicInferenceEngine._STATE_EVENTS
    }
    loop.run_until_complete(dynamic_engine.DynamicInferenceEngine.shutdown(shutdown_engine))
    assert shutdown_engine.state == dynamic_engine.EngineState.STOPPED
    assert pending_future.cancelled() is True
    assert all(socket.closed for socket in shutdown_sockets)
    assert "close-ep" in calls and "close-world" in calls and "term-zmq" in calls
    assert shutdown_engine._state_events[dynamic_engine.EngineState.STOPPED].is_set()

    loop.close()


def test_text_generation_server_generate_endpoint_validation_and_success(monkeypatch):
    from megatron.core.inference.text_generation_server import (
        text_generation_server as generation_server,
    )

    calls = []

    class _Request:
        remote_addr = "127.0.0.1"

        def __init__(self, payload):
            self.payload = payload

        def get_json(self):
            return self.payload

    def call(payload):
        monkeypatch.setattr(generation_server, "request", _Request(payload), raising=False)
        endpoint = generation_server.MegatronGenerate(
            "engine", SimpleNamespace(inference_flask_server_logging=True)
        )
        return endpoint.put()

    monkeypatch.setattr(generation_server, "LOCK", _ReusableNullContext())
    monkeypatch.setattr(generation_server, "jsonify", lambda obj: ("json", obj), raising=False)
    monkeypatch.setattr(generation_server, "send_do_generate", lambda: calls.append("send"))
    monkeypatch.setattr(
        generation_server,
        "run_mcore_engine",
        lambda *args, **kwargs: calls.append(("run", args[1:])) or {"text": ["ok"]},
    )
    monkeypatch.setattr(generation_server.logging, "info", lambda message: calls.append(("log", message)))

    invalid_payloads = [
        ({}, "prompts argument required"),
        ({"prompts": ["x"], "max_len": 4}, "max_len is no longer used"),
        ({"prompts": ["x"], "sentences": ["x"]}, "sentences is no longer used"),
        ({"prompts": ["x"], "beam_width": 1}, "Beam search is no longer supported."),
        ({"prompts": "x"}, "prompts is not a list of strings"),
        ({"prompts": []}, "prompts is empty"),
        ({"prompts": ["x"] * 129}, "Maximum number of prompts is 128"),
        ({"prompts": ["x"], "tokens_to_generate": "bad"}, "tokens_to_generate must be"),
        ({"prompts": ["x"], "tokens_to_generate": -1}, "tokens_to_generate must be"),
        ({"prompts": ["x"], "tokens_to_generate": 0}, "tokens_to_generate=0 implies"),
        ({"prompts": ["x"], "logprobs": "bad"}, "logprobs must be"),
        ({"prompts": ["x"], "temperature": "hot"}, "temperature must be"),
        ({"prompts": ["x"], "temperature": 0.0}, "temperature must be"),
        ({"prompts": ["x"], "top_k": 1.5}, "top_k must be"),
        ({"prompts": ["x"], "top_k": 1001}, "top_k must be"),
        ({"prompts": ["x"], "top_p": 1}, "top_p must be"),
        ({"prompts": ["x"], "top_k": 1, "top_p": 0.5}, "cannot set both"),
        ({"prompts": ["x"], "top_p": 1.5}, "top_p must be less"),
        ({"prompts": ["x"], "top_p_decay": "bad"}, "top_p_decay must be"),
        ({"prompts": ["x"], "top_p_decay": 0.5}, "top_p_decay cannot"),
        ({"prompts": ["x"], "top_p": 0.8, "top_p_decay": 1.5}, "top_p_decay must be"),
        ({"prompts": ["x"], "top_p_bound": "bad"}, "top_p_bound must be"),
        ({"prompts": ["x"], "top_p_bound": 0.5}, "top_p_bound cannot"),
        ({"prompts": ["x"], "top_p": 0.4, "top_p_bound": 0.5}, "top_p_bound must be"),
        ({"prompts": ["x"], "add_BOS": "bad"}, "add_BOS must be"),
        ({"prompts": [""]}, "Empty prompts require"),
        ({"prompts": ["x"], "stop_on_double_eol": 1}, "stop_on_double_eol must be"),
        ({"prompts": ["x"], "stop_on_eol": 1}, "stop_on_eol must be"),
        ({"prompts": ["x"], "prevent_newline_after_colon": 1}, "prevent_newline_after_colon"),
        ({"prompts": ["x"], "random_seed": "seed"}, "random_seed must be"),
        ({"prompts": ["x"], "random_seed": -1}, "random_seed must be"),
        ({"prompts": ["x"], "stop_token": "eod"}, "stop_token must be"),
        ({"prompts": ["x"], "length_penalty": 1}, "length_penalty must be"),
    ]
    for payload, expected in invalid_payloads:
        response = call(payload)
        message = response[0] if isinstance(response, tuple) else response
        assert expected in message

    response = call(
        {
            "prompts": ["x"],
            "tokens_to_generate": 2,
            "logprobs": True,
            "temperature": 0.5,
            "top_p": 0.8,
            "top_p_decay": 0.1,
            "top_p_bound": 0.2,
            "add_BOS": True,
            "stop_on_double_eol": False,
            "stop_on_eol": False,
            "prevent_newline_after_colon": False,
            "random_seed": 4,
            "stop_token": 5,
            "length_penalty": 1.0,
        }
    )
    assert response == ("json", {"text": ["ok"]})
    assert "send" in calls
    assert any(item[0] == "run" for item in calls if isinstance(item, tuple))

    monkeypatch.setattr(
        generation_server,
        "run_mcore_engine",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("engine failed")),
    )
    assert call({"prompts": ["x"]}) == "engine failed"

    monkeypatch.setattr(generation_server, "HAVE_FLASK", False)
    with pytest.raises(RuntimeError, match="flask"):
        generation_server.MegatronServer("model")


def test_text_generation_completions_endpoint_detokenize_and_post(monkeypatch):
    from megatron.core.inference.text_generation_server.endpoints import (
        completions as completions_endpoint,
    )

    class _Tokenizer:
        eod = 9

        def detokenize(self, tokens):
            if isinstance(tokens, int):
                return f"tok{tokens}"
            if isinstance(tokens, list):
                return "".join(f"tok{token}" for token in tokens)
            return str(tokens)

        def offsets(self, tokens, text):
            offsets = []
            cursor = 0
            for token in tokens:
                offsets.append(cursor)
                cursor += len(self.detokenize([token]))
            return offsets

    tokenizer = _Tokenizer()
    assert completions_endpoint.detokenize("prompt", tokenizer) == ["prompt"]
    assert completions_endpoint.detokenize(["a", "b"], tokenizer) == ["a", "b"]
    assert completions_endpoint.detokenize([1, 2], tokenizer) == ["tok1"]
    assert completions_endpoint.detokenize([[1, 2], [3]], tokenizer) == ["tok1tok2", "tok3"]
    for bad_prompt in ([], [1, "bad"], object()):
        with pytest.raises(ValueError):
            completions_endpoint.detokenize(bad_prompt, tokenizer)

    calls = []

    class _Request:
        def __init__(self, payload):
            self.payload = payload

        def get_json(self):
            return self.payload

    def call(payload):
        monkeypatch.setattr(completions_endpoint, "request", _Request(payload), raising=False)
        endpoint = completions_endpoint.MegatronCompletions(
            SimpleNamespace(controller=SimpleNamespace(tokenizer=tokenizer)), SimpleNamespace()
        )
        return endpoint.post()

    monkeypatch.setattr(completions_endpoint, "HAVE_FLASK", True)
    monkeypatch.setattr(completions_endpoint, "LOCK", _ReusableNullContext())
    monkeypatch.setattr(completions_endpoint, "send_do_generate", lambda: calls.append("send"))
    monkeypatch.setattr(completions_endpoint, "jsonify", lambda obj: ("json", obj), raising=False)

    monkeypatch.setattr(
        completions_endpoint,
        "run_mcore_engine",
        lambda *args, **kwargs: calls.append(("run", args, kwargs))
        or {
            "text": ["prompttok1tok2STOP"],
            "segments": [[["prompt", "tok1", "tok2", "STOP"]]],
            "logprobs": [[-0.1, -0.2, -0.3]],
            "tokens": [[1, 2, 3]],
            "top_n_logprobs": [[{"tok1": -0.1}, {"tok2": -0.2}, {"STOP": -0.3}]],
        },
    )

    assert call({"prompt": "x", "max_tokens": 0, "echo": False}) == (
        "echo=False not supported when tokens_to_generate == 0",
        400,
    )
    assert call({"prompt": "x", "max_tokens": 1, "best_of": 2}) == (
        "best_of > 1 not supported",
        400,
    )
    assert call({"prompt": "x", "max_tokens": 1, "n": 2}) == (
        "num_completions > 1 not supported",
        400,
    )
    assert call({"prompt": "x", "max_tokens": 1, "logprobs": 1}) == (
        "cannot return top-k unless tokens_to_generate=0 at this time",
        400,
    )
    assert call({"prompt": "x", "max_tokens": 0, "echo": True, "logprobs": 11}) == (
        "return_topk_logprobs > 10 not supported",
        400,
    )

    response = call(
        {
            "prompt": "prompt",
            "max_tokens": 0,
            "temperature": 0,
            "top_k": 4,
            "top_p": 0.7,
            "logprobs": 1,
            "echo": True,
            "seed": 123,
            "stop": "STOP",
        }
    )
    assert response[0] == "json"
    assert response[1]["choices"][0]["index"] == 0
    assert response[1]["choices"][0]["logprobs"]["top_logprobs"][0] is None
    assert "send" in calls
    run_call = next(item for item in calls if isinstance(item, tuple) and item[0] == "run")
    assert run_call[2]["random_seed"] == 123
    assert run_call[1][1] == ["prompt"]
    assert run_call[1][2] == 0.0
    assert run_call[1][3] == 1
    assert run_call[1][4] == 0


def test_dynamic_chat_completions_helpers_normalize_and_sanitize(monkeypatch):
    chat_endpoint = _load_dynamic_endpoint(
        monkeypatch,
        "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions",
    )

    redacted = chat_endpoint._redact_token_id_lists_for_logging(
        {
            "prompt_tokens": [1, 2, 3],
            "nested": [{"moe_topk_indices": [[1, 2], [3, 4]], "tpot": [1.0, 2.0]}],
            "keep": ["not", "numeric"],
        }
    )
    assert redacted["prompt_tokens"] == "...truncated..."
    assert redacted["nested"][0]["moe_topk_indices"] == "...truncated..."
    assert redacted["nested"][0]["tpot"] == "...truncated..."
    assert redacted["keep"] == ["not", "numeric"]

    tools = [
        {
            "function": {
                "name": "update_reservation_flights",
                "parameters": {
                    "properties": {
                        "legs": {"type": "array"},
                        "meta": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                    }
                },
            }
        }
    ]
    normalized = chat_endpoint._normalize_tool_calls(
        [
            {
                "id": 7,
                "function": {
                    "name": "update_reservation_flights",
                    "arguments": '{"legs": "[1, 2]", "meta": "{\\"a\\": 3}"}',
                },
            },
            {"function": {"name": "cancel_reservation", "arguments": {"id": 1}}},
            {"function": {"arguments": 5}},
        ],
        tools=tools,
    )
    assert len(normalized) == 1
    assert normalized[0]["id"] == "7"
    parsed_args = json.loads(normalized[0]["function"]["arguments"])
    assert parsed_args == {"legs": [1, 2], "meta": {"a": 3}}

    assert chat_endpoint._normalize_assistant_content(
        "before transfer",
        [{"function": {"name": "transfer_to_human_agents", "arguments": "{}"}}],
    ) == "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."

    messages = chat_endpoint._sanitize_messages_for_template(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, {"text": " world"}],
            },
            {
                "role": "assistant",
                "content": {"text": "answer"},
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": '{"city": "Paris"}'}},
                    {"function": {"name": "bad", "arguments": "[1, 2]"}},
                ],
            },
            "passthrough",
        ]
    )
    assert messages[0]["content"] == "hello world"
    assert messages[1]["content"] == "answer"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert messages[1]["tool_calls"][1]["function"]["arguments"] == {}
    assert messages[2] == "passthrough"

    tools_for_template = chat_endpoint._sanitize_tools_for_template(
        [{"function": {"name": "lookup", "parameters": "bad"}}, "drop-me"]
    )
    assert tools_for_template == [
        {"function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    ]

    reconstructed = chat_endpoint._reconstruct_reasoning_content(
        [{"role": "assistant", "content": "final", "reasoning_content": "think"}]
    )
    assert reconstructed[0]["content"] == "<think>think</think>final"

    replaced = chat_endpoint._replace_prefix_tokens(
        99,
        [1, 2, 99],
        [8, 99],
        [7, 99, 10, 11],
    )
    assert replaced == [1, 2, 99, 10, 11]


def test_dynamic_chat_completions_endpoint_success_errors_and_fallback(monkeypatch):
    chat_endpoint = _load_dynamic_endpoint(
        monkeypatch,
        "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions",
    )
    monkeypatch.setattr(chat_endpoint, "HAVE_ORJSON", False, raising=False)
    monkeypatch.setattr(chat_endpoint, "jsonify", lambda obj: ("json", obj), raising=False)

    calls = []

    class _Tokenizer:
        chat_template = "template"
        bos = 101
        eos_id = 99

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, tools=None, **kwargs):
            calls.append(("template", messages, add_generation_prompt, tools, kwargs))
            return [101, 1, 2, 3]

        def tokenize(self, text):
            calls.append(("tokenize", text))
            return [5, 6]

        def detokenize(self, tokens):
            return "".join(f"T{token}" for token in tokens)

    class _Client:
        async def add_request(self, prompt_tokens, sampling_params):
            calls.append(("request", list(prompt_tokens), sampling_params))
            return {
                "status": "COMPLETED",
                "prompt_tokens": list(prompt_tokens),
                "generated_text": "answer",
                "generated_tokens": [20, 21],
                "log_probs": [-0.1, -0.2],
                "generated_top_n_logprobs": [{"A": -0.1}, {"B": -0.2}],
                "generated_log_probs": [-0.1, -0.2],
                "sampling_params": {"num_tokens_to_generate": 2},
                "prompt": "prompt",
                "policy_epoch": 1,
                "kv_cache_epoch": 2,
                "events": [{"type": "EVICT"}, {"type": "OTHER"}],
                "routing_indices": [[0], [1], [2], [3]],
            }

    class _Parser:
        @staticmethod
        def parse(message_text, tools=None):
            return (
                "parsed",
                {
                    "reasoning": "chain",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"items": "[1, 2]"},
                            }
                        }
                    ],
                },
            )

    monkeypatch.setattr(chat_endpoint, "PARSER_MAPPING", {"parser": _Parser})
    monkeypatch.setattr(
        chat_endpoint,
        "current_app",
        SimpleNamespace(
            config={
                "client": _Client(),
                "tokenizer": _Tokenizer(),
                "parsers": ["parser"],
                "verbose": True,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        chat_endpoint,
        "request",
        _AsyncJsonRequest(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                ],
                "tools": [
                    {
                        "function": {
                            "name": "lookup",
                            "parameters": {"properties": {"items": {"type": "array"}}},
                        }
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0.0,
                "top_p": 0.7,
                "top_k": 3,
                "n": 2,
                "logprobs": True,
                "top_logprobs": 1,
                "skip_prompt_log_probs": False,
                "add_BOS": True,
                "max_completion_tokens": 2,
            }
        ),
        raising=False,
    )

    response = asyncio.run(chat_endpoint.chat_completions())
    assert response[0] == "json"
    payload = response[1]
    assert payload["object"] == "chat.completion"
    assert len(payload["choices"]) == 2
    assert payload["choices"][0]["message"]["content"] == "parsed"
    assert payload["choices"][0]["message"]["reasoning_content"] == "chain"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["choices"][0]["num_evictions"] == 1
    assert payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["token"] == "A"
    assert payload["usage"]["completion_tokens"] == 4
    first_request = next(item for item in calls if item[0] == "request")
    assert first_request[1] == [101, 1, 2, 3]
    assert first_request[2].top_k == 1
    assert first_request[2].top_p == 0.0

    monkeypatch.setattr(chat_endpoint, "request", _AsyncJsonRequest({}), raising=False)
    missing = asyncio.run(chat_endpoint.chat_completions())
    assert isinstance(missing, _FakeQuartResponse)
    assert missing.status == 400
    assert "messages" in missing.body

    monkeypatch.setattr(
        chat_endpoint,
        "request",
        _AsyncJsonRequest({"messages": "bad"}),
        raising=False,
    )
    bad_messages = asyncio.run(chat_endpoint.chat_completions())
    assert bad_messages.status == 400
    assert "must be a list" in bad_messages.body

    class _BadTokenizer(_Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("template failed")

    monkeypatch.setattr(
        chat_endpoint,
        "current_app",
        SimpleNamespace(
            config={
                "client": _Client(),
                "tokenizer": _BadTokenizer(),
                "parsers": [],
                "verbose": False,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        chat_endpoint,
        "request",
        _AsyncJsonRequest({"messages": [{"role": "user", "content": "hi"}]}),
        raising=False,
    )
    template_error = asyncio.run(chat_endpoint.chat_completions())
    assert template_error.status == 500
    assert "template failed" in template_error.body

    class _FallbackTokenizer:
        chat_template = None
        bos = None

        def tokenize(self, text):
            return [9, 8]

        def detokenize(self, tokens):
            return "fallback"

    class _FailingClient:
        async def add_request(self, prompt_tokens, sampling_params):
            raise RuntimeError("engine failed")

    monkeypatch.setattr(
        chat_endpoint,
        "current_app",
        SimpleNamespace(
            config={
                "client": _FailingClient(),
                "tokenizer": _FallbackTokenizer(),
                "parsers": [],
                "verbose": False,
            }
        ),
        raising=False,
    )
    engine_error = asyncio.run(chat_endpoint.chat_completions())
    assert engine_error.status == 500
    assert "engine failed" in engine_error.body

    class _FailedClient:
        async def add_request(self, prompt_tokens, sampling_params):
            return {
                "status": "FAILED",
                "events": [{"type": "ERROR_NONTRANSIENT", "payload": "bad request"}],
            }

    monkeypatch.setattr(
        chat_endpoint,
        "current_app",
        SimpleNamespace(
            config={
                "client": _FailedClient(),
                "tokenizer": _FallbackTokenizer(),
                "parsers": [],
                "verbose": False,
            }
        ),
        raising=False,
    )
    failed_record = asyncio.run(chat_endpoint.chat_completions())
    assert failed_record.status == 400
    assert "bad request" in failed_record.body


def test_dynamic_completions_endpoint_prompt_sampling_logprobs_and_errors(monkeypatch):
    completions_endpoint = _load_dynamic_endpoint(
        monkeypatch,
        "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions",
    )
    monkeypatch.setattr(completions_endpoint, "jsonify", lambda obj: ("json", obj), raising=False)

    calls = []

    class _Tokenizer:
        def tokenize(self, text):
            if text == "explode":
                raise RuntimeError("tokenizer failed")
            return [ord(ch) % 10 for ch in text]

        def detokenize(self, tokens):
            return "".join(f"T{token}" for token in tokens)

    class _Client:
        async def add_request(self, prompt_tokens, sampling_params):
            calls.append(("request", list(prompt_tokens), sampling_params))
            return {
                "status": "COMPLETED",
                "prompt_tokens": list(prompt_tokens),
                "generated_text": "gen",
                "generated_tokens": [7, 8],
                "prompt_log_probs": [-0.5],
                "prompt_top_n_logprobs": [{"T1": -0.5}],
                "generated_log_probs": [-0.1, -0.2],
                "generated_top_n_logprobs": [{"T7": -0.1}, {"T8": -0.2}],
                "routing_indices": ["p0", "g0", "g1"],
            }

    def set_request(payload, client=None, verbose=False):
        monkeypatch.setattr(
            completions_endpoint,
            "current_app",
            SimpleNamespace(
                config={
                    "client": client or _Client(),
                    "tokenizer": _Tokenizer(),
                    "verbose": verbose,
                }
            ),
            raising=False,
        )
        monkeypatch.setattr(completions_endpoint, "request", _AsyncJsonRequest(payload), raising=False)

    set_request(None)
    assert asyncio.run(completions_endpoint.completions()) == ("Invalid or missing JSON body", 400)

    for payload, expected in (
        ({}, "prompt"),
        ({"prompt": []}, "prompt"),
        ({"prompt": {"bad": True}}, "type"),
        ({"prompt": [1, "bad"]}, "format"),
    ):
        set_request(payload)
        response = asyncio.run(completions_endpoint.completions())
        assert response[1] == 400
        assert expected in response[0]

    set_request({"prompt": "explode"})
    token_error = asyncio.run(completions_endpoint.completions())
    assert token_error[1] == 500
    assert "tokenizer failed" in token_error[0]

    set_request({"prompt": "abc", "temperature": "hot"})
    sampling_error = asyncio.run(completions_endpoint.completions())
    assert sampling_error[1] == 400
    assert "Invalid sampling parameter" in sampling_error[0]

    set_request(
        {
            "prompt": [[1, 2], [3]],
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 3,
            "echo": True,
            "logprobs": 1,
            "max_tokens": 2,
            "stop": "END",
        },
        verbose=True,
    )
    response = asyncio.run(completions_endpoint.completions())
    assert response[0] == "json"
    payload = response[1]
    assert len(payload["choices"]) == 2
    assert payload["choices"][0]["text"].startswith("T1T2")
    assert payload["choices"][0]["logprobs"]["tokens"][0] == "T1"
    assert payload["choices"][0]["prompt_moe_topk_indices"] == ["p0", "g0"]
    assert payload["choices"][0]["moe_topk_indices"] == ["p0", "g0", "g1"]
    assert calls[0][2].top_k == 1
    assert calls[0][2].top_p == 0.0
    assert calls[0][2].skip_prompt_log_probs is False
    assert calls[0][2].stop_words == ["END"]

    class _TransientFailedClient:
        async def add_request(self, prompt_tokens, sampling_params):
            return {"status": "FAILED", "events": [{"type": "ERROR_TRANSIENT", "payload": "retry"}]}

    set_request({"prompt": "abc"}, client=_TransientFailedClient())
    transient = asyncio.run(completions_endpoint.completions())
    assert transient[1] == 500
    assert "retry" in transient[0]

    class _NonTransientFailedClient:
        async def add_request(self, prompt_tokens, sampling_params):
            return {
                "status": "FAILED",
                "events": [{"type": "ERROR_NONTRANSIENT", "payload": "invalid"}],
            }

    set_request({"prompt": "abc"}, client=_NonTransientFailedClient())
    nontransient = asyncio.run(completions_endpoint.completions())
    assert nontransient[1] == 400
    assert "invalid" in nontransient[0]

    class _EngineErrorClient:
        async def add_request(self, prompt_tokens, sampling_params):
            raise RuntimeError("engine boom")

    set_request({"prompt": "abc"}, client=_EngineErrorClient())
    engine_error = asyncio.run(completions_endpoint.completions())
    assert engine_error[1] == 500
    assert "engine boom" in engine_error[0]


def test_masked_wordpiece_dataset_config_cache_and_masking_paths(monkeypatch, tmp_path):
    from megatron.core.datasets import masked_dataset

    class _Tokenizer:
        cls = 101
        sep = 102
        mask = 103
        pad = 0
        eos = 104
        eod = 105
        special_tokens_dict = {"pad_token": pad, "eos_token": eos}
        inv_vocab = {
            101: "[CLS]",
            102: "[SEP]",
            11: "hello",
            12: "##ly",
            13: "world",
            14: "again",
            15: "##s",
            16: "tail",
        }

    config = masked_dataset.MaskedWordPieceDatasetConfig(
        random_seed=123,
        sequence_length=8,
        tokenizer=_Tokenizer(),
        path_to_cache=str(tmp_path),
        masking_probability=0.5,
        short_sequence_probability=0.25,
        masking_max_ngram=3,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=True,
        masking_use_geometric_distribution=False,
    )
    assert config.mock is True

    with pytest.raises(AssertionError):
        masked_dataset.MaskedWordPieceDatasetConfig(
            random_seed=1,
            sequence_length=8,
            tokenizer=_Tokenizer(),
            masking_probability=0.5,
            short_sequence_probability=0.1,
            masking_max_ngram=2,
            masking_do_full_word=True,
            masking_do_permutation=True,
            masking_use_longer_ngrams=False,
            masking_use_geometric_distribution=True,
        )

    log_calls = []
    monkeypatch.setattr(
        masked_dataset,
        "log_single_rank",
        lambda logger, level, message: log_calls.append((level, message)),
    )
    masked_dataset.MaskedWordPieceDatasetConfig(
        random_seed=1,
        sequence_length=8,
        tokenizer=_Tokenizer(),
        masking_probability=0.5,
        short_sequence_probability=0.1,
        masking_max_ngram=2,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=True,
        masking_use_geometric_distribution=True,
    )
    assert any("geometric distribution overrides" in message for _, message in log_calls)

    class _LowLevelDataset:
        document_indices = torch.tensor([0, 2, 5])

    assert masked_dataset.MaskedWordPieceDataset.numel_low_level_dataset(_LowLevelDataset()) == 2
    monkeypatch.setattr(masked_dataset, "IndexedDataset", lambda path: ("indexed", path))
    assert masked_dataset.MaskedWordPieceDataset.build_low_level_dataset("prefix", config) == (
        "indexed",
        "prefix",
    )
    assert "masking_probability" in masked_dataset.MaskedWordPieceDataset._key_config_attributes()

    class _ToyMaskedDataset(masked_dataset.MaskedWordPieceDataset):
        def __getitem__(self, idx):
            return idx

        def _get_token_mask(self, numpy_random_state):
            return self.config.tokenizer.mask

    dataset = object.__new__(_ToyMaskedDataset)
    dataset.config = config
    dataset.sample_index = torch.arange(3)
    assert len(dataset) == 3

    masked_tokens, positions, labels, boundaries, spans = dataset._create_masked_lm_predictions(
        [101, 11, 12, 13, 102, 14, 15, 16, 102],
        target_sequence_length=8,
        numpy_random_state=masked_dataset.numpy.random.RandomState(7),
    )
    assert masked_tokens[0] == 101 and masked_tokens[-1] == 102
    assert positions == sorted(positions)
    assert all(label in {11, 12, 13, 14, 15, 16} for label in labels)
    assert boundaries == [1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert spans == sorted(spans, key=lambda item: item[0][0])

    permutation_config = masked_dataset.MaskedWordPieceDatasetConfig(
        random_seed=123,
        sequence_length=8,
        tokenizer=_Tokenizer(),
        path_to_cache=str(tmp_path),
        masking_probability=0.5,
        short_sequence_probability=0.25,
        masking_max_ngram=2,
        masking_do_full_word=False,
        masking_do_permutation=True,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=False,
    )
    permutation_dataset = object.__new__(_ToyMaskedDataset)
    permutation_dataset.config = permutation_config
    permutation_dataset._create_masked_lm_predictions(
        [101, 11, 12, 13, 102, 14, 15, 16, 102],
        target_sequence_length=8,
        numpy_random_state=masked_dataset.numpy.random.RandomState(11),
    )

    cached_dataset = object.__new__(_ToyMaskedDataset)
    cached_dataset.config = config
    cached_dataset.dataset = SimpleNamespace(path_prefix="prefix", document_indices=torch.arange(5), sequence_lengths=torch.arange(5))
    cached_dataset.indices = masked_dataset.numpy.array([0, 1, 2])
    cached_dataset.num_samples = None
    cached_dataset.index_split = dataset_utils.Split.train
    cached_dataset.unique_description_hash = "hash"
    cached_dataset.unique_description = "description"
    cache_dir = tmp_path
    (cache_dir / "hash-_ToyMaskedDataset-description.txt").write_text("description", encoding="utf-8")
    saved_index = masked_dataset.numpy.array([[0, 1], [1, 2]])
    masked_dataset.numpy.save(cache_dir / "hash-_ToyMaskedDataset-sample_index.npy", saved_index)
    loaded = cached_dataset._build_sample_index(sequence_length=8, min_sentences_per_sample=1)
    assert loaded.shape == saved_index.shape


def test_blended_dataset_indices_cache_defer_and_getitem(monkeypatch, tmp_path):
    import numpy

    from megatron.core.datasets import blended_dataset

    class _ToyDataset:
        index_split = dataset_utils.Split.train

        def __init__(self, name, length=6):
            self.name = name
            self.length = length
            self.unique_identifiers = {"name": name}

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {"sample": f"{self.name}-{idx}"}

    def build_blending_indices(dataset_index, dataset_sample_index, weights, num_datasets, size, verbose):
        assert num_datasets == 2
        dataset_index[:] = numpy.array([0, 1, 0, 1], dtype=numpy.int16)
        dataset_sample_index[:] = numpy.array([0, 0, 1, 1], dtype=numpy.int64)

    def build_exhaustive_indices(dataset_index, dataset_sample_index, weights, num_datasets):
        dataset_index[:] = numpy.array([0, 0, 1], dtype=numpy.int16)
        dataset_sample_index[:] = numpy.array([0, 1, 0], dtype=numpy.int64)

    fake_helpers = ModuleType("megatron.core.datasets.helpers")
    fake_helpers.build_blending_indices = build_blending_indices
    fake_helpers.build_exhaustive_blending_indices = build_exhaustive_indices
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.helpers", fake_helpers)
    monkeypatch.setattr(
        sys.modules["megatron.core.datasets"], "helpers", fake_helpers, raising=False
    )
    monkeypatch.setattr(blended_dataset, "is_built_on_zero_rank", lambda: True)
    monkeypatch.setattr(blended_dataset.time, "time", lambda: 1.0)

    config = SimpleNamespace(
        defer_npy_index_mmap=False,
        path_to_cache=None,
        fast_cache_load=False,
        mid_level_dataset_surplus=0.5,
    )
    dataset = blended_dataset.BlendedDataset(
        [_ToyDataset("a"), _ToyDataset("b")], [0.25, 0.75], 4, config
    )
    assert len(dataset) == 4
    assert dataset[1] == {"dataset_id": 1, "sample": "b-0"}
    assert dataset.dataset_index.tolist() == [0, 1, 0, 1]

    exhaustive = blended_dataset.BlendedDataset(
        [_ToyDataset("a"), _ToyDataset("b")], [2, 1], None, config
    )
    assert len(exhaustive) == 3
    assert exhaustive[2] == {"dataset_id": 1, "sample": "b-0"}

    cache_config = SimpleNamespace(
        defer_npy_index_mmap=False,
        path_to_cache=str(tmp_path),
        fast_cache_load=False,
        mid_level_dataset_surplus=0.5,
    )
    cached = blended_dataset.BlendedDataset(
        [_ToyDataset("a"), _ToyDataset("b")], [0.5, 0.5], 4, cache_config
    )
    hash_prefix = f"{cached.unique_description_hash}-BlendedDataset-train"
    assert (tmp_path / f"{hash_prefix}-description.txt").is_file()

    cache_hit = blended_dataset.BlendedDataset(
        [_ToyDataset("a"), _ToyDataset("b")], [0.5, 0.5], 4, cache_config
    )
    assert cache_hit.dataset_index.shape[0] == 4

    defer_config = SimpleNamespace(
        defer_npy_index_mmap=True,
        path_to_cache=str(tmp_path),
        fast_cache_load=False,
        mid_level_dataset_surplus=0.5,
    )
    deferred = blended_dataset.BlendedDataset(
        [_ToyDataset("a"), _ToyDataset("b")], [2, 1], None, defer_config
    )
    assert len(deferred) == 3
    numpy.save(deferred.path_to_dataset_index, numpy.array([1], dtype=numpy.int16))
    numpy.save(deferred.path_to_dataset_sample_index, numpy.array([2], dtype=numpy.int64))
    assert deferred[0] == {"dataset_id": 1, "sample": "b-2"}

    def oversample_indices(dataset_index, dataset_sample_index, weights, num_datasets, size, verbose):
        dataset_index[:] = numpy.array([0, 0, 0, 0], dtype=numpy.int16)
        dataset_sample_index[:] = numpy.arange(size, dtype=numpy.int64)

    fake_helpers.build_blending_indices = oversample_indices
    with pytest.raises(IndexError, match="oversamples"):
        blended_dataset.BlendedDataset([_ToyDataset("small", length=1)], [1.0], 4, config)


def test_blended_megatron_dataset_builder_mock_blend_and_per_split_paths(monkeypatch):
    from megatron.core.datasets import blended_megatron_dataset_builder as builder_module
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
        _get_size_per_split_per_dataset,
    )

    class _ToyMidDataset:
        def __init__(self, low_level_dataset, dataset_path, indexed_indices, num_samples, split, config):
            self.low_level_dataset = low_level_dataset
            self.dataset_path = dataset_path
            self.indexed_indices = indexed_indices
            self.num_samples = num_samples
            self.index_split = split
            self.config = config
            self.unique_identifiers = {"path": dataset_path, "split": split.name}

        def __len__(self):
            return 10 if self.num_samples is None else max(self.num_samples, 1)

        @staticmethod
        def build_low_level_dataset(dataset_path, config):
            return list(range(12)) if dataset_path is not None else list(range(9))

        @staticmethod
        def numel_low_level_dataset(low_level_dataset):
            return len(low_level_dataset)

    monkeypatch.setattr(builder_module.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(builder_module, "is_built_on_zero_rank", lambda: True)
    monkeypatch.setattr(builder_module.cur_platform, "device_count", lambda: 4)

    class _FakeBlendedDataset:
        def __init__(self, datasets, weights, size, config):
            self.datasets = datasets
            self.weights = weights
            self.size = size
            self.config = config
            self.args = (datasets, weights, size, config)

        def __len__(self):
            return sum(len(dataset) for dataset in self.datasets) if self.size is None else self.size

    monkeypatch.setattr(builder_module, "BlendedDataset", _FakeBlendedDataset)

    config = SimpleNamespace(
        mock=True,
        split_matrix=[(0.0, 0.5), (0.5, 0.75), None],
        blend=None,
        blend_per_split=None,
        multiple_validation_sets=False,
        full_validation=False,
        num_dataset_builder_threads=1,
        mid_level_dataset_surplus=0.25,
        fast_cache_load=False,
    )
    builder = BlendedMegatronDatasetBuilder(_ToyMidDataset, [4, 2, None], lambda: True, config)
    built = builder.build()
    assert [dataset.index_split.name if dataset else None for dataset in built] == [
        "train",
        "valid",
        None,
    ]
    assert built[0].indexed_indices.tolist() == [0, 1, 2, 3]
    assert built[1].indexed_indices.tolist() == [4, 5, 6]

    single_config = SimpleNamespace(
        mock=False,
        split_matrix=[(0.0, 0.5), None, None],
        blend=(["prefix"], None),
        blend_per_split=None,
        multiple_validation_sets=False,
        full_validation=False,
        num_dataset_builder_threads=1,
        mid_level_dataset_surplus=0.25,
        fast_cache_load=False,
    )
    single = BlendedMegatronDatasetBuilder(_ToyMidDataset, [3, None, None], lambda: True, single_config)
    single_built = single._build_blended_dataset_splits()
    assert single_built[0].dataset_path == "prefix"
    assert single_built[1] is None

    class _FakeDataset:
        def __init__(self, name, length):
            self.name = name
            self.length = length
            self.num_samples = None
            self.unique_identifiers = {"name": name}
            self.index_split = dataset_utils.Split.train

        def __len__(self):
            return self.length

    generic_calls = []

    def fake_generic(cls, is_built_on_rank, synchronize_ranks, *args):
        generic_calls.append((cls, synchronize_ranks, args))
        if cls is builder_module.BlendedDataset:
            return cls(*args)
        return cls(*args)

    weighted_config = SimpleNamespace(
        mock=False,
        split_matrix=[(0.0, 0.5), (0.5, 1.0), None],
        blend=(["a", "b"], [1.0, 3.0]),
        blend_per_split=None,
        multiple_validation_sets=False,
        full_validation=False,
        num_dataset_builder_threads=1,
        mid_level_dataset_surplus=0.5,
        fast_cache_load=False,
    )
    weighted = BlendedMegatronDatasetBuilder(
        _ToyMidDataset, [8, 4, 0], lambda: True, weighted_config
    )
    monkeypatch.setattr(
        weighted,
        "_build_megatron_datasets_parallel",
        lambda prefixes, split, sizes: [
            [_FakeDataset("a-train", 10), _FakeDataset("b-train", 10)],
            [_FakeDataset("a-valid", 10), _FakeDataset("b-valid", 10)],
            [],
        ],
    )
    monkeypatch.setattr(weighted, "build_generic_dataset", fake_generic)
    weighted_built = weighted._build_blended_dataset_splits()
    assert weighted_built[0].size == 8
    assert weighted_built[1].size == 4
    assert generic_calls[0][2][1] == [0.25, 0.75]

    unweighted_config = SimpleNamespace(
        mock=False,
        split_matrix=[(0.0, 1.0), None, None],
        blend=(["a", "b"], None),
        blend_per_split=None,
        multiple_validation_sets=False,
        full_validation=False,
        num_dataset_builder_threads=1,
        mid_level_dataset_surplus=0.0,
        fast_cache_load=False,
    )
    unweighted = BlendedMegatronDatasetBuilder(
        _ToyMidDataset, [7, None, None], lambda: True, unweighted_config
    )
    monkeypatch.setattr(
        unweighted,
        "_build_megatron_datasets_parallel",
        lambda prefixes, split, sizes: [
            [_FakeDataset("a-train", 3), _FakeDataset("b-train", 5)],
            [],
            [],
        ],
    )
    monkeypatch.setattr(unweighted, "build_generic_dataset", fake_generic)
    unweighted_built = unweighted._build_blended_dataset_splits()
    assert unweighted_built[0].size == 7
    assert unweighted_built[0].args[1] == [3, 5]

    per_split_config = SimpleNamespace(
        mock=False,
        split_matrix=[None, None, None],
        blend=None,
        blend_per_split=[
            (["train-a", "train-b"], [0.5, 0.5]),
            (["valid-a", "valid-b"], None),
            (["test-only"], None),
        ],
        multiple_validation_sets=True,
        full_validation=True,
        num_dataset_builder_threads=1,
        mid_level_dataset_surplus=0.0,
        fast_cache_load=False,
    )
    per_split = BlendedMegatronDatasetBuilder(
        _ToyMidDataset, [6, None, 2], lambda: True, per_split_config
    )
    monkeypatch.setattr(
        per_split,
        "_build_megatron_datasets_parallel",
        lambda prefixes, split, sizes: [
            [_FakeDataset(f"{prefixes[0]}-train", 3), _FakeDataset(f"{prefixes[-1]}-train", 3)],
            [],
            [],
        ],
    )
    monkeypatch.setattr(per_split, "build_generic_dataset", fake_generic)
    per_split_built = per_split._build_blended_dataset_splits()
    assert per_split_built[0].size == 6
    assert [dataset.dataset_path for dataset in per_split_built[1]] == ["valid-a", "valid-b"]
    assert per_split_built[2].dataset_path == "test-only"

    assert _get_size_per_split_per_dataset([0.25, 0.75], [8, 4, 0], surplus=0.5) == [
        [3, 2, 0],
        [9, 5, 0],
    ]


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"fp16": True, "bf16": True}, ValueError),
        ({"num_attention_heads": 3, "tensor_model_parallel_size": 2}, ValueError),
        ({"num_query_groups": 3, "tensor_model_parallel_size": 2}, ValueError),
        ({"experimental_attention_variant": "gated_delta_net"}, AssertionError),
        (
            {
                "experimental_attention_variant": "gated_delta_net",
                "linear_attention_freq": 1,
                "linear_conv_kernel_dim": 3,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 3,
            },
            AssertionError,
        ),
        (
            {
                "experimental_attention_variant": "gated_delta_net",
                "linear_attention_freq": 1,
                "linear_conv_kernel_dim": 3,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "context_parallel_size": 2,
            },
            AssertionError,
        ),
        (
            {"fp8": True, "first_last_layers_bf16": True, "fp8_recipe": Fp8Recipe.delayed},
            ValueError,
        ),
        (
            {
                "fp8": True,
                "first_last_layers_bf16": True,
                "fp8_recipe": Fp8Recipe.blockwise,
                "num_layers_at_start_in_bf16": 5,
            },
            ValueError,
        ),
        ({"fp8": True, "fp8_recipe": Fp8Recipe.custom}, ValueError),
        ({"fp8_param": True}, ValueError),
        ({"fp4_param": True}, ValueError),
        ({"fp4": True, "fp8": True}, ValueError),
        ({"fp4": True, "fp4_recipe": Fp4Recipe.custom}, ValueError),
        ({"expert_model_parallel_size": 2}, ValueError),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 2,
                "expert_tensor_parallel_size": 2,
            },
            ValueError,
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 2,
                "moe_expert_capacity_factor": 1.0,
            },
            ValueError,
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 2,
                "moe_router_dtype": "fp16",
            },
            ValueError,
        ),
        (
            {
                "transformer_impl": "inference_optimized",
                "num_moe_experts": 2,
                "cuda_graph_impl": "local",
                "inference_grouped_gemm_backend": "te",
            },
            ValueError,
        ),
        ({"num_moe_experts": 0}, ValueError),
        ({"moe_ffn_hidden_size": 32}, AssertionError),
        ({"moe_enable_deepep": True, "moe_token_dispatcher_type": "alltoall"}, ValueError),
        (
            {
                "moe_enable_deepep": True,
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
            },
            ValueError,
        ),
        (
            {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "deepep",
                "moe_pad_expert_input_to_capacity": True,
                "moe_expert_capacity_factor": 1.0,
            },
            ValueError,
        ),
        ({"moe_shared_expert_intermediate_size": 0}, ValueError),
        (
            {
                "moe_shared_expert_intermediate_size": 8,
                "moe_shared_expert_overlap": True,
                "moe_token_dispatcher_type": "allgather",
            },
            ValueError,
        ),
        (
            {
                "moe_router_load_balancing_type": ["aux_loss", "none"],
                "moe_aux_loss_coeff": [0.1],
            },
            AssertionError,
        ),
        (
            {
                "moe_expert_capacity_factor": 1.0,
                "moe_router_load_balancing_type": "sinkhorn",
            },
            ValueError,
        ),
        ({"moe_pad_expert_input_to_capacity": True}, ValueError),
        ({"cpu_offloading": True, "cpu_offloading_num_layers": 4}, ValueError),
        ({"cpu_offloading": True, "cpu_offloading_num_layers": 1, "pipeline_model_parallel_size": 2}, ValueError),
        ({"cpu_offloading": True, "cpu_offloading_num_layers": 1, "recompute_granularity": "full"}, ValueError),
        ({"recompute_granularity": "bad"}, ValueError),
        ({"recompute_granularity": "full", "recompute_method": "bad"}, ValueError),
        ({"recompute_granularity": "full", "recompute_method": "block"}, ValueError),
        ({"recompute_granularity": "selective", "recompute_num_layers": 1}, ValueError),
        (
            {
                "recompute_granularity": "full",
                "recompute_method": "block",
                "recompute_num_layers": 1,
                "distribute_saved_activations": True,
                "sequence_parallel": True,
            },
            ValueError,
        ),
        ({"recompute_granularity": "selective", "recompute_modules": ["not_supported"]}, AssertionError),
        ({"recompute_granularity": "selective", "recompute_modules": ["moe_act"]}, ValueError),
        ({"recompute_granularity": "selective", "recompute_modules": ["mla_up_proj"]}, ValueError),
        (
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["shared_experts"],
                "moe_shared_expert_intermediate_size": 8,
                "moe_shared_expert_overlap": True,
                "moe_token_dispatcher_type": "alltoall",
            },
            ValueError,
        ),
        ({"moe_layer_recompute": True, "recompute_granularity": "full"}, ValueError),
        ({"fine_grained_activation_offloading": True, "cpu_offloading": True}, AssertionError),
        ({"fine_grained_activation_offloading": True, "offload_modules": []}, AssertionError),
        ({"fine_grained_activation_offloading": True, "offload_modules": ["bad"]}, AssertionError),
        ({"fine_grained_activation_offloading": True, "offload_modules": ["attn_proj"]}, ValueError),
        (
            {
                "num_layers_in_first_pipeline_stage": 1,
                "account_for_embedding_in_pipeline_split": True,
            },
            ValueError,
        ),
        (
            {
                "pipeline_model_parallel_layout": "t|t",
                "account_for_loss_in_pipeline_split": True,
            },
            ValueError,
        ),
        ({"num_layers_in_first_pipeline_stage": 0, "pipeline_model_parallel_size": 2}, ValueError),
        (
            {
                "num_layers_in_first_pipeline_stage": 3,
                "virtual_pipeline_model_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
            },
            ValueError,
        ),
        ({"num_layers_in_last_pipeline_stage": 0, "pipeline_model_parallel_size": 2}, ValueError),
        (
            {
                "num_layers_in_first_pipeline_stage": 1,
                "pipeline_model_parallel_size": 4,
            },
            ValueError,
        ),
        (
            {
                "account_for_embedding_in_pipeline_split": True,
                "pipeline_model_parallel_size": 3,
            },
            ValueError,
        ),
        ({"bias_activation_fusion": True, "activation_func": torch.nn.functional.relu}, ValueError),
        (
            {
                "bias_activation_fusion": True,
                "add_bias_linear": False,
            },
            ValueError,
        ),
        (
            {
                "bias_activation_fusion": True,
                "activation_func": quick_gelu,
                "gated_linear_unit": False,
            },
            ValueError,
        ),
        (
            {
                "bias_activation_fusion": True,
                "activation_func": torch.nn.functional.gelu,
                "glu_linear_offset": 0.5,
            },
            ValueError,
        ),
        ({"bias_activation_fusion": True, "use_te_activation_func": True}, ValueError),
        ({"fused_residual_rmsnorm": True, "normalization": "LayerNorm"}, ValueError),
        ({"use_te_activation_func": True, "activation_func": torch.tanh}, ValueError),
        ({"activation_func_fp8_input_store": True}, ValueError),
    ],
)
def test_transformer_config_validation_rejects_incompatible_options(overrides, expected_error):
    kwargs = {
        "num_layers": 4,
        "hidden_size": 16,
        "num_attention_heads": 4,
    }
    kwargs.update(overrides)

    with pytest.raises(expected_error):
        TransformerConfig(**kwargs)


def test_transformer_config_validation_mutates_supported_legacy_options():
    scaled = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        fp32_residual_connection=True,
        pipeline_dtype=torch.float16,
        apply_query_key_layer_scaling=True,
    )
    assert scaled.pipeline_dtype is torch.float
    assert scaled.attention_softmax_in_fp32 is True

    with pytest.warns(UserWarning, match="moe_ffn_hidden_size is not set"):
        moe = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2, num_moe_experts=2)
    assert moe.moe_ffn_hidden_size == moe.ffn_hidden_size

    capacity = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        moe_expert_capacity_factor=-1.0,
    )
    assert capacity.moe_expert_capacity_factor is None

    with pytest.warns(UserWarning, match="moe_enable_deepep is deprecated"):
        deepep = TransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            moe_enable_deepep=True,
            moe_token_dispatcher_type="flex",
        )
    assert deepep.moe_flex_dispatcher_backend == "deepep"

    with pytest.warns(UserWarning, match="moe-layer-recompute is deprecated"):
        recompute = TransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            moe_layer_recompute=True,
        )
    assert recompute.recompute_granularity == "selective"
    assert "moe" in recompute.recompute_modules


def test_tensor_aware_state_dict_tensor_lifecycle(monkeypatch):
    class _FakeShardedTensor:
        def __init__(self, key, data):
            self.key = key
            self.data = data
            self.init_calls = []

        def init_data(self, device):
            self.init_calls.append(device)
            self.data = torch.empty(2, device=device)

    class _FakeShardedObject:
        def __init__(self, key, data):
            self.key = key
            self.data = data

    monkeypatch.setattr(tensor_aware, "ShardedTensor", _FakeShardedTensor)
    monkeypatch.setattr(tensor_aware, "ShardedObject", _FakeShardedObject)

    tensor_a = torch.tensor([1.0, 2.0])
    tensor_b = torch.tensor([3.0, 4.0])
    sharded_a = _FakeShardedTensor("a", tensor_a)
    sharded_b = _FakeShardedTensor("b", tensor_b)
    state = tensor_aware.MCoreTensorAwareStateDict(
        common={"metadata": {"step": 1}},
        sharded_state_dict={"layer": [sharded_a, {"nested": sharded_b}]},
    )

    assert state.is_hollow is False
    assert state.common_state_dict["metadata"]["step"] == 1
    tensors = list(state.tensors)
    assert tensors[0] is tensor_a
    assert tensors[1] is tensor_b

    popped = state.pop_tensors()
    assert popped[0] is tensor_a
    assert popped[1] is tensor_b
    assert state.is_hollow is True
    assert sharded_a.data is None
    assert sharded_a.orig_device == "cpu"
    assert list(state._sharded_tensors) == [sharded_a, sharded_b]

    state.insert_tensors([tensor_a, tensor_b])
    assert state.is_hollow is False
    assert sharded_a.data is tensor_a
    assert not hasattr(sharded_a, "orig_device")

    state.pop_tensors()
    state.init_tensors()
    assert state.is_hollow is False
    assert sharded_a.init_calls == ["cpu"]
    assert sharded_b.init_calls == ["cpu"]

    old_data = sharded_a.data
    state.copy_tensors_to_cpu(non_blocking=True)
    assert sharded_a.data.device.type == "cpu"
    assert sharded_a.data is not old_data
    state.restore_tensor_device(non_blocking=False)
    assert sharded_a.data.device.type == "cpu"

    tensor_aware.MCoreTensorAwareStateDict._validate_params("atomic")
    tensor_aware.MCoreTensorAwareStateDict._validate_params("fully_parallel")
    with pytest.raises(NotImplementedError, match="Only"):
        tensor_aware.MCoreTensorAwareStateDict._validate_params("other")

    cached_distribution = ({"a": 0}, {"b": 1}, "unused-a", "unused-b")
    assert (
        tensor_aware.MCoreTensorAwareStateDict._get_distribution(
            False, state.sharded_state_dict, None
        )
        == (None, None, None, None)
    )
    assert (
        tensor_aware.MCoreTensorAwareStateDict._get_distribution(
            True, state.sharded_state_dict, None, cached_distribution
        )
        is cached_distribution
    )


def test_pipeline_utils_rank_helpers_streams_and_schedule_node(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_utils, "get_pg_rank", lambda group: group["rank"])
    monkeypatch.setattr(pipeline_utils, "get_pg_size", lambda group: group["size"])
    monkeypatch.setattr(
        pipeline_utils.torch.distributed,
        "get_process_group_ranks",
        lambda group: group["ranks"],
    )

    first_group = {"rank": 0, "size": 4, "ranks": [10, 11, 12, 13]}
    middle_group = {"rank": 2, "size": 4, "ranks": [10, 11, 12, 13]}
    last_group = {"rank": 3, "size": 4, "ranks": [10, 11, 12, 13]}
    assert pipeline_utils.is_pp_first_stage(first_group) is True
    assert pipeline_utils.is_pp_last_stage(first_group) is False
    assert pipeline_utils.is_pp_last_stage(last_group) is True
    assert pipeline_utils.get_pp_first_rank(middle_group) == 10
    assert pipeline_utils.get_pp_last_rank(middle_group) == 13
    assert pipeline_utils.get_pp_next_rank(middle_group) == 13
    assert pipeline_utils.get_pp_prev_rank(middle_group) == 11
    assert pipeline_utils.get_pp_next_rank(last_group) is None
    assert pipeline_utils.get_pp_prev_rank(first_group) is None

    assert pipeline_utils.is_vp_first_stage(None, None) is True
    assert pipeline_utils.is_vp_first_stage(0, 1) is True
    assert pipeline_utils.is_vp_first_stage(0, 4) is True
    assert pipeline_utils.is_vp_first_stage(2, 4) is False
    assert pipeline_utils.is_vp_last_stage(3, 4) is True
    assert pipeline_utils.is_vp_last_stage(1, 4) is False
    with pytest.raises(AssertionError, match="Expected vp_stage"):
        pipeline_utils.is_vp_first_stage(1, None)
    with pytest.raises(AssertionError, match="Expected vp_stage"):
        pipeline_utils.is_vp_last_stage(2, 1)

    noop = pipeline_utils.NoopScheduleNode()
    assert noop.forward("x") == "x"
    assert noop.backward("grad") == "grad"

    monkeypatch.setattr(pipeline_utils, "cur_platform", _FakePlatform(calls))
    pipeline_utils._COMM_STREAM = None
    pipeline_utils.set_streams(comm_stream="comm-a")
    assert pipeline_utils.get_comm_stream() == "comm-a"
    pipeline_utils.set_streams(comm_stream="comm-b")
    assert pipeline_utils.get_comm_stream() == "comm-a"
    assert pipeline_utils.get_comp_stream() == "current-stream"

    monkeypatch.setattr(pipeline_utils, "make_viewless", lambda value: value)
    tensor = torch.tensor([1.0], requires_grad=True)
    grad = torch.tensor([2.0])
    event = _FakeEvent(calls)
    node = pipeline_utils.ScheduleNode(
        forward_func=lambda value: value * 3,
        stream=lambda: "node-stream",
        event=event,
        backward_func=lambda outputs, output_grad: (None,),
        name="unit-node",
    )
    output = node.forward(tensor)
    assert torch.allclose(output, torch.tensor([3.0]))
    assert node.get_output() is output
    node.inputs[0].grad = grad
    assert node.backward(torch.tensor([1.0])) is grad
    assert node.inputs is None
    assert not hasattr(node, "forward_func")


def test_exchange_utils_pure_distribution_and_object_gather(monkeypatch):
    class _Shard:
        def __init__(self, key, shape=(2, 3), dtype=torch.float32, data=None):
            self.key = key
            self.local_shape = shape
            self.dtype = dtype
            self.data = data
            self.init_calls = []

        def init_data(self, device):
            self.init_calls.append(device)
            self.data = torch.empty(self.local_shape, dtype=self.dtype, device="cpu")

    assert exchange_utils._shard_size(_Shard("a", shape=(2, 3), dtype=torch.float32)) == 24
    assert exchange_utils.is_float8tensor(torch.tensor([1.0])) is False

    assignment = exchange_utils.distribute_shards_to_ranks(
        shard_to_ranks={"a": [0, 1], "b": [1], "c": [0, 1], "d": [0]},
        shard_to_size={"a": 4, "b": 8, "c": 2, "d": 1},
        num_ranks=2,
        cross_parallelization_group_loads={"c"},
    )
    assert assignment["b"] == 1
    assert assignment["d"] == 0
    assert assignment["c"] in {0, 1}

    needed = {"need": _Shard("need")}
    unneeded = {"drop": _Shard("drop"), "drop-loaded": _Shard("drop-loaded", data=torch.ones(2))}
    loaded = {}
    monkeypatch.setattr(
        exchange_utils,
        "cur_platform",
        SimpleNamespace(device_name=lambda: "cpu", device=lambda: torch.device("cpu"), synchronize=lambda: None),
    )
    tensor, orig_device = exchange_utils._get_empty_tensor_for_exchange(
        "need", needed, unneeded, loaded
    )
    assert tensor.shape == torch.Size([2, 3])
    assert orig_device == torch.device("cpu")
    assert loaded["need"] is tensor
    tensor, orig_device = exchange_utils._get_empty_tensor_for_exchange(
        "drop", needed, unneeded, loaded
    )
    assert unneeded["drop"].data is None
    assert orig_device is None
    tensor, orig_device = exchange_utils._get_empty_tensor_for_exchange(
        "drop-loaded", needed, unneeded, loaded
    )
    assert tensor.shape == torch.Size([2])
    assert orig_device is None

    def fake_all_gather_object(target, payload, group=None):
        target[:] = [{"a": 1}, {"b": 2}]

    monkeypatch.setattr(exchange_utils.torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(exchange_utils.torch.distributed, "all_gather_object", fake_all_gather_object)
    assert exchange_utils.exchange_loaded_objects_gather_object({"local": 0}) == {"a": 1, "b": 2}

    def duplicate_all_gather_object(target, payload, group=None):
        target[:] = [{"dup": 1}, {"dup": 2}]

    monkeypatch.setattr(exchange_utils.torch.distributed, "all_gather_object", duplicate_all_gather_object)
    with pytest.raises(exchange_utils.CheckpointingException, match="Duplicate shard ids"):
        exchange_utils.exchange_loaded_objects_gather_object({"dup": 1})

    distribution = exchange_utils.ShardDistribution(
        main_rank_for_shard={("a",): 0},
        shards_in_this_group={("a",)},
        shard_to_metadata={("a",): _Shard("a")},
        all_ranks_for_shard={("a",): [0]},
    )
    monkeypatch.setattr(
        exchange_utils,
        "exchange_loaded_tensors_gather_object",
        lambda loaded_tensors, unloaded_shards, shard_distribution, parallelization_group=None: {
            **loaded_tensors,
            "algo": "gather_object",
        },
    )
    monkeypatch.setattr(
        exchange_utils,
        "exchange_loaded_tensors_gather_rounds",
        lambda loaded_tensors, unloaded_shards, shard_distribution, parallelization_group=None: {
            **loaded_tensors,
            "algo": "gather_rounds",
        },
    )
    monkeypatch.setattr(
        exchange_utils,
        "exchange_loaded_tensors_broadcast",
        lambda loaded_tensors, unloaded_shards, shard_distribution, parallelization_group=None: {
            **loaded_tensors,
            "algo": "broadcast",
        },
    )
    assert exchange_utils.exchange_by_distribution({}, {}, distribution, exchange_algo="gather_object") == {
        "algo": "gather_object"
    }
    assert exchange_utils.exchange_by_distribution({}, {}, distribution, exchange_algo="gather_rounds") == {
        "algo": "gather_rounds"
    }
    assert exchange_utils.exchange_by_distribution({}, {}, distribution, exchange_algo="broadcast") == {
        "algo": "broadcast"
    }
    with pytest.raises(AssertionError, match="Expecting distribution"):
        exchange_utils.exchange_by_distribution({}, {}, None)
    with pytest.raises(NotImplementedError, match="Unrecognized"):
        exchange_utils.exchange_by_distribution({}, {}, distribution, exchange_algo="unknown")


def test_dist_checkpointing_dict_utils_nested_operations():
    nested = {
        "keep": 2,
        "drop": 1,
        "nested": [3, {"keep2": 4, "drop2": 5}, []],
    }
    matching, nonmatching = checkpoint_dict_utils.extract_matching_values(
        nested, lambda value: isinstance(value, int) and value % 2 == 0
    )
    assert matching == {"keep": 2, "nested": [{"keep2": 4}]}
    assert nonmatching == {"drop": 1, "nested": [3, {"drop2": 5}, []]}

    matching, nonmatching = checkpoint_dict_utils.extract_matching_values(
        [1, 2, {"x": 4, "y": 5}], lambda value: isinstance(value, int) and value > 1, True
    )
    assert matching == {1: 2, 2: {"x": 4, "y": 5}}
    assert nonmatching == {0: 1}
    with pytest.raises(ValueError, match="Unexpected top-level"):
        checkpoint_dict_utils.extract_matching_values("bad", lambda value: True)

    left = {"a": torch.tensor([1, 2]), "b": [1, 2], "c": {"same": "x"}}
    right = {"a": torch.tensor([1, 3]), "b": [1, 2, 3], "d": 4, "c": {"same": "x"}}
    only_left, only_right, mismatch = checkpoint_dict_utils.diff(left, right)
    assert ("d",) in only_right
    assert ("a",) == mismatch[0][0]
    assert only_left == []
    assert 2 in only_right

    class _Replica:
        def __init__(self, data):
            self.replica_id = 0
            self.data = data

    _, _, replica_mismatch = checkpoint_dict_utils.diff(_Replica(1), _Replica(2))
    assert replica_mismatch[0][0] == (_Replica,)

    mapped = {"a": [1, {"b": 2}]}
    checkpoint_dict_utils.dict_map(lambda value: value * 10, mapped)
    assert mapped == {"a": [10, {"b": 20}]}
    checkpoint_dict_utils.dict_map_with_key(lambda key, value: f"{key}:{value}", mapped)
    assert mapped == {"a": ["0:10", {"b": "b:20"}]}

    inplace = {"a": [1, 2]}
    assert checkpoint_dict_utils.dict_list_map_inplace(lambda value: value + 1, inplace) == {
        "a": [2, 3]
    }
    outplace = checkpoint_dict_utils.dict_list_map_outplace(lambda value: value * 2, inplace)
    assert outplace == {"a": [4, 6]}
    assert inplace == {"a": [2, 3]}

    merged = {"a": {"x": 1}, "b": [{"left": 1}, {"z": 2}]}
    checkpoint_dict_utils.merge(merged, {"a": {"y": 2}, "b": [{"right": 3}, {"w": 4}]})
    assert merged == {
        "a": {"x": 1, "y": 2},
        "b": [{"left": 1, "right": 3}, {"z": 2, "w": 4}],
    }
    with pytest.raises(ValueError, match="different lengths"):
        checkpoint_dict_utils.merge([1], [1, 2])
    with pytest.raises(ValueError, match="Duplicate"):
        checkpoint_dict_utils.merge({"a": 1}, {"a": 2})

    reduced = checkpoint_dict_utils.map_reduce(
        ["apple", "ape", "bear"],
        key_fn=lambda value: value[0],
        value_fn=len,
        reduce_fn=sum,
    )
    assert reduced == {"a": 8, "b": 4}
    assert list(checkpoint_dict_utils.nested_values({"a": [1, {"b": 2}]})) == [1, 2]
    assert [item[1:] for item in checkpoint_dict_utils.nested_items_iter({"a": [1]})] == [(0, 1)]


def test_dist_checkpointing_utils_prefixes_filters_and_logging(monkeypatch, caplog):
    class _Base:
        def __init__(self, key, data=None, flattened_range=None):
            self.key = key
            self.data = data
            self.global_offset = (0,)
            self.global_shape = (1,)
            self.flattened_range = flattened_range

    class _Tensor(_Base):
        pass

    class _Factory(_Base):
        pass

    class _Object(_Base):
        pass

    class _Nonpersistent:
        def __init__(self, data):
            self.data = data

    monkeypatch.setattr(checkpoint_utils, "ShardedBase", _Base)
    monkeypatch.setattr(checkpoint_utils, "ShardedTensor", _Tensor)
    monkeypatch.setattr(checkpoint_utils, "ShardedTensorFactory", _Factory)
    monkeypatch.setattr(checkpoint_utils, "ShardedObject", _Object)
    monkeypatch.setattr(checkpoint_utils, "LocalNonpersistentObject", _Nonpersistent)
    monkeypatch.setattr(checkpoint_state_utils, "ShardedTensor", _Tensor)

    tensor = _Tensor("old.weight", torch.tensor([1.0]))
    factory = _Factory("old.factory")
    obj = _Object("old.object", {"x": 1})
    nonpersistent = _Nonpersistent("local")
    state = {"layer": [tensor, {"factory": factory, "object": obj, "local": nonpersistent}], "plain": 3}

    sharded, rest = checkpoint_utils.extract_sharded_tensors(state)
    assert sharded == {"layer": [tensor]}
    assert rest["plain"] == 3
    sharded_factories, rest = checkpoint_utils.extract_sharded_tensors_and_factories(state)
    assert sharded_factories["layer"][1]["factory"] is factory
    sharded_or_local, _ = checkpoint_utils.extract_sharded_tensors_or_nonpersistent(state)
    assert sharded_or_local["layer"][1]["local"] is nonpersistent
    base_only, common = checkpoint_utils.extract_sharded_base(state)
    assert base_only["layer"][1]["object"] is obj
    assert common["plain"] == 3
    local_only, _ = checkpoint_utils.extract_nonpersistent(state)
    assert local_only["layer"][0]["local"] is nonpersistent

    checkpoint_utils.add_prefix_for_sharding(state, "prefix.")
    assert tensor.key == "prefix.old.weight"
    checkpoint_utils.replace_prefix_for_sharding(state, "prefix.old.", "new.")
    assert tensor.key == "new.weight"
    with pytest.raises(ValueError, match="Expected"):
        checkpoint_utils.replace_prefix_for_sharding({"bad": _Tensor("wrong.weight")}, "new.", "other.")

    checkpoint_utils.apply_prefix_mapping(state, {"new.": "mapped.", "missing.": "unused."})
    assert tensor.key == "mapped.weight"
    assert checkpoint_utils._sharded_tensor_shard_id(tensor) == ("mapped.weight", (0,), None)
    assert checkpoint_utils._sharded_object_id(obj) == (obj.key, (0,), (1,))

    assert list(checkpoint_utils.zip_strict([1, 2], ["a", "b"])) == [(1, "a"), (2, "b")]
    with pytest.raises(AssertionError, match="unequal lengths"):
        list(checkpoint_utils.zip_strict([1], [1, 2]))

    empty = _Tensor("empty", flattened_range=slice(2, 2))
    nonempty = _Tensor("nonempty", flattened_range=slice(2, 3))
    filtered = checkpoint_state_utils.filter_out_empty_flatten_tensor({"a": empty, "b": nonempty})
    assert filtered == {"b": nonempty}
    assert checkpoint_utils._sharded_tensor_shard_id(nonempty) == ("nonempty", (0,), (2, 3))

    assert checkpoint_utils._clean_metadata_for_serialization(None) is None
    metadata = {"dp_cp_group": object(), "keep": 1}
    assert checkpoint_utils._clean_metadata_for_serialization(metadata) == {"keep": 1}

    logger = logging.getLogger("test_dist_checkpointing_utils")
    caplog.set_level(logging.DEBUG, logger="test_dist_checkpointing_utils")
    with checkpoint_utils.logger_stack("outer", logger):
        with checkpoint_utils.logger_stack("inner", logger):
            checkpoint_utils.debug_msg("hello")
    assert "outer.inner hello" in caplog.text


def test_transformer_config_safe_post_init_branches(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.log_single_rank",
        lambda *args, **kwargs: None,
    )

    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4)
    assert config.ffn_hidden_size == 64
    assert config.kv_channels == 4
    assert config.num_query_groups == 4

    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        fp32_residual_connection=True,
        pipeline_dtype=torch.float16,
    )
    assert config.pipeline_dtype == torch.float

    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=False,
    )
    assert config.attention_softmax_in_fp32 is True

    with pytest.raises(ValueError, match="Only one"):
        TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4, fp16=True, bf16=True)

    with pytest.raises(ValueError, match="num_attention_heads"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=3,
            tensor_model_parallel_size=2,
        )

    with pytest.raises(ValueError, match="num_query_groups"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            num_query_groups=3,
            tensor_model_parallel_size=2,
        )

    with pytest.raises(AssertionError, match="linear_attention_freq"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=None,
        )

    with pytest.raises(AssertionError, match="linear_num_value_heads"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=2,
            linear_num_key_heads=3,
            linear_num_value_heads=4,
        )

    with pytest.raises(ValueError, match="Delayed scaling"):
        TransformerConfig(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=4,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.delayed,
            first_last_layers_bf16=True,
        )

    with pytest.raises(ValueError, match="fp8_quantizer_factory"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.custom,
        )

    with pytest.raises(ValueError, match="fp8_param"):
        TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4, fp8_param=True)

    with pytest.raises(ValueError, match="fp4_param"):
        TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4, fp4_param=True)

    with pytest.raises(ValueError, match="fp4 and fp8"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            fp8="e4m3",
            fp4="e2m1",
        )

    with pytest.raises(ValueError, match="fp4_quantizer_factory"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            fp4="e2m1",
            fp4_recipe=Fp4Recipe.custom,
        )

    with pytest.raises(ValueError, match="num_moe_experts"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            expert_model_parallel_size=2,
        )

    with pytest.raises(ValueError, match="non-negative"):
        TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4, num_moe_experts=0)


def test_rerun_quick_stats_data_iterator_and_error_injector(monkeypatch):
    stats = rerun.QuickStats(max_size=3)
    stats.record(0.0)
    stats.record(1.0)
    stats.record(2.0)
    stats.record(3.0)
    stats.record(4.0)
    other = rerun.QuickStats()
    other.record(5.0)
    other.record(0.0)
    stats.combine([other])

    printed = stats.print_stats()
    assert "total/identical samples" in printed
    assert stats.max == 5.0
    stats.reset()
    assert stats.print_stats() == "0 samples, all identical"

    machine = SimpleNamespace(get_mode=lambda: rerun.RerunMode.VALIDATE_RESULTS)
    monkeypatch.setattr(rerun, "get_rerun_state_machine", lambda: machine)
    iterator = rerun.RerunDataIterator(iter(["a", "b", "c"]))
    assert next(iterator) == "a"
    assert next(iterator) == "b"
    assert iterator.saved_microbatches == ["a", "b"]
    iterator.rewind()
    assert next(iterator) == "a"
    saved = iterator.state_dict()
    iterator.advance()
    assert iterator.saved_microbatches == []
    iterator.load_state_dict(saved)
    assert iterator.replaying is True
    assert iterator.replay_pos == 1

    monkeypatch.setattr(rerun.random, "randint", lambda start, stop: 0)
    monkeypatch.setattr(rerun, "safe_get_rank", lambda: 0)
    injector = rerun.RerunErrorInjector(
        error_injection_rate=1,
        error_injection_type=rerun.RerunDiagnostic.TRANSIENT_ERROR,
    )
    assert injector.maybe_inject() is True
    assert injector.maybe_inject() is False
    assert (
        injector.maybe_miscompare(lambda left, right: 0.25, 1.0, 2.0, rerun.RerunState.RERUNNING_IN_PLACE)
        == rerun.COMPARISON_MISMATCH
    )
    assert injector.injected_error_type is None
    assert injector.maybe_miscompare(lambda left, right: 0.25, 1.0, 2.0, rerun.RerunState.RERUNNING_IN_PLACE) == 0.25

    persistent = rerun.RerunErrorInjector(
        error_injection_rate=1,
        error_injection_type=rerun.RerunDiagnostic.PERSISTENT_ERROR,
    )
    persistent.injected_error_type = rerun.RerunDiagnostic.PERSISTENT_ERROR
    assert (
        persistent.maybe_miscompare(lambda left, right: 0.0, 1.0, 1.0, rerun.RerunState.RERUNNING_IN_PLACE)
        == rerun.COMPARISON_MATCH
    )
    assert (
        persistent.maybe_miscompare(lambda left, right: 0.0, 1.0, 2.0, rerun.RerunState.RERUNNING_FROM_CHECKPOINT)
        == rerun.COMPARISON_MISMATCH
    )

    checkpoint = persistent.state_dict()
    reloaded = rerun.RerunErrorInjector()
    reloaded.load_state_dict(checkpoint)
    assert reloaded.error_injection_type == rerun.RerunDiagnostic.PERSISTENT_ERROR


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_rerun_state_machine_validation_paths_and_checkpoint_helpers(monkeypatch, tmp_path):
    restored = []
    monkeypatch.setattr(rerun, "log_single_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(rerun, "safe_get_rank", lambda: 0)
    monkeypatch.setattr(rerun, "cur_platform", SimpleNamespace(current_device=lambda: 0))

    disabled = rerun.RerunStateMachine(mode=rerun.RerunMode.DISABLED)
    assert disabled.should_run_forward_backward(None) is True
    assert disabled.should_run_forward_backward(None) is False
    assert disabled.should_checkpoint_and_exit() == (False, False, 0)
    disabled.validate_result("bad", lambda _: True, "nonfatal", fatal=False)
    with pytest.raises(RuntimeError, match="Unexpected result"):
        disabled.validate_result("bad", lambda _: True, "fatal", fatal=True)

    machine = rerun.RerunStateMachine(
        state_save_func=lambda: {"custom": "state"},
        state_restore_func=restored.append,
        mode=rerun.RerunMode.VALIDATE_RESULTS,
    )
    monkeypatch.setattr(rerun, "get_rerun_state_machine", lambda: machine)
    monkeypatch.setattr(machine, "_save_state", lambda: setattr(machine, "saved_state", {"custom": "state"}))
    monkeypatch.setattr(machine, "_restore_state", lambda: restored.append(machine.saved_state))
    monkeypatch.setattr(machine, "_reduce_any", lambda value: tuple(value) if isinstance(value, list) else value)

    data_iterator = rerun.RerunDataIterator(iter(["microbatch"]))
    assert machine.should_run_forward_backward(data_iterator) is True
    assert next(data_iterator) == "microbatch"
    machine.first_iteration_complete = True
    machine.validate_result(1.0, lambda _: True, "loss spike", fatal=False)
    assert machine.rerun_requested is True

    assert machine.should_run_forward_backward(data_iterator) is True
    assert data_iterator.replaying is True
    machine.validate_result(
        2.0,
        lambda _: False,
        "loss spike",
        comparison_func=lambda initial, result: rerun.COMPARISON_MISMATCH,
        fatal=False,
    )
    assert machine.continue_requested is True
    assert machine.should_run_forward_backward(data_iterator) is False
    assert machine.state == rerun.RerunState.NOT_RUNNING_YET
    assert restored

    assert machine.is_unexpectedly_large(_Scalar(1.0), threshold=10.0, context="loss", num_samples=2) is False
    assert machine.is_unexpectedly_large(_Scalar(2.0), threshold=10.0, context="loss", num_samples=2) is False
    assert machine.is_unexpectedly_large(_Scalar(25.0), threshold=10.0, context="loss", num_samples=2) is True
    assert machine.is_unexpectedly_large(_Scalar(float("nan")), threshold=10.0, context="nan") is False
    assert machine.is_unexpectedly_large(_Scalar(float("inf")), threshold=10.0, context="inf") is False
    assert machine.is_unexpectedly_large(_Scalar(3.0), threshold=10.0, context="loss", resample=True) is False

    assert machine.validate_state_dict(None) is False
    assert machine.validate_state_dict({"state": rerun.RerunState.NOT_RUNNING_YET}) is False
    assert machine.validate_state_dict({"state": rerun.RerunState.INITIAL_RUN}) is True
    machine.load_state_dict(
        {
            "mode": rerun.RerunMode.VALIDATE_RESULTS,
            "state": rerun.RerunState.WILL_RERUN_FROM_CHECKPOINT,
            "current_iteration": 7,
            "sharded": {
                "rerun_requested": True,
                "checkpoint_requested": True,
                "restart_again_requested": False,
                "continue_requested": False,
                "error_injector_checkpoint": rerun.RerunErrorInjector().state_dict(),
                "failed_validation_call": "call",
                "initial_result": 3.0,
                "suspicious_node": "node",
                "suspicious_device": 0,
                "data_iterator_checkpoints": [{"saved_microbatches": [], "replaying": False, "replay_pos": 0}],
                "large_value_counts": {"loss": 2},
                "max_values": {"loss": 2.0},
            },
        }
    )
    assert machine.current_iteration == 7
    assert machine.data_iterator_checkpoints[0]["replay_pos"] == 0

    tracker = tmp_path / "rerun-tracker.log"
    tracker.write_text(
        "\n".join(
            [
                "ts=now node=node device=0 jobID=job-a rank=0 iteration=4 status=RerunValidationStatus.RERUN_DISABLED result=nan message='loss'",
                "ts=now node=node device=0 jobID=job-a rank=0 iteration=4 status=RerunValidationStatus.RERUN_DISABLED result=nan message='loss'",
                "ts=now node=node device=0 jobID=job-b rank=0 iteration=5 status=RerunValidationStatus.FIRST_RERUN_REPRODUCIBLE result=nan message='loss'",
                "ts=now node=node device=0 jobID=job-b rank=1 iteration=5 status=RerunValidationStatus.FIRST_RERUN_REPRODUCIBLE result=nan message='loss'",
            ]
        )
    )
    assert rerun.RerunStateMachine.get_skipped_iterations_from_tracker_file(str(tracker)) == [4]


def test_fine_grained_offload_pool_group_handler_and_manager_paths(monkeypatch):
    calls = []
    fake_platform = _FakePlatform(calls)
    monkeypatch.setattr(offload, "cur_platform", fake_platform)
    monkeypatch.setattr(offload, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(offload.torch.cuda, "nvtx", SimpleNamespace(range_push=lambda name: calls.append(("nvtx-push", name)), range_pop=lambda: calls.append(("nvtx-pop",))))
    monkeypatch.setattr(offload.PipelineOffloadManager, "OFFLOAD_MGR", None)

    pool = offload.GPUTensorPool(device="cpu", pin_memory=False)
    first = pool.allocate((2, 3), torch.float32)
    assert first.shape == (2, 3)
    assert pool.get_pool_status((2, 3), torch.float32)["allocated_count"] == 1
    pool.free(first)
    assert pool.get_pool_status((2, 3), torch.float32)["free_count"] == 1
    second = pool.allocate((2, 3), torch.float32)
    assert second is first
    assert pool.get_pool_status()["global_stats"]["pool_hits"] == 1
    with pytest.raises(ValueError, match="dtype must be specified"):
        pool.get_pool_status((2, 3))
    with pytest.raises(ValueError, match="No pool exists"):
        pool.get_pool_status((4,), torch.float16)
    with pytest.raises(ValueError, match="doesn't belong"):
        pool.free(torch.empty(2, 3))
    pool.reset()
    assert pool.get_pool_status((2, 3), torch.float32)["allocated_count"] == 0
    pool.clear()
    assert pool.get_pool_status()["pools"] == {}

    group = offload.OffloadTensorGroup("attention")
    tensor = torch.arange(4, dtype=torch.float32)
    group.push_tensor(("tag", 0), tensor)
    assert group.pop_tensor(("tag", 0)) is tensor
    group.record_offload_event("d2h")
    group.wait_offload_event(_FakeStream(calls, "consumer"))
    group.record_reload_event("h2d")
    group.wait_reload_event(_FakeStream(calls, "compute"))
    group.update_offload_info(tensor)
    assert group.total_tensor_count == 1
    assert group.total_offload_bytes == tensor.numel() * tensor.element_size()
    assert offload.OffloadTensorGroup("expert_fc1").use_cpu_pool is False
    assert offload.OffloadTensorGroup("moe_act").use_cpu_pool is False

    manager = offload.PipelineOffloadManager.get_instance()
    manager._cpu_tensor_pool = offload.GPUTensorPool(device="cpu", pin_memory=False)
    manager.init_model_chunk_offload_handler(vp_size=2, vp_stage=0, min_offloaded_tensor_size=2)
    first_chunk = manager.cur_forward_chunk()
    manager.init_model_chunk_offload_handler(vp_size=2, vp_stage=1, min_offloaded_tensor_size=2)
    second_chunk = manager.cur_forward_chunk()
    assert first_chunk.vpp_rank == 0
    assert second_chunk.vpp_rank == 1
    assert first_chunk in manager._cached_chunks_forward
    assert second_chunk in manager._cached_chunks_backward

    manager.disable_offload()
    assert manager.do_offload is False
    assert all(not chunk.do_offload for chunk in manager._cached_chunks_forward)
    manager.enable_offload()
    assert manager.do_offload is True
    assert all(chunk.do_offload for chunk in manager._cached_chunks_forward)

    released = torch.ones(4)
    manager.mark_not_offloadable(released)
    assert released.offloading_activation is False

    delayed_calls = []
    manager.push_offload_groups(lambda forced: delayed_calls.append(("first", tuple(forced))), [1])
    manager.push_offload_groups(lambda forced: delayed_calls.append(("second", tuple(forced))), [2])
    manager.flush_delayed_groups()
    assert delayed_calls == [("second", (2,)), ("first", (1,))]
    assert manager._delayed_offload_groups == []

    chunk = offload.ChunkOffloadHandler(min_offloaded_tensor_size=2, cpu_tensor_pool=manager.cpu_tensor_pool)
    assert chunk.is_empty_chunk()
    chunk.on_group_start_forward("mlp")
    assert chunk.find_group_with_name("mlp") is chunk.offload_groups[0]
    pushed_tag = chunk.tensor_push(torch.arange(4, dtype=torch.float32))
    assert pushed_tag == (1, 0)
    assert chunk.tensor_need_offloading_checker(torch.ones(1)) is False
    large = torch.ones(3)
    large.offloading_activation = False
    assert chunk.tensor_need_offloading_checker(large) is False
    assert chunk.tensor_need_offloading_checker(torch.ones(3)) is True
    assert chunk.get_max_deduplicated_groups() == 1
    assert chunk.finish_all_groups("missing") is True

    state = chunk.offload(torch.arange(3, dtype=torch.float32), pin_memory=False, use_cpu_pool=True)
    restored = chunk.reload(state, non_blocking=False)
    assert torch.equal(restored, torch.arange(3, dtype=torch.float32))
    raw_state = chunk.offload(torch.arange(2, dtype=torch.float32), pin_memory=False, use_cpu_pool=False)
    assert raw_state[2] is False
    assert torch.equal(chunk.reload(raw_state, non_blocking=False), torch.arange(2, dtype=torch.float32))

    class _RecordableTensor:
        shape = torch.Size([4])
        dtype = torch.float32

        def __init__(self, payload):
            self.payload = payload
            self.streams = []

        def numel(self):
            return self.payload.numel()

        def element_size(self):
            return self.payload.element_size()

        def is_contiguous(self):
            return True

        def record_stream(self, stream):
            self.streams.append(stream)

    recordable = _RecordableTensor(torch.arange(4, dtype=torch.float32))
    chunk.offload_groups[0]._tensors = {pushed_tag: recordable}
    monkeypatch.setattr(chunk, "offload", lambda tensor, pin_memory=True, use_cpu_pool=True: ("cpu", tensor.payload.clone(), use_cpu_pool))
    monkeypatch.setattr(chunk, "reload", lambda state, non_blocking=None: state[1].clone())
    chunk.bulk_offload([])
    assert chunk._groups_to_offload == []
    assert len(chunk._groups_to_reload) == 1
    chunk.bulk_reload_group()
    assert chunk._groups_to_reload == []
    assert len(chunk._reloading_group) == 1
    recovered = chunk.tensor_pop(pushed_tag)
    assert torch.equal(recovered, torch.arange(4, dtype=torch.float32))
    chunk.reset()
    assert chunk._offloaded_group_index == 0
    assert chunk._groups_to_offload == []

    manager._cached_chunks_backward = [chunk]
    manager._cached_chunks_index_backward = 0
    assert manager.front_backward_chunk("mlp") is chunk
    manager.pop_backward_chunk("mlp")
    assert manager.cur_backward_chunk() is chunk
    with pytest.raises(AssertionError, match="No non-empty chunk"):
        manager.pop_backward_chunk("absent")

    manager._cached_chunks_forward = [chunk]
    manager._cached_chunks_backward = [chunk]
    chunk.is_warmup = True
    chunk.offload_groups[0].offload = True
    chunk.offload_groups[0].total_tensor_count = 2
    chunk.offload_groups[0].total_offload_bytes = 32
    monkeypatch.setattr(chunk, "get_max_deduplicated_groups", lambda: 0)
    monkeypatch.setattr(offload, "print_offload_summary_table", lambda table: calls.append(("summary", dict(table))))
    manager._is_warmup = True
    manager._offload_margin = 0
    manager.post_warmup_callback()
    assert manager.offload_summary_bytes == {"mlp": 32}
    assert manager.offload_summary_total_bytes == 32
    assert ("summary", {"mlp": 32}) in calls

    assert offload.fine_grained_offloading_group_commit((), "none") == ()
    assert offload.fine_grained_offloading_group_commit([], "none") == []
    manager._cur_forward_chunk = None
    passthrough = torch.tensor([1.0])
    assert offload.fine_grained_offloading_group_commit(passthrough, "none") is passthrough
    assert offload.fine_grained_offloading_group_start(passthrough, "none") is passthrough
    offload.fine_grained_offloading_disable_offload()
    assert offload.PipelineOffloadManager.get_instance().do_offload is False
    offload.fine_grained_offloading_enable_offload()
    assert offload.PipelineOffloadManager.get_instance().do_offload is True
    with offload.FineGrainedActivationOffloadingInterface.get_context(False):
        pass
    offload.FineGrainedActivationOffloadingInterface.reset()
    offload.PipelineOffloadManager.OFFLOAD_MGR = None


def test_dist_checkpointing_mapping_tensor_object_factory_and_errors():
    data = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    shard = checkpoint_mapping.ShardedTensor.from_rank_offsets(
        "weight",
        data,
        (1, 1, 2),
        replica_id=(0, 0),
        prepend_axis_num=1,
    )
    assert shard.has_regular_grid is True
    assert shard.global_shape == (1, 4, 3)
    assert shard.global_offset == (0, 2, 0)
    assert shard.global_slice() == (0, slice(2, 4), slice(0, 3))
    assert shard.local_chunk_offset_in_global() == (0, 1, 0)
    assert shard.max_allowed_chunks() == (1, 2, 3)
    assert shard.without_data().data is None

    empty = shard.without_data()
    init_calls = []
    empty.init_data("cpu", init_fn=lambda shape, dtype, device: init_calls.append((shape, dtype, device)) or torch.zeros(shape, dtype=dtype))
    assert torch.equal(empty.data, torch.zeros(2, 3))
    assert init_calls == [((2, 3), torch.float32, "cpu")]
    empty.init_data("cpu", init_fn=lambda *args, **kwargs: pytest.fail("init_data should not reinitialize existing data"))

    narrowed = shard.narrow(dim=1, start=1, length=2)
    assert len(narrowed) == 1
    assert torch.equal(narrowed[0].data, data.narrow(1, 1, 2))
    assert narrowed[0].local_shape == torch.Size([2, 2])

    with pytest.raises(ValueError, match="from_rank_offsets_flat"):
        checkpoint_mapping.ShardedTensor.from_rank_offsets("bad", data, flattened_range=slice(0, 1))
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Invalid rank offsets"):
        checkpoint_mapping.ShardedTensor.from_rank_offsets("bad", data, (0, 2, 2))
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Data dtype"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data.to(torch.float16),
            dtype=torch.float32,
            local_shape=tuple(data.shape),
            global_shape=tuple(data.shape),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
        )
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Data shape"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data,
            dtype=data.dtype,
            local_shape=(3, 2),
            global_shape=(3, 2),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
        )
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Global offset dimensions"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data,
            dtype=data.dtype,
            local_shape=tuple(data.shape),
            global_shape=(2, 3),
            global_offset=(0,),
            axis_fragmentations=(1, 1),
        )
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Local shape together"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data,
            dtype=data.dtype,
            local_shape=tuple(data.shape),
            global_shape=(1, 2, 3, 4),
            global_offset=(0, 0, 0, 0),
            axis_fragmentations=(1, 1, 1, 1),
        )
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Global offset"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data,
            dtype=data.dtype,
            local_shape=tuple(data.shape),
            global_shape=(4, 3),
            global_offset=(1, 0),
            axis_fragmentations=(2, 1),
        )
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Axis shape"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data,
            dtype=data.dtype,
            local_shape=tuple(data.shape),
            global_shape=(5, 3),
            global_offset=(0, 0),
            axis_fragmentations=(2, 1),
        ).max_allowed_chunks()
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="flattened_range"):
        checkpoint_mapping.ShardedTensor(
            key="bad",
            data=data,
            dtype=data.dtype,
            local_shape=tuple(data.shape),
            global_shape=tuple(data.shape),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            flattened_range=slice(0, 1),
        )

    assert checkpoint_mapping.is_main_replica(0) is True
    assert checkpoint_mapping.is_main_replica(1) is False
    assert checkpoint_mapping.is_main_replica((0, 0, 0)) is True
    assert checkpoint_mapping.is_main_replica((0, 1)) is False
    wrapped = checkpoint_mapping.LocalNonpersistentObject({"local": True})
    assert wrapped.unwrap() == {"local": True}

    obj = checkpoint_mapping.ShardedObject("obj", {"payload": 1}, (2, 3), (1, 2), replica_id=(0, 1))
    assert obj.unique_key == "obj/shard_1.2_2.3"
    assert str(obj) == "ShardedObject(key='obj')"
    assert obj.without_data().data is None
    assert checkpoint_mapping.ShardedObject.empty_from_unique_key(obj.unique_key).global_shape == (2, 3)
    assert checkpoint_mapping.ShardedObject.empty_from_unique_key("legacy/shard_0.1.2_3.4").global_shape == (3, 4, -1)
    with pytest.raises(checkpoint_mapping.CheckpointingException, match="Global offset dimensions"):
        checkpoint_mapping.ShardedObject("bad", None, (1,), (1, 2))

    factory = checkpoint_mapping.ShardedTensorFactory(
        "factory",
        torch.tensor([1.0, 2.0]),
        build_fn=lambda key, tensor, replica_id, flattened_range: {
            "left": checkpoint_mapping.ShardedTensor.from_rank_offsets(key + ".left", tensor[:1]),
            "right": checkpoint_mapping.ShardedTensor.from_rank_offsets(key + ".right", tensor[1:]),
        },
        merge_fn=lambda loaded: torch.cat([loaded["left"], loaded["right"]]),
    )
    built = factory.build()
    assert sorted(built) == ["left", "right"]
    assert factory.without_data().data is None
    state = {"outer": factory}
    checkpoint_mapping.apply_factories(state)
    assert isinstance(state["outer"]["left"], checkpoint_mapping.ShardedTensor)

    merge_template = {"outer": checkpoint_mapping.ShardedTensorFactory("merge", torch.tensor([0.0]), lambda *args: {}, lambda loaded: loaded["x"] + loaded["y"])}
    loaded = {"outer": {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}}
    assert torch.equal(checkpoint_mapping.apply_factory_merges(loaded, merge_template)["outer"], torch.tensor([3.0]))
    nested_loaded = {"a": [torch.tensor([1]), torch.tensor([2])]}
    nested_template = {"a": {1: checkpoint_mapping.ShardedTensorFactory("merge", torch.tensor([0]), lambda *args: {}, lambda value: value + 3)}}
    assert torch.equal(checkpoint_mapping.apply_factory_merges(nested_loaded, nested_template)["a"][1], torch.tensor([5]))

    with pytest.raises(ValueError, match="Different dict keys"):
        checkpoint_mapping.apply_factory_merges({}, {"missing": factory})
    with pytest.raises(ValueError, match="different lengths"):
        checkpoint_mapping.apply_factory_merges([1], [factory, factory])
    with pytest.raises(ValueError, match="non-integer"):
        checkpoint_mapping.apply_factory_merges([1], {"bad": factory})
    with pytest.raises(ValueError, match="out of bound"):
        checkpoint_mapping.apply_factory_merges([1], {3: factory})
    with pytest.raises(ValueError, match="Duplicate non-dict"):
        checkpoint_mapping.apply_factory_merges(1, 2)


def test_inference_batch_dimensions_builder_matching_and_sampling(monkeypatch):
    dims = batch_dimensions_utils.InferenceBatchDimensions(8, 2, 3)
    assert str(dims) == "[8]: 2 P + 3 D"
    assert dims.req_count == 5
    assert dims == batch_dimensions_utils.InferenceBatchDimensions(8, 2, 3)
    assert dims != None  # noqa: E711
    assert len({dims, batch_dimensions_utils.InferenceBatchDimensions(8, 2, 3)}) == 1

    decode_graph = batch_dimensions_utils.InferenceBatchDimensions(6, 0, 3)
    decode_real = batch_dimensions_utils.InferenceBatchDimensions(4, 0, 2)
    assert decode_graph.is_applicable_for_batch_dim(decode_real) is True
    assert batch_dimensions_utils.InferenceBatchDimensions(6, 1, 2).is_applicable_for_batch_dim(decode_real) is False
    mixed_graph = batch_dimensions_utils.InferenceBatchDimensions(12, 3, 2)
    mixed_real = batch_dimensions_utils.InferenceBatchDimensions(8, 2, 3)
    assert mixed_graph.is_applicable_for_batch_dim(mixed_real, strict=False) is True
    assert mixed_graph.is_applicable_for_batch_dim(mixed_real, strict=True) is False

    assert batch_dimensions_utils.InferenceBatchDimensions(4, 2, 2).is_valid(4, 8, 0) is True
    assert batch_dimensions_utils.InferenceBatchDimensions(4, 3, 2).is_valid(4, 8, 0) is False
    assert batch_dimensions_utils.InferenceBatchDimensions(4, -1, 1).is_valid(4, 8, 0) is False
    assert batch_dimensions_utils.InferenceBatchDimensions(2, 1, 2).is_valid(4, 8, 0) is False
    assert batch_dimensions_utils.InferenceBatchDimensions(20, 1, 0).is_valid(4, 8, 0) is False

    builder = batch_dimensions_utils.CUDAGraphBatchDimensionBuilder
    assert builder._calculate_cuda_graph_token_counts(tp_size=2, num_cuda_graphs=1, cuda_graph_max_tokens=18) == [18]
    assert builder._calculate_cuda_graph_token_counts(tp_size=2, num_cuda_graphs=3, cuda_graph_max_tokens=24) == [24, 16, 8]
    auto_counts = builder._calculate_cuda_graph_token_counts(tp_size=4, num_cuda_graphs=-1, cuda_graph_max_tokens=20)
    assert auto_counts[0] == 20
    assert auto_counts[-1] == 4
    with pytest.raises(AssertionError, match="num_cuda_graphs"):
        builder._calculate_cuda_graph_token_counts(tp_size=1, num_cuda_graphs=0, cuda_graph_max_tokens=8)
    with pytest.raises(AssertionError, match="cuda_graph_max_tokens"):
        builder._calculate_cuda_graph_token_counts(tp_size=1, num_cuda_graphs=1, cuda_graph_max_tokens=0)

    decode_dims, decode_counts = builder.generate_cuda_graph_batch_dimensions_list(
        tp_size=2,
        num_cuda_graphs=3,
        cuda_graph_max_tokens=8,
        cuda_graph_mixed_prefill_request_count=0,
        max_requests=4,
        max_tokens=8,
        max_sequence_length=16,
        use_cuda_graphs_for_non_decode_steps=False,
        num_speculative_tokens=1,
    )
    assert decode_counts == sorted({dim.token_count for dim in decode_dims}, reverse=True)
    assert decode_counts == [8]
    assert all(dim.prefill_req_count == 0 for dim in decode_dims)

    mixed_dims, mixed_counts = builder.generate_cuda_graph_batch_dimensions_list(
        tp_size=2,
        num_cuda_graphs=2,
        cuda_graph_max_tokens=8,
        cuda_graph_mixed_prefill_request_count=2,
        max_requests=4,
        max_tokens=8,
        max_sequence_length=16,
        use_cuda_graphs_for_non_decode_steps=True,
        num_speculative_tokens=1,
    )
    assert mixed_counts == sorted({dim.token_count for dim in mixed_dims}, reverse=True)
    assert mixed_counts == [8]
    assert any(dim.prefill_req_count > 0 for dim in mixed_dims)
    assert mixed_dims == sorted(
        mixed_dims,
        key=lambda item: (item.token_count - item.decode_req_count * 2, item.decode_req_count),
        reverse=True,
    )
    none_dims, none_counts = builder.generate_cuda_graph_batch_dimensions_list(
        tp_size=2,
        num_cuda_graphs=None,
        cuda_graph_max_tokens=8,
        cuda_graph_mixed_prefill_request_count=2,
        max_requests=4,
        max_tokens=8,
        max_sequence_length=16,
        use_cuda_graphs_for_non_decode_steps=True,
    )
    assert none_dims == []
    assert none_counts is None
    with pytest.raises(AssertionError, match="must equal"):
        builder.generate_cuda_graph_batch_dimensions_list(
            tp_size=2,
            num_cuda_graphs=1,
            cuda_graph_max_tokens=6,
            cuda_graph_mixed_prefill_request_count=0,
            max_requests=4,
            max_tokens=8,
            max_sequence_length=16,
            use_cuda_graphs_for_non_decode_steps=False,
            num_speculative_tokens=1,
        )

    monkeypatch.setattr(batch_dimensions_utils, "get_pg_size", lambda group=None: 1)
    assert (
        batch_dimensions_utils.InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
            mixed_real,
            strict=False,
            decode_only_cuda_graphs=False,
            smallest_non_decode_cuda_graph_size=16,
        )
        is mixed_real
    )
    assert builder.match_graph_config(
        real_batch_dim=decode_real,
        cuda_graph_batch_dimensions_list=decode_dims,
        smallest_non_decode_cuda_graph_size=8,
    ) == min(dim for dim in decode_dims if dim.is_applicable_for_batch_dim(decode_real))
    assert builder.match_graph_config(
        real_batch_dim=batch_dimensions_utils.InferenceBatchDimensions(100, 10, 0),
        cuda_graph_batch_dimensions_list=decode_dims,
        smallest_non_decode_cuda_graph_size=8,
    ) is None
    monkeypatch.setattr(
        batch_dimensions_utils.InferenceBatchDimensions,
        "adjust_batch_dims_for_expert_parallelism",
        staticmethod(lambda *args, **kwargs: None),
    )
    assert builder.match_graph_config(decode_real, decode_dims, 8, decode_only_cuda_graphs=True) is None

    with pytest.warns(DeprecationWarning):
        legacy = SamplingParams(return_prompt_top_n_logprobs=True, top_n_logprobs=2)
    assert legacy.return_prompt_top_n_logprobs is True
    with pytest.raises(AssertionError, match="requires"):
        SamplingParams(return_prompt_top_n_logprobs=True, skip_prompt_log_probs=True)
    params = SamplingParams(top_n_logprobs=4, skip_prompt_log_probs=True)
    assert params.return_prompt_top_n_logprobs is False
    params.add_attributes({"top_n_logprobs": 2, "skip_prompt_log_probs": False, "custom": "value"})
    assert params.return_prompt_top_n_logprobs is True
    assert params.custom == "value"
    restored = SamplingParams.deserialize(params.serialize())
    assert restored.serialize()["custom"] == "value"


def test_spec_rank_and_tensor_parallel_utils_cpu_paths(monkeypatch):
    class ToyModule:
        def __init__(self, value, scale=1, submodules=None):
            self.value = value
            self.scale = scale
            self.submodules = submodules

    def helper_function():
        return "helper"

    assert spec_utils.get_module(ToyModule) is ToyModule
    assert spec_utils.get_module(spec_utils.ModuleSpec(module=ToyModule)) is ToyModule
    assert spec_utils.build_module(helper_function) is helper_function
    assert spec_utils.build_module(spec_utils.ModuleSpec(module=helper_function)) is helper_function
    module = spec_utils.build_module(
        spec_utils.ModuleSpec(module=ToyModule, params={"scale": 2}, submodules={"inner": True}),
        5,
    )
    assert (module.value, module.scale, module.submodules) == (5, 2, {"inner": True})
    module_with_kwargs = spec_utils.build_module(
        spec_utils.ModuleSpec(module=ToyModule, params={"submodules": {"base": True}}),
        5,
        scale=7,
    )
    assert (module_with_kwargs.scale, module_with_kwargs.submodules) == (7, {"base": True})
    spec = spec_utils.ModuleSpec(module=ToyModule, params={"scale": 3})
    assert spec(4).scale == 3
    assert spec_utils.get_submodules(spec_utils.ModuleSpec(module=ToyModule, submodules="sub")) == "sub"
    with pytest.raises(ValueError, match="Could not find"):
        spec_utils.get_submodules(object())

    monkeypatch.setattr(spec_utils, "import_module", lambda module_path: helper_function if module_path == ("pkg", "helper") else ToyModule)
    assert spec_utils.get_module(spec_utils.ModuleSpec(module=("pkg", "helper"))) is helper_function
    assert spec_utils.build_module(spec_utils.ModuleSpec(module=("pkg", "toy"), params={"scale": 9}), 1).scale == 9
    bad_spec = spec_utils.ModuleSpec(module=ToyModule, params={"unknown": True})
    with pytest.raises(TypeError, match="when instantiating ToyModule"):
        spec_utils.build_module(bad_spec, 1)

    monkeypatch.setattr(rank_utils.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "5")
    assert rank_utils.safe_get_rank() == 5
    monkeypatch.setenv("RANK", "not-an-int")
    assert rank_utils.safe_get_rank() == 0
    monkeypatch.setattr(rank_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(rank_utils.torch.distributed, "get_rank", lambda: 3)
    assert rank_utils.safe_get_rank() == 3
    log_calls = []
    fake_logger = SimpleNamespace(log=lambda *args, **kwargs: log_calls.append((args, kwargs)))
    rank_utils.log_single_rank(fake_logger, logging.INFO, "visible", rank=3, marker=True)
    rank_utils.log_single_rank(fake_logger, logging.INFO, "hidden", rank=0)
    assert log_calls == [((logging.INFO, "visible"), {"marker": True})]

    matrix = torch.arange(12).reshape(3, 4)
    chunks = tensor_parallel_utils.split_tensor_along_last_dim(matrix, 2)
    assert [chunk.tolist() for chunk in chunks] == [
        [[0, 1], [4, 5], [8, 9]],
        [[2, 3], [6, 7], [10, 11]],
    ]
    contiguous_chunks = tensor_parallel_utils.split_tensor_along_last_dim(matrix, 2, contiguous_split_chunks=True)
    assert all(chunk.is_contiguous() for chunk in contiguous_chunks)

    class _FakeTensorParallelGroup:
        def __init__(self, rank, size):
            self._rank = rank
            self._size = size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    group = _FakeTensorParallelGroup(rank=1, size=3)
    assert torch.equal(tensor_parallel_utils.split_tensor_into_1d_equal_chunks(torch.arange(12), tp_group=group), torch.arange(4, 8))
    monkeypatch.setattr(tensor_parallel_utils.cur_platform, "current_device", lambda: "cpu")
    new_buffer = tensor_parallel_utils.split_tensor_into_1d_equal_chunks(torch.arange(12), new_buffer=True, tp_group=group)
    assert torch.equal(new_buffer, torch.arange(4, 8))
    assert new_buffer._base is None
    gathered_calls = []

    def fake_all_gather(gathered, tensor, group):
        gathered.copy_(torch.cat([tensor + 10 * i for i in range(group.size())]))
        gathered_calls.append(group)

    monkeypatch.setattr(
        tensor_parallel_utils,
        "dist_all_gather_func",
        fake_all_gather,
    )
    assert torch.equal(tensor_parallel_utils.gather_split_1d_tensor(torch.tensor([1, 2]), tp_group=group), torch.tensor([1, 2, 11, 12, 21, 22]))
    assert gathered_calls == [group]
    assert tensor_parallel_utils.VocabUtility.vocab_range_from_per_partition_vocab_size(8, rank=2, world_size=4) == (16, 24)
    assert tensor_parallel_utils.VocabUtility.vocab_range_from_global_vocab_size(32, rank=3, world_size=4) == (24, 32)


def test_core_utils_cpu_helpers_wrappers_versions_and_memory(monkeypatch):
    core_utils._te_version = None
    core_utils._flashinfer_version = None
    monkeypatch.setattr(core_utils, "HAVE_PACKAGING", True)
    monkeypatch.setattr(core_utils, "version", lambda name: {"transformer-engine": "2.8.0", "flashinfer": "1.2.3"}[name])
    monkeypatch.setitem(sys.modules, "transformer_engine", SimpleNamespace(__version__="0.1.0+te2.9.1"))
    assert str(core_utils.get_te_version()) == "2.9.1"
    assert core_utils.is_te_min_version("2.9.0") is True
    assert core_utils.is_te_min_version("2.9.1", check_equality=False) is False
    core_utils._te_version = None
    monkeypatch.setitem(sys.modules, "transformer_engine", SimpleNamespace())
    assert str(core_utils.get_te_version()) == "2.8.0"

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__="1.2.4"))
    assert str(core_utils.get_flashinfer_version()) == "1.2.4"
    assert core_utils.is_flashinfer_min_version("1.2.0") is True
    core_utils._flashinfer_version = None

    assert core_utils.divide(12, 3) == 4
    with pytest.raises(AssertionError, match="not divisible"):
        core_utils.ensure_divisibility(10, 3)

    class _Group:
        def __init__(self, rank, size):
            self._rank = rank
            self._size = size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    monkeypatch.setattr(core_utils.torch.distributed, "is_initialized", lambda: False)
    assert core_utils.get_pg_size(_Group(1, 8)) == 1
    assert core_utils.get_pg_rank(_Group(1, 8)) == 0
    assert core_utils.get_pg_src_rank(_Group(1, 8)) == 0
    monkeypatch.setattr(core_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(core_utils.torch.distributed, "get_process_group_ranks", lambda group: [5, 6, 7])
    assert core_utils.get_pg_size(_Group(2, 9)) == 9
    assert core_utils.get_pg_rank(_Group(2, 9)) == 2
    assert core_utils.get_pg_size([_Group(3, 4)]) == 4
    assert core_utils.get_pg_rank([_Group(3, 4)]) == 3
    assert core_utils.get_pg_src_rank(_Group(2, 9)) == 5

    class _Inner:
        model_type = "gpt"
        config = {"hidden": 8}
        xattn_needed = True

    wrapped = SimpleNamespace(module=SimpleNamespace(module=_Inner()))
    assert core_utils.get_attr_wrapped_model(wrapped, "model_type") == "gpt"
    assert isinstance(core_utils.get_attr_wrapped_model(wrapped, "config", allow_none=False, return_model_obj=True), _Inner)
    assert core_utils.get_model_type(wrapped) == "gpt"
    assert core_utils.get_model_config(wrapped) == {"hidden": 8}
    assert core_utils.get_model_xattn(wrapped) is True
    assert core_utils.get_model_xattn(SimpleNamespace()) is False
    with pytest.raises(RuntimeError, match="given a list"):
        core_utils.get_attr_wrapped_model([wrapped], "model_type")
    with pytest.raises(RuntimeError, match="couldn't find"):
        core_utils.get_attr_wrapped_model(SimpleNamespace(), "missing")

    monkeypatch.setattr(core_utils.cur_platform, "current_device", lambda: "cpu")
    buffer = core_utils.GlobalMemoryBuffer()
    alloc_context_calls = []

    class _AllocContext:
        def __enter__(self):
            alloc_context_calls.append("enter")

        def __exit__(self, *exc):
            alloc_context_calls.append("exit")

    first = buffer.get_tensor((2, 2), torch.float32, "workspace", mem_alloc_context=_AllocContext)
    first.fill_(3)
    second = buffer.get_tensor((1, 4), torch.float32, "workspace")
    assert torch.equal(second, torch.full((1, 4), 3.0))
    bigger = buffer.get_tensor((3, 3), torch.float32, "workspace")
    assert bigger.shape == (3, 3)
    assert alloc_context_calls == ["enter", "exit"]

    wrapped_tensor = core_utils.WrappedTensor(torch.tensor([1.0]))
    assert torch.equal(wrapped_tensor.unwrap(), torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="already been unwrapped"):
        wrapped_tensor.unwrap()

    base = torch.arange(6.0, requires_grad=True)
    view = base.view(2, 3)
    viewless = core_utils.make_viewless_tensor(view, requires_grad=True, keep_graph=False)
    assert viewless._base is None
    assert torch.equal(viewless, view)
    graph_viewless = core_utils.make_viewless_tensor(view, requires_grad=True, keep_graph=True)
    assert graph_viewless._base is None
    assert torch.equal(graph_viewless, view)
    assert core_utils.make_viewless_tensor(base, requires_grad=True, keep_graph=False) is base
    assert core_utils.assert_viewless_tensor(viewless) is viewless
    assert core_utils.assert_viewless_tensor([viewless]) == [viewless]
    assert core_utils.assert_viewless_tensor("not-a-tensor") == "not-a-tensor"
    with pytest.raises(AssertionError, match="Ensure tensor._base"):
        core_utils.assert_viewless_tensor(view, extra_msg="extra")
    replacement = torch.ones_like(viewless)
    core_utils.safely_set_viewless_tensor_data(viewless, replacement)
    assert torch.equal(viewless, replacement)
    with pytest.raises(AssertionError, match="FYI"):
        core_utils.safely_set_viewless_tensor_data(view, replacement)

    init_target = torch.empty(4)
    core_utils.init_method_normal(0.0)(init_target)
    assert torch.equal(init_target, torch.zeros(4))
    scaled_target = torch.empty(4)
    core_utils.scaled_init_method_normal(0.0, num_layers=2)(scaled_target)
    assert torch.equal(scaled_target, torch.zeros(4))
    mup_target = torch.empty(4)
    core_utils.mup_scaled_init_method_normal(0.0, num_layers=2, width_mult=4)(mup_target)
    assert torch.equal(mup_target, torch.zeros(4))

    log_calls = []
    logger = SimpleNamespace(log=lambda *args, **kwargs: log_calls.append((args, kwargs)))
    monkeypatch.setattr(core_utils.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(core_utils.parallel_state, "get_data_parallel_rank", lambda with_context_parallel=True: 0)
    core_utils.log_on_each_pipeline_stage(logger, logging.INFO, "stage")
    assert log_calls == [((logging.INFO, "stage"), {})]
    log_calls.clear()
    monkeypatch.setattr(core_utils.parallel_state, "get_data_parallel_rank", lambda with_context_parallel=True: 1)
    core_utils.log_on_each_pipeline_stage(logger, logging.INFO, "hidden")
    assert log_calls == []
    with pytest.raises(ValueError, match="must be provided"):
        core_utils.log_on_each_pipeline_stage(logger, logging.INFO, "bad", tp_group=_Group(0, 1))


def test_core_utils_optional_package_version_helpers(monkeypatch):
    monkeypatch.setattr(core_utils, "HAVE_PACKAGING", True)
    monkeypatch.setattr(core_utils, "_fa_version", None)
    monkeypatch.setattr(core_utils, "_mamba_ssm_version", None)
    monkeypatch.setattr(core_utils, "_causal_conv1d_version", None)

    fallback_versions = {
        "flash-attn": "2.5.0",
        "mamba_ssm": "1.1.0",
        "causal_conv1d": "1.2.0",
    }
    monkeypatch.setattr(core_utils, "version", lambda name: fallback_versions[name])

    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(__version__="2.6.1"))
    assert str(core_utils.get_fa_version()) == "2.6.1"
    assert core_utils.get_fa_version() is core_utils.get_fa_version()
    assert core_utils.is_fa_min_version("2.6.0") is True
    assert core_utils.is_fa_min_version("2.6.1", check_equality=False) is False

    monkeypatch.setattr(core_utils, "_fa_version", None)
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace())
    assert str(core_utils.get_fa_version()) == "2.5.0"

    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace(__version__="1.2.3"))
    assert str(core_utils.get_mamba_version()) == "1.2.3"
    assert core_utils.is_mamba_min_version("1.2.0") is True
    assert core_utils.is_mamba_min_version("1.2.3", check_equality=False) is False

    monkeypatch.setattr(core_utils, "_mamba_ssm_version", None)
    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace())
    assert str(core_utils.get_mamba_version()) == "1.1.0"

    monkeypatch.setitem(sys.modules, "causal_conv1d", SimpleNamespace(__version__="1.3.0"))
    assert str(core_utils.get_causal_conv1d_version()) == "1.3.0"
    assert core_utils.is_causal_conv1d_min_version("1.2.9") is True
    assert core_utils.is_causal_conv1d_min_version("1.3.0", check_equality=False) is False

    monkeypatch.setattr(core_utils, "_causal_conv1d_version", None)
    monkeypatch.setitem(sys.modules, "causal_conv1d", SimpleNamespace())
    assert str(core_utils.get_causal_conv1d_version()) == "1.2.0"

    monkeypatch.setattr(core_utils, "HAVE_PACKAGING", False)
    for helper in (
        core_utils.is_torch_min_version,
        core_utils.get_fa_version,
        core_utils.is_fa_min_version,
        core_utils.get_mamba_version,
        core_utils.is_mamba_min_version,
        core_utils.get_causal_conv1d_version,
        core_utils.is_causal_conv1d_min_version,
        core_utils.get_flashinfer_version,
        core_utils.is_flashinfer_min_version,
    ):
        with pytest.raises(ImportError, match="packaging is not installed"):
            helper("1.0") if helper.__name__.startswith("is_") else helper()


def test_tensor_aware_state_dict_cpu_lifecycle_and_conversion(monkeypatch):
    tensor_a = checkpoint_mapping.ShardedTensor.from_rank_offsets(
        "a",
        torch.tensor([1.0, 2.0]),
    )
    tensor_b = checkpoint_mapping.ShardedTensor.from_rank_offsets(
        "b",
        torch.tensor([3.0]),
    )
    sharded_object = checkpoint_mapping.ShardedObject("obj", {"payload": True}, (1,), (0,))
    tasd = tensor_aware.MCoreTensorAwareStateDict(
        common={"meta": {"step": 7}},
        sharded_state_dict={"layer": {"a": tensor_a, "b": [tensor_b], "obj": sharded_object}},
    )

    assert tasd.is_hollow is False
    assert [tensor.tolist() for tensor in tasd.tensors] == [[1.0, 2.0], [3.0]]
    assert tasd.common_state_dict == {"meta": {"step": 7}}
    popped = tasd.pop_tensors()
    assert [tensor.tolist() for tensor in popped] == [[1.0, 2.0], [3.0]]
    assert tasd.is_hollow is True
    assert tensor_a.data is None
    assert hasattr(tensor_a, "orig_device")
    with pytest.raises(AssertionError):
        list(tasd.tensors)

    tasd.insert_tensors([torch.tensor([5.0, 6.0]), torch.tensor([7.0])])
    assert tasd.is_hollow is False
    assert torch.equal(tensor_a.data, torch.tensor([5.0, 6.0]))
    assert not hasattr(tensor_a, "orig_device")
    with pytest.raises(AssertionError):
        tasd.insert_tensors([torch.tensor([1.0])])

    popped_again = tasd.pop_tensors()
    assert len(popped_again) == 2
    tasd.init_tensors()
    assert tasd.is_hollow is False
    assert tensor_a.data.shape == (2,)
    assert tensor_b.data.shape == (1,)

    original_a = tensor_a.data
    tasd.copy_tensors_to_cpu(non_blocking=False)
    assert tensor_a.data.device.type == "cpu"
    assert tensor_a.data is not original_a
    tensor_a.orig_device = "cpu"
    tasd.restore_tensor_device(non_blocking=False)
    assert not hasattr(tensor_a, "orig_device")

    sharded_template = {
        "layer": {
            "a": checkpoint_mapping.ShardedTensor.from_rank_offsets("a", torch.zeros(2)),
            "b": [checkpoint_mapping.ShardedTensor.from_rank_offsets("b", torch.zeros(1))],
            "obj": checkpoint_mapping.ShardedObject("obj", None, (1,), (0,)),
        }
    }
    monkeypatch.setattr(tensor_aware, "load_preprocess", lambda state: (state, {"local": {"keep": 1}}, {}))
    monkeypatch.setattr(tensor_aware, "parse_strict_flag", lambda strict: strict)
    monkeypatch.setattr(tensor_aware.StrictHandling, "requires_explicit_ckpt_mismatch_check", staticmethod(lambda strict: False))
    monkeypatch.setattr(tensor_aware.StrictHandling, "requires_global_app_metadata", staticmethod(lambda strict: False))
    monkeypatch.setattr(tensor_aware, "determine_global_metadata", lambda state: pytest.fail("global metadata should not be needed"))
    monkeypatch.setattr(
        tensor_aware,
        "validate_integrity_and_strict_load",
        lambda sharded_part, strict, validate_access_integrity, local_metadata, global_metadata, ckpt_sharded_metadata: (
            sharded_part,
            ["missing"],
            ["unexpected"],
        ),
    )
    recreated, missing, unexpected = tasd.to_state_dict(
        sharded_template,
        algo="atomic",
        validate_access_integrity=False,
        strict=object(),
        return_mismatch_keys=True,
    )
    assert recreated["meta"] == {"step": 7}
    assert recreated["local"] == {"keep": 1}
    assert torch.equal(recreated["layer"]["a"], tensor_a.data)
    assert torch.equal(recreated["layer"]["b"][0], tensor_b.data)
    assert recreated["layer"]["obj"] == {"payload": True}
    assert missing == ["missing"]
    assert unexpected == ["unexpected"]

    monkeypatch.setattr(tensor_aware, "HAVE_NVRX", False)
    with pytest.raises(ImportError, match="nvidia_resiliency_ext"):
        tensor_aware.MCoreTensorAwareStateDict.from_state_dict({})
    with pytest.raises(NotImplementedError, match="atomic"):
        tensor_aware.MCoreTensorAwareStateDict._validate_params("unsupported")


def test_dynamic_inference_requests_hashes_events_records_and_identity_ops(monkeypatch):
    nvtx_calls = []
    monkeypatch.setattr(
        inference_request_module.torch.cuda,
        "nvtx",
        SimpleNamespace(
            range_push=lambda name: nvtx_calls.append(("push", name)),
            range_pop=lambda: nvtx_calls.append(("pop",)),
        ),
    )
    monkeypatch.setattr(inference_request_module.torch.distributed, "is_initialized", lambda: False)
    inference_request_module._hash_powers = None

    def _as_msgpack_round_trip(value):
        if isinstance(value, tuple):
            return [_as_msgpack_round_trip(item) for item in value]
        if isinstance(value, list):
            return [_as_msgpack_round_trip(item) for item in value]
        if isinstance(value, dict):
            return {key: _as_msgpack_round_trip(item) for key, item in value.items()}
        return value

    prompt_tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    assert inference_request_module.compute_block_hashes_batched(prompt_tokens, block_size=8) == []
    hashes = inference_request_module.compute_block_hashes_batched(prompt_tokens, block_size=2)
    expected_first = (1 * 1 + 2 * inference_request_module.HASH_BASE) % inference_request_module.HASH_PRIME
    expected_second_token_hash = (3 * 1 + 4 * inference_request_module.HASH_BASE) % inference_request_module.HASH_PRIME
    expected_second = ((expected_first + 1) * inference_request_module.HASH_BASE + expected_second_token_hash) % inference_request_module.HASH_PRIME + 1
    assert hashes == [expected_first + 1, expected_second]

    tensor = torch.tensor([[1, 2], [3, 4]])
    assert inference_request_module.serialize_tensor(tensor) == [[1, 2], [3, 4]]
    assert torch.equal(inference_request_module.deserialize_tensor([[5, 6]]), torch.tensor([[5, 6]]))
    assert inference_request_module.unwrap_serialized_tensors({"a": ("tensor", [1]), "b": 2}) == {"a": [1], "b": 2}

    with pytest.warns(UserWarning, match="renamed"):
        legacy = inference_request_module.InferenceRequest(
            request_id=1,
            prompt="hello",
            inference_parameters=SamplingParams(top_k=3),
            status=Status.ACTIVE_AND_GENERATING_TOKENS,
            generated_tokens=torch.tensor([7, 8]),
        )
    serialized = legacy.serialize()
    assert serialized["status"] == "ACTIVE_AND_GENERATING_TOKENS"
    assert serialized["generated_tokens"][0] == "tensor"
    round_tripped = inference_request_module.InferenceRequest.deserialize(_as_msgpack_round_trip(serialized))
    assert round_tripped.status == Status.ACTIVE_AND_GENERATING_TOKENS
    assert round_tripped.sampling_params.top_k == 3
    assert torch.equal(round_tripped.generated_tokens, torch.tensor([7, 8]))

    event = inference_request_module.DynamicInferenceEvent(
        timestamp=1.25,
        type=DynamicInferenceEventType.GENERATED_TOKEN,
        payload={"token_id": 99},
    )
    assert str(event) == "[1.250] GENERATED_TOKEN, token=99"
    assert inference_request_module.DynamicInferenceEvent.deserialize(event.serialize()).payload == {"token_id": 99}
    assert str(inference_request_module.DynamicInferenceEvent(timestamp=2.0, type=DynamicInferenceEventType.PAUSE)) == "[2.000] PAUSE"
    with pytest.raises(AssertionError):
        inference_request_module.DynamicInferenceEvent(type="bad")
    with pytest.raises(AssertionError):
        inference_request_module.DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN)
    with pytest.raises(AssertionError):
        inference_request_module.DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_TRANSIENT)

    sampling = SamplingParams(num_tokens_to_generate=5, top_k=2)
    request = DynamicInferenceRequest(
        request_id=11,
        prompt="prompt",
        prompt_tokens=torch.tensor([10, 11, 12, 13]),
        sampling_params=sampling,
        status=Status.WAITING_IN_QUEUE,
        block_size_tokens=2,
        enable_prefix_caching=True,
    )
    assert request.remaining_prompt_tokens is request.prompt_tokens
    assert request.remaining_prompt_length == 4
    assert request.precomputed_block_hashes == inference_request_module.compute_block_hashes_batched(torch.tensor([10, 11, 12, 13]), 2)
    assert request.sampling_params is not sampling
    assert "id 11" in str(request)

    with pytest.warns(UserWarning, match="termination_id"):
        tracked = request.tracked_metadata
    assert request.sampling_params.termination_id == -1
    assert tracked[:3] == [1.0, 2, 0.0]
    assert [name for name, _, _ in DynamicInferenceRequest.get_metadata_types()] == [
        "temperature",
        "top_k",
        "top_p",
        "termination_id",
        "return_log_probs",
        "skip_prompt_log_probs",
        "top_n_logprobs",
    ]
    assert request.add_event_add_engine().type == DynamicInferenceEventType.ADD_ENGINE
    assert request.add_event_add_context().type == DynamicInferenceEventType.ADD_CONTEXT
    generated_event = request.add_event_generated_token(
        42,
        blocks_total=8,
        blocks_hashed_total=4,
        blocks_hashed_active=3,
        blocks_ref_count=2,
        pre_fwd_active_token_count=6,
        pre_fwd_step_count=1,
    )
    assert generated_event.payload["token_id"] == 42
    assert generated_event.payload["blocks_total"] == 8
    assert request.add_event_pause().type == DynamicInferenceEventType.PAUSE
    assert request.add_event_evict().type == DynamicInferenceEventType.EVICT
    assert request.add_event_finish().type == DynamicInferenceEventType.FINISH
    assert request.add_event_fail().type == DynamicInferenceEventType.FAIL
    request.status = Status.COMPLETED
    assert request.succeeded() is True
    request.status = Status.FAILED
    assert request.failed() is True

    request.generated_tokens = [14, 15]
    request.generated_text = "ab"
    request.generated_log_probs = [0.1]
    request.generated_top_n_logprobs = [{"a": -0.1}]
    request.tpot = [1]
    request.policy_epoch = [(0, 1)]
    request.kv_cache_epoch = [(2, 3)]
    request.routing_indices = torch.zeros((5, 1, 1), dtype=torch.int64)
    serialized_request = request.serialize()
    assert "event_add_engine" not in serialized_request
    assert len(serialized_request["events"]) == len(request.events)
    request_copy = DynamicInferenceRequest.deserialize(_as_msgpack_round_trip(serialized_request))
    assert request_copy.events[2].payload["token_id"] == 42

    bad_routing = DynamicInferenceRequest(
        request_id=12,
        prompt_tokens=torch.tensor([1, 2]),
        sampling_params=SamplingParams(),
        generated_tokens=[3],
        routing_indices=torch.zeros((5, 1, 1), dtype=torch.int64),
    )
    with pytest.raises(AssertionError, match="routing_indices"):
        bad_routing.serialize()

    record = DynamicInferenceRequestRecord.from_request(request)
    assert record.request_id == 11
    record.checkpoint()
    assert len(record.requests) == 2
    assert torch.equal(record.requests[1].prompt_tokens, torch.tensor([10, 11, 12, 13, 14, 15]))
    assert record.requests[1].sampling_params.num_tokens_to_generate == 3
    assert record.requests[1].event_add_engine is request.event_add_engine
    record.requests[1].generated_tokens = [16]
    record.requests[1].generated_text = "c"
    record.requests[1].generated_log_probs = [0.2]
    record.requests[1].generated_top_n_logprobs = [{"c": -0.2}]
    record.requests[1].tpot = [2]
    record.requests[1].events = [inference_request_module.DynamicInferenceEvent(timestamp=4.0, type=DynamicInferenceEventType.FINISH)]
    record.requests[1].status = Status.COMPLETED
    request.routing_indices = None
    record.latency = 9.5
    merged = record.merge()
    assert merged.generated_tokens == [14, 15, 16]
    assert merged.generated_text == "abc"
    assert merged.generated_length == 3
    assert merged.status == Status.COMPLETED
    assert merged.latency == 9.5
    record_serialized = record.serialize()
    record_round_trip = DynamicInferenceRequestRecord.deserialize(_as_msgpack_round_trip(record_serialized))
    assert record_round_trip.requests[0].request_id == 11

    vlm = inference_request_module.VLMInferenceRequest(
        request_id=21,
        prompt="describe",
        num_img_embeddings_per_tile=2,
        imgs=torch.zeros(1, 3, 4, 4),
        num_tiles=torch.tensor([1]),
        decoder_seq_length=8,
    )
    assert vlm.decoder_seq_length == 8

    identity = IdentityOp()
    payload = {"x": 1}
    assert identity(payload, ignored=True) is payload
    identity_func = IdentityFuncOp()()
    assert identity_func("kept", "dropped") == "kept"

    packed = PackedSeqParams(cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32), total_tokens=7)
    assert torch.equal(packed.seq_idx, torch.tensor([[0, 0, 1, 1, 1, 2, 2]], dtype=torch.int32))
    padded = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 10], dtype=torch.int32),
        total_tokens=5,
    )
    assert torch.equal(
        padded.seq_idx,
        torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32),
    )
    no_tokens = PackedSeqParams(cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32))
    assert no_tokens.seq_idx is None


def test_state_dict_utils_preprocess_filter_factories_and_nonpersistent(monkeypatch):
    factory_source = torch.tensor([1.0, 2.0])

    def build_fn(key, tensor, replica_id, flattened_range):
        return {
            "left": checkpoint_mapping.ShardedTensor.from_rank_offsets(key + ".left", tensor[:1]),
            "right": checkpoint_mapping.ShardedTensor.from_rank_offsets(key + ".right", tensor[1:]),
        }

    factory = checkpoint_mapping.ShardedTensorFactory(
        "factory",
        factory_source,
        build_fn=build_fn,
        merge_fn=lambda loaded: torch.cat([loaded["left"], loaded["right"]]),
    )
    nonpersistent = checkpoint_mapping.LocalNonpersistentObject({"rank-local": 1})
    state = {
        "factory": factory,
        "plain": 3,
        "local": nonpersistent,
        "content_metadata": {"tuple": (1, 2), "set": {"a", "b"}},
    }

    validated = []
    monkeypatch.setattr(checkpoint_state_utils, "determine_global_metadata", lambda sharded_part: ("local", {"global": sorted(sharded_part)}))
    monkeypatch.setattr(
        checkpoint_state_utils,
        "validate_sharding_integrity",
        lambda global_metadata, common_state_dict=None: validated.append((global_metadata, common_state_dict)),
    )
    sharded_part, common = checkpoint_state_utils.save_preprocess(
        state,
        validate_access_integrity=True,
        preprocess_common_before_consistancy_check=lambda common_state: {**common_state, "preprocessed": True},
    )
    assert sorted(sharded_part["factory"]) == ["left", "right"]
    assert common["plain"] == 3
    assert "local" not in common
    assert validated[0][0] == {"global": ["factory"]}
    assert validated[0][1]["preprocessed"] is True

    load_factory = checkpoint_mapping.ShardedTensorFactory(
        "load_factory",
        torch.tensor([5.0, 6.0]),
        build_fn=build_fn,
        merge_fn=lambda loaded: torch.cat([loaded["left"], loaded["right"]]),
    )
    load_state = {
        "layer": [load_factory, checkpoint_mapping.LocalNonpersistentObject("local-value")],
        "common": "kept",
    }
    loaded_sharded, loaded_local, factories = checkpoint_state_utils.load_preprocess(load_state)
    assert isinstance(loaded_sharded["layer"][0]["left"], checkpoint_mapping.ShardedTensor)
    assert loaded_local == {"layer": ["local-value"]}
    assert factories["layer"][0].data is None
    assert load_factory.data is not None
    assert load_state["layer"][1].unwrap() == "local-value"

    empty_flat = checkpoint_mapping.ShardedTensor.from_rank_offsets("empty", torch.empty(0))
    empty_flat.flattened_range = slice(0, 0)
    non_empty_flat = checkpoint_mapping.ShardedTensor.from_rank_offsets("nonempty", torch.ones(1))
    non_empty_flat.flattened_range = slice(0, 1)
    filtered = checkpoint_state_utils.filter_out_empty_flatten_tensor(
        {"keep": non_empty_flat, "drop": empty_flat, "nested": [empty_flat, non_empty_flat]}
    )
    assert "drop" not in filtered
    assert filtered["keep"] is non_empty_flat
    assert filtered["nested"] == [non_empty_flat]


def test_inference_dataset_quantization_and_fp4_utility_paths(monkeypatch, tmp_path):
    counter = inference_utils.Counter(start=3)
    assert next(counter) == 3
    assert next(counter) == 4
    counter.reset()
    assert next(counter) == 0

    class _FakeDispatcher:
        def __init__(self):
            self.drop_and_pad = False
            self.config = SimpleNamespace()
            self.input_splits = [1]
            self.output_splits = [2]
            self.output_splits_tp = [3]
            self.tokens_per_expert = [4]
            self.num_global_tokens_per_local_expert = [5]
            self.reversed_local_input_permutation_mapping = [6]
            self.capacity = 7
            self.cuda_sync_point = "sync"

    class _FakeRouter:
        def __init__(self):
            self.config = SimpleNamespace()

    class _FakeMoELayer:
        def __init__(self):
            self.token_dispatcher = _FakeDispatcher()
            self.router = _FakeRouter()
            self.capture_calls = []

        def children(self):
            return []

        def set_inference_cuda_graphed_iteration(self):
            self.capture_calls.append("set")

        def unset_inference_cuda_graphed_iteration(self):
            self.capture_calls.append("unset")

    class _FakeModel:
        def __init__(self, child):
            self.config = SimpleNamespace(
                moe_pad_expert_input_to_capacity=False,
                moe_expert_capacity_factor=None,
            )
            self._child = child

        def children(self):
            return [self._child]

    moe_layer = _FakeMoELayer()
    model = _FakeModel(moe_layer)
    monkeypatch.setattr(inference_utils, "MoELayer", _FakeMoELayer)
    inference_utils.moe_layer_cache = None
    inference_utils.set_decode_expert_padding(model, set_to=True, capacity_factor=2)
    assert model.config.moe_pad_expert_input_to_capacity is True
    assert model.config.moe_expert_capacity_factor == 2
    assert moe_layer.token_dispatcher.drop_and_pad is True
    assert moe_layer.token_dispatcher.input_splits is None
    assert moe_layer.token_dispatcher.output_splits is None
    assert moe_layer.token_dispatcher.cuda_sync_point == "no_sync"
    assert moe_layer.router.moe_expert_capacity_factor == 2
    assert inference_utils.moe_layer_cache == [moe_layer]
    inference_utils.set_decode_expert_padding(model, set_to=False, capacity_factor=None)
    assert moe_layer.token_dispatcher.drop_and_pad is False
    inference_utils.set_inference_cuda_graphed_iteration_for_ep_inference(model)
    inference_utils.unset_inference_cuda_graphed_iteration_for_ep_inference(model)
    assert moe_layer.capture_calls == ["set", "unset"]

    values = torch.tensor([10, 20, 30, 40])
    inference_utils.tensor_swap(values, torch.tensor([0, 2]), torch.tensor([1, 3]))
    assert torch.equal(values, torch.tensor([20, 10, 40, 30]))

    monkeypatch.setattr(inference_utils, "FLASHINFER_JIT_CACHE_VERSION", "1.0.0")
    inference_utils.check_flashinfer_jit_cache_installed(log_version=True)
    monkeypatch.setattr(inference_utils, "FLASHINFER_JIT_CACHE_VERSION", None)
    monkeypatch.setattr(inference_utils.torch.version, "cuda", "12.9")
    with pytest.raises(RuntimeError, match="cu129"):
        inference_utils.check_flashinfer_jit_cache_installed()
    monkeypatch.setattr(inference_utils.torch.version, "cuda", "13.0")
    with pytest.raises(RuntimeError, match="cu130"):
        inference_utils.check_flashinfer_jit_cache_installed()
    monkeypatch.setattr(inference_utils.torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="flashinfer-jit-cache"):
        inference_utils.check_flashinfer_jit_cache_installed()

    async def _exercise_queue_and_process_helpers():
        queue = inference_utils.asyncio_Queue()
        queue.put_nowait("item")
        assert await queue.get() == "item"
        queue.shutdown()
        with pytest.raises(inference_utils.asyncio_QueueShutDown):
            await queue.get()
        with pytest.raises(inference_utils.asyncio_QueueShutDown):
            queue.put_nowait("late")
        if hasattr(inference_utils, "_SHUTDOWN_SENTINEL"):
            fresh_queue = inference_utils.asyncio_Queue()
            with pytest.raises(ValueError, match="reserved"):
                fresh_queue.put_nowait(inference_utils._SHUTDOWN_SENTINEL)

        attempts = {"count": 0}

        def wait_call(timeout):
            attempts["count"] += 1
            return attempts["count"] >= 2

        alive_process = SimpleNamespace(is_alive=lambda: True, name="alive", pid=123)
        await inference_utils.await_process_call(wait_call, alive_process, timeout=0.001)
        dead_process = SimpleNamespace(is_alive=lambda: False, name="dead", pid=456)
        with pytest.raises(RuntimeError, match="dead"):
            await inference_utils.await_process_call(lambda timeout: False, dead_process, timeout=0.001)

    import asyncio

    asyncio.run(_exercise_queue_and_process_helpers())

    assert dataset_utils.Split.train.value == 0
    assert dataset_utils.normalize([1.0, 1.0, 2.0]) == [0.25, 0.25, 0.5]
    assert dataset_utils.get_blend_from_list(None) is None
    assert dataset_utils.get_blend_from_list(["a", "b", "c"]) == (["a", "b", "c"], None)
    assert dataset_utils.get_blend_from_list(["0.25", " a ", "0.75", " b "]) == (["a", "b"], [0.25, 0.75])
    assert dataset_utils.get_blend_from_list(["a", "b"]) == (["a", "b"], None)

    matcher = GlobMatcher(pattern="*layers.3*fc1*", config_key="fp4")
    matching_context = MatchContext(module_path="decoder.layers.3.mlp.fc1", layer_number=3)
    assert matcher.match(matching_context) == "fp4"
    assert matcher.match(MatchContext(module_path="decoder.layers.4.mlp.fc2", layer_number=4)) is None
    assert "GlobMatcher" in repr(matcher)

    recipe = RecipeConfig.from_config_dict(
        {
            "configs": {"fp4": {"bits": 4}, "fallback": {"bits": 8}},
            "matchers": {
                "disabled": {
                    "enabled": False,
                    "type": "glob",
                    "pattern": "*disabled*",
                    "config": "fallback",
                },
                "fc1": {
                    "enabled": True,
                    "type": "glob",
                    "pattern": "*fc1*",
                    "config": "fp4",
                },
            },
        }
    )
    quant_config = recipe.match(matching_context)
    assert quant_config.config == {"bits": 4}
    assert quant_config.match_input == matching_context
    assert quant_config.config_key == "fp4"
    quant_config.config["bits"] = 2
    assert recipe.configs["fp4"] == {"bits": 4}
    assert recipe.match(MatchContext(module_path="decoder.layers.3.mlp.fc2", layer_number=3)) is None
    assert "RecipeConfig" in repr(recipe)
    with pytest.raises(AssertionError, match="type"):
        RecipeConfig.from_config_dict({"matchers": {"bad": {"enabled": True}}})
    with pytest.raises(AssertionError, match="pattern"):
        RecipeConfig.from_config_dict({"matchers": {"bad": {"enabled": True, "type": "glob", "config": "x"}}})
    with pytest.raises(AssertionError, match="config"):
        RecipeConfig.from_config_dict({"matchers": {"bad": {"enabled": True, "type": "glob", "pattern": "*"}}})
    with pytest.raises(NotImplementedError, match="Match type"):
        RecipeConfig.from_config_dict({"matchers": {"bad": {"enabled": True, "type": "regex"}}})

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "\n".join(
            [
                "configs:",
                "  default:",
                "    bits: 8",
                "matchers:",
                "  all:",
                "    enabled: true",
                "    type: glob",
                "    pattern: '*'",
                "    config: default",
            ]
        )
    )
    loaded_recipe = quantization_utils.load_quantization_recipe(str(recipe_path))
    assert loaded_recipe.match(MatchContext("anything.layers.9", 9)).config == {"bits": 8}
    kitchen = quantization_utils.kitchen_quantization_recipe_config(7)
    assert quantization_utils.get_quant_config_or_none("module.layers.12.linear", kitchen).config["recipe_idx"] == 7
    assert quantization_utils.get_quant_config_or_none("module.no_layer.linear", None) is None
    matched_no_layer = quantization_utils.get_quant_config_or_none("module.no_layer.linear", kitchen)
    assert matched_no_layer.match_input.layer_number is None

    assert fp4_utils.is_nvfp4tensor(torch.tensor([1.0])) is False
    assert fp4_utils.get_nvfp4_rowwise_packed_shape(torch.Size([])) == torch.Size([])
    assert fp4_utils.get_nvfp4_rowwise_packed_shape(torch.Size([2, 8])) == torch.Size([2, 4])
    with pytest.raises(AssertionError, match="inner dimension"):
        fp4_utils.get_nvfp4_rowwise_packed_shape(torch.Size([3]))
    with pytest.raises(ValueError, match="NVFP4"):
        fp4_utils.modify_nvfp4_rowwise_storage(torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8))
    monkeypatch.setattr(fp4_utils, "HAVE_TE_FP4_TENSOR_CLASS", True)
    monkeypatch.setattr(fp4_utils, "FP4_TENSOR_CLASS", object)
    fake_fp4 = SimpleNamespace(_rowwise_data=torch.tensor([1, 2], dtype=torch.uint8))
    new_storage = torch.empty(2, dtype=torch.uint8)
    fp4_utils.modify_nvfp4_rowwise_storage(fake_fp4, new_storage)
    assert fake_fp4._rowwise_data is new_storage
    assert torch.equal(new_storage, torch.tensor([1, 2], dtype=torch.uint8))
    fake_missing_storage = SimpleNamespace()
    with pytest.raises(RuntimeError, match="missing rowwise"):
        fp4_utils.modify_nvfp4_rowwise_storage(fake_missing_storage, torch.empty(1, dtype=torch.uint8))
    fake_bad_dtype = SimpleNamespace(_rowwise_data=torch.tensor([1], dtype=torch.int32))
    with pytest.raises(AssertionError, match="uint8"):
        fp4_utils.modify_nvfp4_rowwise_storage(fake_bad_dtype, torch.empty(1, dtype=torch.uint8))
    assert fp4_utils.get_fp4_align_size(Fp4Recipe.nvfp4) == 128
    monkeypatch.setattr(fp4_utils, "is_te_min_version", lambda version: True)
    fake_dequant = SimpleNamespace(dequantize=lambda: torch.tensor([9.0]))
    assert torch.equal(fp4_utils.dequantize_fp4_tensor(fake_dequant), torch.tensor([9.0]))
    monkeypatch.setattr(fp4_utils, "is_te_min_version", lambda version: False)
    with pytest.raises(RuntimeError, match="FP4 dequantization"):
        fp4_utils.dequantize_fp4_tensor(fake_dequant)
    if not fp4_utils.HAVE_TE:
        assert fp4_utils.get_fp4_recipe(SimpleNamespace()) is None
        with fp4_utils.get_fp4_context(SimpleNamespace(), layer_no=0):
            pass


def test_core_utils_deprecated_args_submodules_wgrad_and_context_parallel(monkeypatch):
    @core_utils.deprecate_args("old", message="{name} is gone")
    def _new_api(**kwargs):
        return kwargs

    assert _new_api(new=1) == {"new": 1}
    with pytest.raises(TypeError, match="old is gone"):
        _new_api(old=1)

    sentinel = object()
    with pytest.warns(UserWarning, match="renamed"):
        assert core_utils.deprecate_inference_params(None, sentinel) is sentinel
    assert core_utils.deprecate_inference_params("context", sentinel) == "context"

    grad_output = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
    gathered_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
    prepared_grad, prepared_input = core_utils.prepare_input_tensors_for_wgrad_compute(
        grad_output,
        gathered_input,
    )
    assert prepared_grad.shape == (6, 4)
    assert prepared_input.shape == (6, 4)
    assert prepared_grad.is_contiguous()
    two_d_grad, two_d_input = core_utils.prepare_input_tensors_for_wgrad_compute(
        torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
    )
    assert two_d_grad.shape == (3, 2)
    assert two_d_grad.is_contiguous()
    assert two_d_input.is_contiguous()

    src = torch.tensor([1.0, 2.0])
    dst = torch.empty_like(src)
    core_utils.local_multi_tensor_scale(0, None, [[src], [dst]], scale=3.0)
    assert torch.equal(dst, torch.tensor([3.0, 6.0]))
    assert core_utils.local_multi_tensor_applier(
        lambda chunk_size, noop, tensor_lists, value: (chunk_size, noop, tensor_lists, value),
        "noop",
        [[src]],
        "arg",
    ) == (2048 * 32, "noop", [[src]], "arg")

    parent = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    child = parent[0]
    assert core_utils.is_submodule(child, parent) is True
    assert core_utils.is_submodule(parent, parent) is False
    assert core_utils.is_submodule(parent, parent, strict=False) is True
    assert core_utils.is_submodule(torch.nn.Linear(2, 2), parent) is False

    class _ContextGroup:
        def __init__(self, rank, size):
            self._rank = rank
            self._size = size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    batch = {
        "tokens": torch.arange(16).reshape(1, 8, 2),
        "labels": torch.arange(16, 32).reshape(1, 8, 2),
        "attention_mask": torch.arange(64).reshape(1, 1, 8, 8),
        "optional": None,
    }
    sliced = core_utils.get_batch_on_this_cp_rank(batch, cp_group=_ContextGroup(rank=1, size=2))
    assert torch.equal(sliced["tokens"], torch.cat([torch.arange(4, 8).reshape(1, 2, 2), torch.arange(8, 12).reshape(1, 2, 2)], dim=1))
    assert torch.equal(sliced["labels"], torch.cat([torch.arange(20, 24).reshape(1, 2, 2), torch.arange(24, 28).reshape(1, 2, 2)], dim=1))
    assert sliced["attention_mask"].shape == (1, 1, 4, 8)
    assert sliced["optional"] is None

    unchanged = {"tokens": torch.arange(4).reshape(1, 2, 2)}
    assert core_utils.get_batch_on_this_cp_rank(unchanged, cp_group=_ContextGroup(rank=0, size=1)) is unchanged

    value_a = core_utils._ValueWithRank(1.234, 2, "ms")
    value_b = core_utils._ValueWithRank(2.0, 3, "ms")
    assert value_a < value_b
    assert value_b > value_a
    assert value_a() == (1.234, 2, "ms")
    assert str(value_a) == "1.23ms/2"
    detector = core_utils.StragglerDetector()
    detector._off = True
    assert detector.elapsed() == (0, 0, 0, 0, 0, 0)
    assert detector.enabled is False
    detector.world = 8
    detector.rank = 4
    assert detector.world_size == 8
    assert detector.my_rank == 4
    detector.null_method()
    with detector:
        pass
    assert detector(bdata=True) is detector
