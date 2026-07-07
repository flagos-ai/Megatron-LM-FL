# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import module as transformer_module
from megatron.core.transformer.module import (
    Float16Module,
    GraphableMegatronModule,
    MegatronModule,
    conversion_helper,
    float16_to_fp32,
    fp32_to_float16,
    param_is_not_shared,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


class DummyModule(MegatronModule):
    # def __init__(self, config: TransformerConfig, share_embeddings_and_output_weights=True):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.linear = torch.nn.modules.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


class TestMegatronModule:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.megatron_module = DummyModule(config=transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_megatron_module(self):
        megatron_module = self.megatron_module
        assert megatron_module
        assert megatron_module.config.hidden_size == 12
        assert megatron_module.config.ffn_hidden_size == 48
        assert megatron_module.linear.weight.dtype == torch.float32

        x = torch.ones((2, 2)).cuda()
        assert megatron_module(x).dtype == torch.float32

        # TODO: test bad configs actually fail
        # failed_module = megatron_module
        # failed_module.fp16 = True
        # failed_module.bf16 = True


class TestFloat16Module:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.megatron_module = DummyModule(config=self.transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fp16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.fp16 = True
        fp16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert fp16_module
        assert fp16_module.config.hidden_size == 12
        assert fp16_module.config.ffn_hidden_size == 48
        assert fp16_module.module.linear.weight.dtype == torch.float16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to fp16 then outputs are converted to fp32
        assert fp16_module(x).dtype == torch.float32

    pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason='bfloat16 is not supported on this device',
    )

    def test_bf16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.bf16 = True
        bf16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert bf16_module
        assert bf16_module.config.hidden_size == 12
        assert bf16_module.config.ffn_hidden_size == 48
        assert bf16_module.module.linear.weight.dtype == torch.bfloat16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to bf16 then outputs are converted to fp32
        assert bf16_module(x).dtype == torch.float32


def _cpu_transformer_config(**overrides):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_megatron_module_cpu_cache_and_conversion_paths(monkeypatch):
    config = _cpu_transformer_config(fp8="hybrid")
    module = MegatronModule(config)

    child_with_flag = torch.nn.Module()
    child_with_flag.is_first_microbatch = False
    child_without_flag = torch.nn.Module()
    module.add_module("flagged", child_with_flag)
    module.add_module("plain", child_without_flag)

    module.set_is_first_microbatch()
    assert child_with_flag.is_first_microbatch is True
    assert module.modules_with_is_first_microbatch == [child_with_flag]

    child_with_ar = torch.nn.Module()
    child_with_ar.symmetric_ar_type = "two_shot"
    nested = torch.nn.Sequential(child_with_ar)
    module.add_module("nested", nested)
    module.set_symmetric_ar("one_shot")
    assert child_with_ar._symmetric_ar_cache == "one_shot"
    module.set_symmetric_ar(None)
    assert child_with_ar._symmetric_ar_cache is None
    with pytest.raises(AssertionError):
        module.set_symmetric_ar("bad")

    shared = torch.nn.Parameter(torch.ones(1))
    shared.shared = True
    assert param_is_not_shared(shared) is False
    assert param_is_not_shared(torch.nn.Parameter(torch.ones(1))) is True

    nested_values = (torch.tensor([1.0]), [torch.tensor([2.0]), "keep"])
    converted = conversion_helper(nested_values, lambda value: value + 1 if torch.is_tensor(value) else value)
    assert torch.equal(converted[0], torch.tensor([2.0]))
    assert torch.equal(converted[1][0], torch.tensor([3.0]))
    half_values = fp32_to_float16(nested_values, lambda value: value.half())
    assert half_values[0].dtype == torch.float16
    assert float16_to_fp32(half_values)[0].dtype == torch.float32


def test_graphable_megatron_module_cpu_cuda_graph_paths(monkeypatch):
    monkeypatch.setattr(
        transformer_module,
        "cur_platform",
        SimpleNamespace(current_device=lambda: "cpu"),
    )

    class _Graphable(GraphableMegatronModule):
        def __init__(self, config):
            super().__init__(config)
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, hidden_states, scale=None, is_first_microbatch=False):
            output = hidden_states + self.weight
            if scale is not None:
                output = output * scale
            if is_first_microbatch:
                output = output + 1
            return output

    config = _cpu_transformer_config(
        cuda_graph_impl="transformer_engine",
        cuda_graph_scope=[],
        context_parallel_size=2,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
    )
    graphable = _Graphable(config)
    graphable.train()
    static_inputs = graphable.get_layer_static_inputs(seq_length=8, micro_batch_size=3)
    assert static_inputs["hidden_states"].shape == torch.Size([2, 3, 8])
    assert static_inputs["hidden_states"].requires_grad is True

    hooks = []
    graphable.setup_manual_hooks(lambda: lambda module: hooks.append(module))
    assert graphable.cuda_graph_manual_hooks

    class _Graph:
        def __init__(self):
            self.backward_calls = 0

        def __call__(self, *args, **kwargs):
            return args[0] + kwargs["scale"]

        def backward_dw(self):
            self.backward_calls += 1

    graph = _Graph()
    graphable.cuda_graphs = [graph]
    graphable.current_microbatch = 0
    replayed = graphable._te_cuda_graph_replay(torch.tensor([2.0]), scale=torch.tensor([3.0]))
    assert torch.equal(replayed, torch.tensor([5.0]))
    assert hooks == [graphable]
    graphable._te_cuda_graph_backward_dw_graph(0)
    assert graph.backward_calls == 1
    graphable.cuda_graphs = [SimpleNamespace()]
    graphable._te_cuda_graph_backward_dw_graph(0)

    args, kwargs = graphable._get_te_cuda_graph_replay_args(
        hidden_states=torch.tensor([1.0]), scale=torch.tensor([2.0])
    )
    assert torch.equal(args[0], torch.tensor([1.0]))
    assert kwargs["is_first_microbatch"] is True
    with pytest.raises(AssertionError, match="hidden_states should only"):
        graphable._get_te_cuda_graph_replay_args(torch.tensor([1.0]), hidden_states=torch.tensor([1.0]))
    with pytest.raises(AssertionError, match="CUDA graph accepts only Tensor"):
        graphable._te_cuda_graph_replay("not-a-tensor")


def test_float16_module_cpu_forward_and_state_paths(monkeypatch):
    monkeypatch.setattr(
        transformer_module.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: "pp",
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.utils.is_pp_first_stage",
        lambda group: True,
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.utils.is_pp_last_stage",
        lambda group: True,
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.utils.is_vp_first_stage",
        lambda vp_stage, vp_size: True,
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.utils.is_vp_last_stage",
        lambda vp_stage, vp_size: True,
    )

    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.inputs = []

        def set_input_tensor(self, value):
            self.inputs.append(value)
            return "set"

        def forward(self, value, extra=None):
            assert value.dtype == torch.float16
            if extra is not None:
                assert extra.dtype == torch.float16
            return value + (extra if extra is not None else 1)

        def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
            return {"saved": prefix, "keep": keep_vars}

        def sharded_state_dict(self, prefix="", *args, **kwargs):
            return {"sharded": prefix}

    config = _cpu_transformer_config(fp16=True)
    wrapped = _Wrapped()
    fp16_module = Float16Module(config, wrapped)
    assert fp16_module.set_input_tensor("input") == "set"
    assert wrapped.inputs == ["input"]
    output = fp16_module(torch.ones(2), extra=torch.ones(2), fp32_output=True)
    assert output.dtype == torch.float32
    half_output = fp16_module(torch.ones(2), fp32_output=False)
    assert half_output.dtype == torch.float16
    assert "linear.weight" in fp16_module.state_dict()
    assert fp16_module.state_dict_for_save_checkpoint(prefix="p.", keep_vars=True) == {
        "saved": "p.",
        "keep": True,
    }
    assert fp16_module.sharded_state_dict(prefix="s.") == {"sharded": "s."}
    fp16_module.load_state_dict(wrapped.state_dict())
