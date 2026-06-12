# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.pipeline_parallel.utils as pp_utils
import megatron.core.transformer.mlp as mlp_module
import megatron.core.transformer.module as module_lib
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.mlp import MLP, MLPSubmodules, apply_swiglu_sharded_factory
from megatron.core.transformer.transformer_config import TransformerConfig


def _cfg(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        ffn_hidden_size=8,
        add_bias_linear=True,
        bias_activation_fusion=False,
        gated_linear_unit=False,
        activation_func=F.gelu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _Linear:
    def __init__(self, output_size, bias=True, stride=1):
        self.output_size = output_size
        self.bias = bias
        self.stride = stride
        self.backward_calls = 0
        self.sharded_calls = []

    def __call__(self, hidden_states):
        base = hidden_states.mean(dim=-1, keepdim=True).expand(*hidden_states.shape[:-1], self.output_size)
        bias = torch.ones(self.output_size, dtype=hidden_states.dtype) if self.bias else None
        return base.contiguous(), bias

    def backward_dw(self):
        self.backward_calls += 1

    def sharded_state_dict(self, prefix, sharded_offsets=(), metadata=None):
        self.sharded_calls.append((prefix, sharded_offsets, metadata))
        return {
            f"{prefix}weight": SimpleNamespace(
                key=f"{prefix}weight",
                data=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                local_shape=(4, 2),
                global_offset=(0, 0),
                axis_fragmentations=(1, 1),
                replica_id=(0, 0, 0),
                flattened_range=None,
            )
        }


class _LinearBuilder:
    def __init__(self):
        self.calls = []

    def __call__(self, input_size, output_size, **kwargs):
        self.calls.append((input_size, output_size, kwargs))
        return _Linear(output_size, bias=kwargs.get("bias", True), stride=kwargs.get("stride", 1))


class _ActivationBuilder:
    def __call__(self, *, config):
        return lambda tensor: tensor + 3


def _submodules():
    return MLPSubmodules(
        linear_fc1=_LinearBuilder(),
        linear_fc2=_LinearBuilder(),
        activation_func=_ActivationBuilder(),
    )


def test_mlp_forward_plain_gated_te_and_per_token_scale_paths(monkeypatch):
    monkeypatch.setattr(mlp_module, "get_tensor_model_parallel_group_if_none", lambda group=None, **kwargs: group)
    monkeypatch.setattr(mlp_module, "nvtx_range_push", lambda **kwargs: None)
    monkeypatch.setattr(mlp_module, "nvtx_range_pop", lambda **kwargs: None)
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    plain = MLP(_cfg(), _submodules())
    out, bias = plain(hidden)
    assert out.shape == hidden.shape
    assert bias.shape == (4,)

    gated = MLP(
        _cfg(
            gated_linear_unit=True,
            activation_func=F.silu,
            activation_func_clamp_value=1.5,
            glu_linear_offset=0.25,
        ),
        _submodules(),
    )
    out, bias = gated(hidden, per_token_scale=torch.ones(3, 2))
    assert out.shape == hidden.shape
    assert bias is None
    assert gated.linear_fc1.stride == 2

    kitchen = MLP(
        _cfg(gated_linear_unit=True, activation_func=F.silu, use_kitchen=True),
        _submodules(),
    )
    assert kitchen.linear_fc1.stride == 1

    te_activation = MLP(_cfg(use_te_activation_func=True), _submodules())
    out, bias = te_activation(hidden, per_token_scale=torch.ones(3, 2))
    assert out.shape == hidden.shape
    assert bias is None

    with pytest.warns(DeprecationWarning):
        fallback = MLP(_cfg(), _submodules(), ffn_hidden_size=None)
    assert fallback.linear_fc1.output_size == 8
    with pytest.raises(ValueError, match="MoE MLP requires"):
        MLP(_cfg(), _submodules(), is_expert=True, ffn_hidden_size=None)

    latent = MLP(
        _cfg(moe_latent_size=2, ffn_hidden_size=6),
        _submodules(),
        is_expert=True,
        ffn_hidden_size=6,
    )
    assert latent.linear_fc1.output_size == 6
    assert latent.linear_fc2.output_size == 2
    latent.backward_dw()
    assert latent.linear_fc1.backward_calls == 1
    assert latent.linear_fc2.backward_calls == 1


def test_mlp_bias_activation_fusion_paths(monkeypatch):
    monkeypatch.setattr(mlp_module, "get_tensor_model_parallel_group_if_none", lambda group=None, **kwargs: group)
    monkeypatch.setattr(mlp_module, "nvtx_range_push", lambda **kwargs: None)
    monkeypatch.setattr(mlp_module, "nvtx_range_pop", lambda **kwargs: None)
    monkeypatch.setattr(mlp_module, "bias_gelu_impl", lambda tensor, bias: tensor + bias)
    monkeypatch.setattr(mlp_module, "bias_geglu_impl", lambda tensor, bias: torch.chunk(tensor + bias, 2, dim=-1)[0])
    monkeypatch.setattr(mlp_module, "bias_swiglu_impl", lambda tensor, bias, *args: torch.chunk(tensor + bias, 2, dim=-1)[0])
    monkeypatch.setattr(
        mlp_module,
        "weighted_bias_swiglu_impl",
        lambda tensor, bias, scale, *args: torch.chunk(tensor + bias, 2, dim=-1)[0] * scale,
    )
    monkeypatch.setattr(
        mlp_module,
        "weighted_bias_quick_geglu_impl",
        lambda tensor, bias, scale, *args: torch.chunk(tensor + bias, 2, dim=-1)[0] * scale,
    )
    hidden = torch.ones(2, 2, 4)

    gelu = MLP(_cfg(bias_activation_fusion=True, activation_func=F.gelu), _submodules())
    assert gelu(hidden)[0].shape == hidden.shape

    geglu = MLP(
        _cfg(bias_activation_fusion=True, activation_func=F.gelu, gated_linear_unit=True),
        _submodules(),
    )
    assert geglu(hidden)[0].shape == hidden.shape

    swiglu = MLP(
        _cfg(bias_activation_fusion=True, activation_func=F.silu, gated_linear_unit=True),
        _submodules(),
    )
    assert swiglu(hidden, per_token_scale=torch.ones(2, 2))[0].shape == hidden.shape

    quick = MLP(
        _cfg(
            bias_activation_fusion=True,
            activation_func=mlp_module.quick_gelu,
            gated_linear_unit=True,
        ),
        _submodules(),
    )
    assert quick(hidden, per_token_scale=torch.ones(2, 2))[0].shape == hidden.shape

    unsupported = MLP(
        _cfg(bias_activation_fusion=True, activation_func=torch.relu, gated_linear_unit=True),
        _submodules(),
    )
    with pytest.raises(ValueError, match="Only support fusion"):
        unsupported(hidden, per_token_scale=torch.ones(2, 2))
    with pytest.raises(ValueError, match="Only support fusion"):
        unsupported(hidden)


def test_mlp_sharded_state_dict_and_swiglu_factory_paths(monkeypatch):
    monkeypatch.setattr(mlp_module, "get_tensor_model_parallel_group_if_none", lambda group=None, **kwargs: group)
    captured = []
    monkeypatch.setattr(
        mlp_module.ShardedTensor,
        "from_rank_offsets",
        staticmethod(lambda key, tensor, *offsets, **kwargs: captured.append((key, tensor.clone(), offsets, kwargs)) or SimpleNamespace(key=key, tensor=tensor)),
    )
    mlp = MLP(_cfg(gated_linear_unit=True, activation_func=F.silu), _submodules())
    state = mlp.sharded_state_dict(prefix="mlp.", metadata={})
    assert "mlp.linear_fc1.weight" in state
    factory = state["mlp.linear_fc1.weight"]
    built = factory.build()
    assert len(built) == 2
    assert captured[0][0] == "mlp.linear_fc1.weight"
    assert captured[1][0] == "mlp.linear_fc1.weight"
    assert torch.equal(factory.merge_fn([torch.ones(1, 2), torch.zeros(1, 2)]), torch.tensor([[1.0, 1.0], [0.0, 0.0]]))

    singleton = apply_swiglu_sharded_factory(
        SimpleNamespace(
            key="single",
            data=torch.arange(8, dtype=torch.float32).reshape(4, 2),
            local_shape=(4, 2),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=(0, 0, 0),
            flattened_range=None,
        ),
        (),
        singleton_local_shards=True,
    )
    singleton.build()
    assert captured[-2][0] == "single_w"
    assert captured[-1][0] == "single_v"


class _Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.is_first_microbatch = False
        self.symmetric_ar_type = "two_shot"

    def forward(self, x):
        return x + self.weight


class _ToyMegatron(module_lib.GraphableMegatronModule):
    def __init__(self, config, vp_stage=None):
        super().__init__(config, vp_stage=vp_stage)
        self.child = _Child()

    def create_mcore_cudagraph_manager(self, config):
        self.cudagraph_manager = lambda module, args, kwargs: ("local", args, kwargs)

    def forward(self, hidden_states, **kwargs):
        return self.child(hidden_states)


def test_megatron_and_graphable_module_cuda_graph_cpu_paths(monkeypatch):
    monkeypatch.setattr(module_lib.cur_platform, "current_device", lambda: "cpu")
    cfg = _cfg(fp8="e4m3", cuda_graph_impl="transformer_engine")
    module = _ToyMegatron(cfg)
    module.set_is_first_microbatch()
    assert module.child.is_first_microbatch is True
    module.set_symmetric_ar("one_shot")
    assert module.child._symmetric_ar_cache == "one_shot"
    with pytest.raises(AssertionError):
        module.set_symmetric_ar("bad")

    static_inputs = module.get_layer_static_inputs(seq_length=8, micro_batch_size=2)
    assert static_inputs["hidden_states"].shape == (8, 2, 4)

    hooks = []
    module.setup_manual_hooks(lambda: lambda child: hooks.append(child))
    assert len(module.cuda_graph_manual_hooks) == 1
    args, kwargs = module._get_te_cuda_graph_replay_args(hidden_states=torch.ones(1))
    assert args[0].shape == (1,)
    assert kwargs["is_first_microbatch"] is True
    with pytest.raises(AssertionError, match="hidden_states"):
        module._get_te_cuda_graph_replay_args(torch.ones(1), hidden_states=torch.ones(1))

    calls = []
    module.cuda_graphs = [lambda hidden_states, **kwargs: calls.append((hidden_states, kwargs)) or hidden_states + 2]
    module.current_microbatch = 0
    out = module._te_cuda_graph_replay(torch.ones(1))
    assert torch.equal(out, torch.tensor([3.0]))
    assert hooks == [module.child]
    with pytest.raises(AssertionError, match="Tensor inputs"):
        module._te_cuda_graph_replay("bad")

    module.cuda_graphs = [SimpleNamespace(backward_dw=lambda: calls.append("dw"))]
    module._te_cuda_graph_backward_dw_graph(0)
    assert "dw" in calls
    module.cuda_graphs = [SimpleNamespace()]
    assert module._te_cuda_graph_backward_dw_graph(0) is None

    local_cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=[CudaGraphScope.attn])
    local = _ToyMegatron(local_cfg)
    assert local._should_call_local_cudagraph() is True
    assert local(torch.ones(1))[0] == "local"


def test_float16_module_conversion_forward_and_state_paths(monkeypatch):
    monkeypatch.setattr(pp_utils, "is_pp_first_stage", lambda group=None: True)
    monkeypatch.setattr(pp_utils, "is_pp_last_stage", lambda group=None: True)
    monkeypatch.setattr(pp_utils, "is_vp_first_stage", lambda vp_stage, vp_size: True)
    monkeypatch.setattr(pp_utils, "is_vp_last_stage", lambda vp_stage, vp_size: True)
    monkeypatch.setattr(module_lib.parallel_state, "get_pipeline_model_parallel_group", lambda: "pp")

    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
            self.vp_stage = 0

        def set_input_tensor(self, tensor):
            self.input_tensor = tensor
            return "set"

        def forward(self, x, extra=None):
            assert x.dtype in (torch.float16, torch.bfloat16)
            if extra is not None:
                assert extra.dtype == x.dtype
            return x + 1, {"extra": extra}

        def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
            return {"saved": prefix}

        def sharded_state_dict(self, prefix="", *args, **kwargs):
            return {"sharded": prefix}

    wrapped = _Wrapped()
    fp16_module = module_lib.Float16Module(
        _cfg(fp16=True, virtual_pipeline_model_parallel_size=2),
        wrapped,
    )
    assert fp16_module.set_input_tensor(torch.ones(1)) == "set"
    out, meta = fp16_module(torch.ones(2, dtype=torch.float32), extra=torch.ones(2))
    assert out.dtype == torch.float32
    assert meta["extra"].dtype == torch.float16
    out, _ = fp16_module(torch.ones(2), fp32_output=False)
    assert out.dtype == torch.float16
    assert fp16_module.state_dict_for_save_checkpoint(prefix="p") == {"saved": "p"}
    assert fp16_module.sharded_state_dict(prefix="s") == {"sharded": "s"}
    fp16_module.load_state_dict(fp16_module.state_dict())

    bf16_module = module_lib.Float16Module(_cfg(bf16=True), _Wrapped())
    out, _ = bf16_module(torch.ones(2))
    assert out.dtype == torch.float32
    with pytest.raises(Exception, match="Either config.fp16 or config.bf16"):
        module_lib.Float16Module(_cfg(), _Wrapped())

    val = (torch.ones(1, dtype=torch.float32), [torch.ones(1, dtype=torch.float32)])
    converted = module_lib.fp32_to_float16(val, lambda tensor: tensor.half())
    assert converted[0].dtype == torch.float16
    assert converted[1][0].dtype == torch.float16
    restored = module_lib.float16_to_fp32(converted)
    assert restored[0].dtype == torch.float32
    assert module_lib.param_is_not_shared(torch.nn.Parameter(torch.ones(1))) is True
    shared = torch.nn.Parameter(torch.ones(1))
    shared.shared = True
    assert module_lib.param_is_not_shared(shared) is False
