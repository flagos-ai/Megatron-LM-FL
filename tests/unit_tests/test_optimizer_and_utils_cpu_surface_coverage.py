# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import megatron.core.optimizer as optimizer_pkg
import megatron.core.utils as core_utils
from megatron.core.optimizer import OptimizerConfig


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.bias_only = torch.nn.Parameter(torch.ones(4))
        self.expert = torch.nn.Parameter(torch.ones(2, 2))
        self.expert.allreduce = False
        self.engram = torch.nn.Parameter(torch.ones(2, 2))
        self.engram.is_engram_embedding = True
        self.embed = torch.nn.Parameter(torch.ones(3, 3))
        self.embed.is_embedding_or_output_parameter = True
        self.buffers = ["dense-buffer"]
        self.expert_parallel_buffers = ["expert-buffer"]
        self.engram_embedding_buffers = ["engram-buffer"]
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=False)


class _Group:
    def __init__(self, size=2, rank=1):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _patch_optimizer_distributed(monkeypatch):
    monkeypatch.setattr(optimizer_pkg.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        optimizer_pkg.torch.distributed,
        "all_gather_object",
        lambda gathered, local: gathered.__setitem__(0, local),
    )
    monkeypatch.setattr(
        optimizer_pkg.parallel_state,
        "get_tensor_model_parallel_group",
        lambda: _Group(size=1, rank=0),
    )


def test_optimizer_param_group_overrides_and_buffers_cpu_paths(monkeypatch):
    _patch_optimizer_distributed(monkeypatch)
    model = _Model()
    config = OptimizerConfig(lr=0.1, min_lr=0.01, decoupled_lr=0.2, decoupled_min_lr=0.02)

    standard = optimizer_pkg.get_standard_config_overrides(config)
    assert any(override.get("wd_mult") == 0.0 for override in standard.values())
    assert any(override.get("max_lr") == 0.2 for override in standard.values())

    standard_qk = optimizer_pkg.get_standard_config_overrides(
        OptimizerConfig(apply_wd_to_qk_layernorm=True)
    )
    assert len(standard_qk) == 1

    groups = optimizer_pkg._get_param_groups([model], config, None)
    assert groups
    assert any(group["is_expert_parallel"] for group in groups)
    assert any(group["is_engram_parallel"] for group in groups)
    assert any(group["default_config"] is False for group in groups)

    dense, dense_buffers = optimizer_pkg._get_param_groups_and_buffers(
        [model],
        model_chunk_offset=3,
        config=config,
        config_overrides={},
        filter_fn=lambda group: not group["is_expert_parallel"] and not group["is_engram_parallel"],
        buffer_name="buffers",
    )
    assert dense
    assert dense_buffers == {3: ["dense-buffer"]}

    expert, expert_buffers = optimizer_pkg._get_param_groups_and_buffers(
        [model],
        model_chunk_offset=0,
        config=config,
        config_overrides={},
        filter_fn=lambda group: group["is_expert_parallel"],
        buffer_name="expert_parallel_buffers",
    )
    assert expert
    assert expert_buffers == {0: ["expert-buffer"]}

    with pytest.raises(AssertionError, match="'params' should not be"):
        optimizer_pkg._get_param_groups(
            [model],
            config,
            {optimizer_pkg.ParamKey(name="linear.weight"): {"params": ()}},
        )


def test_optimizer_mup_override_predicates_cover_adam_sgd_muon_paths(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        optimizer_pkg,
        "log_single_rank",
        lambda logger, level, message, *args, **kwargs: warnings.append(message),
    )

    matrix = torch.nn.Parameter(torch.ones(2, 2))
    vector = torch.nn.Parameter(torch.ones(2))
    embedding = torch.nn.Parameter(torch.ones(2, 2))
    embedding.is_embedding_or_output_parameter = True

    adam_config = OptimizerConfig(
        lr=0.12,
        min_lr=0.03,
        adam_eps=1.0e-6,
        decoupled_lr=0.2,
        decoupled_min_lr=0.02,
    )
    adam_overrides = optimizer_pkg.get_mup_config_overrides(adam_config, 3.0, "adam")
    assert warnings
    assert len(adam_overrides) == 2
    matches = [key.matches(matrix, "decoder.layers.0.mlp.weight") for key in adam_overrides]
    assert all(matches)
    assert not any(key.matches(vector, "decoder.layers.0.bias") for key in adam_overrides)
    assert not any(key.matches(embedding, "word_embeddings.weight") for key in adam_overrides)

    sgd_overrides = optimizer_pkg.get_mup_config_overrides(
        OptimizerConfig(optimizer="sgd", lr=0.1, min_lr=0.01), 4.0, "sgd"
    )
    assert len(sgd_overrides) == 1
    sgd_key = next(iter(sgd_overrides))
    assert sgd_key.matches(vector, "layernorm.weight")
    assert not sgd_key.matches(matrix, "dense.weight")

    muon_matrix = torch.nn.Parameter(torch.ones(2, 2))
    muon_overrides = optimizer_pkg.get_mup_config_overrides(
        OptimizerConfig(lr=0.1, min_lr=0.01, muon_scale_mode="spectral"), 2.0, "muon"
    )
    assert len(muon_overrides) == 1
    muon_key = next(iter(muon_overrides))
    assert not muon_key.matches(muon_matrix, "dense.weight")
    assert optimizer_pkg.get_mup_config_overrides(OptimizerConfig(lr=0.1), 1.0, "adam") == {}


def test_optimizer_factory_wrapper_selection_cpu_paths(monkeypatch):
    _patch_optimizer_distributed(monkeypatch)
    model = _Model()
    captured = []

    class _BaseOptimizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.param_groups = kwargs["params"]
            self.state = {param: {} for group in self.param_groups for param in group["params"]}

    class _Wrapper:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.tp_group = None
            captured.append((self.__class__.__name__, args, kwargs))

    class _FP32(_Wrapper):
        pass

    class _FP16(_Wrapper):
        pass

    class _Dist(_Wrapper):
        pass

    class _Hybrid:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups
            self.kwargs = kwargs

    monkeypatch.setattr(optimizer_pkg, "USING_PYTORCH_OPTIMIZER", False)
    monkeypatch.setattr(optimizer_pkg, "Adam", _BaseOptimizer)
    monkeypatch.setattr(optimizer_pkg, "SGD", _BaseOptimizer)
    monkeypatch.setattr(optimizer_pkg, "HybridDeviceOptimizer", _Hybrid)
    monkeypatch.setattr(optimizer_pkg, "FP32Optimizer", _FP32)
    monkeypatch.setattr(optimizer_pkg, "Float16OptimizerWithFloat16Params", _FP16)
    monkeypatch.setattr(optimizer_pkg, "DistributedOptimizer", _Dist)
    monkeypatch.setattr(optimizer_pkg, "is_te_min_version", lambda version: True)

    param_groups = optimizer_pkg._get_param_groups([model], OptimizerConfig(lr=0.1), {})
    opt = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="adam", lr=0.1, use_precision_aware_optimizer=True),
        [model],
        param_groups,
        model_parallel_group="mp",
    )
    assert isinstance(opt, _FP32)
    assert opt.tp_group.size() == 1

    opt = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="sgd", lr=0.1, bf16=True),
        [model],
        param_groups,
        model_parallel_group="mp",
    )
    assert isinstance(opt, _FP16)

    opt = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(optimizer="sgd", lr=0.1, use_distributed_optimizer=True),
        [model],
        param_groups,
        per_model_buffers={0: ["buffer"]},
        model_parallel_group="mp",
        data_parallel_group="dp",
        data_parallel_group_gloo="gloo",
        data_parallel_group_idx=0,
        intra_dist_opt_group="intra",
    )
    assert isinstance(opt, _Dist)
    assert getattr(opt, "grad_stats_parallel_group") == "intra"

    offload = optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
        OptimizerConfig(
            optimizer="adam",
            lr=0.1,
            optimizer_cpu_offload=True,
            decoupled_weight_decay=True,
            use_torch_optimizer_for_cpu_offload=True,
        ),
        [model],
        param_groups,
        model_parallel_group="mp",
    )
    assert isinstance(offload.args[0], _Hybrid)

    if not optimizer_pkg.HAVE_EO_V02:
        with pytest.raises(ImportError, match="Lion optimizer"):
            optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
                OptimizerConfig(optimizer="lion", lr=0.1),
                [model],
                param_groups,
            )
    with pytest.raises(Exception, match="not supported"):
        optimizer_pkg._get_megatron_optimizer_based_on_param_groups(
            OptimizerConfig(optimizer="unknown", lr=0.1),
            [model],
            param_groups,
        )


def test_optimizer_get_megatron_optimizer_split_chaining_and_dump_paths(monkeypatch, tmp_path):
    _patch_optimizer_distributed(monkeypatch)
    model = _Model()
    config = OptimizerConfig(lr=0.1, overlap_param_gather_with_optimizer_step=True)
    groups = {
        "dp_cp_group": _Group(size=2, rank=1),
        "intra_dp_cp_group": _Group(size=1, rank=0),
        "intra_expt_dp_group": _Group(size=1, rank=0),
        "mp_group": _Group(size=1, rank=0),
        "expt_tp_pp_group": _Group(size=1, rank=0),
        "intra_dp_cp_group_gloo": "dp-gloo",
        "intra_expt_dp_group_gloo": "expert-gloo",
        "intra_dist_opt_group": "intra",
        "inter_dist_opt_group": _Group(size=2, rank=1),
        "engram_dp_group": _Group(size=1, rank=0),
        "engram_mp_group": _Group(size=1, rank=0),
        "engram_dp_group_gloo": "engram-gloo",
    }
    built = []

    monkeypatch.setattr(
        optimizer_pkg.ProcessGroupCollection,
        "setup_process_groups_for_optimizer",
        staticmethod(lambda pg_collection, model_chunks, use_gloo_process_groups: groups),
    )
    monkeypatch.setattr(optimizer_pkg, "get_global_unique_param_name", lambda models, param: f"id-{id(param)}")
    monkeypatch.setattr(
        optimizer_pkg.torch.distributed.checkpoint,
        "save",
        lambda state_dict, checkpoint_id: built.append(("dump", checkpoint_id, len(state_dict))),
    )

    class _FakeOpt:
        def __init__(self, label):
            self.label = label

    def _fake_get(**kwargs):
        label = (
            len(kwargs["param_groups"]),
            kwargs["data_parallel_group_idx"],
            kwargs["data_parallel_group_gloo"],
        )
        built.append(label)
        return _FakeOpt(label)

    monkeypatch.setattr(optimizer_pkg, "_get_megatron_optimizer_based_on_param_groups", _fake_get)
    monkeypatch.setattr(
        optimizer_pkg,
        "ChainedOptimizer",
        lambda optimizers: SimpleNamespace(optimizers=optimizers),
    )

    result = optimizer_pkg.get_megatron_optimizer(
        config,
        [model, _Model()],
        config_overrides={},
        dump_param_to_param_group_map=str(tmp_path / "map"),
    )
    assert len(result.optimizers) >= 3
    assert any(item[0] == "dump" for item in built if isinstance(item, tuple))
    assert model.overlap_param_gather_with_optimizer_step is True

    with pytest.raises(ValueError, match="should not be overriden"):
        optimizer_pkg.check_config_overrides_consistency(
            OptimizerConfig(optimizer="adam"),
            {optimizer_pkg.ParamKey(name="*"): {"optimizer": "sgd"}},
        )
    assert optimizer_pkg.check_config_overrides_consistency(OptimizerConfig(), None) is True


def test_core_utils_versions_wrappers_and_tensor_helpers(monkeypatch):
    fake_te = ModuleType("transformer_engine")
    fake_te.__version__ = "0.1.0+te2.9.0"
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_te)
    monkeypatch.setattr(core_utils, "_te_version", None)
    assert str(core_utils.get_te_version()).startswith("2.9")
    assert core_utils.is_te_min_version("2.0")
    assert not core_utils.is_te_min_version("9.0")

    fake_fa = ModuleType("flash_attn")
    fake_fa.__version__ = "2.6.1"
    monkeypatch.setitem(sys.modules, "flash_attn", fake_fa)
    monkeypatch.setattr(core_utils, "_fa_version", None)
    assert core_utils.is_fa_min_version("2.0")

    fake_mamba = ModuleType("mamba_ssm")
    fake_mamba.__version__ = "1.2.3"
    monkeypatch.setitem(sys.modules, "mamba_ssm", fake_mamba)
    monkeypatch.setattr(core_utils, "_mamba_ssm_version", None)
    assert core_utils.is_mamba_min_version("1.0")

    fake_conv = ModuleType("causal_conv1d")
    fake_conv.__version__ = "1.4.0"
    monkeypatch.setitem(sys.modules, "causal_conv1d", fake_conv)
    monkeypatch.setattr(core_utils, "_causal_conv1d_version", None)
    assert core_utils.is_causal_conv1d_min_version("1.0")

    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setattr(core_utils, "_flashinfer_version", None)
    assert core_utils.get_flashinfer_version() is None
    assert core_utils.is_flashinfer_min_version("1.0") is False

    assert core_utils.divide(8, 2) == 4
    with pytest.raises(AssertionError):
        core_utils.ensure_divisibility(7, 3)

    wrapped = SimpleNamespace(module=SimpleNamespace(config="cfg", model_type="type"))
    assert core_utils.get_attr_wrapped_model(wrapped, "config") == "cfg"
    assert core_utils.get_attr_wrapped_model(wrapped, "config", return_model_obj=True).config == "cfg"
    assert core_utils.get_model_type(wrapped) == "type"
    assert core_utils.get_model_config(wrapped) == "cfg"
    assert core_utils.get_model_xattn(SimpleNamespace()) is False
    with pytest.raises(RuntimeError, match="list"):
        core_utils.get_attr_wrapped_model([wrapped], "config")
    with pytest.raises(RuntimeError, match="couldn't find"):
        core_utils.get_attr_wrapped_model(SimpleNamespace(), "missing")

    base = torch.arange(4.0)
    view = base[:2]
    viewless = core_utils.make_viewless_tensor(view, requires_grad=True, keep_graph=False)
    assert viewless._base is None
    viewless_graph = core_utils.make_viewless_tensor(view, requires_grad=True, keep_graph=True)
    assert viewless_graph._base is None
    assert core_utils.make_viewless_tensor(base, requires_grad=False, keep_graph=False) is base
    assert core_utils.assert_viewless_tensor([base]) == [base]
    with pytest.raises(AssertionError, match="memory leak"):
        core_utils.assert_viewless_tensor(view)
    new_data = torch.ones_like(base)
    core_utils.safely_set_viewless_tensor_data(base, new_data)
    assert torch.equal(base, new_data)

    wrapped_tensor = core_utils.WrappedTensor(torch.tensor([1]))
    assert wrapped_tensor.unwrap().item() == 1
    with pytest.raises(RuntimeError, match="already been unwrapped"):
        wrapped_tensor.unwrap()


def test_core_utils_logging_pg_and_checkpoint_tensor_helpers(monkeypatch):
    group = _Group(size=4, rank=2)
    monkeypatch.setattr(core_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(core_utils.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(core_utils.torch.distributed, "get_process_group_ranks", lambda group: [3, 4])
    monkeypatch.setattr(core_utils.parallel_state, "is_initialized", lambda: True)
    monkeypatch.setattr(core_utils.parallel_state, "get_tensor_model_parallel_group", lambda **kwargs: group)
    monkeypatch.setattr(core_utils.parallel_state, "get_expert_tensor_parallel_group", lambda **kwargs: "expert")
    monkeypatch.setattr(core_utils.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        core_utils.parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 0,
    )

    assert core_utils.get_pg_size(group) == 4
    assert core_utils.get_pg_rank(group) == 2
    assert core_utils.get_pg_src_rank(group) == 3
    assert core_utils.get_pg_size([group]) == 4
    assert core_utils.get_pg_rank([group]) == 2
    with pytest.warns(DeprecationWarning):
        assert core_utils.get_tensor_model_parallel_group_if_none(None) is group
    assert core_utils.get_tensor_model_parallel_group_if_none(None, is_expert=True) == "expert"
    assert core_utils.get_tensor_model_parallel_group_if_none("explicit") == "explicit"
    monkeypatch.setattr(core_utils.torch.distributed, "is_initialized", lambda: False)
    assert core_utils.get_pg_size(group) == 1
    assert core_utils.get_pg_rank(group) == 0
    assert core_utils.get_pg_src_rank(group) == 0
    assert core_utils.get_tensor_model_parallel_group_if_none(None) is None

    monkeypatch.setattr(core_utils.torch.distributed, "is_initialized", lambda: True)
    logger = logging.getLogger("test_core_utils_logging")
    records = []
    monkeypatch.setattr(logger, "log", lambda *args, **kwargs: records.append((args, kwargs)))
    core_utils.log_on_each_pipeline_stage(logger, logging.INFO, "hello")
    core_utils.log_on_each_pipeline_stage(logger, logging.INFO, "hello2", tp_group=group, dp_cp_group=_Group(rank=0))
    assert len(records) == 1
    with pytest.raises(ValueError, match="must be provided"):
        core_utils.log_on_each_pipeline_stage(logger, logging.INFO, "bad", tp_group=group)

    tensor = torch.arange(8).reshape(2, 4)
    captured = {}

    def _from_offsets(key, tensor, *offsets, **kwargs):
        captured["args"] = (key, tensor, offsets, kwargs)
        return SimpleNamespace(key=key, tensor=tensor, offsets=offsets, kwargs=kwargs)

    monkeypatch.setattr(core_utils.ShardedTensor, "from_rank_offsets", staticmethod(_from_offsets))
    tp_shard = core_utils.make_tp_sharded_tensor_for_checkpoint(
        tensor, "tp", tp_axis=1, tp_group=group, dp_cp_group=_Group(size=2, rank=1)
    )
    assert tp_shard.key == "tp"
    assert captured["args"][2][0] == (1, 2, 4)
    replicated = core_utils.make_sharded_tensor_for_checkpoint(
        tensor, "replicated", tp_group=group, dp_cp_group=_Group(size=2, rank=1)
    )
    assert replicated.kwargs["replica_id"] == (0, 2, 1)

    grad_output, gathered_input = core_utils.prepare_input_tensors_for_wgrad_compute(
        torch.ones(2, 3, 4).transpose(0, 1),
        torch.ones(2, 3, 4).transpose(0, 1),
    )
    assert grad_output.shape == (6, 4)
    assert gathered_input.is_contiguous()
