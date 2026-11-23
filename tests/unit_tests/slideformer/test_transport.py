"""Explicit DP ownership must fail closed and keep legacy defaults unchanged."""
from dataclasses import replace
from types import SimpleNamespace
import os

import pytest
import torch

from megatron.plugin.slideformer.engine import MegatronSlideFormerEngineConfig
from megatron.plugin.slideformer.session import FullFineTuneSession
from megatron.plugin.slideformer.transport import DataParallelTransport


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in os.environ:
        if name.startswith("MEGATRON_SLIDEFORMER_"):
            monkeypatch.delenv(name)


def test_no_implicit_change_to_legacy_default():
    assert DataParallelTransport.from_env() is None


@pytest.mark.parametrize("preset,gradient,default_h2d", [
    ("dense-dp", "gpu_reduce", "shard"), ("hsr-dp", "hsr", "full")])
def test_preset_default_and_explicit_transfer_override(monkeypatch, preset, gradient, default_h2d):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_DP_PRESET", preset)
    assert DataParallelTransport.from_env() == DataParallelTransport(gradient, default_h2d)
    for flag, expected in [("0", "full"), ("1", "shard")]:
        monkeypatch.setenv("MEGATRON_SLIDEFORMER_SHARD_H2D", flag)
        assert DataParallelTransport.from_env() == DataParallelTransport(gradient, expected)


def test_explicit_session_default_and_legacy_env_remain_distinct(monkeypatch):
    assert DataParallelTransport() == DataParallelTransport("gpu_reduce", "shard")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_SHARDED_CPU_OPT", "1")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_EAGER_SHARDED_UPDATE", "1")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_GPU_REDUCE", "1")
    assert DataParallelTransport.from_env() == DataParallelTransport("gpu_reduce", "full")


@pytest.mark.parametrize("preset,gradient", [("dense-dp", "gpu_reduce"), ("hsr-dp", "hsr")])
def test_presets_do_not_mutate_global_environment(monkeypatch, preset, gradient):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_DP_PRESET", preset)
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_SHARD_H2D", "1")
    before = dict(os.environ)
    assert DataParallelTransport.from_env() == DataParallelTransport(gradient, "shard")
    assert dict(os.environ) == before


@pytest.mark.parametrize("name,value", [("SHARDED_CPU_OPT", "0"), ("EAGER_SHARDED_UPDATE", "0"), ("GPU_REDUCE", "0")])
def test_conflicting_legacy_flags_are_not_silently_ignored(monkeypatch, name, value):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_DP_PRESET", "dense-dp")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_" + name, value)
    with pytest.raises(ValueError, match="conflicts"):
        DataParallelTransport.from_env()


@pytest.mark.parametrize("field,value", [("context_parallel_size", 2), ("tensor_model_parallel_size", 2),
    ("expert_model_parallel_size", 2), ("pipeline_model_parallel_size", 2), ("num_moe_experts", 4)])
def test_unqualified_compositions_rejected_before_allocation(field, value):
    native = SimpleNamespace(**{field: value})
    with pytest.raises(NotImplementedError):
        DataParallelTransport().prepare(SimpleNamespace(config=native), None, None)


@pytest.mark.parametrize("change", [dict(defer_optimizer_step=False), dict(fp32_master_params=False),
    dict(fp32_optimizer_state=False), dict(nvme_offload_fraction=0.1), dict(skip_grad_d2h=True)])
def test_incompatible_storage_rejected_before_collectives(change):
    native = SimpleNamespace(calculate_per_token_loss=True, params_dtype=torch.bfloat16)
    config = replace(MegatronSlideFormerEngineConfig(defer_optimizer_step=True), **change)
    with pytest.raises(ValueError, match="FP32 CPU"):
        DataParallelTransport().prepare(SimpleNamespace(config=native), None, config)


def test_partial_owner_coverage_cannot_report_success():
    engine = SimpleNamespace(enable_sharded_cpu_optimizer=lambda *a, **k: True,
        managed_layers=[SimpleNamespace(params=[object()])], _sharded_opt=SimpleNamespace(owners=[]))
    with pytest.raises(RuntimeError, match="every trainable owner"):
        DataParallelTransport().attach(engine, SimpleNamespace(dp_cp=None, tp=None, pp=None))


@pytest.mark.parametrize("micro,tokens", [(2, torch.tensor(8)), (1, None)])
def test_missing_early_scale_rejected_before_zeroing_or_backward(micro, tokens):
    session = FullFineTuneSession.__new__(FullFineTuneSession)
    session._phase, session.transport = "ready", DataParallelTransport()
    with pytest.raises(ValueError, match="one batch"):
        session.begin_step(micro, num_tokens=tokens)
    assert session._phase == "ready"
