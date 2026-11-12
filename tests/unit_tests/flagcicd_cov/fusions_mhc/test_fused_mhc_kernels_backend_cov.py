# Copyright (c) 2026, BAAI. All rights reserved.
"""mHC fused backend validation, availability, selection, and once-per-policy logging."""

import logging

import pytest
import torch

import megatron.core.fusions.fused_mhc_kernels as mhc

AUTO_POLICY_STATUS = (
    "policy=auto; sinkhorn=triton; h_aggregate=fwd:triton,bwd:native; "
    "h_post_bda=fwd:triton,bwd:triton; proj_rms_compute_h=native"
)
NATIVE_POLICY_STATUS = (
    "policy=native; sinkhorn=native; h_aggregate=fwd:native,bwd:native; "
    "h_post_bda=fwd:native,bwd:native; proj_rms_compute_h=native"
)
CUTILE_POLICY_STATUS = (
    "policy=cutile; sinkhorn=cutile; h_aggregate=fwd:cutile,bwd:cutile; "
    "h_post_bda=fwd:cutile,bwd:cutile; proj_rms_compute_h=cutile"
)


def _write_tileiras(directory, body):
    compiler = directory / "tileiras"
    compiler.write_text(f"#!/bin/sh\n{body}")
    compiler.chmod(0o755)
    return compiler


def _simulate_cuda_tile(monkeypatch):
    monkeypatch.setattr(mhc, "_CUTILE_AVAILABLE", True)
    monkeypatch.setattr(mhc, "_CUTILE_DEVICE_SUPPORT_CACHE", None)
    monkeypatch.setattr(mhc, "_CUTILE_DEVICE_SUPPORT_ERROR", None)


def test_fused_sinkhorn_rejects_unknown_backend_policy():
    logits = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="Unknown mHC fused backend"):
        mhc.fused_sinkhorn(logits, 1, backend="webgpu")


@pytest.mark.parametrize(
    "source", ["path", "cuda_home", "missing"], ids=["on-path", "in-cuda-home", "absent"]
)
def test_get_tileiras_path_locates_compiler_from_path_or_cuda_home(source, monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    monkeypatch.setenv("PATH", str(search_dir))
    if source == "path":
        compiler = search_dir / "tileiras"
        compiler.write_text("")
        compiler.chmod(0o755)
        assert mhc._get_tileiras_path() == str(compiler)
    else:
        cuda_home = tmp_path / "cuda-home"
        (cuda_home / "bin").mkdir(parents=True)
        monkeypatch.setenv("CUDA_HOME", str(cuda_home))
        if source == "cuda_home":
            compiler = cuda_home / "bin" / "tileiras"
            compiler.write_text("")
            compiler.chmod(0o755)
            assert mhc._get_tileiras_path() == str(compiler)
        else:
            assert mhc._get_tileiras_path() is None


def test_cutile_supports_current_device_reports_false_when_cuda_tile_is_missing():
    assert mhc._cutile_supports_current_device() is False
    assert mhc.is_cutile_available() is False
    with pytest.raises(RuntimeError, match="current environment: cuTile is unavailable"):
        mhc.fused_sinkhorn(torch.zeros(1, 1, 2, 2), 1, backend="cutile")


def test_mhc_backend_status_reports_triton_frontends_when_cutile_unavailable():
    message, all_native = mhc._mhc_backend_status("auto")
    assert message == AUTO_POLICY_STATUS
    assert all_native is False


def test_mhc_backend_status_reports_all_native_for_explicit_native_policy():
    message, all_native = mhc._mhc_backend_status("native")
    assert message == NATIVE_POLICY_STATUS
    assert all_native is True


def test_mhc_backend_selection_returns_the_backend_status_message():
    assert mhc._mhc_backend_selection("auto") == AUTO_POLICY_STATUS
    assert mhc._mhc_backend_selection("native") == NATIVE_POLICY_STATUS


def test_log_fused_mhc_backend_once_emits_each_policy_exactly_once(monkeypatch):
    logged = []

    def record_log(_logger, level, message):
        logged.append((level, message))

    monkeypatch.setattr(mhc, "_BACKEND_INFO_LOGGED", set())
    monkeypatch.setattr(mhc, "log_single_rank", record_log)
    mhc.log_fused_mhc_backend_once("native")
    mhc.log_fused_mhc_backend_once("native")
    mhc.log_fused_mhc_backend_once("auto")

    assert logged == [
        (logging.INFO, f"[mHC] fused backend selection: {NATIVE_POLICY_STATUS}"),
        (logging.INFO, f"[mHC] fused backend selection: {AUTO_POLICY_STATUS}"),
    ]


def test_cutile_supports_current_device_caches_the_first_probe_result(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_tileiras(bin_dir, "exit 0\n")
    _simulate_cuda_tile(monkeypatch)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert mhc._cutile_supports_current_device() is True
    assert mhc.is_cutile_available() is True

    empty_dir = tmp_path / "empty-bin"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    assert mhc._cutile_supports_current_device() is True


def test_cutile_supports_current_device_reports_missing_compiler_when_tileiras_is_absent(
    monkeypatch, tmp_path
):
    empty_dir = tmp_path / "empty-bin"
    empty_dir.mkdir()
    _simulate_cuda_tile(monkeypatch)
    monkeypatch.setenv("PATH", str(empty_dir))
    monkeypatch.setenv("CUDA_HOME", str(empty_dir))

    assert mhc._cutile_supports_current_device() is False
    with pytest.raises(RuntimeError, match="current environment: tileiras compiler was not found"):
        mhc.fused_sinkhorn(torch.zeros(1, 1, 2, 2), 1, backend="cutile")


def test_cutile_supports_current_device_reports_compiler_diagnostic_when_arch_is_rejected(
    monkeypatch, tmp_path
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_tileiras(bin_dir, "echo \"tileiras: error: Cannot find option named '$2'\"\n")
    _simulate_cuda_tile(monkeypatch)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert mhc._cutile_supports_current_device() is False
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    with pytest.raises(
        RuntimeError,
        match=f"current environment: tileiras: error: Cannot find option named '{arch}'",
    ):
        mhc.fused_sinkhorn(torch.zeros(1, 1, 2, 2), 1, backend="cutile")


def test_cutile_supports_current_device_reports_launch_failure_when_tileiras_is_broken(
    monkeypatch, tmp_path
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    compiler = bin_dir / "tileiras"
    compiler.write_text("#!/no/such/interpreter\n")
    compiler.chmod(0o755)
    _simulate_cuda_tile(monkeypatch)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert mhc._cutile_supports_current_device() is False
    with pytest.raises(RuntimeError, match="No such file or directory") as caught:
        mhc.fused_sinkhorn(torch.zeros(1, 1, 2, 2), 1, backend="cutile")
    assert str(compiler) in str(caught.value)


def test_mhc_backend_status_reports_cutile_for_every_operation_when_the_device_is_supported(
    monkeypatch, tmp_path
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_tileiras(bin_dir, "exit 0\n")
    _simulate_cuda_tile(monkeypatch)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    message, all_native = mhc._mhc_backend_status("cutile")
    assert message == CUTILE_POLICY_STATUS
    assert all_native is False
