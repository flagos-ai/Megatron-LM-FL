# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.test_utils.runners import run_megalens_gate as gate

RUNNER = Path(__file__).parents[1] / "test_utils/runners/run_megalens_gate.py"
REPOSITORY_ROOT = Path(__file__).parents[2]


def _write_script(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def _run_runner(
    root: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--root", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_runner_returns_zero_when_all_targets_pass(tmp_path: Path) -> None:
    _write_script(tmp_path / "pass.py", "raise SystemExit(0)\n")
    result = _run_runner(tmp_path, "--target", "pass.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_profile_wires_static_contract_before_cpu_tests() -> None:
    checks = gate._profile_checks("cpu", REPOSITORY_ROOT, sys.executable, 300.0)

    assert [check.name for check in checks] == [
        "megalens-static-probe-contract",
        "megalens-cpu-contracts",
    ]
    assert checks[0].argv[-2:] == (
        "--fixture",
        str(REPOSITORY_ROOT / "tests/megalens/fixtures/probe_scan_gate.json"),
    )
    assert {
        "tests/megalens/test_bert_encoder_observability.py",
        "tests/megalens/test_gpt_loss_observability.py",
        "tests/megalens/test_gpt_model_phase_observability.py",
        "tests/megalens/test_transformer_layer_phase_observability.py",
        "tests/megalens/test_dual_node_probe_profiles.py",
        "tests/megalens/test_moe_capacity_probe_contract.py",
        "tests/megalens/test_moe_flex_deepep_probe_contract.py",
        "tests/megalens/test_moe_shared_expert_probe_contract.py",
        "tests/megalens/test_bridge_observability.py",
        "tests/megalens/test_mimo_pretrain_probe_contract.py",
    } <= set(checks[1].argv)


def test_cpu_profile_can_wire_locked_source_and_compatibility_argument() -> None:
    source_repo = REPOSITORY_ROOT.parent / "MixedPara"
    contract = REPOSITORY_ROOT / "tests/megalens/fixtures/ignored-compatibility-input.json"

    checks = gate._profile_checks(
        "cpu",
        REPOSITORY_ROOT,
        sys.executable,
        300.0,
        source_repo=source_repo,
        contract_path=contract,
    )

    assert checks[0].argv == (
        sys.executable,
        str(REPOSITORY_ROOT / "tools/probe_contract_scan.py"),
        "gate",
        "--source-repo",
        str(source_repo),
        "--target-repo",
        str(REPOSITORY_ROOT),
        "--fixture",
        str(REPOSITORY_ROOT / "tests/megalens/fixtures/probe_scan_gate.json"),
        "--contract",
        str(contract),
    )


def test_runner_rejects_contract_options_for_custom_targets(tmp_path: Path) -> None:
    passing = _write_script(tmp_path / "pass.py", "raise SystemExit(0)\n")

    result = _run_runner(tmp_path, "--target", str(passing), "--source-repo", str(tmp_path))

    assert result.returncode == 2
    assert "only apply to profile checks" in result.stderr


def test_runner_preserves_first_failure_and_runs_later_targets(tmp_path: Path) -> None:
    passing = _write_script(tmp_path / "pass.py", "raise SystemExit(0)\n")
    failing = _write_script(tmp_path / "fail.py", "raise SystemExit(23)\n")
    sentinel = tmp_path / "after-ran"
    after = _write_script(
        tmp_path / "after.py",
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
    )

    result = _run_runner(
        tmp_path, "--target", str(passing), "--target", str(failing), "--target", str(after)
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert sentinel.read_text() == "ran"


def test_runner_reports_timeout_as_124(tmp_path: Path) -> None:
    slow = _write_script(tmp_path / "slow.py", "import time\ntime.sleep(5)\n")
    result = _run_runner(tmp_path, "--timeout", "0.05", "--target", str(slow))
    assert result.returncode == 124, result.stdout + result.stderr


def test_runner_reports_missing_target_as_127(tmp_path: Path) -> None:
    result = _run_runner(tmp_path, "--target", str(tmp_path / "missing.py"))
    assert result.returncode == 127, result.stdout + result.stderr


def test_source_target_can_import_from_repository_root(tmp_path: Path) -> None:
    package = tmp_path / "source_package"
    package.mkdir()
    _write_script(package / "__init__.py", "VALUE = 17\n")
    checks = tmp_path / "nested" / "checks"
    checks.mkdir(parents=True)
    target = _write_script(
        checks / "import_source.py", "from source_package import VALUE\nassert VALUE == 17\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = ""

    result = _run_runner(tmp_path, "--target", str(target), env=env)

    assert result.returncode == 0, result.stdout + result.stderr


def test_runner_tolerates_process_group_exiting_during_timeout(tmp_path: Path, monkeypatch) -> None:
    class _ExitedProcess:
        pid = 137

        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self, timeout=None) -> int:
            self.wait_count += 1
            if self.wait_count <= 2:
                raise subprocess.TimeoutExpired([sys.executable], timeout)
            return 0

    process = _ExitedProcess()
    signals = []
    monkeypatch.setattr(gate.subprocess, "Popen", lambda *args, **kwargs: process)

    def _already_exited(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr(gate.os, "killpg", _already_exited)

    result = gate.run_check(
        gate.Check("timeout-race", (sys.executable,), 0.01), cwd=tmp_path, env={}
    )

    assert result.returncode == 124
    assert signals == [(process.pid, gate.signal.SIGTERM), (process.pid, gate.signal.SIGKILL)]
    assert process.wait_count == 3
