# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-closed MegaLens test runner."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Check:
    name: str
    argv: tuple[str, ...]
    timeout: float
    required_path: Path | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    returncode: int


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _check_environment(root: Path) -> dict[str, str]:
    """Expose the selected repository root to source-tree checks."""
    env = os.environ.copy()
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join((str(root), inherited)) if inherited else str(root)
    return env


def _profile_checks(
    profile: str,
    root: Path,
    python: str,
    timeout: float,
    *,
    source_repo: Path | None = None,
) -> list[Check]:
    scanner = root / "tools/probe_contract_scan.py"
    scanner_fixture = root / "tests/megalens/fixtures/probe_scan_gate.json"
    static_argv = [python, str(scanner), "gate" if source_repo is not None else "target"]
    if source_repo is not None:
        static_argv.extend(("--source-repo", str(source_repo)))
    static_argv.extend(("--target-repo", str(root), "--fixture", str(scanner_fixture)))
    static = Check(
        name="megalens-static-probe-contract",
        argv=tuple(static_argv),
        timeout=timeout,
        required_path=scanner,
    )
    cpu = Check(
        name="megalens-cpu-contracts",
        argv=(
            python,
            "-m",
            "pytest",
            "-q",
            "tests/megalens/test_observability.py",
            "tests/megalens/test_training_optimizer_observability.py",
            "tests/megalens/test_core_adapter.py",
            "tests/megalens/test_bert_encoder_observability.py",
            "tests/megalens/test_gpt_loss_observability.py",
            "tests/megalens/test_gpt_model_phase_observability.py",
            "tests/megalens/test_transformer_layer_phase_observability.py",
            "tests/megalens/test_runtime_lifecycle.py",
            "tests/megalens/test_offline_contracts.py",
            "tests/megalens/test_event_catalog.py",
            "tests/megalens/test_nested_aggregation.py",
            "tests/megalens/test_dp_probe_contract.py",
            "tests/megalens/test_cp_probe_contract.py",
            "tests/megalens/test_dp_analyzer.py",
            "tests/megalens/test_tp_analyzer.py",
            "tests/megalens/test_pp_analyzer.py",
            "tests/megalens/test_ep_analyzer.py",
            "tests/megalens/test_hybrid_analyzer.py",
            "tests/megalens/test_analyzer.py",
            "tests/megalens/test_probe_contract_scan.py",
            "tests/megalens/test_run_manifest.py",
            "tests/megalens/test_flagscale_run_gate.py",
            "tests/megalens/test_combined_1f1b_probe_contract.py",
            "tests/megalens/test_legacy_checkpoint_comparator.py",
            "tests/megalens/test_dual_node_probe_profiles.py",
            "tests/megalens/test_tp_allreduce_observability.py",
            "tests/megalens/test_tp_allgather_observability.py",
            "tests/megalens/test_tp_reduce_scatter_observability.py",
            "tests/megalens/test_tp_linear_observability.py",
            "tests/megalens/test_dp_grad_sync_observability.py",
            "tests/megalens/test_all_grads_sync_observability.py",
            "tests/megalens/test_sp_layernorm_allreduce_observability.py",
            "tests/megalens/test_embedding_grads_allreduce_observability.py",
            "tests/megalens/test_moe_phase_observability.py",
            "tests/megalens/test_moe_recompute_fp8_probe_contract.py",
            "tests/megalens/test_deepseek_d0_probe_contract.py",
            "tests/megalens/test_deepseek_d2_ep8_config.py",
            "tests/megalens/test_deepseek_tp2_sp_run_gate.py",
            "tests/megalens/test_deepseek_d1_tp2_sp_probe_contract.py",
            "tests/megalens/test_moe_capacity_probe_contract.py",
            "tests/megalens/test_moe_flex_deepep_probe_contract.py",
            "tests/megalens/test_moe_flex_hybridep_probe_contract.py",
            "tests/megalens/test_moe_shared_expert_overlap_probe_contract.py",
            "tests/megalens/test_moe_shared_expert_probe_contract.py",
            "tests/megalens/test_shared_expert_overlap_observability.py",
            "tests/megalens/test_ep_primitive_observability.py",
            "tests/megalens/test_pipeline_schedule_observability.py",
            "tests/megalens/test_dualpipev_schedule_observability.py",
            "tests/megalens/test_dualpipev_a2a_observability.py",
            "tests/megalens/test_p2p_observability.py",
            "tests/megalens/test_p2p_order_nccl_runner.py",
            "tests/megalens/test_bridge_observability.py",
            "tests/megalens/test_bridge_probe_contract.py",
            "tests/megalens/test_mimo_training_probe_contract.py",
            "tests/megalens/test_mimo_pretrain_probe_contract.py",
            "tests/megalens/test_gate_runner.py",
        ),
        timeout=timeout,
    )
    if profile == "cpu":
        return [static, cpu]
    raise ValueError(f"unknown profile: {profile}")


def _target_checks(targets: Sequence[Path], root: Path, python: str, timeout: float) -> list[Check]:
    checks = []
    for target in targets:
        resolved = target if target.is_absolute() else root / target
        resolved = resolved.resolve()
        checks.append(
            Check(
                name=f"target:{target.name}",
                argv=(python, str(resolved)),
                timeout=timeout,
                required_path=resolved,
            )
        )
    return checks


def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    """Signal a check's process group, tolerating an already-exited child."""
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        # The child may exit between wait() timing out and killpg(). The check
        # still exceeded its deadline, so the caller should report rc=124.
        pass


def run_check(check: Check, *, cwd: Path, env: dict[str, str]) -> CheckResult:
    print(f"[MegaLens gate] START {check.name}", flush=True)
    if check.required_path is not None and not check.required_path.is_file():
        print(f"[MegaLens gate] FAIL  {check.name}: missing {check.required_path}", flush=True)
        return CheckResult(check.name, 127)

    try:
        process = subprocess.Popen(check.argv, cwd=cwd, env=env, start_new_session=True)
        try:
            returncode = process.wait(timeout=check.timeout)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait()
            returncode = 124
    except subprocess.TimeoutExpired:
        returncode = 124
    except FileNotFoundError:
        returncode = 127

    status = "PASS" if returncode == 0 else "FAIL"
    print(f"[MegaLens gate] {status:<5} {check.name} rc={returncode}", flush=True)
    return CheckResult(check.name, returncode)


def run_checks(checks: Sequence[Check], *, cwd: Path, env: dict[str, str]) -> int:
    first_failure = 0
    results: list[CheckResult] = []
    for check in checks:
        result = run_check(check, cwd=cwd, env=env)
        results.append(result)
        if first_failure == 0 and result.returncode != 0:
            first_failure = result.returncode

    passed = sum(result.returncode == 0 for result in results)
    print(f"[MegaLens gate] SUMMARY passed={passed} failed={len(results) - passed}", flush=True)
    return first_failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("cpu",), default="cpu")
    parser.add_argument("--target", action="append", type=Path, default=[])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--root", type=Path, default=_repository_root())
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="recompute the locked MixedPara source snapshot in addition to the target gate",
    )
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.target and args.source_repo is not None:
        parser.error("--source-repo only applies to profile checks")

    root = args.root.resolve()
    source_repo = args.source_repo.resolve() if args.source_repo is not None else None
    checks = (
        _target_checks(args.target, root, args.python, args.timeout)
        if args.target
        else _profile_checks(
            args.profile,
            root,
            args.python,
            args.timeout,
            source_repo=source_repo,
        )
    )
    env = _check_environment(root)
    return run_checks(checks, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
