# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run a FlagScale training profile and record compact MegaLens probe evidence."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tests.test_utils.runners import cp_probe_contract  # noqa: E402
from tests.test_utils.runners import dp_probe_contract  # noqa: E402
from tests.test_utils.runners import gpt_probe_contract  # noqa: E402
from tests.test_utils.runners import p2p_probe_contract  # noqa: E402
from tests.test_utils.runners import tp_probe_contract  # noqa: E402
from tests.test_utils.runners import training_run_contract  # noqa: E402
from tests.test_utils.runners import megalens_run_manifest as manifest  # noqa: E402

CONTAINER_SOURCE_ROOT = "/workspace/Megatron-LM-FL"

CONTAINER_RUN_ROOT = "/artifacts/run"

_COMMON_FIELDS = ("g_rk", "dp_rk", "pp_rk", "tp_rk")


def _events(*names: str) -> tuple[manifest.EventRequirement, ...]:
    return tuple(manifest.EventRequirement(name=name, fields=_COMMON_FIELDS) for name in names)


def _contracts(
    *checks: Callable[[Path], Sequence[manifest.Failure]]
) -> Callable[[Path], tuple[manifest.Failure, ...]]:
    def validate(trace_root: Path) -> tuple[manifest.Failure, ...]:
        return tuple(failure for check in checks for failure in check(trace_root))

    return validate


_PP2_EVENTS = _events(
    "forward-step",
    "decoder",
    "decoder-postprocess",
    "output_layer",
    "loss",
    "p2p-launch",
    "p2p-batch-device-sync",
    "send-forward",
    "recv-forward",
    "send-backward",
    "recv-backward",
    "optimizer",
    "optimizer-step",
    "optimizer-postprocess",
)

_PP2_UNBATCHED_EVENTS = _events(
    "p2p-launch", "send-forward", "recv-forward", "send-backward", "recv-backward"
)

_PP2_UNBATCHED_WARMUP_FLUSH_EVENTS = (
    *_PP2_UNBATCHED_EVENTS,
    *_events("forward-step", "backward-step"),
)

_EP_COMMON_EVENTS = (
    manifest.EventRequirement(
        "moe-router", (*_COMMON_FIELDS, "layer", "num_experts", "router_topk"), "E"
    ),
    manifest.EventRequirement(
        "moe-dispatch",
        (
            *_COMMON_FIELDS,
            "layer",
            "dispatcher",
            "num_tokens",
            "dropped_tokens",
            "drop_rate",
            "expert_cv",
            "top1_expert_share",
            "aux_loss",
            "z_loss",
        ),
        "E",
    ),
    *_events("moe-experts", "moe-combine"),
)

_EP_ROUTE_FIELDS = (*_COMMON_FIELDS, "comm_type", "dispatcher", "group_size")

_DP_FIELDS = (*_COMMON_FIELDS, "op", "data_bytes", "group_size", "operation_id", "payload_role")

_CP_TE_EVENTS = (
    *_events(
        "forward-step",
        "decoder",
        "decoder-postprocess",
        "output_layer",
        "loss",
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "MLP.forward",
        "grad-sync",
        "all-grads-sync",
    ),
    manifest.EventRequirement("dp-allreduce", (*_DP_FIELDS, "group_role", "stage"), "B"),
)

_TP_COLLECTIVE_FIELDS = (*_COMMON_FIELDS, "op", "dim", "data_bytes", "group_size")

_TP2_SP_EVENTS = (
    *(
        manifest.EventRequirement(name, _TP_COLLECTIVE_FIELDS, "B")
        for name in (
            "tp-all-gather-first",
            "tp-all-gather-last",
            "tp-reduce-scatter",
            "tp-reduce-scatter-last",
        )
    ),
    manifest.EventRequirement(
        "tp-linear-async-launch",
        (*_COMMON_FIELDS, "operation_id", "collective_op", "launch_site"),
        "B",
    ),
    manifest.EventRequirement(
        "tp-linear-async-complete",
        (*_COMMON_FIELDS, "operation_id", "collective_op", "completion_kind"),
        "B",
    ),
    manifest.EventRequirement("grad-sync", (*_COMMON_FIELDS, "schedule", "timing_phase"), "B"),
    manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
    manifest.EventRequirement(
        "sp-layernorm-allreduce",
        (*_COMMON_FIELDS, "data_bytes", "group_size", "reduce_op", "grad_bucket"),
        "B",
    ),
)


def _ep_events(dispatcher: str) -> tuple[manifest.EventRequirement, ...]:
    return (
        *_EP_COMMON_EVENTS,
        manifest.EventRequirement(f"ep-{dispatcher}-dispatch", _EP_ROUTE_FIELDS, "E"),
        manifest.EventRequirement(f"ep-{dispatcher}-combine", _EP_ROUTE_FIELDS, "E"),
    )


def _dp_events(profile: str) -> tuple[manifest.EventRequirement, ...]:
    names = (
        ("dp-allreduce",)
        if profile == "standard-ddp"
        else ("dp-reduce-scatter", "dp-param-all-gather")
    )
    return tuple(manifest.EventRequirement(name, _DP_FIELDS, "B") for name in names)


_LAYERWISE_FULL_CUDA_GRAPH_EVENTS = _events(
    "forward-step",
    "backward-step",
    "decoder",
    "decoder-postprocess",
    "output_layer",
    "loss",
    "transformer_layer",
    "_forward_attention",
    "attention",
    "_forward_mlp",
    "MLP.forward",
    "optimizer",
    "optimizer-step",
    "optimizer-postprocess",
)

_GPT_EAGER_EVENTS = _events(
    "forward-step",
    "decoder",
    "decoder-postprocess",
    "output_layer",
    "loss",
    "transformer_layer",
    "_forward_attention",
    "attention",
    "_forward_mlp",
    "MLP.forward",
)

PROFILES: Mapping[str, manifest.TraceProfile] = {
    'pp1': manifest.TraceProfile(
        "pp1", 1, run_contract=training_run_contract.validate_two_iteration_checkpoint
    ),
    'gpt-eager-continuous-cupti': manifest.TraceProfile(
        "gpt-eager-continuous-cupti",
        1,
        _GPT_EAGER_EVENTS,
        gpt_probe_contract.validate_gpt_pp1_eager_continuous_kernel_phases,
        training_run_contract.validate_two_iteration_continuous_cuda_kernel_checkpoint,
    ),
    'cp2-te': manifest.TraceProfile(
        "cp2-te",
        2,
        _CP_TE_EVENTS,
        cp_probe_contract.validate_cp2_te_coexistence,
        training_run_contract.validate_two_iteration_transformer_engine_cp2_checkpoint,
    ),
    'tp2-sp-local': manifest.TraceProfile(
        "tp2-sp-local",
        2,
        _TP2_SP_EVENTS,
        tp_probe_contract.validate_tp2_sp_profile,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    'tp2-pp4-multimicrobatch': manifest.TraceProfile(
        "tp2-pp4-multimicrobatch",
        8,
        (
            *_PP2_UNBATCHED_WARMUP_FLUSH_EVENTS,
            *_events("optimizer", "optimizer-step", "optimizer-postprocess"),
            manifest.EventRequirement(
                "sp-layernorm-allreduce",
                (*_COMMON_FIELDS, "data_bytes", "group_size", "reduce_op", "grad_bucket"),
                "B",
            ),
        ),
        tp_probe_contract.validate_tp2_pp4_multimicrobatch,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    'pp2': manifest.TraceProfile(
        "pp2",
        2,
        _PP2_EVENTS,
        _contracts(
            gpt_probe_contract.validate_gpt_pp2_training_phases,
            p2p_probe_contract.validate_pp2_batched_route,
        ),
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    'ep2-alltoall': manifest.TraceProfile(
        "ep2-alltoall",
        2,
        _ep_events("alltoall"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    'ep2-allgather': manifest.TraceProfile(
        "ep2-allgather",
        2,
        _ep_events("allgather"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    'dp2-standard-ddp-overlap': manifest.TraceProfile(
        "dp2-standard-ddp-overlap",
        2,
        (
            *_dp_events("standard-ddp"),
            manifest.EventRequirement(
                "dp-grad-sync-complete", (*_COMMON_FIELDS, "operation_ids", "completion_kind"), "B"
            ),
        ),
        dp_probe_contract.validate_dp_standard_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    'dp2-distopt-overlap': manifest.TraceProfile(
        "dp2-distopt-overlap",
        2,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement(
                "dp-grad-sync-complete", (*_COMMON_FIELDS, "operation_ids", "completion_kind"), "B"
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete", (*_COMMON_FIELDS, "operation_id", "completion_kind"), "B"
            ),
        ),
        dp_probe_contract.validate_dp_distopt_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    'te-full-cuda-kernels': manifest.TraceProfile(
        "te-full-cuda-kernels",
        1,
        _LAYERWISE_FULL_CUDA_GRAPH_EVENTS,
        gpt_probe_contract.validate_te_full_cuda_graph_kernel_phases,
        training_run_contract.validate_two_iteration_te_full_cuda_graph_kernel_checkpoint,
    ),
}

_CONFIG_PROFILES = {
    'flagscale_single_node_smoke': 'pp1',
    'flagscale_single_node_gpt_eager_continuous_cupti_smoke': 'gpt-eager-continuous-cupti',
    'flagscale_single_node_cp2_te_smoke': 'cp2-te',
    'flagscale_single_node_tp2_sp_local_smoke': 'tp2-sp-local',
    'flagscale_single_node_tp2_pp4_multimicrobatch_smoke': 'tp2-pp4-multimicrobatch',
    'flagscale_single_node_pp2_smoke': 'pp2',
    'flagscale_single_node_ep2_smoke': 'ep2-alltoall',
    'flagscale_single_node_dp2_standard_overlap_smoke': 'dp2-standard-ddp-overlap',
    'flagscale_single_node_dp2_distopt_overlap_smoke': 'dp2-distopt-overlap',
    'flagscale_single_node_te_cuda_graph_full_cupti_smoke': 'te-full-cuda-kernels',
}


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    timed_out: bool = False


def _source_head(source_root: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"cannot read source HEAD: {detail}")
    status = subprocess.run(
        ("git", "-C", str(source_root), "status", "--porcelain=v1", "--untracked-files=normal"),
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        detail = status.stderr.strip() or status.stdout.strip()
        raise ValueError(f"cannot read source status: {detail}")
    if status.stdout.strip():
        raise ValueError("source tree must be clean before running the gate")
    return result.stdout.strip()


def _reserve_loopback_port(requested: int) -> int:
    if requested:
        if not 1 <= requested <= 65535:
            raise ValueError("--rdzv-port must be between 1 and 65535")
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _profile_from_arguments(args: argparse.Namespace) -> manifest.TraceProfile:
    if args.profile is not None:
        return PROFILES[args.profile]
    configured = _CONFIG_PROFILES.get(args.input_config.stem)
    if configured is None:
        raise ValueError(
            "unknown or archived training configuration; select a supported --profile "
            "explicitly for a custom configuration"
        )
    if configured == "ep2-alltoall" and args.ep_dispatcher == "allgather":
        configured = "ep2-allgather"
    return PROFILES[configured]


def _ep_dispatcher(profile: manifest.TraceProfile, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    if profile.name in {"ep2-alltoall", "ep2-allgather"}:
        return profile.name.removeprefix("ep2-")
    return None


def _docker_command(
    *,
    run_dir: Path,
    config_name: str,
    mode: str,
    image: str,
    source_root: Path,
    rdzv_port: int,
    ep_dispatcher: str | None,
    flagscale_training_overlay: Path | None,
    dataset_helper_overlay: Path | None = None,
) -> tuple[str, ...]:
    overlay = ()
    if flagscale_training_overlay is not None:
        overlay = (
            "--volume",
            f"{flagscale_training_overlay}:"
            "/workspace/FlagScale/flagscale/train/megatron/training/training.py:ro",
        )
    dataset_overlay = ()
    if dataset_helper_overlay is not None:
        dataset_overlay = (
            "--volume",
            f"{dataset_helper_overlay}:" f"{CONTAINER_SOURCE_ROOT}/megatron/core/datasets",
        )
    dispatcher = ()
    if ep_dispatcher is not None:
        dispatcher = ("--env", f"MEGALENS_GATE_EP_DISPATCHER={ep_dispatcher}")

    shell = (
        "set -euo pipefail; "
        "source /root/miniconda3/etc/profile.d/conda.sh; "
        "conda activate flagscale-train; "
        "export PYTHONPATH=/workspace/FlagScale:"
        "/workspace/FlagScale/flagscale/train:/workspace/Megatron-LM-FL:"
        "${PYTHONPATH:-}; "
        "cd /workspace/FlagScale; "
        'exec "$@"'
    )
    return (
        "docker",
        "run",
        "--rm",
        "--privileged",
        "--runtime=nvidia",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size=64g",
        "--ulimit",
        "memlock=-1:-1",
        "--volume",
        f"{run_dir}:{CONTAINER_RUN_ROOT}",
        "--volume",
        f"{source_root}:{CONTAINER_SOURCE_ROOT}:ro",
        *dataset_overlay,
        *overlay,
        "--env",
        f"MEGALENS_GATE_CONTAINER_RUN_DIR={CONTAINER_RUN_ROOT}",
        "--env",
        f"MEGALENS_GATE_RDZV_ENDPOINT=127.0.0.1:{rdzv_port}",
        "--env",
        f"MEGALENS_GATE_TRACE={'true' if mode == 'trace-on' else 'false'}",
        *dispatcher,
        image,
        "/bin/bash",
        "-lc",
        shell,
        "_",
        "flagscale",
        "run",
        f"--config-path={CONTAINER_RUN_ROOT}/inputs",
        f"--config-name={config_name}",
        "--action=test",
    )


def run_foreground(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], launcher_log: Path, timeout: float
) -> ExecutionResult:
    try:
        with launcher_log.open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(
                    tuple(argv),
                    cwd=cwd,
                    env=dict(env),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                output.write(f"[FlagScale MegaLens] training timed out after {timeout} seconds\n")
                return ExecutionResult(124, True)
            except FileNotFoundError as error:
                output.write(f"[FlagScale MegaLens] missing executable: {error}\n")
                return ExecutionResult(127)
            except OSError as error:
                output.write(f"[FlagScale MegaLens] launcher error: {error}\n")
                return ExecutionResult(126)
    except OSError:
        return ExecutionResult(126)
    return ExecutionResult(completed.returncode)


def _copy_inputs(source: Path, run_dir: Path) -> Path:
    inputs = run_dir / "inputs"
    inputs.mkdir()
    destination = inputs / source.name
    shutil.copyfile(source, destination)
    return destination


def _prepare_dataset_helper_overlay(source_root: Path, run_dir: Path) -> Path:
    """Copy the dataset helper sources to a writable per-run build directory."""

    source = source_root / "megatron" / "core" / "datasets"
    destination = run_dir / "build" / "megatron-core-datasets"
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "helpers_cpp*.so")
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("trace-on", "trace-off"))
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--image", required=True)
    parser.add_argument("--ep-dispatcher", choices=("alltoall", "allgather"))
    parser.add_argument(
        "--megatron-source-root",
        type=Path,
        default=_REPOSITORY_ROOT,
        help="current source checkout mounted over the image source",
    )
    parser.add_argument("--flagscale-training-overlay", type=Path)
    parser.add_argument(
        "--controller-revision",
        help="accepted for compatibility; the manifest records the mounted source HEAD",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--rdzv-port", type=int, default=0)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    return parser


def _exit_code(execution: ExecutionResult, passed: bool) -> int:
    if execution.returncode == 0:
        return 0 if passed else 1
    if execution.returncode < 0:
        return min(255, 128 - execution.returncode)
    return min(255, execution.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not args.input_config.is_file():
        parser.error("--input-config must be a YAML file")
    if args.input_config.suffix not in {".yaml", ".yml"}:
        parser.error("--input-config must be a YAML file")
    if not args.image.strip():
        parser.error("--image cannot be empty")

    source_root = args.megatron_source_root.resolve()
    if not source_root.is_dir():
        parser.error("--megatron-source-root must be a directory")
    try:
        profile = _profile_from_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        source_head = _source_head(source_root)
        rdzv_port = _reserve_loopback_port(args.rdzv_port)
    except ValueError as error:
        parser.error(str(error))
    run_dir = args.run_dir.resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"[FlagScale MegaLens] run directory already exists: {run_dir}", file=sys.stderr)
        return 2

    started_at = manifest.utc_now()
    copied_config = _copy_inputs(args.input_config.resolve(), run_dir)
    dataset_helper_overlay = _prepare_dataset_helper_overlay(source_root, run_dir)
    launcher_log = run_dir / "launcher.log"
    dispatcher = _ep_dispatcher(profile, args.ep_dispatcher)
    overlay = (
        args.flagscale_training_overlay.resolve()
        if args.flagscale_training_overlay is not None
        else None
    )
    command = _docker_command(
        run_dir=run_dir,
        config_name=copied_config.stem,
        mode=args.mode,
        image=args.image,
        source_root=source_root,
        rdzv_port=rdzv_port,
        ep_dispatcher=dispatcher,
        flagscale_training_overlay=overlay,
        dataset_helper_overlay=dataset_helper_overlay,
    )
    environment = os.environ.copy()
    environment.update(
        MEGALENS_GATE_RUN_DIR=str(run_dir),
        MEGALENS_GATE_MODE=args.mode,
        MEGALENS_GATE_PROFILE=profile.name,
    )
    if dispatcher is not None:
        environment["MEGALENS_GATE_EP_DISPATCHER"] = dispatcher

    execution = run_foreground(
        command,
        cwd=args.cwd.resolve(),
        env=environment,
        launcher_log=launcher_log,
        timeout=args.timeout,
    )
    report = manifest.validate_trace(
        run_dir / "traces", profile, trace_enabled=args.mode == "trace-on"
    )
    report = manifest.validate_run_artifacts(
        run_dir, profile, report, trace_enabled=args.mode == "trace-on"
    )
    payload = manifest.build_manifest(
        run_id=run_dir.name,
        started_at=started_at,
        finished_at=manifest.utc_now(),
        profile=profile,
        mode=args.mode,
        command=command,
        config_path=copied_config,
        source_root=source_root,
        source_head=source_head,
        image=args.image,
        returncode=execution.returncode,
        timed_out=execution.timed_out,
        log_path=launcher_log,
        report=report,
    )
    manifest.write_manifest(run_dir, payload)
    exit_code = _exit_code(execution, report.passed)
    print(
        f"[FlagScale MegaLens] status={payload['status']} "
        f"profile={profile.name} rc={exit_code} manifest={run_dir / manifest.MANIFEST_NAME}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
