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

from tests.test_utils.runners import dp_probe_contract  # noqa: E402
from tests.test_utils.runners import generate_bert_smoke_inputs  # noqa: E402
from tests.test_utils.runners import gpt_probe_contract  # noqa: E402
from tests.test_utils.runners import megalens_run_manifest as manifest  # noqa: E402
from tests.test_utils.runners import p2p_probe_contract  # noqa: E402
from tests.test_utils.runners import tp_probe_contract  # noqa: E402

CONTAINER_SOURCE_ROOT = "/workspace/Megatron-LM-FL"
CONTAINER_RUN_ROOT = "/artifacts/run"

_COMMON_FIELDS = ("g_rk", "dp_rk", "pp_rk", "tp_rk")


def _events(*names: str) -> tuple[manifest.EventRequirement, ...]:
    return tuple(
        manifest.EventRequirement(name=name, fields=_COMMON_FIELDS) for name in names
    )


def _contracts(
    *checks: Callable[[Path], Sequence[manifest.Failure]],
) -> Callable[[Path], tuple[manifest.Failure, ...]]:
    def validate(trace_root: Path) -> tuple[manifest.Failure, ...]:
        return tuple(
            failure
            for check in checks
            for failure in check(trace_root)
        )

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
    "p2p-launch",
    "send-forward",
    "recv-forward",
    "send-backward",
    "recv-backward",
)
_EP_COMMON_EVENTS = (
    manifest.EventRequirement(
        "moe-router",
        (*_COMMON_FIELDS, "layer", "num_experts", "router_topk"),
        "E",
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
_DP_FIELDS = (
    *_COMMON_FIELDS,
    "op",
    "data_bytes",
    "group_size",
    "operation_id",
    "payload_role",
)
_TP_COLLECTIVE_FIELDS = (
    *_COMMON_FIELDS,
    "op",
    "dim",
    "data_bytes",
    "group_size",
)


def _ep_events(
    dispatcher: str,
    *,
    fine_grained: bool = False,
) -> tuple[manifest.EventRequirement, ...]:
    route_events = (
        manifest.EventRequirement(f"ep-{dispatcher}-dispatch", _EP_ROUTE_FIELDS, "E"),
        manifest.EventRequirement(f"ep-{dispatcher}-combine", _EP_ROUTE_FIELDS, "E"),
    )
    scheduler = (
        (
            manifest.EventRequirement(
                "combined-forward-backward-step",
                (*_COMMON_FIELDS, "execution_mode"),
                "B",
            ),
        )
        if fine_grained
        else ()
    )
    return (*_EP_COMMON_EVENTS, *route_events, *scheduler)


def _dp_events(profile: str) -> tuple[manifest.EventRequirement, ...]:
    names = (
        ("dp-allreduce",)
        if profile == "standard-ddp"
        else ("dp-reduce-scatter", "dp-param-all-gather")
    )
    return tuple(manifest.EventRequirement(name, _DP_FIELDS, "B") for name in names)


PROFILES: Mapping[str, manifest.TraceProfile] = {
    "pp1": manifest.TraceProfile("pp1", 1),
    "gpt-eager-full": manifest.TraceProfile(
        "gpt-eager-full",
        1,
        _events(
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
        ),
        gpt_probe_contract.validate_gpt_pp1_eager_phases,
    ),
    "tp2-sp-local": manifest.TraceProfile(
        "tp2-sp-local",
        2,
        (
            *(
                manifest.EventRequirement(
                    name,
                    _TP_COLLECTIVE_FIELDS,
                    "B",
                )
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
                (
                    *_COMMON_FIELDS,
                    "operation_id",
                    "collective_op",
                    "completion_kind",
                ),
                "B",
            ),
            manifest.EventRequirement(
                "grad-sync",
                (*_COMMON_FIELDS, "schedule", "timing_phase"),
                "B",
            ),
            manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
            manifest.EventRequirement(
                "sp-layernorm-allreduce",
                (
                    *_COMMON_FIELDS,
                    "data_bytes",
                    "group_size",
                    "reduce_op",
                    "grad_bucket",
                ),
                "B",
            ),
        ),
        tp_probe_contract.validate_tp2_sp_profile,
    ),
    "pp2": manifest.TraceProfile(
        "pp2",
        2,
        _PP2_EVENTS,
        _contracts(
            gpt_probe_contract.validate_gpt_pp2_training_phases,
            p2p_probe_contract.validate_pp2_batched_route,
        ),
    ),
    "pp2-unbatched": manifest.TraceProfile(
        "pp2-unbatched",
        2,
        _PP2_UNBATCHED_EVENTS,
        p2p_probe_contract.validate_pp2_unbatched_route,
    ),
    "ep2-alltoall": manifest.TraceProfile("ep2-alltoall", 2, _ep_events("alltoall")),
    "ep2-allgather": manifest.TraceProfile("ep2-allgather", 2, _ep_events("allgather")),
    "ep2-fine-grained": manifest.TraceProfile(
        "ep2-fine-grained", 4, _ep_events("alltoall", fine_grained=True)
    ),
    "dp2-standard-ddp": manifest.TraceProfile(
        "dp2-standard-ddp", 2, _dp_events("standard-ddp")
    ),
    "dp2-standard-ddp-overlap": manifest.TraceProfile(
        "dp2-standard-ddp-overlap",
        2,
        (
            *_dp_events("standard-ddp"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_standard_overlap,
    ),
    "dp2-distopt": manifest.TraceProfile("dp2-distopt", 2, _dp_events("distopt")),
    "dp2-distopt-overlap": manifest.TraceProfile(
        "dp2-distopt-overlap",
        2,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_distopt_overlap,
    ),
    "dp2-layerwise-overlap": manifest.TraceProfile(
        "dp2-layerwise-overlap",
        2,
        (
            *_dp_events("standard-ddp"),
            manifest.EventRequirement("dp-param-all-gather", _DP_FIELDS, "B"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_layerwise_overlap,
    ),
    "dp4-distopt-multi-instance-overlap": manifest.TraceProfile(
        "dp4-distopt-multi-instance-overlap",
        4,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement("dp-allreduce", _DP_FIELDS, "B"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_multi_instance_distopt_overlap,
    ),
    "dp8-standard-ddp": manifest.TraceProfile(
        "dp8-standard-ddp", 8, _dp_events("standard-ddp")
    ),
    "dp8-distopt": manifest.TraceProfile("dp8-distopt", 8, _dp_events("distopt")),
    "te-attn-cuda-graph": manifest.TraceProfile(
        "te-attn-cuda-graph", 1, _events("MLP.forward")
    ),
    "te-moe-router-cuda-graph": manifest.TraceProfile(
        "te-moe-router-cuda-graph",
        2,
        (
            *_EP_COMMON_EVENTS,
            manifest.EventRequirement("ep-alltoall-dispatch", _EP_ROUTE_FIELDS, "E"),
            manifest.EventRequirement("ep-alltoall-combine", _EP_ROUTE_FIELDS, "E"),
        ),
    ),
    "bert-encoder": manifest.TraceProfile("bert-encoder", 1, _events("encoder")),
}

_CONFIG_PROFILES = {
    "flagscale_single_node_smoke": "pp1",
    "flagscale_single_node_gpt_eager_full_smoke": "gpt-eager-full",
    "flagscale_single_node_tp2_sp_local_smoke": "tp2-sp-local",
    "flagscale_single_node_pp2_smoke": "pp2",
    "flagscale_single_node_pp2_unbatched_smoke": "pp2-unbatched",
    "flagscale_single_node_ep2_smoke": "ep2-alltoall",
    "flagscale_single_node_ep2_fine_grained_smoke": "ep2-fine-grained",
    "flagscale_single_node_dp2_standard_smoke": "dp2-standard-ddp",
    "flagscale_single_node_dp2_standard_overlap_smoke": (
        "dp2-standard-ddp-overlap"
    ),
    "flagscale_single_node_dp2_distopt_smoke": "dp2-distopt",
    "flagscale_single_node_dp2_distopt_overlap_smoke": "dp2-distopt-overlap",
    "flagscale_single_node_dp2_layerwise_overlap_smoke": "dp2-layerwise-overlap",
    "flagscale_single_node_dp4_distopt_multi_instance_overlap_smoke": (
        "dp4-distopt-multi-instance-overlap"
    ),
    "flagscale_single_node_dp8_standard_smoke": "dp8-standard-ddp",
    "flagscale_single_node_dp8_distopt_smoke": "dp8-distopt",
    "flagscale_single_node_te_cuda_graph_attn_smoke": "te-attn-cuda-graph",
    "flagscale_single_node_te_cuda_graph_moe_router_smoke": (
        "te-moe-router-cuda-graph"
    ),
    "flagscale_single_node_bert_smoke": "bert-encoder",
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
    if configured is not None:
        if configured == "ep2-alltoall" and args.ep_dispatcher == "allgather":
            configured = "ep2-allgather"
        return PROFILES[configured]
    if args.model_profile == "bert-encoder":
        return PROFILES["bert-encoder"]
    if args.cuda_graph_profile == "transformer-engine-attn":
        return PROFILES["te-attn-cuda-graph"]
    if args.cuda_graph_profile == "transformer-engine-moe-router":
        return PROFILES["te-moe-router-cuda-graph"]
    if args.topology == "tp2":
        return PROFILES["tp2-sp-local"]
    if args.topology == "pp2":
        return PROFILES["pp2"]
    if args.topology == "ep2":
        if args.ep_profile == "fine-grained":
            return PROFILES["ep2-fine-grained"]
        return PROFILES[f"ep2-{args.ep_dispatcher or 'alltoall'}"]
    if args.topology == "dp4":
        return PROFILES[
            f"dp4-{args.dp_profile or 'distopt-multi-instance-overlap'}"
        ]
    if args.topology in {"dp2", "dp8"}:
        return PROFILES[f"{args.topology}-{args.dp_profile or 'standard-ddp'}"]
    return PROFILES["pp1"]


def _ep_dispatcher(profile: manifest.TraceProfile, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    if "alltoall" in profile.name or profile.name in {
        "ep2-fine-grained",
        "te-moe-router-cuda-graph",
    }:
        return "alltoall"
    if "allgather" in profile.name:
        return "allgather"
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
) -> tuple[str, ...]:
    overlay = ()
    if flagscale_training_overlay is not None:
        overlay = (
            "--volume",
            f"{flagscale_training_overlay}:"
            "/workspace/FlagScale/flagscale/train/megatron/training/training.py:ro",
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
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    launcher_log: Path,
    timeout: float,
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
                output.write(
                    f"[FlagScale MegaLens] training timed out after {timeout} seconds\n"
                )
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


def _copy_inputs(
    source: Path,
    run_dir: Path,
    profile: manifest.TraceProfile,
) -> Path:
    inputs = run_dir / "inputs"
    inputs.mkdir()
    destination = inputs / source.name
    shutil.copyfile(source, destination)
    if profile.name == "bert-encoder":
        generate_bert_smoke_inputs.generate_inputs(inputs)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("trace-on", "trace-off"))
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--topology",
        choices=("pp1", "pp2", "tp2", "ep2", "dp2", "dp4", "dp8"),
        default=None,
    )
    parser.add_argument("--ep-dispatcher", choices=("alltoall", "allgather"))
    parser.add_argument("--ep-profile", choices=("standard", "fine-grained"))
    parser.add_argument(
        "--dp-profile",
        choices=(
            "standard-ddp",
            "standard-ddp-overlap",
            "distopt",
            "distopt-overlap",
            "layerwise-overlap",
            "distopt-multi-instance-overlap",
        ),
    )
    parser.add_argument(
        "--cuda-graph-profile",
        choices=("transformer-engine-attn", "transformer-engine-moe-router"),
    )
    parser.add_argument("--model-profile", choices=("bert-encoder",))
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
        source_head = _source_head(source_root)
        rdzv_port = _reserve_loopback_port(args.rdzv_port)
    except ValueError as error:
        parser.error(str(error))

    profile = _profile_from_arguments(args)
    run_dir = args.run_dir.resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"[FlagScale MegaLens] run directory already exists: {run_dir}",
            file=sys.stderr,
        )
        return 2

    started_at = manifest.utc_now()
    copied_config = _copy_inputs(args.input_config.resolve(), run_dir, profile)
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
        run_dir / "traces",
        profile,
        trace_enabled=args.mode == "trace-on",
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
