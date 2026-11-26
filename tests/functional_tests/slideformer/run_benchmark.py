"""Four-GPU Qwen2.5-7B benchmark: DP4 SSDP comparison or Megatron-only TP validation."""

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpus", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--ssdp-repo", type=pathlib.Path, required=True)
parser.add_argument("--ssdp-revision", help="Revision of the supplied SSDP checkout/archive")
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--steps", type=int, default=13)
parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2, 4), default=1)
parser.add_argument(
    "--cpu-affinity", help="Optional per-rank CPU sets, e.g. 32-39;40-47;48-55;56-63"
)
parser.add_argument("--order", nargs="+", default=["ssdp_a", "megatron_a", "megatron_b", "ssdp_b"])
args = parser.parse_args()
if len(args.gpus.split(",")) != 4 or len(set(args.gpus.split(","))) != 4:
    parser.error("This protocol requires four distinct GPU IDs")
if args.steps <= 3:
    parser.error("At least four steps are required")
if any(not m.startswith(("ssdp_", "megatron_")) for m in args.order):
    parser.error("Run names must begin with ssdp_ or megatron_")
if args.tensor_parallel_size != 1 and any(not m.startswith("megatron_") for m in args.order):
    parser.error("TP runs are Megatron-only validation; SSDP comparison requires TP=1")

root = args.output.resolve()
if root.exists() and any(root.iterdir()):
    raise FileExistsError("Use a fresh output directory")
(root / "logs").mkdir(parents=True, exist_ok=True)
repo = pathlib.Path(__file__).resolve().parents[3]


def source_manifest(directory):
    """Also works for a source snapshot without .git on the experiment host."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".sh", ".cpp", ".h", ".cu"}
        and not {".git", "__pycache__", ".venv"}.intersection(path.relative_to(directory).parts)
    }


sources = {"megatron": source_manifest(repo), "ssdp": source_manifest(args.ssdp_repo)}
(root / "source_manifest.json").write_text(json.dumps(sources, indent=2))
env = os.environ.copy()
if args.cpu_affinity:
    if len(args.cpu_affinity.split(";")) != 4:
        parser.error("Expected four CPU sets")
    env["SLIDE_BENCH_CPU_AFFINITY"] = args.cpu_affinity
env.update(
    PYTHONPATH=str(repo) + ":" + os.environ.get("PYTHONPATH", ""),
    SLIDEFORMER_GPU_IDS=args.gpus,
    REVIEW_STEPS=str(args.steps),
    OMP_NUM_THREADS="8",
    CUDA_DEVICE_MAX_CONNECTIONS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
model = str(pathlib.Path(args.model).resolve())
env.update(
    SLIDEFORMER_REAL_MODEL=model,
    MEGATRON_SLIDEFORMER_ENABLE="1",
    MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD="1",
    MEGATRON_SLIDEFORMER_KERNEL_POLICY="auto",
    MEGATRON_SLIDEFORMER_STRICT_KERNELS="1",
    NVTE_ALLOW_NONDETERMINISTIC_ALGO="1",
    MEGATRON_SLIDEFORMER_GPU_GRADIENT_PIPELINE="0",
    MEGATRON_SLIDEFORMER_SHARDED_CPU_OPT="1",
    MEGATRON_SLIDEFORMER_EAGER_SHARDED_UPDATE="1",
    MEGATRON_SLIDEFORMER_SPLIT_SWIGLU_THRESHOLD_GIB="0",
    MEGATRON_SLIDEFORMER_GPU_REDUCE="1",
    MEGATRON_SLIDEFORMER_SHARD_H2D="1",
    MEGATRON_SLIDEFORMER_SHARDED_EXEC_CACHE=os.getenv(
        "MEGATRON_SLIDEFORMER_SHARDED_EXEC_CACHE", "0"
    ),
    MEGATRON_SLIDEFORMER_ASYNC_GRAD_ACCUMULATE="1",
    MEGATRON_SLIDEFORMER_CPU_GRAD_BUFFER_COUNT=os.getenv(
        "MEGATRON_SLIDEFORMER_CPU_GRAD_BUFFER_COUNT", "1"
    ),
    MEGATRON_SLIDEFORMER_GPU_GRAD_BUFFER_COUNT=os.getenv(
        "MEGATRON_SLIDEFORMER_GPU_GRAD_BUFFER_COUNT", "1"
    ),
)
native_args = (
    shlex.split(
        """--attention-backend flash --bf16 --calculate-per-token-loss
 --ddp-reduce-scatter-with-fp32-accumulation
 --no-save-optim --no-save-rng --save-interval 100000 --context-parallel-size 1
 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --log-interval 1
 --no-gradient-accumulation-fusion --num-workers 0 --adam-beta1 0.9 --adam-beta2 0.999
 --adam-eps 1e-8 --attention-dropout 0 --hidden-dropout 0 --disable-bias-linear
 --eval-interval 10000 --eval-iters 0 --ffn-hidden-size 18944 --global-batch-size 32
 --micro-batch-size 8 --group-query-attention --hidden-size 3584 --kv-channels 128
 --lr 1e-5 --lr-decay-style constant --lr-warmup-iters 0 --min-lr 1e-5
 --max-position-embeddings 32768 --norm-epsilon 1e-6 --normalization RMSNorm
 --num-attention-heads 28 --num-layers 28 --num-query-groups 4 --position-embedding-type rope
 --rotary-base 1000000 --seed 1234 --seq-length 1024 --swiglu
 --transformer-impl transformer_engine --weight-decay 0 --clip-grad 0 --add-qkv-bias
 --make-vocab-size-divisible-by 128 --tokenizer-type HuggingFaceTokenizer
 --dataloader-type single --distributed-timeout-minutes 4"""
    )
    + ["--tokenizer-model", model, "--train-iters", str(args.steps)]
)
native_args[native_args.index("--tensor-model-parallel-size") + 1] = str(args.tensor_parallel_size)
if args.tensor_parallel_size > 1:
    native_args += ["--sequence-parallel", "--no-create-attention-mask-in-dataloader"]
ssdp_config = root / "ssdp_config.json"
ssdp_config.write_text(
    json.dumps(
        {
            "weight_decay": 0.0,
            "lr": 1e-5,
            "seed": 1234,
            "enable_timing": False,
            "lce_backend": "legacy_lce",
            "num_epochs": 1,
        }
    )
)
ssdp_args = [
    "--config",
    str(ssdp_config),
    "--model-path",
    model,
    "--param-transfer-mode",
    "shard",
    "--grad-sync-mode",
    "gpu_reduce",
    "--batch-size",
    "8",
    "--seq-length",
    "1024",
    "--max-steps",
    str(args.steps),
    "--dummy-dataset-size",
    str(32 * args.steps),
    "--no-auto-policy",
    "--output-dir",
    str(root / "ssdp_output"),
]
(root / "protocol.json").write_text(
    json.dumps(
        {
            "gpus": args.gpus,
            "model": model,
            "ssdp_repo": str(args.ssdp_repo.resolve()),
            "ssdp_revision": args.ssdp_revision,
            "cpu_affinity": args.cpu_affinity,
            "profile_dir": env.get("SLIDE_BENCH_PROFILE_DIR"),
            "source_manifest_sha256": hashlib.sha256(
                (root / "source_manifest.json").read_bytes()
            ).hexdigest(),
            "steps": args.steps,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 4 // args.tensor_parallel_size,
            "sequence_parallel": args.tensor_parallel_size > 1,
            "global_batch_size": 32,
            "microbatch_size": 8,
            "microbatches_per_step": args.tensor_parallel_size,
            "warmup_steps": 3,
            "order": args.order,
            "slide_environment": {
                k: v for k, v in env.items() if k.startswith("MEGATRON_SLIDEFORMER_")
            },
        },
        indent=2,
    )
)
records = []
for mode in args.order:
    # Abort if any selected GPU has another compute process.
    for g in args.gpus.split(","):
        used = subprocess.check_output(
            ["nvidia-smi", "-i", str(g), "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if used:
            raise RuntimeError(f"GPU {g} busy: {used}")
    is_mt = mode.startswith("megatron")
    e = dict(env)
    if not is_mt:
        e["SLIDE_BENCH_SSDP_REPO"] = str(args.ssdp_repo.resolve())
        e["PYTHONPATH"] = str(args.ssdp_repo.resolve()) + ":" + os.environ.get("PYTHONPATH", "")
        e.pop("SSDP_DISABLE_FUSED_CE", None)
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(
            repo
            / "tests/functional_tests/slideformer"
            / ("bench_megatron.py" if is_mt else "bench_ssdp.py")
        ),
    ]
    command += native_args if is_mt else ssdp_args
    start = time.time()
    print("START " + mode, flush=True)
    with (root / "logs" / f"perf_{mode}.log").open("w") as f:
        p = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=20s", "1200", *command],
            cwd=repo,
            env=e,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    record = {"mode": mode, "rc": p.returncode, "wall_s": time.time() - start, "command": command}
    records.append(record)
    (root / "performance_runs.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    if p.returncode:
        raise RuntimeError(f"{mode} failed; see its log")
if sources != {"megatron": source_manifest(repo), "ssdp": source_manifest(args.ssdp_repo)}:
    raise RuntimeError("Source files changed during the benchmark; discard these timings")
