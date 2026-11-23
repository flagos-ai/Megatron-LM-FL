"""Run small four-GPU Slide acceptance jobs and check saved numerical state.

Activate the Megatron environment, then run:
  python tests/functional_tests/slideformer/run_acceptance.py --gpus 4,5,6,7 --output /tmp/slide-check
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error("This acceptance matrix requires four distinct GPU IDs")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh acceptance output directory")
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        SLIDEFORMER_GPU_IDS=args.gpus,
        CUDA_VISIBLE_DEVICES="",
        TP="1",
        PP="1",
        CP="1",
        EP="1",
        GLOBAL_BATCH="8",
        TRAIN_ITERS="5",
        MEGATRON_SLIDEFORMER_GPU_GRADIENT_PIPELINE="0",
        MEGATRON_SLIDEFORMER_SHARDED_CPU_OPT="0",
    )
    smoke = ["bash", "tests/functional_tests/slideformer/run_smoke.sh"]
    shard = {"MEGATRON_SLIDEFORMER_SHARDED_CPU_OPT": "1"}
    fast = dict(
        shard,
        MEGATRON_SLIDEFORMER_EAGER_SHARDED_UPDATE="1",
        MEGATRON_SLIDEFORMER_GPU_REDUCE="1",
        MEGATRON_SLIDEFORMER_SHARD_H2D="1",
        MEGATRON_SLIDEFORMER_ASYNC_GRAD_ACCUMULATE="1",
        PER_TOKEN="1",
        MEGATRON_SLIDEFORMER_GPU_REDUCE_BUCKET_MB="0.125",
    )
    jobs = [
        ("pp_cpu", {"PP": "2", "PER_TOKEN": "1"}, []),
        (
            "pp_gpu",
            {"PP": "2", "PER_TOKEN": "1", "MEGATRON_SLIDEFORMER_GPU_GRADIENT_PIPELINE": "1"},
            [],
        ),
        ("shard_continuous", dict(shard, PER_TOKEN="1"), []),
        ("shard_save3", dict(shard, PER_TOKEN="1"), ["--exit-interval", "3"]),
        ("shard_resume5", dict(shard, PER_TOKEN="1", LOAD_DIR=str(root / "shard_resume")), []),
        ("fast", fast, []),
        ("shard_single", dict(shard, PER_TOKEN="1", GLOBAL_BATCH="4"), []),
        ("fast_single", dict(fast, GLOBAL_BATCH="4"), []),
        ("tp2_reference", dict(shard, TP="2", PER_TOKEN="1"), []),
        ("tp2dp2", dict(fast, TP="2"), []),
        ("cp2dp2", dict(shard, CP="2"), []),
    ]
    available = {name for name, _, _ in jobs}
    if args.cases and not set(args.cases) <= available:
        parser.error(f"Unknown cases: {set(args.cases) - available}")
    records = []
    for name, overrides, extra in jobs:
        if args.cases and name not in args.cases:
            continue
        checkpoint = "shard_resume" if name in {"shard_save3", "shard_resume5"} else name
        if overrides.get("MEGATRON_SLIDEFORMER_GPU_REDUCE") == "1":
            extra = [*extra, "--ddp-reduce-scatter-with-fp32-accumulation"]
        run_env = dict(env, **overrides, CHECKPOINT_DIR=str(root / checkpoint))
        start = time.monotonic()
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(
                ["timeout", "--kill-after=15s", "360", *smoke, *extra],
                cwd=repo,
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record = dict(name=name, returncode=result.returncode, seconds=time.monotonic() - start)
        records.append(record)
        (root / "runs.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")

    # Do not initialize a CUDA context in the parent process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    def load(case, rank):
        path = root / case / "slideformer" / f"rank_{rank:05d}" / "chunk_000" / "iter_0000005.pt"
        return torch.load(path, map_location="cpu", weights_only=False)["engine"]

    def parameters(state):
        return [param for layer in state["layers"] for param in layer["cpu_params"]]

    checks = {}
    ran = {item["name"] for item in records}
    for case in ("pp_cpu", "pp_gpu"):
        if case not in ran:
            continue
        embeddings = []
        for rank in range(4):
            matches = [p for p in parameters(load(case, rank)) if tuple(p.shape) == (1152, 128)]
            assert len(matches) == 1, "Test did not identify the tied embedding"
            embeddings.append(matches[0])
        errors = []
        for first, last in ((0, 2), (1, 3)):
            torch.testing.assert_close(embeddings[first], embeddings[last], atol=0, rtol=0)
            errors.append((embeddings[first] - embeddings[last]).abs().max().item())
        checks[case] = {"tied_embedding_max_abs": errors}
    if {"shard_continuous", "shard_resume5"} <= ran:
        for rank in range(4):
            continuous, resumed = load("shard_continuous", rank), load("shard_resume", rank)
            for a, b in zip(parameters(continuous), parameters(resumed), strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            a, b = continuous["layer_optimizer"], resumed["layer_optimizer"]
            assert a["state_shards"] == b["state_shards"]
            assert a["param_group_steps"] == b["param_group_steps"]
            for field in ("exp_avg_flat", "exp_avg_sq_flat"):
                assert a[field].keys() == b[field].keys()
                for key in a[field]:
                    torch.testing.assert_close(a[field][key], b[field][key], atol=0, rtol=0)
        checks["resume"] = (
            "weights, moments and optimizer steps bitwise identical on all four ranks"
        )
    if {"fast", "shard_continuous"} <= ran:
        errors = []
        for rank in range(4):
            pairs = zip(
                parameters(load("fast", rank)),
                parameters(load("shard_continuous", rank)),
                strict=True,
            )
            for a, b in pairs:
                torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
                errors.append((a - b).abs().max().item())
        checks["gpu_reduce_vs_cpu"] = {"max_abs": max(errors)}
    if {"fast_single", "shard_single"} <= ran:
        errors = []
        for rank in range(4):
            pairs = zip(
                parameters(load("fast_single", rank)),
                parameters(load("shard_single", rank)),
                strict=True,
            )
            for a, b in pairs:
                torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
                errors.append((a - b).abs().max().item())
        checks["pooled_gpu_reduce_vs_cpu"] = {"max_abs": max(errors)}
    if {"tp2_reference", "tp2dp2"} <= ran:
        errors = []
        for rank in range(4):
            pairs = zip(
                parameters(load("tp2dp2", rank)),
                parameters(load("tp2_reference", rank)),
                strict=True,
            )
            for a, b in pairs:
                torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
                errors.append((a - b).abs().max().item())
        checks["tp2_gpu_reduce_vs_cpu"] = {"max_abs": max(errors)}
    (root / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
