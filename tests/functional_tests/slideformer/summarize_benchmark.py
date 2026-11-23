"""Summarize completed runs without discarding measured-step outliers."""

import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol.get("profile_dir"):
        raise ValueError("Profiler runs are diagnostic and cannot establish performance parity")
    runs = json.loads((root / "performance_runs.json").read_text())
    report = {"protocol": protocol, "runs": {}, "aggregate": {}}
    input_hashes = None
    timings = {}
    for run in runs:
        if run["rc"]:
            raise ValueError(f"Incomplete run: {run['mode']}")
        log = (root / "logs" / f"perf_{run['mode']}.log").read_text()

        def records(prefix):
            decoder = json.JSONDecoder()
            return [
                decoder.raw_decode(log[match.end() :])[0]
                for match in re.finditer(re.escape(prefix), log)
            ]

        rows = records("REVIEW_STEP ")
        inputs = {item["rank"]: item["sha256"] for item in records("REVIEW_INPUT ")}
        if len(inputs) != 4 or [item["step"] for item in rows] != list(
            range(1, protocol["steps"] + 1)
        ):
            raise ValueError(f"Missing inputs or training steps: {run['mode']}")
        if input_hashes is None:
            input_hashes = inputs
        elif inputs != input_hashes:
            raise ValueError("Frameworks did not consume the same first batch on every rank")
        values = [item["seconds"] for item in rows if item["step"] > protocol["warmup_steps"]]
        memory = records("REVIEW_MEMORY ")[-1]
        losses = [float(value) for value in re.findall(r"lm loss:\s*([\d.E+-]+)", log)]
        if not losses:
            losses = [row["loss"] for row in rows if row["loss"] is not None]
        result = {
            "measured_steps": len(values),
            "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "min_s": min(values),
            "max_s": max(values),
            "input_tokens_per_s": 32768 / statistics.mean(values),
            "end_sum_pss_gib": memory["sum_pss_mib"] / 1024,
            "peak_allocated_gpu_gib": memory["max_alloc_mib"] / 1024,
            "first_loss": losses[0],
            "last_loss": losses[-1],
        }
        report["runs"][run["mode"]] = result
        backend = run["mode"].split("_", 1)[0]
        timings.setdefault(backend, []).extend(values)
    for backend, values in timings.items():
        report["aggregate"][backend] = {
            "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "input_tokens_per_s": 32768 / statistics.mean(values),
            "measured_steps": len(values),
        }
    if {"megatron", "ssdp"} <= timings.keys():
        report["megatron_to_ssdp_time_ratio"] = statistics.mean(
            timings["megatron"]
        ) / statistics.mean(timings["ssdp"])
    report["input_sha256_by_rank"] = input_hashes
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    result = json.dumps(report, indent=2)
    (args.directory / "summary.json").write_text(result)
    print(result)
