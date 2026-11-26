"""Process placement shared by the native SSDP and Megatron benchmarks."""

import os


def configure_worker():
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLIDEFORMER_GPU_IDS"].split(",")[rank]
    os.environ["LOCAL_RANK"] = "0"
    placement = os.getenv("SLIDE_BENCH_CPU_AFFINITY")
    if placement:
        cpus = set()
        for part in placement.split(";")[rank].split(","):
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
        if not cpus:
            raise ValueError("Empty CPU affinity for benchmark worker")
        os.sched_setaffinity(0, cpus)
