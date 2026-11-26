import hashlib
import importlib.metadata
import json
import os
import socket
import time
import torch
import torch.distributed as dist

SEQ = 1024
MBS = 8
STEPS = int(os.getenv("REVIEW_STEPS", "13"))


def sample(index):
    return torch.randint(100, 30000, (SEQ,), generator=torch.Generator().manual_seed(87123 + index))


def log_input(tokens):
    if dist.get_rank() == 0:
        versions = {}
        for package in ("torch", "transformer-engine", "transformers", "mbridge", "flash-attn", "liger-kernel"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        print("REVIEW_ENVIRONMENT " + json.dumps({
            "host": socket.gethostname(), "packages": versions,
            "torch_cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
        }), flush=True)
    print(
        "REVIEW_INPUT "
        + json.dumps(
            {
                "rank": dist.get_rank(),
                "sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            }
        ),
        flush=True,
    )


def start_step():
    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter()


def end_step(start, step, loss=None):
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - start, device="cuda", dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if loss is not None:
        v = torch.tensor(float(loss), device="cuda", dtype=torch.float64)
        dist.all_reduce(v)
        loss = v.item() / dist.get_world_size()
    if dist.get_rank() == 0:
        print(
            "REVIEW_STEP "
            + json.dumps(
                {
                    "step": step,
                    "seconds": elapsed.item(),
                    "loss": loss,
                    "input_tokens": MBS * SEQ * dist.get_world_size(),
                    "loss_tokens": MBS * (SEQ - 1) * dist.get_world_size(),
                }
            ),
            flush=True,
        )


def memory():
    pss = 0.0
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if line.startswith("Pss:"):
                pss = int(line.split()[1]) / 1024
    v = [pss, torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, v)
    host_stats = (
        torch.cuda.memory.host_memory_stats()
        if hasattr(torch.cuda.memory, "host_memory_stats")
        else {}
    )
    hosts = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, host_stats)
    if dist.get_rank() == 0:
        print(
            "REVIEW_MEMORY "
            + json.dumps(
                {
                    "ranks_pss_alloc_reserved_mib": values,
                    "sum_pss_mib": sum(x[0] for x in values),
                    "max_alloc_mib": max(x[1] for x in values),
                    "ranks_host_allocator": hosts,
                }
            ),
            flush=True,
        )
