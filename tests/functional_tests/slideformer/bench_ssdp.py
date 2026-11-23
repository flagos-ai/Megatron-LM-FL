import os
import sys
from pathlib import Path
from bench_runtime import configure_worker

configure_worker()
sys.path.insert(0, os.environ["SLIDE_BENCH_SSDP_REPO"])
import torch
import torch.distributed as dist
import train
from bench_common import SEQ, MBS, STEPS, sample, log_input, start_step, end_step, memory


def epoch(model, dataloader, config, device, epoch, total_flops, world_size, max_steps=None):
    model.train()
    rank = dist.get_rank()
    batches = [
        torch.stack([sample(s * world_size * MBS + rank * MBS + j) for j in range(MBS)])
        for s in range(STEPS)
    ]
    log_input(batches[0])
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for step, tokens in enumerate(batches, 1):
        profiler = None
        profile_dir = os.getenv("SLIDE_BENCH_PROFILE_DIR")
        if profile_dir and step == 5 and rank == 0:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            )
            profiler.start()
        t = start_step()
        tokens = tokens.to(device)
        out = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=tokens)
        if not config.auto_backward_in_forward:
            out.loss.backward()
        model.wait_for_completion()
        loss = out.loss.detach().item()
        losses.append(loss)
        end_step(t, step, loss)
        if profiler is not None:
            profiler.stop()
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(Path(profile_dir) / "ssdp_rank0_step5.json"))
    memory()
    return {
        "avg_loss": sum(losses) / len(losses),
        "steps": float(STEPS),
        "avg_step_s": 0.0,
        "tokens_per_s": 0.0,
    }


train.train_epoch = epoch
train.main()
